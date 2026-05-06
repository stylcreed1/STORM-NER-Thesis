# coding=utf-8
#
# Copyright 2020-2024 Heinrich Heine University Duesseldorf
#
# Part of this code is based on the source code of BERT-DST
# (arXiv:1907.03040)
# Part of this code is based on the source code of Transformers
# (arXiv:1910.03771)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import autograd_hacks
import copy
import glob
import gzip
import higher
import json
import logging
import math
import os
import pickle
import random
import re
import sys
import torch
import transformers
import warnings

import numpy as np

from accelerate import (Accelerator)
from tensorboardX import (SummaryWriter)
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler)
from tqdm import (tqdm, trange)
from transformers import (
    WEIGHTS_NAME,
    BertConfig, BertModel, BertTokenizer,
    RobertaConfig, RobertaModel, RobertaTokenizer,
    ElectraConfig, ElectraModel, ElectraTokenizer)

from dst.agra_dst import AGRA
from dst.modeling_dst import (TransformerForDST)
from dst.utils_storm_dst import (Results, Filter, batch_to_dict)
from utils_gpu import (from_device, to_device)
from utils_storm import (print_header)

trippy_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'trippy'))
sys.path.insert(0, trippy_path)

from trippy.data_processors import (PROCESSORS)
from trippy.run_dst import (eval_metric, predict_and_format)
from trippy.utils_dst import (convert_examples_to_features)
from trippy.tensorlistdataset import (TensorListDataset)

warnings.filterwarnings("ignore", "Detected call of `lr_scheduler\.step\(\)` before `optimizer\.step\(\)`\.", UserWarning)

MODEL_CLASSES = {
    'bert': (BertConfig, TransformerForDST('bert'), BertModel, BertTokenizer),
    'roberta': (RobertaConfig, TransformerForDST('roberta'), RobertaModel, RobertaTokenizer),
    'electra': (ElectraConfig, TransformerForDST('electra'), ElectraModel, ElectraTokenizer),
}

logger = logging.getLogger(__name__)


def parse_args():
    def list_of_strings(arg):
        return arg.split(',')
    
    parser = argparse.ArgumentParser(description="STORM for dialogue state tracking with TripPy.")

    # Required parameters
    parser.add_argument("--task_name", type=str, default=None, required=True,
                        help="Name of the task (e.g., multiwoz21).")
    parser.add_argument("--data_dir", type=str, default=None, required=True,
                        help="Task database.")
    parser.add_argument("--dataset_config", type=str, default=None, required=True,
                        help="Dataset configuration file.")
    parser.add_argument("--predict_type", type=str, default=None, required=True,
                        help="Portion of the data to perform prediction on (e.g., dev, test).")
    parser.add_argument("--model_type", type=str, default=None, required=True,
                        help="Model type.",
                        choices=list(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path",type=str, default=None, required=True,
                        help="Path to pretrained model or model identifier from huggingface.co/models.")
    parser.add_argument("--output_dir", type=str, default=None, required=True,
                        help="The output directory where the model checkpoints and predictions will be written.")

    # Other parameters
    parser.add_argument("--max_seq_length", type=int, default=180,
                        help="Maximum input length after tokenization. "
                             "Longer sequences will be truncated, shorter ones padded.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the <predict_type> set.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--dropout_rate", type=float, default=0.3,
                        help="Dropout rate for transformer encoder representations.")
    parser.add_argument("--heads_dropout", type=float, default=0.0,
                        help="Dropout rate for classification heads.")
    parser.add_argument("--class_loss_ratio", type=float, default=0.8,
                        help="The ratio applied on class loss in total loss calculation. "
                             "Should be a value in [0.0, 1.0]. "
                             "The ratio applied on token loss is (1-class_loss_ratio)/2. "
                             "The ratio applied on refer loss is (1-class_loss_ratio)/2.")
    parser.add_argument("--token_loss_for_nonpointable", action='store_true',
                        help="Whether the token loss for classes other than copy_value contribute towards total loss.")
    parser.add_argument("--refer_loss_for_nonpointable", action='store_true',
                        help="Whether the refer loss for classes other than refer contribute towards total loss.")

    parser.add_argument("--no_append_history", action='store_true',
                        help="Whether or not to append the dialog history to each turn.")
    parser.add_argument("--no_use_history_labels", action='store_true',
                        help="Whether or not to label the history as well.")
    parser.add_argument("--no_label_value_repetitions", action='store_true',
                        help="Whether or not to label values that have been mentioned before.")
    parser.add_argument("--swap_utterances", action='store_true',
                        help="Whether or not to swap the turn utterances (default: usr|sys, swapped: sys|usr).")
    parser.add_argument("--delexicalize_sys_utts", action='store_true',
                        help="Whether or not to delexicalize the system utterances.")
    parser.add_argument("--class_aux_feats_inform", action='store_true',
                        help="Whether or not to use the identity of informed slots as auxiliary featurs for class prediction.")
    parser.add_argument("--class_aux_feats_ds", action='store_true',
                        help="Whether or not to use the identity of slots in the current dialog state as auxiliary featurs for class prediction.")

    parser.add_argument("--train_batch_size", type=int, default=8,
                        help="Batch size for the training dataloader.")
    parser.add_argument("--eval_batch_size", type=int, default=8,
                        help="Batch size for the evaluation dataloader.")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="Initial learning rate (after the potential warmup period) to use.")
    parser.add_argument("--rescaler_learning_rate", type=float, default=1e-2,
                        help="Initial learning rate (after the potential warmup period) to use.")
    parser.add_argument("--weight_decay", type=float, default=0.0,
                        help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-6,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--num_train_epochs", type=int, default=3,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_train_steps", type=int, default=None,
                        help="Total number of training steps to perform. If provided, overrides num_train_epochs.")
    parser.add_argument("--optimizer", type=str, default='AdamW',
                        help="Optimizer to use.",
                        choices=["Adam", "AdamW"])
    parser.add_argument("--lr_scheduler_type", type=str, default="linear",
                        help="The scheduler type to use.",
                        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"])
    parser.add_argument("--warmup_proportion", type=float, default=0.0,
                        help="Proportion of steps for the warmup in the lr scheduler.")
    parser.add_argument("--svd", type=float, default=0.0,
                        help="Slot value dropout ratio.")

    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--local_files_only', action='store_true',
                        help="Whether to only load local model files (useful when working offline).")
    parser.add_argument('--logging_steps', type=int, default=100,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=0,
                        help="Save checkpoint every X updates steps. Overwritten by --save_epochs.")
    parser.add_argument('--save_epochs', type=int, default=0,
                        help="Save checkpoint every X epochs. Overrides --save_steps.")
    parser.add_argument("--save_stats", action='store_true',
                        help="When set, saves detailed training statistics as gzipped json.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")

    parser.add_argument('--stats_window', type=int, default=1,
                        help="Window size for sample statistics memory (in number of batches).")
    parser.add_argument("--simulate_only", action='store_true',
                        help="When set, rescaling is not actually applied (but still learned).")
    parser.add_argument('--dropout_rounds', type=int, default=2,
                        help="Number of additional forward passes to compute sample statistics.")
    parser.add_argument("--no_cat", action='store_true',
                        help="When set, sample statistics are computed by label-prediction agreement.")
    parser.add_argument("--agra", action='store_true',
                        help="Use original AGRA method instead of rescaler. "
                             "AGRA does not use meta learning.")
    parser.add_argument("--agra_weighted", action='store_true',
                        help="When set, uses class weighting to sample comparison batches.")
    parser.add_argument("--agra_loss", type=str, default="CE",
                        help="Comparison loss for AGRA.",
                        choices=["CE", "F1"])
    parser.add_argument("--meta_dataset", type=str, default="train",
                        help="Dataset used for the meta update step (default for STORM: 'train').",
                        choices=["train", "eval"])
    parser.add_argument('--meta_innerloop_rounds', type=int, default=1,
                        help="Number of inner loop traversals for meta learning.")
    parser.add_argument("--rescaler_binary", action='store_true',
                        help="When set, rescaler will produce binary loss weights, i.e., 0 or 1.")
    parser.add_argument("--rescaler_binary_threshold", type=float, default=0.5,
                        help="Confidence threshold for binary rescaler.",
                        choices=[0.1, 0.2, 0.3, 0.4, 0.5])
    parser.add_argument("--rescaler_feats", type=str, default="default",
                        help="Feature set for rescaler.",
                        choices=["default", "loss"])
    parser.add_argument("--rescaler_feats_no_cat", action='store_true',
                        help="When set, cat is not added to the feature set of the rescaler.")
    parser.add_argument("--no_class_separation", action='store_true',
                        help="When set, sample statistics are not separated by target classes.")
    parser.add_argument("--no_meta_loss_rescaling", action='store_true',
                        help="When set, no meta loss rescaling is used.")

    args = parser.parse_args()

    assert args.warmup_proportion >= 0.0 and args.warmup_proportion <= 1.0
    assert args.svd >= 0.0 and args.svd <= 1.0
    assert not args.class_aux_feats_ds or args.eval_batch_size == 1
    assert not args.class_aux_feats_inform or args.eval_batch_size == 1

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    return args


def train(args, train_dataset, eval_dataset, features, model, tokenizer, processor):
    tb_writer = SummaryWriter()

    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, drop_last=True)

    if args.max_train_steps is not None:
        t_total = args.max_train_steps
        args.num_train_epochs = args.max_train_steps // len(train_dataloader) + 1
    else:
        t_total = len(train_dataloader) * args.num_train_epochs

    if args.save_epochs > 0:
        args.save_steps = t_total // args.num_train_epochs * args.save_epochs

    num_warmup_steps = int(t_total * args.warmup_proportion)

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and not "rescaler" in n],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and not "rescaler" in n],
            "weight_decay": 0.0,
        },
    ]
    inner_optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and not "rescaler" in n],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and not "rescaler" in n],
            "weight_decay": 0.0,
        },
    ]
    outer_optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and "rescaler" in n],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and "rescaler" in n],
            "weight_decay": 0.0,
        },
    ]
    inner_optimizer = torch.optim.Adam(inner_optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    if args.optimizer == "Adam":
        optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        meta_optimizer = torch.optim.Adam(outer_optimizer_grouped_parameters, lr=args.rescaler_learning_rate, eps=args.adam_epsilon)
    else:
        optimizer = transformers.AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        meta_optimizer = transformers.AdamW(outer_optimizer_grouped_parameters, lr=args.rescaler_learning_rate, eps=args.adam_epsilon)

    # This would jointly regulate the lr for optimizer and inner_optimizer
    # if both optimizers would use the same grouped_parameters, i.e.,
    # the lr seems to be tied to the parameters, not to the optimizer.
    scheduler = transformers.get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=inner_optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=t_total,
    )

    if args.agra:
        agra = AGRA(args.agra_loss,
                    model.class_labels,
                    args.agra_weighted,
                    train_dataset,
                    model=model,
                    device=args.device,
                    window_size=args.stats_window)

    # Train!
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Total train batch size = {args.train_batch_size}")
    logger.info(f"  Total optimization steps = {t_total}")
    logger.info(f"  Warmup steps = {num_warmup_steps}")

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=False)

    stat_list = {}
    filter_list_by_epoch = {-1: []}
    for epoch_itr, _ in enumerate(train_iterator):
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=False)

        train_inner_sampler = RandomSampler(train_dataset)
        train_inner_dataloader = DataLoader(
            train_dataset, sampler=train_inner_sampler, batch_size=args.train_batch_size, drop_last=True)

        # Meta dataloader
        if args.meta_dataset == "eval":
            meta_dataset = eval_dataset
        else:
            meta_dataset = train_dataset
        meta_sampler = RandomSampler(meta_dataset)
        meta_dataloader = DataLoader(
            meta_dataset, sampler=meta_sampler, batch_size=args.train_batch_size, drop_last=True)

        if args.agra:
            agra.build_dataloader(args.train_batch_size)

        model.train()

        filter_list_by_epoch[epoch_itr] = []
        # Setting this to 1 will cause Filter to not separate stats by target class
        num_filter_labels = 1 if args.no_class_separation else model.class_labels
        epoch_filter = Filter(num_filter_labels,
                              args.stats_window,
                              args.train_batch_size)
        if not args.no_meta_loss_rescaling:
            meta_epoch_filter = Filter(num_filter_labels,
                                       args.stats_window,
                                       args.train_batch_size)
        for step, orig_batch in enumerate(epoch_iterator):
            orig_batch = to_device(batch_to_dict(orig_batch), args.device)
            batch_size = orig_batch['input_ids'].size(0)

            meta_batch = next(iter(meta_dataloader))
            meta_batch = to_device(batch_to_dict(meta_batch), args.device)
            meta_optimizer.zero_grad()

            total_inner_rescaler_outputs = {}
            with higher.innerloop_ctx(model, inner_optimizer, copy_initial_weights=False, track_higher_grads=True) as (fmodel, diffopt):
                for ir in range(args.meta_innerloop_rounds):
                    if ir > 0:
                        batch = next(iter(train_inner_dataloader))
                        batch = to_device(batch_to_dict(batch), args.device)
                    else:
                        batch = orig_batch

                    if args.agra:
                        grad_scores = agra.agra_step(batch)

                    outputs = fmodel(**batch)

                    # These are the losses we use for backprop
                    class_losses = outputs[2][0]
                    token_losses = outputs[4][0]
                    refer_losses = outputs[7][0]
                    outputs = from_device(outputs)
                    labels = from_device(batch['class_label_id'])
                    step_results = Results(outputs[2], outputs[3], labels)

                    if args.agra:
                        step_results['grad_scores'] = grad_scores

                    # Get batch loss/grad statistics, accumulated across samples
                    epoch_filter.update_batch_stats(step_results)

                    # Get stats
                    kls, ovls = epoch_filter.get_sample_stats(batch,
                                                              step_results,
                                                              args.no_cat)

                    # ---------------------------
                    # Rescale sample losses
                    # ---------------------------

                    total_loss = None # Recompute total loss as in modeling_dst.py
                    filter_ids = {}
                    for slot in model.slot_list:
                        if args.agra:
                            # AGRA baseline
                            inner_rescaler_outputs = (grad_scores[slot] >= 0).float().to(args.device)
                        else:
                            # Get weights for each sample for each slot gate
                            inner_rescaler_outputs = call_rescaler(args, fmodel, slot,
                                                                   epoch_filter, step_results,
                                                                   kls, ovls,
                                                                   class_losses, batch)

                        # Rescale
                        if not args.simulate_only:
                            scaling_factor = class_losses[slot].sum()
                            class_losses[slot] *= inner_rescaler_outputs
                            scaling_factor /= class_losses[slot].sum()
                            if not args.no_meta_loss_rescaling:
                                class_losses[slot] *= scaling_factor
                        if ir == 0:
                            total_inner_rescaler_outputs[slot] = inner_rescaler_outputs

                        if 'refer' in model.class_types:
                            per_example_loss = \
                                (args.class_loss_ratio) * class_losses[slot] + \
                                ((1 - args.class_loss_ratio) / 2) * token_losses[slot] + \
                                ((1 - args.class_loss_ratio) / 2) * refer_losses[slot]
                        else:
                            per_example_loss = \
                                args.class_loss_ratio * class_losses[slot] + \
                                (1 - args.class_loss_ratio) * token_losses[slot]

                        if total_loss is None:
                            total_loss = per_example_loss.sum()
                        else:
                            total_loss += per_example_loss.sum()

                        # Get stats
                        if ir == 0:
                            filter_ids[slot] = batch['ids'][(inner_rescaler_outputs < 0.5).cpu()].tolist()

                    # Get stats
                    if ir == 0:
                        filter_list_by_epoch[epoch_itr] += [x for xs in list(filter_ids.values()) for x in xs]

                        for l_itr in range(batch_size):
                            batch_id = batch['ids'][l_itr].item()

                            # Collect statistics
                            if batch_id not in stat_list:
                                stat_list[batch_id] = []
                            stat_list[batch_id].append(
                                {"is_filtered": {s: batch_id in filter_ids[s] for s in model.slot_list},
                                 "epoch": epoch_itr,
                                 "label": {s: features[batch_id].class_label_id[s] for s in model.slot_list},
                                 "weight": {s: total_inner_rescaler_outputs[s][l_itr].item() for s in model.slot_list},
                                 "guid": features[batch_id].guid,
                                 "pred": {s: step_results["preds"][s][0][l_itr].item() for s in model.slot_list},
                                 "probs": {s: step_results["probs"][s][0][l_itr].tolist() for s in model.slot_list}})

                    diffopt.step(total_loss) # Perform a training step

                    if ir > 0:
                        batch = from_device(batch)

                # End of inner loop(s)

                # Meta update
                if not args.agra:
                    meta_outputs = fmodel(**meta_batch)
                    meta_loss = meta_outputs[0][0]
                    meta_class_losses = meta_outputs[2][0]
                    meta_token_losses = meta_outputs[4][0]
                    meta_refer_losses = meta_outputs[7][0]
                    meta_outputs = from_device(meta_outputs)
                    meta_labels = from_device(meta_batch['class_label_id'])

                    if not args.no_meta_loss_rescaling:
                        meta_step_results = Results(meta_outputs[2], meta_outputs[3], meta_labels)
                        meta_epoch_filter.update_batch_stats(meta_step_results)
                        meta_kls, meta_ovls = meta_epoch_filter.get_sample_stats(meta_batch,
                                                                                 meta_step_results,
                                                                                 args.no_cat)

                        meta_total_loss = None # Recompute total loss as in modeling_dst.py
                        for slot in model.slot_list:
                            rescaler_outputs = call_rescaler(args, fmodel, slot,
                                                             meta_epoch_filter, meta_step_results,
                                                             meta_kls, meta_ovls,
                                                             meta_class_losses, meta_batch)
                            if not args.simulate_only:
                                meta_scaling_factor = meta_class_losses[slot].sum()
                                meta_class_losses[slot] *= rescaler_outputs
                                meta_scaling_factor /= meta_class_losses[slot].sum()
                                meta_class_losses[slot] *= meta_scaling_factor

                            if 'refer' in model.class_types:
                                meta_per_example_loss = \
                                    (args.class_loss_ratio) * meta_class_losses[slot] + \
                                    ((1 - args.class_loss_ratio) / 2) * meta_token_losses[slot] + \
                                    ((1 - args.class_loss_ratio) / 2) * meta_refer_losses[slot]
                            else:
                                meta_per_example_loss = \
                                    args.class_loss_ratio * meta_class_losses[slot] + \
                                    (1 - args.class_loss_ratio) * meta_token_losses[slot]
                            if meta_total_loss is None:
                                meta_total_loss = meta_per_example_loss.sum()
                            else:
                                meta_total_loss += meta_per_example_loss.sum()
                        meta_loss = meta_total_loss

                    meta_loss.backward()

            # Meta update
            if not args.agra:
                meta_optimizer.step()
                meta_optimizer.zero_grad()

            # Update model parameters
            model.zero_grad()
            outputs = model(**orig_batch)
            class_losses = outputs[2][0]
            token_losses = outputs[4][0]
            refer_losses = outputs[7][0]
            outputs = from_device(outputs)
            total_loss = None # Recompute total loss as in modeling_dst.py
            for slot in model.slot_list:
                if not args.simulate_only:
                    class_losses[slot] *= total_inner_rescaler_outputs[slot].clone().detach()
                if 'refer' in model.class_types:
                    per_example_loss = \
                        (args.class_loss_ratio) * class_losses[slot] + \
                        ((1 - args.class_loss_ratio) / 2) * token_losses[slot] + \
                        ((1 - args.class_loss_ratio) / 2) * refer_losses[slot]
                else:
                    per_example_loss = \
                        args.class_loss_ratio * class_losses[slot] + \
                        (1 - args.class_loss_ratio) * token_losses[slot]
                if total_loss is None:
                    total_loss = per_example_loss.sum()
                else:
                    total_loss += per_example_loss.sum()
            tr_loss += total_loss.item()
            total_loss.backward()
            for o, oi in zip(optimizer.param_groups, inner_optimizer.param_groups):
                o['lr'] = oi['lr']
            optimizer.step()
            optimizer.zero_grad()
            if args.agra:
                autograd_hacks.clear_grad1(model)

            orig_batch = from_device(orig_batch)

            scheduler.step()
            global_step += 1
            
            # Log metrics
            if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                tb_writer.add_scalar('lr', scheduler.get_last_lr()[0], global_step)
                tb_writer.add_scalar('loss', (tr_loss - logging_loss) / args.logging_steps, global_step)
                logging_loss = tr_loss

            # Save model checkpoint
            if args.save_steps > 0 and global_step % args.save_steps == 0:
                output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                logger.info("Saving model checkpoint to %s", output_dir)
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                model.save_pretrained(output_dir)

            if args.max_train_steps is not None and global_step > args.max_train_steps:
                epoch_iterator.close()
                break

        logger.info("")
        logger.info("Filtered (loss weight < 0.5) samples in epoch %s: %s of %s" % (epoch_itr,
                                                                                    len(filter_list_by_epoch[epoch_itr]),
                                                                                    len(features) * len(model.slot_list)))
        tb_writer.add_scalars('filtered_samples', {'Filtered samples': len(filter_list_by_epoch[epoch_itr]),
                                                   'Total': len(features) * len(model.slot_list)}, epoch_itr)

        # Evaluate task performance
        if args.evaluate_during_training:
            results = evaluate(args, model, tokenizer, processor, prefix=global_step)
            logger.info(f"epoch {epoch_itr}: {results}")
            for key, value in results.items():
                tb_writer.add_scalar('eval_{}'.format(key), value, global_step)

        if args.max_train_steps is not None and global_step > args.max_train_steps:
            train_iterator.close()
            break

    tb_writer.close()

    if args.output_dir is not None and args.save_stats:
        with gzip.open(os.path.join(args.output_dir, "stat_list.%s.json.gz" % (args.predict_type)), "w") as f:
            f.write(json.dumps(stat_list, indent=2).encode('utf-8'))

    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, processor, prefix=""):
    dataset, features = load_and_cache_examples(args, model, tokenizer, processor, evaluate=True)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    args.eval_batch_size = args.eval_batch_size
    eval_sampler = SequentialSampler(dataset) # Note that DistributedSampler samples randomly
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    all_results = []
    all_preds = []
    ds = {slot: 'none' for slot in model.slot_list}
    with torch.no_grad():
        diag_state = {slot: torch.tensor([0 for _ in range(args.eval_batch_size)]).to(args.device) for slot in model.slot_list}
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = to_device(batch, args.device)

        # Reset dialog state if turn is first in the dialog.
        turn_itrs = [features[i.item()].guid.split('-')[-1] for i in batch[9]]
        reset_diag_state = np.where(np.array(turn_itrs) == '0')[0]
        for slot in model.slot_list:
            for i in reset_diag_state:
                diag_state[slot][i] = 0

        with torch.no_grad():
            inputs = {'input_ids':       batch[0],
                      'input_mask':      batch[1],
                      'segment_ids':     batch[2],
                      'start_pos':       batch[3],
                      'end_pos':         batch[4],
                      'inform_slot_id':  batch[5],
                      'refer_id':        batch[6],
                      'diag_state':      diag_state,
                      'class_label_id':  batch[8],
                      'suppress_dropout_passes': True}
            unique_ids = [features[i.item()].guid for i in batch[9]]
            values = [features[i.item()].values for i in batch[9]]
            input_ids_unmasked = [features[i.item()].input_ids_unmasked for i in batch[9]]
            inform = [features[i.item()].inform for i in batch[9]]
            outputs = model(**inputs)

            # Update dialog state for next turn.
            for slot in model.slot_list:
                updates = outputs[3][0][slot].max(1)[1]
                for i, u in enumerate(updates):
                    if u != 0:
                        diag_state[slot][i] = u

        results = eval_metric(model, inputs, outputs[0][0], outputs[1][0], outputs[3][0], outputs[5][0], outputs[6][0], outputs[8][0])
        preds, ds = predict_and_format(model, tokenizer, inputs, outputs[3][0], outputs[5][0], outputs[6][0], outputs[8][0], unique_ids, input_ids_unmasked, values, inform, prefix, ds)
        all_results.append(results)
        all_preds.append(preds)

    all_preds = [item for sublist in all_preds for item in sublist] # Flatten list

    # Generate final results
    final_results = {}
    for k in all_results[0].keys():
        final_results[k] = torch.stack([r[k] for r in all_results]).mean()

    # Write final predictions (for evaluation with external tool)
    output_prediction_file = os.path.join(args.output_dir, "pred_res.%s.%s.json" % (args.predict_type, prefix))
    with open(output_prediction_file, "w") as f:
        json.dump(all_preds, f, indent=2)

    return final_results


def load_and_cache_examples(args, model, tokenizer, processor, evaluate=False):
    # Load data features from cache or dataset file
    cached_file = os.path.join(os.path.dirname(args.output_dir), 'cached_{}_features'.format(
        args.predict_type if evaluate else 'train'))
    if os.path.exists(cached_file) and not args.overwrite_cache: # and not output_examples:
        logger.info("Loading features from cached file %s", cached_file)
        features = torch.load(cached_file)
    else:
        if args.task_name == "unified":
            logger.info("Creating features from unified data format")
        else:
            logger.info("Creating features from dataset file at %s", args.data_dir)
        processor_args = {'no_append_history': args.no_append_history,
                          'no_use_history_labels': args.no_use_history_labels,
                          'no_label_value_repetitions': args.no_label_value_repetitions,
                          'swap_utterances': args.swap_utterances,
                          'delexicalize_sys_utts': args.delexicalize_sys_utts,
                          'unk_token': '<unk>' if args.model_type == 'roberta' else '[UNK]'}
        if evaluate and args.predict_type == "dev":
            examples = processor.get_dev_examples(args.data_dir, processor_args)
        elif evaluate and args.predict_type == "test":
            examples = processor.get_test_examples(args.data_dir, processor_args)
        else:
            examples = processor.get_train_examples(args.data_dir, processor_args)
        features = convert_examples_to_features(examples=examples,
                                                slot_list=model.slot_list,
                                                class_types=model.class_types,
                                                model_type=args.model_type,
                                                tokenizer=tokenizer,
                                                max_seq_length=args.max_seq_length,
                                                slot_value_dropout=(0.0 if evaluate else args.svd))
        logger.info("Saving features into cached file %s", cached_file)
        torch.save(features, cached_file)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
    f_start_pos = [f.start_pos for f in features]
    f_end_pos = [f.end_pos for f in features]
    f_inform_slot_ids = [f.inform_slot for f in features]
    f_refer_ids = [f.refer_id for f in features]
    f_diag_state = [f.diag_state for f in features]
    f_class_label_ids = [f.class_label_id for f in features]
    all_start_positions = {}
    all_end_positions = {}
    all_inform_slot_ids = {}
    all_refer_ids = {}
    all_diag_state = {}
    all_class_label_ids = {}
    for s in model.slot_list:
        all_start_positions[s] = torch.tensor([f[s] for f in f_start_pos], dtype=torch.long)
        all_end_positions[s] = torch.tensor([f[s] for f in f_end_pos], dtype=torch.long)
        all_inform_slot_ids[s] = torch.tensor([f[s] for f in f_inform_slot_ids], dtype=torch.long)
        all_refer_ids[s] = torch.tensor([f[s] for f in f_refer_ids], dtype=torch.long)
        all_diag_state[s] = torch.tensor([f[s] for f in f_diag_state], dtype=torch.long)
        all_class_label_ids[s] = torch.tensor([f[s] for f in f_class_label_ids], dtype=torch.long)
    dataset = TensorListDataset(all_input_ids, all_input_mask, all_segment_ids,
                                all_start_positions, all_end_positions,
                                all_inform_slot_ids,
                                all_refer_ids,
                                all_diag_state,
                                all_class_label_ids, all_example_index)

    return dataset, features


def call_rescaler(args, model, slot, epoch_filter, step_results, kls, ovls, class_losses, batch):
    d_cat = int(not args.no_cat) * (2 - step_results['agreement'][slot].type(torch.int)) # 0, 1 or 2 (agree or disagree)

    rlbls = torch.full(batch['class_label_id'][slot].size(), 0) if args.no_class_separation else from_device(batch['class_label_id'][slot])
    if args.rescaler_feats == "default":
        feats = torch.cat((step_results['losses_means'][slot].reshape(-1,1),
                           epoch_filter.get_stats_tensor('loss_means_mean', slot)[rlbls, d_cat].reshape(-1,1),
                           epoch_filter.get_stats_tensor('loss_means_std', slot)[rlbls, d_cat].reshape(-1,1),
                           step_results['losses_stds'][slot].reshape(-1,1),
                           epoch_filter.get_stats_tensor('loss_stds_mean', slot)[rlbls, d_cat].reshape(-1,1),
                           epoch_filter.get_stats_tensor('loss_stds_std', slot)[rlbls, d_cat].reshape(-1,1),
                           step_results['probs_means'][slot].reshape(-1,1),
                           epoch_filter.get_stats_tensor('prob_means_mean', slot)[rlbls, d_cat].reshape(-1,1),
                           epoch_filter.get_stats_tensor('prob_means_std', slot)[rlbls, d_cat].reshape(-1,1),
                           step_results['probs_stds'][slot].reshape(-1,1),
                           epoch_filter.get_stats_tensor('prob_stds_mean', slot)[rlbls, d_cat].reshape(-1,1),
                           epoch_filter.get_stats_tensor('prob_stds_std', slot)[rlbls, d_cat].reshape(-1,1),
                           torch.tensor(kls[slot]).reshape(-1,1),
                           epoch_filter.get_stats_tensor('kl_mean', slot)[rlbls, d_cat].reshape(-1,1),
                           epoch_filter.get_stats_tensor('kl_std', slot)[rlbls, d_cat].reshape(-1,1),
                           torch.tensor(ovls[slot]).reshape(-1,1),
                           epoch_filter.get_stats_tensor('ovl_mean', slot)[rlbls, d_cat].reshape(-1,1),
                           epoch_filter.get_stats_tensor('ovl_std', slot)[rlbls, d_cat].reshape(-1,1)),
                          dim=1).type(torch.float).to(args.device)
    elif args.rescaler_feats == "loss":
        feats = class_losses[slot].reshape(-1,1).detach().clone() # Loss w/o grad
    else:
        raise Exception("Unknown meta filter features.")
    if not args.rescaler_feats_no_cat:
        feats = torch.cat((feats, step_results['agreement'][slot].unsqueeze(1).type(torch.float).to(args.device)), dim=1)
    rescaler_outputs = model(feats=feats, mode="rescaler")
    if not args.no_class_separation:
        rescaler_outputs = rescaler_outputs[0]
        new_rescaler_outputs = rescaler_outputs[0] * (step_results['labels'][slot] == 0).unsqueeze(1).to(args.device)
        for c in range(1, len(model.class_types)):
            new_rescaler_outputs += rescaler_outputs[c] * (step_results['labels'][slot] == c).unsqueeze(1).to(args.device)
        rescaler_outputs = (new_rescaler_outputs,)
    rescaler_outputs = rescaler_outputs[0][:,1]
    return rescaler_outputs


def main():
    args = parse_args()

    task_name = args.task_name.lower()
    if task_name not in PROCESSORS:
        raise ValueError("Task not found: %s" % (task_name))

    processor = PROCESSORS[task_name](args.dataset_config)
    dst_slot_list = processor.slot_list
    dst_class_types = processor.class_types
    dst_class_labels = len(dst_class_types)

    # Setup logging, print header
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO)
    logger.setLevel(logging.INFO)
    transformers.utils.logging.set_verbosity_info()

    print_header()

    for a in vars(args):
        logger.info('  {:40s} {:s}'.format(a, str(getattr(args, a))))
    logger.info("")

    # If passed along, set the training seed now.
    if args.seed is not None:
        transformers.set_seed(args.seed)

    # Handle the repository creation
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    args.model_type = args.model_type.lower()
    config_class, model_class, default_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    # Load pretrained model and tokenizer
    config = config_class.from_pretrained(args.model_name_or_path, local_files_only=args.local_files_only)

    # Add DST specific parameters to config
    config.dst_max_seq_length = args.max_seq_length
    config.dst_dropout_rate = args.dropout_rate
    config.dst_heads_dropout_rate = args.heads_dropout
    config.dst_class_loss_ratio = args.class_loss_ratio
    config.dst_token_loss_for_nonpointable = args.token_loss_for_nonpointable
    config.dst_refer_loss_for_nonpointable = args.refer_loss_for_nonpointable
    config.dst_class_aux_feats_inform = args.class_aux_feats_inform
    config.dst_class_aux_feats_ds = args.class_aux_feats_ds
    config.dst_slot_list = dst_slot_list
    config.dst_class_types = dst_class_types
    config.dst_class_labels = dst_class_labels

    # Add STORM specific parameters to config
    config.dropout_rounds = args.dropout_rounds
    config.no_class_separation = args.no_class_separation
    config.rescaler_binary = args.rescaler_binary
    config.rescaler_binary_threshold = args.rescaler_binary_threshold
    if args.rescaler_feats == "default":
        config.rescaler_featnum = 18
    elif args.rescaler_feats == "loss":
        config.rescaler_featnum = 1
    else:
        raise Exception("Unknown rescaler features.")
    if not args.rescaler_feats_no_cat:
        config.rescaler_featnum += 1

    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path,
                                                do_lower_case=args.do_lower_case,
                                                local_files_only=args.local_files_only)
    model = model_class.from_pretrained(args.model_name_or_path,
                                        from_tf=bool('.ckpt' in args.model_name_or_path),
                                        config=config,
                                        local_files_only=args.local_files_only)

    logger.info("Updated model config: %s" % config)

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        train_dataset, features = load_and_cache_examples(args, model, tokenizer, processor, evaluate=False)
        eval_dataset, _ = load_and_cache_examples(args, model, tokenizer, processor, evaluate=True)
        global_step, tr_loss = train(args, train_dataset, eval_dataset, features, model, tokenizer, processor)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

        if args.output_dir is not None and args.save_steps > 0:
            logger.info("Saving model checkpoint to %s", args.output_dir)
            model.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

    # Evaluation
    results = []
    if args.do_eval:
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN) # Reduce model loading logs

        logger.info("Evaluate the following checkpoints: %s", checkpoints)

        for cItr, checkpoint in enumerate(checkpoints):
            # Reload the model
            global_step = checkpoint.split('-')[-1]
            if cItr == len(checkpoints) - 1:
                global_step = "final"
            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)

            # Evaluate
            result = evaluate(args, model, tokenizer, processor, prefix=global_step)
            result_dict = {k: float(v) for k, v in result.items()}
            result_dict["global_step"] = global_step
            results.append(result_dict)

            for key in sorted(result_dict.keys()):
                logger.info("%s = %s", key, str(result_dict[key]))

        output_eval_file = os.path.join(args.output_dir, "eval_res.%s.json.gz" % (args.predict_type))
        with gzip.open(output_eval_file, "w") as f:
            f.write(json.dumps(results, indent=2).encode('utf-8'))

    return results


if __name__ == "__main__":
    main()

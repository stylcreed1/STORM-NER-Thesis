
# TEST: Just checking my workflow! 🚀
# # coding=utf-8
#
# Copyright 2024 Heinrich Heine University Duesseldorf
#
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

# 
from torch.utils.data import Dataset



import argparse
import autograd_hacks
import copy
import gzip
import higher
import json
import logging
import math
import os
import pickle
import random
import re
import torch
import transformers
import warnings

import numpy as np

from sklearn.metrics import (f1_score, matthews_corrcoef)
from tensorboardX import (SummaryWriter)
from torch.utils.data import (DataLoader, WeightedRandomSampler)
from tqdm.auto import (tqdm)
from transformers import (
    BertConfig, BertModel, BertTokenizer,
    RobertaConfig, RobertaModel, RobertaTokenizer,RobertaForTokenClassification,
    ElectraConfig, ElectraModel, ElectraTokenizer)

from agra import (AGRA)
from modeling import (TransformerForGlue)
from utils_gpu import (from_device, to_device)
from utils_storm import (print_header, Results, Filter)

warnings.filterwarnings("ignore", "Detected call of `lr_scheduler\.step\(\)` before `optimizer\.step\(\)`\.", UserWarning)

MODEL_CLASSES = {
    'bert': (BertConfig, TransformerForGlue('bert'), BertModel, BertTokenizer),
    'roberta': (RobertaConfig, TransformerForGlue('roberta'), RobertaModel, RobertaTokenizer),
    'electra': (ElectraConfig, TransformerForGlue('electra'), ElectraModel, ElectraTokenizer),
}

logger = logging.getLogger(__name__)


def parse_args():
    def list_of_strings(arg):
        return arg.split(',')
    
    parser = argparse.ArgumentParser(description="STORM for text classification tasks.")

    # Required parameters
    parser.add_argument("--task_name", type=str, default=None, required=True,
                        help="The name of the glue task to train on.")
    parser.add_argument("--model_type", type=str, default=None, required=True,
                        help="Model type.",
                        choices=list(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", type=str, default=None, required=True,
                        help="Path to pretrained model or model identifier from huggingface.co/models.")
    parser.add_argument("--train_file", type=str, default=None, required=True,
                        help="A tsv file containing the training data.")
    parser.add_argument("--validation_file", type=str, default=None, required=True,
                        help="A tsv file containing the validation data.")

    # Other parameters
    parser.add_argument("--test_file", type=str, default=None,
                        help="A tsv file containing the test data. Loading a test file overwrites --crossvalid_fold.")
    parser.add_argument("--max_length", type=int, default=128,
                        help="The maximum total input sequence length after tokenization. "
                             "Sequences longer than this will be truncated, sequences shorter "
                             "will be padded if `--pad_to_max_lengh` is passed.")
    parser.add_argument("--pad_to_max_length", action="store_true",
                        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.")
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
    parser.add_argument("--num_train_epochs", type=int, default=3,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_train_steps", type=int, default=None,
                        help="Total number of training steps to perform. If provided, overrides num_train_epochs.")
    parser.add_argument("--lr_scheduler_type", type=str, default="constant",
                        help="The scheduler type to use.",
                        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"])
    parser.add_argument("--warmup_proportion", type=float, default=0.0,
                        help="Proportion of steps for the warmup in the lr scheduler.")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=42,
                        help="A seed for reproducible training.")
    parser.add_argument('--local_files_only', action='store_true',
                        help="Whether to only load local model files (useful when working offline).")
    parser.add_argument('--logging_steps', type=int, default=100,
                        help="Log every X updates steps.")
    parser.add_argument("--save_checkpoints", action='store_true',
                        help="When set, saves model checkpoint after every epoch.")
    parser.add_argument("--save_stats", action='store_true',
                        help="When set, saves detailed training statistics as gzipped json.")
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
    parser.add_argument("--agra_layer_groups", type=list_of_strings, default='classifier',
                        help="Layer groups to consider for AGRA.")
    parser.add_argument("--use_tfidf", action='store_true',
                        help="When set, loads TF-IDF features instead of using a transformer encoder. Requires --tfidf_path.")
    parser.add_argument("--tfidf_path", type=str, default=None,
                        help="Path to TF-IDF features to be loaded when using --use_tdidf.")
    parser.add_argument("--evaluate_rescaling", action='store_true',
                        help="When set, evaluate rescaling performance given a dataset with corruption labels "
                              "that indicate whether the original label is noisy. "
                              "Note: You need to ensure that such labels exist in the data, "
                              "as the code does not assert this automatically.")
    parser.add_argument('--crossvalid_fold', type=int, default=-1,
                        help="When set to 0 or 1, uses 2-fold cross-validation given the --validation_file. "
                             "Usage: Run this script separately once for fold 0 and 1 "
                             "and then manually pool your results accordingly.",
                        choices=[-1, 0, 1])
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
    parser.add_argument("--use_class_weights", action="store_true", help="Enable mathematical class weighting")
    parser.add_argument("--use_entity_amplifier", action="store_true", help="Enable the entity density amplifier")
    parser.add_argument("--rescaler_lower_bound", type=float, default=0.0, help="Minimum gradient floor for the rescaler")
    parser.add_argument("--custom_weights", type=str, default="1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0", help="Comma separated mathematical weights")
    args = parser.parse_args()

    assert not args.use_tfidf or args.tfidf_path is not None

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return args

class DictDataset(Dataset):
    def __init__(self, dict_of_tensors):
        self.data = dict_of_tensors
        self.length = self.data['input_ids'].size(0)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        item = {key: tensor[idx] for key, tensor in self.data.items()}
        item['ids'] = torch.tensor(idx, dtype=torch.long)
        return item
    
    
    
def evaluate_task(pred, ref, metrics):
    results = {}
    if "accuracy" in metrics:
        results['accuracy'] = (np.array(pred) == np.array(ref)).mean()
    if "f1" in metrics:
        results['f1'] = f1_score(y_true=ref, y_pred=pred, average='macro')
    if "matthews_correlation" in metrics:
        results['matthews_correlation'] = matthews_corrcoef(y_true=ref, y_pred=pred)
    return results


def load_raw_dataset(input_file, data_specs, has_corruption_labels=False):
    def label_map(label, lbl_map):
        if lbl_map is not None:
            return lbl_map[label]
        else:
            return int(label)

    raw_data = []
    id_to_pos = {}
    unique_labels = set()
    corrupted_ids = []
    with open(input_file, "r", encoding='utf-8-sig') as f:
        for l_itr, line in enumerate(f):
            data_point = {'labels': None, 'sentence1': None, 'sentence2': None, 'ids': None}
            if data_specs['has_header'] and l_itr == 0:
                continue
            raw_data_point = line.strip().split('\t')
            data_point['labels'] = label_map(raw_data_point[data_specs['label']], data_specs['label_map'])
            data_point['sentence1'] = raw_data_point[data_specs['sentence1']]
            data_point['sentence2'] = raw_data_point[data_specs['sentence2']] if data_specs['sentence2'] is not None else None
            data_point['ids'] = l_itr - 1 if data_specs['has_header'] else l_itr
            id_to_pos[data_point['ids']] = len(raw_data)
            raw_data.append(data_point)
            if has_corruption_labels and raw_data_point[-1] == 'True':
                corrupted_ids.append(data_point['ids'])
            unique_labels.add(data_point['labels'])
    return raw_data, list(unique_labels), id_to_pos, corrupted_ids


def process_raw_dataset(dataset, tokenizer, padding, max_length):
    processed_dataset = []
    for i in tqdm(dataset, desc="Running tokenizer on dataset"):
        # Tokenize the texts
        text = ((i['sentence1'],) if i['sentence2'] is None else (i['sentence1'], i['sentence2']))
        result = tokenizer(*text, padding=padding, max_length=max_length, truncation=True)
        if 'ids' in i:
            result['ids'] = i['ids']
        if 'labels' in i:
            result['labels'] = i['labels']
        processed_dataset.append(result)
    return processed_dataset


def add_feats_to_raw_dataset(dataset, feats_path):
    feats_dict = pickle.load(open(feats_path, "rb"))
    for i in dataset:
        ids = str(i['ids'])
        if ids in feats_dict:
            i['feats'] = feats_dict[ids].tolist()
        else:
            raise Exception("Feats not found")


def evaluate_filtering(corrupted_ids, filter_list_by_epoch, epoch, data_size):
    prediction_size = len(filter_list_by_epoch[epoch])
    c_tp = 0
    c_fn = 0
    for i in corrupted_ids:
        if i in filter_list_by_epoch[epoch]:
            c_tp += 1
        else:
            c_fn += 1
    c_fp = prediction_size - c_tp
    c_tn = data_size - c_tp - c_fp - c_fn
    precision = c_tp / (c_tp + c_fp + 1e-8)
    recall = c_tp / (c_tp + c_fn + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    acc = (c_tp + c_tn) / (c_tp + c_fp + c_tn + c_fn + 1e-8)
    specificity = c_tn / (c_tn + c_fp + 1e-8)
    return precision, recall, f1, specificity, acc, c_tp, c_tn, c_fp, c_fn


def call_rescaler(args, model, epoch_filter, step_results, kls, ovls, per_example_loss, batch, num_labels, corrupted_ids):

    d_cat = (int(not args.no_cat) * (2 - step_results['agreement'].type(torch.int))).view(-1) # 0, 1 or 2
    rlbls = torch.zeros_like(d_cat, dtype=torch.long)
    if args.rescaler_feats == "default":
        feats = torch.cat((step_results['losses_means'].reshape(-1,1),
                           epoch_filter.get_stats_tensor('loss_means_mean')[rlbls, d_cat].reshape(-1,1),
                           epoch_filter.get_stats_tensor('loss_means_std')[rlbls, d_cat].reshape(-1,1),
                           step_results['losses_stds'].reshape(-1,1),
                           epoch_filter.get_stats_tensor('loss_stds_mean')[rlbls, d_cat].reshape(-1,1),
                           epoch_filter.get_stats_tensor('loss_stds_std')[rlbls, d_cat].reshape(-1,1),
                           step_results['probs_means'].reshape(-1,1),
                           epoch_filter.get_stats_tensor('prob_means_mean')[rlbls, d_cat].reshape(-1,1),
                           epoch_filter.get_stats_tensor('prob_means_std')[rlbls, d_cat].reshape(-1,1),
                           step_results['probs_stds'].reshape(-1,1),
                           epoch_filter.get_stats_tensor('prob_stds_mean')[rlbls, d_cat].reshape(-1,1),
                           epoch_filter.get_stats_tensor('prob_stds_std')[rlbls, d_cat].reshape(-1,1),
                           torch.tensor(kls).reshape(-1,1),
                           epoch_filter.get_stats_tensor('kl_mean')[rlbls, d_cat].reshape(-1,1),
                           epoch_filter.get_stats_tensor('kl_std')[rlbls, d_cat].reshape(-1,1),
                           torch.tensor(ovls).reshape(-1,1),
                           epoch_filter.get_stats_tensor('ovl_mean')[rlbls, d_cat].reshape(-1,1),
                           epoch_filter.get_stats_tensor('ovl_std')[rlbls, d_cat].reshape(-1,1)),
                          dim=1).type(torch.float).to(args.device)
    elif args.rescaler_feats == "loss":
        feats = per_example_loss.reshape(-1,1).detach().clone() # Loss w/o grad
    else:
        raise Exception("Unknown rescaler features.")
    if not args.rescaler_feats_no_cat:
        feats = torch.cat((feats, step_results['agreement'].unsqueeze(1).type(torch.float).to(args.device)), dim=1)
    rescaler_outputs = model(feats=feats, mode="rescaler")
    if not args.no_class_separation:
        rescaler_outputs = rescaler_outputs[0]
        new_rescaler_outputs = rescaler_outputs[0] * (step_results['labels'] == 0).unsqueeze(1).to(args.device)
        for c in range(1, num_labels):
            new_rescaler_outputs += rescaler_outputs[c] * (step_results['labels'] == c).unsqueeze(1).to(args.device)
        rescaler_outputs = (new_rescaler_outputs,)
    rescaler_outputs = rescaler_outputs[0][:,1]
    return rescaler_outputs


def main():
    args = parse_args()

    # Set up logging, print header
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                        datefmt="%m/%d/%Y %H:%M:%S",
                        level=logging.INFO)
    logger.setLevel(logging.INFO)
    transformers.utils.logging.set_verbosity_info()
    tb_writer = SummaryWriter()

    print_header()

    for a in vars(args):
        logger.info("{:40s} {:s}".format(a, str(getattr(args, a))))
    logger.info("")
        
    # If passed along, set the training seed now.
    if args.seed is not None:
        transformers.set_seed(args.seed)

    # Handle the repository creation
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    # Loading the datasets.
    data_specs = {'rte': {'has_header': True, 'label': 3, 'sentence1': 1, 'sentence2': 2, 'label_map': {'not_entailment': 0, 'entailment': 1}},
                  'mrpc': {'has_header': True, 'label': 0, 'sentence1': 3, 'sentence2': 4, 'label_map': None},
                  'cola': {'has_header': False, 'label': 1, 'sentence1': 3, 'sentence2': None, 'label_map': None},
                  'youtube': {'has_header': True, 'label': 2, 'sentence1': 1, 'sentence2': None, 'label_map': None},
                  'sms': {'has_header': True, 'label': 2, 'sentence1': 1, 'sentence2': None, 'label_map': None},
                  'noisebench': {'has_header': True, 'label': 1, 'sentence1': 0, 'sentence2': None, 'label_map': None}}
    # ==========================================
    # --- NER TENSOR LOADER ---
    # ==========================================
    logger.info("Loading pre-processed PyTorch datasets directly...")
    
    train_dict = torch.load(args.train_file)
    train_dataset = DictDataset(train_dict)
    
    eval_dict = torch.load(args.validation_file)
    eval_dataset = DictDataset(eval_dict)
    
    if args.test_file is not None:
        test_dict = torch.load(args.test_file)
        test_dataset = DictDataset(test_dict)
        
    noise_tensor = train_dict['noise_labels']
    
    corrupted_mask = (noise_tensor == 1).any(dim=1)
    
    corrupted_ids = corrupted_mask.nonzero(as_tuple=True)[0].tolist()
    num_labels = 9

    config_class, model_class, default_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    config = config_class.from_pretrained(args.model_name_or_path, num_labels=num_labels, finetuning_task=args.task_name, local_files_only=args.local_files_only)
    config.dropout_rounds = args.dropout_rounds
    config.use_tfidf = args.use_tfidf
    config.no_class_separation = args.no_class_separation
    config.rescaler_binary = args.rescaler_binary
    config.rescaler_binary_threshold = args.rescaler_binary_threshold
    config.use_class_weights = args.use_class_weights
    config.use_entity_amplifier = args.use_entity_amplifier
    config.rescaler_lower_bound = args.rescaler_lower_bound
    config.custom_weights = [float(w) for w in args.custom_weights.split(",")]
    if args.rescaler_feats == "default":
        config.rescaler_featnum = 18
    elif args.rescaler_feats == "loss":
        config.rescaler_featnum = 1
    else:
        raise Exception("Unknown rescaler features.")
    if not args.rescaler_feats_no_cat:
        config.rescaler_featnum += 1

    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path, local_files_only=args.local_files_only)
    config._attn_implementation = "eager"
    config.num_labels = 9
    # ==========================================

    model = model_class.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        local_files_only=args.local_files_only)

    # DataLoaders creation:
    if args.pad_to_max_length:
        data_collator = transformers.default_data_collator
    else:
        data_collator = transformers.DataCollatorWithPadding(tokenizer, pad_to_multiple_of=(None))

    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.eval_batch_size)
    if args.test_file is not None or args.crossvalid_fold > -1:
        test_dataloader = DataLoader(test_dataset, collate_fn=data_collator, batch_size=args.eval_batch_size)

    no_decay = ["bias", "LayerNorm.weight"]
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
    inner_optimizer = torch.optim.Adam(inner_optimizer_grouped_parameters, lr=args.learning_rate)
    meta_optimizer = torch.optim.Adam(outer_optimizer_grouped_parameters, lr=args.rescaler_learning_rate)

    model.to(args.device)

    num_update_steps_per_epoch = math.ceil(len(train_dataset) / args.train_batch_size)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch # t_total
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    lr_scheduler = transformers.get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=inner_optimizer,
        num_warmup_steps=int(args.max_train_steps * args.warmup_proportion),
        num_training_steps=args.max_train_steps,
    )

    if args.agra:
        agra = AGRA(args.agra_loss,
                    num_labels,
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
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    progress_bar = tqdm(range(args.max_train_steps))
    completed_steps = 0
    tr_loss, logging_loss = 0.0, 0.0

    stat_list = {}
    filter_list_by_epoch = {-1: []}
    for epoch in range(args.num_train_epochs):
        train_dataloader = DataLoader(
            train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.train_batch_size
        )

        train_inner_dataloader = DataLoader(
            train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.train_batch_size
        )

        # Meta dataloader
        if args.meta_dataset == "eval":
            meta_dataset = eval_dataset
        else:
            meta_dataset = train_dataset
        meta_sampler = WeightedRandomSampler(torch.ones(len(meta_dataset)), len(meta_dataset))
        meta_dataloader = DataLoader(
            meta_dataset, collate_fn=data_collator, batch_size=args.train_batch_size, sampler=meta_sampler)

        if args.agra:
            agra.build_dataloader(data_collator, args.train_batch_size)

        model.train()

        filter_list_by_epoch[epoch] = []
        num_filter_labels = 1 if args.no_class_separation else num_labels 
        epoch_filter = Filter(num_filter_labels,
                              args.stats_window,
                              args.train_batch_size)
        if not args.no_meta_loss_rescaling:
            meta_epoch_filter = Filter(num_filter_labels,
                                       args.stats_window,
                                       args.train_batch_size)
        for step, orig_batch in enumerate(train_dataloader):
            orig_batch = to_device(orig_batch, args.device)
            batch_size = orig_batch['input_ids'].size(0)

            meta_batch = next(iter(meta_dataloader))
            meta_batch = to_device(meta_batch, args.device)
            meta_optimizer.zero_grad()

            with higher.innerloop_ctx(model, inner_optimizer, copy_initial_weights=False, track_higher_grads=True) as (fmodel, diffopt):
                for ir in range(args.meta_innerloop_rounds):
                    if ir > 0:
                        batch = next(iter(train_inner_dataloader))
                        batch = to_device(batch, args.device)
                    else:
                        batch = orig_batch

                    if args.agra:
                       grad_scores = agra.agra_step(batch, args.agra_layer_groups)
                       
                    temp_noise = batch.pop("noise_labels", None)
                    outputs = fmodel(**batch)
                    bz = batch['labels'].size(0)
                    sentence_logits = outputs[1].mean(dim=2) 
                    outputs = (outputs[0].view(-1, bz), sentence_logits) + outputs[2:]
                    

                    per_example_loss = outputs[0] 
                    outputs = from_device(outputs)
                    
                    # --- DUMMY SENTENCE LABELS ---
                    dummy_labels = torch.zeros(bz, dtype=torch.long)
                    step_results = Results(outputs[0], outputs[1], dummy_labels)
                    if args.agra:
                        step_results['grad_scores'] = grad_scores

                    orig_loss = per_example_loss.sum().item()

                    epoch_filter.update_batch_stats(step_results)

                    kls, ovls = epoch_filter.get_sample_stats(batch,
                                                              step_results,
                                                              args.no_cat)
                    # ---------------------------
                    # Rescale sample losses
                    # ---------------------------

                    if args.agra:
                        # AGRA baseline
                        inner_rescaler_outputs = (grad_scores >= 0).float().to(args.device)
                    else:
                        # Get weights for each sample
                        inner_rescaler_outputs = call_rescaler(args, fmodel,
                                                               epoch_filter, step_results,
                                                               kls, ovls,
                                                               per_example_loss, batch,
                                                               num_labels, corrupted_ids)
                    if epoch < 1:
                        # Epoch 0: Let RoBERTa learn basic grammar (No filtering)
                        inner_rescaler_outputs = torch.ones_like(inner_rescaler_outputs)
                    else:
                        # force a minimum weight of 0.4 so the model never loses the clean tokens completely
                        lower_bound = args.rescaler_lower_bound
                        inner_rescaler_outputs = torch.clamp(inner_rescaler_outputs, min=lower_bound)
                    # Rescale
                    if not args.simulate_only:
                        per_example_loss *= inner_rescaler_outputs

                    # Get stats
                    if ir == 0:
                        filter_ids = batch['ids'][(inner_rescaler_outputs < 0.5).cpu()].tolist()
                        filter_list_by_epoch[epoch] += filter_ids

                        for l_itr in range(batch_size):
                            #Since a single sentence contains multiple NER classes,  bypassed class-separated tracking to prevent dictionary key crashes.
                            batch_id = batch['ids'][l_itr].item()
                            label = 0
                            d_cat = 0
                            rlbl = 0 

                            # Collect statistics
                            if batch_id not in stat_list:
                                stat_list[batch_id] = []
                            stat_list[batch_id].append(
                                {"is_filtered": batch_id in filter_ids,
                                 "epoch": epoch,
                                 "label": label,
                                 "weight": inner_rescaler_outputs[l_itr].item(),
                                 "guid": batch_id,
                                 "d_preds": step_results['preds'][:,l_itr].tolist(),
                                 "d_losses": step_results['losses'][:,l_itr].tolist(),
                                 "d_losses_mean": step_results['losses_means'][l_itr].item(),
                                 "d_losses_std": step_results['losses_stds'][l_itr].item(),
                                 "d_losses_mean_batch": epoch_filter.get_stats('loss_means_mean')[rlbl][d_cat],
                                 "d_losses_std_batch": epoch_filter.get_stats('loss_stds_mean')[rlbl][d_cat],
                                 "d_probs": step_results['probs'][:,l_itr].max(1)[0].tolist(),
                                 "d_probs_mean": step_results['probs_means'][l_itr].item(),
                                 "d_probs_std": step_results['probs_stds'][l_itr].item(),
                                 "grad_scores": step_results['grad_scores'][l_itr].item() if 'grad_scores' in step_results.examples else 0.0,
                                 "d_probs_mean_batch": epoch_filter.get_stats('prob_means_mean')[rlbl][d_cat],
                                 "d_probs_std_batch": epoch_filter.get_stats('prob_means_std')[rlbl][d_cat],
                                 "g_kl_div": kls[l_itr],
                                 "g_kl_div_batch": epoch_filter.get_stats('kl_mean')[rlbl][d_cat],
                                 "ovl": ovls[l_itr],
                                 "ovl_batch": epoch_filter.get_stats('ovl_mean')[rlbl][d_cat]})
                    loss = per_example_loss.mean()
                    tr_loss += loss.item()

                    diffopt.step(loss) # Perform a training step

                    if ir > 0:
                        batch = from_device(batch)

                # End of inner loop(s)

                # Meta update
                if not args.agra:
                    meta_outputs = fmodel(**meta_batch)
                    mbz = meta_batch['labels'].size(0)
                    
                    # --- NER SHAPE FIX ---
                    meta_sentence_logits = meta_outputs[1].mean(dim=2)
                    meta_outputs = (meta_outputs[0].view(-1, mbz), meta_sentence_logits) + meta_outputs[2:]
                    # ----------------------------
                    
                    meta_per_example_loss = meta_outputs[0] # This is the loss we use for backprop
                    meta_outputs = from_device(meta_outputs)

                    if not args.no_meta_loss_rescaling:
                        meta_bz = meta_batch['labels'].size(0)
                        
                        # Dummy labels for the meta batch
                        meta_dummy_labels = torch.zeros(meta_bz, dtype=torch.long)
                        
                        meta_step_results = Results(meta_outputs[0], meta_outputs[1], meta_dummy_labels)
                        
                        meta_epoch_filter.update_batch_stats(meta_step_results)
                        meta_kls, meta_ovls = meta_epoch_filter.get_sample_stats(meta_batch,
                                                                                 meta_step_results,
                                                                                 args.no_cat)
                        rescaler_outputs = call_rescaler(args, fmodel,
                                                         meta_epoch_filter, meta_step_results,
                                                         meta_kls, meta_ovls,
                                                         meta_per_example_loss, meta_batch,
                                                         num_labels, corrupted_ids)
                        if not args.simulate_only:
                            meta_per_example_loss *= rescaler_outputs

                    meta_loss = meta_per_example_loss.mean()
                    meta_loss.backward()

            # Meta update
            if not args.agra:
                meta_optimizer.step()
                meta_optimizer.zero_grad()

            # Update model parameters
            model.roberta.load_state_dict(fmodel.roberta.state_dict())
            model.classifier.load_state_dict(fmodel.classifier.state_dict())

            lr_scheduler.step()
            progress_bar.update(1)
            completed_steps += 1

            orig_batch = from_device(orig_batch)

            # Log metrics
            if tb_writer and args.logging_steps > 0 and completed_steps % args.logging_steps == 0:
                tb_writer.add_scalar('lr', lr_scheduler.get_last_lr()[0], completed_steps)
                tb_writer.add_scalar('loss', (tr_loss - logging_loss) / args.logging_steps, completed_steps)
                logging_loss = tr_loss

            if completed_steps >= args.max_train_steps:
                break

        model.eval()
        eval_metric = {"predictions": [], "references": []}
        for step, batch in enumerate(eval_dataloader):
            batch = to_device(batch, args.device)
            outputs = model(**batch, suppress_dropout_passes=True)
            predictions = outputs[1][0].argmax(dim=-1)
            active_mask = (batch["labels"] != -100)
            eval_metric["predictions"].extend(predictions[active_mask].tolist())
            eval_metric["references"].extend(batch["labels"][active_mask].tolist())
        if args.test_file is not None or args.crossvalid_fold > -1:
            test_metric = {"predictions": [], "references": []}
            for step, batch in enumerate(test_dataloader):
                batch = to_device(batch, args.device)
                outputs = model(**batch, suppress_dropout_passes=True)
                predictions = outputs[1][0].argmax(dim=-1)
                active_mask = (batch["labels"] != -100)
                test_metric["predictions"].extend(predictions[active_mask].tolist())
                test_metric["references"].extend(batch["labels"][active_mask].tolist())

        logger.info("")
        logger.info("Filtered samples (loss weight < 0.5) in epoch %s: %s of %s" % (epoch,
                                                                                    len(filter_list_by_epoch[epoch]),
                                                                                    len(train_dataset)))
        tb_writer.add_scalars('filtered_samples', {'Filtered samples': len(filter_list_by_epoch[epoch]),
                                                   'Total': len(train_dataset)}, epoch)

        # Evaluate rescaling/filtering performance
        if args.evaluate_rescaling:
            (precision, recall,
             f1, specificity, acc,
             c_tp, c_tn, c_fp, c_fn) = evaluate_filtering(corrupted_ids, filter_list_by_epoch, epoch, len(train_dataset))
            logger.info("Filter performance in epoch %s: " \
                        "Precision %.4f, Recall %.4f, F1 %.4f, Accuracy %.4f, " \
                        "Specificity: %.4f (TP: %s, TN: %s, FP: %s, FN: %s)" % (epoch, precision,
                                                                                recall, f1,
                                                                                acc, specificity,
                                                                                c_tp, c_tn, c_fp, c_fn))
            tb_writer.add_scalars('filter_performance', {'Precision': precision,
                                                         'Recall': recall,
                                                         'F1': f1,
                                                         'Specificity': specificity,
                                                         'Accuracy': acc}, epoch)

        # Evaluate task performance
        eval_result = evaluate_task(eval_metric['predictions'], eval_metric['references'], ["accuracy", "f1", "matthews_correlation"])
        logger.info(f"eval epoch {epoch}: {eval_result}")
        tb_writer.add_scalars('eval_metric', eval_result, epoch)
        if args.test_file is not None or args.crossvalid_fold > -1:
            test_result = evaluate_task(test_metric['predictions'], test_metric['references'], ["accuracy", "f1", "matthews_correlation"])
            logger.info(f"test epoch {epoch}: {test_result}")
            tb_writer.add_scalars('test_metric', test_result, epoch)

        # Save model checkpoint
        if args.output_dir is not None and args.save_checkpoints:
            output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(completed_steps))
            logger.info("Saving model checkpoint to %s", output_dir)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            model.save_pretrained(output_dir)

    # After training, print detailed statistics
    cc_stats = {e: {'cnt': 1e-8, 'filtered': 0} for e in range(args.num_train_epochs)}
    cw_stats = copy.deepcopy(cc_stats)
    wc_stats = copy.deepcopy(cc_stats)
    ww_stats = copy.deepcopy(cc_stats)
    for s in stat_list:
        # Reformat stats
        stats = {k: None for k in stat_list[s][0].keys()}
        for k in stat_list[s][0]:
            if k in ["epoch", "is_filtered", "label", "d_preds"]:
                stats[k] = [i[k] for i in stat_list[s]]
            elif k in ["d_losses", "d_probs"]:
                stats[k] = [["%.4f" % (j) for j in i[k]] for i in stat_list[s]]
            else:
                stats[k] = ["%.4f" % (i[k]) for i in stat_list[s]]
        stats['filtered_epochs'] = [i_itr for i_itr, i in enumerate(stat_list[s]) if i['is_filtered']]

        # Collect global stats
        for ep in stats['epoch']:
            majority_pred = max(set(stats['d_preds'][ep]), key=stats['d_preds'][ep].count)
            majority_pred_count = stats['d_preds'][ep].count(majority_pred)
            # In case multiple classes have the majority count, take the class of the first prediction
            if stats['d_preds'][ep].count(stats['d_preds'][ep][0]) == majority_pred_count:
                majority_pred = stats['d_preds'][ep][0]

            stat_dict = None
            if s not in corrupted_ids and majority_pred == stats['label'][0]:
                stat_dict = cc_stats
            elif s not in corrupted_ids and majority_pred != stats['label'][0]:
                stat_dict = cw_stats
            elif s in corrupted_ids and majority_pred != stats['label'][0]:
                stat_dict = wc_stats
            elif s in corrupted_ids and majority_pred == stats['label'][0]:
                stat_dict = ww_stats
            stat_dict[ep]['cnt'] += 1
            stat_dict[ep]['filtered'] += 1 if ep in stats['filtered_epochs'] else 0

    logger.info("")
    logger.info("Filtered samples (loss weight < 0.5) per epoch by label-prediction agreement:")
    for e in range(args.num_train_epochs):
        logger.info("%d "
                    "cc: %d of %d, cw: %d of %d, "
                    "wc: %d of %d, ww: %d of %d" % (e,
                                                    cc_stats[e]['filtered'], cc_stats[e]['cnt'],
                                                    cw_stats[e]['filtered'], cw_stats[e]['cnt'],
                                                    wc_stats[e]['filtered'], wc_stats[e]['cnt'],
                                                    ww_stats[e]['filtered'], ww_stats[e]['cnt']))

    if tb_writer:
        tb_writer.close()

    if args.output_dir is not None and args.save_checkpoints:
        logger.info("Saving model checkpoint to %s", args.output_dir)
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
    torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

    if args.output_dir is not None and args.save_stats:
        with gzip.open(os.path.join(args.output_dir, "stat_list.json.gz"), "w") as f:
            f.write(json.dumps(stat_list, indent=2).encode('utf-8'))


if __name__ == "__main__":
    # FORCE DISABLE FLASH ATTENTION ---
    try:
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
            main()
    except:
        main()



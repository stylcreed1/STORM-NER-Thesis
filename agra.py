# coding=utf-8
#
# Copyright 2024 Heinrich Heine University Duesseldorf
#
# Part of this code is based on the source code of AGRA
# (arXiv:2306.04502)
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

import autograd_hacks
import torch

import numpy as np

from torch.utils.data import (DataLoader, WeightedRandomSampler)

from utils_gpu import (to_device)


class F1Loss:
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, predictions, labels):
        softmax = torch.nn.Softmax(dim=1)
        all_preds = softmax(predictions)

        if self.num_classes == 2:
            preds = all_preds[:, 1]

            tp = torch.sum(preds * labels)
            fp = torch.sum(preds * (1 - labels))
            fn = torch.sum((1 - preds) * labels)

            f1_loss = 1 - ( (2 * tp) / (2 * tp + fn + fp + 1e-8) ) # tp + fn = number positive labels
        elif self.num_classes > 2:
            f1 = torch.zeros(self.num_classes)

            for label in range(0, self.num_classes):
                labels_bin = copy.deepcopy(labels)
                labels_bin = torch.where(labels_bin == label, 1, 0)

                tp = torch.sum(all_preds[:, label] * labels_bin)
                fp = torch.sum(all_preds[:, label] * (1 - labels_bin))
                fn = torch.sum((1 - all_preds[:, label]) * labels_bin)

                f1[label] = (2 * tp) / (2 * tp + fn + fp + 1e-8)

            f1_loss = 1 - torch.mean(f1) # define loss as 1 - macro F1
        else:
            raise ValueError("Invalid number of classes")

        return f1_loss


class AGRA:
    def __init__(self, comp_loss, num_classes, is_weighted, dataset, model, device, window_size=1):
        self.stats = {}
        self.window_size = window_size # window_size=0 -> unlimited window size
        self.comp_loss = comp_loss
        self.num_classes = num_classes
        self.model = model
        self.device = device
        self.dataset = dataset

        autograd_hacks.add_hooks(self.model)

        self.agra_weights = torch.ones(len(self.dataset))
        if is_weighted:
            self.agra_weights = torch.tensor(self._compute_weights([i['labels'] for i in self.dataset]))


    def _compute_weights(self, train_labels):
        num_samples = len(train_labels)
        _, counts = np.unique(train_labels, return_counts=True)
        assert sum(counts) == num_samples
        weights = np.zeros(num_samples)
        for label in range(0, self.num_classes):
            weights[np.array(train_labels) == label] = 1 / counts[label]
        return weights


    def _get_loss(self):
        if self.comp_loss == 'F1':
            comp_loss = F1Loss(self.num_classes)
            loss_type = 'sum'
        else:
            comp_loss = torch.nn.CrossEntropyLoss(reduction='mean')
            loss_type = 'mean'
        return comp_loss, loss_type


    def _get_comp_grads(self):
        comp_grads = [params[1].grad.reshape(-1).detach().clone().cpu()
                      for params in self.model.named_parameters()
                      if hasattr(params[1], 'grad1') and 'bias' not in params[0]]
        comp_grads = torch.cat(comp_grads).numpy()
        if 'comp_grads' not in self.stats:
            self.stats['comp_grads'] = []
        self.stats['comp_grads'].append(comp_grads)
        self.stats['comp_grads'] = self.stats['comp_grads'][-1 * self.window_size:]
        return np.mean(self.stats['comp_grads'], 0)


    def build_dataloader(self, data_collator, batch_size):
        comp_sampler = WeightedRandomSampler(self.agra_weights, len(self.dataset))
        self.comp_dataloader = DataLoader(
            self.dataset, collate_fn=data_collator, batch_size=batch_size, sampler=comp_sampler)
        self.comp_dataloader = iter(self.comp_dataloader)


    def agra_step(self, batch, agra_layer_groups=['classifier']):
        agra_crit, agra_loss_type = self._get_loss()
        batch_size = batch['input_ids'].size(0)

        # Get comparison gradients
        comp_batch = next(self.comp_dataloader)
        autograd_hacks.clear_grad1(self.model)
        self.model.classifier.weight.retain_grad() # required for _get_comp_grads()
        comp_batch = to_device(comp_batch, self.device)
        comp_outputs = self.model(**comp_batch, suppress_dropout_passes=True)
        comp_loss = agra_crit(comp_outputs[1][0], comp_batch['labels']) # Logits vs. labels
        comp_loss.backward()
        autograd_hacks.compute_grad1(self.model, loss_type=agra_loss_type, layer_groups=agra_layer_groups)
        comp_grads = self._get_comp_grads()

        del self.model.classifier.weight.grad
        autograd_hacks.clear_grad1(self.model)

        # Get sample gradients
        outputs = self.model(**batch)
        labels = batch['labels']
        sample_loss = agra_crit(outputs[1][0], labels) # Logits vs. labels
        sample_loss.backward()
        autograd_hacks.compute_grad1(self.model, loss_type=agra_loss_type, layer_groups=agra_layer_groups)
        grads = [params[1].grad1.detach().clone().cpu() for params in self.model.named_parameters() if
                 hasattr(params[1], 'grad1') and 'bias' not in params[0]]

        # Get gradient scores
        grad_scores = np.zeros(batch_size)
        for l_itr in range(batch_size):
            sample_grads = torch.cat([grad[l_itr].reshape(-1) for grad in grads]).numpy()
            grad_scores[l_itr] = np.sum(sample_grads * comp_grads) / ((np.linalg.norm(sample_grads) * np.linalg.norm(comp_grads)) + 1e-8)

        autograd_hacks.clear_grad1(self.model)

        return torch.tensor(grad_scores)
    

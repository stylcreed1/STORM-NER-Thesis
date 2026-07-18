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
import os
import torch
import sys

import numpy as np

from torch.utils.data import (DataLoader, WeightedRandomSampler)

from agra import (AGRA)
from utils_gpu import (to_device)
from dst.utils_storm_dst import (batch_to_dict)


class AGRA(AGRA):
    # Implementation is limited to the slot gates.
    def _get_comp_grads(self):
        comp_grads = {params[0]: params[1].grad.reshape(-1).detach().clone().cpu()
                      for params in self.model.named_parameters()
                      if hasattr(params[1], 'grad1') and 'bias' not in params[0]}
        if 'comp_grads' not in self.stats:
            self.stats['comp_grads'] = {}
        mean_comp_grads = {}
        for slot in self.model.slot_list:
            if slot not in self.stats['comp_grads']:
                self.stats['comp_grads'][slot] = []
            self.stats['comp_grads'][slot].append(comp_grads['class_' + slot + '.weight'].numpy())
            self.stats['comp_grads'][slot] = self.stats['comp_grads'][slot][-1 * self.window_size:]
            mean_comp_grads[slot] = np.mean(self.stats['comp_grads'][slot], 0)
        return mean_comp_grads


    def build_dataloader(self, batch_size):
        comp_sampler = WeightedRandomSampler(self.agra_weights, len(self.dataset))
        self.comp_dataloader = DataLoader(
            self.dataset, sampler=comp_sampler, batch_size=batch_size, drop_last=True)
        self.comp_dataloader = iter(self.comp_dataloader)


    # Implementation is limited to the slot gates.
    def agra_step(self, batch):
        batch_size = batch['input_ids'].size(0)

        # Get comparison gradients
        comp_batch = next(self.comp_dataloader)
        autograd_hacks.clear_grad1(self.model)
        for slot in self.model.slot_list:
            getattr(self.model, "class_" + slot).weight.retain_grad() # required for _get_comp_grads()
        comp_batch = to_device(batch_to_dict(comp_batch), self.device)
        comp_outputs = self.model(**comp_batch, suppress_dropout_passes=True)
        comp_loss = comp_outputs[0][0]
        comp_loss.backward()
        autograd_hacks.compute_grad1(self.model, loss_type="sum", layer_groups=["class_"])
        comp_grads = self._get_comp_grads()

        for slot in self.model.slot_list:
            del getattr(self.model, "class_" + slot).weight.grad
        autograd_hacks.clear_grad1(self.model)

        # Get sample gradients
        outputs = self.model(**batch)
        sample_loss = outputs[0][0]
        sample_loss.backward()
        autograd_hacks.compute_grad1(self.model, loss_type="sum", layer_groups=["class_"])
        grads = {params[0]: params[1].grad1.detach().clone().cpu() for params in self.model.named_parameters() if
                 hasattr(params[1], 'grad1') and 'bias' not in params[0]}

        # Get gradient scores
        grad_scores = {}
        for slot in self.model.slot_list:
            grad_scores[slot] = np.zeros(batch_size)
            for l_itr in range(batch_size):
                sample_grads = grads["class_" + slot + ".weight"][l_itr].reshape(-1).numpy()
                grad_scores[slot][l_itr] = np.sum(sample_grads * comp_grads[slot]) / ((np.linalg.norm(sample_grads) * np.linalg.norm(comp_grads[slot])) + 1e-8)
            grad_scores[slot] = torch.tensor(grad_scores[slot])

        autograd_hacks.clear_grad1(self.model)

        return grad_scores
    

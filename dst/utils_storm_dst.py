# coding=utf-8
#
# Copyright 2024 Heinrich Heine University Duesseldorf
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

import logging
import torch

import numpy as np

from statistics import (NormalDist)

from utils_storm import (Results, Filter,
                         gaussian_KL)

logger = logging.getLogger(__name__)


def batch_to_dict(batch):
    assert len(batch) == 10
    return  {'input_ids':       batch[0],
             'input_mask':      batch[1], 
             'segment_ids':     batch[2],
             'start_pos':       batch[3],
             'end_pos':         batch[4],
             'inform_slot_id':  batch[5],
             'refer_id':        batch[6],
             'diag_state':      batch[7],
             'class_label_id':  batch[8],
             'ids':             batch[9]}


class Results(Results):
    def _restructure(self, data):
        restructured_data = {slot: [] for slot in data[0]}
        for d in data:
            for s in d:
                restructured_data[s].append(d[s])
        for s in restructured_data:
            restructured_data[s] = torch.stack(restructured_data[s])
        return restructured_data


    def update(self, losses, logits, labels):
        self.examples['losses'] = self._restructure(losses)
        self.examples['logits'] = self._restructure(logits)
        self.examples['labels'] = labels
        self.examples['probs'] = {}
        self.examples['preds'] = {}

        for slot in self.examples['logits']:
            self.examples['probs'][slot] = torch.softmax(self.examples['logits'][slot].float(), dim=2)
            self.examples['preds'][slot] = torch.argmax(self.examples['logits'][slot], dim=2)

        self._update_agreement(labels)
        self._update_means()


    def _update_agreement(self, labels):
        self.examples['agreement'] = {}
        for slot in self.examples['preds']:
            agreement = self.examples['preds'][slot][0] == labels[slot] # [0] is prediction for which we backprop
            dropout_agreement_cnt = (self.examples['preds'][slot] == labels[slot]).sum(0)
            dropout_agreement = dropout_agreement_cnt >= self.examples['preds'][slot].size(0) / 2
            tie = dropout_agreement_cnt == self.examples['preds'][slot].size(0) / 2
            tie_idx = tie.nonzero(as_tuple=True)[0]
            dropout_agreement[tie_idx] = agreement[tie_idx]
            self.examples['agreement'][slot] = dropout_agreement


    def _update_means(self):
        self.examples['losses_means'] = {}
        self.examples['losses_stds'] = {}
        for slot in self.examples['losses']:
            self.examples['losses_means'][slot] = self.examples['losses'][slot].mean(0)
            self.examples['losses_stds'][slot] = self.examples['losses'][slot].std(0).nan_to_num()

        self.examples['probs_means'] = {}
        self.examples['probs_stds'] = {}
        for slot in self.examples['probs']:
            self.examples['probs_means'][slot] = self.examples['probs'][slot].max(2)[0].mean(0)
            self.examples['probs_stds'][slot] = self.examples['probs'][slot].max(2)[0].std(0).nan_to_num()


class Filter(Filter):
    def get_stats(self, name, slot):
        return self.stats[name][slot]


    def get_stats_tensor(self, name, slot):
        return torch.tensor(list(self.stats[name][slot].values()))


    def _append_new_stats(self, stats):
        for e in stats.examples:
            for slot in stats[e]:
                if stats[e][slot].dim() == 1:
                    if e not in self.stats:
                        self.stats[e] = {}
                    if slot not in self.stats[e]:
                        self.stats[e][slot] = {}
                    for l in range(self.num_labels):
                        if l not in self.stats[e][slot] or self.window_size == 0:
                            self.stats[e][slot][l] = torch.tensor([], dtype=stats[e][slot].dtype)
                        if self.num_labels > 1:
                            self.stats[e][slot][l] = torch.cat((self.stats[e][slot][l], stats[e][slot][stats['labels'][slot] == l]))
                        else:
                            self.stats[e][slot][l] = torch.cat((self.stats[e][slot][l], stats[e][slot]))
                        if self.window_size > 0:
                            self.stats[e][slot][l] = self.stats[e][slot][l][-1 * self.window_size * self.batch_size:] # sliding window


    def update_batch_stats(self, stats):
        self._append_new_stats(stats)

        self.stats['loss_means_mean'] = {}
        self.stats['loss_means_std'] = {}
        self.stats['loss_stds_mean'] = {}
        self.stats['loss_stds_std'] = {}
        self.stats['prob_means_mean'] = {}
        self.stats['prob_means_std'] = {}
        self.stats['prob_stds_mean'] = {}
        self.stats['prob_stds_std'] = {}
        self.stats['kl'] = {}
        self.stats['kl_mean'] = {}
        self.stats['kl_std'] = {}
        self.stats['ovl'] = {}
        self.stats['ovl_mean'] = {}
        self.stats['ovl_std'] = {}
        if 'grad_scores' in stats.examples:
            self.stats['grad_scores_mean'] = {}
            self.stats['grad_scores_std'] = {}
        for slot in self.stats['agreement']:
            self.stats['loss_means_mean'][slot] = {}
            self.stats['loss_means_std'][slot] = {}
            self.stats['loss_stds_mean'][slot] = {}
            self.stats['loss_stds_std'][slot] = {}
            self.stats['prob_means_mean'][slot] = {}
            self.stats['prob_means_std'][slot] = {}
            self.stats['prob_stds_mean'][slot] = {}
            self.stats['prob_stds_std'][slot] = {}
            self.stats['kl'][slot] = {}
            self.stats['kl_mean'][slot] = {}
            self.stats['kl_std'][slot] = {}
            self.stats['ovl'][slot] = {}
            self.stats['ovl_mean'][slot] = {}
            self.stats['ovl_std'][slot] = {}
            if 'grad_scores' in stats.examples:
                self.stats['grad_scores_mean'][slot] = {}
                self.stats['grad_scores_std'][slot] = {}
            for l in range(self.num_labels):
                # Get batch loss statistics, separate by sample, then accumulated
                (self.stats['loss_means_mean'][slot][l],
                 self.stats['loss_means_std'][slot][l],
                 self.stats['loss_stds_mean'][slot][l],
                 self.stats['loss_stds_std'][slot][l]) = self._update_batch_stats(self.stats['losses_means'][slot][l],
                                                                                  self.stats['losses_stds'][slot][l],
                                                                                  self.stats['agreement'][slot][l])
                
                # Get batch probability statistics, separate by sample, then accumulated
                (self.stats['prob_means_mean'][slot][l],
                 self.stats['prob_means_std'][slot][l],
                 self.stats['prob_stds_mean'][slot][l],
                 self.stats['prob_stds_std'][slot][l]) = self._update_batch_stats(self.stats['probs_means'][slot][l],
                                                                                  self.stats['probs_stds'][slot][l],
                                                                                  self.stats['agreement'][slot][l])

                # Get batch loss distribution KL divergence statistics
                (self.stats['kl'][slot][l],
                 self.stats['kl_mean'][slot][l],
                 self.stats['kl_std'][slot][l]) = self._update_batch_kl(self.stats['losses_means'][slot][l],
                                                                        self.stats['losses_stds'][slot][l],
                                                                        self.stats['loss_means_mean'][slot][l],
                                                                        self.stats['loss_stds_mean'][slot][l],
                                                                        self.stats['agreement'][slot][l],
                                                                        mode="kl")

                # Get batch loss distribution overlap statistics
                (self.stats['ovl'][slot][l],
                 self.stats['ovl_mean'][slot][l],
                 self.stats['ovl_std'][slot][l]) = self._update_batch_kl(self.stats['losses_means'][slot][l],
                                                                         self.stats['losses_stds'][slot][l],
                                                                         self.stats['loss_means_mean'][slot][l],
                                                                         self.stats['loss_stds_mean'][slot][l],
                                                                         self.stats['agreement'][slot][l],
                                                                         mode="ovl")

                # Get grad statistics
                if 'grad_scores' in stats.examples:
                    (self.stats['grad_scores_mean'][slot][l],
                     self.stats['grad_scores_std'][slot][l]) = self._update_global_batch_stats(self.stats['grad_scores'][slot][l],
                                                                                      self.stats['agreement'][slot][l])


    def get_sample_stats(self, batch, stats, no_cat=False):
        batch_size = batch['input_ids'].size(0)
        kls = {slot: [] for slot in stats['agreement']}
        ovls = {slot: [] for slot in stats['agreement']}
        for l_itr in range(batch_size):
            for slot in stats['agreement']:
                d_cat = int(not no_cat) * (2 - int(stats['agreement'][slot][l_itr])) # 0, 1 or 2 (agree or disagree)

                if self.num_labels > 1:
                    rlbl = batch['class_label_id'][slot][l_itr].item()
                else:
                    rlbl = 0

                d_losses_mean = stats['losses_means'][slot][l_itr].item()
                d_losses_std = stats['losses_stds'][slot][l_itr].item()
                d_probs_mean = stats['probs_means'][slot][l_itr].item()
                d_probs_std = stats['probs_stds'][slot][l_itr].item()
                if 'grad_scores' in stats.examples:
                    d_grad_scores = stats['grad_scores'][slot][l_itr].item()

                g_kl_div = gaussian_KL(d_losses_mean,
                                       d_losses_std,
                                       self.stats['loss_means_mean'][slot][rlbl][d_cat],
                                       self.stats['loss_stds_mean'][slot][rlbl][d_cat])
                ovl = NormalDist(mu=d_losses_mean,
                                 sigma=max(d_losses_std, self.eps)).overlap(
                                     NormalDist(mu=self.stats['loss_means_mean'][slot][rlbl][d_cat],
                                                sigma=max(self.stats['loss_stds_mean'][slot][rlbl][d_cat], self.eps)))

                kls[slot].append(g_kl_div)
                ovls[slot].append(ovl)
        return kls, ovls

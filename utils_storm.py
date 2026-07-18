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

logger = logging.getLogger(__name__)


def print_header():
    logger.info(" ________  _________  ________  ________  _____ ______       ")
    logger.info("|\   ____\|\___   ___\\\   __  \|\   __  \|\   _ \  _   \     ")
    logger.info("\ \  \___|\|___ \  \_\ \  \|\  \ \  \|\  \ \  \\\\\__\ \  \    ")
    logger.info(" \ \_____  \   \ \  \ \ \  \\\\\  \ \   _  _\ \  \\\|__| \  \   ")
    logger.info("  \|____|\  \   \ \  \ \ \  \\\\\  \ \  \\\  \\\ \  \    \ \  \  ")
    logger.info("    ____\_\  \   \ \__\ \ \_______\ \__\\\ _\\\ \__\    \ \__\ ")
    logger.info("   |\_________\   \|__|  \|_______|\|__|\|__|\|__|     \|__| ")
    logger.info("   \|_________|    (c) 2024 Heinrich Heine University        ")
    logger.info("")


def gaussian_KL(P_mu, P_std, Q_mu, Q_std):
    if P_std == 0 or Q_std == 0:
        return 0.0
    x = np.log(Q_std / P_std)
    x += ( pow(P_std, 2) + pow(P_mu - Q_mu, 2) ) / ( 2 * pow(Q_std, 2))
    x -= 0.5
    return x


class Results:
    def __init__(self, losses=None, logits=None, labels=None):
        self.examples = {}
        if losses is not None and logits is not None and labels is not None:
            self.update(losses, logits, labels)


    def __getitem__(self, idx):
        return self.examples[idx]


    def __setitem__(self, idx, item):
        self.examples[idx] = item


    def __repr__(self):
        result = ""
        for e in self.examples:
            result += "%s: %s\n" % (e, self.examples[e])
        return result


    def update(self, losses, logits, labels):
        if logits.dim() == 4: # If shape is [batch, seq_len, classes]
            logits = logits.view(1, -1, logits.size(-1)) 
            labels = labels.view(-1) 
            
        if losses.dim() == 0: # If loss is a single number
            losses = losses.view(1, -1).expand(1, labels.size(0))
        # ----------------------------
        self.examples['losses'] = losses
        self.examples['logits'] = logits
        self.examples['labels'] = labels
        self.examples['probs'] = torch.softmax(self.examples['logits'], dim=2)
        self.examples['preds'] = torch.argmax(self.examples['logits'], dim=2)

        self._update_agreement(labels)
        self._update_means()


    def _update_agreement(self, labels):
        agreement = self.examples['preds'][0] == labels # [0] is prediction for which we backprop
        dropout_agreement_cnt = (self.examples['preds'] == labels).sum(0)
        dropout_agreement = dropout_agreement_cnt >= self.examples['preds'].size(0) / 2
        tie = dropout_agreement_cnt == self.examples['preds'].size(0) / 2
        tie_idx = tie.nonzero(as_tuple=True)[0]
        dropout_agreement[tie_idx] = agreement[tie_idx]
        self.examples['agreement'] = dropout_agreement


    def _update_means(self):
        self.examples['losses_means'] = self.examples['losses'].mean(0)
        self.examples['losses_stds'] = self.examples['losses'].std(0).nan_to_num()
        self.examples['probs_means'] = self.examples['probs'].max(2)[0].mean(0)
        self.examples['probs_stds'] = self.examples['probs'].max(2)[0].std(0).nan_to_num()

        
class Filter():
    def __init__(self, num_labels, window_size=1, batch_size=1):
        self.num_labels = num_labels 
        self.cats = 3
        self.stats = {}
        self.eps = 1e-8
        self.window_size = window_size # window_size=0 -> unlimited window size
        self.batch_size = batch_size


    def get_stats(self, name):
        return self.stats[name]


    def get_stats_tensor(self, name):
        return torch.tensor(list(self.stats[name].values()))


    def __len__(self):
        return self.cats


    def _append_new_stats(self, stats):
        for e in stats.examples:
            if stats[e].dim() == 1:
                if e not in self.stats:
                    self.stats[e] = {}
                for l in range(self.num_labels):
                    if l not in self.stats[e] or self.window_size == 0:
                        self.stats[e][l] = torch.tensor([], dtype=stats[e].dtype)
                    if self.num_labels > 1:
                        self.stats[e][l] = torch.cat((self.stats[e][l], stats[e][stats['labels'] == l]))
                    else:
                        self.stats[e][l] = torch.cat((self.stats[e][l], stats[e]))
                    if self.window_size > 0:
                        self.stats[e][l] = self.stats[e][l][-1 * self.window_size * self.batch_size:] # sliding window


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
        for l in range(self.num_labels):
            # Get batch loss statistics, separate by sample, then accumulated
            (self.stats['loss_means_mean'][l],
             self.stats['loss_means_std'][l],
             self.stats['loss_stds_mean'][l],
             self.stats['loss_stds_std'][l]) = self._update_batch_stats(self.stats['losses_means'][l],
                                                                        self.stats['losses_stds'][l],
                                                                        self.stats['agreement'][l])

            # Get batch probability statistics, separate by sample, then accumulated
            (self.stats['prob_means_mean'][l],
             self.stats['prob_means_std'][l],
             self.stats['prob_stds_mean'][l],
             self.stats['prob_stds_std'][l]) = self._update_batch_stats(self.stats['probs_means'][l],
                                                                        self.stats['probs_stds'][l],
                                                                        self.stats['agreement'][l])

            # Get batch loss distribution KL divergence statistics
            (self.stats['kl'][l],
             self.stats['kl_mean'][l],
             self.stats['kl_std'][l]) = self._update_batch_kl(self.stats['losses_means'][l],
                                                              self.stats['losses_stds'][l],
                                                              self.stats['loss_means_mean'][l],
                                                              self.stats['loss_stds_mean'][l],
                                                              self.stats['agreement'][l],
                                                              mode="kl")

            # Get batch loss distribution overlap statistics
            (self.stats['ovl'][l],
             self.stats['ovl_mean'][l],
             self.stats['ovl_std'][l]) = self._update_batch_kl(self.stats['losses_means'][l],
                                                               self.stats['losses_stds'][l],
                                                               self.stats['loss_means_mean'][l],
                                                               self.stats['loss_stds_mean'][l],
                                                               self.stats['agreement'][l],
                                                               mode="ovl")

            # Get grad statistics
            if 'grad_scores' in stats.examples:
                (self.stats['grad_scores_mean'][l],
                 self.stats['grad_scores_std'][l]) = self._update_global_batch_stats(self.stats['grad_scores'][l],
                                                                                     self.stats['agreement'][l])


    def _update_batch_stats(self, stats_means, stats_stds, agreement):
        stats_means_mean = [None] * self.cats
        stats_means_std = [None] * self.cats
        stats_stds_mean = [None] * self.cats
        stats_stds_std = [None] * self.cats
        for cat in range(self.cats):
            cat_means = self._get_stats_by_agreement_cat(stats_means, cat, agreement)
            cat_stds = self._get_stats_by_agreement_cat(stats_stds, cat, agreement)
            stats_means_mean[cat] = cat_means.mean().tolist()
            stats_means_std[cat] = cat_means.std().tolist() if len(cat_means) > 1 else self.eps
            stats_stds_mean[cat] = cat_stds.mean().tolist()
            stats_stds_std[cat] = cat_stds.std().tolist() if len(cat_stds) > 1 else self.eps
        return (stats_means_mean, stats_means_std, stats_stds_mean, stats_stds_std)


    def _update_global_batch_stats(self, stats, agreement):
        stats_mean = [None] * self.cats
        stats_std = [None] * self.cats
        for cat in range(self.cats):
            cat_stats = self._get_stats_by_agreement_cat(stats, cat, agreement)
            stats_mean[cat] = cat_stats.mean().tolist()
            stats_std[cat] = cat_stats.std().tolist() if len(cat_stats) > 1 else self.eps
        return (stats_mean, stats_std)

    
    def _update_batch_kl(self, stats_means, stats_stds, batch_stats_means_mean, batch_stats_stds_mean, agreement, mode="kl"):
        g_mean = [None] * self.cats
        g_std = [None] * self.cats
        g = [[] for c in range(self.cats)]
        for cat in range(self.cats):
            cat_means = self._get_stats_by_agreement_cat(stats_means, cat, agreement)
            cat_stds = self._get_stats_by_agreement_cat(stats_stds, cat, agreement)
            for l_itr in range(len(cat_means)):
                if mode == "kl":
                    g[cat].append(gaussian_KL(cat_means[l_itr],
                                              cat_stds[l_itr],
                                              batch_stats_means_mean[cat],
                                              batch_stats_stds_mean[cat]))
                elif mode == "ovl":
                    g[cat].append(NormalDist(mu=cat_means[l_itr],
                                             sigma=max(cat_stds[l_itr], self.eps)).overlap(
                                                 NormalDist(mu=batch_stats_means_mean[cat],
                                                            sigma=max(batch_stats_stds_mean[cat], self.eps))))
                else:
                    raise Exception("Unknown mode for _update_batch_kl.")
            g_mean[cat] = torch.tensor(g[cat]).mean().tolist()
            g_std[cat] = torch.tensor(g[cat]).std().tolist() if len(g[cat]) > 1 else self.eps
        return (g, g_mean, g_std)

    
    def _get_stats_by_agreement_cat(self, stats, cat, agreement):
        if cat == 0:
            return stats
        elif cat == 1:
            if len(stats.size()) == 1:
                return stats[agreement.nonzero()].reshape(-1)
            else:
                return stats[:,agreement.nonzero()].reshape(-1)
        else:
            if len(stats.size()) == 1:
                return stats[(~agreement).nonzero()].reshape(-1)
            else:
                return stats[:,(~agreement).nonzero()].reshape(-1)


    def get_sample_stats(self, batch, stats, no_cat=False):
        batch_size = batch['input_ids'].size(0)
        kls = []
        ovls = []
        for l_itr in range(batch_size):
            d_cat = int(not no_cat) * (2 - int(stats['agreement'][l_itr])) # 0, 1 or 2 (agree or disagree)

            if self.num_labels > 1 and batch['labels'][l_itr].numel() == 1:
                rlbl = batch['labels'][l_itr].item()
            else:
                rlbl = 0
                
            d_losses_mean = stats['losses_means'][l_itr].item()
            d_losses_std = stats['losses_stds'][l_itr].item()
            d_probs_mean = stats['probs_means'][l_itr].item()
            d_probs_std = stats['probs_stds'][l_itr].item()
            if 'grad_scores' in stats.examples:
                d_grad_scores = stats['grad_scores'][l_itr].item()

            g_kl_div = gaussian_KL(d_losses_mean,
                                   d_losses_std,
                                   self.stats['loss_means_mean'][rlbl][d_cat],
                                   self.stats['loss_stds_mean'][rlbl][d_cat])
            ovl = NormalDist(mu=d_losses_mean,
                             sigma=max(d_losses_std, self.eps)).overlap(
                                 NormalDist(mu=self.stats['loss_means_mean'][rlbl][d_cat],
                                            sigma=max(self.stats['loss_stds_mean'][rlbl][d_cat], self.eps)))

            kls.append(g_kl_div)
            ovls.append(ovl)
        return kls, ovls

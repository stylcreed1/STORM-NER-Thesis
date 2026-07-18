# coding=utf-8
#
# Copyright 2024 Heinrich Heine University Duesseldorf
#
# Part of this code is based on the source code of TripPy
# (arXiv:2005.02877)
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

import torch

from torch import nn
from torch.nn import (CrossEntropyLoss)
from transformers import (BertModel, BertPreTrainedModel,
                          RobertaModel, RobertaPreTrainedModel,
                          ElectraModel, ElectraPreTrainedModel)

PARENT_CLASSES = {
    'bert': BertPreTrainedModel,
    'roberta': RobertaPreTrainedModel,
    'electra': ElectraPreTrainedModel
}

MODEL_CLASSES = {
    BertPreTrainedModel: BertModel,
    RobertaPreTrainedModel: RobertaModel,
    ElectraPreTrainedModel: ElectraModel
}


class STEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return (input >= 0.5).float()

    @staticmethod
    def backward(ctx, grad_output):
        return torch.nn.functional.hardtanh(grad_output)


class STEFunction01(STEFunction):
    @staticmethod
    def forward(ctx, input):
        return (input >= 0.1).float()


class STEFunction02(STEFunction):
    @staticmethod
    def forward(ctx, input):
        return (input >= 0.2).float()


class STEFunction03(STEFunction):
    @staticmethod
    def forward(ctx, input):
        return (input >= 0.3).float()


class STEFunction04(STEFunction):
    @staticmethod
    def forward(ctx, input):
        return (input >= 0.4).float()


class StraightThroughEstimator(nn.Module):
    def __init__(self, t):
        super(StraightThroughEstimator, self).__init__()
        self.t = t

    def forward(self, x):
        if self.t == 0.1:
            x = STEFunction01.apply(x)
        elif self.t == 0.2:
            x = STEFunction02.apply(x)
        elif self.t == 0.3:
            x = STEFunction03.apply(x)
        elif self.t == 0.4:
            x = STEFunction04.apply(x)
        else:
            x = STEFunction.apply(x)
        return x


class ElectraPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()
        
    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


def TransformerForGlue(parent_name):
    if parent_name not in PARENT_CLASSES:
        raise ValueError("Unknown model %s" % (parent_name))

    class TransformerForGlue(PARENT_CLASSES[parent_name]):
        def __init__(self, config):
            assert config.model_type in PARENT_CLASSES
            assert self.__class__.__bases__[0] in MODEL_CLASSES
            super(TransformerForGlue, self).__init__(config)
            self.model_type = config.model_type
            self.num_labels = config.num_labels
            self.dropout_rounds = config.dropout_rounds
            self.use_tfidf = config.use_tfidf
            self.tfidf_dim = config.tfidf_dim if hasattr(config, "tfidf_dim") else 0
            self.no_class_separation = config.no_class_separation
            self.rescaler_featnum = config.rescaler_featnum
            self.rescaler_binary = config.rescaler_binary
            self.rescaler_binary_threshold = config.rescaler_binary_threshold
            self.config = config

            self.add_module(self.model_type, MODEL_CLASSES[self.__class__.__bases__[0]](config))
            if self.model_type == "electra":
                self.pooler = ElectraPooler(config)

            classifier_dropout = (
                config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
            )
            self.dropout = nn.Dropout(classifier_dropout)

            if self.use_tfidf:
                self.classifier = nn.Linear(self.tfidf_dim, config.num_labels)
            else:
                self.classifier = nn.Linear(config.hidden_size, config.num_labels)

            if not self.no_class_separation:
                for c in range(self.num_labels):
                    self.add_module("rescaler_" + str(c), nn.Sequential(
                        nn.Linear(self.rescaler_featnum, self.rescaler_featnum),
                        nn.BatchNorm1d(self.rescaler_featnum, affine=False),
                        nn.ReLU(),
                        nn.Linear(self.rescaler_featnum, 2),
                        nn.BatchNorm1d(2, affine=False),
                        nn.Softmax(dim=1),
                    ))
            else:
                self.rescaler = nn.Sequential(
                    nn.Linear(self.rescaler_featnum, self.rescaler_featnum),
                    nn.BatchNorm1d(self.rescaler_featnum, affine=False),
                    nn.ReLU(),
                    nn.Linear(self.rescaler_featnum, 2),
                    nn.BatchNorm1d(2, affine=False),
                    nn.Softmax(dim=1),
                )
            if self.rescaler_binary:
                if not self.no_class_separation:
                    for c in range(self.num_labels):
                        getattr(self, "rescaler_" + str(c)).add_module(str(len(getattr(self, "rescaler_" + str(c))) + 1),
                                                                       StraightThroughEstimator(config.rescaler_binary_threshold))
                else:
                    self.rescaler.add_module(str(len(self.rescaler) + 1),
                                             StraightThroughEstimator(config.rescaler_binary_threshold))

            self.post_init()

        def forward(
            self,
            input_ids = None,
            attention_mask = None,
            token_type_ids = None,
            position_ids = None,
            head_mask = None,
            inputs_embeds = None,
            labels = None,
            ids = None,
            feats = None,
            output_attentions = None,
            output_hidden_states = None,
            no_grad = False,
            suppress_dropout_passes = False,
            mode = "default",**kwargs):
            if mode == "rescaler":
                if not self.no_class_separation:
                    logits = tuple([getattr(self, "rescaler_" + str(c))(feats) for c in range(self.num_labels)])
                else:
                    logits = self.rescaler(feats)
                return (logits,)

            dropout_rounds = 1
            if not suppress_dropout_passes:
                dropout_rounds += self.dropout_rounds

            dropout_losses = []
            dropout_logits = []

            for i in range(dropout_rounds):
                if i > 0:
                    torch.set_grad_enabled(False)

                outputs = getattr(self, self.model_type)(
                    input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    position_ids=position_ids,
                    head_mask=head_mask,
                    inputs_embeds=inputs_embeds,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states)
                sequence_output = outputs[0]
                cls_token_state = sequence_output[:, 0, :] 
                sequence_output = self.dropout(sequence_output)
                
                logits = self.classifier(sequence_output)
                if labels is not None:
                    loss_fct = CrossEntropyLoss(reduction='none')
                        
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                    loss_matrix = loss.view(labels.size(0), labels.size(1))
                        
                    active_loss_mask = (labels != -100).float()
                        
                    # ====================================================
                    # Class-Balanced Dictionary Weights
                    # ====================================================
                    safe_labels = torch.where(labels != -100, labels, torch.tensor(8, device=logits.device))                    
                    if getattr(self.config, "use_class_weights", False):
                            class_weights = torch.tensor(self.config.custom_weights, device=logits.device)
                            entity_weights = class_weights[safe_labels]
                    else:
                        entity_weights = torch.ones_like(safe_labels, dtype=torch.float, device=logits.device)
                        
                    weighted_loss_mask = active_loss_mask * entity_weights
                    loss_matrix = loss_matrix * weighted_loss_mask
                        
                    weighted_tokens_per_sentence = weighted_loss_mask.sum(dim=1).clamp(min=1)
                    sentence_loss = loss_matrix.sum(dim=1) / weighted_tokens_per_sentence
                        
                    # ====================================================
                    #  Entity Density Amplifier
                    # ====================================================
                    if getattr(self.config, "use_entity_amplifier", False):
                        entity_count = ((labels >= 0) & (labels < 8)).float().sum(dim=1)
                        
                        total_words = active_loss_mask.sum(dim=1).clamp(min=1)
                        entity_density = entity_count / total_words
                        
                        density_multiplier = 1.0 + (entity_density * 0.5)
                        sentence_loss = sentence_loss * density_multiplier

                    dropout_losses.append(sentence_loss)
                    
                dropout_logits.append(logits)
                if i > 0:
                    torch.set_grad_enabled(True)

            outputs_final = (torch.stack(dropout_losses) if labels is not None else None,
                             torch.stack(dropout_logits),) + outputs[2:]

            return outputs_final

    return TransformerForGlue

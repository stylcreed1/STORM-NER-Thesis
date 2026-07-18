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

from transformers.tokenization_utils_base import (BatchEncoding)


def from_device(batch):
    if isinstance(batch, tuple):
        batch_on_cpu = tuple([from_device(element) for element in batch])
    elif isinstance(batch, list):
        batch_on_cpu = [from_device(element) for element in batch]
    elif isinstance(batch, dict):
        batch_on_cpu = {k: from_device(v) for k, v in batch.items()}
    elif isinstance(batch, BatchEncoding):
        batch_on_cpu = {k: from_device(v) for k, v in batch.items()}
    else:
        batch_on_cpu = batch.detach().cpu() if batch is not None else batch
    return batch_on_cpu


def to_device(batch, device):
    if isinstance(batch, list):
        return to_device_list(batch, device)
    batch_on_device = {}
    for element in batch:
        if isinstance(batch[element], dict):
            batch_on_device[element] = {k: v.to(device) for k, v in batch[element].items()}
        else:
            batch_on_device[element] = batch[element].to(device)
    return batch_on_device


def to_device_list(batch, device):
    batch_on_device = []
    for element in batch:
        if isinstance(element, dict):
            batch_on_device.append({k: v.to(device) for k, v in element.items()})
        else:
            batch_on_device.append(element.to(device))
    return batch_on_device

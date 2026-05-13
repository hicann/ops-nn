#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import random
import torch

from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi

@register("ascend_apply_top_k_top_p")            
class AclnnIndexPut(BaseApi):  
    def __call__(self, input_data: InputDataset, with_output: bool = False):
        ori_dtype = input_data.kwargs['logits'].dtype

        logits = input_data.kwargs['logits']
        p = input_data.kwargs['p'].to(torch.float32)
        k = input_data.kwargs['k']

        logits_sort, logits_idx = logits.sort(dim=-1, descending=False, stable=True)
        kth_idx = logits_sort.size(1) - k.to(torch.long)
        kth_value = logits_sort.gather(1, kth_idx.unsqueeze(dim=1))
        top_k_mask = logits_sort < kth_value
        logits_sort.masked_fill_(top_k_mask, -float("inf"))

        softmax_res = logits_sort.to(torch.float32).softmax(dim=-1)
        cumsum_res = softmax_res.cumsum(dim=-1)
        top_p_mask = cumsum_res <= 1 - p.unsqueeze(dim=1)
        top_p_mask[:, -1] = False
        logits_sort.masked_fill_(top_p_mask, -float("inf"))

        logits = torch.empty_like(logits_sort).scatter_(dim=-1, index=logits_idx, src=logits_sort)
        
        return logits.to(ori_dtype)

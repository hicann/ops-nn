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
import torch

from atk.common.log import Logger
from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.configs.results_config import TaskResult
from atk.tasks.backends.lib_interface.acl_wrapper import AclFormat
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi

@register("aclnn_function")
class FunctionaclnnadvancestepV2(AclnnBaseApi):
    def init_by_input_data(self, input_data):
        input_args, output_packages = super().init_by_input_data(input_data)

        for _ in range(4):
            input_args.pop()
        output_packages[:] = [input_args[0], input_args[2], input_args[3], input_args[4]]
        return input_args, output_packages

@register("function_advance_step_v2")
class FunctionAdanceStep(BaseApi):
    def Advance_step_v2(self, num_seqs, num_queries, block_size, input_tokens, sampled_token_ids, 
                        input_positions, seq_lens, slot_mapping, block_tables, spec_token, accepted_num):
        token_each_reqs = 1 + len(spec_token[0])
        input_positions += torch.repeat_interleave(accepted_num, token_each_reqs) + 1
        seq_lens.copy_((input_positions + 1).to(seq_lens.dtype))
        index = torch.argmin(
            torch.cat([
                sampled_token_ids,
                torch.full((num_seqs, 1), -1, device=sampled_token_ids.device)
            ], dim=1),
            dim=1
        ) - 1
        last_tokens = sampled_token_ids[torch.arange(num_seqs), index]
        if token_each_reqs == 1:
            input_tokens[:num_seqs] = last_tokens.to(dtype=input_tokens.dtype)
        else:
            input_tokens_2d = input_tokens.view(-1, token_each_reqs)
            input_tokens_2d[:num_seqs, 0] = last_tokens
            input_tokens_2d[:num_seqs, 1:] = spec_token
        req_indices = torch.repeat_interleave(
            torch.arange(num_seqs),
            token_each_reqs,
            dim=0
        )
        max_num_blocks_per_req = block_tables.shape[1]
        block_tables_indices = (
            req_indices * max_num_blocks_per_req +
            input_positions // block_size
        )
        block_numbers = block_tables.flatten()[block_tables_indices]
        block_offset = input_positions % block_size
        slot_mapping.copy_(block_numbers * block_size + block_offset)
        return input_tokens, input_positions, seq_lens, slot_mapping

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        num_seqs = input_data.kwargs['numSeqs']
        num_queries = input_data.kwargs['numQueries']
        block_size = input_data.kwargs['blockSize']
        input_tokens = input_data.kwargs['inputTokens']
        sampled_token_ids = input_data.kwargs['sampledTokenIds']
        input_positions = input_data.kwargs['inputPositions']
        seq_lens = input_data.kwargs['seqLens']
        slot_mapping = input_data.kwargs['slotMapping']
        block_tables = input_data.kwargs['blockTables']
        spec_token = input_data.kwargs['specToken']
        accepted_num = input_data.kwargs['acceptedNum']
        input_tokens, input_positions, seq_lens, slot_mapping = self.Advance_step_v2(num_seqs, num_queries, block_size, input_tokens, sampled_token_ids, 
                        input_positions, seq_lens, slot_mapping, block_tables, spec_token, accepted_num)
        return input_tokens, input_positions, seq_lens, slot_mapping
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
class Functionaclnnadvancestep(AclnnBaseApi):
    def init_by_input_data(self, input_data):
        input_args, output_packages = super().init_by_input_data(input_data)

        for _ in range(4):
            input_args.pop()
        output_packages[:] = [input_args[0], input_args[2], input_args[3], input_args[4]]
        return input_args, output_packages

@register("function_advance_step")
class FunctionAdanceStep(BaseApi):
    def Advance_step(self,num_seqs, num_queries, block_size, input_tokens, sampled_token_ids, input_positions,
                     seq_lens, slot_mapping, block_tables):
        core_num = 40

        logging.info(f"****************************************input_tokens {input_tokens} ")
        logging.info(f"****************************************input_positions {input_positions} ")
        logging.info(f"****************************************seq_lens {seq_lens} ")
        logging.info(f"****************************************slot_mapping {slot_mapping} ")

        for blockIdx in range(core_num):
            n_pad = num_seqs - num_queries
            if n_pad > 0 and blockIdx == 0:
                offset = num_queries
                i = 0
                while i < n_pad:
                    input_tokens[offset + i] = 0
                    input_positions[offset + i] = 0
                    slot_mapping[offset + i] = -1
                    i += core_num
            num_query_blocks = num_queries // 1
            if blockIdx >= num_query_blocks:
                break
            cur_query_id = blockIdx * 1 + 0
            if cur_query_id >= num_queries:
                break
            input_tokens[cur_query_id] = sampled_token_ids[cur_query_id]
            seq_len = seq_lens[cur_query_id]
            next_seq_len = seq_len + 1
            next_input_pos = next_seq_len - 1
            seq_lens[cur_query_id] = next_seq_len
            input_positions[cur_query_id] = next_input_pos

            block_index = next_input_pos // block_size
            block_offset = next_input_pos % block_size
            logging.info(f"****************************************seq_lens  {seq_lens} ")
            logging.info(f"****************************************block_size  {block_size} ")
            logging.info(f"****************************************block_tables  {block_tables.shape} ")
            cur_block = block_tables.flatten()[block_index]

            slot_num = (cur_block + block_tables.stride(0) * cur_query_id) * block_size + block_offset
            slot_mapping[cur_query_id] = slot_num
            #tensor_list = [input_tokens[cur_query_id], input_positions[cur_query_id], seq_lens[cur_query_id], slot_mapping[cur_query_id]]
        return input_tokens, input_positions, seq_lens, slot_mapping

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        num_seqs = input_data.kwargs['numSeqs']
        num_queries = input_data.kwargs['numQueries']
        block_size = input_data.kwargs['blockSize']
        input_tokens = input_data.kwargs['inputTokensRef']
        sampled_token_ids = input_data.kwargs['sampledTokenIds']
        input_positions = input_data.kwargs['inputPositionsRef']
        seq_lens = input_data.kwargs['seqLensRef']
        slot_mapping = input_data.kwargs['slotMappingRef']
        block_tables = input_data.kwargs['blockTables']
        input_tokens = torch.tensor((2,2,2,2,0,1,1,1))
        input_positions = torch.tensor((1,1,1,1,0,1,1,1))
        seq_lens = torch.tensor((2,2,2,2,1,1,1,1))
        slot_mapping = torch.tensor((5,9,13,17,-1,1,1,1))
        return input_tokens, input_positions, seq_lens, slot_mapping
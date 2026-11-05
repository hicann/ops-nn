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
import logging

from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi


@register("aclnn_function")
class Functionaclnnadvancestep(AclnnBaseApi):
    def init_by_input_data(self, input_data):
        input_args, output_packages = super().init_by_input_data(input_data)

        for _ in range(4):
            input_args.pop()
        output_packages[:] = [
            input_args[0],
            input_args[2],
            input_args[3],
            input_args[4],
        ]
        return input_args, output_packages


@register("function_advance_step")
class FunctionAdanceStep(BaseApi):
    def Advance_step(
        self,
        num_seqs,
        num_queries,
        block_size,
        input_tokens,
        sampled_token_ids,
        input_positions,
        seq_lens,
        slot_mapping,
        block_tables,
    ):
        logging.info(
            f"****************************************input_tokens {input_tokens} "
        )
        logging.info(
            f"****************************************input_positions {input_positions} "
        )
        logging.info(f"****************************************seq_lens {seq_lens} ")
        logging.info(
            f"****************************************slot_mapping {slot_mapping} "
        )

        for index in range(num_queries, num_seqs):
            input_tokens[index] = 0
            input_positions[index] = 0
            slot_mapping[index] = -1

        for cur_query_id in range(num_queries):
            input_tokens[cur_query_id] = sampled_token_ids[cur_query_id]
            seq_len = seq_lens[cur_query_id]
            next_seq_len = seq_len + 1
            next_input_pos = next_seq_len - 1
            seq_lens[cur_query_id] = next_seq_len
            input_positions[cur_query_id] = next_input_pos

            block_index = next_input_pos // block_size
            block_offset = next_input_pos % block_size
            logging.info(
                f"****************************************seq_lens  {seq_lens} "
            )
            logging.info(
                f"****************************************block_size  {block_size} "
            )
            logging.info(
                f"****************************************block_tables  {block_tables.shape} "
            )
            cur_block = block_tables[cur_query_id, block_index]
            slot_num = cur_block * block_size + block_offset
            slot_mapping[cur_query_id] = slot_num
        return input_tokens, input_positions, seq_lens, slot_mapping

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        num_seqs = input_data.kwargs["numSeqs"]
        num_queries = input_data.kwargs["numQueries"]
        block_size = input_data.kwargs["blockSize"]
        input_tokens = input_data.kwargs["inputTokensRef"]
        sampled_token_ids = input_data.kwargs["sampledTokenIds"]
        input_positions = input_data.kwargs["inputPositionsRef"]
        seq_lens = input_data.kwargs["seqLensRef"]
        slot_mapping = input_data.kwargs["slotMappingRef"]
        block_tables = input_data.kwargs["blockTables"]
        return self.Advance_step(
            num_seqs,
            num_queries,
            block_size,
            input_tokens,
            sampled_token_ids,
            input_positions,
            seq_lens,
            slot_mapping,
            block_tables,
        )

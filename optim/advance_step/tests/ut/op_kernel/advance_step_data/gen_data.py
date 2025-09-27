# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

#!/usr/bin/python3
import sys
import os
import numpy as np
import re
import torch
import tensorflow as tf

def Advance_step(num_seqs, num_queries, block_size, input_tokens, sampled_token_ids, input_positions,
                seq_lens, slot_mapping, block_tables):
    for blockIdx in range(40):
        n_pad = num_seqs - num_queries
        if n_pad > 0 and blockIdx == 0:
            offset = num_queries
            i = 0
            while i < n_pad:
                input_tokens[offset + i] = 0
                input_positions[offset + i] = 0
                slot_mapping[offset + i] = -1
                i += 40

        num_query_blocks = num_queries // 1
        if blockIdx >= num_query_blocks:
            break
        cur_query_id = blockIdx * 1 + 0
        if cur_query_id >= num_queries:
            break

        input_tokens[cur_query_id] = sampled_token_ids[cur_query_id]
        seq_len = seq_lens[cur_query_id]
        next_seq_len = seq_len + 1;
        next_input_pos = next_seq_len - 1;
        seq_lens[cur_query_id] = next_seq_len
        input_positions[cur_query_id] = next_input_pos

        block_index = next_input_pos // block_size
        block_offset = next_input_pos % block_size
        cur_block = block_tables[block_index]
        slot_num = (cur_block + block_tables.stride(0) * cur_query_id) * block_size + block_offset

        slot_mapping[cur_query_id] = slot_num

def parse_str_to_shape_list(shape_str):
    shape_str = shape_str.strip('(').strip(')')
    shape_list = [int(x) for x in shape_str.split(",")]
    return np.array(shape_list), shape_list

def gen_data_and_golden(input_shape_str, output_size_str, d_type="int64"):
    d_type_dict = {
        "int64": np.int64,
    }
    np_type = d_type_dict[d_type]
    input_shape, _ = parse_str_to_shape_list(input_shape_str)
    sample_shape, output_size = parse_str_to_shape_list(output_size_str)

    size = np.prod(input_shape)
    size2 = np.prod(sample_shape)

    input_tokens = np.random.randint(10, size=size).astype(np_type).flatten()
    sampled_token_ids = np.random.randint(10, size=size2).astype(np_type).flatten()
    input_positions = np.random.randint(10, size=size).astype(np_type).flatten()
    seq_lens = np.random.randint(10, size=size).astype(np_type).flatten()
    slot_mapping = np.random.randint(10, size=size).astype(np_type).flatten()
    block_tables = np.random.randint(10, size=size).astype(np_type).flatten()

    input_tokens_tensor = torch.tensor(input_tokens.astype(np.int64), dtype=torch.int64)
    sampled_token_ids_tensor = torch.tensor(sampled_token_ids.astype(np.int64), dtype=torch.int64)
    input_positions_tensor = torch.tensor(input_positions.astype(np.int64), dtype=torch.int64)
    seq_lens_tensor = torch.tensor(seq_lens.astype(np.int64), dtype=torch.int64)
    slot_mapping_tensor = torch.tensor(slot_mapping.astype(np.int64), dtype=torch.int64)
    block_tables_tensor = torch.tensor(block_tables.astype(np.int64), dtype=torch.int64)

    input_tokens_out = np.random.randint(1, size=size).astype(np_type).flatten()
    input_positions_out = np.random.randint(1, size=size).astype(np_type).flatten()
    seq_lens_out = np.random.randint(1, size=size).astype(np_type).flatten()
    slot_mapping_out = np.random.randint(1, size=size).astype(np_type).flatten()

    input_tokens_out_tensor = torch.tensor(input_tokens_out.astype(np.int64), dtype=torch.int64)
    input_positions_out_tensor = torch.tensor(input_positions_out.astype(np.int64), dtype=torch.int64)
    seq_lens_out_tensor = torch.tensor(seq_lens_out.astype(np.int64), dtype=torch.int64)
    slot_mapping_out_tensor = torch.tensor(slot_mapping_out.astype(np.int64), dtype=torch.int64)

    Advance_step(16, 8, 8, input_tokens_tensor, sampled_token_ids_tensor, input_positions_tensor, seq_lens_tensor,
                slot_mapping_tensor, block_tables_tensor)

    input_tokens.astype(np_type).tofile(f"input_tokens_input_advance_step.bin")
    sampled_token_ids.astype(np_type).tofile(f"sampled_token_ids_input_advance_step.bin")
    input_positions.astype(np_type).tofile(f"input_positions_input_advance_step.bin")
    seq_lens.astype(np_type).tofile(f"seq_lens_input_advance_step.bin")
    slot_mapping.astype(np_type).tofile(f"slot_mapping_input_advance_step.bin")
    block_tables.astype(np_type).tofile(f"block_tables_input_advance_step.bin")
    input_tokens_out_tensor.numpy().astype(np_type).tofile(f"1_golden_advance_step.bin")
    input_positions_out_tensor.numpy().astype(np_type).tofile(f"2_golden_advance_step.bin")
    seq_lens_out_tensor.numpy().astype(np_type).tofile(f"3_golden_advance_step.bin")
    slot_mapping_out_tensor.numpy().astype(np_type).tofile(f"4_golden_advance_step.bin")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Param num must be 4, actually is ", len(sys.argv))
        exit(1)
    # 清理bin文件
    os.system("rm -rf *.bin")
    gen_data_and_golden(sys.argv[1], sys.argv[2], sys.argv[3])
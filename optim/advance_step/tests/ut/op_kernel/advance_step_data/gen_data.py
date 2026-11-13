# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import sys
import os
import numpy as np
import torch


def trunc_div_mod(a, b):
    # C++ 截断除法/取余语义（kernel int64 除法取余）：python // 与 % 为向下取整，
    # 负值域下两者不一致（如 a=-9,b=8：trunc 商 -1 余 -1；floor 商 -2 余 7）
    q = abs(a) // abs(b)
    if (a < 0) != (b < 0):
        q = -q
    return q, a - q * b


def Advance_step(
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
    for index in range(num_queries, num_seqs):
        input_tokens[index] = 0
        input_positions[index] = 0
        slot_mapping[index] = -1

    # 查表防御语义与 kernel 对齐：行内列校验 0 ≤ q < stride，越列（含跨行平坦
    # 下标与负值）防御性写 0（块号与余数同时屏蔽）
    bt_stride = block_tables.numel() // block_tables.shape[0]
    for cur_query_id in range(num_queries):
        input_tokens[cur_query_id] = sampled_token_ids[cur_query_id]
        seq_len = seq_lens[cur_query_id].item()
        seq_lens[cur_query_id] = seq_len + 1
        input_positions[cur_query_id] = seq_len

        # C++ 截断除法（kernel int64 除法语义；非负契约域内 trunc == floor）
        block_index, block_offset = trunc_div_mod(seq_len, block_size)
        # 行内列校验（与 kernel 一致）：q∈[0, stride) 正常查表（平坦下标 stride*i+q），
        # 越列（含跨行平坦下标与负值）防御性写 0（块号与余数同时屏蔽）
        if 0 <= block_index < bt_stride:
            physical_block = int(block_tables[bt_stride * cur_query_id + block_index])
            slot_mapping[cur_query_id] = physical_block * block_size + block_offset
        else:
            slot_mapping[cur_query_id] = 0


def advance_step_v2_golden(
    num_seqs,
    spec_num,
    block_size,
    block_tables_stride,
    input_tokens,
    sampled_token_ids,
    input_positions,
    seq_lens,
    slot_mapping,
    block_tables,
    spec_token,
    accepted_num,
):
    # 与 op_kernel/advance_step_spec.h 的 V2 实现逐项对齐：
    # 1) inputPositions 逐元素 += acceptedNum[seq] + 1；seqLens = 更新后位置 + 1
    # 2) slotMapping 按行查表：blockTablesFlat[pos/blockSize + stride*seq]*blockSize + pos%blockSize
    # 3) lastToken = 行末补 -1 后首个 -1 的前一元素（行首 -1 时为 -1）
    # 4) inputTokens 第 0 列写 lastToken，后 specNum 列写 specToken
    token_each_reqs = spec_num + 1
    out_tokens = np.array(input_tokens, dtype=np.int64).reshape(
        num_seqs, token_each_reqs
    )
    out_pos = np.array(input_positions, dtype=np.int64).reshape(
        num_seqs, token_each_reqs
    )
    out_lens = np.array(seq_lens, dtype=np.int64).reshape(num_seqs, token_each_reqs)
    out_slot = np.array(slot_mapping, dtype=np.int64).reshape(num_seqs, token_each_reqs)
    in_pos = np.array(input_positions, dtype=np.int64).reshape(
        num_seqs, token_each_reqs
    )
    in_sampled = np.array(sampled_token_ids, dtype=np.int64).reshape(
        num_seqs, token_each_reqs
    )
    in_spec = np.array(spec_token, dtype=np.int64).reshape(num_seqs, spec_num)
    bt = np.array(block_tables, dtype=np.int64).reshape(num_seqs, block_tables_stride)
    for s in range(num_seqs):
        row = in_sampled[s]
        padded = np.append(row, -1)
        min_index = int(np.argmin(padded))
        last_token = -1 if min_index == 0 else int(row[min_index - 1])
        out_tokens[s, 0] = last_token
        out_tokens[s, 1:] = in_spec[s]
        acc = int(accepted_num[s])
        for j in range(token_each_reqs):
            new_pos = int(in_pos[s][j]) + acc + 1
            out_pos[s, j] = new_pos
            out_lens[s, j] = new_pos + 1
            # 行内列校验（与 kernel qMask 掩码一致）：q∈[0, stride) 正常查表，
            # 越列（含跨行平坦下标）防御性写 0（块号与余数同时屏蔽）
            q, mod = trunc_div_mod(new_pos, block_size)
            if 0 <= q < block_tables_stride:
                out_slot[s, j] = int(bt[s, q]) * block_size + mod
            else:
                out_slot[s, j] = 0
    return (
        out_tokens.flatten(),
        out_pos.flatten(),
        out_lens.flatten(),
        out_slot.flatten(),
    )


def gen_v2_data_and_golden(
    num_seqs=16, spec_num=2, block_size=128, block_tables_stride=1000
):
    token_each_reqs = spec_num + 1
    rng = np.random.default_rng(20261019)

    input_tokens = rng.integers(10, 1000, size=num_seqs * token_each_reqs).astype(
        np.int64
    )
    # 覆盖 pos >= blockSize（q>0）分支：bs=128 时 q 最高 4，pos 恒小于 bs 曾掩盖 flat 计算序回归
    input_positions = rng.integers(0, 600, size=num_seqs * token_each_reqs).astype(
        np.int64
    )
    seq_lens = rng.integers(0, 100, size=num_seqs * token_each_reqs).astype(np.int64)
    slot_mapping = rng.integers(10, 1000, size=num_seqs * token_each_reqs).astype(
        np.int64
    )
    block_tables = rng.integers(0, 1000, size=num_seqs * block_tables_stride).astype(
        np.int64
    )
    spec_token = rng.integers(10, 1000, size=num_seqs * spec_num).astype(np.int64)
    accepted_num = rng.integers(0, 4, size=num_seqs).astype(np.int64)

    sampled_token_ids = rng.integers(10, 1000, size=(num_seqs, token_each_reqs)).astype(
        np.int64
    )
    # 构造哨兵分支：首行行首 -1（lastToken=-1）、次行行内 -1、其余行随机全非 -1
    sampled_token_ids[0][0] = -1
    if token_each_reqs == 3:
        sampled_token_ids[1] = [5, -1, 3]
    else:
        # 其他 te（含 v2spec1 的 te=2）：行内 -1 分支按行宽构造（[5, -1, -1, ...]）
        sampled_token_ids[1][0] = 5
        for k in range(1, token_each_reqs):
            sampled_token_ids[1][k] = -1

    out_tokens, out_pos, out_lens, out_slot = advance_step_v2_golden(
        num_seqs,
        spec_num,
        block_size,
        block_tables_stride,
        input_tokens,
        sampled_token_ids,
        input_positions,
        seq_lens,
        slot_mapping,
        block_tables,
        spec_token,
        accepted_num,
    )

    input_tokens.astype(np.int64).tofile("v2_input_tokens_input_advance_step.bin")
    sampled_token_ids.astype(np.int64).tofile(
        "v2_sampled_token_ids_input_advance_step.bin"
    )
    input_positions.astype(np.int64).tofile("v2_input_positions_input_advance_step.bin")
    seq_lens.astype(np.int64).tofile("v2_seq_lens_input_advance_step.bin")
    slot_mapping.astype(np.int64).tofile("v2_slot_mapping_input_advance_step.bin")
    block_tables.astype(np.int64).tofile("v2_block_tables_input_advance_step.bin")
    spec_token.astype(np.int64).tofile("v2_spec_token_input_advance_step.bin")
    accepted_num.astype(np.int64).tofile("v2_accepted_num_input_advance_step.bin")
    out_tokens.tofile("v2_1_golden_advance_step.bin")
    out_pos.tofile("v2_2_golden_advance_step.bin")
    out_lens.tofile("v2_3_golden_advance_step.bin")
    out_slot.tofile("v2_4_golden_advance_step.bin")


def gen_v1_oob_data_and_golden():
    # V1 契约外值域用例（对应 issue：AdvanceStep kernel 内存越界）：
    # seqLens 定点构造越上界/负值/压界位置，见证防御语义——行内列校验
    # 0 ≤ q < stride（q = trunc(pos/blockSize)）：越列（含跨行平坦下标与负
    # 商）防御性 slot 写 0（块号与余数同时屏蔽）且不读 GM；q 落在行内
    # （含负 pos 的 q=0 分支）正常查表，与 V2 qMask 掩码语义一致。
    # 参数与 test_advance_step_int_1 的 tiling 一致：
    #   numSeqs=16, numQueries=8, blockSize=8, blockTables 16 元素（stride=1），
    #   行内合法列仅 q=0，flat = i + q
    num_seqs, num_queries, block_size = 16, 8, 8
    rng = np.random.default_rng(20261104)

    input_tokens = rng.integers(10, 1000, size=num_seqs).astype(np.int64)
    sampled_token_ids = rng.integers(10, 1000, size=(num_queries, 1)).astype(np.int64)
    input_positions = rng.integers(10, 1000, size=num_seqs).astype(np.int64)
    seq_lens = rng.integers(10, 1000, size=num_seqs).astype(np.int64)
    slot_mapping = rng.integers(10, 1000, size=num_seqs).astype(np.int64)
    block_tables = rng.integers(10, 1000, size=num_seqs).astype(np.int64)

    # query 行位置值（pos = 更新前 seqLens[i]）逐行定点：
    #   [0]=127  → q=15 ≥ stride=1，越列（防御 0）
    #   [1]=128  → q=16，越上界（防御 0）
    #   [2]=10^15 → q 巨值（issue 同款量级，防御 0）
    #   [3]=-9   → trunc 商 -1，负列（防御 0，余 -1 也被屏蔽）
    #   [4]=-3   → trunc 商 0，行内（读 bt[4]，余 -3）
    #   [5..7]   = q=5 越列防御 0；q=0 行内读 bt[6]；q=11 越列防御 0
    pos_plan = [127, 128, 10**15, -9, -3, 40, 0, 95]
    for i in range(num_queries):
        seq_lens[i] = pos_plan[i]

    input_tokens_tensor = torch.tensor(input_tokens, dtype=torch.int64)
    sampled_token_ids_tensor = torch.tensor(sampled_token_ids, dtype=torch.int64)
    input_positions_tensor = torch.tensor(input_positions, dtype=torch.int64)
    seq_lens_tensor = torch.tensor(seq_lens, dtype=torch.int64)
    slot_mapping_tensor = torch.tensor(slot_mapping, dtype=torch.int64)
    block_tables_tensor = torch.tensor(block_tables, dtype=torch.int64)

    Advance_step(
        num_seqs,
        num_queries,
        block_size,
        input_tokens_tensor,
        sampled_token_ids_tensor,
        input_positions_tensor,
        seq_lens_tensor,
        slot_mapping_tensor,
        block_tables_tensor,
    )

    input_tokens.astype(np.int64).tofile("oob_input_tokens_input_advance_step.bin")
    sampled_token_ids.astype(np.int64).tofile(
        "oob_sampled_token_ids_input_advance_step.bin"
    )
    input_positions.astype(np.int64).tofile(
        "oob_input_positions_input_advance_step.bin"
    )
    seq_lens.astype(np.int64).tofile("oob_seq_lens_input_advance_step.bin")
    slot_mapping.astype(np.int64).tofile("oob_slot_mapping_input_advance_step.bin")
    block_tables.astype(np.int64).tofile("oob_block_tables_input_advance_step.bin")
    input_tokens_tensor.numpy().astype(np.int64).tofile("oob_1_golden_advance_step.bin")
    input_positions_tensor.numpy().astype(np.int64).tofile(
        "oob_2_golden_advance_step.bin"
    )
    seq_lens_tensor.numpy().astype(np.int64).tofile("oob_3_golden_advance_step.bin")
    slot_mapping_tensor.numpy().astype(np.int64).tofile("oob_4_golden_advance_step.bin")


def parse_str_to_shape_list(shape_str):
    shape_str = shape_str.strip("(").strip(")")
    shape_list = [int(x) for x in shape_str.split(",")]
    return np.array(shape_list), shape_list


def gen_data_and_golden(input_shape_str, output_size_str, d_type="int64"):
    d_type_dict = {
        "int64": np.int64,
    }
    np_type = d_type_dict[d_type]
    input_shape, _ = parse_str_to_shape_list(input_shape_str)
    sample_shape, _ = parse_str_to_shape_list(output_size_str)

    size = np.prod(input_shape)
    size2 = np.prod(sample_shape)

    input_tokens = np.random.randint(10, size=size).astype(np_type).flatten()
    sampled_token_ids = np.random.randint(10, size=size2).astype(np_type).flatten()
    input_positions = np.random.randint(10, size=size).astype(np_type).flatten()
    seq_lens = np.random.randint(8, size=size).astype(np_type).flatten()
    slot_mapping = np.random.randint(10, size=size).astype(np_type).flatten()
    block_tables = np.random.randint(10, size=size).astype(np_type).flatten()

    input_tokens_tensor = torch.tensor(input_tokens.astype(np.int64), dtype=torch.int64)
    sampled_token_ids_tensor = torch.tensor(
        sampled_token_ids.astype(np.int64), dtype=torch.int64
    )
    input_positions_tensor = torch.tensor(
        input_positions.astype(np.int64), dtype=torch.int64
    )
    seq_lens_tensor = torch.tensor(seq_lens.astype(np.int64), dtype=torch.int64)
    slot_mapping_tensor = torch.tensor(slot_mapping.astype(np.int64), dtype=torch.int64)
    block_tables_tensor = torch.tensor(block_tables.astype(np.int64), dtype=torch.int64)

    Advance_step(
        int(size),
        int(size2),
        8,
        input_tokens_tensor,
        sampled_token_ids_tensor,
        input_positions_tensor,
        seq_lens_tensor,
        slot_mapping_tensor,
        block_tables_tensor,
    )

    input_tokens.astype(np_type).tofile("input_tokens_input_advance_step.bin")
    sampled_token_ids.astype(np_type).tofile("sampled_token_ids_input_advance_step.bin")
    input_positions.astype(np_type).tofile("input_positions_input_advance_step.bin")
    seq_lens.astype(np_type).tofile("seq_lens_input_advance_step.bin")
    slot_mapping.astype(np_type).tofile("slot_mapping_input_advance_step.bin")
    block_tables.astype(np_type).tofile("block_tables_input_advance_step.bin")
    input_tokens_tensor.numpy().astype(np_type).tofile("1_golden_advance_step.bin")
    input_positions_tensor.numpy().astype(np_type).tofile("2_golden_advance_step.bin")
    seq_lens_tensor.numpy().astype(np_type).tofile("3_golden_advance_step.bin")
    slot_mapping_tensor.numpy().astype(np_type).tofile("4_golden_advance_step.bin")


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "v2":
        # V2（投机采样）用例：参数与 test_advance_step_int_2 的 tiling 保持一致
        gen_v2_data_and_golden()
        sys.exit(0)
    if len(sys.argv) >= 2 and sys.argv[1] == "v2spec1":
        # V2 specNum=1（tokenEachReqs=2）用例：覆盖 ComputeInputTokens 的 te==2 向量化
        # 哨兵检索快速路径；参数与 test_advance_step_int_5 的 tiling 保持一致
        gen_v2_data_and_golden(
            num_seqs=16, spec_num=1, block_size=128, block_tables_stride=1000
        )
        sys.exit(0)
    if len(sys.argv) >= 2 and sys.argv[1] == "v1oob":
        # V1 契约外值域（防御语义）用例：参数与 test_advance_step_int_1 的 tiling 保持一致
        # （旧调用携带的 aiv_num 尾参已随 padding 密集化废弃，忽略多余参数）
        gen_v1_oob_data_and_golden()
        sys.exit(0)
    if len(sys.argv) != 4 and len(sys.argv) != 5:
        print("Param num must be 4 or 5, actually is ", len(sys.argv))
        sys.exit(1)
    # 清理bin文件
    os.system("rm -rf *.bin")
    gen_data_and_golden(sys.argv[1], sys.argv[2], sys.argv[3])

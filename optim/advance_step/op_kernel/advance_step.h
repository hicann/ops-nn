/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file advance_step.h
 * \brief
 */

#ifndef ADVANCE_STEP_H
#define ADVANCE_STEP_H

#include <type_traits>
#include "kernel_operator.h"
#include "lib/matmul_intf.h"

namespace AdvanceStepNs {
using namespace AscendC;

template <typename T>
class KernelAdvanceStep {
public:
    TPipe pipe;
    __aicore__ inline KernelAdvanceStep(){};
    __aicore__ inline void Init(GM_ADDR input_tokens, GM_ADDR sampled_token_ids, GM_ADDR input_positions,
                                GM_ADDR seq_lens, GM_ADDR slot_mapping, GM_ADDR block_tables, GM_ADDR workspace,
                                const AdvanceStepTilingData* tilingData);
    __aicore__ inline void Process();

private:
    GlobalTensor<T> input_tokens_ptr;
    GlobalTensor<T> sampled_token_ids_ptr;
    GlobalTensor<T> input_positions_ptr;
    GlobalTensor<T> seq_lens_ptr;
    GlobalTensor<T> slot_mapping_ptr;
    GlobalTensor<T> block_tables_ptr;

    int64_t blockIdx;
    int64_t block_tables_stride;
    int64_t num_seqs;
    int64_t num_queries;
    int64_t block_size;
    int64_t per_core_queries;
};

template <typename T>
__aicore__ inline void KernelAdvanceStep<T>::Init(GM_ADDR input_tokens, GM_ADDR sampled_token_ids,
                                                  GM_ADDR input_positions, GM_ADDR seq_lens, GM_ADDR slot_mapping,
                                                  GM_ADDR block_tables, GM_ADDR workspace,
                                                  const AdvanceStepTilingData* tilingData)
{
    input_tokens_ptr.SetGlobalBuffer((__gm__ T*)input_tokens);
    sampled_token_ids_ptr.SetGlobalBuffer((__gm__ T*)sampled_token_ids);
    input_positions_ptr.SetGlobalBuffer((__gm__ T*)input_positions);
    seq_lens_ptr.SetGlobalBuffer((__gm__ T*)seq_lens);
    slot_mapping_ptr.SetGlobalBuffer((__gm__ T*)slot_mapping);
    block_tables_ptr.SetGlobalBuffer((__gm__ T*)block_tables);

    blockIdx = GetBlockIdx();
    block_tables_stride = tilingData->blockTablesStride;
    num_seqs = tilingData->numSeqs;
    num_queries = tilingData->numQueries;
    block_size = tilingData->blockSize;
    per_core_queries = tilingData->perCoreSeqs;
}

template <typename T>
__aicore__ inline void KernelAdvanceStep<T>::Process()
{
    int64_t n_pad = num_seqs - num_queries;
    if (n_pad && blockIdx == 0) {
        int64_t offset = num_queries;
        for (int64_t i = 0; i < n_pad; ++i) {
            input_tokens_ptr.SetValue(offset + i, 0);
            input_positions_ptr.SetValue(offset + i, 0);
            slot_mapping_ptr.SetValue(offset + i, -1);
        }
    }

    // 更新 [0, numQueries) 的 query 行（REQUIREMENTS §4.1 公式），多核分块切分：核 b 处理
    // [b*perCoreQueries, min((b+1)*perCoreQueries, numQueries))，各核行区间连续且之并为全量。
    // perCoreQueries 来自 tiling（多核时 CeilDiv(numQueries, aivNum)、单核时 = numQueries）。
    // blockTables 查表列号 q = pos/blockSize 由 GM 数据值（seqLens）驱动，host 侧无法校验：
    // 契约要求 max(seqLens)/blockSize < blockTablesStride；契约外输入（越列、跨行平坦
    // 下标、负值）防御性 slot 写 0 且不读 GM——与 V2 路径 qMask 掩码语义一致。
    // 查表公式与 V2 及 issue #6573 修正对齐：物理块号按平坦下标 stride*i + q 读取。
    const int64_t rowStart = blockIdx * per_core_queries;
    if (rowStart >= num_queries) {
        return;
    }
    int64_t rowEnd = rowStart + per_core_queries;
    if (rowEnd > num_queries) {
        rowEnd = num_queries;
    }
    for (int64_t index = rowStart; index < rowEnd; index++) {
        // Update input_tokens
        input_tokens_ptr.SetValue(index, sampled_token_ids_ptr.GetValue(index));

        int64_t seq_len = seq_lens_ptr.GetValue(index);
        int64_t next_seq_len = seq_len + 1;
        int64_t next_input_pos = next_seq_len - 1;

        // Update seq_lens
        seq_lens_ptr.SetValue(index, next_seq_len);
        input_positions_ptr.SetValue(index, next_input_pos);

        int64_t block_index = next_input_pos / block_size; // 第几个q块
        int64_t block_offset = next_input_pos % block_size;

        // 行内列校验（0 ≤ block_index < stride，q 由 GM 数据驱动、host 侧无法校验）：
        // 契约要求 max(seqLens)/blockSize < blockTablesStride；契约外（负值/越列/跨行
        // 平坦下标）防御性写 0——块号与余数同时屏蔽（与 V2 qMask 掩码语义一致，兑现
        // "越界防御性写 0"公开契约），行内蕴含平坦下标落在张量内、不发生 GM 越界读
        const bool inRow = block_index >= 0 && block_index < block_tables_stride;
        int64_t slot_num = 0;
        if (inRow) {
            const int64_t physical_block = block_tables_ptr.GetValue(block_tables_stride * index + block_index);
            slot_num = physical_block * block_size + block_offset;
        }
        // Update slot_mapping
        slot_mapping_ptr.SetValue(index, slot_num);
    }
}
} // namespace AdvanceStepNs
#endif // ADVANCE_STEP_H

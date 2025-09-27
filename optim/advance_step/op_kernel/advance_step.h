/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
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
class KernelAdvanceStep
{
public:
    TPipe pipe;
    __aicore__ inline KernelAdvanceStep(){};
    __aicore__ inline void Init(
        GM_ADDR input_tokens, GM_ADDR sampled_token_ids, GM_ADDR input_positions, GM_ADDR seq_lens,
        GM_ADDR slot_mapping, GM_ADDR block_tables, GM_ADDR workspace, const AdvanceStepTilingData* tilingData);
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
    int64_t total_core_num;
};

template <typename T>
__aicore__ inline void KernelAdvanceStep<T>::Init(
    GM_ADDR input_tokens, GM_ADDR sampled_token_ids, GM_ADDR input_positions, GM_ADDR seq_lens, GM_ADDR slot_mapping,
    GM_ADDR block_tables, GM_ADDR workspace, const AdvanceStepTilingData* tilingData)
{
    input_tokens_ptr.SetGlobalBuffer((__gm__ T*)input_tokens);
    sampled_token_ids_ptr.SetGlobalBuffer((__gm__ T*)sampled_token_ids);
    input_positions_ptr.SetGlobalBuffer((__gm__ T*)input_positions);
    seq_lens_ptr.SetGlobalBuffer((__gm__ T*)seq_lens);
    slot_mapping_ptr.SetGlobalBuffer((__gm__ T*)slot_mapping);
    block_tables_ptr.SetGlobalBuffer((__gm__ T*)block_tables);

    blockIdx = GetBlockIdx();
    total_core_num = tilingData->needCoreNum;
    block_tables_stride = tilingData->blockTablesStride;
    num_seqs = tilingData->numSeqs;
    num_queries = tilingData->numQueries;
    block_size = tilingData->blockSize;
}

template <typename T>
__aicore__ inline void KernelAdvanceStep<T>::Process()
{
    int64_t n_pad = num_seqs - num_queries;
    if (n_pad && blockIdx == 0) {
        int64_t offset = num_queries;
        for (int i = 0; i < n_pad; i += total_core_num) {
            input_tokens_ptr.SetValue(offset + i, 0);
            input_positions_ptr.SetValue(offset + i, 0);
            slot_mapping_ptr.SetValue(offset + i, -1);
        }
    }

    for (int index = 0; index < total_core_num; index++) {
        if (index >= num_queries) {
            return;
        }
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

        int64_t slot_num =
            (block_tables_ptr.GetValue(block_index) + block_tables_stride * index) * block_size + block_offset;
        // Update slot_mapping
        slot_mapping_ptr.SetValue(index, slot_num);
    }
}
} // namespace AdvanceStepNs
#endif // ADVANCE_STEP_H
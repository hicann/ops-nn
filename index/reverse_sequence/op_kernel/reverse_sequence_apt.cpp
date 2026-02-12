/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
* \file reverse_sequence_apt.cpp
* \brief
*/
#include "./arch35/reverse_sequence_simt.h"
#include "./arch35/reverse_sequence_tiling_key.h"

#define TEST_FIRST_KEY 101
#define REVERSE_SEQUENCE_BSA 200001
using namespace ReverseSequence;

template <uint64_t TEMPLATE_MODE, uint64_t DYTPE_MODE, uint64_t ADDR_MODE>
__global__ __aicore__ void reverse_sequence(GM_ADDR x, GM_ADDR seq_lengths, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) 
{
    AscendC::TPipe pipeBase;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(ReverseSequenceSimtTilingData4RegBase);
    if constexpr (TEMPLATE_MODE == TPL_MODE_TEMPLATE_SIMT && DYTPE_MODE == TPL_MODE_DTYPE_B8 && ADDR_MODE == TPL_MODE_ADDR_INT32) {
        REGISTER_TILING_FOR_TILINGKEY("TEMPLATE_MODE == TPL_MODE_TEMPLATE_SIMT && DYTPE_MODE == TPL_MODE_DTYPE_B8 && ADDR_MODE == TPL_MODE_ADDR_INT32", ReverseSequenceSimtTilingData4RegBase);
        GET_TILING_DATA_WITH_STRUCT(ReverseSequenceSimtTilingData4RegBase, tilingData, tiling);
        ReverseSequenceSimt<int8_t, DTYPE_SEQ_LENGTHS, uint32_t> op(&pipeBase, &tilingData);
        op.Init(x, seq_lengths, y);
        op.Process();
    } else if constexpr (TEMPLATE_MODE == TPL_MODE_TEMPLATE_SIMT && DYTPE_MODE == TPL_MODE_DTYPE_B8 && ADDR_MODE == TPL_MODE_ADDR_INT64) {
        REGISTER_TILING_FOR_TILINGKEY("TEMPLATE_MODE == TPL_MODE_TEMPLATE_SIMT && DYTPE_MODE == TPL_MODE_DTYPE_B8 && ADDR_MODE == TPL_MODE_ADDR_INT64", ReverseSequenceSimtTilingData4RegBase);
        GET_TILING_DATA_WITH_STRUCT(ReverseSequenceSimtTilingData4RegBase, tilingData, tiling);
        ReverseSequenceSimt<int8_t, DTYPE_SEQ_LENGTHS, uint64_t> op(&pipeBase, &tilingData);
        op.Init(x, seq_lengths, y);
        op.Process();
    } else if constexpr (TEMPLATE_MODE == TPL_MODE_TEMPLATE_SIMT && DYTPE_MODE == TPL_MODE_DTYPE_B16 && ADDR_MODE == TPL_MODE_ADDR_INT32) {
        REGISTER_TILING_FOR_TILINGKEY("TEMPLATE_MODE == TPL_MODE_TEMPLATE_SIMT && DYTPE_MODE == TPL_MODE_DTYPE_B16 && ADDR_MODE == TPL_MODE_ADDR_INT32", ReverseSequenceSimtTilingData4RegBase);
        GET_TILING_DATA_WITH_STRUCT(ReverseSequenceSimtTilingData4RegBase, tilingData, tiling);
        ReverseSequenceSimt<int16_t, DTYPE_SEQ_LENGTHS, uint32_t> op(&pipeBase, &tilingData);
        op.Init(x, seq_lengths, y);
        op.Process();
    } else if constexpr (TEMPLATE_MODE == TPL_MODE_TEMPLATE_SIMT && DYTPE_MODE == TPL_MODE_DTYPE_B16 && ADDR_MODE == TPL_MODE_ADDR_INT64) {
        REGISTER_TILING_FOR_TILINGKEY("TEMPLATE_MODE == TPL_MODE_TEMPLATE_SIMT && DYTPE_MODE == TPL_MODE_DTYPE_B16 && ADDR_MODE == TPL_MODE_ADDR_INT64", ReverseSequenceSimtTilingData4RegBase);
        GET_TILING_DATA_WITH_STRUCT(ReverseSequenceSimtTilingData4RegBase, tilingData, tiling);
        ReverseSequenceSimt<int16_t, DTYPE_SEQ_LENGTHS, uint64_t> op(&pipeBase, &tilingData);
        op.Init(x, seq_lengths, y);
        op.Process();
    } else if constexpr (TEMPLATE_MODE == TPL_MODE_TEMPLATE_SIMT && DYTPE_MODE == TPL_MODE_DTYPE_B32 && ADDR_MODE == TPL_MODE_ADDR_INT32) {
        REGISTER_TILING_FOR_TILINGKEY("TEMPLATE_MODE == TPL_MODE_TEMPLATE_SIMT && DYTPE_MODE == TPL_MODE_DTYPE_B32 && ADDR_MODE == TPL_MODE_ADDR_INT32", ReverseSequenceSimtTilingData4RegBase);
        GET_TILING_DATA_WITH_STRUCT(ReverseSequenceSimtTilingData4RegBase, tilingData, tiling);
        ReverseSequenceSimt<int32_t, DTYPE_SEQ_LENGTHS, uint32_t> op(&pipeBase, &tilingData);
        op.Init(x, seq_lengths, y);
        op.Process();
    } else if constexpr (TEMPLATE_MODE == TPL_MODE_TEMPLATE_SIMT && DYTPE_MODE == TPL_MODE_DTYPE_B32 && ADDR_MODE == TPL_MODE_ADDR_INT64) {
        REGISTER_TILING_FOR_TILINGKEY("TEMPLATE_MODE == TPL_MODE_TEMPLATE_SIMT && DYTPE_MODE == TPL_MODE_DTYPE_B32 && ADDR_MODE == TPL_MODE_ADDR_INT64", ReverseSequenceSimtTilingData4RegBase);
        GET_TILING_DATA_WITH_STRUCT(ReverseSequenceSimtTilingData4RegBase, tilingData, tiling);
        ReverseSequenceSimt<int32_t, DTYPE_SEQ_LENGTHS, uint64_t> op(&pipeBase, &tilingData);
        op.Init(x, seq_lengths, y);
        op.Process();
    } else if constexpr (TEMPLATE_MODE == TPL_MODE_TEMPLATE_SIMT && DYTPE_MODE == TPL_MODE_DTYPE_B64 && ADDR_MODE == TPL_MODE_ADDR_INT32) {
        REGISTER_TILING_FOR_TILINGKEY("TEMPLATE_MODE == TPL_MODE_TEMPLATE_SIMT && DYTPE_MODE == TPL_MODE_DTYPE_B64 && ADDR_MODE == TPL_MODE_ADDR_INT32", ReverseSequenceSimtTilingData4RegBase);
        GET_TILING_DATA_WITH_STRUCT(ReverseSequenceSimtTilingData4RegBase, tilingData, tiling);
        ReverseSequenceSimt<int64_t, DTYPE_SEQ_LENGTHS, uint32_t> op(&pipeBase, &tilingData);
        op.Init(x, seq_lengths, y);
        op.Process();
    } else if constexpr (TEMPLATE_MODE == TPL_MODE_TEMPLATE_SIMT && DYTPE_MODE == TPL_MODE_DTYPE_B64 && ADDR_MODE == TPL_MODE_ADDR_INT64) {
        REGISTER_TILING_FOR_TILINGKEY("TEMPLATE_MODE == TPL_MODE_TEMPLATE_SIMT && DYTPE_MODE == TPL_MODE_DTYPE_B64 && ADDR_MODE == TPL_MODE_ADDR_INT64", ReverseSequenceSimtTilingData4RegBase);
        GET_TILING_DATA_WITH_STRUCT(ReverseSequenceSimtTilingData4RegBase, tilingData, tiling);
        ReverseSequenceSimt<int64_t, DTYPE_SEQ_LENGTHS, uint64_t> op(&pipeBase, &tilingData);
        op.Init(x, seq_lengths, y);
        op.Process();
    }
}
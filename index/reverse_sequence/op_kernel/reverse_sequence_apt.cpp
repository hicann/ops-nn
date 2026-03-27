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
#include "./arch35/reverse_sequence_bsa.h"
#include "./arch35/reverse_sequence_bas.h"
#include "./arch35/reverse_sequence_bs.h"
#include "./arch35/reverse_sequence_sba.h"
#include "./arch35/reverse_sequence_tiling_key.h"

#define TEST_FIRST_KEY 101
#define REVERSE_SEQUENCE_BSA 200001
using namespace ReverseSequence;

template <uint64_t TEMPLATE_MODE, uint64_t DYTPE_MODE, uint64_t ADDR_MODE>
__global__ __aicore__ void reverse_sequence(GM_ADDR x, GM_ADDR seq_lengths, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) 
{
    AscendC::TPipe pipeBase;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(ReverseSequenceTilingData);
    if constexpr (TEMPLATE_MODE == TPL_MODE_TEMPLATE_SIMT && DYTPE_MODE == TPL_MODE_DTYPE_B8 && ADDR_MODE == TPL_MODE_ADDR_INT32) {
        GET_TILING_DATA_WITH_STRUCT(ReverseSequenceSimtTilingData4RegBase, tilingData, tiling);
        ReverseSequenceSimt<int8_t, DTYPE_SEQ_LENGTHS, uint32_t> op(&pipeBase, &tilingData);
        op.Init(x, seq_lengths, y);
        op.Process();
    } else if constexpr (TEMPLATE_MODE == TPL_MODE_TEMPLATE_SIMT && DYTPE_MODE == TPL_MODE_DTYPE_B8 && ADDR_MODE == TPL_MODE_ADDR_INT64) {
        GET_TILING_DATA_WITH_STRUCT(ReverseSequenceSimtTilingData4RegBase, tilingData, tiling);
        ReverseSequenceSimt<int8_t, DTYPE_SEQ_LENGTHS, uint64_t> op(&pipeBase, &tilingData);
        op.Init(x, seq_lengths, y);
        op.Process();
    } else if constexpr (TEMPLATE_MODE == TPL_MODE_TEMPLATE_SIMT && DYTPE_MODE == TPL_MODE_DTYPE_B16 && ADDR_MODE == TPL_MODE_ADDR_INT32) {
        GET_TILING_DATA_WITH_STRUCT(ReverseSequenceSimtTilingData4RegBase, tilingData, tiling);
        ReverseSequenceSimt<int16_t, DTYPE_SEQ_LENGTHS, uint32_t> op(&pipeBase, &tilingData);
        op.Init(x, seq_lengths, y);
        op.Process();
    } else if constexpr (TEMPLATE_MODE == TPL_MODE_TEMPLATE_SIMT && DYTPE_MODE == TPL_MODE_DTYPE_B16 && ADDR_MODE == TPL_MODE_ADDR_INT64) {
        GET_TILING_DATA_WITH_STRUCT(ReverseSequenceSimtTilingData4RegBase, tilingData, tiling);
        ReverseSequenceSimt<int16_t, DTYPE_SEQ_LENGTHS, uint64_t> op(&pipeBase, &tilingData);
        op.Init(x, seq_lengths, y);
        op.Process();
    } else if constexpr (TEMPLATE_MODE == TPL_MODE_TEMPLATE_SIMT && DYTPE_MODE == TPL_MODE_DTYPE_B32 && ADDR_MODE == TPL_MODE_ADDR_INT32) {
        GET_TILING_DATA_WITH_STRUCT(ReverseSequenceSimtTilingData4RegBase, tilingData, tiling);
        ReverseSequenceSimt<int32_t, DTYPE_SEQ_LENGTHS, uint32_t> op(&pipeBase, &tilingData);
        op.Init(x, seq_lengths, y);
        op.Process();
    } else if constexpr (TEMPLATE_MODE == TPL_MODE_TEMPLATE_SIMT && DYTPE_MODE == TPL_MODE_DTYPE_B32 && ADDR_MODE == TPL_MODE_ADDR_INT64) {
        GET_TILING_DATA_WITH_STRUCT(ReverseSequenceSimtTilingData4RegBase, tilingData, tiling);
        ReverseSequenceSimt<int32_t, DTYPE_SEQ_LENGTHS, uint64_t> op(&pipeBase, &tilingData);
        op.Init(x, seq_lengths, y);
        op.Process();
    } else if constexpr (TEMPLATE_MODE == TPL_MODE_TEMPLATE_SIMT && DYTPE_MODE == TPL_MODE_DTYPE_B64 && ADDR_MODE == TPL_MODE_ADDR_INT32) {
        GET_TILING_DATA_WITH_STRUCT(ReverseSequenceSimtTilingData4RegBase, tilingData, tiling);
        ReverseSequenceSimt<int64_t, DTYPE_SEQ_LENGTHS, uint32_t> op(&pipeBase, &tilingData);
        op.Init(x, seq_lengths, y);
        op.Process();
    } else if constexpr (TEMPLATE_MODE == TPL_MODE_TEMPLATE_SIMT && DYTPE_MODE == TPL_MODE_DTYPE_B64 && ADDR_MODE == TPL_MODE_ADDR_INT64) {
        GET_TILING_DATA_WITH_STRUCT(ReverseSequenceSimtTilingData4RegBase, tilingData, tiling);
        ReverseSequenceSimt<int64_t, DTYPE_SEQ_LENGTHS, uint64_t> op(&pipeBase, &tilingData);
        op.Init(x, seq_lengths, y);
        op.Process();
    } else if constexpr (TEMPLATE_MODE == TPL_MODE_TEMPLATE_BSA && DYTPE_MODE == TPL_MODE_DTYPE_B64 && ADDR_MODE == TPL_MODE_ADDR_INT64) {
        GET_TILING_DATA_WITH_STRUCT(ReverseSequenceBSATilingData, tilingData, tiling);
        ReverseSequenceBSA<int64_t, DTYPE_SEQ_LENGTHS> op(&pipeBase, &tilingData);
        op.Init(x, seq_lengths, y);
        op.Process();
    } else if constexpr (TEMPLATE_MODE == TPL_MODE_TEMPLATE_BSA && DYTPE_MODE == TPL_MODE_DTYPE_B32 && ADDR_MODE == TPL_MODE_ADDR_INT64) {
        GET_TILING_DATA_WITH_STRUCT(ReverseSequenceBSATilingData, tilingData, tiling);
        ReverseSequenceBSA<int32_t, DTYPE_SEQ_LENGTHS> op(&pipeBase, &tilingData);
        op.Init(x, seq_lengths, y);
        op.Process();
    } else if constexpr (TEMPLATE_MODE == TPL_MODE_TEMPLATE_BSA && DYTPE_MODE == TPL_MODE_DTYPE_B16 && ADDR_MODE == TPL_MODE_ADDR_INT64) {
        GET_TILING_DATA_WITH_STRUCT(ReverseSequenceBSATilingData, tilingData, tiling);
        ReverseSequenceBSA<int16_t, DTYPE_SEQ_LENGTHS> op(&pipeBase, &tilingData);
        op.Init(x, seq_lengths, y);
        op.Process();
    } else if constexpr (TEMPLATE_MODE == TPL_MODE_TEMPLATE_BSA && DYTPE_MODE == TPL_MODE_DTYPE_B8 && ADDR_MODE == TPL_MODE_ADDR_INT64) {
        GET_TILING_DATA_WITH_STRUCT(ReverseSequenceBSATilingData, tilingData, tiling);
        ReverseSequenceBSA<int8_t, DTYPE_SEQ_LENGTHS> op(&pipeBase, &tilingData);
        op.Init(x, seq_lengths, y);
        op.Process();
    } else if constexpr (TEMPLATE_MODE == TPL_MODE_TEMPLATE_BS && DYTPE_MODE == TPL_MODE_DTYPE_B8 && ADDR_MODE == TPL_MODE_ADDR_INT32) {
        GET_TILING_DATA_WITH_STRUCT(ReverseSequenceBSTilingData, tilingData, tiling);
        ReverseSequenceBS<int8_t, DTYPE_SEQ_LENGTHS, uint32_t> op(&pipeBase, &tilingData);
        op.Init(x, seq_lengths, y);
        op.Process();
    } else if constexpr (TEMPLATE_MODE == TPL_MODE_TEMPLATE_BS && DYTPE_MODE == TPL_MODE_DTYPE_B8 && ADDR_MODE == TPL_MODE_ADDR_INT64) {
        GET_TILING_DATA_WITH_STRUCT(ReverseSequenceBSTilingData, tilingData, tiling);
        ReverseSequenceBS<int8_t, DTYPE_SEQ_LENGTHS, uint64_t> op(&pipeBase, &tilingData);
        op.Init(x, seq_lengths, y);
        op.Process();
    } else if constexpr (TEMPLATE_MODE == TPL_MODE_TEMPLATE_BS && DYTPE_MODE == TPL_MODE_DTYPE_B16 && ADDR_MODE == TPL_MODE_ADDR_INT32) {
        GET_TILING_DATA_WITH_STRUCT(ReverseSequenceBSTilingData, tilingData, tiling);
        ReverseSequenceBS<int16_t, DTYPE_SEQ_LENGTHS, uint32_t> op(&pipeBase, &tilingData);
        op.Init(x, seq_lengths, y);
        op.Process();
    } else if constexpr (TEMPLATE_MODE == TPL_MODE_TEMPLATE_BS && DYTPE_MODE == TPL_MODE_DTYPE_B16 && ADDR_MODE == TPL_MODE_ADDR_INT64) {
        GET_TILING_DATA_WITH_STRUCT(ReverseSequenceBSTilingData, tilingData, tiling);
        ReverseSequenceBS<int16_t, DTYPE_SEQ_LENGTHS, uint64_t> op(&pipeBase, &tilingData);
        op.Init(x, seq_lengths, y);
        op.Process();
    } else if constexpr (TEMPLATE_MODE == TPL_MODE_TEMPLATE_BS && DYTPE_MODE == TPL_MODE_DTYPE_B32 && ADDR_MODE == TPL_MODE_ADDR_INT32) {
        GET_TILING_DATA_WITH_STRUCT(ReverseSequenceBSTilingData, tilingData, tiling);
        ReverseSequenceBS<int32_t, DTYPE_SEQ_LENGTHS, uint32_t> op(&pipeBase, &tilingData);
        op.Init(x, seq_lengths, y);
        op.Process();
    } else if constexpr (TEMPLATE_MODE == TPL_MODE_TEMPLATE_BS && DYTPE_MODE == TPL_MODE_DTYPE_B32 && ADDR_MODE == TPL_MODE_ADDR_INT64) {
        GET_TILING_DATA_WITH_STRUCT(ReverseSequenceBSTilingData, tilingData, tiling);
        ReverseSequenceBS<int32_t, DTYPE_SEQ_LENGTHS, uint64_t> op(&pipeBase, &tilingData);
        op.Init(x, seq_lengths, y);
        op.Process();
    } else if constexpr (TEMPLATE_MODE == TPL_MODE_TEMPLATE_BS && DYTPE_MODE == TPL_MODE_DTYPE_B64 && ADDR_MODE == TPL_MODE_ADDR_INT32) {
        GET_TILING_DATA_WITH_STRUCT(ReverseSequenceBSTilingData, tilingData, tiling);
        ReverseSequenceBS<int64_t, DTYPE_SEQ_LENGTHS, uint32_t> op(&pipeBase, &tilingData);
        op.Init(x, seq_lengths, y);
        op.Process();
    } else if constexpr (TEMPLATE_MODE == TPL_MODE_TEMPLATE_BS && DYTPE_MODE == TPL_MODE_DTYPE_B64 && ADDR_MODE == TPL_MODE_ADDR_INT64) {
        GET_TILING_DATA_WITH_STRUCT(ReverseSequenceBSTilingData, tilingData, tiling);
        ReverseSequenceBS<int64_t, DTYPE_SEQ_LENGTHS, uint64_t> op(&pipeBase, &tilingData);
        op.Init(x, seq_lengths, y);
        op.Process();
    } else if constexpr (TEMPLATE_MODE == TPL_MODE_TEMPLATE_SBA && DYTPE_MODE == TPL_MODE_DTYPE_B8 && ADDR_MODE == TPL_MODE_ADDR_INT32) {
        GET_TILING_DATA_WITH_STRUCT(ReverseSequenceA1SBATilingData, tilingData, tiling);
        ReverseSequenceSBA<int8_t, DTYPE_SEQ_LENGTHS, uint32_t> op(&pipeBase, &tilingData);
        op.Init(x, seq_lengths, y);
        op.Process();
    } else if constexpr (TEMPLATE_MODE == TPL_MODE_TEMPLATE_SBA && DYTPE_MODE == TPL_MODE_DTYPE_B8 && ADDR_MODE == TPL_MODE_ADDR_INT64) {
        GET_TILING_DATA_WITH_STRUCT(ReverseSequenceA1SBATilingData, tilingData, tiling);
        ReverseSequenceSBA<int8_t, DTYPE_SEQ_LENGTHS, uint64_t> op(&pipeBase, &tilingData);
        op.Init(x, seq_lengths, y);
        op.Process();
    } else if constexpr (TEMPLATE_MODE == TPL_MODE_TEMPLATE_SBA && DYTPE_MODE == TPL_MODE_DTYPE_B16 && ADDR_MODE == TPL_MODE_ADDR_INT32) {
        GET_TILING_DATA_WITH_STRUCT(ReverseSequenceA1SBATilingData, tilingData, tiling);
        ReverseSequenceSBA<int16_t, DTYPE_SEQ_LENGTHS, uint32_t> op(&pipeBase, &tilingData);
        op.Init(x, seq_lengths, y);
        op.Process();
    } else if constexpr (TEMPLATE_MODE == TPL_MODE_TEMPLATE_SBA && DYTPE_MODE == TPL_MODE_DTYPE_B16 && ADDR_MODE == TPL_MODE_ADDR_INT64) {
        GET_TILING_DATA_WITH_STRUCT(ReverseSequenceA1SBATilingData, tilingData, tiling);
        ReverseSequenceSBA<int16_t, DTYPE_SEQ_LENGTHS, uint64_t> op(&pipeBase, &tilingData);
        op.Init(x, seq_lengths, y);
        op.Process();
    } else if constexpr (TEMPLATE_MODE == TPL_MODE_TEMPLATE_SBA && DYTPE_MODE == TPL_MODE_DTYPE_B32 && ADDR_MODE == TPL_MODE_ADDR_INT32) {
        GET_TILING_DATA_WITH_STRUCT(ReverseSequenceA1SBATilingData, tilingData, tiling);
        ReverseSequenceSBA<int32_t, DTYPE_SEQ_LENGTHS, uint32_t> op(&pipeBase, &tilingData);
        op.Init(x, seq_lengths, y);
        op.Process();
    } else if constexpr (TEMPLATE_MODE == TPL_MODE_TEMPLATE_SBA && DYTPE_MODE == TPL_MODE_DTYPE_B32 && ADDR_MODE == TPL_MODE_ADDR_INT64) {
        GET_TILING_DATA_WITH_STRUCT(ReverseSequenceA1SBATilingData, tilingData, tiling);
        ReverseSequenceSBA<int32_t, DTYPE_SEQ_LENGTHS, uint64_t> op(&pipeBase, &tilingData);
        op.Init(x, seq_lengths, y);
        op.Process();
    } else if constexpr (TEMPLATE_MODE == TPL_MODE_TEMPLATE_SBA && DYTPE_MODE == TPL_MODE_DTYPE_B64 && ADDR_MODE == TPL_MODE_ADDR_INT32) {
        GET_TILING_DATA_WITH_STRUCT(ReverseSequenceA1SBATilingData, tilingData, tiling);
        ReverseSequenceSBA<int64_t, DTYPE_SEQ_LENGTHS, uint32_t> op(&pipeBase, &tilingData);
        op.Init(x, seq_lengths, y);
        op.Process();
    } else if constexpr (TEMPLATE_MODE == TPL_MODE_TEMPLATE_SBA && DYTPE_MODE == TPL_MODE_DTYPE_B64 && ADDR_MODE == TPL_MODE_ADDR_INT64) {
        GET_TILING_DATA_WITH_STRUCT(ReverseSequenceA1SBATilingData, tilingData, tiling);
        ReverseSequenceSBA<int64_t, DTYPE_SEQ_LENGTHS, uint64_t> op(&pipeBase, &tilingData);
        op.Init(x, seq_lengths, y);
        op.Process();
    } else if constexpr (TEMPLATE_MODE == TPL_MODE_TEMPLATE_BAS && DYTPE_MODE == TPL_MODE_DTYPE_B64 && ADDR_MODE == TPL_MODE_ADDR_INT64) {
        GET_TILING_DATA_WITH_STRUCT(ReverseSequenceBASTilingData, tilingData, tiling);
        ReverseSequenceBAS<int64_t, DTYPE_SEQ_LENGTHS> op(&pipeBase, &tilingData);
        op.Init(x, seq_lengths, y);
        op.Process();
    } else if constexpr (TEMPLATE_MODE == TPL_MODE_TEMPLATE_BAS && DYTPE_MODE == TPL_MODE_DTYPE_B32 && ADDR_MODE == TPL_MODE_ADDR_INT64) {
        GET_TILING_DATA_WITH_STRUCT(ReverseSequenceBASTilingData, tilingData, tiling);
        ReverseSequenceBAS<int32_t, DTYPE_SEQ_LENGTHS> op(&pipeBase, &tilingData);
        op.Init(x, seq_lengths, y);
        op.Process();
    } else if constexpr (TEMPLATE_MODE == TPL_MODE_TEMPLATE_BAS && DYTPE_MODE == TPL_MODE_DTYPE_B16 && ADDR_MODE == TPL_MODE_ADDR_INT64) {
        GET_TILING_DATA_WITH_STRUCT(ReverseSequenceBASTilingData, tilingData, tiling);
        ReverseSequenceBAS<int16_t, DTYPE_SEQ_LENGTHS> op(&pipeBase, &tilingData);
        op.Init(x, seq_lengths, y);
        op.Process();
    } else if constexpr (TEMPLATE_MODE == TPL_MODE_TEMPLATE_BAS && DYTPE_MODE == TPL_MODE_DTYPE_B8 && ADDR_MODE == TPL_MODE_ADDR_INT64) {
        GET_TILING_DATA_WITH_STRUCT(ReverseSequenceBASTilingData, tilingData, tiling);
        ReverseSequenceBAS<int8_t, DTYPE_SEQ_LENGTHS> op(&pipeBase, &tilingData);
        op.Init(x, seq_lengths, y);
        op.Process();
    }
}
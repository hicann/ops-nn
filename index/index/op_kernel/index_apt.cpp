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
 * \file index.cpp
 * \brief index full load
 */

#include "arch35/index.h"
#include "arch35/index_perf.h"
#include "arch35/index_simd.h"
#include "arch35/index_full_load.h"
#include "arch35/index_no_continuous.h"
#include "arch35/index_perf_no_continuous.h"
#include "arch35/index_tiling_key.h"
#include "arch35/index_tiling_data.h"

using namespace Index;

template <uint32_t SIZE>
struct ComputeTypeBySize;
// X_DTYPE 0 is used by SIMD tiling as a placeholder
template <>
struct ComputeTypeBySize<INDEX_TPL_NONE> {
    using type = int8_t;
};
template <>
struct ComputeTypeBySize<INDEX_TPL_B8> {
    using type = int8_t;
};
template <>
struct ComputeTypeBySize<INDEX_TPL_B16> {
    using type = half;
};
template <>
struct ComputeTypeBySize<INDEX_TPL_B32> {
    using type = float;
};
template <>
struct ComputeTypeBySize<INDEX_TPL_B64> {
    using type = int64_t;
};
template <>
struct ComputeTypeBySize<INDEX_TPL_B128> {
    using type = int4;
};

template <uint32_t X_DTYPE, uint32_t FULL_LOAD_TYPE, bool IS_PERF, bool IS_SIMD, bool IS_NOCON, bool IS_ACCUMULATE,
          bool IS_OVERLENGTH>
__global__ __aicore__ void index(GM_ADDR inputX, GM_ADDR indexedSizes, GM_ADDR indexedStrides, GM_ADDR indices,
                                 GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    if (workspace == nullptr) {
        return;
    }
    SetSysWorkspace(workspace);
    GM_ADDR userWS = GetUserWorkspace(workspace);
    if (userWS == nullptr) {
        return;
    }
    TPipe tPipe;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(IndexSimtTilingData);

    using inputComputeType = typename ComputeTypeBySize<X_DTYPE>::type;
    using indexOffsetType = std::conditional_t<IS_OVERLENGTH, uint64_t, uint32_t>;

    // ---------------- SIMT -------------------------
    if constexpr (FULL_LOAD_TYPE == INDEX_NOT_FULL_LOAD && !IS_NOCON && !IS_PERF && !IS_SIMD) {
        REGISTER_TILING_FOR_TILINGKEY(
            "FULL_LOAD_TYPE == INDEX_NOT_FULL_LOAD && IS_NOCON == false && IS_PERF == false && IS_SIMD == false",
            IndexSimtTilingData);
        GET_TILING_DATA_WITH_STRUCT(IndexSimtTilingData, tilingData, tiling);
        KernelIndex<inputComputeType, IndexAssign<inputComputeType, indexOffsetType>, DTYPE_INDICES, indexOffsetType>
            op;
        op.Init(output, inputX, indexedSizes, indexedStrides, indices, tilingData);
        op.Process();
        // ---------------- Perf SIMT --------------------
    } else if constexpr (FULL_LOAD_TYPE == INDEX_NOT_FULL_LOAD && !IS_NOCON && IS_PERF && !IS_SIMD) {
        REGISTER_TILING_FOR_TILINGKEY(
            "FULL_LOAD_TYPE == INDEX_NOT_FULL_LOAD && IS_NOCON == false && IS_PERF == true && IS_SIMD == false",
            IndexPerfSimtTilingData);
        Process<inputComputeType, DTYPE_INDICES, indexOffsetType>(
            (__gm__ inputComputeType*)output, (__gm__ inputComputeType*)inputX, indices,
            (__gm__ const IndexPerfSimtTilingData* __restrict)tiling);
        // ---------------- SIMD -------------------------
    } else if constexpr (IS_SIMD) {
        REGISTER_TILING_FOR_TILINGKEY("IS_SIMD == true", IndexSimdTilingData);
        GET_TILING_DATA_WITH_STRUCT(IndexSimdTilingData, tilingData, tiling);
        IndexSimd<DTYPE_INDICES> op;
        op.Init(output, inputX, indexedSizes, indexedStrides, indices, &tilingData);
        op.Process();
        // ---------------- Full-load --------------------
    } else if constexpr (FULL_LOAD_TYPE == INDEX_FULL_LOAD_1D) {
        REGISTER_TILING_FOR_TILINGKEY("FULL_LOAD_TYPE == INDEX_FULL_LOAD_1D", IndexFullLoadTilingData);
        GET_TILING_DATA_WITH_STRUCT(IndexFullLoadTilingData, tilingData, tiling);
        KernelIndexFullLoad<inputComputeType, int32_t, DTYPE_INDICES, 1, 1> op(tPipe, tilingData);
        op.Init(inputX, indexedSizes, indices, output);
        op.Process();
    } else if constexpr (FULL_LOAD_TYPE == INDEX_FULL_LOAD_2D_MASK_0) {
        REGISTER_TILING_FOR_TILINGKEY("FULL_LOAD_TYPE == INDEX_FULL_LOAD_2D_MASK_0", IndexFullLoadTilingData);
        GET_TILING_DATA_WITH_STRUCT(IndexFullLoadTilingData, tilingData, tiling);
        KernelIndexFullLoad<inputComputeType, int32_t, DTYPE_INDICES, 0, DIM2> op(tPipe, tilingData);
        op.Init(inputX, indexedSizes, indices, output);
        op.Process();
    } else if constexpr (FULL_LOAD_TYPE == INDEX_FULL_LOAD_2D_MASK_1) {
        REGISTER_TILING_FOR_TILINGKEY("FULL_LOAD_TYPE == INDEX_FULL_LOAD_2D_MASK_1", IndexFullLoadTilingData);
        GET_TILING_DATA_WITH_STRUCT(IndexFullLoadTilingData, tilingData, tiling);
        KernelIndexFullLoad<inputComputeType, int32_t, DTYPE_INDICES, 1, DIM2> op(tPipe, tilingData);
        op.Init(inputX, indexedSizes, indices, output);
        op.Process();
        // ---------------- Non-contiguous ---------------
    } else if constexpr (FULL_LOAD_TYPE == INDEX_NOT_FULL_LOAD && IS_NOCON && !IS_PERF && !IS_SIMD) {
        REGISTER_TILING_FOR_TILINGKEY(
            "FULL_LOAD_TYPE == INDEX_NOT_FULL_LOAD && IS_NOCON == true && IS_PERF == false && IS_SIMD == false",
            IndexNonContinuousTilingData);
        GET_TILING_DATA_WITH_STRUCT(IndexNonContinuousTilingData, tilingData, tiling);
        KernelIndexNoContiguous<inputComputeType, IndexAssign<inputComputeType, indexOffsetType>, DTYPE_INDICES,
                                indexOffsetType>
            op;
        op.Init(output, inputX, indexedSizes, indexedStrides, indices, tilingData);
        op.Process();
    } else if constexpr (FULL_LOAD_TYPE == INDEX_NOT_FULL_LOAD && IS_NOCON && IS_PERF && !IS_SIMD) {
        REGISTER_TILING_FOR_TILINGKEY(
            "FULL_LOAD_TYPE == INDEX_NOT_FULL_LOAD && IS_NOCON == true && IS_PERF == true && IS_SIMD == false",
            IndexNonContinuousTilingData);
        GET_TILING_DATA_WITH_STRUCT(IndexNonContinuousTilingData, tilingData, tiling);
        KernelIndexNoContiguousPerf<inputComputeType, DTYPE_INDICES, indexOffsetType> op(tPipe);
        op.Init(output, inputX, indices, tilingData);
        op.Process();
    }
}

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
 * \file index_put_v2.cpp
 * \brief Ascendc IndexPutV2 kernel
 */

#include <type_traits>
#include "../index/arch35/index.h"
#include "../index/arch35/index_no_continuous.h"
#include "arch35/index_put_v2_simd.h"
#include "arch35/index_put_v2_tiling_key.h"
#include "arch35/index_put_v2_tiling_data.h"

using namespace IndexPutV2;

template <uint32_t SIZE>
struct ComputeTypeBySize;
template <> struct ComputeTypeBySize<INDEX_PUT_V2_TPL_B8> { using type = int8_t; };
template <> struct ComputeTypeBySize<INDEX_PUT_V2_TPL_B16> { using type = half; };
template <> struct ComputeTypeBySize<INDEX_PUT_V2_TPL_B32> { using type = float; };
template <> struct ComputeTypeBySize<INDEX_PUT_V2_TPL_B64> { using type = int64_t; };
template <> struct ComputeTypeBySize<INDEX_PUT_V2_TPL_B128> { using type = int4; };

template <uint32_t X_DTYPE, uint32_t FULL_LOAD_TYPE, bool IS_PERF, bool IS_SIMD, bool IS_NOCON,
    bool IS_ACCUMULATE, bool IS_OVERLENGTH>
__global__ __aicore__ void index_put_v2(GM_ADDR inputX, GM_ADDR value, GM_ADDR indexedSizes,
    GM_ADDR indexedStrides, GM_ADDR indices, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    if (workspace == nullptr) {
        return;
    }
    SetSysWorkspace(workspace);
    GM_ADDR userWS = GetUserWorkspace(workspace);
    if (userWS == nullptr) {
        return;
    }
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(Index::IndexSimtTilingData);

    using indexOffsetType = std::conditional_t<IS_OVERLENGTH, uint64_t, uint32_t>;

    // ---------------- SIMT -------------------------
    if constexpr (!IS_NOCON && !IS_SIMD && !IS_ACCUMULATE) {
        REGISTER_TILING_FOR_TILINGKEY(
            "FULL_LOAD_TYPE == 0 && IS_NOCON == false && IS_SIMD == false && IS_ACCUMULATE == false",
            Index::IndexSimtTilingData);
        GET_TILING_DATA_WITH_STRUCT(Index::IndexSimtTilingData, tilingData, tiling);
        using inputComputeType = typename ComputeTypeBySize<X_DTYPE>::type;
        Index::KernelIndex<inputComputeType, Index::IndexPutAssign<inputComputeType>, DTYPE_INDICES, indexOffsetType> op;
        op.Init(output, inputX, indexedSizes, indexedStrides, indices, tilingData, value);
        op.Process();
    } else if constexpr (!IS_NOCON && !IS_SIMD && IS_ACCUMULATE &&
        !std::is_same<DTYPE_X, int4>::value && !std::is_same<DTYPE_X, int8_t>::value &&
        !std::is_same<DTYPE_X, uint8_t>::value) {
        REGISTER_TILING_FOR_TILINGKEY(
            "FULL_LOAD_TYPE == 0 && IS_NOCON == false && IS_SIMD == false && IS_ACCUMULATE == true",
            Index::IndexSimtTilingData);
        GET_TILING_DATA_WITH_STRUCT(Index::IndexSimtTilingData, tilingData, tiling);
        Index::KernelIndex<DTYPE_X, Index::IndexPutAdd<DTYPE_X>, DTYPE_INDICES, indexOffsetType> op;
        op.Init(output, inputX, indexedSizes, indexedStrides, indices, tilingData, value);
        op.Process();
    // ---------------- Non-contiguous ---------------
    } else if constexpr (IS_NOCON && !IS_ACCUMULATE &&
        !std::is_same<DTYPE_X, int4>::value) {
        REGISTER_TILING_FOR_TILINGKEY(
            "FULL_LOAD_TYPE == 0 && IS_NOCON == true && IS_SIMD == false && IS_ACCUMULATE == false",
            Index::IndexNonContinuousTilingData);
        GET_TILING_DATA_WITH_STRUCT(Index::IndexNonContinuousTilingData, tilingData, tiling);
        Index::KernelIndexNoContiguous<DTYPE_X, Index::IndexPutAssign<DTYPE_X>, DTYPE_INDICES, indexOffsetType> op;
        op.Init(output, inputX, indexedSizes, indexedStrides, indices, tilingData, value);
        op.Process();
    } else if constexpr (IS_NOCON && IS_ACCUMULATE &&
        !std::is_same<DTYPE_X, int4>::value && !std::is_same<DTYPE_X, int8_t>::value &&
        !std::is_same<DTYPE_X, uint8_t>::value) {
        REGISTER_TILING_FOR_TILINGKEY(
            "FULL_LOAD_TYPE == 0 && IS_NOCON == true && IS_SIMD == false && IS_ACCUMULATE == true",
            Index::IndexNonContinuousTilingData);
        GET_TILING_DATA_WITH_STRUCT(Index::IndexNonContinuousTilingData, tilingData, tiling);
        Index::KernelIndexNoContiguous<DTYPE_X, Index::IndexPutAdd<DTYPE_X>, DTYPE_INDICES, indexOffsetType> op;
        op.Init(output, inputX, indexedSizes, indexedStrides, indices, tilingData, value);
        op.Process();
    // ---------------- SIMD -------------------------
    } else if constexpr (IS_SIMD && !IS_ACCUMULATE) {
        REGISTER_TILING_FOR_TILINGKEY(
            "FULL_LOAD_TYPE == 0 && IS_SIMD == true && IS_NOCON == false && IS_ACCUMULATE == false",
            IndexPutV2SimdTilingData);
        GET_TILING_DATA_WITH_STRUCT(IndexPutV2SimdTilingData, tilingData, tiling);
        IndexPutV2Simd<DTYPE_X, DTYPE_INDICES, false> op;
        op.Init(inputX, value, indexedSizes, indexedStrides, indices, tilingData);
        op.Process();
    } else if constexpr (IS_SIMD && IS_ACCUMULATE &&
        !std::is_same<DTYPE_X, uint8_t>::value && !std::is_same<DTYPE_X, int64_t>::value &&
        !std::is_same<DTYPE_X, bool>::value) {
        REGISTER_TILING_FOR_TILINGKEY(
            "FULL_LOAD_TYPE == 0 && IS_SIMD == true && IS_NOCON == false && IS_ACCUMULATE == true",
            IndexPutV2SimdTilingData);
        GET_TILING_DATA_WITH_STRUCT(IndexPutV2SimdTilingData, tilingData, tiling);
        IndexPutV2Simd<DTYPE_X, DTYPE_INDICES, true> op;
        op.Init(inputX, value, indexedSizes, indexedStrides, indices, tilingData);
        op.Process();
    }
}

/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file inplace_index_fill_apt.cpp
 * \brief Kernel entry point with template dispatch
 */

#include "arch35/inplace_index_fill_simt.h"
#include "arch35/inplace_index_fill_simt_dense_indices.h"
#include "arch35/inplace_index_fill_simd.h"
#include "arch35/inplace_index_fill_struct.h"
#include "arch35/inplace_index_fill_tiling_key.h"

using namespace AscendC;
using namespace InplaceIndexFill;

static constexpr int64_t B64 = 8;
template <typename T>
struct ComputeTypeGet {
    using type = typename std::conditional<sizeof(T) == B64, int64_t, T>::type;
};

template <typename xType>
__aicore__ inline void DispatchSimd(
    InplaceIndexFillSimdTilingData& tilingData,
    GM_ADDR x, GM_ADDR indices, GM_ADDR value, GM_ADDR workspace, AscendC::TPipe& pipe)
{
    InplaceIndexFillSimd<xType, DTYPE_INDICES> op(tilingData, pipe);
    op.Init(x, indices, value, workspace);
    op.Process();
}

template <typename xType, typename COM_T>
__aicore__ inline void DispatchSimt(
    InplaceIndexFillSimtTilingData& tilingData,
    GM_ADDR x, GM_ADDR indices, GM_ADDR value, GM_ADDR workspace)
{
    InplaceIndexFillSimt::InplaceIndexFillSimtImpl<xType, DTYPE_INDICES, COM_T> op(&tilingData);
    op.Init(indices, value, workspace);
    op.Process((__gm__ xType*)(x), workspace);
}

template <typename xType, typename COM_T>
__aicore__ inline void DispatchDenseIndices(
    InplaceIndexFillSimtDenseIndicesTilingData& tilingData,
    GM_ADDR x, GM_ADDR indices, GM_ADDR value, GM_ADDR workspace, AscendC::TPipe& pipe)
{
    InplaceIndexFillDenseIndices::InplaceIndexFillDenseIndicesImpl<xType, DTYPE_INDICES, COM_T> op(&tilingData, &pipe);
    op.Init(indices, value, workspace);
    op.Process((__gm__ xType*)(x));
}

template <uint64_t TEMPLATE_MODE, uint64_t DTYPE_MODE>
__global__ __aicore__ void inplace_index_fill(
    GM_ADDR x, GM_ADDR indices, GM_ADDR value, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    AscendC::TPipe pipe;
    using xType = typename ComputeTypeGet<DTYPE_X>::type;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(InplaceIndexFillSimtTilingData);

    if constexpr (TEMPLATE_MODE == TPL_MODE_TEMPLATE_SIMD && DTYPE_MODE == TPL_MODE_DTYPE_B32) {
        REGISTER_TILING_FOR_TILINGKEY(
            "TEMPLATE_MODE == TPL_MODE_TEMPLATE_SIMD && DTYPE_MODE == TPL_MODE_DTYPE_B32",
            InplaceIndexFillSimdTilingData);
        GET_TILING_DATA_WITH_STRUCT(InplaceIndexFillSimdTilingData, tilingData, tiling);
        DispatchSimd<xType>(tilingData, x, indices, value, workspace, pipe);
    } else if constexpr (TEMPLATE_MODE == TPL_MODE_TEMPLATE_SIMD && DTYPE_MODE == TPL_MODE_DTYPE_B64) {
        REGISTER_TILING_FOR_TILINGKEY(
            "TEMPLATE_MODE == TPL_MODE_TEMPLATE_SIMD && DTYPE_MODE == TPL_MODE_DTYPE_B64",
            InplaceIndexFillSimdTilingData);
        GET_TILING_DATA_WITH_STRUCT(InplaceIndexFillSimdTilingData, tilingData, tiling);
        DispatchSimd<xType>(tilingData, x, indices, value, workspace, pipe);
    } else if constexpr (TEMPLATE_MODE == TPL_MODE_TEMPLATE_SIMT && DTYPE_MODE == TPL_MODE_DTYPE_B32) {
        REGISTER_TILING_FOR_TILINGKEY(
            "TEMPLATE_MODE == TPL_MODE_TEMPLATE_SIMT && DTYPE_MODE == TPL_MODE_DTYPE_B32",
            InplaceIndexFillSimtTilingData);
        GET_TILING_DATA_WITH_STRUCT(InplaceIndexFillSimtTilingData, tilingData, tiling);
        DispatchSimt<xType, uint32_t>(tilingData, x, indices, value, workspace);
    } else if constexpr (TEMPLATE_MODE == TPL_MODE_TEMPLATE_SIMT && DTYPE_MODE == TPL_MODE_DTYPE_B64) {
        REGISTER_TILING_FOR_TILINGKEY(
            "TEMPLATE_MODE == TPL_MODE_TEMPLATE_SIMT && DTYPE_MODE == TPL_MODE_DTYPE_B64",
            InplaceIndexFillSimtTilingData);
        GET_TILING_DATA_WITH_STRUCT(InplaceIndexFillSimtTilingData, tilingData, tiling);
        DispatchSimt<xType, uint64_t>(tilingData, x, indices, value, workspace);
    } else if constexpr (TEMPLATE_MODE == TPL_MODE_TEMPLATE_SIMT_DENSE_INDICES && DTYPE_MODE == TPL_MODE_DTYPE_B32) {
        REGISTER_TILING_FOR_TILINGKEY(
            "TEMPLATE_MODE == TPL_MODE_TEMPLATE_SIMT_DENSE_INDICES && DTYPE_MODE == TPL_MODE_DTYPE_B32",
            InplaceIndexFillSimtDenseIndicesTilingData);
        GET_TILING_DATA_WITH_STRUCT(InplaceIndexFillSimtDenseIndicesTilingData, tilingData, tiling);
        DispatchDenseIndices<xType, uint32_t>(tilingData, x, indices, value, workspace, pipe);
    } else if constexpr (TEMPLATE_MODE == TPL_MODE_TEMPLATE_SIMT_DENSE_INDICES && DTYPE_MODE == TPL_MODE_DTYPE_B64) {
        REGISTER_TILING_FOR_TILINGKEY(
            "TEMPLATE_MODE == TPL_MODE_TEMPLATE_SIMT_DENSE_INDICES && DTYPE_MODE == TPL_MODE_DTYPE_B64",
            InplaceIndexFillSimtDenseIndicesTilingData);
        GET_TILING_DATA_WITH_STRUCT(InplaceIndexFillSimtDenseIndicesTilingData, tilingData, tiling);
        DispatchDenseIndices<xType, uint64_t>(tilingData, x, indices, value, workspace, pipe);
    }
}
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
 * \brief
 */

#include "arch35/inplace_index_fill_simt.h"
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

template <uint64_t TEMPLATE_MODE, uint64_t ADDR_MODE>
__global__ __aicore__ void inplace_index_fill(
    GM_ADDR x, GM_ADDR indices, GM_ADDR value, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    AscendC::TPipe pipe;
    using xType = typename ComputeTypeGet<DTYPE_X>::type;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(InplaceIndexFillSimtTilingData);

    if constexpr (TEMPLATE_MODE == TPL_MODE_SIMD && ADDR_MODE == TPL_MODE_ADDR_INT64) {
        REGISTER_TILING_FOR_TILINGKEY(
            "TEMPLATE_MODE == TPL_MODE_SIMD && ADDR_MODE == TPL_MODE_ADDR_INT64",
            InplaceIndexFillSimdTilingData);
        GET_TILING_DATA_WITH_STRUCT(InplaceIndexFillSimdTilingData, tilingData, tiling);
        InplaceIndexFillSimd<xType, DTYPE_INDICES> op(tilingData, pipe);
        op.Init(x, indices, value, workspace);
        op.Process();
    } else if constexpr (TEMPLATE_MODE == TPL_MODE_SIMT && ADDR_MODE == TPL_MODE_ADDR_INT32) {
        REGISTER_TILING_FOR_TILINGKEY(
            "TEMPLATE_MODE == TPL_MODE_SIMT && ADDR_MODE == TPL_MODE_ADDR_INT32",
            InplaceIndexFillSimtTilingData);
        GET_TILING_DATA_WITH_STRUCT(InplaceIndexFillSimtTilingData, tilingData, tiling);
        InplaceIndexFillSimt::InplaceIndexFillSimtImpl<xType, DTYPE_INDICES, uint32_t> op(&tilingData);
        op.Init(value);
        op.Process(x, indices);
    } else if constexpr (TEMPLATE_MODE == TPL_MODE_SIMT && ADDR_MODE == TPL_MODE_ADDR_INT64) {
        REGISTER_TILING_FOR_TILINGKEY(
            "TEMPLATE_MODE == TPL_MODE_SIMT && ADDR_MODE == TPL_MODE_ADDR_INT64",
            InplaceIndexFillSimtTilingData);
        GET_TILING_DATA_WITH_STRUCT(InplaceIndexFillSimtTilingData, tilingData, tiling);
        InplaceIndexFillSimt::InplaceIndexFillSimtImpl<xType, DTYPE_INDICES, uint64_t> op(&tilingData);
        op.Init(value);
        op.Process(x, indices);
    }
}
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
 * \file adaptive_max_pool3d_apt.cpp
 * \brief
 */

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "../adaptive_pool3d_common/arch35/adaptive_max_pool3d_big_kernel.h"
#include "../adaptive_pool3d_common/arch35/adaptive_max_pool3d_simt.h"
#include "../adaptive_pool3d_common/arch35/adaptive_pool3d_tiling_struct.h"

using maskdType = uint8_t;
template <uint64_t TEMPLATE_MODE, uint64_t DYTPE_MODE, uint64_t MULTI_MODE>
__global__ __aicore__ void adaptive_max_pool3d(
    GM_ADDR x, GM_ADDR y, GM_ADDR indices, GM_ADDR workspace, GM_ADDR tiling)
{
    using namespace AdaptivePool3DTiling;
    AscendC::TPipe pipeBase;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(AdaptivePool3DTiling::AdaptivePool3dBigKernelTilingData);
    if constexpr (TEMPLATE_MODE == TPL_MODE_1 && DYTPE_MODE == TPL_DYTPE_0 && MULTI_MODE == TPL_MULTI_MODE_0) {
        REGISTER_TILING_FOR_TILINGKEY(
        "TEMPLATE_MODE == TPL_MODE_1 && DYTPE_MODE == TPL_DYTPE_0 && MULTI_MODE == TPL_MULTI_MODE_0", AdaptivePool3DTiling::AdaptivePool3dBigKernelTilingData);
        GET_TILING_DATA_WITH_STRUCT(AdaptivePool3DTiling::AdaptivePool3dBigKernelTilingData, tilingData, tiling);
        AdaptivePool3d::AdaptiveMaxPool3dBigKernel<DTYPE_X, DTYPE_INDICES> op(tilingData, pipeBase);
        op.Init(x, y, indices);
        op.Process();
 	}else if constexpr (TEMPLATE_MODE == TPL_MODE_2 && DYTPE_MODE == TPL_INT32_UINT32 && MULTI_MODE == TPL_MULTI_MODE_0 ) {
        REGISTER_TILING_FOR_TILINGKEY(
        "TEMPLATE_MODE == TPL_MODE_2 && DYTPE_MODE == TPL_INT32_UINT32 && MULTI_MODE == TPL_MULTI_MODE_0", AdaptivePool3DTiling::AdaptivePool3DSimtTilingData);
        GET_TILING_DATA_WITH_STRUCT(AdaptivePool3DTiling::AdaptivePool3DSimtTilingData, tilingData, tiling);
        AdaptivePool3DWithSimt::AdaptivePool3DSimt<DTYPE_X, DTYPE_INDICES, int32_t, uint32_t> op(&pipeBase, &tilingData);
        op.Init(x, y, indices);
        op.Process();
    }else if constexpr (TEMPLATE_MODE == TPL_MODE_2 && DYTPE_MODE == TPL_INT64_UINT32 && MULTI_MODE == TPL_MULTI_MODE_0 ){
        REGISTER_TILING_FOR_TILINGKEY(
        "TEMPLATE_MODE == TPL_MODE_2 && DYTPE_MODE == TPL_INT64_UINT32 && MULTI_MODE == TPL_MULTI_MODE_0", AdaptivePool3DTiling::AdaptivePool3DSimtTilingData);
        GET_TILING_DATA_WITH_STRUCT(AdaptivePool3DTiling::AdaptivePool3DSimtTilingData, tilingData, tiling);
        AdaptivePool3DWithSimt::AdaptivePool3DSimt<DTYPE_X, DTYPE_INDICES, int64_t, uint32_t> op(&pipeBase, &tilingData);
        op.Init(x, y, indices);
        op.Process();
    }else if constexpr (TEMPLATE_MODE == TPL_MODE_2 && DYTPE_MODE == TPL_INT32_UINT64 && MULTI_MODE == TPL_MULTI_MODE_0 ){
        REGISTER_TILING_FOR_TILINGKEY(
        "TEMPLATE_MODE == TPL_MODE_2 && DYTPE_MODE == TPL_INT32_UINT64 && MULTI_MODE == TPL_MULTI_MODE_0", AdaptivePool3DTiling::AdaptivePool3DSimtTilingData);
        GET_TILING_DATA_WITH_STRUCT(AdaptivePool3DTiling::AdaptivePool3DSimtTilingData, tilingData, tiling);
        AdaptivePool3DWithSimt::AdaptivePool3DSimt<DTYPE_X, DTYPE_INDICES, int32_t, uint64_t> op(&pipeBase, &tilingData);
        op.Init(x, y, indices);
        op.Process();
    }else if constexpr (TEMPLATE_MODE == TPL_MODE_2 && DYTPE_MODE == TPL_INT64_UINT64 && MULTI_MODE == TPL_MULTI_MODE_0 ){
        REGISTER_TILING_FOR_TILINGKEY(
        "TEMPLATE_MODE == TPL_MODE_2 && DYTPE_MODE == TPL_INT64_UINT64 && MULTI_MODE == TPL_MULTI_MODE_0", AdaptivePool3DTiling::AdaptivePool3DSimtTilingData);
        GET_TILING_DATA_WITH_STRUCT(AdaptivePool3DTiling::AdaptivePool3DSimtTilingData, tilingData, tiling);
        AdaptivePool3DWithSimt::AdaptivePool3DSimt<DTYPE_X, DTYPE_INDICES, int64_t, uint64_t> op(&pipeBase, &tilingData);
        op.Init(x, y, indices);
        op.Process();
    }
    return;
}
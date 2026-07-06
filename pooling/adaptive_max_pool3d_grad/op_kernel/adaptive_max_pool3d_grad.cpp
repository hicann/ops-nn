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
 * \file adaptive_max_pool3d_grad.cpp
 * \brief Adaptive MaxPool3D Grad kernel entry
 */

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
#include "arch35/adaptive_max_pool3d_grad_simt.h"
#else
#include "adaptive_max_pool3d_grad_normal.h"
#include "adaptive_max_pool3d_grad_scatter.h"
#include "adaptive_max_pool3d_grad_scatter_overlap.h"
using namespace AdaptiveMaxPool3DGrad;
#endif

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
using namespace AdaptiveMaxPool3dGradOp;
template <uint64_t TEMPLATE_MODE = TPL_SIMT_KERNEL, uint64_t INDEX_DTYPE = TPL_INT32>
__global__ __aicore__ void adaptive_max_pool3d_grad(GM_ADDR x, GM_ADDR grad, GM_ADDR argmax, GM_ADDR y,
                                                    GM_ADDR workspace, GM_ADDR tiling)
{
    if (workspace == nullptr || GetUserWorkspace(workspace) == nullptr || g_coreType == AIC) {
        return;
    }
    TPipe pipe;

    if constexpr (TEMPLATE_MODE == TPL_SIMT_KERNEL && INDEX_DTYPE == TPL_INT32) {
        REGISTER_TILING_DEFAULT(AdaptiveMaxPool3dGradTilingDataV35);
        GET_TILING_DATA_WITH_STRUCT(AdaptiveMaxPool3dGradTilingDataV35, tilingData, tiling);
        AdaptiveMaxPool3dGradSimt<DTYPE_X, int32_t> op(&pipe, &tilingData);
        op.Init(x, grad, argmax, y, workspace);
        op.Process();
    } else if constexpr (TEMPLATE_MODE == TPL_SIMT_KERNEL && INDEX_DTYPE == TPL_INT64) {
        REGISTER_TILING_DEFAULT(AdaptiveMaxPool3dGradTilingDataV35);
        GET_TILING_DATA_WITH_STRUCT(AdaptiveMaxPool3dGradTilingDataV35, tilingData, tiling);
        AdaptiveMaxPool3dGradSimt<DTYPE_X, int64_t> op(&pipe, &tilingData);
        op.Init(x, grad, argmax, y, workspace);
        op.Process();
    }
}

#else

#define GENERAL_OP_IMPL(templateClass, ...)                  \
    do {                                                     \
        GET_TILING_DATA(tilingData, tiling);                 \
        templateClass<__VA_ARGS__> op(&pipe);                \
        op.Init(x, grad, argmax, y, workspace, &tilingData); \
        op.Process();                                        \
    } while (0)

extern "C" __global__ __aicore__ void adaptive_max_pool3d_grad(GM_ADDR x, GM_ADDR grad, GM_ADDR argmax, GM_ADDR y,
                                                               GM_ADDR workspace, GM_ADDR tiling)
{
    if (workspace == nullptr || GetUserWorkspace(workspace) == nullptr || g_coreType == AIC) {
        return;
    }
    TPipe pipe;
    if (TILING_KEY_IS(0)) {
        GENERAL_OP_IMPL(AdaptiveMaxPool3DGradNormal, DTYPE_X, DTYPE_GRAD, DTYPE_ARGMAX, DTYPE_Y, false);
    } else if (TILING_KEY_IS(100)) {
        GENERAL_OP_IMPL(AdaptiveMaxPool3DGradNormal, DTYPE_X, DTYPE_GRAD, DTYPE_ARGMAX, DTYPE_Y, true);
    }
    if (TILING_KEY_IS(2)) {
        GENERAL_OP_IMPL(AdaptiveMaxPool3DGradScatter, DTYPE_X, DTYPE_GRAD, DTYPE_ARGMAX, DTYPE_Y);
    } else if (TILING_KEY_IS(102)) {
        GENERAL_OP_IMPL(AdaptiveMaxPool3DGradScatterOverlap, DTYPE_X, DTYPE_GRAD, DTYPE_ARGMAX, DTYPE_Y);
    }
}

#endif

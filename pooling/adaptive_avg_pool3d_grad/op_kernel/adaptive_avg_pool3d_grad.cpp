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
 * \file adaptive_avg_pool3d_grad.cpp
 * \brief
 */
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
#include "arch35/adaptive_avg_pool3d_grad_simt.h"
#include "arch35/adaptive_avg_pool3d_grad_ncdhw_big_kernel.h"
#include "arch35/adaptive_avg_pool3d_grad_struct.h"
#include "arch35/adaptive_avg_pool3d_grad_ncdhw_small_kernel.h"
#include "arch35/adaptive_avg_pool3d_grad_ksize_one.h"
#else
#include "adaptive_avg_pool3d_grad_float.h"
#include "adaptive_avg_pool3d_grad_cast.h"
#include "adaptive_avg_pool3d_grad_nc_large_cast.h"
#include "adaptive_avg_pool3d_grad_nc_large_float.h"
#endif
using namespace AscendC;
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
using namespace AdaptiveAvgPool3dGradOp;

// 按索引位宽实例化并运行 SIMT 模板。
template <uint64_t INDEX_DTYPE, uint64_t IS_CHANNEL_LAST, uint32_t THREADS>
__aicore__ inline void RunAdaptiveAvgPool3dGradSimt(TPipe* pipe, const AdaptiveAvgPool3dGradTilingDataV35* tilingData,
                                                    GM_ADDR y_grad, GM_ADDR x_grad)
{
    if constexpr (INDEX_DTYPE == TPL_INT32) {
        AdaptiveAvgPool3dGradSimt<DTYPE_X, int32_t, IS_CHANNEL_LAST, THREADS> op(pipe, tilingData);
        op.Init(y_grad, x_grad);
        op.Process();
    } else if constexpr (INDEX_DTYPE == TPL_INT64) {
        AdaptiveAvgPool3dGradSimt<DTYPE_X, int64_t, IS_CHANNEL_LAST, THREADS> op(pipe, tilingData);
        op.Init(y_grad, x_grad);
        op.Process();
    }
}

template <uint64_t TEMPLATE_MODE = TPL_SMALL_KERNEL, uint64_t INDEX_DTYPE = TPL_INT32, uint64_t IS_CHANNEL_LAST = 0>
__global__ __aicore__ void adaptive_avg_pool3d_grad(GM_ADDR y_grad, GM_ADDR x, GM_ADDR x_grad, GM_ADDR workspace,
                                                    GM_ADDR tiling)
{
    if (workspace == nullptr || GetUserWorkspace(workspace) == nullptr || g_coreType == AIC) {
        return;
    }
    TPipe pipe;
    REGISTER_TILING_DEFAULT(AdaptiveAvgPool3dGradTilingDataV35);
    if constexpr (TEMPLATE_MODE == TPL_SIMT_KERNEL) {
        GET_TILING_DATA_WITH_STRUCT(AdaptiveAvgPool3dGradTilingDataV35, tilingData, tiling);
        // 线程宽度按 tiling 下发的档位运行时二选一。
        if (tilingData.threadMode == TPL_THREAD_512) {
            RunAdaptiveAvgPool3dGradSimt<INDEX_DTYPE, IS_CHANNEL_LAST, 512U>(&pipe, &tilingData, y_grad, x_grad);
        } else {
            RunAdaptiveAvgPool3dGradSimt<INDEX_DTYPE, IS_CHANNEL_LAST, 1024U>(&pipe, &tilingData, y_grad, x_grad);
        }
    } else if constexpr (TEMPLATE_MODE == TPL_BIG_KERNEL && IS_CHANNEL_LAST == 0) {
        GET_TILING_DATA_WITH_STRUCT(AdaptiveAvgPool3dNCDHWGradBigKernelTilingDataV35, tilingData, tiling);
        if constexpr (INDEX_DTYPE == TPL_INT32) {
            AdaptiveAvgPool3dGradNCDHWBigKernel<DTYPE_X, int32_t> op;
            op.Init(y_grad, x_grad, pipe, tilingData);
            op.Process();
        } else {
            AdaptiveAvgPool3dGradNCDHWBigKernel<DTYPE_X, int64_t> op;
            op.Init(y_grad, x_grad, pipe, tilingData);
            op.Process();
        }
    } else if constexpr (TEMPLATE_MODE == TPL_SMALL_KERNEL && IS_CHANNEL_LAST == 0) {
        GET_TILING_DATA_WITH_STRUCT(AdaptiveAvgPool3dNCDHWGradSmallKernelTilingDataV35, tilingData, tiling);
        if constexpr (INDEX_DTYPE == TPL_INT32) {
            AdaptiveAvgPool3dGradNCDHWSmallKernel<DTYPE_X, int32_t> op;
            op.Init(y_grad, x_grad, pipe, tilingData);
            op.Process();
        } else {
            AdaptiveAvgPool3dGradNCDHWSmallKernel<DTYPE_X, int64_t> op;
            op.Init(y_grad, x_grad, pipe, tilingData);
            op.Process();
        }
    } else if constexpr (TEMPLATE_MODE == TPL_KSIZE_ONE_KERNEL) {
        GET_TILING_DATA_WITH_STRUCT(AdaptiveAvgPool3dGradKsizeOneTilingDataV35, tilingData, tiling);
        AdaptiveAvgPool3dGradKsizeOne<DTYPE_X> op(&pipe, &tilingData);
        op.Init(y_grad, x_grad);
        op.Process();
    }
}
#else
extern "C" __global__ __aicore__ void adaptive_avg_pool3d_grad(GM_ADDR y_grad, GM_ADDR x, GM_ADDR x_grad,
                                                               GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    if (TILING_KEY_IS(0)) {
        TPipe pipe;
        KernelAdaptiveAvgPool3DGradFloat<float> op;
        op.Init(y_grad, x_grad, workspace, &tiling_data, &pipe);
        op.InitBuffer();
        op.GetLocalTensor();
        op.ClearOutput();
        op.Process();
        op.ReleaseEventID();
    } else if (TILING_KEY_IS(10)) {
        TPipe pipe;
        KernelAdaptiveAvgPool3DGradCast<half> op;
        op.Init(y_grad, x_grad, workspace, &tiling_data, &pipe);
        op.InitBuffer();
        op.GetLocalTensor();
        op.ClearOutput();
        op.Process();
        op.ReleaseEventID();
    } else if (TILING_KEY_IS(20)) {
        TPipe pipe;
        KernelAdaptiveAvgPool3DGradCast<bfloat16_t> op;
        op.Init(y_grad, x_grad, workspace, &tiling_data, &pipe);
        op.InitBuffer();
        op.GetLocalTensor();
        op.ClearOutput();
        op.Process();
        op.ReleaseEventID();
    } else if (TILING_KEY_IS(1)) {
        TPipe pipe;
        KernelAdaptiveAvgPool3DGradNcLargeFloat<float> op;
        op.Init(y_grad, x_grad, workspace, &tiling_data, &pipe);
        op.InitBuffer();
        op.GetLocalTensor();
        op.ClearOutput();
        op.Process();
        op.ReleaseEventID();
    } else if (TILING_KEY_IS(11)) {
        TPipe pipe;
        KernelAdaptiveAvgPool3DGradNcLargeCast<half> op;
        op.Init(y_grad, x_grad, workspace, &tiling_data, &pipe);
        op.InitBuffer();
        op.GetLocalTensor();
        op.ClearOutput();
        op.Process();
        op.ReleaseEventID();
    } else if (TILING_KEY_IS(21)) {
        TPipe pipe;
        KernelAdaptiveAvgPool3DGradNcLargeCast<bfloat16_t> op;
        op.Init(y_grad, x_grad, workspace, &tiling_data, &pipe);
        op.InitBuffer();
        op.GetLocalTensor();
        op.ClearOutput();
        op.Process();
        op.ReleaseEventID();
    }
}
#endif

/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file adaptive_avg_pool2d_grad.cpp
 * \brief
 */
#include "adaptive_avg_pool2d_grad_simt.h"
#include "adaptive_avg_pool2d_grad_struct.h"
#include "adaptive_avg_pool2d_grad_big_kernel.h"

using namespace AdaptiveAvgPool2dGradOp;

template <uint64_t TEMPLATE_MODE = TPL_SMALL_KERNEL, uint64_t INDEX_DTYPE = TPL_INT32, uint64_t IS_CHANNEL_LAST = 0>
__global__ __aicore__ void adaptive_avg_pool2d_grad(GM_ADDR input_grad, GM_ADDR output_grad, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    REGISTER_TILING_DEFAULT(AdaptiveAvgPool2dGradSimtTiling);
    if constexpr (TEMPLATE_MODE == TPL_SIMT_KERNEL) {
        GET_TILING_DATA_WITH_STRUCT(AdaptiveAvgPool2dGradSimtTiling, tilingData, tiling);
        if constexpr (INDEX_DTYPE == TPL_INT32) {
            AdaptiveAvgPool2dGradSimt<DTYPE_INPUT_GRAD, int32_t, IS_CHANNEL_LAST> op(&pipe, &tilingData);
            op.Init(input_grad, output_grad);
            op.Process();
        } else if constexpr (INDEX_DTYPE == TPL_INT64) {
            AdaptiveAvgPool2dGradSimt<DTYPE_INPUT_GRAD, int64_t, IS_CHANNEL_LAST> op(&pipe, &tilingData);
            op.Init(input_grad, output_grad);
            op.Process();
        }
    } else if constexpr (TEMPLATE_MODE == TPL_BIG_KERNEL) {
        GET_TILING_DATA_WITH_STRUCT(AdaptiveAvgPool2dGradBigKernelTiling, tilingData, tiling);
        if constexpr (INDEX_DTYPE == TPL_INT32) {
            AdaptiveAvgPool2dGradBigKernel<DTYPE_INPUT_GRAD, int32_t> op(pipe, tilingData);
            op.Init(input_grad, output_grad);
            op.Process();
        } else {
            AdaptiveAvgPool2dGradBigKernel<DTYPE_INPUT_GRAD, int64_t> op(pipe, tilingData);
            op.Init(input_grad, output_grad);
            op.Process();
        }
    }
}
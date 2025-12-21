/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file max_pool3d_grad_with_argmax.cpp
 * \brief
 */

#include "max_pool3d_grad_with_argmax_normal.h"
#include "max_pool3d_grad_with_argmax_scatter.h"
#include "max_pool3d_grad_with_argmax_scatter_overlap.h"
#include "max_pool3d_grad_with_argmax_cutk_d.h"
#include "max_pool3d_grad_with_argmax_cutk_dh.h"
#include "max_pool3d_grad_with_argmax_cutk_dhw.h"

using namespace MaxPool3DGradWithArgmax;

#define GENERAL_OP_IMPL(templateClass, ...)                  \
    do {                                                     \
        GET_TILING_DATA(tilingData, tiling);                 \
        templateClass<__VA_ARGS__> op(&pipe);                \
        op.Init(x, grad, argmax, y, workspace, &tilingData); \
        op.Process();                                        \
    } while (0)

#define GENERAL_OP_IMPL_CUTNC(templateClass, ...)            \
    do {                                                     \
        GET_TILING_DATA(tilingData, tiling);                 \
        templateClass<__VA_ARGS__> op(&pipe);                \
        op.Init(x, grad, argmax, y, workspace, &tilingData); \
        op.ProcessCutNc();                                   \
    } while (0)

extern "C" __global__ __aicore__ void max_pool3d_grad_with_argmax(
    GM_ADDR x, GM_ADDR grad, GM_ADDR argmax, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    if (workspace == nullptr) {
        return;
    }

    GM_ADDR userWS = GetUserWorkspace(workspace);
    if (userWS == nullptr) {
        return;
    }
    if (g_coreType == AIC) {
        return;
    }

    TPipe pipe;
    // The percentile determines if overlap occurs
    if (TILING_KEY_IS(0)) { // Normal Kernel
        GENERAL_OP_IMPL(MaxPool3DGradWithArgmaxNormal, DTYPE_X, DTYPE_X, int32_t, DTYPE_X, false);
    } else if (TILING_KEY_IS(100)) {
        GENERAL_OP_IMPL(MaxPool3DGradWithArgmaxNormal, DTYPE_X, DTYPE_X, int32_t, DTYPE_X, true);
    } else if (TILING_KEY_IS(2)) { // Scatter Kernel
        GENERAL_OP_IMPL(MaxPoolGradWithArgScatter, DTYPE_X, DTYPE_X, int32_t, DTYPE_X);
    } else if (TILING_KEY_IS(102)) {
        GENERAL_OP_IMPL(MaxPoolGradWithArgScatterOverlap, DTYPE_X, DTYPE_GRAD, int32_t, DTYPE_Y);
    } else if (TILING_KEY_IS(1)) { // CutK Kernel, no cut
        GENERAL_OP_IMPL(MaxPool3DGradWithArgmaxCutKD, DTYPE_X, DTYPE_GRAD, int32_t, DTYPE_Y, false);
    } else if (TILING_KEY_IS(21)) { // CutK Kernel, cut do
        GENERAL_OP_IMPL(MaxPool3DGradWithArgmaxCutKD, DTYPE_X, DTYPE_GRAD, int32_t, DTYPE_Y, false);
    } else if (TILING_KEY_IS(31)) { // CutK Kernel, cut do, kd, ho
        GENERAL_OP_IMPL(MaxPool3DGradWithArgmaxCutKD, DTYPE_X, DTYPE_GRAD, int32_t, DTYPE_Y, false);
    } else if (TILING_KEY_IS(41)) { // CutK Kernel, cut do, kd, ho, kh, wo
        GENERAL_OP_IMPL(MaxPool3DGradWithArgmaxCutKDH, DTYPE_X, DTYPE_GRAD, int32_t, DTYPE_Y, false);
    } else if (TILING_KEY_IS(51)) { // CutK Kernel, cut do, kd
        GENERAL_OP_IMPL(MaxPool3DGradWithArgmaxCutKD, DTYPE_X, DTYPE_GRAD, int32_t, DTYPE_Y, false);
    } else if (TILING_KEY_IS(61)) { // CutK Kernel, cut do, kd, ho, kh
        GENERAL_OP_IMPL(MaxPool3DGradWithArgmaxCutKDH, DTYPE_X, DTYPE_X, int32_t, DTYPE_X, false);
    } else if (TILING_KEY_IS(71)) { // CutK Kernel, cut do, kd, ho, kh, wo, kw
        GENERAL_OP_IMPL(MaxPool3DGradWithArgmaxCutKDHW, DTYPE_X, DTYPE_GRAD, int32_t, DTYPE_Y, false);
    } else if (TILING_KEY_IS(101)) {
        GENERAL_OP_IMPL_CUTNC(MaxPool3DGradWithArgmaxCutKD, DTYPE_X, DTYPE_GRAD, int32_t, DTYPE_Y, true);
    } else if (TILING_KEY_IS(121)) {
        GENERAL_OP_IMPL_CUTNC(MaxPool3DGradWithArgmaxCutKD, DTYPE_X, DTYPE_GRAD, int32_t, DTYPE_Y, true);
    } else if (TILING_KEY_IS(131)) {
        GENERAL_OP_IMPL_CUTNC(MaxPool3DGradWithArgmaxCutKD, DTYPE_X, DTYPE_GRAD, int32_t, DTYPE_Y, true);
    } else if (TILING_KEY_IS(141)) {
        GENERAL_OP_IMPL_CUTNC(MaxPool3DGradWithArgmaxCutKDH, DTYPE_X, DTYPE_GRAD, int32_t, DTYPE_Y, true);
    } else if (TILING_KEY_IS(151)) {
        GENERAL_OP_IMPL_CUTNC(MaxPool3DGradWithArgmaxCutKD, DTYPE_X, DTYPE_X, int32_t, DTYPE_X, true);
    } else if (TILING_KEY_IS(161)) {
        GENERAL_OP_IMPL_CUTNC(MaxPool3DGradWithArgmaxCutKDH, DTYPE_X, DTYPE_X, int32_t, DTYPE_X, true);
    } else if (TILING_KEY_IS(171)) {
        GENERAL_OP_IMPL_CUTNC(MaxPool3DGradWithArgmaxCutKDHW, DTYPE_X, DTYPE_X, int32_t, DTYPE_X, true);
    }
    return;
}
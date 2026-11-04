// clang-format off
/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 *
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of CANN Open Software License Agreement Version 2.0 (the "License").
 *
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 *
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 *
 * See LICENSE in the root of the software repository for the full text of the License.
 */
// clang-format on

/* !
 * \file max_pool2d_with_mask_backward.cpp
 * \brief
 */
#if defined(__DAV_M200__) || defined(__DAV_M200_VEC__) || (__NPU_ARCH__ == 2002)
#include "arch20/max_pool2d_with_mask_backward_310p.h"
#include "arch20/max_pool2d_with_mask_backward_vector_310p.h"
#include "arch20/max_pool2d_with_mask_backward_scatter_310p.h"
using namespace AscendC;
// Isolate Vector local arrays/register lifetimes from the short Scatter entry path.
// This is an experiment in code generation, measured independently with ABBA.
template <class Kernel, class TD>
__aicore__ __attribute__((noinline)) void Run310PVector(GM_ADDR grad, GM_ADDR index, GM_ADDR out, const TD* td,
                                                        TPipe* pipe)
{
    Kernel op;
    op.Init(grad, index, out, td, pipe);
    op.Process();
}
extern "C" __global__ __aicore__ void max_pool2d_with_mask_backward(GM_ADDR x, GM_ADDR grad, GM_ADDR argmax, GM_ADDR y,
                                                                    GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    TPipe pipe;
    if (TILING_KEY_IS(20040)) {
        MaxPool2dMask310P::ScatterKernel<half> op;
        op.Init(grad, argmax, y, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(20041)) {
        MaxPool2dMask310P::ScatterKernel<float> op;
        op.Init(grad, argmax, y, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(20042)) {
        MaxPool2dMask310P::ScatterKernel<half, 32768, 4096> op;
        op.Init(grad, argmax, y, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(20043)) {
        MaxPool2dMask310P::ScatterKernel<float, 32768, 4096> op;
        op.Init(grad, argmax, y, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(20045)) {
        MaxPool2dMask310P::ScatterKernel<float, 25600, 4096, true> op;
        op.Init(grad, argmax, y, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(20030)) {
        Run310PVector<MaxPool2dMask310P::VectorKernel<half>>(grad, argmax, y, &tilingData, &pipe);
    } else if (TILING_KEY_IS(20031)) {
        Run310PVector<MaxPool2dMask310P::VectorKernel<float>>(grad, argmax, y, &tilingData, &pipe);
    } else if (TILING_KEY_IS(20032)) {
        Run310PVector<MaxPool2dMask310P::VectorKernel<half, 8>>(grad, argmax, y, &tilingData, &pipe);
    } else if (TILING_KEY_IS(20033)) {
        Run310PVector<MaxPool2dMask310P::VectorKernel<float, 8>>(grad, argmax, y, &tilingData, &pipe);
    } else if (TILING_KEY_IS(20034)) {
        Run310PVector<MaxPool2dMask310P::VectorKernel<half, 8, 6144, 16384>>(grad, argmax, y, &tilingData, &pipe);
    } else if (TILING_KEY_IS(20035)) {
        Run310PVector<MaxPool2dMask310P::VectorKernel<float, 8, 6144, 16384, true, true>>(grad, argmax, y, &tilingData,
                                                                                          &pipe);
    } else if (TILING_KEY_IS(20036)) {
        Run310PVector<MaxPool2dMask310P::VectorKernel<half, 8, 10240, 8192>>(grad, argmax, y, &tilingData, &pipe);
    } else if (TILING_KEY_IS(20037)) {
        Run310PVector<MaxPool2dMask310P::VectorKernel<float, 8, 10240, 8192>>(grad, argmax, y, &tilingData, &pipe);
    } else if (TILING_KEY_IS(20039)) {
        Run310PVector<MaxPool2dMask310P::VectorKernel<float, 8, 4096, 20480>>(grad, argmax, y, &tilingData, &pipe);
    } else if (TILING_KEY_IS(20047)) {
        Run310PVector<MaxPool2dMask310P::VectorKernel<float, 8, 4096, 20480, true>>(grad, argmax, y, &tilingData,
                                                                                    &pipe);
    } else if (TILING_KEY_IS(20049)) {
        Run310PVector<MaxPool2dMask310P::VectorKernel<float, 8, 6144, 16384, true, true>>(grad, argmax, y, &tilingData,
                                                                                          &pipe);
    } else if (TILING_KEY_IS(20056)) {
        Run310PVector<MaxPool2dMask310P::VectorKernel<half, 8, 4096, 19200>>(grad, argmax, y, &tilingData, &pipe);
    } else if (TILING_KEY_IS(20058)) {
        Run310PVector<MaxPool2dMask310P::VectorKernel<half, 8, 3072, 20480>>(grad, argmax, y, &tilingData, &pipe);
    } else if (TILING_KEY_IS(20073)) {
        Run310PVector<
            MaxPool2dMask310P::VectorKernel<float, 8, 5632, 30464, true, true, false, 5, 3, 4, 2, true, 192, true>>(
            grad, argmax, y, &tilingData, &pipe);
    } else if (TILING_KEY_IS(20071)) {
        Run310PVector<MaxPool2dMask310P::VectorKernel<float, 8, 6400, 30464, true, true, false, 7, 3, 4, 2, true, 264>>(
            grad, argmax, y, &tilingData, &pipe);
    } else if (TILING_KEY_IS(20067)) {
        Run310PVector<MaxPool2dMask310P::VectorKernel<float, 8, 5632, 30464, true, true, false, 5, 3, 4, 2, true, 192>>(
            grad, argmax, y, &tilingData, &pipe);
    } else if (TILING_KEY_IS(20069)) {
        Run310PVector<MaxPool2dMask310P::VectorKernel<float, 8, 5632, 30464, true, true, false, 7, 3, 4, 2, true, 192>>(
            grad, argmax, y, &tilingData, &pipe);
    } else if (TILING_KEY_IS(20068)) {
        Run310PVector<MaxPool2dMask310P::VectorKernel<half, 8, 4608, 30464, true, true, false, 4, 4, 3, 3, true, 192>>(
            grad, argmax, y, &tilingData, &pipe);
    } else if (TILING_KEY_IS(20063)) {
        Run310PVector<MaxPool2dMask310P::VectorKernel<float, 8, 7936, 26112, true, true, false, 0, 0, 0, 0, true>>(
            grad, argmax, y, &tilingData, &pipe);
    } else if (TILING_KEY_IS(20062)) {
        Run310PVector<MaxPool2dMask310P::VectorKernel<half, 8, 6144, 28672, false, false, false, 0, 0, 0, 0, true>>(
            grad, argmax, y, &tilingData, &pipe);
    } else if (TILING_KEY_IS(20061)) {
        Run310PVector<MaxPool2dMask310P::VectorKernel<float, 8, 7168, 28672, true, true, false, 0, 0, 0, 0, true>>(
            grad, argmax, y, &tilingData, &pipe);
    } else if (TILING_KEY_IS(20055)) {
        Run310PVector<MaxPool2dMask310P::VectorKernel<float, 8, 6144, 17408, true, true>>(grad, argmax, y, &tilingData,
                                                                                          &pipe);
    } else if (TILING_KEY_IS(20057)) {
        Run310PVector<MaxPool2dMask310P::VectorKernel<float, 8, 4608, 20480, true, true, true, 5, 3, 4, 2>>(
            grad, argmax, y, &tilingData, &pipe);
    } else if (TILING_KEY_IS(20059)) {
        Run310PVector<MaxPool2dMask310P::VectorKernel<float, 8, 4608, 20480, true, true, true, 7, 3, 4, 2>>(
            grad, argmax, y, &tilingData, &pipe);
    } else if (TILING_KEY_IS(20053)) {
        Run310PVector<MaxPool2dMask310P::VectorKernel<float, 8, 4608, 20480, true, true, true>>(grad, argmax, y,
                                                                                                &tilingData, &pipe);
    } else if (TILING_KEY_IS(20020)) {
        MaxPool2dMask310P::Kernel<half> op;
        op.Init(grad, argmax, y, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(20021)) {
        MaxPool2dMask310P::Kernel<float> op;
        op.Init(grad, argmax, y, &tilingData, &pipe);
        op.Process();
    }
}
#else
// Adapted from pooling/max_pool3d_grad_with_argmax/op_kernel/max_pool3d_grad_with_argmax.cpp.
#include "max_pool2d_with_mask_backward_normal.h"
#include "max_pool2d_with_mask_backward_scatter.h"
#include "max_pool2d_with_mask_backward_scatter_overlap.h"
#include "max_pool2d_with_mask_backward_cutk_d.h"
#include "max_pool2d_with_mask_backward_cutk_dh.h"
#include "max_pool2d_with_mask_backward_cutk_dhw.h"

using namespace MaxPool2dWithMaskBackward;

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
extern "C" __global__ __aicore__ void max_pool2d_with_mask_backward(GM_ADDR x, GM_ADDR grad, GM_ADDR argmax, GM_ADDR y,
                                                                    GM_ADDR workspace, GM_ADDR tiling)
{
    if (g_coreType == AIC) {
        return;
    }
    // Only Normal/non-overlap with complete output coverage uses neither
    // temporary GM storage nor SyncAll. Other paths retain system-workspace
    // protection, even when their user workspace size is zero.
    bool needsWorkspace = true;
    if (TILING_KEY_IS(0)) {
        GET_TILING_DATA(entryTilingData, tiling);
        needsWorkspace = MaxPool2dWithMaskBackwardComm::NeedsOutputInitialization(entryTilingData, false);
    }
    if (needsWorkspace && (workspace == nullptr || GetUserWorkspace(workspace) == nullptr)) {
        return;
    }

    TPipe pipe;
    // The percentile determines if overlap occurs
    if (TILING_KEY_IS(0)) { // Normal Kernel
        GENERAL_OP_IMPL(MaxPool2dWithMaskBackwardNormal, DTYPE_X, DTYPE_X, int32_t, DTYPE_X, false);
    } else if (TILING_KEY_IS(100)) {
        GENERAL_OP_IMPL(MaxPool2dWithMaskBackwardNormal, DTYPE_X, DTYPE_X, int32_t, DTYPE_X, true);
    } else if (TILING_KEY_IS(2)) { // Scatter Kernel
        GENERAL_OP_IMPL(MaxPoolGradWithArgScatter, DTYPE_X, DTYPE_X, int32_t, DTYPE_X);
    } else if (TILING_KEY_IS(102)) {
        GENERAL_OP_IMPL(MaxPoolGradWithArgScatterOverlap, DTYPE_X, DTYPE_GRAD, int32_t, DTYPE_Y);
    } else if (TILING_KEY_IS(1)) { // CutK Kernel, no cut
        GENERAL_OP_IMPL(MaxPool2dWithMaskBackwardCutKD, DTYPE_X, DTYPE_GRAD, int32_t, DTYPE_Y, false);
    } else if (TILING_KEY_IS(21)) { // CutK Kernel, cut do
        GENERAL_OP_IMPL(MaxPool2dWithMaskBackwardCutKD, DTYPE_X, DTYPE_GRAD, int32_t, DTYPE_Y, false);
    } else if (TILING_KEY_IS(31)) { // CutK Kernel, cut do, kd, ho
        GENERAL_OP_IMPL(MaxPool2dWithMaskBackwardCutKD, DTYPE_X, DTYPE_GRAD, int32_t, DTYPE_Y, false);
    } else if (TILING_KEY_IS(41)) { // CutK Kernel, cut do, kd, ho, kh, wo
        GENERAL_OP_IMPL(MaxPool2dWithMaskBackwardCutKDH, DTYPE_X, DTYPE_GRAD, int32_t, DTYPE_Y, false);
    } else if (TILING_KEY_IS(51)) { // CutK Kernel, cut do, kd
        GENERAL_OP_IMPL(MaxPool2dWithMaskBackwardCutKD, DTYPE_X, DTYPE_GRAD, int32_t, DTYPE_Y, false);
    } else if (TILING_KEY_IS(61)) { // CutK Kernel, cut do, kd, ho, kh
        GENERAL_OP_IMPL(MaxPool2dWithMaskBackwardCutKDH, DTYPE_X, DTYPE_X, int32_t, DTYPE_X, false);
    } else if (TILING_KEY_IS(71)) { // CutK Kernel, cut do, kd, ho, kh, wo, kw
        GENERAL_OP_IMPL(MaxPool2dWithMaskBackwardCutKDHW, DTYPE_X, DTYPE_GRAD, int32_t, DTYPE_Y, false);
    } else if (TILING_KEY_IS(101)) {
        GENERAL_OP_IMPL_CUTNC(MaxPool2dWithMaskBackwardCutKD, DTYPE_X, DTYPE_GRAD, int32_t, DTYPE_Y, true);
    } else if (TILING_KEY_IS(121)) {
        GENERAL_OP_IMPL_CUTNC(MaxPool2dWithMaskBackwardCutKD, DTYPE_X, DTYPE_GRAD, int32_t, DTYPE_Y, true);
    } else if (TILING_KEY_IS(131)) {
        GENERAL_OP_IMPL_CUTNC(MaxPool2dWithMaskBackwardCutKD, DTYPE_X, DTYPE_GRAD, int32_t, DTYPE_Y, true);
    } else if (TILING_KEY_IS(141)) {
        GENERAL_OP_IMPL_CUTNC(MaxPool2dWithMaskBackwardCutKDH, DTYPE_X, DTYPE_GRAD, int32_t, DTYPE_Y, true);
    } else if (TILING_KEY_IS(151)) {
        GENERAL_OP_IMPL_CUTNC(MaxPool2dWithMaskBackwardCutKD, DTYPE_X, DTYPE_X, int32_t, DTYPE_X, true);
    } else if (TILING_KEY_IS(161)) {
        GENERAL_OP_IMPL_CUTNC(MaxPool2dWithMaskBackwardCutKDH, DTYPE_X, DTYPE_X, int32_t, DTYPE_X, true);
    } else if (TILING_KEY_IS(171)) {
        GENERAL_OP_IMPL_CUTNC(MaxPool2dWithMaskBackwardCutKDHW, DTYPE_X, DTYPE_X, int32_t, DTYPE_X, true);
    }
    return;
}

#endif // DAV_2002

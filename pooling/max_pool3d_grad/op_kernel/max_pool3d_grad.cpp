/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
 
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "arch35/max_pool3d_grad_small_kernel_impl_scatter.h"
#include "arch35/max_pool3d_grad_small_kernel_impl_gather.h"
#include "arch35/max_pool3d_grad_simt_kernel.h"

constexpr int64_t NCDHW = 0;
constexpr int64_t NDHWC = 1;

using namespace AscendC;
using namespace MaxPool3DSmallKernelNameSpace;

template <uint64_t INDEX_DTYPE = TPL_INT32, uint64_t IS_SIMT = 0, uint64_t IS_CHANNEL_LAST = 0, uint64_t IS_CHECK_RANGE = 0, uint64_t USE_INT64_INDEX = 0>
__global__ __aicore__ void max_pool3d_grad(
    GM_ADDR orig_x, GM_ADDR orig_y, GM_ADDR grads, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    if (workspace == nullptr || GetUserWorkspace(workspace) == nullptr || g_coreType == AIC) {
        return;
    }
    TPipe pipe;
    if constexpr (IS_SIMT == 0 && INDEX_DTYPE == TPL_INT32 && IS_CHECK_RANGE == 1) {
        REGISTER_TILING_DEFAULT(Pool3DGradNCDHWTilingData);
        GET_TILING_DATA_WITH_STRUCT(Pool3DGradNCDHWTilingData, tilingData, tiling);
        Pool3DGradSmallKernel<DTYPE_ORIG_X, int32_t, int32_t, true> op(pipe, tilingData);
        op.Init(orig_x, orig_y, grads, y);
        op.Process();
    } else if (IS_SIMT == 0 && INDEX_DTYPE == TPL_INT64 && IS_CHECK_RANGE == 1) {
        REGISTER_TILING_DEFAULT(Pool3DGradNCDHWTilingData);
        GET_TILING_DATA_WITH_STRUCT(Pool3DGradNCDHWTilingData, tilingData, tiling);
        Pool3DGradSmallKernel<DTYPE_ORIG_X, int64_t, int64_t, true> op(pipe, tilingData);
        op.Init(orig_x, orig_y, grads, y);
        op.Process();
    } else if (IS_SIMT == 0 && INDEX_DTYPE == TPL_INT32 && IS_CHECK_RANGE == 0) {
        REGISTER_TILING_DEFAULT(Pool3DGradNCDHWTilingData);
        GET_TILING_DATA_WITH_STRUCT(Pool3DGradNCDHWTilingData, tilingData, tiling);
        Pool3DGradSmallKernel<DTYPE_ORIG_X, int32_t, int32_t, false> op(pipe, tilingData);
        op.Init(orig_x, orig_y, grads, y);
        op.Process();        
    } else if (IS_SIMT == 0 && INDEX_DTYPE == TPL_INT64 && IS_CHECK_RANGE == 0) {
        REGISTER_TILING_DEFAULT(Pool3DGradNCDHWTilingData);
        GET_TILING_DATA_WITH_STRUCT(Pool3DGradNCDHWTilingData, tilingData, tiling);
        Pool3DGradSmallKernel<DTYPE_ORIG_X, int64_t, int64_t, false> op(pipe, tilingData);
        op.Init(orig_x, orig_y, grads, y);
        op.Process();        
    } else if constexpr (IS_SIMT == 1 && INDEX_DTYPE == TPL_INT32 && IS_CHANNEL_LAST == 0 && USE_INT64_INDEX == 0) {
        REGISTER_TILING_FOR_TILINGKEY("IS_SIMT == 1 && INDEX_DTYPE == TPL_INT32 && IS_CHANNEL_LAST == 0 && USE_INT64_INDEX == 0", MaxPool3DGradSimtTilingData);
        GET_TILING_DATA_WITH_STRUCT(MaxPool3DGradSimtTilingData, tilingData, tiling);
        MaxPool3DGrad::MaxPool3DGradSimtKernel<DTYPE_ORIG_X, int64_t, NCDHW, false, int32_t> op(&pipe, &tilingData);
        op.Init(orig_x, orig_y, grads, y, workspace);
        op.Process();
    } else if constexpr (IS_SIMT == 1 && INDEX_DTYPE == TPL_INT64 && IS_CHANNEL_LAST == 0 && USE_INT64_INDEX == 0) {
        REGISTER_TILING_FOR_TILINGKEY("IS_SIMT == 1 && INDEX_DTYPE == TPL_INT64 && IS_CHANNEL_LAST == 0 && USE_INT64_INDEX == 0", MaxPool3DGradSimtTilingData);
        GET_TILING_DATA_WITH_STRUCT(MaxPool3DGradSimtTilingData, tilingData, tiling);
        MaxPool3DGrad::MaxPool3DGradSimtKernel<DTYPE_ORIG_X, int64_t, NCDHW, false, int64_t> op(&pipe, &tilingData);
        op.Init(orig_x, orig_y, grads, y, workspace);
        op.Process();
    } else if constexpr (IS_SIMT == 1 && INDEX_DTYPE == TPL_INT32 && IS_CHANNEL_LAST == 1 && USE_INT64_INDEX == 0) {
        REGISTER_TILING_FOR_TILINGKEY("IS_SIMT == 1 && INDEX_DTYPE == TPL_INT32 && IS_CHANNEL_LAST == 1 && USE_INT64_INDEX == 0", MaxPool3DGradSimtTilingData);
        GET_TILING_DATA_WITH_STRUCT(MaxPool3DGradSimtTilingData, tilingData, tiling);
        MaxPool3DGrad::MaxPool3DGradSimtKernel<DTYPE_ORIG_X, int64_t, NDHWC, false, int32_t> op(&pipe, &tilingData);
        op.Init(orig_x, orig_y, grads, y, workspace);
        op.Process();
    } else if constexpr (IS_SIMT == 1 && INDEX_DTYPE == TPL_INT64 && IS_CHANNEL_LAST == 1 && USE_INT64_INDEX == 0) {
        REGISTER_TILING_FOR_TILINGKEY("IS_SIMT == 1 && INDEX_DTYPE == TPL_INT64 && IS_CHANNEL_LAST == 1 && USE_INT64_INDEX == 0", MaxPool3DGradSimtTilingData);
        GET_TILING_DATA_WITH_STRUCT(MaxPool3DGradSimtTilingData, tilingData, tiling);
        MaxPool3DGrad::MaxPool3DGradSimtKernel<DTYPE_ORIG_X, int64_t, NDHWC, false, int64_t> op(&pipe, &tilingData);
        op.Init(orig_x, orig_y, grads, y, workspace);
        op.Process();
    } else if constexpr (IS_SIMT == 1 && INDEX_DTYPE == TPL_INT64 && IS_CHANNEL_LAST == 0 && USE_INT64_INDEX == 1) {
        REGISTER_TILING_FOR_TILINGKEY("IS_SIMT == 1 && INDEX_DTYPE == TPL_INT64 && IS_CHANNEL_LAST == 0 && USE_INT64_INDEX == 1", MaxPool3DGradSimtTilingData);
        GET_TILING_DATA_WITH_STRUCT(MaxPool3DGradSimtTilingData, tilingData, tiling);
        MaxPool3DGrad::MaxPool3DGradSimtKernel<DTYPE_ORIG_X, int64_t, NCDHW, true, int64_t> op(&pipe, &tilingData);
        op.Init(orig_x, orig_y, grads, y, workspace);
        op.Process();
    } else if constexpr (IS_SIMT == 1 && INDEX_DTYPE == TPL_INT64 && IS_CHANNEL_LAST == 1 && USE_INT64_INDEX == 1) {
        REGISTER_TILING_FOR_TILINGKEY("IS_SIMT == 1 && INDEX_DTYPE == TPL_INT64 && IS_CHANNEL_LAST == 1 && USE_INT64_INDEX == 1", MaxPool3DGradSimtTilingData);
        GET_TILING_DATA_WITH_STRUCT(MaxPool3DGradSimtTilingData, tilingData, tiling);
        MaxPool3DGrad::MaxPool3DGradSimtKernel<DTYPE_ORIG_X, int64_t, NDHWC, true, int64_t> op(&pipe, &tilingData);
        op.Init(orig_x, orig_y, grads, y, workspace);
        op.Process();
    }
}

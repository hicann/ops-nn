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
 * \file max_pool_grad_with_argmax_v3.cpp
 * \brief max_pool_grad_with_argmax_v3 implied
 */

#include <cstdint>
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"

#include "./arch35/max_pool_grad_with_argmax_v3_nchw_kernel.h"
#include "./arch35/max_pool_grad_with_argmax_v3_nhwc_kernel.h"
#include "./arch35/max_pool_grad_with_argmax_v3_nchw_scalar.h"
#include "./arch35/max_pool_grad_with_argmax_v3_ksize_one.h"
#include "./arch35/max_pool_grad_with_argmax_v3_simt.h"

#define NO_CHECK_RANGE_TILING_KEY_NCHW 100
#define CHECK_RANGE_TILING_KEY_NCHW 101
#define NO_CHECK_RANGE_TILING_KEY_NHWC 200
#define CHECK_RANGE_TILING_KEY_NHWC 201
#define CHECK_RANGE_TILING_KEY_NCHW_SCALAR 301

#define NO_CHECK_RANGE_TILING_KEY_NCHW_INT64 110
#define CHECK_RANGE_TILING_KEY_NCHW_INT64 111
#define NO_CHECK_RANGE_TILING_KEY_NHWC_INT64 210
#define CHECK_RANGE_TILING_KEY_NHWC_INT64 211
#define KSIZE_ONE_TILING_KEY 800
#define SIMT_NCHW_INT32_TILING_KEY  900
#define SIMT_NCHW_INT64_TILING_KEY  901

constexpr int NCHW = 0;
constexpr int NHWC = 1;

extern "C" __global__ __aicore__ void max_pool_grad_with_argmax_v3(
    GM_ADDR x, GM_ADDR grad, GM_ADDR argmax, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    AscendC::TPipe pipeBase;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (TILING_KEY_IS(NO_CHECK_RANGE_TILING_KEY_NCHW)) {
        GET_TILING_DATA_WITH_STRUCT(MaxPoolGradWithArgmaxV3NCHWTilingData, tilingDataIn, tiling);
        MaxPoolGradWithArgmaxV3NCHWNameSpace::MaxPoolGradWithArgmaxV3NCHWKernel<DTYPE_X, DTYPE_ARGMAX, int32_t, 0> op;
        op.Init(x, grad, argmax, y, pipeBase, tilingDataIn);
        op.Process();
    } else if (TILING_KEY_IS(CHECK_RANGE_TILING_KEY_NCHW)) {
        GET_TILING_DATA_WITH_STRUCT(MaxPoolGradWithArgmaxV3NCHWTilingData, tilingDataIn, tiling);
        MaxPoolGradWithArgmaxV3NCHWNameSpace::MaxPoolGradWithArgmaxV3NCHWKernel<DTYPE_X, DTYPE_ARGMAX, int32_t, 1> op;
        op.Init(x, grad, argmax, y, pipeBase, tilingDataIn);
        op.Process();
    } else if (TILING_KEY_IS(NO_CHECK_RANGE_TILING_KEY_NHWC)) {
        GET_TILING_DATA_WITH_STRUCT(MaxPoolGradWithArgmaxV3NHWCTilingData, tilingDataIn, tiling);
        MaxPoolGradWithArgmaxV3NHWCNameSpace::MaxPoolGradWithArgmaxV3KernelNHWC<DTYPE_X, DTYPE_ARGMAX, int32_t, 0> op;
        op.Init(x, grad, argmax, y, pipeBase, tilingDataIn);
        op.Process();
    } else if (TILING_KEY_IS(CHECK_RANGE_TILING_KEY_NHWC)) {
        GET_TILING_DATA_WITH_STRUCT(MaxPoolGradWithArgmaxV3NHWCTilingData, tilingDataIn, tiling);
        MaxPoolGradWithArgmaxV3NHWCNameSpace::MaxPoolGradWithArgmaxV3KernelNHWC<DTYPE_X, DTYPE_ARGMAX, int32_t, 1> op;
        op.Init(x, grad, argmax, y, pipeBase, tilingDataIn);
        op.Process();
    } else if (TILING_KEY_IS(NO_CHECK_RANGE_TILING_KEY_NCHW_INT64)) {
        GET_TILING_DATA_WITH_STRUCT(MaxPoolGradWithArgmaxV3NCHWTilingData, tilingDataIn, tiling);
        MaxPoolGradWithArgmaxV3NCHWNameSpace::MaxPoolGradWithArgmaxV3NCHWKernel<DTYPE_X, DTYPE_ARGMAX, int64_t, 0> op;
        op.Init(x, grad, argmax, y, pipeBase, tilingDataIn);
        op.Process();
    } else if (TILING_KEY_IS(CHECK_RANGE_TILING_KEY_NCHW_INT64)) {
        GET_TILING_DATA_WITH_STRUCT(MaxPoolGradWithArgmaxV3NCHWTilingData, tilingDataIn, tiling);
        MaxPoolGradWithArgmaxV3NCHWNameSpace::MaxPoolGradWithArgmaxV3NCHWKernel<DTYPE_X, DTYPE_ARGMAX, int64_t, 1> op;
        op.Init(x, grad, argmax, y, pipeBase, tilingDataIn);
        op.Process();
    } else if (TILING_KEY_IS(NO_CHECK_RANGE_TILING_KEY_NHWC_INT64)) {
        GET_TILING_DATA_WITH_STRUCT(MaxPoolGradWithArgmaxV3NHWCTilingData, tilingDataIn, tiling);
        MaxPoolGradWithArgmaxV3NHWCNameSpace::MaxPoolGradWithArgmaxV3KernelNHWC<DTYPE_X, DTYPE_ARGMAX, int64_t, 0> op;
        op.Init(x, grad, argmax, y, pipeBase, tilingDataIn);
        op.Process();
    } else if (TILING_KEY_IS(CHECK_RANGE_TILING_KEY_NHWC_INT64)) {
        GET_TILING_DATA_WITH_STRUCT(MaxPoolGradWithArgmaxV3NHWCTilingData, tilingDataIn, tiling);
        MaxPoolGradWithArgmaxV3NHWCNameSpace::MaxPoolGradWithArgmaxV3KernelNHWC<DTYPE_X, DTYPE_ARGMAX, int64_t, 1> op;
        op.Init(x, grad, argmax, y, pipeBase, tilingDataIn);
        op.Process();
    } else if (TILING_KEY_IS(CHECK_RANGE_TILING_KEY_NCHW_SCALAR)) {
        GET_TILING_DATA_WITH_STRUCT(MaxPoolGradWithArgmaxV3NCHWScalarTilingData, tilingDataIn, tiling);
        MaxPoolGradWithArgmaxV3NCHWScalarNameSpace::MaxPoolGradWithArgmaxV3NCHWScalar<DTYPE_X, DTYPE_ARGMAX> op(
            tilingDataIn, pipeBase);
        op.Init(x, grad, argmax, y);
        op.Process();
    } else if (TILING_KEY_IS(KSIZE_ONE_TILING_KEY)) {
        GET_TILING_DATA_WITH_STRUCT(MaxPoolGradWithArgmaxV3KSizeOneTilingData, tilingDataIn, tiling);
        MaxPoolGradWithArgmaxV3KsizeOneNameSpace::MaxPoolGradWithArgmaxV3KsizeOne<DTYPE_X> op(
            tilingDataIn, pipeBase);
        op.Init(x, grad, argmax, y);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_NCHW_INT32_TILING_KEY)) {
        GET_TILING_DATA_WITH_STRUCT(MaxPoolGradWithArgmaxV3SimtTilingData, tilingDataIn, tiling);
        MaxPoolGradWithArgmaxV3Simt<DTYPE_X, DTYPE_ARGMAX, NCHW, int32_t, uint32_t> op(&pipeBase, &tilingDataIn);
        op.Init(x, grad, argmax, y);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_NCHW_INT64_TILING_KEY)) {
        GET_TILING_DATA_WITH_STRUCT(MaxPoolGradWithArgmaxV3SimtTilingData, tilingDataIn, tiling);
        MaxPoolGradWithArgmaxV3Simt<DTYPE_X, DTYPE_ARGMAX, NCHW, int64_t, uint64_t> op(&pipeBase, &tilingDataIn);
        op.Init(x, grad, argmax, y);
        op.Process();
    }
}
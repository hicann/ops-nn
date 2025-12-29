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
 * \file max_pool_with_argmax_v3.cpp
 * \brief max_pool_with_argmax_v3 implied
 */

#include <cstdint>
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "arch35/max_pool_with_argmax_v3_simt.h"
#include "arch35/max_pool_with_argmax_v3_big_kernel.h"
#include "arch35/max_pool_with_argmax_v3_gather_kernel.h"
#include "arch35/max_pool_with_argmax_v3_big_kernel_mul_core.h"
#include "arch35/max_pool_with_argmax_v3_nhwc_big_c.h"
#include "arch35/max_pool_with_argmax_v3_nhwc_small_c.h"

#define BIG_KERNEL_FORMAT_NCHW 311110
#define NO_PADDING_TILING_KEY 300001
#define PADDING_TILING_KEY 300002
#define MAX_POOL_WITH_ARGMAX_V3_TILING_KEY_BIG_KERNEL_SPLIT_FP_INT32 400001
#define MAX_POOL_WITH_ARGMAX_V3_TILING_KEY_BIG_KERNEL_SPLIT_FP_INT64 400002
#define MAX_POOL_WITH_ARGMAX_V3_TILING_KEY_BIG_KERNEL_SPLIT_BF16_INT32 400003
#define MAX_POOL_WITH_ARGMAX_V3_TILING_KEY_BIG_KERNEL_SPLIT_BF16_INT64 400004
#define MAX_POOL_WITH_ARGMAX_V3_TILING_KEY_BIG_KERNEL_SPLIT_HALF_INT32 400005
#define MAX_POOL_WITH_ARGMAX_V3_TILING_KEY_BIG_KERNEL_SPLIT_HALF_INT64 400006
#define MAX_POOL_WITH_ARGMAX_V3_TILING_KEY_SIMT_NCHW 500001
#define MAX_POOL_WITH_ARGMAX_V3_TILING_KEY_SIMT_NHWC 500002
#define MAX_POOL_WITH_ARGMAX_V3_TILING_KEY_SIMT_NCHW_PAD 500011
#define MAX_POOL_WITH_ARGMAX_V3_TILING_KEY_SIMT_NHWC_PAD 500012
#define MAX_POOL_WITH_ARGMAX_V3_TILING_KEY_NHWC_BIG_C 800001
#define MAX_POOL_WITH_ARGMAX_V3_TILING_KEY_NHWC_BIG_C_PAD 800002
#define MAX_POOL_WITH_ARGMAX_V3_TILING_KEY_NHWC_SMALL_C 700001
#define MAX_POOL_WITH_ARGMAX_V3_TILING_KEY_NHWC_SMALL_C_PAD 700002

constexpr uint32_t PAD_DISABLE = 0;
constexpr uint32_t PAD_ENABLE = 1;
constexpr int NCHW = 0;
constexpr int NHWC = 1;

extern "C" __global__ __aicore__ void max_pool_with_argmax_v3(
    GM_ADDR x, GM_ADDR y, GM_ADDR argmax, GM_ADDR workspace, GM_ADDR tiling)
{
    TPipe pipeBase;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    if (TILING_KEY_IS(MAX_POOL_WITH_ARGMAX_V3_TILING_KEY_NHWC_SMALL_C)) {
        GET_TILING_DATA_WITH_STRUCT(MaxPoolWithArgmaxV3NhwcTilingData, tilingDataIn, tiling);
        const MaxPoolWithArgmaxV3NhwcTilingData* __restrict tilingData = &tilingDataIn;
        MaxPoolWithArgmaxV3SmallCNameSpace::MaxPoolWithArgmaxV3SmallC<DTYPE_X, DTYPE_ARGMAX, 0> op(
            &pipeBase, tilingData);
        op.Init(x, y, argmax);
        op.MaxPoolWithArgmaxV3SmallCProcess();
    } else if (TILING_KEY_IS(MAX_POOL_WITH_ARGMAX_V3_TILING_KEY_NHWC_SMALL_C_PAD)) {
        GET_TILING_DATA_WITH_STRUCT(MaxPoolWithArgmaxV3NhwcTilingData, tilingDataIn, tiling);
        const MaxPoolWithArgmaxV3NhwcTilingData* __restrict tilingData = &tilingDataIn;
        MaxPoolWithArgmaxV3SmallCNameSpace::MaxPoolWithArgmaxV3SmallC<DTYPE_X, DTYPE_ARGMAX, 1> op(
            &pipeBase, tilingData);
        op.Init(x, y, argmax);
        op.MaxPoolWithArgmaxV3SmallCProcess();
    } else if (TILING_KEY_IS(MAX_POOL_WITH_ARGMAX_V3_TILING_KEY_NHWC_BIG_C)) {
        GET_TILING_DATA_WITH_STRUCT(MaxPoolWithArgmaxV3NhwcTilingData, tilingDataIn, tiling);
        const MaxPoolWithArgmaxV3NhwcTilingData* __restrict tilingData = &tilingDataIn;
        MaxPoolWithArgmaxV3NHWC::MaxPoolWithArgmaxV3NhwCKernel<DTYPE_X, DTYPE_ARGMAX, false> op(&pipeBase, tilingData);
        op.Init(x, y, argmax);
        op.Process();
    } else if (TILING_KEY_IS(MAX_POOL_WITH_ARGMAX_V3_TILING_KEY_NHWC_BIG_C_PAD)) {
        GET_TILING_DATA_WITH_STRUCT(MaxPoolWithArgmaxV3NhwcTilingData, tilingDataIn, tiling);
        const MaxPoolWithArgmaxV3NhwcTilingData* __restrict tilingData = &tilingDataIn;
        MaxPoolWithArgmaxV3NHWC::MaxPoolWithArgmaxV3NhwCKernel<DTYPE_X, DTYPE_ARGMAX, true> op(&pipeBase, tilingData);
        op.Init(x, y, argmax);
        op.Process();
    } else if (TILING_KEY_IS(MAX_POOL_WITH_ARGMAX_V3_TILING_KEY_SIMT_NCHW)) {
        GET_TILING_DATA_WITH_STRUCT(MaxPoolWithArgmaxV3SimtTilingData, tilingDataIn, tiling);
        const MaxPoolWithArgmaxV3SimtTilingData* __restrict tilingData = &tilingDataIn;
        MaxPoolWithArgmaxV3<DTYPE_X, DTYPE_ARGMAX, NCHW, false> op(tilingData);
        op.Init(x, y, argmax);
        op.Process();
    } else if (TILING_KEY_IS(MAX_POOL_WITH_ARGMAX_V3_TILING_KEY_SIMT_NHWC)) {
        GET_TILING_DATA_WITH_STRUCT(MaxPoolWithArgmaxV3SimtTilingData, tilingDataIn, tiling);
        const MaxPoolWithArgmaxV3SimtTilingData* __restrict tilingData = &tilingDataIn;
        MaxPoolWithArgmaxV3<DTYPE_X, DTYPE_ARGMAX, NHWC, false> op(tilingData);
        op.Init(x, y, argmax);
        op.Process();
    } else if (TILING_KEY_IS(MAX_POOL_WITH_ARGMAX_V3_TILING_KEY_SIMT_NCHW_PAD)) {
        GET_TILING_DATA_WITH_STRUCT(MaxPoolWithArgmaxV3SimtTilingData, tilingDataIn, tiling);
        const MaxPoolWithArgmaxV3SimtTilingData* __restrict tilingData = &tilingDataIn;
        MaxPoolWithArgmaxV3<DTYPE_X, DTYPE_ARGMAX, NCHW, true> op(tilingData);
        op.Init(x, y, argmax);
        op.Process();
    } else if (TILING_KEY_IS(MAX_POOL_WITH_ARGMAX_V3_TILING_KEY_SIMT_NHWC_PAD)) {
        GET_TILING_DATA_WITH_STRUCT(MaxPoolWithArgmaxV3SimtTilingData, tilingDataIn, tiling);
        const MaxPoolWithArgmaxV3SimtTilingData* __restrict tilingData = &tilingDataIn;
        MaxPoolWithArgmaxV3<DTYPE_X, DTYPE_ARGMAX, NHWC, true> op(tilingData);
        op.Init(x, y, argmax);
        op.Process();
    } else if (TILING_KEY_IS(BIG_KERNEL_FORMAT_NCHW)) {
        GET_TILING_DATA_WITH_STRUCT(MaxPoolWithArgmaxV3BigKernelTilingData, tilingDataIn, tiling);
        const MaxPoolWithArgmaxV3BigKernelTilingData* __restrict tilingData = &tilingDataIn;
        MaxPoolWithArgMaxV3BigKernel::MaxPoolWithArgmaxV3BigKernel<DTYPE_X, float, DTYPE_ARGMAX> op(
            &pipeBase, tilingData);
        op.Init(x, y, argmax);
        op.Process();
    } else if (TILING_KEY_IS(NO_PADDING_TILING_KEY)) {
        GET_TILING_DATA_WITH_STRUCT(MaxPoolWithArgmaxV3GatherTilingData, tilingDataIn, tiling);
        MaxPoolWithArgmaxV3GatherNameSpace::MaxPoolWithArgmaxV3GatherKernel<DTYPE_X, DTYPE_ARGMAX, PAD_DISABLE> op(
            pipeBase, tilingDataIn);
        op.Init(x, y, argmax);
        op.Process();
    } else if (TILING_KEY_IS(PADDING_TILING_KEY)) {
        GET_TILING_DATA_WITH_STRUCT(MaxPoolWithArgmaxV3GatherTilingData, tilingDataIn, tiling);
        MaxPoolWithArgmaxV3GatherNameSpace::MaxPoolWithArgmaxV3GatherKernel<DTYPE_X, DTYPE_ARGMAX, PAD_ENABLE> op(
            pipeBase, tilingDataIn);
        op.Init(x, y, argmax);
        op.Process();
    } else if (TILING_KEY_IS(MAX_POOL_WITH_ARGMAX_V3_TILING_KEY_BIG_KERNEL_SPLIT_FP_INT32)) {
        TPipe pipeBase;
        GET_TILING_DATA_WITH_STRUCT(MaxPoolWithArgmaxV3BigKernelMulCoreTilingData, tilingDataIn, tiling);
        const MaxPoolWithArgmaxV3BigKernelMulCoreTilingData* __restrict tilingData = &tilingDataIn;
        MaxPoolWithArgmaxV3BigKernelMulCore::MaxPoolWithArgmaxV3BigKernelMulCore<float, float, int32_t> op;
        op.Init(x, y, argmax, workspace, &pipeBase, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(MAX_POOL_WITH_ARGMAX_V3_TILING_KEY_BIG_KERNEL_SPLIT_FP_INT64)) {
        TPipe pipeBase;
        GET_TILING_DATA_WITH_STRUCT(MaxPoolWithArgmaxV3BigKernelMulCoreTilingData, tilingDataIn, tiling);
        const MaxPoolWithArgmaxV3BigKernelMulCoreTilingData* __restrict tilingData = &tilingDataIn;
        MaxPoolWithArgmaxV3BigKernelMulCore::MaxPoolWithArgmaxV3BigKernelMulCore<float, float, int64_t> op;
        op.Init(x, y, argmax, workspace, &pipeBase, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(MAX_POOL_WITH_ARGMAX_V3_TILING_KEY_BIG_KERNEL_SPLIT_BF16_INT32)) {
        TPipe pipeBase;
        GET_TILING_DATA_WITH_STRUCT(MaxPoolWithArgmaxV3BigKernelMulCoreTilingData, tilingDataIn, tiling);
        const MaxPoolWithArgmaxV3BigKernelMulCoreTilingData* __restrict tilingData = &tilingDataIn;
        MaxPoolWithArgmaxV3BigKernelMulCore::MaxPoolWithArgmaxV3BigKernelMulCore<bfloat16_t, float, int32_t> op;
        op.Init(x, y, argmax, workspace, &pipeBase, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(MAX_POOL_WITH_ARGMAX_V3_TILING_KEY_BIG_KERNEL_SPLIT_BF16_INT64)) {
        TPipe pipeBase;
        GET_TILING_DATA_WITH_STRUCT(MaxPoolWithArgmaxV3BigKernelMulCoreTilingData, tilingDataIn, tiling);
        const MaxPoolWithArgmaxV3BigKernelMulCoreTilingData* __restrict tilingData = &tilingDataIn;
        MaxPoolWithArgmaxV3BigKernelMulCore::MaxPoolWithArgmaxV3BigKernelMulCore<bfloat16_t, float, int64_t> op;
        op.Init(x, y, argmax, workspace, &pipeBase, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(MAX_POOL_WITH_ARGMAX_V3_TILING_KEY_BIG_KERNEL_SPLIT_HALF_INT32)) {
        TPipe pipeBase;
        GET_TILING_DATA_WITH_STRUCT(MaxPoolWithArgmaxV3BigKernelMulCoreTilingData, tilingDataIn, tiling);
        const MaxPoolWithArgmaxV3BigKernelMulCoreTilingData* __restrict tilingData = &tilingDataIn;
        MaxPoolWithArgmaxV3BigKernelMulCore::MaxPoolWithArgmaxV3BigKernelMulCore<half, half, int32_t> op;
        op.Init(x, y, argmax, workspace, &pipeBase, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(MAX_POOL_WITH_ARGMAX_V3_TILING_KEY_BIG_KERNEL_SPLIT_HALF_INT64)) {
        TPipe pipeBase;
        GET_TILING_DATA_WITH_STRUCT(MaxPoolWithArgmaxV3BigKernelMulCoreTilingData, tilingDataIn, tiling);
        const MaxPoolWithArgmaxV3BigKernelMulCoreTilingData* __restrict tilingData = &tilingDataIn;
        MaxPoolWithArgmaxV3BigKernelMulCore::MaxPoolWithArgmaxV3BigKernelMulCore<half, half, int64_t> op;
        op.Init(x, y, argmax, workspace, &pipeBase, tilingData);
        op.Process();
    }
}
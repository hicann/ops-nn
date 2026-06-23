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
 * \file avg_pool_apt.cpp
 * \brief avg_pool implied
 */

#include <cstdint>
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "arch35/avg_pool_nhwc_small_kernel.h"
#include "arch35/avg_pool_nhwc_small_kernel_pad.h"
#include "arch35/avg_pool_nchw_small_kernel.h"
#include "arch35/avg_pool_simt.h"
#include "arch35/avg_pool_nchw_small_kernel_pad.h"
#include "arch35/avg_pool_big_kernel_nhwc.h"
#include "arch35/avg_pool_big_kernel.h"
#include "arch35/avg_pool_struct.h"

#define AVG_POOL_TILING_KEY_SIMT_NCHW_INT32 911100
#define AVG_POOL_TILING_KEY_SIMT_NHWC_INT32 911101
#define AVG_POOL_TILING_KEY_SIMT_NCHW_INT64 911110
#define AVG_POOL_TILING_KEY_SIMT_NHWC_INT64 911111
#define AVG_POOL_TILING_KEY_SMALL_KERNEL_NO_PADDING_NCHW 300001
#define AVG_POOL_TILING_KEY_SMALL_KERNEL_PADDING_NCHW 300002
#define AVG_POOL_TILING_KEY_SMALL_KERNEL_PADDING_OUT_DIV_NCHW 300003
#define AVG_POOL_TILING_KEY_SMALL_KERNEL_NO_PADDING_SPARSE 300004
#define AVG_POOL_TILING_KEY_ONE_KSIZE  100001
#define AVG_POOL_TILING_KEY_BIG_KERNEL_NHWC 411110
#define AVG_POOL_TILING_KEY_SMALL_KERNEL_NHWC      200001
#define AVG_POOL_TILING_KEY_SMALL_KERNEL_NHWC_PAD  211110
#define AVG_POOL_TILING_KEY_BIG_CHANNELS_NHWC      222220
#define AVG_POOL_TILING_KEY_BIG_CHANNELS_NHWC_PAD  222221
#define AVG_POOL_TILING_KEY_SMALL_KERNEL_NHWC_PAD_DIV 211111
#define AVG_POOL_TILING_KEY_BIG_CHANNELS_NHWC_PAD_DIV 222222
#define AVG_POOL_BIG_KERNEL_FORMAT_NCHW 511110

using namespace AvgPool;

constexpr int32_t NCHW = 0;
constexpr int32_t NHWC = 1;

extern "C" __global__ __aicore__ void avg_pool(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    AscendC::TPipe pipeBase;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(AvgPoolTilingData);
    if (TILING_KEY_IS(AVG_POOL_TILING_KEY_SMALL_KERNEL_PADDING_NCHW)) {
        REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 300002", AvgPoolNCHWSmallKernelTilingData);
        GET_TILING_DATA_WITH_STRUCT(AvgPoolNCHWSmallKernelTilingData, tilingData, tiling);
        AvgPool::AvgPoolNCHWSmallPadKernel<DTYPE_X> op(&pipeBase, &tilingData);
        op.Init(x, y);
        op.Process();
    } else if (TILING_KEY_IS(AVG_POOL_TILING_KEY_SMALL_KERNEL_PADDING_OUT_DIV_NCHW)) {
        REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 300003", AvgPoolNCHWSmallKernelTilingData);
        GET_TILING_DATA_WITH_STRUCT(AvgPoolNCHWSmallKernelTilingData, tilingData, tiling);
        AvgPool::AvgPoolNCHWSmallPadKernel<DTYPE_X, true> op(&pipeBase, &tilingData);
        op.Init(x, y);
        op.Process();
    } else if (TILING_KEY_IS(AVG_POOL_TILING_KEY_SMALL_KERNEL_NHWC)) {
        REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 200001", AvgPoolNHWCSmallKernelTilingData);
        GET_TILING_DATA_WITH_STRUCT(AvgPoolNHWCSmallKernelTilingData, tilingData, tiling);
        AvgPool::AvgPoolNHWCSmallKernel<DTYPE_X> op(&pipeBase, &tilingData);
        op.Init(x, y);
        op.Process();
    } else if (TILING_KEY_IS(AVG_POOL_TILING_KEY_SMALL_KERNEL_NHWC_PAD)) {
        REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 211110", AvgPoolNHWCSmallKernelTilingData);
        GET_TILING_DATA_WITH_STRUCT(AvgPoolNHWCSmallKernelTilingData, tilingData, tiling);
        AvgPool::AvgPoolNHWCSmallKernelPad<DTYPE_X> op(&pipeBase, &tilingData);
        op.Init(x, y);
        op.Process();
    }  else if (TILING_KEY_IS(AVG_POOL_TILING_KEY_SMALL_KERNEL_NHWC_PAD_DIV)) {
        REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 211111", AvgPoolNHWCSmallKernelTilingData);
        GET_TILING_DATA_WITH_STRUCT(AvgPoolNHWCSmallKernelTilingData, tilingData, tiling);
        AvgPool::AvgPoolNHWCSmallKernelPad<DTYPE_X, true> op(&pipeBase, &tilingData);
        op.Init(x, y);
        op.Process();
    } else if (TILING_KEY_IS(AVG_POOL_TILING_KEY_SMALL_KERNEL_NO_PADDING_NCHW)) {
        REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 300001", AvgPoolNCHWSmallKernelTilingData);
        GET_TILING_DATA_WITH_STRUCT(AvgPoolNCHWSmallKernelTilingData, tilingData, tiling);
        AvgPool::AvgPoolSmallKernel<DTYPE_X> op(&pipeBase, &tilingData);
        op.Init(x, y);
        op.Process();
    } else if (TILING_KEY_IS(AVG_POOL_TILING_KEY_BIG_KERNEL_NHWC)) {
        REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 411110", AvgPoolBigKernelNhwcTilingData);
        GET_TILING_DATA_WITH_STRUCT(AvgPoolBigKernelNhwcTilingData, tilingData, tiling);
        AvgPoolNhwcBigKernel<DTYPE_X> op(&pipeBase, &tilingData);
        op.Init(x, y);
        op.Process();
    } else if (TILING_KEY_IS(AVG_POOL_BIG_KERNEL_FORMAT_NCHW)) {
        REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 511110", AvgPoolBigKernelTilingData);
        GET_TILING_DATA_WITH_STRUCT(AvgPoolBigKernelTilingData, tilingData, tiling);
        AvgPool::AvgPoolBigKernel<DTYPE_X> op(&pipeBase, &tilingData);
        op.Init(x, y);
        op.Process();
    } else if (TILING_KEY_IS(AVG_POOL_TILING_KEY_SIMT_NCHW_INT32)) {
        REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 911100", AvgPoolSimtTilingData);
        GET_TILING_DATA_WITH_STRUCT(AvgPool::AvgPoolSimtTilingData, tilingData, tiling);
        AvgPoolSimt::AvgPoolSimtImpl<DTYPE_X, int32_t, NCHW> op(&pipeBase, &tilingData);
        op.Init(x, y);
        op.Process();
    } else if (TILING_KEY_IS(AVG_POOL_TILING_KEY_SIMT_NHWC_INT32)) {
        REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 911101", AvgPoolSimtTilingData);
        GET_TILING_DATA_WITH_STRUCT(AvgPool::AvgPoolSimtTilingData, tilingData, tiling);
        AvgPoolSimt::AvgPoolSimtImpl<DTYPE_X, int32_t, NHWC> op(&pipeBase, &tilingData);
        op.Init(x, y);
        op.Process();
    } else if (TILING_KEY_IS(AVG_POOL_TILING_KEY_SIMT_NCHW_INT64)) {
        REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 911110", AvgPoolSimtTilingData);
        GET_TILING_DATA_WITH_STRUCT(AvgPool::AvgPoolSimtTilingData, tilingData, tiling);
        AvgPoolSimt::AvgPoolSimtImpl<DTYPE_X, int64_t, NCHW> op(&pipeBase, &tilingData);
        op.Init(x, y);
        op.Process();
    } else if (TILING_KEY_IS(AVG_POOL_TILING_KEY_SIMT_NHWC_INT64)) {
        REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 911111", AvgPoolSimtTilingData);
        GET_TILING_DATA_WITH_STRUCT(AvgPool::AvgPoolSimtTilingData, tilingData, tiling);
        AvgPoolSimt::AvgPoolSimtImpl<DTYPE_X, int64_t, NHWC> op(&pipeBase, &tilingData);
        op.Init(x, y);
        op.Process();
    }
}
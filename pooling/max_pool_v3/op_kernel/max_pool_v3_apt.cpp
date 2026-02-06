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
 * \file max_pool_v3.cpp
 * \brief max_pool_v3 implied
 */

#include <cstdint>
#include "arch35/max_pool_v3_big_kernel.h"
#include "arch35/max_pool_v3_small_kernel.h"
#include "arch35/max_pool_v3_small_kernel_pad.h"
#include "arch35/max_pool_v3_nhwc_big_kernel.h"
#include "arch35/max_pool_v3_nhwc_small_kernel.h"
#include "arch35/max_pool_v3_nhwc_small_kernel_pad.h"

#define BIG_KERNEL_FORMAT_NCHW 311110
#define SMALL_KERNEL_NO_PADDING_FORMAT_NCHW 300001
#define SMALL_KERNEL_PADDING_FORMAT_NCHW 300002
#define BIG_KERNEL_FORMAT_NHWC 411110
#define SMALL_KERNEL_NO_PADDING_FORMAT_NHWC 400001
#define SMALL_KERNEL_PADDING_FORMAT_NHWC 400002

extern "C" __global__ __aicore__ void max_pool_v3(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    AscendC::TPipe pipeBase;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TILING_KEY_IS(BIG_KERNEL_FORMAT_NCHW);
    TILING_KEY_IS(SMALL_KERNEL_NO_PADDING_FORMAT_NCHW);
    TILING_KEY_IS(SMALL_KERNEL_PADDING_FORMAT_NCHW);
    TILING_KEY_IS(BIG_KERNEL_FORMAT_NHWC);
    TILING_KEY_IS(SMALL_KERNEL_NO_PADDING_FORMAT_NHWC);
    TILING_KEY_IS(SMALL_KERNEL_PADDING_FORMAT_NHWC);
    #if TILING_KEY_X == BIG_KERNEL_FORMAT_NCHW
        GET_TILING_DATA_WITH_STRUCT(MaxPoolV3BigKernelTilingData, tilingData, tiling);
        MaxPoolV3::MaxPoolV3BigKernel<DTYPE_X> op(&pipeBase, &tilingData);
        op.Init(x, y);
        op.Process();
    #endif

    #if TILING_KEY_X == SMALL_KERNEL_NO_PADDING_FORMAT_NCHW
        GET_TILING_DATA_WITH_STRUCT(MaxPoolV3SmallKernelTilingData, tilingData, tiling);
        MaxPoolV3::MaxPoolV3SmallKernel<DTYPE_X> op(&pipeBase, &tilingData);
        op.Init(x, y);
        op.Process();
    #endif

    #if TILING_KEY_X == SMALL_KERNEL_PADDING_FORMAT_NCHW
        GET_TILING_DATA_WITH_STRUCT(MaxPoolV3SmallKernelTilingData, tilingData, tiling);
        MaxPoolV3::MaxPoolV3SmallPadKernel<DTYPE_X> op(&pipeBase, &tilingData);
        op.Init(x, y);
        op.Process();
    #endif

    #if TILING_KEY_X == BIG_KERNEL_FORMAT_NHWC
        GET_TILING_DATA_WITH_STRUCT(MaxPoolV3NHWCBigKernelTilingData, tilingData, tiling);
        MaxPoolV3::MaxPoolV3NHWCBigKernel<DTYPE_X> op(&pipeBase, &tilingData);
        op.Init(x, y);
        op.Process();
    #endif

    #if TILING_KEY_X == SMALL_KERNEL_NO_PADDING_FORMAT_NHWC
        GET_TILING_DATA_WITH_STRUCT(MaxPoolV3NHWCSmallKernelTilingData, tilingData, tiling);
        MaxPoolV3::MaxPoolV3NHWCSmallKernel<DTYPE_X> op(&pipeBase, &tilingData);
        op.Init(x, y);
        op.Process();
    #endif

    #if TILING_KEY_X == SMALL_KERNEL_PADDING_FORMAT_NHWC
        GET_TILING_DATA_WITH_STRUCT(MaxPoolV3NHWCSmallKernelTilingData, tilingData, tiling);
        MaxPoolV3::MaxPoolV3NHWCSmallPadKernel<DTYPE_X> op(&pipeBase, &tilingData);
        op.Init(x, y);
        op.Process();
    #endif
}
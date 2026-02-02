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
 * \file adaptive_avg_pool3d.cpp
 * \brief
 */

#include "kernel_operator.h"

#include "adaptive_avg_pool3d_split_c.h"
#include "adaptive_avg_pool3d_multi_w.h"
#include "adaptive_avg_pool3d_split_w.h"

using namespace AdaptiveAvgPool3d;

#define DISPATCH_OP_IMPL(KernelImpl, ...)              \
    do {                                               \
        KernelImpl<__VA_ARGS__> op;                    \
        TPipe tPipe;                                   \
        op.Init(x, y, workspace, &tilingData, &tPipe); \
        op.Process();                                  \
    } while (0)

extern "C" __global__ __aicore__ void adaptive_avg_pool3d(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    if (TILING_KEY_IS(11)) {
        DISPATCH_OP_IMPL(KernelAdaptiveAvgPool3dSplitC, half, 1);
#if __CCE_AICORE__ >= 220 && !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
    } else if (TILING_KEY_IS(10)) {
        DISPATCH_OP_IMPL(KernelAdaptiveAvgPool3dSplitC, bfloat16_t, 1);
#endif
    } else if (TILING_KEY_IS(12)) {
        DISPATCH_OP_IMPL(KernelAdaptiveAvgPool3dSplitC, float, 1);
#if __CCE_AICORE__ >= 220 && !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
    } else if (TILING_KEY_IS(20)) {
        DISPATCH_OP_IMPL(KernelAdaptiveAvgPool3dSplitW, bfloat16_t, 1);
#endif
    } else if (TILING_KEY_IS(21)) {
        DISPATCH_OP_IMPL(KernelAdaptiveAvgPool3dSplitW, half, 1);
    } else if (TILING_KEY_IS(22)) {
        DISPATCH_OP_IMPL(KernelAdaptiveAvgPool3dSplitW, float, 1);
#if __CCE_AICORE__ >= 220 && !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
    } else if (TILING_KEY_IS(30)) {
        DISPATCH_OP_IMPL(KernelAdaptiveAvgPool3dMultiW, bfloat16_t, 1);
#endif
    } else if (TILING_KEY_IS(31)) {
        DISPATCH_OP_IMPL(KernelAdaptiveAvgPool3dMultiW, half, 1);
    } else if (TILING_KEY_IS(32)) {
        DISPATCH_OP_IMPL(KernelAdaptiveAvgPool3dMultiW, float, 1);
    }
}

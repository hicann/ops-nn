/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file adaptive_avg_pool2d_apt.cpp
 * \brief
 */

#include "arch35/adaptive_avg_pool2d_big_kernel.h"
#include "arch35/adaptive_avg_pool2d_simt.h"
#include "arch35/adaptive_avg_pool2d_small_kernel.h"
#include "arch35/adaptive_avg_pool2d_split_c.h"
#include "arch35/adaptive_avg_pool2d_split_h.h"
#include "arch35/adaptive_avg_pool2d_split_w.h"
#include "arch35/adaptive_avg_pool2d_upsample_h.h"
#include "arch35/adaptive_avg_pool2d_struct.h"

using namespace AdaptiveAvgPool2dOp;

#define DISPATCH_SPLIT_KERNEL(Namespace, KernelClass, TilingDataType)                   \
    do {                                                                                \
        GET_TILING_DATA_WITH_STRUCT(TilingDataType, tilingData, tiling);                \
        if constexpr (DTYPE_MODE == TPL_INT32_UINT32) {                                 \
            Namespace::KernelClass<DTYPE_X, int32_t, NC_FACTOR> op(&tilingData, &pipe); \
            op.Init(x, y);                                                              \
            op.Process();                                                               \
        } else {                                                                        \
            Namespace::KernelClass<DTYPE_X, int64_t, NC_FACTOR> op(&tilingData, &pipe); \
            op.Init(x, y);                                                              \
            op.Process();                                                               \
        }                                                                               \
    } while (0)

template <uint64_t TEMPLATE_MODE = TPL_SIMT_KERNEL, uint64_t DTYPE_MODE = TPL_INT32_UINT32, uint64_t NC_FACTOR,
          uint64_t BIG_KERNEL_COPY_MODE = TPL_BIG_KERNEL_NDDMA>
__global__ __aicore__ void adaptive_avg_pool2d(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    if (workspace == nullptr || GetUserWorkspace(workspace) == nullptr || g_coreType == AIC) {
        return;
    }
    TPipe pipe;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(AdaptivePool2DSimtTilingData);
    if constexpr (TEMPLATE_MODE == TPL_SMALL_KERNEL) {
        DISPATCH_SPLIT_KERNEL(AdaptivePool2dSmallKernelNamespace, AdaptiveAvgPool2dSmallKernel,
                              AdaptivePool2dSmallKernelTilingData);
    } else if constexpr (TEMPLATE_MODE == TPL_SIMT_KERNEL) {
        GET_TILING_DATA_WITH_STRUCT(AdaptivePool2DSimtTilingData, tilingData, tiling);
        if constexpr (DTYPE_MODE == TPL_INT32_UINT32) {
            AdaptiveAvgPool2dSimt<DTYPE_X, uint32_t> op(&pipe, &tilingData);
            op.Init(x, y);
            op.Process();
        } else {
            AdaptiveAvgPool2dSimt<DTYPE_X, uint64_t> op(&pipe, &tilingData);
            op.Init(x, y);
            op.Process();
        }
    } else if constexpr (TEMPLATE_MODE == TPL_BIG_KERNEL) {
        GET_TILING_DATA_WITH_STRUCT(AdaptivePool2dBigKernelTilingData, tilingData, tiling);
        AdaptiveAvgPool2dBigKernel<DTYPE_X, BIG_KERNEL_COPY_MODE> op(tilingData, pipe);
        op.Init(x, y);
        op.Process();
    } else if constexpr (TEMPLATE_MODE == TPL_SPLIT_W_KERNEL) {
        DISPATCH_SPLIT_KERNEL(AdaptivePool2dSplitWNamespace, AdaptiveAvgPool2dSplitW, AdaptivePool2dSplitWTilingData);
    } else if constexpr (TEMPLATE_MODE == TPL_SPLIT_C_KERNEL) {
        DISPATCH_SPLIT_KERNEL(AdaptivePool2dSplitCNamespace, AdaptiveAvgPool2dSplitC, AdaptivePool2dSplitCTilingData);
    } else if constexpr (TEMPLATE_MODE == TPL_SPLIT_H_KERNEL) {
        DISPATCH_SPLIT_KERNEL(AdaptivePool2dSplitHNamespace, AdaptiveAvgPool2dSplitH, AdaptivePool2dSplitHTilingData);
    } else if constexpr (TEMPLATE_MODE == TPL_UPSAMPLE_H_KERNEL) {
        DISPATCH_SPLIT_KERNEL(AdaptivePool2dUpsampleHNamespace, AdaptiveAvgPool2dUpsampleH,
                              AdaptivePool2dUpsampleHTilingData);
    } else {
        GET_TILING_DATA_WITH_STRUCT(AdaptivePool2dBigKernelTilingData, tilingData, tiling);
        AdaptiveAvgPool2dBigKernel<DTYPE_X, BIG_KERNEL_COPY_MODE> op(tilingData, pipe);
        op.Init(x, y);
        op.Process();
    }
}

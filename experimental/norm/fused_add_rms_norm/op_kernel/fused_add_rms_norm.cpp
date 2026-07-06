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
 * \file fused_add_rms_norm.cpp
 * \brief
 */
#include "fused_add_rms_norm.h"
#include "fused_add_rms_norm_split_d.h"
#include "fused_add_rms_norm_merge_n.h"
#include "fused_add_rms_norm_multi_n.h"
#include "fused_add_rms_norm_single_n.h"

using namespace AscendC;

#define GENERAL_OP_IMPL(templateClass)                   \
    do {                                                 \
        templateClass<DTYPE_X1> op(&pipe);               \
        op.Init(x1, x2, gamma, y, rstd, x, &tilingData); \
        op.Process();                                    \
    } while (0)

extern "C" __global__ __aicore__ void fused_add_rms_norm(
    GM_ADDR x1, GM_ADDR x2, GM_ADDR gamma, GM_ADDR y, GM_ADDR rstd, GM_ADDR x, GM_ADDR workspace, GM_ADDR tiling)
{
    TPipe pipe;
    GET_TILING_DATA(tilingData, tiling);
    if (TILING_KEY_IS(0)) {
        GENERAL_OP_IMPL(KernelFusedAddRmsNorm);
    } else if (TILING_KEY_IS(1)) {
        GENERAL_OP_IMPL(KernelFusedAddRmsNormSplitD);
    } else if (TILING_KEY_IS(2)) {
        GENERAL_OP_IMPL(KernelFusedAddRmsNormMergeN);
    } else if (TILING_KEY_IS(3)) {
        GENERAL_OP_IMPL(KernelFusedAddRmsNormSingleN);
    } else if (TILING_KEY_IS(4)) {
        GENERAL_OP_IMPL(KernelFusedAddRmsNormMultiN);
    }
}
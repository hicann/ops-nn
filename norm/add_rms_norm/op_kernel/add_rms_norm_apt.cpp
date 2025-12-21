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
 * \file add_rms_norm.cpp
 * \brief
 */
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
#include "arch35/add_rms_norm_regbase.h"
#include "arch35/add_rms_norm_regbase_split_d.h"
#else
#include "arch35/add_rms_norm.h"
#include "arch35/add_rms_norm_split_d.h"
#include "arch35/add_rms_norm_merge_n.h"
#include "arch35/add_rms_norm_multi_n.h"
#include "arch35/add_rms_norm_single_n.h"
#endif

using namespace AscendC;
using namespace AddRmsNorm;

#define GENERAL_OP_IMPL(templateClass, ...)              \
    do {                                                 \
        templateClass<__VA_ARGS__> op(&pipe);            \
        op.Init(x1, x2, gamma, y, rstd, x, &tilingData); \
        op.Process();                                    \
    } while (0)

extern "C" __global__ __aicore__ void add_rms_norm(
    GM_ADDR x1, GM_ADDR x2, GM_ADDR gamma, GM_ADDR y, GM_ADDR rstd, GM_ADDR x, GM_ADDR workspace, GM_ADDR tiling)
{
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    GET_TILING_DATA_WITH_STRUCT(AddRMSNormRegbaseTilingData, tilingDataIn, tiling);
    if (TILING_KEY_IS(1)) {
        KernelAddRmsNormRegBase<half> op(&pipe);
        op.Init(x1, x2, gamma, y, rstd, x, &tilingDataIn);
        op.Process();
    } else if (TILING_KEY_IS(2)) {
        KernelAddRmsNormRegBase<float> op(&pipe);
        op.Init(x1, x2, gamma, y, rstd, x, &tilingDataIn);
        op.Process();
    } else if (TILING_KEY_IS(3)) {
        KernelAddRmsNormRegBase<bfloat16_t> op(&pipe);
        op.Init(x1, x2, gamma, y, rstd, x, &tilingDataIn);
        op.Process();
    } else if (TILING_KEY_IS(1001)) {
        KernelAddRmsNormRegBaseSplitD<half> op(&pipe);
        op.Init(x1, x2, gamma, y, rstd, x, &tilingDataIn);
        op.Process();
    } else if (TILING_KEY_IS(1002)) {
        KernelAddRmsNormRegBaseSplitD<float> op(&pipe);
        op.Init(x1, x2, gamma, y, rstd, x, &tilingDataIn);
        op.Process();
    } else if (TILING_KEY_IS(1003)) {
        KernelAddRmsNormRegBaseSplitD<bfloat16_t> op(&pipe);
        op.Init(x1, x2, gamma, y, rstd, x, &tilingDataIn);
        op.Process();
    }
#else
    GET_TILING_DATA(tilingData, tiling);
    if (TILING_KEY_IS(10)) {
        GENERAL_OP_IMPL(KernelAddRmsNorm, half);
    } else if (TILING_KEY_IS(20)) {
        GENERAL_OP_IMPL(KernelAddRmsNorm, float);
    } else if (TILING_KEY_IS(30)) {
        GENERAL_OP_IMPL(KernelAddRmsNorm, bfloat16_t);
    } else if (TILING_KEY_IS(11)) {
        GENERAL_OP_IMPL(KernelAddRmsNormSplitD, half);
    } else if (TILING_KEY_IS(21)) {
        GENERAL_OP_IMPL(KernelAddRmsNormSplitD, float);
    } else if (TILING_KEY_IS(31)) {
        GENERAL_OP_IMPL(KernelAddRmsNormSplitD, bfloat16_t);
    } else if (TILING_KEY_IS(12)) {
        GENERAL_OP_IMPL(KernelAddRmsNormMergeN, half);
    } else if (TILING_KEY_IS(22)) {
        GENERAL_OP_IMPL(KernelAddRmsNormMergeN, float);
    } else if (TILING_KEY_IS(32)) {
        GENERAL_OP_IMPL(KernelAddRmsNormMergeN, bfloat16_t);
    } else if (TILING_KEY_IS(13)) {
        GENERAL_OP_IMPL(KernelAddRmsNormSingleN, half);
    } else if (TILING_KEY_IS(23)) {
        GENERAL_OP_IMPL(KernelAddRmsNormSingleN, float);
    } else if (TILING_KEY_IS(33)) {
        GENERAL_OP_IMPL(KernelAddRmsNormSingleN, bfloat16_t);
    } else if (TILING_KEY_IS(14)) {
        GENERAL_OP_IMPL(KernelAddRmsNormMultiN, half);
    }
#endif
}
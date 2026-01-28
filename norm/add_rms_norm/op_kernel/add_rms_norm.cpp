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
 * \file add_rms_norm.cpp
 * \brief
 */
#include "add_rms_norm.h"
#include "add_rms_norm_split_d.h"
#include "add_rms_norm_merge_n.h"
#include "add_rms_norm_multi_n.h"
#include "add_rms_norm_single_n.h"

using namespace AscendC;

#define GENERAL_OP_IMPL(templateClass, ...)              \
    do {                                                 \
        templateClass<__VA_ARGS__> op(&pipe);            \
        op.Init(x1, x2, gamma, y, rstd, x, workspace, &tilingData); \
        op.Process();                                    \
    } while (0)

extern "C" __global__ __aicore__ void add_rms_norm(
    GM_ADDR x1, GM_ADDR x2, GM_ADDR gamma, GM_ADDR y, GM_ADDR rstd, GM_ADDR x, GM_ADDR workspace, GM_ADDR tiling)
{
    TPipe pipe;
    GET_TILING_DATA(tilingData, tiling);
    if (TILING_KEY_IS(10)) {
        GENERAL_OP_IMPL(KernelAddRmsNorm, half, 1);
    } else if (TILING_KEY_IS(20)) {
        GENERAL_OP_IMPL(KernelAddRmsNorm, float, 1);
    } else if (TILING_KEY_IS(30)) {
#if !(defined(__NPU_ARCH__) && __NPU_ARCH__ == 3003)
        GENERAL_OP_IMPL(KernelAddRmsNorm, bfloat16_t, 1);
#endif
    } else if (TILING_KEY_IS(11)) {
        GENERAL_OP_IMPL(KernelAddRmsNormSplitD, half, 1);
    } else if (TILING_KEY_IS(21)) {
        GENERAL_OP_IMPL(KernelAddRmsNormSplitD, float, 1);
    } else if (TILING_KEY_IS(31)) {
#if !(defined(__NPU_ARCH__) && __NPU_ARCH__ == 3003)
        GENERAL_OP_IMPL(KernelAddRmsNormSplitD, bfloat16_t, 1);
#endif
    } else if (TILING_KEY_IS(12)) {
        GENERAL_OP_IMPL(KernelAddRmsNormMergeN, half, 1);
    } else if (TILING_KEY_IS(22)) {
        GENERAL_OP_IMPL(KernelAddRmsNormMergeN, float, 1);
    } else if (TILING_KEY_IS(32)) {
#if !(defined(__NPU_ARCH__) && __NPU_ARCH__ == 3003)
        GENERAL_OP_IMPL(KernelAddRmsNormMergeN, bfloat16_t, 1);
#endif
    } else if (TILING_KEY_IS(13)) {
        GENERAL_OP_IMPL(KernelAddRmsNormSingleN, half, 1);
    } else if (TILING_KEY_IS(23)) {
        GENERAL_OP_IMPL(KernelAddRmsNormSingleN, float, 1);
    } else if (TILING_KEY_IS(33)) {
#if !(defined(__NPU_ARCH__) && __NPU_ARCH__ == 3003)
        GENERAL_OP_IMPL(KernelAddRmsNormSingleN, bfloat16_t, 1);
#endif
    } else if (TILING_KEY_IS(14)) {
        GENERAL_OP_IMPL(KernelAddRmsNormMultiN, half, 1);
    } else if (TILING_KEY_IS(34)) {
        GENERAL_OP_IMPL(KernelAddRmsNormMultiN, bfloat16_t, 1);
    } else if (TILING_KEY_IS(110)) {
        GENERAL_OP_IMPL(KernelAddRmsNorm, half, 2);
    } else if (TILING_KEY_IS(130)) {
#if !(defined(__NPU_ARCH__) && __NPU_ARCH__ == 3003)
        GENERAL_OP_IMPL(KernelAddRmsNorm, bfloat16_t, 2);
#endif
    } else if (TILING_KEY_IS(111)) {
        GENERAL_OP_IMPL(KernelAddRmsNormSplitD, half, 2);
    } else if (TILING_KEY_IS(131)) {
#if !(defined(__NPU_ARCH__) && __NPU_ARCH__ == 3003)
        GENERAL_OP_IMPL(KernelAddRmsNormSplitD, bfloat16_t, 2);
#endif
    } else if (TILING_KEY_IS(112)) {
        GENERAL_OP_IMPL(KernelAddRmsNormMergeN, half, 2);
    } else if (TILING_KEY_IS(132)) {
#if !(defined(__NPU_ARCH__) && __NPU_ARCH__ == 3003)
        GENERAL_OP_IMPL(KernelAddRmsNormMergeN, bfloat16_t, 2);
#endif
    } else if (TILING_KEY_IS(113)) {
        GENERAL_OP_IMPL(KernelAddRmsNormSingleN, half, 2);
    } else if (TILING_KEY_IS(133)) {
#if !(defined(__NPU_ARCH__) && __NPU_ARCH__ == 3003)
        GENERAL_OP_IMPL(KernelAddRmsNormSingleN, bfloat16_t, 2);
#endif
    } else if (TILING_KEY_IS(114)) {
        GENERAL_OP_IMPL(KernelAddRmsNormMultiN, half, 2);
    } else if (TILING_KEY_IS(134)) {
        GENERAL_OP_IMPL(KernelAddRmsNormMultiN, bfloat16_t, 2);
    } else if (TILING_KEY_IS(1010)) {
        GENERAL_OP_IMPL(KernelAddRmsNorm, half, 3);
    } else if (TILING_KEY_IS(1030)) {
#if !(defined(__NPU_ARCH__) && __NPU_ARCH__ == 3003)
        GENERAL_OP_IMPL(KernelAddRmsNorm, bfloat16_t, 3);
#endif
    } else if (TILING_KEY_IS(1011)) {
        GENERAL_OP_IMPL(KernelAddRmsNormSplitD, half, 3);
    } else if (TILING_KEY_IS(1031)) {
#if !(defined(__NPU_ARCH__) && __NPU_ARCH__ == 3003)
        GENERAL_OP_IMPL(KernelAddRmsNormSplitD, bfloat16_t, 3);
#endif
    } else if (TILING_KEY_IS(1012)) {
        GENERAL_OP_IMPL(KernelAddRmsNormMergeN, half, 3);
    } else if (TILING_KEY_IS(1032)) {
#if !(defined(__NPU_ARCH__) && __NPU_ARCH__ == 3003)
        GENERAL_OP_IMPL(KernelAddRmsNormMergeN, bfloat16_t, 3);
#endif
    } else if (TILING_KEY_IS(1013)) {
        GENERAL_OP_IMPL(KernelAddRmsNormSingleN, half, 3);
    } else if (TILING_KEY_IS(1033)) {
#if !(defined(__NPU_ARCH__) && __NPU_ARCH__ == 3003)
        GENERAL_OP_IMPL(KernelAddRmsNormSingleN, bfloat16_t, 3);
#endif
    } else if (TILING_KEY_IS(1014)) {
        GENERAL_OP_IMPL(KernelAddRmsNormMultiN, half, 3);
    } else if (TILING_KEY_IS(1034)) {
        GENERAL_OP_IMPL(KernelAddRmsNormMultiN, bfloat16_t, 3);
    }
}
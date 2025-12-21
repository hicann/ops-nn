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
 * \file gemma_rms_norm.cpp
 * \brief
 */
#include "../rms_norm/rms_norm.h"
#include "../rms_norm/rms_norm_split_d.h"

using namespace RmsNorm;
using namespace AscendC;

#define GENERAL_OP_IMPL(templateClass, ...)      \
    do {                                         \
        templateClass<__VA_ARGS__> op;           \
        op.Init(x, gamma, y, rstd, &tilingData); \
        op.Process();                            \
    } while (0)

extern "C" __global__ __aicore__ void gemma_rms_norm(
    GM_ADDR x, GM_ADDR gamma, GM_ADDR y, GM_ADDR rstd, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    if (TILING_KEY_IS(0)) {
        GENERAL_OP_IMPL(KernelRmsNorm, DTYPE_X, DTYPE_GAMMA);
    } else if (TILING_KEY_IS(1)) {
        GENERAL_OP_IMPL(KernelRmsNormSplitD, DTYPE_X, DTYPE_GAMMA);
    }
}

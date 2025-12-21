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
 * \file fused_linear_online_max_sum.cpp
 * \brief
 */

#include "fused_linear_online_max_sum.h"

using namespace FusedLinearOnlineMaxSum;

extern "C" __global__ __aicore__ void fused_linear_online_max_sum(
    GM_ADDR input, GM_ADDR weight, GM_ADDR target, GM_ADDR logitsMaxLocal, GM_ADDR sumExpLogitsLocal,
    GM_ADDR predictedLogitsLocal, GM_ADDR targetMask, GM_ADDR maskedTarget, GM_ADDR vocabParallelLogitsOut,
    GM_ADDR workspace, GM_ADDR tiling) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    GET_TILING_DATA(tilingData, tiling);
    AscendC::TPipe pipe;
    __gm__ uint8_t *userWorkspace = GetUserWorkspace(workspace);

#define INIT_AND_PROCESS                                                                                \
    op.Init(tilingData, input, weight, target, logitsMaxLocal, sumExpLogitsLocal, predictedLogitsLocal, \
            targetMask, maskedTarget, vocabParallelLogitsOut, userWorkspace, &pipe);                    \
    op.Process()

#if (defined(DTYPE_INPUT) && defined(DTYPE_TARGET))
    if (TILING_KEY_IS(0)) {
        FusedLinearOnlineMaxSum::FusedLinearOnlineMaxSumOp<DTYPE_INPUT, DTYPE_TARGET, DTYPE_INPUT, 0> op;
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(1)) {
        FusedLinearOnlineMaxSum::FusedLinearOnlineMaxSumOp<DTYPE_INPUT, DTYPE_TARGET, DTYPE_INPUT, 1> op;
        INIT_AND_PROCESS;
    }
    return;
#endif
}

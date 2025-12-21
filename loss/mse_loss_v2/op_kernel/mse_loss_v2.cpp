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
 * \file mse_loss_v2.cpp
 * \brief
 */

#include "mse_loss_v2_none.h"
#include "mse_loss_v2_sum.h"
#include "mse_loss_v2_mean.h"

#define INIT_AND_PROCESS(mode, dtype)               \
    MSELossV2##mode<dtype> op(&pipe, &tiling_data); \
    op.Init(output, input, target, usrWorkspace);   \
    op.Process()

extern "C" __global__ __aicore__ void mse_loss_v2(
    GM_ADDR input, GM_ADDR target, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);

    AscendC::TPipe pipe;
    GM_ADDR usrWorkspace = AscendC::GetUserWorkspace(workspace);

    using namespace MSELossV2Namespace;
    if (TILING_KEY_IS(11)) {
        INIT_AND_PROCESS(None, float);
    } else if (TILING_KEY_IS(12)) {
        INIT_AND_PROCESS(None, half);
    } else if (TILING_KEY_IS(21)) {
        INIT_AND_PROCESS(Sum, float);
    } else if (TILING_KEY_IS(22)) {
        INIT_AND_PROCESS(Sum, half);
    } else if (TILING_KEY_IS(31)) {
        INIT_AND_PROCESS(Mean, float);
    } else if (TILING_KEY_IS(32)) {
        INIT_AND_PROCESS(Mean, half);
#if __CCE_AICORE__ != 200
    } else if (TILING_KEY_IS(13)) {
        INIT_AND_PROCESS(None, bfloat16_t);
    } else if (TILING_KEY_IS(23)) {
        INIT_AND_PROCESS(Sum, bfloat16_t);
    } else if (TILING_KEY_IS(33)) {
        INIT_AND_PROCESS(Mean, bfloat16_t);
#endif
    }
}
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
 * \file logit_grad.cpp
 * \brief logit_grad kernel
 */

#include "logit_grad.h"

using namespace AscendC;

using namespace LogitGrad;

extern "C" __global__ __aicore__ void logit_grad(GM_ADDR x, GM_ADDR dy, GM_ADDR dx, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);

    GM_ADDR userWs = nullptr;

#if __CCE_AICORE__ == 220
    if (TILING_KEY_IS(1)) {
        LogitGradND<half> op;
        op.Init(x, dy, dx, userWs, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(2)) {
        LogitGradND<float> op;
        op.Init(x, dy, dx, userWs, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(3)) {
        LogitGradND<bfloat16_t> op;
        op.Init(x, dy, dx, userWs, &tilingData);
        op.Process();
    }
#else
#endif
}
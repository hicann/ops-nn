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
 * \file cross_entropy_loss.cpp
 * \brief
 */

#include "cross_entropy_loss.h"
#include "cross_entropy_loss_fp32.h"

using namespace CrossEntropyLossCustom;

extern "C" __global__ __aicore__ void cross_entropy_loss(
    GM_ADDR input, GM_ADDR target, GM_ADDR weight, GM_ADDR loss, GM_ADDR log_prob, GM_ADDR zloss, GM_ADDR lse_for_zloss,
    GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    GET_TILING_DATA(tilingData, tiling);
    GM_ADDR usrWorkspace = AscendC::GetUserWorkspace(workspace);
    if (TILING_KEY_IS(1)) {
        CrossEntropyLoss<bfloat16_t> crossEntropyLossOp;
        crossEntropyLossOp.Init(input, target, weight, loss, log_prob, zloss, lse_for_zloss, usrWorkspace, tilingData);
        crossEntropyLossOp.Process();
    }

    if (TILING_KEY_IS(2)) {
        CrossEntropyLoss<half> crossEntropyLossOp;
        crossEntropyLossOp.Init(input, target, weight, loss, log_prob, zloss, lse_for_zloss, usrWorkspace, tilingData);
        crossEntropyLossOp.Process();
    }

    if (TILING_KEY_IS(3)) {
        CrossEntropyLoss<float> crossEntropyLossOp;
        crossEntropyLossOp.Init(input, target, weight, loss, log_prob, zloss, lse_for_zloss, usrWorkspace, tilingData);
        crossEntropyLossOp.ProcessFp32();
    }
}

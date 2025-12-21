/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!.0
 * \file group_norm_swish_grad.cpp
 * \brief
 */
#include "group_norm_swish_grad.h"

extern "C" __global__ __aicore__ void group_norm_swish_grad(
    GM_ADDR dy, GM_ADDR mean, GM_ADDR rstd, GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR dx, GM_ADDR dgamma,
    GM_ADDR dbeta, GM_ADDR workspace, GM_ADDR tilingdata)
{
    GM_ADDR usrWorkspace = AscendC::GetUserWorkspace(workspace);
    GET_TILING_DATA(tiling_data, tilingdata);
    if (TILING_KEY_IS(0)) {
        GroupNormSwishGrad<float, false> opFP32(
            dy, mean, rstd, x, gamma, beta, dx, dgamma, dbeta, usrWorkspace, &tiling_data);
        opFP32.Process();
    } else if (TILING_KEY_IS(1)) {
        GroupNormSwishGrad<half, false> opFP16(
            dy, mean, rstd, x, gamma, beta, dx, dgamma, dbeta, usrWorkspace, &tiling_data);
        opFP16.Process();
    } else if (TILING_KEY_IS(2)) {
        GroupNormSwishGrad<bfloat16_t, false> opBF16(
            dy, mean, rstd, x, gamma, beta, dx, dgamma, dbeta, usrWorkspace, &tiling_data);
        opBF16.Process();
    } else if (TILING_KEY_IS(10)) {
        GroupNormSwishGrad<float, true> opFP32Det(
            dy, mean, rstd, x, gamma, beta, dx, dgamma, dbeta, usrWorkspace, &tiling_data);
        opFP32Det.Process();
    } else if (TILING_KEY_IS(11)) {
        GroupNormSwishGrad<half, true> opFP16Det(
            dy, mean, rstd, x, gamma, beta, dx, dgamma, dbeta, usrWorkspace, &tiling_data);
        opFP16Det.Process();
    } else if (TILING_KEY_IS(12)) {
        GroupNormSwishGrad<bfloat16_t, true> opBF16Det(
            dy, mean, rstd, x, gamma, beta, dx, dgamma, dbeta, usrWorkspace, &tiling_data);
        opBF16Det.Process();
    }
}

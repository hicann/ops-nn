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
 * \file top_k_top_p_sample.cpp
 * \brief
 */

#include "kernel_operator.h"
#include "top_k_top_p_sample.h"

using namespace AscendC;
using namespace TopKTopPSample;

extern "C" __global__ __aicore__ void top_k_top_p_sample(
    GM_ADDR logits, GM_ADDR topKs, GM_ADDR topPs, GM_ADDR q,
    GM_ADDR logitsSelectIdx, GM_ADDR logitsTopKpSelect, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    GM_ADDR usrWorkspace = GetUserWorkspace(workspace);
    TPipe pipe;
    if (TILING_KEY_IS(1001)) {
        TopKTopPSampleKernel<half> op;
        op.Init(logits, topKs, topPs, q, logitsSelectIdx, logitsTopKpSelect, usrWorkspace, tilingData, &pipe);
        op.Process();
        pipe.Destroy();
    } else if (TILING_KEY_IS(1027)) {
        TopKTopPSampleKernel<bfloat16_t> op;
        op.Init(logits, topKs, topPs, q, logitsSelectIdx, logitsTopKpSelect, usrWorkspace, tilingData, &pipe);
        op.Process();
        pipe.Destroy();
    } else {
        return;
    }
}
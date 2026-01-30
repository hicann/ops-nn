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
 * \file top_k_top_p_sample_v2.cpp
 * \brief
 */

#include "kernel_operator.h"
#include "top_k_top_p_sample_v2.h"

using namespace AscendC;
using namespace TopKTopPSampleV2;

extern "C" __global__ __aicore__ void top_k_top_p_sample_v2(
    GM_ADDR logits, GM_ADDR topKs, GM_ADDR topPs, GM_ADDR q, GM_ADDR minPs,
    GM_ADDR logitsSelectIdx, GM_ADDR logitsTopKpSelect, GM_ADDR logitsIdx, GM_ADDR logitsSortMasked, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    GM_ADDR usrWorkspace = GetUserWorkspace(workspace);
    TPipe pipe;
    if (TILING_KEY_IS(1001)) {
        TopKTopPSampleV2Kernel<half> op;
        op.Init(logits, topKs, topPs, q, minPs, logitsSelectIdx, logitsTopKpSelect, logitsIdx, logitsSortMasked, usrWorkspace, tilingData, &pipe);
        op.Process();
        pipe.Destroy();
    } else if (TILING_KEY_IS(1027)) {
        TopKTopPSampleV2Kernel<bfloat16_t> op;
        op.Init(logits, topKs, topPs, q, minPs, logitsSelectIdx, logitsTopKpSelect, logitsIdx, logitsSortMasked, usrWorkspace, tilingData, &pipe);
        op.Process();
        pipe.Destroy();
    } 
    else if (TILING_KEY_IS(1000)) {
        TopKTopPSampleV2Kernel<float> op;
        op.Init(logits, topKs, topPs, q, minPs, logitsSelectIdx, logitsTopKpSelect, logitsIdx, logitsSortMasked, usrWorkspace, tilingData, &pipe);
        op.Process();
        pipe.Destroy();
    } else {
        return;
    }
}
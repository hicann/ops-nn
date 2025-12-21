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
 * \file trans_quant_param_v2.cpp
 * \brief
 */
#include "trans_quant_param_v2.h"

using namespace AscendC;

extern "C" __global__ __aicore__ void trans_quant_param_v2(
    GM_ADDR scale, GM_ADDR offset, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    if (workspace == nullptr) {
        return;
    }
    GM_ADDR user1 = GetUserWorkspace(workspace);
    if (user1 == nullptr) {
        return;
    }
    GET_TILING_DATA(tilingData, tiling);
    TPipe tPipe;
    TransQuantParamV2 op;
    op.Init(scale, offset, y, user1, &tilingData, &tPipe);
    op.Process();
}
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
 * \file hard_swish_grad_v2_apt.cpp
 * \brief hard_swish_grad_v2 kernel
 */
#include "hard_swish_grad_v2_base.h"
#include "hard_swish_grad_v2_220.h"

using namespace AscendC;

using namespace HardSwishGradV2;

extern "C" __global__ __aicore__ void hard_swish_grad_v2(GM_ADDR gradOutput,
                                            GM_ADDR self,
                                            GM_ADDR out,
                                            GM_ADDR workspace,
                                            GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);

    GM_ADDR userWs = nullptr;
    HardSwishGradV2220<DTYPE_SELF> op;
    op.Init(gradOutput, self, out, userWs, &tilingData);
    op.Process();
}
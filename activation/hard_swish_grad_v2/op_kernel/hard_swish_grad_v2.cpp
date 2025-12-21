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
 * \file hard_swish_grad_v2.cpp
 * \brief hard_swish_grad_v2 kernel
 */
#include "hard_swish_grad_v2_base.h"

#if __CCE_AICORE__ == 220
#include "hard_swish_grad_v2_220.h"
#elif __CCE_AICORE__ == 100
#include "hard_swish_grad_v2_100.h"
#endif

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

#if __CCE_AICORE__ == 220
        HardSwishGradV2220<DTYPE_SELF> op;
        op.Init(gradOutput, self, out, userWs, &tilingData);
        op.Process();
#elif __CCE_AICORE__ == 100
        HardSwishGradV2100<DTYPE_SELF> op;
        op.Init(gradOutput, self, out, userWs, &tilingData);
        op.Process();
#endif
}
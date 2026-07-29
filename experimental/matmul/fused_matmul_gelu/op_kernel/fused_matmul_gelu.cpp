/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fused_matmul_gelu.cpp
 * \brief AICore kernel for fused matmul + optional bias + gelu.
 */

#include "fused_matmul_gelu_kernel.h"

using namespace FusedMatmulGelu;

extern "C" __global__ __aicore__ void fused_matmul_gelu(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR y,
                                                        GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    GET_TILING_DATA(tilingData, tiling);

    AscendC::TPipe pipe;
    __gm__ uint8_t* userWorkspace = GetUserWorkspace(workspace);

#define INIT_AND_PROCESS(APPR_MODE)                                \
    FusedMatmulGeluOp<DTYPE_X, APPR_MODE> op;                      \
    op.Init(tilingData, x, weight, bias, y, userWorkspace, &pipe); \
    op.Process()

#if defined(DTYPE_X)

    if (TILING_KEY_IS(1)) {
        INIT_AND_PROCESS(1);
    }
#endif

#undef INIT_AND_PROCESS
    return;
}

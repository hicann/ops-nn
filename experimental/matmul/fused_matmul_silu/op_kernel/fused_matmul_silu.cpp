/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "fused_matmul_silu_kernel.h"

extern "C" __global__ __aicore__ void fused_matmul_silu(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR y,
                                                        GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
    REGISTER_TILING_DEFAULT(FusedMatmulSiluTilingData);
    GET_TILING_DATA_WITH_STRUCT(FusedMatmulSiluTilingData, tilingData, tiling);

    GM_ADDR userWorkspace = AscendC::GetUserWorkspace(workspace);
    AscendC::TPipe pipe;
    FusedMatmulSiluKernel::Kernel<bfloat16_t> op;
    op.Init(x, weight, bias, y, userWorkspace, &tilingData, &pipe);
    op.Process();
}

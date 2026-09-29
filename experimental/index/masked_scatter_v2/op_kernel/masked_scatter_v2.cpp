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
 * \file masked_scatter_v2.cpp
 * \brief kernel entry of masked_scatter_v2 (Ascend 950PR, AIV-only)
 *
 * [注] KERNEL_TASK_TYPE_DEFAULT(AIV_ONLY) + InitSocState 保留自 950PR 直调验证版本：
 *      前者标记 AIV-only 任务类型（950PR AIC+AIV 混合架构必需），后者初始化 SoC 状态。
 *      若仓内 CI 构建体系与该标记冲突，由评审决定去除（A2 版入口无这两行）。
 */
#include "masked_scatter_v2.h"

template <uint32_t schMode>
__global__ __aicore__ void masked_scatter_v2(GM_ADDR self, GM_ADDR mask, GM_ADDR source, GM_ADDR out, GM_ADDR workspace,
                                             GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::InitSocState();
    REGISTER_TILING_DEFAULT(MaskedScatterV2TilingData);
    GET_TILING_DATA_WITH_STRUCT(MaskedScatterV2TilingData, tilingData, tiling);
    GM_ADDR usrWorkspace = AscendC::GetUserWorkspace(workspace);
    KernelMaskedScatterV2<DTYPE_X> kernel;
    kernel.Init(self, mask, source, out, usrWorkspace, &tilingData);
    kernel.Process();
}

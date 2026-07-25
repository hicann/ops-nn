/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "nll_loss.h"

#if defined(ASCENDC_CPU_DEBUG)
#include "securec.h"
#ifndef GET_TILING_DATA_WITH_STRUCT
#define GET_TILING_DATA_WITH_STRUCT(structType, name, tilingArg) \
    structType name;                                             \
    (void)memcpy_s(&(name), sizeof(structType), (tilingArg), sizeof(structType))
#endif
#else
#include "nll_loss_tiling_key.h"
#endif

template <uint32_t schMode>
__global__ __aicore__ void nll_loss(GM_ADDR x, GM_ADDR target, GM_ADDR weight, GM_ADDR y, GM_ADDR total_weight,
                                    GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(NllLossTilingData);
    GET_TILING_DATA_WITH_STRUCT(NllLossTilingData, tilingData, tiling);

    GM_ADDR usrWorkspace = AscendC::GetUserWorkspace(workspace);

    if constexpr (schMode == NLLLOSS_TPL_SCH_MODE_0) {
        NsNllLoss::Run<half>(x, target, weight, y, total_weight, usrWorkspace, &tilingData);
    } else if constexpr (schMode == NLLLOSS_TPL_SCH_MODE_1) {
        NsNllLoss::Run<float>(x, target, weight, y, total_weight, usrWorkspace, &tilingData);
    }
}

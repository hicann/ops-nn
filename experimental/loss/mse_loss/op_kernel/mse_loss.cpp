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
 * \file mse_loss.cpp
 * \brief MseLoss 算子 kernel 入口
 */

#include "mse_loss.h"

enum class MseLossTilingKey : uint32_t {
    TILING_KEY_MSELOSS_MODE_0 = 0,
    TILING_KEY_MSELOSS_MODE_1 = 1,
    TILING_KEY_MSELOSS_MODE_2 = 2,
};

template <uint32_t schMode>
__global__ __aicore__ void mse_loss(GM_ADDR predict, GM_ADDR label, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(MseLossTilingData);
    GET_TILING_DATA_WITH_STRUCT(MseLossTilingData, tilingData, tiling);
    GM_ADDR userWorkspace = workspace;
    if (tilingData.reduction != 0 && tilingData.blockNum > 1) {
        if (workspace == nullptr) {
            return;
        }
        AscendC::SetSysWorkspace(workspace);
        userWorkspace = AscendC::GetUserWorkspace(workspace);
        if (userWorkspace == nullptr) {
            return;
        }
    }
    if constexpr (schMode == static_cast<uint32_t>(MseLossTilingKey::TILING_KEY_MSELOSS_MODE_0)) {
        NsMseLoss::MseLoss<half> op;
        op.Init(predict, label, y, userWorkspace, &tilingData);
        op.Process();
    }
    if constexpr (schMode == static_cast<uint32_t>(MseLossTilingKey::TILING_KEY_MSELOSS_MODE_1)) {
        NsMseLoss::MseLoss<float> op;
        op.Init(predict, label, y, userWorkspace, &tilingData);
        op.Process();
    }
    if constexpr (schMode == static_cast<uint32_t>(MseLossTilingKey::TILING_KEY_MSELOSS_MODE_2)) {
        NsMseLoss::MseLoss<bfloat16_t> op;
        op.Init(predict, label, y, userWorkspace, &tilingData);
        op.Process();
    }
}

/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "lp_loss.h"
#include "lp_loss_tiling_key.h"

#define DOUBLE_BUFFER_NUM 2
#define SINGLE_BUFFER_NUM 1

enum class LpLossTilingKey : uint32_t
{
    TILING_KEY_DB = LPLOSS_TPL_SCH_MODE_0,
    TILING_KEY_NDB = LPLOSS_TPL_SCH_MODE_1,
};

template <uint32_t schMode>
__global__ __aicore__ void lp_loss(GM_ADDR predict, GM_ADDR label, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(LpLossTilingData);
    GET_TILING_DATA_WITH_STRUCT(LpLossTilingData, tilingData, tiling);
    AscendC::TPipe pipe;

    if constexpr (schMode == static_cast<uint32_t>(LpLossTilingKey::TILING_KEY_DB)) {
        if (tilingData.reduction == 0) {
            MyLpLoss::KernelLpLoss<DTYPE_PREDICT, DTYPE_Y, DOUBLE_BUFFER_NUM, 0> op;
            op.InitNone(predict, label, output, &tilingData, &pipe);
            op.Process();
        } else {
            GM_ADDR usrWorkspace = AscendC::GetUserWorkspace(workspace);
            MyLpLoss::KernelLpLoss<DTYPE_PREDICT, DTYPE_Y, DOUBLE_BUFFER_NUM, 1> op;
            op.InitReduce(predict, label, output, usrWorkspace, &tilingData, &pipe);
            op.Process();
        }
    }
    if constexpr (schMode == static_cast<uint32_t>(LpLossTilingKey::TILING_KEY_NDB)) {
        if (tilingData.reduction == 0) {
            MyLpLoss::KernelLpLoss<DTYPE_PREDICT, DTYPE_Y, SINGLE_BUFFER_NUM, 0> op;
            op.InitNone(predict, label, output, &tilingData, &pipe);
            op.Process();
        } else {
            GM_ADDR usrWorkspace = AscendC::GetUserWorkspace(workspace);
            MyLpLoss::KernelLpLoss<DTYPE_PREDICT, DTYPE_Y, SINGLE_BUFFER_NUM, 1> op;
            op.InitReduce(predict, label, output, usrWorkspace, &tilingData, &pipe);
            op.Process();
        }
    }
}
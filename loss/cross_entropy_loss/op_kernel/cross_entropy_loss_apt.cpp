/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/* !
 * \file cross_entropy_loss.cpp
 * \brief
 */
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "arch35/cross_entropy_loss_tiling_key.h"
#include "arch35/cross_entropy_loss_tiling_data.h"
#include "arch35/cross_entropy_loss_ar_big_c.h"
#include "arch35/cross_entropy_loss_empty.h"
#include "arch35/cross_entropy_loss_full_load.h"

using namespace AscendC;
using namespace CrossEntropyLoss;

template <uint64_t schId, uint64_t reduction, uint64_t isWeight, uint64_t labelS, uint64_t ignorex>
__global__ __aicore__ void cross_entropy_loss(
    GM_ADDR input, GM_ADDR target, GM_ADDR weight, GM_ADDR loss, GM_ADDR log_prob, GM_ADDR zloss, GM_ADDR lse_for_zloss,
    GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    REGISTER_TILING_DEFAULT(CrossEntropyLossRegBaseTilingData);
    GET_TILING_DATA_WITH_STRUCT(CrossEntropyLossRegBaseTilingData, tilingData, tiling);

    GM_ADDR usrWorkspace = AscendC::GetUserWorkspace(workspace);
    TPipe pipe;
    if constexpr (schId == 0) {
        CrossEntropyLossBigC<DTYPE_INPUT, DTYPE_TARGET, reduction, isWeight, labelS, ignorex> op;
        op.Init(input, target, weight, loss, log_prob, zloss, lse_for_zloss, usrWorkspace, &tilingData, &pipe);
        op.Process();
        return;
    } else if constexpr (schId == 2) {
        CrossEntropyLossFullLoad<DTYPE_INPUT, DTYPE_TARGET, reduction, isWeight, labelS, ignorex> op;
        op.Init(input, target, weight, loss, log_prob, zloss, lse_for_zloss, usrWorkspace, &tilingData, &pipe);
        op.Process();
        return;
    } else {
        CrossEntropyLossEmpty<DTYPE_INPUT, reduction> op;
        op.Init(input, target, weight, loss, log_prob, zloss, lse_for_zloss, usrWorkspace, &tilingData, &pipe);
        op.Process();
        return;
    }
}

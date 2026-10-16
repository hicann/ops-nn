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
 * \file bn3_d_training_update.cpp
 * \brief BN3DTrainingUpdate arch35(Ascend 950)kernel 入口。
 */

#include "kernel_operator.h"
#include "arch35/bn3_d_training_update_channel_first.h"
#include "arch35/bn3_d_training_update_channel_last.h"
#include "arch35/bn3_d_training_update_common.h"
#include "arch35/bn3_d_training_update_stat.h"
#include "arch35/bn3_d_training_update_tiling_data.h"
#include "arch35/bn3_d_training_update_tiling_key.h"

using namespace AscendC;
using namespace Bn3dTrainingUpdateOps;

extern "C" __global__ __aicore__ void bn3_d_training_update(GM_ADDR x, GM_ADDR sum, GM_ADDR square_sum, GM_ADDR scale,
                                                            GM_ADDR offset, GM_ADDR mean, GM_ADDR variance, GM_ADDR y,
                                                            GM_ADDR mean_out, GM_ADDR variance_out, GM_ADDR batch_mean,
                                                            GM_ADDR batch_variance, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace; // 无跨核归约、无中间落盘 ⇒ 不需要 workspace
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    REGISTER_TILING_DEFAULT(Bn3dTrainingUpdateChannelFirstTilingData);
    // TILING_KEY_IS 的实参必须是宏/数字字面量
    if (TILING_KEY_IS(BN3D_TRAINING_UPDATE_KEY_CHANNEL_FIRST)) {
        REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 2000", Bn3dTrainingUpdateChannelFirstTilingData);
        GET_TILING_DATA_WITH_STRUCT(Bn3dTrainingUpdateChannelFirstTilingData, tilingData, tiling);
        Bn3dTrainingUpdateChannelFirst<DTYPE_X> op;
        op.Init(x, sum, square_sum, scale, offset, mean, variance, y, mean_out, variance_out, batch_mean,
                batch_variance, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(BN3D_TRAINING_UPDATE_KEY_CHANNEL_LAST)) {
        REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 3000", Bn3dTrainingUpdateChannelLastTilingData);
        GET_TILING_DATA_WITH_STRUCT(Bn3dTrainingUpdateChannelLastTilingData, tilingData, tiling);
        Bn3dTrainingUpdateChannelLast<DTYPE_X> op;
        op.Init(x, sum, square_sum, scale, offset, mean, variance, y, mean_out, variance_out, batch_mean,
                batch_variance, &tilingData, &pipe);
        op.Process();
    }
}

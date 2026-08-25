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
 * \file bn3d_training_reduce.cpp
 * \brief bn3d_training_reduce kernel entry (Ascend950 / arch35 only)
 */

#include "kernel_operator.h"
#include "arch35/bn3d_training_reduce_tiling_data.h"
#include "arch35/bn3d_training_reduce_common.h"
#include "arch35/bn3d_training_reduce_dense_channel.h"

using namespace AscendC;
using namespace BN3DTrainingReduceOps;

// 每通道归约成 1 个标量（storage NCDHW / NCHW）。
#define TILINGKEY_DENSE_CHANNEL 100000
// 每通道归约成 C0 个标量（storage NDC1HWC0）。搬运模型与上者同构，仅收尾方式不同。
#define TILINGKEY_NDC1HWC0_CHANNEL 200000
// 低通道、超大归约轴多核分段路线。
#define TILINGKEY_SPLIT_REDUCE 300000

extern "C" __global__ __aicore__ void bn3d_training_reduce(GM_ADDR x, GM_ADDR sum, GM_ADDR square_sum,
                                                           GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    KERNEL_TASK_TYPE(TILINGKEY_SPLIT_REDUCE, KERNEL_TYPE_MIX_AIV_1_0);
    REGISTER_TILING_DEFAULT(BN3DTrainingReduceDenseChannelTilingData);
    if (TILING_KEY_IS(TILINGKEY_DENSE_CHANNEL)) {
        GET_TILING_DATA_WITH_STRUCT(BN3DTrainingReduceDenseChannelTilingData, tilingData, tiling);
        BN3DTrainingReduceDenseChannel<DTYPE_X, false> op(&tilingData);
        op.Init(x, sum, square_sum);
        op.Process();
    } else if (TILING_KEY_IS(TILINGKEY_NDC1HWC0_CHANNEL)) {
        GET_TILING_DATA_WITH_STRUCT(BN3DTrainingReduceDenseChannelTilingData, tilingData, tiling);
        BN3DTrainingReduceDenseChannel<DTYPE_X, true> op(&tilingData);
        op.Init(x, sum, square_sum);
        op.Process();
    } else if (TILING_KEY_IS(TILINGKEY_SPLIT_REDUCE)) {
        GET_TILING_DATA_WITH_STRUCT(BN3DTrainingReduceDenseChannelTilingData, tilingData, tiling);
        BN3DTrainingReduceDenseChannel<DTYPE_X, false> op(&tilingData);
        op.InitSplitReduce(x, sum, square_sum, GetUserWorkspace(workspace));
        op.ProcessSplitReduce();
    }
}

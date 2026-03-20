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
 * \file batch_norm_apt.cpp
 * \brief
 */

#include "kernel_operator.h"
#include "arch35/batch_norm_full_reduce.h"
#include "arch35/batch_norm_ra_full_reduce.h"
#include "arch35/batch_norm_block_split_r.h"
#include "arch35/batch_norm_rar_block_split_r.h"
#include "arch35/batch_norm_ra_welford.h"
#include "arch35/batch_norm_welford.h"
#include "arch35/batch_norm_infer.h"
#include "arch35/batch_norm_infer_last_channel.h"

using namespace AscendC;
using namespace BatchNormOps;

namespace
{
#define TILINGKEY_FULL_REDUCE 200000
#define TILINGKEY_RAR_BLOCK_SPLIT_R 250000
#define TILINGKEY_RA_FULL_REDUCE 400000
#define TILINGKEY_WELFORD_REDUCE 300000
#define TILINGKEY_RA_WELFORD 500000
#define TILINGKEY_RA_BLOCK_SPLIT_R 600000
#define TILINGKEY_INFER_LAST_CHANNEL 900000
#define TILINGKEY_INFER 910000
}  // namespace

extern "C" __global__ __aicore__ void batch_norm(GM_ADDR x, GM_ADDR scale, GM_ADDR offset, GM_ADDR mean,
                                                 GM_ADDR variance, GM_ADDR y, GM_ADDR batch_mean,
                                                 GM_ADDR batch_variance, GM_ADDR reserve_space_1,
                                                 GM_ADDR reserve_space_2, GM_ADDR reserve_space_3, GM_ADDR workspace,
                                                 GM_ADDR tiling)
{
    if (g_coreType == AIC) {
        return;
    }

    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    TPipe pipe;

    if (TILING_KEY_IS(TILINGKEY_FULL_REDUCE)) {
        GET_TILING_DATA_WITH_STRUCT(BatchNormFullReduceRegbaseTilingData, tiling_data_in, tiling);
        const BatchNormFullReduceRegbaseTilingData* __restrict tilingData = &tiling_data_in;
        BatchNormFullReduce<DTYPE_X, DTYPE_SCALE> op(tilingData, &pipe);
        op.Init(x, scale, offset, mean, variance, y, batch_mean, batch_variance, reserve_space_1, reserve_space_2);
        op.Process();
    } else if (TILING_KEY_IS(TILINGKEY_RA_FULL_REDUCE)) {
        GET_TILING_DATA_WITH_STRUCT(BatchNormRAFullReduceTilingData, tiling_data_in, tiling);
        const BatchNormRAFullReduceTilingData* __restrict tilingData = &tiling_data_in;
        BatchNormRAFullReduce<DTYPE_X, DTYPE_SCALE> op(tilingData, &pipe);
        op.Init(x, scale, offset, mean, variance, y, batch_mean, batch_variance, reserve_space_1, reserve_space_2);
        op.Process();
    } else if (TILING_KEY_IS(TILINGKEY_WELFORD_REDUCE)) {
        GET_TILING_DATA_WITH_STRUCT(BatchNormWelfordRegbaseTilingData, tiling_data_in, tiling);
        const BatchNormWelfordRegbaseTilingData* __restrict tilingData = &tiling_data_in;
        BatchNormWelford<DTYPE_X, DTYPE_SCALE> op(tilingData, &pipe);
        op.Init(x, scale, offset, mean, variance, y, batch_mean, batch_variance, reserve_space_1, reserve_space_2);
        op.Process();
    } else if (TILING_KEY_IS(TILINGKEY_RA_WELFORD)) {
        GET_TILING_DATA_WITH_STRUCT(BatchNormRAWelfordTilingData, tiling_data_in, tiling);
        const BatchNormRAWelfordTilingData* __restrict tilingData = &tiling_data_in;
        BatchNormRAWelford<DTYPE_X, DTYPE_SCALE> op(tilingData, &pipe);
        op.Init(x, scale, offset, mean, variance, y, batch_mean, batch_variance, reserve_space_1, reserve_space_2);
        op.Process();
    } else if (TILING_KEY_IS(TILINGKEY_RA_BLOCK_SPLIT_R)) {
        GET_TILING_DATA_WITH_STRUCT(BatchNormBlockSplitRTilingData, tiling_data_in, tiling);
        const BatchNormBlockSplitRTilingData* __restrict tilingData = &tiling_data_in;
        BatchNormBlockSplitR<DTYPE_X, DTYPE_SCALE> op(tilingData, &pipe);
        op.Init(x, scale, offset, mean, variance, y, batch_mean, batch_variance, reserve_space_1, reserve_space_2,
                workspace);
        op.Process();
    } else if (TILING_KEY_IS(TILINGKEY_INFER_LAST_CHANNEL)) {
        GET_TILING_DATA_WITH_STRUCT(BatchNormInferLastChannelTilingData, tiling_data_in, tiling);
        const BatchNormInferLastChannelTilingData* __restrict tilingData = &tiling_data_in;
        BatchNormInferLastChannel<DTYPE_X, DTYPE_SCALE> op(tilingData);
        op.Init(x, scale, offset, mean, variance, y, batch_mean, batch_variance, reserve_space_1, reserve_space_2,
                &pipe);
        op.Process();
    } else if (TILING_KEY_IS(TILINGKEY_INFER)) {
        GET_TILING_DATA_WITH_STRUCT(BatchNormInferTilingData, tiling_data_in, tiling);
        const BatchNormInferTilingData* __restrict tilingData = &tiling_data_in;
        BatchNormInfer<DTYPE_X, DTYPE_SCALE> op(tilingData);
        op.Init(x, scale, offset, mean, variance, y, batch_mean, batch_variance, reserve_space_1, reserve_space_2,
                &pipe);
        op.Process();
    } else if (TILING_KEY_IS(TILINGKEY_RAR_BLOCK_SPLIT_R)) {
        GET_TILING_DATA_WITH_STRUCT(BatchNormRARBlockSplitRTilingData, tiling_data_in, tiling);
        const BatchNormRARBlockSplitRTilingData* __restrict tilingData = &tiling_data_in;
        BatchNormRARBlockSplitR<DTYPE_X, DTYPE_SCALE> op(tilingData, &pipe);
        op.Init(x, scale, offset, mean, variance, y, batch_mean, batch_variance, reserve_space_1, reserve_space_2,
                workspace);
        op.Process();
    }
}
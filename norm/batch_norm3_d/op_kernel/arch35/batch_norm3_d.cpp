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
 * \file batch_norm3d.cpp
 * \brief BatchNorm3D kernel entry for arch35 / Ascend950.
 */

#include "kernel_operator.h"
#include "batch_norm3d_full_reduce.h"
#include "batch_norm3d_ra_full_reduce.h"
#include "batch_norm3d_block_split_r.h"
#include "batch_norm3d_rar_block_split_r.h"
#include "batch_norm3d_ra_welford.h"
#include "batch_norm3d_welford.h"
#include "batch_norm3d_infer.h"
#include "batch_norm3d_infer_last_channel.h"
#include "batch_norm3d_infer_last_channel_continuous_a.h"
#include "batch_norm3d_infer_last_channel_small_a.h"
#include "batch_norm3d_infer_small_ab1.h"

using namespace AscendC;
using namespace BatchNorm3DOps;

namespace {
#define TILINGKEY_FULL_REDUCE 200000
#define TILINGKEY_RAR_BLOCK_SPLIT_R 250000
#define TILINGKEY_RA_FULL_REDUCE 400000
#define TILINGKEY_WELFORD_REDUCE 300000
#define TILINGKEY_RA_WELFORD 500000
#define TILINGKEY_RA_BLOCK_SPLIT_R 600000
#define TILINGKEY_INFER_LAST_CHANNEL 900000
#define TILINGKEY_INFER_LAST_CHANNEL_SMALL_A 902000
#define TILINGKEY_INFER_LAST_CHANNEL_CONTINUOUS_A 901000
#define TILINGKEY_INFER 910000
#define TILINGKEY_INFER_SMALL_AB1 911000
} // namespace

extern "C" __global__ __aicore__ void batch_norm3_d(GM_ADDR x, GM_ADDR scale, GM_ADDR offset, GM_ADDR mean,
                                                    GM_ADDR variance, GM_ADDR y, GM_ADDR batch_mean,
                                                    GM_ADDR batch_variance, GM_ADDR reserve_space_1,
                                                    GM_ADDR reserve_space_2, GM_ADDR workspace, GM_ADDR tiling)
{
    if (g_coreType == AIC) {
        return;
    }

    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    REGISTER_TILING_DEFAULT(BatchNorm3DFullReduceRegbaseTilingData);
    TPipe pipe;

    if (TILING_KEY_IS(TILINGKEY_FULL_REDUCE)) {
        REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 200000", BatchNorm3DFullReduceRegbaseTilingData);
        GET_TILING_DATA_WITH_STRUCT(BatchNorm3DFullReduceRegbaseTilingData, tiling_data_in, tiling);
        const BatchNorm3DFullReduceRegbaseTilingData* __restrict tilingData = &tiling_data_in;
        BatchNorm3DFullReduce<DTYPE_X, DTYPE_SCALE> op(tilingData, &pipe);
        op.Init(x, scale, offset, mean, variance, y, batch_mean, batch_variance, reserve_space_1, reserve_space_2);
        op.Process();
    } else if (TILING_KEY_IS(TILINGKEY_RA_FULL_REDUCE)) {
        REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 400000", BatchNorm3DRAFullReduceTilingData);
        GET_TILING_DATA_WITH_STRUCT(BatchNorm3DRAFullReduceTilingData, tiling_data_in, tiling);
        const BatchNorm3DRAFullReduceTilingData* __restrict tilingData = &tiling_data_in;
        BatchNorm3DRAFullReduce<DTYPE_X, DTYPE_SCALE> op(tilingData, &pipe);
        op.Init(x, scale, offset, mean, variance, y, batch_mean, batch_variance, reserve_space_1, reserve_space_2);
        op.Process();
    } else if (TILING_KEY_IS(TILINGKEY_WELFORD_REDUCE)) {
        REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 300000", BatchNorm3DWelfordRegbaseTilingData);
        GET_TILING_DATA_WITH_STRUCT(BatchNorm3DWelfordRegbaseTilingData, tiling_data_in, tiling);
        const BatchNorm3DWelfordRegbaseTilingData* __restrict tilingData = &tiling_data_in;
        BatchNorm3DWelford<DTYPE_X, DTYPE_SCALE> op(tilingData, &pipe);
        op.Init(x, scale, offset, mean, variance, y, batch_mean, batch_variance, reserve_space_1, reserve_space_2);
        op.Process();
    } else if (TILING_KEY_IS(TILINGKEY_RA_WELFORD)) {
        REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 500000", BatchNorm3DRAWelfordTilingData);
        GET_TILING_DATA_WITH_STRUCT(BatchNorm3DRAWelfordTilingData, tiling_data_in, tiling);
        const BatchNorm3DRAWelfordTilingData* __restrict tilingData = &tiling_data_in;
        BatchNorm3DRAWelford<DTYPE_X, DTYPE_SCALE> op(tilingData, &pipe);
        op.Init(x, scale, offset, mean, variance, y, batch_mean, batch_variance, reserve_space_1, reserve_space_2);
        op.Process();
    } else if (TILING_KEY_IS(TILINGKEY_RA_BLOCK_SPLIT_R)) {
        REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 600000", BatchNorm3DBlockSplitRTilingData);
        GET_TILING_DATA_WITH_STRUCT(BatchNorm3DBlockSplitRTilingData, tiling_data_in, tiling);
        const BatchNorm3DBlockSplitRTilingData* __restrict tilingData = &tiling_data_in;
        BatchNorm3DBlockSplitR<DTYPE_X, DTYPE_SCALE> op(tilingData, &pipe);
        op.Init(x, scale, offset, mean, variance, y, batch_mean, batch_variance, reserve_space_1, reserve_space_2,
                workspace);
        op.Process();
    } else if (TILING_KEY_IS(TILINGKEY_INFER_LAST_CHANNEL)) {
        REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 900000", BatchNorm3DInferLastChannelTilingData);
        GET_TILING_DATA_WITH_STRUCT(BatchNorm3DInferLastChannelTilingData, tiling_data_in, tiling);
        const BatchNorm3DInferLastChannelTilingData* __restrict tilingData = &tiling_data_in;
        BatchNorm3DInferLastChannel<DTYPE_X, DTYPE_SCALE> op(tilingData);
        op.Init(x, scale, offset, mean, variance, y, batch_mean, batch_variance, reserve_space_1, reserve_space_2,
                &pipe);
        op.Process();
    } else if (TILING_KEY_IS(TILINGKEY_INFER)) {
        REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 910000", BatchNorm3DInferTilingData);
        GET_TILING_DATA_WITH_STRUCT(BatchNorm3DInferTilingData, tiling_data_in, tiling);
        const BatchNorm3DInferTilingData* __restrict tilingData = &tiling_data_in;
        BatchNorm3DInfer<DTYPE_X, DTYPE_SCALE> op(tilingData);
        op.Init(x, scale, offset, mean, variance, y, batch_mean, batch_variance, reserve_space_1, reserve_space_2,
                &pipe);
        op.Process();
    } else if (TILING_KEY_IS(TILINGKEY_INFER_LAST_CHANNEL_SMALL_A)) {
        REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 902000", BatchNorm3DInferLastChannelTilingData);
        GET_TILING_DATA_WITH_STRUCT(BatchNorm3DInferLastChannelTilingData, tiling_data_in, tiling);
        const BatchNorm3DInferLastChannelTilingData* __restrict tilingData = &tiling_data_in;
        BatchNorm3DInferLastChannelSmallA<DTYPE_X, DTYPE_SCALE, DTYPE_SCALE> op(tilingData);
        op.Init(x, scale, offset, mean, variance, y, batch_mean, batch_variance, reserve_space_1, reserve_space_2,
                &pipe);
        op.Process();
    } else if (TILING_KEY_IS(TILINGKEY_INFER_LAST_CHANNEL_CONTINUOUS_A)) {
        REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 901000", BatchNorm3DInferLastChannelTilingData);
        GET_TILING_DATA_WITH_STRUCT(BatchNorm3DInferLastChannelTilingData, tiling_data_in, tiling);
        const BatchNorm3DInferLastChannelTilingData* __restrict tilingData = &tiling_data_in;
        BatchNorm3DInferLastChannelContinuousA<DTYPE_X, DTYPE_SCALE, DTYPE_SCALE> op(tilingData);
        op.Init(x, scale, offset, mean, variance, y, batch_mean, batch_variance, reserve_space_1, reserve_space_2,
                &pipe);
        op.Process();
    } else if (TILING_KEY_IS(TILINGKEY_INFER_SMALL_AB1)) {
        REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 911000", BatchNorm3DInferTilingData);
        GET_TILING_DATA_WITH_STRUCT(BatchNorm3DInferTilingData, tiling_data_in, tiling);
        const BatchNorm3DInferTilingData* __restrict tilingData = &tiling_data_in;
        BatchNorm3DInferSmallAB1<DTYPE_X, DTYPE_SCALE, DTYPE_SCALE> op(tilingData);
        op.Init(x, scale, offset, mean, variance, y, batch_mean, batch_variance, reserve_space_1, reserve_space_2,
                &pipe);
        op.Process();
    } else if (TILING_KEY_IS(TILINGKEY_RAR_BLOCK_SPLIT_R)) {
        REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 250000", BatchNorm3DRARBlockSplitRTilingData);
        GET_TILING_DATA_WITH_STRUCT(BatchNorm3DRARBlockSplitRTilingData, tiling_data_in, tiling);
        const BatchNorm3DRARBlockSplitRTilingData* __restrict tilingData = &tiling_data_in;
        BatchNorm3DRARBlockSplitR<DTYPE_X, DTYPE_SCALE> op(tilingData, &pipe);
        op.Init(x, scale, offset, mean, variance, y, batch_mean, batch_variance, reserve_space_1, reserve_space_2,
                workspace);
        op.Process();
    }
}

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
 * \file batch_norm_ext2.cpp
 * \brief
 */

#include "kernel_operator.h"
#include "arch35/batch_norm_ext2_full_reduce.h"
#include "arch35/batch_norm_ext2_ra_full_reduce.h"
#include "arch35/batch_norm_ext2_block_split_r.h"
#include "arch35/batch_norm_ext2_rar_block_split_r.h"
#include "arch35/batch_norm_ext2_ra_welford.h"
#include "arch35/batch_norm_ext2_welford.h"
#include "arch35/batch_norm_ext2_infer.h"
#include "arch35/batch_norm_ext2_infer_last_channel.h"
#include "arch35/batch_norm_ext2_infer_last_channel_continuous_a.h"
#include "arch35/batch_norm_ext2_infer_last_channel_small_a.h"
#include "arch35/batch_norm_ext2_infer_small_ab1.h"

using namespace AscendC;
using namespace BatchNormExt2Ops;

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

extern "C" __global__ __aicore__ void batch_norm_ext2(GM_ADDR x, GM_ADDR scale, GM_ADDR offset, GM_ADDR mean,
                                                      GM_ADDR variance, GM_ADDR y, GM_ADDR batch_mean,
                                                      GM_ADDR batch_variance, GM_ADDR reserve_space_1,
                                                      GM_ADDR reserve_space_2, GM_ADDR workspace, GM_ADDR tiling)
{
    if (g_coreType == AIC) {
        return;
    }

    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    TPipe pipe;

    if (TILING_KEY_IS(TILINGKEY_FULL_REDUCE)) {
        GET_TILING_DATA_WITH_STRUCT(BatchNormExt2FullReduceRegbaseTilingData, tiling_data_in, tiling);
        const BatchNormExt2FullReduceRegbaseTilingData* __restrict tilingData = &tiling_data_in;
        BatchNormExt2FullReduce<DTYPE_INPUT_X, DTYPE_INPUT_SCALE> op(tilingData, &pipe);
        op.Init(x, scale, offset, mean, variance, y, batch_mean, batch_variance, reserve_space_1, reserve_space_2);
        op.Process();
    } else if (TILING_KEY_IS(TILINGKEY_RA_FULL_REDUCE)) {
        GET_TILING_DATA_WITH_STRUCT(BatchNormExt2RAFullReduceTilingData, tiling_data_in, tiling);
        const BatchNormExt2RAFullReduceTilingData* __restrict tilingData = &tiling_data_in;
        BatchNormExt2RAFullReduce<DTYPE_INPUT_X, DTYPE_INPUT_SCALE> op(tilingData, &pipe);
        op.Init(x, scale, offset, mean, variance, y, batch_mean, batch_variance, reserve_space_1, reserve_space_2);
        op.Process();
    } else if (TILING_KEY_IS(TILINGKEY_WELFORD_REDUCE)) {
        GET_TILING_DATA_WITH_STRUCT(BatchNormExt2WelfordRegbaseTilingData, tiling_data_in, tiling);
        const BatchNormExt2WelfordRegbaseTilingData* __restrict tilingData = &tiling_data_in;
        BatchNormExt2Welford<DTYPE_INPUT_X, DTYPE_INPUT_SCALE> op(tilingData, &pipe);
        op.Init(x, scale, offset, mean, variance, y, batch_mean, batch_variance, reserve_space_1, reserve_space_2);
        op.Process();
    } else if (TILING_KEY_IS(TILINGKEY_RA_WELFORD)) {
        GET_TILING_DATA_WITH_STRUCT(BatchNormExt2RAWelfordTilingData, tiling_data_in, tiling);
        const BatchNormExt2RAWelfordTilingData* __restrict tilingData = &tiling_data_in;
        BatchNormExt2RAWelford<DTYPE_INPUT_X, DTYPE_INPUT_SCALE> op(tilingData, &pipe);
        op.Init(x, scale, offset, mean, variance, y, batch_mean, batch_variance, reserve_space_1, reserve_space_2);
        op.Process();
    } else if (TILING_KEY_IS(TILINGKEY_RA_BLOCK_SPLIT_R)) {
        GET_TILING_DATA_WITH_STRUCT(BatchNormExt2BlockSplitRTilingData, tiling_data_in, tiling);
        const BatchNormExt2BlockSplitRTilingData* __restrict tilingData = &tiling_data_in;
        BatchNormExt2BlockSplitR<DTYPE_INPUT_X, DTYPE_INPUT_SCALE> op(tilingData, &pipe);
        op.Init(x, scale, offset, mean, variance, y, batch_mean, batch_variance, reserve_space_1, reserve_space_2,
                workspace);
        op.Process();
    } else if (TILING_KEY_IS(TILINGKEY_INFER_LAST_CHANNEL)) {
        GET_TILING_DATA_WITH_STRUCT(BatchNormExt2InferLastChannelTilingData, tiling_data_in, tiling);
        const BatchNormExt2InferLastChannelTilingData* __restrict tilingData = &tiling_data_in;
        BatchNormExt2InferLastChannel<DTYPE_INPUT_X, DTYPE_INPUT_SCALE> op(tilingData);
        op.Init(x, scale, offset, mean, variance, y, batch_mean, batch_variance, reserve_space_1, reserve_space_2,
                &pipe);
        op.Process();
    } else if (TILING_KEY_IS(TILINGKEY_INFER)) {
        GET_TILING_DATA_WITH_STRUCT(BatchNormExt2InferTilingData, tiling_data_in, tiling);
        const BatchNormExt2InferTilingData* __restrict tilingData = &tiling_data_in;
        BatchNormExt2Infer<DTYPE_INPUT_X, DTYPE_INPUT_SCALE> op(tilingData);
        op.Init(x, scale, offset, mean, variance, y, batch_mean, batch_variance, reserve_space_1, reserve_space_2,
                &pipe);
        op.Process();
    } else if (TILING_KEY_IS(TILINGKEY_INFER_LAST_CHANNEL_SMALL_A)) {
        GET_TILING_DATA_WITH_STRUCT(BatchNormExt2InferLastChannelTilingData, tiling_data_in, tiling);
        const BatchNormExt2InferLastChannelTilingData* __restrict tilingData = &tiling_data_in;
        BatchNormExt2InferLastChannelSmallA<DTYPE_INPUT_X, DTYPE_INPUT_SCALE, DTYPE_INPUT_SCALE> op(tilingData);
        op.Init(x, scale, offset, mean, variance, y, batch_mean, batch_variance, reserve_space_1, reserve_space_2,
                &pipe);
        op.Process();
    } else if (TILING_KEY_IS(TILINGKEY_INFER_LAST_CHANNEL_CONTINUOUS_A)) {
        GET_TILING_DATA_WITH_STRUCT(BatchNormExt2InferLastChannelTilingData, tiling_data_in, tiling);
        const BatchNormExt2InferLastChannelTilingData* __restrict tilingData = &tiling_data_in;
        BatchNormExt2InferLastChannelContinuousA<DTYPE_INPUT_X, DTYPE_INPUT_SCALE, DTYPE_INPUT_SCALE> op(tilingData);
        op.Init(x, scale, offset, mean, variance, y, batch_mean, batch_variance, reserve_space_1, reserve_space_2,
                &pipe);
        op.Process();
    } else if (TILING_KEY_IS(TILINGKEY_INFER_SMALL_AB1)) {
        GET_TILING_DATA_WITH_STRUCT(BatchNormExt2InferTilingData, tiling_data_in, tiling);
        const BatchNormExt2InferTilingData* __restrict tilingData = &tiling_data_in;
        BatchNormExt2InferSmallAB1<DTYPE_INPUT_X, DTYPE_INPUT_SCALE, DTYPE_INPUT_SCALE> op(tilingData);
        op.Init(x, scale, offset, mean, variance, y, batch_mean, batch_variance, reserve_space_1, reserve_space_2,
                &pipe);
        op.Process();
    } else if (TILING_KEY_IS(TILINGKEY_RAR_BLOCK_SPLIT_R)) {
        GET_TILING_DATA_WITH_STRUCT(BatchNormExt2RARBlockSplitRTilingData, tiling_data_in, tiling);
        const BatchNormExt2RARBlockSplitRTilingData* __restrict tilingData = &tiling_data_in;
        BatchNormExt2RARBlockSplitR<DTYPE_INPUT_X, DTYPE_INPUT_SCALE> op(tilingData, &pipe);
        op.Init(x, scale, offset, mean, variance, y, batch_mean, batch_variance, reserve_space_1, reserve_space_2,
                workspace);
        op.Process();
    }
}

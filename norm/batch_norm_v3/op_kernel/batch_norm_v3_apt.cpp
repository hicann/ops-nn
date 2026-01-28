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
 * \file batch_norm_v3_apt.cpp
 * \brief
 */

#include "kernel_operator.h"
#include "arch35/batch_norm_v3_ra_full_reduce.h"
#include "arch35/batch_norm_v3_block_split_r.h"
#include "arch35/batch_norm_v3_ra_welford.h"
#include "arch35/batch_norm_v3.h"
#include "arch35/batch_norm_v3_welford.h"
#include "arch35/batch_norm_v3_infer.h"
#include "arch35/batch_norm_v3_infer_last_channel.h"

using namespace AscendC;
using namespace BatchNormV3Ops;

namespace {
#define TILINGKEY_FULL_REDUCE 200000
#define TILINGKEY_RA_FULL_REDUCE 400000
#define TILINGKEY_WELFORD_REDUCE 300000
#define TILINGKEY_RA_WELFORD 500000
#define TILINGKEY_RA_BLOCK_SPLIT_R 600000
#define TILINGKEY_INFER_LAST_CHANNEL 900000
#define TILINGKEY_INFER 910000
} // namespace

extern "C" __global__ __aicore__ void batch_norm_v3(
    GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR mean, GM_ADDR variance, GM_ADDR y, GM_ADDR mean_out,
    GM_ADDR variance_out, GM_ADDR batch_mean, GM_ADDR batch_rstd, GM_ADDR workspace, GM_ADDR tiling)
{
    if (g_coreType == AIC) {
        return;
    }

    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    if (TILING_KEY_IS(TILINGKEY_FULL_REDUCE)) {
        GET_TILING_DATA_WITH_STRUCT(BatchNormV3FullReduceRegbaseTilingData, tiling_data_in, tiling);
        const BatchNormV3FullReduceRegbaseTilingData* __restrict tilingData = &tiling_data_in;
        BatchNormV3FullReduce<DTYPE_X, DTYPE_WEIGHT, DTYPE_RUNNING_MEAN> op(tilingData);
        op.Init(x, weight, bias, mean, variance, y, mean_out, variance_out, batch_mean, batch_rstd);
        op.Process();
    } else if (TILING_KEY_IS(TILINGKEY_WELFORD_REDUCE)) {
        GET_TILING_DATA_WITH_STRUCT(BatchNormV3WelfordRegbaseTilingData, tiling_data_in, tiling);
        const BatchNormV3WelfordRegbaseTilingData* __restrict tilingData = &tiling_data_in;
        BatchNormV3Welford<DTYPE_X, DTYPE_WEIGHT, DTYPE_RUNNING_MEAN> op(tilingData);
        op.Init(x, weight, bias, mean, variance, y, mean_out, variance_out, batch_mean, batch_rstd);
        op.Process();
    } else if (TILING_KEY_IS(TILINGKEY_RA_FULL_REDUCE)) {
        GET_TILING_DATA_WITH_STRUCT(BatchNormV3RAFullReduceTilingData, tiling_data_in, tiling);
        const BatchNormV3RAFullReduceTilingData* __restrict tilingData = &tiling_data_in;
        BatchNormV3RAFullReduce<DTYPE_X, DTYPE_WEIGHT, DTYPE_RUNNING_MEAN> op(tilingData);
        op.Init(x, weight, bias, mean, variance, y, mean_out, variance_out, batch_mean, batch_rstd);
        op.Process();
    } else if (TILING_KEY_IS(TILINGKEY_RA_WELFORD)) {
        GET_TILING_DATA_WITH_STRUCT(BatchNormV3RAWelfordTilingData, tiling_data_in, tiling);
        const BatchNormV3RAWelfordTilingData* __restrict tilingData = &tiling_data_in;
        BatchNormV3RAWelford<DTYPE_X, DTYPE_WEIGHT, DTYPE_RUNNING_MEAN> op(tilingData);
        op.Init(x, weight, bias, mean, variance, y, mean_out, variance_out, batch_mean, batch_rstd);
        op.Process();
    } else if (TILING_KEY_IS(TILINGKEY_INFER_LAST_CHANNEL)) {
        GET_TILING_DATA_WITH_STRUCT(BatchNormV3InferLastChannelTilingData, tiling_data_in, tiling);
        const BatchNormV3InferLastChannelTilingData* __restrict tilingData = &tiling_data_in;
        TPipe pipe;
        BatchNormV3InferLastChannel<DTYPE_X, DTYPE_WEIGHT, DTYPE_RUNNING_MEAN> op(tilingData);
        op.Init(x, weight, bias, mean, variance, y, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(TILINGKEY_INFER)) {
        GET_TILING_DATA_WITH_STRUCT(BatchNormV3InferTilingData, tiling_data_in, tiling);
        const BatchNormV3InferTilingData* __restrict tilingData = &tiling_data_in;
        TPipe pipe;
        BatchNormV3Infer<DTYPE_X, DTYPE_WEIGHT, DTYPE_RUNNING_MEAN> op(tilingData);
        op.Init(x, weight, bias, mean, variance, y, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(TILINGKEY_RA_BLOCK_SPLIT_R)) {
        GET_TILING_DATA_WITH_STRUCT(BatchNormV3BlockSplitRTilingData, tiling_data_in, tiling);
        const BatchNormV3BlockSplitRTilingData* __restrict tilingData = &tiling_data_in;
        BatchNormV3BlockSplitR<DTYPE_X, DTYPE_WEIGHT, DTYPE_RUNNING_MEAN> op(tilingData);
        op.Init(x, weight, bias, mean, variance, y, mean_out, variance_out, batch_mean, batch_rstd, workspace);
        op.Process();
    }
}
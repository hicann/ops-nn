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
 * \file sync_batch_norm_gather_stats_apt.cpp
 * \brief
 */

#include "kernel_operator.h"
#include "arch35/sync_batch_norm_gather_stats_n_full_load.h"
#include "arch35/sync_batch_norm_gather_stats_n_not_full_load.h"

using namespace SyncBatchNormGatherStats;

#define SYNC_BATCH_NORM_GATHER_STATS_N_FULL_LOAD 10001
#define SYNC_BATCH_NORM_GATHER_STATS_N_NOT_FULL_LOAD 20001

extern "C" __global__ __aicore__ void sync_batch_norm_gather_stats(GM_ADDR total_sum, GM_ADDR total_square_sum, GM_ADDR sample_count, 
    GM_ADDR running_mean, GM_ADDR running_var, GM_ADDR batch_mean, GM_ADDR batch_invstd, GM_ADDR running_mean_update, 
    GM_ADDR running_var_update, GM_ADDR workspace, GM_ADDR tiling)
{
    TPipe pipe;

    if (TILING_KEY_IS(SYNC_BATCH_NORM_GATHER_STATS_N_FULL_LOAD)) {
        GET_TILING_DATA_WITH_STRUCT(SyncBatchNormGatherStatsTilingData, tiling_data_in, tiling);
        const SyncBatchNormGatherStatsTilingData* __restrict tilingData = &tiling_data_in;
        SyncBatchNormGatherStatsNFullLoad<DTYPE_TOTAL_SUM> op(tilingData, &pipe);
        op.Init(total_sum, total_square_sum, sample_count, running_mean, running_var, batch_mean, batch_invstd, running_mean_update, running_var_update);
        op.Process();
    } else if (TILING_KEY_IS(SYNC_BATCH_NORM_GATHER_STATS_N_NOT_FULL_LOAD)) {
        GET_TILING_DATA_WITH_STRUCT(SyncBatchNormGatherStatsNNotFullLoadTilingData, tiling_data_in, tiling);
        const SyncBatchNormGatherStatsNNotFullLoadTilingData* __restrict tilingData = &tiling_data_in;
        SyncBatchNormGatherStatsNNotFullLoad<DTYPE_TOTAL_SUM> op(tilingData, &pipe);
        op.Init(total_sum, total_square_sum, sample_count, running_mean, running_var, batch_mean, batch_invstd, running_mean_update, running_var_update);
        op.Process();
    }
}
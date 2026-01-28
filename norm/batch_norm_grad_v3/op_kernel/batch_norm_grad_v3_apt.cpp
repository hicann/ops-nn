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
 * \file batch_norm_grad_v3_apt.cpp
 * \brief
 */
#include "kernel_operator.h"
#include "arch35/batch_norm_grad_v3_full_load_regbase.h"
#include "arch35/batch_norm_grad_v3_recompute_split_r0_regbase.h"
#include "arch35/batch_norm_grad_v3_split_r1_regbase.h"
#include "arch35/batch_norm_grad_v3_ra_full_load_regbase.h"
#include "arch35/batch_norm_grad_v3_ra_recompute_regbase.h"
#include "arch35/batch_norm_grad_v3_ra_split_r_regbase.h"
#include "arch35/batch_norm_grad_v3_infer_channel_last.h"
#include "arch35/batch_norm_grad_v3_infer.h"
#include "arch35/batch_norm_grad_v3_rar_split_core_r1.h"

using namespace BatchNormGradV3;
using namespace BNGV3RARRecomputeSplitR0;

#define BATCH_NORM_GRAD_V3_RAR_FULL_LOAD 10000000UL
#define BATCH_NORM_GRAD_V3_RAR_SPLIT_R1 31000000UL
#define BATCH_NORM_GRAD_V3_RAR_RECOMPUTE_SPLIT_R0 32000000UL
#define BATCH_NORM_GRAD_V3_RA_FULL_LOAD 20000000UL
#define BATCH_NORM_GRAD_V3_RA_RECOMPUTE 40000000UL
#define BATCH_NORM_GRAD_V3_RA_SPLIT_R_TILING_KEY 50000000UL
#define BATCH_NORM_GRAD_V3_RAR_SPLIT_CORE_R1 1000UL
static constexpr int DOUBLE_BUFFER = 2;

// inference
#define BATCH_NORM_GRAD_V3_INFER_CHANNEL_LAST 900000UL
#define BATCH_NORM_GRAD_V3_INFER 910000UL

extern "C" __global__ __aicore__ void batch_norm_grad_v3(
    GM_ADDR dy, GM_ADDR x, GM_ADDR weight, GM_ADDR running_mean, GM_ADDR running_var, GM_ADDR save_mean,
    GM_ADDR save_rstd, GM_ADDR dx, GM_ADDR dweight, GM_ADDR dbias, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    TPipe pipe;
    GM_ADDR usrWorkspace = AscendC::GetUserWorkspace(workspace);
    if (TILING_KEY_IS(BATCH_NORM_GRAD_V3_RAR_FULL_LOAD)) {
        GET_TILING_DATA_WITH_STRUCT(BatchNormGradV3RARFullLoadTilingData, tilingDataIn, tiling);
        const BatchNormGradV3RARFullLoadTilingData* __restrict tilingData = &tilingDataIn;
        BatchNormGradV3RARFullLoad<DTYPE_DY, DTYPE_WEIGHT, DOUBLE_BUFFER> op(&pipe);
        op.Init(dy, x, save_mean, save_rstd, weight, dx, dweight, dbias, usrWorkspace, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(BATCH_NORM_GRAD_V3_RAR_RECOMPUTE_SPLIT_R0)) {
        GET_TILING_DATA_WITH_STRUCT(BatchNormGradV3RARRecomputeTilingData, tilingDataIn, tiling);
        const BatchNormGradV3RARRecomputeTilingData* __restrict tilingData = &tilingDataIn;
        BatchNormGradV3RARRecomputeSplitR0<DTYPE_DY, DTYPE_WEIGHT, DOUBLE_BUFFER> op;
        op.Init(dy, x, save_mean, save_rstd, weight, dx, dweight, dbias, usrWorkspace, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(BATCH_NORM_GRAD_V3_RAR_SPLIT_R1)) {
        GET_TILING_DATA_WITH_STRUCT(BatchNormGradV3RARRecomputeTilingData, tilingDataIn, tiling);
        const BatchNormGradV3RARRecomputeTilingData* __restrict tilingData = &tilingDataIn;
        BatchNormGradV3RARSplitR1<DTYPE_DY, DTYPE_WEIGHT, DOUBLE_BUFFER> op(&pipe);
        op.Init(dy, x, save_mean, save_rstd, weight, dx, dweight, dbias, usrWorkspace, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(BATCH_NORM_GRAD_V3_RA_FULL_LOAD)) {
        GET_TILING_DATA_WITH_STRUCT(BatchNormGradV3RAFullLoadTilingData, tilingDataIn, tiling);
        const BatchNormGradV3RAFullLoadTilingData* __restrict tilingData = &tilingDataIn;
        BatchNormGradV3RAFullLoad<DTYPE_DY, DTYPE_WEIGHT, DOUBLE_BUFFER> op(&pipe);
        op.Init(dy, x, save_mean, save_rstd, weight, dx, dweight, dbias, usrWorkspace, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(BATCH_NORM_GRAD_V3_RA_RECOMPUTE)) {
        GET_TILING_DATA_WITH_STRUCT(BatchNormGradV3RARecomputeTilingData, tilingDataIn, tiling);
        const BatchNormGradV3RARecomputeTilingData* __restrict tilingData = &tilingDataIn;
        BatchNormGradV3RARecompute<DTYPE_DY, DTYPE_WEIGHT, DOUBLE_BUFFER> op(&pipe);
        op.Init(dy, x, save_mean, save_rstd, weight, dx, dweight, dbias, usrWorkspace, tilingData);
        op.Process();
    } else if (
        TILING_KEY_IS(BATCH_NORM_GRAD_V3_INFER_CHANNEL_LAST)) {
        GET_TILING_DATA_WITH_STRUCT(BatchNormGradV3InferChannelLastTilingData, tiling_data_in, tiling);
        const BatchNormGradV3InferChannelLastTilingData* __restrict tilingData = &tiling_data_in;
        BatchNormGradV3InferChannelLast<DTYPE_DY, DTYPE_WEIGHT, DTYPE_RUNNING_VAR> op(tilingData);
        op.Init(dy, weight, running_var, dx, &pipe);
        op.Process();
    } else if (
        TILING_KEY_IS(BATCH_NORM_GRAD_V3_INFER)) {
        GET_TILING_DATA_WITH_STRUCT(BatchNormGradV3InferTilingData, tiling_data_in, tiling);
        const BatchNormGradV3InferTilingData* __restrict tilingData = &tiling_data_in;
        BatchNormGradV3Infer<DTYPE_DY, DTYPE_WEIGHT, DTYPE_RUNNING_VAR> op(tilingData);
        op.Init(dy, weight, running_var, dx, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(BATCH_NORM_GRAD_V3_RAR_SPLIT_CORE_R1)) {
        GET_TILING_DATA_WITH_STRUCT(BatchNormGradV3RARSplitCoreR1TilingData, tiling_data_in, tiling);
        const BatchNormGradV3RARSplitCoreR1TilingData* __restrict tilingData = &tiling_data_in;
        BatchNormGradV3RARSplitCoreR1<DTYPE_DY, DTYPE_WEIGHT> op(&pipe, tilingData);
        op.Init(dy, x, save_mean, save_rstd, weight, dx, dweight, dbias, usrWorkspace);
        op.Process();
    } else if (TILING_KEY_IS(BATCH_NORM_GRAD_V3_RA_SPLIT_R_TILING_KEY)) {
        GET_TILING_DATA_WITH_STRUCT(BatchNormGradV3RASplitRTilingData, tilingDataIn, tiling);
        const BatchNormGradV3RASplitRTilingData* __restrict tilingData = &tilingDataIn;
        BatchNormGradV3RASplitR<DTYPE_DY, DTYPE_WEIGHT> op(&pipe);
        op.Init(dy, x, save_mean, save_rstd, weight, dx, dweight, dbias, usrWorkspace, tilingData);
        op.Process();
    }
}
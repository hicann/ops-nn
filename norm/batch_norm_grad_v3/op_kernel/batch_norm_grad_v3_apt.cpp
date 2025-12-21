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

#define BATCH_NORM_GRAD_V3_RAR_FULL_LOAD_FLOAT 10000001UL
#define BATCH_NORM_GRAD_V3_RAR_FULL_LOAD_FLOAT16 10000002UL
#define BATCH_NORM_GRAD_V3_RAR_FULL_LOAD_BFLOAT16 10000003UL
#define BATCH_NORM_GRAD_V3_RAR_FULL_LOAD_MIXED_FP16_FP32 10000004UL
#define BATCH_NORM_GRAD_V3_RAR_FULL_LOAD_MIXED_BF16_FP32 10000005UL
#define BATCH_NORM_GRAD_V3_RAR_SPLIT_R1_FLOAT 31000001UL
#define BATCH_NORM_GRAD_V3_RAR_SPLIT_R1_FLOAT16 31000002UL
#define BATCH_NORM_GRAD_V3_RAR_SPLIT_R1_BFLOAT16 31000003UL
#define BATCH_NORM_GRAD_V3_RAR_SPLIT_R1_MIXED_FP16_FP32 31000004UL
#define BATCH_NORM_GRAD_V3_RAR_SPLIT_R1_MIXED_BF16_FP32 31000005UL
#define BATCH_NORM_GRAD_V3_RAR_RECOMPUTE_SPLIT_R0_FLOAT 32000001UL
#define BATCH_NORM_GRAD_V3_RAR_RECOMPUTE_SPLIT_R0_FLOAT16 32000002UL
#define BATCH_NORM_GRAD_V3_RAR_RECOMPUTE_SPLIT_R0_BFLOAT16 32000003UL
#define BATCH_NORM_GRAD_V3_RAR_RECOMPUTE_SPLIT_R0_MIXED_FP16_FP32 32000004UL
#define BATCH_NORM_GRAD_V3_RAR_RECOMPUTE_SPLIT_R0_MIXED_BF16_FP32 32000005UL
#define BATCH_NORM_GRAD_V3_RA_FULL_LOAD_FLOAT 20000001UL
#define BATCH_NORM_GRAD_V3_RA_FULL_LOAD_FLOAT16 20000002UL
#define BATCH_NORM_GRAD_V3_RA_FULL_LOAD_BFLOAT16 20000003UL
#define BATCH_NORM_GRAD_V3_RA_FULL_LOAD_MIXED_FP16_FP32 20000004UL
#define BATCH_NORM_GRAD_V3_RA_FULL_LOAD_MIXED_BF16_FP32 20000005UL
#define BATCH_NORM_GRAD_V3_RA_RECOMPUTE_FLOAT 40000001UL
#define BATCH_NORM_GRAD_V3_RA_RECOMPUTE_FLOAT16 40000002UL
#define BATCH_NORM_GRAD_V3_RA_RECOMPUTE_BFLOAT16 40000003UL
#define BATCH_NORM_GRAD_V3_RA_RECOMPUTE_MIXED_FP16_FP32 40000004UL
#define BATCH_NORM_GRAD_V3_RA_RECOMPUTE_MIXED_BF16_FP32 40000005UL
#define BATCH_NORM_GRAD_V3_RA_SPLIT_R_TILING_KEY 50000000UL
#define BATCH_NORM_GRAD_V3_RAR_SPLIT_CORE_R1 1000UL
static constexpr int DOUBLE_BUFFER = 2;

// inference
#define BATCH_NORM_GRAD_V3_INFER_CHANNEL_LAST_FP32_FP32_F32 900001UL
#define BATCH_NORM_GRAD_V3_INFER_CHANNEL_LAST_FP16_FP16_FP16 900002UL
#define BATCH_NORM_GRAD_V3_INFER_CHANNEL_LAST_BF16_BF16_BF16 900003UL
#define BATCH_NORM_GRAD_V3_INFER_CHANNEL_LAST_FP16_FP32_FP16 900004UL
#define BATCH_NORM_GRAD_V3_INFER_CHANNEL_LAST_FP16_FP32_FP32 900005UL
#define BATCH_NORM_GRAD_V3_INFER_CHANNEL_LAST_BF16_FP32_BF16 900006UL
#define BATCH_NORM_GRAD_V3_INFER_CHANNEL_LAST_BF16_FP32_FP32 900007UL
#define BATCH_NORM_GRAD_V3_INFER_CHANNEL_LAST_FP16_FP16_FP32 900008UL
#define BATCH_NORM_GRAD_V3_INFER_CHANNEL_LAST_BF16_BF16_FP32 900009UL

#define BATCH_NORM_GRAD_V3_INFER_FP32_FP32_F32 910001UL
#define BATCH_NORM_GRAD_V3_INFER_FP16_FP16_FP16 910002UL
#define BATCH_NORM_GRAD_V3_INFER_BF16_BF16_BF16 910003UL
#define BATCH_NORM_GRAD_V3_INFER_FP16_FP32_FP16 910004UL
#define BATCH_NORM_GRAD_V3_INFER_FP16_FP32_FP32 910005UL
#define BATCH_NORM_GRAD_V3_INFER_BF16_FP32_BF16 910006UL
#define BATCH_NORM_GRAD_V3_INFER_BF16_FP32_FP32 910007UL
#define BATCH_NORM_GRAD_V3_INFER_FP16_FP16_FP32 910008UL
#define BATCH_NORM_GRAD_V3_INFER_BF16_BF16_FP32 910009UL

extern "C" __global__ __aicore__ void batch_norm_grad_v3(
    GM_ADDR dy, GM_ADDR x, GM_ADDR weight, GM_ADDR running_mean, GM_ADDR running_var, GM_ADDR save_mean,
    GM_ADDR save_rstd, GM_ADDR dx, GM_ADDR dweight, GM_ADDR dbias, GM_ADDR workspace, GM_ADDR tiling)
{
    TPipe pipe;
    GM_ADDR usrWorkspace = AscendC::GetUserWorkspace(workspace);
    if (TILING_KEY_IS(BATCH_NORM_GRAD_V3_RAR_FULL_LOAD_FLOAT)) {
        GET_TILING_DATA_WITH_STRUCT(BatchNormGradV3RARFullLoadTilingData, tilingDataIn, tiling);
        const BatchNormGradV3RARFullLoadTilingData* __restrict tilingData = &tilingDataIn;
        BatchNormGradV3RARFullLoad<float, float, DOUBLE_BUFFER> op(&pipe);
        op.Init(dy, x, save_mean, save_rstd, weight, dx, dweight, dbias, usrWorkspace, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(BATCH_NORM_GRAD_V3_RAR_FULL_LOAD_FLOAT16)) {
        GET_TILING_DATA_WITH_STRUCT(BatchNormGradV3RARFullLoadTilingData, tilingDataIn, tiling);
        const BatchNormGradV3RARFullLoadTilingData* __restrict tilingData = &tilingDataIn;
        BatchNormGradV3RARFullLoad<half, half, DOUBLE_BUFFER> op(&pipe);
        op.Init(dy, x, save_mean, save_rstd, weight, dx, dweight, dbias, usrWorkspace, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(BATCH_NORM_GRAD_V3_RAR_FULL_LOAD_BFLOAT16)) {
        GET_TILING_DATA_WITH_STRUCT(BatchNormGradV3RARFullLoadTilingData, tilingDataIn, tiling);
        const BatchNormGradV3RARFullLoadTilingData* __restrict tilingData = &tilingDataIn;
        BatchNormGradV3RARFullLoad<bfloat16_t, bfloat16_t, DOUBLE_BUFFER> op(&pipe);
        op.Init(dy, x, save_mean, save_rstd, weight, dx, dweight, dbias, usrWorkspace, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(BATCH_NORM_GRAD_V3_RAR_FULL_LOAD_MIXED_FP16_FP32)) {
        GET_TILING_DATA_WITH_STRUCT(BatchNormGradV3RARFullLoadTilingData, tilingDataIn, tiling);
        const BatchNormGradV3RARFullLoadTilingData* __restrict tilingData = &tilingDataIn;
        BatchNormGradV3RARFullLoad<half, float, DOUBLE_BUFFER> op(&pipe);
        op.Init(dy, x, save_mean, save_rstd, weight, dx, dweight, dbias, usrWorkspace, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(BATCH_NORM_GRAD_V3_RAR_FULL_LOAD_MIXED_BF16_FP32)) {
        GET_TILING_DATA_WITH_STRUCT(BatchNormGradV3RARFullLoadTilingData, tilingDataIn, tiling);
        const BatchNormGradV3RARFullLoadTilingData* __restrict tilingData = &tilingDataIn;
        BatchNormGradV3RARFullLoad<bfloat16_t, float, DOUBLE_BUFFER> op(&pipe);
        op.Init(dy, x, save_mean, save_rstd, weight, dx, dweight, dbias, usrWorkspace, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(BATCH_NORM_GRAD_V3_RAR_RECOMPUTE_SPLIT_R0_FLOAT)) {
        GET_TILING_DATA_WITH_STRUCT(BatchNormGradV3RARRecomputeTilingData, tilingDataIn, tiling);
        const BatchNormGradV3RARRecomputeTilingData* __restrict tilingData = &tilingDataIn;
        BatchNormGradV3RARRecomputeSplitR0<float, float, DOUBLE_BUFFER> op;
        op.Init(dy, x, save_mean, save_rstd, weight, dx, dweight, dbias, usrWorkspace, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(BATCH_NORM_GRAD_V3_RAR_RECOMPUTE_SPLIT_R0_FLOAT16)) {
        GET_TILING_DATA_WITH_STRUCT(BatchNormGradV3RARRecomputeTilingData, tilingDataIn, tiling);
        const BatchNormGradV3RARRecomputeTilingData* __restrict tilingData = &tilingDataIn;
        BatchNormGradV3RARRecomputeSplitR0<half, half, DOUBLE_BUFFER> op;
        op.Init(dy, x, save_mean, save_rstd, weight, dx, dweight, dbias, usrWorkspace, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(BATCH_NORM_GRAD_V3_RAR_RECOMPUTE_SPLIT_R0_BFLOAT16)) {
        GET_TILING_DATA_WITH_STRUCT(BatchNormGradV3RARRecomputeTilingData, tilingDataIn, tiling);
        const BatchNormGradV3RARRecomputeTilingData* __restrict tilingData = &tilingDataIn;
        BatchNormGradV3RARRecomputeSplitR0<bfloat16_t, bfloat16_t, DOUBLE_BUFFER> op;
        op.Init(dy, x, save_mean, save_rstd, weight, dx, dweight, dbias, usrWorkspace, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(BATCH_NORM_GRAD_V3_RAR_RECOMPUTE_SPLIT_R0_MIXED_FP16_FP32)) {
        GET_TILING_DATA_WITH_STRUCT(BatchNormGradV3RARRecomputeTilingData, tilingDataIn, tiling);
        const BatchNormGradV3RARRecomputeTilingData* __restrict tilingData = &tilingDataIn;
        BatchNormGradV3RARRecomputeSplitR0<half, float, DOUBLE_BUFFER> op;
        op.Init(dy, x, save_mean, save_rstd, weight, dx, dweight, dbias, usrWorkspace, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(BATCH_NORM_GRAD_V3_RAR_RECOMPUTE_SPLIT_R0_MIXED_BF16_FP32)) {
        GET_TILING_DATA_WITH_STRUCT(BatchNormGradV3RARRecomputeTilingData, tilingDataIn, tiling);
        const BatchNormGradV3RARRecomputeTilingData* __restrict tilingData = &tilingDataIn;
        BatchNormGradV3RARRecomputeSplitR0<bfloat16_t, float, DOUBLE_BUFFER> op;
        op.Init(dy, x, save_mean, save_rstd, weight, dx, dweight, dbias, usrWorkspace, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(BATCH_NORM_GRAD_V3_RAR_SPLIT_R1_FLOAT)) {
        GET_TILING_DATA_WITH_STRUCT(BatchNormGradV3RARRecomputeTilingData, tilingDataIn, tiling);
        const BatchNormGradV3RARRecomputeTilingData* __restrict tilingData = &tilingDataIn;
        BatchNormGradV3RARSplitR1<float, float, DOUBLE_BUFFER> op(&pipe);
        op.Init(dy, x, save_mean, save_rstd, weight, dx, dweight, dbias, usrWorkspace, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(BATCH_NORM_GRAD_V3_RAR_SPLIT_R1_FLOAT16)) {
        GET_TILING_DATA_WITH_STRUCT(BatchNormGradV3RARRecomputeTilingData, tilingDataIn, tiling);
        const BatchNormGradV3RARRecomputeTilingData* __restrict tilingData = &tilingDataIn;
        BatchNormGradV3RARSplitR1<half, half, DOUBLE_BUFFER> op(&pipe);
        op.Init(dy, x, save_mean, save_rstd, weight, dx, dweight, dbias, usrWorkspace, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(BATCH_NORM_GRAD_V3_RAR_SPLIT_R1_BFLOAT16)) {
        GET_TILING_DATA_WITH_STRUCT(BatchNormGradV3RARRecomputeTilingData, tilingDataIn, tiling);
        const BatchNormGradV3RARRecomputeTilingData* __restrict tilingData = &tilingDataIn;
        BatchNormGradV3RARSplitR1<bfloat16_t, bfloat16_t, DOUBLE_BUFFER> op(&pipe);
        op.Init(dy, x, save_mean, save_rstd, weight, dx, dweight, dbias, usrWorkspace, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(BATCH_NORM_GRAD_V3_RAR_SPLIT_R1_MIXED_FP16_FP32)) {
        GET_TILING_DATA_WITH_STRUCT(BatchNormGradV3RARRecomputeTilingData, tilingDataIn, tiling);
        const BatchNormGradV3RARRecomputeTilingData* __restrict tilingData = &tilingDataIn;
        BatchNormGradV3RARSplitR1<half, float, DOUBLE_BUFFER> op(&pipe);
        op.Init(dy, x, save_mean, save_rstd, weight, dx, dweight, dbias, usrWorkspace, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(BATCH_NORM_GRAD_V3_RAR_SPLIT_R1_MIXED_BF16_FP32)) {
        GET_TILING_DATA_WITH_STRUCT(BatchNormGradV3RARRecomputeTilingData, tilingDataIn, tiling);
        const BatchNormGradV3RARRecomputeTilingData* __restrict tilingData = &tilingDataIn;
        BatchNormGradV3RARSplitR1<bfloat16_t, float, DOUBLE_BUFFER> op(&pipe);
        op.Init(dy, x, save_mean, save_rstd, weight, dx, dweight, dbias, usrWorkspace, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(BATCH_NORM_GRAD_V3_RA_FULL_LOAD_FLOAT)) {
        GET_TILING_DATA_WITH_STRUCT(BatchNormGradV3RAFullLoadTilingData, tilingDataIn, tiling);
        const BatchNormGradV3RAFullLoadTilingData* __restrict tilingData = &tilingDataIn;
        BatchNormGradV3RAFullLoad<float, float, DOUBLE_BUFFER> op(&pipe);
        op.Init(dy, x, save_mean, save_rstd, weight, dx, dweight, dbias, usrWorkspace, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(BATCH_NORM_GRAD_V3_RA_FULL_LOAD_FLOAT16)) {
        GET_TILING_DATA_WITH_STRUCT(BatchNormGradV3RAFullLoadTilingData, tilingDataIn, tiling);
        const BatchNormGradV3RAFullLoadTilingData* __restrict tilingData = &tilingDataIn;
        BatchNormGradV3RAFullLoad<half, half, DOUBLE_BUFFER> op(&pipe);
        op.Init(dy, x, save_mean, save_rstd, weight, dx, dweight, dbias, usrWorkspace, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(BATCH_NORM_GRAD_V3_RA_FULL_LOAD_BFLOAT16)) {
        GET_TILING_DATA_WITH_STRUCT(BatchNormGradV3RAFullLoadTilingData, tilingDataIn, tiling);
        const BatchNormGradV3RAFullLoadTilingData* __restrict tilingData = &tilingDataIn;
        BatchNormGradV3RAFullLoad<bfloat16_t, bfloat16_t, DOUBLE_BUFFER> op(&pipe);
        op.Init(dy, x, save_mean, save_rstd, weight, dx, dweight, dbias, usrWorkspace, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(BATCH_NORM_GRAD_V3_RA_FULL_LOAD_MIXED_FP16_FP32)) {
        GET_TILING_DATA_WITH_STRUCT(BatchNormGradV3RAFullLoadTilingData, tilingDataIn, tiling);
        const BatchNormGradV3RAFullLoadTilingData* __restrict tilingData = &tilingDataIn;
        BatchNormGradV3RAFullLoad<half, float, DOUBLE_BUFFER> op(&pipe);
        op.Init(dy, x, save_mean, save_rstd, weight, dx, dweight, dbias, usrWorkspace, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(BATCH_NORM_GRAD_V3_RA_FULL_LOAD_MIXED_BF16_FP32)) {
        GET_TILING_DATA_WITH_STRUCT(BatchNormGradV3RAFullLoadTilingData, tilingDataIn, tiling);
        const BatchNormGradV3RAFullLoadTilingData* __restrict tilingData = &tilingDataIn;
        BatchNormGradV3RAFullLoad<bfloat16_t, float, DOUBLE_BUFFER> op(&pipe);
        op.Init(dy, x, save_mean, save_rstd, weight, dx, dweight, dbias, usrWorkspace, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(BATCH_NORM_GRAD_V3_RA_RECOMPUTE_FLOAT)) {
        GET_TILING_DATA_WITH_STRUCT(BatchNormGradV3RARecomputeTilingData, tilingDataIn, tiling);
        const BatchNormGradV3RARecomputeTilingData* __restrict tilingData = &tilingDataIn;
        BatchNormGradV3RARecompute<float, float, DOUBLE_BUFFER> op(&pipe);
        op.Init(dy, x, save_mean, save_rstd, weight, dx, dweight, dbias, usrWorkspace, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(BATCH_NORM_GRAD_V3_RA_RECOMPUTE_FLOAT16)) {
        GET_TILING_DATA_WITH_STRUCT(BatchNormGradV3RARecomputeTilingData, tilingDataIn, tiling);
        const BatchNormGradV3RARecomputeTilingData* __restrict tilingData = &tilingDataIn;
        BatchNormGradV3RARecompute<half, half, DOUBLE_BUFFER> op(&pipe);
        op.Init(dy, x, save_mean, save_rstd, weight, dx, dweight, dbias, usrWorkspace, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(BATCH_NORM_GRAD_V3_RA_RECOMPUTE_BFLOAT16)) {
        GET_TILING_DATA_WITH_STRUCT(BatchNormGradV3RARecomputeTilingData, tilingDataIn, tiling);
        const BatchNormGradV3RARecomputeTilingData* __restrict tilingData = &tilingDataIn;
        BatchNormGradV3RARecompute<bfloat16_t, bfloat16_t, DOUBLE_BUFFER> op(&pipe);
        op.Init(dy, x, save_mean, save_rstd, weight, dx, dweight, dbias, usrWorkspace, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(BATCH_NORM_GRAD_V3_RA_RECOMPUTE_MIXED_FP16_FP32)) {
        GET_TILING_DATA_WITH_STRUCT(BatchNormGradV3RARecomputeTilingData, tilingDataIn, tiling);
        const BatchNormGradV3RARecomputeTilingData* __restrict tilingData = &tilingDataIn;
        BatchNormGradV3RARecompute<half, float, DOUBLE_BUFFER> op(&pipe);
        op.Init(dy, x, save_mean, save_rstd, weight, dx, dweight, dbias, usrWorkspace, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(BATCH_NORM_GRAD_V3_RA_RECOMPUTE_MIXED_BF16_FP32)) {
        GET_TILING_DATA_WITH_STRUCT(BatchNormGradV3RARecomputeTilingData, tilingDataIn, tiling);
        const BatchNormGradV3RARecomputeTilingData* __restrict tilingData = &tilingDataIn;
        BatchNormGradV3RARecompute<bfloat16_t, float, DOUBLE_BUFFER> op(&pipe);
        op.Init(dy, x, save_mean, save_rstd, weight, dx, dweight, dbias, usrWorkspace, tilingData);
        op.Process();
    } else if (
        TILING_KEY_IS(BATCH_NORM_GRAD_V3_INFER_CHANNEL_LAST_FP32_FP32_F32) ||
        TILING_KEY_IS(BATCH_NORM_GRAD_V3_INFER_CHANNEL_LAST_FP16_FP16_FP16) ||
        TILING_KEY_IS(BATCH_NORM_GRAD_V3_INFER_CHANNEL_LAST_BF16_BF16_BF16) ||
        TILING_KEY_IS(BATCH_NORM_GRAD_V3_INFER_CHANNEL_LAST_FP16_FP32_FP16) ||
        TILING_KEY_IS(BATCH_NORM_GRAD_V3_INFER_CHANNEL_LAST_FP16_FP32_FP32) ||
        TILING_KEY_IS(BATCH_NORM_GRAD_V3_INFER_CHANNEL_LAST_BF16_FP32_BF16) ||
        TILING_KEY_IS(BATCH_NORM_GRAD_V3_INFER_CHANNEL_LAST_BF16_FP32_FP32) ||
        TILING_KEY_IS(BATCH_NORM_GRAD_V3_INFER_CHANNEL_LAST_FP16_FP16_FP32) ||
        TILING_KEY_IS(BATCH_NORM_GRAD_V3_INFER_CHANNEL_LAST_BF16_BF16_FP32)) {
        GET_TILING_DATA_WITH_STRUCT(BatchNormGradV3InferChannelLastTilingData, tiling_data_in, tiling);
        const BatchNormGradV3InferChannelLastTilingData* __restrict tilingData = &tiling_data_in;
        BatchNormGradV3InferChannelLast<DTYPE_DY, DTYPE_WEIGHT, DTYPE_RUNNING_VAR> op(tilingData);
        op.Init(dy, weight, running_var, dx, &pipe);
        op.Process();
    } else if (
        TILING_KEY_IS(BATCH_NORM_GRAD_V3_INFER_FP32_FP32_F32) ||
        TILING_KEY_IS(BATCH_NORM_GRAD_V3_INFER_FP16_FP16_FP16) ||
        TILING_KEY_IS(BATCH_NORM_GRAD_V3_INFER_BF16_BF16_BF16) ||
        TILING_KEY_IS(BATCH_NORM_GRAD_V3_INFER_FP16_FP32_FP16) ||
        TILING_KEY_IS(BATCH_NORM_GRAD_V3_INFER_FP16_FP32_FP32) ||
        TILING_KEY_IS(BATCH_NORM_GRAD_V3_INFER_BF16_FP32_BF16) ||
        TILING_KEY_IS(BATCH_NORM_GRAD_V3_INFER_BF16_FP32_FP32) ||
        TILING_KEY_IS(BATCH_NORM_GRAD_V3_INFER_FP16_FP16_FP32) ||
        TILING_KEY_IS(BATCH_NORM_GRAD_V3_INFER_BF16_BF16_FP32)) {
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
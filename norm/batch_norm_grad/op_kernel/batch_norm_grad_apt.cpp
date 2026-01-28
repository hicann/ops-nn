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
 * \file batch_norm_grad_apt.cpp
 * \brief
 */
#include "kernel_operator.h"
#include "arch35/batch_norm_grad_full_load_regbase.h"
#include "arch35/batch_norm_grad_recompute_split_r0_regbase.h"
#include "arch35/batch_norm_grad_split_r1_regbase.h"
#include "arch35/batch_norm_grad_ra_full_load_regbase.h"
#include "arch35/batch_norm_grad_ra_recompute_regbase.h"
#include "arch35/batch_norm_grad_ra_split_r_regbase.h"
#include "arch35/batch_norm_grad_infer_channel_last.h"
#include "arch35/batch_norm_grad_infer.h"
#include "arch35/batch_norm_grad_rar_split_core_r1.h"

using namespace BatchNormGrad;
using namespace BNGRARRecomputeSplitR0;

#define BATCH_NORM_GRAD_RAR_FULL_LOAD 10000000UL
#define BATCH_NORM_GRAD_RA_FULL_LOAD 20000000UL
#define BATCH_NORM_GRAD_RAR_SPLIT_R1 31000000UL
#define BATCH_NORM_GRAD_RAR_RECOMPUTE_SPLIT_R0 32000000UL
#define BATCH_NORM_GRAD_RA_RECOMPUTE 40000000UL

#define BATCH_NORM_GRAD_RA_SPLIT_R_TILING_KEY 50000000UL
#define BATCH_NORM_GRAD_RAR_SPLIT_CORE_R1 1000UL

#define BATCH_NORM_GRAD_INFER_CHANNEL_LAST 900000UL
#define BATCH_NORM_GRAD_INFER_SPLIT_R1 910001UL
#define BATCH_NORM_GRAD_INFER_SPLIT_R0 910002UL

static constexpr int DOUBLE_BUFFER = 2;

extern "C" __global__ __aicore__ void batch_norm_grad(GM_ADDR y_backprop, GM_ADDR x, GM_ADDR scale, GM_ADDR reserve_space_1,
    GM_ADDR reserve_space_2, GM_ADDR reserve_space_3, GM_ADDR x_backprop, GM_ADDR scale_backprop, GM_ADDR offset_backprop, 
    GM_ADDR reserve_space_4, GM_ADDR reserve_space_5, GM_ADDR workspace, GM_ADDR tiling)
{
    TPipe pipe;
    GM_ADDR usrWorkspace = AscendC::GetUserWorkspace(workspace);

    if (TILING_KEY_IS(BATCH_NORM_GRAD_RAR_FULL_LOAD)) {
        GET_TILING_DATA_WITH_STRUCT(BatchNormGradRARFullLoadTilingData, tilingDataIn, tiling);
        BatchNormGradRARFullLoad<DTYPE_Y_BACKPROP, DTYPE_SCALE, DOUBLE_BUFFER> op(&pipe);
        op.Init(y_backprop, x, reserve_space_1, reserve_space_2, scale, x_backprop, scale_backprop, offset_backprop, usrWorkspace, &tilingDataIn);
        op.Process();
    } else if (TILING_KEY_IS(BATCH_NORM_GRAD_RAR_RECOMPUTE_SPLIT_R0)) {
        GET_TILING_DATA_WITH_STRUCT(BatchNormGradRARRecomputeTilingData, tilingDataIn, tiling);
        BatchNormGradRARRecomputeSplitR0<DTYPE_Y_BACKPROP, DTYPE_SCALE, DOUBLE_BUFFER> op;
        op.Init(y_backprop, x, reserve_space_1, reserve_space_2, scale, x_backprop, scale_backprop, offset_backprop, usrWorkspace, &tilingDataIn);
        op.Process();
    } else if (TILING_KEY_IS(BATCH_NORM_GRAD_RAR_SPLIT_R1)) {
        GET_TILING_DATA_WITH_STRUCT(BatchNormGradRARRecomputeTilingData, tilingDataIn, tiling);
        BatchNormGradRARSplitR1<DTYPE_Y_BACKPROP, DTYPE_SCALE, DOUBLE_BUFFER> op(&pipe);
        op.Init(y_backprop, x, reserve_space_1, reserve_space_2, scale, x_backprop, scale_backprop, offset_backprop, usrWorkspace, &tilingDataIn);
        op.Process();
    } else if (TILING_KEY_IS(BATCH_NORM_GRAD_RA_FULL_LOAD)) {
        GET_TILING_DATA_WITH_STRUCT(BatchNormGradRAFullLoadTilingData, tilingDataIn, tiling);
        BatchNormGradRAFullLoad<DTYPE_Y_BACKPROP, DTYPE_SCALE, DOUBLE_BUFFER> op(&pipe);
        op.Init(y_backprop, x, reserve_space_1, reserve_space_2, scale, x_backprop, scale_backprop, offset_backprop, usrWorkspace, &tilingDataIn);
        op.Process();
    } else if (TILING_KEY_IS(BATCH_NORM_GRAD_RA_RECOMPUTE)) {
        GET_TILING_DATA_WITH_STRUCT(BatchNormGradRARecomputeTilingData, tilingDataIn, tiling);
        BatchNormGradRARecompute<DTYPE_Y_BACKPROP, DTYPE_SCALE, DOUBLE_BUFFER> op(&pipe);
        op.Init(y_backprop, x, reserve_space_1, reserve_space_2, scale, x_backprop, scale_backprop, offset_backprop, usrWorkspace, &tilingDataIn);
        op.Process();
    } else if (TILING_KEY_IS(BATCH_NORM_GRAD_INFER_CHANNEL_LAST)) {
        GET_TILING_DATA_WITH_STRUCT(BatchNormGradInferChannelLastTilingData, tiling_data_in, tiling);
        BatchNormGradInferChannelLast<DTYPE_Y_BACKPROP, DTYPE_SCALE> op(&pipe, &tiling_data_in);
        op.Process(y_backprop, x, scale, reserve_space_1, reserve_space_2, x_backprop, scale_backprop, offset_backprop, usrWorkspace);
    } else if (TILING_KEY_IS(BATCH_NORM_GRAD_INFER_SPLIT_R1)) {
        GET_TILING_DATA_WITH_STRUCT(BatchNormGradInferTilingData, tiling_data_in, tiling);
        BatchNormGradInfer<DTYPE_Y_BACKPROP, DTYPE_SCALE, false> op(&pipe, &tiling_data_in);
        op.Process(y_backprop, x, scale, reserve_space_1, reserve_space_2, x_backprop, scale_backprop, offset_backprop, usrWorkspace);
    } else if (TILING_KEY_IS(BATCH_NORM_GRAD_INFER_SPLIT_R0)) {
        GET_TILING_DATA_WITH_STRUCT(BatchNormGradInferTilingData, tiling_data_in, tiling);
        BatchNormGradInfer<DTYPE_Y_BACKPROP, DTYPE_SCALE, true> op(&pipe, &tiling_data_in);
        op.Process(y_backprop, x, scale, reserve_space_1, reserve_space_2, x_backprop, scale_backprop, offset_backprop, usrWorkspace);
    } else if (TILING_KEY_IS(BATCH_NORM_GRAD_RAR_SPLIT_CORE_R1)) {
        GET_TILING_DATA_WITH_STRUCT(BatchNormGradRARSplitCoreR1TilingData, tiling_data_in, tiling);
        BatchNormGradRARSplitCoreR1<DTYPE_Y_BACKPROP, DTYPE_SCALE> op(&pipe, &tiling_data_in);
        op.Init(y_backprop, x, reserve_space_1, reserve_space_2, scale, x_backprop, scale_backprop, offset_backprop, usrWorkspace);
        op.Process();
    } else if (TILING_KEY_IS(BATCH_NORM_GRAD_RA_SPLIT_R_TILING_KEY)) {
        GET_TILING_DATA_WITH_STRUCT(BatchNormGradRASplitRTilingData, tilingDataIn, tiling);
        BatchNormGradRASplitR<DTYPE_Y_BACKPROP, DTYPE_SCALE> op(&pipe);
        op.Init(y_backprop, x, reserve_space_1, reserve_space_2, scale, x_backprop, scale_backprop, offset_backprop, usrWorkspace, &tilingDataIn);
        op.Process();
    }
}
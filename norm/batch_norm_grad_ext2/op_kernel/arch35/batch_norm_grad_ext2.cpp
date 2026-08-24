/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file batch_norm_grad_ext2.cpp
 * \brief BatchNormGradExt2 kernel entry for arch35 / Ascend950.
 */
#include "kernel_operator.h"
#include "batch_norm_grad_ext2_full_load_regbase.h"
#include "batch_norm_grad_ext2_recompute_split_r0_regbase.h"
#include "batch_norm_grad_ext2_split_r1_regbase.h"
#include "batch_norm_grad_ext2_ra_full_load_regbase.h"
#include "batch_norm_grad_ext2_ra_recompute_regbase.h"
#include "batch_norm_grad_ext2_ra_split_r_regbase.h"
#include "batch_norm_grad_ext2_infer_channel_last.h"
#include "batch_norm_grad_ext2_infer.h"
#include "batch_norm_grad_ext2_rar_split_core_r1.h"
#include "batch_norm_grad_ext2_rar_split_core_r0.h"

using namespace BatchNormGradExt2;
using namespace BNGRARRecomputeSplitR0;

#define BATCH_NORM_GRAD_EXT2_RAR_FULL_LOAD 10000000UL
#define BATCH_NORM_GRAD_EXT2_RA_FULL_LOAD 20000000UL
#define BATCH_NORM_GRAD_EXT2_RAR_SPLIT_R1 31000000UL
#define BATCH_NORM_GRAD_EXT2_RAR_RECOMPUTE_SPLIT_R0 32000000UL
#define BATCH_NORM_GRAD_EXT2_RA_RECOMPUTE 40000000UL

#define BATCH_NORM_GRAD_EXT2_RA_SPLIT_R_TILING_KEY 50000000UL
#define BATCH_NORM_GRAD_EXT2_RAR_SPLIT_CORE_R1 1000UL
#define BATCH_NORM_GRAD_EXT2_RAR_SPLIT_CORE_R0 1100UL

#define BATCH_NORM_GRAD_EXT2_INFER_CHANNEL_LAST 900000UL
#define BATCH_NORM_GRAD_EXT2_INFER_SPLIT_R1 910001UL
#define BATCH_NORM_GRAD_EXT2_INFER_SPLIT_R0 910002UL

static constexpr int DOUBLE_BUFFER = 2;

extern "C" __global__ __aicore__ void batch_norm_grad_ext2(GM_ADDR y_backprop, GM_ADDR x, GM_ADDR scale,
                                                           GM_ADDR reserve_space_1, GM_ADDR reserve_space_2,
                                                           GM_ADDR x_backprop, GM_ADDR scale_backprop,
                                                           GM_ADDR offset_backprop, GM_ADDR reserve_space_3,
                                                           GM_ADDR reserve_space_4, GM_ADDR workspace, GM_ADDR tiling)
{
    TPipe pipe;
    GM_ADDR usrWorkspace = AscendC::GetUserWorkspace(workspace);

    REGISTER_TILING_DEFAULT(BatchNormGradExt2RARFullLoadTilingData);
    if (TILING_KEY_IS(BATCH_NORM_GRAD_EXT2_RAR_FULL_LOAD)) {
        REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 10000000", BatchNormGradExt2RARFullLoadTilingData);
        GET_TILING_DATA_WITH_STRUCT(BatchNormGradExt2RARFullLoadTilingData, tilingDataIn, tiling);
        BatchNormGradExt2RARFullLoad<DTYPE_Y_BACKPROP, DTYPE_SCALE, DOUBLE_BUFFER> op(&pipe);
        op.Init(y_backprop, x, reserve_space_1, reserve_space_2, scale, x_backprop, scale_backprop, offset_backprop,
                usrWorkspace, &tilingDataIn);
        op.Process();
    } else if (TILING_KEY_IS(BATCH_NORM_GRAD_EXT2_RAR_RECOMPUTE_SPLIT_R0)) {
        REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 32000000", BatchNormGradExt2RARRecomputeTilingData);
        GET_TILING_DATA_WITH_STRUCT(BatchNormGradExt2RARRecomputeTilingData, tilingDataIn, tiling);
        BatchNormGradExt2RARRecomputeSplitR0<DTYPE_Y_BACKPROP, DTYPE_SCALE, DOUBLE_BUFFER> op;
        op.Init(y_backprop, x, reserve_space_1, reserve_space_2, scale, x_backprop, scale_backprop, offset_backprop,
                usrWorkspace, &tilingDataIn);
        op.Process();
    } else if (TILING_KEY_IS(BATCH_NORM_GRAD_EXT2_RAR_SPLIT_R1)) {
        REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 31000000", BatchNormGradExt2RARRecomputeTilingData);
        GET_TILING_DATA_WITH_STRUCT(BatchNormGradExt2RARRecomputeTilingData, tilingDataIn, tiling);
        BatchNormGradExt2RARSplitR1<DTYPE_Y_BACKPROP, DTYPE_SCALE, DOUBLE_BUFFER> op(&pipe);
        op.Init(y_backprop, x, reserve_space_1, reserve_space_2, scale, x_backprop, scale_backprop, offset_backprop,
                usrWorkspace, &tilingDataIn);
        op.Process();
    } else if (TILING_KEY_IS(BATCH_NORM_GRAD_EXT2_RA_FULL_LOAD)) {
        REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 20000000", BatchNormGradExt2RAFullLoadTilingData);
        GET_TILING_DATA_WITH_STRUCT(BatchNormGradExt2RAFullLoadTilingData, tilingDataIn, tiling);
        BatchNormGradExt2RAFullLoad<DTYPE_Y_BACKPROP, DTYPE_SCALE, DOUBLE_BUFFER> op(&pipe);
        op.Init(y_backprop, x, reserve_space_1, reserve_space_2, scale, x_backprop, scale_backprop, offset_backprop,
                usrWorkspace, &tilingDataIn);
        op.Process();
    } else if (TILING_KEY_IS(BATCH_NORM_GRAD_EXT2_RA_RECOMPUTE)) {
        REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 40000000", BatchNormGradExt2RARecomputeTilingData);
        GET_TILING_DATA_WITH_STRUCT(BatchNormGradExt2RARecomputeTilingData, tilingDataIn, tiling);
        BatchNormGradExt2RARecompute<DTYPE_Y_BACKPROP, DTYPE_SCALE, DOUBLE_BUFFER> op(&pipe);
        op.Init(y_backprop, x, reserve_space_1, reserve_space_2, scale, x_backprop, scale_backprop, offset_backprop,
                usrWorkspace, &tilingDataIn);
        op.Process();
    } else if (TILING_KEY_IS(BATCH_NORM_GRAD_EXT2_INFER_CHANNEL_LAST)) {
        REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 900000", BatchNormGradExt2InferChannelLastTilingData);
        GET_TILING_DATA_WITH_STRUCT(BatchNormGradExt2InferChannelLastTilingData, tiling_data_in, tiling);
        BatchNormGradExt2InferChannelLast<DTYPE_Y_BACKPROP, DTYPE_SCALE> op(&pipe, &tiling_data_in);
        op.Process(y_backprop, x, scale, reserve_space_1, reserve_space_2, x_backprop, scale_backprop, offset_backprop,
                   usrWorkspace);
    } else if (TILING_KEY_IS(BATCH_NORM_GRAD_EXT2_INFER_SPLIT_R1)) {
        REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 910001", BatchNormGradExt2InferTilingData);
        GET_TILING_DATA_WITH_STRUCT(BatchNormGradExt2InferTilingData, tiling_data_in, tiling);
        BatchNormGradExt2Infer<DTYPE_Y_BACKPROP, DTYPE_SCALE, false> op(&pipe, &tiling_data_in);
        op.Process(y_backprop, x, scale, reserve_space_1, reserve_space_2, x_backprop, scale_backprop, offset_backprop,
                   usrWorkspace);
    } else if (TILING_KEY_IS(BATCH_NORM_GRAD_EXT2_INFER_SPLIT_R0)) {
        REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 910002", BatchNormGradExt2InferTilingData);
        GET_TILING_DATA_WITH_STRUCT(BatchNormGradExt2InferTilingData, tiling_data_in, tiling);
        BatchNormGradExt2Infer<DTYPE_Y_BACKPROP, DTYPE_SCALE, true> op(&pipe, &tiling_data_in);
        op.Process(y_backprop, x, scale, reserve_space_1, reserve_space_2, x_backprop, scale_backprop, offset_backprop,
                   usrWorkspace);
    } else if (TILING_KEY_IS(BATCH_NORM_GRAD_EXT2_RAR_SPLIT_CORE_R1)) {
        REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 1000", BatchNormGradExt2RARSplitCoreR1TilingData);
        GET_TILING_DATA_WITH_STRUCT(BatchNormGradExt2RARSplitCoreR1TilingData, tiling_data_in, tiling);
        BatchNormGradExt2RARSplitCoreR1<DTYPE_Y_BACKPROP, DTYPE_SCALE> op(&pipe, &tiling_data_in);
        op.Init(y_backprop, x, reserve_space_1, reserve_space_2, scale, x_backprop, scale_backprop, offset_backprop,
                usrWorkspace);
        op.Process();
    } else if (TILING_KEY_IS(BATCH_NORM_GRAD_EXT2_RAR_SPLIT_CORE_R0)) {
        REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 1100", BatchNormGradExt2RARSplitCoreR0TilingData);
        GET_TILING_DATA_WITH_STRUCT(BatchNormGradExt2RARSplitCoreR0TilingData, tiling_data_in, tiling);
        BatchNormGradExt2RARSplitCoreR0<DTYPE_Y_BACKPROP, DTYPE_SCALE> op(&pipe, &tiling_data_in);
        op.Init(y_backprop, x, reserve_space_1, reserve_space_2, scale, x_backprop, scale_backprop, offset_backprop,
                usrWorkspace);
        op.Process();
    } else if (TILING_KEY_IS(BATCH_NORM_GRAD_EXT2_RA_SPLIT_R_TILING_KEY)) {
        REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 50000000", BatchNormGradExt2RASplitRTilingData);
        GET_TILING_DATA_WITH_STRUCT(BatchNormGradExt2RASplitRTilingData, tilingDataIn, tiling);
        BatchNormGradExt2RASplitR<DTYPE_Y_BACKPROP, DTYPE_SCALE> op(&pipe);
        op.Init(y_backprop, x, reserve_space_1, reserve_space_2, scale, x_backprop, scale_backprop, offset_backprop,
                usrWorkspace, &tilingDataIn);
        op.Process();
    }
}

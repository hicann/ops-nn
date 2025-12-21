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
 * \file batch_norm_grad_v3.cpp
 * \brief
 */
#include "kernel_operator.h"

#include "batch_norm_grad_v3_infer_fullLoad.h"
#include "batch_norm_grad_v3_infer_splitLoad.h"
#include "batch_norm_grad_v3_infer_splitLoad_crossCore.h"
#include "batch_norm_grad_v3_train_fullLoad.h"
#include "batch_norm_grad_v3_train_splitLoad.h"
#include "batch_norm_grad_v3_train_splitLoad_crossCore.h"

using namespace BatchNormGradV3;

extern "C" __global__ __aicore__ void batch_norm_grad_v3(
    GM_ADDR dy, GM_ADDR x, GM_ADDR weight, GM_ADDR running_mean, GM_ADDR running_var, GM_ADDR save_mean,
    GM_ADDR save_rstd, GM_ADDR dx, GM_ADDR dweight, GM_ADDR dbias, GM_ADDR workspace, GM_ADDR tiling)
{
    if (g_coreType == AIC) {
        return;
    }
    TPipe pipe;
    GM_ADDR usrWorkspace = AscendC::GetUserWorkspace(workspace);
    if (TILING_KEY_IS(10001)) {
        GET_TILING_DATA_WITH_STRUCT(BatchNormGradV3FullLoadTilingData, tiling_data_in, tiling);
        const BatchNormGradV3FullLoadTilingData* __restrict tilingData = &tiling_data_in;
        BatchNormGradV3InferFullload<DTYPE_DY, DTYPE_WEIGHT, DTYPE_RUNNING_VAR> op(tilingData);
        GMStruct gmStruct{dy, x, weight, running_mean, running_var, dx, dweight, dbias, usrWorkspace};
        op.Init(gmStruct, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(20001)) {
        GET_TILING_DATA_WITH_STRUCT(BatchNormGradV3SplitLoadTilingData, tiling_data_in, tiling);
        const BatchNormGradV3SplitLoadTilingData* __restrict tilingData = &tiling_data_in;
        BatchNormGradV3InferSplitLoad<DTYPE_DY, DTYPE_WEIGHT, DTYPE_RUNNING_VAR, BNGV3SplitMode::R0_SPLIT_MODE> op(
            tilingData);
        GMStruct gmStruct{dy, x, weight, running_mean, running_var, dx, dweight, dbias, usrWorkspace};
        op.Init(gmStruct, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(20002)) {
        GET_TILING_DATA_WITH_STRUCT(BatchNormGradV3SplitLoadTilingData, tiling_data_in, tiling);
        const BatchNormGradV3SplitLoadTilingData* __restrict tilingData = &tiling_data_in;
        BatchNormGradV3InferSplitLoad<DTYPE_DY, DTYPE_WEIGHT, DTYPE_RUNNING_VAR, BNGV3SplitMode::R1_SPLIT_MODE> op(
            tilingData);
        GMStruct gmStruct{dy, x, weight, running_mean, running_var, dx, dweight, dbias, usrWorkspace};
        op.Init(gmStruct, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(20011)) {
        GET_TILING_DATA_WITH_STRUCT(BatchNormGradV3SplitLoadCrossCoreTilingData, tiling_data_in, tiling);
        const BatchNormGradV3SplitLoadCrossCoreTilingData* __restrict tilingData = &tiling_data_in;
        BatchNormGradV3InferSplitLoadCrossCore<DTYPE_DY, DTYPE_WEIGHT, DTYPE_RUNNING_VAR, BNGV3SplitMode::R0_SPLIT_MODE>
            op(tilingData);
        GMStruct gmStruct{dy, x, weight, running_mean, running_var, dx, dweight, dbias, usrWorkspace};
        op.Init(gmStruct, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(20012)) {
        GET_TILING_DATA_WITH_STRUCT(BatchNormGradV3SplitLoadCrossCoreTilingData, tiling_data_in, tiling);
        const BatchNormGradV3SplitLoadCrossCoreTilingData* __restrict tilingData = &tiling_data_in;
        BatchNormGradV3InferSplitLoadCrossCore<DTYPE_DY, DTYPE_WEIGHT, DTYPE_RUNNING_VAR, BNGV3SplitMode::R1_SPLIT_MODE>
            op(tilingData);
        GMStruct gmStruct{dy, x, weight, running_mean, running_var, dx, dweight, dbias, usrWorkspace};
        op.Init(gmStruct, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(30001)) {
        GET_TILING_DATA_WITH_STRUCT(BatchNormGradV3FullLoadTilingData, tiling_data_in, tiling);
        const BatchNormGradV3FullLoadTilingData* __restrict tilingData = &tiling_data_in;
        BatchNormGradV3TrainFullload<DTYPE_DY, DTYPE_WEIGHT, DTYPE_SAVE_MEAN> op(tilingData);
        GMStruct gmStruct{dy, x, weight, save_mean, save_rstd, dx, dweight, dbias, usrWorkspace};
        op.Init(gmStruct, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(40001)) {
        GET_TILING_DATA_WITH_STRUCT(BatchNormGradV3SplitLoadTilingData, tiling_data_in, tiling);
        const BatchNormGradV3SplitLoadTilingData* __restrict tilingData = &tiling_data_in;
        BatchNormGradV3TrainSplitLoad<DTYPE_DY, DTYPE_WEIGHT, DTYPE_SAVE_MEAN, BNGV3SplitMode::R0_SPLIT_MODE> op(
            tilingData);
        GMStruct gmStruct{dy, x, weight, save_mean, save_rstd, dx, dweight, dbias, usrWorkspace};
        op.Init(gmStruct, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(40002)) {
        GET_TILING_DATA_WITH_STRUCT(BatchNormGradV3SplitLoadTilingData, tiling_data_in, tiling);
        const BatchNormGradV3SplitLoadTilingData* __restrict tilingData = &tiling_data_in;
        BatchNormGradV3TrainSplitLoad<DTYPE_DY, DTYPE_WEIGHT, DTYPE_SAVE_MEAN, BNGV3SplitMode::R1_SPLIT_MODE> op(
            tilingData);
        GMStruct gmStruct{dy, x, weight, save_mean, save_rstd, dx, dweight, dbias, usrWorkspace};
        op.Init(gmStruct, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(40011)) {
        GET_TILING_DATA_WITH_STRUCT(BatchNormGradV3SplitLoadCrossCoreTilingData, tiling_data_in, tiling);
        const BatchNormGradV3SplitLoadCrossCoreTilingData* __restrict tilingData = &tiling_data_in;
        BatchNormGradV3TrainSplitLoadCrossCore<DTYPE_DY, DTYPE_WEIGHT, DTYPE_SAVE_MEAN, BNGV3SplitMode::R0_SPLIT_MODE>
            op(tilingData);
        GMStruct gmStruct{dy, x, weight, save_mean, save_rstd, dx, dweight, dbias, usrWorkspace};
        op.Init(gmStruct, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(40012)) {
        GET_TILING_DATA_WITH_STRUCT(BatchNormGradV3SplitLoadCrossCoreTilingData, tiling_data_in, tiling);
        const BatchNormGradV3SplitLoadCrossCoreTilingData* __restrict tilingData = &tiling_data_in;
        BatchNormGradV3TrainSplitLoadCrossCore<DTYPE_DY, DTYPE_WEIGHT, DTYPE_SAVE_MEAN, BNGV3SplitMode::R1_SPLIT_MODE>
            op(tilingData);
        GMStruct gmStruct{dy, x, weight, save_mean, save_rstd, dx, dweight, dbias, usrWorkspace};
        op.Init(gmStruct, &pipe);
        op.Process();
    }
}
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
 * \file weight_quant_batch_matmul_experiment.cpp
 * \brief
 */

#include "msd/weight_quant_batch_matmul_experiment_msd_controller.h"
#include "weight_quant_batch_matmul_experiment_tiling_key.h"

using AscendC::GetUserWorkspace;
using WeightQuantBatchMatmulExperimental::WeightQuantBatchMatmulExperimentTilingData;
using WeightQuantBatchMatmulExperimental::WqbmmExpMsdController;
#define WQBMM_EXP_IMPL_CLASS(templateClass)                                                                         \
    do {                                                                                                            \
        GET_TILING_DATA_WITH_STRUCT(WeightQuantBatchMatmulExperimental::WeightQuantBatchMatmulExperimentTilingData, \
                                    tilingDataIn, tiling);                                                          \
        templateClass<DTYPE_X, DTYPE_WEIGHT, MSD_MODE> op;                                                                    \
        op.Init(xGM, weightGM, antiquantScaleGM, yGM, user, &tilingDataIn, &tPipe);                                 \
        op.Process();                                                                                               \
    } while (0)

template <int MSD_MODE>
__global__ __aicore__ void weight_quant_batch_matmul_experiment(GM_ADDR xGM, GM_ADDR weightGM, GM_ADDR antiquantScaleGM,
                                                                GM_ADDR yGM, GM_ADDR workspaceGM, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(WeightQuantBatchMatmulExperimentTilingData);
    AscendC::TPipe tPipe;
    GM_ADDR user = GetUserWorkspace(workspaceGM);
    WQBMM_EXP_IMPL_CLASS(WqbmmExpMsdController);
}

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
 * \file in_training_update_grad.cpp
 * \brief arch35 entry: dispatch by TilingKey; dtype (fp16/fp32) is selected by DTYPE_DY at compile time.
 */

#include "kernel_operator.h"
#include "arch35/in_training_update_grad_tiling_data.h"
#include "arch35/in_training_update_grad_common.h"
#include "arch35/in_training_update_grad_full_load.h"
#include "arch35/in_training_update_grad_stream.h"
#include "arch35/in_training_update_grad_reduce_empty.h"

using namespace AscendC;
using namespace InTrainingUpdateGrad;

#define TILINGKEY_REDUCE_EMPTY 50000
#define TILINGKEY_FULL_LOAD 100000
#define TILINGKEY_STREAM 200000

extern "C" __global__ __aicore__ void in_training_update_grad(GM_ADDR dy, GM_ADDR x, GM_ADDR variance, GM_ADDR mean,
                                                              GM_ADDR res_gamma, GM_ADDR res_beta, GM_ADDR workspace,
                                                              GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(InTrainingUpdateGradFullLoadTilingData);
    if (TILING_KEY_IS(TILINGKEY_FULL_LOAD)) {
        GET_TILING_DATA_WITH_STRUCT(InTrainingUpdateGradFullLoadTilingData, tilingData, tiling);
        InTrainingUpdateGradFullLoad<DTYPE_DY> op(&tilingData);
        op.Init(dy, x, variance, mean, res_gamma, res_beta);
        op.Process();
    } else if (TILING_KEY_IS(TILINGKEY_STREAM)) {
        GET_TILING_DATA_WITH_STRUCT(InTrainingUpdateGradStreamTilingData, tilingData, tiling);
        InTrainingUpdateGradStream<DTYPE_DY> op(&tilingData);
        op.Init(dy, x, variance, mean, res_gamma, res_beta);
        op.Process();
    } else if (TILING_KEY_IS(TILINGKEY_REDUCE_EMPTY)) {
        GET_TILING_DATA_WITH_STRUCT(InTrainingUpdateGradReduceEmptyTilingData, tilingData, tiling);
        InTrainingUpdateGradReduceEmpty op(&tilingData);
        op.Init(res_gamma, res_beta);
        op.Process();
    }
}

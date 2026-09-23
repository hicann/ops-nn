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
 * \file in_training_update_v2.cpp
 * \brief INTrainingUpdateV2 arch35 kernel entry.
 */

#define K_MAX_SHAPE_DIM 0
#include "kernel_operator.h"
#include "in_training_update_v2_empty.h"
#include "in_training_update_v2_nchw.h"
#include "in_training_update_v2_nhwc.h"
#include "in_training_update_v2_tiling_data.h"
#include "in_training_update_v2_tiling_key.h"

using namespace AscendC;
using namespace INTrainingUpdateV2Ops;

REGISTER_TILING_DEFAULT(INTrainingUpdateV2TilingData);

template <int TILING_MODE>
__global__ __aicore__ void in_training_update_v2(GM_ADDR x, GM_ADDR sum, GM_ADDR square_sum, GM_ADDR gamma,
                                                 GM_ADDR beta, GM_ADDR mean, GM_ADDR variance, GM_ADDR y,
                                                 GM_ADDR batch_mean, GM_ADDR batch_variance, GM_ADDR workspace,
                                                 GM_ADDR tiling)
{
    if (g_coreType == AscendC::AIC) {
        return;
    }
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GET_TILING_DATA_WITH_STRUCT(INTrainingUpdateV2TilingData, tilingData, tiling);
    if constexpr (TILING_MODE == IN_TRAINING_UPDATE_V2_EMPTY_KEY) {
        INTrainingUpdateV2Empty op;
        op.Process();
        return;
    }
    const int64_t blockIndex = static_cast<int64_t>(GetBlockIdx());
    if (tilingData.rCores <= 0 || tilingData.unitBlocks <= 0 ||
        blockIndex / tilingData.rCores >= tilingData.unitBlocks) {
        return;
    }

    TPipe pipe;
    if constexpr (TILING_MODE == IN_TRAINING_UPDATE_V2_NCHW_KEY) {
        if (tilingData.hasAffine != 0) {
            if (tilingData.hasRunning != 0) {
                INTrainingUpdateV2Nchw<DTYPE_X, true, true> op;
                op.Init(x, sum, square_sum, gamma, beta, mean, variance, y, batch_mean, batch_variance, &tilingData,
                        &pipe);
                op.Process();
            } else {
                INTrainingUpdateV2Nchw<DTYPE_X, true, false> op;
                op.Init(x, sum, square_sum, gamma, beta, mean, variance, y, batch_mean, batch_variance, &tilingData,
                        &pipe);
                op.Process();
            }
        } else if (tilingData.hasRunning != 0) {
            INTrainingUpdateV2Nchw<DTYPE_X, false, true> op;
            op.Init(x, sum, square_sum, gamma, beta, mean, variance, y, batch_mean, batch_variance, &tilingData, &pipe);
            op.Process();
        } else {
            INTrainingUpdateV2Nchw<DTYPE_X, false, false> op;
            op.Init(x, sum, square_sum, gamma, beta, mean, variance, y, batch_mean, batch_variance, &tilingData, &pipe);
            op.Process();
        }
    } else if constexpr (TILING_MODE == IN_TRAINING_UPDATE_V2_NHWC_KEY) {
        if (tilingData.hasAffine != 0) {
            if (tilingData.hasRunning != 0) {
                INTrainingUpdateV2Nhwc<DTYPE_X, true, true> op;
                op.Init(x, sum, square_sum, gamma, beta, mean, variance, y, batch_mean, batch_variance, &tilingData,
                        &pipe);
                op.Process();
            } else {
                INTrainingUpdateV2Nhwc<DTYPE_X, true, false> op;
                op.Init(x, sum, square_sum, gamma, beta, mean, variance, y, batch_mean, batch_variance, &tilingData,
                        &pipe);
                op.Process();
            }
        } else if (tilingData.hasRunning != 0) {
            INTrainingUpdateV2Nhwc<DTYPE_X, false, true> op;
            op.Init(x, sum, square_sum, gamma, beta, mean, variance, y, batch_mean, batch_variance, &tilingData, &pipe);
            op.Process();
        } else {
            INTrainingUpdateV2Nhwc<DTYPE_X, false, false> op;
            op.Init(x, sum, square_sum, gamma, beta, mean, variance, y, batch_mean, batch_variance, &tilingData, &pipe);
            op.Process();
        }
    }
    (void)workspace;
}

/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel_operator.h"
#include "arch35/bn_training_reduce.h"
#include "arch35/bn_training_reduce_empty.h"
#include "arch35/bn_training_reduce_small_r.h"
#include "arch35/bn_training_reduce_struct.h"
#include "arch35/bn_training_reduce_tiling_data.h"

template <bool templateType, bool isEmptyTensor, bool isTailR, bool isDeterministic>
__global__ __aicore__ void bn_training_reduce(GM_ADDR x, GM_ADDR sum, GM_ADDR squareSum, GM_ADDR workspace,
                                              GM_ADDR tiling)
{
    AscendC::InitSocState();
    REGISTER_TILING_DEFAULT(BNTrainingReduceTilingData);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GET_TILING_DATA_WITH_STRUCT(BNTrainingReduceTilingData, tilingData, tiling);
    if constexpr (isEmptyTensor && isTailR) {
        NsBNTrainingReduce::BNTrainingReduceSmallRKernel<DTYPE_X> op;
        op.Init(x, sum, squareSum, &tilingData);
        op.Process();
        (void)workspace;
    } else if constexpr (isEmptyTensor) {
        NsBNTrainingReduce::BNTrainingReduceEmpty<DTYPE_X> op;
        op.Init(sum, squareSum, &tilingData);
        for (int32_t outputIdx = 0; outputIdx < 2; ++outputIdx) {
            op.Process(outputIdx);
        }
        (void)x;
        (void)workspace;
    } else if constexpr (!templateType) {
        NsBNTrainingReduce::BNTrainingReduceKernel<DTYPE_X, isTailR, isDeterministic> op;
        op.Init(x, sum, squareSum, &tilingData);
        for (int32_t outputIdx = 0; outputIdx < 2; ++outputIdx) {
            op.Process(outputIdx);
        }
    } else if constexpr (templateType) {
        NsBNTrainingReduce::BNTrainingReduceKernel<DTYPE_X, isTailR, isDeterministic> op;
        op.InitGroup(x, sum, squareSum, workspace, &tilingData);
        for (int32_t outputIdx = 0; outputIdx < 2; ++outputIdx) {
            op.ProcessGroup(outputIdx);
            AscendC::SyncAll();
        }
    }
}

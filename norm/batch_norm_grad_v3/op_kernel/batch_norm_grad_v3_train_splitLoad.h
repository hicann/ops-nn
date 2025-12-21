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
 * \file batch_norm_grad_v3_train_splitLoad.h
 * \brief
 */

#ifndef BATCH_NORM_GRAD_V3_TRAIN_SPLITLOAD_H
#define BATCH_NORM_GRAD_V3_TRAIN_SPLITLOAD_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "kernel_utils.h"
#include "batch_norm_grad_v3_base_common.h"
#include "batch_norm_grad_v3_splitLoad_common.h"
#include "batch_norm_grad_v3_splitLoad_crosscore_common.h"

namespace BatchNormGradV3 {
using namespace AscendC;

template <typename T1, typename T2, typename T3, BNGV3SplitMode SPLIT_MODE = BNGV3SplitMode::R0_SPLIT_MODE>
class BatchNormGradV3TrainSplitLoad : public BatchNormGradV3SplitLoadCrossCoreBase<T1, T2, T3, SPLIT_MODE> {
public:
    __aicore__ inline BatchNormGradV3TrainSplitLoad(){};

    __aicore__ inline BatchNormGradV3TrainSplitLoad(const BatchNormGradV3SplitLoadTilingData* tilingDataIn)
    {
        tilingData_ = tilingDataIn;

        this->initSplitLoadTiling(tilingData_);
        this->b0UbBlock = tilingData_->b0UbBlock;
        this->bUbLoopMulti = tilingData_->bUbLoop;
        this->bUbBlockTail = tilingData_->bUbBlockTail;
        this->b0UbBlockTailMulti = tilingData_->b0UbBlockTail;
        meanDenominator = (float)1.0 / (this->b0Dim * this->b1Dim);
    }

    __aicore__ inline void Init(GMStruct gmStruct, TPipe* pipeIn)
    {
        this->InitSplitLoad(gmStruct, pipeIn);
        this->InitSplitLoadMode(gmStruct, pipeIn);
    }

    __aicore__ inline void Process()
    {
        this->blockIdx = GetBlockIdx();
        if (this->blockIdx > this->needCoreNum) {
            return;
        }

        ProcessHandle();
    }

    __aicore__ inline void ProcessHandle()
    {
        for (int64_t j = 0; j < this->totalChannel; j++) {
            int64_t offset = this->blockIdx * this->coreChannelNum;
            this->ProcessLoopCommon(offset, j);
            if constexpr (SPLIT_MODE == BNGV3SplitMode::R0_SPLIT_MODE) {
                computeR0SplitMode(j);
            } else {
                computeR1SplitMode(j);
            }

            this->ComputeAndCopyOutBiasAndWeight(offset, j);

            TPipeSetWaitFlag<HardEvent::V_S>();
            this->dyReduceValue = this->reduceUbTensor.GetValue(0) * meanDenominator;
            this->dyMulInputReduceValue = this->reduceInputUbTensor.GetValue(0);
            this->dyMulInputReduceValue = this->dyMulInputReduceValue * this->rVar * this->rVar / (this->b0Dim * this->b1Dim);
            if constexpr (SPLIT_MODE == BNGV3SplitMode::R0_SPLIT_MODE) {
                ComputeDxForR0(j);
            } else {
                ComputeDxForR1(j);
            }

            this->reduceQueue.template FreeTensor(this->reduceUbTensor);
            this->reduceInputQueue.template FreeTensor(this->reduceInputUbTensor);
        }
    }

private:
    __aicore__ inline void computeR0SplitMode(int64_t cIndex)
    {
        for (int64_t j = 0; j < this->b0Dim; j++) {
            for (int64_t i = 0; i < this->bUbLoopMulti; i++) {
                if (i == this->bUbLoopMulti - 1) {
                    this->cBlockLength = this->bUbBlockTail;
                } else {
                    this->cBlockLength = this->cUbBlock;
                }
                int64_t dyOffset = this->blockIdx * this->coreChannelNum * this->b1Dim + cIndex * this->b1Dim +
                                   j * this->aDim * this->b1Dim + i * this->cUbBlock;
                this->cBlockLengthT1Align = GetAlignValue<T1>(this->cBlockLength);
                this->CopyInXDyR0SplitMode(dyOffset);
                this->ComputeInLineTrain();
            }
        }
    }

    __aicore__ inline void computeR1SplitMode(int64_t cIndex)
    {
        for (int64_t i = 0; i < this->bUbLoopMulti; i++) {
            if (i == this->bUbLoopMulti - 1) {
                this->cBlockLength = this->b0UbBlockTailMulti * this->b1DimAlign;
                this->bUbR1Block = this->b0UbBlockTailMulti;
            } else {
                this->cBlockLength = this->b0UbBlock * this->b1DimAlign;
                this->bUbR1Block = this->b0UbBlock;
            }
            int64_t dyOffset = this->blockIdx * this->coreChannelNum * this->b1Dim + cIndex * this->b1Dim +
                               i * this->aDim * this->b1Dim * this->b0UbBlock;
            this->cBlockLengthT1Align = GetAlignValue<T1>(this->cBlockLength);
            this->CopyInXDyR1SplitMode(dyOffset);
            this->ComputeInLineTrain();
        }
    }

    __aicore__ inline void ComputeDxForR0(int64_t cIndex)
    {
        for (int64_t j = 0; j < this->b0Dim; j++) {
            for (int64_t i = 0; i < this->bUbLoopMulti; i++) {
                if (i == this->bUbLoopMulti - 1) {
                    this->cBlockLength = this->bUbBlockTail;
                } else {
                    this->cBlockLength = this->cUbBlock;
                }
                this->cBlockLengthT1Align = GetAlignValue<T1>(this->cBlockLength);
                int64_t dyOffset = this->blockIdx * this->coreChannelNum * this->b1Dim + cIndex * this->b1Dim +
                                   j * this->aDim * this->b1Dim + i * this->cUbBlock;
                this->CopyInXDyR0SplitMode(dyOffset);
                this->ComputeDxTrain();
                this->CopyOutDxR0SplitMode(dyOffset);
            }
        }
    }

    // 对R1场景，把数据全部拷出到workspace，然后顺序拷入计算会节省时间
    __aicore__ inline void ComputeDxForR1(int64_t cIndex)
    {
        for (int64_t i = 0; i < this->bUbLoopMulti; i++) {
            if (i == this->bUbLoopMulti - 1) {
                this->cBlockLength = this->b0UbBlockTailMulti * this->b1DimAlign;
                this->bUbR1Block = this->b0UbBlockTailMulti;
            } else {
                this->cBlockLength = this->b0UbBlock * this->b1DimAlign;
                this->bUbR1Block = this->b0UbBlock;
            }
            this->cBlockLengthT1Align = GetAlignValue<T1>(this->cBlockLength);
            int64_t dyOffset = this->blockIdx * this->coreChannelNum * this->b1Dim + cIndex * this->b1Dim +
                               i * this->aDim * this->b1Dim * this->b0UbBlock;
            this->CopyInXDyR1SplitMode(dyOffset);
            this->ComputeDxTrain();
            this->CopyOutDxR1SplitMode(dyOffset);
        }
    }
public:
    float meanDenominator;

private:
    const BatchNormGradV3SplitLoadTilingData* tilingData_;

    TPipe* pipe_;

    int64_t cBlockLengthAlign;
};
} // namespace BatchNormGradV3

#endif

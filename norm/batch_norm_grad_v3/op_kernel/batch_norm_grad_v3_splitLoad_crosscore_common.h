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
 * \file batch_norm_grad_v3_common.h
 * \brief
 */
#ifndef __BATCH_NORM_GRAD_V3_BASE_SPLITLOAD_CROSSCORE_H__
#define __BATCH_NORM_GRAD_V3_BASE_SPLITLOAD_CROSSCORE_H__

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "batch_norm_grad_v3_splitLoad_common.h"

namespace BatchNormGradV3 {
using namespace AscendC;

template <typename T1, typename T2, typename T3, BNGV3SplitMode SPLIT_MODE = BNGV3SplitMode::R0_SPLIT_MODE>
class BatchNormGradV3SplitLoadCrossCoreBase : public BatchNormGradV3SplitLoadBase<T1, T2, T3, SPLIT_MODE> {
public:
    __aicore__ inline BatchNormGradV3SplitLoadCrossCoreBase()
    {}

    __aicore__ inline void initSplitLoadCrossCoreTiling(
        const BatchNormGradV3SplitLoadCrossCoreTilingData* crossCoreTilingData_)
    {
        this->b0Dim = crossCoreTilingData_->b0Dim;
        this->b1Dim = crossCoreTilingData_->b1Dim;
        this->aDim = crossCoreTilingData_->aDim;
        this->bAlign = crossCoreTilingData_->bAlign;
        this->coreChannelNum = crossCoreTilingData_->coreChannelNum;
        this->coreChannelNumTail = crossCoreTilingData_->coreChannelNumTail;
        this->cUbBlock = crossCoreTilingData_->bUbBlock;
        this->b1DimAlign = GetAlignValue<T1>(this->b1Dim);
        this->bAlign = this->b1DimAlign * this->b0Dim;
        this->needCoreNum = crossCoreTilingData_->needCoreNum;
        this->epsilon = crossCoreTilingData_->epsilon;

        this->blockIdx = GetBlockIdx();
        if (this->blockIdx < this->needCoreNum) {
            this->totalChannel = this->coreChannelNum;
        }

        this->totalChannel = this->coreChannelNum;
        this->b0UbBlock = crossCoreTilingData_->b0UbBlock;
        this->bUbLoop = crossCoreTilingData_->bUbLoop;
        this->bUbBlockTail = crossCoreTilingData_->bUbBlockTail;
        this->b0UbBlockTail = crossCoreTilingData_->b0UbBlockTail;

        this->groupCore = crossCoreTilingData_->groupCore;
        this->groupBlockNum = crossCoreTilingData_->groupBlockNum;
        this->groupBlockNumTail = crossCoreTilingData_->groupBlockNumTail;
        this->morehalfChannel = crossCoreTilingData_->morehalfChannel;
        this->groupBlockNumHalf = crossCoreTilingData_->groupBlockNumHalf;
        this->groupBlockNumHalfTail = crossCoreTilingData_->groupBlockNumHalfTail;
        this->coreNum = crossCoreTilingData_->coreNum;
        this->bUbTailLoop = crossCoreTilingData_->bUbTailLoop;
        this->b0UbTailBlockTail = crossCoreTilingData_->b0UbTailBlockTail;
        this->bUbLoopHalf = crossCoreTilingData_->bUbLoopHalf;
        this->b0UbBlockTailHalf = crossCoreTilingData_->b0UbBlockTailHalf;
        this->bUbTailLoopHalf = crossCoreTilingData_->bUbTailLoopHalf;
        this->b0UbTailBlockTailHalf = crossCoreTilingData_->b0UbTailBlockTailHalf;
        this->moreMultiChannel = crossCoreTilingData_->moreMultiChannel;
        this->bUbLoopMulti = crossCoreTilingData_->bUbLoopMulti;
        this->b0UbBlockTailMulti = crossCoreTilingData_->b0UbBlockTailMulti;

        this->wsReduceOffset = this->blockIdx * B32_BLOCK_ALIGN_NUM;
        if (this->morehalfChannel || this->moreMultiChannel) {
            this->wsReduceInputOffset = this->coreNum * B32_BLOCK_ALIGN_NUM + this->blockIdx * B32_BLOCK_ALIGN_NUM;
        } else {
            this->wsReduceInputOffset = this->needCoreNum * B32_BLOCK_ALIGN_NUM + this->blockIdx * B32_BLOCK_ALIGN_NUM;
        }
    }

    __aicore__ inline void CopyInWeightMeanVarCrossCore(int64_t cIndex)
    {
        if constexpr (IsSameType<T2, float>::value || IsSameType<T2, half>::value) {
            this->weight = this->weightGm_.GetValue(cIndex);
        } else {
            this->weight = ToFloat(this->weightGm_.GetValue(cIndex));
        }

        if constexpr (IsSameType<T3, float>::value || IsSameType<T3, half>::value) {
            this->mean = this->meanGm_.GetValue(cIndex);
            this->var = this->varGm_.GetValue(cIndex);
        } else {
            this->mean = ToFloat(this->meanGm_.GetValue(cIndex));
            this->var = ToFloat(this->varGm_.GetValue(cIndex));
        }
        float tmpVar = this->var + this->epsilon;
        tmpVar = sqrt(tmpVar);
        this->rVar = tmpVar == 0 ? tmpVar : 1 / tmpVar;
    }

    __aicore__ inline void GetCrossCoreR1Param(int64_t coreIndex, bool morehalfChannel)
    {
        if (morehalfChannel) {
            this->usedBUbLoop = this->bUbLoopHalf;
            if (coreIndex == this->usedGroupCore - 1) {
                this->usedBUbLoop = this->bUbTailLoopHalf;
            }
            this->usedB0UbBlockTail = this->b0UbBlockTailHalf;
            this->usedB0UbTailBlockTail = this->b0UbTailBlockTailHalf;
        } else {
            this->usedBUbLoop = this->bUbLoop;
            if (coreIndex == this->usedGroupCore - 1) {
                this->usedBUbLoop = this->bUbTailLoop;
            }
            this->usedB0UbBlockTail = this->b0UbBlockTail;
            this->usedB0UbTailBlockTail = this->b0UbTailBlockTail;
        }
    }

    __aicore__ inline void ComputeDBiasForCrossCore()
    {
        PipeBarrier<PIPE_V>();
        ReduceSum(this->reduceUbTensor, this->reduceUbTensor, this->reduceUbTensor, this->cUbBlock);
        PipeBarrier<PIPE_V>();
        this->reduceQueue.EnQue(this->reduceUbTensor);
        this->reduceUbTensor = this->reduceQueue.template DeQue<float>();
        PipeBarrier<PIPE_MTE3>();
        DataCopyExtParams copyInParams;
        copyInParams.blockCount = 1;
        copyInParams.blockLen = 1 * sizeof(float);
        copyInParams.srcStride = 0;
        copyInParams.dstStride = 0;
        DataCopyPad(this->workSpaceGm_[this->wsReduceOffset], this->reduceUbTensor, copyInParams);
    }

    __aicore__ inline void ComputeDWeightForCrossCore()
    {
        PipeBarrier<PIPE_V>();
        ReduceSum(this->reduceInputUbTensor, this->reduceInputUbTensor, this->reduceInputUbTensor, this->cUbBlock);
        PipeBarrier<PIPE_V>();
        this->reduceInputQueue.EnQue(this->reduceInputUbTensor);

        this->reduceInputUbTensor = this->reduceInputQueue.template DeQue<float>();
        PipeBarrier<PIPE_MTE3>();
        DataCopyExtParams copyInParams;
        copyInParams.blockCount = 1;
        copyInParams.blockLen = 1 * sizeof(float);
        copyInParams.srcStride = 0;
        copyInParams.dstStride = 0;
        DataCopyPad(this->workSpaceGm_[this->wsReduceInputOffset], this->reduceInputUbTensor, copyInParams);
    }
};
} // namespace BatchNormGradV3
#endif

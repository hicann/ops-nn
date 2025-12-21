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

#ifndef BATCH_NORM_GRAD_V3_TRAIN_SPLITLOAD_CROSSCORE_H
#define BATCH_NORM_GRAD_V3_TRAIN_SPLITLOAD_CROSSCORE_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "kernel_utils.h"
#include "batch_norm_grad_v3_base_common.h"
#include "batch_norm_grad_v3_splitLoad_common.h"
#include "batch_norm_grad_v3_train_splitLoad.h"

namespace BatchNormGradV3 {
using namespace AscendC;

template <typename T1, typename T2, typename T3, BNGV3SplitMode SPLIT_MODE = BNGV3SplitMode::R0_SPLIT_MODE>
class BatchNormGradV3TrainSplitLoadCrossCore : public BatchNormGradV3TrainSplitLoad<T1, T2, T3, SPLIT_MODE> {
public:
    __aicore__ inline BatchNormGradV3TrainSplitLoadCrossCore(){};

    __aicore__ inline BatchNormGradV3TrainSplitLoadCrossCore(
        const BatchNormGradV3SplitLoadCrossCoreTilingData* tilingDataIn)
    {
        tilingData_ = tilingDataIn;

        this->initSplitLoadCrossCoreTiling(tilingData_);
        this->meanDenominator = (float)1.0 / (this->b0Dim * this->b1Dim);
    }

    __aicore__ inline void Init(GMStruct gmStruct, TPipe* pipeIn)
    {
        this->InitSplitLoad(gmStruct, pipeIn);
        this->InitSplitLoadMode(gmStruct, pipeIn);
    }

    __aicore__ inline void Process()
    {
        this->blockIdx = GetBlockIdx();
        if (this->moreMultiChannel) {
            this->ProcessHandle();
        }

        if (this->morehalfChannel) {
            ProcessPreChannel(true);
            SyncAll();
        }
        if (this->blockIdx >= this->needCoreNum) {
            SyncAll();
            return;
        }
        ProcessPreChannel(false);
    }

    __aicore__ inline void ProcessPreChannel(bool morehalfChannel)
    {
        this->usedGroupBlockNum = morehalfChannel ? this->groupBlockNumHalf : this->groupBlockNum;
        this->usedGroupBlockNumTail = morehalfChannel ? this->groupBlockNumHalfTail : this->groupBlockNumTail;
        this->usedGroupCore = morehalfChannel ? TWO : this->groupCore;
        int64_t cIndex = this->blockIdx / this->usedGroupCore; // 计算当前核计算的C坐标

        if (this->moreMultiChannel) {
            cIndex += this->coreNum * this->coreChannelNum;
        }

        if (!morehalfChannel && this->morehalfChannel) {
            cIndex += this->coreNum / TWO;
        }
        int64_t coreIndex = this->blockIdx % this->usedGroupCore; // 计算核计算C中索引
        wsReduceStartOffset = this->wsReduceOffset - coreIndex * B32_BLOCK_ALIGN_NUM;
        wsReduceInputStartOffset = this->wsReduceInputOffset - coreIndex * B32_BLOCK_ALIGN_NUM;
        this->CopyInWeightMeanVarCrossCore(cIndex);

        int64_t offset = this->blockIdx * this->coreChannelNum;
        this->reduceUbTensor = this->reduceQueue.template AllocTensor<float>();
        this->reduceInputUbTensor = this->reduceInputQueue.template AllocTensor<float>();
        Duplicate(this->reduceUbTensor, static_cast<float>(0), this->cUbBlock);
        Duplicate(this->reduceInputUbTensor, static_cast<float>(0), this->cUbBlock);
        if constexpr (SPLIT_MODE == BNGV3SplitMode::R0_SPLIT_MODE) {
            computeR0SplitMode(cIndex, coreIndex);
        } else {
            this->GetCrossCoreR1Param(coreIndex, morehalfChannel);
            computeR1SplitMode(cIndex, coreIndex);
        }
        this->ComputeAndCopyOutBiasAndWeight(cIndex, coreIndex);
        ComputeAndCopyOutDx(cIndex, coreIndex);

        this->reduceQueue.template FreeTensor(this->reduceUbTensor);
        this->reduceInputQueue.template FreeTensor(this->reduceInputUbTensor);
    }

private:
    __aicore__ inline void computeR0SplitMode(int64_t cIndex, int64_t coreIndex)
    {
        int64_t tmpBlockNumCopyIn = this->usedGroupBlockNum;
        if (coreIndex == this->usedGroupCore - 1) {
            tmpBlockNumCopyIn = this->usedGroupBlockNumTail;
        }
        // 循环执行block数量次
        for (int64_t i = 0; i < tmpBlockNumCopyIn; i++) {
            int64_t nIndexCopyIn = (coreIndex * this->usedGroupBlockNum + i) / this->bUbLoop;  // 计算的N索引
            int64_t hwIndexCopyIn = (coreIndex * this->usedGroupBlockNum + i) % this->bUbLoop; // 计算的H和W索引

            // 找到offset
            int64_t dyOffset =
                cIndex * this->b1Dim + nIndexCopyIn * this->aDim * this->b1Dim + hwIndexCopyIn * this->cUbBlock;

            if (hwIndexCopyIn != this->bUbLoop - 1) {
                this->cBlockLength = this->cUbBlock;
            } else {
                this->cBlockLength = this->bUbBlockTail;
            }

            this->cBlockLengthT1Align = GetAlignValue<T1>(this->cBlockLength);
            this->CopyInXDyR0SplitMode(dyOffset);
            this->ComputeInLineTrain();
        }
    }

    __aicore__ inline void computeR1SplitMode(int64_t cIndex, int64_t coreIndex)
    {
        for (int64_t i = 0; i < this->usedBUbLoop; i++) {
            if (i != this->usedBUbLoop - 1) {
                this->cBlockLength = this->b0UbBlock * this->b1DimAlign;
                this->bUbR1Block = this->b0UbBlock;
            } else if (coreIndex != this->usedGroupCore - 1) {
                this->cBlockLength = this->usedB0UbBlockTail * this->b1DimAlign;
                this->bUbR1Block = this->usedB0UbBlockTail;
            } else {
                this->bUbR1Block = this->usedB0UbTailBlockTail;
                this->cBlockLength = this->usedB0UbTailBlockTail * this->b1DimAlign;
            }
            this->cBlockLengthT1Align = GetAlignValue<T1>(this->cBlockLength);
            int64_t dyOffset = cIndex * this->b1Dim + coreIndex * this->usedGroupBlockNum * this->aDim * this->b1Dim +
                               i * this->b0UbBlock * this->aDim * this->b1Dim;

            this->CopyInXDyR1SplitMode(dyOffset);
            this->ComputeInLineTrain();
        }
    }

    __aicore__ inline void ComputeDxForR0(int64_t cIndex, int64_t coreIndex)
    {
        int64_t tmpBlockNum = this->usedGroupBlockNum;
        if (coreIndex == this->usedGroupCore - 1) {
            tmpBlockNum = this->usedGroupBlockNumTail;
        }
        // 循环执行block数量次
        for (int64_t i = 0; i < tmpBlockNum; i++) {
            int64_t nIndex = (coreIndex * this->usedGroupBlockNum + i) / this->bUbLoop;  // 计算的N索引
            int64_t hwIndex = (coreIndex * this->usedGroupBlockNum + i) % this->bUbLoop; // 计算的H和W索引

            // 找到offset
            int64_t dyOffset = cIndex * this->b1Dim + nIndex * this->aDim * this->b1Dim + hwIndex * this->cUbBlock;

            if (hwIndex != this->bUbLoop - 1) {
                this->cBlockLength = this->cUbBlock;
            } else {
                this->cBlockLength = this->bUbBlockTail;
            }
            this->cBlockLengthT1Align = GetAlignValue<T1>(this->cBlockLength);
            this->CopyInXDyR0SplitMode(dyOffset);
            this->ComputeDxTrain();
            this->CopyOutDxR0SplitMode(dyOffset);
        }
    }

    // 对R1场景，把数据全部拷出到workspace，然后顺序拷入计算会节省时间
    __aicore__ inline void ComputeDxForR1(int64_t cIndex, int64_t coreIndex)
    {
        for (int64_t i = 0; i < this->usedBUbLoop; i++) {
            if (i != this->usedBUbLoop - 1) {
                this->cBlockLength = this->b0UbBlock * this->b1DimAlign;
                this->bUbR1Block = this->b0UbBlock;
            } else if (coreIndex != this->usedGroupCore - 1) {
                this->cBlockLength = this->usedB0UbBlockTail * this->b1DimAlign;
                this->bUbR1Block = this->usedB0UbBlockTail;
            } else {
                this->cBlockLength = this->usedB0UbTailBlockTail * this->b1DimAlign;
                this->bUbR1Block = this->usedB0UbTailBlockTail;
            }
            int64_t dyOffset = cIndex * this->b1Dim + coreIndex * this->usedGroupBlockNum * this->aDim * this->b1Dim +
                               i * this->b0UbBlock * this->aDim * this->b1Dim;

            this->cBlockLengthT1Align = GetAlignValue<T1>(this->cBlockLength);

            this->CopyInXDyR1SplitMode(dyOffset);
            this->ComputeDxTrain();
            this->CopyOutDxR1SplitMode(dyOffset);
        }
    }

    __aicore__ inline void ComputeCrossCoreDBias(int64_t cIndex, int64_t coreIndex)
    {
        TPipeSetWaitFlag<HardEvent::MTE3_MTE2>();
        DataCopyExtParams extParam;
        extParam.blockCount = 1;
        extParam.blockLen = B32_BLOCK_ALIGN_NUM * this->usedGroupCore * sizeof(float);
        DataCopyPadExtParams<float> padExtParam;
        padExtParam.isPad = false;
        LoadOneTensor(
            this->workSpaceGm_[wsReduceStartOffset], this->reduceUbTensor, extParam, padExtParam,
            this->usedGroupCore * B32_BLOCK_ALIGN_NUM);
        float sum = 0.0;
        TPipeSetWaitFlag<HardEvent::MTE2_S>();

        for (int16_t i = 0; i < this->usedGroupCore; i++) {
            float tmp = this->reduceUbTensor.GetValue(i * B32_BLOCK_ALIGN_NUM);
            sum += tmp;
        }

        this->reduceUbTensor.SetValue(0, sum);
        if (coreIndex == 0) {
            TPipeSetWaitFlag<HardEvent::S_MTE3>();
            DataCopyExtParams copyInParams;
            copyInParams.blockCount = 1;
            copyInParams.blockLen = 1 * sizeof(float);
            copyInParams.srcStride = 0;
            copyInParams.dstStride = 0;
            StoreOneTensor(this->dBiasGm_[cIndex], this->reduceUbTensor, copyInParams, 1);
        }
    }

    __aicore__ inline void ComputeCrossCoreDWeight(int64_t cIndex, int64_t coreIndex)
    {
        TPipeSetWaitFlag<HardEvent::MTE3_MTE2>();
        DataCopyExtParams extParam;
        extParam.blockCount = 1;
        extParam.blockLen = B32_BLOCK_ALIGN_NUM * this->usedGroupCore * sizeof(float);
        DataCopyPadExtParams<float> padExtParam;
        padExtParam.isPad = false;
        LoadOneTensor(
            this->workSpaceGm_[wsReduceInputStartOffset], this->reduceInputUbTensor, extParam, padExtParam,
            this->usedGroupCore * B32_BLOCK_ALIGN_NUM);
        TPipeSetWaitFlag<HardEvent::MTE2_S>();
        float sum = 0.0;
        for (int16_t i = 0; i < this->usedGroupCore; i++) {
            float tmp = this->reduceInputUbTensor.GetValue(i * B32_BLOCK_ALIGN_NUM);
            sum += tmp;
        }
        sum = sum * this->rVar;
        this->reduceInputUbTensor.SetValue(0, sum);

        if (coreIndex == 0) {
            TPipeSetWaitFlag<HardEvent::S_MTE3>();
            DataCopyExtParams copyInParams;
            copyInParams.blockCount = 1;
            copyInParams.blockLen = 1 * sizeof(float);
            copyInParams.srcStride = 0;
            copyInParams.dstStride = 0;
            StoreOneTensor(this->dWeightGm_[cIndex], this->reduceInputUbTensor, copyInParams, 1);
        }
    }

    __aicore__ inline void ComputeAndCopyOutBiasAndWeight(int64_t cIndex, int64_t coreIndex)
    {
        this->ComputeDBiasForCrossCore();
        this->ComputeDWeightForCrossCore();
        SyncAll();

        ComputeCrossCoreDBias(cIndex, coreIndex);
        ComputeCrossCoreDWeight(cIndex, coreIndex);
    }

    __aicore__ inline void ComputeAndCopyOutDx(int64_t cIndex, int64_t coreIndex)
    {
        this->dyReduceValue = this->reduceUbTensor.GetValue(0) * this->meanDenominator;
        this->dyMulInputReduceValue = this->reduceInputUbTensor.GetValue(0);
        this->dyMulInputReduceValue = this->dyMulInputReduceValue * this->rVar / (this->b0Dim * this->b1Dim);

        if constexpr (SPLIT_MODE == BNGV3SplitMode::R0_SPLIT_MODE) {
            ComputeDxForR0(cIndex, coreIndex);
        } else {
            ComputeDxForR1(cIndex, coreIndex);
        }
    }

private:
    const BatchNormGradV3SplitLoadCrossCoreTilingData* tilingData_;

    TPipe* pipe_;

    int64_t cBlockLengthAlign;

    int64_t wsReduceStartOffset;
    int64_t wsReduceInputStartOffset;
};
} // namespace BatchNormGradV3

#endif

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
 * \file batch_norm_grad_v3_infer_splitLoad.h
 * \brief
 */

#ifndef BATCH_NORM_GRAD_V3_INFER_SPLITLOAD_CROSSCORE_H
#define BATCH_NORM_GRAD_V3_INFER_SPLITLOAD_CROSSCORE_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "kernel_utils.h"
#include "batch_norm_grad_v3_base_common.h"
#include "batch_norm_grad_v3_splitLoad_common.h"
#include "batch_norm_grad_v3_infer_splitLoad.h"

namespace BatchNormGradV3 {
using namespace AscendC;

template <typename T1, typename T2, typename T3, BNGV3SplitMode SPLIT_MODE = BNGV3SplitMode::R0_SPLIT_MODE>
class BatchNormGradV3InferSplitLoadCrossCore : public BatchNormGradV3InferSplitLoad<T1, T2, T3, SPLIT_MODE> {
public:
    __aicore__ inline BatchNormGradV3InferSplitLoadCrossCore(){};

    __aicore__ inline BatchNormGradV3InferSplitLoadCrossCore(
        const BatchNormGradV3SplitLoadCrossCoreTilingData* tilingDataIn)
    {
        tilingData_ = tilingDataIn;

        this->initSplitLoadCrossCoreTiling(tilingData_);
    }

    __aicore__ inline void Process()
    {
        this->blockIdx = GetBlockIdx();
        if (this->moreMultiChannel) {
            this->ProcessHandle();
        }

        if (this->morehalfChannel) {
            ProcessPreChannelInfer(true);
            SyncAll();
        }
        if (this->blockIdx >= this->needCoreNum) {
            SyncAll();
            return;
        }
        ProcessPreChannelInfer(false);
    }

    __aicore__ inline void Init(GMStruct gmStruct, TPipe* pipeIn)
    {
        this->InitSplitLoad(gmStruct, pipeIn);
        this->InitSplitLoadMode(gmStruct, pipeIn);
    }

    __aicore__ inline void ProcessPreChannelInfer(bool morehalfChannel)
    {
        this->usedGroupCore = morehalfChannel ? TWO : this->groupCore;
        this->usedGroupBlockNum = morehalfChannel ? this->groupBlockNumHalf : this->groupBlockNum;
        this->usedGroupBlockNumTail = morehalfChannel ? this->groupBlockNumHalfTail : this->groupBlockNumTail;
        int64_t cIndex = this->blockIdx / this->usedGroupCore; // 计算当前核计算的C坐标
        if (this->moreMultiChannel) {
            cIndex += this->coreNum * this->coreChannelNum;
        }
        if (!morehalfChannel && this->morehalfChannel) {
            cIndex += this->coreNum / TWO;
        }
        int64_t coreIndex = this->blockIdx % this->usedGroupCore; // 计算核计算C中索引
        this->CopyInWeightMeanVarCrossCore(cIndex);
        this->reduceUbTensor = this->reduceQueue.template AllocTensor<float>();
        this->reduceInputUbTensor = this->reduceInputQueue.template AllocTensor<float>();
        Duplicate(this->reduceUbTensor, static_cast<float>(0), this->cUbBlock);
        Duplicate(this->reduceInputUbTensor, static_cast<float>(0), this->cUbBlock);

        if constexpr (SPLIT_MODE == BNGV3SplitMode::R0_SPLIT_MODE) {
            ComputeR0SplitMode(cIndex, coreIndex);
        } else {
            this->GetCrossCoreR1Param(coreIndex, morehalfChannel);
            computeR1SplitMode(cIndex, coreIndex);
        }
        this->ComputeAndCopyOutBiasAndWeight(cIndex, coreIndex);

        this->reduceInputQueue.template FreeTensor(this->reduceInputUbTensor);
        this->reduceQueue.template FreeTensor(this->reduceUbTensor);
    }

private:
    __aicore__ inline void computeR1SplitMode(int64_t cIndex, int64_t coreIndex)
    {
        for (int64_t i = 0; i < this->usedBUbLoop; i++) {
            if (i != this->usedBUbLoop - 1) {
                this->bUbR1Block = this->b0UbBlock;
                this->cBlockLength = this->b0UbBlock * this->b1DimAlign;
            } else if (coreIndex != this->usedGroupCore - 1) {
                this->bUbR1Block = this->usedB0UbBlockTail;
                this->cBlockLength = this->usedB0UbBlockTail * this->b1DimAlign;
            } else {
                this->bUbR1Block = this->usedB0UbTailBlockTail;
                this->cBlockLength = this->usedB0UbTailBlockTail * this->b1DimAlign;
            }
            int64_t dyOffset = cIndex * this->b1Dim + coreIndex * this->usedGroupBlockNum * this->aDim * this->b1Dim +
                               i * this->b0UbBlock * this->aDim * this->b1Dim;
            this->cBlockLengthT1Align = GetAlignValue<T1>(this->cBlockLength);

            this->CopyInXDyR1SplitMode(dyOffset);
            this->ComputeInLineInfer();
            this->CopyOutDxR1SplitMode(dyOffset);
        }
    }

    __aicore__ inline void ComputeR0SplitMode(int64_t cIndex, int64_t coreIndex)
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
            int64_t dyOffsetInfer = cIndex * this->b1Dim + nIndex * this->aDim * this->b1Dim + hwIndex * this->cUbBlock;

            if (hwIndex != this->bUbLoop - 1) {
                this->cBlockLength = this->cUbBlock;
            } else {
                this->cBlockLength = this->bUbBlockTail;
            }
            this->cBlockLengthT1Align = GetAlignValue<T1>(this->cBlockLength);
            this->CopyInXDyR0SplitMode(dyOffsetInfer);
            this->ComputeInLineInfer();
            this->CopyOutDxR0SplitMode(dyOffsetInfer);
        }
    }

    __aicore__ inline void ComputeCrossCoreDBias(int64_t cIndex)
    {
        TPipeSetWaitFlag<HardEvent::MTE3_MTE2>();
        DataCopyExtParams extParam;
        extParam.blockCount = 1;
        extParam.blockLen = B32_BLOCK_ALIGN_NUM * this->usedGroupCore * sizeof(float);
        DataCopyPadExtParams<float> padExtParam;
        padExtParam.isPad = false;
        LoadOneTensor(
            this->workSpaceGm_[this->wsReduceOffset], this->reduceUbTensor, extParam, padExtParam,
            this->usedGroupCore * B32_BLOCK_ALIGN_NUM);
        float sum = 0.0;
        TPipeSetWaitFlag<HardEvent::MTE2_S>();
        for (int16_t i = 0; i < this->usedGroupCore; i++) {
            float tmp = this->reduceUbTensor.GetValue(i * B32_BLOCK_ALIGN_NUM);
            sum += tmp;
        }

        this->reduceUbTensor.SetValue(0, sum);
        TPipeSetWaitFlag<HardEvent::S_MTE3>();
        DataCopyExtParams copyInParams;
        copyInParams.blockCount = 1;
        copyInParams.blockLen = 1 * sizeof(float);
        copyInParams.srcStride = 0;
        copyInParams.dstStride = 0;
        StoreOneTensor(this->dBiasGm_[cIndex], this->reduceUbTensor, copyInParams, 1);
    }

    __aicore__ inline void ComputeCrossCoreDWeight(int64_t cIndex)
    {
        TPipeSetWaitFlag<HardEvent::MTE3_MTE2>();
        DataCopyExtParams extParam;
        extParam.blockCount = 1;
        extParam.blockLen = B32_BLOCK_ALIGN_NUM * this->usedGroupCore * sizeof(float);
        DataCopyPadExtParams<float> padExtParam;
        padExtParam.isPad = false;
        LoadOneTensor(
            this->workSpaceGm_[this->wsReduceInputOffset], this->reduceInputUbTensor, extParam, padExtParam,
            this->usedGroupCore * B32_BLOCK_ALIGN_NUM);
        TPipeSetWaitFlag<HardEvent::MTE2_S>();
        float sum = 0.0;
        for (int16_t i = 0; i < this->usedGroupCore; i++) {
            float tmp = this->reduceInputUbTensor.GetValue(i * B32_BLOCK_ALIGN_NUM);
            sum += tmp;
        }
        sum = sum * this->rVar;
        this->reduceInputUbTensor.SetValue(0, sum);

        TPipeSetWaitFlag<HardEvent::S_MTE3>();
        DataCopyExtParams copyInParams;
        copyInParams.blockCount = 1;
        copyInParams.blockLen = 1 * sizeof(float);
        copyInParams.srcStride = 0;
        copyInParams.dstStride = 0;
        StoreOneTensor(this->dWeightGm_[cIndex], this->reduceInputUbTensor, copyInParams, 1);
    }

    __aicore__ inline void ComputeAndCopyOutBiasAndWeight(int64_t cIndex, int64_t coreIndex)
    {
        this->ComputeDBiasForCrossCore();
        this->ComputeDWeightForCrossCore();
        SyncAll();
        if (coreIndex == 0) {
            ComputeCrossCoreDBias(cIndex);
            ComputeCrossCoreDWeight(cIndex);
        }
    }

private:
    const BatchNormGradV3SplitLoadCrossCoreTilingData* tilingData_;

    TPipe* pipe_;
};
} // namespace BatchNormGradV3

#endif

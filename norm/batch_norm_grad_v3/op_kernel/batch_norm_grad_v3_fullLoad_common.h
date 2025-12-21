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
#ifndef __BATCH_NORM_GRAD_V3_BASE_FULLLOAD_H__
#define __BATCH_NORM_GRAD_V3_BASE_FULLLOAD_H__

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"

namespace BatchNormGradV3 {
using namespace AscendC;

template <typename T1, typename T2, typename T3>
class BatchNormGradV3FullLoadBase : public BatchNormGradV3Base<T1, T2, T3> {
public:
    __aicore__ inline BatchNormGradV3FullLoadBase()
    {}

protected:
    TQue<QuePosition::VECIN, BUFFER_DEPTH> dyQueue_;
    TQue<QuePosition::VECIN, BUFFER_DEPTH> inputQueue;
    TQue<QuePosition::VECIN, BUFFER_DEPTH> meanQueue;
    TBuf<QuePosition::VECCALC> meanBrcbQueue;
    TQue<QuePosition::VECIN, BUFFER_DEPTH> varQueue;
    TBuf<QuePosition::VECCALC> varBrcbQueue;
    TQue<QuePosition::VECOUT, BUFFER_DEPTH> reduceQueue;
    TQue<QuePosition::VECIN, BUFFER_DEPTH> weightQueue;
    TBuf<QuePosition::VECCALC> weightBrcbQueue;

    LocalTensor<float> dyTensor;
    LocalTensor<float> xTensor;
    LocalTensor<float> weightTensor;
    LocalTensor<float> weightBrcbTensor;
    LocalTensor<float> varTensor;
    LocalTensor<float> varBrcbTensor;
    LocalTensor<float> meanTensor;
    LocalTensor<float> meanBrcbTensor;
    LocalTensor<float> reduceUbTensor;

protected:
    __aicore__ inline void initFullLoadTiling(const BatchNormGradV3FullLoadTilingData* tilingData_)
    {
        this->b1Dim = tilingData_->b1Dim;
        this->aDim = tilingData_->aDim;
        this->b0Dim = tilingData_->b0Dim;
        this->bAlign = tilingData_->bAlign;
        this->coreChannelNum = tilingData_->coreChannelNum;
        this->coreChannelNumTail = tilingData_->coreChannelNumTail;
        this->cUbBlock = tilingData_->cUbBlock;
        this->needCoreNum = tilingData_->needCoreNum;
        this->epsilon = tilingData_->epsilon;

        this->b1DimAlign = GetAlignValue<T1>(this->b1Dim);
        this->bAlign = this->b1DimAlign * this->b0Dim;
    }

    __aicore__ inline void InitFullLoad(GMStruct gmStruct, TPipe* pipe_)
    {
        this->dyGm_.SetGlobalBuffer((__gm__ T1*)gmStruct.dy);
        this->xGm_.SetGlobalBuffer((__gm__ T1*)gmStruct.x);
        this->weightGm_.SetGlobalBuffer((__gm__ T2*)gmStruct.weight);
        this->varGm_.SetGlobalBuffer((__gm__ T3*)gmStruct.var);
        this->meanGm_.SetGlobalBuffer((__gm__ T3*)gmStruct.mean);
        this->dxGm_.SetGlobalBuffer((__gm__ T1*)gmStruct.dx);
        this->dWeightGm_.SetGlobalBuffer((__gm__ T2*)gmStruct.dWeight);
        this->dBiasGm_.SetGlobalBuffer((__gm__ T2*)gmStruct.dBias);

        int64_t xShapeLen = this->cUbBlock * this->bAlign;
        pipe_->InitBuffer(dyQueue_, BUFFER_NUM, xShapeLen * sizeof(float));
        pipe_->InitBuffer(reduceQueue, BUFFER_NUM, xShapeLen * sizeof(float));
        pipe_->InitBuffer(inputQueue, BUFFER_NUM, xShapeLen * sizeof(float));

        int64_t tmpCBlock = GetAlignValue<T2>(this->cUbBlock);
        pipe_->InitBuffer(meanQueue, BUFFER_NUM, tmpCBlock * sizeof(float));
        pipe_->InitBuffer(meanBrcbQueue, tmpCBlock * B32_BLOCK_ALIGN_NUM * sizeof(float));
        pipe_->InitBuffer(varQueue, BUFFER_NUM, tmpCBlock * sizeof(float));
        pipe_->InitBuffer(varBrcbQueue, tmpCBlock * B32_BLOCK_ALIGN_NUM * sizeof(float));
        pipe_->InitBuffer(weightQueue, BUFFER_NUM, tmpCBlock * sizeof(float));
        pipe_->InitBuffer(weightBrcbQueue, tmpCBlock * B32_BLOCK_ALIGN_NUM * sizeof(float));
    }

    __aicore__ inline void CopyInXDy(int64_t dyGmOffset)
    {
        LocalTensor<float> dyLocal = dyQueue_.AllocTensor<float>();
        LocalTensor<float> xLocal = inputQueue.AllocTensor<float>();
        DataCopyExtParams copyInParams;
        copyInParams.blockCount = this->b0Dim;
        copyInParams.blockLen = this->b1Dim * sizeof(T1);
        copyInParams.srcStride = this->b1Dim * (this->aDim - 1) * sizeof(T1);
        copyInParams.dstStride = 0;
        DataCopyPadExtParams<T1> dataCopyPadExtParams;
        dataCopyPadExtParams.isPad = true;
        dataCopyPadExtParams.leftPadding = 0;
        dataCopyPadExtParams.rightPadding = this->b1DimAlign - this->b1Dim;
        if constexpr (IsSameType<T1, bfloat16_t>::value) {
            dataCopyPadExtParams.paddingValue = ToBfloat16(0);
        } else {
            dataCopyPadExtParams.paddingValue = (T1)0;
        }
        for (int32_t i = 0; i < this->cBlockLength; i++) {
            int64_t dyGmOffsetForLoop = dyGmOffset + i * this->b1Dim;
            int64_t dyLocalOffset = i * this->b0Dim * this->b1DimAlign;
            LoadOneTensor(
                this->dyGm_[dyGmOffsetForLoop], dyLocal[dyLocalOffset], copyInParams, dataCopyPadExtParams,
                this->b0Dim * this->b1DimAlign);
            LoadOneTensor(
                this->xGm_[dyGmOffsetForLoop], xLocal[dyLocalOffset], copyInParams, dataCopyPadExtParams,
                this->b0Dim * this->b1DimAlign);
        }
        dyQueue_.EnQue(dyLocal);
        inputQueue.EnQue(xLocal);
    }

    __aicore__ inline void CopyInWeightMeanVar(int64_t offset)
    {
        LocalTensor<float> weightLocal = weightQueue.AllocTensor<float>();
        LocalTensor<float> meanLocal = meanQueue.AllocTensor<float>();
        LocalTensor<float> varLocal = varQueue.AllocTensor<float>();

        DataCopyExtParams extParam;
        extParam.blockCount = 1;
        extParam.blockLen = this->cBlockLength * sizeof(T2);
        DataCopyPadExtParams<T2> padExtParam;
        padExtParam.isPad = false;
        LoadOneTensor(this->weightGm_[offset], weightLocal, extParam, padExtParam, this->cBlockLengthT2Align);

        // var
        extParam.blockLen = this->cBlockLength * sizeof(T3);
        DataCopyPadExtParams<T3> padExtParam2;
        padExtParam2.isPad = false;
        LoadOneTensor(this->meanGm_[offset], meanLocal, extParam, padExtParam2, this->cBlockLengthT3Align);
        LoadOneTensor(this->varGm_[offset], varLocal, extParam, padExtParam2, this->cBlockLengthT3Align);

        weightQueue.EnQue(weightLocal);
        meanQueue.EnQue(meanLocal);
        varQueue.EnQue(varLocal);
    }

    __aicore__ inline void CopyOutDx(int64_t dxGmOffset)
    {
        LocalTensor<float> dx = dyQueue_.DeQue<float>();
        DataCopyExtParams copyInParams;
        copyInParams.blockCount = this->b0Dim;
        copyInParams.blockLen = this->b1Dim * sizeof(T1);
        copyInParams.srcStride = 0;
        copyInParams.dstStride = this->b1Dim * (this->aDim - 1) * sizeof(T1);

        for (int32_t i = 0; i < this->cBlockLength; i++) {
            int64_t dxGmOffsetForLoop = dxGmOffset + i * this->b1Dim;
            int64_t dxUbOffset = i * this->b1DimAlign * this->b0Dim;
            StoreOneTensor(
                this->dxGm_[dxGmOffsetForLoop], dx[dxUbOffset], copyInParams, this->b0Dim * this->b1DimAlign);
        }
        dyQueue_.FreeTensor(dx);
    }

    __aicore__ inline bool ProcessGetBlockInfo()
    {
        this->blockIdx = GetBlockIdx();
        if (this->blockIdx > this->needCoreNum) {
            return false;
        }

        this->totalChannel = this->coreChannelNum;
        if (this->blockIdx == this->needCoreNum - 1) {
            this->totalChannel = this->coreChannelNumTail;
        }
        return true;
    }

    __aicore__ inline void ComputeLoopCommonInfo(int64_t channelLoop, int64_t i)
    {
        if (i == channelLoop - 1) {
            this->cBlockLength = this->totalChannel - (channelLoop - 1) * this->cUbBlock;
        } else {
            this->cBlockLength = this->cUbBlock;
        }
        this->cBlockLengthT1Align = GetAlignValue<T1>(this->cBlockLength);
        this->cBlockLengthT2Align = GetAlignValue<T2>(this->cBlockLength);
        this->cBlockLengthT3Align = GetAlignValue<T3>(this->cBlockLength);
    }
};
} // namespace BatchNormGradV3
#endif

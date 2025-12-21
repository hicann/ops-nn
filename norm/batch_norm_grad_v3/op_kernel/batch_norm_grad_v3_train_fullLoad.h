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
 * \file batch_norm_grad_v3_train_fullLoad.h
 * \brief
 */

#ifndef BATCH_NORM_GRAD_V3_TRAIN_FULLLOAD_H
#define BATCH_NORM_GRAD_V3_TRAIN_FULLLOAD_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "kernel_utils.h"
#include "batch_norm_grad_v3_base_common.h"
#include "batch_norm_grad_v3_fullLoad_common.h"

namespace BatchNormGradV3 {
using namespace AscendC;

template <typename T1, typename T2, typename T3>
class BatchNormGradV3TrainFullload : public BatchNormGradV3FullLoadBase<T1, T2, T3> {
public:
    __aicore__ inline BatchNormGradV3TrainFullload(){};

    __aicore__ inline BatchNormGradV3TrainFullload(const BatchNormGradV3FullLoadTilingData* tilingDataIn)
    {
        tilingData_ = tilingDataIn;

        this->initFullLoadTiling(tilingData_);
        meanDenominator = (float)1.0 / (this->b0Dim * this->b1Dim);
    }

    __aicore__ inline void Init(GMStruct gmStruct, TPipe* pipeIn)
    {
        pipe_ = pipeIn;
        this->InitFullLoad(gmStruct, pipe_);

        int64_t xShapeLen = this->cUbBlock * this->bAlign;
        int64_t tmpCBlock = GetAlignValue<T2>(this->cUbBlock);
        pipe_->InitBuffer(dyMulInputQueue, BUFFER_NUM, xShapeLen * sizeof(float));
        pipe_->InitBuffer(dBiasQueue, BUFFER_NUM, tmpCBlock * B32_BLOCK_ALIGN_NUM * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        if (!this->ProcessGetBlockInfo()) {
            return;
        }

        int64_t channelLoop = CeilDiv(this->totalChannel, this->cUbBlock);
        for (int64_t i = 0; i < channelLoop; i++) {
            this->ComputeLoopCommonInfo(channelLoop, i);
            int64_t weightOffset = this->blockIdx * this->coreChannelNum + i * this->cUbBlock;
            int64_t dyOffset = this->blockIdx * this->coreChannelNum * this->b1Dim + i * this->cUbBlock * this->b1Dim;

            this->CopyInXDy(dyOffset);
            this->CopyInWeightMeanVar(weightOffset);
            Compute(weightOffset);
            this->CopyOutDx(dyOffset);
            dyMulInputQueue.FreeTensor(dyMulInputTensor);
            CopyOutDWeight(weightOffset);
        }
    }

private:
    __aicore__ inline void Compute(int64_t weightOffset)
    {
        this->dyTensor = this->dyQueue_.template DeQue<float>();
        this->xTensor = this->inputQueue.template DeQue<float>();
        this->weightTensor = this->weightQueue.template DeQue<float>();
        this->varTensor = this->varQueue.template DeQue<float>();
        this->meanTensor = this->meanQueue.template DeQue<float>();

        this->reduceUbTensor = this->reduceQueue.template AllocTensor<float>();
        this->varBrcbTensor = this->varBrcbQueue.template Get<float>();
        this->meanBrcbTensor = this->meanBrcbQueue.template Get<float>();
        this->weightBrcbTensor = this->weightBrcbQueue.template Get<float>();
        dyMulInputTensor = dyMulInputQueue.template AllocTensor<float>();
        computeDBias();
        CopyOutDBias(weightOffset);
        TPipeSetWaitFlag<HardEvent::MTE2_V>();

        WeightMeanVarStruct weightMeanVarStruct{this->weightTensor,     this->meanTensor,     this->varTensor,
                                                this->weightBrcbTensor, this->meanBrcbTensor, this->varBrcbTensor};
        this->BroadCastWeightMeanVar(weightMeanVarStruct);

        computeDWeight();
        computeDx();

        this->weightQueue.template FreeTensor<float>(this->weightTensor);
        this->varQueue.template FreeTensor<float>(this->varTensor);
        this->meanQueue.template FreeTensor<float>(this->meanTensor);
        this->meanBrcbQueue.template FreeTensor<float>(this->meanBrcbTensor);
        this->reduceQueue.template FreeTensor(this->reduceUbTensor);
    }

    __aicore__ inline void computeDBias()
    {
        LocalTensor dBiasTensor = dBiasQueue.AllocTensor<float>();
        this->DoNormalReduce(
            this->dyTensor, this->reduceUbTensor, this->reduceUbTensor, this->cBlockLength,
            this->b0Dim * this->b1DimAlign, this->b0Dim * this->b1DimAlign);
        int64_t copyNum = GetAlignValue<T2>(this->cBlockLength);
        DataCopy(dBiasTensor, this->reduceUbTensor, copyNum);
        PipeBarrier<PIPE_V>();
        dBiasQueue.template EnQue(dBiasTensor);
    }

    __aicore__ inline void computeDWeight()
    {
        PipeBarrier<PIPE_ALL>();;
        CALC_TWO_TENSOR_REPEAT(Sub, Adds, this->xTensor, this->meanBrcbTensor, CALC_SUB_FLAG, "dyMulx_mul_x");
        PipeBarrier<PIPE_V>();
        CALC_TWO_TENSOR(
            Mul, dyMulInputTensor, this->xTensor, this->dyTensor, this->cBlockLength, this->b1DimAlign * this->b0Dim,
            "x_mul_dy");
        PipeBarrier<PIPE_V>();
        this->DoNormalReduce(
            dyMulInputTensor, dyMulInputTensor, dyMulInputTensor, this->cBlockLength, this->b0Dim * this->b1DimAlign,
            this->b0Dim * this->b1DimAlign);
        PipeBarrier<PIPE_V>();
        Div(this->meanTensor, dyMulInputTensor, this->varTensor, this->cBlockLength);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void computeDx()
    {
        uint8_t repeatTime = CeilDiv(this->cBlockLength, B32_BLOCK_ALIGN_NUM);
        Brcb(this->meanBrcbTensor, dyMulInputTensor, repeatTime, {1, 8});
        PipeBarrier<PIPE_ALL>();;
        Div(this->meanBrcbTensor, this->meanBrcbTensor, this->varBrcbTensor, B32_BLOCK_ALIGN_NUM * this->cBlockLength);
        PipeBarrier<PIPE_ALL>();;
        Div(this->meanBrcbTensor, this->meanBrcbTensor, this->varBrcbTensor, B32_BLOCK_ALIGN_NUM * this->cBlockLength);
        PipeBarrier<PIPE_ALL>();;
        Muls(this->meanBrcbTensor, this->meanBrcbTensor, meanDenominator, B32_BLOCK_ALIGN_NUM * this->cBlockLength);
        PipeBarrier<PIPE_ALL>();;
        // 计算dx需要
        CALC_TWO_TENSOR_REPEAT(Mul, Muls, this->xTensor, this->meanBrcbTensor, 0, "x_sub_mean");
        PipeBarrier<PIPE_ALL>();;
        Muls(this->reduceUbTensor, this->reduceUbTensor, meanDenominator, this->cBlockLength);
        PipeBarrier<PIPE_ALL>();;
        Brcb(this->meanBrcbTensor, this->reduceUbTensor, repeatTime, {1, 8});
        PipeBarrier<PIPE_ALL>();;
        CALC_TWO_TENSOR_REPEAT(Sub, Adds, this->dyTensor, this->meanBrcbTensor, CALC_SUB_FLAG, "gradOut_sub_reduceUb");
        PipeBarrier<PIPE_ALL>();;
        CALC_TWO_TENSOR(
            Sub, this->dyTensor, this->dyTensor, this->xTensor, this->cBlockLength, this->b1DimAlign * this->b0Dim,
            "gradOut_sub_x");
        PipeBarrier<PIPE_ALL>();;
        CALC_TWO_TENSOR_REPEAT(Div, Muls, this->dyTensor, this->varBrcbTensor, CALC_DIV_FLAG, "gradOut_div_var3");
        PipeBarrier<PIPE_ALL>();;
        CALC_TWO_TENSOR_REPEAT(Mul, Muls, this->dyTensor, this->weightBrcbTensor, 0, "gradOut_mul_var");
        PipeBarrier<PIPE_ALL>();;
        this->dyQueue_.template EnQue(this->dyTensor);
    }

    __aicore__ inline void CopyOutDBias(int64_t dBiasGMOffset)
    {
        LocalTensor<float> dBiasTensor = dBiasQueue.template DeQue<float>();
        DataCopyExtParams copyInParams;
        copyInParams.blockCount = 1;
        copyInParams.blockLen = this->cBlockLength * sizeof(T2);
        copyInParams.srcStride = 0;
        copyInParams.dstStride = 0;
        StoreOneTensor(this->dBiasGm_[dBiasGMOffset], dBiasTensor, copyInParams, this->cBlockLengthT2Align);
        TPipeSetWaitFlag<HardEvent::MTE3_V>();
        dBiasQueue.template FreeTensor(dBiasTensor);
    }

    __aicore__ inline void CopyOutDWeight(int64_t dWeightGMOffset)
    {
        DataCopyExtParams copyInParams;
        copyInParams.blockCount = 1;
        copyInParams.blockLen = this->cBlockLength * sizeof(T2);
        copyInParams.srcStride = 0;
        copyInParams.dstStride = 0;
        StoreOneTensor(this->dWeightGm_[dWeightGMOffset], this->meanTensor, copyInParams, this->cBlockLengthT2Align);
        this->inputQueue.template FreeTensor(this->xTensor);
    }

private:
    const BatchNormGradV3FullLoadTilingData* tilingData_;

    TPipe* pipe_;

    TQue<QuePosition::VECOUT, BUFFER_DEPTH> dyMulInputQueue;
    TQue<QuePosition::VECOUT, BUFFER_DEPTH> dBiasQueue;
    LocalTensor<float> dyMulInputTensor;
    LocalTensor<float> dBiasTensor;

    float meanDenominator;
};
} // namespace BatchNormGradV3

#endif

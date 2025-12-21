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
 * \file batch_norm_grad_v3_infer_fullLoad.h
 * \brief
 */

#ifndef BATCH_NORM_GRAD_V3_INFER_FULLLOAD_H
#define BATCH_NORM_GRAD_V3_INFER_FULLLOAD_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "kernel_utils.h"
#include "batch_norm_grad_v3_base_common.h"
#include "batch_norm_grad_v3_fullLoad_common.h"

namespace BatchNormGradV3 {
using namespace AscendC;

template <typename T1, typename T2, typename T3>
class BatchNormGradV3InferFullload : public BatchNormGradV3FullLoadBase<T1, T2, T3> {
    static constexpr int32_t BUFFER_NUM = 1;
    static constexpr int32_t BUFFER_DEPTH = 1;

    static constexpr int64_t BLOCK_SIZE = 32U;

public:
    __aicore__ inline BatchNormGradV3InferFullload(){};

    __aicore__ inline BatchNormGradV3InferFullload(const BatchNormGradV3FullLoadTilingData* tilingDataIn)
    {
        tilingData_ = tilingDataIn;
        this->initFullLoadTiling(tilingData_);
    }

    __aicore__ inline void Init(GMStruct gmStruct, TPipe* pipeIn)
    {
        pipe_ = pipeIn;
        this->InitFullLoad(gmStruct, pipe_);
    }

    __aicore__ inline void Process()
    {
        if (!this->ProcessGetBlockInfo()) {
            return;
        }

        int64_t channelLoop = CeilDiv(this->totalChannel, this->cUbBlock);
        for (int64_t i = 0; i < channelLoop; i++) {
            this->ComputeLoopCommonInfo(channelLoop, i);
            int64_t dyOffset = this->blockIdx * this->coreChannelNum * this->b1Dim + i * this->cUbBlock * this->b1Dim;
            int64_t weightOffset = this->blockIdx * this->coreChannelNum + i * this->cUbBlock;
            this->CopyInXDy(dyOffset);
            this->CopyInWeightMeanVar(weightOffset);
            Compute();
            this->CopyOutDx(dyOffset);
            CopyOutDBias(weightOffset);
            CopyOutDWeight(weightOffset);
        }
    }

private:
    __aicore__ inline void Compute()
    {
        this->dyTensor = this->dyQueue_.template DeQue<float>();
        this->xTensor = this->inputQueue.template DeQue<float>();
        this->weightTensor = this->weightQueue.template DeQue<float>();
        this->varTensor = this->varQueue.template DeQue<float>();
        this->meanTensor = this->meanQueue.template DeQue<float>();
        this->reduceUbTensor = this->reduceQueue.template AllocTensor<float>();
        this->meanBrcbTensor = this->meanBrcbQueue.template Get<float>();
        this->varBrcbTensor = this->varBrcbQueue.template Get<float>();
        this->weightBrcbTensor = this->weightBrcbQueue.template Get<float>();

        computeDBias();
        TPipeSetWaitFlag<HardEvent::MTE2_V>();

        WeightMeanVarStruct weightMeanVarStruct{this->weightTensor,     this->meanTensor,     this->varTensor,
                                                this->weightBrcbTensor, this->meanBrcbTensor, this->varBrcbTensor};
        this->BroadCastWeightMeanVar(weightMeanVarStruct);

        computeDWeight();
        computeDx();

        this->weightQueue.template FreeTensor<float>(this->weightTensor);
        this->varQueue.template FreeTensor<float>(this->varTensor);
        this->meanQueue.template FreeTensor<float>(this->meanTensor);
    }

    __aicore__ inline void computeDBias()
    {
        this->DoNormalReduce(
            this->dyTensor, this->reduceUbTensor, this->reduceUbTensor, this->cBlockLength,
            this->b0Dim * this->b1DimAlign, this->b0Dim * this->b1DimAlign);
        PipeBarrier<PIPE_V>();
        this->reduceQueue.template EnQue(this->reduceUbTensor);
    }

    __aicore__ inline void computeDWeight()
    {
        PipeBarrier<PIPE_ALL>();;
        CALC_TWO_TENSOR_REPEAT(Sub, Adds, this->xTensor, this->meanBrcbTensor, CALC_SUB_FLAG, "x_sub_mean");
        PipeBarrier<PIPE_V>();
        CALC_TWO_TENSOR(
            Mul, this->xTensor, this->xTensor, this->dyTensor, this->cBlockLength, this->b1DimAlign * this->b0Dim,
            "x_mul_dy");
        PipeBarrier<PIPE_V>();
        this->DoNormalReduce(
            this->xTensor, this->xTensor, this->xTensor, this->cBlockLength, this->b0Dim * this->b1DimAlign,
            this->b0Dim * this->b1DimAlign);
        PipeBarrier<PIPE_V>();
        Div(this->xTensor, this->xTensor, this->varTensor, this->cBlockLength);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void computeDx()
    {
        CALC_TWO_TENSOR_REPEAT(Div, Muls, this->dyTensor, this->varBrcbTensor, CALC_DIV_FLAG, "gradOut_div_var");
        PipeBarrier<PIPE_V>();

        CALC_TWO_TENSOR_REPEAT(Mul, Muls, this->dyTensor, this->weightBrcbTensor, 0, "gradOut_mul_var");
        PipeBarrier<PIPE_V>();
        this->dyQueue_.template EnQue(this->dyTensor);
    }

    __aicore__ inline void CopyOutDBias(int64_t dBiasGMOffset)
    {
        this->reduceUbTensor = this->reduceQueue.template DeQue<float>();
        DataCopyExtParams copyInParams;
        copyInParams.blockCount = 1;
        copyInParams.blockLen = this->cBlockLength * sizeof(T2);
        copyInParams.srcStride = 0;
        copyInParams.dstStride = 0;
        StoreOneTensor(this->dBiasGm_[dBiasGMOffset], this->reduceUbTensor, copyInParams, this->cBlockLengthT2Align);
        this->reduceQueue.template FreeTensor(this->reduceUbTensor);
    }

    __aicore__ inline void CopyOutDWeight(int64_t dWeightGMOffset)
    {
        DataCopyExtParams copyInParams;
        copyInParams.blockCount = 1;
        copyInParams.blockLen = this->cBlockLength * sizeof(T2);
        copyInParams.srcStride = 0;
        copyInParams.dstStride = 0;
        StoreOneTensor(this->dWeightGm_[dWeightGMOffset], this->xTensor, copyInParams, this->cBlockLengthT2Align);
        this->inputQueue.template FreeTensor(this->xTensor);
    }

private:
    const BatchNormGradV3FullLoadTilingData* tilingData_;

    TPipe* pipe_;
};
} // namespace BatchNormGradV3

#endif

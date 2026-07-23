/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file selu_grad.h
 * \brief SeluGrad 算子 kernel 类定义
 */

#ifndef SELUGRAD_H
#define SELUGRAD_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "selu_grad_tiling_data.h"
#include "selu_grad_tiling_key.h"

#include <cstdint>

namespace NsSeluGrad {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
constexpr int64_t VECTOR_BYTES = 256;
constexpr float SCALE = 1.0507009873554804934193349852946f;
constexpr float SCALE_ALPHA_PRODUCT = 1.7580993408473768599402175208123f;

template <typename RawT, typename ComputeT, bool NeedCast>
class SeluGrad {
public:
    __aicore__ inline SeluGrad(){};

    __aicore__ inline void Init(TPipe& pipe, GM_ADDR gradients, GM_ADDR outputs, GM_ADDR y,
                                const SeluGradTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum);
    __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum);
    __aicore__ inline void Compute(int64_t currentNum);

private:
    TQue<QuePosition::VECIN, 1> gradientsQueue_;
    TQue<QuePosition::VECIN, 1> outputsQueue_;
    TQue<QuePosition::VECOUT, 1> yQueue_;

    TBuf<TPosition::VECCALC> gradientsComputeBuf_;
    TBuf<TPosition::VECCALC> outputsComputeBuf_;
    TBuf<TPosition::VECCALC> yComputeBuf_;
    TBuf<TPosition::VECCALC> tmpBuf_;
    TBuf<TPosition::VECCALC> maskBuf_;

    GlobalTensor<RawT> gradientsGm_;
    GlobalTensor<RawT> outputsGm_;
    GlobalTensor<RawT> yGm_;

    int64_t blockOffset_ = 0;
    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
};

template <typename RawT, typename ComputeT, bool NeedCast>
__aicore__ inline void SeluGrad<RawT, ComputeT, NeedCast>::Init(TPipe& pipe, GM_ADDR gradients, GM_ADDR outputs,
                                                                GM_ADDR y, const SeluGradTilingData* tilingData)
{
    if (tilingData == nullptr || tilingData->totalNum == 0 || tilingData->ubFactor == 0) {
        return;
    }

    const int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
    blockOffset_ = static_cast<int64_t>(tilingData->blockFactor) * blockIdx;
    if (static_cast<uint64_t>(blockOffset_) >= tilingData->totalNum) {
        return;
    }
    blockLength_ = static_cast<int64_t>(tilingData->totalNum - static_cast<uint64_t>(blockOffset_));
    if (blockLength_ > static_cast<int64_t>(tilingData->blockFactor)) {
        blockLength_ = static_cast<int64_t>(tilingData->blockFactor);
    }
    ubLength_ = static_cast<int64_t>(tilingData->ubFactor);

    gradientsGm_.SetGlobalBuffer((__gm__ RawT*)gradients + blockOffset_, blockLength_);
    outputsGm_.SetGlobalBuffer((__gm__ RawT*)outputs + blockOffset_, blockLength_);
    yGm_.SetGlobalBuffer((__gm__ RawT*)y + blockOffset_, blockLength_);

    const int32_t bufferNum = blockLength_ > ubLength_ ? BUFFER_NUM : 1;
    pipe.InitBuffer(gradientsQueue_, bufferNum, ubLength_ * sizeof(RawT));
    pipe.InitBuffer(outputsQueue_, bufferNum, ubLength_ * sizeof(RawT));
    pipe.InitBuffer(yQueue_, bufferNum, ubLength_ * sizeof(RawT));
    if constexpr (NeedCast) {
        pipe.InitBuffer(gradientsComputeBuf_, ubLength_ * sizeof(ComputeT));
        pipe.InitBuffer(outputsComputeBuf_, ubLength_ * sizeof(ComputeT));
        pipe.InitBuffer(yComputeBuf_, ubLength_ * sizeof(ComputeT));
    }
    pipe.InitBuffer(tmpBuf_, ubLength_ * sizeof(ComputeT));
    pipe.InitBuffer(maskBuf_, ubLength_ * sizeof(uint8_t));
}

template <typename RawT, typename ComputeT, bool NeedCast>
__aicore__ inline void SeluGrad<RawT, ComputeT, NeedCast>::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<RawT> gradientsLocal = gradientsQueue_.template AllocTensor<RawT>();
    LocalTensor<RawT> outputsLocal = outputsQueue_.template AllocTensor<RawT>();
    const uint32_t copyBytes = static_cast<uint32_t>(currentNum * sizeof(RawT));
    const uint32_t alignedBytes = (copyBytes + 31U) / 32U * 32U;
    const uint8_t rightPadding = static_cast<uint8_t>((alignedBytes - copyBytes) / sizeof(RawT));
    const DataCopyExtParams copyParams = {1, copyBytes, 0, 0, 0};
    const DataCopyPadExtParams<RawT> padParams = {rightPadding != 0, 0, rightPadding, static_cast<RawT>(0)};
    DataCopyPad(gradientsLocal, gradientsGm_[progress], copyParams, padParams);
    DataCopyPad(outputsLocal, outputsGm_[progress], copyParams, padParams);
    gradientsQueue_.EnQue(gradientsLocal);
    outputsQueue_.EnQue(outputsLocal);
}

template <typename RawT, typename ComputeT, bool NeedCast>
__aicore__ inline void SeluGrad<RawT, ComputeT, NeedCast>::Compute(int64_t currentNum)
{
    LocalTensor<RawT> gradientsRaw = gradientsQueue_.template DeQue<RawT>();
    LocalTensor<RawT> outputsRaw = outputsQueue_.template DeQue<RawT>();
    LocalTensor<RawT> yRaw = yQueue_.template AllocTensor<RawT>();

    LocalTensor<ComputeT> gradientsCompute;
    LocalTensor<ComputeT> outputsCompute;
    LocalTensor<ComputeT> yCompute;
    if constexpr (NeedCast) {
        gradientsCompute = gradientsComputeBuf_.template Get<ComputeT>();
        outputsCompute = outputsComputeBuf_.template Get<ComputeT>();
        yCompute = yComputeBuf_.template Get<ComputeT>();
        Cast(gradientsCompute, gradientsRaw, RoundMode::CAST_NONE, currentNum);
        Cast(outputsCompute, outputsRaw, RoundMode::CAST_NONE, currentNum);
        PipeBarrier<PIPE_V>();
    } else {
        gradientsCompute = gradientsRaw;
        outputsCompute = outputsRaw;
        yCompute = yRaw;
    }

    LocalTensor<ComputeT> tmp = tmpBuf_.template Get<ComputeT>();
    LocalTensor<uint8_t> mask = maskBuf_.template Get<uint8_t>();

    constexpr int64_t vectorAlign = VECTOR_BYTES / sizeof(ComputeT);
    const int64_t vectorNum = currentNum / vectorAlign * vectorAlign;
    if (vectorNum > 0) {
        CompareScalar(mask, outputsCompute, static_cast<ComputeT>(0), CMPMODE::LT, vectorNum);
        Adds(tmp, outputsCompute, static_cast<ComputeT>(SCALE_ALPHA_PRODUCT), vectorNum);
        PipeBarrier<PIPE_V>();
        Select(outputsCompute, mask, tmp, static_cast<ComputeT>(SCALE), SELMODE::VSEL_TENSOR_SCALAR_MODE, vectorNum);
        PipeBarrier<PIPE_V>();
        Mul(yCompute, gradientsCompute, outputsCompute, vectorNum);
    }

    for (int64_t i = vectorNum; i < currentNum; ++i) {
        const float gradientValue = static_cast<float>(gradientsCompute.GetValue(i));
        const float outputValue = static_cast<float>(outputsCompute.GetValue(i));
        float factorValue;
        if (outputValue < 0.0f) {
            if constexpr (sizeof(ComputeT) == sizeof(half)) {
                const ComputeT roundedFactor = static_cast<ComputeT>(
                    outputValue + static_cast<float>(static_cast<ComputeT>(SCALE_ALPHA_PRODUCT)));
                factorValue = static_cast<float>(roundedFactor);
            } else {
                factorValue = outputValue + SCALE_ALPHA_PRODUCT;
            }
        } else {
            factorValue = static_cast<float>(static_cast<ComputeT>(SCALE));
        }
        yCompute.SetValue(i, static_cast<ComputeT>(gradientValue * factorValue));
    }

    if constexpr (NeedCast) {
        if (vectorNum > 0) {
            PipeBarrier<PIPE_V>();
        }
        Cast(yRaw, yCompute, RoundMode::CAST_RINT, currentNum);
    }

    yQueue_.EnQue(yRaw);
    gradientsQueue_.FreeTensor(gradientsRaw);
    outputsQueue_.FreeTensor(outputsRaw);
}

template <typename RawT, typename ComputeT, bool NeedCast>
__aicore__ inline void SeluGrad<RawT, ComputeT, NeedCast>::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<RawT> yLocal = yQueue_.template DeQue<RawT>();
    const DataCopyExtParams copyParams = {1, static_cast<uint32_t>(currentNum * sizeof(RawT)), 0, 0, 0};
    DataCopyPad(yGm_[progress], yLocal, copyParams);
    yQueue_.FreeTensor(yLocal);
}

template <typename RawT, typename ComputeT, bool NeedCast>
__aicore__ inline void SeluGrad<RawT, ComputeT, NeedCast>::Process()
{
    if (blockLength_ <= 0 || ubLength_ <= 0) {
        return;
    }

    for (int64_t progress = 0; progress < blockLength_;) {
        int64_t currentNum = blockLength_ - progress;
        if (currentNum > ubLength_) {
            currentNum = ubLength_;
        }
        CopyIn(progress, currentNum);
        Compute(currentNum);
        CopyOut(progress, currentNum);
        progress += currentNum;
    }
}

} // namespace NsSeluGrad
#endif // SELUGRAD_H

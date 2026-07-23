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
 * \file hard_swish_grad.h
 * \brief HardSwishGrad 算子 kernel 类定义
 */

#ifndef HARDSWISHGRAD_H
#define HARDSWISHGRAD_H

#include <type_traits>
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "hard_swish_grad_tiling_data.h"
#include "hard_swish_grad_tiling_key.h"

namespace NsHardSwishGrad {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 1;
constexpr int64_t COMPUTE_ALIGN_NUM = 64;

template <typename T>
class HardSwishGrad {
public:
    __aicore__ inline HardSwishGrad(){};

    __aicore__ inline void Init(GM_ADDR grad, GM_ADDR x, GM_ADDR y, const HardSwishGradTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum);
    __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum);
    __aicore__ inline void Compute(int64_t currentNum);

private:
    __aicore__ inline int64_t AlignComputeNum(int64_t currentNum) const;

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueGrad;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;

    TBuf<QuePosition::VECCALC> xFp32Buf;
    TBuf<QuePosition::VECCALC> gradFp32Buf;
    TBuf<QuePosition::VECCALC> slopeBuf;
    TBuf<QuePosition::VECCALC> lowerBuf;
    TBuf<QuePosition::VECCALC> upperBuf;
    TBuf<QuePosition::VECCALC> zeroBuf;
    TBuf<QuePosition::VECCALC> oneBuf;
    TBuf<QuePosition::VECCALC> greaterMaskBuf;
    TBuf<QuePosition::VECCALC> lessMaskBuf;

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> inputGMGrad;
    GlobalTensor<T> outputGMY;

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
};

template <typename T>
__aicore__ inline int64_t HardSwishGrad<T>::AlignComputeNum(int64_t currentNum) const
{
    return (currentNum + COMPUTE_ALIGN_NUM - 1) / COMPUTE_ALIGN_NUM * COMPUTE_ALIGN_NUM;
}

template <typename T>
__aicore__ inline void HardSwishGrad<T>::Init(GM_ADDR grad, GM_ADDR x, GM_ADDR y,
                                              const HardSwishGradTilingData* tilingData)
{
    int64_t blockOffset = tilingData->blockFactor * AscendC::GetBlockIdx();
    int64_t remainderLength = tilingData->totalNum - blockOffset;
    blockLength_ = remainderLength > tilingData->blockFactor ? tilingData->blockFactor : remainderLength;
    if (blockLength_ < 0) {
        blockLength_ = 0;
    }
    ubLength_ = tilingData->ubFactor;

    inputGMX.SetGlobalBuffer((__gm__ T*)x + blockOffset, blockLength_);
    inputGMGrad.SetGlobalBuffer((__gm__ T*)grad + blockOffset, blockLength_);
    outputGMY.SetGlobalBuffer((__gm__ T*)y + blockOffset, blockLength_);

    if (blockLength_ <= 0 || ubLength_ <= 0) {
        return;
    }

    pipe.InitBuffer(inputQueueX, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(inputQueueGrad, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, ubLength_ * sizeof(T));

    if constexpr (!std::is_same_v<T, float>) {
        pipe.InitBuffer(xFp32Buf, ubLength_ * sizeof(float));
        pipe.InitBuffer(gradFp32Buf, ubLength_ * sizeof(float));
    }
    pipe.InitBuffer(slopeBuf, ubLength_ * sizeof(float));
    pipe.InitBuffer(lowerBuf, ubLength_ * sizeof(float));
    pipe.InitBuffer(upperBuf, ubLength_ * sizeof(float));
    pipe.InitBuffer(zeroBuf, ubLength_ * sizeof(float));
    pipe.InitBuffer(oneBuf, ubLength_ * sizeof(float));

    int64_t maskBytes = ((ubLength_ + 7) / 8 + 255) / 256 * 256;
    if (maskBytes < 256) {
        maskBytes = 256;
    }
    pipe.InitBuffer(greaterMaskBuf, maskBytes);
    pipe.InitBuffer(lessMaskBuf, maskBytes);

    LocalTensor<float> lowerLocal = lowerBuf.Get<float>();
    LocalTensor<float> upperLocal = upperBuf.Get<float>();
    LocalTensor<float> zeroLocal = zeroBuf.Get<float>();
    LocalTensor<float> oneLocal = oneBuf.Get<float>();
    Duplicate(lowerLocal, -3.0f, ubLength_);
    Duplicate(upperLocal, 3.0f, ubLength_);
    Duplicate(zeroLocal, 0.0f, ubLength_);
    Duplicate(oneLocal, 1.0f, ubLength_);
    PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline void HardSwishGrad<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.template AllocTensor<T>();
    LocalTensor<T> gradLocal = inputQueueGrad.template AllocTensor<T>();

    DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = currentNum * sizeof(T);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;

    int64_t offset = progress * ubLength_;
    DataCopyPad(xLocal, inputGMX[offset], copyParams, {false, 0, 0, 0});
    DataCopyPad(gradLocal, inputGMGrad[offset], copyParams, {false, 0, 0, 0});

    inputQueueX.EnQue(xLocal);
    inputQueueGrad.EnQue(gradLocal);
}

template <typename T>
__aicore__ inline void HardSwishGrad<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.template DeQue<T>();
    LocalTensor<T> gradLocal = inputQueueGrad.template DeQue<T>();
    LocalTensor<T> yLocal = outputQueueY.template AllocTensor<T>();

    LocalTensor<float> xFp32;
    LocalTensor<float> gradFp32;
    if constexpr (std::is_same_v<T, float>) {
        xFp32 = xLocal.template ReinterpretCast<float>();
        gradFp32 = gradLocal.template ReinterpretCast<float>();
    } else {
        xFp32 = xFp32Buf.Get<float>();
        gradFp32 = gradFp32Buf.Get<float>();
        Cast(xFp32, xLocal, RoundMode::CAST_NONE, currentNum);
        Cast(gradFp32, gradLocal, RoundMode::CAST_NONE, currentNum);
        PipeBarrier<PIPE_V>();
    }

    LocalTensor<float> slope = slopeBuf.Get<float>();
    LocalTensor<float> lowerLocal = lowerBuf.Get<float>();
    LocalTensor<float> upperLocal = upperBuf.Get<float>();
    LocalTensor<float> zeroLocal = zeroBuf.Get<float>();
    LocalTensor<float> oneLocal = oneBuf.Get<float>();
    LocalTensor<uint8_t> greaterMask = greaterMaskBuf.Get<uint8_t>();
    LocalTensor<uint8_t> lessMask = lessMaskBuf.Get<uint8_t>();

    int64_t computeNum = AlignComputeNum(currentNum);

    Compare(greaterMask, xFp32, lowerLocal, CMPMODE::GT, computeNum);
    Compare(lessMask, xFp32, upperLocal, CMPMODE::LT, computeNum);
    PipeBarrier<PIPE_V>();

    Muls(slope, xFp32, 0.333333343f, computeNum);
    Adds(slope, slope, 0.5f, computeNum);
    PipeBarrier<PIPE_V>();

    Select(slope, greaterMask, slope, zeroLocal, SELMODE::VSEL_TENSOR_TENSOR_MODE, computeNum);
    PipeBarrier<PIPE_V>();
    Select(slope, lessMask, slope, oneLocal, SELMODE::VSEL_TENSOR_TENSOR_MODE, computeNum);
    PipeBarrier<PIPE_V>();

    if constexpr (std::is_same_v<T, float>) {
        LocalTensor<float> yFp32 = yLocal.template ReinterpretCast<float>();
        Mul(yFp32, gradFp32, slope, computeNum);
    } else {
        Mul(slope, gradFp32, slope, computeNum);
        PipeBarrier<PIPE_V>();
        if constexpr (std::is_same_v<T, bfloat16_t>) {
            Cast(yLocal, slope, RoundMode::CAST_RINT, currentNum);
        } else {
            Cast(yLocal, slope, RoundMode::CAST_NONE, currentNum);
        }
    }

    outputQueueY.template EnQue<T>(yLocal);
    inputQueueX.FreeTensor(xLocal);
    inputQueueGrad.FreeTensor(gradLocal);
}

template <typename T>
__aicore__ inline void HardSwishGrad<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> yLocal = outputQueueY.template DeQue<T>();

    DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = currentNum * sizeof(T);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;

    DataCopyPad(outputGMY[progress * ubLength_], yLocal, copyParams);
    outputQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void HardSwishGrad<T>::Process()
{
    if (blockLength_ <= 0 || ubLength_ <= 0) {
        return;
    }

    int64_t loopCount = (blockLength_ + ubLength_ - 1) / ubLength_;
    for (int64_t i = 0; i < loopCount; ++i) {
        int64_t currentNum = (i == loopCount - 1) ? (blockLength_ - ubLength_ * i) : ubLength_;
        CopyIn(i, currentNum);
        Compute(currentNum);
        CopyOut(i, currentNum);
    }
}

} // namespace NsHardSwishGrad
#endif // HARDSWISHGRAD_H

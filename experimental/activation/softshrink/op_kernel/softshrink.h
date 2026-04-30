/** 
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. 
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License"). 
 * Please refer to the License for details. You may not use this file except in compliance with the License. 
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. 
 * See LICENSE in the root of the software repository for the full text of the License. 
 *
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

#ifndef SOFTSHRINK_H
#define SOFTSHRINK_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "softshrink_tiling_data.h"
#include "softshrink_tiling_key.h"

namespace NsSoftshrink {

using namespace AscendC;

// SoftShrink forward:
//   y[i] = x[i] - lambd,  if x[i] > lambd
//   y[i] = x[i] + lambd,  if x[i] < -lambd
//   y[i] = 0,              otherwise

template <typename T, int BUFFER_MODE>
class Softshrink {
    static constexpr int32_t BUFFER_NUM = BUFFER_MODE ? 2 : 1;
    static constexpr bool NEED_CAST = std::is_same_v<T, half> || std::is_same_v<T, bfloat16_t>;
    using ComputeType = std::conditional_t<NEED_CAST, float, T>;

public:
    __aicore__ inline Softshrink() {};
    __aicore__ inline void Init(GM_ADDR inputX, GM_ADDR outputY,
        const SoftshrinkTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum);
    __aicore__ inline void Compute(int64_t currentNum);
    __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueue;
    TBuf<QuePosition::VECCALC> tmpBuf;
    TBuf<QuePosition::VECCALC> maskBuf;

    GlobalTensor<T> inputGM;
    GlobalTensor<T> outputGM;

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
    float lambd_;
};

template <typename T, int BUFFER_MODE>
__aicore__ inline void Softshrink<T, BUFFER_MODE>::Init(
    GM_ADDR inputX, GM_ADDR outputY,
    const SoftshrinkTilingData* tilingData)
{
    int64_t remainderLength =
        tilingData->totalNum - tilingData->blockFactor * GetBlockIdx();
    blockLength_ = (remainderLength > tilingData->blockFactor)
        ? tilingData->blockFactor : remainderLength;
    ubLength_ = tilingData->ubFactor;
    lambd_ = tilingData->lambd;

    int64_t offset = tilingData->blockFactor * GetBlockIdx();
    inputGM.SetGlobalBuffer((__gm__ T*)inputX + offset, blockLength_);
    outputGM.SetGlobalBuffer((__gm__ T*)outputY + offset, blockLength_);

    pipe.InitBuffer(inputQueue, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(outputQueue, BUFFER_NUM, ubLength_ * sizeof(T));

    if constexpr (NEED_CAST) {
        // fp16/bf16: need fp32Buf for computation (reused from tmpBuf allocation)
        // tmpBuf serves as both fp32 workspace and tmp buffer
        // Allocate enough for 2 fp32 buffers (fp32Buf + tmp) back-to-back
        pipe.InitBuffer(tmpBuf, 2 * ubLength_ * sizeof(float));
    } else {
        pipe.InitBuffer(tmpBuf, ubLength_ * sizeof(T));
    }

    int64_t maskBytes = (ubLength_ + 7) / 8;
    maskBytes = (maskBytes + 255) / 256 * 256;
    pipe.InitBuffer(maskBuf, maskBytes);
}

template <typename T, int BUFFER_MODE>
__aicore__ inline void Softshrink<T, BUFFER_MODE>::CopyIn(
    int64_t progress, int64_t currentNum)
{
    LocalTensor<T> inputLocal = inputQueue.template AllocTensor<T>();

    DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = currentNum * sizeof(T);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;

    DataCopyPad(inputLocal, inputGM[progress * ubLength_], copyParams,
        {false, 0, 0, 0});

    inputQueue.EnQue(inputLocal);
}

template <typename T, int BUFFER_MODE>
__aicore__ inline void Softshrink<T, BUFFER_MODE>::Compute(
    int64_t currentNum)
{
    LocalTensor<T> inputLocal = inputQueue.template DeQue<T>();
    LocalTensor<T> resultLocal = outputQueue.template AllocTensor<T>();

    constexpr int64_t ALIGN_ELEM = 256 / static_cast<int64_t>(sizeof(ComputeType));
    int64_t alignedNum = (currentNum + ALIGN_ELEM - 1) / ALIGN_ELEM * ALIGN_ELEM;

    LocalTensor<uint8_t> maskLocal = maskBuf.Get<uint8_t>();

    if constexpr (NEED_CAST) {
        // fp16/bf16: cast to fp32, compute, cast back
        // tmpBuf is allocated as 2 * ubLength_ * sizeof(float)
        // Split into fp32Buf (first half) and fp32Tmp (second half)
        LocalTensor<float> fp32Buf = tmpBuf.Get<float>();
        LocalTensor<float> fp32Tmp = fp32Buf[ubLength_];

        // Cast input to fp32
        Cast(fp32Buf, inputLocal, RoundMode::CAST_NONE, alignedNum);
        PipeBarrier<PIPE_V>();

        // Step 1: Positive branch — mask = x > lambd; tmp = x - lambd; select
        Compares(maskLocal, fp32Buf, lambd_, CMPMODE::GT, alignedNum);
        Adds(fp32Tmp, fp32Buf, -lambd_, alignedNum);
        PipeBarrier<PIPE_V>();
        Select(fp32Tmp, maskLocal, fp32Tmp, 0.0f,
            SELMODE::VSEL_TENSOR_SCALAR_MODE, alignedNum);
        PipeBarrier<PIPE_V>();

        // Step 2: Negative branch — mask = x < -lambd; fp32Buf = x + lambd; select
        Compares(maskLocal, fp32Buf, -lambd_, CMPMODE::LT, alignedNum);
        Adds(fp32Buf, fp32Buf, lambd_, alignedNum);
        PipeBarrier<PIPE_V>();
        Select(fp32Buf, maskLocal, fp32Buf, 0.0f,
            SELMODE::VSEL_TENSOR_SCALAR_MODE, alignedNum);
        PipeBarrier<PIPE_V>();

        // Step 3: Merge — result = positive_branch + negative_branch
        Add(fp32Buf, fp32Buf, fp32Tmp, alignedNum);
        PipeBarrier<PIPE_V>();

        // Step 4: NaN handling — reuse fp32Tmp to recast input for NaN detection
        Cast(fp32Tmp, inputLocal, RoundMode::CAST_NONE, alignedNum);
        PipeBarrier<PIPE_V>();
        Compare(maskLocal, fp32Tmp, fp32Tmp, CMPMODE::NE, alignedNum);
        PipeBarrier<PIPE_V>();
        Select(fp32Buf, maskLocal, fp32Tmp, fp32Buf,
            SELMODE::VSEL_TENSOR_TENSOR_MODE, alignedNum);
        PipeBarrier<PIPE_V>();

        // Cast back to original type
        Cast(resultLocal, fp32Buf, RoundMode::CAST_ROUND, alignedNum);
        PipeBarrier<PIPE_V>();
    } else {
        // fp32: compute directly
        LocalTensor<T> tmpLocal = tmpBuf.Get<T>();

        // Step 1: Positive branch — mask = x > lambd; tmp = x - lambd; select
        Compares(maskLocal, inputLocal, lambd_, CMPMODE::GT, alignedNum);
        Adds(tmpLocal, inputLocal, (T)(-lambd_), alignedNum);
        PipeBarrier<PIPE_V>();
        Select(tmpLocal, maskLocal, tmpLocal, (T)0.0,
            SELMODE::VSEL_TENSOR_SCALAR_MODE, alignedNum);
        PipeBarrier<PIPE_V>();

        // Step 2: Negative branch — mask = x < -lambd; result = x + lambd; select
        Compares(maskLocal, inputLocal, (T)(-lambd_), CMPMODE::LT, alignedNum);
        Adds(resultLocal, inputLocal, (T)(lambd_), alignedNum);
        PipeBarrier<PIPE_V>();
        Select(resultLocal, maskLocal, resultLocal, (T)0.0,
            SELMODE::VSEL_TENSOR_SCALAR_MODE, alignedNum);
        PipeBarrier<PIPE_V>();

        // Step 3: Merge — result = positive_branch + negative_branch
        Add(resultLocal, resultLocal, tmpLocal, alignedNum);
        PipeBarrier<PIPE_V>();
        // Step 4: NaN handling — preserve NaN values from input
        Compare(maskLocal, inputLocal, inputLocal, CMPMODE::NE, alignedNum);
        PipeBarrier<PIPE_V>();
        Select(resultLocal, maskLocal, inputLocal, resultLocal, SELMODE::VSEL_TENSOR_TENSOR_MODE, alignedNum);
        PipeBarrier<PIPE_V>();
    }

    outputQueue.template EnQue<T>(resultLocal);
    inputQueue.FreeTensor(inputLocal);
}

template <typename T, int BUFFER_MODE>
__aicore__ inline void Softshrink<T, BUFFER_MODE>::CopyOut(
    int64_t progress, int64_t currentNum)
{
    LocalTensor<T> resultLocal = outputQueue.template DeQue<T>();
    DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = currentNum * sizeof(T);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    DataCopyPad(outputGM[progress * ubLength_], resultLocal, copyParams);
    outputQueue.FreeTensor(resultLocal);
}

template <typename T, int BUFFER_MODE>
__aicore__ inline void Softshrink<T, BUFFER_MODE>::Process()
{
    int64_t loopCount = (blockLength_ + ubLength_ - 1) / ubLength_;

    if constexpr (BUFFER_NUM == 2) {
        int64_t curNum0 = (loopCount == 1) ? blockLength_ : ubLength_;
        if (loopCount >= 2) {
            CopyIn(0, curNum0);
            for (int64_t i = 1; i < loopCount; i++) {
                int64_t prevNum = (i == 1) ? curNum0 :
                    ((i - 1 == loopCount - 1) ? (blockLength_ - ubLength_ * (i - 1)) : ubLength_);
                int64_t curNum = (i == loopCount - 1)
                    ? (blockLength_ - ubLength_ * i) : ubLength_;
                CopyIn(i, curNum);
                Compute(prevNum);
                CopyOut(i - 1, prevNum);
            }
            int64_t lastNum = blockLength_ - ubLength_ * (loopCount - 1);
            Compute(lastNum);
            CopyOut(loopCount - 1, lastNum);
        } else if (loopCount == 1) {
            CopyIn(0, curNum0);
            Compute(curNum0);
            CopyOut(0, curNum0);
        }
    } else {
        for (int64_t i = 0; i < loopCount; i++) {
            int64_t currentNum = (i == (loopCount - 1))
                ? (blockLength_ - ubLength_ * i) : ubLength_;
            CopyIn(i, currentNum);
            Compute(currentNum);
            CopyOut(i, currentNum);
        }
    }
}

} // namespace NsSoftshrink
#endif // SOFTSHRINK_H

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

/*!
 * \file hard_swish_grad.h
 * \brief HardSwishGrad kernel class definition (arch35)
 *
 * Computes: grad_input = grad_output * HardSwish'(self)
 * where HardSwish'(x) = 0 if x <= -3, (2x+3)/6 if -3 < x < 3, 1 if x >= 3
 *
 * Template parameters:
 *   - T: data type (float / half / bfloat16_t)
 *   - BUFFER_MODE: buffer mode (0=single buffer, 1=double buffer)
 *
 * For fp16/bf16, input data is cast to fp32 for intermediate computation
 * to meet precision requirements, then cast back to the original type for output.
 */

#ifndef HARD_SWISH_GRAD_H
#define HARD_SWISH_GRAD_H

#include "kernel_operator.h"
#ifndef DTYPE_X
#include "kernel_tiling/kernel_tiling.h"
#include "hard_swish_grad_tiling_data.h"
#include "hard_swish_grad_tiling_key.h"
#endif

namespace NsHardSwishGrad {

using namespace AscendC;

template <typename T, int BUFFER_MODE>
class HardSwishGrad {
    static constexpr int32_t BUFFER_NUM = BUFFER_MODE ? 2 : 1;
    // fp16 needs promotion to fp32 for computation
    static constexpr bool NEED_CAST = std::is_same_v<T, half> || std::is_same_v<T, bfloat16_t>;
    using ComputeType = std::conditional_t<NEED_CAST, float, T>;

public:
    __aicore__ inline HardSwishGrad() {};
    __aicore__ inline void Init(GM_ADDR gradOutput, GM_ADDR self,
        GM_ADDR gradInput, const HardSwishGradTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum);
    __aicore__ inline void Compute(int64_t currentNum);
    __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> gradOutputQueue;
    TQue<QuePosition::VECIN, BUFFER_NUM> selfQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> gradInputQueue;
    TBuf<QuePosition::VECCALC> tmpBuf;    // fp32: derivative; fp16: selfFp32/gradOutFp32
    TBuf<QuePosition::VECCALC> tmpBuf2;   // fp32: mask; fp16: derivFp32
    TBuf<QuePosition::VECCALC> maskBuf;   // mask buffer for Compares/Select

    GlobalTensor<T> gradOutputGM;
    GlobalTensor<T> selfGM;
    GlobalTensor<T> gradInputGM;

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
};

template <typename T, int BUFFER_MODE>
__aicore__ inline void HardSwishGrad<T, BUFFER_MODE>::Init(
    GM_ADDR gradOutput, GM_ADDR self, GM_ADDR gradInput,
    const HardSwishGradTilingData* tilingData)
{
    int64_t remainderLength =
        tilingData->totalNum - tilingData->blockFactor * GetBlockIdx();
    blockLength_ = (remainderLength > tilingData->blockFactor)
        ? tilingData->blockFactor : remainderLength;
    ubLength_ = tilingData->ubFactor;

    int64_t offset = tilingData->blockFactor * GetBlockIdx();
    gradOutputGM.SetGlobalBuffer((__gm__ T*)gradOutput + offset, blockLength_);
    selfGM.SetGlobalBuffer((__gm__ T*)self + offset, blockLength_);
    gradInputGM.SetGlobalBuffer((__gm__ T*)gradInput + offset, blockLength_);

    pipe.InitBuffer(gradOutputQueue, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(selfQueue, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(gradInputQueue, BUFFER_NUM, ubLength_ * sizeof(T));

    // Temporary buffer for intermediate computation
    if constexpr (NEED_CAST) {
        // fp16: two separate fp32 buffers + mask
        pipe.InitBuffer(tmpBuf, ubLength_ * sizeof(float));
        pipe.InitBuffer(tmpBuf2, ubLength_ * sizeof(float));
    } else {
        // fp32: 1 derivative buffer
        pipe.InitBuffer(tmpBuf, ubLength_ * sizeof(T));
    }
    // Mask buffer for Compares/Select: 1 bit per element, aligned to 256 bytes
    int64_t maskBytes = (ubLength_ + 7) / 8;
    maskBytes = (maskBytes + 255) / 256 * 256;
    pipe.InitBuffer(maskBuf, maskBytes);
}

template <typename T, int BUFFER_MODE>
__aicore__ inline void HardSwishGrad<T, BUFFER_MODE>::CopyIn(
    int64_t progress, int64_t currentNum)
{
    LocalTensor<T> gradOutLocal = gradOutputQueue.template AllocTensor<T>();
    LocalTensor<T> selfLocal = selfQueue.template AllocTensor<T>();

    DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = currentNum * sizeof(T);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;

    DataCopyPad(gradOutLocal, gradOutputGM[progress * ubLength_], copyParams,
        {false, 0, 0, 0});
    DataCopyPad(selfLocal, selfGM[progress * ubLength_], copyParams,
        {false, 0, 0, 0});

    gradOutputQueue.EnQue(gradOutLocal);
    selfQueue.EnQue(selfLocal);
}

template <typename T, int BUFFER_MODE>
__aicore__ inline void HardSwishGrad<T, BUFFER_MODE>::Compute(
    int64_t currentNum)
{
    LocalTensor<T> gradOutLocal = gradOutputQueue.template DeQue<T>();
    LocalTensor<T> selfLocal = selfQueue.template DeQue<T>();
    LocalTensor<T> resultLocal = gradInputQueue.template AllocTensor<T>();

    // Align to 256 bytes for vector instructions (Compares/Select require it)
    constexpr int64_t ALIGN_ELEM = 256 / static_cast<int64_t>(sizeof(ComputeType));
    int64_t alignedNum = (currentNum + ALIGN_ELEM - 1) / ALIGN_ELEM * ALIGN_ELEM;

    if constexpr (NEED_CAST) {
        // fp16/bf16 path: cast to fp32 for computation
        LocalTensor<float> selfFp32 = tmpBuf.Get<float>();
        LocalTensor<float> derivFp32 = tmpBuf2.Get<float>();
        LocalTensor<uint8_t> maskLocal = maskBuf.Get<uint8_t>();

        // Group 1: Cast self → fp32, then linear part + first Compares (independent)
        Cast(selfFp32, selfLocal, RoundMode::CAST_NONE, alignedNum);
           // Muls & Compares both read selfFp32
        Muls(derivFp32, selfFp32, (1.0f / 3.0f), alignedNum);
        Adds(derivFp32, derivFp32, 0.5f, alignedNum);                     // chain on derivFp32
        Compares(maskLocal, selfFp32, -3.0f, CMPMODE::GT, alignedNum);    // reads selfFp32 only
           // Select reads both derivFp32 and maskLocal

        // Group 2: first Select + second Compares (WAR on mask, independent of deriv write)
        Select(derivFp32, maskLocal, derivFp32, 0.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, alignedNum);
        Compares(maskLocal, selfFp32, 3.0f, CMPMODE::LT, alignedNum);
           // next Select reads both mask and deriv

        // Group 3: second Select, then reuse selfFp32 for gradOut cast
        Select(derivFp32, maskLocal, derivFp32, 1.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, alignedNum);
           // ensure all reads of selfFp32 done before overwrite
        Cast(selfFp32, gradOutLocal, RoundMode::CAST_NONE, alignedNum);
        Mul(derivFp32, selfFp32, derivFp32, alignedNum);
        Cast(resultLocal, derivFp32, RoundMode::CAST_ROUND, alignedNum);
           // ensure result ready before EnQue
    } else {
        // fp32 path: compute directly
        LocalTensor<T> derivLocal = tmpBuf.Get<T>();
        LocalTensor<uint8_t> maskLocal = maskBuf.Get<uint8_t>();

        // Group 1: linear part + first Compares (Compares reads selfLocal, independent of deriv)
        Muls(derivLocal, selfLocal, (T)(1.0 / 3.0), alignedNum);
        Adds(derivLocal, derivLocal, (T)0.5, alignedNum);                   // chain on derivLocal
        Compares(maskLocal, selfLocal, (T)(-3.0), CMPMODE::GT, alignedNum); // reads selfLocal only
           // Select reads both derivLocal and maskLocal

        // Group 2: first Select + second Compares (WAR on mask)
        Select(derivLocal, maskLocal, derivLocal, (T)0.0, SELMODE::VSEL_TENSOR_SCALAR_MODE, alignedNum);
        Compares(maskLocal, selfLocal, (T)3.0, CMPMODE::LT, alignedNum);
           // next Select reads both mask and deriv

        // Group 3: second Select + final multiply
        Select(derivLocal, maskLocal, derivLocal, (T)1.0, SELMODE::VSEL_TENSOR_SCALAR_MODE, alignedNum);
        Mul(resultLocal, gradOutLocal, derivLocal, alignedNum);
           // ensure result ready before EnQue
    }

    gradInputQueue.template EnQue<T>(resultLocal);
    gradOutputQueue.FreeTensor(gradOutLocal);
    selfQueue.FreeTensor(selfLocal);
}

template <typename T, int BUFFER_MODE>
__aicore__ inline void HardSwishGrad<T, BUFFER_MODE>::CopyOut(
    int64_t progress, int64_t currentNum)
{
    LocalTensor<T> resultLocal = gradInputQueue.template DeQue<T>();
    DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = currentNum * sizeof(T);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    DataCopyPad(gradInputGM[progress * ubLength_], resultLocal, copyParams);
    gradInputQueue.FreeTensor(resultLocal);
}

template <typename T, int BUFFER_MODE>
__aicore__ inline void HardSwishGrad<T, BUFFER_MODE>::Process()
{
    int64_t loopCount = (blockLength_ + ubLength_ - 1) / ubLength_;
    for (int64_t i = 0; i < loopCount; i++) {
        int64_t currentNum = (i == (loopCount - 1))
            ? (blockLength_ - ubLength_ * i) : ubLength_;
        CopyIn(i, currentNum);
        Compute(currentNum);
        CopyOut(i, currentNum);
    }
}

} // namespace NsHardSwishGrad
#endif // HARD_SWISH_GRAD_H

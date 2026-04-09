/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

/*!
 * \file hard_sigmoid_grad_bf16.h
 * \brief HardSigmoidGrad bfloat16 kernel class definition (arch35)
 *
 * bfloat16 computation flow:
 * 1. CopyIn: DataCopyPad bf16 gradOutput and self from GM to UB
 * 2. Compute:
 *    a. Cast bf16 -> fp32 for both inputs
 *    b. tmp = alpha * selfFp32 + beta (in fp32)
 *    c. result = gradOutputFp32 * alpha (in fp32)
 *    d. mask = (tmp > 0) && (tmp < 1) via two-pass CompareScalar+Select
 *    e. Cast fp32 result -> bf16
 * 3. CopyOut: DataCopyPad bf16 result from UB to GM
 *
 * Buffer plan (7 data buffers + 1 mask buffer):
 *   Queued: inQueueGradOutput(bf16), inQueueSelf(bf16), outQueueResult(bf16)
 *   Temp:   gradFp32Buf(fp32), selfFp32Buf(fp32), tmpBuf(fp32), maskBuf(uint8)
 */
#ifndef HARD_SIGMOID_GRAD_BF16_H
#define HARD_SIGMOID_GRAD_BF16_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "hard_sigmoid_grad_tiling_data.h"

namespace NsHardSigmoidGrad {

using namespace AscendC;

class HardSigmoidGradBf16 {
    static constexpr int32_t BUFFER_NUM = 2; // double buffer

public:
    __aicore__ inline HardSigmoidGradBf16() {};

    __aicore__ inline void Init(GM_ADDR gradOutput, GM_ADDR self, GM_ADDR gradInput,
                                 const HardSigmoidGradTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum);
    __aicore__ inline void Compute(int64_t currentNum);
    __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueGradOutput;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueSelf;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueResult;
    // fp32 temp buffers for Cast + compute
    TBuf<QuePosition::VECCALC> gradFp32Buf;
    TBuf<QuePosition::VECCALC> selfFp32Buf;
    TBuf<QuePosition::VECCALC> tmpBuf;
    // mask buffer for CompareScalar bitmask
    TBuf<QuePosition::VECCALC> maskBuf;

    GlobalTensor<bfloat16_t> gmGradOutput;
    GlobalTensor<bfloat16_t> gmSelf;
    GlobalTensor<bfloat16_t> gmGradInput;

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
    float alpha_ = 0.16666666f;
    float beta_ = 0.5f;
};

__aicore__ inline void HardSigmoidGradBf16::Init(
    GM_ADDR gradOutput, GM_ADDR self, GM_ADDR gradInput,
    const HardSigmoidGradTilingData* tilingData)
{
    int64_t remainderLength = tilingData->totalLength - tilingData->blockFactor * AscendC::GetBlockIdx();
    blockLength_ = (remainderLength > tilingData->blockFactor) ? tilingData->blockFactor : remainderLength;
    if (blockLength_ <= 0) {
        return;
    }
    ubLength_ = tilingData->ubFactor;
    alpha_ = tilingData->alpha;
    beta_ = tilingData->beta;

    gmGradOutput.SetGlobalBuffer((__gm__ bfloat16_t*)gradOutput + tilingData->blockFactor * AscendC::GetBlockIdx(), blockLength_);
    gmSelf.SetGlobalBuffer((__gm__ bfloat16_t*)self + tilingData->blockFactor * AscendC::GetBlockIdx(), blockLength_);
    gmGradInput.SetGlobalBuffer((__gm__ bfloat16_t*)gradInput + tilingData->blockFactor * AscendC::GetBlockIdx(), blockLength_);

    // bf16 input/output queues (bf16 element size)
    pipe.InitBuffer(inQueueGradOutput, BUFFER_NUM, ubLength_ * sizeof(bfloat16_t));
    pipe.InitBuffer(inQueueSelf, BUFFER_NUM, ubLength_ * sizeof(bfloat16_t));
    pipe.InitBuffer(outQueueResult, BUFFER_NUM, ubLength_ * sizeof(bfloat16_t));
    // fp32 temp buffers (fp32 element size)
    pipe.InitBuffer(gradFp32Buf, ubLength_ * sizeof(float));
    pipe.InitBuffer(selfFp32Buf, ubLength_ * sizeof(float));
    pipe.InitBuffer(tmpBuf, ubLength_ * sizeof(float));
    // mask buffer: N fp32 elements => N/8 bytes bitmask, 32-byte aligned
    int64_t maskBufSize = ((ubLength_ + 7) / 8 + 31) / 32 * 32;
    if (maskBufSize < 32) maskBufSize = 32;
    pipe.InitBuffer(maskBuf, maskBufSize);
}

__aicore__ inline void HardSigmoidGradBf16::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<bfloat16_t> gradLocal = inQueueGradOutput.AllocTensor<bfloat16_t>();
    LocalTensor<bfloat16_t> selfLocal = inQueueSelf.AllocTensor<bfloat16_t>();

    DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = currentNum * sizeof(bfloat16_t);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;

    DataCopyPad(gradLocal, gmGradOutput[progress * ubLength_], copyParams, {false, 0, 0, 0});
    DataCopyPad(selfLocal, gmSelf[progress * ubLength_], copyParams, {false, 0, 0, 0});

    inQueueGradOutput.EnQue(gradLocal);
    inQueueSelf.EnQue(selfLocal);
}

__aicore__ inline void HardSigmoidGradBf16::Compute(int64_t currentNum)
{
    LocalTensor<bfloat16_t> gradLocal = inQueueGradOutput.DeQue<bfloat16_t>();
    LocalTensor<bfloat16_t> selfLocal = inQueueSelf.DeQue<bfloat16_t>();
    LocalTensor<bfloat16_t> resultLocal = outQueueResult.AllocTensor<bfloat16_t>();

    LocalTensor<float> gradFp32 = gradFp32Buf.Get<float>();
    LocalTensor<float> selfFp32 = selfFp32Buf.Get<float>();
    LocalTensor<float> tmp = tmpBuf.Get<float>();
    LocalTensor<uint8_t> mask = maskBuf.Get<uint8_t>();

    // Align count to 256-byte boundary for fp32 CompareScalar (256/4 = 64 elements)
    constexpr int64_t COMPARE_ALIGN = 64;
    int64_t alignedCount = (currentNum + COMPARE_ALIGN - 1) / COMPARE_ALIGN * COMPARE_ALIGN;

    // Step 1: Cast bf16 -> fp32
    Cast(gradFp32, gradLocal, RoundMode::CAST_NONE, currentNum);
    Cast(selfFp32, selfLocal, RoundMode::CAST_NONE, currentNum);

    // Step 2: tmp = alpha * selfFp32 + beta (in fp32)
    Muls(tmp, selfFp32, alpha_, currentNum);
    Adds(tmp, tmp, beta_, currentNum);

    // Step 3: result = gradOutputFp32 * alpha (reuse selfFp32 buffer)
    Muls(selfFp32, gradFp32, alpha_, currentNum);

    // Step 4: mask = (tmp > 0), select
    CompareScalar(mask, tmp, 0.0f, CMPMODE::GT, alignedCount);
    Select(selfFp32, mask, selfFp32, 0.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, currentNum);

    // Step 5: mask = (tmp < 1), select
    CompareScalar(mask, tmp, 1.0f, CMPMODE::LT, alignedCount);
    Select(selfFp32, mask, selfFp32, 0.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, currentNum);

    // Step 6: Cast fp32 result -> bf16
    Cast(resultLocal, selfFp32, RoundMode::CAST_ROUND, currentNum);

    outQueueResult.EnQue(resultLocal);
    inQueueGradOutput.FreeTensor(gradLocal);
    inQueueSelf.FreeTensor(selfLocal);
}

__aicore__ inline void HardSigmoidGradBf16::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<bfloat16_t> resultLocal = outQueueResult.DeQue<bfloat16_t>();

    DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = currentNum * sizeof(bfloat16_t);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;

    DataCopyPad(gmGradInput[progress * ubLength_], resultLocal, copyParams);
    outQueueResult.FreeTensor(resultLocal);
}

__aicore__ inline void HardSigmoidGradBf16::Process()
{
    if (blockLength_ <= 0) {
        return;
    }
    int64_t loopCount = (blockLength_ + ubLength_ - 1) / ubLength_;
    for (int64_t i = 0; i < loopCount; i++) {
        int64_t currentNum = (i == (loopCount - 1)) ? (blockLength_ - ubLength_ * i) : ubLength_;
        CopyIn(i, currentNum);
        Compute(currentNum);
        CopyOut(i, currentNum);
    }
}

} // namespace NsHardSigmoidGrad
#endif // HARD_SIGMOID_GRAD_BF16_H

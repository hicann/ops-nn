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

/**
 * \file hard_shrink_grad.h
 * \brief HardShrinkGrad kernel class definitions (arch32 architecture)
 *
 * Computation: output_i = grad_output_i if |self_i| > lambd, else 0
 *
 * Two kernel implementations:
 *   - HardShrinkGradDirect<T, BUFFER_MODE>: for T=float, computes directly in T
 *   - HardShrinkGradCastFp32<T, BUFFER_MODE>: for T=half/bfloat16_t, casts to fp32
 *     for Abs/Compare/Select to avoid incorrect Compare results on arch32
 *
 * Data flow (Direct, fp32):
 *   CopyIn:  GM(grad_output, self) -> UB
 *   Compute: Abs(self) -> absLocal
 *            Compare(absLocal, lambdLocal, GT) -> cmpMask
 *            Select(cmpMask, grad, zero) -> output
 *   CopyOut: UB(output) -> GM
 *
 * Data flow (CastFp32, fp16):
 *   CopyIn:  GM(grad_output_fp16, self_fp16) -> UB
 *   Compute: Cast(fp16 -> fp32)
 *            Abs(self_fp32) -> absLocal_fp32
 *            Compare(absLocal_fp32, lambdLocal_fp32, GT) -> cmpMask
 *            Select(cmpMask, grad_fp32, zero_fp32) -> out_fp32
 *            Cast(fp32 -> fp16)
 *   CopyOut: UB(output_fp16) -> GM
 */
#ifndef HARD_SHRINK_GRAD_H
#define HARD_SHRINK_GRAD_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "hard_shrink_grad_tiling_data.h"
#include "hard_shrink_grad_tiling_key.h"

namespace NsHardShrinkGrad {

using namespace AscendC;

// ============================================================================
// HardShrinkGradDirect: Direct computation in native type T (for fp32)
// ============================================================================
template <typename T, int BUFFER_MODE>
class HardShrinkGradDirect {
    static constexpr int32_t BUFFER_NUM = BUFFER_MODE ? 2 : 1;

public:
    __aicore__ inline HardShrinkGradDirect() {};

    __aicore__ inline void Init(GM_ADDR gradOutput, GM_ADDR self, GM_ADDR output,
                                 const HardShrinkGradTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum);
    __aicore__ inline void Compute(int64_t currentNum);
    __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueGrad;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueSelf;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueue;
    TBuf<QuePosition::VECCALC> absBuf;
    TBuf<QuePosition::VECCALC> lambdBuf;
    TBuf<QuePosition::VECCALC> zeroBuf;
    TBuf<QuePosition::VECCALC> cmpMaskBuf;

    GlobalTensor<T> inputGMGrad;
    GlobalTensor<T> inputGMSelf;
    GlobalTensor<T> outputGM;

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
    float lambd_ = 0.5f;
};

template <typename T, int BUFFER_MODE>
__aicore__ inline void HardShrinkGradDirect<T, BUFFER_MODE>::Init(
    GM_ADDR gradOutput, GM_ADDR self, GM_ADDR output,
    const HardShrinkGradTilingData* tilingData)
{
    int64_t remainderLength = tilingData->totalNum - tilingData->blockFactor * AscendC::GetBlockIdx();
    blockLength_ = (remainderLength > tilingData->blockFactor) ? tilingData->blockFactor : remainderLength;
    ubLength_ = tilingData->ubFactor;
    lambd_ = tilingData->lambd;

    int64_t offset = tilingData->blockFactor * AscendC::GetBlockIdx();
    inputGMGrad.SetGlobalBuffer((__gm__ T*)gradOutput + offset, blockLength_);
    inputGMSelf.SetGlobalBuffer((__gm__ T*)self + offset, blockLength_);
    outputGM.SetGlobalBuffer((__gm__ T*)output + offset, blockLength_);

    pipe.InitBuffer(inputQueueGrad, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(inputQueueSelf, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(outputQueue, BUFFER_NUM, ubLength_ * sizeof(T));

    pipe.InitBuffer(absBuf, ubLength_ * sizeof(T));
    pipe.InitBuffer(lambdBuf, ubLength_ * sizeof(T));
    pipe.InitBuffer(zeroBuf, ubLength_ * sizeof(T));

    int64_t cmpMaskBytes = ((ubLength_ + 7) / 8 + 255) / 256 * 256;
    if (cmpMaskBytes < 256) cmpMaskBytes = 256;
    pipe.InitBuffer(cmpMaskBuf, cmpMaskBytes);

    if (blockLength_ > 0) {
        LocalTensor<T> lambdLocal = lambdBuf.Get<T>();
        LocalTensor<T> zeroLocal = zeroBuf.Get<T>();
        Duplicate(lambdLocal, static_cast<T>(lambd_), ubLength_);
        Duplicate(zeroLocal, static_cast<T>(0), ubLength_);
        PipeBarrier<PIPE_V>();
    }
}

template <typename T, int BUFFER_MODE>
__aicore__ inline void HardShrinkGradDirect<T, BUFFER_MODE>::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> gradLocal = inputQueueGrad.template AllocTensor<T>();
    LocalTensor<T> selfLocal = inputQueueSelf.template AllocTensor<T>();

    DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = currentNum * sizeof(T);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;

    DataCopyPad(gradLocal, inputGMGrad[progress * ubLength_], copyParams, {false, 0, 0, 0});
    DataCopyPad(selfLocal, inputGMSelf[progress * ubLength_], copyParams, {false, 0, 0, 0});

    inputQueueGrad.EnQue(gradLocal);
    inputQueueSelf.EnQue(selfLocal);
}

template <typename T, int BUFFER_MODE>
__aicore__ inline void HardShrinkGradDirect<T, BUFFER_MODE>::Compute(int64_t currentNum)
{
    LocalTensor<T> gradLocal = inputQueueGrad.template DeQue<T>();
    LocalTensor<T> selfLocal = inputQueueSelf.template DeQue<T>();
    LocalTensor<T> outLocal = outputQueue.template AllocTensor<T>();

    LocalTensor<T> absLocal = absBuf.Get<T>();
    LocalTensor<T> lambdLocal = lambdBuf.Get<T>();
    LocalTensor<T> zeroLocal = zeroBuf.Get<T>();
    LocalTensor<uint8_t> cmpMask = cmpMaskBuf.Get<uint8_t>();

    int64_t computeNum = ubLength_;

    Abs(absLocal, selfLocal, computeNum);
    Compare(cmpMask, absLocal, lambdLocal, CMPMODE::GT, computeNum);
    Select(outLocal, cmpMask, gradLocal, zeroLocal, SELMODE::VSEL_TENSOR_TENSOR_MODE, computeNum);

    outputQueue.template EnQue<T>(outLocal);
    inputQueueGrad.FreeTensor(gradLocal);
    inputQueueSelf.FreeTensor(selfLocal);
}

template <typename T, int BUFFER_MODE>
__aicore__ inline void HardShrinkGradDirect<T, BUFFER_MODE>::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> outLocal = outputQueue.template DeQue<T>();

    DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = currentNum * sizeof(T);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;

    DataCopyPad(outputGM[progress * ubLength_], outLocal, copyParams);
    outputQueue.FreeTensor(outLocal);
}

template <typename T, int BUFFER_MODE>
__aicore__ inline void HardShrinkGradDirect<T, BUFFER_MODE>::Process()
{
    if (blockLength_ <= 0) return;

    int64_t loopCount = (blockLength_ + ubLength_ - 1) / ubLength_;
    for (int64_t i = 0; i < loopCount; i++) {
        int64_t currentNum = (i == (loopCount - 1)) ? (blockLength_ - ubLength_ * i) : ubLength_;
        CopyIn(i, currentNum);
        Compute(currentNum);
        CopyOut(i, currentNum);
    }
}

// ============================================================================
// HardShrinkGradCastFp32: Cast to fp32 for compute (for fp16/bf16)
// On arch32, Compare API produces incorrect results on half type.
// Solution: Cast fp16->fp32, compute in fp32, Cast fp32->fp16.
// ============================================================================
template <typename T, int BUFFER_MODE>
class HardShrinkGradCastFp32 {
    static constexpr int32_t BUFFER_NUM = BUFFER_MODE ? 2 : 1;

public:
    __aicore__ inline HardShrinkGradCastFp32() {};

    __aicore__ inline void Init(GM_ADDR gradOutput, GM_ADDR self, GM_ADDR output,
                                 const HardShrinkGradTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum);
    __aicore__ inline void Compute(int64_t currentNum);
    __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum);

private:
    TPipe pipe;

    // Input/output queues in native type T (fp16)
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueGrad;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueSelf;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueue;

    // fp32 computation buffers (single-buffered)
    TBuf<QuePosition::VECCALC> gradFp32Buf;
    TBuf<QuePosition::VECCALC> selfFp32Buf;
    TBuf<QuePosition::VECCALC> outFp32Buf;
    TBuf<QuePosition::VECCALC> absBuf;      // fp32
    TBuf<QuePosition::VECCALC> lambdBuf;    // fp32
    TBuf<QuePosition::VECCALC> zeroBuf;     // fp32
    TBuf<QuePosition::VECCALC> cmpMaskBuf;

    GlobalTensor<T> inputGMGrad;
    GlobalTensor<T> inputGMSelf;
    GlobalTensor<T> outputGM;

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
    float lambd_ = 0.5f;
};

template <typename T, int BUFFER_MODE>
__aicore__ inline void HardShrinkGradCastFp32<T, BUFFER_MODE>::Init(
    GM_ADDR gradOutput, GM_ADDR self, GM_ADDR output,
    const HardShrinkGradTilingData* tilingData)
{
    int64_t remainderLength = tilingData->totalNum - tilingData->blockFactor * AscendC::GetBlockIdx();
    blockLength_ = (remainderLength > tilingData->blockFactor) ? tilingData->blockFactor : remainderLength;
    ubLength_ = tilingData->ubFactor;
    lambd_ = tilingData->lambd;

    int64_t offset = tilingData->blockFactor * AscendC::GetBlockIdx();
    inputGMGrad.SetGlobalBuffer((__gm__ T*)gradOutput + offset, blockLength_);
    inputGMSelf.SetGlobalBuffer((__gm__ T*)self + offset, blockLength_);
    outputGM.SetGlobalBuffer((__gm__ T*)output + offset, blockLength_);

    // Input/output queues in native type T (fp16)
    pipe.InitBuffer(inputQueueGrad, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(inputQueueSelf, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(outputQueue, BUFFER_NUM, ubLength_ * sizeof(T));

    // All computation buffers in fp32
    pipe.InitBuffer(gradFp32Buf, ubLength_ * sizeof(float));
    pipe.InitBuffer(selfFp32Buf, ubLength_ * sizeof(float));
    pipe.InitBuffer(outFp32Buf, ubLength_ * sizeof(float));
    pipe.InitBuffer(absBuf, ubLength_ * sizeof(float));
    pipe.InitBuffer(lambdBuf, ubLength_ * sizeof(float));
    pipe.InitBuffer(zeroBuf, ubLength_ * sizeof(float));

    int64_t cmpMaskBytes = ((ubLength_ + 7) / 8 + 255) / 256 * 256;
    if (cmpMaskBytes < 256) cmpMaskBytes = 256;
    pipe.InitBuffer(cmpMaskBuf, cmpMaskBytes);

    // Pre-fill constant buffers as fp32
    if (blockLength_ > 0) {
        LocalTensor<float> lambdLocal = lambdBuf.Get<float>();
        LocalTensor<float> zeroLocal = zeroBuf.Get<float>();
        Duplicate(lambdLocal, lambd_, ubLength_);
        Duplicate(zeroLocal, 0.0f, ubLength_);
        PipeBarrier<PIPE_V>();
    }
}

template <typename T, int BUFFER_MODE>
__aicore__ inline void HardShrinkGradCastFp32<T, BUFFER_MODE>::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> gradLocal = inputQueueGrad.template AllocTensor<T>();
    LocalTensor<T> selfLocal = inputQueueSelf.template AllocTensor<T>();

    DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = currentNum * sizeof(T);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;

    DataCopyPad(gradLocal, inputGMGrad[progress * ubLength_], copyParams, {false, 0, 0, 0});
    DataCopyPad(selfLocal, inputGMSelf[progress * ubLength_], copyParams, {false, 0, 0, 0});

    inputQueueGrad.EnQue(gradLocal);
    inputQueueSelf.EnQue(selfLocal);
}

template <typename T, int BUFFER_MODE>
__aicore__ inline void HardShrinkGradCastFp32<T, BUFFER_MODE>::Compute(int64_t currentNum)
{
    // Dequeue fp16 inputs
    LocalTensor<T> gradLocal = inputQueueGrad.template DeQue<T>();
    LocalTensor<T> selfLocal = inputQueueSelf.template DeQue<T>();
    LocalTensor<T> outLocal = outputQueue.template AllocTensor<T>();

    // Get fp32 computation buffers
    LocalTensor<float> gradFp32 = gradFp32Buf.Get<float>();
    LocalTensor<float> selfFp32 = selfFp32Buf.Get<float>();
    LocalTensor<float> outFp32 = outFp32Buf.Get<float>();
    LocalTensor<float> absLocal = absBuf.Get<float>();
    LocalTensor<float> lambdLocal = lambdBuf.Get<float>();
    LocalTensor<float> zeroLocal = zeroBuf.Get<float>();
    LocalTensor<uint8_t> cmpMask = cmpMaskBuf.Get<uint8_t>();

    int64_t computeNum = ubLength_;

    // Step 0: Cast T (fp16) -> fp32
    Cast(gradFp32, gradLocal, RoundMode::CAST_NONE, computeNum);
    Cast(selfFp32, selfLocal, RoundMode::CAST_NONE, computeNum);

    // Step 1: absLocal = |selfFp32|
    Abs(absLocal, selfFp32, computeNum);

    // Step 2: Compare(absLocal, lambdLocal, GT) -> cmpMask
    Compare(cmpMask, absLocal, lambdLocal, CMPMODE::GT, computeNum);

    // Step 3: Select(cmpMask, gradFp32, zeroLocal) -> outFp32
    Select(outFp32, cmpMask, gradFp32, zeroLocal, SELMODE::VSEL_TENSOR_TENSOR_MODE, computeNum);

    // Step 4: Cast fp32 -> T (fp16)
    Cast(outLocal, outFp32, RoundMode::CAST_ROUND, computeNum);

    outputQueue.template EnQue<T>(outLocal);
    inputQueueGrad.FreeTensor(gradLocal);
    inputQueueSelf.FreeTensor(selfLocal);
}

template <typename T, int BUFFER_MODE>
__aicore__ inline void HardShrinkGradCastFp32<T, BUFFER_MODE>::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> outLocal = outputQueue.template DeQue<T>();

    DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = currentNum * sizeof(T);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;

    DataCopyPad(outputGM[progress * ubLength_], outLocal, copyParams);
    outputQueue.FreeTensor(outLocal);
}

template <typename T, int BUFFER_MODE>
__aicore__ inline void HardShrinkGradCastFp32<T, BUFFER_MODE>::Process()
{
    if (blockLength_ <= 0) return;

    int64_t loopCount = (blockLength_ + ubLength_ - 1) / ubLength_;
    for (int64_t i = 0; i < loopCount; i++) {
        int64_t currentNum = (i == (loopCount - 1)) ? (blockLength_ - ubLength_ * i) : ubLength_;
        CopyIn(i, currentNum);
        Compute(currentNum);
        CopyOut(i, currentNum);
    }
}

} // namespace NsHardShrinkGrad
#endif // HARD_SHRINK_GRAD_H

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

#ifndef BNLL_H
#define BNLL_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "bnll_tiling_data.h"
#include "bnll_tiling_key.h"

namespace NsBnll {

using namespace AscendC;

template <typename T_IN, typename T_COMPUTE, int BUFFER_MODE>
class KernelBnll {
    static constexpr int32_t BUFFER_NUM = BUFFER_MODE ? 2 : 1;
    static constexpr bool NEED_CAST = !std::is_same<T_IN, T_COMPUTE>::value;

public:
    __aicore__ inline KernelBnll() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const BnllTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum);
    __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum);
    __aicore__ inline void Compute(int64_t currentNum);

private:
    TPipe pipe_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY_;

    // Work buffers for fp32 computation
    TBuf<QuePosition::VECCALC> workBuf1_;  // negX / expInput
    TBuf<QuePosition::VECCALC> workBuf2_;  // expResult / expPlusOne / lnResult
    TBuf<QuePosition::VECCALC> workBuf3_;  // zeroTensor / addResult
    TBuf<QuePosition::VECCALC> cmpBuf_;    // Compare mask (uint8_t)
    TBuf<QuePosition::VECCALC> castBuf_;   // Cast intermediate buffer (only for fp16/bf16)

    GlobalTensor<T_IN> xGm_;
    GlobalTensor<T_IN> yGm_;

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
};

template <typename T_IN, typename T_COMPUTE, int BUFFER_MODE>
__aicore__ inline void KernelBnll<T_IN, T_COMPUTE, BUFFER_MODE>::Init(
    GM_ADDR x, GM_ADDR y, const BnllTilingData* tilingData)
{
    int64_t startOffset = tilingData->blockFactor * static_cast<int64_t>(AscendC::GetBlockIdx());
    int64_t remaining = tilingData->totalNum - startOffset;
    if (remaining <= 0) {
        blockLength_ = 0;
        ubLength_ = 0;
        return;
    }
    blockLength_ = (remaining > tilingData->blockFactor) ? tilingData->blockFactor : remaining;
    ubLength_ = tilingData->ubFactor;

    xGm_.SetGlobalBuffer((__gm__ T_IN*)x + startOffset, blockLength_);
    yGm_.SetGlobalBuffer((__gm__ T_IN*)y + startOffset, blockLength_);

    // Input/output queues: allocate in input data type size for DMA
    pipe_.InitBuffer(inQueueX_, BUFFER_NUM, ubLength_ * static_cast<int64_t>(sizeof(T_IN)));
    pipe_.InitBuffer(outQueueY_, BUFFER_NUM, ubLength_ * static_cast<int64_t>(sizeof(T_IN)));

    // Work buffers: always in T_COMPUTE (float) size
    pipe_.InitBuffer(workBuf1_, ubLength_ * static_cast<int64_t>(sizeof(T_COMPUTE)));
    pipe_.InitBuffer(workBuf2_, ubLength_ * static_cast<int64_t>(sizeof(T_COMPUTE)));
    pipe_.InitBuffer(workBuf3_, ubLength_ * static_cast<int64_t>(sizeof(T_COMPUTE)));

    // Compare mask buffer: 1 bit per element -> N/8 bytes, align to 256 bytes
    int64_t cmpBufSize = ((ubLength_ / 8 + 255) / 256) * 256;
    if (cmpBufSize < 256) {
        cmpBufSize = 256;
    }
    pipe_.InitBuffer(cmpBuf_, cmpBufSize);

    // Cast buffer: only needed for fp16/bf16 paths
    if constexpr (NEED_CAST) {
        pipe_.InitBuffer(castBuf_, ubLength_ * static_cast<int64_t>(sizeof(T_COMPUTE)));
    }
}

template <typename T_IN, typename T_COMPUTE, int BUFFER_MODE>
__aicore__ inline void KernelBnll<T_IN, T_COMPUTE, BUFFER_MODE>::CopyIn(
    int64_t progress, int64_t currentNum)
{
    LocalTensor<T_IN> xLocal = inQueueX_.template AllocTensor<T_IN>();
    DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = static_cast<uint32_t>(currentNum * static_cast<int64_t>(sizeof(T_IN)));
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    DataCopyPadExtParams<T_IN> padParams{false, 0, 0, 0};
    DataCopyPad(xLocal, xGm_[progress * ubLength_], copyParams, padParams);
    inQueueX_.EnQue(xLocal);
}

template <typename T_IN, typename T_COMPUTE, int BUFFER_MODE>
__aicore__ inline void KernelBnll<T_IN, T_COMPUTE, BUFFER_MODE>::CopyOut(
    int64_t progress, int64_t currentNum)
{
    LocalTensor<T_IN> yLocal = outQueueY_.template DeQue<T_IN>();
    DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = static_cast<uint32_t>(currentNum * static_cast<int64_t>(sizeof(T_IN)));
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    DataCopyPad(yGm_[progress * ubLength_], yLocal, copyParams);
    outQueueY_.FreeTensor(yLocal);
}

template <typename T_IN, typename T_COMPUTE, int BUFFER_MODE>
__aicore__ inline void KernelBnll<T_IN, T_COMPUTE, BUFFER_MODE>::Compute(int64_t currentNum)
{
    LocalTensor<T_IN> xLocal = inQueueX_.template DeQue<T_IN>();
    LocalTensor<T_IN> yLocal = outQueueY_.template AllocTensor<T_IN>();

    // Get work buffers (always in T_COMPUTE = float)
    LocalTensor<T_COMPUTE> work1 = workBuf1_.Get<T_COMPUTE>();   // negX / expInput
    LocalTensor<T_COMPUTE> work2 = workBuf2_.Get<T_COMPUTE>();   // expResult / lnResult
    LocalTensor<T_COMPUTE> work3 = workBuf3_.Get<T_COMPUTE>();   // zeroTensor / addResult
    LocalTensor<uint8_t> cmpMask = cmpBuf_.Get<uint8_t>();

    uint32_t elemCount = static_cast<uint32_t>(currentNum);
    // Compare API requires count * sizeof(T_COMPUTE) to be 256-byte aligned.
    // For float (4 bytes): align to 64 elements. For half (2 bytes): align to 128 elements.
    // Buffer was allocated with ubLength_ which may be larger than currentNum,
    // so using aligned count is safe (extra elements are padding, result ignored).
    constexpr uint32_t COMPARE_ALIGN_ELEMENTS = 256 / static_cast<uint32_t>(sizeof(T_COMPUTE));
    uint32_t alignedCount = (elemCount + COMPARE_ALIGN_ELEMENTS - 1) / COMPARE_ALIGN_ELEMENTS * COMPARE_ALIGN_ELEMENTS;

    // Prepare input in T_COMPUTE precision
    LocalTensor<T_COMPUTE> inputFp32;
    LocalTensor<T_COMPUTE> outputFp32;
    if constexpr (NEED_CAST) {
        inputFp32 = castBuf_.Get<T_COMPUTE>();
        // Cast T_IN -> T_COMPUTE (fp16/bf16 -> fp32)
        Cast(inputFp32, xLocal, RoundMode::CAST_NONE, elemCount);
        // Use work3 as outputFp32 storage (will be reused after zero and compare)
        outputFp32 = work3;
    } else {
        // For float32, xLocal IS the fp32 input
        // We need to reinterpret since xLocal is T_IN which equals T_COMPUTE
        inputFp32 = xLocal.template ReinterpretCast<T_COMPUTE>();
        outputFp32 = yLocal.template ReinterpretCast<T_COMPUTE>();
    }

    // Step 1: Duplicate zero constant for Compare
    // Use work3 temporarily for zero (will be overwritten by addResult later)
    LocalTensor<T_COMPUTE> zeroTensor = work3;
    Duplicate<T_COMPUTE>(zeroTensor, static_cast<T_COMPUTE>(0), elemCount);

    // Step 2: negX = inputFp32 * (-1)
    Muls<T_COMPUTE>(work1, inputFp32, static_cast<T_COMPUTE>(-1), elemCount);

    // Step 3: Compare(inputFp32, zero, GT) -> cmpMask
    // Compare requires count * sizeof(T_COMPUTE) to be 256-byte aligned
    Compare(cmpMask, inputFp32, zeroTensor, CMPMODE::GT, alignedCount);

    // Step 4: Select expInput: x>0 ? negX : x
    Select(work2, cmpMask, work1, inputFp32, SELMODE::VSEL_TENSOR_TENSOR_MODE, elemCount);

    // Step 5: Exp(expInput) -> expResult (store in work1, reuse)
    Exp(work1, work2, elemCount);

    // Step 6: expResult + 1.0 -> expPlusOne (store in work1, in-place)
    Adds(work1, work1, static_cast<T_COMPUTE>(1), elemCount);

    // Step 7: Ln(expPlusOne) -> lnResult (store in work2)
    Ln(work2, work1, elemCount);

    // Step 8: inputFp32 + lnResult -> addResult (store in work3)
    Add(work3, inputFp32, work2, elemCount);

    // Step 9: Select: x>0 ? addResult : lnResult -> output
    if constexpr (NEED_CAST) {
        Select(work1, cmpMask, work3, work2, SELMODE::VSEL_TENSOR_TENSOR_MODE, elemCount);
        // Cast T_COMPUTE -> T_IN (fp32 -> fp16/bf16)
        Cast(yLocal, work1, RoundMode::CAST_ROUND, elemCount);
    } else {
        // For fp32: select directly into outputFp32
        Select(outputFp32, cmpMask, work3, work2, SELMODE::VSEL_TENSOR_TENSOR_MODE, elemCount);
    }

    outQueueY_.template EnQue<T_IN>(yLocal);
    inQueueX_.FreeTensor(xLocal);
}

template <typename T_IN, typename T_COMPUTE, int BUFFER_MODE>
__aicore__ inline void KernelBnll<T_IN, T_COMPUTE, BUFFER_MODE>::Process()
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

} // namespace NsBnll

#endif // BNLL_H

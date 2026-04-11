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
 * \file softplus_v2.h
 * \brief SoftplusV2 kernel implementation (arch32)
 *
 * Iteration 3: Full coverage (FP32 + FP16 + BF16 + edge cases)
 *
 * Edge case handling:
 *   - Empty tensor (0 elements): early return in Process/Init
 *   - Scalar input (0-dim): handled by Host Tiling EnsureNotScalar -> {1}
 *   - Non-aligned data: DataCopyPad handles non-32B alignment
 *   - beta=0: invBeta = +inf, output = inf for all elements (aligned with PyTorch)
 *   - NaN input: propagated through Exp/Log/Select (IEEE 754 behavior, same as PyTorch)
 *
 * TilingKey_0 (FP32 direct):
 *   - 5 buffers + cmpMask
 *   - Compute chain: Muls(beta*x) -> Duplicate+Compare(<=threshold) -> Exp -> Adds(+1) -> Log -> Muls(invBeta) -> Select
 *
 * TilingKey_1 (FP16 cast to FP32) / TilingKey_2 (BF16 cast to FP32):
 *   - 4 buffers (in/out depth=2 at sizeof(T)=2) + castBuf(fp32) + tmpBuf(fp32) + cmpMask
 *   - Key probe findings applied:
 *     1. Compare does not support scalar; use Duplicate + two-tensor Compare
 *     2. Select<bfloat16_t> not available on dav_c220; Select performed in fp32 space
 *     3. Two-pass Cast approach: after Duplicate overwrites castBuf, re-Cast from inQueue to recover fp32 x
 *     4. UB capacity accounts for TPipe overhead (~2KB safety margin via 8KB Select reserve)
 */

#ifndef SOFTPLUS_V2_H
#define SOFTPLUS_V2_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "softplus_v2_tiling_data.h"
#include "softplus_v2_tiling_key.h"

namespace NsSoftplusV2 {

using namespace AscendC;

template <typename T>
class SoftplusV2 {
    static constexpr int32_t BUFFER_NUM = 2; // Double Buffer

public:
    __aicore__ inline SoftplusV2() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const SoftplusV2TilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum);
    __aicore__ inline void Compute(int64_t currentNum);
    __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum);

    // FP32 direct compute path (TilingKey_0)
    __aicore__ inline void ComputeFp32(AscendC::LocalTensor<float>& xLocal,
                                       AscendC::LocalTensor<float>& yLocal,
                                       int64_t currentNum);

    // FP16/BF16 cast-to-fp32 compute path (TilingKey_1 / TilingKey_2)
    // Uses two-pass Cast approach validated by probe results
    __aicore__ inline void ComputeCast(AscendC::LocalTensor<T>& xLocal,
                                       AscendC::LocalTensor<T>& yLocal,
                                       int64_t currentNum);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue;
    TBuf<TPosition::VECCALC> tmpBuf;
    TBuf<TPosition::VECCALC> cmpMaskBuf;
    TBuf<TPosition::VECCALC> castBuf;  // FP16/BF16 only: fp32 workspace for Cast results

    GlobalTensor<T> xGm;
    GlobalTensor<T> yGm;

    int64_t blockLength_ = 0;
    int64_t tileLength_ = 0;
    float beta_ = 1.0f;
    float threshold_ = 20.0f;
    float invBeta_ = 1.0f;
};

template <typename T>
__aicore__ inline void SoftplusV2<T>::Init(GM_ADDR x, GM_ADDR y, const SoftplusV2TilingData* tilingData)
{
    int64_t blockIdx = AscendC::GetBlockIdx();
    int64_t remainderLength = tilingData->totalLength - tilingData->blockFactor * blockIdx;
    blockLength_ = (remainderLength > tilingData->blockFactor) ? tilingData->blockFactor : remainderLength;
    tileLength_ = tilingData->tileLength;
    beta_ = tilingData->beta;
    threshold_ = tilingData->threshold;
    invBeta_ = tilingData->invBeta;

    xGm.SetGlobalBuffer((__gm__ T*)x + tilingData->blockFactor * blockIdx, blockLength_);
    yGm.SetGlobalBuffer((__gm__ T*)y + tilingData->blockFactor * blockIdx, blockLength_);

    // Skip buffer allocation for empty tensor (blockLength == 0 or tileLength == 0)
    if (blockLength_ <= 0 || tileLength_ <= 0) {
        return;
    }

    // Init buffers based on data type
    if constexpr (std::is_same_v<T, float>) {
        // FP32 path: inQueue(2*tile*4) + outQueue(2*tile*4) + tmpBuf(tile*4) + cmpMask
        // castBuf not used for FP32 path, allocate minimum to avoid uninitialized member
        pipe.InitBuffer(inQueue, BUFFER_NUM, tileLength_ * sizeof(float));
        pipe.InitBuffer(outQueue, BUFFER_NUM, tileLength_ * sizeof(float));
        pipe.InitBuffer(tmpBuf, tileLength_ * sizeof(float));
        uint32_t cmpMaskBytes = static_cast<uint32_t>(((tileLength_ / 8 + 255) / 256) * 256);
        pipe.InitBuffer(cmpMaskBuf, cmpMaskBytes);
    } else {
        // FP16/BF16 path: inQueue(2*tile*2) + outQueue(2*tile*2) + castBuf(tile*4) + tmpBuf(tile*4) + cmpMask
        // Total: 4*tile + 4*tile + 4*tile + 4*tile + cmpMask = 16*tile + cmpMask
        pipe.InitBuffer(inQueue, BUFFER_NUM, tileLength_ * sizeof(T));
        pipe.InitBuffer(outQueue, BUFFER_NUM, tileLength_ * sizeof(T));
        pipe.InitBuffer(castBuf, tileLength_ * sizeof(float));
        pipe.InitBuffer(tmpBuf, tileLength_ * sizeof(float));
        uint32_t cmpMaskBytes = static_cast<uint32_t>(((tileLength_ / 8 + 255) / 256) * 256);
        pipe.InitBuffer(cmpMaskBuf, cmpMaskBytes);
    }
}

template <typename T>
__aicore__ inline void SoftplusV2<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    AscendC::LocalTensor<T> xLocal = inQueue.template AllocTensor<T>();
    AscendC::DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = static_cast<uint32_t>(currentNum * sizeof(T));
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    AscendC::DataCopyPad(xLocal, xGm[progress * tileLength_], copyParams, {false, 0, 0, 0});
    inQueue.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void SoftplusV2<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    AscendC::LocalTensor<T> yLocal = outQueue.template DeQue<T>();
    AscendC::DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = static_cast<uint32_t>(currentNum * sizeof(T));
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    AscendC::DataCopyPad(yGm[progress * tileLength_], yLocal, copyParams);
    outQueue.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void SoftplusV2<T>::ComputeFp32(AscendC::LocalTensor<float>& xLocal,
                                                   AscendC::LocalTensor<float>& yLocal,
                                                   int64_t currentNum)
{
    AscendC::LocalTensor<float> tmp = tmpBuf.Get<float>();
    AscendC::LocalTensor<uint8_t> cmpMask = cmpMaskBuf.Get<uint8_t>();

    // Align count to 64 elements for Compare 256B alignment
    int64_t alignedCount = ((currentNum + 63) / 64) * 64;
    uint32_t computeCount = static_cast<uint32_t>(
        (alignedCount > tileLength_) ? tileLength_ : alignedCount);

    // Step 1: tmp = beta * x
    AscendC::Muls<float>(tmp, xLocal, beta_, computeCount);

    // Step 2: Compare(cmpMask, tmp, thresholdTensor, LE) -- beta*x <= threshold
    // Probe finding: Compare does not support scalar; use Duplicate + two-tensor Compare
    AscendC::Duplicate<float>(yLocal, threshold_, computeCount);
    AscendC::Compare<float>(cmpMask, tmp, yLocal, AscendC::CMPMODE::LE, computeCount);

    // Step 3: tmp = exp(beta * x)
    AscendC::Exp<float>(tmp, tmp, computeCount);

    // Step 4: tmp = 1 + exp(beta * x)
    AscendC::Adds<float>(tmp, tmp, 1.0f, computeCount);

    // Step 5: tmp = ln(1 + exp(beta * x))
    AscendC::Log<float>(tmp, tmp, computeCount);

    // Step 6: tmp = (1/beta) * ln(1 + exp(beta * x))
    AscendC::Muls<float>(tmp, tmp, invBeta_, computeCount);

    // Step 7: y = Select(cmpMask ? tmp : x)
    // Compare LE: bit=1 when beta*x <= threshold -> select softplus (src0=tmp)
    //             bit=0 when beta*x >  threshold -> select x (src1=xLocal)
    AscendC::Select<float>(yLocal, cmpMask, tmp, xLocal,
                           AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE,
                           computeCount);
}

template <typename T>
__aicore__ inline void SoftplusV2<T>::ComputeCast(AscendC::LocalTensor<T>& xLocal,
                                                   AscendC::LocalTensor<T>& yLocal,
                                                   int64_t currentNum)
{
    // FP16/BF16 path: Cast to fp32, compute in fp32, Cast back
    // Validated by fp16_cast_probe and bf16_cast_probe (both successful)
    //
    // Key probe findings applied:
    //   1. Compare requires two tensors -- use Duplicate to fill threshold into castLocal
    //   2. Select<bfloat16_t> not available on dav_c220 -- Select in fp32 space
    //   3. After Duplicate overwrites castLocal, re-Cast from xLocal to recover fp32 x
    //   4. UB capacity includes TPipe overhead via 8KB Select reserve in Tiling

    AscendC::LocalTensor<float> castLocal = castBuf.Get<float>();
    AscendC::LocalTensor<float> tmp = tmpBuf.Get<float>();
    AscendC::LocalTensor<uint8_t> cmpMask = cmpMaskBuf.Get<uint8_t>();

    // Align count to 64 elements for Compare 256B alignment (fp32 compute)
    int64_t alignedCount = ((currentNum + 63) / 64) * 64;
    uint32_t computeCount = static_cast<uint32_t>(
        (alignedCount > tileLength_) ? tileLength_ : alignedCount);

    // Step 1: Cast T -> fp32
    AscendC::Cast<float, T>(castLocal, xLocal, AscendC::RoundMode::CAST_NONE, computeCount);

    // Step 2: tmp = beta * x (in fp32)
    AscendC::Muls<float>(tmp, castLocal, beta_, computeCount);

    // Step 3: Duplicate threshold into castLocal (temporarily overwrites fp32 x)
    // Probe finding: Compare does not support scalar parameter
    AscendC::Duplicate<float>(castLocal, threshold_, computeCount);

    // Step 4: Compare: beta*x <= threshold (two-tensor version)
    AscendC::Compare<float>(cmpMask, tmp, castLocal, AscendC::CMPMODE::LE, computeCount);

    // Step 5: Exp(beta*x)
    AscendC::Exp<float>(tmp, tmp, computeCount);

    // Step 6: 1 + exp(beta*x)
    AscendC::Adds<float>(tmp, tmp, 1.0f, computeCount);

    // Step 7: ln(1 + exp(beta*x))
    AscendC::Log<float>(tmp, tmp, computeCount);

    // Step 8: (1/beta) * ln(...)
    AscendC::Muls<float>(tmp, tmp, invBeta_, computeCount);

    // Step 9: Re-Cast T -> fp32 to recover original x (castLocal was overwritten by Duplicate)
    // Probe finding: two-pass Cast approach -- second Cast restores fp32 x for Select
    AscendC::Cast<float, T>(castLocal, xLocal, AscendC::RoundMode::CAST_NONE, computeCount);

    // Step 10: Select in fp32 space
    // Probe finding: Select<bfloat16_t> not available on dav_c220, must do Select in fp32
    // bit=1 -> src0(tmp, softplus result), bit=0 -> src1(castLocal, original x)
    AscendC::Select<float>(castLocal, cmpMask, tmp, castLocal,
                           AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE,
                           computeCount);

    // Step 11: Cast fp32 -> T (output)
    AscendC::Cast<T, float>(yLocal, castLocal, AscendC::RoundMode::CAST_ROUND, computeCount);
}

template <typename T>
__aicore__ inline void SoftplusV2<T>::Compute(int64_t currentNum)
{
    AscendC::LocalTensor<T> xLocal = inQueue.template DeQue<T>();
    AscendC::LocalTensor<T> yLocal = outQueue.template AllocTensor<T>();

    if constexpr (std::is_same_v<T, float>) {
        ComputeFp32(xLocal, yLocal, currentNum);
    } else {
        // FP16 (half) and BF16 (bfloat16_t) both use the cast-to-fp32 path
        ComputeCast(xLocal, yLocal, currentNum);
    }

    outQueue.template EnQue<T>(yLocal);
    inQueue.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void SoftplusV2<T>::Process()
{
    // Guard against empty tensor or zero tileLength (set by Host when totalLength==0)
    if (blockLength_ <= 0 || tileLength_ <= 0) {
        return;
    }
    int64_t loopCount = (blockLength_ + tileLength_ - 1) / tileLength_;
    for (int64_t i = 0; i < loopCount; i++) {
        int64_t currentNum = (i == (loopCount - 1)) ? (blockLength_ - tileLength_ * i) : tileLength_;
        CopyIn(i, currentNum);
        Compute(currentNum);
        CopyOut(i, currentNum);
    }
}

} // namespace NsSoftplusV2

#endif // SOFTPLUS_V2_H

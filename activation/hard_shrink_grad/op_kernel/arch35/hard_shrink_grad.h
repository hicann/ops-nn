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
 * \file hard_shrink_grad.h
 * \brief HardShrinkGrad Kernel class (arch35)
 *
 * Template parameters:
 *   - T: data type (half / float / bfloat16_t)
 *   - BUFFER_MODE: buffer mode (0=single, 1=double)
 *
 * Computation: dx = (|x| <= lambd) ? 0 : dy
 *
 * Strategy (aligned with probe-validated patterns):
 *   - fp16: Abs/Compares/Select directly on half (probe_fp16_sb validated)
 *   - fp32: Abs/Compares/Select directly on float (probe_fp32_db validated)
 *   - bf16: Cast bf16->float, Abs/Compares/Select in float, Cast back (probe_bf16_db validated)
 *
 * Uses Compares (scalar comparison) + Select (VSEL_TENSOR_SCALAR_MODE).
 */

#ifndef HARD_SHRINK_GRAD_H
#define HARD_SHRINK_GRAD_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "hard_shrink_grad_tiling_data.h"
#include "hard_shrink_grad_tiling_key.h"

namespace NsHardShrinkGrad {

using namespace AscendC;

// bf16 needs Cast to fp32 for computation on arch35; fp16/fp32 compute directly
template <typename T>
static constexpr bool NEEDS_FP32_CAST = std::is_same_v<T, bfloat16_t>;

// ComputeT: the type used for vector ops (Abs/Compares/Select)
template <typename T>
using ComputeType = std::conditional_t<std::is_same_v<T, bfloat16_t>, float, T>;

template <typename T, int BUFFER_MODE>
class HardShrinkGrad {
    static constexpr int32_t BUFFER_NUM = BUFFER_MODE ? 2 : 1;
    static constexpr bool CAST_TO_FP32 = NEEDS_FP32_CAST<T>;
    using ComputeT = ComputeType<T>;

public:
    __aicore__ inline HardShrinkGrad() {};

    __aicore__ inline void Init(GM_ADDR selfGrad, GM_ADDR self, GM_ADDR out,
                                const HardShrinkGradTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum);
    __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum);
    __aicore__ inline void Compute(int64_t currentNum);

private:
    TPipe pipe_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueGrad_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueSelf_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueue_;

    // Mask buffer for Compares output (bitmask: 1 bit per element)
    TBuf<TPosition::VECCALC> maskBuf_;

    // Extra fp32 buffers for bf16 cast path only
    TBuf<TPosition::VECCALC> selfFloatBuf_;
    TBuf<TPosition::VECCALC> gradFloatBuf_;

    GlobalTensor<T> gmGrad_;
    GlobalTensor<T> gmSelf_;
    GlobalTensor<T> gmOut_;

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
    float lambd_ = 0.0f;
};

template <typename T, int BUFFER_MODE>
__aicore__ inline void HardShrinkGrad<T, BUFFER_MODE>::Init(
    GM_ADDR selfGrad, GM_ADDR self, GM_ADDR out,
    const HardShrinkGradTilingData* tilingData)
{
    int64_t remainderLength = tilingData->totalNum - tilingData->blockFactor * AscendC::GetBlockIdx();
    blockLength_ = (remainderLength > tilingData->blockFactor) ? tilingData->blockFactor : remainderLength;
    ubLength_ = tilingData->ubFactor;
    lambd_ = tilingData->lambd;

    int64_t offset = tilingData->blockFactor * AscendC::GetBlockIdx();
    gmGrad_.SetGlobalBuffer((__gm__ T*)selfGrad + offset, blockLength_);
    gmSelf_.SetGlobalBuffer((__gm__ T*)self + offset, blockLength_);
    gmOut_.SetGlobalBuffer((__gm__ T*)out + offset, blockLength_);

    // Input/output queues in native type T
    pipe_.InitBuffer(inputQueueGrad_, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe_.InitBuffer(inputQueueSelf_, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe_.InitBuffer(outputQueue_, BUFFER_NUM, ubLength_ * sizeof(T));

    // Mask buffer: ubLength/8 bytes, 32-byte aligned
    int64_t maskBytes = (ubLength_ / 8 + 31) & ~31;
    if (maskBytes < 32) maskBytes = 32;
    pipe_.InitBuffer(maskBuf_, maskBytes);

    // bf16 path: extra fp32 intermediate buffers (NOT double buffered)
    if constexpr (CAST_TO_FP32) {
        pipe_.InitBuffer(selfFloatBuf_, ubLength_ * sizeof(float));
        pipe_.InitBuffer(gradFloatBuf_, ubLength_ * sizeof(float));
    }
}

template <typename T, int BUFFER_MODE>
__aicore__ inline void HardShrinkGrad<T, BUFFER_MODE>::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> gradLocal = inputQueueGrad_.template AllocTensor<T>();
    LocalTensor<T> selfLocal = inputQueueSelf_.template AllocTensor<T>();

    DataCopyPad(gradLocal, gmGrad_[progress * ubLength_],
        {1, static_cast<uint16_t>(currentNum * sizeof(T)), 0, 0},
        {false, 0, 0, 0});
    DataCopyPad(selfLocal, gmSelf_[progress * ubLength_],
        {1, static_cast<uint16_t>(currentNum * sizeof(T)), 0, 0},
        {false, 0, 0, 0});

    inputQueueGrad_.EnQue(gradLocal);
    inputQueueSelf_.EnQue(selfLocal);
}

template <typename T, int BUFFER_MODE>
__aicore__ inline void HardShrinkGrad<T, BUFFER_MODE>::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> outLocal = outputQueue_.template DeQue<T>();

    DataCopyPad(gmOut_[progress * ubLength_], outLocal,
        {1, static_cast<uint16_t>(currentNum * sizeof(T)), 0, 0});

    outputQueue_.FreeTensor(outLocal);
}

template <typename T, int BUFFER_MODE>
__aicore__ inline void HardShrinkGrad<T, BUFFER_MODE>::Compute(int64_t currentNum)
{
    LocalTensor<T> gradLocal = inputQueueGrad_.template DeQue<T>();
    LocalTensor<T> selfLocal = inputQueueSelf_.template DeQue<T>();
    LocalTensor<T> outLocal = outputQueue_.template AllocTensor<T>();
    LocalTensor<uint8_t> maskLocal = maskBuf_.Get<uint8_t>();

    // Use aligned ubLength_ for vector ops (ubFactor is pre-aligned by tiling)
    int64_t computeNum = ubLength_;

    if constexpr (CAST_TO_FP32) {
        // bf16 path: Cast to fp32, compute in fp32, Cast back to bf16
        LocalTensor<float> selfFloatLocal = selfFloatBuf_.Get<float>();
        LocalTensor<float> gradFloatLocal = gradFloatBuf_.Get<float>();

        // Step 1: Cast bf16 -> float
        Cast(selfFloatLocal, selfLocal, RoundMode::CAST_NONE, computeNum);
        Cast(gradFloatLocal, gradLocal, RoundMode::CAST_NONE, computeNum);

        // Step 2: Abs in-place on selfFloat
        Abs(selfFloatLocal, selfFloatLocal, computeNum);

        // Step 3: Compares |x| <= lambd -> bitmask (mask=1 means dead zone)
        Compares(maskLocal, selfFloatLocal, lambd_, CMPMODE::LE, computeNum);

        // Step 4: Select - mask=1 (dead zone) picks 0, mask=0 picks gradFloat
        // Fill selfFloatLocal with 0 (reuse as zero tensor, no longer needed)
        Duplicate(selfFloatLocal, 0.0f, computeNum);
        Select(gradFloatLocal, maskLocal, selfFloatLocal, gradFloatLocal,
               SELMODE::VSEL_TENSOR_TENSOR_MODE, computeNum);

        // Step 5: Cast float -> bf16
        Cast(outLocal, gradFloatLocal, RoundMode::CAST_RINT, computeNum);
    } else {
        // fp16/fp32 direct path: Abs/Compares/Select on native type

        // Step 1: Abs in-place on selfLocal
        Abs(selfLocal, selfLocal, computeNum);

        // Step 2: Compares |x| <= lambd -> bitmask (mask=1 means dead zone)
        ComputeT lambdScalar = static_cast<ComputeT>(lambd_);
        Compares(maskLocal, selfLocal, lambdScalar, CMPMODE::LE, computeNum);

        // Step 3: Select - mask=1 (dead zone) picks 0, mask=0 picks grad
        // Fill selfLocal with 0 (reuse as zero tensor, no longer needed after Abs)
        Duplicate(selfLocal, static_cast<ComputeT>(0.0f), computeNum);
        Select(outLocal, maskLocal, selfLocal, gradLocal,
               SELMODE::VSEL_TENSOR_TENSOR_MODE, computeNum);
    }

    outputQueue_.template EnQue<T>(outLocal);
    inputQueueGrad_.FreeTensor(gradLocal);
    inputQueueSelf_.FreeTensor(selfLocal);
}

template <typename T, int BUFFER_MODE>
__aicore__ inline void HardShrinkGrad<T, BUFFER_MODE>::Process()
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

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
 * \file situ_glu_grad.hpp
 * \brief SiTU gated linear unit backward (float32, front-back split).
 *
 * Single template class covering all three dtypes:
 *   - SituGluGradBase<float>        -> fp32 native compute (no Cast)
 *   - SituGluGradBase<half>         -> Cast->float32 compute -> Cast back (CAST_NONE)
 *   - SituGluGradBase<bfloat16_t>   -> Cast->float32 compute -> Cast back (CAST_RINT)
 *
 * Given grad_y (shape [..., h]) and x (shape [..., 2h]), compute grad_x (shape [..., 2h]).
 *
 * Forward:  y = situ_a(gate) * up'(up)
 * Backward: grad_gate = grad_situ_a * s * [(1-t^2) + beta*t*(1-s)]
 *           grad_up    = grad_up_prime * (1 - tanh^2(up/linear_beta))   (linear_beta > 0)
 *           grad_up    = grad_up_prime                                  (linear_beta <= 0)
 *           where:  t = tanh(gate/beta),  s = sigmoid(gate)
 *                    situ_a = beta*t*s,  up' = linear_beta*tanh(up/linear_beta) or up
 *                    grad_situ_a = grad_y * up',  grad_up_prime = grad_y * situ_a
 */
#ifndef OPP_SITU_GLU_GRAD_HPP
#define OPP_SITU_GLU_GRAD_HPP
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"

namespace SituGluGradOps {
using namespace AscendC;
constexpr static int64_t DB_BUFFER = 2;
constexpr static int64_t BLOCK_SIZE = 32;
constexpr static int64_t BLOCK_ELEM = BLOCK_SIZE / sizeof(float);
constexpr static int64_t SWI_FACTOR = 2;
constexpr static int64_t NUM_TMP = 5; // tmp1..tmp5

// Type trait: true when T is bfloat16_t.
// bf16 requires CAST_RINT on all platforms; CAST_NONE produces garbage on non-A2 (e.g. Ascend950).
template <typename T>
struct SituIsBf16 : std::false_type {};
template <>
struct SituIsBf16<bfloat16_t> : std::true_type {};

template <typename T>
class SituGluGradBase {
    static constexpr bool kIsFp32 = std::is_same<T, float>::value;

public:
    __aicore__ inline SituGluGradBase(TPipe* pipe) { pipe_ = pipe; };
    __aicore__ inline void Init(GM_ADDR gradY, GM_ADDR x, GM_ADDR gradX, const SituGluGradTilingData* tilingData);
    __aicore__ inline int64_t AlignBytes(int64_t number);
    __aicore__ inline void Process();
    __aicore__ inline void CalTilingParam();
    __aicore__ inline void ProcessSingleLoop(int64_t gradYOffset, int64_t xOffset);
    __aicore__ inline void CopyIn(int64_t gradYOffset, int64_t xOffset);
    __aicore__ inline void CopyInShortH(LocalTensor<T>& gradYLocal, LocalTensor<T>& xLocal, int64_t gradYOffset,
                                        int64_t xOffset);
    __aicore__ inline void CopyInLongH(LocalTensor<T>& gradYLocal, LocalTensor<T>& xLocal, int64_t gradYOffset,
                                       int64_t xOffset);
    __aicore__ inline void Compute(LocalTensor<T>& gradYLocal, LocalTensor<T>& xLocal, LocalTensor<float>& tmp1,
                                   LocalTensor<float>& tmp2, LocalTensor<float>& tmp3, LocalTensor<float>& tmp4,
                                   LocalTensor<float>& tmp5, LocalTensor<T>& gradXLocal);
    __aicore__ inline void CopyOut(int64_t xOffset);

private:
    // Common backward compute body. All operands are LocalTensor<float>.
    // gate/up are the forward gate/up values (float).
    // gradY is grad_y (float); reused internally as scratch (1-s, term2, t_u).
    // gradGateOut / gradUpOut receive the final grad_gate / grad_up (float).
    // In the mixed path, gateF is passed as both gate and gradGateOut (safe:
    // gate input is fully consumed before gradGateOut is written).
    __aicore__ inline void GradCore(LocalTensor<float> gate, LocalTensor<float> up, LocalTensor<float> gradY,
                                    LocalTensor<float> tmp1, LocalTensor<float> tmp2, LocalTensor<float> tmp3,
                                    LocalTensor<float> tmp4, LocalTensor<float> tmp5, LocalTensor<float> gradGateOut,
                                    LocalTensor<float> gradUpOut, int64_t n);

protected:
    GlobalTensor<T> gradYGm_;
    GlobalTensor<T> xGm_;
    GlobalTensor<T> gradXGm_;

    TPipe* pipe_ = nullptr;
    TQue<QuePosition::VECIN, DB_BUFFER> gradYQueue_;
    TQue<QuePosition::VECIN, DB_BUFFER> xQueue_;
    TQue<QuePosition::VECOUT, 1> gradXQueue_;
    TBuf<TPosition::VECCALC> tmpBufs_[NUM_TMP];
    // Cast-path scratch (only InitBuffer'd when T != float; declared unconditionally
    // because TBuf is a thin handle and unused declarations have no UB footprint).
    TBuf<TPosition::VECCALC> gateFBuf_;
    TBuf<TPosition::VECCALC> upFBuf_;
    TBuf<TPosition::VECCALC> gradYFBuf_;

    uint32_t blockIdx_ = GetBlockIdx();
    uint32_t usedCoreNum_ = 0;
    int64_t dimBatchSize_ = 0;
    int64_t dim2H_ = 0;
    int64_t dimH_ = 0;
    int64_t isLongH_ = 0;
    int64_t ubMaxPair_ = 0;
    int64_t blockOffset_ = 0;
    int64_t loopOffset_ = 0;
    int64_t loopTime_ = 0;
    int64_t pairFrontLoop_ = 0;
    int64_t pairLastLoop_ = 0;
    int64_t pairNum_ = 0;
    int64_t batchPreBlock_ = 0;
    int64_t halfOff_ = 0;
    int64_t xQueSpace_ = 0;
    int64_t gradXQueSpace_ = 0;
    float beta_ = 1.0f;
    float linearBeta_ = 0.0f;
    int64_t activateLeft_ = 1;
    const SituGluGradTilingData* tl_ = nullptr;
};

template <typename T>
__aicore__ inline void SituGluGradBase<T>::Init(GM_ADDR gradY, GM_ADDR x, GM_ADDR gradX,
                                                const SituGluGradTilingData* tilingData)
{
    tl_ = tilingData;
    dimBatchSize_ = tl_->dimBatchSize;
    dim2H_ = tl_->dim2H;
    dimH_ = dim2H_ / SWI_FACTOR;
    isLongH_ = tl_->isLongH;
    ubMaxPair_ = tl_->ubMaxPair;
    beta_ = tl_->beta;
    linearBeta_ = tl_->linearBeta;
    activateLeft_ = tl_->activateLeft;

    xQueSpace_ = SWI_FACTOR * AlignBytes(ubMaxPair_ * static_cast<int64_t>(sizeof(T)));
    halfOff_ = xQueSpace_ / static_cast<int64_t>(sizeof(T)) / SWI_FACTOR;
    gradXQueSpace_ = SWI_FACTOR * AlignBytes(ubMaxPair_ * static_cast<int64_t>(sizeof(T)));

    gradYGm_.SetGlobalBuffer((__gm__ T*)gradY);
    xGm_.SetGlobalBuffer((__gm__ T*)x);
    gradXGm_.SetGlobalBuffer((__gm__ T*)gradX);

    pipe_->InitBuffer(gradYQueue_, DB_BUFFER, AlignBytes(ubMaxPair_ * static_cast<int64_t>(sizeof(T))));
    pipe_->InitBuffer(xQueue_, DB_BUFFER, xQueSpace_);
    pipe_->InitBuffer(gradXQueue_, 1, gradXQueSpace_);
    for (int64_t i = 0; i < NUM_TMP; ++i) {
        pipe_->InitBuffer(tmpBufs_[i], AlignBytes(ubMaxPair_ * static_cast<int64_t>(sizeof(float))));
    }
    if constexpr (!kIsFp32) {
        pipe_->InitBuffer(gateFBuf_, AlignBytes(ubMaxPair_ * static_cast<int64_t>(sizeof(float))));
        pipe_->InitBuffer(upFBuf_, AlignBytes(ubMaxPair_ * static_cast<int64_t>(sizeof(float))));
        pipe_->InitBuffer(gradYFBuf_, AlignBytes(ubMaxPair_ * static_cast<int64_t>(sizeof(float))));
    }
}

template <typename T>
__aicore__ inline int64_t SituGluGradBase<T>::AlignBytes(int64_t number)
{
    return (number + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
}

template <typename T>
__aicore__ inline void SituGluGradBase<T>::Process()
{
    if (dimBatchSize_ <= 0 || dim2H_ <= 0) {
        return;
    }
    CalTilingParam();

    if (blockIdx_ < usedCoreNum_) {
        if (isLongH_ == 1) {
            for (int64_t batchIdx = 0; batchIdx < batchPreBlock_; ++batchIdx) {
                int64_t xOffset = blockIdx_ * blockOffset_ + batchIdx * dim2H_;
                int64_t gradYOffset = blockIdx_ * blockOffset_ / SWI_FACTOR + batchIdx * dimH_;
                for (int64_t loopIdx = 0; loopIdx < loopTime_; ++loopIdx) {
                    pairNum_ = loopIdx == (loopTime_ - 1) ? pairLastLoop_ : pairFrontLoop_;
                    ProcessSingleLoop(gradYOffset, xOffset);
                    xOffset += loopOffset_;
                    gradYOffset += loopOffset_;
                }
            }
        } else {
            int64_t xOffset = blockIdx_ * blockOffset_;
            int64_t gradYOffset = blockIdx_ * blockOffset_ / SWI_FACTOR;
            for (int64_t loopIdx = 0; loopIdx < loopTime_; ++loopIdx) {
                pairNum_ = loopIdx == (loopTime_ - 1) ? pairLastLoop_ : pairFrontLoop_;
                ProcessSingleLoop(gradYOffset, xOffset);
                xOffset += loopOffset_;
                gradYOffset += loopOffset_ / SWI_FACTOR;
            }
        }
    }
}

template <typename T>
__aicore__ inline void SituGluGradBase<T>::CalTilingParam()
{
    int64_t coreNum = tl_->coreNumAll;
    if (coreNum <= 0) {
        coreNum = 1;
    }
    int64_t batchFrontBlock = (dimBatchSize_ + coreNum - 1) / coreNum;
    if (batchFrontBlock <= 0) {
        batchFrontBlock = 1;
    }
    usedCoreNum_ = static_cast<uint32_t>((dimBatchSize_ + batchFrontBlock - 1) / batchFrontBlock);
    int64_t batchLastBlock = dimBatchSize_ - batchFrontBlock * (usedCoreNum_ - 1);
    batchPreBlock_ = blockIdx_ == (usedCoreNum_ - 1) ? batchLastBlock : batchFrontBlock;
    blockOffset_ = batchFrontBlock * dim2H_;
    if (isLongH_ == 0) {
        int64_t ubMaxBatch = ubMaxPair_ / dimH_;
        if (ubMaxBatch < 1) {
            ubMaxBatch = 1;
        }
        loopTime_ = (batchPreBlock_ + ubMaxBatch - 1) / ubMaxBatch;
        int64_t batchLastLoop = batchPreBlock_ - ubMaxBatch * (loopTime_ - 1);
        pairFrontLoop_ = ubMaxBatch * dimH_;
        pairLastLoop_ = batchLastLoop * dimH_;
        loopOffset_ = ubMaxBatch * dim2H_;
    } else {
        loopTime_ = (dimH_ + ubMaxPair_ - 1) / ubMaxPair_;
        pairLastLoop_ = dimH_ - ubMaxPair_ * (loopTime_ - 1);
        pairFrontLoop_ = ubMaxPair_;
        loopOffset_ = ubMaxPair_;
    }
}

template <typename T>
__aicore__ inline void SituGluGradBase<T>::ProcessSingleLoop(int64_t gradYOffset, int64_t xOffset)
{
    CopyIn(gradYOffset, xOffset);
    LocalTensor<T> gradYLocal = gradYQueue_.DeQue<T>();
    LocalTensor<T> xLocal = xQueue_.DeQue<T>();
    LocalTensor<float> tmp1 = tmpBufs_[0].Get<float>();
    LocalTensor<float> tmp2 = tmpBufs_[1].Get<float>();
    LocalTensor<float> tmp3 = tmpBufs_[2].Get<float>();
    LocalTensor<float> tmp4 = tmpBufs_[3].Get<float>();
    LocalTensor<float> tmp5 = tmpBufs_[4].Get<float>();
    LocalTensor<T> gradXLocal = gradXQueue_.AllocTensor<T>();
    Compute(gradYLocal, xLocal, tmp1, tmp2, tmp3, tmp4, tmp5, gradXLocal);
    gradXQueue_.EnQue<T>(gradXLocal);
    CopyOut(xOffset);
}

template <typename T>
__aicore__ inline void SituGluGradBase<T>::CopyIn(int64_t gradYOffset, int64_t xOffset)
{
    LocalTensor<T> gradYLocal = gradYQueue_.AllocTensor<T>();
    LocalTensor<T> xLocal = xQueue_.AllocTensor<T>();
    if (isLongH_ == 0) {
        CopyInShortH(gradYLocal, xLocal, gradYOffset, xOffset);
    } else {
        CopyInLongH(gradYLocal, xLocal, gradYOffset, xOffset);
    }
    gradYQueue_.EnQue(gradYLocal);
    xQueue_.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void SituGluGradBase<T>::CopyInShortH(LocalTensor<T>& gradYLocal, LocalTensor<T>& xLocal,
                                                        int64_t gradYOffset, int64_t xOffset)
{
    DataCopyPadParams padParams{false, 0, 0, 0};
    // grad_y: contiguous rows of dimH
    DataCopyParams gradYParams;
    gradYParams.blockCount = pairNum_ / dimH_;
    gradYParams.blockLen = dimH_ * sizeof(T);
    gradYParams.srcStride = 0;
    gradYParams.dstStride = 0;
    DataCopyPad(gradYLocal, gradYGm_[gradYOffset], gradYParams, padParams);

    // x: gate + up halves (strided rows, skip up half between rows)
    DataCopyParams xParams;
    xParams.blockCount = pairNum_ / dimH_;
    xParams.blockLen = dimH_ * sizeof(T);
    xParams.srcStride = dimH_ * sizeof(T);
    xParams.dstStride = 0;
    DataCopyPad(xLocal, xGm_[xOffset], xParams, padParams);
    DataCopyPad(xLocal[halfOff_], xGm_[xOffset + dimH_], xParams, padParams);
}

template <typename T>
__aicore__ inline void SituGluGradBase<T>::CopyInLongH(LocalTensor<T>& gradYLocal, LocalTensor<T>& xLocal,
                                                       int64_t gradYOffset, int64_t xOffset)
{
    DataCopyPadParams padParams{false, 0, 0, 0};
    DataCopyParams gradYParams;
    gradYParams.blockCount = 1;
    gradYParams.blockLen = pairNum_ * sizeof(T);
    gradYParams.srcStride = 0;
    gradYParams.dstStride = 0;
    DataCopyPad(gradYLocal, gradYGm_[gradYOffset], gradYParams, padParams);

    DataCopyParams xParams;
    xParams.blockCount = 1;
    xParams.blockLen = pairNum_ * sizeof(T);
    xParams.srcStride = 0;
    xParams.dstStride = 0;
    DataCopyPad(xLocal, xGm_[xOffset], xParams, padParams);
    DataCopyPad(xLocal[halfOff_], xGm_[xOffset + dimH_], xParams, padParams);
}

// Common backward compute body. All operands are LocalTensor<float>.
// gate/up: forward gate/up (float). gradY: grad_y (float), reused as scratch.
// gradGateOut / gradUpOut: final grad_gate / grad_up (float).
template <typename T>
__aicore__ inline void SituGluGradBase<T>::GradCore(LocalTensor<float> gate, LocalTensor<float> up,
                                                    LocalTensor<float> gradY, LocalTensor<float> tmp1,
                                                    LocalTensor<float> tmp2, LocalTensor<float> tmp3,
                                                    LocalTensor<float> tmp4, LocalTensor<float> tmp5,
                                                    LocalTensor<float> gradGateOut, LocalTensor<float> gradUpOut,
                                                    int64_t n)
{
    // 1. t = tanh(gate / beta)
    Muls(tmp1, gate, 1.0f / beta_, n);
    PipeBarrier<PIPE_V>();
    Tanh(tmp1, tmp1, n);
    PipeBarrier<PIPE_V>();

    // 2. s = sigmoid(gate)
    Sigmoid(tmp2, gate, n);
    PipeBarrier<PIPE_V>();
    // tmp2 = s = sigmoid(gate)

    // 3. situ_a = beta * t * s
    Muls(tmp3, tmp1, beta_, n);
    PipeBarrier<PIPE_V>();
    Mul(tmp3, tmp3, tmp2, n);
    PipeBarrier<PIPE_V>();
    // tmp3 = situ_a

    // 4. up_prime
    if (linearBeta_ > 0.0f) {
        Muls(tmp4, up, 1.0f / linearBeta_, n);
        PipeBarrier<PIPE_V>();
        Tanh(tmp4, tmp4, n);
        PipeBarrier<PIPE_V>();
        Muls(tmp4, tmp4, linearBeta_, n);
        PipeBarrier<PIPE_V>();
        // tmp4 = up_prime
    }
    // else: up_prime = up (use up directly)

    // 5. grad_situ_a = grad_y * up_prime
    if (linearBeta_ > 0.0f) {
        Mul(tmp5, gradY, tmp4, n);
    } else {
        Mul(tmp5, gradY, up, n);
    }
    PipeBarrier<PIPE_V>();
    // tmp5 = grad_situ_a

    // 6. grad_up_prime = grad_y * situ_a
    Mul(tmp4, gradY, tmp3, n);
    PipeBarrier<PIPE_V>();
    // tmp4 = grad_up_prime, tmp3 free, grad_y no longer needed

    // 7. grad_gate = grad_situ_a * s * [(1-t^2) + beta*t*(1-s)]
    // (1 - t^2) in tmp3
    Mul(tmp3, tmp1, tmp1, n);
    PipeBarrier<PIPE_V>();
    Muls(tmp3, tmp3, -1.0f, n);
    PipeBarrier<PIPE_V>();
    Adds(tmp3, tmp3, 1.0f, n);
    PipeBarrier<PIPE_V>();
    // tmp3 = 1 - t^2

    // (1 - s) in gradY (reused after grad_y is consumed)
    Muls(gradY, tmp2, -1.0f, n);
    PipeBarrier<PIPE_V>();
    Adds(gradY, gradY, 1.0f, n);
    PipeBarrier<PIPE_V>();
    // gradY = 1 - s

    // term2 = beta * t * (1-s) in gradY
    Mul(gradY, gradY, tmp1, n);
    PipeBarrier<PIPE_V>();
    Muls(gradY, gradY, beta_, n);
    PipeBarrier<PIPE_V>();
    // gradY = beta * t * (1-s) = term2

    // bracket = (1-t^2) + term2 in tmp3
    Add(tmp3, tmp3, gradY, n);
    PipeBarrier<PIPE_V>();
    // tmp3 = bracket

    // grad_gate = grad_situ_a * s * bracket
    Mul(gradGateOut, tmp5, tmp2, n);
    PipeBarrier<PIPE_V>();
    Mul(gradGateOut, gradGateOut, tmp3, n);
    PipeBarrier<PIPE_V>();
    // gradGateOut = grad_gate

    // 8. grad_up
    if (linearBeta_ > 0.0f) {
        // t_u = tanh(up / linear_beta) in gradY (reused)
        Muls(gradY, up, 1.0f / linearBeta_, n);
        PipeBarrier<PIPE_V>();
        Tanh(gradY, gradY, n);
        PipeBarrier<PIPE_V>();
        // gradY = t_u

        // (1 - t_u^2) in tmp3 (reused)
        Mul(tmp3, gradY, gradY, n);
        PipeBarrier<PIPE_V>();
        Muls(tmp3, tmp3, -1.0f, n);
        PipeBarrier<PIPE_V>();
        Adds(tmp3, tmp3, 1.0f, n);
        PipeBarrier<PIPE_V>();
        // tmp3 = 1 - t_u^2

        Mul(gradUpOut, tmp4, tmp3, n);
        PipeBarrier<PIPE_V>();
    } else {
        Muls(gradUpOut, tmp4, 1.0f, n);
        PipeBarrier<PIPE_V>();
    }
    // gradUpOut = grad_up
}

template <typename T>
__aicore__ inline void SituGluGradBase<T>::Compute(LocalTensor<T>& gradYLocal, LocalTensor<T>& xLocal,
                                                   LocalTensor<float>& tmp1, LocalTensor<float>& tmp2,
                                                   LocalTensor<float>& tmp3, LocalTensor<float>& tmp4,
                                                   LocalTensor<float>& tmp5, LocalTensor<T>& gradXLocal)
{
    int64_t n = pairNum_;
    int64_t gateOff = (activateLeft_ == 1) ? 0 : halfOff_;
    int64_t upOff = (activateLeft_ == 1) ? halfOff_ : 0;

    if constexpr (kIsFp32) {
        // fp32 native path: gate/up from xLocal directly, gradY from gradYLocal,
        // grad_gate / grad_up written directly to gradXLocal
        GradCore(xLocal[gateOff], xLocal[upOff], gradYLocal, tmp1, tmp2, tmp3, tmp4, tmp5, gradXLocal[gateOff],
                 gradXLocal[upOff], n);
        gradYQueue_.FreeTensor(gradYLocal);
        xQueue_.FreeTensor(xLocal);
    } else {
        // mixed precision path (half / bf16): Cast T->float32, compute, Cast float32->T
        LocalTensor<float> gateF = gateFBuf_.Get<float>();
        LocalTensor<float> upF = upFBuf_.Get<float>();
        LocalTensor<float> gradYF = gradYFBuf_.Get<float>();

        // T -> float32 (lossless widening for both half and bf16)
        Cast(gateF, xLocal[gateOff], RoundMode::CAST_NONE, n);
        PipeBarrier<PIPE_V>();
        Cast(upF, xLocal[upOff], RoundMode::CAST_NONE, n);
        PipeBarrier<PIPE_V>();
        Cast(gradYF, gradYLocal, RoundMode::CAST_NONE, n);
        PipeBarrier<PIPE_V>();
        xQueue_.FreeTensor(xLocal);
        gradYQueue_.FreeTensor(gradYLocal);

        // grad_gate (float) reuses gateF, grad_up (float) reuses upF
        // (safe: gate input fully consumed before grad_gate output is written)
        GradCore(gateF, upF, gradYF, tmp1, tmp2, tmp3, tmp4, tmp5, gateF, upF, n);

        // float32 -> T
        // bf16 requires CAST_RINT on all platforms; CAST_NONE produces garbage on non-A2 (e.g. Ascend950)
        if constexpr (SituIsBf16<T>::value) {
            Cast(gradXLocal[gateOff], gateF, RoundMode::CAST_RINT, n);
        } else {
            Cast(gradXLocal[gateOff], gateF, RoundMode::CAST_NONE, n);
        }
        PipeBarrier<PIPE_V>();
        if constexpr (SituIsBf16<T>::value) {
            Cast(gradXLocal[upOff], upF, RoundMode::CAST_RINT, n);
        } else {
            Cast(gradXLocal[upOff], upF, RoundMode::CAST_NONE, n);
        }
        PipeBarrier<PIPE_V>();
    }
}

template <typename T>
__aicore__ inline void SituGluGradBase<T>::CopyOut(int64_t xOffset)
{
    LocalTensor<T> gradXLocal = gradXQueue_.DeQue<T>();
    DataCopyParams params;
    int64_t gateOff = (activateLeft_ == 1) ? 0 : halfOff_;
    int64_t upOff = (activateLeft_ == 1) ? halfOff_ : 0;
    int64_t gateGmOff = (activateLeft_ == 1) ? 0 : dimH_;
    int64_t upGmOff = (activateLeft_ == 1) ? dimH_ : 0;

    if (isLongH_ == 0) {
        params.blockCount = pairNum_ / dimH_;
        params.blockLen = dimH_ * sizeof(T);
        params.srcStride = 0;
        params.dstStride = dimH_ * sizeof(T);
        DataCopyPad(gradXGm_[xOffset + gateGmOff], gradXLocal[gateOff], params);
        DataCopyPad(gradXGm_[xOffset + upGmOff], gradXLocal[upOff], params);
    } else {
        params.blockCount = 1;
        params.blockLen = pairNum_ * sizeof(T);
        params.srcStride = 0;
        params.dstStride = 0;
        DataCopyPad(gradXGm_[xOffset + gateGmOff], gradXLocal[gateOff], params);
        DataCopyPad(gradXGm_[xOffset + upGmOff], gradXLocal[upOff], params);
    }
    gradXQueue_.FreeTensor(gradXLocal);
}

} // namespace SituGluGradOps
#endif // OPP_SITU_GLU_GRAD_HPP

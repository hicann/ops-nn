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
 * \file situ_glu.hpp
 * \brief SiTU gated linear unit (float32, front-back split).
 *
 * Single template class covering all three dtypes:
 *   - SituGluBase<float>        -> fp32 native compute (no Cast)
 *   - SituGluBase<half>         -> Cast->float32 compute -> Cast back (CAST_NONE)
 *   - SituGluBase<bfloat16_t>   -> Cast->float32 compute -> Cast back (CAST_RINT)
 *
 * Compute (per element pair, activate_left controls which half is the gate):
 *   situ_a = beta * tanh(gate / beta) * sigmoid(gate)
 *   if linear_beta > 0: up = linear_beta * tanh(up / linear_beta)
 *   y = situ_a * up
 */
#ifndef OPP_SITU_GLU_HPP
#define OPP_SITU_GLU_HPP
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"

namespace SituGluOps {
using namespace AscendC;
constexpr static int64_t DB_BUFFER = 2;
constexpr static int64_t BLOCK_SIZE = 32;
constexpr static int64_t BLOCK_ELEM = BLOCK_SIZE / sizeof(float);
constexpr static int64_t SWI_FACTOR = 2;

// Type trait: true when T is bfloat16_t.
// bf16 requires CAST_RINT on all platforms; CAST_NONE produces garbage on non-A2 (e.g. Ascend950).
template <typename T>
struct SituIsBf16 : std::false_type {};
template <>
struct SituIsBf16<bfloat16_t> : std::true_type {};

template <typename T>
class SituGluBase {
    static constexpr bool kIsFp32 = std::is_same<T, float>::value;

public:
    __aicore__ inline SituGluBase(TPipe* pipe) { pipe_ = pipe; };
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const SituGluTilingData* tilingData);
    __aicore__ inline int64_t AlignBytes(int64_t number);
    __aicore__ inline void Process();
    __aicore__ inline void CalTilingParam();
    __aicore__ inline void ProcessSingleLoop(int64_t xOffset, int64_t yOffset);
    __aicore__ inline void CopyIn(int64_t xOffset);
    __aicore__ inline void CopyInShortH(LocalTensor<T>& xLocal, int64_t xOffset);
    __aicore__ inline void CopyInLongH(LocalTensor<T>& xLocal, int64_t xOffset);
    __aicore__ inline void Compute(LocalTensor<T>& xLocal, LocalTensor<float>& tmp1, LocalTensor<float>& tmp2,
                                   LocalTensor<T>& yLocal);
    __aicore__ inline void CopyOut(int64_t yOffset);

private:
    __aicore__ inline void SituCore(LocalTensor<float> gate, LocalTensor<float> up, LocalTensor<float> tmp1,
                                    LocalTensor<float> tmp2, LocalTensor<float> y, int64_t n);

protected:
    /* global memory address */
    GlobalTensor<T> xGm_;
    GlobalTensor<T> yGm_;

    /* ascendc variable */
    TPipe* pipe_ = nullptr;
    TQue<QuePosition::VECIN, DB_BUFFER> xQueue_;
    TQue<QuePosition::VECOUT, 1> yQueue_;
    TBuf<TPosition::VECCALC> tmpBuf1_;
    TBuf<TPosition::VECCALC> tmpBuf2_;
    // Cast-path scratch (only InitBuffer'd when T != float; declared unconditionally
    // because TBuf is a thin handle and unused declarations have no UB footprint).
    TBuf<TPosition::VECCALC> gateFBuf_;
    TBuf<TPosition::VECCALC> upFBuf_;
    TBuf<TPosition::VECCALC> yFBbuf_;

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
    int64_t halfOff_ = 0;   // up half offset inside one xQueue buffer (unit: T element)
    int64_t xQueSpace_ = 0; // one xQueue buffer size (unit: byte)
    float beta_ = 1.0f;
    float linearBeta_ = 0.0f;
    int64_t activateLeft_ = 1;
    const SituGluTilingData* tl_ = nullptr;
};

template <typename T>
__aicore__ inline void SituGluBase<T>::Init(GM_ADDR x, GM_ADDR y, const SituGluTilingData* tilingData)
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
    // one xQueue buffer holds gate + up (2 * ubMaxPair T elements), 32B aligned
    xQueSpace_ = SWI_FACTOR * AlignBytes(ubMaxPair_ * static_cast<int64_t>(sizeof(T)));
    halfOff_ = xQueSpace_ / static_cast<int64_t>(sizeof(T)) / SWI_FACTOR;
    xGm_.SetGlobalBuffer((__gm__ T*)x);
    yGm_.SetGlobalBuffer((__gm__ T*)y);
    pipe_->InitBuffer(xQueue_, DB_BUFFER, xQueSpace_);
    pipe_->InitBuffer(yQueue_, 1, AlignBytes(ubMaxPair_ * static_cast<int64_t>(sizeof(T))));
    pipe_->InitBuffer(tmpBuf1_, AlignBytes(ubMaxPair_ * static_cast<int64_t>(sizeof(float))));
    pipe_->InitBuffer(tmpBuf2_, AlignBytes(ubMaxPair_ * static_cast<int64_t>(sizeof(float))));
    if constexpr (!kIsFp32) {
        pipe_->InitBuffer(gateFBuf_, AlignBytes(ubMaxPair_ * static_cast<int64_t>(sizeof(float))));
        pipe_->InitBuffer(upFBuf_, AlignBytes(ubMaxPair_ * static_cast<int64_t>(sizeof(float))));
        pipe_->InitBuffer(yFBbuf_, AlignBytes(ubMaxPair_ * static_cast<int64_t>(sizeof(float))));
    }
}

template <typename T>
__aicore__ inline int64_t SituGluBase<T>::AlignBytes(int64_t number)
{
    return (number + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
}

template <typename T>
__aicore__ inline void SituGluBase<T>::Process()
{
    if (dimBatchSize_ <= 0 || dim2H_ <= 0) {
        return;
    }
    CalTilingParam();

    if (blockIdx_ < usedCoreNum_) {
        if (isLongH_ == 1) {
            // long H: outer loop over batch, inner loop over H tiles
            for (int64_t batchIdx = 0; batchIdx < batchPreBlock_; ++batchIdx) {
                int64_t xOffset = blockIdx_ * blockOffset_ + batchIdx * dim2H_;
                int64_t yOffset = blockIdx_ * blockOffset_ / SWI_FACTOR + batchIdx * dimH_;
                for (int64_t loopIdx = 0; loopIdx < loopTime_; ++loopIdx) {
                    pairNum_ = loopIdx == (loopTime_ - 1) ? pairLastLoop_ : pairFrontLoop_;
                    ProcessSingleLoop(xOffset, yOffset);
                    xOffset += loopOffset_;
                    yOffset += loopOffset_;
                }
            }
        } else {
            // short H: tile along batch
            int64_t xOffset = blockIdx_ * blockOffset_;
            int64_t yOffset = blockIdx_ * blockOffset_ / SWI_FACTOR;
            for (int64_t loopIdx = 0; loopIdx < loopTime_; ++loopIdx) {
                pairNum_ = loopIdx == (loopTime_ - 1) ? pairLastLoop_ : pairFrontLoop_;
                ProcessSingleLoop(xOffset, yOffset);
                xOffset += loopOffset_;
                yOffset += loopOffset_ / SWI_FACTOR;
            }
        }
    }
}

template <typename T>
__aicore__ inline void SituGluBase<T>::CalTilingParam()
{
    int64_t coreNum = tl_->coreNumAll;
    if (coreNum <= 0) {
        coreNum = 1;
    }
    // inter-core batch split
    int64_t batchFrontBlock = (dimBatchSize_ + coreNum - 1) / coreNum;
    if (batchFrontBlock <= 0) {
        batchFrontBlock = 1;
    }
    usedCoreNum_ = static_cast<uint32_t>((dimBatchSize_ + batchFrontBlock - 1) / batchFrontBlock);
    int64_t batchLastBlock = dimBatchSize_ - batchFrontBlock * (usedCoreNum_ - 1);
    batchPreBlock_ = blockIdx_ == (usedCoreNum_ - 1) ? batchLastBlock : batchFrontBlock;
    blockOffset_ = batchFrontBlock * dim2H_;
    if (isLongH_ == 0) { // dim2H small: intra-core batch split
        int64_t ubMaxBatch = ubMaxPair_ / dimH_;
        if (ubMaxBatch < 1) {
            ubMaxBatch = 1;
        }
        loopTime_ = (batchPreBlock_ + ubMaxBatch - 1) / ubMaxBatch;
        int64_t batchLastLoop = batchPreBlock_ - ubMaxBatch * (loopTime_ - 1);
        pairFrontLoop_ = ubMaxBatch * dimH_;
        pairLastLoop_ = batchLastLoop * dimH_;
        loopOffset_ = ubMaxBatch * dim2H_;
    } else { // dim2H large: intra-core dim2H split
        loopTime_ = (dimH_ + ubMaxPair_ - 1) / ubMaxPair_;
        pairLastLoop_ = dimH_ - ubMaxPair_ * (loopTime_ - 1);
        pairFrontLoop_ = ubMaxPair_;
        loopOffset_ = ubMaxPair_;
    }
}

template <typename T>
__aicore__ inline void SituGluBase<T>::ProcessSingleLoop(int64_t xOffset, int64_t yOffset)
{
    CopyIn(xOffset);
    LocalTensor<T> xLocal = xQueue_.DeQue<T>();
    LocalTensor<float> tmp1 = tmpBuf1_.Get<float>();
    LocalTensor<float> tmp2 = tmpBuf2_.Get<float>();
    LocalTensor<T> yLocal = yQueue_.AllocTensor<T>();
    Compute(xLocal, tmp1, tmp2, yLocal);
    yQueue_.EnQue<T>(yLocal);
    CopyOut(yOffset);
}

template <typename T>
__aicore__ inline void SituGluBase<T>::CopyIn(int64_t xOffset)
{
    LocalTensor<T> xLocal = xQueue_.AllocTensor<T>();
    if (isLongH_ == 0) {
        CopyInShortH(xLocal, xOffset);
    } else {
        CopyInLongH(xLocal, xOffset);
    }
    xQueue_.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void SituGluBase<T>::CopyInShortH(LocalTensor<T>& xLocal, int64_t xOffset)
{
    DataCopyPadParams padParams{false, 0, 0, 0};
    DataCopyParams dataCopyXParams;
    dataCopyXParams.blockCount = pairNum_ / dimH_; // batch rows in this tile
    dataCopyXParams.blockLen = dimH_ * sizeof(T);  // half row
    dataCopyXParams.srcStride = dimH_ * sizeof(T); // skip up half between rows
    dataCopyXParams.dstStride = 0;
    DataCopyPad(xLocal, xGm_[xOffset], dataCopyXParams, padParams);
    DataCopyPad(xLocal[halfOff_], xGm_[xOffset + dimH_], dataCopyXParams, padParams);
}

template <typename T>
__aicore__ inline void SituGluBase<T>::CopyInLongH(LocalTensor<T>& xLocal, int64_t xOffset)
{
    DataCopyPadParams padParams{false, 0, 0, 0};
    DataCopyParams dataCopyXParams;
    dataCopyXParams.blockCount = 1;
    dataCopyXParams.blockLen = pairNum_ * sizeof(T);
    dataCopyXParams.srcStride = 0;
    dataCopyXParams.dstStride = 0;
    DataCopyPad(xLocal, xGm_[xOffset], dataCopyXParams, padParams);
    DataCopyPad(xLocal[halfOff_], xGm_[xOffset + dimH_], dataCopyXParams, padParams);
}

// Common core: situ_a = beta * tanh(gate/beta) * sigmoid(gate); y = situ_a * up'
// gate, up, y are always LocalTensor<float> regardless of IO dtype T.
template <typename T>
__aicore__ inline void SituGluBase<T>::SituCore(LocalTensor<float> gate, LocalTensor<float> up, LocalTensor<float> tmp1,
                                                LocalTensor<float> tmp2, LocalTensor<float> y, int64_t n)
{
    // situ_a = beta * tanh(gate/beta) * sigmoid(gate)
    Muls(tmp1, gate, 1.0f / beta_, n);
    PipeBarrier<PIPE_V>();
    Tanh(tmp1, tmp1, n);
    PipeBarrier<PIPE_V>();
    Muls(tmp1, tmp1, beta_, n);
    PipeBarrier<PIPE_V>();

    Sigmoid(tmp2, gate, n);
    PipeBarrier<PIPE_V>();
    Mul(y, tmp1, tmp2, n);
    PipeBarrier<PIPE_V>();

    if (linearBeta_ > 0.0f) {
        Muls(tmp1, up, 1.0f / linearBeta_, n);
        PipeBarrier<PIPE_V>();
        Tanh(tmp1, tmp1, n);
        PipeBarrier<PIPE_V>();
        Muls(tmp1, tmp1, linearBeta_, n);
        PipeBarrier<PIPE_V>();
        Mul(y, y, tmp1, n);
        PipeBarrier<PIPE_V>();
    } else {
        Mul(y, y, up, n);
        PipeBarrier<PIPE_V>();
    }
}

template <typename T>
__aicore__ inline void SituGluBase<T>::Compute(LocalTensor<T>& xLocal, LocalTensor<float>& tmp1,
                                               LocalTensor<float>& tmp2, LocalTensor<T>& yLocal)
{
    // front half of row -> xLocal[0:pairNum_]; back half -> xLocal[halfOff_:halfOff_+pairNum_]
    int64_t gateOff = (activateLeft_ == 1) ? 0 : halfOff_;
    int64_t upOff = (activateLeft_ == 1) ? halfOff_ : 0;

    if constexpr (kIsFp32) {
        // fp32 native path: gate/up come directly from xLocal, result goes to yLocal
        LocalTensor<float> gateLocal = xLocal[gateOff];
        LocalTensor<float> upLocal = xLocal[upOff];
        SituCore(gateLocal, upLocal, tmp1, tmp2, yLocal, pairNum_);
        xQueue_.FreeTensor(xLocal);
    } else {
        // mixed precision path (half / bf16): Cast T->float32, compute, Cast float32->T
        LocalTensor<float> gateF = gateFBuf_.Get<float>();
        LocalTensor<float> upF = upFBuf_.Get<float>();
        LocalTensor<float> yF = yFBbuf_.Get<float>();

        // T -> float32 (lossless widening for both half and bf16)
        Cast(gateF, xLocal[gateOff], RoundMode::CAST_NONE, pairNum_);
        PipeBarrier<PIPE_V>();
        Cast(upF, xLocal[upOff], RoundMode::CAST_NONE, pairNum_);
        PipeBarrier<PIPE_V>();
        xQueue_.FreeTensor(xLocal);

        SituCore(gateF, upF, tmp1, tmp2, yF, pairNum_);

        // float32 -> T
        // bf16 requires CAST_RINT on all platforms; CAST_NONE produces garbage on non-A2 (e.g. Ascend950)
        if constexpr (SituIsBf16<T>::value) {
            Cast(yLocal, yF, RoundMode::CAST_RINT, pairNum_);
        } else {
            Cast(yLocal, yF, RoundMode::CAST_NONE, pairNum_);
        }
        PipeBarrier<PIPE_V>();
    }
}

template <typename T>
__aicore__ inline void SituGluBase<T>::CopyOut(int64_t yOffset)
{
    LocalTensor<T> yLocal = yQueue_.DeQue<T>();
    DataCopyParams dataCopyParams;
    if (isLongH_ == 0) { // short H: strided output rows
        dataCopyParams.blockCount = pairNum_ / dimH_;
        dataCopyParams.blockLen = dimH_ * sizeof(T);
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = 0;
    } else { // long H: contiguous output tile
        dataCopyParams.blockCount = 1;
        dataCopyParams.blockLen = pairNum_ * sizeof(T);
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = 0;
    }
    DataCopyPad(yGm_[yOffset], yLocal, dataCopyParams);
    yQueue_.FreeTensor(yLocal);
}

} // namespace SituGluOps
#endif // OPP_SITU_GLU_HPP

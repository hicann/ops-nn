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
 * \file clipped_swiglu_grad.h
 * \brief ClippedSwigluGrad kernel implementation (unified, all platforms)
 *
 * 反向公式（与 golden 对齐）：
 *   A = clamp(a, max=limit);  B = clamp(b, -limit, limit);  s = sigmoid(alpha * A)
 *   maskA = (a <= limit);     maskB = (-limit <= b <= limit)
 *   da = dy * (B + bias) * s * (1 + alpha * A * (1 - s)) * maskA
 *   db = dy * A * s * maskB
 *   dx 散回：interleaved -> Scatter(da,even) + Scatter(db,odd);
 *           front/back  -> Copy(da,first_half) + Copy(db,second_half)
 *
 * Buffer layout (half = xQueSpace_ / sizeof(float) / SWI_FACTOR):
 *   tmpBuf1_[0..half]    : tmpA (a -> A_clamped)
 *   tmpBuf1_[half..2*half]: tmpB (b -> B_clamped+bias)
 *   xFloatLocal[0..half]  : sBuf (s = sigmoid)
 *   xFloatLocal[half..2*half]: scratch (db -> da intermediate)
 *   tmpBuf2_[0..half]    : Gather offsets (GetAB), da staging (interleaved ScatterResult)
 *   tmpBuf2_[half..2*half]: db storage (interleaved only, float)
 *   tmpBuf1_ (ScatterResult 阶段复用): interleaved 交错索引表 (uint32 字节偏移)
 *   dxFloatLocal (ScatterResult 阶段复用): 索引表临时区, 随后被 Gather 结果覆盖
 *   maskBufA_ / maskBufB_ : CompareScalar bitmasks (small)
 *   groupBuf_             : group_index (isGroup only)
 */
#ifndef OPP_CLIPPED_SWIGLU_GRAD_H
#define OPP_CLIPPED_SWIGLU_GRAD_H
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"

namespace ClippedSwigluGradOps {
using namespace AscendC;
constexpr static int64_t DB_BUFFER = 2;
constexpr static int64_t BLOCK_SIZE = 32;
constexpr static int64_t BLOCK_ELEM = BLOCK_SIZE / sizeof(float);
constexpr static int64_t BITS_PER_BYTE = 8;
constexpr static int64_t SWI_FACTOR = 2;
constexpr static int64_t GATHER_REG_ELEM = 64;
constexpr static int64_t ZERO_CHUNK_BYTES = 65535 / BLOCK_SIZE * BLOCK_SIZE;

template <typename T, bool isInterleaved, bool isGroup>
class ClippedSwigluGradBase {
public:
    __aicore__ inline ClippedSwigluGradBase(const ClippedSwigluGradTilingData* tilingData, TPipe* pipe)
        : tiling_(tilingData), pipe_(pipe){};
    __aicore__ inline void Init(GM_ADDR gradY, GM_ADDR x, GM_ADDR groupIndex, GM_ADDR gradXOut);
    __aicore__ inline int64_t AlignBytes(int64_t number);
    __aicore__ inline void Process();
    __aicore__ inline void ComputeRealBatchSize();
    __aicore__ inline void ProcessMainLoop();
    __aicore__ inline void CalTilingParam();
    __aicore__ inline void ProcessSingleLoop(int64_t xOffset, int64_t dyOffset, int64_t dxOffset);
    __aicore__ inline void CopyIn(int64_t xOffset, int64_t dyOffset);
    __aicore__ inline void CopyInHalfShortH(LocalTensor<T>& xDTypeLocal, LocalTensor<T>& dyDTypeLocal, int64_t xOffset,
                                            int64_t dyOffset);
    __aicore__ inline void CopyInHalfLongH(LocalTensor<T>& xDTypeLocal, LocalTensor<T>& dyDTypeLocal, int64_t xOffset,
                                           int64_t dyOffset);
    __aicore__ inline void CopyInInterLeaved(LocalTensor<T>& xDTypeLocal, LocalTensor<T>& dyDTypeLocal, int64_t xOffset,
                                             int64_t dyOffset);
    __aicore__ inline void Compute(LocalTensor<float>& xFloatLocal, LocalTensor<float>& dyFloatLocal,
                                   LocalTensor<float>& tmpUbF32, LocalTensor<float>& dxFloatLocal);
    __aicore__ inline void ComputeMasks(LocalTensor<float>& tmpA, LocalTensor<float>& tmpB, LocalTensor<uint8_t>& maskA,
                                        LocalTensor<uint8_t>& maskB);
    __aicore__ inline void ComputeSigmoid(LocalTensor<float>& tmpA, LocalTensor<float>& sBuf,
                                          LocalTensor<float>& scratch);
    __aicore__ inline void ComputeDb(LocalTensor<float>& tmpA, LocalTensor<float>& sBuf,
                                     LocalTensor<float>& dyFloatLocal, LocalTensor<float>& scratch,
                                     LocalTensor<float>& tmpB, LocalTensor<uint8_t>& maskB,
                                     LocalTensor<float>& dxFloatLocal);
    __aicore__ inline void ComputeDa(LocalTensor<float>& tmpA, LocalTensor<float>& sBuf, LocalTensor<float>& tmpB,
                                     LocalTensor<float>& dyFloatLocal, LocalTensor<float>& scratch,
                                     LocalTensor<uint8_t>& maskA);
    __aicore__ inline void ScatterResult(LocalTensor<float>& scratch, LocalTensor<float>& dxFloatLocal);
    __aicore__ inline void GetAB(LocalTensor<float>& tmpA, LocalTensor<float>& tmpB, LocalTensor<float>& xFloatLocal);
    __aicore__ inline void CopyOut(int64_t dxOffset);
    __aicore__ inline void InitZeroBuffer();
    __aicore__ inline void ZeroInvalidRows();

protected:
    /* global memory address */
    GlobalTensor<T> xGm_;
    GlobalTensor<T> gradYGm_;
    GlobalTensor<int64_t> groupIndexGm_;
    GlobalTensor<T> gradXOutGm_;

    /* ascendc variable */
    TPipe* pipe_ = nullptr;
    TQue<QuePosition::VECIN, DB_BUFFER> xQueue_;
    TQue<QuePosition::VECIN, DB_BUFFER> dyQueue_;
    TQue<QuePosition::VECOUT, 1> dxQueue_;
    TBuf<TPosition::VECCALC> tmpBuf1_;
    TBuf<TPosition::VECCALC> tmpBuf2_;
    TBuf<TPosition::VECCALC> maskBufA_;
    TBuf<TPosition::VECCALC> maskBufB_;
    TBuf<TPosition::VECCALC> groupBuf_;
    TBuf<TPosition::VECCALC> zeroBuf_;

    uint32_t blockIdx_ = GetBlockIdx();
    uint32_t usedCoreNum_ = 0;
    int64_t realBatchSize_ = 0;
    int64_t blockOffset_ = 0;
    int64_t loopOffset_ = 0;
    int64_t loopTime_ = 0;
    int64_t pairFrontLoop_ = 0;
    int64_t pairLastLoop_ = 0;
    int64_t pairNum_ = 0;
    int64_t batchPreBlock_ = 0;
    int64_t dimH_ = 0;
    int64_t ubMaxPair_ = 0;
    int64_t xLocalOffset1_ = 0;
    int64_t xLocalOffset2_ = 0;
    int64_t dyLocalOffset_ = 0;
    int64_t dxDbOffset_ = 0;
    int64_t xQueSpace_ = 0;
    int64_t dyQueSpace_ = 0;
    int64_t half_ = 0;
    int64_t calPairFrontLoop_ = 0;
    int64_t calPairLastLoop_ = 0;
    int64_t calPairNum_ = 0;
    const ClippedSwigluGradTilingData* tiling_ = nullptr;
};

template <typename T, bool isInterleaved, bool isGroup>
__aicore__ inline void ClippedSwigluGradBase<T, isInterleaved, isGroup>::Init(GM_ADDR gradY, GM_ADDR x,
                                                                              GM_ADDR groupIndex, GM_ADDR gradXOut)
{
    ubMaxPair_ = tiling_->ubMaxPair;
    dimH_ = tiling_->dim2H / SWI_FACTOR;
    xQueSpace_ = SWI_FACTOR * AlignBytes(ubMaxPair_ * static_cast<int64_t>(sizeof(float)));
    dyQueSpace_ = AlignBytes(ubMaxPair_ * static_cast<int64_t>(sizeof(float)));
    half_ = xQueSpace_ / sizeof(float) / SWI_FACTOR;
    xGm_.SetGlobalBuffer((__gm__ T*)x);
    gradYGm_.SetGlobalBuffer((__gm__ T*)gradY);
    gradXOutGm_.SetGlobalBuffer((__gm__ T*)gradXOut);
    if constexpr (isGroup) {
        groupIndexGm_.SetGlobalBuffer((__gm__ int64_t*)groupIndex);
    }
    pipe_->InitBuffer(xQueue_, DB_BUFFER, xQueSpace_);
    pipe_->InitBuffer(dyQueue_, DB_BUFFER, dyQueSpace_);
    pipe_->InitBuffer(dxQueue_, 1, xQueSpace_);
    pipe_->InitBuffer(tmpBuf1_, xQueSpace_);
    pipe_->InitBuffer(tmpBuf2_, xQueSpace_);
    int64_t maskBytes = (ubMaxPair_ + BITS_PER_BYTE - 1) / BITS_PER_BYTE;
    int64_t maskBufSize = AlignBytes(maskBytes);
    if (maskBufSize < BLOCK_SIZE) {
        maskBufSize = BLOCK_SIZE;
    }
    pipe_->InitBuffer(maskBufA_, maskBufSize);
    pipe_->InitBuffer(maskBufB_, maskBufSize);
}

template <typename T, bool isInterleaved, bool isGroup>
__aicore__ inline int64_t ClippedSwigluGradBase<T, isInterleaved, isGroup>::AlignBytes(int64_t number)
{
    return (number + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
}

template <typename T, bool isInterleaved, bool isGroup>
__aicore__ inline void ClippedSwigluGradBase<T, isInterleaved, isGroup>::Process()
{
    ComputeRealBatchSize();
    CalTilingParam();

    if (blockIdx_ < usedCoreNum_) {
        ProcessMainLoop();
    }

    SyncAll();

    if constexpr (isGroup) {
        if (realBatchSize_ < tiling_->dimBatchSize) {
            InitZeroBuffer();
            ZeroInvalidRows();
        }
    }
}

template <typename T, bool isInterleaved, bool isGroup>
__aicore__ inline void ClippedSwigluGradBase<T, isInterleaved, isGroup>::ComputeRealBatchSize()
{
    if constexpr (!isGroup) {
        realBatchSize_ = tiling_->dimBatchSize;
    } else {
        int64_t groupSum = 0;
        for (int64_t i = 0; i < tiling_->groupNum; ++i) {
            groupSum += groupIndexGm_.GetValue(i);
        }
        realBatchSize_ = groupSum < tiling_->dimBatchSize ? groupSum : tiling_->dimBatchSize;
    }
}

template <typename T, bool isInterleaved, bool isGroup>
__aicore__ inline void ClippedSwigluGradBase<T, isInterleaved, isGroup>::ProcessMainLoop()
{
    int64_t xOffset = 0;
    int64_t dyOffset = 0;
    int64_t dxOffset = 0;
    if constexpr (!isInterleaved) {
        if (tiling_->isLongH == 1) {
            for (int64_t batchIdx = 0; batchIdx < batchPreBlock_; ++batchIdx) {
                xOffset = blockOffset_ + batchIdx * tiling_->dim2H;
                dyOffset = blockOffset_ / SWI_FACTOR + batchIdx * dimH_;
                dxOffset = blockOffset_ + batchIdx * tiling_->dim2H;
                for (int64_t loopIdx = 0; loopIdx < loopTime_; ++loopIdx) {
                    pairNum_ = loopIdx == (loopTime_ - 1) ? pairLastLoop_ : pairFrontLoop_;
                    calPairNum_ = loopIdx == (loopTime_ - 1) ? calPairLastLoop_ : calPairFrontLoop_;
                    ProcessSingleLoop(xOffset, dyOffset, dxOffset);
                    xOffset += loopOffset_;
                    dyOffset += loopOffset_;
                    dxOffset += loopOffset_;
                }
            }
            return;
        }
    }

    xOffset = blockOffset_;
    dyOffset = blockOffset_ / SWI_FACTOR;
    dxOffset = blockOffset_;
    for (int64_t loopIdx = 0; loopIdx < loopTime_; ++loopIdx) {
        pairNum_ = loopIdx == (loopTime_ - 1) ? pairLastLoop_ : pairFrontLoop_;
        calPairNum_ = loopIdx == (loopTime_ - 1) ? calPairLastLoop_ : calPairFrontLoop_;
        ProcessSingleLoop(xOffset, dyOffset, dxOffset);
        xOffset += loopOffset_;
        dyOffset += loopOffset_ / SWI_FACTOR;
        dxOffset += loopOffset_;
    }
}

template <typename T, bool isInterleaved, bool isGroup>
__aicore__ inline void ClippedSwigluGradBase<T, isInterleaved, isGroup>::CalTilingParam()
{
    int64_t coreNum = static_cast<int64_t>(tiling_->coreNumAll);
    int64_t blockIdx = static_cast<int64_t>(blockIdx_);

    if constexpr (!isInterleaved) {
        // 前后切分：按 batch 行均衡分配
        int64_t base = realBatchSize_ / coreNum;
        int64_t remainder = realBatchSize_ % coreNum;
        usedCoreNum_ = static_cast<uint32_t>(realBatchSize_ < coreNum ? realBatchSize_ : coreNum);
        batchPreBlock_ = base + (blockIdx < remainder ? 1 : 0);
        int64_t coreStartRow = blockIdx * base + (blockIdx < remainder ? blockIdx : remainder);
        blockOffset_ = coreStartRow * tiling_->dim2H;

        if (tiling_->isLongH == 0) {
            int64_t batchSpace = SWI_FACTOR * AlignBytes(dimH_ * static_cast<int64_t>(sizeof(float)));
            int64_t ubMaxBatch = xQueSpace_ / batchSpace;
            loopTime_ = (batchPreBlock_ + ubMaxBatch - 1) / ubMaxBatch;
            int64_t batchLastLoop = batchPreBlock_ - ubMaxBatch * (loopTime_ - 1);
            pairFrontLoop_ = ubMaxBatch * dimH_;
            pairLastLoop_ = batchLastLoop * dimH_;
            loopOffset_ = ubMaxBatch * tiling_->dim2H;
            calPairFrontLoop_ = ubMaxBatch * batchSpace / SWI_FACTOR / sizeof(float);
            calPairLastLoop_ = batchLastLoop * batchSpace / SWI_FACTOR / sizeof(float);
        } else {
            loopTime_ = (dimH_ + ubMaxPair_ - 1) / ubMaxPair_;
            pairLastLoop_ = dimH_ - ubMaxPair_ * (loopTime_ - 1);
            pairFrontLoop_ = ubMaxPair_;
            loopOffset_ = ubMaxPair_;
            calPairFrontLoop_ = pairFrontLoop_;
            calPairLastLoop_ = pairLastLoop_;
        }
    } else {
        // 奇偶切分：按 pair 均衡分配
        int64_t pairTotal = tiling_->dim2H * realBatchSize_ / SWI_FACTOR;
        int64_t base = pairTotal / coreNum;
        int64_t remainder = pairTotal % coreNum;
        usedCoreNum_ = static_cast<uint32_t>(pairTotal < coreNum ? pairTotal : coreNum);
        int64_t pairPreBlock = base + (blockIdx < remainder ? 1 : 0);
        int64_t coreStartPair = blockIdx * base + (blockIdx < remainder ? blockIdx : remainder);
        blockOffset_ = coreStartPair * SWI_FACTOR;

        loopTime_ = (pairPreBlock + ubMaxPair_ - 1) / ubMaxPair_;
        pairLastLoop_ = pairPreBlock - ubMaxPair_ * (loopTime_ - 1);
        pairFrontLoop_ = ubMaxPair_;
        loopOffset_ = SWI_FACTOR * ubMaxPair_;
        calPairFrontLoop_ = pairFrontLoop_;
        calPairLastLoop_ = pairLastLoop_;
    }
}

template <typename T, bool isInterleaved, bool isGroup>
__aicore__ inline void ClippedSwigluGradBase<T, isInterleaved, isGroup>::ProcessSingleLoop(int64_t xOffset,
                                                                                           int64_t dyOffset,
                                                                                           int64_t dxOffset)
{
    CopyIn(xOffset, dyOffset);
    LocalTensor<T> xDTypeLocal = xQueue_.DeQue<T>();
    LocalTensor<T> dyDTypeLocal = dyQueue_.DeQue<T>();
    LocalTensor<float> xFloatLocal = xDTypeLocal.template ReinterpretCast<float>();
    LocalTensor<float> dyFloatLocal = dyDTypeLocal.template ReinterpretCast<float>();

    if constexpr (!std::is_same_v<T, float>) {
        if constexpr (!isInterleaved) {
            Cast(xFloatLocal, xDTypeLocal[xLocalOffset1_], RoundMode::CAST_NONE, calPairNum_);
            PipeBarrier<PIPE_V>();
            Cast(xFloatLocal[half_], xDTypeLocal[xLocalOffset1_ + xLocalOffset2_], RoundMode::CAST_NONE, calPairNum_);
            PipeBarrier<PIPE_V>();
        } else {
            Cast(xFloatLocal, xDTypeLocal[xLocalOffset1_], RoundMode::CAST_NONE, calPairNum_ * SWI_FACTOR);
            PipeBarrier<PIPE_V>();
        }
        Cast(dyFloatLocal, dyDTypeLocal[dyLocalOffset_], RoundMode::CAST_NONE, calPairNum_);
        PipeBarrier<PIPE_V>();
    }

    LocalTensor<float> tmpUbF32 = tmpBuf1_.Get<float>();
    LocalTensor<float> dxFloatLocal = dxQueue_.AllocTensor<float>();
    Compute(xFloatLocal, dyFloatLocal, tmpUbF32, dxFloatLocal);

    LocalTensor<T> dxDTypeLocal = dxFloatLocal.template ReinterpretCast<T>();
    if constexpr (std::is_same_v<T, bfloat16_t>) {
        if constexpr (!isInterleaved) {
            Cast(dxDTypeLocal, dxFloatLocal, RoundMode::CAST_RINT, calPairNum_);
            PipeBarrier<PIPE_V>();
            Cast(dxDTypeLocal[dxDbOffset_], dxFloatLocal[half_], RoundMode::CAST_RINT, calPairNum_);
        } else {
            Cast(dxDTypeLocal, dxFloatLocal, RoundMode::CAST_RINT, calPairNum_ * SWI_FACTOR);
        }
    } else if constexpr (std::is_same_v<T, half>) {
        if constexpr (!isInterleaved) {
            Cast(dxDTypeLocal, dxFloatLocal, RoundMode::CAST_NONE, calPairNum_);
            PipeBarrier<PIPE_V>();
            Cast(dxDTypeLocal[dxDbOffset_], dxFloatLocal[half_], RoundMode::CAST_NONE, calPairNum_);
        } else {
            Cast(dxDTypeLocal, dxFloatLocal, RoundMode::CAST_NONE, calPairNum_ * SWI_FACTOR);
        }
    }
    dxQueue_.EnQue<T>(dxDTypeLocal);
    CopyOut(dxOffset);
}

template <typename T, bool isInterleaved, bool isGroup>
__aicore__ inline void ClippedSwigluGradBase<T, isInterleaved, isGroup>::CopyIn(int64_t xOffset, int64_t dyOffset)
{
    if constexpr (!std::is_same_v<T, float>) {
        int64_t blockElem = BLOCK_SIZE / sizeof(T);
        xLocalOffset1_ = (xQueSpace_ / SWI_FACTOR / static_cast<int64_t>(sizeof(T)) + blockElem - 1) / blockElem *
                         blockElem;
        xLocalOffset2_ = (xLocalOffset1_ / SWI_FACTOR + blockElem - 1) / blockElem * blockElem;
        dyLocalOffset_ = (dyQueSpace_ / static_cast<int64_t>(sizeof(T)) / SWI_FACTOR + blockElem - 1) / blockElem *
                         blockElem;
        dxDbOffset_ = (calPairNum_ * static_cast<int64_t>(sizeof(T)) + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE /
                      sizeof(T);
    } else {
        xLocalOffset1_ = 0;
        xLocalOffset2_ = xQueSpace_ / static_cast<int64_t>(sizeof(float)) / SWI_FACTOR;
        dyLocalOffset_ = 0;
        dxDbOffset_ = half_;
    }
    LocalTensor<T> xDTypeLocal = xQueue_.AllocTensor<T>();
    LocalTensor<T> dyDTypeLocal = dyQueue_.AllocTensor<T>();
    if constexpr (isInterleaved) {
        CopyInInterLeaved(xDTypeLocal, dyDTypeLocal, xOffset, dyOffset);
    } else {
        if (tiling_->isLongH == 0) {
            CopyInHalfShortH(xDTypeLocal, dyDTypeLocal, xOffset, dyOffset);
        } else {
            CopyInHalfLongH(xDTypeLocal, dyDTypeLocal, xOffset, dyOffset);
        }
    }
    xQueue_.EnQue(xDTypeLocal);
    dyQueue_.EnQue(dyDTypeLocal);
}

template <typename T, bool isInterleaved, bool isGroup>
__aicore__ inline void ClippedSwigluGradBase<T, isInterleaved, isGroup>::CopyInHalfShortH(LocalTensor<T>& xDTypeLocal,
                                                                                          LocalTensor<T>& dyDTypeLocal,
                                                                                          int64_t xOffset,
                                                                                          int64_t dyOffset)
{
    DataCopyPadParams padParams{false, 0, 0, 0};
    DataCopyParams dataCopyXParams;
    dataCopyXParams.blockCount = pairNum_ / dimH_;
    dataCopyXParams.blockLen = dimH_ * sizeof(T);
    dataCopyXParams.srcStride = dimH_ * sizeof(T);
    dataCopyXParams.dstStride = 0;
    DataCopyPad(xDTypeLocal[xLocalOffset1_], xGm_[xOffset], dataCopyXParams, padParams);
    DataCopyPad(xDTypeLocal[xLocalOffset1_ + xLocalOffset2_], xGm_[xOffset + dimH_], dataCopyXParams, padParams);
    DataCopyParams dataCopyDyParams;
    dataCopyDyParams.blockCount = pairNum_ / dimH_;
    dataCopyDyParams.blockLen = dimH_ * sizeof(T);
    dataCopyDyParams.srcStride = 0;
    dataCopyDyParams.dstStride = 0;
    DataCopyPad(dyDTypeLocal[dyLocalOffset_], gradYGm_[dyOffset], dataCopyDyParams, padParams);
}

template <typename T, bool isInterleaved, bool isGroup>
__aicore__ inline void ClippedSwigluGradBase<T, isInterleaved, isGroup>::CopyInHalfLongH(LocalTensor<T>& xDTypeLocal,
                                                                                         LocalTensor<T>& dyDTypeLocal,
                                                                                         int64_t xOffset,
                                                                                         int64_t dyOffset)
{
    DataCopyPadParams padParams{false, 0, 0, 0};
    DataCopyParams dataCopyXParams;
    dataCopyXParams.blockCount = 1;
    dataCopyXParams.blockLen = AlignBytes(pairNum_ * sizeof(T));
    dataCopyXParams.srcStride = 0;
    dataCopyXParams.dstStride = 0;
    DataCopyPad(xDTypeLocal[xLocalOffset1_], xGm_[xOffset], dataCopyXParams, padParams);
    DataCopyPad(xDTypeLocal[xLocalOffset1_ + xLocalOffset2_], xGm_[xOffset + dimH_], dataCopyXParams, padParams);
    DataCopyParams dataCopyDyParams;
    dataCopyDyParams.blockCount = 1;
    dataCopyDyParams.blockLen = AlignBytes(pairNum_ * sizeof(T));
    dataCopyDyParams.srcStride = 0;
    dataCopyDyParams.dstStride = 0;
    DataCopyPad(dyDTypeLocal[dyLocalOffset_], gradYGm_[dyOffset], dataCopyDyParams, padParams);
}

template <typename T, bool isInterleaved, bool isGroup>
__aicore__ inline void ClippedSwigluGradBase<T, isInterleaved, isGroup>::CopyInInterLeaved(LocalTensor<T>& xDTypeLocal,
                                                                                           LocalTensor<T>& dyDTypeLocal,
                                                                                           int64_t xOffset,
                                                                                           int64_t dyOffset)
{
    DataCopyPadParams padParams{false, 0, 0, 0};
    DataCopyParams dataCopyXParams;
    dataCopyXParams.blockCount = 1;
    dataCopyXParams.blockLen = SWI_FACTOR * pairNum_ * sizeof(T);
    dataCopyXParams.srcStride = 0;
    dataCopyXParams.dstStride = 0;
    DataCopyPad(xDTypeLocal[xLocalOffset1_], xGm_[xOffset], dataCopyXParams, padParams);
    DataCopyParams dataCopyDyParams;
    dataCopyDyParams.blockCount = 1;
    dataCopyDyParams.blockLen = pairNum_ * sizeof(T);
    dataCopyDyParams.srcStride = 0;
    dataCopyDyParams.dstStride = 0;
    DataCopyPad(dyDTypeLocal[dyLocalOffset_], gradYGm_[dyOffset], dataCopyDyParams, padParams);
}

template <typename T, bool isInterleaved, bool isGroup>
__aicore__ inline void ClippedSwigluGradBase<T, isInterleaved, isGroup>::Compute(LocalTensor<float>& xFloatLocal,
                                                                                 LocalTensor<float>& dyFloatLocal,
                                                                                 LocalTensor<float>& tmpUbF32,
                                                                                 LocalTensor<float>& dxFloatLocal)
{
    LocalTensor<float> tmpA = tmpUbF32;
    LocalTensor<float> tmpB = tmpUbF32[half_];
    LocalTensor<float> sBuf = xFloatLocal;
    LocalTensor<float> scratch = xFloatLocal[half_];
    LocalTensor<uint8_t> maskA = maskBufA_.Get<uint8_t>();
    LocalTensor<uint8_t> maskB = maskBufB_.Get<uint8_t>();

    GetAB(tmpA, tmpB, xFloatLocal);
    ComputeMasks(tmpA, tmpB, maskA, maskB);
    ComputeSigmoid(tmpA, sBuf, scratch);
    ComputeDb(tmpA, sBuf, dyFloatLocal, scratch, tmpB, maskB, dxFloatLocal);
    ComputeDa(tmpA, sBuf, tmpB, dyFloatLocal, scratch, maskA);
    ScatterResult(scratch, dxFloatLocal);

    xQueue_.FreeTensor(xFloatLocal);
    dyQueue_.FreeTensor(dyFloatLocal);
}

template <typename T, bool isInterleaved, bool isGroup>
__aicore__ inline void ClippedSwigluGradBase<T, isInterleaved, isGroup>::ComputeMasks(LocalTensor<float>& tmpA,
                                                                                      LocalTensor<float>& tmpB,
                                                                                      LocalTensor<uint8_t>& maskA,
                                                                                      LocalTensor<uint8_t>& maskB)
{
    constexpr int64_t CMP_ALIGN = 64;
    int64_t alignedCount = (calPairNum_ + CMP_ALIGN - 1) / CMP_ALIGN * CMP_ALIGN;
    CompareScalar(maskA, tmpA, tiling_->limit, CMPMODE::LE, alignedCount);
    CompareScalar(maskB, tmpB, tiling_->limit, CMPMODE::LE, alignedCount);
}

template <typename T, bool isInterleaved, bool isGroup>
__aicore__ inline void ClippedSwigluGradBase<T, isInterleaved, isGroup>::ComputeSigmoid(LocalTensor<float>& tmpA,
                                                                                        LocalTensor<float>& sBuf,
                                                                                        LocalTensor<float>& scratch)
{
    Mins(tmpA, tmpA, tiling_->limit, calPairNum_);
    PipeBarrier<PIPE_V>();
    Muls(sBuf, tmpA, -1 * tiling_->alpha, calPairNum_);
    PipeBarrier<PIPE_V>();
    Exp(sBuf, sBuf, calPairNum_);
    PipeBarrier<PIPE_V>();
    Adds(sBuf, sBuf, (float)1.0, calPairNum_);
    PipeBarrier<PIPE_V>();
    Duplicate(scratch, (float)1.0, calPairNum_);
    Div(sBuf, scratch, sBuf, calPairNum_);
    PipeBarrier<PIPE_V>();
}

template <typename T, bool isInterleaved, bool isGroup>
__aicore__ inline void ClippedSwigluGradBase<T, isInterleaved, isGroup>::ComputeDb(
    LocalTensor<float>& tmpA, LocalTensor<float>& sBuf, LocalTensor<float>& dyFloatLocal, LocalTensor<float>& scratch,
    LocalTensor<float>& tmpB, LocalTensor<uint8_t>& maskB, LocalTensor<float>& dxFloatLocal)
{
    Mul(scratch, tmpA, sBuf, calPairNum_);
    PipeBarrier<PIPE_V>();
    Mul(scratch, scratch, dyFloatLocal, calPairNum_);
    PipeBarrier<PIPE_V>();
    Select(scratch, maskB, scratch, (float)0.0, SELMODE::VSEL_TENSOR_SCALAR_MODE, calPairNum_);
    PipeBarrier<PIPE_V>();
    constexpr int64_t CMP_ALIGN = 64;
    int64_t alignedCount = (calPairNum_ + CMP_ALIGN - 1) / CMP_ALIGN * CMP_ALIGN;
    CompareScalar(maskB, tmpB, -1 * tiling_->limit, CMPMODE::GE, alignedCount);
    Select(scratch, maskB, scratch, (float)0.0, SELMODE::VSEL_TENSOR_SCALAR_MODE, calPairNum_);
    PipeBarrier<PIPE_V>();

    if constexpr (!isInterleaved) {
        SetMaskCount();
        SetVectorMask<float, MaskMode::COUNTER>(calPairNum_);
        Copy<float, false>(dxFloatLocal[half_], scratch, AscendC::MASK_PLACEHOLDER, 1, {1, 1, 0, 0});
        SetMaskNorm();
        ResetMask();
    } else {
        LocalTensor<float> dbStorage = tmpBuf2_.Get<float>();
        SetMaskCount();
        SetVectorMask<float, MaskMode::COUNTER>(calPairNum_);
        Copy<float, false>(dbStorage[half_], scratch, AscendC::MASK_PLACEHOLDER, 1, {1, 1, 0, 0});
        SetMaskNorm();
        ResetMask();
    }
}

template <typename T, bool isInterleaved, bool isGroup>
__aicore__ inline void ClippedSwigluGradBase<T, isInterleaved, isGroup>::ComputeDa(
    LocalTensor<float>& tmpA, LocalTensor<float>& sBuf, LocalTensor<float>& tmpB, LocalTensor<float>& dyFloatLocal,
    LocalTensor<float>& scratch, LocalTensor<uint8_t>& maskA)
{
    Mins(tmpB, tmpB, tiling_->limit, calPairNum_);
    PipeBarrier<PIPE_V>();
    Maxs(tmpB, tmpB, -1 * tiling_->limit, calPairNum_);
    PipeBarrier<PIPE_V>();
    Adds(tmpB, tmpB, tiling_->bias, calPairNum_);
    PipeBarrier<PIPE_V>();

    Muls(scratch, sBuf, (float)-1.0, calPairNum_);
    PipeBarrier<PIPE_V>();
    Adds(scratch, scratch, (float)1.0, calPairNum_);
    PipeBarrier<PIPE_V>();
    Mul(scratch, scratch, tmpA, calPairNum_);
    PipeBarrier<PIPE_V>();
    Muls(scratch, scratch, tiling_->alpha, calPairNum_);
    PipeBarrier<PIPE_V>();
    Adds(scratch, scratch, (float)1.0, calPairNum_);
    PipeBarrier<PIPE_V>();
    Mul(scratch, scratch, sBuf, calPairNum_);
    PipeBarrier<PIPE_V>();
    Mul(scratch, scratch, tmpB, calPairNum_);
    PipeBarrier<PIPE_V>();
    Mul(scratch, scratch, dyFloatLocal, calPairNum_);
    PipeBarrier<PIPE_V>();
    Select(scratch, maskA, scratch, (float)0.0, SELMODE::VSEL_TENSOR_SCALAR_MODE, calPairNum_);
    PipeBarrier<PIPE_V>();
}

template <typename T, bool isInterleaved, bool isGroup>
__aicore__ inline void ClippedSwigluGradBase<T, isInterleaved, isGroup>::ScatterResult(LocalTensor<float>& scratch,
                                                                                       LocalTensor<float>& dxFloatLocal)
{
    if constexpr (!isInterleaved) {
        SetMaskCount();
        SetVectorMask<float, MaskMode::COUNTER>(calPairNum_);
        Copy<float, false>(dxFloatLocal, scratch, AscendC::MASK_PLACEHOLDER, 1, {1, 1, 0, 0});
        SetMaskNorm();
        ResetMask();
    } else {
        LocalTensor<float> dbStorage = tmpBuf2_.Get<float>();
        SetMaskCount();
        SetVectorMask<float, MaskMode::COUNTER>(calPairNum_);
        Copy<float, false>(dbStorage, scratch, AscendC::MASK_PLACEHOLDER, 1, {1, 1, 0, 0});
        SetMaskNorm();
        ResetMask();
        PipeBarrier<PIPE_V>();

        int64_t cnt = SWI_FACTOR * calPairNum_;
        int64_t cntAligned = (cnt + GATHER_REG_ELEM - 1) / GATHER_REG_ELEM * GATHER_REG_ELEM;
        LocalTensor<int32_t> interleaveIdx = tmpBuf1_.Get<int32_t>();
        LocalTensor<int32_t> idxTmp = dxFloatLocal.template ReinterpretCast<int32_t>();
        ArithProgression(interleaveIdx, static_cast<int32_t>(0), static_cast<int32_t>(1),
                         static_cast<int32_t>(GATHER_REG_ELEM));
        PipeBarrier<PIPE_V>();
        ShiftRight(idxTmp, interleaveIdx, static_cast<int32_t>(1), static_cast<int32_t>(GATHER_REG_ELEM));
        PipeBarrier<PIPE_V>();
        Muls(idxTmp, idxTmp, static_cast<int32_t>(8 * half_ - 4), static_cast<int32_t>(GATHER_REG_ELEM));
        PipeBarrier<PIPE_V>();
        Muls(interleaveIdx, interleaveIdx, static_cast<int32_t>(4 * half_), static_cast<int32_t>(GATHER_REG_ELEM));
        PipeBarrier<PIPE_V>();
        Sub(interleaveIdx, interleaveIdx, idxTmp, static_cast<int32_t>(GATHER_REG_ELEM));
        PipeBarrier<PIPE_V>();
        for (int64_t built = GATHER_REG_ELEM; built < cntAligned; built *= SWI_FACTOR) {
            int64_t len = cntAligned - built < built ? cntAligned - built : built;
            Adds(interleaveIdx[built], interleaveIdx, static_cast<int32_t>(SWI_FACTOR * built),
                 static_cast<int32_t>(len));
            PipeBarrier<PIPE_V>();
        }
        Gather(dxFloatLocal, dbStorage, interleaveIdx.template ReinterpretCast<uint32_t>(), static_cast<uint32_t>(0),
               static_cast<uint32_t>(cnt));
        PipeBarrier<PIPE_V>();
    }
}

template <typename T, bool isInterleaved, bool isGroup>
__aicore__ inline void ClippedSwigluGradBase<T, isInterleaved, isGroup>::GetAB(LocalTensor<float>& tmpA,
                                                                               LocalTensor<float>& tmpB,
                                                                               LocalTensor<float>& xFloatLocal)
{
    if constexpr (!isInterleaved) {
        SetMaskCount();
        SetVectorMask<float, MaskMode::COUNTER>(calPairNum_);
        Copy<float, false>(tmpA, xFloatLocal, AscendC::MASK_PLACEHOLDER, 1, {1, 1, 0, 0});
        Copy<float, false>(tmpB, xFloatLocal[half_], AscendC::MASK_PLACEHOLDER, 1, {1, 1, 0, 0});
        SetMaskNorm();
        ResetMask();
    } else {
        LocalTensor<int32_t> xOffsetLocalI32 = tmpBuf2_.Get<int32_t>();
        ArithProgression(xOffsetLocalI32, static_cast<int32_t>(0), static_cast<int32_t>(sizeof(float) * SWI_FACTOR),
                         static_cast<int32_t>(ubMaxPair_));
        PipeBarrier<PIPE_V>();
        LocalTensor<uint32_t> xOffsetLocalU32 = xOffsetLocalI32.template ReinterpretCast<uint32_t>();
        Gather(tmpB, xFloatLocal, xOffsetLocalU32, static_cast<uint32_t>(4), pairNum_);
        Gather(tmpA, xFloatLocal, xOffsetLocalU32, static_cast<uint32_t>(0), pairNum_);
        PipeBarrier<PIPE_V>();
    }
}

template <typename T, bool isInterleaved, bool isGroup>
__aicore__ inline void ClippedSwigluGradBase<T, isInterleaved, isGroup>::CopyOut(int64_t dxOffset)
{
    LocalTensor<T> dxDTypeLocal = dxQueue_.DeQue<T>();

    DataCopyParams params;
    if constexpr (!isInterleaved) {
        if (tiling_->isLongH == 0) {
            params.blockCount = pairNum_ / dimH_;
            params.blockLen = dimH_ * sizeof(T);
            params.srcStride = 0;
            params.dstStride = dimH_ * sizeof(T);
        } else {
            params.blockCount = 1;
            params.blockLen = pairNum_ * sizeof(T);
            params.srcStride = 0;
            params.dstStride = 0;
        }
        DataCopyPad(gradXOutGm_[dxOffset], dxDTypeLocal, params);
        DataCopyPad(gradXOutGm_[dxOffset + dimH_], dxDTypeLocal[dxDbOffset_], params);
    } else {
        params.blockCount = 1;
        params.blockLen = pairNum_ * SWI_FACTOR * sizeof(T);
        params.srcStride = 0;
        params.dstStride = 0;
        DataCopyPad(gradXOutGm_[dxOffset], dxDTypeLocal, params);
    }
    dxQueue_.FreeTensor(dxDTypeLocal);
}

template <typename T, bool isInterleaved, bool isGroup>
__aicore__ inline void ClippedSwigluGradBase<T, isInterleaved, isGroup>::InitZeroBuffer()
{
    pipe_->Reset();
    int64_t elemBytes = static_cast<int64_t>(sizeof(T));
    int64_t chunkElems = ZERO_CHUNK_BYTES / elemBytes;
    int64_t zeroBufSize = AlignBytes(ZERO_CHUNK_BYTES);
    pipe_->InitBuffer(zeroBuf_, zeroBufSize);

    LocalTensor<T> zeroLocal = zeroBuf_.Get<T>();
    Duplicate(zeroLocal, static_cast<T>(0), chunkElems);
    event_t vToMte3 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>());
    SetFlag<HardEvent::V_MTE3>(vToMte3);
    WaitFlag<HardEvent::V_MTE3>(vToMte3);
    GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::V_MTE3>(vToMte3);
}

template <typename T, bool isInterleaved, bool isGroup>
__aicore__ inline void ClippedSwigluGradBase<T, isInterleaved, isGroup>::ZeroInvalidRows()
{
    int64_t invalidRows = tiling_->dimBatchSize - realBatchSize_;
    if (invalidRows <= 0) {
        return;
    }

    int64_t coreNum = static_cast<int64_t>(tiling_->coreNumAll);
    int64_t blockIdx = static_cast<int64_t>(blockIdx_);
    int64_t base = invalidRows / coreNum;
    int64_t remainder = invalidRows % coreNum;
    int64_t rowsToZero = base + (blockIdx < remainder ? 1 : 0);
    if (rowsToZero <= 0) {
        return;
    }
    int64_t zeroStartRow = realBatchSize_ + blockIdx * base + (blockIdx < remainder ? blockIdx : remainder);

    LocalTensor<T> zeroLocal = zeroBuf_.Get<T>();
    DataCopyParams params;
    params.blockCount = 1;
    params.srcStride = 0;
    params.dstStride = 0;
    int64_t elemBytes = static_cast<int64_t>(sizeof(T));
    int64_t chunkElems = ZERO_CHUNK_BYTES / elemBytes;
    int64_t dim2H = tiling_->dim2H;
    int64_t fullChunks = dim2H / chunkElems;
    int64_t tailElems = dim2H % chunkElems;

    for (int64_t row = 0; row < rowsToZero; ++row) {
        int64_t rowBase = (zeroStartRow + row) * dim2H;
        int64_t off = 0;
        for (int64_t c = 0; c < fullChunks; ++c) {
            params.blockLen = ZERO_CHUNK_BYTES;
            DataCopyPad(gradXOutGm_[rowBase + off], zeroLocal, params);
            off += chunkElems;
        }
        if (tailElems > 0) {
            params.blockLen = tailElems * elemBytes;
            DataCopyPad(gradXOutGm_[rowBase + off], zeroLocal, params);
        }
    }
}

} // namespace ClippedSwigluGradOps
#endif // OPP_CLIPPED_SWIGLU_GRAD_H

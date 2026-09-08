/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file clipped_swiglu_grad_base.h
 * \brief ClippedSwigluGrad common scheduling base (CRTP, shared by 910B and arch35 kernels)
 *
 * 提供公共调度框架：分核切分(CalTilingParam)、主循环(ProcessMainLoop)、
 * 按切分模式的数据搬入/搬出(CopyIn/CopyOut)、group 场景补零(ZeroInvalidRows)。
 * 计算路径由派生类通过 ProcessSingleLoop 实现（CRTP 静态分派，无虚函数开销）。
 */
#ifndef OPP_CLIPPED_SWIGLU_GRAD_BASE_H
#define OPP_CLIPPED_SWIGLU_GRAD_BASE_H
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"

namespace ClippedSwigluGradOps {
using namespace AscendC;

constexpr static int64_t DB_BUFFER = 2;
constexpr static int64_t BLOCK_SIZE = 32;
constexpr static int64_t SWI_FACTOR = 2;
constexpr static int64_t ZERO_CHUNK_BYTES = 65535 / BLOCK_SIZE * BLOCK_SIZE;
constexpr static int64_t BLOCK_ELEM = BLOCK_SIZE / sizeof(float);

template <typename T, bool isInterleaved, bool isGroup, typename Derived>
class ClippedSwigluGradSchedBase {
public:
    __aicore__ inline void Process();

protected:
    __aicore__ inline ClippedSwigluGradSchedBase(const ClippedSwigluGradTilingData* tilingData, TPipe* pipe)
        : tiling_(tilingData), pipe_(pipe){};
    __aicore__ inline void InitCommon(GM_ADDR gradY, GM_ADDR x, GM_ADDR groupIndex, GM_ADDR gradXOut);
    __aicore__ inline int64_t AlignBytes(int64_t number);
    __aicore__ inline void ComputeRealBatchSize();
    __aicore__ inline void CalTilingParam();
    __aicore__ inline void ProcessMainLoop();
    __aicore__ inline void CopyIn(int64_t xOffset, int64_t dyOffset);
    __aicore__ inline void CopyInHalfShortH(LocalTensor<T>& xDTypeLocal, LocalTensor<T>& dyDTypeLocal, int64_t xOffset,
                                            int64_t dyOffset);
    __aicore__ inline void CopyInHalfLongH(LocalTensor<T>& xDTypeLocal, LocalTensor<T>& dyDTypeLocal, int64_t xOffset,
                                           int64_t dyOffset);
    __aicore__ inline void CopyInInterLeaved(LocalTensor<T>& xDTypeLocal, LocalTensor<T>& dyDTypeLocal, int64_t xOffset,
                                             int64_t dyOffset);
    __aicore__ inline void CopyOut(int64_t dxOffset);
    __aicore__ inline void InitZeroBuffer();
    __aicore__ inline void ZeroInvalidRows();

    /* global memory address */
    GlobalTensor<T> xGm_;
    GlobalTensor<T> gradYGm_;
    GlobalTensor<int64_t> groupIndexGm_;
    GlobalTensor<T> gradXOutGm_;

    /* ascendc variable */
    TPipe* pipe_ = nullptr;
    const ClippedSwigluGradTilingData* tiling_ = nullptr;
    TQue<QuePosition::VECIN, DB_BUFFER> xQueue_;
    TQue<QuePosition::VECIN, DB_BUFFER> dyQueue_;
    TQue<QuePosition::VECOUT, 1> dxQueue_;
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
    int64_t xQueSpace_ = 0;
    int64_t dyQueSpace_ = 0;
    int64_t half_ = 0;
    int64_t calPairFrontLoop_ = 0;
    int64_t calPairLastLoop_ = 0;
    int64_t calPairNum_ = 0;
    int64_t xLocalOffset1_ = 0;
    int64_t xLocalOffset2_ = 0;
    int64_t dyLocalOffset_ = 0;
    int64_t dxDbOffset_ = 0;
};

template <typename T, bool isInterleaved, bool isGroup, typename Derived>
__aicore__ inline void ClippedSwigluGradSchedBase<T, isInterleaved, isGroup, Derived>::InitCommon(GM_ADDR gradY,
                                                                                                  GM_ADDR x,
                                                                                                  GM_ADDR groupIndex,
                                                                                                  GM_ADDR gradXOut)
{
    ubMaxPair_ = tiling_->ubMaxPair;
    dimH_ = tiling_->dim2H / SWI_FACTOR;
    xQueSpace_ = SWI_FACTOR * AlignBytes(ubMaxPair_ * static_cast<int64_t>(sizeof(float)));
    dyQueSpace_ = AlignBytes(ubMaxPair_ * static_cast<int64_t>(sizeof(float)));
    half_ = xQueSpace_ / sizeof(float) / SWI_FACTOR;
    xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x));
    gradYGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(gradY));
    gradXOutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(gradXOut));
    if constexpr (isGroup) {
        groupIndexGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(groupIndex));
    }
    pipe_->InitBuffer(xQueue_, DB_BUFFER, xQueSpace_);
    pipe_->InitBuffer(dyQueue_, DB_BUFFER, dyQueSpace_);
    pipe_->InitBuffer(dxQueue_, 1, xQueSpace_);
}

template <typename T, bool isInterleaved, bool isGroup, typename Derived>
__aicore__ inline int64_t ClippedSwigluGradSchedBase<T, isInterleaved, isGroup, Derived>::AlignBytes(int64_t number)
{
    return (number + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
}

template <typename T, bool isInterleaved, bool isGroup, typename Derived>
__aicore__ inline void ClippedSwigluGradSchedBase<T, isInterleaved, isGroup, Derived>::Process()
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

template <typename T, bool isInterleaved, bool isGroup, typename Derived>
__aicore__ inline void ClippedSwigluGradSchedBase<T, isInterleaved, isGroup, Derived>::ComputeRealBatchSize()
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

template <typename T, bool isInterleaved, bool isGroup, typename Derived>
__aicore__ inline void ClippedSwigluGradSchedBase<T, isInterleaved, isGroup, Derived>::CalTilingParam()
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

template <typename T, bool isInterleaved, bool isGroup, typename Derived>
__aicore__ inline void ClippedSwigluGradSchedBase<T, isInterleaved, isGroup, Derived>::ProcessMainLoop()
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
                    static_cast<Derived*>(this)->ProcessSingleLoop(xOffset, dyOffset, dxOffset);
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
        static_cast<Derived*>(this)->ProcessSingleLoop(xOffset, dyOffset, dxOffset);
        xOffset += loopOffset_;
        dyOffset += loopOffset_ / SWI_FACTOR;
        dxOffset += loopOffset_;
    }
}

template <typename T, bool isInterleaved, bool isGroup, typename Derived>
__aicore__ inline void ClippedSwigluGradSchedBase<T, isInterleaved, isGroup, Derived>::CopyIn(int64_t xOffset,
                                                                                              int64_t dyOffset)
{
    if constexpr (!std::is_same_v<T, float>) {
        xLocalOffset1_ = (xQueSpace_ / SWI_FACTOR / static_cast<int64_t>(sizeof(T)) + BLOCK_ELEM - 1) / BLOCK_ELEM *
                         BLOCK_ELEM;
        xLocalOffset2_ = (xLocalOffset1_ / SWI_FACTOR + BLOCK_ELEM - 1) / BLOCK_ELEM * BLOCK_ELEM;
        dyLocalOffset_ = (dyQueSpace_ / static_cast<int64_t>(sizeof(T)) / SWI_FACTOR + BLOCK_ELEM - 1) / BLOCK_ELEM *
                         BLOCK_ELEM;
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

template <typename T, bool isInterleaved, bool isGroup, typename Derived>
__aicore__ inline void ClippedSwigluGradSchedBase<T, isInterleaved, isGroup, Derived>::CopyInHalfShortH(
    LocalTensor<T>& xDTypeLocal, LocalTensor<T>& dyDTypeLocal, int64_t xOffset, int64_t dyOffset)
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

template <typename T, bool isInterleaved, bool isGroup, typename Derived>
__aicore__ inline void ClippedSwigluGradSchedBase<T, isInterleaved, isGroup, Derived>::CopyInHalfLongH(
    LocalTensor<T>& xDTypeLocal, LocalTensor<T>& dyDTypeLocal, int64_t xOffset, int64_t dyOffset)
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

template <typename T, bool isInterleaved, bool isGroup, typename Derived>
__aicore__ inline void ClippedSwigluGradSchedBase<T, isInterleaved, isGroup, Derived>::CopyInInterLeaved(
    LocalTensor<T>& xDTypeLocal, LocalTensor<T>& dyDTypeLocal, int64_t xOffset, int64_t dyOffset)
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

template <typename T, bool isInterleaved, bool isGroup, typename Derived>
__aicore__ inline void ClippedSwigluGradSchedBase<T, isInterleaved, isGroup, Derived>::CopyOut(int64_t dxOffset)
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

template <typename T, bool isInterleaved, bool isGroup, typename Derived>
__aicore__ inline void ClippedSwigluGradSchedBase<T, isInterleaved, isGroup, Derived>::InitZeroBuffer()
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

template <typename T, bool isInterleaved, bool isGroup, typename Derived>
__aicore__ inline void ClippedSwigluGradSchedBase<T, isInterleaved, isGroup, Derived>::ZeroInvalidRows()
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
#endif // OPP_CLIPPED_SWIGLU_GRAD_BASE_H

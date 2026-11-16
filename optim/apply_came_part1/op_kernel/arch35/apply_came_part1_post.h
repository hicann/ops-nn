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
 * \file apply_came_part1_post.h
 * \brief
 */
#ifndef APPLY_CAME_PART1_POST
#define APPLY_CAME_PART1_POST

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "apply_came_part1_common.h"
#include "apply_came_part1_c_reduce_copy_out.h"

namespace ApplyCamePart1 {

using namespace AscendC;

template <typename T>
class ApplyCamePart1Post {
public:
    __aicore__ inline ApplyCamePart1Post(){};
    __aicore__ inline void Init(GM_ADDR grad, GM_ADDR eps, GM_ADDR sum_grad_r, GM_ADDR sum_grad_c, GM_ADDR sum_grad_rc,
                                GM_ADDR workspace, const ApplyCamePart1TilingData* tilingData,
                                LocalTensor<float> arena);
    __aicore__ inline void ProcessRShard(int64_t batchIdx, int64_t nCoreIdx);
    __aicore__ inline void ProcessCShard(int64_t batchIdx, int64_t mCoreIdx);
    __aicore__ inline void ProcessBatch(int64_t batchIdx);

private:
    __aicore__ inline void ParseTilingData(const ApplyCamePart1TilingData* tilingData);
    __aicore__ inline void Pre_Core_Compute(uint64_t gmOffsets, uint64_t cal_m, uint64_t base_m);
    __aicore__ inline void ReduceNarrowC(uint64_t gmOffsets, uint64_t outputCount, uint64_t rowCount,
                                         uint64_t rowsPerChunk);
    __aicore__ inline void ReduceWidth2C(uint64_t gmOffsets, uint64_t rowCount);
    __aicore__ inline void ReduceWorkspaceRPartial(LocalTensor<float> inputLocal, int64_t base, int64_t partialRows);
    __aicore__ inline void ReduceWorkspaceRShard(int64_t nCoreIdx);
    __aicore__ inline void ReduceWorkspaceRCPair(LocalTensor<float> high, LocalTensor<float> highRight,
                                                 LocalTensor<float> low, LocalTensor<float> lowRight,
                                                 LocalTensor<float> sumTmp, LocalTensor<float> recoveredTmp,
                                                 uint64_t leftOffset, uint64_t rightOffset, uint64_t copyCount);
    __aicore__ inline void ReduceWorkspaceRCOdd(LocalTensor<float> inputLocal, uint64_t oddIndex, uint64_t destination);
    __aicore__ inline void ReduceWorkspaceRCFinal(LocalTensor<float> inputLocal);
    __aicore__ inline void ReduceWorkspaceRC();
    __aicore__ inline void CopyWorkspaceCChunk(LocalTensor<float> highLocal, LocalTensor<float> lowLocal,
                                               uint64_t rowBase, uint64_t rowCount, uint64_t gmOffsets, uint64_t calM);
    __aicore__ inline void ReduceAdd(LocalTensor<float> accuUb, int64_t n, int64_t m);
    __aicore__ inline void StoreFloatAligned(GlobalTensor<float>& output, int64_t index, float value,
                                             LocalTensor<float> scratch);

private:
    int64_t postUbElements_{32768};
    LocalTensor<float> inputLocal_;
    LocalTensor<float> outputScratchLocal_;

    // Keep GM read-modify-write scratch separate from reduction input. Reusing
    // inputBuf_ here would overwrite values that the caller still needs to read.

    GlobalTensor<float> gmGrad_;
    GlobalTensor<float> gmEps_;
    GlobalTensor<float> gmSumGradR_;
    GlobalTensor<float> gmSumGradC_;
    GlobalTensor<float> gmSumGradRC_;

    GlobalTensor<float> workspaceSumGradR_;
    GlobalTensor<float> workspaceSumGradRLow_;
    GlobalTensor<float> workspaceSumGradC_;
    GlobalTensor<float> workspaceSumGradCLow_;
    GlobalTensor<float> workspaceSumGradRC_;
    GlobalTensor<float> workspaceSumGradRCLow_;

    GM_ADDR workspaceAddr_;

    // tiling params
    int64_t N{0};
    int64_t M{0};

    int64_t nLoopNormCore_{0};
    int64_t nLoopTailCore_{0};

    int64_t nNormalCoreNum_{0};
    int64_t nTailCoreNum_{0};

    int64_t mNormalCoreNum_{0};
    int64_t mTailCoreNum_{0};

    int64_t totalCoreNum_{0};
    int64_t usedCoreNum_{0};

    int64_t nCoreNum_{0};
    int64_t mCoreNum_{0};
    int64_t mLoopNumCore_{0};
    int64_t totalColumnTileCount_{0};
    int64_t cWorkspaceSlotSize_{0};

    bool IsContainsTailN{false};
    bool IsContainsTailM{false};
    int64_t batchIdx_{0};
    float epsValue_{0.0f};

    int64_t ONCE_HANDLE_NUM64{64};
    const int64_t ONCE_HANDLE_NUM512{512};
    const int64_t ONCE_ONE_SIZE8{8};
};

template <typename T>
__aicore__ inline void ApplyCamePart1Post<T>::ParseTilingData(const ApplyCamePart1TilingData* tilingData)
{
    // 总维度[N, M]
    ONCE_HANDLE_NUM64 = tilingData->tileSize;
    postUbElements_ = tilingData->postUbElements;
    N = tilingData->N;
    M = tilingData->M;

    // 单核矩阵维度 [nNormalCoreNum_, nTailCoreNum_]
    nNormalCoreNum_ = tilingData->nNormalCoreNum;
    nTailCoreNum_ = tilingData->nTailCoreNum;

    // 单核矩阵维度 [mNormalCoreNum_, mTailCoreNum_]
    mNormalCoreNum_ = tilingData->mNormalCoreNum;
    mTailCoreNum_ = tilingData->mTailCoreNum;

    // 循环次数
    nLoopNormCore_ = tilingData->nLoopNormCore;
    nLoopTailCore_ = tilingData->nLoopTailCore;
    mLoopNumCore_ = tilingData->mLoopNumCore;
    totalColumnTileCount_ = CeilDivApplyCamePart1(M, ONCE_HANDLE_NUM64);
    cWorkspaceSlotSize_ = M == 2 ? ONCE_ONE_SIZE8 : ONCE_HANDLE_NUM64;

    // 使用核数 && 总核数 [totalCoreNum_, usedCoreNum_]
    totalCoreNum_ = tilingData->totalCoreNum;
    usedCoreNum_ = tilingData->usedCoreNum;

    // 行列方向的核 [nCoreNum_, mCoreNum_]
    nCoreNum_ = tilingData->nCoreNum;
    mCoreNum_ = tilingData->mCoreNum;
}

template <typename T>
__aicore__ inline void ApplyCamePart1Post<T>::Init(GM_ADDR grad, GM_ADDR eps, GM_ADDR sum_grad_r, GM_ADDR sum_grad_c,
                                                   GM_ADDR sum_grad_rc, GM_ADDR workspace,
                                                   const ApplyCamePart1TilingData* tilingData, LocalTensor<float> arena)
{
    (void)grad;
    ParseTilingData(tilingData);

    gmEps_.SetGlobalBuffer((__gm__ float*)eps, 1);
    gmSumGradR_.SetGlobalBuffer((__gm__ float*)sum_grad_r);
    gmSumGradC_.SetGlobalBuffer((__gm__ float*)sum_grad_c);
    gmSumGradRC_.SetGlobalBuffer((__gm__ float*)sum_grad_rc);

    const int64_t groupIdx = GetBlockIdx() / (nCoreNum_ * mCoreNum_);
    int64_t workspaceOffsets = 0;
    workspaceSumGradRC_.SetGlobalBuffer((__gm__ float*)workspace + workspaceOffsets);
    int64_t nLoopCount = (nCoreNum_ - 1) * nLoopNormCore_ + nLoopTailCore_;
    constexpr int64_t kScalarSlotSize = 8;
    int64_t rcPartialCount = nLoopCount * totalColumnTileCount_;
    int64_t rPartialCount = nLoopCount * mCoreNum_;
    int64_t rcOffsets = (rcPartialCount * kScalarSlotSize + 128 - 1) / 128 * 128;
    workspaceOffsets = workspaceOffsets + rcOffsets;
    workspaceSumGradRCLow_.SetGlobalBuffer((__gm__ float*)workspace + workspaceOffsets);
    workspaceOffsets = workspaceOffsets + rcOffsets;
    // Keep workspace offsets 64-bit to avoid overflow.
    int64_t rOffsets = (rPartialCount * ONCE_HANDLE_NUM64 + 128 - 1) / 128 * 128;
    workspaceSumGradR_.SetGlobalBuffer((__gm__ float*)workspace + workspaceOffsets);
    workspaceOffsets = workspaceOffsets + rOffsets;
    workspaceSumGradRLow_.SetGlobalBuffer((__gm__ float*)workspace + workspaceOffsets);
    workspaceOffsets = workspaceOffsets + rOffsets;
    int64_t cOffsets = (rcPartialCount * cWorkspaceSlotSize_ + 128 - 1) / 128 * 128;
    workspaceSumGradC_.SetGlobalBuffer((__gm__ float*)workspace + workspaceOffsets);
    workspaceOffsets = workspaceOffsets + cOffsets;
    workspaceSumGradCLow_.SetGlobalBuffer((__gm__ float*)workspace + workspaceOffsets);
    int64_t regionFloats = workspaceOffsets + cOffsets;
    if (groupIdx > 0) {
        auto* regionBase = reinterpret_cast<__gm__ float*>(workspace) + groupIdx * regionFloats;
        workspaceSumGradRC_.SetGlobalBuffer(regionBase);
        workspaceSumGradRCLow_.SetGlobalBuffer(regionBase + rcOffsets);
        workspaceSumGradR_.SetGlobalBuffer(regionBase + 2 * rcOffsets);
        workspaceSumGradRLow_.SetGlobalBuffer(regionBase + 2 * rcOffsets + rOffsets);
        workspaceSumGradC_.SetGlobalBuffer(regionBase + 2 * rcOffsets + 2 * rOffsets);
        workspaceSumGradCLow_.SetGlobalBuffer(regionBase + 2 * rcOffsets + 2 * rOffsets + cOffsets);
    }

    inputLocal_ = arena;
    outputScratchLocal_ = arena[postUbElements_];
    DataCopyPad(inputLocal_, gmEps_, {1, sizeof(float), 0, 0, 0}, {false, 0, 0, 0});
    event_t eventIdMte2ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(eventIdMte2ToS);
    WaitFlag<HardEvent::MTE2_S>(eventIdMte2ToS);
    epsValue_ = inputLocal_.GetValue(0);
}

template <typename T>
__aicore__ inline void ApplyCamePart1Post<T>::StoreFloatAligned(GlobalTensor<float>& output, int64_t index, float value,
                                                                LocalTensor<float> scratch)
{
    (void)scratch;
    AtomicAdd(output.GetPhyAddr(static_cast<uint64_t>(index)), value);
    event_t eventIdSToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
    SetFlag<HardEvent::S_MTE3>(eventIdSToMte3);
    WaitFlag<HardEvent::S_MTE3>(eventIdSToMte3);
}

template <typename T>
__aicore__ inline void ApplyCamePart1Post<T>::ProcessBatch(int64_t batchIdx)
{
    batchIdx_ = batchIdx;
    ReduceWorkspaceRC();
}

template <typename T>
__aicore__ inline void ApplyCamePart1Post<T>::ProcessCShard(int64_t batchIdx, int64_t mCoreIdx)
{
    if (M == 1 || mCoreIdx < 0 || mCoreIdx >= mCoreNum_) {
        return;
    }
    batchIdx_ = batchIdx;
    const uint64_t totalNPartials = (nCoreNum_ - 1) * nLoopNormCore_ + nLoopTailCore_;
    // A single N partial keeps C high/low plus three vector scratch rows alive
    // while eps is merged.  Bound the column chunk by that five-row live set.
    // Multi-partial C needs at least separate high/low rows while loading.
    const uint64_t liveRows = totalNPartials == 1 ? 5 : 2;
    uint64_t columnTilesPerChunk = postUbElements_ / (liveRows * ONCE_HANDLE_NUM64);
    columnTilesPerChunk = columnTilesPerChunk > 0 ? columnTilesPerChunk : 1;
    const uint64_t baseTilesPerCore = totalColumnTileCount_ / mCoreNum_;
    const uint64_t extraTileCoreCount = totalColumnTileCount_ % mCoreNum_;
    const uint64_t shardTileCount = baseTilesPerCore + (static_cast<uint64_t>(mCoreIdx) < extraTileCoreCount ? 1 : 0);
    const uint64_t shardTileStart = static_cast<uint64_t>(mCoreIdx) * baseTilesPerCore +
                                    (static_cast<uint64_t>(mCoreIdx) < extraTileCoreCount ?
                                         static_cast<uint64_t>(mCoreIdx) :
                                         extraTileCoreCount);
    uint64_t shardTileOffset = 0;

    while (shardTileOffset < shardTileCount) {
        const uint64_t chunkTiles = shardTileCount - shardTileOffset < columnTilesPerChunk ?
                                        shardTileCount - shardTileOffset :
                                        columnTilesPerChunk;
        const uint64_t gmOffsets = (shardTileStart + shardTileOffset) * ONCE_HANDLE_NUM64;
        const uint64_t calM = chunkTiles * ONCE_HANDLE_NUM64;
        const uint64_t baseM = M - gmOffsets < calM ? M - gmOffsets : calM;
        Pre_Core_Compute(gmOffsets, calM, baseM);
        shardTileOffset += chunkTiles;
    }
}

template <typename T>
__aicore__ inline void ApplyCamePart1Post<T>::ProcessRShard(int64_t batchIdx, int64_t nCoreIdx)
{
    batchIdx_ = batchIdx;
    ReduceWorkspaceRShard(nCoreIdx);
}

template <typename T>
__aicore__ inline void ApplyCamePart1Post<T>::ReduceWorkspaceRPartial(LocalTensor<float> inputLocal, int64_t base,
                                                                      int64_t partialRows)
{
    const int64_t rowWidth = ONCE_HANDLE_NUM64;
    const int64_t maxRowsPerCopy = postUbElements_ / (6 * rowWidth);
    int64_t activeRows = partialRows;
    while (activeRows > 1) {
        const int64_t rightRows = (activeRows + 1) / 2;
        const int64_t pairRows = activeRows - rightRows;
        for (int64_t offset = 0; offset < pairRows; offset += maxRowsPerCopy) {
            const int64_t copyRows = (pairRows - offset < maxRowsPerCopy) ? pairRows - offset : maxRowsPerCopy;
            const int64_t copyElements = copyRows * rowWidth;
            LocalTensor<float> high = inputLocal;
            LocalTensor<float> highRight = inputLocal[copyElements];
            LocalTensor<float> low = inputLocal[2 * copyElements];
            LocalTensor<float> lowRight = inputLocal[3 * copyElements];
            LocalTensor<float> sumTmp = inputLocal[4 * copyElements];
            LocalTensor<float> recoveredTmp = inputLocal[5 * copyElements];
            DataCopy(high, workspaceSumGradR_[base + offset * rowWidth], copyElements);
            DataCopy(highRight, workspaceSumGradR_[base + (rightRows + offset) * rowWidth], copyElements);
            DataCopy(low, workspaceSumGradRLow_[base + offset * rowWidth], copyElements);
            DataCopy(lowRight, workspaceSumGradRLow_[base + (rightRows + offset) * rowWidth], copyElements);
            event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
            WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);

            Add(sumTmp, high, highRight, copyElements);
            PipeBarrier<PIPE_V>();
            Sub(recoveredTmp, sumTmp, high, copyElements);
            PipeBarrier<PIPE_V>();
            Sub(highRight, highRight, recoveredTmp, copyElements);
            Sub(recoveredTmp, sumTmp, recoveredTmp, copyElements);
            PipeBarrier<PIPE_V>();
            Sub(high, high, recoveredTmp, copyElements);
            PipeBarrier<PIPE_V>();
            Add(high, high, highRight, copyElements);
            Add(low, low, lowRight, copyElements);
            PipeBarrier<PIPE_V>();
            Add(low, low, high, copyElements);
            Adds(high, sumTmp, static_cast<float>(0), copyElements);
            PipeBarrier<PIPE_V>();
            NormalizeCompensationApplyCamePart1(high, low, highRight, recoveredTmp, copyElements);
            event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
            SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
            WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
            DataCopy(workspaceSumGradR_[base + offset * rowWidth], high, copyElements);
            DataCopy(workspaceSumGradRLow_[base + offset * rowWidth], low, copyElements);
            event_t eventIdMte3ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
            SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
            WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
        }
        if ((activeRows & 1) != 0) {
            const int64_t oddIndex = rightRows - 1;
            DataCopy(inputLocal, workspaceSumGradR_[base + oddIndex * rowWidth], rowWidth);
            DataCopy(inputLocal[rowWidth], workspaceSumGradRLow_[base + oddIndex * rowWidth], rowWidth);
            event_t eventIdMte2ToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3));
            SetFlag<HardEvent::MTE2_MTE3>(eventIdMte2ToMte3);
            WaitFlag<HardEvent::MTE2_MTE3>(eventIdMte2ToMte3);
            DataCopy(workspaceSumGradR_[base + pairRows * rowWidth], inputLocal, rowWidth);
            DataCopy(workspaceSumGradRLow_[base + pairRows * rowWidth], inputLocal[rowWidth], rowWidth);
            event_t eventIdMte3ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
            SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
            WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
        }
        activeRows = rightRows;
    }
}

template <typename T>
__aicore__ inline void ApplyCamePart1Post<T>::ReduceWorkspaceRShard(int64_t nCoreIdx)
{
    if (nCoreIdx < 0 || nCoreIdx >= nCoreNum_) {
        return;
    }
    LocalTensor<float> inputLocal = inputLocal_;
    const int64_t rowWidth = ONCE_HANDLE_NUM64;
    const int64_t partialRows = mCoreNum_;

    const int64_t loopCount = (nCoreIdx == nCoreNum_ - 1) ? nLoopTailCore_ : nLoopNormCore_;
    const int64_t rowsInCore = (nCoreIdx == nCoreNum_ - 1) ? nTailCoreNum_ : nNormalCoreNum_;
    for (int64_t nLoopIdx = 0; nLoopIdx < loopCount; ++nLoopIdx) {
        const int64_t base = (nCoreIdx * nLoopNormCore_ + nLoopIdx) * partialRows * rowWidth;
        ReduceWorkspaceRPartial(inputLocal, base, partialRows);
        const int64_t rowOffset = batchIdx_ * N + nCoreIdx * nNormalCoreNum_ + nLoopIdx * rowWidth;
        const int64_t rowCount = (rowsInCore - nLoopIdx * rowWidth < rowWidth) ? rowsInCore - nLoopIdx * rowWidth :
                                                                                 rowWidth;
        DataCopy(inputLocal, workspaceSumGradR_[base], rowWidth);
        DataCopy(inputLocal[rowWidth], workspaceSumGradRLow_[base], rowWidth);
        event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        // Fold eps into the compensation before the final rounding.  Adding it
        // after high + low introduces a second FP32 rounding and can lose a
        // small eps contribution for short rows with a large squared sum.
        float epsHigh = 0.0f;
        float epsLow = 0.0f;
        ScaleEpsApplyCamePart1(epsValue_, M, 1, epsHigh, epsLow);
        AddCompensatedScalarApplyCamePart1(inputLocal, inputLocal[rowWidth], inputLocal[2 * rowWidth],
                                           inputLocal[3 * rowWidth], inputLocal[4 * rowWidth], epsHigh, epsLow,
                                           rowWidth);
        Add(inputLocal, inputLocal, inputLocal[rowWidth], rowWidth);
        // Atomics keep adjacent unaligned GM blocks safe when different batch
        // groups publish concurrently.
        event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventIdVToS);
        WaitFlag<HardEvent::V_S>(eventIdVToS);
        const uint64_t prefix = ((8 - rowOffset % 8) % 8 < rowCount ? (8 - rowOffset % 8) % 8 : rowCount);
        for (uint64_t i = 0; i < prefix; ++i) {
            AtomicAdd(gmSumGradR_.GetPhyAddr(rowOffset + i), inputLocal.GetValue(i));
        }
        const uint64_t body = (rowCount - prefix) / 8 * 8;
        LocalTensor<float> alignedBody = inputLocal_[rowWidth];
        for (uint64_t i = 0; i < body; ++i) {
            alignedBody.SetValue(i, inputLocal.GetValue(prefix + i));
        }
        event_t scalarToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
        SetFlag<HardEvent::S_MTE3>(scalarToMte3);
        WaitFlag<HardEvent::S_MTE3>(scalarToMte3);
        AtomicAddAlignedCReduceResult(gmSumGradR_, alignedBody, rowOffset + prefix, body);
        for (uint64_t i = prefix + body; i < static_cast<uint64_t>(rowCount); ++i) {
            AtomicAdd(gmSumGradR_.GetPhyAddr(rowOffset + i), inputLocal.GetValue(i));
        }
        event_t eventIdSToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE2));
        SetFlag<HardEvent::S_MTE2>(eventIdSToMte2);
        WaitFlag<HardEvent::S_MTE2>(eventIdSToMte2);
    }
}

template <typename T>
__aicore__ inline void ApplyCamePart1Post<T>::ReduceWorkspaceRCPair(
    LocalTensor<float> high, LocalTensor<float> highRight, LocalTensor<float> low, LocalTensor<float> lowRight,
    LocalTensor<float> sumTmp, LocalTensor<float> recoveredTmp, uint64_t leftOffset, uint64_t rightOffset,
    uint64_t copyCount)
{
    constexpr uint64_t kScalarSlotSize = 8;
    const uint64_t copyElements = copyCount * kScalarSlotSize;
    DataCopy(high, workspaceSumGradRC_[leftOffset * kScalarSlotSize], copyElements);
    DataCopy(highRight, workspaceSumGradRC_[rightOffset * kScalarSlotSize], copyElements);
    DataCopy(low, workspaceSumGradRCLow_[leftOffset * kScalarSlotSize], copyElements);
    DataCopy(lowRight, workspaceSumGradRCLow_[rightOffset * kScalarSlotSize], copyElements);
    event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);

    Add(sumTmp, high, highRight, copyElements);
    PipeBarrier<PIPE_V>();
    Sub(recoveredTmp, sumTmp, high, copyElements);
    PipeBarrier<PIPE_V>();
    Sub(highRight, highRight, recoveredTmp, copyElements);
    Sub(recoveredTmp, sumTmp, recoveredTmp, copyElements);
    PipeBarrier<PIPE_V>();
    Sub(high, high, recoveredTmp, copyElements);
    PipeBarrier<PIPE_V>();
    Add(high, high, highRight, copyElements);
    Add(low, low, lowRight, copyElements);
    PipeBarrier<PIPE_V>();
    Add(low, low, high, copyElements);
    Adds(high, sumTmp, static_cast<float>(0), copyElements);
    event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    PipeBarrier<PIPE_V>();
    NormalizeCompensationApplyCamePart1(high, low, highRight, recoveredTmp, copyElements);
    SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
    WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);

    DataCopy(workspaceSumGradRC_[leftOffset * kScalarSlotSize], high, copyElements);
    DataCopy(workspaceSumGradRCLow_[leftOffset * kScalarSlotSize], low, copyElements);
    event_t eventIdMte3ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
    WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
}

template <typename T>
__aicore__ inline void ApplyCamePart1Post<T>::ReduceWorkspaceRCOdd(LocalTensor<float> inputLocal, uint64_t oddIndex,
                                                                   uint64_t destination)
{
    constexpr uint64_t kScalarSlotSize = 8;
    DataCopy(inputLocal, workspaceSumGradRC_[oddIndex * kScalarSlotSize], kScalarSlotSize);
    LocalTensor<float> lowLocal = inputLocal[kScalarSlotSize];
    DataCopy(lowLocal, workspaceSumGradRCLow_[oddIndex * kScalarSlotSize], kScalarSlotSize);
    event_t eventIdMte2ToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3));
    SetFlag<HardEvent::MTE2_MTE3>(eventIdMte2ToMte3);
    WaitFlag<HardEvent::MTE2_MTE3>(eventIdMte2ToMte3);
    DataCopy(workspaceSumGradRC_[destination * kScalarSlotSize], inputLocal, kScalarSlotSize);
    DataCopy(workspaceSumGradRCLow_[destination * kScalarSlotSize], lowLocal, kScalarSlotSize);
    event_t eventIdMte3ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
    WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
}

template <typename T>
__aicore__ inline void ApplyCamePart1Post<T>::ReduceWorkspaceRCFinal(LocalTensor<float> inputLocal)
{
    constexpr uint64_t kScalarSlotSize = 8;
    LocalTensor<float> lowLocal = inputLocal[kScalarSlotSize];
    DataCopy(inputLocal, workspaceSumGradRC_, kScalarSlotSize);
    DataCopy(lowLocal, workspaceSumGradRCLow_, kScalarSlotSize);
    event_t eventIdMte2ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(eventIdMte2ToS);
    WaitFlag<HardEvent::MTE2_S>(eventIdMte2ToS);
    float epsHigh = 0.0f;
    float epsLow = 0.0f;
    ScaleEpsApplyCamePart1(epsValue_, N, M, epsHigh, epsLow);
    float rcHigh = 0.0f;
    float rcLow = 0.0f;
    CompensatedAddApplyCamePart1(inputLocal.GetValue(0), lowLocal.GetValue(0), epsHigh, epsLow, rcHigh, rcLow);
    const float rcValue = rcHigh + rcLow;
    StoreFloatAligned(gmSumGradRC_, batchIdx_, rcValue, outputScratchLocal_);
    if (M == 1) {
        StoreFloatAligned(gmSumGradC_, batchIdx_, rcValue, outputScratchLocal_);
    }
    event_t eventIdSToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE2));
    SetFlag<HardEvent::S_MTE2>(eventIdSToMte2);
    WaitFlag<HardEvent::S_MTE2>(eventIdSToMte2);
}

template <typename T>
__aicore__ inline void ApplyCamePart1Post<T>::ReduceWorkspaceRC()
{
    LocalTensor<float> inputLocal = inputLocal_;
    constexpr uint64_t kScalarSlotSize = 8;
    const uint64_t kMaxPairCount = static_cast<uint64_t>(postUbElements_) / (6 * 8);
    const uint64_t kMaxPairElements = kMaxPairCount * kScalarSlotSize;
    LocalTensor<float> high = inputLocal;
    LocalTensor<float> highRight = inputLocal[kMaxPairElements];
    LocalTensor<float> low = inputLocal[2 * kMaxPairElements];
    LocalTensor<float> lowRight = inputLocal[3 * kMaxPairElements];
    LocalTensor<float> sumTmp = inputLocal[4 * kMaxPairElements];
    LocalTensor<float> recoveredTmp = inputLocal[5 * kMaxPairElements];
    const uint64_t nLoopCount = (nCoreNum_ - 1) * nLoopNormCore_ + nLoopTailCore_;
    uint64_t activeCount = nLoopCount * totalColumnTileCount_;

    while (activeCount > 1) {
        const uint64_t rightCount = (activeCount + 1) / 2;
        const uint64_t pairCount = activeCount - rightCount;
        for (uint64_t offset = 0; offset < pairCount; offset += kMaxPairCount) {
            const uint64_t copyCount = (pairCount - offset < kMaxPairCount) ? pairCount - offset : kMaxPairCount;
            ReduceWorkspaceRCPair(high, highRight, low, lowRight, sumTmp, recoveredTmp, offset, rightCount + offset,
                                  copyCount);
        }
        if ((activeCount & 1U) != 0U) {
            ReduceWorkspaceRCOdd(inputLocal, rightCount - 1, pairCount);
        }
        activeCount = rightCount;
    }
    ReduceWorkspaceRCFinal(inputLocal);
}

template <typename T>
__aicore__ inline void ApplyCamePart1Post<T>::CopyWorkspaceCChunk(LocalTensor<float> highLocal,
                                                                  LocalTensor<float> lowLocal, uint64_t rowBase,
                                                                  uint64_t rowCount, uint64_t gmOffsets, uint64_t calM)
{
    const uint64_t workspaceRowStride = totalColumnTileCount_ * cWorkspaceSlotSize_;
    for (uint64_t row = 0; row < rowCount; ++row) {
        const uint64_t workspaceOffset = (rowBase + row) * workspaceRowStride + gmOffsets;
        DataCopy(highLocal[row * calM], workspaceSumGradC_[workspaceOffset], calM);
        DataCopy(lowLocal[row * calM], workspaceSumGradCLow_[workspaceOffset], calM);
    }
}

template <typename T>
__aicore__ inline void ApplyCamePart1Post<T>::ReduceWidth2C(uint64_t gmOffsets, uint64_t rowCount)
{
    constexpr uint64_t width = 8;
    constexpr uint64_t columns = 2;
    constexpr uint64_t vectorLanes = 64;
    constexpr uint64_t scalarSlotSize = 8;
    const uint64_t elementsPerRow = 2 * width + 4;
    uint64_t rowsPerChunk = (static_cast<uint64_t>(postUbElements_) - 2 * scalarSlotSize) / elementsPerRow;
    rowsPerChunk = rowsPerChunk / vectorLanes * vectorLanes;
    if (rowsPerChunk == 0) {
        const uint64_t fallbackRows = (static_cast<uint64_t>(postUbElements_) - 2 * width) / (2 * width);
        ReduceNarrowC(gmOffsets, columns, rowCount, fallbackRows);
        return;
    }

    const uint64_t sourceElements = rowsPerChunk * width;
    LocalTensor<float> sourceHigh = inputLocal_;
    LocalTensor<float> sourceLow = sourceHigh[sourceElements];
    LocalTensor<float> gatheredHigh = sourceLow[sourceElements];
    LocalTensor<float> gatheredLow = gatheredHigh[rowsPerChunk];
    LocalTensor<float> sumTmp = gatheredLow[rowsPerChunk];
    LocalTensor<float> recoveredTmp = sumTmp[rowsPerChunk];
    LocalTensor<float> reduced = recoveredTmp[rowsPerChunk];

    float accumulatedHigh[columns] = {0.0f, 0.0f};
    float accumulatedLow[columns] = {0.0f, 0.0f};
    for (uint64_t start = 0; start < rowCount; start += rowsPerChunk) {
        const uint64_t count = rowCount - start < rowsPerChunk ? rowCount - start : rowsPerChunk;
        CopyWorkspaceCChunk(sourceHigh, sourceLow, start, count, gmOffsets, width);
        event_t mte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(mte2ToV);
        WaitFlag<HardEvent::MTE2_V>(mte2ToV);
        for (uint64_t column = 0; column < columns; ++column) {
            Duplicate(gatheredHigh, 0.0f, rowsPerChunk);
            Duplicate(gatheredLow, 0.0f, rowsPerChunk);
            PipeBarrier<PIPE_V>();
            GatherStridedColumnApplyCamePart1Vf(reinterpret_cast<__ubuf__ float*>(gatheredHigh.GetPhyAddr()),
                                                reinterpret_cast<__ubuf__ float*>(sourceHigh.GetPhyAddr()),
                                                static_cast<uint32_t>(count), static_cast<uint32_t>(width),
                                                static_cast<uint32_t>(column));
            GatherStridedColumnApplyCamePart1Vf(reinterpret_cast<__ubuf__ float*>(gatheredLow.GetPhyAddr()),
                                                reinterpret_cast<__ubuf__ float*>(sourceLow.GetPhyAddr()),
                                                static_cast<uint32_t>(count), static_cast<uint32_t>(width),
                                                static_cast<uint32_t>(column));
            PipeBarrier<PIPE_V>();
            Duplicate(reduced, 0.0f, 2 * scalarSlotSize);
            CompensatedReduceApplyCamePart1(reduced, gatheredHigh, gatheredLow, sumTmp, recoveredTmp, count);
            event_t vToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
            SetFlag<HardEvent::V_S>(vToS);
            WaitFlag<HardEvent::V_S>(vToS);
            float high = 0.0f;
            float low = 0.0f;
            CompensatedAddApplyCamePart1(accumulatedHigh[column], accumulatedLow[column], reduced.GetValue(0),
                                         reduced.GetValue(scalarSlotSize), high, low);
            accumulatedHigh[column] = high;
            accumulatedLow[column] = low;
            event_t sToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
            SetFlag<HardEvent::S_V>(sToV);
            WaitFlag<HardEvent::S_V>(sToV);
        }
    }

    float epsHigh = 0.0f;
    float epsLow = 0.0f;
    ScaleEpsApplyCamePart1(epsValue_, N, 1, epsHigh, epsLow);
    Duplicate(sourceHigh, 0.0f, scalarSlotSize);
    event_t vToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(vToS);
    WaitFlag<HardEvent::V_S>(vToS);
    for (uint64_t column = 0; column < columns; ++column) {
        float high = 0.0f;
        float low = 0.0f;
        CompensatedAddApplyCamePart1(accumulatedHigh[column], accumulatedLow[column], epsHigh, epsLow, high, low);
        sourceHigh.SetValue(column, high + low);
    }
    event_t sToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
    SetFlag<HardEvent::S_MTE3>(sToMte3);
    WaitFlag<HardEvent::S_MTE3>(sToMte3);
    AtomicAddCReduceResult(gmSumGradC_, sourceHigh, batchIdx_ * M + gmOffsets, columns);
}

template <typename T>
__aicore__ inline void ApplyCamePart1Post<T>::ReduceNarrowC(uint64_t gmOffsets, uint64_t outputCount, uint64_t rowCount,
                                                            uint64_t rowsPerChunk)
{
    const uint64_t width = static_cast<uint64_t>(cWorkspaceSlotSize_);
    LocalTensor<float> input = inputLocal_;
    LocalTensor<float> inputLow = input[rowsPerChunk * width];
    LocalTensor<float> high = input[2 * rowsPerChunk * width];
    LocalTensor<float> low = input[(2 * rowsPerChunk + 1) * width];
    Duplicate(high, 0.0f, width);
    Duplicate(low, 0.0f, width);
    for (uint64_t start = 0; start < rowCount; start += rowsPerChunk) {
        Duplicate(input, 0.0f, rowsPerChunk * width);
        event_t vToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        SetFlag<HardEvent::V_MTE2>(vToMte2);
        WaitFlag<HardEvent::V_MTE2>(vToMte2);
        const uint64_t count = rowCount - start < rowsPerChunk ? rowCount - start : rowsPerChunk;
        CopyWorkspaceCChunk(input, inputLow, start, count, gmOffsets, width);
        event_t mte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(mte2ToV);
        WaitFlag<HardEvent::MTE2_V>(mte2ToV);
        Add(input, input, inputLow, rowsPerChunk * width);
        ReduceAdd(input, rowsPerChunk, width);
        event_t vToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(vToS);
        WaitFlag<HardEvent::V_S>(vToS);
        for (uint64_t col = 0; col < outputCount; ++col) {
            float h = 0.0f;
            float l = 0.0f;
            CompensatedAddApplyCamePart1(high.GetValue(col), low.GetValue(col), input.GetValue(col), 0.0f, h, l);
            high.SetValue(col, h);
            low.SetValue(col, l);
        }
        event_t sToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(sToV);
        WaitFlag<HardEvent::S_V>(sToV);
    }
    float epsHigh = 0.0f;
    float epsLow = 0.0f;
    ScaleEpsApplyCamePart1(epsValue_, N, 1, epsHigh, epsLow);
    AddCompensatedScalarApplyCamePart1(high, low, input, input[width], input[2 * width], epsHigh, epsLow, width);
    Add(input, high, low, width);
    AtomicAddCReduceResult(gmSumGradC_, input, batchIdx_ * M + gmOffsets, outputCount);
}

template <typename T>
__aicore__ inline void ApplyCamePart1Post<T>::Pre_Core_Compute(uint64_t gmOffsets, uint64_t cal_m, uint64_t base_m)
{
    uint64_t total_n = (nCoreNum_ - 1) * nLoopNormCore_ + nLoopTailCore_;
    if (M == 2 && total_n > 1) {
        ReduceWidth2C(gmOffsets, total_n);
        return;
    }
    if (cal_m == static_cast<uint64_t>(ONCE_HANDLE_NUM64) && total_n > 512) {
        // Reserve two aligned rows for compensated accumulation in the same arena.
        const uint64_t width = static_cast<uint64_t>(ONCE_HANDLE_NUM64);
        const uint64_t availableRows = static_cast<uint64_t>(postUbElements_) / width;
        if (availableRows >= 4) {
            uint64_t rowsPerChunk = 1;
            while (rowsPerChunk < 256 && rowsPerChunk * 4 <= availableRows - 2) {
                rowsPerChunk *= 2;
            }
            const uint64_t outputCount = (M - gmOffsets < base_m) ? M - gmOffsets : base_m;
            ReduceNarrowC(gmOffsets, outputCount, total_n, rowsPerChunk);
            return;
        }
    }
    if (total_n == 1) {
        const uint64_t compute_m = M == 2 ? static_cast<uint64_t>(cWorkspaceSlotSize_) : cal_m;
        LocalTensor<float> high = inputLocal_;
        LocalTensor<float> low = inputLocal_[compute_m];
        DataCopy(high, workspaceSumGradC_[gmOffsets], compute_m);
        DataCopy(low, workspaceSumGradCLow_[gmOffsets], compute_m);
        event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        float epsHigh = 0.0f;
        float epsLow = 0.0f;
        ScaleEpsApplyCamePart1(epsValue_, N, 1, epsHigh, epsLow);
        AddCompensatedScalarApplyCamePart1(high, low, inputLocal_[2 * compute_m], inputLocal_[3 * compute_m],
                                           inputLocal_[4 * compute_m], epsHigh, epsLow, compute_m);
        Add(high, high, low, compute_m);
        const uint64_t outputCount = (M - gmOffsets < base_m) ? M - gmOffsets : base_m;
        AtomicAddCReduceResult(gmSumGradC_, high, batchIdx_ * M + gmOffsets, outputCount);
        return;
    }

    uint64_t pre_loop_n = 1;
    while (pre_loop_n < total_n) {
        pre_loop_n = pre_loop_n << 1;
        if (2 * pre_loop_n * cal_m > static_cast<uint64_t>(postUbElements_)) {
            pre_loop_n = pre_loop_n >> 1;
            break;
        }
        if (pre_loop_n >= total_n) {
            break;
        }
    }
    uint64_t loop_time = (total_n + pre_loop_n - 1) / pre_loop_n;
    uint64_t last_loop_n = total_n - (loop_time - 1) * pre_loop_n;
    // Keep the original row grouping, which is based on the logical 128-wide
    // tile, while operating only on the aligned compact C slot for M == 2.
    const uint64_t compute_m = M == 2 ? static_cast<uint64_t>(cWorkspaceSlotSize_) : cal_m;
    LocalTensor<float> inputLocal = inputLocal_;
    event_t eventIdSToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE2));
    SetFlag<HardEvent::S_MTE2>(eventIdSToMte2);
    WaitFlag<HardEvent::S_MTE2>(eventIdSToMte2);

    for (int64_t i = 0; i < loop_time - 1; i++) {
        LocalTensor<float> lowLocal = inputLocal[pre_loop_n * compute_m];
        CopyWorkspaceCChunk(inputLocal, lowLocal, i * pre_loop_n, pre_loop_n, gmOffsets, compute_m);
        event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        Add(inputLocal, inputLocal, lowLocal, pre_loop_n * compute_m);
        ReduceAdd(inputLocal, pre_loop_n, compute_m);
        const uint64_t outputCount = (M - gmOffsets < base_m) ? M - gmOffsets : base_m;
        AtomicAddCReduceResult(gmSumGradC_, inputLocal, batchIdx_ * M + gmOffsets, outputCount);
    }

    constexpr float scalarValue = 0;
    Duplicate(inputLocal, scalarValue, 2 * pre_loop_n * compute_m);
    event_t eventIdVToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
    SetFlag<HardEvent::V_MTE2>(eventIdVToMte2);
    WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2);
    LocalTensor<float> lastLowLocal = inputLocal[pre_loop_n * compute_m];
    CopyWorkspaceCChunk(inputLocal, lastLowLocal, (loop_time - 1) * pre_loop_n, last_loop_n, gmOffsets, compute_m);
    event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    Add(inputLocal, inputLocal, lastLowLocal, pre_loop_n * compute_m);
    ReduceAdd(inputLocal, pre_loop_n, compute_m);
    float epsHigh = 0.0f;
    float epsLow = 0.0f;
    ScaleEpsApplyCamePart1(epsValue_, N, 1, epsHigh, epsLow);
    if (5 * compute_m <= static_cast<uint64_t>(postUbElements_)) {
        LocalTensor<float> low = inputLocal[compute_m];
        Duplicate(low, 0.0f, compute_m);
        PipeBarrier<PIPE_V>();
        AddCompensatedScalarApplyCamePart1(inputLocal, low, inputLocal[2 * compute_m], inputLocal[3 * compute_m],
                                           inputLocal[4 * compute_m], epsHigh, epsLow, compute_m);
        Add(inputLocal, inputLocal, low, compute_m);
    } else {
        Adds(inputLocal, inputLocal, epsHigh, compute_m);
        Adds(inputLocal, inputLocal, epsLow, compute_m);
    }
    const uint64_t outputCount = (M - gmOffsets < base_m) ? M - gmOffsets : base_m;
    AtomicAddCReduceResult(gmSumGradC_, inputLocal, batchIdx_ * M + gmOffsets, outputCount);
}

template <typename T>
__aicore__ inline void ApplyCamePart1Post<T>::ReduceAdd(LocalTensor<float> accuUb, int64_t n, int64_t m)
{
    BinaryReduceRowsApplyCamePart1(accuUb, n, m);
}

} // namespace ApplyCamePart1
#endif // APPLY_CAME_PART1_POST

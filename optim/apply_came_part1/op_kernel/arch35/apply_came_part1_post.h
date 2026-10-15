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

namespace ApplyCamePart1 {

using namespace AscendC;

template <typename T>
class ApplyCamePart1Post {
public:
    __aicore__ inline ApplyCamePart1Post(){};
    __aicore__ inline void Init(GM_ADDR grad, GM_ADDR eps, GM_ADDR sum_grad_r, GM_ADDR sum_grad_c, GM_ADDR sum_grad_rc,
                                GM_ADDR workspace, const ApplyCamePart1TilingData* tilingData);
    __aicore__ inline void ProcessBatch(int64_t batchIdx);

private:
    __aicore__ inline void ParseTilingData(const ApplyCamePart1TilingData* tilingData);
    __aicore__ inline void Pre_Core_Compute(uint64_t gmOffsets, uint64_t cal_m, uint64_t base_m);
    __aicore__ inline void ReduceWorkspaceRPartial(LocalTensor<float> inputLocal, int64_t base, int64_t partialRows);
    __aicore__ inline void ReduceWorkspaceR();
    __aicore__ inline void ReduceWorkspaceRCPair(LocalTensor<float> high, LocalTensor<float> highRight,
                                                 LocalTensor<float> low, LocalTensor<float> lowRight,
                                                 LocalTensor<float> sumTmp, LocalTensor<float> recoveredTmp,
                                                 uint64_t leftOffset, uint64_t rightOffset, uint64_t copyCount);
    __aicore__ inline void ReduceWorkspaceRCOdd(LocalTensor<float> inputLocal, uint64_t oddIndex, uint64_t destination);
    __aicore__ inline void ReduceWorkspaceRCFinal(LocalTensor<float> inputLocal);
    __aicore__ inline void ReduceWorkspaceRC();
    __aicore__ inline void CopyWorkspaceCChunk(LocalTensor<float> inputLocal, uint64_t rowBase, uint64_t rowCount,
                                               uint64_t gmOffsets, uint64_t calM);
    __aicore__ inline void ReduceAdd(LocalTensor<float> accuUb, int64_t n, int64_t m);
    __aicore__ inline void StoreFloatAligned(GlobalTensor<float>& output, int64_t index, float value,
                                             LocalTensor<float> scratch);

private:
    TPipe pipe;
    TBuf<QuePosition::VECCALC> inputBuf_;
    // Keep GM read-modify-write scratch separate from reduction input. Reusing
    // inputBuf_ here would overwrite values that the caller still needs to read.
    TBuf<QuePosition::VECCALC> outputScratchBuf_;

    GlobalTensor<float> gmGrad_;
    GlobalTensor<float> gmEps_;
    GlobalTensor<float> gmSumGradR_;
    GlobalTensor<float> gmSumGradC_;
    GlobalTensor<float> gmSumGradRC_;

    GlobalTensor<float> workspaceSumGradR_;
    GlobalTensor<float> workspaceSumGradC_;
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

    bool IsContainsTailN{false};
    bool IsContainsTailM{false};
    int64_t batchIdx_{0};

    const int64_t ONCE_HANDLE_NUM64{64};
    const int64_t ONCE_HANDLE_NUM512{512};
    const int64_t ONCE_ONE_SIZE8{8};
};

template <typename T>
__aicore__ inline void ApplyCamePart1Post<T>::ParseTilingData(const ApplyCamePart1TilingData* tilingData)
{
    // 总维度[N, M]
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
                                                   const ApplyCamePart1TilingData* tilingData)
{
    (void)grad;
    (void)eps;
    ParseTilingData(tilingData);

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
    int64_t cOffsets = (rcPartialCount * ONCE_HANDLE_NUM64 + 128 - 1) / 128 * 128;
    workspaceSumGradC_.SetGlobalBuffer((__gm__ float*)workspace + workspaceOffsets);
    int64_t regionFloats = workspaceOffsets + cOffsets;
    if (groupIdx > 0) {
        auto* regionBase = reinterpret_cast<__gm__ float*>(workspace) + groupIdx * regionFloats;
        workspaceSumGradRC_.SetGlobalBuffer(regionBase);
        workspaceSumGradRCLow_.SetGlobalBuffer(regionBase + rcOffsets);
        workspaceSumGradR_.SetGlobalBuffer(regionBase + 2 * rcOffsets);
        workspaceSumGradC_.SetGlobalBuffer(regionBase + 2 * rcOffsets + rOffsets);
    }

    pipe.InitBuffer(inputBuf_, ONCE_HANDLE_NUM512 * ONCE_HANDLE_NUM64 * sizeof(float));
    pipe.InitBuffer(outputScratchBuf_, 2 * ONCE_ONE_SIZE8 * sizeof(float));
}

template <typename T>
__aicore__ inline void ApplyCamePart1Post<T>::StoreFloatAligned(GlobalTensor<float>& output, int64_t index, float value,
                                                                LocalTensor<float> scratch)
{
    (void)scratch;
    // Use atomic updates because adjacent batch groups may share a GM block.
    AtomicAdd(output.GetPhyAddr(static_cast<uint64_t>(index)), value);
    event_t eventIdSToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
    SetFlag<HardEvent::S_MTE3>(eventIdSToMte3);
    WaitFlag<HardEvent::S_MTE3>(eventIdSToMte3);
}

template <typename T>
__aicore__ inline void ApplyCamePart1Post<T>::ProcessBatch(int64_t batchIdx)
{
    // 由本组的首核调用: 归约本组 workspace 分区, 写出该 batch 的三个输出.
    // 产者/归约阶段的同步由内核入口统一负责, 此处不再做全局屏障.
    batchIdx_ = batchIdx;
    ReduceWorkspaceR();
    uint64_t core_loop = (totalColumnTileCount_ + ONCE_HANDLE_NUM512 - 1) / ONCE_HANDLE_NUM512;
    uint64_t pre_core_m = (totalColumnTileCount_ + core_loop - 1) / core_loop * ONCE_HANDLE_NUM64;
    uint64_t last_core_m = (totalColumnTileCount_ - pre_core_m * (core_loop - 1) / ONCE_HANDLE_NUM64) *
                           ONCE_HANDLE_NUM64;
    uint64_t gmOffsets = 0;
    uint64_t base_m = 0;

    for (int64_t core_loop_idx = 0; core_loop_idx < core_loop - 1; core_loop_idx++) {
        gmOffsets = core_loop_idx * pre_core_m;
        Pre_Core_Compute(gmOffsets, pre_core_m, pre_core_m);
    }
    gmOffsets = (core_loop - 1) * pre_core_m;
    base_m = M - pre_core_m * (core_loop - 1);
    Pre_Core_Compute(gmOffsets, last_core_m, base_m);
    ReduceWorkspaceRC();
}

template <typename T>
__aicore__ inline void ApplyCamePart1Post<T>::ReduceWorkspaceRPartial(LocalTensor<float> inputLocal, int64_t base,
                                                                      int64_t partialRows)
{
    constexpr int64_t rowWidth = 64;
    constexpr int64_t maxRowsPerCopy = 256;
    int64_t activeRows = partialRows;
    while (activeRows > 1) {
        const int64_t rightRows = (activeRows + 1) / 2;
        const int64_t pairRows = activeRows - rightRows;
        for (int64_t offset = 0; offset < pairRows; offset += maxRowsPerCopy) {
            const int64_t copyRows = (pairRows - offset < maxRowsPerCopy) ? pairRows - offset : maxRowsPerCopy;
            DataCopy(inputLocal, workspaceSumGradR_[base + offset * rowWidth], copyRows * rowWidth);
            DataCopy(inputLocal[copyRows * rowWidth], workspaceSumGradR_[base + (rightRows + offset) * rowWidth],
                     copyRows * rowWidth);
            event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
            WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
            Add(inputLocal, inputLocal[copyRows * rowWidth], inputLocal, copyRows * rowWidth);
            event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
            SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
            WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
            DataCopy(workspaceSumGradR_[base + offset * rowWidth], inputLocal, copyRows * rowWidth);
            event_t eventIdMte3ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
            SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
            WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
        }
        if ((activeRows & 1) != 0) {
            const int64_t oddIndex = rightRows - 1;
            DataCopy(inputLocal, workspaceSumGradR_[base + oddIndex * rowWidth], rowWidth);
            event_t eventIdMte2ToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3));
            SetFlag<HardEvent::MTE2_MTE3>(eventIdMte2ToMte3);
            WaitFlag<HardEvent::MTE2_MTE3>(eventIdMte2ToMte3);
            DataCopy(workspaceSumGradR_[base + pairRows * rowWidth], inputLocal, rowWidth);
            event_t eventIdMte3ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
            SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
            WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
        }
        activeRows = rightRows;
    }
}

template <typename T>
__aicore__ inline void ApplyCamePart1Post<T>::ReduceWorkspaceR()
{
    LocalTensor<float> inputLocal = inputBuf_.Get<float>(ONCE_HANDLE_NUM512 * ONCE_HANDLE_NUM64);
    constexpr int64_t rowWidth = 64;
    const int64_t partialRows = mCoreNum_;

    for (int64_t nCoreIdx = 0; nCoreIdx < nCoreNum_; ++nCoreIdx) {
        const int64_t loopCount = (nCoreIdx == nCoreNum_ - 1) ? nLoopTailCore_ : nLoopNormCore_;
        const int64_t rowsInCore = (nCoreIdx == nCoreNum_ - 1) ? nTailCoreNum_ : nNormalCoreNum_;
        for (int64_t nLoopIdx = 0; nLoopIdx < loopCount; ++nLoopIdx) {
            const int64_t base = (nCoreIdx * nLoopNormCore_ + nLoopIdx) * partialRows * rowWidth;
            ReduceWorkspaceRPartial(inputLocal, base, partialRows);
            const int64_t rowOffset = batchIdx_ * N + nCoreIdx * nNormalCoreNum_ + nLoopIdx * rowWidth;
            const int64_t rowCount = (rowsInCore - nLoopIdx * rowWidth < rowWidth) ? rowsInCore - nLoopIdx * rowWidth :
                                                                                     rowWidth;
            DataCopy(inputLocal, workspaceSumGradR_[base], rowWidth);
            // The partial is produced in GM by the previous kernel phase.
            // Fence the GM-to-UB transfer before scalar extraction, matching
            // Part3's explicit MTE2-to-scalar synchronization.
            event_t eventIdMte2ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
            SetFlag<HardEvent::MTE2_S>(eventIdMte2ToS);
            WaitFlag<HardEvent::MTE2_S>(eventIdMte2ToS);
            for (int64_t i = 0; i < rowCount; ++i) {
                StoreFloatAligned(gmSumGradR_, rowOffset + i, inputLocal.GetValue(i), outputScratchBuf_.Get<float>());
            }
            event_t eventIdSToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE2));
            SetFlag<HardEvent::S_MTE2>(eventIdSToMte2);
            WaitFlag<HardEvent::S_MTE2>(eventIdSToMte2);
        }
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
    event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    Add(inputLocal, inputLocal, lowLocal, kScalarSlotSize);
    event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIdVToS);
    WaitFlag<HardEvent::V_S>(eventIdVToS);
    StoreFloatAligned(gmSumGradRC_, batchIdx_, inputLocal.GetValue(0), outputScratchBuf_.Get<float>());
    event_t eventIdSToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE2));
    SetFlag<HardEvent::S_MTE2>(eventIdSToMte2);
    WaitFlag<HardEvent::S_MTE2>(eventIdSToMte2);
}

template <typename T>
__aicore__ inline void ApplyCamePart1Post<T>::ReduceWorkspaceRC()
{
    LocalTensor<float> inputLocal = inputBuf_.Get<float>(ONCE_HANDLE_NUM512 * ONCE_HANDLE_NUM64);
    constexpr uint64_t kScalarSlotSize = 8;
    constexpr uint64_t kMaxPairCount = 512;
    constexpr uint64_t kMaxPairElements = kMaxPairCount * kScalarSlotSize;
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
__aicore__ inline void CopyOutApplyCamePart1C(GlobalTensor<float>& output, LocalTensor<float> input, uint64_t offset,
                                              uint64_t count, LocalTensor<float> scratch, bool accumulate)
{
    (void)scratch;
    (void)accumulate;
    event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIdVToS);
    WaitFlag<HardEvent::V_S>(eventIdVToS);
    for (uint64_t i = 0; i < count; ++i) {
        AtomicAdd(output.GetPhyAddr(offset + i), input.GetValue(i));
    }
    event_t eventIdSToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
    SetFlag<HardEvent::S_MTE3>(eventIdSToMte3);
    WaitFlag<HardEvent::S_MTE3>(eventIdSToMte3);
    event_t eventIdSToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE2));
    SetFlag<HardEvent::S_MTE2>(eventIdSToMte2);
    WaitFlag<HardEvent::S_MTE2>(eventIdSToMte2);
}

template <typename T>
__aicore__ inline void ApplyCamePart1Post<T>::CopyWorkspaceCChunk(LocalTensor<float> inputLocal, uint64_t rowBase,
                                                                  uint64_t rowCount, uint64_t gmOffsets, uint64_t calM)
{
    const uint64_t workspaceRowStride = totalColumnTileCount_ * ONCE_HANDLE_NUM64;
    for (uint64_t row = 0; row < rowCount; ++row) {
        DataCopy(inputLocal[row * calM], workspaceSumGradC_[(rowBase + row) * workspaceRowStride + gmOffsets], calM);
    }
}

template <typename T>
__aicore__ inline void ApplyCamePart1Post<T>::Pre_Core_Compute(uint64_t gmOffsets, uint64_t cal_m, uint64_t base_m)
{
    uint64_t total_n = (nCoreNum_ - 1) * nLoopNormCore_ + nLoopTailCore_;
    uint64_t pre_loop_n = 1;
    while (pre_loop_n < total_n) {
        pre_loop_n = pre_loop_n << 1;
        if (pre_loop_n * cal_m > ONCE_HANDLE_NUM512 * ONCE_HANDLE_NUM64) {
            pre_loop_n = pre_loop_n >> 1;
            break;
        }
        if (pre_loop_n >= total_n) {
            break;
        }
    }
    uint64_t loop_time = (total_n + pre_loop_n - 1) / pre_loop_n;
    uint64_t last_loop_n = total_n - (loop_time - 1) * pre_loop_n;
    LocalTensor<float> inputLocal = inputBuf_.Get<float>(ONCE_HANDLE_NUM512 * ONCE_HANDLE_NUM64);
    event_t eventIdSToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE2));
    SetFlag<HardEvent::S_MTE2>(eventIdSToMte2);
    WaitFlag<HardEvent::S_MTE2>(eventIdSToMte2);

    for (int64_t i = 0; i < loop_time - 1; i++) {
        CopyWorkspaceCChunk(inputLocal, i * pre_loop_n, pre_loop_n, gmOffsets, cal_m);
        event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        ReduceAdd(inputLocal, pre_loop_n, cal_m);
        const uint64_t outputCount = (M - gmOffsets < base_m) ? M - gmOffsets : base_m;
        CopyOutApplyCamePart1C<T>(gmSumGradC_, inputLocal, batchIdx_ * M + gmOffsets, outputCount,
                                  outputScratchBuf_.Get<float>(), i != 0);
    }

    constexpr float scalarValue = 0;
    Duplicate(inputLocal, scalarValue, pre_loop_n * cal_m);
    event_t eventIdVToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
    SetFlag<HardEvent::V_MTE2>(eventIdVToMte2);
    WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2);
    CopyWorkspaceCChunk(inputLocal, (loop_time - 1) * pre_loop_n, last_loop_n, gmOffsets, cal_m);
    event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    ReduceAdd(inputLocal, pre_loop_n, cal_m);
    const uint64_t outputCount = (M - gmOffsets < base_m) ? M - gmOffsets : base_m;
    CopyOutApplyCamePart1C<T>(gmSumGradC_, inputLocal, batchIdx_ * M + gmOffsets, outputCount,
                              outputScratchBuf_.Get<float>(), loop_time != 1);
}

template <typename T>
__aicore__ inline void ApplyCamePart1Post<T>::ReduceAdd(LocalTensor<float> accuUb, int64_t n, int64_t m)
{
    BinaryReduceRowsApplyCamePart1(accuUb, n, m);
}

} // namespace ApplyCamePart1
#endif // APPLY_CAME_PART1_POST

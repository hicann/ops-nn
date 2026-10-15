/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OPS_NN_APPLY_CAME_PART1_COMMON_H
#define OPS_NN_APPLY_CAME_PART1_COMMON_H

#include "kernel_operator.h"

namespace ApplyCamePart1 {
using namespace AscendC;

constexpr float kApplyCamePart1MaxFinite = 3.402823466e+38F;

__aicore__ inline bool IsFiniteApplyCamePart1(float value)
{
    return value >= -kApplyCamePart1MaxFinite && value <= kApplyCamePart1MaxFinite;
}

// Error-free transforms contain expressions such as Inf - Inf. Keep the
// compensation only while the leading sum is finite; this preserves the
// ordinary IEEE propagation of Inf and NaN without changing finite inputs.
__aicore__ inline void CompensatedAddApplyCamePart1(float leftHigh, float leftLow, float rightHigh, float rightLow,
                                                    float& resultHigh, float& resultLow)
{
    const float sum = leftHigh + rightHigh;
    if (!IsFiniteApplyCamePart1(sum)) {
        resultHigh = sum;
        resultLow = 0.0f;
        return;
    }
    const float split = sum - leftHigh;
    const float roundError = (leftHigh - (sum - split)) + (rightHigh - split);
    resultHigh = sum;
    resultLow = leftLow + rightLow + roundError;
}

__aicore__ inline void NormalizeCompensationApplyCamePart1(LocalTensor<float> high, LocalTensor<float> low,
                                                           LocalTensor<float> scratch, LocalTensor<float> maskStorage,
                                                           uint64_t count)
{
    LocalTensor<uint8_t> finiteMask = maskStorage.ReinterpretCast<uint8_t>();
    Abs(scratch, high, count);
    PipeBarrier<PIPE_V>();
    Compares(finiteMask, scratch, kApplyCamePart1MaxFinite, CMPMODE::LE, count);
    PipeBarrier<PIPE_V>();
    Duplicate(scratch, static_cast<float>(0), count);
    PipeBarrier<PIPE_V>();
    Select(low, finiteMask, low, scratch, SELMODE::VSEL_TENSOR_TENSOR_MODE, count);
    PipeBarrier<PIPE_V>();
}

__aicore__ inline bool HasApplyCamePart1Tail(int64_t value, int64_t tileSize) { return value % tileSize != 0; }

__aicore__ inline int64_t CeilDivApplyCamePart1(int64_t value, int64_t factor)
{
    return factor <= 0 ? value : value / factor + (value % factor == 0 ? 0 : 1);
}

// Apply eps to the valid columns in each row.
__aicore__ inline void AddEpsApplyCamePart1(LocalTensor<float> values, float eps, int64_t tileColumns,
                                            int64_t curRepeatTimes)
{
    constexpr int64_t tileSize = 64;
    if (tileColumns >= tileSize) {
        Adds(values, values, eps, curRepeatTimes * tileSize);
        return;
    }
    if (tileColumns < 8) {
        event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventIdVToS);
        WaitFlag<HardEvent::V_S>(eventIdVToS);
        for (int64_t i = 0; i < curRepeatTimes; ++i) {
            const int64_t rowOffset = i * tileSize;
            for (int64_t j = 0; j < tileColumns; ++j) {
                values.SetValue(rowOffset + j, values.GetValue(rowOffset + j) + eps);
            }
        }
        event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(eventIdSToV);
        WaitFlag<HardEvent::S_V>(eventIdSToV);
    } else {
        for (int64_t i = 0; i < curRepeatTimes; ++i) {
            Adds(values[i * tileSize], values[i * tileSize], eps, static_cast<uint16_t>(tileColumns));
        }
    }
}

template <typename Op>
__aicore__ inline void ProcessApplyCamePart1(Op& op, int64_t localBlockIdx, int64_t sliceCoreNum, int64_t nCoreNum,
                                             int64_t mCoreNum, int64_t nLoopNormCore, int64_t nLoopTailCore,
                                             int64_t nTailCoreNum, int64_t mLoopNumCore, int64_t onceHandleNum)
{
    if (localBlockIdx >= sliceCoreNum) {
        return;
    }

    if (((localBlockIdx / mCoreNum) + 1) != nCoreNum) {
        for (int64_t n = 0; n < nLoopNormCore; n++) {
            for (int64_t m = 0; m < mLoopNumCore; m++) {
                op.ProcessTile(n, m, onceHandleNum);
            }
        }
    } else {
        for (int64_t n = 0; n < nLoopTailCore - 1; n++) {
            for (int64_t m = 0; m < mLoopNumCore; m++) {
                op.ProcessTile(n, m, onceHandleNum);
            }
        }

        int64_t nTailCoreLastLoop = nTailCoreNum - (nLoopTailCore - 1) * onceHandleNum;
        for (int64_t m = 0; m < mLoopNumCore; m++) {
            op.ProcessTile(nLoopTailCore - 1, m, nTailCoreLastLoop);
        }
    }
}

template <typename T>
__aicore__ inline void CopyInApplyCamePart1Last(GlobalTensor<T>& gmGrad, LocalTensor<T> gradLocal, int64_t nLoopIdx,
                                                int64_t mLoopIdx, int64_t curRepeatTimes, int64_t tileColumns,
                                                int64_t columnCount, int64_t onceHandleNum, int64_t inputBase)
{
    const int64_t baseOffset = inputBase + nLoopIdx * onceHandleNum * columnCount + mLoopIdx * onceHandleNum;
    Duplicate(gradLocal, static_cast<T>(0), curRepeatTimes * onceHandleNum);
    event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIdVToS);
    WaitFlag<HardEvent::V_S>(eventIdVToS);
    // Use scalar loads for the unaligned tail.
    for (int64_t i = 0; i < curRepeatTimes; i++) {
        const int64_t rowOffset = i * onceHandleNum;
        for (int64_t j = 0; j < tileColumns; ++j) {
            gradLocal.SetValue(rowOffset + j, gmGrad.GetValue(baseOffset + i * columnCount + j));
        }
    }
    event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventIdSToV);
    WaitFlag<HardEvent::S_V>(eventIdSToV);
}

template <typename T>
__aicore__ inline void CopyInApplyCamePart1Normal(GlobalTensor<T>& gmGrad, LocalTensor<T> gradLocal, int64_t nLoopIdx,
                                                  int64_t mLoopIdx, int64_t curRepeatTimes, int64_t columnCount,
                                                  int64_t onceHandleNum, int64_t inputBase)
{
    const int64_t baseOffset = inputBase + nLoopIdx * onceHandleNum * columnCount + mLoopIdx * onceHandleNum;
    // Match Part3's per-row DataCopyPad semantics. A 2D transfer with a
    // synthetic stride is avoided because the row start may be unaligned even
    // when the batch base is aligned.
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(onceHandleNum * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};
    for (int64_t i = 0; i < curRepeatTimes; ++i) {
        DataCopyPad(gradLocal[i * onceHandleNum], gmGrad[baseOffset + i * columnCount], copyParams, padParams);
    }
}

__aicore__ inline void ZeroFillTailApplyCamePart1(LocalTensor<float>& tensor, int64_t firstRow, int64_t slotWidth)
{
    const int64_t alignedStart = (firstRow + 7) / 8 * 8;
    for (int64_t i = firstRow; i < alignedStart && i < slotWidth; ++i) {
        tensor.SetValue(i, static_cast<float>(0));
    }
    if (alignedStart < slotWidth) {
        Duplicate(tensor[alignedStart], static_cast<float>(0), slotWidth - alignedStart);
    }
}

// Clear disjoint aligned output ranges before concurrent atomic updates.
__aicore__ inline void ZeroOutputRegionApplyCamePart1(GlobalTensor<float>& output, int64_t totalElements,
                                                      int64_t blockIdx, int64_t blockNum, LocalTensor<float> zeroUb,
                                                      int64_t zeroCount)
{
    if (totalElements <= 0 || blockNum <= 0) {
        return;
    }
    const int64_t alignedTotal = CeilDivApplyCamePart1(totalElements, 8) * 8;
    const int64_t chunk = CeilDivApplyCamePart1(alignedTotal / 8, blockNum) * 8;
    const int64_t begin = blockIdx * chunk;
    const int64_t end = (begin + chunk) < alignedTotal ? (begin + chunk) : alignedTotal;
    if (begin >= end) {
        return;
    }
    const int64_t validEnd = end < totalElements ? end : totalElements;
    const int64_t alignedValidEnd = begin + (validEnd - begin) / 8 * 8;
    for (int64_t offset = begin; offset < alignedValidEnd; offset += zeroCount) {
        const int64_t remain = alignedValidEnd - offset;
        const int64_t count = remain < zeroCount ? remain : zeroCount;
        DataCopy(output[offset], zeroUb, count);
    }
    if (alignedValidEnd < validEnd) {
        // Use an ordered byte-granularity copy for the unaligned tail.
        DataCopyPad(output[alignedValidEnd], zeroUb,
                    {1, static_cast<uint16_t>((validEnd - alignedValidEnd) * sizeof(float)), 0, 0});
    }
}

__aicore__ inline void ComputeRApplyCamePart1(TQue<QuePosition::VECOUT, 1>& sumGradRQueue, int64_t curRepeatTimes,
                                              int64_t mLoopIdx, int64_t coreLoopCount, LocalTensor<float> gradSqrtTmpUb,
                                              LocalTensor<float> rowAccumulator, LocalTensor<float> workLocal,
                                              LocalTensor<float> reduceDst, int64_t onceHandleNum)
{
    LocalTensor<float> high = rowAccumulator;
    LocalTensor<float> low = rowAccumulator[onceHandleNum];
    for (int64_t row = 0; row < curRepeatTimes; ++row) {
        ReduceSum(reduceDst[row * 8], gradSqrtTmpUb[row * onceHandleNum], workLocal, onceHandleNum);
        PipeBarrier<PIPE_V>();
    }
    event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIdVToS);
    WaitFlag<HardEvent::V_S>(eventIdVToS);
    for (int64_t row = 0; row < curRepeatTimes; ++row) {
        float resultHigh = 0.0f;
        float resultLow = 0.0f;
        CompensatedAddApplyCamePart1(high.GetValue(row), low.GetValue(row), reduceDst.GetValue(row * 8), 0.0f,
                                     resultHigh, resultLow);
        high.SetValue(row, resultHigh);
        low.SetValue(row, resultLow);
    }
    if (mLoopIdx != coreLoopCount - 1) {
        return;
    }

    LocalTensor<float> sumGradRLocal = sumGradRQueue.AllocTensor<float>();
    for (int64_t row = 0; row < curRepeatTimes; ++row) {
        sumGradRLocal.SetValue(row, high.GetValue(row) + low.GetValue(row));
    }
    if (curRepeatTimes < onceHandleNum) {
        ZeroFillTailApplyCamePart1(sumGradRLocal, curRepeatTimes, onceHandleNum);
    }
    event_t eventIdSToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
    SetFlag<HardEvent::S_MTE3>(eventIdSToMte3);
    WaitFlag<HardEvent::S_MTE3>(eventIdSToMte3);
    sumGradRQueue.EnQue<float>(sumGradRLocal);
}

__aicore__ inline void CopyOutSumGradRWorkspaceApplyCamePart1(TQue<QuePosition::VECOUT, 1>& sumGradRQueue,
                                                              GlobalTensor<float>& workspaceSumGradR, int64_t nLoopIdx,
                                                              int64_t curRepeatTimes, int64_t mCoreNum,
                                                              int64_t nLoopNormCore, int64_t onceHandleNum,
                                                              int64_t localBlockIdx)
{
    LocalTensor<float> sumGradRLocal = sumGradRQueue.DeQue<float>();
    (void)curRepeatTimes;
    // Publish scalar row results after the tail clear is complete.
    event_t eventIdSToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
    SetFlag<HardEvent::S_MTE3>(eventIdSToMte3);
    WaitFlag<HardEvent::S_MTE3>(eventIdSToMte3);
    const int64_t nCoreIdx = localBlockIdx / mCoreNum;
    const int64_t mCoreIdx = localBlockIdx % mCoreNum;
    const int64_t partialIdx = (nCoreIdx * nLoopNormCore + nLoopIdx) * mCoreNum + mCoreIdx;
    DataCopy(workspaceSumGradR[partialIdx * onceHandleNum], sumGradRLocal, onceHandleNum);
    sumGradRQueue.FreeTensor(sumGradRLocal);
}

__aicore__ inline void CopyOutReductionWorkspaceApplyCamePart1(TQue<QuePosition::VECOUT, 1>& sumGradRCQueue,
                                                               TQue<QuePosition::VECOUT, 1>& sumGradCQueue,
                                                               GlobalTensor<float>& workspaceSumGradRC,
                                                               GlobalTensor<float>& workspaceSumGradRCLow,
                                                               GlobalTensor<float>& workspaceSumGradC, int64_t offset,
                                                               int64_t scalarSlotSize, int64_t columnTileSize)
{
    LocalTensor<float> sumGradRCLocal = sumGradRCQueue.DeQue<float>();
    event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
    WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
    const int64_t rcOffset = offset * scalarSlotSize;
    DataCopy(workspaceSumGradRC[rcOffset], sumGradRCLocal, scalarSlotSize);
    DataCopy(workspaceSumGradRCLow[rcOffset], sumGradRCLocal[scalarSlotSize], scalarSlotSize);
    sumGradRCQueue.FreeTensor(sumGradRCLocal);

    LocalTensor<float> sumGradCLocal = sumGradCQueue.DeQue<float>();
    event_t eventIdSToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
    SetFlag<HardEvent::S_MTE3>(eventIdSToMte3);
    WaitFlag<HardEvent::S_MTE3>(eventIdSToMte3);
    DataCopy(workspaceSumGradC[offset * columnTileSize], sumGradCLocal, columnTileSize);
    sumGradCQueue.FreeTensor(sumGradCLocal);
    event_t eventIdMte3ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
    WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
}

__aicore__ inline void ComputeCApplyCamePart1(TQue<QuePosition::VECOUT, 1>& sumGradCQueue, int64_t curRepeatTimes,
                                              LocalTensor<float> gradSqrtTmpUb, int64_t mCoreNum, int64_t nCoreNum,
                                              int64_t onceHandleNum, int64_t validColumns, int64_t localBlockIdx)
{
    LocalTensor<float> sumGradCLocal = sumGradCQueue.AllocTensor<float>();
    if (((localBlockIdx / mCoreNum + 1) == nCoreNum) && (curRepeatTimes < onceHandleNum)) {
        PipeBarrier<PIPE_V>();
        Duplicate(gradSqrtTmpUb[onceHandleNum * curRepeatTimes], static_cast<float>(0),
                  (onceHandleNum - curRepeatTimes) * onceHandleNum);
    }
    event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIdVToS);
    WaitFlag<HardEvent::V_S>(eventIdVToS);
    // Process only valid columns in the current tile.
    if (validColumns < onceHandleNum) {
        ZeroFillTailApplyCamePart1(sumGradCLocal, validColumns, onceHandleNum);
        event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventIdVToS);
        WaitFlag<HardEvent::V_S>(eventIdVToS);
    }
    for (int64_t col = 0; col < validColumns; ++col) {
        float sum = 0.0f;
        float correction = 0.0f;
        for (int64_t row = 0; row < curRepeatTimes; ++row) {
            const float value = gradSqrtTmpUb.GetValue(row * onceHandleNum + col);
            CompensatedAddApplyCamePart1(sum, correction, value, 0.0f, sum, correction);
        }
        sumGradCLocal.SetValue(col, sum + correction);
    }
    // Publish scalar stores before the vector reduction consumes them.
    event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventIdSToV);
    WaitFlag<HardEvent::S_V>(eventIdSToV);
    sumGradCQueue.EnQue<float>(sumGradCLocal);
}

__aicore__ inline void BinaryReduceRowsApplyCamePart1(LocalTensor<float> values, int64_t rowCount, int64_t rowWidth)
{
    int64_t activeRows = rowCount;
    while (activeRows > 1) {
        const int64_t rightRows = (activeRows + 1) / 2;
        const int64_t pairRows = activeRows - rightRows;
        if (pairRows > 0) {
            PipeBarrier<PIPE_V>();
            Add(values, values[rightRows * rowWidth], values, pairRows * rowWidth);
        }
        activeRows = rightRows;
    }
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void CompensatedReduceApplyCamePart1(LocalTensor<float> output, LocalTensor<float> high,
                                                       LocalTensor<float> low, LocalTensor<float> sumTmp,
                                                       LocalTensor<float> recoveredTmp, uint64_t count)
{
    constexpr uint64_t kScalarTail = 8;
    const uint64_t alignedCount = (count + kScalarTail - 1) / kScalarTail * kScalarTail;
    uint64_t currentCount = alignedCount;
    while (currentCount > kScalarTail) {
        const uint64_t leftCount = ((currentCount + 2 * kScalarTail - 1) / (2 * kScalarTail)) * kScalarTail;
        const uint64_t pairCount = currentCount - leftCount;

        Add(sumTmp, high, high[leftCount], pairCount);
        PipeBarrier<PIPE_V>();
        Sub(recoveredTmp, sumTmp, high, pairCount);
        PipeBarrier<PIPE_V>();
        Sub(high[leftCount], high[leftCount], recoveredTmp, pairCount);
        Sub(recoveredTmp, sumTmp, recoveredTmp, pairCount);
        PipeBarrier<PIPE_V>();
        Sub(high, high, recoveredTmp, pairCount);
        PipeBarrier<PIPE_V>();
        Add(high, high, high[leftCount], pairCount);
        Add(low, low, low[leftCount], pairCount);
        PipeBarrier<PIPE_V>();
        Add(low, low, high, pairCount);
        Adds(high, sumTmp, static_cast<float>(0), pairCount);
        PipeBarrier<PIPE_V>();
        NormalizeCompensationApplyCamePart1(high, low, high[leftCount], recoveredTmp, pairCount);
        currentCount = leftCount;
    }

    event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIdVToS);
    WaitFlag<HardEvent::V_S>(eventIdVToS);
    while (currentCount > 1) {
        const uint64_t rightCount = (currentCount + 1) / 2;
        const uint64_t pairCount = currentCount - rightCount;
        for (uint64_t i = 0; i < pairCount; ++i) {
            const float left = high.GetValue(i);
            const float right = high.GetValue(rightCount + i);
            float sum = 0.0f;
            float lowSum = 0.0f;
            CompensatedAddApplyCamePart1(left, low.GetValue(i), right, low.GetValue(rightCount + i), sum, lowSum);
            high.SetValue(i, sum);
            low.SetValue(i, lowSum);
        }
        currentCount = rightCount;
    }
    const float highValue = high.GetValue(0);
    const float lowValue = low.GetValue(0);
    event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventIdSToV);
    WaitFlag<HardEvent::S_V>(eventIdSToV);
    Duplicate(output, highValue, 1);
    Duplicate(output[kScalarTail], lowValue, 1);
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void FinishApplyCamePart1Reduction(LocalTensor<float> sumGradRCLocal,
                                                     LocalTensor<float> sumGradCLocal, LocalTensor<float> mComTmpUb,
                                                     int64_t onceHandleNum)
{
    constexpr int64_t kScalarSlotSize = 8;
    LocalTensor<float> high = mComTmpUb;
    LocalTensor<float> low = mComTmpUb[onceHandleNum];
    LocalTensor<float> sumTmp = mComTmpUb[2 * onceHandleNum];
    LocalTensor<float> recoveredTmp = mComTmpUb[3 * onceHandleNum];
    Duplicate(sumGradRCLocal, static_cast<float>(0), 2 * kScalarSlotSize);
    Adds(high, sumGradCLocal, static_cast<float>(0), onceHandleNum);
    Duplicate(low, static_cast<float>(0), onceHandleNum);
    PipeBarrier<PIPE_V>();
    CompensatedReduceApplyCamePart1(sumGradRCLocal, high, low, sumTmp, recoveredTmp, onceHandleNum);
}

__aicore__ inline int64_t InitApplyCamePart1OutputBuffers(
    GlobalTensor<float>& sumGradR, GlobalTensor<float>& sumGradC, GlobalTensor<float>& sumGradRC,
    GlobalTensor<float>& workspaceSumGradR, GlobalTensor<float>& workspaceSumGradRC,
    GlobalTensor<float>& workspaceSumGradRCLow, GlobalTensor<float>& workspaceSumGradC, GM_ADDR sumGradRAddr,
    GM_ADDR sumGradCAddr, GM_ADDR sumGradRCAddr, GM_ADDR workspaceAddr, int64_t columnCount, int64_t mCoreNum,
    int64_t nCoreNum, int64_t nLoopNormCore, int64_t nLoopTailCore, int64_t rowTileSize, int64_t groupIdx)
{
    sumGradR.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(sumGradRAddr));
    sumGradC.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(sumGradCAddr));
    sumGradRC.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(sumGradRCAddr));
    if (mCoreNum <= 0 || nCoreNum <= 0) {
        return 0;
    }
    // Each batch group owns an independent reduction workspace region.
    const int64_t nLoopCount = (nCoreNum - 1) * nLoopNormCore + nLoopTailCore;
    constexpr int64_t kScalarSlotSize = 8;
    const int64_t totalColumnTileCount = CeilDivApplyCamePart1(columnCount, rowTileSize);
    const int64_t rcPartialCount = nLoopCount * totalColumnTileCount;
    const int64_t rPartialCount = nLoopCount * mCoreNum;
    const int64_t rcAlignedCount = (rcPartialCount * kScalarSlotSize + 128 - 1) / 128 * 128;
    const int64_t rAlignedCount = (rPartialCount * rowTileSize + 128 - 1) / 128 * 128;
    const int64_t cAlignedCount = (rcPartialCount * rowTileSize + 128 - 1) / 128 * 128;
    const int64_t regionFloats = 2 * rcAlignedCount + rAlignedCount + cAlignedCount;
    auto* regionBase = reinterpret_cast<__gm__ float*>(workspaceAddr) + groupIdx * regionFloats;
    workspaceSumGradRC.SetGlobalBuffer(regionBase);
    workspaceSumGradRCLow.SetGlobalBuffer(regionBase + rcAlignedCount);
    workspaceSumGradR.SetGlobalBuffer(regionBase + 2 * rcAlignedCount);
    workspaceSumGradC.SetGlobalBuffer(regionBase + 2 * rcAlignedCount + rAlignedCount);
    return regionFloats;
}

} // namespace ApplyCamePart1
#endif // OPS_NN_APPLY_CAME_PART1_COMMON_H

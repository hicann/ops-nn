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

__aicore__ inline bool HasApplyCamePart1Tail(int64_t value, int64_t tileSize) { return value % tileSize != 0; }

__aicore__ inline void AddEpsApplyCamePart1(LocalTensor<float> values, float eps, int64_t mLoopIdx,
                                            int64_t mLoopNumCore, bool hasColumnTail, int64_t mTailCoreNum,
                                            int64_t mNormalCoreNum, int64_t curRepeatTimes)
{
    constexpr int64_t tileSize = 64;
    if (mLoopIdx == (mLoopNumCore - 1) && hasColumnTail) {
        const int64_t tailCount = (mTailCoreNum > 0) ? mTailCoreNum : mNormalCoreNum - (mLoopNumCore - 1) * tileSize;
        if (tailCount < 8) {
            event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
            SetFlag<HardEvent::V_S>(eventIdVToS);
            WaitFlag<HardEvent::V_S>(eventIdVToS);
            for (int64_t i = 0; i < curRepeatTimes; ++i) {
                const int64_t rowOffset = i * tileSize;
                for (int64_t j = 0; j < tailCount; ++j) {
                    values.SetValue(rowOffset + j, values.GetValue(rowOffset + j) + eps);
                }
            }
        } else {
            for (int64_t i = 0; i < curRepeatTimes; ++i) {
                Adds(values[i * tileSize], values[i * tileSize], eps, static_cast<uint16_t>(tailCount));
            }
        }
    } else {
        Adds(values, values, eps, curRepeatTimes * tileSize);
    }
}

template <typename Op>
__aicore__ inline void ProcessApplyCamePart1(Op& op, int64_t blockIdx, int64_t usedCoreNum, int64_t nCoreNum,
                                             int64_t mCoreNum, int64_t nLoopNormCore, int64_t nLoopTailCore,
                                             int64_t nTailCoreNum, int64_t mLoopNumCore, int64_t onceHandleNum)
{
    if (blockIdx >= usedCoreNum) {
        return;
    }

    if (((blockIdx / mCoreNum) + 1) != nCoreNum) {
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
                                                int64_t mLoopIdx, int64_t curRepeatTimes, int64_t mNormalCoreNum,
                                                int64_t mTailCoreNum, int64_t columnCount, int64_t onceHandleNum,
                                                int64_t inputBase)
{
    const int64_t tailNum = (mTailCoreNum > 0) ? mTailCoreNum : (mNormalCoreNum % onceHandleNum);
    const int64_t baseOffset = inputBase + nLoopIdx * onceHandleNum * columnCount + mLoopIdx * onceHandleNum;
    Duplicate(gradLocal, static_cast<T>(0), curRepeatTimes * onceHandleNum);
    // Match Part3's exact tail contract. Scalar GM loads are used for the
    // non-block-aligned suffix so no DMA transaction can truncate the tail.
    for (int64_t i = 0; i < curRepeatTimes; i++) {
        const int64_t rowOffset = i * onceHandleNum;
        for (int64_t j = 0; j < tailNum; ++j) {
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

__aicore__ inline void ComputeRApplyCamePart1(TQue<QuePosition::VECOUT, 1>& sumGradRQueue, int64_t curRepeatTimes,
                                              int64_t mLoopIdx, int64_t mLoopNumCore, LocalTensor<float> gradSqrtTmpUb,
                                              LocalTensor<float> rowTree, LocalTensor<float> workLocal,
                                              int64_t onceHandleNum)
{
    int64_t treeLevelCount = 1;
    uint64_t levelProbe = static_cast<uint64_t>(mLoopNumCore);
    while ((levelProbe >>= 1U) != 0U) {
        ++treeLevelCount;
    }
    const int64_t lowBase = treeLevelCount * onceHandleNum;
    LocalTensor<float> sumGradRLocal = sumGradRQueue.AllocTensor<float>();
    for (int64_t row = 0; row < curRepeatTimes; ++row) {
        ReduceSum(sumGradRLocal[row], gradSqrtTmpUb[row * onceHandleNum], workLocal, onceHandleNum);
        PipeBarrier<PIPE_V>();
    }
    event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIdVToS);
    WaitFlag<HardEvent::V_S>(eventIdVToS);
    for (int64_t row = 0; row < curRepeatTimes; ++row) {
        float valueHigh = sumGradRLocal.GetValue(row);
        float valueLow = 0.0f;
        uint64_t carry = static_cast<uint64_t>(mLoopIdx);
        int64_t level = 0;
        while ((carry & 1U) != 0U) {
            const float leftHigh = rowTree.GetValue(level * onceHandleNum + row);
            const float leftLow = rowTree.GetValue(lowBase + level * onceHandleNum + row);
            const float sum = leftHigh + valueHigh;
            const float split = sum - leftHigh;
            const float roundError = (leftHigh - (sum - split)) + (valueHigh - split);
            valueHigh = sum;
            valueLow += leftLow + roundError;
            carry >>= 1U;
            ++level;
        }
        rowTree.SetValue(level * onceHandleNum + row, valueHigh);
        rowTree.SetValue(lowBase + level * onceHandleNum + row, valueLow);
    }
    PipeBarrier<PIPE_ALL>();
    if (mLoopIdx == mLoopNumCore - 1) {
        int64_t highestLevel = 0;
        uint64_t probe = static_cast<uint64_t>(mLoopNumCore);
        while ((probe >>= 1U) != 0U) {
            ++highestLevel;
        }
        for (int64_t row = 0; row < curRepeatTimes; ++row) {
            float totalHigh = rowTree.GetValue(highestLevel * onceHandleNum + row);
            float totalLow = rowTree.GetValue(lowBase + highestLevel * onceHandleNum + row);
            for (int64_t level = highestLevel - 1; level >= 0; --level) {
                if ((static_cast<uint64_t>(mLoopNumCore) & (1ULL << level)) != 0U) {
                    const float rightHigh = rowTree.GetValue(level * onceHandleNum + row);
                    const float rightLow = rowTree.GetValue(lowBase + level * onceHandleNum + row);
                    const float sum = totalHigh + rightHigh;
                    const float split = sum - totalHigh;
                    const float roundError = (totalHigh - (sum - split)) + (rightHigh - split);
                    totalHigh = sum;
                    totalLow += rightLow + roundError;
                }
            }
            sumGradRLocal.SetValue(row, totalHigh + totalLow);
        }
        for (int64_t row = curRepeatTimes; row < onceHandleNum; ++row) {
            sumGradRLocal.SetValue(row, static_cast<float>(0));
        }
        PipeBarrier<PIPE_ALL>();
        sumGradRQueue.EnQue<float>(sumGradRLocal);
    } else {
        sumGradRQueue.FreeTensor(sumGradRLocal);
    }
}

__aicore__ inline void CopyOutSumGradRWorkspaceApplyCamePart1(TQue<QuePosition::VECOUT, 1>& sumGradRQueue,
                                                              GlobalTensor<float>& workspaceSumGradR, int64_t nLoopIdx,
                                                              int64_t curRepeatTimes, int64_t mCoreNum,
                                                              int64_t nLoopNormCore, int64_t onceHandleNum)
{
    LocalTensor<float> sumGradRLocal = sumGradRQueue.DeQue<float>();
    if (curRepeatTimes < onceHandleNum) {
        event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventIdVToS);
        WaitFlag<HardEvent::V_S>(eventIdVToS);
        for (int64_t i = curRepeatTimes; i < onceHandleNum; ++i) {
            sumGradRLocal.SetValue(i, static_cast<float>(0));
        }
    }
    // ComputeR finishes the row partial with scalar stores.  Fence the
    // scalar-to-MTE3 dependency before publishing it to GM, as in Part3.
    event_t eventIdSToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
    SetFlag<HardEvent::S_MTE3>(eventIdSToMte3);
    WaitFlag<HardEvent::S_MTE3>(eventIdSToMte3);
    const int64_t nCoreIdx = GetBlockIdx() / mCoreNum;
    const int64_t mCoreIdx = GetBlockIdx() % mCoreNum;
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
    event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIdVToS);
    WaitFlag<HardEvent::V_S>(eventIdVToS);
    event_t eventIdSToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
    SetFlag<HardEvent::S_MTE3>(eventIdSToMte3);
    WaitFlag<HardEvent::S_MTE3>(eventIdSToMte3);
    const int64_t rcOffset = offset * scalarSlotSize;
    DataCopy(workspaceSumGradRC[rcOffset], sumGradRCLocal, scalarSlotSize);
    DataCopy(workspaceSumGradRCLow[rcOffset], sumGradRCLocal[scalarSlotSize], scalarSlotSize);
    sumGradRCQueue.FreeTensor(sumGradRCLocal);

    LocalTensor<float> sumGradCLocal = sumGradCQueue.DeQue<float>();
    event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
    WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
    DataCopy(workspaceSumGradC[offset * columnTileSize], sumGradCLocal, columnTileSize);
    sumGradCQueue.FreeTensor(sumGradCLocal);
    event_t eventIdMte3ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
    WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
}

__aicore__ inline void ComputeCApplyCamePart1(TQue<QuePosition::VECOUT, 1>& sumGradCQueue, int64_t curRepeatTimes,
                                              LocalTensor<float> gradSqrtTmpUb, int64_t mCoreNum, int64_t nCoreNum,
                                              int64_t onceHandleNum)
{
    LocalTensor<float> sumGradCLocal = sumGradCQueue.AllocTensor<float>();
    if (((GetBlockIdx() / mCoreNum + 1) == nCoreNum) && (curRepeatTimes < onceHandleNum)) {
        PipeBarrier<PIPE_V>();
        Duplicate(gradSqrtTmpUb[onceHandleNum * curRepeatTimes], static_cast<float>(0),
                  (onceHandleNum - curRepeatTimes) * onceHandleNum);
    }
    event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIdVToS);
    WaitFlag<HardEvent::V_S>(eventIdVToS);
    for (int64_t col = 0; col < onceHandleNum; ++col) {
        float sum = 0.0f;
        float correction = 0.0f;
        for (int64_t row = 0; row < onceHandleNum; ++row) {
            const float value = gradSqrtTmpUb.GetValue(row * onceHandleNum + col);
            const float adjusted = value - correction;
            const float next = sum + adjusted;
            correction = (next - sum) - adjusted;
            sum = next;
        }
        sumGradCLocal.SetValue(col, sum);
    }
    PipeBarrier<PIPE_ALL>();
    sumGradCQueue.EnQue<float>(sumGradCLocal);
}

__aicore__ inline void BinaryReduceRowsApplyCamePart1(LocalTensor<float> values, int64_t rowCount, int64_t rowWidth)
{
    int64_t activeRows = rowCount;
    while (activeRows > 1) {
        const int64_t rightRows = (activeRows + 1) / 2;
        const int64_t pairRows = activeRows - rightRows;
        if (pairRows > 0) {
            PipeBarrier<PIPE_ALL>();
            Add(values, values[rightRows * rowWidth], values, pairRows * rowWidth);
        }
        activeRows = rightRows;
    }
    PipeBarrier<PIPE_ALL>();
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
            const float sum = left + right;
            const float split = sum - left;
            const float error = (left - (sum - split)) + (right - split);
            float lowSum = low.GetValue(i) + low.GetValue(rightCount + i);
            lowSum += error;
            high.SetValue(i, sum);
            low.SetValue(i, lowSum);
        }
        currentCount = rightCount;
    }
    Duplicate(output, high.GetValue(0), 1);
    Duplicate(output[kScalarTail], low.GetValue(0), 1);
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

__aicore__ inline void InitApplyCamePart1OutputBuffers(
    GlobalTensor<float>& sumGradR, GlobalTensor<float>& sumGradC, GlobalTensor<float>& sumGradRC,
    GlobalTensor<float>& workspaceSumGradR, GlobalTensor<float>& workspaceSumGradRC,
    GlobalTensor<float>& workspaceSumGradRCLow, GlobalTensor<float>& workspaceSumGradC, GM_ADDR sumGradRAddr,
    GM_ADDR sumGradCAddr, GM_ADDR sumGradRCAddr, GM_ADDR workspaceAddr, int64_t mCoreNum, int64_t nLoopNormCore,
    int64_t usedCoreNum, int64_t nLoopTailCore, int64_t mLoopNumCore, int64_t rowTileSize)
{
    sumGradR.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(sumGradRAddr) +
                             GetBlockIdx() / mCoreNum * rowTileSize * nLoopNormCore);
    sumGradC.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(sumGradCAddr));
    sumGradRC.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(sumGradRCAddr));
    if (mCoreNum <= 0 || usedCoreNum <= 0) {
        return;
    }
    const int64_t nCoreNum = usedCoreNum / mCoreNum;
    const int64_t nLoopCount = (nCoreNum - 1) * nLoopNormCore + nLoopTailCore;
    constexpr int64_t kScalarSlotSize = 8;
    const int64_t rcPartialCount = nLoopCount * mCoreNum * mLoopNumCore;
    const int64_t rPartialCount = nLoopCount * mCoreNum;
    const int64_t rcOffsets = rcPartialCount * kScalarSlotSize + 128 - 1;
    const int64_t rcAlignedCount = rcOffsets / 128 * 128;
    const int64_t rOffsets = rPartialCount * rowTileSize + 128 - 1;
    const int64_t rAlignedCount = rOffsets / 128 * 128;
    workspaceSumGradRC.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspaceAddr));
    workspaceSumGradRCLow.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspaceAddr) + rcAlignedCount);
    workspaceSumGradR.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspaceAddr) + 2 * rcAlignedCount);
    workspaceSumGradC.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspaceAddr) + 2 * rcAlignedCount +
                                      rAlignedCount);
}

} // namespace ApplyCamePart1
#endif // OPS_NN_APPLY_CAME_PART1_COMMON_H

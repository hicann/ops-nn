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
    uint32_t bits = 0;
    __builtin_memcpy(&bits, &value, sizeof(bits));
    return (bits & 0x7F800000U) != 0x7F800000U;
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
    const float residual = leftLow + rightLow + roundError;
    const float normalizedHigh = sum + residual;
    if (!IsFiniteApplyCamePart1(normalizedHigh)) {
        resultHigh = normalizedHigh;
        resultLow = 0.0f;
        return;
    }
    resultHigh = normalizedHigh;
    resultLow = residual - (normalizedHigh - sum);
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
                                            int64_t curRepeatTimes, int64_t tileSize = 64)
{
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
    event_t eventIdVToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
    SetFlag<HardEvent::V_MTE2>(eventIdVToMte2);
    WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2);
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(tileColumns * sizeof(T)), 0, 0, 0};
    constexpr int64_t blockElements = ONE_BLK_SIZE / sizeof(T);
    const uint8_t rightPadding = static_cast<uint8_t>((blockElements - tileColumns % blockElements) % blockElements);
    DataCopyPadExtParams<T> padParams{true, 0, rightPadding, static_cast<T>(0)};
    for (int64_t row = 0; row < curRepeatTimes; ++row) {
        DataCopyPad(gradLocal[row * onceHandleNum], gmGrad[baseOffset + row * columnCount], copyParams, padParams);
    }
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

__aicore__ inline void ZeroFillTailApplyCamePart1(LocalTensor<float> tensor, int64_t firstRow, int64_t slotWidth)
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

// Short rows are sensitive to a one-ULP error from the fixed-width ReduceSum tree.
// Keep the rounding error for each row while accumulating its valid columns.  The
// register lanes process independent rows, so this also avoids scalar UB traffic.
__aicore__ inline void SequentialCompensatedRowsApplyCamePart1(LocalTensor<float> output, LocalTensor<float> values,
                                                               uint32_t rows, uint32_t width, uint32_t columns)
{
    __ubuf__ float* inputAddr = reinterpret_cast<__ubuf__ float*>(values.GetPhyAddr());
    __ubuf__ float* outputAddr = reinterpret_cast<__ubuf__ float*>(output.GetPhyAddr());
    __VEC_SCOPE__
    {
        constexpr uint32_t lanes = 64;
        Reg::RegTensor<float> high, low, value, sum, split, leftError, rightError, error, zero;
        Reg::RegTensor<uint32_t> lane, rowIndex, base, index;
        Reg::MaskReg mask, finite, lowerFinite;
        Reg::MaskReg all = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
        Reg::Arange(reinterpret_cast<Reg::RegTensor<int32_t>&>(lane), static_cast<int32_t>(0));
        for (uint16_t block = 0; block < static_cast<uint16_t>((rows + lanes - 1) / lanes); ++block) {
            const uint32_t blockOffset = static_cast<uint32_t>(block) * lanes;
            uint32_t remaining = rows - blockOffset;
            mask = Reg::UpdateMask<float>(remaining);
            Reg::Adds(rowIndex, lane, blockOffset, all);
            Reg::Muls(base, rowIndex, width, all);
            Reg::Duplicate(high, 0.0f, mask);
            Reg::Duplicate(low, 0.0f, mask);
            Reg::Duplicate(zero, 0.0f, mask);
            for (uint16_t column = 0; column < static_cast<uint16_t>(columns); ++column) {
                Reg::Adds(index, base, static_cast<uint32_t>(column), all);
                Reg::Gather(value, inputAddr, index, mask);
                Reg::Add(sum, high, value, mask);
                Reg::Sub(split, sum, high, mask);
                Reg::Sub(leftError, sum, split, mask);
                Reg::Sub(leftError, high, leftError, mask);
                Reg::Sub(rightError, value, split, mask);
                Reg::Add(error, leftError, rightError, mask);
                Reg::Add(low, low, zero, mask);
                Reg::Add(low, low, error, mask);
                Reg::Compares<float, CMPMODE::LE>(finite, sum, kApplyCamePart1MaxFinite, mask);
                Reg::Compares<float, CMPMODE::GE>(lowerFinite, sum, -kApplyCamePart1MaxFinite, mask);
                Reg::MaskAnd(finite, finite, lowerFinite, mask);
                Reg::Select(low, low, zero, finite);
                // Match the scalar helper's first guard: publish the leading
                // sum and discard its compensation once that sum is non-finite.
                // The high/low pair is normalized when the partials are merged.
                Reg::Add(high, sum, zero, mask);
            }
            Reg::StoreAlign(outputAddr + blockOffset, high, mask);
            Reg::StoreAlign(outputAddr + width + blockOffset, low, mask);
        }
    }
}

// A full 128-column tile previously used ReduceSum and discarded the low part
// of its reduction.  That is visible when a following short tail is merged
// into the row accumulator (for example 128 + 1 columns).  Reduce the two
// 64-lane halves with error-free TwoSum operations and carry the low component
// through the complete register tree.  This keeps the common full-tile path
// vectorized and does not require another UB buffer.
__aicore__ inline void CompensatedFullTileRowsApplyCamePart1(LocalTensor<float> output, LocalTensor<float> values,
                                                             uint32_t rows, uint32_t width)
{
    __ubuf__ float* inputAddr = reinterpret_cast<__ubuf__ float*>(values.GetPhyAddr());
    __ubuf__ float* outputAddr = reinterpret_cast<__ubuf__ float*>(output.GetPhyAddr());
    __VEC_SCOPE__
    {
        constexpr uint32_t lanes = 64;
        Reg::RegTensor<float> high, low, lhsHigh, rhsHigh, lhsLow, rhsLow, split, error, temp, zero;
        Reg::RegTensor<int32_t> evenIndex, oddIndex;
        Reg::MaskReg finite, lowerFinite;
        Reg::MaskReg all = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
        Reg::MaskReg one = Reg::CreateMask<float, Reg::MaskPattern::VL1>();
        Reg::Duplicate(zero, 0.0f, all);
        Reg::Arange(evenIndex, static_cast<int32_t>(0));
        Reg::ShiftLefts(evenIndex, evenIndex, static_cast<int16_t>(1), all);
        Reg::Adds(oddIndex, evenIndex, static_cast<int32_t>(1), all);

        for (uint16_t row = 0; row < static_cast<uint16_t>(rows); ++row) {
            Reg::LoadAlign(lhsHigh, inputAddr + static_cast<uint32_t>(row) * width);
            Reg::LoadAlign(rhsHigh, inputAddr + static_cast<uint32_t>(row) * width + lanes);

            Reg::Add(high, lhsHigh, rhsHigh, all);
            Reg::Sub(split, high, lhsHigh, all);
            Reg::Sub(error, high, split, all);
            Reg::Sub(error, lhsHigh, error, all);
            Reg::Sub(temp, rhsHigh, split, all);
            Reg::Add(low, error, temp, all);
            Reg::Compares<float, CMPMODE::LE>(finite, high, kApplyCamePart1MaxFinite, all);
            Reg::Compares<float, CMPMODE::GE>(lowerFinite, high, -kApplyCamePart1MaxFinite, all);
            Reg::MaskAnd(finite, finite, lowerFinite, all);
            Reg::Select(low, low, zero, finite);

            for (uint16_t activeCount = lanes / 2; activeCount > 0; activeCount /= 2) {
                uint32_t activeWidth = activeCount;
                Reg::MaskReg active = Reg::UpdateMask<float>(activeWidth);
                Reg::Gather<float, uint32_t>(lhsHigh, high, reinterpret_cast<Reg::RegTensor<uint32_t>&>(evenIndex));
                Reg::Gather<float, uint32_t>(rhsHigh, high, reinterpret_cast<Reg::RegTensor<uint32_t>&>(oddIndex));
                Reg::Gather<float, uint32_t>(lhsLow, low, reinterpret_cast<Reg::RegTensor<uint32_t>&>(evenIndex));
                Reg::Gather<float, uint32_t>(rhsLow, low, reinterpret_cast<Reg::RegTensor<uint32_t>&>(oddIndex));

                Reg::Add(high, lhsHigh, rhsHigh, active);
                Reg::Sub(split, high, lhsHigh, active);
                Reg::Sub(error, high, split, active);
                Reg::Sub(error, lhsHigh, error, active);
                Reg::Sub(temp, rhsHigh, split, active);
                Reg::Add(error, error, temp, active);
                Reg::Compares<float, CMPMODE::LE>(finite, high, kApplyCamePart1MaxFinite, active);
                Reg::Compares<float, CMPMODE::GE>(lowerFinite, high, -kApplyCamePart1MaxFinite, active);
                Reg::MaskAnd(finite, finite, lowerFinite, active);
                Reg::Select(error, error, zero, finite);
                Reg::Add(low, lhsLow, rhsLow, active);
                Reg::Add(low, low, error, active);
            }
            Reg::DataCopy<float, Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(outputAddr + row * 8, high, one);
            Reg::DataCopy<float, Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(outputAddr + row * 8 + 1, low, one);
        }
    }
}

__aicore__ inline void ComputeRApplyCamePart1(TQue<QuePosition::VECOUT, 1>& sumGradRQueue, int64_t curRepeatTimes,
                                              int64_t mLoopIdx, int64_t coreLoopCount, LocalTensor<float> gradSqrtTmpUb,
                                              LocalTensor<float> rowAccumulator, LocalTensor<float> workLocal,
                                              LocalTensor<float> reduceDst, int64_t onceHandleNum, int64_t validColumns)
{
    LocalTensor<float> high = rowAccumulator;
    LocalTensor<float> low = rowAccumulator[onceHandleNum];
    constexpr int64_t kAccurateShortRowColumns = 8;
    const bool useAccurateShortRow = validColumns > 0 && validColumns <= kAccurateShortRowColumns;
    const bool useCompensatedFullTile = validColumns == 128 && onceHandleNum == 128;
    if (useAccurateShortRow) {
        SequentialCompensatedRowsApplyCamePart1(reduceDst, gradSqrtTmpUb, static_cast<uint32_t>(curRepeatTimes),
                                                static_cast<uint32_t>(onceHandleNum),
                                                static_cast<uint32_t>(validColumns));
        PipeBarrier<PIPE_V>();
    } else if (useCompensatedFullTile) {
        CompensatedFullTileRowsApplyCamePart1(reduceDst, gradSqrtTmpUb, static_cast<uint32_t>(curRepeatTimes),
                                              static_cast<uint32_t>(onceHandleNum));
        PipeBarrier<PIPE_V>();
    } else {
        for (int64_t row = 0; row < curRepeatTimes; ++row) {
            ReduceSum(reduceDst[row * 8], gradSqrtTmpUb[row * onceHandleNum], workLocal, onceHandleNum);
            PipeBarrier<PIPE_V>();
        }
    }
    event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIdVToS);
    WaitFlag<HardEvent::V_S>(eventIdVToS);
    for (int64_t row = 0; row < curRepeatTimes; ++row) {
        float resultHigh = 0.0f;
        float resultLow = 0.0f;
        const float tileHigh = useAccurateShortRow ? reduceDst.GetValue(row) : reduceDst.GetValue(row * 8);
        const float tileLow = useAccurateShortRow ? reduceDst.GetValue(onceHandleNum + row) :
                                                    (useCompensatedFullTile ? reduceDst.GetValue(row * 8 + 1) : 0.0f);
        CompensatedAddApplyCamePart1(high.GetValue(row), low.GetValue(row), tileHigh, tileLow, resultHigh, resultLow);
        high.SetValue(row, resultHigh);
        low.SetValue(row, resultLow);
    }
    if (mLoopIdx != coreLoopCount - 1) {
        return;
    }

    LocalTensor<float> sumGradRLocal = sumGradRQueue.AllocTensor<float>();
    for (int64_t row = 0; row < curRepeatTimes; ++row) {
        sumGradRLocal.SetValue(row, high.GetValue(row));
        sumGradRLocal.SetValue(onceHandleNum + row, low.GetValue(row));
    }
    if (curRepeatTimes < onceHandleNum) {
        ZeroFillTailApplyCamePart1(sumGradRLocal, curRepeatTimes, onceHandleNum);
        ZeroFillTailApplyCamePart1(sumGradRLocal[onceHandleNum], curRepeatTimes, onceHandleNum);
    }
    event_t eventIdSToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
    SetFlag<HardEvent::S_MTE3>(eventIdSToMte3);
    WaitFlag<HardEvent::S_MTE3>(eventIdSToMte3);
    sumGradRQueue.EnQue<float>(sumGradRLocal);
}

// M == 2 is a common optimizer shape. Keep the two values of every row
// compact in UB and reduce 64 rows in parallel. The six additions with +0
// reproduce the remaining levels of the 128-wide reduction tree; the final
// addition reproduces high + low from the compensated row accumulator.
__aicore__ inline void ComputeRWidth2ApplyCamePart1(TQue<QuePosition::VECOUT, 1>& sumGradRQueue,
                                                    LocalTensor<float> compactValues, uint32_t rows,
                                                    uint32_t rowTileSize)
{
    LocalTensor<float> output = sumGradRQueue.AllocTensor<float>();
    Duplicate(output, 0.0f, 2 * rowTileSize);
    PipeBarrier<PIPE_V>();
    __ubuf__ float* inputAddr = reinterpret_cast<__ubuf__ float*>(compactValues.GetPhyAddr());
    __ubuf__ float* outputAddr = reinterpret_cast<__ubuf__ float*>(output.GetPhyAddr());
    __VEC_SCOPE__
    {
        constexpr uint32_t lanes = 64;
        Reg::RegTensor<float> left, right, sum, split, leftError, rightError, low, zero;
        Reg::RegTensor<uint32_t> evenIndex, oddIndex;
        Reg::MaskReg mask;
        Reg::MaskReg all = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
        Reg::Arange(reinterpret_cast<Reg::RegTensor<int32_t>&>(evenIndex), static_cast<int32_t>(0));
        Reg::Muls(evenIndex, evenIndex, static_cast<uint32_t>(2), all);
        Reg::Adds(oddIndex, evenIndex, static_cast<uint32_t>(1), all);
        Reg::Duplicate(zero, 0.0f, all);
        uint32_t remaining = rows;
        for (uint16_t block = 0; block < static_cast<uint16_t>((rows + lanes - 1) / lanes); ++block) {
            mask = Reg::UpdateMask<float>(remaining);
            __ubuf__ float* blockInput = inputAddr + block * lanes * 2;
            Reg::Gather(left, blockInput, evenIndex, mask);
            Reg::Gather(right, blockInput, oddIndex, mask);
            Reg::Add(sum, left, right, mask);
            Reg::Sub(split, sum, left, mask);
            Reg::Sub(leftError, sum, split, mask);
            Reg::Sub(leftError, left, leftError, mask);
            Reg::Sub(rightError, right, split, mask);
            Reg::Add(low, leftError, rightError, mask);
            for (uint16_t level = 0; level < 7; ++level) {
                Reg::Add(sum, sum, zero, mask);
                Reg::Add(low, low, zero, mask);
            }
            Reg::StoreAlign(outputAddr + block * lanes, sum, mask);
            Reg::StoreAlign(outputAddr + rowTileSize + block * lanes, low, mask);
        }
    }
    PipeBarrier<PIPE_V>();
    sumGradRQueue.EnQue<float>(output);
}

// Produce several consecutive M == 2 row tiles in one vector scope.  The
// arithmetic for every row is identical to ComputeRWidth2ApplyCamePart1; only
// queue allocation and workspace publication are moved outside the tile loop.
__aicore__ inline void ComputeRWidth2BulkApplyCamePart1(LocalTensor<float> output, LocalTensor<float> compactValues,
                                                        uint32_t rows, uint32_t alignedRows)
{
    Duplicate(output, 0.0f, alignedRows);
    PipeBarrier<PIPE_V>();
    __ubuf__ float* inputAddr = reinterpret_cast<__ubuf__ float*>(compactValues.GetPhyAddr());
    __ubuf__ float* outputAddr = reinterpret_cast<__ubuf__ float*>(output.GetPhyAddr());
    __VEC_SCOPE__
    {
        constexpr uint32_t lanes = 64;
        Reg::RegTensor<float> left, right, sum, split, leftError, rightError, low, zero;
        Reg::RegTensor<uint32_t> evenIndex, oddIndex;
        Reg::MaskReg mask;
        Reg::MaskReg all = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
        Reg::Arange(reinterpret_cast<Reg::RegTensor<int32_t>&>(evenIndex), static_cast<int32_t>(0));
        Reg::Muls(evenIndex, evenIndex, static_cast<uint32_t>(2), all);
        Reg::Adds(oddIndex, evenIndex, static_cast<uint32_t>(1), all);
        Reg::Duplicate(zero, 0.0f, all);
        uint32_t remaining = rows;
        for (uint16_t block = 0; block < static_cast<uint16_t>((rows + lanes - 1) / lanes); ++block) {
            mask = Reg::UpdateMask<float>(remaining);
            __ubuf__ float* blockInput = inputAddr + block * lanes * 2;
            Reg::Gather(left, blockInput, evenIndex, mask);
            Reg::Gather(right, blockInput, oddIndex, mask);
            Reg::Add(sum, left, right, mask);
            Reg::Sub(split, sum, left, mask);
            Reg::Sub(leftError, sum, split, mask);
            Reg::Sub(leftError, left, leftError, mask);
            Reg::Sub(rightError, right, split, mask);
            Reg::Add(low, leftError, rightError, mask);
            for (uint16_t level = 0; level < 7; ++level) {
                Reg::Add(sum, sum, zero, mask);
                Reg::Add(low, low, zero, mask);
            }
            Reg::StoreAlign(outputAddr + block * lanes, sum, mask);
            Reg::StoreAlign(outputAddr + alignedRows + block * lanes, low, mask);
        }
    }
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void FinishRWidth2DirectApplyCamePart1(LocalTensor<float> values, uint32_t rows, uint32_t alignedRows,
                                                         float epsHigh, float epsLow)
{
    __ubuf__ float* outputAddr = reinterpret_cast<__ubuf__ float*>(values.GetPhyAddr());
    __VEC_SCOPE__
    {
        constexpr uint32_t lanes = 64;
        Reg::RegTensor<float> high, low, scalarHigh, scalarLow, sum, split, leftError, rightError, error;
        Reg::MaskReg mask;
        uint32_t remaining = rows;
        for (uint16_t block = 0; block < static_cast<uint16_t>((rows + lanes - 1) / lanes); ++block) {
            mask = Reg::UpdateMask<float>(remaining);
            Reg::LoadAlign(high, outputAddr + block * lanes);
            Reg::LoadAlign(low, outputAddr + alignedRows + block * lanes);
            Reg::Duplicate(scalarHigh, epsHigh, mask);
            Reg::Duplicate(scalarLow, epsLow, mask);
            Reg::Add(sum, high, scalarHigh, mask);
            Reg::Sub(split, sum, high, mask);
            Reg::Sub(leftError, sum, split, mask);
            Reg::Sub(leftError, high, leftError, mask);
            Reg::Sub(rightError, scalarHigh, split, mask);
            Reg::Add(error, leftError, rightError, mask);
            Reg::Add(low, low, scalarLow, mask);
            Reg::Add(low, low, error, mask);
            Reg::Add(sum, sum, low, mask);
            Reg::StoreAlign(outputAddr + block * lanes, sum, mask);
        }
    }
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void ComputeCCompactApplyCamePart1(TQue<QuePosition::VECOUT, 1>& sumGradCQueue,
                                                     LocalTensor<float> compactValues, uint32_t rows, uint32_t columns,
                                                     uint32_t columnTileSize)
{
    LocalTensor<float> output = sumGradCQueue.AllocTensor<float>();
    Duplicate(output, 0.0f, 2 * columnTileSize);
    event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIdVToS);
    WaitFlag<HardEvent::V_S>(eventIdVToS);
    for (uint32_t col = 0; col < columns; ++col) {
        float high = 0.0f;
        float low = 0.0f;
        for (uint32_t row = 0; row < rows; ++row) {
            float nextHigh = 0.0f;
            float nextLow = 0.0f;
            CompensatedAddApplyCamePart1(high, low, compactValues.GetValue(row * columns + col), 0.0f, nextHigh,
                                         nextLow);
            high = nextHigh;
            low = nextLow;
        }
        output.SetValue(col, high);
        output.SetValue(columnTileSize + col, low);
    }
    event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventIdSToV);
    WaitFlag<HardEvent::S_V>(eventIdSToV);
    sumGradCQueue.EnQue<float>(output);
}

// Accumulate up to 32 independent 128x2 tiles at once. Lanes are laid out as
// [tile0-col0, tile0-col1, tile1-col0, ...], so every lane retains the exact
// row order of the scalar compensated sum while all tiles execute in parallel.
__aicore__ inline void ComputeCWidth2FullTilesApplyCamePart1(LocalTensor<float> output,
                                                             LocalTensor<float> compactValues, uint32_t tileCount,
                                                             uint32_t rowTileSize)
{
    Duplicate(output, 0.0f, tileCount * 2);
    PipeBarrier<PIPE_V>();
    __ubuf__ float* inputAddr = reinterpret_cast<__ubuf__ float*>(compactValues.GetPhyAddr());
    __ubuf__ float* outputAddr = reinterpret_cast<__ubuf__ float*>(output.GetPhyAddr());
    __VEC_SCOPE__
    {
        constexpr uint32_t tilesPerBlock = 32;
        constexpr uint32_t lanes = 64;
        Reg::RegTensor<float> high, low, value, sum, split, leftError, rightError, error, zero;
        Reg::RegTensor<uint32_t> lane, tile, doubledTile, column, base, index;
        Reg::MaskReg mask, finite, lowerFinite;
        Reg::MaskReg all = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
        Reg::Arange(reinterpret_cast<Reg::RegTensor<int32_t>&>(lane), static_cast<int32_t>(0));
        Reg::ShiftRights(tile, lane, static_cast<int16_t>(1), all);
        Reg::ShiftLefts(doubledTile, tile, static_cast<int16_t>(1), all);
        Reg::Sub(column, lane, doubledTile, all);
        Reg::Muls(base, tile, rowTileSize * 2, all);
        Reg::Add(base, base, column, all);
        const uint16_t blockCount = static_cast<uint16_t>((tileCount + tilesPerBlock - 1) / tilesPerBlock);
        for (uint16_t block = 0; block < blockCount; ++block) {
            uint32_t blockTiles = tileCount - block * tilesPerBlock < tilesPerBlock ?
                                      tileCount - block * tilesPerBlock :
                                      tilesPerBlock;
            uint32_t activeLanes = blockTiles * 2;
            mask = Reg::UpdateMask<float>(activeLanes);
            Reg::Duplicate(high, 0.0f, mask);
            Reg::Duplicate(low, 0.0f, mask);
            Reg::Duplicate(zero, 0.0f, mask);
            __ubuf__ float* blockInput = inputAddr + block * tilesPerBlock * rowTileSize * 2;
            for (uint16_t row = 0; row < static_cast<uint16_t>(rowTileSize); ++row) {
                Reg::Adds(index, base, static_cast<uint32_t>(row) * 2, mask);
                Reg::Gather(value, blockInput, index, mask);
                Reg::Add(sum, high, value, mask);
                Reg::Sub(split, sum, high, mask);
                Reg::Sub(leftError, sum, split, mask);
                Reg::Sub(leftError, high, leftError, mask);
                Reg::Sub(rightError, value, split, mask);
                Reg::Add(error, leftError, rightError, mask);
                Reg::Add(low, low, zero, mask);
                Reg::Add(low, low, error, mask);
                Reg::Compares<float, CMPMODE::LE>(finite, sum, kApplyCamePart1MaxFinite, mask);
                Reg::Compares<float, CMPMODE::GE>(lowerFinite, sum, -kApplyCamePart1MaxFinite, mask);
                Reg::MaskAnd(finite, finite, lowerFinite, mask);
                Reg::Select(low, low, zero, finite);
                Reg::Select(high, sum, sum, mask);
            }
            Reg::Add(sum, high, low, mask);
            Reg::StoreAlign(outputAddr + block * lanes, sum, mask);
        }
    }
    PipeBarrier<PIPE_V>();
}

// Reproduce the 128-slot compensated RC reduction for up to 32 M == 2 tiles
// per register block.  Each column value is combined with +0 through the six
// zero-only tree levels, then column 0 and column 1 are combined at the root.
// The compact result keeps one high and one low value per tile; callers expand
// them to the eight-float workspace slots in bulk.
__aicore__ inline void ComputeRCWidth2TilesApplyCamePart1(LocalTensor<float> output, LocalTensor<float> compactC,
                                                          uint32_t tileCount)
{
    constexpr uint32_t kVectorStoreAlignmentElements = 8;
    const uint32_t lowOffset = (tileCount + kVectorStoreAlignmentElements - 1) / kVectorStoreAlignmentElements *
                               kVectorStoreAlignmentElements;
    Duplicate(output, 0.0f, lowOffset + tileCount);
    PipeBarrier<PIPE_V>();
    __ubuf__ float* inputAddr = reinterpret_cast<__ubuf__ float*>(compactC.GetPhyAddr());
    __ubuf__ float* highOutputAddr = reinterpret_cast<__ubuf__ float*>(output.GetPhyAddr());
    __ubuf__ float* lowOutputAddr = highOutputAddr + lowOffset;
    __VEC_SCOPE__
    {
        constexpr uint32_t tilesPerBlock = 32;
        Reg::RegTensor<float> leftHigh, leftLow, rightHigh, rightLow;
        Reg::RegTensor<float> sum, split, leftError, rightError, error, zero, resultLow;
        Reg::RegTensor<uint32_t> lane, evenIndex, oddIndex;
        Reg::MaskReg mask, finite, lowerFinite;
        Reg::MaskReg all = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
        Reg::Arange(reinterpret_cast<Reg::RegTensor<int32_t>&>(lane), static_cast<int32_t>(0));
        Reg::Muls(evenIndex, lane, static_cast<uint32_t>(2), all);
        Reg::Adds(oddIndex, evenIndex, static_cast<uint32_t>(1), all);
        const uint16_t blockCount = static_cast<uint16_t>((tileCount + tilesPerBlock - 1) / tilesPerBlock);
        for (uint16_t block = 0; block < blockCount; ++block) {
            uint32_t blockTiles = tileCount - block * tilesPerBlock < tilesPerBlock ?
                                      tileCount - block * tilesPerBlock :
                                      tilesPerBlock;
            mask = Reg::UpdateMask<float>(blockTiles);
            __ubuf__ float* blockInput = inputAddr + block * tilesPerBlock * 2;
            Reg::Gather(leftHigh, blockInput, evenIndex, mask);
            Reg::Gather(rightHigh, blockInput, oddIndex, mask);
            Reg::Duplicate(leftLow, 0.0f, mask);
            Reg::Duplicate(rightLow, 0.0f, mask);
            Reg::Duplicate(zero, 0.0f, mask);

            for (uint16_t level = 0; level < 6; ++level) {
                Reg::Add(sum, leftHigh, zero, mask);
                Reg::Sub(split, sum, leftHigh, mask);
                Reg::Sub(leftError, sum, split, mask);
                Reg::Sub(leftError, leftHigh, leftError, mask);
                Reg::Sub(rightError, zero, split, mask);
                Reg::Add(error, leftError, rightError, mask);
                Reg::Add(leftLow, leftLow, zero, mask);
                Reg::Add(leftLow, leftLow, error, mask);
                Reg::Compares<float, CMPMODE::LE>(finite, sum, kApplyCamePart1MaxFinite, mask);
                Reg::Compares<float, CMPMODE::GE>(lowerFinite, sum, -kApplyCamePart1MaxFinite, mask);
                Reg::MaskAnd(finite, finite, lowerFinite, mask);
                Reg::Select(leftLow, leftLow, zero, finite);
                Reg::Select(leftHigh, sum, sum, mask);

                Reg::Add(sum, rightHigh, zero, mask);
                Reg::Sub(split, sum, rightHigh, mask);
                Reg::Sub(leftError, sum, split, mask);
                Reg::Sub(leftError, rightHigh, leftError, mask);
                Reg::Sub(rightError, zero, split, mask);
                Reg::Add(error, leftError, rightError, mask);
                Reg::Add(rightLow, rightLow, zero, mask);
                Reg::Add(rightLow, rightLow, error, mask);
                Reg::Compares<float, CMPMODE::LE>(finite, sum, kApplyCamePart1MaxFinite, mask);
                Reg::Compares<float, CMPMODE::GE>(lowerFinite, sum, -kApplyCamePart1MaxFinite, mask);
                Reg::MaskAnd(finite, finite, lowerFinite, mask);
                Reg::Select(rightLow, rightLow, zero, finite);
                Reg::Select(rightHigh, sum, sum, mask);
            }

            Reg::Add(sum, leftHigh, rightHigh, mask);
            Reg::Sub(split, sum, leftHigh, mask);
            Reg::Sub(leftError, sum, split, mask);
            Reg::Sub(leftError, leftHigh, leftError, mask);
            Reg::Sub(rightError, rightHigh, split, mask);
            Reg::Add(error, leftError, rightError, mask);
            Reg::Add(resultLow, leftLow, rightLow, mask);
            Reg::Add(resultLow, resultLow, error, mask);
            Reg::Compares<float, CMPMODE::LE>(finite, sum, kApplyCamePart1MaxFinite, mask);
            Reg::Compares<float, CMPMODE::GE>(lowerFinite, sum, -kApplyCamePart1MaxFinite, mask);
            Reg::MaskAnd(finite, finite, lowerFinite, mask);
            Reg::Select(resultLow, resultLow, zero, finite);
            Reg::StoreAlign(highOutputAddr + block * tilesPerBlock, sum, mask);
            Reg::StoreAlign(lowOutputAddr + block * tilesPerBlock, resultLow, mask);
        }
    }
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void CopyOutSumGradRWorkspaceApplyCamePart1(TQue<QuePosition::VECOUT, 1>& sumGradRQueue,
                                                              GlobalTensor<float>& workspaceSumGradR,
                                                              GlobalTensor<float>& workspaceSumGradRLow,
                                                              int64_t nLoopIdx, int64_t curRepeatTimes,
                                                              int64_t mCoreNum, int64_t nLoopNormCore,
                                                              int64_t onceHandleNum, int64_t localBlockIdx)
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
    DataCopy(workspaceSumGradRLow[partialIdx * onceHandleNum], sumGradRLocal[onceHandleNum], onceHandleNum);
    sumGradRQueue.FreeTensor(sumGradRLocal);
}

__aicore__ inline void CopyOutReductionWorkspaceApplyCamePart1(
    TQue<QuePosition::VECOUT, 1>& sumGradRCQueue, TQue<QuePosition::VECOUT, 1>& sumGradCQueue,
    GlobalTensor<float>& workspaceSumGradRC, GlobalTensor<float>& workspaceSumGradRCLow,
    GlobalTensor<float>& workspaceSumGradC, GlobalTensor<float>& workspaceSumGradCLow, int64_t offset,
    int64_t scalarSlotSize, int64_t columnLocalLowOffset, int64_t columnWorkspaceSlotSize)
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
    DataCopy(workspaceSumGradC[offset * columnWorkspaceSlotSize], sumGradCLocal, columnWorkspaceSlotSize);
    DataCopy(workspaceSumGradCLow[offset * columnWorkspaceSlotSize], sumGradCLocal[columnLocalLowOffset],
             columnWorkspaceSlotSize);
    sumGradCQueue.FreeTensor(sumGradCLocal);
    event_t eventIdMte3ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
    WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
}

// Columns are independent; each lane retains the scalar row accumulation order.
__aicore__ inline void SequentialCompensatedColumnsApplyCamePart1(LocalTensor<float> output, LocalTensor<float> values,
                                                                  uint32_t rows, uint32_t width, uint32_t columns)
{
    __ubuf__ float* inputAddr = reinterpret_cast<__ubuf__ float*>(values.GetPhyAddr());
    __ubuf__ float* outputAddr = reinterpret_cast<__ubuf__ float*>(output.GetPhyAddr());
    __VEC_SCOPE__
    {
        constexpr uint32_t lanes = 64;
        Reg::RegTensor<float> high, low, value, sum, split, leftError, rightError, error, zero;
        Reg::MaskReg mask, finite, lowerFinite;
        uint32_t remaining = columns;
        for (uint16_t block = 0; block < static_cast<uint16_t>((columns + lanes - 1) / lanes); ++block) {
            mask = Reg::UpdateMask<float>(remaining);
            Reg::Duplicate(high, 0.0f, mask);
            Reg::Duplicate(low, 0.0f, mask);
            Reg::Duplicate(zero, 0.0f, mask);
            for (uint16_t row = 0; row < static_cast<uint16_t>(rows); ++row) {
                Reg::LoadAlign(value, inputAddr + row * width + block * lanes);
                Reg::Add(sum, high, value, mask);
                Reg::Sub(split, sum, high, mask);
                Reg::Sub(leftError, sum, split, mask);
                Reg::Sub(leftError, high, leftError, mask);
                Reg::Sub(rightError, value, split, mask);
                Reg::Add(error, leftError, rightError, mask);
                // Match leftLow + rightLow + roundError, with rightLow = +0.
                Reg::Add(low, low, zero, mask);
                Reg::Add(low, low, error, mask);
                Reg::Compares<float, CMPMODE::LE>(finite, sum, kApplyCamePart1MaxFinite, mask);
                Reg::Compares<float, CMPMODE::GE>(lowerFinite, sum, -kApplyCamePart1MaxFinite, mask);
                Reg::MaskAnd(finite, finite, lowerFinite, mask);
                Reg::Select(low, low, zero, finite);
                Reg::Select(high, sum, sum, mask);
            }
            Reg::StoreAlign(outputAddr + block * lanes, high, mask);
            Reg::StoreAlign(outputAddr + width + block * lanes, low, mask);
        }
    }
}

__aicore__ inline void ComputeCApplyCamePart1(TQue<QuePosition::VECOUT, 1>& sumGradCQueue, int64_t curRepeatTimes,
                                              LocalTensor<float> gradSqrtTmpUb, int64_t mCoreNum, int64_t nCoreNum,
                                              int64_t onceHandleNum, int64_t validColumns, int64_t localBlockIdx)
{
    (void)mCoreNum;
    (void)nCoreNum;
    (void)localBlockIdx;
    LocalTensor<float> sumGradCLocal = sumGradCQueue.AllocTensor<float>();
    Duplicate(sumGradCLocal, 0.0f, 2 * onceHandleNum);
    PipeBarrier<PIPE_V>();
    SequentialCompensatedColumnsApplyCamePart1(sumGradCLocal, gradSqrtTmpUb, static_cast<uint32_t>(curRepeatTimes),
                                               static_cast<uint32_t>(onceHandleNum),
                                               static_cast<uint32_t>(validColumns));
    PipeBarrier<PIPE_V>();
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

__simd_vf__ inline void GatherStridedColumnApplyCamePart1Vf(__ubuf__ float* output, __ubuf__ float* input,
                                                            uint32_t rowCount, uint32_t rowStride, uint32_t column)
{
    Reg::RegTensor<uint32_t> lane;
    Reg::RegTensor<uint32_t> index;
    Reg::RegTensor<float> value;
    auto all = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
    Reg::Arange(reinterpret_cast<Reg::RegTensor<int32_t>&>(lane), 0);
    for (uint32_t offset = 0; offset < rowCount; offset += 64) {
        uint32_t remaining = rowCount - offset;
        auto mask = Reg::UpdateMask<float>(remaining);
        Reg::Adds(index, lane, offset, all);
        Reg::Muls(index, index, rowStride, all);
        Reg::Adds(index, index, column, all);
        Reg::Gather(value, input, index, mask);
        Reg::StoreAlign(output + offset, value, mask);
    }
}

__simd_vf__ inline void GatherM1RowsApplyCamePart1Vf(__ubuf__ float* output, __ubuf__ float* input, uint32_t rowCount,
                                                     uint32_t rowStride)
{
    Reg::RegTensor<uint32_t> lane;
    Reg::RegTensor<uint32_t> index;
    Reg::RegTensor<float> value;
    auto all = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
    Reg::Arange(reinterpret_cast<Reg::RegTensor<int32_t>&>(lane), 0);
    for (uint32_t offset = 0; offset < rowCount; offset += 64) {
        uint32_t remaining = rowCount - offset;
        auto mask = Reg::UpdateMask<float>(remaining);
        Reg::Adds(index, lane, offset, all);
        Reg::Muls(index, index, rowStride, all);
        Reg::Gather(value, input, index, mask);
        Reg::StoreAlign(output + offset, value, mask);
    }
}

// M == 1 has a single column, so C and RC are the same reduction. Gather the
// strided column and keep its high/low pair instead of first rounding C to one
// float and reducing that rounded value again for RC.
__aicore__ inline void FinishM1ReductionApplyCamePart1(LocalTensor<float> sumGradRCLocal,
                                                       LocalTensor<float> gradSquared, LocalTensor<float> scratch,
                                                       int64_t rowCount, int64_t rowStride)
{
    constexpr int64_t kScalarSlotSize = 8;
    LocalTensor<float> high = scratch;
    LocalTensor<float> low = scratch[rowStride];
    LocalTensor<float> sumTmp = scratch[2 * rowStride];
    LocalTensor<float> recoveredTmp = scratch[3 * rowStride];
    Duplicate(high, 0.0f, rowStride);
    Duplicate(low, 0.0f, rowStride);
    PipeBarrier<PIPE_V>();
    GatherM1RowsApplyCamePart1Vf(reinterpret_cast<__ubuf__ float*>(high.GetPhyAddr()),
                                 reinterpret_cast<__ubuf__ float*>(gradSquared.GetPhyAddr()),
                                 static_cast<uint32_t>(rowCount), static_cast<uint32_t>(rowStride));
    PipeBarrier<PIPE_V>();
    Duplicate(sumGradRCLocal, 0.0f, 2 * kScalarSlotSize);
    CompensatedReduceApplyCamePart1(sumGradRCLocal, high, low, sumTmp, recoveredTmp, static_cast<uint64_t>(rowCount));
}

__aicore__ inline void TwoProductApplyCamePart1(float left, float right, float& high, float& low)
{
    constexpr float splitter = 4097.0f;
    constexpr float scaleDown = 0.0001220703125f;
    constexpr float scaleUp = 8192.0f;
    constexpr uint32_t scaleExponent = static_cast<uint32_t>(127 + 115) << 23;
    high = left * right;
    if (!IsFiniteApplyCamePart1(high)) {
        low = 0.0f;
        return;
    }
    uint32_t leftBits = 0;
    uint32_t rightBits = 0;
    __builtin_memcpy(&leftBits, &left, sizeof(leftBits));
    __builtin_memcpy(&rightBits, &right, sizeof(rightBits));
    const bool scaleLeft = (leftBits & 0x7F800000U) >= scaleExponent;
    const bool scaleRight = (rightBits & 0x7F800000U) >= scaleExponent;
    const float leftForSplit = scaleLeft ? left * scaleDown : left;
    const float rightForSplit = scaleRight ? right * scaleDown : right;
    const float leftSplit = splitter * leftForSplit;
    float leftHigh = leftSplit - (leftSplit - leftForSplit);
    float leftLow = leftForSplit - leftHigh;
    const float rightSplit = splitter * rightForSplit;
    float rightHigh = rightSplit - (rightSplit - rightForSplit);
    float rightLow = rightForSplit - rightHigh;
    if (scaleLeft) {
        leftHigh *= scaleUp;
        leftLow *= scaleUp;
    }
    if (scaleRight) {
        rightHigh *= scaleUp;
        rightLow *= scaleUp;
    }
    low = ((leftHigh * rightHigh - high) + leftHigh * rightLow + leftLow * rightHigh) + leftLow * rightLow;
}

__aicore__ inline void ScaleEpsApplyCamePart1(float eps, int64_t first, int64_t second, float& high, float& low)
{
    float firstHigh = 0.0f;
    float firstLow = 0.0f;
    TwoProductApplyCamePart1(eps, static_cast<float>(first), firstHigh, firstLow);
    if (second == 1) {
        high = firstHigh;
        low = firstLow;
        return;
    }
    float productHigh = 0.0f;
    float productLow = 0.0f;
    float lowHigh = 0.0f;
    float lowLow = 0.0f;
    TwoProductApplyCamePart1(firstHigh, static_cast<float>(second), productHigh, productLow);
    TwoProductApplyCamePart1(firstLow, static_cast<float>(second), lowHigh, lowLow);
    CompensatedAddApplyCamePart1(productHigh, productLow, lowHigh, lowLow, high, low);
}

__aicore__ inline void AddCompensatedScalarApplyCamePart1(LocalTensor<float> high, LocalTensor<float> low,
                                                          LocalTensor<float> sumTmp, LocalTensor<float> recoveredTmp,
                                                          LocalTensor<float> errorTmp, float scalarHigh,
                                                          float scalarLow, uint64_t count)
{
    Adds(sumTmp, high, scalarHigh, count);
    PipeBarrier<PIPE_V>();
    Sub(recoveredTmp, sumTmp, high, count);
    PipeBarrier<PIPE_V>();
    Sub(errorTmp, sumTmp, recoveredTmp, count);
    PipeBarrier<PIPE_V>();
    Sub(errorTmp, high, errorTmp, count);
    Adds(recoveredTmp, recoveredTmp, -scalarHigh, count);
    PipeBarrier<PIPE_V>();
    Muls(recoveredTmp, recoveredTmp, -1.0f, count);
    Add(errorTmp, errorTmp, recoveredTmp, count);
    Adds(low, low, scalarLow, count);
    PipeBarrier<PIPE_V>();
    Add(low, low, errorTmp, count);
    Adds(high, sumTmp, 0.0f, count);
    PipeBarrier<PIPE_V>();
    NormalizeCompensationApplyCamePart1(high, low, recoveredTmp, errorTmp, count);
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
    Adds(low, sumGradCLocal[onceHandleNum], static_cast<float>(0), onceHandleNum);
    PipeBarrier<PIPE_V>();
    CompensatedReduceApplyCamePart1(sumGradRCLocal, high, low, sumTmp, recoveredTmp, onceHandleNum);
}

__aicore__ inline int64_t InitApplyCamePart1OutputBuffers(
    GlobalTensor<float>& sumGradR, GlobalTensor<float>& sumGradC, GlobalTensor<float>& sumGradRC,
    GlobalTensor<float>& workspaceSumGradR, GlobalTensor<float>& workspaceSumGradRLow,
    GlobalTensor<float>& workspaceSumGradRC, GlobalTensor<float>& workspaceSumGradRCLow,
    GlobalTensor<float>& workspaceSumGradC, GlobalTensor<float>& workspaceSumGradCLow, GM_ADDR sumGradRAddr,
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
    const int64_t cWorkspaceSlotSize = columnCount == 2 ? kScalarSlotSize : rowTileSize;
    const int64_t totalColumnTileCount = CeilDivApplyCamePart1(columnCount, rowTileSize);
    const int64_t rcPartialCount = nLoopCount * totalColumnTileCount;
    const int64_t rPartialCount = nLoopCount * mCoreNum;
    const int64_t rcAlignedCount = (rcPartialCount * kScalarSlotSize + 128 - 1) / 128 * 128;
    const int64_t rAlignedCount = (rPartialCount * rowTileSize + 128 - 1) / 128 * 128;
    const int64_t cAlignedCount = (rcPartialCount * cWorkspaceSlotSize + 128 - 1) / 128 * 128;
    const int64_t regionFloats = 2 * rcAlignedCount + 2 * rAlignedCount + 2 * cAlignedCount;
    auto* regionBase = reinterpret_cast<__gm__ float*>(workspaceAddr) + groupIdx * regionFloats;
    workspaceSumGradRC.SetGlobalBuffer(regionBase);
    workspaceSumGradRCLow.SetGlobalBuffer(regionBase + rcAlignedCount);
    workspaceSumGradR.SetGlobalBuffer(regionBase + 2 * rcAlignedCount);
    workspaceSumGradRLow.SetGlobalBuffer(regionBase + 2 * rcAlignedCount + rAlignedCount);
    workspaceSumGradC.SetGlobalBuffer(regionBase + 2 * rcAlignedCount + 2 * rAlignedCount);
    workspaceSumGradCLow.SetGlobalBuffer(regionBase + 2 * rcAlignedCount + 2 * rAlignedCount + cAlignedCount);
    return regionFloats;
}

} // namespace ApplyCamePart1
#endif // OPS_NN_APPLY_CAME_PART1_COMMON_H

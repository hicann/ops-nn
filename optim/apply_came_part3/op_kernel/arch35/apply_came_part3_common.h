/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef _APPLY_CAME_PART3_COMMON_H_
#define _APPLY_CAME_PART3_COMMON_H_

#include "kernel_operator.h"
#include "apply_came_part3_tiling_data.h"

constexpr int64_t SCALAR_INPUT_SIZE = 8; // 32B / 4B
constexpr int64_t DEFAULT_QUEUE_BUFFE_SIZE = 2;
constexpr int64_t ONE_VECTOR_BLOCK_SIZE = 256;
constexpr int64_t CAME_ONE_BLOCK_SIZE = 32;
constexpr int64_t REP_BLOCK_STRIDE = 8;
constexpr int64_t FP16_ONE_BLOCK_COUNT = 16;
constexpr int64_t FP32_ONE_BLOCK_COUNT = 8;
constexpr int64_t BUFFER_SIZE = 3;
constexpr int64_t ONE_BLOCK_INT32_COUNT = 8;
constexpr int64_t SPLIT_PART = 2;
constexpr int64_t ONE_VECTOR_FP32_SIZE = 64;
constexpr int64_t MAX_REPEAT_TIME = 255;
constexpr int64_t INT64_ONE_BLOCK_COUNT = 4;
constexpr int64_t MAX_POST_BUFFER_SIZE = 16384;
constexpr uint32_t DET_WORKSPACE_SIZE = 392;  // 8 * 49
constexpr uint32_t DET_WORKSPACE_BYTE = 1568; // 32B * 49

__aicore__ inline void ApplyCamePart3WaitMte3ToS()
{
    event_t eventIdMte3ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_S));
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(eventIdMte3ToS);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(eventIdMte3ToS);
}

struct CamePart3InOut {
    GM_ADDR u;
    GM_ADDR mIn;
    GM_ADDR eps;
    GM_ADDR beta1;
    GM_ADDR clipThreshold;
    GM_ADDR sumSquareU;
    GM_ADDR globalShape;
    GM_ADDR mOut;
    GM_ADDR sumUR;
    GM_ADDR sumUC;
    GM_ADDR sumURC;
};

__aicore__ inline int64_t ApplyCamePart3Ceil(int64_t value, int64_t factor)
{
    return factor == 0 ? value : (value + factor - 1) / factor * factor;
}

__aicore__ inline void ApplyCamePart3SumUCTail(AscendC::LocalTensor<float>& ubLocal4,
                                               AscendC::LocalTensor<float>& ubLocal3, int64_t baseM, int64_t rowNum,
                                               int64_t calcSize, int64_t tailBlockStride)
{
    uint64_t tailMask = baseM % ONE_VECTOR_FP32_SIZE;
    uint64_t lastOffset = baseM / ONE_VECTOR_FP32_SIZE * ONE_VECTOR_FP32_SIZE;
    int64_t repeatTimes = rowNum > MAX_REPEAT_TIME ? MAX_REPEAT_TIME : rowNum;
    int64_t remaining = rowNum;
    int64_t offset = lastOffset;
    AscendC::Add(ubLocal4[lastOffset], ubLocal3[lastOffset], ubLocal4[calcSize + lastOffset], tailMask, repeatTimes,
                 {1, 1, 1, static_cast<uint8_t>(tailBlockStride), static_cast<uint8_t>(tailBlockStride),
                  static_cast<uint8_t>(tailBlockStride)});
    remaining -= repeatTimes;
    offset += tailBlockStride * FP32_ONE_BLOCK_COUNT * repeatTimes;
    while (remaining > 0) {
        repeatTimes = remaining > MAX_REPEAT_TIME ? MAX_REPEAT_TIME : remaining;
        AscendC::Add(ubLocal4[offset], ubLocal3[offset], ubLocal4[calcSize + offset], tailMask, repeatTimes,
                     {1, 1, 1, static_cast<uint8_t>(tailBlockStride), static_cast<uint8_t>(tailBlockStride),
                      static_cast<uint8_t>(tailBlockStride)});
        remaining -= repeatTimes;
        offset += tailBlockStride * FP32_ONE_BLOCK_COUNT * repeatTimes;
    }
}

__aicore__ inline void ApplyCamePart3CalcSumUC(AscendC::LocalTensor<float>& ubLocal2,
                                               AscendC::LocalTensor<float>& ubLocal3,
                                               AscendC::LocalTensor<float>& ubLocal4, int64_t baseM, int64_t baseN,
                                               int64_t bufSize, int64_t tailBlockStride)
{
    int64_t rowNum = baseN;
    AscendC::Muls(ubLocal4, ubLocal2, static_cast<float>(1.0), bufSize);
    AscendC::Muls(ubLocal3, ubLocal2, static_cast<float>(1.0), bufSize);
    AscendC::PipeBarrier<PIPE_V>();
    int64_t blockSize = ApplyCamePart3Ceil(baseM, FP32_ONE_BLOCK_COUNT);
    int64_t calcSize = baseN / SPLIT_PART * blockSize;
    uint64_t mask = baseM < ONE_VECTOR_FP32_SIZE ? blockSize : ONE_VECTOR_FP32_SIZE;
    int64_t tailOffset = 0;
    bool hasTail = false;
    while (rowNum > 1) {
        if (rowNum % SPLIT_PART) {
            if (!hasTail) {
                hasTail = true;
                tailOffset = (rowNum - 1) * blockSize;
            } else {
                int64_t currentTail = (rowNum - 1) * blockSize;
                AscendC::Add(ubLocal3[tailOffset], ubLocal4[currentTail], ubLocal3[tailOffset], baseM);
            }
            calcSize = rowNum / SPLIT_PART * blockSize;
        }
        rowNum /= SPLIT_PART;
        AscendC::PipeBarrier<PIPE_V>();
        if (baseM % ONE_VECTOR_FP32_SIZE == 0) {
            int64_t repeatTimes = rowNum * baseM / ONE_VECTOR_FP32_SIZE;
            AscendC::Add(ubLocal4, ubLocal3, ubLocal4[calcSize], mask, repeatTimes,
                         {1, 1, 1, REP_BLOCK_STRIDE, REP_BLOCK_STRIDE, REP_BLOCK_STRIDE});
        } else {
            int64_t fullVectors = baseM / ONE_VECTOR_FP32_SIZE;
            for (int64_t vectorIdx = 0; vectorIdx < fullVectors; ++vectorIdx) {
                int64_t offset = vectorIdx * ONE_VECTOR_FP32_SIZE;
                AscendC::Add(ubLocal4[offset], ubLocal3[offset], ubLocal4[calcSize + offset], mask, rowNum,
                             {1, 1, 1, static_cast<uint8_t>(tailBlockStride), static_cast<uint8_t>(tailBlockStride),
                              static_cast<uint8_t>(tailBlockStride)});
            }
            if (baseM % ONE_VECTOR_FP32_SIZE) {
                ApplyCamePart3SumUCTail(ubLocal4, ubLocal3, baseM, rowNum, calcSize, tailBlockStride);
            }
        }
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(ubLocal3, ubLocal4, static_cast<float>(1.0), calcSize);
        AscendC::PipeBarrier<PIPE_V>();
        calcSize /= SPLIT_PART;
    }
    if (hasTail) {
        AscendC::Add(ubLocal4, ubLocal3[tailOffset], ubLocal4, baseM);
    }
    AscendC::PipeBarrier<PIPE_V>();
}

// Reduce a row-major [rowCount, rowWidth] tile without assuming that the
// row width is vector-block aligned.  Workspace C stores compact rows, so the
// reduction stride must be the actual element count rather than an aligned
// byte width.
__aicore__ inline void ApplyCamePart3ReduceRows(AscendC::LocalTensor<float>& values, int64_t rowCount, int64_t rowWidth)
{
    if (rowCount <= 1 || rowWidth <= 0) {
        return;
    }
    int64_t activeRows = rowCount;
    while (activeRows > 1) {
        const int64_t pairRows = activeRows / 2;
        for (int64_t row = 0; row < pairRows; ++row) {
            const int64_t dstOffset = row * rowWidth;
            const int64_t leftOffset = (row * 2) * rowWidth;
            const int64_t rightOffset = leftOffset + rowWidth;
            AscendC::Add(values[dstOffset], values[leftOffset], values[rightOffset], rowWidth);
        }
        if (activeRows % 2 != 0) {
            const int64_t dstOffset = pairRows * rowWidth;
            const int64_t srcOffset = (activeRows - 1) * rowWidth;
            AscendC::Muls(values[dstOffset], values[srcOffset], static_cast<float>(1.0), rowWidth);
        }
        AscendC::PipeBarrier<PIPE_V>();
        activeRows = pairRows + (activeRows % 2);
    }
}

__aicore__ inline void ApplyCamePart3ReduceVector(AscendC::LocalTensor<float>& values,
                                                  AscendC::LocalTensor<float>& input, int64_t count)
{
    const int64_t alignedCount = ApplyCamePart3Ceil(count, FP32_ONE_BLOCK_COUNT);
    AscendC::Duplicate(values, static_cast<float>(0.0), alignedCount);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Muls(values, input, static_cast<float>(1.0), count);
    AscendC::PipeBarrier<PIPE_V>();

    int64_t activeCount = alignedCount;
    while (activeCount > FP32_ONE_BLOCK_COUNT) {
        const int64_t leftCount = ApplyCamePart3Ceil(activeCount, 2 * FP32_ONE_BLOCK_COUNT) / 2;
        const int64_t pairCount = activeCount - leftCount;
        AscendC::Add(values, values[leftCount], values, pairCount);
        AscendC::PipeBarrier<PIPE_V>();
        activeCount = leftCount;
    }

    event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_S));
    AscendC::SetFlag<AscendC::HardEvent::V_S>(eventIdVToS);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(eventIdVToS);
    float sum = 0.0f;
    for (int64_t index = 0; index < activeCount; ++index) {
        sum += values.GetValue(index);
    }
    AscendC::Duplicate(values, sum, 1);
    AscendC::PipeBarrier<PIPE_V>();
}

// Preserve the roundoff discarded by each FP32 add. This follows the
// high/low reduction used by ApplyCamePart1 for its scalar output.
__aicore__ inline void ApplyCamePart3CompensatedReduce(AscendC::LocalTensor<float> high,
                                                       AscendC::LocalTensor<float> low,
                                                       AscendC::LocalTensor<float> sumTmp,
                                                       AscendC::LocalTensor<float> recoveredTmp, uint64_t count)
{
    const uint64_t alignedCount = ApplyCamePart3Ceil(count, FP32_ONE_BLOCK_COUNT);
    uint64_t currentCount = alignedCount;
    while (currentCount > FP32_ONE_BLOCK_COUNT) {
        const uint64_t leftCount = ApplyCamePart3Ceil(currentCount, 2 * FP32_ONE_BLOCK_COUNT) / 2;
        const uint64_t pairCount = currentCount - leftCount;

        AscendC::Add(sumTmp, high, high[leftCount], pairCount);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sub(recoveredTmp, sumTmp, high, pairCount);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sub(high[leftCount], high[leftCount], recoveredTmp, pairCount);
        AscendC::Sub(recoveredTmp, sumTmp, recoveredTmp, pairCount);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sub(high, high, recoveredTmp, pairCount);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(high, high, high[leftCount], pairCount);
        AscendC::Add(low, low, low[leftCount], pairCount);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(low, low, high, pairCount);
        AscendC::Adds(high, sumTmp, static_cast<float>(0), pairCount);
        AscendC::PipeBarrier<PIPE_V>();
        currentCount = leftCount;
    }

    event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_S));
    AscendC::SetFlag<AscendC::HardEvent::V_S>(eventIdVToS);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(eventIdVToS);
    while (currentCount > 1) {
        const uint64_t rightCount = (currentCount + 1) / 2;
        const uint64_t pairCount = currentCount - rightCount;
        for (uint64_t index = 0; index < pairCount; ++index) {
            const float left = high.GetValue(index);
            const float right = high.GetValue(rightCount + index);
            const float sum = left + right;
            const float split = sum - left;
            const float error = (left - (sum - split)) + (right - split);
            const float lowSum = low.GetValue(index) + low.GetValue(rightCount + index) + error;
            high.SetValue(index, sum);
            low.SetValue(index, lowSum);
        }
        currentCount = rightCount;
    }
}

#endif

/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OPS_NORM_CENTRALIZATION_GENERIC_IRREGULAR_IMPL_H_
#define OPS_NORM_CENTRALIZATION_GENERIC_IRREGULAR_IMPL_H_

__aicore__ inline void ZeroIrregularVectorSums(LocalTensor<float>& sum, int64_t validKeepElements, int64_t sumStride,
                                               int64_t validElements)
{
    for (int64_t keepLocal = 0; keepLocal < validKeepElements; ++keepLocal) {
        ZeroSumTileAt(sum, keepLocal * sumStride, validElements);
    }
}

__aicore__ inline void AccumulateIrregularVectorTile(int64_t keepBase, int64_t keepStride, int64_t innerOffset,
                                                     int64_t validKeepElements, int64_t sumStride,
                                                     int64_t validElements, LocalTensor<float>& sum)
{
    int64_t reduceCoords[8] = {};
    int64_t currentReduceOffset = 0;
    for (int64_t r = 0; r < tiling_->reduceSegmentCount; ++r) {
        const int64_t reduceOffset = tiling_->useReduceOffsetTable != 0 ? tiling_->reduceSegmentOffsets[r] :
                                                                          currentReduceOffset;
        for (int64_t keepLocal = 0; keepLocal < validKeepElements; ++keepLocal) {
            const int64_t sumOffset = keepLocal * sumStride;
            const int64_t offset = keepBase + keepLocal * keepStride + reduceOffset + innerOffset;
            CopyInContiguous(offset, validElements);
            LocalTensor<T> input = inQueue_.DeQue<T>();
            AccumulateContiguousAt(input, sum, sumOffset, validElements);
            PipeBarrier<PIPE_V>();
            inQueue_.FreeTensor(input);
        }
        if (tiling_->useReduceOffsetTable == 0) {
            AdvanceReduceOffset(currentReduceOffset, reduceCoords, tiling_->reduceRank);
        }
    }
}

__aicore__ inline void ScaleIrregularVectorSums(LocalTensor<float>& sum, int64_t validKeepElements, int64_t sumStride,
                                                int64_t validElements)
{
    for (int64_t keepLocal = 0; keepLocal < validKeepElements; ++keepLocal) {
        ScaleSumToMeanAt(sum, keepLocal * sumStride, validElements);
    }
}

__aicore__ inline void WriteIrregularVectorTile(int64_t keepBase, int64_t keepStride, int64_t innerOffset,
                                                int64_t validKeepElements, int64_t sumStride, int64_t validElements,
                                                LocalTensor<float>& sum)
{
    int64_t reduceCoordsSecond[8] = {};
    int64_t currentReduceOffset = 0;
    for (int64_t r = 0; r < tiling_->reduceSegmentCount; ++r) {
        const int64_t reduceOffset = tiling_->useReduceOffsetTable != 0 ? tiling_->reduceSegmentOffsets[r] :
                                                                          currentReduceOffset;
        for (int64_t keepLocal = 0; keepLocal < validKeepElements; ++keepLocal) {
            const int64_t meanOffset = keepLocal * sumStride;
            const int64_t offset = keepBase + keepLocal * keepStride + reduceOffset + innerOffset;
            CopyInContiguous(offset, validElements);
            LocalTensor<T> input = inQueue_.DeQue<T>();
            LocalTensor<T> output = outQueue_.AllocTensor<T>();
            CentralizeContiguousAt(input, output, sum, meanOffset, validElements);
            PipeBarrier<PIPE_V>();
            outQueue_.EnQue(output);
            inQueue_.FreeTensor(input);
            CopyOutContiguous(offset, validElements);
        }
        if (tiling_->useReduceOffsetTable == 0) {
            AdvanceReduceOffset(currentReduceOffset, reduceCoordsSecond, tiling_->reduceRank);
        }
    }
}

__aicore__ inline void ProcessIrregularVectorTileTask(int64_t keepTileStart, int64_t validKeepElements,
                                                      int64_t innerOffset, int64_t validElements)
{
    LocalTensor<float> sum = sumBuf_.Get<float>();
    const int64_t sumStride = AlignToVector(validElements);
    ZeroIrregularVectorSums(sum, validKeepElements, sumStride, validElements);
    PipeBarrier<PIPE_V>();

    const int64_t keepBase = IrregularKeepOuterBase(keepTileStart);
    const int64_t keepStride = tiling_->irregularKeepOuterRank > 0 ?
                                   tiling_->keepStrides[tiling_->irregularKeepOuterRank - 1] :
                                   0;
    AccumulateIrregularVectorTile(keepBase, keepStride, innerOffset, validKeepElements, sumStride, validElements, sum);
    ScaleIrregularVectorSums(sum, validKeepElements, sumStride, validElements);
    PipeBarrier<PIPE_V>();
    WriteIrregularVectorTile(keepBase, keepStride, innerOffset, validKeepElements, sumStride, validElements, sum);
}

__aicore__ inline void ProcessIrregularVector(uint32_t core, uint32_t cores)
{
    if (tiling_->irregularVectorizable == 2) {
        ProcessIrregularReduceSuffix(core, cores);
        return;
    }
    const int64_t tileElements = tiling_->irregularVectorTileElements > 0 ?
                                     Min(tiling_->irregularVectorTileElements, kLargeChunkElements) :
                                     kLargeChunkElements;
    const int64_t innerTileCount = tiling_->irregularVectorTileCount;
    const int64_t keepTileCountPerPrefix = tiling_->irregularKeepTileCountPerPrefix;
    const int64_t taskCount = tiling_->irregularVectorTaskCount;
    for (int64_t task = core; task < taskCount; task += cores) {
        const int64_t keepTileTask = task / innerTileCount;
        const int64_t innerTile = task - keepTileTask * innerTileCount;
        const int64_t keepPrefix = keepTileTask / keepTileCountPerPrefix;
        const int64_t keepTile = keepTileTask - keepPrefix * keepTileCountPerPrefix;
        const int64_t keepTileStart = keepPrefix * tiling_->irregularKeepInnerCount +
                                      keepTile * tiling_->irregularKeepTileElements;
        const int64_t validKeepElements = Min(
            tiling_->irregularKeepTileElements,
            tiling_->irregularKeepInnerCount - keepTile * tiling_->irregularKeepTileElements);
        const int64_t innerOffset = innerTile * tileElements;
        const int64_t validElements = Min(tileElements, tiling_->irregularVectorInnerCount - innerOffset);
        ProcessIrregularVectorTileTask(keepTileStart, validKeepElements, innerOffset, validElements);
    }
}

__aicore__ inline void ProcessIrregularReduceSuffixTask(int64_t keepOuter)
{
    float sum = 0.0f;
    const int64_t keepBase = IrregularKeepOuterBase(keepOuter);
    for (int64_t reduceOuter = 0; reduceOuter < tiling_->irregularReduceOuterCount; ++reduceOuter) {
        const int64_t segmentBase = keepBase + IrregularReduceOuterOffset(reduceOuter);
        for (int64_t segmentOffset = 0; segmentOffset < tiling_->irregularReduceSuffixElements;
             segmentOffset += kLargeChunkElements) {
            const int64_t validElements = Min(kLargeChunkElements,
                                              tiling_->irregularReduceSuffixElements - segmentOffset);
            CopyInContiguous(segmentBase + segmentOffset, validElements);
            LocalTensor<T> input = inQueue_.DeQue<T>();
            sum += ReduceSegmentToScalar(input, validElements);
            inQueue_.FreeTensor(input);
        }
    }
    const float mean = sum / static_cast<float>(tiling_->reduceCount);

    for (int64_t reduceOuter = 0; reduceOuter < tiling_->irregularReduceOuterCount; ++reduceOuter) {
        const int64_t segmentBase = keepBase + IrregularReduceOuterOffset(reduceOuter);
        for (int64_t segmentOffset = 0; segmentOffset < tiling_->irregularReduceSuffixElements;
             segmentOffset += kLargeChunkElements) {
            const int64_t validElements = Min(kLargeChunkElements,
                                              tiling_->irregularReduceSuffixElements - segmentOffset);
            CopyInContiguous(segmentBase + segmentOffset, validElements);
            LocalTensor<T> input = inQueue_.DeQue<T>();
            LocalTensor<T> output = outQueue_.AllocTensor<T>();
            CentralizeSegmentScalar(input, output, validElements, mean);
            PipeBarrier<PIPE_V>();
            outQueue_.EnQue(output);
            inQueue_.FreeTensor(input);
            CopyOutContiguous(segmentBase + segmentOffset, validElements);
        }
    }
}

__aicore__ inline void AccumulateIrregularReduceSuffixTile(int64_t keepBase, int64_t keepStride,
                                                           int64_t validKeepElements, float* sums)
{
    int64_t reduceCoords[8] = {};
    int64_t currentReduceOffset = 0;
    for (int64_t reduceOuter = 0; reduceOuter < tiling_->reduceSegmentCount; ++reduceOuter) {
        const int64_t reduceOffset = tiling_->useReduceOffsetTable != 0 ? tiling_->reduceSegmentOffsets[reduceOuter] :
                                                                          currentReduceOffset;
        for (int64_t keepLocal = 0; keepLocal < validKeepElements; ++keepLocal) {
            const int64_t segmentBase = keepBase + keepLocal * keepStride + reduceOffset;
            for (int64_t segmentOffset = 0; segmentOffset < tiling_->irregularReduceSuffixElements;
                 segmentOffset += kLargeChunkElements) {
                const int64_t validElements = Min(kLargeChunkElements,
                                                  tiling_->irregularReduceSuffixElements - segmentOffset);
                CopyInContiguous(segmentBase + segmentOffset, validElements);
                LocalTensor<T> input = inQueue_.DeQue<T>();
                sums[keepLocal] += ReduceSegmentToScalar(input, validElements);
                inQueue_.FreeTensor(input);
            }
        }
        if (tiling_->useReduceOffsetTable == 0) {
            AdvanceReduceOffset(currentReduceOffset, reduceCoords, tiling_->irregularReduceOuterRank);
        }
    }
}

__aicore__ inline void ScaleIrregularReduceSuffixSums(float* sums, int64_t validKeepElements)
{
    for (int64_t keepLocal = 0; keepLocal < validKeepElements; ++keepLocal) {
        sums[keepLocal] /= static_cast<float>(tiling_->reduceCount);
    }
}

__aicore__ inline void WriteIrregularReduceSuffixTile(int64_t keepBase, int64_t keepStride, int64_t validKeepElements,
                                                      const float* sums)
{
    int64_t reduceCoordsSecond[8] = {};
    int64_t currentReduceOffset = 0;
    for (int64_t reduceOuter = 0; reduceOuter < tiling_->reduceSegmentCount; ++reduceOuter) {
        const int64_t reduceOffset = tiling_->useReduceOffsetTable != 0 ? tiling_->reduceSegmentOffsets[reduceOuter] :
                                                                          currentReduceOffset;
        for (int64_t keepLocal = 0; keepLocal < validKeepElements; ++keepLocal) {
            const int64_t segmentBase = keepBase + keepLocal * keepStride + reduceOffset;
            for (int64_t segmentOffset = 0; segmentOffset < tiling_->irregularReduceSuffixElements;
                 segmentOffset += kLargeChunkElements) {
                const int64_t validElements = Min(kLargeChunkElements,
                                                  tiling_->irregularReduceSuffixElements - segmentOffset);
                CopyInContiguous(segmentBase + segmentOffset, validElements);
                LocalTensor<T> input = inQueue_.DeQue<T>();
                LocalTensor<T> output = outQueue_.AllocTensor<T>();
                CentralizeSegmentScalar(input, output, validElements, sums[keepLocal]);
                PipeBarrier<PIPE_V>();
                outQueue_.EnQue(output);
                inQueue_.FreeTensor(input);
                CopyOutContiguous(segmentBase + segmentOffset, validElements);
            }
        }
        if (tiling_->useReduceOffsetTable == 0) {
            AdvanceReduceOffset(currentReduceOffset, reduceCoordsSecond, tiling_->irregularReduceOuterRank);
        }
    }
}

__aicore__ inline void ProcessIrregularReduceSuffixTileTask(int64_t keepTileStart, int64_t validKeepElements)
{
    float sums[kMaxIrregularScalarKeepTile] = {};
    const int64_t keepBase = IrregularKeepOuterBase(keepTileStart);
    const int64_t keepStride = tiling_->irregularKeepOuterRank > 0 ?
                                   tiling_->keepStrides[tiling_->irregularKeepOuterRank - 1] :
                                   0;
    AccumulateIrregularReduceSuffixTile(keepBase, keepStride, validKeepElements, sums);
    ScaleIrregularReduceSuffixSums(sums, validKeepElements);
    WriteIrregularReduceSuffixTile(keepBase, keepStride, validKeepElements, sums);
}

__aicore__ inline void ProcessIrregularReduceSuffix(uint32_t core, uint32_t cores)
{
    const int64_t keepTileCountPerPrefix = tiling_->irregularKeepTileCountPerPrefix;
    const int64_t taskCount = tiling_->irregularKeepTileTaskCount;
    for (int64_t task = core; task < taskCount; task += cores) {
        const int64_t keepPrefix = task / keepTileCountPerPrefix;
        const int64_t keepTile = task - keepPrefix * keepTileCountPerPrefix;
        const int64_t keepTileStart = keepPrefix * tiling_->irregularKeepInnerCount +
                                      keepTile * tiling_->irregularKeepTileElements;
        const int64_t validKeepElements = Min(
            tiling_->irregularKeepTileElements,
            tiling_->irregularKeepInnerCount - keepTile * tiling_->irregularKeepTileElements);
        ProcessIrregularReduceSuffixTileTask(keepTileStart, validKeepElements);
    }
}

#endif

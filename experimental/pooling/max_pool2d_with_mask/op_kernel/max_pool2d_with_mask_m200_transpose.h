/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef MAX_POOL2D_WITH_MASK_M200_TRANSPOSE_H
#define MAX_POOL2D_WITH_MASK_M200_TRANSPOSE_H
#include "kernel_operator.h"
#include "max_pool2d_with_mask_tiling_data.h"

namespace MaxPool2dWithMask {
using namespace AscendC;

template <typename T, uint32_t NC = 16, bool ENABLE_SEPARABLE = false, bool PARITY_ROWS = false,
          bool CACHE_HORIZONTAL_ROW = false, bool REUSE_FLAT_BUFFERS = false, bool HALF_INDICES = false,
          bool FLAT_SEPARABLE = false>
class M200TransposeKernel {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR out, GM_ADDR indices, const MaxPool2dWithMaskTilingData* tiling)
    {
        t_ = tiling;
        x_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x));
        out_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(out));
        indices_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(indices));
        groupCount_ = t_->totalSlices / NC;
        const int64_t tiles = (t_->outputHeight + t_->m200NcRows - 1) / t_->m200NcRows;
        const int64_t tasks = tiles * groupCount_;
        const int64_t tasksPerCore = (tasks + GetBlockNum() - 1) / GetBlockNum();
        firstTask_ = GetBlockIdx() * tasksPerCore;
        lastTask_ = firstTask_ + tasksPerCore;
        if (lastTask_ > tasks)
            lastTask_ = tasks;
        maxInputRows_ = (t_->m200NcRows - 1) * t_->strideH + t_->kernelH;
        paddedWidth_ = (t_->outputWidth - 1) * t_->strideW + t_->kernelW;
        inputPlane_ = t_->inputHeight * t_->inputWidth;
        outputPlane_ = t_->outputHeight * t_->outputWidth;
        packed_ = inputPlane_ * sizeof(T) % 32 != 0 || outputPlane_ * sizeof(T) % 32 != 0;
        inputSpace_ = Align(packed_ ? inputPlane_ : maxInputRows_ * t_->inputWidth + 64 / sizeof(T), 16);
        outputSpace_ = Align(t_->m200NcRows * t_->outputWidth, 16);
        vectorCount_ = outputSpace_ * NC;
        // With unit stride the window source and the running maximum can share the padded
        // row pitch, so one flat instruction covers every output row at once and the
        // per-row gather into a contiguous candidate buffer disappears entirely.
        flatWindow_ = t_->strideH == 1 && t_->strideW == 1;
        reuseFlatBuffers_ = REUSE_FLAT_BUFFERS && flatWindow_ && !packed_;
        compactAlias_ = reuseFlatBuffers_ && inputSpace_ * NC * sizeof(T) >= vectorCount_ * sizeof(int32_t);
        // Every tile must be entirely inside the input before the padded buffer
        // can be omitted. In ceil mode the final window may otherwise overrun it.
        padStatic_ = !flatWindow_ && t_->padH == 0 && t_->padW == 0 && paddedWidth_ == t_->inputWidth &&
                     (t_->inputWidth * sizeof(T)) % 32 == 0 &&
                     (t_->outputHeight - 1) * t_->strideH + t_->kernelH <= t_->inputHeight;
        poolSpace_ = flatWindow_ ? Align(t_->m200NcRows * paddedWidth_, 16) : outputSpace_;
        if (poolSpace_ < outputSpace_)
            poolSpace_ = outputSpace_;
        poolCount_ = poolSpace_ * NC;
        // Only alias non-packed input: keeping the early release of packed staging
        // preserves overlap on small planes. The host budgets the same allocation.
        padAlias_ = !padStatic_ && !packed_;
        const int64_t paddedElems = maxInputRows_ * paddedWidth_ + t_->kernelW + 16;
        // A horizontal reduction can reuse the dead transpose storage. Save its
        // comparison masks in the output-index buffer until padded input is dead,
        // then replay those masks into the raw/padded allocation as horizontal
        // indices. Every bound uses the largest tile, including alignment tails.
        const int64_t rawElems = padAlias_ && paddedElems > inputSpace_ ? paddedElems : inputSpace_;
        if constexpr (ENABLE_SEPARABLE) {
            horizontalCount_ = Align(maxInputRows_ * t_->outputWidth * NC, 256);
            horizontalMaskBytes_ = horizontalCount_ / 8;
            separable_ = NC == 16 && padAlias_ && t_->kernelH == t_->kernelW &&
                         (t_->kernelH == 7 || (PARITY_ROWS && t_->kernelH == 3)) && t_->strideH == 2 &&
                         t_->strideW == 2 && 2 * horizontalCount_ * sizeof(T) <= inputSpace_ * NC * sizeof(T) &&
                         horizontalCount_ * sizeof(int32_t) <= rawElems * NC * sizeof(T) &&
                         (t_->kernelW - 1) * horizontalMaskBytes_ <= poolCount_ * sizeof(int32_t) &&
                         horizontalCount_ * sizeof(T) >= poolCount_ * sizeof(int32_t) &&
                         (!HALF_INDICES || (t_->kernelH - 1) * t_->inputWidth + t_->kernelW - 1 <= 2048) &&
                         (!PARITY_ROWS ||
                          (maxInputRows_ - t_->m200NcRows) * t_->outputWidth * NC + vectorCount_ <= horizontalCount_);
        }
        if (!packed_) {
            pipe_.InitBuffer(rawBuffer_, rawElems * NC * sizeof(T));
        }
        int64_t transposeElements = inputSpace_ * NC;
        if constexpr (FLAT_SEPARABLE) {
            const int64_t horizontal = Align(Align(t_->m200NcRows * paddedWidth_ * NC, 64) + 2 * paddedWidth_ * NC, 64);
            if (horizontal > transposeElements)
                transposeElements = horizontal;
        }
        pipe_.InitBuffer(transposeBuffer_, transposeElements * sizeof(T));
        // The flat form reads one window shift past the last row, so the buffer carries
        // kernelW extra positions; they are never used as results.
        if (!padStatic_ && !padAlias_)
            pipe_.InitBuffer(paddedBuffer_, (maxInputRows_ * paddedWidth_ + t_->kernelW + 16) * NC * sizeof(T));
        pipe_.InitBuffer(maximumBuffer_, poolCount_ * sizeof(T));
        if (!reuseFlatBuffers_)
            pipe_.InitBuffer(candidateBuffer_, poolCount_ * sizeof(T));
        pipe_.InitBuffer(indexBuffer_, poolCount_ * sizeof(int32_t));
        pipe_.InitBuffer(indexTemplateBuffer_, poolCount_ * sizeof(int32_t));
        if (!packed_)
            pipe_.InitBuffer(outputTransposeBuffer_, vectorCount_ * sizeof(int32_t));
        pipe_.InitBuffer(maskBuffer_, Align(poolCount_ / 8, 32));
        if (!packed_ && flatWindow_ && !compactAlias_)
            pipe_.InitBuffer(compactBuffer_, vectorCount_ * sizeof(int32_t));
        pipe_.InitBuffer(zeroBuffer_, 8192);
        if constexpr (CACHE_HORIZONTAL_ROW) {
            cacheEnabled_ = separable_ && PARITY_ROWS && t_->kernelH == 3;
            if (cacheEnabled_) {
                pipe_.InitBuffer(cachedValuesBuffer_, t_->outputWidth * NC * sizeof(T));
                pipe_.InitBuffer(cachedIndicesBuffer_, t_->outputWidth * NC * sizeof(int32_t));
            }
        }
        rawReuseEvent_ = static_cast<event_t>(pipe_.FetchEventID(HardEvent::V_MTE2));
        SetFlag<HardEvent::V_MTE2>(rawReuseEvent_);
        zeroNext_ = GetBlockIdx() * t_->zeroBytesPerCore;
        zeroEnd_ = zeroNext_ + t_->zeroBytesPerCore;
        const int64_t tailBytes = t_->containerBytes - t_->validIndexBytes;
        if (zeroEnd_ > tailBytes)
            zeroEnd_ = tailBytes;
        if (zeroNext_ < zeroEnd_) {
            Duplicate(zeroBuffer_.Get<int32_t>(), 0, 8192 / sizeof(int32_t));
            Sync<HardEvent::V_MTE3>();
        }
        if (packed_) {
            packedOutputs_ = Align(outputPlane_ * NC, 128);
            int64_t packedBytes = inputPlane_ * NC * sizeof(T);
            if (packedBytes < packedOutputs_ * 4)
                packedBytes = packedOutputs_ * 4;
            pipe_.InitBuffer(packedBuffer_, packedBytes);
            pipe_.InitBuffer(inputOffsetsBuffer_, inputSpace_ * NC * 4);
            pipe_.InitBuffer(outputOffsetsBuffer_, packedOutputs_ * 4);
            pipe_.InitBuffer(indexOffsetsBuffer_, packedOutputs_ * 4);
            auto inputOffsets = inputOffsetsBuffer_.Get<uint32_t>();
            auto outputOffsets = outputOffsetsBuffer_.Get<uint32_t>();
            auto indexOffsets = indexOffsetsBuffer_.Get<uint32_t>();
            // Seed a short offset pattern once and translate it for later slices.
            // A group must contain a whole 32-byte block of int32 offsets so every
            // Adds destination is aligned even when the output plane has 49 elements.
            const int64_t offsetGroup = outputPlane_ % 8 == 0 ? 1 :
                                        outputPlane_ % 4 == 0 ? 2 :
                                        outputPlane_ % 2 == 0 ? 4 :
                                                                8;
            const int64_t groupElements = offsetGroup * outputPlane_;
            Duplicate(outputOffsets, uint32_t(0), packedOutputs_);
            Duplicate(indexOffsets, uint32_t(0), packedOutputs_);
            Sync<HardEvent::V_S>();
            for (int64_t nc = 0; nc < NC; ++nc) {
                inputOffsets.SetValue(nc, nc * inputPlane_ * sizeof(T));
            }
            const int64_t poolPitch = flatWindow_ ? paddedWidth_ : t_->outputWidth;
            for (int64_t i = 0; i < groupElements; ++i) {
                const int64_t spatial = i % outputPlane_;
                const int64_t element = (spatial / t_->outputWidth * poolPitch + spatial % t_->outputWidth) * NC +
                                        i / outputPlane_;
                outputOffsets.SetValue(i, element * sizeof(T));
                indexOffsets.SetValue(i, element * 4);
            }
            Sync<HardEvent::S_V>();
            auto inputPattern = inputOffsets.ReinterpretCast<int32_t>();
            auto outputPattern = outputOffsets.ReinterpretCast<int32_t>();
            auto indexPattern = indexOffsets.ReinterpretCast<int32_t>();
            for (int64_t spatial = 1; spatial < inputSpace_; ++spatial) {
                if (spatial < inputPlane_) {
                    Adds(inputPattern[spatial * NC], inputPattern, static_cast<int32_t>(spatial * sizeof(T)), NC);
                } else {
                    Duplicate(inputPattern[spatial * NC], int32_t(0), NC);
                }
            }
            for (int64_t nc = offsetGroup; nc < NC; nc += offsetGroup) {
                Adds(outputPattern[nc * outputPlane_], outputPattern, static_cast<int32_t>(nc * sizeof(T)),
                     groupElements);
                Adds(indexPattern[nc * outputPlane_], indexPattern, static_cast<int32_t>(nc * 4), groupElements);
            }
        }
        uint32_t negativeBits = sizeof(T) == 2 ? 0x0000fc00U : 0xff800000U;
        negativeInfinity_ = *reinterpret_cast<T*>(&negativeBits);
        Duplicate(maximumBuffer_.Get<T>(), negativeInfinity_, vectorCount_);
        if (!reuseFlatBuffers_)
            Duplicate(candidateBuffer_.Get<T>(), negativeInfinity_, vectorCount_);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void Process()
    {
        if constexpr (CACHE_HORIZONTAL_ROW) {
            if (cacheEnabled_) {
                int64_t remaining = lastTask_ - firstTask_;
                // Preserve each core's original task interval, but visit adjacent
                // row tiles of one channel group together so their common row survives.
                for (int64_t group = 0; group < groupCount_; ++group) {
                    const int64_t begin = (firstTask_ - group + groupCount_ - 1) / groupCount_;
                    const int64_t end = (lastTask_ - group + groupCount_ - 1) / groupCount_;
                    cacheValid_ = false;
                    for (int64_t tile = begin; tile < end; ++tile) {
                        const int64_t row = tile * t_->m200NcRows;
                        const int64_t rows = row + t_->m200NcRows > t_->outputHeight ? t_->outputHeight - row :
                                                                                       t_->m200NcRows;
                        Prepare(row, rows);
                        ClearTailForTask(remaining--);
                        Compute(group, row, rows);
                        cacheValid_ = true;
                    }
                }
                ClearTailForTask(1);
                WaitFlag<HardEvent::V_MTE2>(rawReuseEvent_);
                PipeBarrier<PIPE_ALL>();
                return;
            }
        }
        // Partition both spatial tiles and channels. One prepared spatial
        // template is shared by all channel groups assigned to this core.
        for (int64_t task = firstTask_; task < lastTask_;) {
            const int64_t row = task / groupCount_ * t_->m200NcRows;
            const int64_t firstGroup = task % groupCount_;
            int64_t groups = groupCount_ - firstGroup;
            if (groups > lastTask_ - task)
                groups = lastTask_ - task;
            int64_t rows = t_->m200NcRows;
            if (row + rows > t_->outputHeight)
                rows = t_->outputHeight - row;
            Prepare(row, rows);
            for (int64_t group = firstGroup; group < firstGroup + groups; ++group) {
                // Packed input and output share staging. Order its preceding
                // output DMA before the next input DMA, but place that dependency
                // before the disjoint tail stores so they can overlap input work.
                if (packed_) {
                    Sync<HardEvent::MTE3_MTE2>();
                }
                ClearTailForTask(lastTask_ - task - (group - firstGroup));
                Compute(group, row, rows);
            }
            task += groups;
        }
        // Some launched cores may have no pooling work but still own a tail range.
        ClearTailForTask(1);
        WaitFlag<HardEvent::V_MTE2>(rawReuseEvent_);
        PipeBarrier<PIPE_ALL>();
    }

private:
    // Bit pattern of 1.0f. The window offset is carried through Select as a scalar,
    // and Select only moves bits, so biasing keeps every stored value inside the
    // exponent range of a normal float and no pattern can be flushed to zero.
    // The coordinate template carries -INDEX_BIAS so one Add restores the plane index.
    static constexpr int32_t INDEX_BIAS = 0x3F800000;

    __aicore__ inline int64_t Align(int64_t value, int64_t alignment)
    {
        return (value + alignment - 1) / alignment * alignment;
    }
    // Each NC slice is a contiguous run. Use one descriptor when its lengths
    // and gaps are representable in 32-byte blocks; otherwise keep per-slice copies.
    template <typename U>
    __aicore__ inline void StoreSlices(GlobalTensor<U> dst, LocalTensor<U> src, int64_t group, int64_t row,
                                       int64_t rows)
    {
        const int64_t elements = rows * t_->outputWidth;
        const int64_t run = elements * sizeof(U);
        const int64_t srcGap = (outputSpace_ - elements) * sizeof(U);
        const int64_t dstGap = (outputPlane_ - elements) * sizeof(U);
        if (run % 32 == 0 && srcGap % 32 == 0 && dstGap % 32 == 0 && run / 32 <= 65535 && srcGap / 32 <= 65535 &&
            dstGap / 32 <= 65535) {
            DataCopyParams params;
            params.blockCount = NC;
            params.blockLen = static_cast<uint16_t>(run / 32);
            params.srcStride = static_cast<uint16_t>(srcGap / 32);
            params.dstStride = static_cast<uint16_t>(dstGap / 32);
            DataCopy(dst[group * NC * outputPlane_ + row * t_->outputWidth], src, params);
        } else {
            for (int64_t nc = 0; nc < NC; ++nc) {
                const int64_t offset = (group * NC + nc) * outputPlane_ + row * t_->outputWidth;
                DataCopy(dst[offset], src[nc * outputSpace_], static_cast<uint32_t>(elements));
            }
        }
    }
    template <HardEvent event>
    __aicore__ inline void Sync()
    {
        const event_t id = static_cast<event_t>(pipe_.FetchEventID(event));
        SetFlag<event>(id);
        WaitFlag<event>(id);
    }

    __aicore__ inline void Prepare(int64_t row, int64_t rows)
    {
        const int64_t origin = row * t_->strideH - t_->padH;
        skippedInputRows_ = cacheEnabled_ && cacheValid_ ? 1 : 0;
        firstInputRow_ = origin < 0 ? 0 : origin + skippedInputRows_;
        topPadding_ = origin < 0 ? -origin : 0;
        inputRows_ = (rows - 1) * t_->strideH + t_->kernelH;
        int64_t lastRow = origin + inputRows_;
        if (lastRow > t_->inputHeight)
            lastRow = t_->inputHeight;
        validInputRows_ = lastRow - firstInputRow_;
        const int64_t blockElements = 32 / sizeof(T);
        alignedInputStart_ = firstInputRow_ * t_->inputWidth / blockElements * blockElements;
        inputPrefix_ = firstInputRow_ * t_->inputWidth - alignedInputStart_;
        inputCount_ = Align(lastRow * t_->inputWidth, blockElements) - alignedInputStart_;
        auto base = indexTemplateBuffer_.Get<int32_t>();
        if (cacheEnabled_ && templateReady_) {
            Adds(base, base, static_cast<int32_t>((row - preparedRow_) * t_->strideH * t_->inputWidth), poolCount_);
            PipeBarrier<PIPE_V>();
            preparedRow_ = row;
            return;
        }
        const int32_t bias = HALF_INDICES && separable_ ? 0 : INDEX_BIAS;
        Duplicate(base, -bias, poolCount_);
        PipeBarrier<PIPE_V>();
        // The template follows whatever pitch the pooling uses; the columns past the
        // output width exist only so the rows line up and are dropped when compacting.
        const int64_t poolPitch = flatWindow_ ? paddedWidth_ : t_->outputWidth;
        const int64_t templateRows = cacheEnabled_ ? t_->m200NcRows : rows;
        for (int64_t r = 0; r < templateRows; ++r) {
            for (int64_t c = 0; c < t_->outputWidth; ++c) {
                const int32_t position = ((row + r) * t_->strideH - t_->padH) * t_->inputWidth + c * t_->strideW -
                                         t_->padW;
                Duplicate(base[(r * poolPitch + c) * NC], position - bias, NC);
            }
        }
        PipeBarrier<PIPE_V>();
        templateReady_ = cacheEnabled_;
        preparedRow_ = row;
    }

    __aicore__ inline void ComputeSeparableHalfIndices(int64_t cachedPrefix, int64_t activeCount)
    {
        auto horizontal = transposeBuffer_.Get<T>();
        auto horizontalCandidate = horizontal[horizontalCount_];
        auto savedMasks = indexBuffer_.Get<uint8_t>();
        // The horizontal stage finished reading padded/raw input. A half now
        // stores the exact column offset, not a converted input value.
        auto horizontalIndices = rawBuffer_.Get<half>();
        Duplicate(horizontalIndices, static_cast<half>(0), horizontalCount_);
        PipeBarrier<PIPE_V>();
        for (int32_t kw = 1; kw < t_->kernelW; ++kw) {
            Select(horizontalIndices[cachedPrefix], savedMasks[(kw - 1) * horizontalMaskBytes_],
                   horizontalIndices[cachedPrefix], static_cast<half>(kw), SELMODE::VSEL_TENSOR_SCALAR_MODE,
                   activeCount);
            PipeBarrier<PIPE_V>();
        }
        if (cacheEnabled_) {
            const uint32_t count = t_->outputWidth * NC;
            if (cacheValid_) {
                DataCopy(horizontal, cachedValuesBuffer_.Get<T>(), count);
                DataCopy(horizontalIndices, cachedIndicesBuffer_.Get<half>(), count);
                PipeBarrier<PIPE_V>();
            }
            const int64_t lastEven = (inputRows_ - 1) / 2 * count;
            DataCopy(cachedValuesBuffer_.Get<T>(), horizontal[lastEven], count);
            DataCopy(cachedIndicesBuffer_.Get<half>(), horizontalIndices[lastEven], count);
            PipeBarrier<PIPE_V>();
        }
        // All masks have been replayed before their storage becomes winners.
        auto winners = indexBuffer_.Get<half>();
        auto candidateIndices = horizontalCandidate.template ReinterpretCast<half>();
        auto maximum = maximumBuffer_.Get<T>();
        auto mask = maskBuffer_.Get<uint8_t>();
        DataCopy(maximum, horizontal, vectorCount_);
        DataCopy(winners, horizontalIndices, vectorCount_);
        PipeBarrier<PIPE_V>();
        for (int32_t kh = 1; kh < t_->kernelH; ++kh) {
            const int64_t sourceRow = kh / 2 + (kh % 2) * ((maxInputRows_ + 1) / 2);
            const int64_t offset = sourceRow * t_->outputWidth * NC;
            Adds(candidateIndices, horizontalIndices[offset], static_cast<half>(kh * t_->inputWidth), vectorCount_);
            Compare(mask, horizontal[offset], maximum, CMPMODE::LE, vectorCount_);
            PipeBarrier<PIPE_V>();
            Select(maximum, mask, maximum, horizontal[offset], SELMODE::VSEL_TENSOR_TENSOR_MODE, vectorCount_);
            Select(winners, mask, winners, candidateIndices, SELMODE::VSEL_TENSOR_TENSOR_MODE, vectorCount_);
            PipeBarrier<PIPE_V>();
        }
        // Candidate indices are dead after the last Select. The existing
        // capacity guard reserves at least vectorCount_ * 4 bytes here even
        // when T is half. This is separate from the half winner source.
        Cast(horizontalCandidate.template ReinterpretCast<int32_t>(), winners, RoundMode::CAST_RINT, vectorCount_);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ComputeSeparable(LocalTensor<T> padded, int64_t rows)
    {
        auto horizontal = transposeBuffer_.Get<T>();
        auto horizontalCandidate = horizontal[horizontalCount_];
        auto savedMasks = indexBuffer_.Get<uint8_t>();
        const int64_t cachedPrefix = skippedInputRows_ * t_->outputWidth * NC;
        const int64_t activeCount = cachedPrefix == 0 ?
                                        horizontalCount_ :
                                        Align((maxInputRows_ - skippedInputRows_) * t_->outputWidth * NC, 256);
        Duplicate(horizontal, negativeInfinity_, horizontalCount_);
        Duplicate(horizontalCandidate, negativeInfinity_, horizontalCount_);
        PipeBarrier<PIPE_V>();
        DataCopyParams columns;
        columns.blockCount = static_cast<uint16_t>(t_->outputWidth);
        columns.blockLen = NC * sizeof(T) / 32;
        columns.srcStride = columns.blockLen; // strideW == 2
        columns.dstStride = 0;
        for (int64_t kw = 0; kw < t_->kernelW; ++kw) {
            auto target = kw == 0 ? horizontal : horizontalCandidate;
            for (int64_t r = skippedInputRows_; r < inputRows_; ++r) {
                const int64_t targetRow = PARITY_ROWS ? r / 2 + (r % 2) * ((maxInputRows_ + 1) / 2) : r;
                DataCopy(target[targetRow * t_->outputWidth * NC], padded[(r * paddedWidth_ + kw) * NC], columns);
            }
            PipeBarrier<PIPE_V>();
            if (kw != 0) {
                auto saved = savedMasks[(kw - 1) * horizontalMaskBytes_];
                Compare(saved, horizontalCandidate[cachedPrefix], horizontal[cachedPrefix], CMPMODE::LE, activeCount);
                PipeBarrier<PIPE_V>();
                Select(horizontal[cachedPrefix], saved, horizontal[cachedPrefix], horizontalCandidate[cachedPrefix],
                       SELMODE::VSEL_TENSOR_TENSOR_MODE, activeCount);
                PipeBarrier<PIPE_V>();
            }
        }

        if constexpr (HALF_INDICES) {
            ComputeSeparableHalfIndices(cachedPrefix, activeCount);
            return;
        }

        // All padded input reads have completed. Keep V_MTE2 unreleased: this
        // allocation now contains indices that the vertical stage still reads.
        auto horizontalIndices = rawBuffer_.Get<int32_t>();
        Duplicate(horizontalIndices, INDEX_BIAS, horizontalCount_);
        PipeBarrier<PIPE_V>();
        for (int32_t kw = 1; kw < t_->kernelW; ++kw) {
            const int32_t encoded = INDEX_BIAS + kw;
            Select(horizontalIndices[cachedPrefix].template ReinterpretCast<float>(),
                   savedMasks[(kw - 1) * horizontalMaskBytes_],
                   horizontalIndices[cachedPrefix].template ReinterpretCast<float>(), *((float*)&encoded),
                   SELMODE::VSEL_TENSOR_SCALAR_MODE, activeCount);
            PipeBarrier<PIPE_V>();
        }

        if constexpr (CACHE_HORIZONTAL_ROW) {
            if (cacheEnabled_) {
                const uint32_t count = t_->outputWidth * NC;
                if (cacheValid_) {
                    DataCopy(horizontal, cachedValuesBuffer_.Get<T>(), count);
                    DataCopy(horizontalIndices, cachedIndicesBuffer_.Get<int32_t>(), count);
                    PipeBarrier<PIPE_V>(); // Finish cache reads before overwriting it.
                }
                const int64_t lastEven = (inputRows_ - 1) / 2 * count;
                DataCopy(cachedValuesBuffer_.Get<T>(), horizontal[lastEven], count);
                DataCopy(cachedIndicesBuffer_.Get<int32_t>(), horizontalIndices[lastEven], count);
                PipeBarrier<PIPE_V>();
            }
        }

        // The saved masks and horizontal candidate values are both dead now.
        // Reuse them for vertical winners and candidate indices, respectively.
        auto maximum = maximumBuffer_.Get<T>();
        auto candidate = candidateBuffer_.Get<T>();
        auto indices = indexBuffer_.Get<int32_t>();
        auto candidateIndices = horizontalCandidate.template ReinterpretCast<int32_t>();
        auto mask = maskBuffer_.Get<uint8_t>();
        if constexpr (PARITY_ROWS) {
            // A vertical stride of two is now a contiguous interval within the
            // even or odd half. Preserve row-major ties by visiting kh in order.
            // The capacity guard includes the rounded vector tail of the last
            // odd-row interval, not just the valid elements of the last tile.
            DataCopy(maximum, horizontal, vectorCount_);
            DataCopy(indices, horizontalIndices, vectorCount_);
            PipeBarrier<PIPE_V>();
            for (int32_t kh = 1; kh < t_->kernelH; ++kh) {
                const int64_t sourceRow = kh / 2 + (kh % 2) * ((maxInputRows_ + 1) / 2);
                const int64_t offset = sourceRow * t_->outputWidth * NC;
                Adds(candidateIndices, horizontalIndices[offset], static_cast<int32_t>(kh * t_->inputWidth),
                     vectorCount_);
                Compare(mask, horizontal[offset], maximum, CMPMODE::LE, vectorCount_);
                PipeBarrier<PIPE_V>();
                Select(maximum, mask, maximum, horizontal[offset], SELMODE::VSEL_TENSOR_TENSOR_MODE, vectorCount_);
                Select(indices.ReinterpretCast<float>(), mask, indices.ReinterpretCast<float>(),
                       candidateIndices.template ReinterpretCast<float>(), SELMODE::VSEL_TENSOR_TENSOR_MODE,
                       vectorCount_);
                PipeBarrier<PIPE_V>();
            }
            return;
        }
        Duplicate(indices, INDEX_BIAS, vectorCount_);
        PipeBarrier<PIPE_V>();
        DataCopyParams valueRows;
        valueRows.blockCount = static_cast<uint16_t>(rows);
        valueRows.blockLen = static_cast<uint16_t>(t_->outputWidth * NC * sizeof(T) / 32);
        valueRows.srcStride = valueRows.blockLen; // strideH == 2
        valueRows.dstStride = 0;
        DataCopyParams indexRows = valueRows;
        indexRows.blockLen = static_cast<uint16_t>(t_->outputWidth * NC * sizeof(int32_t) / 32);
        indexRows.srcStride = indexRows.blockLen;
        for (int32_t kh = 0; kh < 7; ++kh) {
            auto target = kh == 0 ? maximum : candidate;
            auto targetIndices = kh == 0 ? indices : candidateIndices;
            DataCopy(target, horizontal[kh * t_->outputWidth * NC], valueRows);
            DataCopy(targetIndices, horizontalIndices[kh * t_->outputWidth * NC], indexRows);
            PipeBarrier<PIPE_V>();
            if (kh != 0) {
                Adds(candidateIndices, candidateIndices, static_cast<int32_t>(kh * t_->inputWidth), vectorCount_);
                Compare(mask, candidate, maximum, CMPMODE::LE, vectorCount_);
                PipeBarrier<PIPE_V>();
                Select(maximum, mask, maximum, candidate, SELMODE::VSEL_TENSOR_TENSOR_MODE, vectorCount_);
                Select(indices.ReinterpretCast<float>(), mask, indices.ReinterpretCast<float>(),
                       candidateIndices.template ReinterpretCast<float>(), SELMODE::VSEL_TENSOR_TENSOR_MODE,
                       vectorCount_);
                PipeBarrier<PIPE_V>();
            }
        }
    }

    __aicore__ inline void ComputeFlatSeparable(LocalTensor<T> padded, uint32_t count)
    {
        const uint32_t pitch = paddedWidth_ * NC;
        const uint32_t horizontalCount = Align(count + 2 * pitch, 64);
        const uint32_t maskBytes = Align(horizontalCount / 8, 32);
        auto horizontal = transposeBuffer_.Get<T>();
        auto saved = indexBuffer_.Get<uint8_t>();
        DataCopy(horizontal, padded, horizontalCount);
        PipeBarrier<PIPE_V>();
        for (uint32_t kw = 1; kw < 3; ++kw) {
            Compare(saved[(kw - 1) * maskBytes], padded[kw * NC], horizontal, CMPMODE::LE, horizontalCount);
            PipeBarrier<PIPE_V>();
            Select(horizontal, saved[(kw - 1) * maskBytes], horizontal, padded[kw * NC],
                   SELMODE::VSEL_TENSOR_TENSOR_MODE, horizontalCount);
            PipeBarrier<PIPE_V>();
        }
        // Input is dead only after both horizontal shifts have been consumed.
        // Replay the saved decisions into its allocation as exact column offsets.
        auto hi = rawBuffer_.Get<int32_t>();
        Duplicate(hi, INDEX_BIAS, horizontalCount);
        PipeBarrier<PIPE_V>();
        for (int32_t kw = 1; kw < 3; ++kw) {
            const int32_t encoded = INDEX_BIAS + kw;
            Select(hi.ReinterpretCast<float>(), saved[(kw - 1) * maskBytes], hi.ReinterpretCast<float>(),
                   *reinterpret_cast<const float*>(&encoded), SELMODE::VSEL_TENSOR_SCALAR_MODE, horizontalCount);
            PipeBarrier<PIPE_V>();
        }
        auto maximum = maximumBuffer_.Get<T>();
        auto indices = indexBuffer_.Get<int32_t>();
        auto mask = maskBuffer_.Get<uint8_t>();
        DataCopy(maximum, horizontal, count);
        DataCopy(indices, hi, count);
        PipeBarrier<PIPE_V>();
        for (int32_t kh = 1; kh < 3; ++kh) {
            const uint32_t offset = kh * pitch;
            Adds(hi[offset], hi[offset], int32_t(kh * t_->inputWidth), count);
            Compare(mask, horizontal[offset], maximum, CMPMODE::LE, count);
            PipeBarrier<PIPE_V>();
            Select(maximum, mask, maximum, horizontal[offset], SELMODE::VSEL_TENSOR_TENSOR_MODE, count);
            Select(indices.ReinterpretCast<float>(), mask, indices.ReinterpretCast<float>(),
                   hi[offset].ReinterpretCast<float>(), SELMODE::VSEL_TENSOR_TENSOR_MODE, count);
            PipeBarrier<PIPE_V>();
            if (kh == 1) {
                // The next shifted interval overlaps this one. Restore its
                // column offsets before applying the next row's displacement.
                Adds(hi[offset], hi[offset], -int32_t(kh * t_->inputWidth), count);
                PipeBarrier<PIPE_V>();
            }
        }
    }

    __aicore__ inline void Compute(int64_t group, int64_t row, int64_t rows)
    {
        auto transposed = transposeBuffer_.Get<T>();
        auto padded = padStatic_ ? transposed : (padAlias_ ? rawBuffer_.Get<T>() : paddedBuffer_.Get<T>());
        auto maximum = maximumBuffer_.Get<T>();
        // The flat window compares padded input directly; its candidate is unused.
        auto candidate = reuseFlatBuffers_ ? maximum : candidateBuffer_.Get<T>();
        auto indices = indexBuffer_.Get<int32_t>();
        if constexpr (HALF_INDICES) {
            if (separable_) {
                indices = transposed[horizontalCount_].template ReinterpretCast<int32_t>();
            }
        }
        auto base = indexTemplateBuffer_.Get<int32_t>();
        auto mask = maskBuffer_.Get<uint8_t>();
        // Process has already ordered the packed staging buffer's preceding
        // output DMA. rawReuseEvent_ separately protects the vector input reads;
        // output buffers retain MTE3_V ordering at their own reuse points.
        // With nothing to pad, the transposed input already has the layout the window
        // traversal expects. Building a separate padded copy then costs two full-plane
        // passes - an -inf fill and a row-by-row copy - which for a 2x2 window is about
        // as much work as the pooling itself.
        const bool padInPlace = padStatic_;
        if (padInPlace) {
            padded = transposed;
        } else if (!padAlias_) {
            Duplicate(padded, negativeInfinity_, (inputRows_ * paddedWidth_ + t_->kernelW + 16) * NC);
        }
        // The previous input layout conversion can still read raw or the packed
        // staging buffer, which the next MTE2 copy is about to overwrite.
        WaitFlag<HardEvent::V_MTE2>(rawReuseEvent_);
        if (packed_) {
            auto packedInput = packedBuffer_.Get<T>();
            DataCopy(packedInput, x_[group * NC * inputPlane_], static_cast<uint32_t>(inputPlane_ * NC));
            Sync<HardEvent::MTE2_V>();
            // Gather directly into [spatial, NC], rather than first gathering into
            // a padded [NC, spatial] tensor and transposing that tensor again.
            Gather(transposed, packedInput, inputOffsetsBuffer_.Get<uint32_t>(), 0,
                   static_cast<uint32_t>(inputSpace_ * NC));
            PipeBarrier<PIPE_V>();
        } else {
            auto raw = rawBuffer_.Get<T>();
            // A ceil tail may use only the cached final input row. Do not issue
            // zero-length DMA or transpose an input tile that was not loaded.
            if (!CACHE_HORIZONTAL_ROW || inputCount_ != 0) {
                for (int64_t nc = 0; nc < NC; ++nc) {
                    DataCopy(raw[nc * inputSpace_], x_[(group * NC + nc) * inputPlane_ + alignedInputStart_],
                             static_cast<uint32_t>(inputCount_));
                }
                Sync<HardEvent::MTE2_V>();
                Transpose<T>(transposed, raw, NC, inputSpace_);
                PipeBarrier<PIPE_V>();
            }
        }
        if (!padAlias_) {
            SetFlag<HardEvent::V_MTE2>(rawReuseEvent_);
        } else {
            // Transpose has finished reading raw; it can now hold the padded tensor.
            // Do not release it to MTE2 until the window loop finishes reading it.
            Duplicate(padded, negativeInfinity_, (inputRows_ * paddedWidth_ + t_->kernelW + 16) * NC);
            PipeBarrier<PIPE_V>();
        }
        if (!padInPlace) {
            int64_t copyWidth = paddedWidth_ - t_->padW;
            if (copyWidth > t_->inputWidth)
                copyWidth = t_->inputWidth;
            for (int64_t r = 0; r < validInputRows_; ++r) {
                DataCopy(padded[((r + topPadding_ + skippedInputRows_) * paddedWidth_ + t_->padW) * NC],
                         transposed[((packed_ ? 0 : inputPrefix_) + r * t_->inputWidth) * NC],
                         static_cast<uint32_t>(copyWidth * NC));
            }
            PipeBarrier<PIPE_V>();
        }
        DataCopyParams copy;
        copy.blockCount = t_->outputWidth;
        copy.blockLen = NC * sizeof(T) / 32;
        copy.srcStride = (t_->strideW - 1) * copy.blockLen;
        copy.dstStride = 0;
        // The mask a Compare writes is consumed by a Select on int32, which packs a
        // different number of lanes per repeat. The generic path rounds to 256;
        // the FP32-only separable path uses 64 lanes for both values and indices.
        // Extra lanes land in columns that output compaction drops.
        const int64_t poolCnt = flatWindow_ ? Align(rows * paddedWidth_ * NC, FLAT_SEPARABLE ? 64 : 256) : vectorCount_;
        int64_t firstKh = 0;
        if constexpr (FLAT_SEPARABLE) {
            ComputeFlatSeparable(padded, poolCnt);
            firstKh = t_->kernelH;
        }
        if constexpr (ENABLE_SEPARABLE) {
            if (separable_) {
                ComputeSeparable(padded, rows);
                firstKh = t_->kernelH;
            }
        }
        for (int64_t kh = firstKh; kh < t_->kernelH; ++kh) {
            for (int64_t kw = 0; kw < t_->kernelW; ++kw) {
                const bool first = kh == 0 && kw == 0;
                if (flatWindow_) {
                    // Unit stride makes the candidate for this window the padded buffer
                    // itself, shifted by a constant, with the same row pitch as the running
                    // maximum. One instruction then covers every output row, where the
                    // strided form needed one gather per row before it could compare.
                    const int64_t shift = (kh * paddedWidth_ + kw) * NC;
                    if (first) {
                        DataCopy(maximum, padded, static_cast<uint32_t>(poolCnt));
                        Duplicate(indices, INDEX_BIAS, poolCnt);
                        PipeBarrier<PIPE_V>();
                    } else {
                        Compare(mask, padded[shift], maximum, CMPMODE::LE, poolCnt);
                        PipeBarrier<PIPE_V>();
                        Select(maximum, mask, maximum, padded[shift], SELMODE::VSEL_TENSOR_TENSOR_MODE, poolCnt);
                        const int32_t windowOffset = INDEX_BIAS + static_cast<int32_t>(kh * t_->inputWidth + kw);
                        Select(indices.template ReinterpretCast<float>(), mask,
                               indices.template ReinterpretCast<float>(), *((float*)&windowOffset),
                               SELMODE::VSEL_TENSOR_SCALAR_MODE, poolCnt);
                        PipeBarrier<PIPE_V>();
                    }
                    continue;
                }
                auto target = first ? maximum : candidate;
                for (int64_t r = 0; r < rows; ++r) {
                    DataCopy(target[r * t_->outputWidth * NC],
                             padded[((r * t_->strideH + kh) * paddedWidth_ + kw) * NC], copy);
                }
                if (first) {
                    Duplicate(indices, INDEX_BIAS, vectorCount_);
                    PipeBarrier<PIPE_V>();
                } else {
                    PipeBarrier<PIPE_V>();
                    // cmp_le: the mask marks the lanes that keep the current winner, so the
                    // candidate's own offset is a plain scalar operand of the select and the
                    // per-window candidate index tensor disappears. LE also keeps the first
                    // maximum on ties, which is the semantics the task Golden requires.
                    Compare(mask, candidate, maximum, CMPMODE::LE, vectorCount_);
                    PipeBarrier<PIPE_V>();
                    Select(maximum, mask, maximum, candidate, SELMODE::VSEL_TENSOR_TENSOR_MODE, vectorCount_);
                    const int32_t windowOffset = INDEX_BIAS + static_cast<int32_t>(kh * t_->inputWidth + kw);
                    Select(indices.template ReinterpretCast<float>(), mask, indices.template ReinterpretCast<float>(),
                           *((float*)&windowOffset), SELMODE::VSEL_TENSOR_SCALAR_MODE, vectorCount_);
                    PipeBarrier<PIPE_V>();
                }
            }
        }
        // One Add per output block turns the carried window offset into the absolute
        // plane index: the template holds position - INDEX_BIAS and the running value
        // holds INDEX_BIAS + offset, so the bias cancels exactly. HALF_INDICES
        // instead expands an exact small offset into int32 and uses an unbiased
        // int32 template; no absolute plane coordinate is represented in half.
        if (padAlias_) {
            SetFlag<HardEvent::V_MTE2>(rawReuseEvent_);
        }
        Add(indices, indices, base, poolCnt);
        PipeBarrier<PIPE_V>();
        if (packed_) {
            // The offsets also remove the flat-window row padding, so neither
            // output compaction nor an output transpose is needed on this path.
            Sync<HardEvent::MTE3_V>();
            auto packedOutput = packedBuffer_.Get<T>();
            Gather(packedOutput, maximum, outputOffsetsBuffer_.Get<uint32_t>(), 0,
                   static_cast<uint32_t>(packedOutputs_));
            Sync<HardEvent::V_MTE3>();
            DataCopy(out_[group * NC * outputPlane_], packedOutput, static_cast<uint32_t>(outputPlane_ * NC));
            Sync<HardEvent::MTE3_V>();
            auto packedIndices = packedBuffer_.Get<int32_t>();
            Gather(packedIndices.ReinterpretCast<float>(), indices.template ReinterpretCast<float>(),
                   indexOffsetsBuffer_.Get<uint32_t>(), 0, static_cast<uint32_t>(packedOutputs_));
            Sync<HardEvent::V_MTE3>();
            DataCopy(indices_[group * NC * outputPlane_], packedIndices, static_cast<uint32_t>(outputPlane_ * NC));
            return;
        }
        // Drop the columns that only existed to keep the row pitch uniform.
        auto poolValues = maximum;
        if (flatWindow_) {
            // Input transposition and padding have completed before pooling.
            // Compaction can reuse that allocation; all its reads finish in the
            // output transpose before the next Compute uses it for input again.
            poolValues = compactAlias_ ? transposeBuffer_.Get<T>() : compactBuffer_.Get<T>();
            for (int64_t r = 0; r < rows; ++r) {
                DataCopy(poolValues[r * t_->outputWidth * NC], maximum[r * paddedWidth_ * NC],
                         static_cast<uint32_t>(t_->outputWidth * NC));
            }
            PipeBarrier<PIPE_V>();
        }
        Sync<HardEvent::MTE3_V>();
        auto transOutput = outputTransposeBuffer_.Get<T>();
        Transpose<T>(transOutput, poolValues, outputSpace_, NC);
        Sync<HardEvent::V_MTE3>();
        StoreSlices(out_, transOutput, group, row, rows);
        Sync<HardEvent::MTE3_V>();
        auto poolIndices = indices;
        if (flatWindow_) {
            poolIndices = compactAlias_ ? transposeBuffer_.Get<int32_t>() : compactBuffer_.Get<int32_t>();
            for (int64_t r = 0; r < rows; ++r) {
                DataCopy(poolIndices[r * t_->outputWidth * NC], indices[r * paddedWidth_ * NC],
                         static_cast<uint32_t>(t_->outputWidth * NC));
            }
            PipeBarrier<PIPE_V>();
        }
        auto transIndices = outputTransposeBuffer_.Get<int32_t>();
        Transpose<float>(transIndices.ReinterpretCast<float>(), poolIndices.template ReinterpretCast<float>(),
                         outputSpace_, NC);
        Sync<HardEvent::V_MTE3>();
        StoreSlices(indices_, transIndices, group, row, rows);
    }

    __aicore__ inline void ClearTailForTask(int64_t remainingTasks)
    {
        if (zeroNext_ >= zeroEnd_)
            return;
        const int64_t remainingBytes = zeroEnd_ - zeroNext_;
        int64_t bytesThisTask = Align((remainingBytes + remainingTasks - 1) / remainingTasks, 32);
        if (bytesThisTask > remainingBytes)
            bytesThisTask = remainingBytes;
        const int64_t taskEnd = zeroNext_ + bytesThisTask;
        const auto zeros = zeroBuffer_.Get<int32_t>();
        // The immutable zero buffer is disjoint from every pooling buffer. Issue
        // its stores before this task's vector work to overlap the two pipelines.
        while (zeroNext_ < taskEnd) {
            const uint32_t bytes = taskEnd - zeroNext_ < 8192 ? taskEnd - zeroNext_ : 8192;
            DataCopy(indices_[(t_->validIndexBytes + zeroNext_) / sizeof(int32_t)], zeros, bytes / sizeof(int32_t));
            zeroNext_ += bytes;
        }
    }

    // TransDataTo5HD layout adapted from the existing A2 implementation.
    template <typename U>
    __aicore__ inline void Transpose(LocalTensor<U> xTLocal, LocalTensor<U> xLocal, uint64_t rowNum, uint64_t colNum)
    {
        uint64_t blockAlign = 32 / sizeof(U);
        uint64_t int32DivUSize = sizeof(int32_t) / sizeof(U);
        uint64_t srcAddrList[16];
        uint64_t dstAddrList[16];
        struct TransDataTo5HDParams transDataParams;
        uint64_t colRepeateNum = colNum / blockAlign;
        uint64_t rowRepeateNum = rowNum / 16;
        if (colRepeateNum >= rowRepeateNum) {
            uint64_t inputOffset = 0;
            uint64_t outputOffset = 0;
            transDataParams.repeatTimes = colRepeateNum;
            if (transDataParams.repeatTimes != 1) {
                transDataParams.srcRepStride = 1;
                transDataParams.dstRepStride = rowNum;
            } else {
                transDataParams.srcRepStride = 0;
                transDataParams.dstRepStride = 0;
            }
            for (uint64_t loop = 0; loop < rowRepeateNum; loop++) {
                uint64_t inputOffsetTmp = inputOffset;
                uint64_t outputOffsetTmp = outputOffset;
                for (uint64_t i = 0; i < 16; i += 2) {
                    srcAddrList[i] = (uint64_t)(xLocal[inputOffsetTmp].GetPhyAddr());
                    dstAddrList[i] = (uint64_t)(xTLocal[outputOffsetTmp].GetPhyAddr());
                    inputOffsetTmp += 2 * colNum;
                    outputOffsetTmp += int32DivUSize * rowNum;
                }
                inputOffsetTmp = inputOffset + colNum;
                if (int32DivUSize == 2) {
                    outputOffsetTmp = outputOffset + rowNum;
                } else {
                    outputOffsetTmp = outputOffset + blockAlign;
                }
                for (uint64_t i = 1; i < 16; i += 2) {
                    srcAddrList[i] = (uint64_t)(xLocal[inputOffsetTmp].GetPhyAddr());
                    dstAddrList[i] = (uint64_t)(xTLocal[outputOffsetTmp].GetPhyAddr());
                    inputOffsetTmp += 2 * colNum;
                    outputOffsetTmp += int32DivUSize * rowNum;
                }
                TransDataTo5HD<U>(dstAddrList, srcAddrList, transDataParams);
                inputOffset += 16 * colNum;
                outputOffset += 16;
            }
        } else {
            uint64_t inputOffset = 0;
            uint64_t outputOffset = 0;
            transDataParams.repeatTimes = rowRepeateNum;
            if (transDataParams.repeatTimes != 1) {
                transDataParams.srcRepStride = 16 * colRepeateNum;
                transDataParams.dstRepStride = 16 / blockAlign;
            } else {
                transDataParams.srcRepStride = 0;
                transDataParams.dstRepStride = 0;
            }
            for (uint64_t loop = 0; loop < colRepeateNum; loop++) {
                uint64_t inputOffsetTmp = inputOffset;
                uint64_t outputOffsetTmp = outputOffset;
                for (uint64_t i = 0; i < 16; i += 2) {
                    srcAddrList[i] = (uint64_t)(xLocal[inputOffsetTmp].GetPhyAddr());
                    dstAddrList[i] = (uint64_t)(xTLocal[outputOffsetTmp].GetPhyAddr());
                    inputOffsetTmp += 2 * colNum;
                    outputOffsetTmp += int32DivUSize * rowNum;
                }
                inputOffsetTmp = inputOffset + colNum;
                if (int32DivUSize == 2) {
                    outputOffsetTmp = outputOffset + rowNum;
                } else {
                    outputOffsetTmp = outputOffset + blockAlign;
                }
                for (uint64_t i = 1; i < 16; i += 2) {
                    srcAddrList[i] = (uint64_t)(xLocal[inputOffsetTmp].GetPhyAddr());
                    dstAddrList[i] = (uint64_t)(xTLocal[outputOffsetTmp].GetPhyAddr());
                    inputOffsetTmp += 2 * colNum;
                    outputOffsetTmp += int32DivUSize * rowNum;
                }
                TransDataTo5HD<U>(dstAddrList, srcAddrList, transDataParams);
                inputOffset += blockAlign;
                outputOffset += blockAlign * rowNum;
            }
        }
    }
    int64_t zeroNext_ = 0;
    int64_t zeroEnd_ = 0;
    TPipe pipe_;
    TBuf<QuePosition::VECCALC> rawBuffer_;
    TBuf<QuePosition::VECCALC> transposeBuffer_;
    TBuf<QuePosition::VECCALC> paddedBuffer_;
    TBuf<QuePosition::VECCALC> maximumBuffer_;
    TBuf<QuePosition::VECCALC> candidateBuffer_;
    TBuf<QuePosition::VECCALC> indexBuffer_;
    TBuf<QuePosition::VECCALC> indexTemplateBuffer_;
    TBuf<QuePosition::VECCALC> outputTransposeBuffer_;
    TBuf<QuePosition::VECCALC> compactBuffer_;
    // Set when raw and the packed staging buffer become dead in one Compute, waited on
    // before the next Compute's input copy overwrites them. Splitting the set from the
    // wait is what keeps the load overlapped with this call's pooling; a plain Sync here
    // would stall MTE2 behind all of the vector work instead.
    event_t rawReuseEvent_;
    bool flatWindow_ = false;
    bool reuseFlatBuffers_ = false;
    bool compactAlias_ = false;
    bool padStatic_ = false;
    bool padAlias_ = false;
    bool separable_ = false;
    bool cacheEnabled_ = false;
    bool cacheValid_ = false;
    bool templateReady_ = false;
    int64_t skippedInputRows_ = 0;
    int64_t preparedRow_ = 0;
    TBuf<QuePosition::VECCALC> cachedValuesBuffer_;
    TBuf<QuePosition::VECCALC> cachedIndicesBuffer_;
    int64_t horizontalCount_ = 0;
    int64_t horizontalMaskBytes_ = 0;
    int64_t poolSpace_ = 0;
    int64_t poolCount_ = 0;
    TBuf<QuePosition::VECCALC> maskBuffer_;
    TBuf<QuePosition::VECCALC> zeroBuffer_;
    TBuf<QuePosition::VECCALC> packedBuffer_;
    TBuf<QuePosition::VECCALC> inputOffsetsBuffer_;
    TBuf<QuePosition::VECCALC> outputOffsetsBuffer_;
    TBuf<QuePosition::VECCALC> indexOffsetsBuffer_;
    GlobalTensor<T> x_;
    GlobalTensor<T> out_;
    GlobalTensor<int32_t> indices_;
    const MaxPool2dWithMaskTilingData* t_;
    T negativeInfinity_;
    int64_t firstTask_;
    int64_t lastTask_;
    int64_t groupCount_;
    int64_t maxInputRows_;
    int64_t inputSpace_;
    int64_t outputSpace_;
    int64_t paddedWidth_;
    int64_t firstInputRow_;
    int64_t topPadding_;
    int64_t inputRows_;
    int64_t validInputRows_;
    uint32_t vectorCount_;
    bool packed_;
    int64_t inputPlane_;
    int64_t outputPlane_;
    int64_t packedOutputs_;
    int64_t alignedInputStart_;
    int64_t inputPrefix_;
    int64_t inputCount_;
};
} // namespace MaxPool2dWithMask
#endif

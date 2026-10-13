/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef MAX_POOL2D_WITH_MASK_M200_SPATIAL_SEPARABLE_H
#define MAX_POOL2D_WITH_MASK_M200_SPATIAL_SEPARABLE_H

#include "kernel_operator.h"
#include "max_pool2d_with_mask_tiling_data.h"

namespace MaxPool2dWithMask {
using namespace AscendC;

// Even, aligned planes with 3x3/stride2/padding1. Keep NCHW spatial order:
// horizontal winners first, then vertical winners, then one contiguous store.
template <typename T, uint32_t STATIC_HEIGHT = 0, uint32_t STATIC_WIDTH = 0, uint32_t COUNT_ALIGNMENT = 256>
class M200SpatialSeparableKernel {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR out, GM_ADDR indices, const MaxPool2dWithMaskTilingData* tiling)
    {
        t_ = tiling;
        input_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x));
        output_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(out));
        indices_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(indices));
        horizontalCount_ = Align((2 * OutputHeight() + 1) * OutputWidth(), COUNT_ALIGNMENT);
        outputCount_ = Align(OutputHeight() * OutputWidth(), COUNT_ALIGNMENT);
        const uint32_t maskBytes = Align(HorizontalCount() / 8, 32);
        pipe_.InitBuffer(rawBuffer_, 2 * HorizontalCount() * sizeof(T));
        pipe_.InitBuffer(nextRawBuffer_, 2 * HorizontalCount() * sizeof(T));
        pipe_.InitBuffer(horizontalBuffer_, HorizontalCount() * sizeof(T));
        pipe_.InitBuffer(candidateBuffer_, HorizontalCount() * sizeof(T));
        pipe_.InitBuffer(horizontalIndexBuffer_, HorizontalCount() * sizeof(int32_t));
        pipe_.InitBuffer(offsetBuffer_, HorizontalCount() * sizeof(int32_t));
        pipe_.InitBuffer(leftMaskBuffer_, maskBytes);
        pipe_.InitBuffer(compareMaskBuffer_, maskBytes);
        pipe_.InitBuffer(templateBuffer_, OutputCount() * sizeof(int32_t));
        // The constant first-column index is shared across all slices. After
        // pooling finishes this allocation becomes the index-container zero pad.
        const uint32_t constantBytes = HorizontalCount() * sizeof(int32_t);
        pipe_.InitBuffer(zeroBuffer_, constantBytes > 8192 ? constantBytes : 8192);
        uint32_t bits = sizeof(T) == 2 ? 0xfc00U : 0xff800000U;
        negativeInfinity_ = *reinterpret_cast<T*>(&bits);
        auto offsets = offsetBuffer_.Get<int32_t>();
        for (int32_t i = 0; i < 32; ++i) {
            offsets.SetValue(i, (2 * i - 1) * sizeof(T));
        }
        auto leftMask = leftMaskBuffer_.Get<uint32_t>();
        for (int32_t word = 0; word < maskBytes / sizeof(uint32_t); ++word) {
            // The host admits outputWidth >= 32, so at most one row boundary
            // occurs in this word. Compute it once instead of testing 32 bits.
            const int32_t column = (word * 32) % OutputWidth();
            const int32_t firstBoundary = column == 0 ? 0 : OutputWidth() - column;
            uint32_t mask = 0xffffffffU;
            if (firstBoundary < 32) {
                mask &= ~(1U << firstBoundary);
            }
            leftMask.SetValue(word, mask);
        }
        auto positions = templateBuffer_.Get<int32_t>();
        for (int32_t c = 0; c < OutputWidth(); ++c) {
            positions.SetValue(c, -static_cast<int32_t>(InputWidth()) + 2 * c - 1 - INDEX_BIAS);
        }
        Sync<HardEvent::S_V>();
        for (int32_t i = 32; i < HorizontalCount(); i *= 2) {
            const uint32_t count = HorizontalCount() - i < i ? HorizontalCount() - i : i;
            Adds(offsets[i], offsets, static_cast<int32_t>(2 * i * sizeof(T)), count);
            // The next expansion consumes the newly generated prefix.
            PipeBarrier<PIPE_V>();
        }
        for (int32_t i = OutputWidth(); i < OutputCount(); i *= 2) {
            const uint32_t count = OutputCount() - i < i ? OutputCount() - i : i;
            Adds(positions[i], positions, static_cast<int32_t>(2 * (i / OutputWidth()) * InputWidth()), count);
            PipeBarrier<PIPE_V>();
        }
        Duplicate(candidateBuffer_.Get<T>(), negativeInfinity_, HorizontalCount());
        // A complete even-height plane needs one padded top row. Output and
        // scratch aliases stay inside the real-input interval, leaving both
        // this row and the rounded tail unchanged across all slice iterations.
        const uint32_t filled = (InputHeight() + 1) * InputWidth();
        for (int32_t slot = 0; slot < 2; ++slot) {
            auto raw = slot == 0 ? rawBuffer_.Get<T>() : nextRawBuffer_.Get<T>();
            Duplicate(raw, negativeInfinity_, InputWidth());
            if (2 * HorizontalCount() > filled) {
                Duplicate(raw[filled], negativeInfinity_, 2 * HorizontalCount() - filled);
            }
        }
        Duplicate(zeroBuffer_.Get<int32_t>(), INDEX_BIAS, HorizontalCount());
        Sync<HardEvent::V_S>();
        // The left column is masked out, but Gather must still use a valid address.
        offsets.SetValue(0, 0);
        Sync<HardEvent::S_V>();
        Sync<HardEvent::V_MTE2>();
        Sync<HardEvent::V_MTE3>();
        // Alloc reserves distinct events for the two simultaneously live slots.
        for (int32_t slot = 0; slot < 2; ++slot) {
            inputReuse_[slot] = static_cast<event_t>(pipe_.AllocEventID<HardEvent::MTE3_MTE2>());
            inputReady_[slot] = static_cast<event_t>(pipe_.AllocEventID<HardEvent::MTE2_V>());
            SetFlag<HardEvent::MTE3_MTE2>(inputReuse_[slot]);
        }
    }

    __aicore__ inline void Process()
    {
        const int64_t first = GetBlockIdx() * t_->slicesPerCore;
        const int64_t end = first + t_->slicesPerCore < t_->totalSlices ? first + t_->slicesPerCore : t_->totalSlices;
        int32_t slot = 0;
        if (first < end) {
            LoadInput(first, slot);
            for (int64_t slice = first; slice < end; ++slice) {
                WaitFlag<HardEvent::MTE2_V>(inputReady_[slot]);
                if (slice + 1 < end) {
                    LoadInput(slice + 1, 1 - slot);
                }
                Compute(slice, 0, OutputHeight(), slot);
                slot = 1 - slot;
            }
        }
        for (int32_t slot = 0; slot < 2; ++slot) {
            WaitFlag<HardEvent::MTE3_MTE2>(inputReuse_[slot]);
            pipe_.ReleaseEventID<HardEvent::MTE3_MTE2>(inputReuse_[slot]);
            pipe_.ReleaseEventID<HardEvent::MTE2_V>(inputReady_[slot]);
        }
        // All horizontal Select reads precede this write on V. MTE3 must not
        // copy the repurposed buffer until the zero fill has completed.
        Duplicate(zeroBuffer_.Get<int32_t>(), 0, 8192 / sizeof(int32_t));
        Sync<HardEvent::V_MTE3>();
        const int64_t tailBytes = t_->containerBytes - t_->validIndexBytes;
        int64_t start = GetBlockIdx() * t_->zeroBytesPerCore;
        const int64_t tailEnd = start + t_->zeroBytesPerCore < tailBytes ? start + t_->zeroBytesPerCore : tailBytes;
        for (; start < tailEnd; start += 8192) {
            const uint32_t bytes = tailEnd - start < 8192 ? tailEnd - start : 8192;
            DataCopy(indices_[(t_->validIndexBytes + start) / 4], zeroBuffer_.Get<int32_t>(), bytes / 4);
        }
        PipeBarrier<PIPE_ALL>();
    }

private:
    static constexpr int32_t INDEX_BIAS = 0x3f800000;
    static_assert((STATIC_HEIGHT == 0) == (STATIC_WIDTH == 0), "Both dimensions must be static or dynamic");
    static_assert(COUNT_ALIGNMENT == 256 || COUNT_ALIGNMENT == 64,
                  "Unsupported count alignment; host admits fine counts only for FP32");
    __aicore__ inline int64_t InputHeight() const
    {
        if constexpr (STATIC_HEIGHT != 0)
            return STATIC_HEIGHT;
        return t_->inputHeight;
    }
    __aicore__ inline int64_t InputWidth() const
    {
        if constexpr (STATIC_WIDTH != 0)
            return STATIC_WIDTH;
        return t_->inputWidth;
    }
    __aicore__ inline int64_t OutputHeight() const
    {
        if constexpr (STATIC_HEIGHT != 0)
            return STATIC_HEIGHT / 2;
        return t_->outputHeight;
    }
    __aicore__ inline int64_t OutputWidth() const
    {
        if constexpr (STATIC_WIDTH != 0)
            return STATIC_WIDTH / 2;
        return t_->outputWidth;
    }
    __aicore__ inline int64_t HorizontalCount() const
    {
        if constexpr (STATIC_HEIGHT != 0)
            return (((STATIC_HEIGHT + 1) * (STATIC_WIDTH / 2) + COUNT_ALIGNMENT - 1) / COUNT_ALIGNMENT) *
                   COUNT_ALIGNMENT;
        return horizontalCount_;
    }
    __aicore__ inline int64_t OutputCount() const
    {
        if constexpr (STATIC_HEIGHT != 0)
            return (((STATIC_HEIGHT / 2) * (STATIC_WIDTH / 2) + COUNT_ALIGNMENT - 1) / COUNT_ALIGNMENT) *
                   COUNT_ALIGNMENT;
        return outputCount_;
    }
    __aicore__ inline int64_t Align(int64_t n, int64_t a) { return (n + a - 1) / a * a; }
    template <HardEvent event>
    __aicore__ inline void Sync()
    {
        const event_t id = static_cast<event_t>(pipe_.FetchEventID(event));
        SetFlag<event>(id);
        WaitFlag<event>(id);
    }

    __aicore__ inline void LoadInput(int64_t slice, int32_t slot)
    {
        auto raw = slot == 0 ? rawBuffer_.Get<T>() : nextRawBuffer_.Get<T>();
        // Only MTE2 waits for this slot's preceding stores. No V padding fill
        // is needed here, so the other slot's current reduction can proceed.
        WaitFlag<HardEvent::MTE3_MTE2>(inputReuse_[slot]);
        DataCopy(raw[InputWidth()], input_[slice * InputHeight() * InputWidth()],
                 static_cast<uint32_t>(InputHeight() * InputWidth()));
        SetFlag<HardEvent::MTE2_V>(inputReady_[slot]);
    }

    __aicore__ inline void Compute(int64_t slice, int64_t row, int64_t rows, int32_t slot)
    {
        if constexpr (STATIC_HEIGHT != 0) {
            // This kernel always processes the complete plane in one tile.
            row = 0;
            rows = STATIC_HEIGHT / 2;
        }
        auto raw = slot == 0 ? rawBuffer_.Get<T>() : nextRawBuffer_.Get<T>();
        auto horizontal = horizontalBuffer_.Get<T>();
        auto candidate = candidateBuffer_.Get<T>();
        auto horizontalIndex = horizontalIndexBuffer_.Get<int32_t>();
        auto maximum = raw[InputWidth()];
        auto outputIndex = maximum[OutputCount()].template ReinterpretCast<int32_t>();
        auto mask = compareMaskBuffer_.Get<uint8_t>();
        const uint32_t horizontalCount = Align((2 * rows + 1) * OutputWidth(), COUNT_ALIGNMENT);
        const uint32_t outputCount = Align(rows * OutputWidth(), COUNT_ALIGNMENT);
        Gather(horizontal, raw, offsetBuffer_.Get<uint32_t>(), 0, horizontalCount);
        PipeBarrier<PIPE_V>();
        Select(horizontal, leftMaskBuffer_.Get<uint8_t>(), horizontal, negativeInfinity_,
               SELMODE::VSEL_TENSOR_SCALAR_MODE, horizontalCount);
        PipeBarrier<PIPE_V>();
        for (int32_t kw = 1; kw < 3; ++kw) {
            uint64_t reserved = 0;
            const uint8_t repeats = 2 * horizontalCount * sizeof(T) / 256;
            GatherMask(candidate, raw, static_cast<uint8_t>(kw), false, 0, {1, repeats, 8, 0}, reserved);
            PipeBarrier<PIPE_V>();
            Compare(mask, candidate, horizontal, CMPMODE::LE, horizontalCount);
            PipeBarrier<PIPE_V>();
            Select(horizontal, mask, horizontal, candidate, SELMODE::VSEL_TENSOR_TENSOR_MODE, horizontalCount);
            const int32_t encoded = INDEX_BIAS + kw;
            // kw=1 initializes every index lane from a persistent constant;
            // only kw=2 consumes this slice's previous horizontal indices.
            auto previousIndex = kw == 1 ? zeroBuffer_.Get<float>() : horizontalIndex.template ReinterpretCast<float>();
            Select(horizontalIndex.template ReinterpretCast<float>(), mask, previousIndex,
                   *reinterpret_cast<const float*>(&encoded), SELMODE::VSEL_TENSOR_SCALAR_MODE, horizontalCount);
            PipeBarrier<PIPE_V>();
        }
        // Every raw input read has completed. Reuse only its real-input
        // interval for maximum, index and candidate index, in disjoint ranges.
        // MTE2_V above also implies this slot's previous stores are complete.
        auto candidateIndex = outputIndex[OutputCount()];
        DataCopyParams valuesCopy;
        valuesCopy.blockCount = static_cast<uint16_t>(rows);
        valuesCopy.blockLen = static_cast<uint16_t>(OutputWidth() * sizeof(T) / 32);
        valuesCopy.srcStride = valuesCopy.blockLen;
        valuesCopy.dstStride = 0;
        DataCopyParams indicesCopy = valuesCopy;
        indicesCopy.blockLen = static_cast<uint16_t>(OutputWidth() * 4 / 32);
        indicesCopy.srcStride = indicesCopy.blockLen;
        for (int32_t kh = 0; kh < 3; ++kh) {
            if (kh == 0) {
                DataCopy(maximum, horizontal, valuesCopy);
                DataCopy(outputIndex, horizontalIndex, indicesCopy);
            } else {
                DataCopy(candidate, horizontal[kh * OutputWidth()], valuesCopy);
                DataCopy(candidateIndex, horizontalIndex[kh * OutputWidth()], indicesCopy);
                PipeBarrier<PIPE_V>();
                Adds(candidateIndex, candidateIndex, static_cast<int32_t>(kh * InputWidth()), outputCount);
                Compare(mask, candidate, maximum, CMPMODE::LE, outputCount);
                PipeBarrier<PIPE_V>();
                Select(maximum, mask, maximum, candidate, SELMODE::VSEL_TENSOR_TENSOR_MODE, outputCount);
                Select(outputIndex.template ReinterpretCast<float>(), mask,
                       outputIndex.template ReinterpretCast<float>(), candidateIndex.template ReinterpretCast<float>(),
                       SELMODE::VSEL_TENSOR_TENSOR_MODE, outputCount);
            }
            PipeBarrier<PIPE_V>();
        }
        Add(outputIndex, outputIndex, templateBuffer_.Get<int32_t>(), outputCount);
        Sync<HardEvent::V_MTE3>();
        const int64_t offset = slice * OutputHeight() * OutputWidth() + row * OutputWidth();
        DataCopy(output_[offset], maximum, static_cast<uint32_t>(rows * OutputWidth()));
        DataCopy(indices_[offset], outputIndex, static_cast<uint32_t>(rows * OutputWidth()));
        // The input slot now owns outstanding output transfers. Only MTE3 may
        // release it, after both stores, rather than V after its last read.
        SetFlag<HardEvent::MTE3_MTE2>(inputReuse_[slot]);
    }

    TPipe pipe_;
    TBuf<QuePosition::VECCALC> rawBuffer_;
    TBuf<QuePosition::VECCALC> horizontalBuffer_;
    TBuf<QuePosition::VECCALC> candidateBuffer_;
    TBuf<QuePosition::VECCALC> horizontalIndexBuffer_;
    TBuf<QuePosition::VECCALC> offsetBuffer_;
    TBuf<QuePosition::VECCALC> leftMaskBuffer_;
    TBuf<QuePosition::VECCALC> compareMaskBuffer_;
    TBuf<QuePosition::VECCALC> templateBuffer_;
    TBuf<QuePosition::VECCALC> zeroBuffer_;
    TBuf<QuePosition::VECCALC> nextRawBuffer_;
    GlobalTensor<T> input_;
    GlobalTensor<T> output_;
    GlobalTensor<int32_t> indices_;
    const MaxPool2dWithMaskTilingData* t_;
    int64_t horizontalCount_;
    int64_t outputCount_;
    T negativeInfinity_;
    event_t inputReuse_[2], inputReady_[2];
};
} // namespace MaxPool2dWithMask
#endif

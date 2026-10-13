/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef MAX_POOL2D_WITH_MASK_M200_SPATIAL_HALF_INPUT_H
#define MAX_POOL2D_WITH_MASK_M200_SPATIAL_HALF_INPUT_H

#include "kernel_operator.h"
#include "max_pool2d_with_mask_tiling_data.h"

namespace MaxPool2dWithMask {
using namespace AscendC;

// Fully covered aligned 2x2/s2 windows. Even and odd input rows have independent
// lifetimes: prefetch the next slice's half after its two column Gathers finish.
// The host admits only 224x224 FP32, one slice per Compute, and two 56-row tiles.
template <typename T>
class M200SpatialHalfInputKernel {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR out, GM_ADDR indices, const MaxPool2dWithMaskTilingData* tiling)
    {
        t_ = tiling;
        input_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x));
        output_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(out));
        indices_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(indices));
        capacity_ = (t_->m200SpatialRows * t_->outputWidth + 63) / 64 * 64;
        pipe_.InitBuffer(inputBuffer_, 4 * capacity_ * sizeof(T));
        pipe_.InitBuffer(maximumBuffer_, capacity_ * sizeof(T));
        pipe_.InitBuffer(nextMaximumBuffer_, capacity_ * sizeof(T));
        pipe_.InitBuffer(indexBuffer_, capacity_ * sizeof(int32_t));
        pipe_.InitBuffer(nextIndexBuffer_, capacity_ * sizeof(int32_t));
        pipe_.InitBuffer(candidateBuffer_, capacity_ * sizeof(T));
        pipe_.InitBuffer(templateBuffer_, capacity_ * sizeof(int32_t));
        pipe_.InitBuffer(maskBuffer_, (capacity_ / 8 + 31) / 32 * 32);

        auto positions = templateBuffer_.Get<int32_t>();
        const int32_t firstRow = GetBlockNum() % 2 == 0 ? (GetBlockIdx() % 2) * t_->m200SpatialRows : 0;
        for (int64_t c = 0; c < t_->outputWidth; ++c) {
            positions.SetValue(c, static_cast<int32_t>(2 * c) - INDEX_BIAS + 2 * firstRow * t_->inputWidth);
        }
        Sync<HardEvent::S_V>();
        for (int64_t offset = t_->outputWidth; offset < capacity_; offset *= 2) {
            const uint32_t count = capacity_ - offset < offset ? capacity_ - offset : offset;
            Adds(positions[offset], positions, static_cast<int32_t>(2 * (offset / t_->outputWidth) * t_->inputWidth),
                 count);
            PipeBarrier<PIPE_V>();
        }
        // The admitted two tiles contain exactly capacity_ real outputs. DMA
        // defines both complete input halves before any Gather reads them.
        for (uint32_t half = 0; half < 2; ++half) {
            inputReuse_[half] = static_cast<event_t>(pipe_.AllocEventID<HardEvent::V_MTE2>());
            inputReady_[half] = static_cast<event_t>(pipe_.AllocEventID<HardEvent::MTE2_V>());
            outputReuse_[half] = static_cast<event_t>(pipe_.AllocEventID<HardEvent::MTE3_V>());
            SetFlag<HardEvent::V_MTE2>(inputReuse_[half]);
            SetFlag<HardEvent::MTE3_V>(outputReuse_[half]);
        }
    }

    __aicore__ inline void Process()
    {
        const bool splitRows = GetBlockNum() % 2 == 0;
        const int64_t first = splitRows ? GetBlockIdx() / 2 : GetBlockIdx();
        const int64_t step = splitRows ? GetBlockNum() / 2 : GetBlockNum();
        const int64_t last = t_->totalSlices;
        for (int64_t row = splitRows ? (GetBlockIdx() % 2) * t_->m200SpatialRows : 0;
             row < t_->outputHeight && first < last; row += (splitRows ? 2 : 1) * t_->m200SpatialRows) {
            if (!splitRows && row != 0) {
                auto positions = templateBuffer_.Get<int32_t>();
                Adds(positions, positions, static_cast<int32_t>(2 * t_->m200SpatialRows * t_->inputWidth), capacity_);
                PipeBarrier<PIPE_V>();
            }
            LoadHalf(first, row, 0);
            LoadHalf(first, row, 1);
            bool second = false;
            for (int64_t slice = first; slice < last; slice += step) {
                Compute(slice, row, slice + step < last, second);
                second = !second;
            }
        }
        // Consume the final releases (or the initial releases on empty cores).
        for (uint32_t half = 0; half < 2; ++half) {
            WaitFlag<HardEvent::V_MTE2>(inputReuse_[half]);
            pipe_.ReleaseEventID<HardEvent::V_MTE2>(inputReuse_[half]);
            pipe_.ReleaseEventID<HardEvent::MTE2_V>(inputReady_[half]);
        }
        // Both input halves are dead after the final release events were
        // drained. Reuse their full allocation for longer zero-tail DMAs.
        auto zero = inputBuffer_.Get<int32_t>();
        const uint32_t zeroBytes = 4 * capacity_ * sizeof(T);
        Duplicate(zero, 0, zeroBytes / sizeof(int32_t));
        Sync<HardEvent::V_MTE3>();
        for (uint32_t half = 0; half < 2; ++half) {
            WaitFlag<HardEvent::MTE3_V>(outputReuse_[half]);
            pipe_.ReleaseEventID<HardEvent::MTE3_V>(outputReuse_[half]);
        }
        const int64_t tail = t_->containerBytes - t_->validIndexBytes;
        int64_t start = GetBlockIdx() * t_->zeroBytesPerCore;
        int64_t end = start + t_->zeroBytesPerCore;
        if (end > tail)
            end = tail;
        for (; start < end; start += zeroBytes) {
            const uint32_t bytes = end - start < zeroBytes ? end - start : zeroBytes;
            DataCopy(indices_[(t_->validIndexBytes + start) / sizeof(int32_t)], zero, bytes / sizeof(int32_t));
        }
        PipeBarrier<PIPE_ALL>();
    }

private:
    static constexpr int32_t INDEX_BIAS = 0x3F800000;
    template <HardEvent event>
    __aicore__ inline void Sync()
    {
        const event_t id = static_cast<event_t>(pipe_.FetchEventID(event));
        SetFlag<event>(id);
        WaitFlag<event>(id);
    }

    __aicore__ inline void LoadHalf(int64_t slice, int64_t row, uint32_t half)
    {
        WaitFlag<HardEvent::V_MTE2>(inputReuse_[half]);
        DataCopyParams copy;
        copy.blockCount = static_cast<uint16_t>(t_->m200SpatialRows);
        copy.blockLen = static_cast<uint16_t>(t_->inputWidth * sizeof(T) / 32);
        copy.srcStride = copy.blockLen;
        copy.dstStride = 0;
        const int64_t start = slice * t_->inputHeight * t_->inputWidth + (2 * row + half) * t_->inputWidth;
        DataCopy(inputBuffer_.Get<T>()[half * 2 * capacity_], input_[start], copy);
        SetFlag<HardEvent::MTE2_V>(inputReady_[half]);
    }

    __aicore__ inline void Compute(int64_t slice, int64_t row, bool prefetch, bool second)
    {
        auto maximum = second ? nextMaximumBuffer_.Get<T>() : maximumBuffer_.Get<T>();
        auto indices = second ? nextIndexBuffer_.Get<int32_t>() : indexBuffer_.Get<int32_t>();
        auto candidate = candidateBuffer_.Get<T>();
        auto mask = maskBuffer_.Get<uint8_t>();
        WaitFlag<HardEvent::MTE3_V>(outputReuse_[second ? 1 : 0]);
        const uint8_t repeats = 2 * capacity_ * sizeof(T) / 256;
        for (uint32_t kh = 0; kh < 2; ++kh) {
            WaitFlag<HardEvent::MTE2_V>(inputReady_[kh]);
            auto selectedRows = inputBuffer_.Get<T>()[kh * 2 * capacity_];
            for (int32_t kw = 0; kw < 2; ++kw) {
                const bool first = kh == 0 && kw == 0;
                auto target = first ? maximum : candidate;
                uint64_t reserved = 0;
                GatherMask(target, selectedRows, static_cast<uint8_t>(kw + 1), false, 0, {1, repeats, 8, 0}, reserved);
                PipeBarrier<PIPE_V>();
                if (kw == 1) {
                    // Both Gathers have consumed this half. Compare/Select
                    // below read independent UB tensors, not selectedRows.
                    SetFlag<HardEvent::V_MTE2>(inputReuse_[kh]);
                    if (prefetch)
                        LoadHalf(slice + (GetBlockNum() % 2 == 0 ? GetBlockNum() / 2 : GetBlockNum()), row, kh);
                }
                if (first) {
                    Duplicate(indices, INDEX_BIAS, capacity_);
                } else {
                    Compare(mask, candidate, maximum, CMPMODE::LE, capacity_);
                    PipeBarrier<PIPE_V>();
                    Select(maximum, mask, maximum, candidate, SELMODE::VSEL_TENSOR_TENSOR_MODE, capacity_);
                    const int32_t encoded = INDEX_BIAS + kh * t_->inputWidth + kw;
                    Select(indices.template ReinterpretCast<float>(), mask, indices.template ReinterpretCast<float>(),
                           *((float*)&encoded), SELMODE::VSEL_TENSOR_SCALAR_MODE, capacity_);
                }
                PipeBarrier<PIPE_V>();
            }
        }
        Add(indices, indices, templateBuffer_.Get<int32_t>(), capacity_);
        Sync<HardEvent::V_MTE3>();
        const int64_t offset = slice * t_->outputHeight * t_->outputWidth + row * t_->outputWidth;
        DataCopy(output_[offset], maximum, static_cast<uint32_t>(capacity_));
        DataCopy(indices_[offset], indices, static_cast<uint32_t>(capacity_));
        SetFlag<HardEvent::MTE3_V>(outputReuse_[second ? 1 : 0]);
    }

    TPipe pipe_;
    event_t inputReuse_[2], inputReady_[2], outputReuse_[2];
    TBuf<QuePosition::VECCALC> inputBuffer_;
    TBuf<QuePosition::VECCALC> maximumBuffer_;
    TBuf<QuePosition::VECCALC> nextMaximumBuffer_;
    TBuf<QuePosition::VECCALC> indexBuffer_;
    TBuf<QuePosition::VECCALC> nextIndexBuffer_;
    TBuf<QuePosition::VECCALC> candidateBuffer_;
    TBuf<QuePosition::VECCALC> templateBuffer_;
    TBuf<QuePosition::VECCALC> maskBuffer_;
    GlobalTensor<T> input_;
    GlobalTensor<T> output_;
    GlobalTensor<int32_t> indices_;
    const MaxPool2dWithMaskTilingData* t_;
    int64_t capacity_;
};
} // namespace MaxPool2dWithMask
#endif

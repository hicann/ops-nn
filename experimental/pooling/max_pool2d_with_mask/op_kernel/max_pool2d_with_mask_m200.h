/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MAX_POOL2D_WITH_MASK_M200_H
#define MAX_POOL2D_WITH_MASK_M200_H

#include "kernel_operator.h"
#include "max_pool2d_with_mask_tiling_data.h"

namespace MaxPool2dWithMask {
using namespace AscendC;

// Full-tile vector reduction. Clamped padding repeats an existing boundary
// element and preserves the first occurrence of every valid row-major index.
template <typename T, bool SPATIAL_TASKS = false>
class M200VectorKernel {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR out, GM_ADDR indices, const MaxPool2dWithMaskTilingData* tiling)
    {
        t_ = tiling;
        input_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x));
        output_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(out));
        indices_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(indices));
        firstSlice_ = GetBlockIdx() * t_->slicesPerCore;
        lastSlice_ = firstSlice_ + t_->slicesPerCore;
        if (lastSlice_ > t_->totalSlices)
            lastSlice_ = t_->totalSlices;
        capacity_ = Align(t_->m200MapRows * t_->outputWidth, 128);
        const int64_t inputRows = (t_->m200MapRows - 1) * t_->strideH + t_->kernelH;
        pipe_.InitBuffer(inputBuffer_, Align(inputRows * t_->inputWidth * sizeof(T), 32) + 64);
        // Extra initialized lanes make the full-width comparison safe on the final block.
        pipe_.InitBuffer(outputBuffer_, (capacity_ + 128) * sizeof(T));
        pipe_.InitBuffer(candidateBuffer_, capacity_ * sizeof(T));
        pipe_.InitBuffer(maskBuffer_, Align(capacity_ / 8, 32));
        pipe_.InitBuffer(indexBuffer_, capacity_ * sizeof(int32_t));
        pipe_.InitBuffer(mapBuffer_, capacity_ * t_->kernelH * t_->kernelW * sizeof(uint32_t) * 2);
        pipe_.InitBuffer(rowBuffer_, capacity_ * sizeof(float));
        pipe_.InitBuffer(columnBuffer_, capacity_ * sizeof(float));
        pipe_.InitBuffer(scratch_, capacity_ * sizeof(float) * 2);
        pipe_.InitBuffer(zeroBuffer_, 8192);
        // Relative coordinates are invariant across row tiles and channels.
        auto rowBase = rowBuffer_.Get<float>();
        auto columnBase = columnBuffer_.Get<float>();
        for (int64_t r = 0; r < t_->m200MapRows; ++r) {
            for (int64_t c = 0; c < t_->outputWidth; ++c) {
                rowBase.SetValue(r * t_->outputWidth + c, static_cast<float>(r * t_->strideH - t_->padH));
                columnBase.SetValue(r * t_->outputWidth + c, static_cast<float>(c * t_->strideW - t_->padW));
            }
        }
        Sync<HardEvent::S_V>();
    }

    __aicore__ inline void Process()
    {
        if constexpr (SPATIAL_TASKS) {
            const int64_t tiles = (t_->outputHeight + t_->m200MapRows - 1) / t_->m200MapRows;
            const int64_t tasks = tiles * t_->totalSlices;
            const int64_t first = tasks * GetBlockIdx() / GetBlockNum();
            const int64_t last = tasks * (GetBlockIdx() + 1) / GetBlockNum();
            int64_t preparedRow = -1;
            for (int64_t task = first; task < last; ++task) {
                const int64_t row = (task / t_->totalSlices) * t_->m200MapRows;
                const int64_t rows = row + t_->m200MapRows > t_->outputHeight ? t_->outputHeight - row :
                                                                                t_->m200MapRows;
                if (row != preparedRow) {
                    PrepareOffsets(row, rows);
                    preparedRow = row;
                }
                Compute(task % t_->totalSlices, row, rows);
            }
        } else if (firstSlice_ < lastSlice_) {
            for (int64_t row = 0; row < t_->outputHeight; row += t_->m200MapRows) {
                int64_t rows = t_->m200MapRows;
                if (row + rows > t_->outputHeight)
                    rows = t_->outputHeight - row;
                PrepareOffsets(row, rows);
                for (int64_t slice = firstSlice_; slice < lastSlice_; ++slice) {
                    Compute(slice, row, rows);
                }
            }
        }
        ClearTail();
    }

private:
    __aicore__ inline int64_t Align(int64_t value, int64_t alignment)
    {
        return (value + alignment - 1) / alignment * alignment;
    }

    template <HardEvent event>
    __aicore__ inline void Sync()
    {
        const event_t id = static_cast<event_t>(pipe_.FetchEventID(event));
        SetFlag<event>(id);
        WaitFlag<event>(id);
    }

    __aicore__ inline void PrepareOffsets(int64_t row, int64_t rows)
    {
        firstInputRow_ = row * t_->strideH - t_->padH;
        if (firstInputRow_ < 0)
            firstInputRow_ = 0;
        int64_t lastInputRow = (row + rows - 1) * t_->strideH - t_->padH + t_->kernelH;
        if (lastInputRow > t_->inputHeight)
            lastInputRow = t_->inputHeight;
        alignedInputStart_ = firstInputRow_ * t_->inputWidth / (32 / sizeof(T)) * (32 / sizeof(T));
        inputCount_ = Align(lastInputRow * t_->inputWidth, 32 / sizeof(T)) - alignedInputStart_;
        const int64_t count = rows * t_->outputWidth;
        auto rowBase = rowBuffer_.Get<float>();
        auto columnBase = columnBuffer_.Get<float>();
        auto rowTemp = scratch_.Get<float>();
        auto colTemp = rowTemp[capacity_];
        auto maps = mapBuffer_.Get<int32_t>();
        Duplicate(maps, 0, static_cast<uint32_t>(capacity_ * t_->kernelH * t_->kernelW * 2));
        PipeBarrier<PIPE_V>();
        for (int64_t kh = 0; kh < t_->kernelH; ++kh) {
            for (int64_t kw = 0; kw < t_->kernelW; ++kw) {
                const int64_t window = kh * t_->kernelW + kw;
                const uint32_t lanes = count;
                Adds(rowTemp, rowBase, static_cast<float>(row * t_->strideH + kh), lanes);
                Adds(colTemp, columnBase, static_cast<float>(kw), lanes);
                PipeBarrier<PIPE_V>();
                Maxs(rowTemp, rowTemp, 0.0f, lanes);
                Maxs(colTemp, colTemp, 0.0f, lanes);
                PipeBarrier<PIPE_V>();
                Mins(rowTemp, rowTemp, static_cast<float>(t_->inputHeight - 1), lanes);
                Mins(colTemp, colTemp, static_cast<float>(t_->inputWidth - 1), lanes);
                PipeBarrier<PIPE_V>();
                Muls(rowTemp, rowTemp, static_cast<float>(t_->inputWidth), lanes);
                PipeBarrier<PIPE_V>();
                Add(rowTemp, rowTemp, colTemp, lanes);
                PipeBarrier<PIPE_V>();
                Cast(maps[(t_->kernelH * t_->kernelW + window) * capacity_], rowTemp, RoundMode::CAST_RINT, lanes);
                PipeBarrier<PIPE_V>();
                Adds(rowTemp, rowTemp, -static_cast<float>(alignedInputStart_), lanes);
                PipeBarrier<PIPE_V>();
                Muls(rowTemp, rowTemp, (sizeof(T) == 2 ? 2.0f : 4.0f), lanes);
                PipeBarrier<PIPE_V>();
                Cast(maps[window * capacity_], rowTemp, RoundMode::CAST_RINT, lanes);
                PipeBarrier<PIPE_V>();
            }
        }
    }

    __aicore__ inline void Compute(int64_t slice, int64_t row, int64_t rows)
    {
        auto input = inputBuffer_.Get<T>();
        auto output = outputBuffer_.Get<T>();
        auto indices = indexBuffer_.Get<int32_t>();
        auto maps = mapBuffer_.Get<uint32_t>();
        auto candidate = candidateBuffer_.Get<T>();
        auto mask = maskBuffer_.Get<uint8_t>();
        const int64_t count = rows * t_->outputWidth;
        const uint32_t vectorCount = Align(count, 128);
        DataCopy(input, input_[slice * t_->inputHeight * t_->inputWidth + alignedInputStart_],
                 static_cast<uint32_t>(inputCount_));
        Sync<HardEvent::MTE2_V>();
        // Each instruction handles a complete tile. Window-major traversal avoids
        // repeating scalar dispatch and pipeline barriers for every 64 outputs.
        for (int64_t window = 0; window < t_->kernelH * t_->kernelW; ++window) {
            auto offsets = maps[window * capacity_];
            auto positions = maps[(t_->kernelH * t_->kernelW + window) * capacity_].template ReinterpretCast<int32_t>();
            auto target = window == 0 ? output : candidate;
            Gather(target, input, offsets, 0, vectorCount);
            if (window == 0) {
                Adds(indices, positions, 0, vectorCount);
            }
            PipeBarrier<PIPE_V>();
            if (window != 0) {
                Compare(mask, candidate, output, CMPMODE::GT, vectorCount);
                PipeBarrier<PIPE_V>();
                Select(output, mask, candidate, output, SELMODE::VSEL_TENSOR_TENSOR_MODE, vectorCount);
                Select(indices.template ReinterpretCast<float>(), mask, positions.template ReinterpretCast<float>(),
                       indices.template ReinterpretCast<float>(), SELMODE::VSEL_TENSOR_TENSOR_MODE, vectorCount);
                PipeBarrier<PIPE_V>();
            }
        }
        Sync<HardEvent::V_MTE3>();
        const int64_t offset = slice * t_->outputHeight * t_->outputWidth + row * t_->outputWidth;
        DataCopy(output_[offset], output, static_cast<uint32_t>(count));
        DataCopy(indices_[offset], indices, static_cast<uint32_t>(count));
        // The next channel reuses all input/output buffers.
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void ClearTail()
    {
        const int64_t tail = t_->containerBytes - t_->validIndexBytes;
        int64_t start = GetBlockIdx() * t_->zeroBytesPerCore;
        int64_t end = start + t_->zeroBytesPerCore;
        if (end > tail)
            end = tail;
        if (start >= end)
            return;
        auto zeros = zeroBuffer_.Get<int32_t>();
        Duplicate(zeros, 0, 8192 / sizeof(int32_t));
        Sync<HardEvent::V_MTE3>();
        for (; start < end; start += 8192) {
            const uint32_t bytes = end - start < 8192 ? end - start : 8192;
            DataCopy(indices_[(t_->validIndexBytes + start) / sizeof(int32_t)], zeros, bytes / sizeof(int32_t));
        }
        PipeBarrier<PIPE_ALL>();
    }

    TPipe pipe_;
    TBuf<QuePosition::VECCALC> inputBuffer_;
    TBuf<QuePosition::VECCALC> outputBuffer_;
    TBuf<QuePosition::VECCALC> indexBuffer_;
    TBuf<QuePosition::VECCALC> mapBuffer_;
    TBuf<QuePosition::VECCALC> candidateBuffer_;
    TBuf<QuePosition::VECCALC> maskBuffer_;
    TBuf<QuePosition::VECCALC> rowBuffer_;
    TBuf<QuePosition::VECCALC> columnBuffer_;
    TBuf<QuePosition::VECCALC> scratch_;
    TBuf<QuePosition::VECCALC> zeroBuffer_;
    GlobalTensor<T> input_;
    GlobalTensor<T> output_;
    GlobalTensor<int32_t> indices_;
    const MaxPool2dWithMaskTilingData* t_;
    int64_t firstSlice_;
    int64_t lastSlice_;
    int64_t capacity_;
    int64_t firstInputRow_;
    int64_t alignedInputStart_;
    int64_t inputCount_;
};
} // namespace MaxPool2dWithMask
#endif

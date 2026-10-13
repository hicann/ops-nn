/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef MAX_POOL2D_WITH_MASK_M200_SPATIAL_H
#define MAX_POOL2D_WITH_MASK_M200_SPATIAL_H

#include "kernel_operator.h"
#include "max_pool2d_with_mask_tiling_data.h"

namespace MaxPool2dWithMask {
using namespace AscendC;

// Aligned, fully covered 2x2/stride2 windows. Select rows by strided UB copy
// and columns by fixed GatherMask patterns, without per-element address maps.
template <typename T, bool INPUT_PIPELINE = false, bool OUTPUT_PIPELINE = false, bool ROWS_ON_LOAD = false>
class M200SpatialKernel {
    static_assert(!ROWS_ON_LOAD || (INPUT_PIPELINE && OUTPUT_PIPELINE),
                  "The strided-load variant requires both pipeline slot pairs");

public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR out, GM_ADDR indices, const MaxPool2dWithMaskTilingData* tiling)
    {
        t_ = tiling;
        input_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x));
        output_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(out));
        indices_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(indices));
        capacity_ = Align(t_->m200SpatialRows * t_->outputWidth * t_->m200SpatialBatch, 256);
        const uint32_t inputBytes = ROWS_ON_LOAD ?
                                        4 * capacity_ * sizeof(T) :
                                        2 * t_->m200SpatialRows * t_->inputWidth * t_->m200SpatialBatch * sizeof(T);
        pipe_.InitBuffer(inputBuffer_, inputBytes);
        if constexpr (INPUT_PIPELINE) {
            pipe_.InitBuffer(nextInputBuffer_, inputBytes);
        }
        if constexpr (!ROWS_ON_LOAD) {
            pipe_.InitBuffer(rowsBuffer_, 2 * capacity_ * sizeof(T));
        }
        pipe_.InitBuffer(maximumBuffer_, capacity_ * sizeof(T));
        pipe_.InitBuffer(candidateBuffer_, capacity_ * sizeof(T));
        pipe_.InitBuffer(indexBuffer_, capacity_ * sizeof(int32_t));
        if constexpr (OUTPUT_PIPELINE) {
            pipe_.InitBuffer(nextMaximumBuffer_, capacity_ * sizeof(T));
            pipe_.InitBuffer(nextIndexBuffer_, capacity_ * sizeof(int32_t));
        }
        pipe_.InitBuffer(templateBuffer_, capacity_ * sizeof(int32_t));
        pipe_.InitBuffer(maskBuffer_, Align(capacity_ / 8, 32));
        pipe_.InitBuffer(zeroBuffer_, 8192);
        auto positions = templateBuffer_.Get<int32_t>();
        // Eligible widths give four or eight int32 positions per 32-byte block.
        // Seed one aligned row group, then translate it with vector Adds instead
        // of performing division, remainder, and scalar stores for every tile lane.
        const int64_t templateRows = t_->outputWidth % 8 == 0 ? 1 : 2;
        const int64_t templateElements = templateRows * t_->outputWidth;
        if constexpr (ROWS_ON_LOAD) {
            for (int64_t r = 0; r < templateRows; ++r) {
                for (int64_t c = 0; c < t_->outputWidth; ++c) {
                    positions.SetValue(r * t_->outputWidth + c,
                                       static_cast<int32_t>(2 * r * t_->inputWidth + 2 * c) - INDEX_BIAS);
                }
            }
        } else {
            for (int64_t i = 0; i < templateElements; ++i) {
                const int64_t r = i / t_->outputWidth;
                const int64_t c = i % t_->outputWidth;
                positions.SetValue(i, static_cast<int32_t>(2 * r * t_->inputWidth + 2 * c) - INDEX_BIAS);
            }
        }
        Sync<HardEvent::S_V>();
        if constexpr (ROWS_ON_LOAD) {
            // This path admits one NC plane per Compute. Expand an aligned
            // prefix instead of issuing one Adds per output row. Aligned tail
            // positions remain defined but are never copied to GM.
            for (int64_t i = templateElements; i < capacity_; i *= 2) {
                const uint32_t count = capacity_ - i < i ? capacity_ - i : i;
                Adds(positions[i], positions, static_cast<int32_t>(2 * (i / t_->outputWidth) * t_->inputWidth), count);
                PipeBarrier<PIPE_V>();
            }
        } else {
            for (int64_t i = templateElements; i < capacity_; i += templateElements) {
                const uint32_t count = capacity_ - i < templateElements ? capacity_ - i : templateElements;
                // Argmax is relative to its own input plane, not to the NC batch.
                const int64_t r = (i / t_->outputWidth) % t_->outputHeight;
                Adds(positions[i], positions, static_cast<int32_t>(2 * r * t_->inputWidth), count);
            }
        }
        if constexpr (ROWS_ON_LOAD) {
            // DMA defines each real even/odd row prefix before Gather. Only
            // the rounded tails need an initial value. Later short row tiles
            // reuse a slot already loaded by the first full-sized row tile.
            const uint32_t valid = 2 * t_->m200SpatialRows * t_->outputWidth * t_->m200SpatialBatch;
            const uint32_t padding = 2 * capacity_ - valid;
            if (padding != 0) {
                Duplicate(inputBuffer_.Get<T>()[valid], static_cast<T>(0), padding);
                Duplicate(inputBuffer_.Get<T>()[2 * capacity_ + valid], static_cast<T>(0), padding);
                if constexpr (INPUT_PIPELINE) {
                    Duplicate(nextInputBuffer_.Get<T>()[valid], static_cast<T>(0), padding);
                    Duplicate(nextInputBuffer_.Get<T>()[2 * capacity_ + valid], static_cast<T>(0), padding);
                }
            }
        } else {
            Duplicate(rowsBuffer_.Get<T>(), static_cast<T>(0), 2 * capacity_);
        }
        Duplicate(zeroBuffer_.Get<int32_t>(), 0, 8192 / sizeof(int32_t));
        Sync<HardEvent::V_MTE3>();
        if constexpr (INPUT_PIPELINE) {
            inputReuseEvent_ = static_cast<event_t>(pipe_.AllocEventID<HardEvent::V_MTE2>());
        } else {
            inputReuseEvent_ = static_cast<event_t>(pipe_.FetchEventID(HardEvent::V_MTE2));
        }
        SetFlag<HardEvent::V_MTE2>(inputReuseEvent_);
        if constexpr (INPUT_PIPELINE) {
            nextInputReuseEvent_ = static_cast<event_t>(pipe_.AllocEventID<HardEvent::V_MTE2>());
            SetFlag<HardEvent::V_MTE2>(nextInputReuseEvent_);
            inputReadyEvent_ = static_cast<event_t>(pipe_.AllocEventID<HardEvent::MTE2_V>());
            nextInputReadyEvent_ = static_cast<event_t>(pipe_.AllocEventID<HardEvent::MTE2_V>());
        }
        if constexpr (OUTPUT_PIPELINE) {
            outputReuseEvent_ = static_cast<event_t>(pipe_.AllocEventID<HardEvent::MTE3_V>());
            nextOutputReuseEvent_ = static_cast<event_t>(pipe_.AllocEventID<HardEvent::MTE3_V>());
            SetFlag<HardEvent::MTE3_V>(outputReuseEvent_);
            SetFlag<HardEvent::MTE3_V>(nextOutputReuseEvent_);
        }
    }

    __aicore__ inline void Process()
    {
        const int64_t first = GetBlockIdx() * t_->slicesPerCore;
        int64_t last = first + t_->slicesPerCore;
        if (last > t_->totalSlices)
            last = t_->totalSlices;
        for (int64_t row = 0; row < t_->outputHeight && first < last; row += t_->m200SpatialRows) {
            int64_t rows = t_->m200SpatialRows;
            if (row + rows > t_->outputHeight)
                rows = t_->outputHeight - row;
            if (row != 0) {
                auto positions = templateBuffer_.Get<int32_t>();
                Adds(positions, positions, static_cast<int32_t>(2 * t_->m200SpatialRows * t_->inputWidth), capacity_);
                PipeBarrier<PIPE_V>();
            }
            bool second = false;
            if constexpr (INPUT_PIPELINE) {
                const int64_t slices = last - first < t_->m200SpatialBatch ? last - first : t_->m200SpatialBatch;
                LoadInput(first, row, rows, slices, false);
            }
            for (int64_t slice = first; slice < last; slice += t_->m200SpatialBatch) {
                const int64_t slices = last - slice < t_->m200SpatialBatch ? last - slice : t_->m200SpatialBatch;
                if constexpr (INPUT_PIPELINE) {
                    WaitFlag<HardEvent::MTE2_V>(second ? nextInputReadyEvent_ : inputReadyEvent_);
                    const int64_t next = slice + t_->m200SpatialBatch;
                    if (next < last) {
                        const int64_t nextSlices = last - next < t_->m200SpatialBatch ? last - next :
                                                                                        t_->m200SpatialBatch;
                        LoadInput(next, row, rows, nextSlices, !second);
                    }
                }
                Compute(slice, row, rows, slices, second);
                if constexpr (INPUT_PIPELINE) {
                    second = !second;
                }
            }
        }
        // Drain the initial or final release, including cores with no input work.
        WaitFlag<HardEvent::V_MTE2>(inputReuseEvent_);
        if constexpr (INPUT_PIPELINE) {
            WaitFlag<HardEvent::V_MTE2>(nextInputReuseEvent_);
            pipe_.ReleaseEventID<HardEvent::V_MTE2>(inputReuseEvent_);
            pipe_.ReleaseEventID<HardEvent::V_MTE2>(nextInputReuseEvent_);
            pipe_.ReleaseEventID<HardEvent::MTE2_V>(inputReadyEvent_);
            pipe_.ReleaseEventID<HardEvent::MTE2_V>(nextInputReadyEvent_);
        }
        if constexpr (OUTPUT_PIPELINE) {
            WaitFlag<HardEvent::MTE3_V>(outputReuseEvent_);
            WaitFlag<HardEvent::MTE3_V>(nextOutputReuseEvent_);
            pipe_.ReleaseEventID<HardEvent::MTE3_V>(outputReuseEvent_);
            pipe_.ReleaseEventID<HardEvent::MTE3_V>(nextOutputReuseEvent_);
        }
        const int64_t tail = t_->containerBytes - t_->validIndexBytes;
        int64_t start = GetBlockIdx() * t_->zeroBytesPerCore;
        int64_t end = start + t_->zeroBytesPerCore;
        if (end > tail)
            end = tail;
        for (; start < end; start += 8192) {
            const uint32_t bytes = end - start < 8192 ? end - start : 8192;
            DataCopy(indices_[(t_->validIndexBytes + start) / sizeof(int32_t)], zeroBuffer_.Get<int32_t>(),
                     bytes / sizeof(int32_t));
        }
        PipeBarrier<PIPE_ALL>();
    }

private:
    static constexpr int32_t INDEX_BIAS = 0x3F800000;
    __aicore__ inline int64_t Align(int64_t value, int64_t multiple)
    {
        return (value + multiple - 1) / multiple * multiple;
    }
    template <HardEvent event>
    __aicore__ inline void Sync()
    {
        const event_t id = static_cast<event_t>(pipe_.FetchEventID(event));
        SetFlag<event>(id);
        WaitFlag<event>(id);
    }
    __aicore__ inline void LoadInput(int64_t slice, int64_t row, int64_t rows, int64_t slices, bool second)
    {
        WaitFlag<HardEvent::V_MTE2>(second ? nextInputReuseEvent_ : inputReuseEvent_);
        auto input = second ? nextInputBuffer_.Get<T>() : inputBuffer_.Get<T>();
        const int64_t start = slice * t_->inputHeight * t_->inputWidth + 2 * row * t_->inputWidth;
        if constexpr (ROWS_ON_LOAD) {
            DataCopyParams copy;
            copy.blockCount = static_cast<uint16_t>(rows * slices);
            copy.blockLen = static_cast<uint16_t>(t_->inputWidth * sizeof(T) / 32);
            copy.srcStride = copy.blockLen;
            copy.dstStride = 0;
            DataCopy(input, input_[start], copy);
            DataCopy(input[2 * capacity_], input_[start + t_->inputWidth], copy);
        } else {
            DataCopy(input, input_[start], static_cast<uint32_t>(2 * rows * t_->inputWidth * slices));
        }
        SetFlag<HardEvent::MTE2_V>(second ? nextInputReadyEvent_ : inputReadyEvent_);
    }
    __aicore__ inline void Compute(int64_t slice, int64_t row, int64_t rows, int64_t slices, bool second)
    {
        auto input = inputBuffer_.Get<T>();
        if constexpr (INPUT_PIPELINE) {
            if (second) {
                input = nextInputBuffer_.Get<T>();
            }
        }
        LocalTensor<T> selectedRows;
        if constexpr (!ROWS_ON_LOAD) {
            selectedRows = rowsBuffer_.Get<T>();
        }
        auto maximum = maximumBuffer_.Get<T>();
        auto candidate = candidateBuffer_.Get<T>();
        auto indices = indexBuffer_.Get<int32_t>();
        if constexpr (OUTPUT_PIPELINE) {
            if (second) {
                maximum = nextMaximumBuffer_.Get<T>();
                indices = nextIndexBuffer_.Get<int32_t>();
            }
        }
        auto mask = maskBuffer_.Get<uint8_t>();
        // The preceding slice may still copy rows from input on the vector pipe.
        if constexpr (!INPUT_PIPELINE) {
            WaitFlag<HardEvent::V_MTE2>(inputReuseEvent_);
            DataCopy(input, input_[slice * t_->inputHeight * t_->inputWidth + 2 * row * t_->inputWidth],
                     static_cast<uint32_t>(2 * rows * t_->inputWidth * slices));
            Sync<HardEvent::MTE2_V>();
        }
        // MTE3 reads maximum and indices from the previous slice. Protect those
        // outputs before overwriting them, while allowing input prefetch above.
        if constexpr (OUTPUT_PIPELINE) {
            // Only wait for this output slot, used two Compute calls ago. The
            // immediately preceding slice can still write from its other slot.
            WaitFlag<HardEvent::MTE3_V>(second ? nextOutputReuseEvent_ : outputReuseEvent_);
        } else {
            Sync<HardEvent::MTE3_V>();
        }
        DataCopyParams copy;
        copy.blockCount = static_cast<uint16_t>(rows * slices);
        copy.blockLen = static_cast<uint16_t>(t_->inputWidth * sizeof(T) / 32);
        copy.srcStride = copy.blockLen;
        copy.dstStride = 0;
        const uint32_t count = Align(rows * t_->outputWidth * slices, 256);
        const uint8_t repeats = 2 * count * sizeof(T) / 256;
        for (int32_t kh = 0; kh < 2; ++kh) {
            if constexpr (ROWS_ON_LOAD) {
                selectedRows = input[kh * 2 * capacity_];
            } else {
                DataCopy(selectedRows, input[kh * t_->inputWidth], copy);
                PipeBarrier<PIPE_V>();
            }
            if (!ROWS_ON_LOAD && kh == 1) {
                // This is the last input read. The remaining GatherMask/Compare/
                // Select work uses selectedRows, so the next load can reuse input.
                if constexpr (INPUT_PIPELINE) {
                    SetFlag<HardEvent::V_MTE2>(second ? nextInputReuseEvent_ : inputReuseEvent_);
                } else {
                    SetFlag<HardEvent::V_MTE2>(inputReuseEvent_);
                }
            }
            for (int32_t kw = 0; kw < 2; ++kw) {
                const bool first = kh == 0 && kw == 0;
                auto target = first ? maximum : candidate;
                uint64_t reserved = 0;
                GatherMask(target, selectedRows, static_cast<uint8_t>(kw + 1), false, 0, {1, repeats, 8, 0}, reserved);
                PipeBarrier<PIPE_V>();
                if constexpr (ROWS_ON_LOAD) {
                    // Gather now reads the DMA slot itself, so release only
                    // after the final Gather, not after selecting the last row.
                    if (kh == 1 && kw == 1) {
                        SetFlag<HardEvent::V_MTE2>(second ? nextInputReuseEvent_ : inputReuseEvent_);
                    }
                }
                if (first) {
                    Duplicate(indices, INDEX_BIAS, count);
                } else {
                    Compare(mask, candidate, maximum, CMPMODE::LE, count);
                    PipeBarrier<PIPE_V>();
                    Select(maximum, mask, maximum, candidate, SELMODE::VSEL_TENSOR_TENSOR_MODE, count);
                    const int32_t encoded = INDEX_BIAS + kh * t_->inputWidth + kw;
                    Select(indices.template ReinterpretCast<float>(), mask, indices.template ReinterpretCast<float>(),
                           *((float*)&encoded), SELMODE::VSEL_TENSOR_SCALAR_MODE, count);
                }
                PipeBarrier<PIPE_V>();
            }
        }
        Add(indices, indices, templateBuffer_.Get<int32_t>(), count);
        Sync<HardEvent::V_MTE3>();
        const int64_t offset = slice * t_->outputHeight * t_->outputWidth + row * t_->outputWidth;
        DataCopy(output_[offset], maximum, static_cast<uint32_t>(rows * t_->outputWidth * slices));
        DataCopy(indices_[offset], indices, static_cast<uint32_t>(rows * t_->outputWidth * slices));
        if constexpr (OUTPUT_PIPELINE) {
            SetFlag<HardEvent::MTE3_V>(second ? nextOutputReuseEvent_ : outputReuseEvent_);
        }
    }
    TPipe pipe_;
    event_t inputReuseEvent_;
    event_t nextInputReuseEvent_;
    event_t inputReadyEvent_;
    event_t nextInputReadyEvent_;
    event_t outputReuseEvent_;
    event_t nextOutputReuseEvent_;
    TBuf<QuePosition::VECCALC> nextInputBuffer_;
    TBuf<QuePosition::VECCALC> nextMaximumBuffer_;
    TBuf<QuePosition::VECCALC> nextIndexBuffer_;
    TBuf<QuePosition::VECCALC> inputBuffer_;
    TBuf<QuePosition::VECCALC> rowsBuffer_;
    TBuf<QuePosition::VECCALC> maximumBuffer_;
    TBuf<QuePosition::VECCALC> candidateBuffer_;
    TBuf<QuePosition::VECCALC> indexBuffer_;
    TBuf<QuePosition::VECCALC> templateBuffer_;
    TBuf<QuePosition::VECCALC> maskBuffer_;
    TBuf<QuePosition::VECCALC> zeroBuffer_;
    GlobalTensor<T> input_;
    GlobalTensor<T> output_;
    GlobalTensor<int32_t> indices_;
    const MaxPool2dWithMaskTilingData* t_;
    int64_t capacity_;
};
} // namespace MaxPool2dWithMask
#endif

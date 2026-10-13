/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef MAX_POOL2D_WITH_MASK_M200_PACKED_INTERVALS_H
#define MAX_POOL2D_WITH_MASK_M200_PACKED_INTERVALS_H
#include "kernel_operator.h"
#include "max_pool2d_with_mask_tiling_data.h"

namespace MaxPool2dWithMask {
using namespace AscendC;

// The four clipped length-seven windows are [0,3], [0,5], [1,6], [3,6].
// Nine merges of ordered disjoint intervals compute all four stable winners.
// Horizontal merges precede vertical merges, preserving row-major tie order.
template <typename T, uint32_t NC>
class M200PackedIntervalsKernel {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR out, GM_ADDR indices, const MaxPool2dWithMaskTilingData* tiling)
    {
        t_ = tiling;
        x_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x));
        out_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(out));
        indices_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(indices));
        pipe_.InitBuffer(raw_, 49 * NC * sizeof(T));
        pipe_.InitBuffer(input_, 56 * NC * sizeof(T));
        pipe_.InitBuffer(horizontal_, 32 * NC * sizeof(T));
        pipe_.InitBuffer(horizontalIndices_, 32 * NC * 4);
        pipe_.InitBuffer(rows_, 28 * NC * sizeof(T));
        pipe_.InitBuffer(rowIndices_, 28 * NC * 4);
        pipe_.InitBuffer(values_, 16 * NC * sizeof(T));
        pipe_.InitBuffer(valueIndices_, 16 * NC * 4);
        pipe_.InitBuffer(inputOffsets_, 56 * NC * 4);
        pipe_.InitBuffer(outputOffsets_, 16 * NC * 4);
        pipe_.InitBuffer(indexOffsets_, 16 * NC * 4);
        pipe_.InitBuffer(rowBase_, 8 * NC * 4);
        pipe_.InitBuffer(mask_, NC);
        pipe_.InitBuffer(zero_, 8192);
        auto in = inputOffsets_.Get<int32_t>();
        auto outOffsets = outputOffsets_.Get<int32_t>();
        auto idxOffsets = indexOffsets_.Get<int32_t>();
        for (uint32_t c = 0; c < NC; ++c) {
            in.SetValue(c, c * 49 * sizeof(T));
        }
        for (uint32_t i = 0; i < 16; ++i) {
            outOffsets.SetValue(i, i * NC * sizeof(T));
            idxOffsets.SetValue(i, i * NC * sizeof(int32_t));
        }
        Sync<HardEvent::S_V>();
        for (uint32_t col = 0; col < 7; ++col) {
            for (uint32_t row = 0; row < 8; ++row) {
                if (col == 0 && row == 0)
                    continue;
                const int32_t spatial = row < 7 ? row * 7 + col : 0;
                Adds(in[(col * 8 + row) * NC], in, int32_t(spatial * sizeof(T)), NC);
            }
        }
        for (uint32_t c = 1; c < NC; ++c) {
            Adds(outOffsets[c * 16], outOffsets, int32_t(c * sizeof(T)), 16);
            Adds(idxOffsets[c * 16], idxOffsets, int32_t(c * sizeof(int32_t)), 16);
        }
        for (uint32_t row = 0; row < 8; ++row) {
            Duplicate(rowBase_.Get<int32_t>()[row * NC], int32_t(row * 7), NC);
        }
        Duplicate(zero_.Get<int32_t>(), int32_t(0), 2048);
        Sync<HardEvent::V_MTE3>();
        zeroNext_ = GetBlockIdx() * t_->zeroBytesPerCore;
        zeroEnd_ = zeroNext_ + t_->zeroBytesPerCore;
        const int64_t tail = t_->containerBytes - t_->validIndexBytes;
        if (zeroEnd_ > tail)
            zeroEnd_ = tail;
    }

    __aicore__ inline void Process()
    {
        const int64_t groups = t_->totalSlices / NC;
        const int64_t first = groups * GetBlockIdx() / GetBlockNum();
        const int64_t last = groups * (GetBlockIdx() + 1) / GetBlockNum();
        for (int64_t group = first; group < last; ++group) {
            // Only staging depends on the preceding result stores. Independent
            // tail stores are issued after this dependency, as in the baseline.
            Sync<HardEvent::MTE3_MTE2>();
            ClearTail(last - group);
            DataCopy(raw_.Get<T>(), x_[group * NC * 49], uint32_t(NC * 49));
            Sync<HardEvent::MTE2_V>();
            Gather(input_.Get<T>(), raw_.Get<T>(), inputOffsets_.Get<uint32_t>(), 0, uint32_t(56 * NC));
            PipeBarrier<PIPE_V>();
            Horizontal();
            Vertical();
            auto idx = valueIndices_.Get<int32_t>();
            Adds(idx, idx, -INDEX_BIAS, 16 * NC);
            PipeBarrier<PIPE_V>();
            Gather(raw_.Get<T>(), values_.Get<T>(), outputOffsets_.Get<uint32_t>(), 0, uint32_t(16 * NC));
            Sync<HardEvent::V_MTE3>();
            DataCopy(out_[group * NC * 16], raw_.Get<T>(), uint32_t(16 * NC));
            Sync<HardEvent::MTE3_V>();
            Gather(raw_.Get<float>(), valueIndices_.Get<float>(), indexOffsets_.Get<uint32_t>(), 0, uint32_t(16 * NC));
            Sync<HardEvent::V_MTE3>();
            DataCopy(indices_[group * NC * 16], raw_.Get<int32_t>(), uint32_t(16 * NC));
        }
        ClearTail(1);
        PipeBarrier<PIPE_ALL>();
    }

private:
    static constexpr int32_t INDEX_BIAS = 0x3f800000;
    template <HardEvent event>
    __aicore__ inline void Sync()
    {
        const event_t id = static_cast<event_t>(pipe_.FetchEventID(event));
        SetFlag<event>(id);
        WaitFlag<event>(id);
    }

    __aicore__ inline void Merge(LocalTensor<T> dst, LocalTensor<int32_t> di, LocalTensor<T> left,
                                 LocalTensor<int32_t> li, LocalTensor<T> right, LocalTensor<int32_t> ri, uint32_t count)
    {
        auto mask = mask_.Get<uint8_t>();
        Compare(mask, right, left, CMPMODE::GT, count);
        PipeBarrier<PIPE_V>();
        Select(dst, mask, right, left, SELMODE::VSEL_TENSOR_TENSOR_MODE, count);
        Select(di.template ReinterpretCast<float>(), mask, ri.template ReinterpretCast<float>(),
               li.template ReinterpretCast<float>(), SELMODE::VSEL_TENSOR_TENSOR_MODE, count);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void MergeRightLeaf(LocalTensor<T> dst, LocalTensor<int32_t> di, LocalTensor<T> left,
                                          LocalTensor<int32_t> li, LocalTensor<T> right, int32_t column, uint32_t count)
    {
        auto mask = mask_.Get<uint8_t>();
        Compare(mask, right, left, CMPMODE::LE, count);
        PipeBarrier<PIPE_V>();
        Select(dst, mask, left, right, SELMODE::VSEL_TENSOR_TENSOR_MODE, count);
        const int32_t encoded = INDEX_BIAS + column;
        Select(di.template ReinterpretCast<float>(), mask, li.template ReinterpretCast<float>(),
               *reinterpret_cast<const float*>(&encoded), SELMODE::VSEL_TENSOR_SCALAR_MODE, count);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void MergeLeftLeaf(LocalTensor<T> dst, LocalTensor<int32_t> di, LocalTensor<T> left,
                                         int32_t column, LocalTensor<T> right, LocalTensor<int32_t> ri, uint32_t count)
    {
        auto mask = mask_.Get<uint8_t>();
        Compare(mask, right, left, CMPMODE::GT, count);
        PipeBarrier<PIPE_V>();
        Select(dst, mask, right, left, SELMODE::VSEL_TENSOR_TENSOR_MODE, count);
        const int32_t encoded = INDEX_BIAS + column;
        Select(di.template ReinterpretCast<float>(), mask, ri.template ReinterpretCast<float>(),
               *reinterpret_cast<const float*>(&encoded), SELMODE::VSEL_TENSOR_SCALAR_MODE, count);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void Horizontal()
    {
        constexpr uint32_t count = 8 * NC;
        auto x = input_.Get<T>();
        auto hi = horizontalIndices_.Get<int32_t>();
        auto h = horizontal_.Get<T>();
        // Input DMA and Gather have finished; the staging allocation can now
        // hold three index summaries. 49*NC*sizeof(T) >= 3*count*sizeof(int32_t).
        auto scratch = raw_.Get<int32_t>();
        for (uint32_t pair = 0; pair < 3; ++pair) {
            Duplicate(scratch[pair * count], INDEX_BIAS + int32_t(2 * pair), count);
        }
        PipeBarrier<PIPE_V>();
        MergeRightLeaf(x, scratch, x, scratch, x[count], 1, count);
        MergeRightLeaf(x[2 * count], scratch[count], x[2 * count], scratch[count], x[3 * count], 3, count);
        MergeRightLeaf(x[4 * count], scratch[2 * count], x[4 * count], scratch[2 * count], x[5 * count], 5, count);
        Merge(h, hi, x, scratch, x[2 * count], scratch[count], count);
        Merge(h[count], hi[count], h, hi, x[4 * count], scratch[2 * count], count);
        MergeRightLeaf(x[4 * count], scratch[2 * count], x[4 * count], scratch[2 * count], x[6 * count], 6, count);
        MergeLeftLeaf(h[3 * count], hi[3 * count], x[3 * count], 3, x[4 * count], scratch[2 * count], count);
        Merge(x[2 * count], scratch[count], x[2 * count], scratch[count], x[4 * count], scratch[2 * count], count);
        MergeLeftLeaf(h[2 * count], hi[2 * count], x[count], 1, x[2 * count], scratch[count], count);
        for (uint32_t col = 0; col < 4; ++col) {
            Add(hi[col * count], hi[col * count], rowBase_.Get<int32_t>(), count);
        }
        PipeBarrier<PIPE_V>();
        DataCopyParams valuesCopy{4, uint16_t(NC * sizeof(T) / 32), uint16_t(7 * NC * sizeof(T) / 32), 0};
        DataCopyParams indicesCopy{4, uint16_t(NC * 4 / 32), uint16_t(7 * NC * 4 / 32), 0};
        for (uint32_t row = 0; row < 7; ++row) {
            DataCopy(rows_.Get<T>()[row * 4 * NC], h[row * NC], valuesCopy);
            DataCopy(rowIndices_.Get<int32_t>()[row * 4 * NC], hi[row * NC], indicesCopy);
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void Vertical()
    {
        constexpr uint32_t n = 4 * NC;
        auto x = rows_.Get<T>();
        auto i = rowIndices_.Get<int32_t>();
        auto v = values_.Get<T>();
        auto vi = valueIndices_.Get<int32_t>();
        Merge(x, i, x, i, x[n], i[n], n);
        Merge(x[2 * n], i[2 * n], x[2 * n], i[2 * n], x[3 * n], i[3 * n], n);
        Merge(x[4 * n], i[4 * n], x[4 * n], i[4 * n], x[5 * n], i[5 * n], n);
        Merge(v, vi, x, i, x[2 * n], i[2 * n], n);
        Merge(v[n], vi[n], v, vi, x[4 * n], i[4 * n], n);
        Merge(x[4 * n], i[4 * n], x[4 * n], i[4 * n], x[6 * n], i[6 * n], n);
        Merge(v[3 * n], vi[3 * n], x[3 * n], i[3 * n], x[4 * n], i[4 * n], n);
        Merge(x[2 * n], i[2 * n], x[2 * n], i[2 * n], x[4 * n], i[4 * n], n);
        Merge(v[2 * n], vi[2 * n], x[n], i[n], x[2 * n], i[2 * n], n);
    }

    __aicore__ inline void ClearTail(int64_t remaining)
    {
        const int64_t chunks = (zeroEnd_ - zeroNext_ + 8191) / 8192;
        const int64_t count = (chunks + remaining - 1) / remaining;
        for (int64_t chunk = 0; chunk < count && zeroNext_ < zeroEnd_; ++chunk) {
            const uint32_t bytes = zeroEnd_ - zeroNext_ < 8192 ? zeroEnd_ - zeroNext_ : 8192;
            DataCopy(indices_[(t_->validIndexBytes + zeroNext_) / 4], zero_.Get<int32_t>(), bytes / 4);
            zeroNext_ += bytes;
        }
    }

    TPipe pipe_;
    TBuf<QuePosition::VECCALC> raw_;
    TBuf<QuePosition::VECCALC> input_;
    TBuf<QuePosition::VECCALC> horizontal_;
    TBuf<QuePosition::VECCALC> horizontalIndices_;
    TBuf<QuePosition::VECCALC> rows_;
    TBuf<QuePosition::VECCALC> rowIndices_;
    TBuf<QuePosition::VECCALC> values_, valueIndices_, inputOffsets_, outputOffsets_, indexOffsets_, rowBase_, mask_,
        zero_;
    GlobalTensor<T> x_;
    GlobalTensor<T> out_;
    GlobalTensor<int32_t> indices_;
    const MaxPool2dWithMaskTilingData* t_;
    int64_t zeroNext_;
    int64_t zeroEnd_;
};
} // namespace MaxPool2dWithMask
#endif

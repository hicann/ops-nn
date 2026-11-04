// clang-format off
/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 *
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of CANN Open Software License Agreement Version 2.0 (the "License").
 *
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 *
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 *
 * See LICENSE in the root of the software repository for the full text of the License.
 */
// clang-format on

#ifndef MAX_POOL2D_WITH_MASK_BACKWARD_SCATTER_310P_H
#define MAX_POOL2D_WITH_MASK_BACKWARD_SCATTER_310P_H
#include "copy32_310p.h"

namespace MaxPool2dMask310P {
using namespace AscendC;

// Planar full-width stripes, aligned to pooling rows. The host admits only
// non-overlapping windows without padding, so every scatter target is unique
// and remains inside this stripe. No transpose, atomics, or cross-core sync.
template <typename T, int64_t OUT_ELEMENTS = 16384, int64_t IN_ELEMENTS = 2048, bool PIPE_OUTPUT = false>
class ScatterKernel {
public:
    template <class TD>
    __aicore__ inline void Init(GM_ADDR grad, GM_ADDR index, GM_ADDR out, const TD* t, TPipe* pipe)
    {
        pipe_ = pipe;
        hi_ = t->hiDim;
        wi_ = t->wiDim;
        ho_ = t->hoDim;
        wo_ = t->woDim;
        sh_ = t->sh;
        nc_ = t->ncDim;
        tileH_ = t->baseHo;
        stripes_ = (hi_ + tileH_ - 1) / tileH_;
        tiles_ = nc_ * stripes_;
        grad_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(grad));
        index_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(index));
        out_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(out));
        pipe->InitBuffer(accBuf_, OUTPUT_CAPACITY * sizeof(float) * (PIPE_OUTPUT ? 2 : 1));
        if constexpr (sizeof(T) == 2)
            pipe->InitBuffer(outBuf_, OUTPUT_CAPACITY * sizeof(T));
        pipe->InitBuffer(gradBuf_, INPUT_CAPACITY * sizeof(T));
        if constexpr (sizeof(T) == 2)
            pipe->InitBuffer(floatBuf_, INPUT_CAPACITY * sizeof(float));
        pipe->InitBuffer(indexBuf_, INPUT_CAPACITY * sizeof(int32_t));
        copy_.Init(pipe);
        indexCopy_.Init(pipe);
        if constexpr (PIPE_OUTPUT) {
            outputEvents_[0] = pipe->AllocEventID<HardEvent::MTE3_V>();
            outputEvents_[1] = pipe->AllocEventID<HardEvent::MTE3_V>();
        }
    }

    __aicore__ inline void Process()
    {
        // If every stripe has a whole number of core groups, interleave NC
        // within a stripe. All cores then receive the same full/tail mix.
        // NC-major scheduling can pin short stripes to only a subset of cores.
        const bool stripeMajor = OUT_ELEMENTS > 16384 && stripes_ > 1 && nc_ % GetBlockNum() == 0;
        int32_t outputSlot = 0;
        for (int64_t tile = GetBlockIdx(); tile < tiles_; tile += GetBlockNum()) {
            const int64_t nc = stripeMajor ? tile % nc_ : tile / stripes_;
            const int64_t y = (stripeMajor ? tile / nc_ : tile % stripes_) * tileH_;
            const int64_t height = Min(tileH_, hi_ - y);
            const int64_t length = height * wi_;
            const int64_t first = y / sh_;
            const int64_t end = Min(ho_, (y + height + sh_ - 1) / sh_);
            if constexpr (PIPE_OUTPUT) {
                if (outputPending_[outputSlot]) {
                    WaitFlag<HardEvent::MTE3_V>(outputEvents_[outputSlot]);
                    outputPending_[outputSlot] = false;
                }
            }
            auto acc = accBuf_.Get<float>()[PIPE_OUTPUT ? outputSlot * OUTPUT_CAPACITY : 0];
            Duplicate(acc, 0.0f, static_cast<int32_t>(length));
            PipeBarrier<PIPE_V>();
            const int64_t gradTotal = nc_ * ho_ * wo_;
            for (int64_t q = first * wo_; q < end * wo_; q += INPUT_CAPACITY) {
                const int64_t count = Min(INPUT_CAPACITY, end * wo_ - q);
                const int64_t offset = nc * ho_ * wo_ + q;
                auto grad = gradBuf_.Get<T>();
                auto idx = indexBuf_.Get<int32_t>();
                if (count % (32 / sizeof(T)) == 0) {
                    // Both input buffers share one MTE2-to-vector dependency.
                    DataCopy(grad, grad_[offset], static_cast<uint32_t>(count));
                    DataCopy(idx, index_[offset], static_cast<uint32_t>(count));
                    Sync<HardEvent::MTE2_V>();
                } else {
                    if (!copy_.In(grad, grad_, offset, count, gradTotal, Max(gradTotal, 32 / sizeof(T)),
                                  INPUT_CAPACITY) ||
                        !indexCopy_.In(idx, index_, offset, count, gradTotal, Max(gradTotal, 8), INPUT_CAPACITY))
                        return;
                }
                auto values = grad.template ReinterpretCast<float>();
                if constexpr (sizeof(T) == sizeof(float)) {
                    values = grad.template ReinterpretCast<float>();
                } else {
                    values = floatBuf_.Get<float>();
                    Cast(values, grad, RoundMode::CAST_NONE, static_cast<int32_t>(count));
                }
                Adds(idx, idx, static_cast<int32_t>(-y * wi_), static_cast<int32_t>(count));
                PipeBarrier<PIPE_V>();
                Muls(idx, idx, static_cast<int32_t>(sizeof(float)), static_cast<int32_t>(count));
                PipeBarrier<PIPE_V>();
                // Byte offsets are relative to this output stripe, not GM.
                // <=4096 FP32 inputs gives <=64 repeats (the API limit is 255).
                Scatter(acc, values, idx.template ReinterpretCast<uint32_t>(), 0, static_cast<uint32_t>(count));
                Sync<HardEvent::V_MTE2>();
            }
            auto result = acc.template ReinterpretCast<T>();
            if constexpr (sizeof(T) == sizeof(float)) {
                result = acc.template ReinterpretCast<T>();
            } else {
                result = outBuf_.Get<T>();
                PipeBarrier<PIPE_V>();
                Cast(result, acc, RoundMode::CAST_NONE, static_cast<int32_t>(length));
                PipeBarrier<PIPE_V>();
            }
            // One contiguous write per stripe plus at most one in-stripe tail.
            // Copy32 waits for MTE3 before any output buffer is reused.
            if (length % (32 / sizeof(T)) == 0) {
                // No tail mask or scalar UB access is needed for full blocks.
                Sync<HardEvent::V_MTE3>();
                DataCopy(out_[(nc * hi_ + y) * wi_], result, static_cast<uint32_t>(length));
                if constexpr (PIPE_OUTPUT) {
                    SetFlag<HardEvent::MTE3_V>(outputEvents_[outputSlot]);
                    outputPending_[outputSlot] = true;
                } else {
                    Sync<HardEvent::MTE3_V>();
                }
            } else {
                if (!copy_.Out(out_, (nc * hi_ + y) * wi_, result, length, nc_ * hi_ * wi_, nc_ * hi_ * wi_,
                               OUTPUT_CAPACITY))
                    return;
            }
            if constexpr (PIPE_OUTPUT)
                outputSlot ^= 1;
        }
        if constexpr (PIPE_OUTPUT) {
            for (int32_t slot = 0; slot < 2; ++slot) {
                if (outputPending_[slot])
                    WaitFlag<HardEvent::MTE3_V>(outputEvents_[slot]);
                pipe_->ReleaseEventID<HardEvent::MTE3_V>(outputEvents_[slot]);
            }
        }
    }

private:
    static constexpr int64_t OUTPUT_CAPACITY = OUT_ELEMENTS;
    static constexpr int64_t INPUT_CAPACITY = IN_ELEMENTS;
    static_assert(!PIPE_OUTPUT || sizeof(T) == 4, "Pipelined output is FP32 only");
    static_assert(OUT_ELEMENTS * (sizeof(T) == 2 ? 6 : (PIPE_OUTPUT ? 8 : 4)) +
                          IN_ELEMENTS * (sizeof(T) == 2 ? 10 : 8) + 192 <=
                      245760,
                  "Scatter UB budget exceeded");
    TEventID outputEvents_[2];
    bool outputPending_[2] = {false, false};
    __aicore__ inline int64_t Min(int64_t a, int64_t b) { return a < b ? a : b; }
    __aicore__ inline int64_t Max(int64_t a, int64_t b) { return a > b ? a : b; }
    template <HardEvent E>
    __aicore__ inline void Sync()
    {
        auto e = pipe_->FetchEventID(E);
        SetFlag<E>(e);
        WaitFlag<E>(e);
    }
    TPipe* pipe_;
    GlobalTensor<T> grad_, out_;
    GlobalTensor<int32_t> index_;
    TBuf<TPosition::VECCALC> accBuf_, outBuf_, gradBuf_, floatBuf_, indexBuf_;
    MaxPool310P::Copy32<T, true> copy_;
    MaxPool310P::Copy32<int32_t, true> indexCopy_;
    int64_t hi_, wi_, ho_, wo_, sh_, nc_, tileH_, stripes_, tiles_;
};
} // namespace MaxPool2dMask310P
#endif

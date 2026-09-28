/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SINGLE_LAYER_LSTM_GRAD_WIDE_ADAPTER_H
#define SINGLE_LAYER_LSTM_GRAD_WIDE_ADAPTER_H
#include "kernel_operator.h"
#include "single_layer_lstm_grad_wide_workspace.h"
#include "../../single_layer_lstm/arch35/single_layer_lstm_gate_math.h"

namespace LstmGradWide {
namespace Micro = AscendC::MicroAPI;

template <AscendC::HardEvent Event>
__aicore__ inline void SyncPipe()
{
    const auto event = static_cast<event_t>(GetTPipePtr()->FetchEventID(Event));
    AscendC::SetFlag<Event>(event);
    AscendC::WaitFlag<Event>(event);
}

template <typename T>
class Adapter {
public:
    Workspace layout;
    __aicore__ inline void Init(GM_ADDR workspace, int64_t time, int64_t batch, int64_t input, int64_t hidden,
                                int64_t parts, int64_t direction, int64_t order, AscendC::TPipe* pipe)
    {
        base_ = workspace;
        t_ = time;
        b_ = batch;
        i_ = input;
        h_ = hidden;
        parts_ = parts;
        reverse_ = direction != 0;
        order_ = order;
        pipe_ = pipe;
        layout.Fill(time, batch, input, hidden, parts);
        gm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(base_));
    }

    __aicore__ inline GM_ADDR At(uint64_t offset) const { return base_ + offset; }
    __aicore__ inline GM_ADDR Plane(uint32_t plane) const
    {
        return At(layout.cache + plane * layout.planeElements * sizeof(float));
    }

    __aicore__ inline void Prepare(GM_ADDR x, GM_ADDR w, GM_ADDR bias, GM_ADDR initH, GM_ADDR initC, GM_ADDR dy,
                                   GM_ADDR dh, GM_ADDR dc)
    {
        if (g_coreType == AscendC::AIV) {
            pipe_->InitBuffer(scratch_, SCRATCH_BYTES);
            Widen(x, layout.x, layout.inputElements);
            Widen(w, layout.w, layout.weightElements);
            Widen(bias, layout.bias, static_cast<uint64_t>(parts_) * 4 * h_);
            Widen(initH, layout.initH, layout.stateElements);
            Widen(initC, layout.initC, layout.stateElements);
            Widen(dy, layout.dy, layout.planeElements);
            Widen(dh, layout.dh, layout.stateElements);
            Widen(dc, layout.dc, layout.stateElements);
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::DataCacheCleanAndInvalid<float, AscendC::CacheLine::ENTIRE_DATA_CACHE>(gm_);
        }
        AscendC::SyncAll();
        for (int64_t step = 0; step < t_; ++step) {
            if (g_coreType == AscendC::AIV) {
                const int64_t tiles = (h_ + TILE - 1) / TILE;
                for (int64_t task = AscendC::GetBlockIdx(); task < b_ * tiles; task += AscendC::GetBlockNum() * 2) {
                    ReplayTile(step, task / tiles, (task % tiles) * TILE);
                }
                AscendC::PipeBarrier<PIPE_ALL>();
                AscendC::DataCacheCleanAndInvalid<float, AscendC::CacheLine::ENTIRE_DATA_CACHE>(gm_);
            }
            // A previous hidden row is complete before any core reads it at the next time step.
            AscendC::SyncAll();
        }
        pipe_->Reset();
    }

    __aicore__ inline void Finish(GM_ADDR dx, GM_ADDR dw, GM_ADDR db, GM_ADDR dh, GM_ADDR dc)
    {
        AscendC::SyncAll();
        if (g_coreType != AscendC::AIV) {
            return;
        }
        pipe_->Reset();
        pipe_->InitBuffer(scratch_, SCRATCH_BYTES);
        Narrow(dx, layout.dx, layout.inputElements);
        Narrow(dw, layout.dw, layout.weightElements);
        if (parts_ != 0) {
            Narrow(db, layout.db, static_cast<uint64_t>(4) * h_);
        }
        Narrow(dh, layout.dhPrev, layout.stateElements);
        Narrow(dc, layout.dcPrev, layout.stateElements);
    }

private:
    static constexpr uint32_t TILE = 64;
    static constexpr uint32_t COPY_ELEMENTS = 4096;
    static constexpr uint32_t SCRATCH_BYTES = 32 * 1024;
    static constexpr uint32_t RAW = 0;
    static constexpr uint32_t WIDE = 8192;
    static constexpr uint32_t WEIGHTS = 0; // 64x64 FP32 tile, reused after whole-input conversion
    static constexpr uint32_t INPUT = 16384;
    static constexpr uint32_t GATES = INPUT + TILE * sizeof(float);
    static constexpr uint32_t CELL = GATES + 4 * TILE * sizeof(float);
    static constexpr uint32_t HIDDEN = CELL + TILE * sizeof(float);
    static constexpr uint32_t TANHC = HIDDEN + TILE * sizeof(float);
    static constexpr uint32_t TMP1 = TANHC + TILE * sizeof(float);
    static constexpr uint32_t TMP2 = TMP1 + TILE * sizeof(float);
    static constexpr uint32_t TMP3 = TMP2 + TILE * sizeof(float);
    static constexpr uint32_t MASK = TMP3 + TILE * sizeof(float);
    GM_ADDR base_;
    int64_t t_, b_, i_, h_, parts_, order_;
    bool reverse_;
    AscendC::TPipe* pipe_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> scratch_;
    AscendC::GlobalTensor<float> gm_;

    template <typename U = float>
    __aicore__ inline AscendC::LocalTensor<U> Local(uint32_t byteOffset)
    {
        return scratch_.Get<uint8_t>()[byteOffset].template ReinterpretCast<U>();
    }

    __aicore__ inline void Widen(GM_ADDR input, uint64_t offset, uint64_t elements)
    {
        if (elements == 0) {
            return;
        }
        AscendC::GlobalTensor<T> src;
        src.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(input));
        for (uint64_t pos = static_cast<uint64_t>(AscendC::GetBlockIdx()) * COPY_ELEMENTS; pos < elements;
             pos += AscendC::GetBlockNum() * 2 * COPY_ELEMENTS) {
            const uint32_t n = elements - pos < COPY_ELEMENTS ? elements - pos : COPY_ELEMENTS;
            AscendC::PipeBarrier<PIPE_ALL>();
            const AscendC::DataCopyExtParams narrowCopy{1, static_cast<uint32_t>(n * sizeof(T)), 0, 0, 0};
            const AscendC::DataCopyExtParams floatCopy{1, n * uint32_t{sizeof(float)}, 0, 0, 0};
            AscendC::DataCopyPad(Local<T>(RAW), src[pos], narrowCopy, AscendC::DataCopyPadExtParams<T>{false, 0, 0, 0});
            SyncPipe<AscendC::HardEvent::MTE2_V>();
            AscendC::Cast(Local(WIDE), Local<T>(RAW), AscendC::RoundMode::CAST_NONE, n);
            SyncPipe<AscendC::HardEvent::V_MTE3>();
            AscendC::DataCopyPad(gm_[offset / sizeof(float) + pos], Local(WIDE), floatCopy);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void Narrow(GM_ADDR output, uint64_t offset, uint64_t elements)
    {
        AscendC::GlobalTensor<T> dst;
        dst.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(output));
        for (uint64_t pos = static_cast<uint64_t>(AscendC::GetBlockIdx()) * COPY_ELEMENTS; pos < elements;
             pos += AscendC::GetBlockNum() * 2 * COPY_ELEMENTS) {
            const uint32_t n = elements - pos < COPY_ELEMENTS ? elements - pos : COPY_ELEMENTS;
            AscendC::PipeBarrier<PIPE_ALL>();
            const AscendC::DataCopyExtParams floatCopy{1, n * uint32_t{sizeof(float)}, 0, 0, 0};
            const AscendC::DataCopyExtParams narrowCopy{1, static_cast<uint32_t>(n * sizeof(T)), 0, 0, 0};
            AscendC::DataCopyPad(Local(WIDE), gm_[offset / sizeof(float) + pos], floatCopy,
                                 AscendC::DataCopyPadExtParams<float>{false, 0, 0, 0});
            SyncPipe<AscendC::HardEvent::MTE2_V>();
            AscendC::Cast(Local<T>(RAW), Local(WIDE), AscendC::RoundMode::CAST_RINT, n);
            SyncPipe<AscendC::HardEvent::V_MTE3>();
            AscendC::DataCopyPad(dst[pos], Local<T>(RAW), narrowCopy);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void Load(uint32_t target, uint64_t sourceBytes, uint32_t n)
    {
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::Duplicate(Local(target), 0.0f, TILE);
        SyncPipe<AscendC::HardEvent::V_MTE2>();
        const AscendC::DataCopyExtParams copy{1, n * uint32_t{sizeof(float)}, 0, 0, 0};
        AscendC::DataCopyPad(Local(target), gm_[sourceBytes / sizeof(float)], copy,
                             AscendC::DataCopyPadExtParams<float>{false, 0, 0, 0});
        SyncPipe<AscendC::HardEvent::MTE2_V>();
    }

    __aicore__ inline uint64_t CacheOffset(uint32_t plane, int64_t time, int64_t batch, int64_t col) const
    {
        return layout.cache + (plane * layout.planeElements + (time * b_ + batch) * h_ + col) * sizeof(float);
    }

    __aicore__ inline void Dot(uint32_t gateOffset, int64_t slot, int64_t step, int64_t batch, int64_t h0,
                               uint32_t rows)
    {
        const int64_t time = reverse_ ? t_ - 1 - step : step;
        for (int32_t source = 0; source < 2; ++source) {
            const int64_t width = source == 0 ? i_ : h_;
            for (int64_t col = 0; col < width; col += TILE) {
                const uint32_t n = width - col < TILE ? width - col : TILE;
                uint64_t input = layout.x + ((time * b_ + batch) * i_ + col) * sizeof(float);
                if (source != 0) {
                    input = step == 0 ? layout.initH + (batch * h_ + col) * sizeof(float) :
                                        CacheOffset(6, reverse_ ? time + 1 : time - 1, batch, col);
                }
                Load(INPUT, input, n);
                SyncPipe<AscendC::HardEvent::V_MTE2>();
                const uint64_t w = layout.w / sizeof(float) + (slot * h_ + h0) * (i_ + h_) + (source == 0 ? 0 : i_) +
                                   col;
                const AscendC::DataCopyExtParams copy{static_cast<uint16_t>(rows), n * uint32_t{sizeof(float)},
                                                      static_cast<uint32_t>((i_ + h_ - n) * sizeof(float)), 0, 0};
                AscendC::DataCopyPad(Local(WEIGHTS), gm_[w], copy,
                                     AscendC::DataCopyPadExtParams<float>{false, 0, 0, 0});
                SyncPipe<AscendC::HardEvent::MTE2_V>();
                auto* xp = (__local_mem__ float*)Local(INPUT).GetPhyAddr();
                auto* wp = (__local_mem__ float*)Local(WEIGHTS).GetPhyAddr();
                auto* gp = (__local_mem__ float*)Local(gateOffset).GetPhyAddr();
                const uint32_t pitch = (n + 7) / 8 * 8;
                __VEC_SCOPE__
                {
                    uint32_t count = n;
                    Micro::MaskReg mask = Micro::UpdateMask<float>(count);
                    Micro::MaskReg one = Micro::CreateMask<float, Micro::MaskPattern::VL1>();
                    Micro::RegTensor<float> x, weight, product, partial, sum;
                    Micro::DataCopy<float, Micro::LoadDist::DIST_NORM>(x, xp);
                    for (uint16_t row = 0; row < static_cast<uint16_t>(rows); ++row) {
                        Micro::DataCopy<float, Micro::LoadDist::DIST_NORM>(weight, wp + row * pitch);
                        Micro::Mul(product, x, weight, mask);
                        Micro::ReduceSum(partial, product, mask);
                        Micro::DataCopy<float, Micro::LoadDist::DIST_BRC_B32>(sum, gp + row);
                        Micro::Add(sum, sum, partial, one);
                        Micro::DataCopy<float, Micro::StoreDist::DIST_FIRST_ELEMENT_B32>(gp + row, sum, one);
                    }
                }
            }
        }
    }

    __aicore__ inline void ReplayTile(int64_t step, int64_t batch, int64_t h0)
    {
        const int64_t time = reverse_ ? t_ - 1 - step : step;
        const uint32_t n = h_ - h0 < TILE ? h_ - h0 : TILE;
        const int64_t slots[4] = {0, order_ == 0 ? 1 : 2, order_ == 0 ? 2 : 1, 3};
        for (int32_t gate = 0; gate < 4; ++gate) {
            const uint32_t off = GATES + gate * TILE * sizeof(float);
            AscendC::Duplicate(Local(off), 0.0f, TILE);
            for (int64_t part = 0; part < parts_; ++part) {
                Load(INPUT, layout.bias + (part * 4 * h_ + slots[gate] * h_ + h0) * sizeof(float), n);
                AscendC::Add(Local(off), Local(off), Local(INPUT), n);
                AscendC::PipeBarrier<PIPE_V>();
            }
            Dot(off, slots[gate], step, batch, h0, n);
        }
        const uint64_t previousC = step == 0 ? layout.initC + (batch * h_ + h0) * sizeof(float) :
                                               CacheOffset(5, reverse_ ? time + 1 : time - 1, batch, h0);
        Load(CELL, previousC, n);
        SingleLayerLstmVec::SingleLayerLstmGates(
            Local(GATES), Local(GATES + 2 * TILE * sizeof(float)), Local(GATES + TILE * sizeof(float)),
            Local(GATES + 3 * TILE * sizeof(float)), Local(CELL), Local(HIDDEN), Local(TANHC), Local(TMP1), Local(TMP2),
            Local(TMP3), Local<uint8_t>(MASK), TILE);
        SyncPipe<AscendC::HardEvent::V_MTE3>();
        const uint32_t offsets[7] = {GATES,
                                     GATES + TILE * sizeof(float),
                                     GATES + 2 * TILE * sizeof(float),
                                     GATES + 3 * TILE * sizeof(float),
                                     TANHC,
                                     CELL,
                                     HIDDEN};
        for (uint32_t plane = 0; plane < 7; ++plane) {
            const AscendC::DataCopyExtParams copy{1, n * uint32_t{sizeof(float)}, 0, 0, 0};
            AscendC::DataCopyPad(gm_[CacheOffset(plane, time, batch, h0) / sizeof(float)], Local(offsets[plane]), copy);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }
};
} // namespace LstmGradWide
#endif

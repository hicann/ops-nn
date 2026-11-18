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

namespace LstmGradWide {
constexpr int32_t VECTOR_CORES_PER_CUBE = 2;
constexpr int32_t GATE_SLOT_F = 2;
constexpr int32_t GATE_SLOT_O = 3;

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
    static constexpr uint32_t COPY_ELEMENTS = 4096;
    static constexpr uint32_t SCRATCH_BYTES = 32 * 1024;
    static constexpr uint32_t RAW = 0;
    static constexpr uint32_t WIDE = 8192;

public:
    Workspace layout;
    __aicore__ inline void Init(GM_ADDR workspace, int64_t time, int64_t batch, int64_t input, int64_t hidden,
                                int64_t parts, int64_t direction, int64_t order, AscendC::TPipe* pipe)
    {
        base_ = workspace;
        h_ = hidden;
        parts_ = parts;
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
                                   GM_ADDR dh, GM_ADDR dc, GM_ADDR i, GM_ADDR j, GM_ADDR f, GM_ADDR o, GM_ADDR tanhc,
                                   GM_ADDR c, GM_ADDR h)
    {
        if (g_coreType == AscendC::AIV) {
            pipe_->InitBuffer(scratch_, SCRATCH_BYTES);
            Widen(x, layout.x, layout.inputElements);
            Widen(w, layout.w, layout.weightElements);
            Widen(bias, layout.bias, static_cast<uint64_t>(parts_) * GATE_COUNT * h_);
            Widen(initH, layout.initH, layout.stateElements);
            Widen(initC, layout.initC, layout.stateElements);
            Widen(dy, layout.dy, layout.planeElements);
            Widen(dh, layout.dh, layout.stateElements);
            Widen(dc, layout.dc, layout.stateElements);
            // Preserve the forward values supplied by the caller while computing backward in FP32.
            GM_ADDR saved[Workspace::PLANES] = {i, j, f, o, tanhc, c, h};
            for (uint32_t plane = 0; plane < Workspace::PLANES; ++plane) {
                Widen(saved[plane], layout.cache + plane * layout.planeElements * sizeof(float), layout.planeElements);
            }
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::DataCacheCleanAndInvalid<float, AscendC::CacheLine::ENTIRE_DATA_CACHE>(gm_);
        }
        AscendC::SyncAll();
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
            Narrow(db, layout.db, static_cast<uint64_t>(GATE_COUNT) * h_);
        }
        Narrow(dh, layout.dhPrev, layout.stateElements);
        Narrow(dc, layout.dcPrev, layout.stateElements);
    }

private:
    GM_ADDR base_;
    int64_t h_, parts_;
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
             pos += AscendC::GetBlockNum() * VECTOR_CORES_PER_CUBE * COPY_ELEMENTS) {
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
             pos += AscendC::GetBlockNum() * VECTOR_CORES_PER_CUBE * COPY_ELEMENTS) {
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
};
} // namespace LstmGradWide
#endif

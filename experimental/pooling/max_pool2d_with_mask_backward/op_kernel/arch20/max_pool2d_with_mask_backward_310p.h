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

#ifndef MAX_POOL2D_WITH_MASK_BACKWARD_310P_H
#define MAX_POOL2D_WITH_MASK_BACKWARD_310P_H
#include "kernel_operator.h"
#include "copy32_310p.h"

namespace MaxPool2dMask310P {
using namespace AscendC;

// Own complete contiguous output tiles, including their tail blocks. All
// overlapping contributions are accumulated in FP32 UB by the same writer.
// No global atomics or A2-only hard SyncAll/DataCopyPad instructions are needed.
template <typename T>
class Kernel {
public:
    template <typename Tiling>
    __aicore__ inline void Init(GM_ADDR grad, GM_ADDR argmax, GM_ADDR y, const Tiling* t, TPipe* pipe)
    {
        hi_ = t->hiDim;
        wi_ = t->wiDim;
        ho_ = t->hoDim;
        wo_ = t->woDim;
        kh_ = t->kh;
        sh_ = t->sh;
        ph_ = t->padHTop;
        sw_ = t->sw;
        kernelOne_ = t->kh == 1 && t->kw == 1;
        spatial_ = hi_ * wi_;
        gradSpatial_ = ho_ * wo_;
        total_ = t->ncDim * spatial_;
        gradTotal_ = t->ncDim * gradSpatial_;
        tile_ = t->baseWo;
        tiles_ = t->totalCnt;
        grad_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(grad));
        index_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(argmax));
        out_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y));
        pipe_ = pipe;
        pipe->InitBuffer(accBuf_, 2 * tile_ * sizeof(float));
        pipe->InitBuffer(outBuf_, 2 * tile_ * sizeof(T));
        pipe->InitBuffer(gradBuf_, (INPUT_TILE + 16) * sizeof(T));
        pipe->InitBuffer(floatBuf_, (INPUT_TILE + 16) * sizeof(float));
        pipe->InitBuffer(indexBuf_, (INPUT_TILE + 16) * sizeof(int32_t));
        dataCopy_.Init(pipe);
        indexCopy_.Init(pipe);
    }

    __aicore__ inline void Process()
    {
        for (int64_t id = GetBlockIdx(); id < tiles_; id += GetBlockNum()) {
            const int64_t begin = id * tile_;
            const int64_t length = id + 1 == tiles_ ? total_ - begin : tile_;
            auto acc = accBuf_.Get<float>();
            Duplicate(acc, 0.0f, static_cast<int32_t>((length + 7) / 8 * 8));
            Sync<HardEvent::V_S>();
            for (int64_t nc = begin / spatial_; nc <= (begin + length - 1) / spatial_; ++nc) {
                Accumulate(nc, begin, length, acc);
            }
            Write(begin, length, acc);
        }
    }

private:
    static constexpr int64_t INPUT_TILE = 1024;
    template <HardEvent E>
    __aicore__ inline void Sync()
    {
        auto event = pipe_->FetchEventID(E);
        SetFlag<E>(event);
        WaitFlag<E>(event);
    }
    __aicore__ inline int64_t Min(int64_t a, int64_t b) { return a < b ? a : b; }
    __aicore__ inline int64_t Max(int64_t a, int64_t b) { return a > b ? a : b; }

    __aicore__ inline void Accumulate(int64_t nc, int64_t begin, int64_t length, const LocalTensor<float>& acc)
    {
        const int64_t low = Max(begin - nc * spatial_, 0);
        const int64_t high = Min(begin + length - nc * spatial_, spatial_);
        // Only pooling rows whose receptive fields intersect this output tile.
        const int64_t firstNumerator = low / wi_ + ph_ - kh_ + 1;
        const int64_t firstRow = firstNumerator > 0 ? (firstNumerator + sh_ - 1) / sh_ : 0;
        const int64_t lastRow = Min(ho_, ((high - 1) / wi_ + ph_) / sh_ + 1);
        auto grad = gradBuf_.Get<T>();
        auto idx = indexBuf_.Get<int32_t>();
        auto values = floatBuf_.Get<float>();
        for (int64_t q = firstRow * wo_; q < lastRow * wo_; q += INPUT_TILE) {
            const int64_t count = Min(INPUT_TILE, lastRow * wo_ - q);
            if (!dataCopy_.In(grad, grad_, nc * gradSpatial_ + q, count, gradTotal_, Max(gradTotal_, 32 / sizeof(T)),
                              INPUT_TILE + 16) ||
                (!kernelOne_ && !indexCopy_.In(idx, index_, nc * gradSpatial_ + q, count, gradTotal_,
                                               Max(gradTotal_, 8), INPUT_TILE + 16))) {
                return;
            }
            if constexpr (sizeof(T) == sizeof(float)) {
                values = grad.template ReinterpretCast<float>();
            } else {
                Sync<HardEvent::MTE2_V>();
                Cast(values, grad, RoundMode::CAST_NONE, static_cast<int32_t>((count + 15) / 16 * 16));
                Sync<HardEvent::V_S>();
            }
            for (int64_t j = 0; j < count; ++j) {
                const int64_t target = kernelOne_ ? ((q + j) / wo_) * sh_ * wi_ + ((q + j) % wo_) * sw_ :
                                                    idx.GetValue(j);
                if (target >= low && target < high) {
                    const int64_t local = nc * spatial_ + target - begin;
                    acc.SetValue(local, acc.GetValue(local) + values.GetValue(j));
                }
            }
            // Do not recycle the source UB before scalar reads finish.
            Sync<HardEvent::S_MTE2>();
        }
    }

    __aicore__ inline void Write(int64_t begin, int64_t length, const LocalTensor<float>& acc)
    {
        constexpr int64_t block = 32 / sizeof(T);
        auto result = outBuf_.Get<T>();
        if constexpr (sizeof(T) == sizeof(float)) {
            result = acc.template ReinterpretCast<T>();
            Sync<HardEvent::S_MTE3>();
        } else {
            Sync<HardEvent::S_V>();
            // One final conversion after every overlapping contribution.
            Cast(result, acc, RoundMode::CAST_NONE, static_cast<int32_t>(length));
            Sync<HardEvent::V_MTE3>();
            Sync<HardEvent::V_S>();
        }
        // The last tile absorbs the remainder, so every non-short write owns
        // at least one full block. L0 reserves backing for a whole short tensor.
        const bool copied = dataCopy_.Out(out_, begin, result, length, total_, Max(total_, block), 2 * tile_);
        if (!copied)
            return;
    }

    TPipe* pipe_ = nullptr;
    GlobalTensor<T> grad_, out_;
    GlobalTensor<int32_t> index_;
    TBuf<TPosition::VECCALC> accBuf_, outBuf_, gradBuf_, floatBuf_, indexBuf_;
    MaxPool310P::Copy32<T> dataCopy_;
    MaxPool310P::Copy32<int32_t> indexCopy_;
    bool kernelOne_;
    int64_t sw_;
    int64_t hi_, wi_, ho_, wo_, kh_, sh_, ph_, spatial_, gradSpatial_, total_, gradTotal_, tile_, tiles_;
};
} // namespace MaxPool2dMask310P
#endif

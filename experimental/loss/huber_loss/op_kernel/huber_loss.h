/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef HUBER_LOSS_H_
#define HUBER_LOSS_H_
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "huber_loss_tiling_data.h"
#include <type_traits>
namespace NsHuberLoss {
using namespace AscendC;
constexpr int32_t kBufferNum = 2;
template <typename T>
class KernelHuberLoss {
    using IoT = T;
    static constexpr bool kUpcast = !std::is_same<IoT, float>::value;

public:
    __aicore__ inline void Init(GM_ADDR predictions, GM_ADDR targets, GM_ADDR loss, const HuberLossTilingData* t)
    {
        const uint32_t idx = GetBlockIdx();
        tile_ = t->tileDataNum;
        delta_ = t->delta;
        if (idx < t->tailBlockNum) {
            count_ = t->bigCoreDataNum;
            tiles_ = t->finalBigTileNum;
            tail_ = t->bigTailDataNum;
            offset_ = static_cast<uint64_t>(idx) * t->bigCoreDataNum;
        } else {
            count_ = t->smallCoreDataNum;
            tiles_ = t->finalSmallTileNum;
            tail_ = t->smallTailDataNum;
            offset_ = static_cast<uint64_t>(t->tailBlockNum) * t->bigCoreDataNum +
                      static_cast<uint64_t>(idx - t->tailBlockNum) * t->smallCoreDataNum;
        }
        predictionsGM_.SetGlobalBuffer((__gm__ IoT*)predictions + offset_, count_);
        targetsGM_.SetGlobalBuffer((__gm__ IoT*)targets + offset_, count_);
        lossGM_.SetGlobalBuffer((__gm__ IoT*)loss + offset_, count_);
        pipe_.InitBuffer(predictionsQueue_, kBufferNum, tile_ * sizeof(IoT));
        pipe_.InitBuffer(targetsQueue_, kBufferNum, tile_ * sizeof(IoT));
        pipe_.InitBuffer(lossQueue_, kBufferNum, tile_ * sizeof(IoT));
        pipe_.InitBuffer(diffBuf_, tile_ * sizeof(float));
        pipe_.InitBuffer(absBuf_, tile_ * sizeof(float));
        pipe_.InitBuffer(quadraticBuf_, tile_ * sizeof(float));
        pipe_.InitBuffer(linearBuf_, tile_ * sizeof(float));
        pipe_.InitBuffer(maskBuf_, tile_ * sizeof(uint8_t));
        if constexpr (kUpcast) {
            pipe_.InitBuffer(predictionsFloatBuf_, tile_ * sizeof(float));
            pipe_.InitBuffer(targetsFloatBuf_, tile_ * sizeof(float));
        }
    }
    __aicore__ inline void Process()
    {
        for (uint32_t i = 0; i < tiles_; ++i) {
            const uint32_t n = i + 1 == tiles_ ? tail_ : tile_;
            CopyIn(i, n);
            Compute(n);
            CopyOut(i, n);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t progress, uint32_t n)
    {
        auto p = predictionsQueue_.AllocTensor<IoT>();
        auto t = targetsQueue_.AllocTensor<IoT>();
        DataCopyExtParams params{1, n * static_cast<uint32_t>(sizeof(IoT)), 0, 0, 0};
        DataCopyPadExtParams<IoT> pad{true, 0, 0, static_cast<IoT>(0)};
        DataCopyPad(p, predictionsGM_[progress * tile_], params, pad);
        DataCopyPad(t, targetsGM_[progress * tile_], params, pad);
        predictionsQueue_.EnQue(p);
        targetsQueue_.EnQue(t);
    }
    __aicore__ inline void Compute(uint32_t n)
    {
        auto p = predictionsQueue_.DeQue<IoT>();
        auto t = targetsQueue_.DeQue<IoT>();
        auto out = lossQueue_.AllocTensor<IoT>();
        auto diff = diffBuf_.Get<float>();
        auto abs = absBuf_.Get<float>();
        auto quadratic = quadraticBuf_.Get<float>();
        auto linear = linearBuf_.Get<float>();
        auto mask = maskBuf_.Get<uint8_t>();
        if constexpr (kUpcast) {
            auto pf = predictionsFloatBuf_.Get<float>();
            auto tf = targetsFloatBuf_.Get<float>();
            Cast(pf, p, RoundMode::CAST_NONE, n);
            Cast(tf, t, RoundMode::CAST_NONE, n);
            PipeBarrier<PIPE_V>();
            Sub(diff, pf, tf, n);
        } else {
            Sub(diff, p, t, n);
        }
        PipeBarrier<PIPE_V>();
        Abs(abs, diff, n);
        PipeBarrier<PIPE_V>();
        Mul(quadratic, diff, diff, n);
        PipeBarrier<PIPE_V>();
        Muls(quadratic, quadratic, 0.5f, n);
        PipeBarrier<PIPE_V>();
        Adds(linear, abs, -0.5f * delta_, n);
        PipeBarrier<PIPE_V>();
        Muls(linear, linear, delta_, n);
        PipeBarrier<PIPE_V>();
        constexpr uint64_t kMask = 64;
        const uint64_t width = n < kMask ? n : kMask;
        const int32_t repeat = static_cast<int32_t>((n + width - 1) / width);
        UnaryRepeatParams cmpParams{1, 1, 8, 8};
        CompareScalar(mask, abs, delta_, CMPMODE::LE, width, repeat, cmpParams);
        PipeBarrier<PIPE_V>();
        BinaryRepeatParams selParams{1, 1, 1, 8, 8, 8};
        if constexpr (kUpcast) {
            Select(diff, mask, quadratic, linear, SELMODE::VSEL_TENSOR_TENSOR_MODE, width, repeat, selParams);
            PipeBarrier<PIPE_V>();
            Cast(out, diff, RoundMode::CAST_RINT, n);
        } else {
            Select(out, mask, quadratic, linear, SELMODE::VSEL_TENSOR_TENSOR_MODE, width, repeat, selParams);
        }
        lossQueue_.EnQue(out);
        predictionsQueue_.FreeTensor(p);
        targetsQueue_.FreeTensor(t);
    }
    __aicore__ inline void CopyOut(uint32_t progress, uint32_t n)
    {
        auto out = lossQueue_.DeQue<IoT>();
        DataCopyExtParams params{1, n * static_cast<uint32_t>(sizeof(IoT)), 0, 0, 0};
        DataCopyPad(lossGM_[progress * tile_], out, params);
        lossQueue_.FreeTensor(out);
    }
    TPipe pipe_;
    TQue<QuePosition::VECIN, kBufferNum> predictionsQueue_, targetsQueue_;
    TQue<QuePosition::VECOUT, kBufferNum> lossQueue_;
    TBuf<TPosition::VECCALC> predictionsFloatBuf_, targetsFloatBuf_, diffBuf_, absBuf_, quadraticBuf_, linearBuf_,
        maskBuf_;
    GlobalTensor<IoT> predictionsGM_, targetsGM_, lossGM_;
    uint64_t offset_ = 0;
    uint32_t count_ = 0, tiles_ = 0, tile_ = 0, tail_ = 0;
    float delta_ = 1.0f;
};
} // namespace NsHuberLoss
#endif

/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file apply_gradient_descent.h
 * \brief apply_gradient_descent classic (ascend910b) elementwise kernel:
 *        var_out = var - alpha * delta, fully elementwise, computed in fp32.
 */

#ifndef APPLY_GRADIENT_DESCENT_H
#define APPLY_GRADIENT_DESCENT_H

#include <type_traits>
#include "kernel_operator.h"
#include "apply_gradient_descent_tiling_data.h"

namespace ApplyGradientDescentClassic {
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t ONE_BLK_SIZE = 32;

template <AscendC::HardEvent hardEvent>
__aicore__ inline void PipeSync()
{
    int32_t eventID = static_cast<int32_t>(GetTPipePtr()->FetchEventID(hardEvent));
    AscendC::SetFlag<hardEvent>(eventID);
    AscendC::WaitFlag<hardEvent>(eventID);
}

template <typename T>
class ApplyGradientDescentKernel {
public:
    __aicore__ inline ApplyGradientDescentKernel(TPipe* pipe) : pipe_(pipe){};
    __aicore__ inline void Init(GM_ADDR var, GM_ADDR alpha, GM_ADDR delta, GM_ADDR var_out,
                                const ApplyGradientDescentTilingData& tiling);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ReadAlpha(GM_ADDR alpha);
    __aicore__ inline void ComputeTile(uint64_t offset, uint64_t count);

    TPipe* pipe_;
    TQue<QuePosition::VECIN, BUFFER_NUM> varInQue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> deltaInQue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQue_;
    TBuf<TPosition::VECCALC> varF32Buf_;
    TBuf<TPosition::VECCALC> deltaF32Buf_;
    TBuf<TPosition::VECCALC> alphaTBuf_;
    TBuf<TPosition::VECCALC> alphaF32Buf_;

    GlobalTensor<T> gmVar_;
    GlobalTensor<T> gmDelta_;
    GlobalTensor<T> gmVarOut_;

    float alphaVal_ = 0.0f;
    uint64_t startElem_ = 0;
    uint64_t coreDataCount_ = 0;
    uint64_t tileDataCount_ = 0;
};

template <typename T>
__aicore__ inline void ApplyGradientDescentKernel<T>::ReadAlpha(GM_ADDR alpha)
{
    GlobalTensor<T> gmAlpha;
    gmAlpha.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(alpha), 1);
    if constexpr (std::is_same_v<T, bfloat16_t>) {
        // BiSheng rejects bfloat16_t -> float C++ scalar casts, so route the single
        // element through the vector Cast path (mirrors apply_add_sign / apply_adadelta).
        LocalTensor<T> alphaT = alphaTBuf_.Get<T>();
        LocalTensor<float> alphaF = alphaF32Buf_.Get<float>();
        DataCopyParams copyParams = {1, static_cast<uint16_t>(sizeof(T)), 0, 0};
        DataCopyPadParams padParams = {false, 0, 0, 0};
        DataCopyPad(alphaT, gmAlpha, copyParams, padParams);
        PipeSync<AscendC::HardEvent::MTE2_V>();
        Cast(alphaF, alphaT, RoundMode::CAST_NONE, 1);
        PipeSync<AscendC::HardEvent::V_S>();
        alphaVal_ = alphaF.GetValue(0);
    } else {
        // float / half: direct scalar load from GM (half has an implicit float conversion).
        alphaVal_ = static_cast<float>(gmAlpha.GetValue(0));
    }
}

template <typename T>
__aicore__ inline void ApplyGradientDescentKernel<T>::Init(GM_ADDR var, GM_ADDR alpha, GM_ADDR delta, GM_ADDR var_out,
                                                           const ApplyGradientDescentTilingData& tiling)
{
    uint32_t blockIdx = GetBlockIdx();
    tileDataCount_ = tiling.tileDataCount;

    uint64_t startBlock;
    uint64_t myBlocks;
    if (blockIdx < tiling.remCoreNum) {
        myBlocks = tiling.blocksPerCore + 1;
        startBlock = static_cast<uint64_t>(blockIdx) * myBlocks;
    } else {
        myBlocks = tiling.blocksPerCore;
        startBlock = static_cast<uint64_t>(blockIdx) * tiling.blocksPerCore + tiling.remCoreNum;
    }
    startElem_ = startBlock * tiling.blockElems;
    coreDataCount_ = myBlocks * tiling.blockElems;

    if (startElem_ >= tiling.totalDataCount) {
        startElem_ = tiling.totalDataCount;
        coreDataCount_ = 0;
    } else if (startElem_ + coreDataCount_ > tiling.totalDataCount) {
        coreDataCount_ = tiling.totalDataCount - startElem_;
    }

    gmVar_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(var));
    gmDelta_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(delta));
    gmVarOut_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(var_out));

    pipe_->InitBuffer(varInQue_, BUFFER_NUM, tileDataCount_ * sizeof(T));
    pipe_->InitBuffer(deltaInQue_, BUFFER_NUM, tileDataCount_ * sizeof(T));
    pipe_->InitBuffer(outQue_, BUFFER_NUM, tileDataCount_ * sizeof(T));
    if constexpr (!std::is_same_v<T, float>) {
        pipe_->InitBuffer(varF32Buf_, tileDataCount_ * sizeof(float));
        pipe_->InitBuffer(deltaF32Buf_, tileDataCount_ * sizeof(float));
    }
    if constexpr (std::is_same_v<T, bfloat16_t>) {
        pipe_->InitBuffer(alphaTBuf_, ONE_BLK_SIZE);
        pipe_->InitBuffer(alphaF32Buf_, ONE_BLK_SIZE);
    }

    ReadAlpha(alpha);
}

template <typename T>
__aicore__ inline void ApplyGradientDescentKernel<T>::ComputeTile(uint64_t offset, uint64_t count)
{
    DataCopyParams copyParams = {1, static_cast<uint16_t>(count * sizeof(T)), 0, 0};
    DataCopyPadParams padParams = {false, 0, 0, 0};

    LocalTensor<T> varLocal = varInQue_.AllocTensor<T>();
    LocalTensor<T> deltaLocal = deltaInQue_.AllocTensor<T>();
    DataCopyPad(varLocal, gmVar_[offset], copyParams, padParams);
    DataCopyPad(deltaLocal, gmDelta_[offset], copyParams, padParams);
    varInQue_.EnQue(varLocal);
    deltaInQue_.EnQue(deltaLocal);

    varLocal = varInQue_.DeQue<T>();
    deltaLocal = deltaInQue_.DeQue<T>();
    LocalTensor<T> outLocal = outQue_.AllocTensor<T>();

    if constexpr (std::is_same_v<T, float>) {
        // var_out = var - alpha * delta, all in fp32
        Muls(deltaLocal, deltaLocal, alphaVal_, count);
        PipeBarrier<PIPE_V>();
        Sub(outLocal, varLocal, deltaLocal, count);
        PipeBarrier<PIPE_V>();
    } else {
        LocalTensor<float> varF32 = varF32Buf_.Get<float>();
        LocalTensor<float> deltaF32 = deltaF32Buf_.Get<float>();
        Cast(varF32, varLocal, RoundMode::CAST_NONE, count);
        Cast(deltaF32, deltaLocal, RoundMode::CAST_NONE, count);
        PipeBarrier<PIPE_V>();
        // Fuse Muls + Sub into one MAC: varF32 = deltaF32 * (-alpha) + varF32 = var - alpha * delta.
        // One fewer vector op and one fewer PipeBarrier than the separate Muls/Sub chain, which
        // shortens the (fp32-precision) compute chain that competes with MTE2 on the 2-byte path.
        Axpy(varF32, deltaF32, -alphaVal_, static_cast<int32_t>(count));
        PipeBarrier<PIPE_V>();
        Cast(outLocal, varF32, RoundMode::CAST_RINT, count);
        PipeBarrier<PIPE_V>();
    }

    varInQue_.FreeTensor(varLocal);
    deltaInQue_.FreeTensor(deltaLocal);
    outQue_.EnQue(outLocal);

    outLocal = outQue_.DeQue<T>();
    DataCopyParams outCopyParams = {1, static_cast<uint16_t>(count * sizeof(T)), 0, 0};
    DataCopyPad(gmVarOut_[offset], outLocal, outCopyParams);
    outQue_.FreeTensor(outLocal);
}

template <typename T>
__aicore__ inline void ApplyGradientDescentKernel<T>::Process()
{
    if (coreDataCount_ == 0) {
        return;
    }
    uint64_t loopNum = (coreDataCount_ + tileDataCount_ - 1) / tileDataCount_;
    for (uint64_t i = 0; i < loopNum; i++) {
        uint64_t offset = startElem_ + i * tileDataCount_;
        uint64_t count = tileDataCount_;
        uint64_t remain = coreDataCount_ - i * tileDataCount_;
        if (remain < count) {
            count = remain;
        }
        ComputeTile(offset, count);
    }
}

} // namespace ApplyGradientDescentClassic

#endif // APPLY_GRADIENT_DESCENT_H

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
 * \file swiglu_static_group_quant_hifp8.h
 * \brief
 */

#ifndef SWIGLU_STATIC_GROUP_QUANT_HIFP8_H
#define SWIGLU_STATIC_GROUP_QUANT_HIFP8_H

#include "swiglu_group_quant_hifp8_base.h"

namespace SwigluGroupQuantStaticHifp8Ops {
using namespace AscendC;

template <typename T>
class SwigluGroupQuantStaticHifp8Kernel
    : public SwigluGroupQuantHifp8BaseOps::SwigluGroupQuantHifp8KernelBase<SwigluGroupQuantStaticHifp8Kernel<T>, T> {
public:
    __aicore__ inline SwigluGroupQuantStaticHifp8Kernel() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR groupIndex, GM_ADDR scale, GM_ADDR y,
                                GM_ADDR yOrigin, GM_ADDR workspace, const SwigluGroupQuantHifp8TilingData* tilingData,
                                TPipe* pipe);

public:
    __aicore__ inline void ProcessGroupQuant();
    __aicore__ inline void ProcessNonGroupQuant();
    __aicore__ inline void ProcessGroup(int64_t groupIdx, int64_t tokenStart, int64_t tokenEnd);

protected:
    __aicore__ inline void ProcessTokensWithScale(int64_t tokenStart, int64_t tokenEnd, float invScale);
    __aicore__ inline void WaitMte3ToV();

    GlobalTensor<float> scaleGm_;
};

template <typename T>
__aicore__ inline void SwigluGroupQuantStaticHifp8Kernel<T>::Init(GM_ADDR x, GM_ADDR weight, GM_ADDR groupIndex,
                                                                  GM_ADDR scale, GM_ADDR y, GM_ADDR yOrigin,
                                                                  GM_ADDR workspace,
                                                                  const SwigluGroupQuantHifp8TilingData* tilingData,
                                                                  TPipe* pipe)
{
    this->InitBase(x, weight, groupIndex, y, yOrigin, workspace, tilingData, pipe);
    scaleGm_.SetGlobalBuffer((__gm__ float*)scale);
}

template <typename T>
__aicore__ inline void SwigluGroupQuantStaticHifp8Kernel<T>::ProcessNonGroupQuant()
{
    this->CalcNonGroupTokenRange();

    float invScale = scaleGm_.GetValue(0);
    ProcessTokensWithScale(this->tokenStart_, this->tokenEnd_, invScale);
}

template <typename T>
__aicore__ inline void SwigluGroupQuantStaticHifp8Kernel<T>::ProcessGroupQuant()
{
    this->ForEachGroup();
    this->FillRemainingTokens();
}

template <typename T>
__aicore__ inline void SwigluGroupQuantStaticHifp8Kernel<T>::ProcessGroup(int64_t groupIdx, int64_t tokenStart,
                                                                          int64_t tokenEnd)
{
    float invScale = scaleGm_.GetValue(groupIdx);
    ProcessTokensWithScale(tokenStart, tokenEnd, invScale);
}

template <typename T>
__aicore__ inline void SwigluGroupQuantStaticHifp8Kernel<T>::ProcessTokensWithScale(int64_t tokenStart,
                                                                                    int64_t tokenEnd, float invScale)
{
    int64_t tokenIdx = tokenStart;
    while (tokenIdx < tokenEnd) {
        int64_t tileEnd = AscendC::Std::min(tokenIdx + this->tileTokens_, tokenEnd);
        int64_t curTileTokens = tileEnd - tokenIdx;

        this->CopyIn(tokenIdx, curTileTokens);
        LocalTensor<float> xFloatLocalTensor = this->xQueue_.template DeQue<float>();
        this->ComputeSwiGLU(xFloatLocalTensor, curTileTokens);
        if (this->outputOrigin_) {
            this->CopyOutOrigin(xFloatLocalTensor, tokenIdx, curTileTokens);
            PipeBarrier<PIPE_MTE3>();
            WaitMte3ToV();
        }
        this->QuantizeOut(xFloatLocalTensor, tokenIdx, curTileTokens, invScale);
        this->xQueue_.template FreeTensor<float>(xFloatLocalTensor);
        tokenIdx += curTileTokens;
    }
}

template <typename T>
__aicore__ inline void SwigluGroupQuantStaticHifp8Kernel<T>::WaitMte3ToV()
{
    event_t event = static_cast<event_t>(this->pipe_->template AllocEventID<HardEvent::MTE3_V>());
    SetFlag<HardEvent::MTE3_V>(event);
    WaitFlag<HardEvent::MTE3_V>(event);
    this->pipe_->template ReleaseEventID<AscendC::HardEvent::MTE3_V>(event);
}

} // namespace SwigluGroupQuantStaticHifp8Ops
#endif // SWIGLU_STATIC_GROUP_QUANT_HIFP8_H
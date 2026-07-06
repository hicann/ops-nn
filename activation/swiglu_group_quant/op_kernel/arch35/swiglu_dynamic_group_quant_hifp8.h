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
 * \file swiglu_dynamic_group_quant_hifp8.h
 * \brief
 */

#ifndef SWIGLU_DYNAMIC_GROUP_QUANT_HIFP8_H
#define SWIGLU_DYNAMIC_GROUP_QUANT_HIFP8_H

#include "swiglu_group_quant_hifp8_base.h"

namespace SwigluGroupQuantDynamicHifp8Ops {
using namespace AscendC;

constexpr float EPS_NON_GROUP = 1e-8f;

template <typename T>
class SwigluGroupQuantDynamicHifp8Kernel
    : public SwigluGroupQuantHifp8BaseOps::SwigluGroupQuantHifp8KernelBase<SwigluGroupQuantDynamicHifp8Kernel<T>, T> {
public:
    __aicore__ inline SwigluGroupQuantDynamicHifp8Kernel() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR groupIndex, GM_ADDR y, GM_ADDR yScale,
                                GM_ADDR yOrigin, GM_ADDR workspace, const SwigluGroupQuantHifp8TilingData* tilingData,
                                TPipe* pipe);

public:
    __aicore__ inline void ProcessGroupQuant();
    __aicore__ inline void ProcessNonGroupQuant();
    __aicore__ inline void ProcessGroup(int64_t groupIdx, int64_t tokenStart, int64_t tokenEnd);

protected:
    __aicore__ inline void ProcessNonGroupQuantAndScale(int64_t tokenStart, int64_t tokenEnd);
    __aicore__ inline void ProcessCoreMax(int64_t tokenStart, int64_t tokenEnd, GlobalTensor<float> coreMaxGm);
    __aicore__ inline void ProcessSwigluQuant(int64_t tokenStart, int64_t tokenEnd);
    __aicore__ inline void ComputeTileMax(LocalTensor<float>& reduceMaxLocalTensor,
                                          LocalTensor<float>& xFloatLocalTensor, int64_t tileIdx,
                                          int64_t curTileTokens);
    __aicore__ inline void CopyOutCoreMax(LocalTensor<float>& reduceMaxLocalTensor, GlobalTensor<float> coreMaxGm);
    __aicore__ inline void ComputeGlobalMax();
    __aicore__ inline void WaitVToS();
    __aicore__ inline void WaitSToMte3();
    __aicore__ inline void WaitMte3ToS();

    GlobalTensor<float> yScaleGm_;
    GlobalTensor<float> coreMaxGm_;
    TQue<TPosition::VECIN, 1> reduceMaxQueue_;
    float globalScale_ = 0.;
    float dstTypeMax_ = 15.0f;
};

template <typename T>
__aicore__ inline void SwigluGroupQuantDynamicHifp8Kernel<T>::Init(GM_ADDR x, GM_ADDR weight, GM_ADDR groupIndex,
                                                                   GM_ADDR y, GM_ADDR yScale, GM_ADDR yOrigin,
                                                                   GM_ADDR workspace,
                                                                   const SwigluGroupQuantHifp8TilingData* tilingData,
                                                                   TPipe* pipe)
{
    dstTypeMax_ = tilingData->dstTypeMax;
    this->InitBase(x, weight, groupIndex, y, yOrigin, workspace, tilingData, pipe);
    this->pipe_->InitBuffer(reduceMaxQueue_, 1, this->tileLength_ * sizeof(float));
    yScaleGm_.SetGlobalBuffer((__gm__ float*)yScale);
    if (!this->isGroup_) {
        coreMaxGm_.SetGlobalBuffer((__gm__ float*)(this->workspace_));
    }
}

template <typename T>
__aicore__ inline void SwigluGroupQuantDynamicHifp8Kernel<T>::ProcessNonGroupQuant()
{
    this->CalcNonGroupTokenRange();
    this->numTiles_ = this->CeilDiv(this->tokenEnd_ - this->tokenStart_, this->tileTokens_);

    ProcessNonGroupQuantAndScale(this->tokenStart_, this->tokenEnd_);
}

template <typename T>
__aicore__ inline void SwigluGroupQuantDynamicHifp8Kernel<T>::ProcessNonGroupQuantAndScale(int64_t tokenStart,
                                                                                           int64_t tokenEnd)
{
    ProcessCoreMax(tokenStart, tokenEnd, coreMaxGm_[this->blockIdx_]);
    SyncAll();

    ComputeGlobalMax();
    SyncAll();

    ProcessSwigluQuant(tokenStart, tokenEnd);
}

template <typename T>
__aicore__ inline void SwigluGroupQuantDynamicHifp8Kernel<T>::ProcessCoreMax(int64_t tokenStart, int64_t tokenEnd,
                                                                             GlobalTensor<float> coreMaxGm)
{
    LocalTensor<float> reduceMaxLocalTensor = reduceMaxQueue_.template AllocTensor<float>();
    int64_t tokenIdx = tokenStart, tileIdx = 0;
    while (tokenIdx < tokenEnd) {
        int64_t tileEnd = AscendC::Std::min(tokenIdx + this->tileTokens_, tokenEnd);
        int64_t curTileTokens = tileEnd - tokenIdx;

        this->CopyIn(tokenIdx, curTileTokens);
        LocalTensor<float> xFloatLocalTensor = this->xQueue_.template DeQue<float>();
        this->ComputeSwiGLU(xFloatLocalTensor, curTileTokens);
        ComputeTileMax(reduceMaxLocalTensor, xFloatLocalTensor, tileIdx, curTileTokens);
        if (this->outputOrigin_) {
            this->CopyOutOrigin(xFloatLocalTensor, tokenIdx, curTileTokens);
            PipeBarrier<PIPE_MTE3>();
        }
        this->xQueue_.template FreeTensor<float>(xFloatLocalTensor);
        tokenIdx += curTileTokens;
        tileIdx += 1;
    }
    if (this->isGroup_) {
        WaitVToS();
        globalScale_ = AscendC::Std::max(reduceMaxLocalTensor.GetValue(0), EPS_NON_GROUP);
        reduceMaxLocalTensor.SetValue(0, globalScale_ / dstTypeMax_);
        WaitSToMte3();
    }
    CopyOutCoreMax(reduceMaxLocalTensor, coreMaxGm);
    reduceMaxQueue_.template FreeTensor<float>(reduceMaxLocalTensor);
}

template <typename T>
__aicore__ inline void SwigluGroupQuantDynamicHifp8Kernel<T>::ProcessSwigluQuant(int64_t tokenStart, int64_t tokenEnd)
{
    int64_t tokenIdx = tokenStart;
    while (tokenIdx < tokenEnd) {
        int64_t tileEnd = AscendC::Std::min(tokenIdx + this->tileTokens_, tokenEnd);
        int64_t curTileTokens = tileEnd - tokenIdx;

        this->CopyIn(tokenIdx, curTileTokens);
        LocalTensor<float> xFloatLocalTensor = this->xQueue_.template DeQue<float>();
        this->ComputeSwiGLU(xFloatLocalTensor, curTileTokens);
        float divScale;
        if (this->isGroup_) {
            divScale = dstTypeMax_ / globalScale_;
        } else {
            divScale = 1.0f / yScaleGm_.GetValue(0);
        }
        this->QuantizeOut(xFloatLocalTensor, tokenIdx, curTileTokens, divScale);
        this->xQueue_.template FreeTensor<float>(xFloatLocalTensor);
        tokenIdx += curTileTokens;
    }
}

template <typename T>
__aicore__ inline void SwigluGroupQuantDynamicHifp8Kernel<T>::ProcessGroupQuant()
{
    this->ForEachGroup();
    this->FillRemainingTokens();
}

template <typename T>
__aicore__ inline void SwigluGroupQuantDynamicHifp8Kernel<T>::ProcessGroup(int64_t groupIdx, int64_t tokenStart,
                                                                           int64_t tokenEnd)
{
    int64_t tokensInGroup = tokenEnd - tokenStart;
    this->numTiles_ = this->CeilDiv(tokensInGroup, this->tileTokens_);
    ProcessCoreMax(tokenStart, tokenEnd, yScaleGm_[groupIdx]);
    if (this->outputOrigin_) {
        PipeBarrier<PIPE_MTE3>();
    }
    ProcessSwigluQuant(tokenStart, tokenEnd);
}

template <typename T>
__aicore__ inline void SwigluGroupQuantDynamicHifp8Kernel<T>::ComputeTileMax(LocalTensor<float>& reduceMaxLocalTensor,
                                                                             LocalTensor<float>& xFloatLocalTensor,
                                                                             int64_t tileIdx, int64_t curTileTokens)
{
    int64_t computeSize = curTileTokens * this->dimH_;
    LocalTensor<float> x0FloatLocalTensor = xFloatLocalTensor;
    LocalTensor<float>
        tmpLocalTensor = xFloatLocalTensor[this->tileLength_ * SwigluGroupQuantHifp8BaseOps::TMP_BUFFER_INDEX];
    Abs(tmpLocalTensor, x0FloatLocalTensor, static_cast<uint32_t>(computeSize));
    PipeBarrier<PIPE_V>();
    Maxs(tmpLocalTensor, tmpLocalTensor, EPS_NON_GROUP, static_cast<uint32_t>(computeSize));
    PipeBarrier<PIPE_V>();
    if (tileIdx == 0) {
        Copy(reduceMaxLocalTensor, tmpLocalTensor, static_cast<uint32_t>(computeSize));
        PipeBarrier<PIPE_V>();
    } else {
        Max(reduceMaxLocalTensor, reduceMaxLocalTensor, tmpLocalTensor, static_cast<uint32_t>(computeSize));
        PipeBarrier<PIPE_V>();
    }

    if (tileIdx == this->numTiles_ - 1) {
        uint32_t reduceCount = (this->numTiles_ > 1) ? static_cast<uint32_t>(this->tileLength_) :
                                                       static_cast<uint32_t>(computeSize);
        ReduceMax<float>(reduceMaxLocalTensor, reduceMaxLocalTensor, tmpLocalTensor, reduceCount, false);
        PipeBarrier<PIPE_V>();
    }
}

template <typename T>
__aicore__ inline void SwigluGroupQuantDynamicHifp8Kernel<T>::CopyOutCoreMax(LocalTensor<float>& reduceMaxLocalTensor,
                                                                             GlobalTensor<float> coreMaxGm)
{
    this->WaitVToMte3();
    DataCopyParams copyParams(1, sizeof(float), 0, 0);
    DataCopyPad(coreMaxGm, reduceMaxLocalTensor, copyParams);
    WaitMte3ToS();
}

template <typename T>
__aicore__ inline void SwigluGroupQuantDynamicHifp8Kernel<T>::ComputeGlobalMax()
{
    if (this->blockIdx_ == 0) {
        LocalTensor<float> reduceMaxLocalTensor = reduceMaxQueue_.template AllocTensor<float>();
        LocalTensor<float> tmpLocalTensor = this->xQueue_.template AllocTensor<float>();
        DataCopyParams coreMaxCopyParams(1, static_cast<uint32_t>(this->usedCoreNum_) * sizeof(float), 0, 0);
        DataCopyPadParams padParams{false, 0, 0, 0};
        DataCopyPad(reduceMaxLocalTensor, coreMaxGm_, coreMaxCopyParams, padParams);
        reduceMaxQueue_.EnQue<float>(reduceMaxLocalTensor);
        reduceMaxLocalTensor = reduceMaxQueue_.template DeQue<float>();
        ReduceMax<float>(reduceMaxLocalTensor, reduceMaxLocalTensor, tmpLocalTensor,
                         static_cast<uint32_t>(this->usedCoreNum_), false);
        PipeBarrier<PIPE_V>();
        Divs(reduceMaxLocalTensor, reduceMaxLocalTensor, dstTypeMax_, 1);
        this->WaitVToMte3();
        DataCopyParams yScaleCopyOutParams(1, sizeof(float), 0, 0);
        DataCopyPad(yScaleGm_, reduceMaxLocalTensor, yScaleCopyOutParams);
        WaitMte3ToS();
        reduceMaxQueue_.template FreeTensor<float>(reduceMaxLocalTensor);
        this->xQueue_.template FreeTensor<float>(tmpLocalTensor);
    }
}

template <typename T>
__aicore__ inline void SwigluGroupQuantDynamicHifp8Kernel<T>::WaitVToS()
{
    event_t event = static_cast<event_t>(this->pipe_->template AllocEventID<HardEvent::V_S>());
    SetFlag<HardEvent::V_S>(event);
    WaitFlag<HardEvent::V_S>(event);
    this->pipe_->template ReleaseEventID<AscendC::HardEvent::V_S>(event);
}

template <typename T>
__aicore__ inline void SwigluGroupQuantDynamicHifp8Kernel<T>::WaitSToMte3()
{
    event_t event = static_cast<event_t>(this->pipe_->template AllocEventID<HardEvent::S_MTE3>());
    SetFlag<HardEvent::S_MTE3>(event);
    WaitFlag<HardEvent::S_MTE3>(event);
    this->pipe_->template ReleaseEventID<AscendC::HardEvent::S_MTE3>(event);
}

template <typename T>
__aicore__ inline void SwigluGroupQuantDynamicHifp8Kernel<T>::WaitMte3ToS()
{
    event_t event = static_cast<event_t>(this->pipe_->template AllocEventID<HardEvent::MTE3_S>());
    SetFlag<HardEvent::MTE3_S>(event);
    WaitFlag<HardEvent::MTE3_S>(event);
    this->pipe_->template ReleaseEventID<AscendC::HardEvent::MTE3_S>(event);
}

} // namespace SwigluGroupQuantDynamicHifp8Ops
#endif // SWIGLU_DYNAMIC_GROUP_QUANT_HIFP8_H

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
 * \file swiglu_group_quant_hifp8_base.h
 * \brief
 */

#ifndef SWIGLU_GROUP_QUANT_HIFP8_BASE_H
#define SWIGLU_GROUP_QUANT_HIFP8_BASE_H

#include "kernel_operator.h"

namespace SwigluGroupQuantHifp8BaseOps {
using namespace AscendC;

constexpr int64_t MAX_CORE_COUNT = 64;
constexpr int64_t INT64_MAX_VAL = 0x7FFFFFFFFFFFFFFF;
constexpr uint32_t TMP_BUFFER_INDEX = 2;
constexpr uint32_t QUANT_TILE_BUFFER_NUM = 3;
constexpr uint32_t DB_BUFFER = 2;

template <typename Derived, typename T>
class SwigluGroupQuantHifp8KernelBase {
public:
    __aicore__ inline SwigluGroupQuantHifp8KernelBase() {}

    __aicore__ inline void InitBase(GM_ADDR x, GM_ADDR weight, GM_ADDR groupIndex, GM_ADDR y, GM_ADDR yOrigin,
                                    GM_ADDR workspace, const SwigluGroupQuantHifp8TilingData* tilingData, TPipe* pipe);
    __aicore__ inline void Process();

protected:
    __aicore__ inline void CalcCoreGroupDistribution();
    __aicore__ inline void CalcMinLoadCoreIdx();
    __aicore__ inline void CalcNonGroupTokenRange();
    __aicore__ inline void WaitVToMte3();
    __aicore__ inline void WaitMte2ToS();
    __aicore__ inline void WaitSToV();
    __aicore__ inline void WaitMte3ToMte2();
    __aicore__ inline static int64_t CeilDiv(int64_t x, int64_t y) { return y == 0 ? 0 : (x + y - 1) / y; }

    __aicore__ inline void CopyIn(int64_t tokenIdx, int64_t curTileTokens);
    __aicore__ inline void ComputeSwiGLU(LocalTensor<float>& xFloatLocalTensor, int64_t curTileTokens);
    __aicore__ inline void QuantizeOut(LocalTensor<float>& xFloatLocalTensor, int64_t tokenIdx, int64_t curTileTokens,
                                       float divScale);
    __aicore__ inline void CopyOutOrigin(LocalTensor<float>& xFloatLocalTensor, int64_t tokenIdx,
                                         int64_t curTileTokens);
    __aicore__ inline void FillYZero(int64_t tokenStart, int64_t tokenEnd);
    __aicore__ inline void ProcessOutputOrigin(int64_t tokenStart, int64_t tokenEnd);
    __aicore__ inline void ForEachGroup();
    __aicore__ inline void FillRemainingTokens();

    GlobalTensor<T> xGm_;
    GlobalTensor<T> yOriginGm_;
    GlobalTensor<float> weightGm_;
    GlobalTensor<hifloat8_t> yGm_;
    GlobalTensor<int64_t> groupIndexGm_;

    TPipe* pipe_ = nullptr;
    TQue<TPosition::VECIN, DB_BUFFER> xQueue_;
    TQue<TPosition::VECIN, DB_BUFFER> weightQueue_;

    GM_ADDR workspace_ = nullptr;
    int64_t tileLength_ = 0;
    int64_t numTiles_ = 0;
    int64_t totalTokens_ = 0;
    int64_t dim2H_ = 0;
    int64_t dimH_ = 0;
    int64_t tileTokens_ = 0;
    int64_t usedCoreNum_ = 0;
    int64_t minLoadCoreIdx_ = 0;
    int64_t groupTokensSum_ = 0;
    uint32_t outputOrigin_ = 0;
    uint32_t isGroup_ = 0;
    uint32_t hasWeight_ = 0;
    uint32_t hasClamp_ = 0;
    uint32_t blockIdx_ = 0;
    float clampLimit_ = -1.0f;
    int64_t tokenStart_ = 0;
    int64_t tokenEnd_ = 0;
    int64_t coreGroupStartArr_[MAX_CORE_COUNT] = {0};
    int64_t coreGroupCountArr_[MAX_CORE_COUNT] = {0};
    const SwigluGroupQuantHifp8TilingData* tilingData_ = nullptr;
};

template <typename Derived, typename T>
__aicore__ inline void SwigluGroupQuantHifp8KernelBase<Derived, T>::WaitVToMte3()
{
    event_t event = static_cast<event_t>(pipe_->AllocEventID<HardEvent::V_MTE3>());
    SetFlag<HardEvent::V_MTE3>(event);
    WaitFlag<HardEvent::V_MTE3>(event);
    pipe_->ReleaseEventID<AscendC::HardEvent::V_MTE3>(event);
}

template <typename Derived, typename T>
__aicore__ inline void SwigluGroupQuantHifp8KernelBase<Derived, T>::WaitMte2ToS()
{
    event_t event = static_cast<event_t>(pipe_->AllocEventID<HardEvent::MTE2_S>());
    SetFlag<HardEvent::MTE2_S>(event);
    WaitFlag<HardEvent::MTE2_S>(event);
    pipe_->ReleaseEventID<AscendC::HardEvent::MTE2_S>(event);
}

template <typename Derived, typename T>
__aicore__ inline void SwigluGroupQuantHifp8KernelBase<Derived, T>::WaitSToV()
{
    event_t event = static_cast<event_t>(pipe_->AllocEventID<HardEvent::S_V>());
    SetFlag<HardEvent::S_V>(event);
    WaitFlag<HardEvent::S_V>(event);
    pipe_->ReleaseEventID<AscendC::HardEvent::S_V>(event);
}

template <typename Derived, typename T>
__aicore__ inline void SwigluGroupQuantHifp8KernelBase<Derived, T>::WaitMte3ToMte2()
{
    event_t event = static_cast<event_t>(pipe_->AllocEventID<HardEvent::MTE3_MTE2>());
    SetFlag<HardEvent::MTE3_MTE2>(event);
    WaitFlag<HardEvent::MTE3_MTE2>(event);
    pipe_->ReleaseEventID<AscendC::HardEvent::MTE3_MTE2>(event);
}

template <typename Derived, typename T>
__aicore__ inline void SwigluGroupQuantHifp8KernelBase<Derived, T>::CalcMinLoadCoreIdx()
{
    int64_t minLoad = INT64_MAX_VAL;
    int64_t minCoreIdx = 0;
    for (int64_t core = 0; core < usedCoreNum_; core++) {
        int64_t coreLoad = 0;
        for (int64_t gg = coreGroupStartArr_[core]; gg < coreGroupStartArr_[core] + coreGroupCountArr_[core]; gg++) {
            coreLoad += groupIndexGm_.GetValue(gg);
        }
        if (coreLoad < minLoad) {
            minLoad = coreLoad;
            minCoreIdx = core;
        }
    }
    minLoadCoreIdx_ = minCoreIdx;
}

template <typename Derived, typename T>
__aicore__ inline void SwigluGroupQuantHifp8KernelBase<Derived, T>::CalcNonGroupTokenRange()
{
    int64_t tokensPerCore = tilingData_->tokensPerCore;
    int64_t remainder = totalTokens_ - tokensPerCore * usedCoreNum_;
    if (blockIdx_ < remainder) {
        tokenStart_ = static_cast<int64_t>(blockIdx_) * (tokensPerCore + 1);
        tokenEnd_ = tokenStart_ + tokensPerCore + 1;
    } else {
        tokenStart_ = remainder * (tokensPerCore + 1) + (static_cast<int64_t>(blockIdx_) - remainder) * tokensPerCore;
        tokenEnd_ = tokenStart_ + tokensPerCore;
    }
}

template <typename Derived, typename T>
__aicore__ inline void SwigluGroupQuantHifp8KernelBase<Derived, T>::ForEachGroup()
{
    if (blockIdx_ >= usedCoreNum_) {
        return;
    }

    int64_t groupStart = coreGroupStartArr_[blockIdx_];
    int64_t groupCount = coreGroupCountArr_[blockIdx_];
    if (groupCount > 0) {
        int64_t tokenStart = 0;
        for (int64_t g = 0; g < groupStart; g++) {
            tokenStart += groupIndexGm_.GetValue(g);
        }

        for (int64_t g = 0; g < groupCount; g++) {
            int64_t groupIdx = groupStart + g;
            int64_t tokensInGroup = groupIndexGm_.GetValue(groupIdx);
            if (tokensInGroup == 0) {
                continue;
            }
            int64_t tokenEnd = tokenStart + tokensInGroup;
            auto* derived = static_cast<Derived*>(this);
            derived->ProcessGroup(groupIdx, tokenStart, tokenEnd);
            tokenStart = tokenEnd;
        }
    }
}

template <typename Derived, typename T>
__aicore__ inline void SwigluGroupQuantHifp8KernelBase<Derived, T>::FillRemainingTokens()
{
    if (blockIdx_ == minLoadCoreIdx_ && groupTokensSum_ < totalTokens_) {
        FillYZero(groupTokensSum_, totalTokens_);
        if (outputOrigin_) {
            ProcessOutputOrigin(groupTokensSum_, totalTokens_);
        }
    }
}

template <typename Derived, typename T>
__aicore__ inline void SwigluGroupQuantHifp8KernelBase<Derived, T>::CalcCoreGroupDistribution()
{
    int64_t groupNum = tilingData_->groupNum;
    for (int64_t i = 0; i < MAX_CORE_COUNT; i++) {
        coreGroupStartArr_[i] = 0;
        coreGroupCountArr_[i] = 0;
    }
    if (usedCoreNum_ == 0 || groupNum == 0) {
        return;
    }

    int64_t totalGroupTokens = 0;
    for (int64_t g = 0; g < groupNum; g++) {
        totalGroupTokens += groupIndexGm_.GetValue(g);
    }
    int64_t targetLoad = CeilDiv(totalGroupTokens, usedCoreNum_);

    int64_t g = 0;
    for (int64_t core = 0; core < usedCoreNum_; core++) {
        int64_t start = g;
        int64_t remainingCores = usedCoreNum_ - core - 1;
        if (core == usedCoreNum_ - 1) {
            g = groupNum;
        } else {
            int64_t coreLoad = 0;
            while (g < groupNum) {
                int64_t remainingGroups = groupNum - g;
                if (remainingGroups <= remainingCores) {
                    break;
                }
                coreLoad += groupIndexGm_.GetValue(g);
                g++;
                if (coreLoad >= targetLoad) {
                    break;
                }
            }
        }
        coreGroupStartArr_[core] = start;
        coreGroupCountArr_[core] = g - start;
    }

    groupTokensSum_ = totalGroupTokens;
    CalcMinLoadCoreIdx();
}

template <typename Derived, typename T>
__aicore__ inline void SwigluGroupQuantHifp8KernelBase<Derived, T>::CopyIn(int64_t tokenIdx, int64_t curTileTokens)
{
    LocalTensor<T> xTLocalTensor = xQueue_.AllocTensor<T>();
    int64_t copySize = curTileTokens * dimH_;
    DataCopyParams copyParams(static_cast<uint32_t>(curTileTokens), static_cast<uint32_t>(dimH_) * sizeof(T),
                              static_cast<uint32_t>(dim2H_ - dimH_) * sizeof(T), 0);
    DataCopyPadParams padParams{false, 0, 0, 0};
    int64_t x0GmOffset = tokenIdx * dim2H_;
    int64_t x1GmOffset = tokenIdx * dim2H_ + dimH_;
    if constexpr (std::is_same_v<T, float>) {
        DataCopyPad(xTLocalTensor, xGm_[x0GmOffset], copyParams, padParams);
        DataCopyPad(xTLocalTensor[tileLength_], xGm_[x1GmOffset], copyParams, padParams);
        xQueue_.EnQue<float>(xTLocalTensor);
    } else {
        int64_t fp16Base = TMP_BUFFER_INDEX * static_cast<int64_t>(tileLength_) * sizeof(float) / sizeof(T);
        DataCopyPad(xTLocalTensor[fp16Base], xGm_[x0GmOffset], copyParams, padParams);
        DataCopyPad(xTLocalTensor[fp16Base + copySize], xGm_[x1GmOffset], copyParams, padParams);
        xQueue_.EnQue<T>(xTLocalTensor);
        xTLocalTensor = xQueue_.DeQue<T>();
        LocalTensor<float> xFloatLocalTensor = xTLocalTensor.template ReinterpretCast<float>();
        Cast(xFloatLocalTensor, xTLocalTensor[fp16Base], RoundMode::CAST_NONE, static_cast<uint32_t>(copySize));
        Cast(xFloatLocalTensor[tileLength_], xTLocalTensor[fp16Base + copySize], RoundMode::CAST_NONE,
             static_cast<uint32_t>(copySize));
        PipeBarrier<PIPE_V>();
        xQueue_.EnQue<float>(xFloatLocalTensor);
    }

    if (hasWeight_) {
        LocalTensor<float> weightLocalTensor = weightQueue_.AllocTensor<float>();
        DataCopyParams weightCopyParams(1, static_cast<uint32_t>(curTileTokens) * sizeof(float), 0, 0);
        DataCopyPadParams weightPadParams{false, 0, 0, 0};
        DataCopyPad(weightLocalTensor, weightGm_[tokenIdx], weightCopyParams, weightPadParams);
        weightQueue_.EnQue<float>(weightLocalTensor);
    }
}

template <typename Derived, typename T>
__aicore__ inline void SwigluGroupQuantHifp8KernelBase<Derived, T>::ComputeSwiGLU(LocalTensor<float>& xFloatLocalTensor,
                                                                                  int64_t curTileTokens)
{
    int64_t computeSize = curTileTokens * dimH_;

    LocalTensor<float> x0FloatLocalTensor = xFloatLocalTensor;
    LocalTensor<float> x1FloatLocalTensor = xFloatLocalTensor[tileLength_];

    if (hasClamp_) {
        Mins(x0FloatLocalTensor, x0FloatLocalTensor, clampLimit_, static_cast<uint32_t>(computeSize));
        PipeBarrier<PIPE_V>();
        Maxs(x1FloatLocalTensor, x1FloatLocalTensor, -clampLimit_, static_cast<uint32_t>(computeSize));
        PipeBarrier<PIPE_V>();
        Mins(x1FloatLocalTensor, x1FloatLocalTensor, clampLimit_, static_cast<uint32_t>(computeSize));
        PipeBarrier<PIPE_V>();
    }

    LocalTensor<float> tmpLocalTensor = xFloatLocalTensor[tileLength_ * TMP_BUFFER_INDEX];
    Muls(tmpLocalTensor, x0FloatLocalTensor, -1.0f, static_cast<uint32_t>(computeSize));
    PipeBarrier<PIPE_V>();
    Exp(tmpLocalTensor, tmpLocalTensor, static_cast<uint32_t>(computeSize));
    PipeBarrier<PIPE_V>();
    Adds(tmpLocalTensor, tmpLocalTensor, 1.0f, static_cast<uint32_t>(computeSize));
    PipeBarrier<PIPE_V>();
    Div(x0FloatLocalTensor, x0FloatLocalTensor, tmpLocalTensor, static_cast<uint32_t>(computeSize));
    PipeBarrier<PIPE_V>();
    Mul(x0FloatLocalTensor, x0FloatLocalTensor, x1FloatLocalTensor, static_cast<uint32_t>(computeSize));
    PipeBarrier<PIPE_V>();

    if (hasWeight_) {
        LocalTensor<float> weightLocalTensor = weightQueue_.DeQue<float>();
        WaitMte2ToS();
        for (int64_t t = 0; t < curTileTokens; t++) {
            float weightVal = weightLocalTensor.GetValue(static_cast<uint32_t>(t));
            WaitSToV();
            Muls(x0FloatLocalTensor[t * dimH_], x0FloatLocalTensor[t * dimH_], weightVal, static_cast<uint32_t>(dimH_));
            PipeBarrier<PIPE_V>();
        }
        weightQueue_.FreeTensor(weightLocalTensor);
    }
}

template <typename Derived, typename T>
__aicore__ inline void SwigluGroupQuantHifp8KernelBase<Derived, T>::QuantizeOut(LocalTensor<float>& xFloatLocalTensor,
                                                                                int64_t tokenIdx, int64_t curTileTokens,
                                                                                float divScale)
{
    int64_t computeSize = curTileTokens * dimH_;
    WaitSToV();
    Muls(xFloatLocalTensor, xFloatLocalTensor, divScale, static_cast<uint32_t>(computeSize));
    PipeBarrier<PIPE_V>();
    LocalTensor<float> x0FloatLocalTensor = xFloatLocalTensor;
    LocalTensor<hifloat8_t> x0TLocalTensor = xFloatLocalTensor[tileLength_ * TMP_BUFFER_INDEX]
                                                 .template ReinterpretCast<hifloat8_t>();
    Cast(x0TLocalTensor, x0FloatLocalTensor, RoundMode::CAST_ROUND, static_cast<uint32_t>(computeSize));
    WaitVToMte3();
    int64_t gmOffset = tokenIdx * dimH_;
    DataCopyParams outCopyParams(1, static_cast<uint32_t>(computeSize) * sizeof(hifloat8_t), 0, 0);
    DataCopyPad(yGm_[gmOffset], x0TLocalTensor, outCopyParams);
    WaitMte3ToMte2();
}

template <typename Derived, typename T>
__aicore__ inline void SwigluGroupQuantHifp8KernelBase<Derived, T>::CopyOutOrigin(LocalTensor<float>& xFloatLocalTensor,
                                                                                  int64_t tokenIdx,
                                                                                  int64_t curTileTokens)
{
    int64_t gmOffset = tokenIdx * dimH_;
    int64_t copySize = curTileTokens * dimH_;
    DataCopyParams outCopyParams(1, static_cast<uint32_t>(copySize) * sizeof(T), 0, 0);
    LocalTensor<float> x0FloatLocalTensor = xFloatLocalTensor;

    if constexpr (std::is_same_v<T, float>) {
        WaitVToMte3();
        DataCopyPad(yOriginGm_[gmOffset], x0FloatLocalTensor, outCopyParams);
    } else {
        LocalTensor<T> x0TLocalTensor = xFloatLocalTensor[tileLength_ * TMP_BUFFER_INDEX].template ReinterpretCast<T>();
        Cast(x0TLocalTensor, x0FloatLocalTensor, RoundMode::CAST_RINT, static_cast<uint32_t>(copySize));
        WaitVToMte3();
        DataCopyPad(yOriginGm_[gmOffset], x0TLocalTensor, outCopyParams);
    }
    WaitMte3ToMte2();
}

template <typename Derived, typename T>
__aicore__ inline void SwigluGroupQuantHifp8KernelBase<Derived, T>::FillYZero(int64_t tokenStart, int64_t tokenEnd)
{
    LocalTensor<float> zeroFloatLocalTensor = xQueue_.AllocTensor<float>();
    Duplicate(zeroFloatLocalTensor, 0.0f, static_cast<uint32_t>(tileLength_));
    PipeBarrier<PIPE_V>();
    LocalTensor<hifloat8_t> zeroHif8LocalTensor = zeroFloatLocalTensor[tileLength_ * TMP_BUFFER_INDEX]
                                                      .template ReinterpretCast<hifloat8_t>();
    Cast(zeroHif8LocalTensor, zeroFloatLocalTensor, RoundMode::CAST_ROUND, static_cast<uint32_t>(tileLength_));
    WaitVToMte3();

    int64_t tokenIdx = tokenStart;
    while (tokenIdx < tokenEnd) {
        int64_t tileEnd = AscendC::Std::min(tokenIdx + tileTokens_, tokenEnd);
        int64_t copySize = (tileEnd - tokenIdx) * dimH_;
        DataCopyParams outCopyParams(1, static_cast<uint32_t>(copySize) * sizeof(hifloat8_t), 0, 0);
        DataCopyPad(yGm_[tokenIdx * dimH_], zeroHif8LocalTensor, outCopyParams);
        tokenIdx = tileEnd;
    }
    WaitMte3ToMte2();
    xQueue_.FreeTensor<float>(zeroFloatLocalTensor);
}

template <typename Derived, typename T>
__aicore__ inline void SwigluGroupQuantHifp8KernelBase<Derived, T>::ProcessOutputOrigin(int64_t tokenStart,
                                                                                        int64_t tokenEnd)
{
    int64_t tokenIdx = tokenStart;
    while (tokenIdx < tokenEnd) {
        int64_t tileEnd = AscendC::Std::min(tokenIdx + tileTokens_, tokenEnd);
        int64_t curTileTokens = tileEnd - tokenIdx;

        CopyIn(tokenIdx, curTileTokens);
        LocalTensor<float> xFloatLocalTensor = xQueue_.DeQue<float>();
        ComputeSwiGLU(xFloatLocalTensor, curTileTokens);
        CopyOutOrigin(xFloatLocalTensor, tokenIdx, curTileTokens);
        PipeBarrier<PIPE_MTE3>();
        xQueue_.FreeTensor<float>(xFloatLocalTensor);
        tokenIdx += curTileTokens;
    }
}

template <typename Derived, typename T>
__aicore__ inline void SwigluGroupQuantHifp8KernelBase<Derived, T>::InitBase(
    GM_ADDR x, GM_ADDR weight, GM_ADDR groupIndex, GM_ADDR y, GM_ADDR yOrigin, GM_ADDR workspace,
    const SwigluGroupQuantHifp8TilingData* tilingData, TPipe* pipe)
{
    this->tilingData_ = tilingData;
    this->pipe_ = pipe;
    this->totalTokens_ = tilingData->totalTokens;
    this->dim2H_ = tilingData->dim2H;
    this->dimH_ = tilingData->dimH;
    this->isGroup_ = tilingData->isGroup;
    this->hasWeight_ = tilingData->hasWeight;
    this->hasClamp_ = tilingData->hasClamp;
    this->clampLimit_ = tilingData->clampLimit;
    this->tileTokens_ = tilingData->tileTokens;
    this->usedCoreNum_ = tilingData->usedCoreNum;
    this->blockIdx_ = GetBlockIdx();
    this->workspace_ = workspace;
    this->outputOrigin_ = tilingData->outputOrigin;

    this->xGm_.SetGlobalBuffer((__gm__ T*)x);
    if (this->hasWeight_) {
        this->weightGm_.SetGlobalBuffer((__gm__ float*)weight);
    }
    this->yGm_.SetGlobalBuffer((__gm__ hifloat8_t*)y);
    if (this->isGroup_) {
        this->groupIndexGm_.SetGlobalBuffer((__gm__ int64_t*)groupIndex);
    }
    if (this->outputOrigin_ && yOrigin != nullptr) {
        this->yOriginGm_.SetGlobalBuffer((__gm__ T*)yOrigin);
    }

    this->tileLength_ = tilingData->tileLength;
    this->pipe_->InitBuffer(this->xQueue_, DB_BUFFER, this->tileLength_ * QUANT_TILE_BUFFER_NUM * sizeof(float));
    if (this->hasWeight_) {
        this->pipe_->InitBuffer(this->weightQueue_, DB_BUFFER, this->tileTokens_ * sizeof(float));
    }

    if (this->isGroup_) {
        this->CalcCoreGroupDistribution();
    }
}

template <typename Derived, typename T>
__aicore__ inline void SwigluGroupQuantHifp8KernelBase<Derived, T>::Process()
{
    auto* derived = static_cast<Derived*>(this);
    this->isGroup_ ? derived->ProcessGroupQuant() : derived->ProcessNonGroupQuant();
}

} // namespace SwigluGroupQuantHifp8BaseOps
#endif // SWIGLU_GROUP_QUANT_HIFP8_BASE_H

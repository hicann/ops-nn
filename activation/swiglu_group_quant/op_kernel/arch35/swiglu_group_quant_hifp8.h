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
 * \file swiglu_group_quant_hifp8.h
 * \brief
 */

#ifndef SWIGLU_GROUP_QUANT_HIFP8_H
#define SWIGLU_GROUP_QUANT_HIFP8_H

#include "kernel_operator.h"

namespace SwigluGroupQuantHifp8Ops {
using namespace AscendC;

constexpr int64_t MAX_CORE_COUNT = 64;
constexpr int64_t INT64_MAX_VAL = 0x7FFFFFFFFFFFFFFF;
constexpr float EPS_NON_GROUP = 1e-8f;
constexpr uint32_t TMP_BUFFER_INDEX = 2;
constexpr uint32_t QUANT_TILE_BUFFER_NUM = 3;
constexpr uint32_t DB_BUFFER = 2;

template <typename T>
class SwigluGroupQuantHifp8Kernel {
public:
    __aicore__ inline SwigluGroupQuantHifp8Kernel() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR groupIndex, GM_ADDR y, GM_ADDR yScale,
                                GM_ADDR yOrigin, GM_ADDR workspace,
                                const SwigluGroupQuantHifp8TilingData* tilingData, TPipe* pipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ProcessGroupQuant();
    __aicore__ inline void ProcessNonGroupOut();
    __aicore__ inline void ProcessOutputOrigin(int64_t tokenStart, int64_t tokenEnd);
    __aicore__ inline void ProcessNonGroupQuant(int64_t tokenStart, int64_t tokenEnd);
    __aicore__ inline void ProcessCoreMax(int64_t tokenStart, int64_t tokenEnd, GlobalTensor<float> coreMaxGm);
    __aicore__ inline void ProcessSwigluQuant(int64_t tokenStart, int64_t tokenEnd);
    __aicore__ inline void CopyIn(int64_t tokenIdx, int64_t curTileTokens);
    __aicore__ inline void ComputeSwiGLU(LocalTensor<float>& xFloatLocalTensor, int64_t curTileTokens);
    __aicore__ inline void ComputeTileMax(LocalTensor<float>& reduceMaxLocalTensor,
                                          LocalTensor<float>& xFloatLocalTensor,
                                          int64_t tileIdx, int64_t curTileTokens);
    __aicore__ inline void QuantizeOut(LocalTensor<float>& xFloatLocalTensor, int64_t tokenStart,
                                       int64_t curTileTokens, float divScale);
    __aicore__ inline void CopyOutOrigin(LocalTensor<float>& xFloatLocalTensor, int64_t tokenIdx,
                                         int64_t curTileTokens);
    __aicore__ inline void CopyOutCoreMax(LocalTensor<float>& reduceMaxLocalTensor, GlobalTensor<float> coreMaxGm);
    __aicore__ inline void ComputeGlobalMax();
    __aicore__ inline void FillYZero(int64_t tokenStart, int64_t tokenEnd);
    __aicore__ inline void CalcCoreGroupDistribution();
    __aicore__ inline void CalcMinLoadCoreIdx();
    __aicore__ inline event_t AllocAndWaitVToMte3Event();
    __aicore__ inline void ReleaseVToMte3Event(event_t event);
    __aicore__ inline static int64_t CeilDiv(int64_t x, int64_t y) { return y == 0 ? 0 : (x + y - 1) / y; }

    GlobalTensor<T> xGm_, yOriginGm_;
    GlobalTensor<float> weightGm_, yScaleGm_, coreMaxGm_;
    GlobalTensor<hifloat8_t> yGm_;
    GlobalTensor<int64_t> groupIndexGm_;

    TPipe* pipe_ = nullptr;
    TQue<TPosition::VECIN, DB_BUFFER> xQueue_;
    TQue<TPosition::VECIN, DB_BUFFER> weightQueue_;
    TQue<TPosition::VECIN, 1> reduceMaxQueue_;

    GM_ADDR workspace_ = nullptr;
    int64_t tileLength_ = 0, numTiles_ = 0, totalTokens_ = 0, dim2H_ = 0, dimH_ = 0;
    int64_t tileTokens_ = 0, usedCoreNum_ = 0, minLoadCoreIdx_ = 0, groupTokensSum_ = 0;
    uint32_t outputOrigin_ = 0, isGroup_ = 0, hasWeight_ = 0, hasClamp_ = 0, blockIdx_ = 0;
    float globalScale_ = 0., clampLimit_ = 0.0f, dstTypeMax_ = 15.0f;
    int64_t coreGroupStartArr_[MAX_CORE_COUNT] = {0}, coreGroupCountArr_[MAX_CORE_COUNT] = {0};
    const SwigluGroupQuantHifp8TilingData* tilingData_ = nullptr;
};

template <typename T>
__aicore__ inline void SwigluGroupQuantHifp8Kernel<T>::Init(
    GM_ADDR x, GM_ADDR weight, GM_ADDR groupIndex, GM_ADDR y, GM_ADDR yScale, GM_ADDR yOrigin,
    GM_ADDR workspace, const SwigluGroupQuantHifp8TilingData* tilingData, TPipe* pipe)
{
    tilingData_ = tilingData;
    pipe_ = pipe;
    totalTokens_ = tilingData->totalTokens;
    dim2H_ = tilingData->dim2H;
    dimH_ = tilingData->dimH;
    isGroup_ = tilingData->isGroup;
    hasWeight_ = tilingData->hasWeight;
    hasClamp_ = tilingData->hasClamp;
    clampLimit_ = tilingData->clampLimit;
    dstTypeMax_ = tilingData->dstTypeMax;
    tileTokens_ = tilingData->tileTokens;
    usedCoreNum_ = tilingData->usedCoreNum;
    blockIdx_ = GetBlockIdx();

    workspace_ = workspace;
    outputOrigin_ = tilingData->outputOrigin;

    xGm_.SetGlobalBuffer((__gm__ T*)x);
    if (hasWeight_) {
        weightGm_.SetGlobalBuffer((__gm__ float*)weight);
    }

    yGm_.SetGlobalBuffer((__gm__ hifloat8_t*)y);
    yScaleGm_.SetGlobalBuffer((__gm__ float*)yScale);
    if (isGroup_) {
        groupIndexGm_.SetGlobalBuffer((__gm__ int64_t*)groupIndex);
    } else {
        coreMaxGm_.SetGlobalBuffer((__gm__ float*)(workspace_));
    }
    if (outputOrigin_ && yOrigin != nullptr) {
        yOriginGm_.SetGlobalBuffer((__gm__ T*)yOrigin);
    }

    tileLength_ = tilingData->tileLength;
    pipe_->InitBuffer(xQueue_, DB_BUFFER, tileLength_ * QUANT_TILE_BUFFER_NUM * sizeof(float));
    pipe_->InitBuffer(reduceMaxQueue_, 1, tileLength_ * sizeof(float));
    if (hasWeight_) {
        pipe_->InitBuffer(weightQueue_, DB_BUFFER, tileTokens_ * sizeof(float));
    }

    if (isGroup_) {
        CalcCoreGroupDistribution();
    }
}

template <typename T>
__aicore__ inline void SwigluGroupQuantHifp8Kernel<T>::Process()
{
    isGroup_ ? ProcessGroupQuant() : ProcessNonGroupOut();
}

template <typename T>
__aicore__ inline void SwigluGroupQuantHifp8Kernel<T>::ProcessNonGroupOut()
{
    int64_t tokensPerCore = tilingData_->tokensPerCore;
    int64_t remainder = totalTokens_ - tokensPerCore * usedCoreNum_;
    int64_t tokenStart, tokenEnd;
    if (blockIdx_ < remainder) {
        tokenStart = static_cast<int64_t>(blockIdx_) * (tokensPerCore + 1);
        tokenEnd = tokenStart + tokensPerCore + 1;
    } else {
        tokenStart = remainder * (tokensPerCore + 1) + (static_cast<int64_t>(blockIdx_) - remainder) * tokensPerCore;
        tokenEnd = tokenStart + tokensPerCore;
    }
    numTiles_ = CeilDiv(tokenEnd - tokenStart, tileTokens_);

    ProcessNonGroupQuant(tokenStart, tokenEnd);
}

template <typename T>
__aicore__ inline void SwigluGroupQuantHifp8Kernel<T>::ProcessOutputOrigin(int64_t tokenStart,
    int64_t tokenEnd)
{
    int64_t tokenIdx = tokenStart, tileIdx = 0;
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
        tileIdx += 1;
    }
}

template <typename T>
__aicore__ inline void SwigluGroupQuantHifp8Kernel<T>::ProcessNonGroupQuant(int64_t tokenStart,
    int64_t tokenEnd)
{
    ProcessCoreMax(tokenStart, tokenEnd, coreMaxGm_[blockIdx_]);
    SyncAll();

    ComputeGlobalMax();
    SyncAll();

    ProcessSwigluQuant(tokenStart, tokenEnd);
}

template <typename T>
__aicore__ inline void SwigluGroupQuantHifp8Kernel<T>::ProcessCoreMax(int64_t tokenStart,
    int64_t tokenEnd, GlobalTensor<float> coreMaxGm)
{
    LocalTensor<float> reduceMaxLocalTensor = reduceMaxQueue_.AllocTensor<float>();
    int64_t tokenIdx = tokenStart, tileIdx = 0;
    while (tokenIdx < tokenEnd) {
        int64_t tileEnd = AscendC::Std::min(tokenIdx + tileTokens_, tokenEnd);
        int64_t curTileTokens = tileEnd - tokenIdx;

        CopyIn(tokenIdx, curTileTokens);
        LocalTensor<float> xFloatLocalTensor = xQueue_.DeQue<float>();
        ComputeSwiGLU(xFloatLocalTensor, curTileTokens);
        ComputeTileMax(reduceMaxLocalTensor, xFloatLocalTensor, tileIdx, curTileTokens);
        if (outputOrigin_) {
            CopyOutOrigin(xFloatLocalTensor, tokenIdx, curTileTokens);
            PipeBarrier<PIPE_MTE3>();
        }
        xQueue_.FreeTensor<float>(xFloatLocalTensor);
        tokenIdx += curTileTokens;
        tileIdx += 1;
    }
    if (isGroup_) {
        globalScale_ = AscendC::Std::max(reduceMaxLocalTensor.GetValue(0), EPS_NON_GROUP);
        reduceMaxLocalTensor.SetValue(0, globalScale_ / dstTypeMax_);
        PipeBarrier<PIPE_V>();
    }
    CopyOutCoreMax(reduceMaxLocalTensor, coreMaxGm);
    reduceMaxQueue_.FreeTensor<float>(reduceMaxLocalTensor);
}

template <typename T>
__aicore__ inline void SwigluGroupQuantHifp8Kernel<T>::ProcessSwigluQuant(int64_t tokenStart,
    int64_t tokenEnd)
{
    int64_t tokenIdx = tokenStart, tileIdx = 0;
    while (tokenIdx < tokenEnd) {
        int64_t tileEnd = AscendC::Std::min(tokenIdx + tileTokens_, tokenEnd);
        int64_t curTileTokens = tileEnd - tokenIdx;

        CopyIn(tokenIdx, curTileTokens);
        LocalTensor<float> xFloatLocalTensor = xQueue_.DeQue<float>();
        ComputeSwiGLU(xFloatLocalTensor, curTileTokens);
        float divScale;
        if (isGroup_) {
            divScale = dstTypeMax_ / globalScale_;
        } else {
            divScale = 1.0f / yScaleGm_.GetValue(0);
        }
        QuantizeOut(xFloatLocalTensor, tokenIdx, curTileTokens, divScale);
        xQueue_.FreeTensor<float>(xFloatLocalTensor);
        tokenIdx += curTileTokens;
        tileIdx += 1;
    }
}

template <typename T>
__aicore__ inline void SwigluGroupQuantHifp8Kernel<T>::ProcessGroupQuant()
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
            numTiles_ = CeilDiv(tokensInGroup, tileTokens_);
            ProcessCoreMax(tokenStart, tokenEnd, yScaleGm_[groupIdx]);
            if (outputOrigin_) {
                PipeBarrier<PIPE_MTE3>();
            }
            ProcessSwigluQuant(tokenStart, tokenEnd);
            tokenStart = tokenEnd;
        }
    }

    if (blockIdx_ == minLoadCoreIdx_ && groupTokensSum_ < totalTokens_) {
        FillYZero(groupTokensSum_, totalTokens_);
        if (outputOrigin_) {
            ProcessOutputOrigin(groupTokensSum_, totalTokens_);
        }
    }
}

template <typename T>
__aicore__ inline void SwigluGroupQuantHifp8Kernel<T>::FillYZero(int64_t tokenStart, int64_t tokenEnd)
{
    LocalTensor<float> zeroFloatLocalTensor = xQueue_.AllocTensor<float>();
    Duplicate(zeroFloatLocalTensor, 0.0f, static_cast<uint32_t>(tileLength_));
    PipeBarrier<PIPE_V>();
    LocalTensor<hifloat8_t> zeroHif8LocalTensor =
        zeroFloatLocalTensor[tileLength_ * TMP_BUFFER_INDEX].template ReinterpretCast<hifloat8_t>();
    Cast(zeroHif8LocalTensor, zeroFloatLocalTensor, RoundMode::CAST_ROUND, static_cast<uint32_t>(tileLength_));
    event_t vToMte3 = AllocAndWaitVToMte3Event();

    int64_t tokenIdx = tokenStart;
    while (tokenIdx < tokenEnd) {
        int64_t tileEnd = AscendC::Std::min(tokenIdx + tileTokens_, tokenEnd);
        int64_t copySize = (tileEnd - tokenIdx) * dimH_;
        DataCopyParams outCopyParams(1, static_cast<uint32_t>(copySize) * sizeof(hifloat8_t), 0, 0);
        DataCopyPad(yGm_[tokenIdx * dimH_], zeroHif8LocalTensor, outCopyParams);
        tokenIdx = tileEnd;
    }
    ReleaseVToMte3Event(vToMte3);
    xQueue_.FreeTensor<float>(zeroFloatLocalTensor);
}

template <typename T>
__aicore__ inline void SwigluGroupQuantHifp8Kernel<T>::CopyIn(int64_t tokenIdx, int64_t curTileTokens)
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
        Cast(xFloatLocalTensor[tileLength_], xTLocalTensor[fp16Base + copySize], RoundMode::CAST_NONE, static_cast<uint32_t>(copySize));
        PipeBarrier<PIPE_V>();
        xQueue_.EnQue<float>(xFloatLocalTensor);
    }

    if (hasWeight_) {
        LocalTensor<float> weightLocalTensor = weightQueue_.AllocTensor<float>();
        DataCopyParams copyParams(1, static_cast<uint32_t>(curTileTokens) * sizeof(float), 0, 0);
        DataCopyPadParams padParams{false, 0, 0, 0};
        DataCopyPad(weightLocalTensor, weightGm_[tokenIdx], copyParams, padParams);
        weightQueue_.EnQue<float>(weightLocalTensor);
    }
}

template <typename T>
__aicore__ inline void SwigluGroupQuantHifp8Kernel<T>::ComputeSwiGLU(LocalTensor<float>& xFloatLocalTensor,
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
        for (int64_t t = 0; t < curTileTokens; t++) {
            float weightVal = weightLocalTensor.GetValue(static_cast<uint32_t>(t));
            Muls(x0FloatLocalTensor[t * dimH_], x0FloatLocalTensor[t * dimH_], weightVal, static_cast<uint32_t>(dimH_));
            PipeBarrier<PIPE_V>();
        }
        weightQueue_.FreeTensor(weightLocalTensor);
    }
}

template <typename T>
__aicore__ inline void SwigluGroupQuantHifp8Kernel<T>::ComputeTileMax(LocalTensor<float>& reduceMaxLocalTensor,
    LocalTensor<float>& xFloatLocalTensor, int64_t tileIdx, int64_t curTileTokens)
{
    int64_t computeSize = curTileTokens * dimH_;
    LocalTensor<float> x0FloatLocalTensor = xFloatLocalTensor;
    LocalTensor<float> tmpLocalTensor = xFloatLocalTensor[tileLength_ * TMP_BUFFER_INDEX];
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

    if (tileIdx == numTiles_ - 1) {
        uint32_t reduceCount = (numTiles_ > 1) ? static_cast<uint32_t>(tileLength_) : static_cast<uint32_t>(computeSize);
        ReduceMax<float>(reduceMaxLocalTensor, reduceMaxLocalTensor, tmpLocalTensor, reduceCount, false);
        PipeBarrier<PIPE_V>();
    }
}

template <typename T>
__aicore__ inline void SwigluGroupQuantHifp8Kernel<T>::CopyOutCoreMax(LocalTensor<float>& reduceMaxLocalTensor,
    GlobalTensor<float> coreMaxGm)
{
    event_t vToMte3 = AllocAndWaitVToMte3Event();
    DataCopyParams copyParams(1, sizeof(float), 0, 0);
    DataCopyPad(coreMaxGm, reduceMaxLocalTensor, copyParams);
    ReleaseVToMte3Event(vToMte3);
}

template <typename T>
__aicore__ inline void SwigluGroupQuantHifp8Kernel<T>::ComputeGlobalMax()
{
    if (blockIdx_ == 0) {
        LocalTensor<float> reduceMaxLocalTensor = reduceMaxQueue_.AllocTensor<float>();
        LocalTensor<float> tmpLocalTensor = xQueue_.AllocTensor<float>();
        DataCopyParams coreMaxCopyParams(1, static_cast<uint32_t>(usedCoreNum_) * sizeof(float), 0, 0);
        DataCopyPadParams padParams{false, 0, 0, 0};
        DataCopyPad(reduceMaxLocalTensor, coreMaxGm_, coreMaxCopyParams, padParams);
        reduceMaxQueue_.EnQue<float>(reduceMaxLocalTensor);
        reduceMaxLocalTensor = reduceMaxQueue_.DeQue<float>();
        ReduceMax<float>(reduceMaxLocalTensor, reduceMaxLocalTensor, tmpLocalTensor, static_cast<uint32_t>(usedCoreNum_), false);
        PipeBarrier<PIPE_V>();
        Divs(reduceMaxLocalTensor, reduceMaxLocalTensor, dstTypeMax_, 1);
        event_t vToMte3 = AllocAndWaitVToMte3Event();
        DataCopyParams yScaleCopyOutParams(1, sizeof(float), 0, 0);
        DataCopyPad(yScaleGm_, reduceMaxLocalTensor, yScaleCopyOutParams);
        ReleaseVToMte3Event(vToMte3);
        reduceMaxQueue_.FreeTensor<float>(reduceMaxLocalTensor);
        xQueue_.FreeTensor<float>(tmpLocalTensor);
    }
}

template <typename T>
__aicore__ inline void SwigluGroupQuantHifp8Kernel<T>::QuantizeOut(LocalTensor<float>& xFloatLocalTensor,
    int64_t tokenIdx, int64_t curTileTokens, float divScale)
{
    int64_t computeSize = curTileTokens * dimH_;
    Muls(xFloatLocalTensor, xFloatLocalTensor, divScale, static_cast<uint32_t>(computeSize));
    PipeBarrier<PIPE_V>();
    LocalTensor<float> x0FloatLocalTensor = xFloatLocalTensor;
    LocalTensor<hifloat8_t> x0TLocalTensor =
        xFloatLocalTensor[tileLength_ * TMP_BUFFER_INDEX].template ReinterpretCast<hifloat8_t>();
    Cast(x0TLocalTensor, x0FloatLocalTensor, RoundMode::CAST_ROUND, static_cast<uint32_t>(computeSize));
    event_t vToMte3 = AllocAndWaitVToMte3Event();
    int64_t gmOffset = tokenIdx * dimH_;
    DataCopyParams outCopyParams(1, static_cast<uint32_t>(computeSize) * sizeof(hifloat8_t), 0, 0);
    DataCopyPad(yGm_[gmOffset], x0TLocalTensor, outCopyParams);
    ReleaseVToMte3Event(vToMte3);
}

template <typename T>
__aicore__ inline void SwigluGroupQuantHifp8Kernel<T>::CopyOutOrigin(LocalTensor<float>& xFloatLocalTensor,
    int64_t tokenIdx, int64_t curTileTokens)
{
    int64_t gmOffset = tokenIdx * dimH_;
    int64_t copySize = curTileTokens * dimH_;
    DataCopyParams outCopyParams(1, static_cast<uint32_t>(copySize) * sizeof(T), 0, 0);
    event_t vToMte3 = AllocAndWaitVToMte3Event();
    LocalTensor<float> x0FloatLocalTensor = xFloatLocalTensor;

    if constexpr (std::is_same_v<T, float>) {
        DataCopyPad(yOriginGm_[gmOffset], x0FloatLocalTensor, outCopyParams);
    } else {
        LocalTensor<T> x0TLocalTensor =
            xFloatLocalTensor[tileLength_ * TMP_BUFFER_INDEX].template ReinterpretCast<T>();
        Cast(x0TLocalTensor, x0FloatLocalTensor, RoundMode::CAST_RINT, static_cast<uint32_t>(copySize));
        SetFlag<HardEvent::V_MTE3>(vToMte3);
        WaitFlag<HardEvent::V_MTE3>(vToMte3);
        DataCopyPad(yOriginGm_[gmOffset], x0TLocalTensor, outCopyParams);
    }
    ReleaseVToMte3Event(vToMte3);
}

template <typename T>
__aicore__ inline event_t SwigluGroupQuantHifp8Kernel<T>::AllocAndWaitVToMte3Event()
{
    event_t event = static_cast<event_t>(pipe_->AllocEventID<HardEvent::V_MTE3>());
    SetFlag<HardEvent::V_MTE3>(event);
    WaitFlag<HardEvent::V_MTE3>(event);
    return event;
}

template <typename T>
__aicore__ inline void SwigluGroupQuantHifp8Kernel<T>::ReleaseVToMte3Event(event_t event)
{
    pipe_->ReleaseEventID<AscendC::HardEvent::V_MTE3>(event);
}

template <typename T>
__aicore__ inline void SwigluGroupQuantHifp8Kernel<T>::CalcMinLoadCoreIdx()
{
    int64_t minLoad = INT64_MAX_VAL;
    int64_t minCoreIdx = 0;
    for (int64_t core = 0; core < usedCoreNum_; core++) {
        int64_t coreLoad = 0;
        for (int64_t gg = coreGroupStartArr_[core];
             gg < coreGroupStartArr_[core] + coreGroupCountArr_[core]; gg++) {
            coreLoad += groupIndexGm_.GetValue(gg);
        }
        if (coreLoad < minLoad) {
            minLoad = coreLoad;
            minCoreIdx = core;
        }
    }
    minLoadCoreIdx_ = minCoreIdx;
}

template <typename T>
__aicore__ inline void SwigluGroupQuantHifp8Kernel<T>::CalcCoreGroupDistribution()
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

} // namespace SwigluGroupQuantHifp8Ops
#endif // SWIGLU_GROUP_QUANT_HIFP8_H

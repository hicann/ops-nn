/**
 * Copyright (c) 2026 Huawei Technologies
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file swiglu_group_quant.h
 * \brief Unified kernel for SwiGLU Group Quant
 *
 * outputOrigin does NOT change the main flow; it only adds y_origin output
 *   on top of the quantization flow (CopyOutOrigin in ProcessCoreMax, plus
 *   tail ProcessOutputOrigin in group mode).
 * isGroup=false:  two-pass: Pass1(SwiGLU+amax->coreMax, optional CopyOutOrigin) + Pass2(global scale + SwiGLU+quant->y, y_scale[1])
 * isGroup=true:   per-group two-pass, each group: Pass1(SwiGLU+amax->localScale, optional CopyOutOrigin) + Pass2(SwiGLU+quant->y, y_scale[groupIdx])
 *                 no workspace needed, scale written directly to yScaleGm[groupIdx]
 */

#ifndef SWIGLU_GROUP_QUANT_H
#define SWIGLU_GROUP_QUANT_H

#include "kernel_operator.h"

namespace SwigluGroupQuantOps {
using namespace AscendC;

constexpr float EPS_NON_GROUP = 1e-8f;
constexpr uint32_t TMP_BUFFER_INDEX = 2;
constexpr uint32_t QUANT_TILE_BUFFER_NUM = 3;
constexpr uint32_t DB_BUFFER = 2;

template <typename T>
class SwigluGroupQuantKernel {
public:
    __aicore__ inline SwigluGroupQuantKernel() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR groupIndex, GM_ADDR y, GM_ADDR yScale,
                                GM_ADDR yOrigin, GM_ADDR workspace, const SwigluGroupQuantTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ProcessGroupQuant();
    __aicore__ inline void ProcessNonGroupOut();
    __aicore__ inline void ProcessOutputOrigin(uint32_t tokenStart, uint32_t tokenEnd);
    __aicore__ inline void ProcessNonGroupQuant(bool hasWork, uint32_t tokenStart, uint32_t tokenEnd);
    __aicore__ inline void ProcessCoreMax(uint32_t tokenStart, uint32_t tokenEnd, GlobalTensor<float> coreMaxGm);
    __aicore__ inline void ProcessSwigluQuant(uint32_t tokenStart, uint32_t tokenEnd);
    __aicore__ inline void CopyIn(uint32_t tokenIdx, uint32_t curTileTokens);
    __aicore__ inline void ComputeSwiGLU(LocalTensor<float>& xFloatLocalTensor, uint32_t curTileTokens);
    __aicore__ inline void ComputeTileMax(LocalTensor<float>& reduceMaxLocalTensor,
                                          LocalTensor<float>& xFloatLocalTensor,
                                          uint32_t tileIdx, uint32_t curTileTokens);
    __aicore__ inline void QuantizeOut(LocalTensor<float>& xFloatLocalTensor, uint32_t tokenStart, uint32_t curTileTokens, float divScale);
    __aicore__ inline void CopyOutOrigin(LocalTensor<float>& xFloatLocalTensor, uint32_t tokenIdx, uint32_t curTileTokens);
    __aicore__ inline void CopyOutCoreMax(LocalTensor<float>& reduceMaxLocalTensor, GlobalTensor<float> coreMaxGm);
    __aicore__ inline void ComputeGlobalMax();
    __aicore__ inline void FillYZero(uint32_t tokenStart, uint32_t tokenEnd);
    __aicore__ inline static uint32_t CeilDiv(uint32_t x, uint32_t y) { return y == 0 ? 0 : (x + y - 1) / y; }

    GlobalTensor<T> xGm_;
    GlobalTensor<float> weightGm_;
    GlobalTensor<hifloat8_t> yGm_;
    GlobalTensor<float> yScaleGm_;
    GlobalTensor<T> yOriginGm_;
    GlobalTensor<int64_t> groupIndexGm_;
    GlobalTensor<float> coreMaxGm_;

    TPipe pipe_;
    TQue<TPosition::VECIN, DB_BUFFER> xQueue_;
    TQue<TPosition::VECIN, DB_BUFFER> weightQueue_;
    TQue<TPosition::VECIN, 1> reduceMaxQueue_;

    GM_ADDR workspace_ = nullptr;
    uint32_t outputOrigin_ = 0;
    uint32_t tileLength_ = 0;
    uint32_t numTiles_ = 0;
    float globalScale_ = 0.;

    uint32_t totalTokens_ = 0;
    uint32_t dim2H_ = 0;
    uint32_t dimH_ = 0;
    uint32_t isGroup_ = 0;
    uint32_t hasWeight_ = 0;
    uint32_t hasClamp_ = 0;
    float clampLimit_ = 0.0f;
    float dstTypeMaxFinite_ = 0.0f;
    uint32_t tileTokens_ = 0;
    uint32_t usedCoreNum_ = 0;
    uint32_t blockIdx_ = 0;
    uint32_t minLoadCoreIdx_ = 0;
    uint32_t groupTokensSum_ = 0;
    const SwigluGroupQuantTilingData* tilingData_ = nullptr;
};

template <typename T>
__aicore__ inline void SwigluGroupQuantKernel<T>::Init(
    GM_ADDR x, GM_ADDR weight, GM_ADDR groupIndex, GM_ADDR y, GM_ADDR yScale, GM_ADDR yOrigin,
    GM_ADDR workspace, const SwigluGroupQuantTilingData* tilingData)
{
    tilingData_ = tilingData;
    totalTokens_ = tilingData->totalTokens;
    dim2H_ = tilingData->dim2H;
    dimH_ = tilingData->dimH;
    isGroup_ = tilingData->isGroup;
    hasWeight_ = tilingData->hasWeight;
    hasClamp_ = tilingData->hasClamp;
    clampLimit_ = tilingData->clampLimit;
    dstTypeMaxFinite_ = tilingData->dstTypeMaxFinite;
    tileTokens_ = tilingData->tileTokens;
    usedCoreNum_ = tilingData->usedCoreNum;
    minLoadCoreIdx_ = tilingData->minLoadCoreIdx;
    groupTokensSum_ = tilingData->groupTokensSum;
    blockIdx_ = GetBlockIdx();

    workspace_ = workspace;
    outputOrigin_ = tilingData->outputOrigin;

    xGm_.SetGlobalBuffer((__gm__ T*)x);
    if (hasWeight_) {
        weightGm_.SetGlobalBuffer((__gm__ float*)weight);
    }

    if (isGroup_) {
        yGm_.SetGlobalBuffer((__gm__ hifloat8_t*)y);
        yScaleGm_.SetGlobalBuffer((__gm__ float*)yScale);
        groupIndexGm_.SetGlobalBuffer((__gm__ int64_t*)groupIndex);
    } else {
        yGm_.SetGlobalBuffer((__gm__ hifloat8_t*)y);
        yScaleGm_.SetGlobalBuffer((__gm__ float*)yScale);
        coreMaxGm_.SetGlobalBuffer((__gm__ float*)(workspace_));
    }
    if (outputOrigin_ && yOrigin != nullptr) {
        yOriginGm_.SetGlobalBuffer((__gm__ T*)yOrigin);
    }

    tileLength_ = tileTokens_ * dimH_;
    pipe_.InitBuffer(xQueue_, DB_BUFFER, tileLength_ * QUANT_TILE_BUFFER_NUM * sizeof(float));
    pipe_.InitBuffer(reduceMaxQueue_, 1, tileLength_ * sizeof(float));
    if (hasWeight_) {
        pipe_.InitBuffer(weightQueue_, DB_BUFFER, tileTokens_ * sizeof(float));
    }
}

template <typename T>
__aicore__ inline void SwigluGroupQuantKernel<T>::Process()
{
    if (isGroup_) {
        ProcessGroupQuant();
        return;
    }

    ProcessNonGroupOut();
}

template <typename T>
__aicore__ inline void SwigluGroupQuantKernel<T>::ProcessNonGroupOut()
{
    uint32_t tokensPerCore = tilingData_->tokensPerCore;
    uint32_t tokenStart = blockIdx_ * tokensPerCore;
    uint32_t tokenEnd = AscendC::Std::min(tokenStart + tokensPerCore, totalTokens_);
    numTiles_ = CeilDiv(tokenEnd - tokenStart, tileTokens_);
    
    bool hasWork = (blockIdx_ < usedCoreNum_) && (tokenStart < totalTokens_);

    ProcessNonGroupQuant(hasWork, tokenStart, tokenEnd);
}

template <typename T>
__aicore__ inline void SwigluGroupQuantKernel<T>::ProcessOutputOrigin(uint32_t tokenStart, uint32_t tokenEnd)
{
    uint32_t tokenIdx = tokenStart;
    uint32_t tileIdx = 0;
    while (tokenIdx < tokenEnd) {
        uint32_t tileEnd = AscendC::Std::min(tokenIdx + tileTokens_, tokenEnd);
        uint32_t curTileTokens = tileEnd - tokenIdx;

        CopyIn(tokenIdx, curTileTokens);
        LocalTensor<float> xFloatLocalTensor = xQueue_.DeQue<float>();
        ComputeSwiGLU(xFloatLocalTensor, curTileTokens);
        CopyOutOrigin(xFloatLocalTensor, tokenIdx, curTileTokens);
        xQueue_.FreeTensor<float>(xFloatLocalTensor);
        tokenIdx += curTileTokens;
        tileIdx += 1;
    }
}

template <typename T>
__aicore__ inline void SwigluGroupQuantKernel<T>::ProcessNonGroupQuant(bool hasWork, uint32_t tokenStart, uint32_t tokenEnd)
{
    if (hasWork) {
        ProcessCoreMax(tokenStart, tokenEnd, coreMaxGm_[blockIdx_]);
    }
    SyncAll();
    
    ComputeGlobalMax();
    SyncAll();
    
    if (hasWork) {
        ProcessSwigluQuant(tokenStart, tokenEnd);
    }
}

template <typename T>
__aicore__ inline void SwigluGroupQuantKernel<T>::ProcessCoreMax(uint32_t tokenStart, uint32_t tokenEnd, GlobalTensor<float> coreMaxGm)
{
    LocalTensor<float> reduceMaxLocalTensor = reduceMaxQueue_.AllocTensor<float>();
    uint32_t tokenIdx = tokenStart;
    uint32_t tileIdx = 0;
    while (tokenIdx < tokenEnd) {
        uint32_t tileEnd = AscendC::Std::min(tokenIdx + tileTokens_, tokenEnd);
        uint32_t curTileTokens = tileEnd - tokenIdx;

        CopyIn(tokenIdx, curTileTokens);
        LocalTensor<float> xFloatLocalTensor = xQueue_.DeQue<float>();
        ComputeSwiGLU(xFloatLocalTensor, curTileTokens);
        ComputeTileMax(reduceMaxLocalTensor, xFloatLocalTensor, tileIdx, curTileTokens);
        if (outputOrigin_) {
            CopyOutOrigin(xFloatLocalTensor, tokenIdx, curTileTokens);
        }
        xQueue_.FreeTensor<float>(xFloatLocalTensor);
        tokenIdx += curTileTokens;
        tileIdx += 1;
    }
    if (isGroup_) {
        globalScale_ = AscendC::Std::max(reduceMaxLocalTensor.GetValue(0), EPS_NON_GROUP);
        reduceMaxLocalTensor.SetValue(0, globalScale_ / dstTypeMaxFinite_);
        PipeBarrier<PIPE_V>();
    }
    CopyOutCoreMax(reduceMaxLocalTensor, coreMaxGm);
    reduceMaxQueue_.FreeTensor<float>(reduceMaxLocalTensor);
}

template <typename T>
__aicore__ inline void SwigluGroupQuantKernel<T>::ProcessSwigluQuant(uint32_t tokenStart, uint32_t tokenEnd)
{
    uint32_t tokenIdx = tokenStart;
    uint32_t tileIdx = 0;
    while (tokenIdx < tokenEnd) {
        uint32_t tileEnd = AscendC::Std::min(tokenIdx + tileTokens_, tokenEnd);
        uint32_t curTileTokens = tileEnd - tokenIdx;

        CopyIn(tokenIdx, curTileTokens);
        LocalTensor<float> xFloatLocalTensor = xQueue_.DeQue<float>();
        ComputeSwiGLU(xFloatLocalTensor, curTileTokens);
        float divScale;
        if (isGroup_) {
            divScale = dstTypeMaxFinite_ / globalScale_;
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
__aicore__ inline void SwigluGroupQuantKernel<T>::ProcessGroupQuant()
{
    if (blockIdx_ >= usedCoreNum_) {
        return;
    }

    uint32_t groupStart = tilingData_->coreGroupStartArr[blockIdx_];
    uint32_t groupCount = tilingData_->coreGroupCountArr[blockIdx_];
    if (groupCount > 0) {
        // tokenStart = sum of group_index[0..groupStart-1]
        uint32_t tokenStart = 0;
        for (uint32_t g = 0; g < groupStart; g++) {
            tokenStart += static_cast<uint32_t>(groupIndexGm_.GetValue(g));
        }

        for (uint32_t g = 0; g < groupCount; g++) {
            uint32_t groupIdx = groupStart + g;
            uint32_t tokensInGroup = static_cast<uint32_t>(groupIndexGm_.GetValue(groupIdx));
            if (tokensInGroup == 0) {
                continue;
            }
            uint32_t tokenEnd = tokenStart + tokensInGroup;
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
__aicore__ inline void SwigluGroupQuantKernel<T>::FillYZero(uint32_t tokenStart, uint32_t tokenEnd)
{
    LocalTensor<float> zeroFloatLocalTensor = xQueue_.AllocTensor<float>();
    Duplicate(zeroFloatLocalTensor, 0.0f, tileLength_);
    PipeBarrier<PIPE_V>();
    LocalTensor<hifloat8_t> zeroHif8LocalTensor =
        zeroFloatLocalTensor[tileLength_ * TMP_BUFFER_INDEX].template ReinterpretCast<hifloat8_t>();
    Cast(zeroHif8LocalTensor, zeroFloatLocalTensor, RoundMode::CAST_ROUND, tileLength_);
    event_t vToMte3 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>());
    SetFlag<HardEvent::V_MTE3>(vToMte3);
    WaitFlag<HardEvent::V_MTE3>(vToMte3);

    uint32_t tokenIdx = tokenStart;
    while (tokenIdx < tokenEnd) {
        uint32_t tileEnd = AscendC::Std::min(tokenIdx + tileTokens_, tokenEnd);
        uint32_t copySize = (tileEnd - tokenIdx) * dimH_;
        DataCopyParams outCopyParams;
        outCopyParams.blockCount = 1;
        outCopyParams.blockLen = copySize * sizeof(hifloat8_t);
        outCopyParams.srcStride = 0;
        outCopyParams.dstStride = 0;
        DataCopyPad(yGm_[tokenIdx * dimH_], zeroHif8LocalTensor, outCopyParams);
        tokenIdx = tileEnd;
    }
    GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::V_MTE3>(vToMte3);
    xQueue_.FreeTensor<float>(zeroFloatLocalTensor);
}

template <typename T>
__aicore__ inline void SwigluGroupQuantKernel<T>::CopyIn(uint32_t tokenIdx, uint32_t curTileTokens)
{
    LocalTensor<T> xTLocalTensor = xQueue_.AllocTensor<T>();
    uint32_t copySize = curTileTokens * dimH_;
    DataCopyParams copyParams;
    copyParams.blockCount = curTileTokens;
    copyParams.blockLen = dimH_ * sizeof(T);
    copyParams.srcStride = (dim2H_ - dimH_) * sizeof(T);
    copyParams.dstStride = 0;
    DataCopyPadParams padParams{false, 0, 0, 0};
    if constexpr (std::is_same_v<T, float>) {
        uint32_t x0GmOffset = tokenIdx * dim2H_;
        DataCopyPad(xTLocalTensor, xGm_[x0GmOffset], copyParams, padParams); 
        uint32_t x1GmOffset = tokenIdx * dim2H_ + dimH_;
        DataCopyPad(xTLocalTensor[tileLength_], xGm_[x1GmOffset], copyParams, padParams);
        xQueue_.EnQue<float>(xTLocalTensor);
    } else {
        // fp16 源加载到 TMP 块(block2), 与 fp32 输出(block0/block1)物理隔离, 避免原地拓宽 Cast 踩踏源
        uint32_t fp16Base = TMP_BUFFER_INDEX * tileLength_ * sizeof(float) / sizeof(T);
        uint32_t x0GmOffset = tokenIdx * dim2H_;
        DataCopyPad(xTLocalTensor[fp16Base], xGm_[x0GmOffset], copyParams, padParams);
        uint32_t x1GmOffset = tokenIdx * dim2H_ + dimH_;
        DataCopyPad(xTLocalTensor[fp16Base + copySize], xGm_[x1GmOffset], copyParams, padParams);
        xQueue_.EnQue<T>(xTLocalTensor);
        xTLocalTensor = xQueue_.DeQue<T>();
        LocalTensor<float> xFloatLocalTensor = xTLocalTensor.template ReinterpretCast<float>();
        Cast(xFloatLocalTensor, xTLocalTensor[fp16Base], RoundMode::CAST_NONE, copySize);
        Cast(xFloatLocalTensor[tileLength_], xTLocalTensor[fp16Base + copySize], RoundMode::CAST_NONE, copySize);
        PipeBarrier<PIPE_V>();
        xQueue_.EnQue<float>(xFloatLocalTensor);
    }

    if (hasWeight_) {
        LocalTensor<float> weightLocalTensor = weightQueue_.AllocTensor<float>(); 
        DataCopyParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = curTileTokens * sizeof(float);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        DataCopyPadParams padParams{false, 0, 0, 0};
        DataCopyPad(weightLocalTensor, weightGm_[tokenIdx], copyParams, padParams);
        weightQueue_.EnQue<float>(weightLocalTensor);
    }
}

template <typename T>
__aicore__ inline void SwigluGroupQuantKernel<T>::ComputeSwiGLU(LocalTensor<float>& xFloatLocalTensor, uint32_t curTileTokens)
{
    uint32_t computeSize = curTileTokens * dimH_;
    
    LocalTensor<float> x0FloatLocalTensor = xFloatLocalTensor;
    LocalTensor<float> x1FloatLocalTensor = xFloatLocalTensor[tileLength_];

    if (hasClamp_) {
        Mins(x0FloatLocalTensor, x0FloatLocalTensor, clampLimit_, computeSize);
        PipeBarrier<PIPE_V>();
        Maxs(x1FloatLocalTensor, x1FloatLocalTensor, -clampLimit_, computeSize);
        PipeBarrier<PIPE_V>();
        Mins(x1FloatLocalTensor, x1FloatLocalTensor, clampLimit_, computeSize);
        PipeBarrier<PIPE_V>();
    }
    
    LocalTensor<float> tmpLocalTensor = xFloatLocalTensor[tileLength_ * TMP_BUFFER_INDEX];
    // silu(x0) = x0 / (1 + exp(-x0))
    // Step 1: tmp = -x0
    Muls(tmpLocalTensor, x0FloatLocalTensor, -1.0f, computeSize);
    PipeBarrier<PIPE_V>();
    // Step 2: tmp = exp(-x0)
    Exp(tmpLocalTensor, tmpLocalTensor, computeSize);
    PipeBarrier<PIPE_V>();
    // Step 3: tmp = 1 + exp(-x0)
    Adds(tmpLocalTensor, tmpLocalTensor, 1.0f, computeSize);
    PipeBarrier<PIPE_V>();
    // Step 4: siluOut = x0 / (1 + exp(-x0))
    Div(x0FloatLocalTensor, x0FloatLocalTensor, tmpLocalTensor, computeSize);
    PipeBarrier<PIPE_V>();
    Mul(x0FloatLocalTensor, x0FloatLocalTensor, x1FloatLocalTensor, computeSize);
    PipeBarrier<PIPE_V>();

    if (hasWeight_) {
        LocalTensor<float> weightLocalTensor = weightQueue_.DeQue<float>();
        for (uint32_t t = 0; t < curTileTokens; t++) {
            float weightVal = weightLocalTensor.GetValue(t);
            Muls(x0FloatLocalTensor[t * dimH_], x0FloatLocalTensor[t * dimH_], weightVal, dimH_);
            PipeBarrier<PIPE_V>();
        }
        weightQueue_.FreeTensor(weightLocalTensor);
    }
}

template <typename T>
__aicore__ inline void SwigluGroupQuantKernel<T>::ComputeTileMax(LocalTensor<float>& reduceMaxLocalTensor,
                                                                  LocalTensor<float>& xFloatLocalTensor,
                                                                  uint32_t tileIdx,
                                                                  uint32_t curTileTokens)
{
    uint32_t computeSize = curTileTokens * dimH_;
    LocalTensor<float> x0FloatLocalTensor = xFloatLocalTensor;
    LocalTensor<float> tmpLocalTensor = xFloatLocalTensor[tileLength_ * TMP_BUFFER_INDEX];
    Abs(tmpLocalTensor, x0FloatLocalTensor, computeSize);
    PipeBarrier<PIPE_V>();
    Maxs(tmpLocalTensor, tmpLocalTensor, EPS_NON_GROUP, computeSize);
    PipeBarrier<PIPE_V>();
    if (tileIdx == 0) {
        Copy(reduceMaxLocalTensor, tmpLocalTensor, computeSize);
        PipeBarrier<PIPE_V>();
    } else {
        Max(reduceMaxLocalTensor, reduceMaxLocalTensor, tmpLocalTensor, computeSize);
        PipeBarrier<PIPE_V>();
    }

    if (tileIdx == numTiles_ - 1) {
        uint32_t reduceCount = (numTiles_ > 1) ? tileLength_ : computeSize;
        ReduceMax<float>(reduceMaxLocalTensor, reduceMaxLocalTensor, tmpLocalTensor, reduceCount, false);
        PipeBarrier<PIPE_V>();
    }
}

template <typename T>
__aicore__ inline void SwigluGroupQuantKernel<T>::CopyOutCoreMax(LocalTensor<float>& reduceMaxLocalTensor, GlobalTensor<float> coreMaxGm)
{
    event_t vToMte3 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>());
 	SetFlag<HardEvent::V_MTE3>(vToMte3);
 	WaitFlag<HardEvent::V_MTE3>(vToMte3);
    DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = 1 * sizeof(float);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    DataCopyPad(coreMaxGm, reduceMaxLocalTensor, copyParams);
    GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::V_MTE3>(vToMte3);
}

template <typename T>
__aicore__ inline void SwigluGroupQuantKernel<T>::ComputeGlobalMax()
{
    if (blockIdx_ == 0) {
        LocalTensor<float> reduceMaxLocalTensor = reduceMaxQueue_.AllocTensor<float>();
        LocalTensor<float> tmpLocalTensor = xQueue_.AllocTensor<float>();
        DataCopyParams coreMaxCopyParams;
        coreMaxCopyParams.blockCount = 1;
        coreMaxCopyParams.blockLen = usedCoreNum_ * sizeof(float);
        coreMaxCopyParams.srcStride = 0;
        coreMaxCopyParams.dstStride = 0;
        DataCopyPadParams padParams{false, 0, 0, 0};
        DataCopyPad(reduceMaxLocalTensor, coreMaxGm_, coreMaxCopyParams, padParams);
        reduceMaxQueue_.EnQue<float>(reduceMaxLocalTensor);
        reduceMaxLocalTensor = reduceMaxQueue_.DeQue<float>();
        ReduceMax<float>(reduceMaxLocalTensor, reduceMaxLocalTensor, tmpLocalTensor, usedCoreNum_, false);
        PipeBarrier<PIPE_V>();
        Divs(reduceMaxLocalTensor, reduceMaxLocalTensor, dstTypeMaxFinite_, 1);
        event_t vToMte3 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>());
        SetFlag<HardEvent::V_MTE3>(vToMte3);
        WaitFlag<HardEvent::V_MTE3>(vToMte3);
        DataCopyParams yScaleCopyOutParams;
        yScaleCopyOutParams.blockCount = 1;
        yScaleCopyOutParams.blockLen = 1 * sizeof(float);
        yScaleCopyOutParams.srcStride = 0;
        yScaleCopyOutParams.dstStride = 0;
        DataCopyPad(yScaleGm_, reduceMaxLocalTensor, yScaleCopyOutParams);
        GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::V_MTE3>(vToMte3);
        reduceMaxQueue_.FreeTensor<float>(reduceMaxLocalTensor);
        xQueue_.FreeTensor<float>(tmpLocalTensor);
    }
}

template <typename T>
__aicore__ inline void SwigluGroupQuantKernel<T>::QuantizeOut(LocalTensor<float>& xFloatLocalTensor, uint32_t tokenIdx, uint32_t curTileTokens, float divScale)
{
    uint32_t computeSize = curTileTokens * dimH_;
    Muls(xFloatLocalTensor, xFloatLocalTensor, divScale, computeSize);
    PipeBarrier<PIPE_V>();
    LocalTensor<float> x0FloatLocalTensor = xFloatLocalTensor;
    // hif8 输出落到 TMP 块(block2), 与 fp32 源(block0)物理隔离, 避免原地窄化 Cast 踩踏; 对齐 group 版本独立 outQueue 写法
    LocalTensor<hifloat8_t> x0TLocalTensor =
        xFloatLocalTensor[tileLength_ * TMP_BUFFER_INDEX].template ReinterpretCast<hifloat8_t>();
    Cast(x0TLocalTensor, x0FloatLocalTensor, RoundMode::CAST_ROUND, computeSize);
    event_t vToMte3 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>());
 	SetFlag<HardEvent::V_MTE3>(vToMte3);
 	WaitFlag<HardEvent::V_MTE3>(vToMte3);
    uint32_t gmOffset = tokenIdx * dimH_;
    DataCopyParams outCopyParams;
    outCopyParams.blockCount = 1;
    outCopyParams.blockLen = computeSize * sizeof(hifloat8_t);
    outCopyParams.srcStride = 0;
    outCopyParams.dstStride = 0;
    DataCopyPad(yGm_[gmOffset], x0TLocalTensor, outCopyParams);
    GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::V_MTE3>(vToMte3);
}

template <typename T>
__aicore__ inline void SwigluGroupQuantKernel<T>::CopyOutOrigin(LocalTensor<float>& xFloatLocalTensor,
    uint32_t tokenIdx, uint32_t curTileTokens)
{
    uint32_t gmOffset = tokenIdx * dimH_;
    uint32_t copySize = curTileTokens * dimH_;
    DataCopyParams outCopyParams;
    outCopyParams.blockCount = 1;
    outCopyParams.blockLen = copySize * sizeof(T);
    outCopyParams.srcStride = 0;
    outCopyParams.dstStride = 0;

    event_t vToMte3 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>());
 	SetFlag<HardEvent::V_MTE3>(vToMte3);
 	WaitFlag<HardEvent::V_MTE3>(vToMte3);

    if constexpr (std::is_same_v<T, float>) {
        LocalTensor<float> x0FloatLocalTensor = xFloatLocalTensor;
        DataCopyPad(yOriginGm_[gmOffset], x0FloatLocalTensor, outCopyParams);
    } else {
        LocalTensor<float> x0FloatLocalTensor = xFloatLocalTensor;
        LocalTensor<T> x0TLocalTensor = xFloatLocalTensor[tileLength_ * TMP_BUFFER_INDEX].template ReinterpretCast<T>();
        Cast(x0TLocalTensor, x0FloatLocalTensor, RoundMode::CAST_RINT, copySize);
        SetFlag<HardEvent::V_MTE3>(vToMte3);
        WaitFlag<HardEvent::V_MTE3>(vToMte3);
        DataCopyPad(yOriginGm_[gmOffset], x0TLocalTensor, outCopyParams);
    }
    GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::V_MTE3>(vToMte3);
}

} // namespace SwigluGroupQuantOps
#endif // SWIGLU_GROUP_QUANT_H

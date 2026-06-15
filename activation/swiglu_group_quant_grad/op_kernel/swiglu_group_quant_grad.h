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
 * \file swiglu_group_quant_grad.h
 * \brief SwiGLU Group Dynamic Quant Backward kernel for Ascend 950 (A5)
 */

#ifndef SWIGLU_GROUP_QUANT_GRAD_H
#define SWIGLU_GROUP_QUANT_GRAD_H

#include "kernel_operator.h"
#include "swiglu_group_quant_grad_base.h"

namespace SwigluGroupQuantGradOp {
using namespace AscendC;

template <typename T>
class SwigluGroupQuantGrad : public SwigluGroupQuantGradBase {
public:
    __aicore__ inline SwigluGroupQuantGrad() {}
    
    __aicore__ inline void Init(GM_ADDR gradY, GM_ADDR x, GM_ADDR weight, GM_ADDR yOrigin,
                                GM_ADDR groupIndex, GM_ADDR gradX, GM_ADDR gradWeight,
                                GM_ADDR workspace, GM_ADDR tiling);
    
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyInGradY(uint32_t tokenIdx, uint32_t hTileIdx, uint32_t computeSize);
    __aicore__ inline void CopyInX(uint32_t tokenIdx, uint32_t hTileIdx, uint32_t actualTokens, uint32_t currentTileH);
    __aicore__ inline void CopyInTopkWeight(uint32_t tokenIdx, uint32_t currentTileTokens);
    __aicore__ inline void CopyInYOrigin(uint32_t tokenIdx, uint32_t hTileIdx, uint32_t computeSize);
    
    __aicore__ inline void ClampX(LocalTensor<float>& xFloatLocalTensor, uint32_t computeSize);
    __aicore__ inline void ComputeSiLUGrad(LocalTensor<float>& xFloatLocalTensor, uint32_t computeSize);
    __aicore__ inline void ComputeGradX(LocalTensor<float>& xFloatLocalTensor,
                                        LocalTensor<float>& gradYFloatLocalTensor, uint32_t computeSize);
    __aicore__ inline void ApplyClampMask(LocalTensor<float>& xFloatLocalTensor, uint32_t computeSize);
    __aicore__ inline void AccumulateGradWeight(LocalTensor<float>& xFloatLocalTensor,
                                                LocalTensor<float>& gradYFloatLocalTensor,
                                                LocalTensor<float>& yOriginFloatLocalTensor,
                                                LocalTensor<float>& weightLocalTensor,
                                                uint32_t currentTileTokens, uint32_t currentTileH, uint32_t hIdx);
    __aicore__ inline void UpdateGradY(LocalTensor<float>& gradYFloatLocalTensor,
                                       LocalTensor<float>& weightLocalTensor,
                                       uint32_t currentTileTokens,
                                       uint32_t currentTileH);
    __aicore__ inline void CopyOutGradWeight(LocalTensor<float>& weightLocalTensor,
                                             uint32_t tokenIdx,
                                             uint32_t currentTileTokens);
    __aicore__ inline void CopyOutGradX(LocalTensor<float>& xFloatLocalTensor,
                                        uint32_t tokenIdx, uint32_t hTileIdx, uint32_t currentTileTokens,
                                        uint32_t currentTileH);
    __aicore__ inline void ZeroOutTrunc();

    __aicore__ inline void ProcessTile(LocalTensor<float>& weightLocalTensor, uint32_t tokenIdx, uint32_t hTileIdx,
                                       uint32_t currentTileTokens, uint32_t currentTileH);
    
    GlobalTensor<T> gradYGm;
    GlobalTensor<T> xGm;
    GlobalTensor<float> weightGm;
    GlobalTensor<T> yOriginGm;
    GlobalTensor<int64_t> groupIndexGm;
    GlobalTensor<T> gradXGm;
    GlobalTensor<float> gradWeightGm;
};

template <typename T>
__aicore__ inline void SwigluGroupQuantGrad<T>::Init(GM_ADDR gradY, GM_ADDR x, GM_ADDR weight,
                                                     GM_ADDR yOrigin, GM_ADDR groupIndex, GM_ADDR gradX,
                                                     GM_ADDR gradWeight, GM_ADDR workspace, GM_ADDR tiling)
{
    ParseTilingData(tiling);
    
    gradYGm.SetGlobalBuffer((__gm__ T *)gradY, totalTokens * dimH);
    xGm.SetGlobalBuffer((__gm__ T *)x, totalTokens * dim2H);
    gradXGm.SetGlobalBuffer((__gm__ T *)gradX, totalTokens * dim2H);
    
    if (hasWeight) {
        weightGm.SetGlobalBuffer((__gm__ float *)weight, totalTokens);
        gradWeightGm.SetGlobalBuffer((__gm__ float *)gradWeight, totalTokens);
    }
    
    if (hasYOrigin) {
        yOriginGm.SetGlobalBuffer((__gm__ T *)yOrigin, totalTokens * dimH);
    }
    
    if (hasGroupIndex) {
        groupIndexGm.SetGlobalBuffer((__gm__ int64_t *)groupIndex, groupNum);
    }
    
    InitBuffer();
}

template <typename T>
__aicore__ inline void SwigluGroupQuantGrad<T>::CopyInGradY(uint32_t tokenIdx, uint32_t hTileIdx, uint32_t computeSize)
{
    LocalTensor<T> gradYTLocalTensor = gradYQueue.AllocTensor<T>();
    uint32_t gmOffset = tokenIdx * dimH + hTileIdx * tileH;
    DataCopyParams gradYCopyParams;
    gradYCopyParams.blockCount = 1;
    gradYCopyParams.blockLen = computeSize * sizeof(T);
    gradYCopyParams.srcStride = 0;
    gradYCopyParams.dstStride = 0;
    DataCopyPadParams padParams{false, 0, 0, 0};
    if constexpr (std::is_same_v<T, float>) {
        DataCopyPad(gradYTLocalTensor, gradYGm[gmOffset], gradYCopyParams, padParams);
        gradYQueue.EnQue<float>(gradYTLocalTensor);
    } else {
        DataCopyPad(gradYTLocalTensor, gradYGm[gmOffset], gradYCopyParams, padParams);
        gradYQueue.EnQue<T>(gradYTLocalTensor);
        gradYTLocalTensor = gradYQueue.DeQue<T>();
        LocalTensor<float> gradYFloatLocalTensor = gradYTLocalTensor.template ReinterpretCast<float>();
        Cast(gradYFloatLocalTensor, gradYTLocalTensor, RoundMode::CAST_NONE, computeSize);
        PipeBarrier<PIPE_V>();
        gradYQueue.EnQue<float>(gradYFloatLocalTensor);
    }
}

template <typename T>
__aicore__ inline void SwigluGroupQuantGrad<T>::CopyInX(uint32_t tokenIdx, uint32_t hTileIdx,
                                                        uint32_t currentTileTokens, uint32_t currentTileH)
{
    LocalTensor<T> xTLocalTensor = xQueue.AllocTensor<T>();
    uint32_t copySize = currentTileTokens * currentTileH;
    DataCopyParams copyParams;
    copyParams.blockCount = currentTileTokens;
    copyParams.blockLen = currentTileH * sizeof(T);
    copyParams.srcStride = (dim2H - currentTileH) * sizeof(T);
    copyParams.dstStride = 0;
    DataCopyPadParams padParams{false, 0, 0, 0};
    if constexpr (std::is_same_v<T, float>) {
        uint32_t x0GmOffset = tokenIdx * dim2H + hTileIdx * tileH;
        DataCopyPad(xTLocalTensor, xGm[x0GmOffset], copyParams, padParams); 
        uint32_t x1GmOffset = tokenIdx * dim2H + dimH + hTileIdx * tileH;
        DataCopyPad(xTLocalTensor[tileLength], xGm[x1GmOffset], copyParams, padParams);
        xQueue.EnQue<float>(xTLocalTensor);
    } else {
        uint32_t x0GmOffset = tokenIdx * dim2H + hTileIdx * tileH;
        DataCopyPad(xTLocalTensor, xGm[x0GmOffset], copyParams, padParams); 
        uint32_t x1GmOffset = tokenIdx * dim2H + dimH + hTileIdx * tileH;
        DataCopyPad(xTLocalTensor[tileLength * sizeof(float) / sizeof(T)], xGm[x1GmOffset], copyParams, padParams);
        xQueue.EnQue<T>(xTLocalTensor);
        xTLocalTensor = xQueue.DeQue<T>();
        LocalTensor<float> xFloatLocalTensor = xTLocalTensor.template ReinterpretCast<float>();
        Cast(xFloatLocalTensor, xTLocalTensor, RoundMode::CAST_NONE, copySize);
        Cast(xFloatLocalTensor[tileLength], xTLocalTensor[tileLength * sizeof(float) / sizeof(T)],
             RoundMode::CAST_NONE, copySize);
        PipeBarrier<PIPE_V>();
        xQueue.EnQue<float>(xFloatLocalTensor);
    }
}

template <typename T>
__aicore__ inline void SwigluGroupQuantGrad<T>::ClampX(LocalTensor<float>& xFloatLocalTensor, uint32_t computeSize)
{
    LocalTensor<float> x0FloatLocalTensor = xFloatLocalTensor;
    LocalTensor<float> x1FloatLocalTensor = xFloatLocalTensor[tileLength];
    LocalTensor<float> x0TruncatedLocalTensor = xFloatLocalTensor[tileLength * CLAMP_BUFFER_INDEX];
    LocalTensor<float> x1TruncatedLocalTensor = xFloatLocalTensor[tileLength * CLAMP_BUFFER_INDEX + tileLength];
    Copy(x0TruncatedLocalTensor, x0FloatLocalTensor, computeSize);
    PipeBarrier<PIPE_V>();
    Copy(x1TruncatedLocalTensor, x1FloatLocalTensor, computeSize);
    PipeBarrier<PIPE_V>();

    Mins(x0FloatLocalTensor, x0FloatLocalTensor, clampLimit, computeSize);
    PipeBarrier<PIPE_V>();
    Maxs(x1FloatLocalTensor, x1FloatLocalTensor, -clampLimit, computeSize);
    PipeBarrier<PIPE_V>();
    Mins(x1FloatLocalTensor, x1FloatLocalTensor, clampLimit, computeSize);
    PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline void SwigluGroupQuantGrad<T>::CopyInTopkWeight(uint32_t tokenIdx, uint32_t currentTileTokens)
{
    LocalTensor<float> weightLocalTensor = weightQueue.AllocTensor<float>(); 
    DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = currentTileTokens * sizeof(float);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    DataCopyPadParams padParams{false, 0, 0, 0};
    DataCopyPad(weightLocalTensor, weightGm[tokenIdx], copyParams, padParams);
    weightQueue.EnQue<float>(weightLocalTensor);
}

template <typename T>
__aicore__ inline void SwigluGroupQuantGrad<T>::CopyInYOrigin(uint32_t tokenIdx, uint32_t hTileIdx,
                                                              uint32_t computeSize)
{
    LocalTensor<T> yOriginTLocalTensor = yOriginQueue.AllocTensor<T>();
    uint32_t gmOffset = tokenIdx * dimH + hTileIdx * tileH;
    DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = computeSize * sizeof(T);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    DataCopyPadParams padParams{false, 0, 0, 0};

    if constexpr (std::is_same_v<T, float>) {
        DataCopyPad(yOriginTLocalTensor, yOriginGm[gmOffset], copyParams, padParams);
        yOriginQueue.EnQue<float>(yOriginTLocalTensor);
    } else {
        DataCopyPad(yOriginTLocalTensor, yOriginGm[gmOffset], copyParams, padParams);
        yOriginQueue.EnQue<T>(yOriginTLocalTensor);
        yOriginTLocalTensor = yOriginQueue.DeQue<T>();
        LocalTensor<float> yOriginFloatLocalTensor = yOriginTLocalTensor.template ReinterpretCast<float>();
        Cast(yOriginFloatLocalTensor, yOriginTLocalTensor, RoundMode::CAST_NONE, computeSize);
        PipeBarrier<PIPE_V>();
        yOriginQueue.EnQue<float>(yOriginFloatLocalTensor);
    }
}

template <typename T>
__aicore__ inline void SwigluGroupQuantGrad<T>::ComputeSiLUGrad(LocalTensor<float>& xFloatLocalTensor,
                                                                uint32_t computeSize)
{
    LocalTensor<float> x0FloatLocalTensor = xFloatLocalTensor;
    LocalTensor<float> x1FloatLocalTensor = xFloatLocalTensor[tileLength];

    LocalTensor<float> sigmoidX0LocalTensor = xFloatLocalTensor[tileLength * SILU_GRAD_BUFFER_INDEX];
    LocalTensor<float> siluX0LocalTensor = xFloatLocalTensor[tileLength * SILU_GRAD_BUFFER_INDEX + tileLength];
    LocalTensor<float> siluGradX0LocalTensor = xFloatLocalTensor[tileLength * SILU_GRAD_BUFFER_INDEX + 2 * tileLength];
    LocalTensor<float> tmpLocalTensor = xFloatLocalTensor[tileLength * TMP_BUFFER_INDEX];

    Exp(tmpLocalTensor, x0FloatLocalTensor, computeSize);
    PipeBarrier<PIPE_V>();
    Copy(sigmoidX0LocalTensor, tmpLocalTensor, computeSize);
    PipeBarrier<PIPE_V>();
    Adds(tmpLocalTensor, tmpLocalTensor, 1.0f, computeSize);
    PipeBarrier<PIPE_V>();
    Div(sigmoidX0LocalTensor, sigmoidX0LocalTensor, tmpLocalTensor, computeSize);
    PipeBarrier<PIPE_V>();
    
    Mul(siluX0LocalTensor, x0FloatLocalTensor, sigmoidX0LocalTensor, computeSize);
    PipeBarrier<PIPE_V>();
    
    Subs(tmpLocalTensor, (float)1.0, sigmoidX0LocalTensor, computeSize);
    PipeBarrier<PIPE_V>();
    Mul(tmpLocalTensor, tmpLocalTensor, x0FloatLocalTensor, computeSize);
    PipeBarrier<PIPE_V>();
    Adds(tmpLocalTensor, tmpLocalTensor, 1.0f, computeSize);
    PipeBarrier<PIPE_V>();
    Mul(siluGradX0LocalTensor, sigmoidX0LocalTensor, tmpLocalTensor, computeSize);
    PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline void SwigluGroupQuantGrad<T>::ComputeGradX(LocalTensor<float>& xFloatLocalTensor,
                                                             LocalTensor<float>& gradYFloatLocalTensor,
                                                             uint32_t computeSize)
{
    LocalTensor<float> x0FloatLocalTensor = xFloatLocalTensor;
    LocalTensor<float> x1FloatLocalTensor = xFloatLocalTensor[tileLength];
    LocalTensor<float> siluX0LocalTensor = xFloatLocalTensor[tileLength * SILU_GRAD_BUFFER_INDEX + tileLength];
    LocalTensor<float> siluGradX0LocalTensor = xFloatLocalTensor[tileLength * SILU_GRAD_BUFFER_INDEX + 2 * tileLength];

    // update x0 to x0Grad
    Mul(x0FloatLocalTensor, gradYFloatLocalTensor, x1FloatLocalTensor, computeSize);
    PipeBarrier<PIPE_V>();
    Mul(x0FloatLocalTensor, x0FloatLocalTensor, siluGradX0LocalTensor, computeSize);
    PipeBarrier<PIPE_V>();
    // update x1 to x1Grad
    Mul(x1FloatLocalTensor, gradYFloatLocalTensor, siluX0LocalTensor, computeSize);
    PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline void SwigluGroupQuantGrad<T>::ApplyClampMask(LocalTensor<float>& xFloatLocalTensor,
                                                               uint32_t computeSize)
{
    LocalTensor<float> tmpLocalTensor = xFloatLocalTensor[tileLength * TMP_BUFFER_INDEX];
    LocalTensor<uint8_t> maskX0U8Local = tmpLocalTensor.template ReinterpretCast<uint8_t>();
    LocalTensor<uint8_t> maskX1LeftU8Local = tmpLocalTensor[tileLength].template ReinterpretCast<uint8_t>();
    LocalTensor<uint8_t> maskX1RightU8Local = tmpLocalTensor[tileLength * 2].template ReinterpretCast<uint8_t>();
    LocalTensor<uint8_t> maskX1U8Local = maskX1LeftU8Local;

    LocalTensor<float> x0FloatLocalTensor = xFloatLocalTensor;
    LocalTensor<float> x1FloatLocalTensor = xFloatLocalTensor[tileLength];
    LocalTensor<float> x0TruncatedLocalTensor = xFloatLocalTensor[tileLength * CLAMP_BUFFER_INDEX];
    LocalTensor<float> x1TruncatedLocalTensor = xFloatLocalTensor[tileLength * CLAMP_BUFFER_INDEX + tileLength];

    // reuse x0TruncatedLocalTensor/x1TruncatedLocalTensor
    LocalTensor<float> maskX0Local = x0TruncatedLocalTensor;
    LocalTensor<float> maskX1Local = x1TruncatedLocalTensor;
    
    CompareScalar(maskX0U8Local, x0TruncatedLocalTensor, clampLimit, CMPMODE::LT, computeSize);
    PipeBarrier<PIPE_V>();
    CompareScalar(maskX1LeftU8Local, x1TruncatedLocalTensor, -clampLimit, CMPMODE::GT, computeSize);
    PipeBarrier<PIPE_V>();
    CompareScalar(maskX1RightU8Local, x1TruncatedLocalTensor, clampLimit, CMPMODE::LT, computeSize);
    PipeBarrier<PIPE_V>();
    And(maskX1U8Local, maskX1LeftU8Local, maskX1RightU8Local, computeSize);
    PipeBarrier<PIPE_V>();

    LocalTensor<float> onesTensor = tmpLocalTensor[tileLength * 2];
    Duplicate(onesTensor, (float)1.0, computeSize);
    PipeBarrier<PIPE_V>();

    Select(maskX0Local, maskX0U8Local, onesTensor, static_cast<float>(0), SELMODE::VSEL_TENSOR_SCALAR_MODE,
           computeSize);
    PipeBarrier<PIPE_V>();
    Select(maskX1Local, maskX1U8Local, onesTensor, static_cast<float>(0), SELMODE::VSEL_TENSOR_SCALAR_MODE,
           computeSize);
    PipeBarrier<PIPE_V>();

    Mul(x0FloatLocalTensor, x0FloatLocalTensor, maskX0Local, computeSize);
    PipeBarrier<PIPE_V>();
    Mul(x1FloatLocalTensor, x1FloatLocalTensor, maskX1Local, computeSize);
    PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline void SwigluGroupQuantGrad<T>::AccumulateGradWeight(LocalTensor<float>& xFloatLocalTensor,
    LocalTensor<float>& gradYFloatLocalTensor, LocalTensor<float>& yOriginFloatLocalTensor,
    LocalTensor<float>& weightLocalTensor, uint32_t currentTileTokens, uint32_t currentTileH, uint32_t hIdx)
{
    LocalTensor<float> gradWeightAccumLocalTensor = weightLocalTensor[AlignUp(tileTokens, FP32_32B_ALIGN_NUM)];
    LocalTensor<float> tmpLocalTensor = xFloatLocalTensor[tileLength * TMP_BUFFER_INDEX];
    uint32_t computeSize = currentTileTokens * currentTileH;
    Mul(yOriginFloatLocalTensor, yOriginFloatLocalTensor, gradYFloatLocalTensor, computeSize);
    PipeBarrier<PIPE_V>();
    
    if (hIdx == 0) {
        Copy(gradWeightAccumLocalTensor, yOriginFloatLocalTensor, computeSize);
        PipeBarrier<PIPE_V>();
    } else {
        Add(gradWeightAccumLocalTensor, gradWeightAccumLocalTensor, yOriginFloatLocalTensor, computeSize);
        PipeBarrier<PIPE_V>();
    }

    if (hIdx == numHTiles - 1) {
        for (uint32_t t = 0; t < currentTileTokens; t++) {
            ReduceSum<float>(gradWeightAccumLocalTensor[t * tileH], gradWeightAccumLocalTensor[t * tileH],
                             tmpLocalTensor, tileH);
            PipeBarrier<PIPE_V>();
        }
    }
}

template <typename T>
__aicore__ inline void SwigluGroupQuantGrad<T>::UpdateGradY(LocalTensor<float>& gradYFloatLocalTensor,
                                                            LocalTensor<float>& weightLocalTensor,
                                                            uint32_t currentTileTokens, uint32_t currentTileH)
{
    for (uint32_t t = 0; t < currentTileTokens; t++) {
        float weightVal = weightLocalTensor.GetValue(t);
        Muls(gradYFloatLocalTensor[t * currentTileH], gradYFloatLocalTensor[t * currentTileH], weightVal, currentTileH);
        PipeBarrier<PIPE_V>();
    }
}

template <typename T>
__aicore__ inline void SwigluGroupQuantGrad<T>::CopyOutGradWeight(LocalTensor<float>& weightLocalTensor,
                                                                  uint32_t tokenIdx, uint32_t currentTileTokens)
{
    event_t vToMte3 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>());
 	SetFlag<HardEvent::V_MTE3>(vToMte3);
 	WaitFlag<HardEvent::V_MTE3>(vToMte3);
    LocalTensor<float> gradWeightAccumLocalTensor = weightLocalTensor[AlignUp(tileTokens, FP32_32B_ALIGN_NUM)];
    DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = 1 * sizeof(float);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    for (uint32_t t = 0; t < currentTileTokens; t++) {
        DataCopyPad(gradWeightGm[tokenIdx + t], gradWeightAccumLocalTensor[t * tileH], copyParams);
    }
    GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::V_MTE3>(vToMte3);
}

template <typename T>
__aicore__ inline void SwigluGroupQuantGrad<T>::ZeroOutTrunc()
{
    if (truncValue >= totalTokens) {
        return;
    }
    
    uint32_t zeroTokenStart = truncValue + blockIdx;
    uint32_t zeroTokenStep = usedCoreNum;

    LocalTensor<T> zeroXLocal = zeroQueue.AllocTensor<T>();
    LocalTensor<float> zeroXFloatLocal;
    LocalTensor<float> zeroWeightLocal;
    if constexpr (std::is_same_v<T, float>) {
        zeroWeightLocal = zeroXLocal[dim2H];
    } else {
        zeroXFloatLocal = zeroXLocal.template ReinterpretCast<float>();
        zeroWeightLocal = zeroXFloatLocal[dim2H];
    }
    Duplicate(zeroXLocal, (T)0, dim2H);
    Duplicate(zeroWeightLocal, (float)0.0, 1);
    event_t vToMte3 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>());
    SetFlag<HardEvent::V_MTE3>(vToMte3);
 	WaitFlag<HardEvent::V_MTE3>(vToMte3);
    for (uint32_t t = zeroTokenStart; t < totalTokens; t += zeroTokenStep) {
        DataCopyParams gradXCopyParams;
        gradXCopyParams.blockCount = 1;
        gradXCopyParams.blockLen = dim2H * sizeof(T);
        gradXCopyParams.srcStride = 0;
        gradXCopyParams.dstStride = 0;
        DataCopyPad(gradXGm[t * dim2H], zeroXLocal, gradXCopyParams);
        if (hasWeight) {
            DataCopyParams gradWightCopyParams{1, 1 * sizeof(float), 0, 0};
            DataCopyPad(gradWeightGm[t], zeroWeightLocal, gradWightCopyParams);
        }
    }
    zeroQueue.FreeTensor<T>(zeroXLocal);
    GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::V_MTE3>(vToMte3);
}

template <typename T>
__aicore__ inline void SwigluGroupQuantGrad<T>::CopyOutGradX(LocalTensor<float>& xFloatLocalTensor, uint32_t tokenIdx,
                                                             uint32_t hTileIdx, uint32_t currentTileTokens,
                                                             uint32_t currentTileH)
{
    uint32_t gmOffset0 = tokenIdx * dim2H + hTileIdx * tileH;
    uint32_t gmOffset1 = tokenIdx * dim2H + dimH + hTileIdx * tileH;
    uint32_t currentTileLength = currentTileTokens * currentTileH;
    DataCopyParams outCopyParams;
    outCopyParams.blockCount = currentTileTokens;
    outCopyParams.blockLen = currentTileH * sizeof(T);
    outCopyParams.srcStride = 0;
    outCopyParams.dstStride = (dim2H - currentTileH) * sizeof(T);

    event_t vToMte3 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>());
 	SetFlag<HardEvent::V_MTE3>(vToMte3);
 	WaitFlag<HardEvent::V_MTE3>(vToMte3);

    if constexpr (std::is_same_v<T, float>) {
        LocalTensor<float> x0FloatLocalTensor = xFloatLocalTensor;
        LocalTensor<float> x1FloatLocalTensor = xFloatLocalTensor[tileLength];
        DataCopyPad(gradXGm[gmOffset0], x0FloatLocalTensor, outCopyParams);
        DataCopyPad(gradXGm[gmOffset1], x1FloatLocalTensor, outCopyParams);
    } else {
        LocalTensor<float> x0FloatLocalTensor = xFloatLocalTensor;
        LocalTensor<float> x1FloatLocalTensor = xFloatLocalTensor[tileLength];
        LocalTensor<T> x0TLocalTensor = x0FloatLocalTensor.template ReinterpretCast<T>();
        Cast(x0TLocalTensor, x0FloatLocalTensor, RoundMode::CAST_RINT, currentTileLength);
        PipeBarrier<PIPE_V>();
        LocalTensor<T> x1TLocalTensor = x1FloatLocalTensor.template ReinterpretCast<T>();
        Cast(x1TLocalTensor, x1FloatLocalTensor, RoundMode::CAST_RINT, currentTileLength);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE3>(vToMte3);
        WaitFlag<HardEvent::V_MTE3>(vToMte3);
        DataCopyPad(gradXGm[gmOffset0], x0TLocalTensor, outCopyParams);
        DataCopyPad(gradXGm[gmOffset1], x1TLocalTensor, outCopyParams);
    }
    GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::V_MTE3>(vToMte3);
}

template <typename T>
__aicore__ inline void SwigluGroupQuantGrad<T>::ProcessTile(LocalTensor<float>& weightLocalTensor, uint32_t tokenIdx,
                                                            uint32_t hTileIdx, uint32_t currentTileTokens,
                                                            uint32_t currentTileH)
{
    uint32_t computeSize = currentTileTokens * currentTileH;
    uint32_t copySize = currentTileTokens * tileH;
    // 1.copy in grad_y, x
    CopyInGradY(tokenIdx, hTileIdx, computeSize);
    CopyInX(tokenIdx, hTileIdx, currentTileTokens, currentTileH);
    LocalTensor<float> gradYFloatLocalTensor = gradYQueue.DeQue<float>();
    LocalTensor<float> xFloatLocalTensor = xQueue.DeQue<float>();

    // 2.update grad_y with weight and compute grad_weight
    if (hasWeight) {
        CopyInYOrigin(tokenIdx, hTileIdx, computeSize);
        LocalTensor<float> yOriginFloatLocalTensor = yOriginQueue.DeQue<float>();
        AccumulateGradWeight(xFloatLocalTensor, gradYFloatLocalTensor, yOriginFloatLocalTensor, weightLocalTensor,
                             currentTileTokens, currentTileH, hTileIdx);
        yOriginQueue.FreeTensor<float>(yOriginFloatLocalTensor);
        UpdateGradY(gradYFloatLocalTensor, weightLocalTensor, currentTileTokens, currentTileH);
    }

    // 3.clamp x0 x1
    if (hasClampLimit) {
        ClampX(xFloatLocalTensor, computeSize);
    }

    // 4.compute silugrad
    ComputeSiLUGrad(xFloatLocalTensor, computeSize);
    // 5.compute grad_x
    ComputeGradX(xFloatLocalTensor, gradYFloatLocalTensor, computeSize);
    // 6.grad_x clamp mask
    if (hasClampLimit) {
        ApplyClampMask(xFloatLocalTensor, computeSize);
    }

    // 7.copy out grad_x
    CopyOutGradX(xFloatLocalTensor, tokenIdx, hTileIdx, currentTileTokens, currentTileH);
    gradYQueue.FreeTensor<float>(gradYFloatLocalTensor);
    xQueue.FreeTensor<float>(xFloatLocalTensor);
}

template <typename T>
__aicore__ inline void SwigluGroupQuantGrad<T>::Process()
{
    if (blockIdx >= usedCoreNum) {
        return;
    }
    
    uint32_t tokenEnd = tokenStart + tokensPerCore;
    if (tokenEnd > truncValue) {
        tokenEnd = truncValue;
    }
    uint32_t tokenIdx = tokenStart;
    while (tokenIdx < tokenEnd) {
        uint32_t currentTileTokens = tileTokens;
        if (tokenIdx + tileTokens > tokenEnd) {
            currentTileTokens = tokenEnd - tokenIdx;
        }

        LocalTensor<float> weightLocalTensor;
        if (hasWeight) {
            CopyInTopkWeight(tokenIdx, currentTileTokens);
            weightLocalTensor = weightQueue.DeQue<float>();
        }

        for (uint32_t hTileIdx = 0; hTileIdx < numHTiles; hTileIdx++) {
            uint32_t currentTileH = tileH;
            if (hTileIdx == numHTiles - 1) {
                currentTileH = dimH - hTileIdx * tileH;
            }
            ProcessTile(weightLocalTensor, tokenIdx, hTileIdx, currentTileTokens, currentTileH);
        }
        
        if (hasWeight) {
            CopyOutGradWeight(weightLocalTensor, tokenIdx, currentTileTokens);
            weightQueue.FreeTensor<float>(weightLocalTensor);
        }
        tokenIdx += currentTileTokens;
    }
    SyncAll();
    ZeroOutTrunc();
}

} // namespace SwigluGroupQuantGradOp

#endif // SWIGLU_GROUP_QUANT_GRAD_H
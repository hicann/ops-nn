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
 * \file ada_layer_norm_impl.h
 * \brief
 */
#ifndef ADA_LAYER_NORM_IMPL_H
#define ADA_LAYER_NORM_IMPL_H

#include "ada_layer_norm_full_load.h"
#include "ada_layer_norm_welford.h"

using namespace AdaLayerNormNS;

template <typename T, typename U, typename Y, uint8_t OP_CODE>
__aicore__ inline void AdaLayerNormWelford<T, U, Y, OP_CODE>::WelfordInitialize(uint32_t computeLength)
{
    meanTmpLocal = scaleQueue.AllocTensor<float>();
    varTmpLocal = shiftQueue.AllocTensor<float>();
    __ubuf__ float* meanAddr = (__ubuf__ float*)meanTmpLocal.GetPhyAddr();
    __ubuf__ float* varAddr = (__ubuf__ float*)varTmpLocal.GetPhyAddr();
    uint16_t colLoopTimes = static_cast<uint16_t>(CeilA2B(computeLength, V_LENGTH));
    __VEC_SCOPE__
    {
        RegTensor<float> x;
        MaskReg pregLoop;
        Duplicate(x, 0.0f);
        for (uint16_t j = 0; j < colLoopTimes; j++) {
            pregLoop = UpdateMask<float>(computeLength);
            DataCopy(meanAddr + j * V_LENGTH, x, pregLoop);
            DataCopy(varAddr + j * V_LENGTH, x, pregLoop);
        }
    }
}

template <typename T, typename U, typename Y, uint8_t OP_CODE>
__aicore__ inline void AdaLayerNormWelford<T, U, Y, OP_CODE>::ProcessWelfordUpdate(int64_t computeLength, int64_t welfordCount)
{
    LocalTensor<T> xLocal = xQueue.DeQue<T>();
    WelfordUpdateParam para;
    para.rnLength = 1;
    para.abLength = sliceSize;
    para.abComputeLength = computeLength;
    para.nRec = 1.0f / static_cast<float>(welfordCount);
    WelfordUpdate<T, float, false>(meanTmpLocal, varTmpLocal, meanTmpLocal, varTmpLocal, xLocal, tmpLocal, para);
    PipeBarrier<PIPE_V>();
    xQueue.FreeTensor(xLocal);
}

template <typename T, typename U, typename Y, uint8_t OP_CODE>
__aicore__ inline void AdaLayerNormWelford<T, U, Y, OP_CODE>::ProcessWelfordFinalize(int64_t welfordCount, int64_t batchCount)
{
    WelfordFinalizePara para;
    para.rnLength = welfordCount;
    para.abLength = sliceSize;
    para.headCount = welfordCount;
    para.headCountLength = tailSize;
    para.tailCount = (tailSize > 0) ? (welfordCount - 1) : welfordCount;
    para.tailCountLength = sliceSize - tailSize;
    para.abRec = 1.0f / static_cast<float>(sliceSize);
    para.rRec = 1.0f / static_cast<float>(hiddenDim);
    WelfordFinalize<true>(meanLocal[batchCount], varLocal[batchCount], meanTmpLocal, varTmpLocal, tmpLocal, para);
    PipeBarrier<PIPE_V>();
    scaleQueue.FreeTensor(meanTmpLocal);
    shiftQueue.FreeTensor(varTmpLocal);
}

template <typename T, typename U, typename Y, uint8_t OP_CODE>
__aicore__ inline void AdaLayerNormWelford<T, U, Y, OP_CODE>::ProcessNormalize(uint32_t dataCount, int64_t batchCount)
{
    LocalTensor<float> xLocal;
    if constexpr (std::is_same_v<T, float>) {
        xLocal = xQueue.DeQue<float>();
    } else {
        LocalTensor<T> xFp16 = xQueue.DeQue<T>();
        xLocal = xFp16.template ReinterpretCast<float>();
        Cast(xLocal, xFp16[WELFORD_COUNT], RoundMode::CAST_NONE, dataCount);
        PipeBarrier<PIPE_V>();
    }
    LocalTensor<float> weightLocal = normLocal;
    if (hasWeight) {
        if constexpr (std::is_same_v<U, float>) {
            weightLocal = scaleQueue.DeQue<float>();
        } else {
            LocalTensor<U> weightFp16 = scaleQueue.DeQue<U>();
            weightLocal = weightFp16.template ReinterpretCast<float>();
            Cast(weightLocal, weightFp16[WELFORD_COUNT], RoundMode::CAST_NONE, dataCount);
            PipeBarrier<PIPE_V>();
        }
    }
    LocalTensor<float> biasLocal = normLocal;
    if (hasBias) {
        if constexpr (std::is_same_v<U, float>) {
            biasLocal = shiftQueue.DeQue<float>();
        } else {
            LocalTensor<U> biasFp16 = shiftQueue.DeQue<U>();
            biasLocal = biasFp16.template ReinterpretCast<float>();
            Cast(biasLocal, biasFp16[WELFORD_COUNT], RoundMode::CAST_NONE, dataCount);
            PipeBarrier<PIPE_V>();
        }
    }

    NormalizePara para{1, dataCount, WELFORD_COUNT};
    if (hasWeight && hasBias) {
        Normalize<float, float, false, hasGammaBetaNormalizeConfig>(normLocal, rstdLocal[batchCount], meanLocal[batchCount], 
            varLocal[batchCount], xLocal, weightLocal, biasLocal, tmpLocal, epsilon, para);
    } else if (hasWeight && !hasBias) {
        Normalize<float, float, false, hasGammaNoBetaNormalizeConfig>(normLocal, rstdLocal[batchCount], meanLocal[batchCount], 
            varLocal[batchCount], xLocal, weightLocal, biasLocal, tmpLocal, epsilon, para);
    } else if (!hasWeight && hasBias) {
        Normalize<float, float, false, noGammaHasBetaNormalizeConfig>(normLocal, rstdLocal[batchCount], meanLocal[batchCount], 
            varLocal[batchCount], xLocal, weightLocal, biasLocal, tmpLocal, epsilon, para);
    } else if (!hasWeight && !hasBias) {
        Normalize<float, float, false, noGammaNoBetaNormalizeConfig>(normLocal, rstdLocal[batchCount], meanLocal[batchCount], 
            varLocal[batchCount], xLocal, weightLocal, biasLocal, tmpLocal, epsilon, para);
    }
    PipeBarrier<PIPE_V>();
    xQueue.FreeTensor(xLocal);
    scaleQueue.FreeTensor(weightLocal);
    shiftQueue.FreeTensor(biasLocal);
}

template <typename T, typename U, typename Y, uint8_t OP_CODE>
__aicore__ inline void AdaLayerNormWelford<T, U, Y, OP_CODE>::CalculateScale(int64_t batchCount)
{
    __ubuf__ float* maxTmpAddr = (__ubuf__ float*)maxTmpLocal.GetPhyAddr();
    __ubuf__ float* quantScaleAddr = (__ubuf__ float*)quantScaleLocal[batchCount].GetPhyAddr();

    __VEC_SCOPE__
    {
        RegTensor<float> quantScale;
        MaskReg pregMerge = CreateMask<float, MaskPattern::VL1>();

        DataCopy<float, LoadDist::DIST_BRC_B32>(quantScale, maxTmpAddr);
        Muls(quantScale, quantScale, quantFactor, pregMerge);
        // 拷出量化系数
        DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(quantScaleAddr, quantScale, pregMerge);
    }
}

template <typename T, typename U, typename Y, uint8_t OP_CODE>
__aicore__ inline void AdaLayerNormFullLoad<T, U, Y, OP_CODE>::CopyInOtherData()
{
    if (hasWeight) {
        CopyInAndCast(weightLocal, weightGm, hiddenDim, DATA_COUNT);
    }
    if (hasBias) {
        CopyInAndCast(biasLocal, biasGm, hiddenDim, DATA_COUNT);
    }
    if constexpr (OP_CODE == QUANT_OP_CODE) {
        if (hasSmooth) {
            DataCopyExtParams copyInParams{1, static_cast<uint32_t>(hiddenDim * sizeof(T)), 0, 0, 0};
            DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
            DataCopyPad(smoothLocal, smoothGm, copyInParams, padParams);
            PIPE_MTE2_V();
        }
    }
}

template <typename T, typename U, typename Y, uint8_t OP_CODE>
__aicore__ inline void AdaLayerNormFullLoad<T, U, Y, OP_CODE>::CopyInScaleShift(int64_t offset, int64_t localOffset, int64_t repeatNum)
{
    PIPE_V_MTE2();
    DataCopyExtParams copyInParams{1, static_cast<uint32_t>(hiddenDim * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    DataCopyPad(scaleLocal[localOffset], scaleGm[offset], copyInParams, padParams);
    DataCopyPad(shiftLocal[localOffset], shiftGm[offset], copyInParams, padParams);
    PIPE_MTE2_V();
    for (int i = 1;i < repeatNum;i ++) {
        Copy(scaleLocal[localOffset + i * hiddenDimCeil], scaleLocal[localOffset], hiddenDim);
        Copy(shiftLocal[localOffset + i * hiddenDimCeil], shiftLocal[localOffset], hiddenDim);
    }
    PipeBarrier<PIPE_V>();
}

template <typename T, typename U, typename Y, uint8_t OP_CODE>
__aicore__ inline void AdaLayerNormFullLoad<T, U, Y, OP_CODE>::CopyInX(int64_t offset, uint16_t blockCount)
{
    LocalTensor<T> xLocal = xQueue.AllocTensor<T>();
    DataCopyExtParams copyInParams{blockCount, static_cast<uint32_t>(hiddenDim * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    if (blockCount > 1 && hiddenDim < hiddenDimCeil) {
        padParams.isPad = true;
        padParams.paddingValue = 0;
        padParams.rightPadding = hiddenDimCeil - hiddenDim;
    }
    if constexpr (std::is_same_v<T, float>) {
        DataCopyPad(xLocal, xGm[offset], copyInParams, padParams);
    } else {
        DataCopyPad(xLocal[DATA_COUNT], xGm[offset], copyInParams, padParams);
    }
    xQueue.EnQue(xLocal);
}

template <typename T, typename U, typename Y, uint8_t OP_CODE>
__aicore__ inline void AdaLayerNormFullLoad<T, U, Y, OP_CODE>::BaseCopyOut(int64_t offset, uint16_t blockCount)
{
    LocalTensor<T> out = outQueue.DeQue<T>();
    DataCopyExtParams copyOutParams{blockCount, static_cast<uint32_t>(hiddenDim * sizeof(T)), 0, 0, 0};
    DataCopyPad(outGm[offset], out, copyOutParams);
    outQueue.FreeTensor(out);
}

template <typename T, typename U, typename Y, uint8_t OP_CODE>
__aicore__ inline void AdaLayerNormFullLoad<T, U, Y, OP_CODE>::QuantCopyOut(int64_t offset, uint16_t blockCount)
{
    LocalTensor<Y> quantOutLocal = quantOutQueue.DeQue<Y>();
    DataCopyExtParams copyOutParams{blockCount, static_cast<uint32_t>(hiddenDim * sizeof(Y)), 0, 0, 0};
    DataCopyPad(quantOutGm[offset], quantOutLocal, copyOutParams);
    quantOutQueue.FreeTensor(quantOutLocal);
}

template <typename T, typename U, typename Y, uint8_t OP_CODE>
__aicore__ inline void AdaLayerNormFullLoad<T, U, Y, OP_CODE>::CopyMeanRstdOut(int64_t offset, int64_t len)
{
    LocalTensor<T> meanOut = meanQueue.DeQue<T>();
    LocalTensor<T> rstdOut = rstdQueue.DeQue<T>();
    DataCopyExtParams copyOutParams{1, static_cast<uint32_t>(len * sizeof(T)), 0, 0, 0};
    DataCopyPad(meanGm[offset], meanOut, copyOutParams);
    DataCopyPad(rstdGm[offset], rstdOut, copyOutParams);
    meanQueue.FreeTensor(meanOut);
    rstdQueue.FreeTensor(rstdOut);
}

template <typename T, typename U, typename Y, uint8_t OP_CODE>
__aicore__ inline void AdaLayerNormFullLoad<T, U, Y, OP_CODE>::CopyScaleOut(int64_t offset, int64_t len)
{
    LocalTensor<float> quantScaleLocal = quantScaleQueue.DeQue<float>();
    DataCopyExtParams copyOutParams{1, static_cast<uint32_t>(len * sizeof(float)), 0, 0, 0};
    DataCopyPad(quantScaleGm[offset], quantScaleLocal, copyOutParams);
    quantScaleQueue.FreeTensor(quantScaleLocal);
}

template <typename T, typename U, typename Y, uint8_t OP_CODE>
__aicore__ inline void AdaLayerNormWelford<T, U, Y, OP_CODE>::CopyInOtherData(int64_t offset, int64_t len)
{
    if (hasWeight) {
        LocalTensor<U> weightLocal = scaleQueue.AllocTensor<U>();
        DataCopyExtParams copyInParams{1, static_cast<uint32_t>(len * sizeof(U)), 0, 0, 0};
        DataCopyPadExtParams<U> padParams{false, 0, 0, 0};
        if constexpr (std::is_same_v<U, float>) {
            DataCopyPad(weightLocal, weightGm[offset], copyInParams, padParams);
        } else {
            DataCopyPad(weightLocal[WELFORD_COUNT], weightGm[offset], copyInParams, padParams);
        }
        scaleQueue.EnQue(weightLocal);
    }
    if (hasBias) {
        LocalTensor<U> biasLocal = shiftQueue.AllocTensor<U>();
        DataCopyExtParams copyInParams{1, static_cast<uint32_t>(len * sizeof(U)), 0, 0, 0};
        DataCopyPadExtParams<U> padParams{false, 0, 0, 0};
        if constexpr (std::is_same_v<U, float>) {
            DataCopyPad(biasLocal, biasGm[offset], copyInParams, padParams);
        } else {
            DataCopyPad(biasLocal[WELFORD_COUNT], biasGm[offset], copyInParams, padParams);
        }
        shiftQueue.EnQue(biasLocal);
    }
}

template <typename T, typename U, typename Y, uint8_t OP_CODE>
__aicore__ inline void AdaLayerNormWelford<T, U, Y, OP_CODE>::CopyInScaleShift(int64_t offset, int64_t smoothOffset, int64_t len)
{
    LocalTensor<T> scaleLocal = scaleQueue.AllocTensor<T>();
    LocalTensor<T> shiftLocal = shiftQueue.AllocTensor<T>();
    DataCopyExtParams copyInParams{1, static_cast<uint32_t>(len * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    DataCopyPad(scaleLocal, scaleGm[offset], copyInParams, padParams);
    DataCopyPad(shiftLocal, shiftGm[offset], copyInParams, padParams);
    scaleQueue.EnQue(scaleLocal);
    shiftQueue.EnQue(shiftLocal);
    if constexpr (OP_CODE == QUANT_OP_CODE) {
        if (hasSmooth) {
            LocalTensor<T> smoothLocal = xQueue.AllocTensor<T>();
            DataCopyPad(smoothLocal, smoothGm[smoothOffset], copyInParams, padParams);
            xQueue.EnQue(smoothLocal);
        }
    }
}

template <typename T, typename U, typename Y, uint8_t OP_CODE>
__aicore__ inline void AdaLayerNormWelford<T, U, Y, OP_CODE>::CopyInX(int64_t offset, int64_t len)
{
    LocalTensor<T> xLocal = xQueue.AllocTensor<T>();
    DataCopyExtParams copyInParams{1, static_cast<uint32_t>(len * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    if constexpr (std::is_same_v<T, float>) {
        DataCopyPad(xLocal, xGm[offset], copyInParams, padParams);
    } else {
        DataCopyPad(xLocal[WELFORD_COUNT], xGm[offset], copyInParams, padParams);
    }
    xQueue.EnQue(xLocal);
}

template <typename T, typename U, typename Y, uint8_t OP_CODE>
__aicore__ inline void AdaLayerNormWelford<T, U, Y, OP_CODE>::CopyInNorm(int64_t offset, int64_t len)
{
    LocalTensor<float> xLocal = xQueue.AllocTensor<float>();
    DataCopyExtParams copyInParams{1, static_cast<uint32_t>(len * sizeof(float)), 0, 0, 0};
    DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
    DataCopyPad(xLocal, normGm[offset], copyInParams, padParams);
    xQueue.EnQue(xLocal);
}

template <typename T, typename U, typename Y, uint8_t OP_CODE>
__aicore__ inline void AdaLayerNormWelford<T, U, Y, OP_CODE>::BaseCopyOut(int64_t offset, int64_t len)
{
    LocalTensor<T> out = outQueue.DeQue<T>();
    DataCopyExtParams copyOutParams{1, static_cast<uint32_t>(len * sizeof(T)), 0, 0, 0};
    DataCopyPad(outGm[offset], out, copyOutParams);
    outQueue.FreeTensor(out);
}

template <typename T, typename U, typename Y, uint8_t OP_CODE>
__aicore__ inline void AdaLayerNormWelford<T, U, Y, OP_CODE>::CopyNormOut(int64_t offset, int64_t len)
{
    LocalTensor<float> out = outQueue.DeQue<float>();
    DataCopyExtParams copyOutParams{1, static_cast<uint32_t>(len * sizeof(float)), 0, 0, 0};
    DataCopyPad(normGm[offset], out, copyOutParams);
    outQueue.FreeTensor(out);
}

template <typename T, typename U, typename Y, uint8_t OP_CODE>
__aicore__ inline void AdaLayerNormWelford<T, U, Y, OP_CODE>::QuantCopyOut(int64_t offset, int64_t len)
{
    LocalTensor<Y> quantOutLocal = quantOutQueue.DeQue<Y>();
    DataCopyExtParams copyOutParams{1, static_cast<uint32_t>(len * sizeof(Y)), 0, 0, 0};
    DataCopyPad(quantOutGm[offset], quantOutLocal, copyOutParams);
    quantOutQueue.FreeTensor(quantOutLocal);
}

template <typename T, typename U, typename Y, uint8_t OP_CODE>
__aicore__ inline void AdaLayerNormWelford<T, U, Y, OP_CODE>::CopyScaleOut(int64_t offset, int64_t len)
{
    LocalTensor<float> quantScaleLocal = quantScaleQueue.DeQue<float>();
    DataCopyExtParams copyOutParams{1, static_cast<uint32_t>(len * sizeof(float)), 0, 0, 0};
    DataCopyPad(quantScaleGm[offset], quantScaleLocal, copyOutParams);
    quantScaleQueue.FreeTensor(quantScaleLocal);
}

template <typename T, typename U, typename Y, uint8_t OP_CODE>
__aicore__ inline void AdaLayerNormWelford<T, U, Y, OP_CODE>::CopyMeanRstdOut(int64_t offset, int64_t len)
{
    LocalTensor<T> meanOut = meanQueue.DeQue<T>();
    LocalTensor<T> rstdOut = rstdQueue.DeQue<T>();
    DataCopyExtParams copyOutParams{1, static_cast<uint32_t>(len * sizeof(T)), 0, 0, 0};
    DataCopyPad(meanGm[offset], meanOut, copyOutParams);
    DataCopyPad(rstdGm[offset], rstdOut, copyOutParams);
    meanQueue.FreeTensor(meanOut);
    rstdQueue.FreeTensor(rstdOut);
}

#endif  // ADA_LAYER_NORM_IMPL_H
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
 * \file ada_layer_norm_welford.h
 * \brief
 */
#ifndef ADA_LAYER_NORM_WELFORD_H
#define ADA_LAYER_NORM_WELFORD_H

#include "ada_layer_norm_common.h"

namespace AdaLayerNormNS {
using namespace AscendC;

template <typename T, typename U, typename Y, uint8_t OP_CODE>
class AdaLayerNormWelford {
public:
    __aicore__ inline AdaLayerNormWelford(){};
    __aicore__ inline void InitV2(const GmAddr* gmAddr, const AdaLayerNormTilingData *tilingData);
    __aicore__ inline void InitQuant(const GmAddr* gmAddr, GM_ADDR workspace, const AdaLayerNormTilingData *tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ParseTilingData(const AdaLayerNormTilingData *tilingData);
    __aicore__ inline void SliceProcess();
    __aicore__ inline void ComputeMeanVar(int64_t offset, int64_t batchCount);
    __aicore__ inline void ComputeAdaLayerNorm(int64_t offset, int64_t scaleOffset, int64_t batchCount);
    __aicore__ inline void ProcessWelfordUpdate(int64_t computeLength, int64_t welfordCount);
    __aicore__ inline void ProcessWelfordFinalize(int64_t welfordCount, int64_t batchCount);
    __aicore__ inline void ProcessNormalize(int64_t dataCount, int64_t batchCount);
    __aicore__ inline void Adaption(int64_t dataCount, int64_t batchCount);
    __aicore__ inline void DynamicQuant(int64_t offset, int64_t batchCount);
    __aicore__ inline void CalculateScale(int64_t batchCount);
    __aicore__ inline void ProcessQuant(int64_t dataCount, int64_t batchCount);

    __aicore__ inline void CopyInOtherData(int64_t offset, int64_t len);
    __aicore__ inline void CopyInScaleShift(int64_t offset, int64_t len);
    __aicore__ inline void CopyInX(int64_t offset, int64_t len);
    __aicore__ inline void CopyInNorm(int64_t offset, int64_t len);
    __aicore__ inline void BaseCopyOut(int64_t offset, int64_t len);
    __aicore__ inline void CopyNormOut(int64_t offset, int64_t len);
    __aicore__ inline void QuantCopyOut(int64_t offset, int64_t len);
    __aicore__ inline void CopyMeanRstdOut(int64_t offset, int64_t len);
    __aicore__ inline void CopyScaleOut(int64_t offset, int64_t len);

private:
    using OUT_DTYPE = std::conditional_t<OP_CODE == QUANT_OP_CODE, float, T>;
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> xQueue;
    TQue<QuePosition::VECIN, 1> scaleQueue;
    TQue<QuePosition::VECIN, 1> shiftQueue;
    TQue<QuePosition::VECIN, 1> smoothQueue;
    TBuf<TPosition::VECCALC> weightBuf;
    TBuf<TPosition::VECCALC> biasBuf;
    TBuf<TPosition::VECCALC> normBuf;
    TBuf<TPosition::VECCALC> tmpBuf;
    TBuf<TPosition::VECCALC> varBuf;
    TBuf<TPosition::VECCALC> maxBuf;
    TQue<QuePosition::VECOUT, 1> outQueue;
    TQue<QuePosition::VECOUT, 1> quantOutQueue;
    TQue<QuePosition::VECOUT, 1> meanQueue;
    TQue<QuePosition::VECOUT, 1> rstdQueue;
    TQue<QuePosition::VECOUT, 1> quantScaleQueue;

    GlobalTensor<T> xGm;
    GlobalTensor<T> scaleGm;
    GlobalTensor<T> shiftGm;
    GlobalTensor<U> weightGm;
    GlobalTensor<U> biasGm;
    GlobalTensor<T> smoothGm;
    GlobalTensor<T> outGm;
    GlobalTensor<T> meanGm;
    GlobalTensor<T> rstdGm;
    GlobalTensor<Y> quantOutGm;
    GlobalTensor<float> quantScaleGm;
    GlobalTensor<float> normGm;

    LocalTensor<float> weightLocal;
    LocalTensor<float> biasLocal;
    LocalTensor<float> normLocal;
    LocalTensor<uint8_t> tmpLocal;
    LocalTensor<float> meanTmpLocal;
    LocalTensor<float> varTmpLocal;
    LocalTensor<float> varLocal;
    LocalTensor<float> maxTmpLocal;
    LocalTensor<float> meanLocal;
    LocalTensor<float> rstdLocal;
    LocalTensor<float> quantScaleLocal;

    int64_t sliceSize = 0;
    int64_t sliceCount = 0;
    int64_t tailSize = 0;
    int64_t seqLen = 0;
    int64_t hiddenDim = 0;
    int64_t hiddenDimCeil = 0;
    int64_t outIntCeil = 0;
    int64_t rowStart = 0;
    int64_t rowEnd = 0;
    int64_t normOffset = 0;
    int64_t tmpBufferSize = 0;
    float epsilon = 0.0f;
    int32_t hasWeight = 0;
    int32_t hasBias = 0;
    int32_t hasSmooth = 0;
    float quantFactor = 0.0f;
};

template <typename T, typename U, typename Y, uint8_t OP_CODE>
__aicore__ inline void AdaLayerNormWelford<T, U, Y, OP_CODE>::InitV2(const GmAddr* gmAddr, const AdaLayerNormTilingData *tilingData)
{
    ParseTilingData(tilingData);
    pipe.InitBuffer(xQueue, 1, DATA_COUNT * sizeof(float));
    pipe.InitBuffer(scaleQueue, 1, DATA_COUNT * sizeof(float));
    pipe.InitBuffer(shiftQueue, 1, DATA_COUNT * sizeof(float));
    pipe.InitBuffer(weightBuf, DATA_COUNT * sizeof(float));
    pipe.InitBuffer(biasBuf, DATA_COUNT * sizeof(float));
    pipe.InitBuffer(varBuf, BATCH_COUNT * sizeof(float));
    pipe.InitBuffer(normBuf, DATA_COUNT * sizeof(float));
    if (tmpBufferSize > 0) {
        pipe.InitBuffer(tmpBuf, tmpBufferSize);
    }
    pipe.InitBuffer(outQueue, 1, DATA_COUNT * sizeof(float));
    pipe.InitBuffer(meanQueue, 1, BATCH_COUNT * sizeof(float));
    pipe.InitBuffer(rstdQueue, 1, BATCH_COUNT * sizeof(float));

    xGm.SetGlobalBuffer((__gm__ T*)gmAddr->x);
    scaleGm.SetGlobalBuffer((__gm__ T*)gmAddr->scale);
    shiftGm.SetGlobalBuffer((__gm__ T*)gmAddr->shift);
    weightGm.SetGlobalBuffer((__gm__ U*)gmAddr->weight);
    biasGm.SetGlobalBuffer((__gm__ U*)gmAddr->bias);
    outGm.SetGlobalBuffer((__gm__ T*)gmAddr->out);
    meanGm.SetGlobalBuffer((__gm__ T*)gmAddr->mean);
    rstdGm.SetGlobalBuffer((__gm__ T*)gmAddr->rstd);
}

template <typename T, typename U, typename Y, uint8_t OP_CODE>
__aicore__ inline void AdaLayerNormWelford<T, U, Y, OP_CODE>::InitQuant(const GmAddr* gmAddr, GM_ADDR workspace,
    const AdaLayerNormTilingData *tilingData)
{
    ParseTilingData(tilingData);
    pipe.InitBuffer(xQueue, 1, DATA_COUNT * sizeof(float));
    pipe.InitBuffer(scaleQueue, 1, DATA_COUNT * sizeof(float));
    pipe.InitBuffer(shiftQueue, 1, DATA_COUNT * sizeof(float));
    pipe.InitBuffer(smoothQueue, 1, DATA_COUNT * sizeof(float));
    pipe.InitBuffer(weightBuf, DATA_COUNT * sizeof(float));
    pipe.InitBuffer(biasBuf, DATA_COUNT * sizeof(float));
    pipe.InitBuffer(varBuf, BATCH_COUNT * sizeof(float));
    pipe.InitBuffer(maxBuf, V_LENGTH * sizeof(float));
    pipe.InitBuffer(normBuf, DATA_COUNT * sizeof(float));
    if (tmpBufferSize > 0) {
        pipe.InitBuffer(tmpBuf, tmpBufferSize);
    }
    pipe.InitBuffer(outQueue, 1, DATA_COUNT * sizeof(float));
    pipe.InitBuffer(quantOutQueue, 1, DATA_COUNT * sizeof(int8_t));
    pipe.InitBuffer(meanQueue, 1, BATCH_COUNT * sizeof(float));
    pipe.InitBuffer(rstdQueue, 1, BATCH_COUNT * sizeof(float));
    pipe.InitBuffer(quantScaleQueue, 1, BATCH_COUNT * sizeof(float));

    xGm.SetGlobalBuffer((__gm__ T*)gmAddr->x);
    scaleGm.SetGlobalBuffer((__gm__ T*)gmAddr->scale);
    shiftGm.SetGlobalBuffer((__gm__ T*)gmAddr->shift);
    weightGm.SetGlobalBuffer((__gm__ U*)gmAddr->weight);
    biasGm.SetGlobalBuffer((__gm__ U*)gmAddr->bias);
    smoothGm.SetGlobalBuffer((__gm__ T*)gmAddr->smooth_scales);
    normGm.SetGlobalBuffer((__gm__ float*)workspace);
    quantOutGm.SetGlobalBuffer((__gm__ Y*)gmAddr->out);
    quantScaleGm.SetGlobalBuffer((__gm__ float*)gmAddr->quant_scale);
}

template <typename T, typename U, typename Y, uint8_t OP_CODE>
__aicore__ inline void AdaLayerNormWelford<T, U, Y, OP_CODE>::Process()
{
    if (rowStart >= rowEnd) {
        return;
    }

    weightLocal = weightBuf.Get<float>();
    biasLocal = biasBuf.Get<float>();
    normLocal = normBuf.Get<float>();
    tmpLocal = tmpBuf.Get<uint8_t>();
    meanTmpLocal = weightLocal;
    varTmpLocal = biasLocal;
    varLocal = varBuf.Get<float>();
    meanLocal = meanQueue.AllocTensor<float>();
    rstdLocal = rstdQueue.AllocTensor<float>();
    if constexpr (OP_CODE == QUANT_OP_CODE) {
        quantScaleLocal = quantScaleQueue.AllocTensor<float>();
        maxTmpLocal = maxBuf.Get<float>();
    }
    SliceProcess();
}

template <typename T, typename U, typename Y, uint8_t OP_CODE>
__aicore__ inline void AdaLayerNormWelford<T, U, Y, OP_CODE>::ParseTilingData(const AdaLayerNormTilingData *tilingData)
{
    int32_t blockNum = HALF_BLOCK_NUM;
    if constexpr (std::is_same_v<T, float>) {
        blockNum = FLOAT_BLOCK_NUM;
    }
    quantFactor = GetQuantFactor<Y>();
    seqLen = tilingData->seqLen;
    hiddenDim = tilingData->hiddenDim;
    hiddenDimCeil = CeilA2B(hiddenDim, blockNum) * blockNum;
    outIntCeil = CeilA2B(hiddenDim, INT8_BLOCK_NUM) * INT8_BLOCK_NUM;
    epsilon = tilingData->epsilon;
    hasWeight = tilingData->hasWeight;
    hasBias = tilingData->hasBias;
    hasSmooth = tilingData->hasSmooth;
    sliceSize = tilingData->sliceSize;
    sliceCount = hiddenDim / sliceSize;
    tailSize = hiddenDim % sliceSize;
    tmpBufferSize = tilingData->tmpBufferSize;

    int64_t singleCoreNum = tilingData->singleCoreNum;
    int64_t tailNum = tilingData->tailNum;
    int64_t blockIdx = GetBlockIdx();
    rowStart = singleCoreNum * blockIdx + (blockIdx < tailNum ? blockIdx : tailNum);
    rowEnd = rowStart + singleCoreNum + (blockIdx < tailNum ? 1 : 0);
    normOffset = blockIdx * outIntCeil;
}

template <typename T, typename U, typename Y, uint8_t OP_CODE>
__aicore__ inline void AdaLayerNormWelford<T, U, Y, OP_CODE>::SliceProcess()
{
    int64_t batchCount = 0;
    int64_t rowIdx = rowStart;
    while (rowIdx < rowEnd) {
        int64_t offset = rowIdx * hiddenDim;
        int64_t scaleOffset = (rowIdx / seqLen) * hiddenDim;
        // 计算均值和方差
        ComputeMeanVar(offset, batchCount);
        ComputeAdaLayerNorm(offset, scaleOffset, batchCount);
        if constexpr (OP_CODE == QUANT_OP_CODE) {
            DynamicQuant(offset, batchCount);
        }
        batchCount++;
        rowIdx++;
        if (batchCount == BATCH_COUNT || rowIdx == rowEnd) {
            if constexpr (OP_CODE == QUANT_OP_CODE) {
                quantScaleQueue.EnQue<float>(quantScaleLocal);
                CopyScaleOut(rowIdx - batchCount, batchCount);
                if (rowIdx != rowEnd) {
                    quantScaleLocal = quantScaleQueue.AllocTensor<float>();
                }
            } else {
                if constexpr (!std::is_same_v<T, float>) {
                    LocalTensor<T> meanOut = meanLocal.ReinterpretCast<T>();
                    LocalTensor<T> rstdOut = rstdLocal.ReinterpretCast<T>();
                    Cast(meanOut, meanLocal, RoundMode::CAST_RINT, batchCount);
                    Cast(rstdOut, rstdLocal, RoundMode::CAST_RINT, batchCount);
                    meanQueue.EnQue<T>(meanOut);
                    rstdQueue.EnQue<T>(rstdOut);
                } else {
                    meanQueue.EnQue<float>(meanLocal);
                    rstdQueue.EnQue<float>(rstdLocal);
                }
                CopyMeanRstdOut(rowIdx - batchCount, batchCount);
                if (rowIdx < rowEnd) {
                    meanLocal = meanQueue.AllocTensor<float>();
                    rstdLocal = rstdQueue.AllocTensor<float>();
                }
            }
            batchCount = 0;
        }
    }
}

template <typename T, typename U, typename Y, uint8_t OP_CODE>
__aicore__ inline void AdaLayerNormWelford<T, U, Y, OP_CODE>::ComputeMeanVar(int64_t offset, int64_t batchCount)
{
    Duplicate(meanTmpLocal, 0.0f, sliceSize);
    Duplicate(varTmpLocal, 0.0f, sliceSize);
    PipeBarrier<PIPE_V>();
    int64_t welfordCount = 0;
    while (welfordCount < sliceCount) {
        welfordCount ++;
        LocalTensor<T> xLocal = xQueue.AllocTensor<T>();
        DataCopyPad(xLocal, xGm[offset], {1, static_cast<uint32_t>(sliceSize * sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});
        xQueue.EnQue(xLocal);
        ProcessWelfordUpdate(sliceSize, welfordCount);
        offset += sliceSize;
    }
    if (tailSize > 0) {
        welfordCount ++;
        LocalTensor<T> xLocal = xQueue.AllocTensor<T>();
        DataCopyPad(xLocal, xGm[offset], {1, static_cast<uint32_t>(tailSize * sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});
        xQueue.EnQue(xLocal);
        ProcessWelfordUpdate(tailSize, welfordCount);
    }
    ProcessWelfordFinalize(welfordCount, batchCount);
}

template <typename T, typename U, typename Y, uint8_t OP_CODE>
__aicore__ inline void AdaLayerNormWelford<T, U, Y, OP_CODE>::ComputeAdaLayerNorm(int64_t offset, int64_t scaleOffset, int64_t batchCount)
{
    if constexpr (OP_CODE == QUANT_OP_CODE) {
        Duplicate(maxTmpLocal, 0.0f, sliceSize);
        PipeBarrier<PIPE_V>();
    }
    int64_t h = 0;
    for (int64_t i = 0; i < sliceCount; i ++) {
        CopyInOtherData(h, sliceSize);
        CopyInX(offset + h, sliceSize);
        ProcessNormalize(sliceSize, batchCount);
        CopyInScaleShift(scaleOffset + h, sliceSize);
        Adaption(sliceSize, batchCount);
        if constexpr (OP_CODE == QUANT_OP_CODE) {
            CopyNormOut(normOffset + h, sliceSize);
        } else {
            BaseCopyOut(offset + h, sliceSize);
        }
        h += sliceSize;
    }
    if (tailSize > 0) {
        CopyInOtherData(h, tailSize);
        CopyInX(offset + h, tailSize);
        ProcessNormalize(tailSize, batchCount);
        CopyInScaleShift(scaleOffset + h, tailSize);
        Adaption(tailSize, batchCount);
        if constexpr (OP_CODE == QUANT_OP_CODE) {
            CopyNormOut(normOffset + h, tailSize);
        } else {
            BaseCopyOut(offset + h, tailSize);
        }
    }
}

template <typename T, typename U, typename Y, uint8_t OP_CODE>
__aicore__ inline void AdaLayerNormWelford<T, U, Y, OP_CODE>::DynamicQuant(int64_t offset, int64_t batchCount)
{
    CalculateScale(batchCount);
    event_t eventIdMte3ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
    WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
    int64_t h = 0;
    for (int64_t i = 0; i < sliceCount; i ++) {
        CopyInNorm(normOffset + h, sliceSize);
        ProcessQuant(sliceSize, batchCount);
        QuantCopyOut(offset + h, sliceSize);
        h += sliceSize;
    }
    if (tailSize > 0) {
        CopyInNorm(normOffset + h, tailSize);
        ProcessQuant(tailSize, batchCount);
        QuantCopyOut(offset + h, tailSize);
    }
}

template <typename T, typename U, typename Y, uint8_t OP_CODE>
__aicore__ inline void AdaLayerNormWelford<T, U, Y, OP_CODE>::Adaption(int64_t dataCount, int64_t batchCount)
{
    LocalTensor<T> scaleLocal = scaleQueue.DeQue<T>();
    LocalTensor<T> shiftLocal = shiftQueue.DeQue<T>();
    LocalTensor<T> smoothLocal;
    LocalTensor<OUT_DTYPE> outLocal = outQueue.AllocTensor<OUT_DTYPE>();

    __local_mem__ float* normAddr = (__ubuf__ float*)normLocal.GetPhyAddr();
    __local_mem__ OUT_DTYPE* outAddr = (__ubuf__ OUT_DTYPE*)outLocal.GetPhyAddr();
    __local_mem__ T* scaleAddr = (__ubuf__ T*)scaleLocal.GetPhyAddr();
    __local_mem__ T* shiftAddr = (__ubuf__ T*)shiftLocal.GetPhyAddr();

    __local_mem__ T* smoothAddr;
    __local_mem__ float* maxTmpAddr;
    if constexpr (OP_CODE == QUANT_OP_CODE) {
        if (hasSmooth) {
            smoothLocal = smoothQueue.DeQue<T>();
            smoothAddr = (__ubuf__ T*)smoothLocal.GetPhyAddr();
        }
        maxTmpAddr = (__ubuf__ float*)maxTmpLocal.GetPhyAddr();
    }

    uint16_t tailLength = static_cast<uint16_t>(dataCount % TWO_V_LENGTH);
    uint16_t colLoopTimes = static_cast<uint16_t>(dataCount / TWO_V_LENGTH) + (tailLength > V_LENGTH ? 1 : 0);
    uint32_t rightLength = static_cast<uint32_t>(dataCount - colLoopTimes * V_LENGTH);
    __VEC_SCOPE__
    {
        RegTensor<float> x1;
        RegTensor<float> x2;
        RegTensor<float> scale1;
        RegTensor<float> scale2;
        RegTensor<float> shift1;
        RegTensor<float> shift2;
        RegTensor<float> tmpMax;

        MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
        MaskReg pregMerge = CreateMask<float, MaskPattern::VL1>();
        MaskReg pregLoop;
        uint32_t sreg2 = rightLength;
        if constexpr (OP_CODE == QUANT_OP_CODE) {
            DataCopy<float, LoadDist::DIST_BRC_B32>(tmpMax, maxTmpAddr);
        }
        for (uint16_t j = 0; j < colLoopTimes;j ++) {
            pregLoop = UpdateMask<float>(sreg2);
            DataCopy(x1, normAddr + j * TWO_V_LENGTH);
            DataCopy(x2, normAddr + j * TWO_V_LENGTH + V_LENGTH);
            LoadTensor(scale1, scaleAddr + j * TWO_V_LENGTH, pregFull);
            LoadTensor(scale2, scaleAddr + j * TWO_V_LENGTH + V_LENGTH, pregLoop);
            LoadTensor(shift1, shiftAddr + j * TWO_V_LENGTH, pregFull);
            LoadTensor(shift2, shiftAddr + j * TWO_V_LENGTH + V_LENGTH, pregLoop);
            Adds(scale1, scale1, 1.0f, pregFull);
            Adds(scale2, scale2, 1.0f, pregLoop);
            FusedMulDstAdd(x1, scale1, shift1, pregFull);
            FusedMulDstAdd(x2, scale2, shift2, pregLoop);
            if constexpr (OP_CODE == QUANT_OP_CODE) {
                if (hasSmooth) {
                    RegTensor<float> smooth1;
                    RegTensor<float> smooth2;
                    LoadTensor(smooth1, smoothAddr + j * TWO_V_LENGTH, pregFull);
                    LoadTensor(smooth2, smoothAddr + j * TWO_V_LENGTH + V_LENGTH, pregLoop);
                    Mul(x1, x1, smooth1, pregFull);
                    Mul(x2, x2, smooth2, pregLoop);
                }
                CopyToTensor(outAddr + j * TWO_V_LENGTH, x1, pregFull);
                CopyToTensor(outAddr + j * TWO_V_LENGTH + V_LENGTH, x2, pregLoop);
                Abs(x1, x1, pregFull);
                Abs(x2, x2, pregLoop);
                Max(x1, x1, x2, pregFull);
                Max(tmpMax, tmpMax, x1, pregFull);
            } else {
                CopyToTensor(outAddr + j * TWO_V_LENGTH, x1, pregFull);
                CopyToTensor(outAddr + j * TWO_V_LENGTH + V_LENGTH, x2, pregLoop);
            }
        }
        if (tailLength > 0 && tailLength <= V_LENGTH) {
            pregLoop = UpdateMask<float>(sreg2);
            DataCopy(x1, normAddr + colLoopTimes * TWO_V_LENGTH);
            LoadTensor(scale1, scaleAddr + colLoopTimes * TWO_V_LENGTH, pregLoop);
            LoadTensor(shift1, shiftAddr + colLoopTimes * TWO_V_LENGTH, pregLoop);
            Adds(scale1, scale1, 1.0f, pregLoop);
            FusedMulDstAdd(x1, scale1, shift1, pregLoop);
            if constexpr (OP_CODE == QUANT_OP_CODE) {
                if (hasSmooth) {
                    RegTensor<float> smooth1;
                    LoadTensor(smooth1, smoothAddr + colLoopTimes * TWO_V_LENGTH, pregLoop);
                    Mul(x1, x1, smooth1, pregLoop);
                }
                CopyToTensor(outAddr + colLoopTimes * TWO_V_LENGTH, x1, pregLoop);
                Abs(x1, x1, pregLoop);
                Max(tmpMax, tmpMax, x1, pregFull);
            } else {
                CopyToTensor(outAddr + colLoopTimes * TWO_V_LENGTH, x1, pregLoop);
            }
        }
        if constexpr (OP_CODE == QUANT_OP_CODE) {
            ReduceMax(tmpMax, tmpMax, pregFull);
            DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(maxTmpAddr, tmpMax, pregMerge);
        }
    }
    scaleQueue.FreeTensor(scaleLocal);
    shiftQueue.FreeTensor(shiftLocal);
    if constexpr (OP_CODE == QUANT_OP_CODE) {
        if (hasSmooth) {
            smoothQueue.FreeTensor(smoothLocal);
        }
    }
    outQueue.EnQue<OUT_DTYPE>(outLocal);
}

template <typename T, typename U, typename Y, uint8_t OP_CODE>
__aicore__ inline void AdaLayerNormWelford<T, U, Y, OP_CODE>::ProcessQuant(int64_t dataCount, int64_t batchCount)
{
    LocalTensor<float> xLocal = xQueue.DeQue<float>();
    LocalTensor<Y> quantOutLocal = quantOutQueue.AllocTensor<Y>();

    __local_mem__ float* xAddr = (__ubuf__ float*)xLocal.GetPhyAddr();
    __local_mem__ Y* quantOutAddr = (__ubuf__ Y*)quantOutLocal.GetPhyAddr();
    __local_mem__ float* quantScaleAddr = (__ubuf__ float*)quantScaleLocal[batchCount].GetPhyAddr();

    uint16_t tailLength = static_cast<uint16_t>(dataCount % TWO_V_LENGTH);
    uint16_t colLoopTimes = static_cast<uint16_t>(dataCount / TWO_V_LENGTH) + (tailLength > V_LENGTH ? 1 : 0);
    uint32_t rightLength = static_cast<uint32_t>(dataCount - colLoopTimes * V_LENGTH);
    __VEC_SCOPE__
    {
        RegTensor<float> x1;
        RegTensor<float> x2;
        RegTensor<float> quantScale;
        RegTensor<int16_t> y1Int16;
        RegTensor<int16_t> y2Int16;
        RegTensor<half> y1Fp16;
        RegTensor<half> y2Fp16;
        RegTensor<Y> y1;
        RegTensor<Y> y2;

        MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
        MaskReg pregMerge = CreateMask<float, MaskPattern::VL1>();
        MaskReg pregLoop;

        DataCopy<float, LoadDist::DIST_BRC_B32>(quantScale, quantScaleAddr);
        uint32_t sreg2 = rightLength;
        for (uint16_t j = 0; j < colLoopTimes;j ++) {
            pregLoop = UpdateMask<float>(sreg2);
            DataCopy(x1, xAddr + j * TWO_V_LENGTH);
            DataCopy(x2, xAddr + j * TWO_V_LENGTH + V_LENGTH);
            Div(x1, x1, quantScale, pregFull);
            Div(x2, x2, quantScale, pregLoop);
            if constexpr (std::is_same_v<Y, int8_t>) {
                Cast<int16_t, float, castTraitF32ToI16>(y1Int16, x1, pregFull);
                Cast<int16_t, float, castTraitF32ToI16>(y2Int16, x2, pregLoop);
                Cast<half, int16_t, castTraitI16ToF16>(y1Fp16, y1Int16, pregFull);
                Cast<half, int16_t, castTraitI16ToF16>(y2Fp16, y2Int16, pregLoop);
                Cast<Y, half, castTraitF16ToI8>(y1, y1Fp16, pregFull);
                Cast<Y, half, castTraitF16ToI8>(y2, y2Fp16, pregLoop);
            } else if constexpr (std::is_same_v<Y, hifloat8_t>) {
                Cast<Y, float, castTraitF32Toh8>(y1, x1, pregFull);
                Cast<Y, float, castTraitF32Toh8>(y2, x2, pregLoop);
            } else {
                Cast<Y, float, castTraitF32Tofp8>(y1, x1, pregFull);
                Cast<Y, float, castTraitF32Tofp8>(y2, x2, pregLoop);
            }
            DataCopy<Y, StoreDist::DIST_PACK4_B32>(quantOutAddr + j * TWO_V_LENGTH, y1, pregFull);
            DataCopy<Y, StoreDist::DIST_PACK4_B32>(quantOutAddr + j * TWO_V_LENGTH + V_LENGTH, y2, pregLoop);
        }
        if (tailLength > 0 && tailLength <= V_LENGTH) {
            pregLoop = UpdateMask<float>(sreg2);
            DataCopy(x1, xAddr + colLoopTimes * TWO_V_LENGTH);
            Div(x1, x1, quantScale, pregLoop);
            if constexpr (std::is_same_v<Y, int8_t>) {
                Cast<int16_t, float, castTraitF32ToI16>(y1Int16, x1, pregLoop);
                Cast<half, int16_t, castTraitI16ToF16>(y1Fp16, y1Int16, pregLoop);
                Cast<Y, half, castTraitF16ToI8>(y1, y1Fp16, pregLoop);
            } else if constexpr (std::is_same_v<Y, hifloat8_t>) {
                Cast<Y, float, castTraitF32Toh8>(y1, x1, pregLoop);
            } else {
                Cast<Y, float, castTraitF32Tofp8>(y1, x1, pregLoop);
            }
            DataCopy<Y, StoreDist::DIST_PACK4_B32>(quantOutAddr + colLoopTimes * TWO_V_LENGTH, y1, pregLoop);
        }
    }

    quantOutQueue.EnQue<Y>(quantOutLocal);
    xQueue.FreeTensor(xLocal);
}
}  // namespace AdaLayerNormNS

#endif  // ADA_LAYER_NORM_WELFORD_H
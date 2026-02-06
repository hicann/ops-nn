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
    __aicore__ inline void WelfordInitialize(uint32_t computeLength);
    __aicore__ inline void ProcessWelfordUpdate(int64_t computeLength, int64_t welfordCount);
    __aicore__ inline void ProcessWelfordFinalize(int64_t welfordCount, int64_t batchCount);
    __aicore__ inline void ProcessNormalize(uint32_t dataCount, int64_t batchCount);
    __aicore__ inline void Adaption(uint32_t dataCount);
    __aicore__ inline void DynamicQuant(int64_t offset, int64_t batchCount);
    __aicore__ inline void CalculateScale(int64_t batchCount);
    __aicore__ inline void ProcessQuant(uint32_t dataCount, int64_t batchCount);

    __aicore__ inline void CopyInOtherData(int64_t offset, int64_t len);
    __aicore__ inline void CopyInScaleShift(int64_t offset, int64_t smoothOffset, int64_t len);
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
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> xQueue;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> scaleQueue;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> shiftQueue;
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
    pipe.InitBuffer(xQueue, DOUBLE_BUFFER, WELFORD_COUNT * sizeof(float));
    pipe.InitBuffer(scaleQueue, DOUBLE_BUFFER, WELFORD_COUNT * sizeof(float));
    pipe.InitBuffer(shiftQueue, DOUBLE_BUFFER, WELFORD_COUNT * sizeof(float));
    pipe.InitBuffer(varBuf, BATCH_COUNT * sizeof(float));
    pipe.InitBuffer(normBuf, WELFORD_COUNT * sizeof(float));
    if (tmpBufferSize > 0) {
        pipe.InitBuffer(tmpBuf, tmpBufferSize);
    }
    pipe.InitBuffer(outQueue, DOUBLE_BUFFER, WELFORD_COUNT * sizeof(float));
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
    pipe.InitBuffer(xQueue, DOUBLE_BUFFER, WELFORD_COUNT * sizeof(float));
    pipe.InitBuffer(scaleQueue, DOUBLE_BUFFER, WELFORD_COUNT * sizeof(float));
    pipe.InitBuffer(shiftQueue, DOUBLE_BUFFER, WELFORD_COUNT * sizeof(float));
    pipe.InitBuffer(varBuf, BATCH_COUNT * sizeof(float));
    pipe.InitBuffer(maxBuf, V_LENGTH * sizeof(float));
    pipe.InitBuffer(normBuf, WELFORD_COUNT * sizeof(float));
    if (tmpBufferSize > 0) {
        pipe.InitBuffer(tmpBuf, tmpBufferSize);
    }
    pipe.InitBuffer(outQueue, DOUBLE_BUFFER, WELFORD_COUNT * sizeof(float));
    pipe.InitBuffer(quantOutQueue, DOUBLE_BUFFER, WELFORD_COUNT * sizeof(int8_t));
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

    normLocal = normBuf.Get<float>();
    tmpLocal = tmpBuf.Get<uint8_t>();
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
    WelfordInitialize(sliceSize);
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
        Duplicate(maxTmpLocal, 0.0f, V_LENGTH);
        PipeBarrier<PIPE_V>();
    }
    int64_t h = 0;
    for (int64_t i = 0; i < sliceCount; i ++) {
        CopyInX(offset + h, sliceSize);
        CopyInOtherData(h, sliceSize);
        ProcessNormalize(sliceSize, batchCount);
        CopyInScaleShift(scaleOffset + h, h, sliceSize);
        Adaption(sliceSize);
        if constexpr (OP_CODE == QUANT_OP_CODE) {
            CopyNormOut(normOffset + h, sliceSize);
        } else {
            BaseCopyOut(offset + h, sliceSize);
        }
        h += sliceSize;
    }
    if (tailSize > 0) {
        CopyInX(offset + h, tailSize);
        CopyInOtherData(h, tailSize);
        ProcessNormalize(tailSize, batchCount);
        CopyInScaleShift(scaleOffset + h, h, tailSize);
        Adaption(tailSize);
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
__aicore__ inline void AdaLayerNormWelford<T, U, Y, OP_CODE>::Adaption(uint32_t dataCount)
{
    LocalTensor<T> scaleLocal = scaleQueue.DeQue<T>();
    LocalTensor<T> shiftLocal = shiftQueue.DeQue<T>();
    LocalTensor<T> smoothLocal;
    LocalTensor<OUT_DTYPE> outLocal = outQueue.AllocTensor<OUT_DTYPE>();

    __ubuf__ float* normAddr = (__ubuf__ float*)normLocal.GetPhyAddr();
    __ubuf__ OUT_DTYPE* outAddr = (__ubuf__ OUT_DTYPE*)outLocal.GetPhyAddr();
    __ubuf__ T* scaleAddr = (__ubuf__ T*)scaleLocal.GetPhyAddr();
    __ubuf__ T* shiftAddr = (__ubuf__ T*)shiftLocal.GetPhyAddr();
    __ubuf__ T* smoothAddr;
    __ubuf__ float* maxTmpAddr;
    if constexpr (OP_CODE == QUANT_OP_CODE) {
        if (hasSmooth) {
            smoothLocal = xQueue.DeQue<T>();
            smoothAddr = (__ubuf__ T*)smoothLocal.GetPhyAddr();
        }
        maxTmpAddr = (__ubuf__ float*)maxTmpLocal.GetPhyAddr();
    }

    if (hasSmooth) {
        WelfordAdaptionVF<T, OUT_DTYPE, OP_CODE, true>(dataCount, normAddr, scaleAddr, shiftAddr, 
                                                       smoothAddr, maxTmpAddr, outAddr);
    } else {
        WelfordAdaptionVF<T, OUT_DTYPE, OP_CODE, false>(dataCount, normAddr, scaleAddr, shiftAddr, 
                                                        nullptr, maxTmpAddr, outAddr);
    }
    scaleQueue.FreeTensor(scaleLocal);
    shiftQueue.FreeTensor(shiftLocal);
    if constexpr (OP_CODE == QUANT_OP_CODE) {
        if (hasSmooth) {
            xQueue.FreeTensor(smoothLocal);
        }
    }
    outQueue.EnQue<OUT_DTYPE>(outLocal);
}

template <typename T, typename U, typename Y, uint8_t OP_CODE>
__aicore__ inline void AdaLayerNormWelford<T, U, Y, OP_CODE>::ProcessQuant(uint32_t dataCount, int64_t batchCount)
{
    LocalTensor<float> xLocal = xQueue.DeQue<float>();
    LocalTensor<Y> quantOutLocal = quantOutQueue.AllocTensor<Y>();
    __ubuf__ float* xAddr = (__ubuf__ float*)xLocal.GetPhyAddr();
    __ubuf__ Y* quantOutAddr = (__ubuf__ Y*)quantOutLocal.GetPhyAddr();
    __ubuf__ float* quantScaleAddr = (__ubuf__ float*)quantScaleLocal[batchCount].GetPhyAddr();

    uint16_t colLoopTimes = static_cast<uint16_t>(CeilA2B(dataCount, V_LENGTH));
    __VEC_SCOPE__
    {
        RegTensor<float> x;
        RegTensor<float> quantScale;
        RegTensor<int16_t> yInt16;
        RegTensor<half> yFp16;
        RegTensor<Y> y;

        MaskReg pregMerge = CreateMask<float, MaskPattern::VL1>();
        MaskReg pregLoop;
        DataCopy<float, LoadDist::DIST_BRC_B32>(quantScale, quantScaleAddr);
        for (uint16_t j = 0; j < colLoopTimes;j ++) {
            pregLoop = UpdateMask<float>(dataCount);
            DataCopy(x, xAddr + j * V_LENGTH);
            Div(x, x, quantScale, pregLoop);
            if constexpr (std::is_same_v<Y, int8_t>) {
                Cast<int16_t, float, castTraitF32ToI16>(yInt16, x, pregLoop);
                Cast<half, int16_t, castTraitI16ToF16>(yFp16, yInt16, pregLoop);
                Cast<Y, half, castTraitF16ToI8>(y, yFp16, pregLoop);
            } else if constexpr (std::is_same_v<Y, hifloat8_t>) {
                Cast<Y, float, castTraitF32Toh8>(y, x, pregLoop);
            } else {
                Cast<Y, float, castTraitF32Tofp8>(y, x, pregLoop);
            }
            DataCopy<Y, StoreDist::DIST_PACK4_B32>(quantOutAddr + j * V_LENGTH, y, pregLoop);
        }
    }

    quantOutQueue.EnQue<Y>(quantOutLocal);
    xQueue.FreeTensor(xLocal);
}
}  // namespace AdaLayerNormNS

#endif  // ADA_LAYER_NORM_WELFORD_H
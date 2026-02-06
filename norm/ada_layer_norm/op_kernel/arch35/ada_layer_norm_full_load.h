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
 * \file ada_layer_norm_full_load.h
 * \brief
 */
#ifndef ADA_LAYER_NORM_FULL_LOAD_H
#define ADA_LAYER_NORM_FULL_LOAD_H

#include "ada_layer_norm_common.h"

namespace AdaLayerNormNS {
using namespace AscendC;

template <typename T, typename U, typename Y, uint8_t OP_CODE>
class AdaLayerNormFullLoad {
public:
    __aicore__ inline AdaLayerNormFullLoad(){};
    __aicore__ inline void InitV2(const GmAddr* gmAddr, const AdaLayerNormTilingData *tilingData);
    __aicore__ inline void InitQuant(const GmAddr* gmAddr, GM_ADDR workspace, const AdaLayerNormTilingData *tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ParseTilingData(const AdaLayerNormTilingData *tilingData);
    __aicore__ inline void FastProcess();
    __aicore__ inline void ProcessLayerNorm(RowRange range, int64_t batchCount);
    __aicore__ inline void Adaption(uint16_t rowNum, int64_t batchCount);
    __aicore__ inline void DynamicQuant(uint16_t rowNum, int64_t batchCount);

    __aicore__ inline void CopyInOtherData();
    __aicore__ inline void CopyInScaleShift(int64_t offset, int64_t localOffset, int64_t repeatNum);
    __aicore__ inline void CopyInX(int64_t offset, uint16_t blockCount);
    __aicore__ inline void BaseCopyOut(int64_t offset, uint16_t blockCount);
    __aicore__ inline void QuantCopyOut(int64_t offset, uint16_t blockCount);
    __aicore__ inline void CopyMeanRstdOut(int64_t offset, int64_t len);
    __aicore__ inline void CopyScaleOut(int64_t offset, int64_t len);

private:
    using OUT_DTYPE = std::conditional_t<OP_CODE == QUANT_OP_CODE, float, T>;
    TPipe pipe;
    RowRange range;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> xQueue;
    TBuf<TPosition::VECCALC> scaleBuf;
    TBuf<TPosition::VECCALC> shiftBuf;
    TBuf<TPosition::VECCALC> smoothBuf;
    TBuf<TPosition::VECCALC> weightBuf;
    TBuf<TPosition::VECCALC> biasBuf;
    TBuf<TPosition::VECCALC> normBuf;
    TBuf<TPosition::VECCALC> tmpBuf;
    TQue<QuePosition::VECOUT, DOUBLE_BUFFER> outQueue;
    TQue<QuePosition::VECOUT, DOUBLE_BUFFER> quantOutQueue;
    TQue<QuePosition::VECOUT, 1> meanQueue;
    TQue<QuePosition::VECOUT, 1> rstdQueue;
    TQue<QuePosition::VECOUT, 1> quantScaleQueue;

    GlobalTensor<T> xGm;
    GlobalTensor<T> scaleGm;
    GlobalTensor<T> shiftGm;
    GlobalTensor<T> smoothGm;
    GlobalTensor<U> weightGm;
    GlobalTensor<U> biasGm;
    GlobalTensor<T> outGm;
    GlobalTensor<T> meanGm;
    GlobalTensor<T> rstdGm;
    GlobalTensor<Y> quantOutGm;
    GlobalTensor<float> quantScaleGm;

    LocalTensor<float> weightLocal;
    LocalTensor<float> biasLocal;
    LocalTensor<T> smoothLocal;
    LocalTensor<T> scaleLocal;
    LocalTensor<T> shiftLocal;
    LocalTensor<float> normLocal;
    LocalTensor<uint8_t> tmpLocal;
    LocalTensor<float> meanLocal;
    LocalTensor<float> rstdLocal;
    LocalTensor<float> quantScaleLocal;

    int64_t rowNum = 0;
    int64_t seqLen = 0;
    uint32_t hiddenDim = 0;
    uint32_t hiddenDimCeil = 0;
    uint32_t outIntCeil = 0;
    int64_t rowStart = 0;
    int64_t rowEnd = 0;
    int64_t tmpBufferSize = 0;
    float epsilon = 0.0f;
    int32_t hasWeight = 0;
    int32_t hasBias = 0;
    int32_t hasSmooth = 0;
    float quantFactor = 0.0f;
    LayerNormSeparateTiling layerNormTiling;
};

template <typename T, typename U, typename Y, uint8_t OP_CODE>
__aicore__ inline void AdaLayerNormFullLoad<T, U, Y, OP_CODE>::InitV2(const GmAddr* gmAddr, const AdaLayerNormTilingData *tilingData)
{
    ParseTilingData(tilingData);
    pipe.InitBuffer(xQueue, DOUBLE_BUFFER, DATA_COUNT * sizeof(float));
    pipe.InitBuffer(scaleBuf, DATA_COUNT * sizeof(float));
    pipe.InitBuffer(shiftBuf, DATA_COUNT * sizeof(float));
    pipe.InitBuffer(weightBuf, DATA_COUNT * sizeof(float));
    pipe.InitBuffer(biasBuf, DATA_COUNT * sizeof(float));
    pipe.InitBuffer(normBuf, DATA_COUNT * sizeof(float));
    if (tmpBufferSize > 0) {
        pipe.InitBuffer(tmpBuf, tmpBufferSize);
    }
    pipe.InitBuffer(outQueue, DOUBLE_BUFFER, DATA_COUNT * sizeof(float));
    pipe.InitBuffer(meanQueue, 1, BATCH_COUNT * sizeof(float));
    pipe.InitBuffer(rstdQueue, 1, BATCH_COUNT * sizeof(float));

    xGm.SetGlobalBuffer((__gm__ T*)gmAddr->x);
    weightGm.SetGlobalBuffer((__gm__ U*)gmAddr->weight);
    biasGm.SetGlobalBuffer((__gm__ U*)gmAddr->bias);
    scaleGm.SetGlobalBuffer((__gm__ T*)gmAddr->scale);
    shiftGm.SetGlobalBuffer((__gm__ T*)gmAddr->shift);
    outGm.SetGlobalBuffer((__gm__ T*)gmAddr->out);
    meanGm.SetGlobalBuffer((__gm__ T*)gmAddr->mean);
    rstdGm.SetGlobalBuffer((__gm__ T*)gmAddr->rstd);
}

template <typename T, typename U, typename Y, uint8_t OP_CODE>
__aicore__ inline void AdaLayerNormFullLoad<T, U, Y, OP_CODE>::InitQuant(const GmAddr* gmAddr, GM_ADDR workspace,
    const AdaLayerNormTilingData *tilingData)
{
    ParseTilingData(tilingData);
    pipe.InitBuffer(xQueue, DOUBLE_BUFFER, DATA_COUNT * sizeof(float));
    pipe.InitBuffer(scaleBuf, DATA_COUNT * sizeof(float));
    pipe.InitBuffer(shiftBuf, DATA_COUNT * sizeof(float));
    pipe.InitBuffer(smoothBuf, DATA_COUNT * sizeof(float));
    pipe.InitBuffer(weightBuf, DATA_COUNT * sizeof(float));
    pipe.InitBuffer(biasBuf, DATA_COUNT * sizeof(float));
    pipe.InitBuffer(normBuf, DATA_COUNT * sizeof(float));
    if (tmpBufferSize > 0) {
        pipe.InitBuffer(tmpBuf, tmpBufferSize);
    }
    pipe.InitBuffer(outQueue, 1, DATA_COUNT * sizeof(float));
    pipe.InitBuffer(quantOutQueue, DOUBLE_BUFFER, DATA_COUNT * sizeof(int8_t));
    pipe.InitBuffer(meanQueue, 1, BATCH_COUNT * sizeof(float));
    pipe.InitBuffer(rstdQueue, 1, BATCH_COUNT * sizeof(float));
    pipe.InitBuffer(quantScaleQueue, 1, BATCH_COUNT * sizeof(float));

    xGm.SetGlobalBuffer((__gm__ T*)gmAddr->x);
    scaleGm.SetGlobalBuffer((__gm__ T*)gmAddr->scale);
    shiftGm.SetGlobalBuffer((__gm__ T*)gmAddr->shift);
    weightGm.SetGlobalBuffer((__gm__ U*)gmAddr->weight);
    biasGm.SetGlobalBuffer((__gm__ U*)gmAddr->bias);
    smoothGm.SetGlobalBuffer((__gm__ T*)gmAddr->smooth_scales);
    quantOutGm.SetGlobalBuffer((__gm__ Y*)gmAddr->out);
    quantScaleGm.SetGlobalBuffer((__gm__ float*)gmAddr->quant_scale);
}

template <typename T, typename U, typename Y, uint8_t OP_CODE>
__aicore__ inline void AdaLayerNormFullLoad<T, U, Y, OP_CODE>::Process()
{
    if (rowStart >= rowEnd) {
        return;
    }

    scaleLocal = scaleBuf.Get<T>();
    shiftLocal = shiftBuf.Get<T>();
    weightLocal = weightBuf.Get<float>();
    biasLocal = biasBuf.Get<float>();
    normLocal = normBuf.Get<float>();
    tmpLocal = tmpBuf.Get<uint8_t>();
    meanLocal = meanQueue.AllocTensor<float>();
    rstdLocal = rstdQueue.AllocTensor<float>();
    if constexpr (OP_CODE == QUANT_OP_CODE) {
        smoothLocal = smoothBuf.Get<T>();
        quantScaleLocal = quantScaleQueue.AllocTensor<float>();
    }
    CopyInOtherData();
    FastProcess();
}

template <typename T, typename U, typename Y, uint8_t OP_CODE>
__aicore__ inline void AdaLayerNormFullLoad<T, U, Y, OP_CODE>::ParseTilingData(const AdaLayerNormTilingData *tilingData)
{
    int32_t blockNum = HALF_BLOCK_NUM;
    if constexpr (std::is_same_v<T, float>) {
        blockNum = FLOAT_BLOCK_NUM;
    }
    seqLen = tilingData->seqLen;
    hiddenDim = tilingData->hiddenDim;
    hiddenDimCeil = CeilA2B(hiddenDim, blockNum) * blockNum;
    outIntCeil = CeilA2B(hiddenDim, INT8_BLOCK_NUM) * INT8_BLOCK_NUM;
    epsilon = tilingData->epsilon;
    hasWeight = tilingData->hasWeight;
    hasBias = tilingData->hasBias;
    hasSmooth = tilingData->hasSmooth;
    rowNum = tilingData->rowNum;
    tmpBufferSize = tilingData->tmpBufferSize;
    layerNormTiling = tilingData->layerNormTiling;
    quantFactor = GetQuantFactor<Y>();

    int64_t singleCoreNum = tilingData->singleCoreNum;
    int64_t tailNum = tilingData->tailNum;
    int64_t blockIdx = GetBlockIdx();
    rowStart = singleCoreNum * blockIdx + (blockIdx < tailNum ? blockIdx : tailNum);
    rowEnd = rowStart + singleCoreNum + (blockIdx < tailNum ? 1 : 0);
}

template <typename T, typename U, typename Y, uint8_t OP_CODE>
__aicore__ inline void AdaLayerNormFullLoad<T, U, Y, OP_CODE>::FastProcess()
{
    int64_t lastBatchStart = -1;
    int64_t lastBatchEnd = -1;
    int64_t rowIdx = rowStart;
    int64_t batchCount = 0;
    while (rowIdx < rowEnd) {
        range.rowStart = rowIdx;
        range.rowEnd = Min(rowEnd, range.rowStart + rowNum);
        range.batchStart = range.rowStart / seqLen;
        range.batchEnd = (range.rowEnd - 1) / seqLen + 1;
        range.actualRowNum = range.rowEnd - range.rowStart;
        range.dataCount = range.actualRowNum > 1 ? range.actualRowNum * hiddenDimCeil : hiddenDim;

        // 拷入
        bool scaleShiftReuse = (lastBatchStart == range.batchStart && lastBatchEnd == range.batchEnd);
        if (!scaleShiftReuse) {
            int64_t rowNum = 0;
            for (int64_t batchIdx = range.batchStart; batchIdx < range.batchEnd; batchIdx++) {
                int64_t repeatNum = seqLen;
                if (batchIdx == range.batchEnd - 1) {
                    repeatNum = range.actualRowNum - rowNum;
                } else if (batchIdx == range.batchStart) {
                    repeatNum = Min(seqLen - (range.rowStart % seqLen), range.actualRowNum);
                }
                CopyInScaleShift(batchIdx * hiddenDim, rowNum * hiddenDimCeil, repeatNum);
                rowNum += repeatNum;
            }
            lastBatchStart = range.batchStart;
            lastBatchEnd = range.batchEnd;
        }
        CopyInX(range.rowStart * hiddenDim, range.actualRowNum);

        // 计算 & 拷出
        ProcessLayerNorm(range, batchCount);
        Adaption(range.actualRowNum, batchCount);
        if constexpr (OP_CODE == QUANT_OP_CODE) {
            DynamicQuant(range.actualRowNum, batchCount);
            QuantCopyOut(range.rowStart * hiddenDim, range.actualRowNum);
        } else {
            BaseCopyOut(range.rowStart * hiddenDim, range.actualRowNum);
        }

        batchCount += range.actualRowNum;
        rowIdx = range.rowEnd;
        if (batchCount + rowNum > BATCH_COUNT || rowIdx == rowEnd) {
            if constexpr (OP_CODE == QUANT_OP_CODE) {
                quantScaleQueue.EnQue<float>(quantScaleLocal);
                CopyScaleOut(rowIdx - batchCount, batchCount);
                if (rowIdx != rowEnd) {
                    quantScaleLocal = quantScaleQueue.AllocTensor<float>();
                }
            } else {
                if constexpr (std::is_same_v<T, float>) {
                    meanQueue.EnQue<float>(meanLocal);
                    rstdQueue.EnQue<float>(rstdLocal);
                } else {
                    LocalTensor<T> meanOut = meanLocal.ReinterpretCast<T>();
                    LocalTensor<T> rstdOut = rstdLocal.ReinterpretCast<T>();
                    Cast(meanOut, meanLocal, RoundMode::CAST_RINT, batchCount);
                    Cast(rstdOut, rstdLocal, RoundMode::CAST_RINT, batchCount);
                    meanQueue.EnQue<T>(meanOut);
                    rstdQueue.EnQue<T>(rstdOut);
                }
                CopyMeanRstdOut(rowIdx - batchCount, batchCount);
                if (rowIdx != rowEnd) {
                    meanLocal = meanQueue.AllocTensor<float>();
                    rstdLocal = rstdQueue.AllocTensor<float>();
                }
            }
            batchCount = 0;
        }
    }
}

template <typename T, typename U, typename Y, uint8_t OP_CODE>
__aicore__ inline void AdaLayerNormFullLoad<T, U, Y, OP_CODE>::ProcessLayerNorm(RowRange range, int64_t batchCount)
{
    LocalTensor<float> xLocal;
    if constexpr (std::is_same_v<T, float>) {
        xLocal = xQueue.DeQue<float>();
    } else {
        LocalTensor<T> xFp16 = xQueue.DeQue<T>();
        xLocal = xFp16.template ReinterpretCast<float>();
        Cast(xLocal, xFp16[DATA_COUNT], RoundMode::CAST_NONE, range.dataCount);
        PipeBarrier<PIPE_V>();
    }

    LayerNormPara para;
    para.aLength = range.actualRowNum;
    para.rLength = hiddenDim;
    para.rLengthWithPadding = hiddenDimCeil;
    if (hasWeight && hasBias) {
        LayerNorm<float, float, true, hasGammaBetaConfig>(
            normLocal, meanLocal[batchCount], rstdLocal[batchCount], xLocal, weightLocal, biasLocal, 
            epsilon, tmpLocal, para, layerNormTiling);
    } else if (!hasWeight && hasBias) {
        LayerNorm<float, float, true, noGammaHasBetaConfig>(
            normLocal, meanLocal[batchCount], rstdLocal[batchCount], xLocal, weightLocal, biasLocal, 
            epsilon, tmpLocal, para, layerNormTiling);
    } else if (hasWeight && !hasBias) {
        LayerNorm<float, float, true, hasGammaNoBetaConfig>(
            normLocal, meanLocal[batchCount], rstdLocal[batchCount], xLocal, weightLocal, biasLocal, 
            epsilon, tmpLocal, para, layerNormTiling);
    } else {
        LayerNorm<float, float, true, noGammaNoBetaConfig>(
            normLocal, meanLocal[batchCount], rstdLocal[batchCount], xLocal, weightLocal, biasLocal, 
            epsilon, tmpLocal, para, layerNormTiling);
    }
    PipeBarrier<PIPE_V>();
    xQueue.FreeTensor(xLocal);
}

template <typename T, typename U, typename Y, uint8_t OP_CODE>
__aicore__ inline void AdaLayerNormFullLoad<T, U, Y, OP_CODE>::Adaption(uint16_t rowNum, int64_t batchCount)
{
    LocalTensor<OUT_DTYPE> outLocal = outQueue.AllocTensor<OUT_DTYPE>();
    __ubuf__ float* normAddr = (__ubuf__ float*)normLocal.GetPhyAddr();
    __ubuf__ OUT_DTYPE* outAddr = (__ubuf__ OUT_DTYPE*)outLocal.GetPhyAddr();
    __ubuf__ T* scaleAddr = (__ubuf__ T*)scaleLocal.GetPhyAddr();
    __ubuf__ T* shiftAddr = (__ubuf__ T*)shiftLocal.GetPhyAddr();
    __ubuf__ T* smoothAddr;
    __ubuf__ float* quantScaleAddr;
    if constexpr (OP_CODE == QUANT_OP_CODE) {
        smoothAddr = (__ubuf__ T*)smoothLocal.GetPhyAddr();
        quantScaleAddr = (__ubuf__ float*)quantScaleLocal[batchCount].GetPhyAddr();
    }

    if (hasSmooth) {
        AdaptionVF<T, OUT_DTYPE, OP_CODE, true>(rowNum, hiddenDim, hiddenDimCeil, quantFactor, normAddr, 
            scaleAddr, shiftAddr, smoothAddr, outAddr, quantScaleAddr);
    } else {
        AdaptionVF<T, OUT_DTYPE, OP_CODE, false>(rowNum, hiddenDim, hiddenDimCeil, quantFactor, normAddr, 
            scaleAddr, shiftAddr, smoothAddr, outAddr, quantScaleAddr);
    }
    outQueue.EnQue<OUT_DTYPE>(outLocal);
}

template <typename T, typename U, typename Y, uint8_t OP_CODE>
__aicore__ inline void AdaLayerNormFullLoad<T, U, Y, OP_CODE>::DynamicQuant(uint16_t rowNum, int64_t batchCount) {
    uint16_t rowLoopTimes = rowNum >> 1;
    uint16_t tailLoopTimes = rowNum & 1;
    uint16_t colLoopTimes = static_cast<uint16_t>(CeilA2B(hiddenDim, V_LENGTH));

    LocalTensor<OUT_DTYPE> outLocal = outQueue.DeQue<OUT_DTYPE>();
    LocalTensor<Y> quantOutLocal = quantOutQueue.AllocTensor<Y>();
    __ubuf__ OUT_DTYPE* outAddr = (__ubuf__ OUT_DTYPE*)outLocal.GetPhyAddr();
    __ubuf__ OUT_DTYPE* outAddr2 = outAddr + rowLoopTimes * hiddenDimCeil;
    __ubuf__ Y* quantOutAddr = (__ubuf__ Y*)quantOutLocal.GetPhyAddr();
    __ubuf__ Y* quantOutAddr2 = quantOutAddr + rowLoopTimes * outIntCeil;
    __ubuf__ float* quantScaleAddr = (__ubuf__ float*)quantScaleLocal[batchCount].GetPhyAddr();
    __VEC_SCOPE__
    {
        RegTensor<float> x1;
        RegTensor<float> x2;
        RegTensor<float> quantScale1;
        RegTensor<float> quantScale2;
        RegTensor<int16_t> y1Int16;
        RegTensor<int16_t> y2Int16;
        RegTensor<half> y1Fp16;
        RegTensor<half> y2Fp16;
        RegTensor<Y> y1;
        RegTensor<Y> y2;

        MaskReg pregLoop;
        for (uint16_t i = 0; i < rowLoopTimes; i++) {
            DataCopy<float, LoadDist::DIST_BRC_B32>(quantScale1, quantScaleAddr + i);
            DataCopy<float, LoadDist::DIST_BRC_B32>(quantScale2, quantScaleAddr + i + rowLoopTimes);
            uint32_t sreg = hiddenDim;
            for (uint16_t j = 0; j < colLoopTimes;j ++) {
                pregLoop = UpdateMask<float>(sreg);
                DataCopy(x1, outAddr + j * V_LENGTH);
                DataCopy(x2, outAddr2 + j * V_LENGTH);
                Div(x1, x1, quantScale1, pregLoop);
                Div(x2, x2, quantScale2, pregLoop);
                if constexpr (std::is_same_v<Y, hifloat8_t>) {
                    Cast<Y, float, castTraitF32Toh8>(y1, x1, pregLoop);
                    Cast<Y, float, castTraitF32Toh8>(y2, x2, pregLoop);
                } else if constexpr (std::is_same_v<Y, int8_t>) {
                    Cast<int16_t, float, castTraitF32ToI16>(y1Int16, x1, pregLoop);
                    Cast<int16_t, float, castTraitF32ToI16>(y2Int16, x2, pregLoop);
                    Cast<half, int16_t, castTraitI16ToF16>(y1Fp16, y1Int16, pregLoop);
                    Cast<half, int16_t, castTraitI16ToF16>(y2Fp16, y2Int16, pregLoop);
                    Cast<Y, half, castTraitF16ToI8>(y1, y1Fp16, pregLoop);
                    Cast<Y, half, castTraitF16ToI8>(y2, y2Fp16, pregLoop);
                } else {
                    Cast<Y, float, castTraitF32Tofp8>(y1, x1, pregLoop);
                    Cast<Y, float, castTraitF32Tofp8>(y2, x2, pregLoop);
                }
                DataCopy<Y, StoreDist::DIST_PACK4_B32>(quantOutAddr + j * V_LENGTH, y1, pregLoop);
                DataCopy<Y, StoreDist::DIST_PACK4_B32>(quantOutAddr2 + j * V_LENGTH, y2, pregLoop);
            }
            outAddr += hiddenDimCeil;
            outAddr2 += hiddenDimCeil;
            quantOutAddr += outIntCeil;
            quantOutAddr2 += outIntCeil;
        }

        for (uint16_t i = 0; i < tailLoopTimes; i++) {
            DataCopy<float, LoadDist::DIST_BRC_B32>(quantScale1, quantScaleAddr + rowLoopTimes + rowLoopTimes);
            uint32_t sreg = hiddenDim;
            for (uint16_t j = 0; j < colLoopTimes;j ++) {
                pregLoop = UpdateMask<float>(sreg);
                DataCopy(x1, outAddr2 + j * V_LENGTH);
                Div(x1, x1, quantScale1, pregLoop);
                if constexpr (std::is_same_v<Y, hifloat8_t>) {
                    Cast<Y, float, castTraitF32Toh8>(y1, x1, pregLoop);
                } else if constexpr (std::is_same_v<Y, int8_t>) {
                    Cast<int16_t, float, castTraitF32ToI16>(y1Int16, x1, pregLoop);
                    Cast<half, int16_t, castTraitI16ToF16>(y1Fp16, y1Int16, pregLoop);
                    Cast<Y, half, castTraitF16ToI8>(y1, y1Fp16, pregLoop);
                } else {
                    Cast<Y, float, castTraitF32Tofp8>(y1, x1, pregLoop);
                }
                DataCopy<Y, StoreDist::DIST_PACK4_B32>(quantOutAddr2 + j * V_LENGTH, y1, pregLoop);
            }
        }
    }
    
    quantOutQueue.EnQue<Y>(quantOutLocal);
    outQueue.FreeTensor(outLocal);
}
}  // namespace AdaLayerNormNS

#endif  // ADA_LAYER_NORM_FULL_LOAD_H
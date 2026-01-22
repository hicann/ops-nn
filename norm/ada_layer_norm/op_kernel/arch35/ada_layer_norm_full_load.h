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
    __aicore__ inline void Adaption(RowRange range, int64_t batchCount);

    __aicore__ inline void CopyInOtherData();
    __aicore__ inline void CopyInScaleShift(int64_t offset, uint16_t blockCount, int64_t len);
    __aicore__ inline void CopyInX(int64_t offset, uint16_t blockCount, int64_t len);
    __aicore__ inline void BaseCopyOut(int64_t offset, uint16_t blockCount, int64_t len);
    __aicore__ inline void QuantCopyOut(int64_t offset, uint16_t blockCount, int64_t len);
    __aicore__ inline void CopyMeanRstdOut(int64_t offset, int64_t len);
    __aicore__ inline void CopyScaleOut(int64_t offset, int64_t len);

private:
    using OUT_DTYPE = std::conditional_t<OP_CODE == QUANT_OP_CODE, float, T>;
    TPipe pipe;
    RowRange range;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> xQueue;
    TQue<QuePosition::VECIN, 1> scaleQueue;
    TQue<QuePosition::VECIN, 1> shiftQueue;
    TQue<QuePosition::VECIN, 1> smoothQueue;
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
    int64_t hiddenDim = 0;
    int64_t hiddenDimCeil = 0;
    int64_t outIntCeil = 0;
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
    pipe.InitBuffer(scaleQueue, 1, DATA_COUNT * sizeof(float));
    pipe.InitBuffer(shiftQueue, 1, DATA_COUNT * sizeof(float));
    pipe.InitBuffer(weightBuf, DATA_COUNT * sizeof(float));
    pipe.InitBuffer(biasBuf, DATA_COUNT * sizeof(float));
    pipe.InitBuffer(normBuf, DATA_COUNT * sizeof(float));
    if (tmpBufferSize > 0) {
        pipe.InitBuffer(tmpBuf, tmpBufferSize);
    }
    pipe.InitBuffer(outQueue, DOUBLE_BUFFER, DATA_COUNT * sizeof(float));
    pipe.InitBuffer(meanQueue, DOUBLE_BUFFER, BATCH_COUNT * sizeof(float));
    pipe.InitBuffer(rstdQueue, DOUBLE_BUFFER, BATCH_COUNT * sizeof(float));

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
    pipe.InitBuffer(scaleQueue, 1, DATA_COUNT * sizeof(float));
    pipe.InitBuffer(shiftQueue, 1, DATA_COUNT * sizeof(float));
    pipe.InitBuffer(smoothQueue, 1, DATA_COUNT * sizeof(float));
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

    weightLocal = weightBuf.Get<float>();
    biasLocal = biasBuf.Get<float>();
    normLocal = normBuf.Get<float>();
    tmpLocal = tmpBuf.Get<uint8_t>();
    meanLocal = meanQueue.AllocTensor<float>();
    rstdLocal = rstdQueue.AllocTensor<float>();
    CopyInOtherData();
    if constexpr (OP_CODE == QUANT_OP_CODE) {
        if (hasSmooth) {
            smoothLocal = smoothQueue.DeQue<T>();
        }
        quantScaleLocal = quantScaleQueue.AllocTensor<float>();
    }
    FastProcess();
    scaleQueue.FreeTensor(scaleLocal);
    shiftQueue.FreeTensor(shiftLocal);
    if constexpr (OP_CODE == QUANT_OP_CODE) {
        if (hasSmooth) {
            smoothQueue.FreeTensor(smoothLocal);
        }
    }
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
            if (lastBatchStart != -1) {
                scaleQueue.FreeTensor(scaleLocal);
                shiftQueue.FreeTensor(shiftLocal);
            }
            CopyInScaleShift(range.batchStart * hiddenDim, range.batchEnd - range.batchStart, hiddenDim);
            scaleLocal = scaleQueue.DeQue<T>();
            shiftLocal = shiftQueue.DeQue<T>();
            lastBatchStart = range.batchStart;
            lastBatchEnd = range.batchEnd;
        }
        CopyInX(range.rowStart * hiddenDim, range.actualRowNum, hiddenDim);

        // 计算 & 拷出
        ProcessLayerNorm(range, batchCount);
        Adaption(range, batchCount);
        if constexpr (OP_CODE == QUANT_OP_CODE) {
            QuantCopyOut(range.rowStart * hiddenDim, range.actualRowNum, hiddenDim);
        } else {
            BaseCopyOut(range.rowStart * hiddenDim, range.actualRowNum, hiddenDim);
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
__aicore__ inline void AdaLayerNormFullLoad<T, U, Y, OP_CODE>::Adaption(RowRange range, int64_t batchCount)
{
    LocalTensor<OUT_DTYPE> outLocal = outQueue.AllocTensor<OUT_DTYPE>();
    __local_mem__ float* normAddr = (__ubuf__ float*)normLocal.GetPhyAddr();
    __local_mem__ OUT_DTYPE* outAddr = (__ubuf__ OUT_DTYPE*)outLocal.GetPhyAddr();
    __local_mem__ T* scaleAddr = (__ubuf__ T*)scaleLocal.GetPhyAddr();
    __local_mem__ T* shiftAddr = (__ubuf__ T*)shiftLocal.GetPhyAddr();

    LocalTensor<Y> quantOutLocal;
    __local_mem__ Y* quantOutAddr;
    __local_mem__ T* smoothAddr;
    __local_mem__ float* quantScaleAddr;
    if constexpr (OP_CODE == QUANT_OP_CODE) {
        quantOutLocal = quantOutQueue.AllocTensor<Y>();
        quantOutAddr = (__ubuf__ Y*)quantOutLocal.GetPhyAddr();
        smoothAddr = (__ubuf__ T*)smoothLocal.GetPhyAddr();
        quantScaleAddr = (__ubuf__ float*)quantScaleLocal[batchCount].GetPhyAddr();
    }

    uint16_t rowLoopTimes = static_cast<uint16_t>(range.actualRowNum);
    uint16_t tailLength = static_cast<uint16_t>(hiddenDim % TWO_V_LENGTH);
    uint16_t colLoopTimes = static_cast<uint16_t>(hiddenDim / TWO_V_LENGTH) + (tailLength > V_LENGTH ? 1 : 0);
    uint32_t rightLength = static_cast<uint32_t>(hiddenDim - colLoopTimes * V_LENGTH);
    __VEC_SCOPE__
    {
        RegTensor<float> x1;
        RegTensor<float> x2;
        RegTensor<float> scale1;
        RegTensor<float> scale2;
        RegTensor<float> shift1;
        RegTensor<float> shift2;
        RegTensor<float> tmpMax;
        RegTensor<float> quantScale;
        RegTensor<float> quantScaleBroad;
        RegTensor<int16_t> y1Int16;
        RegTensor<int16_t> y2Int16;
        RegTensor<half> y1Fp16;
        RegTensor<half> y2Fp16;
        RegTensor<Y> y1;
        RegTensor<Y> y2;

        MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
        MaskReg pregMerge = CreateMask<float, MaskPattern::VL1>();
        MaskReg pregLoop;
        __local_mem__ OUT_DTYPE* outAddr1 = outAddr;
        __local_mem__ OUT_DTYPE* outAddr2 = outAddr;
        for (uint16_t i = 0; i < rowLoopTimes; i++) {
            uint32_t batchIdx = (static_cast<uint32_t>(range.rowStart) + i) / static_cast<uint32_t>(seqLen);
            uint32_t scaleOffset = (batchIdx - static_cast<uint32_t>(range.batchStart)) * static_cast<uint32_t>(hiddenDimCeil);
            uint32_t sreg1 = rightLength;
            if constexpr (OP_CODE == QUANT_OP_CODE) {
                Duplicate(tmpMax, 0.0f, pregFull);
            }
            for (uint16_t j = 0; j < colLoopTimes;j ++) {
                pregLoop = UpdateMask<float>(sreg1);
                DataCopy(x1, normAddr + j * TWO_V_LENGTH);
                DataCopy(x2, normAddr + j * TWO_V_LENGTH + V_LENGTH);
                LoadTensor(scale1, scaleAddr + scaleOffset + j * TWO_V_LENGTH, pregFull);
                LoadTensor(scale2, scaleAddr + scaleOffset + j * TWO_V_LENGTH + V_LENGTH, pregLoop);
                LoadTensor(shift1, shiftAddr + scaleOffset + j * TWO_V_LENGTH, pregFull);
                LoadTensor(shift2, shiftAddr + scaleOffset + j * TWO_V_LENGTH + V_LENGTH, pregLoop);
                Adds(scale1, scale1, 1.0f, pregFull);
                Adds(scale2, scale2, 1.0f, pregLoop);
                FusedMulDstAdd(x1, scale1, shift1, pregFull);
                FusedMulDstAdd(x2, scale2, shift2, pregLoop);
                if constexpr (OP_CODE == BASE_V2_OP_CODE) {
                    CopyToTensor(outAddr1 + j * TWO_V_LENGTH, x1, pregFull);
                    CopyToTensor(outAddr1 + j * TWO_V_LENGTH + V_LENGTH, x2, pregLoop);
                } else {
                    if (hasSmooth) {
                        RegTensor<float> smooth1;
                        RegTensor<float> smooth2;
                        LoadTensor(smooth1, smoothAddr + j * TWO_V_LENGTH, pregFull);
                        LoadTensor(smooth2, smoothAddr + j * TWO_V_LENGTH + V_LENGTH, pregLoop);
                        Mul(x1, x1, smooth1, pregFull);
                        Mul(x2, x2, smooth2, pregLoop);
                    }
                    CopyToTensor(outAddr1 + j * TWO_V_LENGTH, x1, pregFull);
                    CopyToTensor(outAddr1 + j * TWO_V_LENGTH + V_LENGTH, x2, pregLoop);
                    Abs(x1, x1, pregFull);
                    Abs(x2, x2, pregLoop);
                    Max(x1, x1, x2, pregFull);
                    Max(tmpMax, tmpMax, x1, pregFull);
                }
            }
            if (tailLength > 0 && tailLength <= V_LENGTH) {
                pregLoop = UpdateMask<float>(sreg1);
                DataCopy(x1, normAddr + colLoopTimes * TWO_V_LENGTH);
                LoadTensor(scale1, scaleAddr + scaleOffset + colLoopTimes * TWO_V_LENGTH, pregLoop);
                LoadTensor(shift1, shiftAddr + scaleOffset + colLoopTimes * TWO_V_LENGTH, pregLoop);
                Adds(scale1, scale1, 1.0f, pregLoop);
                FusedMulDstAdd(x1, scale1, shift1, pregLoop);
                if constexpr (OP_CODE == QUANT_OP_CODE) {
                    if (hasSmooth) {
                        RegTensor<float> smooth1;
                        LoadTensor(smooth1, smoothAddr + colLoopTimes * TWO_V_LENGTH, pregLoop);
                        Mul(x1, x1, smooth1, pregLoop);
                    }
                    CopyToTensor(outAddr1 + colLoopTimes * TWO_V_LENGTH, x1, pregLoop);
                    Abs(x1, x1, pregLoop);
                    Max(tmpMax, tmpMax, x1, pregFull);
                } else {
                    CopyToTensor(outAddr1 + colLoopTimes * TWO_V_LENGTH, x1, pregLoop);
                }
            }
            if constexpr (OP_CODE == QUANT_OP_CODE) {
                ReduceMax(quantScale, tmpMax, pregFull);
                Muls(quantScale, quantScale, quantFactor, pregMerge);
                Duplicate(quantScaleBroad, quantScale, pregFull);
                LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
                uint32_t sreg2 = rightLength;
                for (uint16_t j = 0; j < colLoopTimes;j ++) {
                    pregLoop = UpdateMask<float>(sreg2);
                    DataCopy(x1, outAddr2 + j * TWO_V_LENGTH);
                    DataCopy(x2, outAddr2 + j * TWO_V_LENGTH + V_LENGTH);
                    Div(x1, x1, quantScaleBroad, pregFull);
                    Div(x2, x2, quantScaleBroad, pregLoop);
                    if constexpr (std::is_same_v<Y, hifloat8_t>) {
                        Cast<Y, float, castTraitF32Toh8>(y1, x1, pregFull);
                        Cast<Y, float, castTraitF32Toh8>(y2, x2, pregLoop);
                    } else if constexpr (std::is_same_v<Y, int8_t>) {
                        Cast<int16_t, float, castTraitF32ToI16>(y1Int16, x1, pregFull);
                        Cast<int16_t, float, castTraitF32ToI16>(y2Int16, x2, pregLoop);
                        Cast<half, int16_t, castTraitI16ToF16>(y1Fp16, y1Int16, pregFull);
                        Cast<half, int16_t, castTraitI16ToF16>(y2Fp16, y2Int16, pregLoop);
                        Cast<Y, half, castTraitF16ToI8>(y1, y1Fp16, pregFull);
                        Cast<Y, half, castTraitF16ToI8>(y2, y2Fp16, pregLoop);
                    } else {
                        Cast<Y, float, castTraitF32Tofp8>(y1, x1, pregFull);
                        Cast<Y, float, castTraitF32Tofp8>(y2, x2, pregLoop);
                    }
                    DataCopy<Y, StoreDist::DIST_PACK4_B32>(quantOutAddr + j * TWO_V_LENGTH, y1, pregFull);
                    DataCopy<Y, StoreDist::DIST_PACK4_B32>(quantOutAddr + j * TWO_V_LENGTH + V_LENGTH, y2, pregLoop);
                }
                if (tailLength > 0 && tailLength <= V_LENGTH) {
                    pregLoop = UpdateMask<float>(sreg2);
                    DataCopy(x1, outAddr2 + colLoopTimes * TWO_V_LENGTH);
                    Div(x1, x1, quantScaleBroad, pregLoop);
                    if constexpr (std::is_same_v<Y, hifloat8_t>) {
                        Cast<Y, float, castTraitF32Toh8>(y1, x1, pregLoop);
                    } else if constexpr (std::is_same_v<Y, int8_t>) {
                        Cast<int16_t, float, castTraitF32ToI16>(y1Int16, x1, pregLoop);
                        Cast<half, int16_t, castTraitI16ToF16>(y1Fp16, y1Int16, pregLoop);
                        Cast<Y, half, castTraitF16ToI8>(y1, y1Fp16, pregLoop);
                    } else {
                        Cast<Y, float, castTraitF32Tofp8>(y1, x1, pregLoop);
                    }
                    DataCopy<Y, StoreDist::DIST_PACK4_B32>(quantOutAddr + colLoopTimes * TWO_V_LENGTH, y1, pregLoop);
                }
                // 拷出量化系数
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(quantScaleAddr + i, quantScale, pregMerge);

                outAddr2 += static_cast<uint32_t>(hiddenDimCeil);
                quantOutAddr += static_cast<uint32_t>(outIntCeil);
            }
            normAddr += static_cast<uint32_t>(hiddenDimCeil);
            outAddr1 += static_cast<uint32_t>(hiddenDimCeil);
        }
    }
    if constexpr (OP_CODE == QUANT_OP_CODE) {
        quantOutQueue.EnQue<Y>(quantOutLocal);
        outQueue.FreeTensor(outLocal);
    } else {
        outQueue.EnQue<OUT_DTYPE>(outLocal);
    }
}
}  // namespace AdaLayerNormNS

#endif  // ADA_LAYER_NORM_FULL_LOAD_H
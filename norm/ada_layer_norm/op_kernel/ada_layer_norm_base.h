/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file ada_layer_norm_base.h
 * \brief
 */
#ifndef ADA_LAYER_NORM_BASE_H
#define ADA_LAYER_NORM_BASE_H

#include "kernel_operator.h"

namespace AdaLayerNormNS {
using namespace AscendC;

constexpr uint8_t BASE_OP_CODE = 1;
constexpr uint8_t BASE_V2_OP_CODE = 12;
constexpr uint8_t QUANT_OP_CODE = 2;
constexpr int32_t MAX_X_SIZE = 16384;
constexpr int32_t TENSOR_NUM = 7;
constexpr int32_t DATA_COUNT = 6144;
constexpr int32_t BATCH_COUNT = 1024;
constexpr int32_t INT8_BLOCK_NUM = 32;
constexpr int32_t HALF_BLOCK_NUM = 16;
constexpr int32_t FLOAT_BLOCK_NUM = 8;
constexpr float MAX_INT8 = 127.0f;
constexpr float ONE_FLOAT = 1.0f;
constexpr float FACTOR_INT8 = 1.0f / 127.0f;

struct RowRange {
    int64_t rowStart;
    int64_t rowEnd;
    int64_t actualRowNum;
    int64_t batchStart;
    int64_t batchEnd;
    int64_t dataCount;
};

struct GmAddr {
    const GM_ADDR x = nullptr;
    const GM_ADDR scale = nullptr;
    const GM_ADDR shift = nullptr;
    const GM_ADDR weight = nullptr;
    const GM_ADDR bias = nullptr;
    const GM_ADDR smooth_scales = nullptr;
    const GM_ADDR out = nullptr;
    const GM_ADDR mean = nullptr;
    const GM_ADDR rstd = nullptr;
    const GM_ADDR quant_scale = nullptr;
};

template <typename X_DTYPE, typename WEIGHT_DTYPE, uint8_t OP_CODE>
class AdaLayerNormND {
public:
    __aicore__ inline AdaLayerNormND(){};
    __aicore__ inline void Init(const GmAddr* gmAddr, const AdaLayerNormTilingData *tilingData);
    __aicore__ inline void InitV2(const GmAddr* gmAddr, const AdaLayerNormTilingData *tilingData);
    __aicore__ inline void InitQuant(const GmAddr* gmAddr, GM_ADDR workspace, const AdaLayerNormTilingData *tilingData);
    __aicore__ inline void Process();

private:
    template <typename T1, typename T2>
    __aicore__ inline T1 CeilA2B(T1 a, T2 b)
    {
        return (b != 0) ? (a + b - 1) / b : a;
    };

    template <typename T1, typename T2>
    __aicore__ inline T1 Min(T1 a, T2 b)
    {
        return (a < b) ? a : b;
    };

    __aicore__ inline void InitTensor();
    __aicore__ inline void InitEventId();
    __aicore__ inline void ReleaseEventId();
    __aicore__ inline void ParseTilingData(const AdaLayerNormTilingData *tilingData);
    __aicore__ inline void SliceProcess();
    __aicore__ inline void FastProcess();

    __aicore__ inline void ComputeMean(int64_t dataCount, float &meanValue);
    __aicore__ inline void ComputeVar(int64_t dataCount, float meanValue, float &varValue);
    __aicore__ inline void BaseSliceCompute(int64_t rowIdx, int64_t batchIdx, float meanValue, float rstdValue);
    __aicore__ inline void QuantSliceCompute(int64_t rowIdx, int64_t batchIdx, float meanValue, float rstdValue);
    __aicore__ inline void ComputeSliceNorm(float meanValue, float rstdValue, int64_t dataCount);
    __aicore__ inline void ComputeSliceQuant(float maxValue, int64_t dataCount);
    __aicore__ inline void MultiCalMeanRstd(RowRange range, int64_t batchIdx);
    __aicore__ inline void MultiLayerNorm(RowRange range, int64_t batchIdx);
    __aicore__ inline void SingleLayerNorm(int64_t batchIdx);
    __aicore__ inline void Adaption(RowRange range);
    __aicore__ inline void DynamicQuant(RowRange range, int64_t batchIdx);

    __aicore__ inline void CopyInData(LocalTensor<float> inputFloat, GlobalTensor<X_DTYPE> inputGm, int64_t len);
    __aicore__ inline void CopyInWeightBias(LocalTensor<float> inputFloat, GlobalTensor<WEIGHT_DTYPE> inputGm, int64_t len);
    __aicore__ inline void CopyInOtherData(int64_t offset, int64_t len);
    __aicore__ inline void CopyInScaleShift(int64_t offset, uint16_t blockCount, int64_t len);
    __aicore__ inline void CopyInSlice(int64_t offset, int64_t scaleOffset, int64_t h, int64_t len);
    __aicore__ inline void CopyInSliceX(int64_t offset, int64_t len);
    __aicore__ inline void CopyInX(int64_t offset, uint16_t blockCount, int64_t len);
    __aicore__ inline void BaseCopyOut(int64_t offset, uint16_t blockCount, int64_t len);
    __aicore__ inline void QuantCopyOut(int64_t offset, uint16_t blockCount, int64_t len);
    __aicore__ inline void CopyScaleOut(int64_t offset, int64_t len);
    __aicore__ inline void CopyMeanRstdOut(int64_t offset, int64_t len);
    __aicore__ inline void CopyNormOut(int64_t h, int64_t len);
    __aicore__ inline void CopyInNorm(int64_t h, int64_t len);

private:
    TPipe pipe;
    RowRange range;
    TBuf<TPosition::VECCALC> tmpBuf;        // 7 * 4 * 6K = 168K (+ 6K)
    TBuf<TPosition::VECCALC> reduceBuf;     // 4 * 1024
    TBuf<TPosition::VECCALC> meanRstdBuf;   // 4 * 1024 + 4 * 1024
    TBuf<TPosition::VECCALC> quantScaleBuf; // 4 * 1024

    GlobalTensor<X_DTYPE> xGm;
    GlobalTensor<X_DTYPE> shiftGm;
    GlobalTensor<X_DTYPE> scaleGm;
    GlobalTensor<WEIGHT_DTYPE> weightGm;
    GlobalTensor<WEIGHT_DTYPE> biasGm;
    GlobalTensor<X_DTYPE> smoothGm;
    GlobalTensor<X_DTYPE> outGm;
    GlobalTensor<X_DTYPE> meanGm;
    GlobalTensor<X_DTYPE> rstdGm;
    GlobalTensor<int8_t> quantOutGm;
    GlobalTensor<float> quantScaleGm;
    GlobalTensor<float> normGm;

    LocalTensor<float> xFloat;
    LocalTensor<float> weightFloat;
    LocalTensor<float> biasFloat;
    LocalTensor<float> scaleFloat;
    LocalTensor<float> shiftFloat;
    LocalTensor<float> meanFloat;
    LocalTensor<float> yFloat;
    LocalTensor<float> tmpFloat;
    LocalTensor<float> smoothFloat;
    LocalTensor<float> meanOutFloat;
    LocalTensor<float> rstdOutFloat;
    LocalTensor<half> yHalf;
    LocalTensor<int8_t> yInt;
    LocalTensor<float> quantScaleFloat;
    LocalTensor<float> reduceFloat;

    event_t eventIdVToMte2;
    event_t eventIdMte2ToV;
    event_t eventIdVToMte3;
    event_t eventIdMte3ToV;
    event_t eventIdVToS;
    event_t eventIdMte3ToS;
    event_t eventIdMte3ToMte2;

    int64_t sliceSize = 0;
    int64_t rowNum = 0;
    int64_t batchSize = 0;
    int64_t seqLen = 0;
    int64_t hiddenDim = 0;
    int64_t hiddenDimCeil = 0;
    int64_t outIntCeil = 0;
    int64_t rowStart = 0;
    int64_t rowEnd = 0;
    int64_t normOffset = 0;
    float epsilon = 0.0f;
    float aveFactor = 0.0f;
    int32_t hasWeight = 0;
    int32_t hasBias = 0;
    int32_t hasSmooth = 0;
};

template <typename X_DTYPE, typename WEIGHT_DTYPE, uint8_t OP_CODE>
__aicore__ inline void AdaLayerNormND<X_DTYPE, WEIGHT_DTYPE, OP_CODE>::Init(const GmAddr* gmAddr, const AdaLayerNormTilingData *tilingData)
{
    ParseTilingData(tilingData);
    pipe.InitBuffer(tmpBuf, TENSOR_NUM * DATA_COUNT * sizeof(float));
    pipe.InitBuffer(reduceBuf, BATCH_COUNT * sizeof(float));

    xGm.SetGlobalBuffer((__gm__ X_DTYPE*)gmAddr->x);
    scaleGm.SetGlobalBuffer((__gm__ X_DTYPE*)gmAddr->scale);
    shiftGm.SetGlobalBuffer((__gm__ X_DTYPE*)gmAddr->shift);
    weightGm.SetGlobalBuffer((__gm__ WEIGHT_DTYPE*)gmAddr->weight);
    biasGm.SetGlobalBuffer((__gm__ WEIGHT_DTYPE*)gmAddr->bias);
    outGm.SetGlobalBuffer((__gm__ X_DTYPE*)gmAddr->out);
}

template <typename X_DTYPE, typename WEIGHT_DTYPE, uint8_t OP_CODE>
__aicore__ inline void AdaLayerNormND<X_DTYPE, WEIGHT_DTYPE, OP_CODE>::InitV2(const GmAddr* gmAddr, const AdaLayerNormTilingData *tilingData)
{
    ParseTilingData(tilingData);
    pipe.InitBuffer(tmpBuf, TENSOR_NUM * DATA_COUNT * sizeof(float));
    pipe.InitBuffer(reduceBuf, BATCH_COUNT * sizeof(float));
    pipe.InitBuffer(meanRstdBuf, (BATCH_COUNT + BATCH_COUNT) * sizeof(float));

    xGm.SetGlobalBuffer((__gm__ X_DTYPE*)gmAddr->x);
    scaleGm.SetGlobalBuffer((__gm__ X_DTYPE*)gmAddr->scale);
    shiftGm.SetGlobalBuffer((__gm__ X_DTYPE*)gmAddr->shift);
    weightGm.SetGlobalBuffer((__gm__ WEIGHT_DTYPE*)gmAddr->weight);
    biasGm.SetGlobalBuffer((__gm__ WEIGHT_DTYPE*)gmAddr->bias);
    outGm.SetGlobalBuffer((__gm__ X_DTYPE*)gmAddr->out);
    meanGm.SetGlobalBuffer((__gm__ X_DTYPE*)gmAddr->mean);
    rstdGm.SetGlobalBuffer((__gm__ X_DTYPE*)gmAddr->rstd);
}

template <typename X_DTYPE, typename WEIGHT_DTYPE, uint8_t OP_CODE>
__aicore__ inline void AdaLayerNormND<X_DTYPE, WEIGHT_DTYPE, OP_CODE>::InitQuant(const GmAddr* gmAddr, GM_ADDR workspace,
    const AdaLayerNormTilingData *tilingData)
{
    ParseTilingData(tilingData);
    pipe.InitBuffer(tmpBuf, TENSOR_NUM * DATA_COUNT * sizeof(float) + DATA_COUNT * sizeof(int8_t));
    pipe.InitBuffer(reduceBuf, BATCH_COUNT * sizeof(float));
    pipe.InitBuffer(quantScaleBuf, BATCH_COUNT * sizeof(float));

    xGm.SetGlobalBuffer((__gm__ X_DTYPE*)gmAddr->x);
    scaleGm.SetGlobalBuffer((__gm__ X_DTYPE*)gmAddr->scale);
    shiftGm.SetGlobalBuffer((__gm__ X_DTYPE*)gmAddr->shift);
    weightGm.SetGlobalBuffer((__gm__ WEIGHT_DTYPE*)gmAddr->weight);
    biasGm.SetGlobalBuffer((__gm__ WEIGHT_DTYPE*)gmAddr->bias);
    smoothGm.SetGlobalBuffer((__gm__ X_DTYPE*)gmAddr->smooth_scales);
    normGm.SetGlobalBuffer((__gm__ float*)workspace);
    quantOutGm.SetGlobalBuffer((__gm__ int8_t*)gmAddr->out);
    quantScaleGm.SetGlobalBuffer((__gm__ float*)gmAddr->quant_scale);
}

template <typename X_DTYPE, typename WEIGHT_DTYPE, uint8_t OP_CODE>
__aicore__ inline void AdaLayerNormND<X_DTYPE, WEIGHT_DTYPE, OP_CODE>::Process()
{
    if (rowStart >= rowEnd) {
        return;
    }
    InitTensor();
    InitEventId();

    if (sliceSize < hiddenDim) {
        SliceProcess();
    } else {
        CopyInOtherData(0, hiddenDim);
        FastProcess();
    }
    ReleaseEventId();
}

template <typename X_DTYPE, typename WEIGHT_DTYPE, uint8_t OP_CODE>
__aicore__ inline void AdaLayerNormND<X_DTYPE, WEIGHT_DTYPE, OP_CODE>::ParseTilingData(const AdaLayerNormTilingData *tilingData)
{
    int32_t blockNum = HALF_BLOCK_NUM;
    if constexpr (std::is_same_v<X_DTYPE, float>) {
        blockNum = FLOAT_BLOCK_NUM;
    }
    batchSize = tilingData->batchSize;
    seqLen = tilingData->seqLen;
    hiddenDim = tilingData->hiddenDim;
    hiddenDimCeil = CeilA2B(hiddenDim, blockNum) * blockNum;
    outIntCeil = CeilA2B(hiddenDim, INT8_BLOCK_NUM) * INT8_BLOCK_NUM;
    epsilon = tilingData->epsilon;
    aveFactor = tilingData->aveFactor;
    hasWeight = tilingData->hasWeight;
    hasBias = tilingData->hasBias;
    hasSmooth = tilingData->hasSmooth;
    sliceSize = tilingData->sliceSize;
    rowNum = tilingData->rowNum;

    int64_t singleCoreNum = tilingData->singleCoreNum;
    int64_t tailNum = tilingData->tailNum;
    int64_t blockIdx = GetBlockIdx();
    rowStart = singleCoreNum * blockIdx + (blockIdx < tailNum ? blockIdx : tailNum);
    rowEnd = rowStart + singleCoreNum + (blockIdx < tailNum ? 1 : 0);
    normOffset = blockIdx * outIntCeil;
}

template <typename X_DTYPE, typename WEIGHT_DTYPE, uint8_t OP_CODE>
__aicore__ inline void AdaLayerNormND<X_DTYPE, WEIGHT_DTYPE, OP_CODE>::InitTensor()
{
    xFloat = tmpBuf.Get<float>();
    weightFloat = xFloat[DATA_COUNT];
    biasFloat = weightFloat[DATA_COUNT];
    scaleFloat = biasFloat[DATA_COUNT];
    shiftFloat = scaleFloat[DATA_COUNT];
    meanFloat = shiftFloat[DATA_COUNT];
    tmpFloat = xFloat[MAX_X_SIZE];
    reduceFloat = reduceBuf.Get<float>();
    if constexpr (OP_CODE == QUANT_OP_CODE) {
        yFloat = meanFloat;
        smoothFloat = yFloat[DATA_COUNT];
        yHalf = yFloat.ReinterpretCast<half>();
        yInt = smoothFloat[DATA_COUNT].ReinterpretCast<int8_t>();
        quantScaleFloat = quantScaleBuf.Get<float>();
    } else {
        yFloat = meanFloat[DATA_COUNT];
    }
    if constexpr (OP_CODE == BASE_V2_OP_CODE) {
        meanOutFloat = meanRstdBuf.Get<float>();
        rstdOutFloat = meanOutFloat[BATCH_COUNT];
    }
}

template <typename X_DTYPE, typename WEIGHT_DTYPE, uint8_t OP_CODE>
__aicore__ inline void AdaLayerNormND<X_DTYPE, WEIGHT_DTYPE, OP_CODE>::InitEventId()
{
    eventIdVToMte2 = static_cast<event_t>(pipe.AllocEventID<HardEvent::V_MTE2>());
    eventIdMte2ToV = static_cast<event_t>(pipe.AllocEventID<HardEvent::MTE2_V>());
    eventIdVToMte3 = static_cast<event_t>(pipe.AllocEventID<HardEvent::V_MTE3>());
    eventIdMte3ToV = static_cast<event_t>(pipe.AllocEventID<HardEvent::MTE3_V>());
    eventIdVToS = static_cast<event_t>(pipe.AllocEventID<HardEvent::V_S>());
    eventIdMte3ToS = static_cast<event_t>(pipe.AllocEventID<HardEvent::MTE3_S>());
    eventIdMte3ToMte2 = static_cast<event_t>(pipe.AllocEventID<HardEvent::MTE3_MTE2>());
}

template <typename X_DTYPE, typename WEIGHT_DTYPE, uint8_t OP_CODE>
__aicore__ inline void AdaLayerNormND<X_DTYPE, WEIGHT_DTYPE, OP_CODE>::SliceProcess()
{
    int64_t batchCount = 0;
    int64_t rowIdx = rowStart;
    while (rowIdx < rowEnd) {
        int64_t offset = rowIdx * hiddenDim;
        int64_t scaleOffset = (rowIdx / seqLen) * hiddenDim;
        // 计算均值和方差
        float meanValue = 0.0f;
        float varValue = 0.0f;
        if (hiddenDim > MAX_X_SIZE) {
            for (int64_t h = 0; h < hiddenDim; h += MAX_X_SIZE) {
                int64_t dataCount = Min(MAX_X_SIZE, hiddenDim - h);
                CopyInSliceX(offset + h, dataCount);
                ComputeMean(dataCount, meanValue);
            }
            for (int64_t h = 0; h < hiddenDim; h += MAX_X_SIZE) {
                int64_t dataCount = Min(MAX_X_SIZE, hiddenDim - h);
                CopyInSliceX(offset + h, dataCount);
                ComputeVar(dataCount, meanValue, varValue);
            }
        } else {
            CopyInSliceX(offset, hiddenDim);
            ComputeMean(hiddenDim, meanValue);
            ComputeVar(hiddenDim, meanValue, varValue);
        }
        float rstdValue = 1.0f / sqrt(varValue + epsilon);

        if constexpr (OP_CODE == QUANT_OP_CODE) {
            QuantSliceCompute(rowIdx, batchCount, meanValue, rstdValue);
        } else {
            BaseSliceCompute(rowIdx, batchCount, meanValue, rstdValue);
        }
        batchCount++;
        rowIdx++;
        if (batchCount == BATCH_COUNT || rowIdx == rowEnd) {
            if constexpr (OP_CODE == QUANT_OP_CODE) {
                CopyScaleOut(rowIdx - batchCount, batchCount);
            }
            if constexpr (OP_CODE == BASE_V2_OP_CODE) {
                CopyMeanRstdOut(rowIdx - batchCount, batchCount);
            }
            batchCount = 0;
        }
    }
}

template <typename X_DTYPE, typename WEIGHT_DTYPE, uint8_t OP_CODE>
__aicore__ inline void AdaLayerNormND<X_DTYPE, WEIGHT_DTYPE, OP_CODE>::FastProcess()
{
    int64_t lastBatchStart = -1;
    int64_t lastBatchEnd = -1;
    int64_t rowIdx = rowStart;
    int64_t batchCount = 0;
    while (rowIdx < rowEnd) {
        range.rowStart = rowIdx;
        range.rowEnd = Min(rowEnd, range.rowStart + rowNum);
        range.actualRowNum = range.rowEnd - range.rowStart;
        range.dataCount = range.actualRowNum > 1 ? range.actualRowNum * hiddenDimCeil : hiddenDim;
        range.batchStart = range.rowStart / seqLen;
        range.batchEnd = (range.rowEnd - 1) / seqLen + 1;

        // 拷入
        bool scaleShiftReuse = (lastBatchStart == range.batchStart && lastBatchEnd == range.batchEnd);
        if (!scaleShiftReuse) {
            SetFlag<HardEvent::V_MTE2>(eventIdVToMte2);
            WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2);
            CopyInScaleShift(range.batchStart * hiddenDim, range.batchEnd - range.batchStart, hiddenDim);
        }
        CopyInX(range.rowStart * hiddenDim, range.actualRowNum, hiddenDim);

        // 计算 & 拷出
        if (range.actualRowNum > 1) {
            MultiCalMeanRstd(range, batchCount);
            MultiLayerNorm(range, batchCount);
        } else {
            SingleLayerNorm(batchCount);
        }
        Adaption(range);
        if constexpr (OP_CODE == QUANT_OP_CODE) {
            DynamicQuant(range, batchCount);
            QuantCopyOut(range.rowStart * hiddenDim, range.actualRowNum, hiddenDim);
        } else {
            BaseCopyOut(range.rowStart * hiddenDim, range.actualRowNum, hiddenDim);
        }

        batchCount += range.actualRowNum;
        rowIdx = range.rowEnd;
        if (batchCount + rowNum > BATCH_COUNT || rowIdx == rowEnd) {
            if constexpr (OP_CODE == QUANT_OP_CODE) {
                CopyScaleOut(rowIdx - batchCount, batchCount);
            }
            if constexpr (OP_CODE == BASE_V2_OP_CODE) {
                CopyMeanRstdOut(rowIdx - batchCount, batchCount);
            }
            batchCount = 0;
        }
        lastBatchStart = range.batchStart;
        lastBatchEnd = range.batchEnd;
    }
}

template <typename X_DTYPE, typename WEIGHT_DTYPE, uint8_t OP_CODE>
__aicore__ inline void AdaLayerNormND<X_DTYPE, WEIGHT_DTYPE, OP_CODE>::ReleaseEventId()
{
    pipe.ReleaseEventID<HardEvent::V_MTE2>(eventIdVToMte2);
    pipe.ReleaseEventID<HardEvent::MTE2_V>(eventIdMte2ToV);
    pipe.ReleaseEventID<HardEvent::V_MTE3>(eventIdVToMte3);
    pipe.ReleaseEventID<HardEvent::MTE3_V>(eventIdMte3ToV);
    pipe.ReleaseEventID<HardEvent::V_S>(eventIdVToS);
    pipe.ReleaseEventID<HardEvent::MTE3_S>(eventIdMte3ToS);
    pipe.ReleaseEventID<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
}
}  // namespace AdaLayerNormNS

#endif  // ADA_LAYER_NORM_BASE_H
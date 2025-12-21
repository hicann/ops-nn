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
 * \file ada_layer_norm_base_v1.h
 * \brief
 */
#include "ada_layer_norm_base.h"

using namespace AdaLayerNormNS;

template <typename X_DTYPE, typename WEIGHT_DTYPE, uint8_t OP_CODE>
__aicore__ inline void AdaLayerNormND<X_DTYPE, WEIGHT_DTYPE, OP_CODE>::ComputeMean(int64_t dataCount, float &meanValue)
{
    Muls(tmpFloat, xFloat, aveFactor, dataCount);
    PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::V_MTE2>(eventIdVToMte2);
    WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2);
    ReduceSum(tmpFloat, tmpFloat, tmpFloat, dataCount);
    SetFlag<HardEvent::V_S>(eventIdVToS);
    WaitFlag<HardEvent::V_S>(eventIdVToS);
    meanValue += tmpFloat.GetValue(0);
}

template <typename X_DTYPE, typename WEIGHT_DTYPE, uint8_t OP_CODE>
__aicore__ inline void AdaLayerNormND<X_DTYPE, WEIGHT_DTYPE, OP_CODE>::ComputeVar(int64_t dataCount, float meanValue, float &varValue)
{
    Adds(tmpFloat, xFloat, -meanValue, dataCount);
    PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::V_MTE2>(eventIdVToMte2);
    WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2);
    Mul(tmpFloat, tmpFloat, tmpFloat, dataCount);
    PipeBarrier<PIPE_V>();
    Muls(tmpFloat, tmpFloat, aveFactor, dataCount);
    PipeBarrier<PIPE_V>();
    ReduceSum(tmpFloat, tmpFloat, tmpFloat, dataCount);
    SetFlag<HardEvent::V_S>(eventIdVToS);
    WaitFlag<HardEvent::V_S>(eventIdVToS);
    varValue += tmpFloat.GetValue(0);
}

template <typename X_DTYPE, typename WEIGHT_DTYPE, uint8_t OP_CODE>
__aicore__ inline void AdaLayerNormND<X_DTYPE, WEIGHT_DTYPE, OP_CODE>::BaseSliceCompute(int64_t rowIdx, int64_t batchIdx, float meanValue, float rstdValue)
{
    if constexpr (OP_CODE == BASE_V2_OP_CODE) {
        meanOutFloat.SetValue(batchIdx, meanValue);
        rstdOutFloat.SetValue(batchIdx, rstdValue);
    }
    int64_t offset = rowIdx * hiddenDim;
    int64_t scaleOffset = (rowIdx / seqLen) * hiddenDim;
    for (int64_t h = 0; h < hiddenDim; h += sliceSize) {
        int64_t dataCount = Min(sliceSize, hiddenDim - h);
        CopyInSlice(offset, scaleOffset + h, h, dataCount);
        ComputeSliceNorm(meanValue, rstdValue, dataCount);
        SetFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
        WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
        Add(yFloat, meanFloat, shiftFloat, dataCount);
        SetFlag<HardEvent::V_MTE2>(eventIdVToMte2);
        WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2);
        BaseCopyOut(offset + h, 1, dataCount);
    }
}

template <typename X_DTYPE, typename WEIGHT_DTYPE, uint8_t OP_CODE>
__aicore__ inline void AdaLayerNormND<X_DTYPE, WEIGHT_DTYPE, OP_CODE>::QuantSliceCompute(
    int64_t rowIdx, int64_t batchIdx, float meanValue, float rstdValue)
{
    int64_t offset = rowIdx * hiddenDim;
    int64_t scaleOffset = (rowIdx / seqLen) * hiddenDim;
    // 计算量化系数
    float maxValue = 0.0f;
    for (int64_t h = 0; h < hiddenDim; h += sliceSize) {
        int64_t dataCount = Min(sliceSize, hiddenDim - h);
        CopyInSlice(offset, scaleOffset + h, h, dataCount);
        SetFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
        WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
        ComputeSliceNorm(meanValue, rstdValue, dataCount);
        Add(xFloat, meanFloat, shiftFloat, dataCount);
        PipeBarrier<PIPE_V>();
        if (hasSmooth) {
            Mul(xFloat, xFloat, smoothFloat, dataCount);
            PipeBarrier<PIPE_V>();
        }
        Abs(yFloat, xFloat, dataCount);
        PipeBarrier<PIPE_V>();
        CopyNormOut(h, dataCount);
        SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
        WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
        ReduceMax(reduceFloat, yFloat, yFloat, dataCount);
        SetFlag<HardEvent::V_S>(eventIdVToS);
        WaitFlag<HardEvent::V_S>(eventIdVToS);
        float tmpMax = reduceFloat.GetValue(0);
        maxValue = (tmpMax != tmpMax) ? tmpMax : (tmpMax > maxValue ? tmpMax : maxValue);
    }
    quantScaleFloat.SetValue(batchIdx, maxValue / MAX_INT8);
    // 计算量化输出
    for (int64_t h = 0; h < hiddenDim; h += sliceSize) {
        int64_t dataCount = Min(sliceSize, hiddenDim - h);
        CopyInNorm(h, dataCount);
        ComputeSliceQuant(maxValue, dataCount);
        QuantCopyOut(offset + h, 1, dataCount);
    }
}

template <typename X_DTYPE, typename WEIGHT_DTYPE, uint8_t OP_CODE>
__aicore__ inline void AdaLayerNormND<X_DTYPE, WEIGHT_DTYPE, OP_CODE>::ComputeSliceNorm(float meanValue, float rstdValue, int64_t dataCount)
{
    Adds(meanFloat, xFloat, -meanValue, dataCount);
    PipeBarrier<PIPE_V>();
    Muls(meanFloat, meanFloat, rstdValue, dataCount);
    PipeBarrier<PIPE_V>();
    if (hasWeight) {
        Mul(meanFloat, meanFloat, weightFloat, dataCount);
        PipeBarrier<PIPE_V>();
    }
    if (hasBias) {
        Add(meanFloat, meanFloat, biasFloat, dataCount);
        PipeBarrier<PIPE_V>();
    }
    Mul(meanFloat, meanFloat, scaleFloat, dataCount);
    PipeBarrier<PIPE_V>();
}

template <typename X_DTYPE, typename WEIGHT_DTYPE, uint8_t OP_CODE>
__aicore__ inline void AdaLayerNormND<X_DTYPE, WEIGHT_DTYPE, OP_CODE>::ComputeSliceQuant(float maxValue, int64_t dataCount)
{
    Muls(yFloat, xFloat, MAX_INT8 / maxValue, dataCount);
    PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::V_MTE2>(eventIdVToMte2);
    WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2);
    Cast(yHalf, yFloat, RoundMode::CAST_RINT, dataCount);
    PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
    WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
    Cast(yInt, yHalf, RoundMode::CAST_ROUND, dataCount);
}

template <typename X_DTYPE, typename WEIGHT_DTYPE, uint8_t OP_CODE>
__aicore__ inline void AdaLayerNormND<X_DTYPE, WEIGHT_DTYPE, OP_CODE>::MultiCalMeanRstd(RowRange range, int64_t batchIdx)
{
    uint32_t srcShape[2] = {static_cast<uint32_t>(range.actualRowNum), 1};
    uint32_t dstShape[2] = {static_cast<uint32_t>(range.actualRowNum), static_cast<uint32_t>(hiddenDimCeil)};
    Muls(meanFloat, xFloat, aveFactor, range.dataCount);
    PipeBarrier<PIPE_V>();
    for (int64_t rowIdx = 0; rowIdx < range.actualRowNum; rowIdx++) {
        ReduceSum(reduceFloat[rowIdx], meanFloat[rowIdx * hiddenDimCeil], meanFloat[rowIdx * hiddenDimCeil], hiddenDim);
    }
    PipeBarrier<PIPE_V>();
    if constexpr (OP_CODE == BASE_V2_OP_CODE) {
        SetFlag<HardEvent::V_S>(eventIdVToS);
        WaitFlag<HardEvent::V_S>(eventIdVToS);
        for (int64_t rowIdx = 0; rowIdx < range.actualRowNum; rowIdx++) {
            meanOutFloat.SetValue(batchIdx + rowIdx, reduceFloat.GetValue(rowIdx));
        }
    }
    BroadCast<float, 2, 1>(meanFloat, reduceFloat, dstShape, srcShape);
    PipeBarrier<PIPE_V>();
    Sub(meanFloat, xFloat, meanFloat, range.dataCount);
    PipeBarrier<PIPE_V>();
    Mul(xFloat, meanFloat, meanFloat, range.dataCount);
    PipeBarrier<PIPE_V>();
    Muls(xFloat, xFloat, aveFactor, range.dataCount);
    PipeBarrier<PIPE_V>();
    for (int64_t rowIdx = 0; rowIdx < range.actualRowNum; rowIdx++) {
        ReduceSum(reduceFloat[rowIdx], xFloat[rowIdx * hiddenDimCeil], xFloat[rowIdx * hiddenDimCeil], hiddenDim);
    }
    PipeBarrier<PIPE_V>();
    Adds(reduceFloat, reduceFloat, epsilon, range.actualRowNum);
    PipeBarrier<PIPE_V>();
    Sqrt(reduceFloat, reduceFloat, range.actualRowNum);
    PipeBarrier<PIPE_V>();
    if constexpr (OP_CODE == BASE_V2_OP_CODE) {
        SetFlag<HardEvent::V_S>(eventIdVToS);
        WaitFlag<HardEvent::V_S>(eventIdVToS);
        for (int64_t rowIdx = 0; rowIdx < range.actualRowNum; rowIdx++) {
            rstdOutFloat.SetValue(batchIdx + rowIdx, ONE_FLOAT / reduceFloat.GetValue(rowIdx));
        }
    }
    BroadCast<float, 2, 1>(xFloat, reduceFloat, dstShape, srcShape);
    PipeBarrier<PIPE_V>();
}

template <typename X_DTYPE, typename WEIGHT_DTYPE, uint8_t OP_CODE>
__aicore__ inline void AdaLayerNormND<X_DTYPE, WEIGHT_DTYPE, OP_CODE>::MultiLayerNorm(RowRange range, int64_t batchIdx)
{
    SetFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
    WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
    Div(yFloat, meanFloat, xFloat, range.dataCount);
    PipeBarrier<PIPE_V>();
    if constexpr (OP_CODE != QUANT_OP_CODE) {
        SetFlag<HardEvent::V_MTE2>(eventIdVToMte2);
        WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2);
    }
    if (hasWeight) {
        for (int64_t rowIdx = 0; rowIdx < range.actualRowNum; rowIdx++) {
            Mul(yFloat[rowIdx * hiddenDimCeil], yFloat[rowIdx * hiddenDimCeil], weightFloat, hiddenDim);
        }
        PipeBarrier<PIPE_V>();
    }
    if (hasBias) {
        for (int64_t rowIdx = 0; rowIdx < range.actualRowNum; rowIdx++) {
            Add(yFloat[rowIdx * hiddenDimCeil], yFloat[rowIdx * hiddenDimCeil], biasFloat, hiddenDim);
        }
        PipeBarrier<PIPE_V>();
    }
}

template <typename X_DTYPE, typename WEIGHT_DTYPE, uint8_t OP_CODE>
__aicore__ inline void AdaLayerNormND<X_DTYPE, WEIGHT_DTYPE, OP_CODE>::SingleLayerNorm(int64_t batchIdx)
{
    Muls(meanFloat, xFloat, aveFactor, hiddenDim);
    PipeBarrier<PIPE_V>();
    ReduceSum(reduceFloat, meanFloat, meanFloat, hiddenDim);
    SetFlag<HardEvent::V_S>(eventIdVToS);
    WaitFlag<HardEvent::V_S>(eventIdVToS);
    float meanValue = reduceFloat.GetValue(0);
    if constexpr (OP_CODE == BASE_V2_OP_CODE) {
        meanOutFloat.SetValue(batchIdx, meanValue);
    }

    Adds(meanFloat, xFloat, -meanValue, hiddenDim);
    PipeBarrier<PIPE_V>();
    Mul(xFloat, meanFloat, meanFloat, hiddenDim);
    PipeBarrier<PIPE_V>();
    Muls(xFloat, xFloat, aveFactor, hiddenDim);
    PipeBarrier<PIPE_V>();
    ReduceSum(reduceFloat, xFloat, xFloat, hiddenDim);
    if constexpr (OP_CODE != QUANT_OP_CODE) {
        SetFlag<HardEvent::V_MTE2>(eventIdVToMte2);
        WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2);
    }
    SetFlag<HardEvent::V_S>(eventIdVToS);
    WaitFlag<HardEvent::V_S>(eventIdVToS);
    float rstdValue = 1.0f / sqrt(reduceFloat.GetValue(0) + epsilon);
    if constexpr (OP_CODE == BASE_V2_OP_CODE) {
        rstdOutFloat.SetValue(batchIdx, rstdValue);
    }

    SetFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
    WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
    Muls(yFloat, meanFloat, rstdValue, hiddenDim);
    PipeBarrier<PIPE_V>();
    if (hasWeight) {
        Mul(yFloat, yFloat, weightFloat, hiddenDim);
        PipeBarrier<PIPE_V>();
    }
    if (hasBias) {
        Add(yFloat, yFloat, biasFloat, hiddenDim);
        PipeBarrier<PIPE_V>();
    }
}

template <typename X_DTYPE, typename WEIGHT_DTYPE, uint8_t OP_CODE>
__aicore__ inline void AdaLayerNormND<X_DTYPE, WEIGHT_DTYPE, OP_CODE>::Adaption(RowRange range)
{
    if (range.actualRowNum > 1) {
        int64_t yOffset = 0;
        for (int64_t rowIdx = range.rowStart; rowIdx < range.rowEnd; rowIdx++) {
            int64_t scaleOffset = (rowIdx / seqLen - range.batchStart) * hiddenDimCeil;
            Mul(yFloat[yOffset], yFloat[yOffset], scaleFloat[scaleOffset], hiddenDim);
            yOffset += hiddenDimCeil;
        }
        PipeBarrier<PIPE_V>();
        yOffset = 0;
        for (int64_t rowIdx = range.rowStart; rowIdx < range.rowEnd; rowIdx++) {
            int64_t shiftOffset = (rowIdx / seqLen - range.batchStart) * hiddenDimCeil;
            Add(yFloat[yOffset], yFloat[yOffset], shiftFloat[shiftOffset], hiddenDim);
            yOffset += hiddenDimCeil;
        }
    } else {
        Mul(yFloat, yFloat, scaleFloat, range.dataCount);
        PipeBarrier<PIPE_V>();
        Add(yFloat, yFloat, shiftFloat, range.dataCount);
    }
}

template <typename X_DTYPE, typename WEIGHT_DTYPE, uint8_t OP_CODE>
__aicore__ inline void AdaLayerNormND<X_DTYPE, WEIGHT_DTYPE, OP_CODE>::DynamicQuant(RowRange range, int64_t batchIdx)
{
    PipeBarrier<PIPE_V>();
    if (hasSmooth) {
        for (int64_t rowIdx = 0; rowIdx < range.actualRowNum; rowIdx++) {
            Mul(yFloat[rowIdx * hiddenDimCeil], yFloat[rowIdx * hiddenDimCeil], smoothFloat, hiddenDim);
        }
        PipeBarrier<PIPE_V>();
    }
    Abs(xFloat, yFloat, range.dataCount);
    PipeBarrier<PIPE_V>();
    if (range.actualRowNum > 1) {
        for (int64_t rowIdx = 0; rowIdx < range.actualRowNum; rowIdx++) {
            ReduceMax(
                reduceFloat[rowIdx], xFloat[rowIdx * hiddenDimCeil], xFloat[rowIdx * hiddenDimCeil], hiddenDim, false);
        }
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_S>(eventIdVToS);
        WaitFlag<HardEvent::V_S>(eventIdVToS);
        for (int64_t rowIdx = 0; rowIdx < range.actualRowNum; rowIdx++) {
            quantScaleFloat.SetValue(batchIdx + rowIdx, reduceFloat.GetValue(rowIdx) / MAX_INT8);
        }
        Muls(reduceFloat, reduceFloat, FACTOR_INT8, range.actualRowNum);
        PipeBarrier<PIPE_V>();
        uint32_t srcShape[2] = {static_cast<uint32_t>(range.actualRowNum), 1};
        uint32_t dstShape[2] = {static_cast<uint32_t>(range.actualRowNum), static_cast<uint32_t>(hiddenDimCeil)};
        BroadCast<float, 2, 1>(xFloat, reduceFloat, dstShape, srcShape);
        PipeBarrier<PIPE_V>();
        Div(yFloat, yFloat, xFloat, range.dataCount);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE2>(eventIdVToMte2);
        WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2);
    } else {
        ReduceMax(reduceFloat, xFloat, xFloat, hiddenDim, false);
        SetFlag<HardEvent::V_MTE2>(eventIdVToMte2);
        WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2);
        SetFlag<HardEvent::V_S>(eventIdVToS);
        WaitFlag<HardEvent::V_S>(eventIdVToS);
        float maxValue = reduceFloat.GetValue(0);
        quantScaleFloat.SetValue(batchIdx, maxValue / MAX_INT8);
        Muls(yFloat, yFloat, MAX_INT8 / maxValue, range.dataCount);
        PipeBarrier<PIPE_V>();
    }
    Cast(yHalf, yFloat, RoundMode::CAST_RINT, range.dataCount);
    PipeBarrier<PIPE_V>();
    for (int64_t rowIdx = 0; rowIdx < range.actualRowNum; rowIdx++) {
        Cast(yInt[rowIdx * outIntCeil], yHalf[rowIdx * hiddenDimCeil], RoundMode::CAST_ROUND, hiddenDim);
    }
}

template <typename X_DTYPE, typename WEIGHT_DTYPE, uint8_t OP_CODE>
__aicore__ inline void AdaLayerNormND<X_DTYPE, WEIGHT_DTYPE, OP_CODE>::CopyInData(
    LocalTensor<float> inputFloat, GlobalTensor<X_DTYPE> inputGm, int64_t len)
{
    DataCopyExtParams copyInParams{1, static_cast<uint32_t>(len * sizeof(X_DTYPE)), 0, 0, 0};
    DataCopyPadExtParams<X_DTYPE> padParams{false, 0, 0, 0};
    if constexpr (std::is_same_v<X_DTYPE, float>) {
        DataCopyPad(inputFloat, inputGm, copyInParams, padParams);
    } else {
        DataCopyPad(inputFloat.ReinterpretCast<X_DTYPE>()[DATA_COUNT], inputGm, copyInParams, padParams);
    }
    SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    if constexpr (!std::is_same_v<X_DTYPE, float>) {
        Cast(inputFloat, inputFloat.ReinterpretCast<X_DTYPE>()[DATA_COUNT], RoundMode::CAST_NONE, len);
        PipeBarrier<PIPE_V>();
    }
}

template <typename X_DTYPE, typename WEIGHT_DTYPE, uint8_t OP_CODE>
__aicore__ inline void AdaLayerNormND<X_DTYPE, WEIGHT_DTYPE, OP_CODE>::CopyInWeightBias(
    LocalTensor<float> inputFloat, GlobalTensor<WEIGHT_DTYPE> inputGm, int64_t len)
{
    DataCopyExtParams copyInParams{1, static_cast<uint32_t>(len * sizeof(WEIGHT_DTYPE)), 0, 0, 0};
    DataCopyPadExtParams<WEIGHT_DTYPE> padParams{false, 0, 0, 0};
    if constexpr (std::is_same_v<WEIGHT_DTYPE, float>) {
        DataCopyPad(inputFloat, inputGm, copyInParams, padParams);
    } else {
        DataCopyPad(inputFloat.ReinterpretCast<WEIGHT_DTYPE>()[DATA_COUNT], inputGm, copyInParams, padParams);
    }
    SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    if constexpr (!std::is_same_v<WEIGHT_DTYPE, float>) {
        Cast(inputFloat, inputFloat.ReinterpretCast<WEIGHT_DTYPE>()[DATA_COUNT], RoundMode::CAST_NONE, len);
        PipeBarrier<PIPE_V>();
    }
}

template <typename X_DTYPE, typename WEIGHT_DTYPE, uint8_t OP_CODE>
__aicore__ inline void AdaLayerNormND<X_DTYPE, WEIGHT_DTYPE, OP_CODE>::CopyInOtherData(int64_t offset, int64_t len)
{
    if (hasWeight) {
        CopyInWeightBias(weightFloat, weightGm[offset], len);
    }
    if (hasBias) {
        CopyInWeightBias(biasFloat, biasGm[offset], len);
    }
    if constexpr (OP_CODE == QUANT_OP_CODE) {
        if (hasSmooth) {
            CopyInData(smoothFloat, smoothGm[offset], len);
        }
    }
}

template <typename X_DTYPE, typename WEIGHT_DTYPE, uint8_t OP_CODE>
__aicore__ inline void AdaLayerNormND<X_DTYPE, WEIGHT_DTYPE, OP_CODE>::CopyInScaleShift(int64_t offset, uint16_t blockCount, int64_t len)
{
    DataCopyExtParams copyInParams{blockCount, static_cast<uint32_t>(len * sizeof(X_DTYPE)), 0, 0, 0};
    DataCopyPadExtParams<X_DTYPE> padParams{false, 0, 0, 0};
    if (blockCount > 1 && len < hiddenDimCeil) {
        padParams.isPad = true;
        padParams.paddingValue = 0;
        padParams.rightPadding = hiddenDimCeil - len;
    }
    if constexpr (std::is_same_v<X_DTYPE, float>) {
        DataCopyPad(scaleFloat, scaleGm[offset], copyInParams, padParams);
        DataCopyPad(shiftFloat, shiftGm[offset], copyInParams, padParams);
    } else {
        DataCopyPad(scaleFloat.ReinterpretCast<X_DTYPE>()[DATA_COUNT], scaleGm[offset], copyInParams, padParams);
        DataCopyPad(shiftFloat.ReinterpretCast<X_DTYPE>()[DATA_COUNT], shiftGm[offset], copyInParams, padParams);
    }
    SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    int64_t scaleCount = blockCount > 1 ? blockCount * hiddenDimCeil : len;
    if constexpr (!std::is_same_v<X_DTYPE, float>) {
        Cast(scaleFloat, scaleFloat.ReinterpretCast<X_DTYPE>()[DATA_COUNT], RoundMode::CAST_NONE, scaleCount);
        Cast(shiftFloat, shiftFloat.ReinterpretCast<X_DTYPE>()[DATA_COUNT], RoundMode::CAST_NONE, scaleCount);
        PipeBarrier<PIPE_V>();
    }
    Adds(scaleFloat, scaleFloat, static_cast<float>(1), scaleCount);
    PipeBarrier<PIPE_V>();
}

template <typename X_DTYPE, typename WEIGHT_DTYPE, uint8_t OP_CODE>
__aicore__ inline void AdaLayerNormND<X_DTYPE, WEIGHT_DTYPE, OP_CODE>::CopyInSlice(
    int64_t offset, int64_t scaleOffset, int64_t h, int64_t len)
{
    CopyInData(xFloat, xGm[offset + h], len);
    CopyInScaleShift(scaleOffset, 1, len);
    CopyInOtherData(h, len);
}

template <typename X_DTYPE, typename WEIGHT_DTYPE, uint8_t OP_CODE>
__aicore__ inline void AdaLayerNormND<X_DTYPE, WEIGHT_DTYPE, OP_CODE>::CopyInSliceX(int64_t offset, int64_t len)
{
    DataCopyExtParams copyInParams{1, static_cast<uint32_t>(len * sizeof(X_DTYPE)), 0, 0, 0};
    DataCopyPadExtParams<X_DTYPE> padParams{false, 0, 0, 0};
    if constexpr (std::is_same_v<X_DTYPE, float>) {
        DataCopyPad(xFloat, xGm[offset], copyInParams, padParams);
    } else {
        DataCopyPad(xFloat.ReinterpretCast<X_DTYPE>()[MAX_X_SIZE], xGm[offset], copyInParams, padParams);
    }
    SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    if constexpr (!std::is_same_v<X_DTYPE, float>) {
        Cast(xFloat, xFloat.ReinterpretCast<X_DTYPE>()[MAX_X_SIZE], RoundMode::CAST_NONE, len);
        PipeBarrier<PIPE_V>();
    }
}

template <typename X_DTYPE, typename WEIGHT_DTYPE, uint8_t OP_CODE>
__aicore__ inline void AdaLayerNormND<X_DTYPE, WEIGHT_DTYPE, OP_CODE>::CopyInX(int64_t offset, uint16_t blockCount, int64_t len)
{
    DataCopyExtParams copyInParams{blockCount, static_cast<uint32_t>(len * sizeof(X_DTYPE)), 0, 0, 0};
    DataCopyPadExtParams<X_DTYPE> padParams{false, 0, 0, 0};
    if (blockCount > 1 && len < hiddenDimCeil) {
        padParams.isPad = true;
        padParams.paddingValue = 0;
        padParams.rightPadding = hiddenDimCeil - len;
    }
    if constexpr (std::is_same_v<X_DTYPE, float>) {
        DataCopyPad(xFloat, xGm[offset], copyInParams, padParams);
    } else {
        DataCopyPad(xFloat.ReinterpretCast<X_DTYPE>()[DATA_COUNT], xGm[offset], copyInParams, padParams);
    }
    SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    if constexpr (!std::is_same_v<X_DTYPE, float>) {
        Cast(xFloat, xFloat.ReinterpretCast<X_DTYPE>()[DATA_COUNT], RoundMode::CAST_NONE, blockCount * hiddenDimCeil);
        PipeBarrier<PIPE_V>();
    }
}

template <typename X_DTYPE, typename WEIGHT_DTYPE, uint8_t OP_CODE>
__aicore__ inline void AdaLayerNormND<X_DTYPE, WEIGHT_DTYPE, OP_CODE>::BaseCopyOut(int64_t offset, uint16_t blockCount, int64_t len)
{
    if constexpr (!std::is_same_v<X_DTYPE, float>) {
        PipeBarrier<PIPE_V>();
        int64_t dataCount = blockCount > 1 ? blockCount * hiddenDimCeil : len;
        Cast(yFloat.ReinterpretCast<X_DTYPE>(), yFloat, RoundMode::CAST_RINT, dataCount);
    }
    SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
    WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
    DataCopyExtParams copyOutParams{blockCount, static_cast<uint32_t>(len * sizeof(X_DTYPE)), 0, 0, 0};
    DataCopyPad(outGm[offset], yFloat.ReinterpretCast<X_DTYPE>(), copyOutParams);
}

template <typename X_DTYPE, typename WEIGHT_DTYPE, uint8_t OP_CODE>
__aicore__ inline void AdaLayerNormND<X_DTYPE, WEIGHT_DTYPE, OP_CODE>::QuantCopyOut(int64_t offset, uint16_t blockCount, int64_t len)
{
    SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
    WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
    DataCopyExtParams copyOutParams{blockCount, static_cast<uint32_t>(len * sizeof(int8_t)), 0, 0, 0};
    DataCopyPad(quantOutGm[offset], yInt, copyOutParams);
}

template <typename X_DTYPE, typename WEIGHT_DTYPE, uint8_t OP_CODE>
__aicore__ inline void AdaLayerNormND<X_DTYPE, WEIGHT_DTYPE, OP_CODE>::CopyScaleOut(int64_t offset, int64_t len)
{
    DataCopyExtParams copyOutParams{1, static_cast<uint32_t>(len * sizeof(float)), 0, 0, 0};
    DataCopyPad(quantScaleGm[offset], quantScaleFloat, copyOutParams);
    SetFlag<HardEvent::MTE3_S>(eventIdMte3ToS);
    WaitFlag<HardEvent::MTE3_S>(eventIdMte3ToS);
}

template <typename X_DTYPE, typename WEIGHT_DTYPE, uint8_t OP_CODE>
__aicore__ inline void AdaLayerNormND<X_DTYPE, WEIGHT_DTYPE, OP_CODE>::CopyMeanRstdOut(int64_t offset, int64_t len)
{
    if constexpr (!std::is_same_v<X_DTYPE, float>) {
        Cast(meanOutFloat.ReinterpretCast<X_DTYPE>(), meanOutFloat, RoundMode::CAST_RINT, len);
        Cast(rstdOutFloat.ReinterpretCast<X_DTYPE>(), rstdOutFloat, RoundMode::CAST_RINT, len);
        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
    }
    DataCopyExtParams copyOutParams{1, static_cast<uint32_t>(len * sizeof(X_DTYPE)), 0, 0, 0};
    DataCopyPad(meanGm[offset], meanOutFloat.ReinterpretCast<X_DTYPE>(), copyOutParams);
    DataCopyPad(rstdGm[offset], rstdOutFloat.ReinterpretCast<X_DTYPE>(), copyOutParams);
    SetFlag<HardEvent::MTE3_S>(eventIdMte3ToS);
    WaitFlag<HardEvent::MTE3_S>(eventIdMte3ToS);
}

template <typename X_DTYPE, typename WEIGHT_DTYPE, uint8_t OP_CODE>
__aicore__ inline void AdaLayerNormND<X_DTYPE, WEIGHT_DTYPE, OP_CODE>::CopyNormOut(int64_t h, int64_t dataCount)
{
    SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
    WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
    DataCopyExtParams copyOutParams{1, static_cast<uint32_t>(dataCount * sizeof(float)), 0, 0, 0};
    DataCopyPad(normGm[normOffset + h], xFloat, copyOutParams);
}

template <typename X_DTYPE, typename WEIGHT_DTYPE, uint8_t OP_CODE>
__aicore__ inline void AdaLayerNormND<X_DTYPE, WEIGHT_DTYPE, OP_CODE>::CopyInNorm(int64_t h, int64_t dataCount)
{
    DataCopyExtParams copyInParams{1, static_cast<uint32_t>(dataCount * sizeof(float)), 0, 0, 0};
    DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
    DataCopyPad(xFloat, normGm[normOffset + h], copyInParams, padParams);
    SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
}


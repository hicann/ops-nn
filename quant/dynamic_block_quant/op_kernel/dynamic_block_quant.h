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
 * \file dynamic_block_quant.h
 * \brief
 */
#ifndef DYNAMIC_BLOCK_QUANT
#define DYNAMIC_BLOCK_QUANT

#include "kernel_operator.h"

namespace DynamicBlockQuant {
using namespace AscendC;

constexpr float NUM_INT_8_MAX = 127.0f;
constexpr float BLOCK_MAX_VALUE_MIN = 1e-4f;
constexpr int64_t SINGLE_DATA_BLOCK_SIZE = 32;
constexpr int64_t ONE_K_BYTES = 1024;

constexpr int64_t NUM_TWO = 2;
constexpr int64_t NUM_EIGHT = 8;
constexpr int64_t NUM_ONE_SIX = 16;
constexpr int64_t NUM_THREE_TWO = 32;
constexpr int64_t NUM_SIX_FOUR = 64;
constexpr int64_t NUM_ONE_TWO_EIGHT = 128;

template <class T>
struct xCalType {
    using type = half;
};

template <>
struct xCalType<bfloat16_t> {
    using type = float;
};

template <typename T>
class DynamicBlockQuantND {
    using XCALTYPE = typename xCalType<T>::type;

public:
    __aicore__ inline DynamicBlockQuantND(){};

    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR y, GM_ADDR scale, GM_ADDR workspace, DynamicBlockQuantTilingData* tilingData);

    __aicore__ inline void Process();

private:
    __aicore__ inline void ParseTilingData(DynamicBlockQuantTilingData* tilingData);

    __aicore__ inline void InitGlobalTensors(GM_ADDR x, GM_ADDR y, GM_ADDR scale);
    __aicore__ inline void InitScalars();
    __aicore__ inline void InitLocalTensors();
    __aicore__ inline void InitEventId();

    __aicore__ inline void ProcessWholeRows(int64_t rowIdx, int64_t perLoopRow, int64_t startRowIdx);
    __aicore__ inline void ProcessSplitRow(int64_t rowIdx, int64_t perLoopRow, int64_t startRowIdx);

    __aicore__ inline void ProcessTail(int64_t rowIdx, int64_t colBlockStart, int64_t colBlockEnd);

    __aicore__ inline void Compute(int64_t calcNum, int64_t startBlockIdx);
    __aicore__ inline void ComputeReduceBf16(int64_t calcNum, int64_t startBlockIdx, int64_t scaleCount);
    __aicore__ inline void ComputeReduceFp16(int64_t calcNum, int64_t startBlockIdx, int64_t scaleCount);
    __aicore__ inline void ComputeQuant(int64_t calcNum, int64_t startBlockIdx);
    __aicore__ inline void CopyIn(int64_t rowStart, int64_t rowCount);
    __aicore__ inline void CopyOutY(int64_t rowStart, int64_t rowCount);
    __aicore__ inline void CopyInTail(int64_t startRowIdx, int64_t colStartBlock, int64_t colEndBlock);
    __aicore__ inline void CopyOutYTail(int64_t startRowIdx, int64_t colStartBlock, int64_t colEndBlock);
    __aicore__ inline void CopyOutScale(int64_t startIdx, int64_t copyNum);

    template <typename T1>
    __aicore__ inline T1 CeilDiv(T1 x, T1 y)
    {
        return y != 0 ? (x + y - 1) / y : x;
    };
    template <typename T1>
    __aicore__ inline T1 GetMax(T1 x, T1 y)
    {
        return x > y ? x : y;
    };
    template <typename T1>
    __aicore__ inline T1 GetMin(T1 x, T1 y)
    {
        return x > y ? y : x;
    };

private:
    TPipe pipe;
    GlobalTensor<T> xGM;
    GlobalTensor<int8_t> yGM;
    GlobalTensor<float> scaleGM;

    TBuf<QuePosition::VECCALC> bufQueue;

    LocalTensor<T> xLocal;
    LocalTensor<float> xLocalTmp;
    LocalTensor<XCALTYPE> xLocalAbs;
    LocalTensor<float> scaleLocal;
    LocalTensor<float> quantScale;
    LocalTensor<float> quantScaleBrcb;
    LocalTensor<float> maskTmp;
    LocalTensor<float> quantMask;
    LocalTensor<float> quantMask2;
    LocalTensor<float> quantMask3;            // 填充127.0的tensor
    LocalTensor<float> blockMaxValueMinLimit; // 填充inputMin的限制,防止max值过小
    LocalTensor<T> clearUbZero;               // 全0tensor，用于clear ub

    LocalTensor<int8_t> xLocalTmpInt8; // 存放量化为int8的结果

    event_t eventIdMTE2ToV;
    event_t eventIdVToMTE2;
    event_t eventIdVToMTE3;
    event_t eventIdMTE3ToV;

    int64_t coreIdx = 0;
    int64_t totalCoreNum = 0;
    int64_t usedCoreNum = 0;

    int64_t rowNum = 0;
    int64_t colNum = 0;
    int64_t colPadNum = 0; // 列pad后的总数据量
    int64_t blockSizeRow = 1;
    int64_t blockSizeCol = 1;
    int64_t colBlockTotalNum = 0;   // 列的block数量
    int64_t colPadExtNum = 0;       // 列最后一个block须pad的数据量
    int64_t colPadToBlockNum = 0;   // 列最后一个block中已有数据pad到32kb需要的数据量
    int64_t colClearExtNum = 0;     // 列最后一个block中pad后须清零的数据量
    int64_t perCoreRowNum = 0;      // 每个核平均处理的行数
    int64_t singleLoopMaxBlock = 0; // 单次循环最大处理的block数量

    int64_t tailRow = 0;
    int64_t tailColBlockStart = 0;
    int64_t tailColBlockEnd = 0;

    float blockMaxValueMin = BLOCK_MAX_VALUE_MIN;

    int64_t ubSize = 0;

    bool isFirst = true; // 是否为第一次进行计算
};

template <typename T>
__aicore__ inline void DynamicBlockQuantND<T>::Init(
    GM_ADDR x, GM_ADDR y, GM_ADDR scale, GM_ADDR workspace, DynamicBlockQuantTilingData* tilingData)
{
    // 获取当前核的索引
    coreIdx = GetBlockIdx();

    // 加载传入的参数
    ParseTilingData(tilingData);
    InitScalars();
    InitGlobalTensors(x, y, scale);
    InitLocalTensors();
    InitEventId();
}

template <typename T>
__aicore__ inline void DynamicBlockQuantND<T>::InitScalars()
{
    colBlockTotalNum = CeilDiv(colNum, blockSizeCol);
    colPadNum = colBlockTotalNum * blockSizeCol;
    colPadExtNum = colPadNum - colNum;
    colPadToBlockNum = colPadExtNum % (SINGLE_DATA_BLOCK_SIZE / sizeof(T));
    colClearExtNum = colPadExtNum - colPadToBlockNum;
    singleLoopMaxBlock = NUM_THREE_TWO * ONE_K_BYTES / sizeof(T) / blockSizeCol;
}

template <typename T>
__aicore__ inline void DynamicBlockQuantND<T>::InitGlobalTensors(GM_ADDR x, GM_ADDR y, GM_ADDR scale)
{
    xGM.SetGlobalBuffer((__gm__ T*)x);
    yGM.SetGlobalBuffer((__gm__ int8_t*)y);
    scaleGM.SetGlobalBuffer((__gm__ float*)scale);
}

template <typename T>
__aicore__ inline void DynamicBlockQuantND<T>::InitLocalTensors()
{
    pipe.InitBuffer(bufQueue, ubSize);
    LocalTensor<uint8_t> tmp = bufQueue.Get<uint8_t>();
    int64_t offset = 0;
    xLocal = tmp[offset].ReinterpretCast<T>();
    offset += NUM_THREE_TWO * ONE_K_BYTES;
    xLocalTmp = tmp[offset].ReinterpretCast<float>();
    xLocalTmpInt8 = xLocalTmp.template ReinterpretCast<int8_t>();
    offset += NUM_SIX_FOUR * ONE_K_BYTES;
    xLocalAbs = tmp[offset].ReinterpretCast<XCALTYPE>();
    offset += NUM_SIX_FOUR * ONE_K_BYTES;
    scaleLocal = tmp[offset].ReinterpretCast<float>();
    offset += 1 * ONE_K_BYTES;
    quantScale = tmp[offset].ReinterpretCast<float>();
    offset += NUM_EIGHT * ONE_K_BYTES;

    quantScaleBrcb = tmp[offset].ReinterpretCast<float>();
    offset += 4 * ONE_K_BYTES;

    quantMask = tmp[offset].ReinterpretCast<float>();
    offset += NUM_EIGHT * sizeof(float);
    quantMask2 = tmp[offset].ReinterpretCast<float>();
    offset += NUM_EIGHT * sizeof(float);
    quantMask3 = tmp[offset].ReinterpretCast<float>();
    offset += NUM_EIGHT * sizeof(float);

    Duplicate(quantMask, (float)(1.0), NUM_EIGHT);
    Duplicate(quantMask2, (float)(1.0) / NUM_INT_8_MAX, NUM_EIGHT);
    Duplicate(quantMask3, NUM_INT_8_MAX, NUM_EIGHT);

    if (colClearExtNum > 0) {
        clearUbZero = tmp[offset].ReinterpretCast<T>();
        offset += colClearExtNum * sizeof(T);
        Duplicate(clearUbZero, (T)(0), colClearExtNum);
    }
}

template <typename T>
__aicore__ inline void DynamicBlockQuantND<T>::InitEventId()
{
    eventIdMTE2ToV = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());
    eventIdVToMTE2 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>());
    eventIdVToMTE3 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>());
    eventIdMTE3ToV = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>());
}

template <typename T>
__aicore__ inline void DynamicBlockQuantND<T>::Process()
{
    // 单次处理的行数
    int64_t perLoopRow = GetMin(singleLoopMaxBlock / colBlockTotalNum, perCoreRowNum);
    int64_t startRowIdx = coreIdx * perCoreRowNum;
    for (int64_t i = startRowIdx; i < startRowIdx + perCoreRowNum; i += GetMax(perLoopRow, (int64_t)1)) {
        if (perLoopRow >= 1) {
            ProcessWholeRows(i, GetMin(perLoopRow, startRowIdx + perCoreRowNum - i), startRowIdx);

        } else {
            ProcessSplitRow(i, 0, colBlockTotalNum);
        }
    }
    ProcessSplitRow(tailRow, tailColBlockStart, tailColBlockEnd);
}

template <typename T>
__aicore__ inline void DynamicBlockQuantND<T>::ProcessWholeRows(int64_t rowIdx, int64_t perLoopRow, int64_t startRowIdx)
{
    // 处理整行
    CopyIn(rowIdx, perLoopRow);
    if (!isFirst) {
        SetFlag<HardEvent::MTE3_V>(eventIdMTE3ToV);
        WaitFlag<HardEvent::MTE3_V>(eventIdMTE3ToV);
    }
    SetFlag<HardEvent::MTE2_V>(eventIdMTE2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIdMTE2ToV);
    Compute(perLoopRow * colPadNum, rowIdx * colBlockTotalNum);
    SetFlag<HardEvent::V_MTE3>(eventIdVToMTE3);
    WaitFlag<HardEvent::V_MTE3>(eventIdVToMTE3);
    CopyOutY(rowIdx, perLoopRow);
    if (isFirst) {
        isFirst = false;
    }
}

template <typename T>
__aicore__ inline void DynamicBlockQuantND<T>::ProcessSplitRow(
    int64_t rowIdx, int64_t colBlockStart, int64_t colBlockEnd)
{
    for (int64_t i = colBlockStart; i < colBlockEnd; i += singleLoopMaxBlock) {
        ProcessTail(rowIdx, i, GetMin(i + singleLoopMaxBlock, colBlockEnd));
        if (isFirst) {
            isFirst = false;
        }
    }
}

template <typename T>
__aicore__ inline void DynamicBlockQuantND<T>::ProcessTail(int64_t rowIdx, int64_t colBlockStart, int64_t colBlockEnd)
{
    // 单次处理小于一行
    if (colBlockEnd <= colBlockStart) {
        return;
    }
    CopyInTail(rowIdx, colBlockStart, colBlockEnd);

    if (!isFirst) {
        SetFlag<HardEvent::MTE3_V>(eventIdMTE3ToV);
        WaitFlag<HardEvent::MTE3_V>(eventIdMTE3ToV);
    }
    SetFlag<HardEvent::MTE2_V>(eventIdMTE2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIdMTE2ToV);
    Compute((colBlockEnd - colBlockStart) * blockSizeCol, rowIdx * colBlockTotalNum + colBlockStart);
    SetFlag<HardEvent::V_MTE3>(eventIdVToMTE3);
    WaitFlag<HardEvent::V_MTE3>(eventIdVToMTE3);
    CopyOutYTail(rowIdx, colBlockStart, colBlockEnd);
}

template <typename T>
__aicore__ inline void DynamicBlockQuantND<T>::Compute(int64_t calcNum, int64_t startBlockIdx)
{
    int64_t scaleCount = calcNum / NUM_ONE_TWO_EIGHT;
    if constexpr (IsSameType<T, bfloat16_t>::value) {
        ComputeReduceBf16(calcNum, startBlockIdx, scaleCount);
    } else {
        ComputeReduceFp16(calcNum, startBlockIdx, scaleCount);
    }
    ComputeQuant(calcNum, startBlockIdx);
}

template <typename T>
__aicore__ inline void DynamicBlockQuantND<T>::ComputeReduceBf16(
    int64_t calcNum, int64_t startBlockIdx, int64_t scaleCount)
{
    Cast(xLocalTmp, xLocal, RoundMode::CAST_NONE, calcNum);
    SetFlag<HardEvent::V_MTE2>(eventIdVToMTE2);
    WaitFlag<HardEvent::V_MTE2>(eventIdVToMTE2);
    PipeBarrier<PIPE_V>();
    Abs(xLocalAbs, xLocalTmp, static_cast<int32_t>(calcNum));
    PipeBarrier<PIPE_V>();
    // 计算每个block的最大值
    WholeReduceMax(scaleLocal, xLocalAbs, NUM_SIX_FOUR, scaleCount, 1, 1, NUM_ONE_SIX, ReduceOrder::ORDER_ONLY_VALUE);
    WholeReduceMax(
        scaleLocal[NUM_ONE_TWO_EIGHT], xLocalAbs[NUM_SIX_FOUR], NUM_SIX_FOUR, scaleCount, 1, 1, NUM_ONE_SIX,
        ReduceOrder::ORDER_ONLY_VALUE);
    PipeBarrier<PIPE_V>();
    Max(scaleLocal, scaleLocal, scaleLocal[NUM_ONE_TWO_EIGHT], scaleCount);
    PipeBarrier<PIPE_V>();
    // bf16场景因xLocakAbs已为float，可先div再扩展
    Div(scaleLocal[NUM_ONE_TWO_EIGHT], quantMask, scaleLocal, NUM_SIX_FOUR, CeilDiv(scaleCount, NUM_SIX_FOUR),
        {1, 0, 1, NUM_EIGHT, 0, NUM_EIGHT});
    PipeBarrier<PIPE_V>();
    Mul(scaleLocal[NUM_ONE_TWO_EIGHT], scaleLocal[NUM_ONE_TWO_EIGHT], quantMask3, NUM_SIX_FOUR,
        CeilDiv(scaleCount, NUM_SIX_FOUR), {1, 1, 0, NUM_EIGHT, NUM_EIGHT, 0});
    PipeBarrier<PIPE_V>();

    // 计算scale的输出
    Mul(scaleLocal, scaleLocal, quantMask2, NUM_SIX_FOUR, CeilDiv(scaleCount, NUM_SIX_FOUR),
        {1, 1, 0, NUM_EIGHT, NUM_EIGHT, 0});
    CopyOutScale(startBlockIdx, scaleCount);

    // 扩展quantScale用于量化x
    Brcb(quantScaleBrcb, scaleLocal[NUM_ONE_TWO_EIGHT], CeilDiv(scaleCount, NUM_EIGHT), {1, NUM_EIGHT});
    PipeBarrier<PIPE_V>();
    Copy(
        quantScale, quantScaleBrcb, NUM_SIX_FOUR, CeilDiv(scaleCount * NUM_EIGHT, NUM_SIX_FOUR),
        {NUM_TWO, 1, NUM_ONE_SIX, NUM_EIGHT});
    Copy(
        quantScale[NUM_EIGHT], quantScaleBrcb, NUM_SIX_FOUR, CeilDiv(scaleCount * NUM_EIGHT, NUM_SIX_FOUR),
        {NUM_TWO, 1, NUM_ONE_SIX, NUM_EIGHT});
    PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline void DynamicBlockQuantND<T>::ComputeReduceFp16(
    int64_t calcNum, int64_t startBlockIdx, int64_t scaleCount)
{
    LocalTensor<T> scaleLocalT = scaleLocal.template ReinterpretCast<T>()[NUM_ONE_TWO_EIGHT];
    LocalTensor<T> quantScaleLocalT = quantScale.template ReinterpretCast<T>()[NUM_ONE_TWO_EIGHT * NUM_ONE_SIX];
    Abs(xLocalAbs, xLocal, calcNum);
    Cast(xLocalTmp, xLocal, RoundMode::CAST_NONE, calcNum);
    PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::V_MTE2>(eventIdVToMTE2);
    WaitFlag<HardEvent::V_MTE2>(eventIdVToMTE2);
    WholeReduceMax(
        scaleLocalT, xLocalAbs, NUM_ONE_TWO_EIGHT, scaleCount, 1, 1, NUM_EIGHT, ReduceOrder::ORDER_ONLY_VALUE);
    PipeBarrier<PIPE_V>();
    // fp16场景因xLocakAbs不为float，先扩展再div
    Brcb(quantScaleLocalT, scaleLocalT, CeilDiv(calcNum / NUM_ONE_TWO_EIGHT, NUM_EIGHT), {1, NUM_EIGHT});
    PipeBarrier<PIPE_V>();
    Cast(quantScale, quantScaleLocalT, RoundMode::CAST_NONE, calcNum / NUM_ONE_TWO_EIGHT * NUM_ONE_SIX);
    Cast(scaleLocal, scaleLocalT, RoundMode::CAST_NONE, calcNum / NUM_ONE_TWO_EIGHT);
    PipeBarrier<PIPE_V>();

    Mul(scaleLocal, scaleLocal, quantMask2, NUM_SIX_FOUR, CeilDiv(scaleCount, NUM_SIX_FOUR),
        {1, 1, 0, NUM_EIGHT, NUM_EIGHT, 0});
    CopyOutScale(startBlockIdx, scaleCount);
    Div(quantScale, quantMask, quantScale, NUM_SIX_FOUR,
        CeilDiv(calcNum / NUM_ONE_TWO_EIGHT * NUM_ONE_SIX, NUM_SIX_FOUR), {1, 0, 1, NUM_EIGHT, 0, NUM_EIGHT});
    PipeBarrier<PIPE_V>();
    Mul(quantScale, quantScale, quantMask3, NUM_SIX_FOUR,
        CeilDiv(calcNum / NUM_ONE_TWO_EIGHT * NUM_ONE_SIX, NUM_SIX_FOUR), {1, 1, 0, NUM_EIGHT, NUM_EIGHT, 0});
    PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline void DynamicBlockQuantND<T>::ComputeQuant(int64_t calcNum, int64_t startBlockIdx)
{
    LocalTensor<half> xLocalTmpHalf = xLocalTmp.template ReinterpretCast<half>();
    if (calcNum / NUM_SIX_FOUR > 255) {
        Mul(xLocalTmp, xLocalTmp, quantScale, NUM_SIX_FOUR, calcNum / NUM_SIX_FOUR / NUM_TWO,
            {1, 1, 0, NUM_EIGHT, NUM_EIGHT, 1});
        Mul(xLocalTmp[calcNum / NUM_TWO], xLocalTmp[calcNum / NUM_TWO],
            quantScale[calcNum / NUM_SIX_FOUR / NUM_TWO * NUM_EIGHT], NUM_SIX_FOUR, calcNum / NUM_SIX_FOUR / NUM_TWO,
            {1, 1, 0, NUM_EIGHT, NUM_EIGHT, 1});
    } else {
        Mul(xLocalTmp, xLocalTmp, quantScale, NUM_SIX_FOUR, calcNum / NUM_SIX_FOUR, {1, 1, 0, NUM_EIGHT, NUM_EIGHT, 1});
    }
    PipeBarrier<PIPE_V>();

    Cast(xLocalTmpHalf.template ReinterpretCast<int16_t>(), xLocalTmp, RoundMode::CAST_RINT, calcNum);
    PipeBarrier<PIPE_V>();

    Cast(xLocalTmpHalf, xLocalTmpHalf.template ReinterpretCast<int16_t>(), RoundMode::CAST_NONE, calcNum);
    PipeBarrier<PIPE_V>();

    Cast(xLocalTmpInt8, xLocalTmpHalf, RoundMode::CAST_NONE, calcNum);
    PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline void DynamicBlockQuantND<T>::CopyIn(int64_t rowStart, int64_t rowCount)
{
    if (colPadExtNum == 0) {
        DataCopy(xLocal, xGM[rowStart * colNum], rowCount * colNum);
    } else if (colPadToBlockNum == 0) {
        DataCopyParams copyParams{
            static_cast<uint16_t>(rowCount), static_cast<uint16_t>(colNum * sizeof(T) / SINGLE_DATA_BLOCK_SIZE), 0,
            static_cast<uint16_t>(colPadExtNum * sizeof(T) / SINGLE_DATA_BLOCK_SIZE)};
        DataCopy(xLocal, xGM[rowStart * colNum], copyParams);
        if (colClearExtNum > 0) {
            Copy(
                xLocal[colNum], clearUbZero, colClearExtNum, rowCount,
                {1, 1, static_cast<uint16_t>(colPadNum / (SINGLE_DATA_BLOCK_SIZE / sizeof(T))), 0});
            PipeBarrier<PIPE_V>();
        }
    } else {
        DataCopyExtParams copyExtParams{
            static_cast<uint16_t>(rowCount), static_cast<uint16_t>(colNum * sizeof(T)), 0,
            static_cast<uint16_t>(colPadExtNum * sizeof(T) / SINGLE_DATA_BLOCK_SIZE), 0};
        DataCopyPadExtParams<T> padParams{static_cast<uint16_t>(1), 0, static_cast<uint8_t>(colPadToBlockNum), 0};
        DataCopyPad(xLocal, xGM[rowStart * colNum], copyExtParams, padParams);
        if (colClearExtNum > 0) {
            Copy(
                xLocal[colNum + colPadToBlockNum], clearUbZero, colClearExtNum, rowCount,
                {1, 1, static_cast<uint16_t>(colPadNum / (SINGLE_DATA_BLOCK_SIZE / sizeof(T))), 0});
            PipeBarrier<PIPE_V>();
        }
    }
}

template <typename T>
__aicore__ inline void DynamicBlockQuantND<T>::CopyOutY(int64_t rowStart, int64_t rowCount)
{
    if (colPadExtNum * sizeof(int8_t) % SINGLE_DATA_BLOCK_SIZE == 0) {
        DataCopyParams copyParams{
            static_cast<uint16_t>(rowCount), static_cast<uint16_t>(colNum * sizeof(int8_t) / SINGLE_DATA_BLOCK_SIZE),
            static_cast<uint16_t>(colPadExtNum * sizeof(int8_t) / SINGLE_DATA_BLOCK_SIZE), 0};
        DataCopy(yGM[rowStart * colNum], xLocalTmpInt8, copyParams);
    } else {
        DataCopyExtParams copyExtParams{
            static_cast<uint16_t>(rowCount), static_cast<uint16_t>(colNum * sizeof(int8_t)),
            static_cast<uint16_t>(colPadExtNum * sizeof(int8_t) / SINGLE_DATA_BLOCK_SIZE), 0, 0};
        DataCopyPad(yGM[rowStart * colNum], xLocalTmpInt8, copyExtParams);
    }
}

template <typename T>
__aicore__ inline void DynamicBlockQuantND<T>::CopyInTail(
    int64_t startRowIdx, int64_t colStartBlock, int64_t colEndBlock)
{
    int64_t copyNum = (colEndBlock - colStartBlock) * blockSizeCol;
    int64_t startIdx = startRowIdx * colNum + colStartBlock * blockSizeCol;
    if (colEndBlock == colBlockTotalNum) {
        copyNum -= colPadExtNum;
    }
    if (colEndBlock < colBlockTotalNum) {
        DataCopy(xLocal, xGM[startIdx], copyNum);
    } else if (colPadToBlockNum == 0) {
        DataCopy(xLocal, xGM[startIdx], copyNum);
        if (colClearExtNum > 0) {
            Duplicate(xLocal[copyNum], (T)0, colClearExtNum);
            PipeBarrier<PIPE_V>();
        }
    } else {
        DataCopyExtParams copyExtParams{1, static_cast<uint16_t>(copyNum * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{static_cast<uint16_t>(1), 0, static_cast<uint8_t>(colPadToBlockNum), 0};
        DataCopyPad(xLocal, xGM[startIdx], copyExtParams, padParams);
        if (colClearExtNum > 0) {
            Duplicate(xLocal[copyNum + colPadToBlockNum], (T)0, colClearExtNum);
            PipeBarrier<PIPE_V>();
        }
    }
}

template <typename T>
__aicore__ inline void DynamicBlockQuantND<T>::CopyOutYTail(
    int64_t startRowIdx, int64_t colStartBlock, int64_t colEndBlock)
{
    int64_t copyNum = (colEndBlock - colStartBlock) * blockSizeCol;
    int64_t startIdx = startRowIdx * colNum + colStartBlock * blockSizeCol;
    if (colEndBlock == colBlockTotalNum) {
        copyNum -= colPadExtNum;
    }
    if (colEndBlock < colBlockTotalNum || colPadExtNum * sizeof(int8_t) % SINGLE_DATA_BLOCK_SIZE == 0) {
        DataCopy(yGM[startIdx], xLocalTmpInt8, copyNum);
    } else {
        DataCopyExtParams copyExtParams{1, static_cast<uint16_t>(copyNum * sizeof(int8_t)), 0, 0, 0};
        DataCopyPad(yGM[startIdx], xLocalTmpInt8, copyExtParams);
    }
}

template <typename T>
__aicore__ inline void DynamicBlockQuantND<T>::CopyOutScale(int64_t startIdx, int64_t copyNum)
{
    SetFlag<HardEvent::V_MTE3>(eventIdVToMTE3);
    WaitFlag<HardEvent::V_MTE3>(eventIdVToMTE3);

    if (copyNum * sizeof(float) % SINGLE_DATA_BLOCK_SIZE == 0) {
        DataCopy(scaleGM[startIdx], scaleLocal, copyNum);
    } else {
        DataCopyExtParams copyExtParams{
            static_cast<uint16_t>(1), static_cast<uint16_t>(copyNum * sizeof(float)), 0, 0, 0};
        DataCopyPad(scaleGM[startIdx], scaleLocal, copyExtParams);
    }
}

template <typename T>
__aicore__ inline void DynamicBlockQuantND<T>::ParseTilingData(DynamicBlockQuantTilingData* tilingData)
{
    totalCoreNum = tilingData->totalCoreNum;
    usedCoreNum = tilingData->usedCoreNum;
    rowNum = tilingData->rowNum;
    colNum = tilingData->colNum;
    blockSizeRow = tilingData->blockSizeRow;
    blockSizeCol = tilingData->blockSizeCol;
    perCoreRowNum = tilingData->perCoreRowNum;

    tailRow = tilingData->tailRowList[coreIdx];
    tailColBlockStart = tilingData->tailColBlockStartList[coreIdx];
    tailColBlockEnd = tilingData->tailColBlockEndList[coreIdx];

    ubSize = tilingData->ubSize;
}
} // namespace DynamicBlockQuant
#endif
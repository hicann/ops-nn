/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file scatter_elements_v2.h
 * \brief
 */
#ifndef SCATTER_ELEMENTS_V2_H
#define SCATTER_ELEMENTS_V2_H
#include "kernel_operator.h"

#define IS_CAST_FLOAT (((is_same<T, half>::value) || (is_same<T, bfloat16_t>::value)) && MODE == 2)
#define IS_CAST_INT (is_same<U, int64_t>::value)
using namespace AscendC;

template <typename Tp, Tp v>
struct integral_constant {
    static constexpr Tp value = v;
};
using false_type = integral_constant<bool, false>;
using true_type = integral_constant<bool, true>;
template <typename, typename>
struct is_same : public false_type {
};
template <typename Tp>
struct is_same<Tp, Tp> : public true_type {
};

constexpr int INT32_OFFSET = 31;
constexpr uint32_t BUFFER_NUM = 1;
constexpr uint32_t SMALL_MODE = 1;

template <typename T, typename U, const uint32_t MODE>
class KernelScatterElementsV2
{
public:
    __aicore__ inline KernelScatterElementsV2()
    {}
    __aicore__ inline void Init(
        const ScatterElementsV2TilingData* __restrict tiling_data, TPipe* tmpPipe, GM_ADDR input, GM_ADDR indices,
        GM_ADDR updates)
    {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");

        pipe = tmpPipe;
        coreId = GetBlockIdx();
        usedCoreNum = tiling_data->usedCoreNum;
        eachNum = tiling_data->eachNum;
        uint32_t extraTaskCore = tiling_data->extraTaskCore;
        inputCount = tiling_data->inputCount;
        indicesCount = tiling_data->indicesCount;
        updatesCount = tiling_data->updatesCount;
        inputOneTime = tiling_data->inputOneTime;
        indicesOneTime = tiling_data->indicesOneTime;
        updatesOneTime = tiling_data->updatesOneTime;
        inputLoop = tiling_data->inputLoop;
        indicesLoop = tiling_data->indicesLoop;
        inputEach = tiling_data->inputEach;
        indicesEach = tiling_data->indicesEach;
        inputLast = tiling_data->inputLast;
        indicesLast = tiling_data->indicesLast;
        inputAlign = tiling_data->inputAlign;
        indicesAlign = tiling_data->indicesAlign;
        updatesAlign = tiling_data->updatesAlign;
        inputOnePiece = tiling_data->inputOnePiece;
        modeFlag = tiling_data->modeFlag;
        lastIndicesLoop = tiling_data->lastIndicesLoop;
        lastIndicesEach = tiling_data->lastIndicesEach;
        lastIndicesLast = tiling_data->lastIndicesLast;
        oneTime = tiling_data->oneTime;

        inputGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(input), inputCount);
        indicesGm.SetGlobalBuffer(reinterpret_cast<__gm__ U*>(indices), indicesCount);
        updatesGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(updates), updatesCount);

        if (modeFlag == SMALL_MODE) {
            indicesLoop = coreId == (usedCoreNum - 1) ? lastIndicesLoop : indicesLoop;
            indicesEach = coreId == (usedCoreNum - 1) ? lastIndicesEach : indicesEach;
            indicesLast = coreId == (usedCoreNum - 1) ? lastIndicesLast : indicesLast;

            inputAlign = (indicesEach * inputOneTime + dataAlign - 1) / dataAlign * dataAlign;
            indicesAlign = (indicesEach * indicesOneTime + dataAlign - 1) / dataAlign * dataAlign;
            updatesAlign = (indicesEach * updatesOneTime + dataAlign - 1) / dataAlign * dataAlign;

            pipe->InitBuffer(inQueueSelf, BUFFER_NUM, inputAlign * sizeof(T));
            if constexpr (IS_CAST_INT) {
                pipe->InitBuffer(inQueueIndics, BUFFER_NUM, indicesAlign * sizeof(U));
            }
            pipe->InitBuffer(inQueueUpdates, BUFFER_NUM, updatesAlign * sizeof(T));
            if constexpr (IS_CAST_FLOAT) {
                pipe->InitBuffer(calcSelfBuf, inputAlign * sizeof(float));
                pipe->InitBuffer(calcUpdatesBuf, updatesAlign * sizeof(float));
                inputTemp = calcSelfBuf.Get<float>();
                updatesTemp = calcUpdatesBuf.Get<float>();
            }
        } else {
            pieceEach = inputEach;
            pieceLast = inputLast;
            if (eachNum == 0) {
                uint32_t eachPiece = tiling_data->eachPiece;
                start = coreId / eachPiece;
                currentPiece = coreId % eachPiece;
                currentNum = 1;
                if (currentPiece == eachPiece - 1) {
                    auto tmpOnePiece = inputOneTime - inputOnePiece * (eachPiece - 1);
                    pieceEach = (tmpOnePiece + inputLoop - 1) / inputLoop;
                    pieceLast = tmpOnePiece - pieceEach * (inputLoop - 1);
                }
            } else {
                currentPiece = 0;
                currentNum = coreId < extraTaskCore ? (eachNum + 1) : eachNum;
                start = coreId * eachNum + (coreId < extraTaskCore ? coreId : extraTaskCore);
            }

            if constexpr (IS_CAST_FLOAT) {
                pipe->InitBuffer(calcSelfBuf, inputAlign * sizeof(float));
                pipe->InitBuffer(calcUpdatesBuf, updatesAlign * sizeof(float));
                inputTemp = calcSelfBuf.Get<float>();
                updatesTemp = calcUpdatesBuf.Get<float>();
            }

            pipe->InitBuffer(inQueueSelf, BUFFER_NUM, inputAlign * sizeof(T));
            if constexpr (IS_CAST_INT) {
                pipe->InitBuffer(inQueueIndics, BUFFER_NUM, indicesAlign * sizeof(U));
            }
            pipe->InitBuffer(inQueueUpdates, BUFFER_NUM, updatesAlign * sizeof(T));
        }

        pipe->InitBuffer(calcIndices32Buf, indicesAlign * sizeof(int));
        indices32Local = calcIndices32Buf.Get<int>();
        inputLocal = inQueueSelf.AllocTensor<T>();
        if constexpr (IS_CAST_INT) {
            indicesLocal = inQueueIndics.AllocTensor<U>();
        }
        updatesLocal = inQueueUpdates.AllocTensor<T>();

        padParams = {false, 0, 0, 0};
        tPadParams = {false, 0, 0, static_cast<T>(0)};
        uPadParams = {false, 0, 0, static_cast<U>(0)};
    }

    __aicore__ inline void CopyInIndex(int indicesIndex)
    {
        if constexpr (IS_CAST_INT) {
            DataCopyPadGm2UBImpl(
                (__ubuf__ uint32_t*)indicesLocal.GetPhyAddr(), (__gm__ uint32_t*)indicesGm[indicesIndex].GetPhyAddr(),
                indicesExtParams, padParams); // datacopypad int64
        } else {
            DataCopyPad(indices32Local, indicesGm[indicesIndex], indicesExtParams, uPadParams);
        }
    }

    __aicore__ inline void ScatterSetValue(int k, int kIndex)
    {
        if constexpr (MODE == 1) {
            inputLocal.SetValue(kIndex, updatesLocal.GetValue(k));
        } else if constexpr (IS_CAST_FLOAT) {
            inputTemp.SetValue(kIndex, inputTemp.GetValue(kIndex) + updatesTemp.GetValue(k));
        } else {
            inputLocal.SetValue(kIndex, inputLocal.GetValue(kIndex) + updatesLocal.GetValue(k));
        }
    }

    __aicore__ inline void ProcessSmall()
    {
        for (uint64_t index = 0; index < indicesLoop; ++index) {
            uint64_t baseIndex = coreId * oneTime + index * indicesEach;
            uint64_t indicesIndex = baseIndex * indicesOneTime;
            uint64_t inputIndex = baseIndex * inputOneTime;
            uint64_t updatesIndex = baseIndex * updatesOneTime;
            uint64_t currentIndices = indicesEach;
            if (index == indicesLoop - 1) {
                currentIndices = indicesLast;
            }
            inputExtParams = {(uint16_t)1, static_cast<uint32_t>(currentIndices * inputOneTime * sizeof(T)), 0, 0, 0};
            indicesExtParams = {
                (uint16_t)1, static_cast<uint32_t>(currentIndices * indicesOneTime * sizeof(U)), 0, 0, 0};
            updatesExtParams = {
                (uint16_t)1, static_cast<uint32_t>(currentIndices * updatesOneTime * sizeof(T)), 0, 0, 0};
            PIPE_MTE3_MTE2();
            CopyInIndex(indicesIndex);
            DataCopyPad(updatesLocal, updatesGm[updatesIndex], updatesExtParams, tPadParams);
            DataCopyPad(inputLocal, inputGm[inputIndex], inputExtParams, tPadParams);
            if constexpr (IS_CAST_FLOAT) {
                PIPE_MTE2_V();
                Cast(inputTemp, inputLocal, RoundMode::CAST_NONE, inputAlign);
                Cast(updatesTemp, updatesLocal, RoundMode::CAST_NONE, indicesAlign);
            }
            if constexpr (IS_CAST_INT) {
                PIPE_MTE2_V();
                Cast<int, U>(indices32Local, indicesLocal, RoundMode::CAST_NONE, indicesAlign);
            }
            PipeBarrier<PIPE_ALL>();
            for (uint64_t j = 0; j < currentIndices; ++j) {
                for (uint64_t k = 0; k < indicesOneTime; ++k) {
                    auto upIndex = j * updatesOneTime + k;
                    auto inIndex = j * inputOneTime + indices32Local.GetValue(j * indicesOneTime + k);
                    ScatterSetValue(upIndex, inIndex);
                }
            }
            PipeBarrier<PIPE_ALL>();
            if constexpr (IS_CAST_FLOAT) {
                Cast(inputLocal, inputTemp, RoundMode::CAST_RINT, inputAlign);
                PIPE_V_MTE3();
            }
            DataCopyPad(inputGm[inputIndex], inputLocal, inputExtParams);
        }
        inQueueSelf.FreeTensor(inputLocal);
        if constexpr (IS_CAST_INT) {
            inQueueIndics.FreeTensor(indicesLocal);
        }
        inQueueUpdates.FreeTensor(updatesLocal);
    }

    __aicore__ inline void ProcessScatter()
    {
        for (uint64_t index = start; index < start + currentNum; ++index) {
            uint64_t inputIndex = index * inputOneTime + currentPiece * inputOnePiece;
            uint64_t indicesIndex = index * indicesOneTime;
            uint64_t updatesIndex = index * updatesOneTime;

            for (uint64_t i = 0; i < inputLoop; ++i) {
                PIPE_MTE3_MTE2();
                uint64_t currentInput = pieceEach;
                if (i == inputLoop - 1) {
                    currentInput = pieceLast;
                }
                inputExtParams = {(uint16_t)1, static_cast<uint32_t>(currentInput * sizeof(T)), 0, 0, 0};
                DataCopyPad(inputLocal, inputGm[inputIndex + i * pieceEach], inputExtParams, tPadParams);

                if constexpr (IS_CAST_FLOAT) {
                    PIPE_MTE2_V();
                    Cast(inputTemp, inputLocal, RoundMode::CAST_NONE, inputAlign);
                }

                for (uint64_t j = 0; j < indicesLoop; ++j) {
                    uint64_t currentIndices = indicesEach;
                    if (j == indicesLoop - 1) {
                        currentIndices = indicesLast;
                    }
                    indicesExtParams = {(uint16_t)1, static_cast<uint32_t>(currentIndices * sizeof(U)), 0, 0, 0};
                    updatesExtParams = {(uint16_t)1, static_cast<uint32_t>(currentIndices * sizeof(T)), 0, 0, 0};
                    CopyInIndex(indicesIndex + j * indicesEach);
                    DataCopyPad(updatesLocal, updatesGm[updatesIndex + j * indicesEach], updatesExtParams, tPadParams);
                    PIPE_MTE2_V();
                    if constexpr (IS_CAST_INT) {
                        Cast<int, U>(indices32Local, indicesLocal, RoundMode::CAST_NONE, indicesAlign);
                        PipeBarrier<PIPE_V>();
                    }
                    if constexpr (IS_CAST_FLOAT) {
                        Cast(updatesTemp, updatesLocal, RoundMode::CAST_NONE, updatesAlign);
                    }
                    Adds(
                        indices32Local, indices32Local, static_cast<int>(-i * pieceEach - currentPiece * inputOnePiece),
                        static_cast<int>(indicesAlign));
                    PIPE_V_S();
                    for (uint64_t k = 0; k < currentIndices; ++k) {
                        auto kIndex = indices32Local.GetValue(k);
                        if (kIndex < 0 || kIndex >= currentInput) {
                            continue;
                        }
                        ScatterSetValue(k, kIndex);
                    }
                }
                PipeBarrier<PIPE_ALL>();
                if constexpr (IS_CAST_FLOAT) {
                    Cast(inputLocal, inputTemp, RoundMode::CAST_RINT, inputAlign);
                    PIPE_V_MTE3();
                }
                DataCopyPad(inputGm[inputIndex + i * pieceEach], inputLocal, inputExtParams);
            }
        }
        inQueueSelf.FreeTensor(inputLocal);
        if constexpr (IS_CAST_INT) {
            inQueueIndics.FreeTensor(indicesLocal);
        }
        inQueueUpdates.FreeTensor(updatesLocal);
    }

    __aicore__ inline void PIPE_MTE3_MTE2() {
        int32_t eventIDMTE3ToMTE2 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
        WaitFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
    }

    __aicore__ inline void PIPE_MTE2_V() {
        int32_t eventIDMTE2ToV = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
    }

    __aicore__ inline void PIPE_V_MTE3() {
        int32_t eventIDVToMTE3 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
        WaitFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
    }

    __aicore__ inline void PIPE_V_S() {
        int32_t eventIDVToS = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventIDVToS);
        WaitFlag<HardEvent::V_S>(eventIDVToS);
    }

private:
    TPipe* pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueSelf, inQueueIndics, inQueueUpdates;
    TBuf<QuePosition::VECCALC> calcSelfBuf, calcUpdatesBuf, calcIndices32Buf;
    GlobalTensor<T> inputGm, updatesGm;
    GlobalTensor<U> indicesGm;
    LocalTensor<float> inputTemp, updatesTemp;
    LocalTensor<int> indicesTemp, indices32Local;
    LocalTensor<U> indicesLocal;
    LocalTensor<T> inputLocal, updatesLocal;
    DataCopyPadExtParams<uint32_t> padParams;
    DataCopyPadExtParams<T> tPadParams;
    DataCopyPadExtParams<U> uPadParams;
    DataCopyExtParams inputExtParams, indicesExtParams, updatesExtParams;
    uint32_t coreId;
    uint64_t usedCoreNum;
    uint64_t modeFlag;
    uint64_t currentNum;
    uint64_t eachNum;
    uint64_t start;
    uint64_t inputAlign;
    uint64_t indicesAlign;
    uint64_t updatesAlign;
    uint64_t inputCount;
    uint64_t indicesCount;
    uint64_t updatesCount;
    uint64_t inputOneTime;
    uint64_t indicesOneTime;
    uint64_t updatesOneTime;
    uint64_t inputLoop;
    uint64_t indicesLoop;
    uint64_t inputEach;
    uint64_t indicesEach;
    uint64_t inputLast;
    uint64_t indicesLast;
    uint64_t currentPiece;
    uint64_t inputOnePiece;
    uint64_t pieceEach;
    uint64_t pieceLast;
    uint64_t lastIndicesLoop;
    uint64_t lastIndicesEach;
    uint64_t lastIndicesLast;
    uint64_t oneTime;
    uint32_t dataAlign = 32;
};
#endif // SCATTER_ELEMENTS_V2_H

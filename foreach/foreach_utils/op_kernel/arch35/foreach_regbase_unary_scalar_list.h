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
 * \file foreach_regbase_unary_scalar_list.h
 * \brief
 */
#ifndef FOREACH_REGBASE_UNARY_SCALAR_LIST_H
#define FOREACH_REGBASE_UNARY_SCALAR_LIST_H

#include "foreach_regbase_common.h"

using namespace AscendC;

template <typename T, typename Tiling, typename Predicate>
class ForeachRegbaseUnaryScalarList
{
public:
    __aicore__ inline ForeachRegbaseUnaryScalarList(Predicate& p) : pred(p){};
    __aicore__ inline void Init(
        GM_ADDR inputs, GM_ADDR outputs, GM_ADDR workspace, const Tiling* tilingData, TPipe* tPipe)
    {
        blockIdx = GetBlockIdx();
        inDesc = ListTensorDesc((__gm__ void*)inputs);
        outDesc = ListTensorDesc((__gm__ void*)outputs);
        ParseTilingData(tilingData);
        tPipe->InitBuffer(inQueue, BUFFER_NUM, inputsTensorUbSize * sizeof(T));
        tPipe->InitBuffer(outQueue, BUFFER_NUM, inputsTensorUbSize * sizeof(T));
        maxDataCount = inputsTensorUbSize;
    }

    __aicore__ inline void Process()
    {
        for (uint16_t i = tensorStart; i <= tensorEnd; i++) {
            int64_t cursorStart = 0;
            int64_t cursorEnd = tensorDataCountList[i] - 1;
            int64_t dataCount = 0;
            if (i == tensorStart) {
                cursorStart = tensorStartOffset;
            }
            if (i == tensorEnd) {
                cursorEnd = tensorEndOffset;
            }
            dataCount = cursorEnd - cursorStart + 1;
            inTensorGM.SetGlobalBuffer(GetInputTensorAddr(i) + cursorStart);
            outTensorGM.SetGlobalBuffer(GetOutputTensorAddr(i) + cursorStart);
            SingleTensorProcess(i, dataCount);
        }
    }

    template <typename T1, typename T2>
    __aicore__ inline T1 CEIL_DIV(T1 a, T2 b)
    {
        return (a + b - 1) / b;
    };

    __aicore__ inline void ParseTilingData(const Tiling* tilingData)
    {
        inputsTensorUbSize = tilingData->inputsTensorUbSize;
        tensorDataCountList = (int64_t*)tilingData->tensorDataCountList;
        tensorStart = tilingData->tensorStartList[blockIdx];
        tensorEnd = tilingData->tensorEndList[blockIdx];
        tensorStartOffset = tilingData->tensorStartOffsetList[blockIdx];
        tensorEndOffset = tilingData->tensorEndOffsetList[blockIdx];
    }
    __aicore__ inline void SingleTensorProcess(int64_t tensorIndex, int64_t dataCount)
    {
        // Batch handling and calculation.
        int64_t quotient = CEIL_DIV(dataCount, maxDataCount);
        for (int64_t i = 0; i < quotient; i++) {
            int64_t currentDataCount =
                (i == (quotient - 1)) ? (dataCount - (quotient - 1) * maxDataCount) : maxDataCount;
            CopyIn(i, currentDataCount);
            Compute(tensorIndex, currentDataCount);
            CopyOut(i, currentDataCount);
        }
    }

    __aicore__ inline void CopyIn(int64_t index, int64_t dataCount)
    {
        LocalTensor<T> dataLocal = inQueue.AllocTensor<T>();
        DataCopyPadExtParams<T> dataCopyPadExtParams;
        dataCopyPadExtParams.isPad = false;
        dataCopyPadExtParams.leftPadding = 0;
        dataCopyPadExtParams.rightPadding = 0;
        dataCopyPadExtParams.paddingValue = 0;
        DataCopyExtParams copyInParams;
        copyInParams.blockCount = 1;
        copyInParams.blockLen = dataCount * sizeof(T);
        copyInParams.srcStride = 0;
        copyInParams.dstStride = 0;
        DataCopyPad(dataLocal, inTensorGM[index * maxDataCount], copyInParams, dataCopyPadExtParams);
        inQueue.EnQue(dataLocal);
    }

    __aicore__ inline void Compute(int64_t tensorIndex, int64_t dataCount)
    {
        LocalTensor<T> inLocal = inQueue.template DeQue<T>();
        LocalTensor<T> outLocal = outQueue.template AllocTensor<T>();
        pred.Compute(inLocal, outLocal, tensorIndex, dataCount);
        inQueue.FreeTensor(inLocal);
        outQueue.template EnQue<T>(outLocal);
    }

    __aicore__ inline void CopyOut(int64_t index, int64_t dataCount)
    {
        LocalTensor<T> retLocal = outQueue.DeQue<T>();
        DataCopyExtParams copyInParams;
        copyInParams.blockCount = 1;
        copyInParams.blockLen = dataCount * sizeof(T);
        copyInParams.srcStride = 0;
        copyInParams.dstStride = 0;
        DataCopyPad(outTensorGM[index * maxDataCount], retLocal, copyInParams);
        outQueue.FreeTensor(retLocal);
    }

    __aicore__ inline __gm__ T* GetInputTensorAddr(uint16_t index)
    {
        return (__gm__ T*)inDesc.GetDataPtr<__gm__ T>(index);
    }

    __aicore__ inline __gm__ T* GetOutputTensorAddr(uint16_t index)
    {
        return (__gm__ T*)outDesc.GetDataPtr<__gm__ T>(index);
    }

protected:
    static constexpr int32_t BUFFER_NUM = 2;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue;

    GlobalTensor<T> inTensorGM;
    GlobalTensor<T> outTensorGM;

    int64_t blockIdx = 0;

    uint32_t maxDataCount = {0};
    // tiling params
    uint64_t inputsTensorUbSize = 0;
    int64_t* tensorDataCountList = nullptr;
    uint16_t tensorStart = {0};
    uint16_t tensorEnd = {0};
    int64_t tensorStartOffset = {0};
    int64_t tensorEndOffset = {0};

    ListTensorDesc inDesc;
    ListTensorDesc outDesc;

private:
    Predicate& pred;
};

#endif // FOREACH_REGBASE_UNARY_SCALAR_LIST_H
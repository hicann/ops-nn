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
 * \file scatter_add_int_with_sorted.h
 * \brief
 */
#ifndef SCATTER_ADD_INT_WITH_SORTED_H
#define SCATTER_ADD_INT_WITH_SORTED_H
#include "kernel_operator.h"
using namespace AscendC;

template <typename T>
class KernelScatterAddIntWithSorted
{
public:
    __aicore__ inline KernelScatterAddIntWithSorted()
    {}

    __aicore__ inline void Init(
        const ScatterAddWithSortedTilingData* __restrict tiling_data, TPipe* tmpPipe, GM_ADDR var, GM_ADDR updates,
        GM_ADDR indices, GM_ADDR output)
    {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");

        pipe = tmpPipe;
        coreId = GetBlockIdx();
        usedCoreNum = tiling_data->usedCoreNum;
        eachNum = tiling_data->eachNum;
        inputCount = tiling_data->inputCount;
        indicesCount = tiling_data->indicesCount;
        updatesCount = tiling_data->updatesCount;
        inputOneTime = tiling_data->inputOneTime;
        updatesOneTime = tiling_data->updatesOneTime;
        updatesLoop = tiling_data->updatesLoop;
        updatesEach = tiling_data->updatesEach;
        updatesLast = tiling_data->updatesLast;
        updatesAlign = tiling_data->updatesAlign;

        uint32_t extraTaskCore = tiling_data->extraTaskCore;
        currentNum = coreId < extraTaskCore ? (eachNum + 1) : eachNum;
        start = coreId * eachNum + (coreId < extraTaskCore ? coreId : extraTaskCore);

        inputGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(var), inputCount);
        indicesGm.SetGlobalBuffer(reinterpret_cast<__gm__ int*>(indices), indicesCount);
        updatesGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(updates), updatesCount);

        pipe->InitBuffer(inQueueIndics, BUFFER_NUM, dataAlign * sizeof(int));
        pipe->InitBuffer(inQueueUpdates, BUFFER_NUM, updatesAlign * sizeof(T));

        updatesLocal = inQueueUpdates.AllocTensor<T>();
        indicesLocal = inQueueIndics.AllocTensor<int>();
        indicesExtParams = {(uint16_t)1, static_cast<uint32_t>(sizeof(int)), 0, 0, 0};
        tPadParams = {false, 0, 0, static_cast<T>(0)};
        uPadParams = {false, 0, 0, static_cast<int>(0)};
    }

    __aicore__ inline void Process()
    {
        SetAtomicAdd<T>();
        for (uint64_t index = start; index < start + currentNum; ++index) {
            uint64_t indicesIndex = index;
            uint64_t updatesIndex = index * updatesOneTime;
            DataCopyPad(indicesLocal, indicesGm[indicesIndex], indicesExtParams, uPadParams);
            PIPE_MTE2_S();

            int iIndex = indicesLocal.GetValue(0); // get indicesLocal[0]
            for (uint64_t i = 0; i < updatesLoop; ++i) {
                uint64_t currentUpdates = updatesEach;
                if (i == updatesLoop - 1) {
                    currentUpdates = updatesLast;
                }
                PIPE_MTE3_MTE2();
                updatesExtParams = {(uint16_t)1, static_cast<uint32_t>(currentUpdates * sizeof(T)), 0, 0, 0};
                DataCopyPad(updatesLocal, updatesGm[updatesIndex + i * updatesEach], updatesExtParams, tPadParams);
                PIPE_MTE2_MTE3();
                DataCopyPad(inputGm[iIndex * inputOneTime + i * updatesEach], updatesLocal, updatesExtParams);
            }
        }
        SetAtomicNone();
        inQueueIndics.FreeTensor(indicesLocal);
        inQueueUpdates.FreeTensor(updatesLocal);
    }

    __aicore__ inline void PIPE_MTE3_MTE2() {
        int32_t eventIDMTE3ToMTE2 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
        WaitFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
    }

    __aicore__ inline void PIPE_MTE2_S() {
        int32_t eventIDMTE2ToS = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
        SetFlag<HardEvent::MTE2_S>(eventIDMTE2ToS);
        WaitFlag<HardEvent::MTE2_S>(eventIDMTE2ToS);
    }

    __aicore__ inline void PIPE_MTE2_MTE3() {
        int32_t eventIDMTE2ToMTE3 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3));
        SetFlag<HardEvent::MTE2_MTE3>(eventIDMTE2ToMTE3);
        WaitFlag<HardEvent::MTE2_MTE3>(eventIDMTE2ToMTE3);
    }

private:
    TPipe* pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueIndics, inQueueUpdates;
    GlobalTensor<T> inputGm, updatesGm;
    GlobalTensor<int> indicesGm;
    LocalTensor<T> updatesLocal;
    LocalTensor<int> indicesLocal;
    DataCopyExtParams indicesExtParams, updatesExtParams;
    DataCopyPadExtParams<T> tPadParams;
    DataCopyPadExtParams<int> uPadParams;
    uint32_t coreId;
    uint64_t usedCoreNum;
    uint64_t currentNum;
    uint64_t start;
    uint64_t eachNum;
    uint64_t lastNum;
    uint64_t inputCount;
    uint64_t indicesCount;
    uint64_t updatesCount;
    uint64_t inputOneTime;
    uint64_t updatesOneTime;
    uint64_t updatesLoop;
    uint64_t updatesEach;
    uint64_t updatesLast;
    uint64_t updatesAlign;
    uint64_t dataAlign = 32;
};

#endif // SCATTER_ADD_INT_WITH_SORTED_H

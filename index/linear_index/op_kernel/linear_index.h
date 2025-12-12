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
 * \file linear_index.h
 * \brief
 */
#ifndef LINEAR_INDEX_H
#define LINEAR_INDEX_H
#include "kernel_operator.h"
#define IS_CAST_INT (isSame<T, int64_t>::value)
using namespace AscendC;

template <typename Tp, Tp v>
struct integralConstant {
    static constexpr Tp value = v;
};
using true_type = integralConstant<bool, true>;
using false_type = integralConstant<bool, false>;
template <typename, typename>
struct isSame : public false_type {
};
template <typename Tp>
struct isSame<Tp, Tp> : public true_type {
};

constexpr uint32_t BUFFER_NUM = 1;
constexpr int INT32_OFFSET = 31;

template <typename T, const uint32_t MODE>
class KernelLinearIndex
{
public:
    __aicore__ inline KernelLinearIndex()
    {}
    __aicore__ inline void Init(
        const LinearIndexTilingData* __restrict tiling_data, TPipe* tmpPipe, GM_ADDR indices, GM_ADDR output)
    {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");

        pipe = tmpPipe;
        coreId = GetBlockIdx();
        usedCoreNum = tiling_data->usedCoreNum;
        indicesCount = tiling_data->indicesCount;
        indicesAlign = tiling_data->indicesAlign;
        eachCount = tiling_data->eachCount;
        lastCount = tiling_data->lastCount;
        eachNum = tiling_data->eachNum;
        eachLoop = tiling_data->eachLoop;
        eachTail = tiling_data->eachTail;
        lastNum = tiling_data->lastNum;
        lastLoop = tiling_data->lastLoop;
        lastTail = tiling_data->lastTail;
        target = tiling_data->target;
        selfStride = static_cast<int>(tiling_data->selfStride);
        indicesStride = static_cast<int>(tiling_data->indicesStride);

        currentEach = coreId == (usedCoreNum - 1) ? lastNum : eachNum;
        currentLoop = coreId == (usedCoreNum - 1) ? lastLoop : eachLoop;
        currentTail = coreId == (usedCoreNum - 1) ? lastTail : eachTail;

        indicesGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(indices), indicesCount);
        outputGm.SetGlobalBuffer(reinterpret_cast<__gm__ int*>(output), indicesCount);

        pipe->InitBuffer(calcIndices32Buf, indicesAlign * sizeof(int));
        pipe->InitBuffer(calcIndicesBuf, indicesAlign * sizeof(int));

        if constexpr (IS_CAST_INT) {
            pipe->InitBuffer(inQueueIndics, BUFFER_NUM, indicesAlign * sizeof(T));
            indicesLocal = inQueueIndics.AllocTensor<T>();
        }
        indices32Local = calcIndices32Buf.Get<int>();
        indicesTemp = calcIndicesBuf.Get<int>();

        if constexpr (MODE == 1) {
            pipe->InitBuffer(arangeBuffer, indicesAlign * sizeof(float));
            pipe->InitBuffer(arangeIntBuffer, indicesAlign * sizeof(int));
            arangeLocal = arangeBuffer.Get<float>();
            arangeIntLocal = arangeIntBuffer.Get<int>();
        }
        if constexpr (MODE == 2) {
            pipe->InitBuffer(arangeBuffer, indicesAlign * sizeof(float));
            arangeLocal = arangeBuffer.Get<float>();
        }

        padParams = {false, 0, 0, 0};
        tPadParams = {false, 0, 0, static_cast<T>(0)};
    }

    __aicore__ inline void Process()
    {
        for (size_t i = 0; i < currentLoop; ++i) {
            uint64_t indicesOffset = eachCount * coreId + i * currentEach;
            int offset = static_cast<int>(indicesOffset);
            currentNum = currentEach;
            if (i == currentLoop - 1) {
                currentNum = currentTail;
            }
            indicesExtParams = {(uint16_t)1, static_cast<uint32_t>(currentNum * sizeof(T)), 0, 0, 0};
            outExtParams = {(uint16_t)1, static_cast<uint32_t>(currentNum * sizeof(int)), 0, 0, 0};
            int32_t eventIDMTE3ToMTE2 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
            SetFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
            WaitFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
            if constexpr (IS_CAST_INT) {
                DataCopyPadGm2UBImpl(
                    (__ubuf__ uint32_t*)indicesLocal.GetPhyAddr(),
                    (__gm__ uint32_t*)indicesGm[indicesOffset].GetPhyAddr(), indicesExtParams,
                    padParams); // datacopypad int64
            } else {
                DataCopyPad(indices32Local, indicesGm[indicesOffset], indicesExtParams, tPadParams);
            }
            int32_t eventIDMTE2ToV = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
            WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
            if constexpr (IS_CAST_INT) {
                Cast<int, T>(indices32Local, indicesLocal, RoundMode::CAST_NONE, indicesAlign);
                PipeBarrier<PIPE_V>();
            }
            // 以下代码用于负数索引转正数
            // 左移31位，负数索引结果为-1，正数索引结果为0
            ShiftRight(indicesTemp, indices32Local, INT32_OFFSET, static_cast<int>(indicesAlign));
            PipeBarrier<PIPE_V>();
            // 乘以边界值
            Muls(indicesTemp, indicesTemp, static_cast<int>(target), static_cast<int>(indicesAlign));
            PipeBarrier<PIPE_V>();
            // 负数索引加上边界值
            Sub(indices32Local, indices32Local, indicesTemp, static_cast<int>(indicesAlign));

            // 以下代码用于合轴
            if constexpr (MODE == 1) { // 二维dim = 0 场景
                // 构造一个{indicesOffset, indicesOffset+1, indicesOffset+2, ……}的tensor
                ArithProgression(arangeLocal, static_cast<float>(offset), static_cast<float>(1), indicesAlign);
                PipeBarrier<PIPE_V>();
                Cast(arangeIntLocal, arangeLocal, RoundMode::CAST_FLOOR, indicesAlign);
                // 计算除以indicesStride的余数
                Muls(arangeLocal, arangeLocal, 1 / static_cast<float>(indicesStride), indicesAlign);
                PipeBarrier<PIPE_V>();
                Cast(indicesTemp, arangeLocal, RoundMode::CAST_FLOOR, indicesAlign);
                PipeBarrier<PIPE_V>();
                Muls(indicesTemp, indicesTemp, static_cast<int>(indicesStride), indicesAlign);
                PipeBarrier<PIPE_V>();
                Sub(indicesTemp, arangeIntLocal, indicesTemp, indicesAlign);
                // indices32Local = indices32Local * selfStride + 余数
                Muls(indices32Local, indices32Local, selfStride, indicesAlign);
                PipeBarrier<PIPE_V>();
                Add(indices32Local, indices32Local, indicesTemp, indicesAlign);
            }
            if constexpr (MODE == 2) { // 二维dim = 1 场景
                // 构造一个{indicesOffset, indicesOffset+1, indicesOffset+2, ……}的tensor
                ArithProgression(arangeLocal, static_cast<float>(offset), static_cast<float>(1), indicesAlign);
                PipeBarrier<PIPE_V>();
                // 计算除以indicesStride的商
                Muls(arangeLocal, arangeLocal, 1 / static_cast<float>(indicesStride), indicesAlign);
                PipeBarrier<PIPE_V>();
                Cast(indicesTemp, arangeLocal, RoundMode::CAST_FLOOR, indicesAlign);
                PipeBarrier<PIPE_V>();
                // indices32Local = indices32Local  + selfStride * 商
                Muls(indicesTemp, indicesTemp, selfStride, indicesAlign);
                PipeBarrier<PIPE_V>();
                Add(indices32Local, indicesTemp, indices32Local, indicesAlign);
            }
            int32_t eventIDVToMTE3 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
            SetFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
            WaitFlag<HardEvent::V_MTE3>(eventIDVToMTE3);

            DataCopyPad(outputGm[indicesOffset], indices32Local, outExtParams);
        }
        if constexpr (IS_CAST_INT) {
            inQueueIndics.FreeTensor(indicesLocal);
        }
    }

private:
    TPipe* pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueIndics;
    TBuf<QuePosition::VECCALC> calcIndices32Buf, calcIndicesBuf, arangeBuffer, arangeIntBuffer;
    GlobalTensor<T> indicesGm;
    GlobalTensor<int> outputGm;
    LocalTensor<T> indicesLocal;
    LocalTensor<int> indices32Local, indicesTemp, arangeIntLocal;
    LocalTensor<float> arangeLocal;
    DataCopyExtParams indicesExtParams, outExtParams;
    DataCopyPadExtParams<T> tPadParams;
    DataCopyPadExtParams<uint32_t> padParams;
    uint32_t coreId;
    uint64_t usedCoreNum;
    uint64_t currentNum;
    uint64_t currentEach;
    uint64_t currentLoop;
    uint64_t currentTail;
    uint64_t indicesCount;
    uint64_t indicesAlign;
    uint64_t eachCount;
    uint64_t lastCount;
    uint64_t eachNum;
    uint64_t eachLoop;
    uint64_t eachTail;
    uint64_t lastNum;
    uint64_t lastLoop;
    uint64_t lastTail;
    uint64_t target;
    int selfStride;
    int indicesStride;
    uint32_t dataAlign = 32;
};

#endif // LINEAR_INDEX_H

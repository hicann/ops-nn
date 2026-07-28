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
 * \file sigmoid.h
 * \brief
 */
#ifndef SIGMOID_H
#define SIGMOID_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "sigmoid_tiling_data.h"
#include "sigmoid_tiling_key.h"

namespace MySigmoid {

using namespace AscendC;

template <typename TYPE_X, uint64_t BUFFER_NUM>
class KernelSigmoid {
public:
    __aicore__ inline KernelSigmoid(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const SigmoidTilingData* tilingData, TPipe* pipeIn);
    __aicore__ inline void Process();

private:
    __aicore__ inline void Compute(uint64_t offset);

private:
    TPipe* pipe;
    TQueBind<TPosition::VECIN, TPosition::VECOUT, BUFFER_NUM> queBind;
    TBuf<QuePosition::VECCALC> tmpCommon;

    GlobalTensor<TYPE_X> xGm;
    GlobalTensor<TYPE_X> yGm;
    uint64_t coreDataNum;
    uint64_t tileNum;
    uint64_t tileDataNum;
    uint64_t tailDataNum;
    uint64_t processDataNum;
};

template <typename TYPE_X, uint64_t BUFFER_NUM>
__aicore__ inline void KernelSigmoid<TYPE_X, BUFFER_NUM>::Init(GM_ADDR x, GM_ADDR y,
                                                               const SigmoidTilingData* tilingData, TPipe* pipeIn)
{
    ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");

    uint64_t coreId = GetBlockIdx();
    uint64_t globalBufferIndex = tilingData->bigCoreDataNum * coreId;
    this->tileDataNum = tilingData->tileDataNum;

    if (coreId < tilingData->tailBlockNum) {
        this->coreDataNum = tilingData->bigCoreDataNum;
        this->tileNum = tilingData->finalBigTileNum;
        this->tailDataNum = tilingData->bigTailDataNum;
    } else {
        this->coreDataNum = tilingData->smallCoreDataNum;
        this->tileNum = tilingData->finalSmallTileNum;
        this->tailDataNum = tilingData->smallTailDataNum;
        globalBufferIndex -= (tilingData->bigCoreDataNum - tilingData->smallCoreDataNum) *
                             (coreId - tilingData->tailBlockNum);
    }

    xGm.SetGlobalBuffer((__gm__ TYPE_X*)x + globalBufferIndex, this->coreDataNum);
    yGm.SetGlobalBuffer((__gm__ TYPE_X*)y + globalBufferIndex, this->coreDataNum);

    this->pipe = pipeIn;
    pipe->InitBuffer(queBind, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_X) * 2);

    if constexpr (std::is_same_v<TYPE_X, bfloat16_t> || std::is_same_v<TYPE_X, half> ||
                  std::is_same_v<TYPE_X, int16_t>) {
        pipe->InitBuffer(tmpCommon, tileDataNum * sizeof(float));
        LocalTensor<float> oneLocal = tmpCommon.Get<float>();
        Duplicate(oneLocal, 1.0f, tileDataNum);
    } else if constexpr (std::is_same_v<TYPE_X, float>) {
        pipe->InitBuffer(tmpCommon, tileDataNum * sizeof(float));
        LocalTensor<float> oneLocal = tmpCommon.Get<float>();
        Duplicate(oneLocal, 1.0f, tileDataNum);
    } else if constexpr (std::is_same_v<TYPE_X, int8_t> || std::is_same_v<TYPE_X, uint8_t>) {
        pipe->InitBuffer(tmpCommon, tileDataNum * sizeof(half));
        LocalTensor<half> oneLocal = tmpCommon.Get<half>();
        Duplicate(oneLocal, static_cast<half>(1.0f), tileDataNum);
    }
}

template <typename TYPE_X, uint64_t BUFFER_NUM>
__aicore__ inline void KernelSigmoid<TYPE_X, BUFFER_NUM>::Compute(uint64_t offset)
{
    LocalTensor<TYPE_X> yLocal = queBind.template AllocTensor<TYPE_X>();
    LocalTensor<TYPE_X> xLocal = yLocal[this->processDataNum];
    int32_t eventIDMTE2ToV = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    int32_t eventIDVToMTE3 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));

    DataCopy(xLocal, xGm[offset], this->processDataNum);
    SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);

    if constexpr (std::is_same_v<TYPE_X, bfloat16_t> || std::is_same_v<TYPE_X, half> ||
                  std::is_same_v<TYPE_X, int16_t>) {
        LocalTensor<float> computeTensor = yLocal.template ReinterpretCast<float>();
        LocalTensor<float> oneLocal = tmpCommon.Get<float>();

        Cast(computeTensor, xLocal, RoundMode::CAST_NONE, processDataNum);
        Muls(computeTensor, computeTensor, -1.0f, processDataNum);
        Exp(computeTensor, computeTensor, processDataNum);
        Adds(computeTensor, computeTensor, 1.0f, processDataNum);
        Div(computeTensor, oneLocal, computeTensor, processDataNum);
        Cast(yLocal, computeTensor, RoundMode::CAST_RINT, processDataNum);
    } else if constexpr (std::is_same_v<TYPE_X, float>) {
        LocalTensor<float> oneLocal = tmpCommon.Get<float>();
        Muls(xLocal, xLocal, -1.0f, processDataNum);
        Exp(xLocal, xLocal, processDataNum);
        Adds(xLocal, xLocal, 1.0f, processDataNum);
        Div(yLocal, oneLocal, xLocal, processDataNum);
    } else if constexpr (std::is_same_v<TYPE_X, int8_t> || std::is_same_v<TYPE_X, uint8_t>) {
        LocalTensor<half> computeTensor = yLocal.template ReinterpretCast<half>();
        LocalTensor<half> oneLocal = tmpCommon.Get<half>();

        Cast(computeTensor, xLocal, RoundMode::CAST_NONE, processDataNum);
        Muls(computeTensor, computeTensor, static_cast<half>(-1.0f), processDataNum);
        Exp(computeTensor, computeTensor, processDataNum);
        Adds(computeTensor, computeTensor, static_cast<half>(1.0f), processDataNum);
        Div(computeTensor, oneLocal, computeTensor, processDataNum);
        Cast(yLocal, computeTensor, RoundMode::CAST_RINT, processDataNum);
    }

    SetFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
    WaitFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
    DataCopy(yGm[offset], yLocal, this->processDataNum);
    queBind.template FreeTensor(yLocal);
}

template <typename TYPE_X, uint64_t BUFFER_NUM>
__aicore__ inline void KernelSigmoid<TYPE_X, BUFFER_NUM>::Process()
{
    uint64_t loopCount = this->tileNum;
    this->processDataNum = this->tileDataNum;

    uint64_t offset = 0;
    for (uint64_t i = 0; i < loopCount - 1; i++, offset += this->tileDataNum) {
        Compute(offset);
    }
    this->processDataNum = this->tailDataNum;
    Compute(offset);
}

} // namespace MySigmoid
#endif // SIGMOID_H

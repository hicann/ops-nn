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
 * \file index_fill_d.h
 * \brief
 */
#ifndef INDEX_FILL_D_H
#define INDEX_FILL_D_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "index_fill_d_tiling_data.h"
#include "index_fill_d_tiling_key.h"

namespace MyIndexFillD {

using namespace AscendC;

template <typename TYPE_X, typename TYPE_Y, uint64_t BUFFER_NUM>
class KernelIndexFillD {
public:
    __aicore__ inline KernelIndexFillD(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR assist1, GM_ADDR assist2, GM_ADDR y,
                                const IndexFillDTilingData* tilingData, TPipe* pipeIn);
    __aicore__ inline void Process();

private:
    __aicore__ inline void DoRunOp(int32_t offset);

private:
    TPipe* pipe;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, BUFFER_NUM> bindQue;
    TBuf<QuePosition::VECCALC> castBuf, maskBuf;
    GlobalTensor<TYPE_X> xGm;
    GlobalTensor<TYPE_X> assist1Gm;
    GlobalTensor<TYPE_X> assist2Gm;
    GlobalTensor<TYPE_Y> yGm;
    uint32_t coreDataNum = 0;
    uint32_t tileNum = 0;
    uint32_t tileDataNum = 0;
    uint32_t tailDataNum = 0;
    uint32_t processDataNum = 0;
};

template <typename TYPE_X, typename TYPE_Y, uint64_t BUFFER_NUM>
__aicore__ inline void KernelIndexFillD<TYPE_X, TYPE_Y, BUFFER_NUM>::Init(GM_ADDR x, GM_ADDR assist1, GM_ADDR assist2,
                                                                          GM_ADDR y,
                                                                          const IndexFillDTilingData* tilingData,
                                                                          TPipe* pipeIn)
{
    ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
    uint32_t coreNum = GetBlockIdx();
    uint32_t globalBufferIndex = tilingData->bigCoreDataNum * GetBlockIdx();
    this->tileDataNum = tilingData->tileDataNum;
    this->pipe = pipeIn;
    if (coreNum < tilingData->tailBlockNum) {
        this->coreDataNum = tilingData->bigCoreDataNum;
        this->tileNum = tilingData->finalBigTileNum;
        this->tailDataNum = tilingData->bigTailDataNum;
    } else {
        this->coreDataNum = tilingData->smallCoreDataNum;
        this->tileNum = tilingData->finalSmallTileNum;
        this->tailDataNum = tilingData->smallTailDataNum;
        globalBufferIndex -= (tilingData->bigCoreDataNum - tilingData->smallCoreDataNum) *
                             (GetBlockIdx() - tilingData->tailBlockNum);
    }
    xGm.SetGlobalBuffer((__gm__ TYPE_X*)x + globalBufferIndex, this->coreDataNum);
    assist1Gm.SetGlobalBuffer((__gm__ TYPE_X*)assist1 + globalBufferIndex, this->coreDataNum);
    assist2Gm.SetGlobalBuffer((__gm__ TYPE_X*)assist2 + globalBufferIndex, this->coreDataNum);
    yGm.SetGlobalBuffer((__gm__ TYPE_Y*)y + globalBufferIndex, this->coreDataNum);
    pipe->InitBuffer(bindQue, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_X) * 3);
    pipe->InitBuffer(maskBuf, this->tileDataNum * sizeof(uint8_t));
    if constexpr (IsSameType<TYPE_X, bfloat16_t>::value) {
        pipe->InitBuffer(castBuf, this->tileDataNum * sizeof(float));
    }
    if constexpr (IsSameType<TYPE_X, int8_t>::value) {
        pipe->InitBuffer(castBuf, this->tileDataNum * sizeof(int16_t));
    }
}

template <typename TYPE_X, typename TYPE_Y, uint64_t BUFFER_NUM>
__aicore__ inline void KernelIndexFillD<TYPE_X, TYPE_Y, BUFFER_NUM>::DoRunOp(int32_t offset)
{
    LocalTensor<TYPE_X> xLocal = bindQue.template AllocTensor<TYPE_X>();
    LocalTensor<TYPE_X> assist1Local = xLocal[this->processDataNum];
    LocalTensor<TYPE_X> assist2Local = assist1Local[this->processDataNum];
    LocalTensor<TYPE_X> yLocal = assist1Local[this->processDataNum];
    int32_t eventIDMTE2ToV = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    int32_t eventIDVToMTE3 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    DataCopy(xLocal, xGm[offset], this->processDataNum);
    DataCopy(assist1Local, assist1Gm[offset], this->processDataNum);
    DataCopy(assist2Local, assist2Gm[offset], this->processDataNum);
    SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
    if constexpr (IsSameType<TYPE_X, bfloat16_t>::value) {
        auto maskLocal = maskBuf.Get<uint8_t>();
        auto castLocal = castBuf.Get<float>();
        auto yLocalFp = xLocal.template ReinterpretCast<float>();

        Cast(castLocal, assist1Local, RoundMode::CAST_NONE, this->processDataNum);
        CompareScalar(maskLocal, castLocal, (float)0.0, CMPMODE::GT, this->processDataNum);
        Cast(castLocal, xLocal, RoundMode::CAST_NONE, this->processDataNum);
        Cast(yLocalFp, assist2Local, RoundMode::CAST_NONE, this->processDataNum);
        Select(yLocalFp, maskLocal, castLocal, yLocalFp, SELMODE::VSEL_TENSOR_TENSOR_MODE, this->processDataNum);
        Cast(yLocal, yLocalFp, RoundMode::CAST_RINT, this->processDataNum);
    } else if constexpr (IsSameType<TYPE_X, float>::value || IsSameType<TYPE_X, half>::value) {
        auto maskLocal = maskBuf.Get<uint8_t>();
        CompareScalar(maskLocal, assist1Local, (TYPE_X)0.0, CMPMODE::GT, this->processDataNum);
        Select(yLocal, maskLocal, xLocal, assist2Local, SELMODE::VSEL_TENSOR_TENSOR_MODE, this->processDataNum);
    } else if constexpr (IsSameType<TYPE_X, int32_t>::value) {
        Mul(xLocal, xLocal, assist1Local, this->processDataNum);
        Add(yLocal, xLocal, assist2Local, this->processDataNum);
    } else if constexpr (IsSameType<TYPE_X, int8_t>::value) {
        auto castLocal = castBuf.Get<half>();
        auto yLocalHalf = xLocal.template ReinterpretCast<half>();
        Cast(castLocal, xLocal, RoundMode::CAST_NONE, this->processDataNum);
        Cast(yLocalHalf, assist1Local, RoundMode::CAST_NONE, this->processDataNum);
        Mul(yLocalHalf, yLocalHalf, castLocal, this->processDataNum);
        Cast(castLocal, assist2Local, RoundMode::CAST_NONE, this->processDataNum);
        Add(yLocalHalf, yLocalHalf, castLocal, this->processDataNum);
        Cast(yLocal, yLocalHalf, RoundMode::CAST_RINT, this->processDataNum);
    }
    SetFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
    WaitFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
    DataCopy(yGm[offset], yLocal, this->processDataNum);
    bindQue.template FreeTensor(xLocal);
}

template <typename TYPE_X, typename TYPE_Y, uint64_t BUFFER_NUM>
__aicore__ inline void KernelIndexFillD<TYPE_X, TYPE_Y, BUFFER_NUM>::Process()
{
    int32_t loopCount = this->tileNum - 1;
    this->processDataNum = this->tileDataNum;
    int32_t offset = 0;
    for (int32_t i = 0; i < loopCount; i++, offset += this->tileDataNum) {
        DoRunOp(offset);
    }
    this->processDataNum = this->tailDataNum;
    DoRunOp(offset);
}

} // namespace MyIndexFillD
#endif // INDEX_FILL_D_H
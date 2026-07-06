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
 * \file gelu.h
 * \brief
 */
#ifndef GELU_H
#define GELU_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "gelu_tiling_data.h"
#include "gelu_tiling_key.h"

namespace MyGelu {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

// Gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
// Approximate: x / (1 + exp(-1.595769122 * (x + 0.0455399241 * x^3)))
constexpr float GELU_PARAM1 = 0.0455399241f;
constexpr float GELU_PARAM2 = -1.595769122f;

template <typename TYPE_X, typename TYPE_Y>
class KernelGelu {
public:
    __aicore__ inline KernelGelu(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const GeluTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int32_t progress);
    __aicore__ inline void CopyOut(int32_t progress);
    __aicore__ inline void Compute(int32_t progress);
    __aicore__ inline void ComputeFp16(LocalTensor<TYPE_X>& xLocal, LocalTensor<TYPE_Y>& yLocal);
    __aicore__ inline void ComputeFp32(LocalTensor<TYPE_X>& xLocal, LocalTensor<TYPE_Y>& yLocal);

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::TBuf<QuePosition::VECCALC> tmpBuffer1, tmpBuffer2;
    AscendC::TBuf<QuePosition::VECCALC> QueueTmpX, QueueTmpY;
    AscendC::GlobalTensor<TYPE_X> xGm;
    AscendC::GlobalTensor<TYPE_Y> yGm;
    uint64_t coreDataNum = 0;
    uint64_t tileNum = 0;
    uint64_t tileDataNum = 0;
    uint64_t tailDataNum = 0;
    uint64_t processDataNum = 0;
};

template <typename TYPE_X, typename TYPE_Y>
__aicore__ inline void KernelGelu<TYPE_X, TYPE_Y>::Init(GM_ADDR x, GM_ADDR y, const GeluTilingData* tilingData)
{
    ASSERT(AscendC::GetBlockNum() != 0 && "block dim can not be zero!");
    uint64_t coreId = AscendC::GetBlockIdx();
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
    yGm.SetGlobalBuffer((__gm__ TYPE_Y*)y + globalBufferIndex, this->coreDataNum);
    pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_X));
    pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_Y));

    pipe.InitBuffer(tmpBuffer1, this->tileDataNum * sizeof(float));
    pipe.InitBuffer(tmpBuffer2, this->tileDataNum * sizeof(float));
    pipe.InitBuffer(QueueTmpX, this->tileDataNum * sizeof(float));
    pipe.InitBuffer(QueueTmpY, this->tileDataNum * sizeof(float));
}

template <typename TYPE_X, typename TYPE_Y>
__aicore__ inline void KernelGelu<TYPE_X, TYPE_Y>::CopyIn(int32_t progress)
{
    AscendC::LocalTensor<TYPE_X> xLocal = inQueueX.AllocTensor<TYPE_X>();
    AscendC::DataCopy(xLocal, xGm[progress * this->tileDataNum], this->processDataNum);
    inQueueX.EnQue(xLocal);
}

template <typename TYPE_X, typename TYPE_Y>
__aicore__ inline void KernelGelu<TYPE_X, TYPE_Y>::CopyOut(int32_t progress)
{
    AscendC::LocalTensor<TYPE_Y> yLocal = outQueueY.DeQue<TYPE_Y>();
    AscendC::DataCopy(yGm[progress * this->tileDataNum], yLocal, this->processDataNum);
    outQueueY.FreeTensor(yLocal);
}

template <typename TYPE_X, typename TYPE_Y>
__aicore__ inline void KernelGelu<TYPE_X, TYPE_Y>::ComputeFp16(LocalTensor<TYPE_X>& xLocal, LocalTensor<TYPE_Y>& yLocal)
{
    AscendC::LocalTensor<float> tmp1Local = tmpBuffer1.Get<float>();
    AscendC::LocalTensor<float> tmp2Local = tmpBuffer2.Get<float>();
    AscendC::LocalTensor<float> xLocalfp32 = QueueTmpX.Get<float>();
    AscendC::LocalTensor<float> yLocalfp32 = QueueTmpY.Get<float>();

    AscendC::Cast(xLocalfp32, xLocal, RoundMode::CAST_NONE, this->processDataNum);

    // Gelu(x) = x / (1 + exp(-1.595769122 * (x + 0.0455399241 * x^3)))
    AscendC::Mul(tmp1Local, xLocalfp32, xLocalfp32, this->processDataNum);
    AscendC::Mul(tmp1Local, tmp1Local, xLocalfp32, this->processDataNum);
    AscendC::Muls(tmp1Local, tmp1Local, float(GELU_PARAM1), this->processDataNum);
    AscendC::Add(tmp1Local, tmp1Local, xLocalfp32, this->processDataNum);
    AscendC::Muls(tmp1Local, tmp1Local, float(GELU_PARAM2), this->processDataNum);
    AscendC::Exp(tmp2Local, tmp1Local, this->processDataNum);
    AscendC::Adds(tmp2Local, tmp2Local, float(1), this->processDataNum);
    AscendC::Div(yLocalfp32, xLocalfp32, tmp2Local, this->processDataNum);

    AscendC::Cast(yLocal, yLocalfp32, RoundMode::CAST_RINT, this->processDataNum);
}

template <typename TYPE_X, typename TYPE_Y>
__aicore__ inline void KernelGelu<TYPE_X, TYPE_Y>::ComputeFp32(LocalTensor<TYPE_X>& xLocal, LocalTensor<TYPE_Y>& yLocal)
{
    AscendC::LocalTensor<TYPE_X> tmp1Local = tmpBuffer1.Get<TYPE_X>();
    AscendC::LocalTensor<TYPE_X> tmp2Local = tmpBuffer2.Get<TYPE_X>();

    // Gelu(x) = x / (1 + exp(-1.595769122 * (x + 0.0455399241 * x^3)))
    AscendC::Mul(tmp1Local, xLocal, xLocal, this->processDataNum);
    AscendC::Mul(tmp1Local, tmp1Local, xLocal, this->processDataNum);
    AscendC::Muls(tmp1Local, tmp1Local, TYPE_X(GELU_PARAM1), this->processDataNum);
    AscendC::Add(tmp1Local, tmp1Local, xLocal, this->processDataNum);
    AscendC::Muls(tmp1Local, tmp1Local, TYPE_X(GELU_PARAM2), this->processDataNum);
    AscendC::Exp(tmp2Local, tmp1Local, this->processDataNum);
    AscendC::Adds(tmp2Local, tmp2Local, TYPE_X(1), this->processDataNum);
    AscendC::Div(yLocal, xLocal, tmp2Local, this->processDataNum);
}

template <typename TYPE_X, typename TYPE_Y>
__aicore__ inline void KernelGelu<TYPE_X, TYPE_Y>::Compute(int32_t progress)
{
    AscendC::LocalTensor<TYPE_X> xLocal = inQueueX.DeQue<TYPE_X>();
    AscendC::LocalTensor<TYPE_Y> yLocal = outQueueY.AllocTensor<TYPE_Y>();

    if constexpr (std::is_same_v<TYPE_X, __bf16> || std::is_same_v<TYPE_X, half>) {
        ComputeFp16(xLocal, yLocal);
    } else if constexpr (std::is_same_v<TYPE_X, float>) {
        ComputeFp32(xLocal, yLocal);
    }

    outQueueY.EnQue<TYPE_Y>(yLocal);
    inQueueX.FreeTensor(xLocal);
}

template <typename TYPE_X, typename TYPE_Y>
__aicore__ inline void KernelGelu<TYPE_X, TYPE_Y>::Process()
{
    int32_t loopCount = this->tileNum;
    this->processDataNum = this->tileDataNum;
    for (int32_t i = 0; i < loopCount - 1; i++) {
        CopyIn(i);
        Compute(i);
        CopyOut(i);
    }
    this->processDataNum = this->tailDataNum;
    CopyIn(loopCount - 1);
    Compute(loopCount - 1);
    CopyOut(loopCount - 1);
}

} // namespace MyGelu

#endif // GELU_H

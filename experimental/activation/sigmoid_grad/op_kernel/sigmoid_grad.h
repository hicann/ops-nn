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
 * \file sigmoid_grad.h
 * \brief
 */
#ifndef SIGMOID_GRAD_H
#define SIGMOID_GRAD_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "sigmoid_grad_tiling_data.h"
#include "sigmoid_grad_tiling_key.h"

namespace NsSigmoidGrad {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

template <typename TYPE_X>
class KernelSigmoidGrad {
public:
    __aicore__ inline KernelSigmoidGrad(){};

    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR y, GM_ADDR z, const SigmoidGradTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int32_t progress);
    __aicore__ inline void CopyOut(int32_t progress);
    __aicore__ inline void Compute(int32_t progress);

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueY;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueZ;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpQueue0;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpQueue1;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpQueue2;
    AscendC::GlobalTensor<TYPE_X> xGm;
    AscendC::GlobalTensor<TYPE_X> yGm;
    AscendC::GlobalTensor<TYPE_X> zGm;
    uint64_t coreDataNum;
    uint64_t tileNum;
    uint64_t tileDataNum;
    uint64_t tailDataNum;
    uint64_t processDataNum;
};

template <typename TYPE_X>
__aicore__ inline void KernelSigmoidGrad<TYPE_X>::Init(
    GM_ADDR x, GM_ADDR y, GM_ADDR z, const SigmoidGradTilingData* tilingData)
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
        globalBufferIndex -= (tilingData->bigCoreDataNum - tilingData->smallCoreDataNum) * (coreId - tilingData->tailBlockNum);
    }
    xGm.SetGlobalBuffer((__gm__ TYPE_X*)x + globalBufferIndex, this->coreDataNum);
    yGm.SetGlobalBuffer((__gm__ TYPE_X*)y + globalBufferIndex, this->coreDataNum);
    zGm.SetGlobalBuffer((__gm__ TYPE_X*)z + globalBufferIndex, this->coreDataNum);
    pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_X));
    pipe.InitBuffer(inQueueY, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_X));
    pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_X));
    pipe.InitBuffer(tmpQueue0, this->tileDataNum * sizeof(float));
    if constexpr (!std::is_same_v<TYPE_X, float32_t>) {
        pipe.InitBuffer(tmpQueue1, this->tileDataNum * sizeof(float));
        pipe.InitBuffer(tmpQueue2, this->tileDataNum * sizeof(float));
    }
}

template <typename TYPE_X>
__aicore__ inline void KernelSigmoidGrad<TYPE_X>::CopyIn(int32_t progress)
{
    AscendC::LocalTensor<TYPE_X> xLocal = inQueueX.AllocTensor<TYPE_X>();
    AscendC::LocalTensor<TYPE_X> yLocal = inQueueY.AllocTensor<TYPE_X>();
    AscendC::DataCopy(xLocal, xGm[progress * this->tileDataNum], this->processDataNum);
    AscendC::DataCopy(yLocal, yGm[progress * this->tileDataNum], this->processDataNum);
    inQueueX.EnQue(xLocal);
    inQueueY.EnQue(yLocal);
}

template <typename TYPE_X>
__aicore__ inline void KernelSigmoidGrad<TYPE_X>::CopyOut(int32_t progress)
{
    AscendC::LocalTensor<TYPE_X> zLocal = outQueueZ.DeQue<TYPE_X>();
    AscendC::DataCopy(zGm[progress * this->tileDataNum], zLocal, this->processDataNum);
    outQueueZ.FreeTensor(zLocal);
}

template <typename TYPE_X>
__aicore__ inline void KernelSigmoidGrad<TYPE_X>::Compute(int32_t progress)
{
    AscendC::LocalTensor<TYPE_X> xLocal = inQueueX.DeQue<TYPE_X>();
    AscendC::LocalTensor<TYPE_X> yLocal = inQueueY.DeQue<TYPE_X>();
    AscendC::LocalTensor<TYPE_X> zLocal = outQueueZ.AllocTensor<TYPE_X>();
    AscendC::LocalTensor<float> tmp0Local = tmpQueue0.AllocTensor<float>();
    if constexpr (!std::is_same_v<TYPE_X, float32_t>) {
        AscendC::LocalTensor<float> xFP32 = tmpQueue1.AllocTensor<float>();
        AscendC::LocalTensor<float> yFP32 = tmpQueue2.AllocTensor<float>();
        AscendC::Duplicate(tmp0Local, static_cast<float>(1.0), this->processDataNum);
        AscendC::Cast(xFP32, xLocal, AscendC::RoundMode::CAST_NONE, this->processDataNum);
        AscendC::Cast(yFP32, yLocal, AscendC::RoundMode::CAST_NONE, this->processDataNum);
        AscendC::Sub(tmp0Local, tmp0Local, xFP32, this->processDataNum);
        AscendC::Mul(tmp0Local, tmp0Local, xFP32, this->processDataNum);
        AscendC::Mul(yFP32, yFP32, tmp0Local, this->processDataNum);
        AscendC::Cast(zLocal, yFP32, AscendC::RoundMode::CAST_RINT, this->processDataNum);
        outQueueZ.EnQue<TYPE_X>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    } else {
        AscendC::Duplicate(tmp0Local, static_cast<float>(1.0), this->processDataNum);
        AscendC::Sub(tmp0Local, tmp0Local, xLocal, this->processDataNum);
        AscendC::Mul(tmp0Local, tmp0Local, xLocal, this->processDataNum);
        AscendC::Mul(zLocal, yLocal, tmp0Local, this->processDataNum);
        outQueueZ.EnQue<TYPE_X>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

}

template <typename TYPE_X>
__aicore__ inline void KernelSigmoidGrad<TYPE_X>::Process()
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

} // namespace NsSigmoidGrad
#endif // SIGMOID_GRAD_H

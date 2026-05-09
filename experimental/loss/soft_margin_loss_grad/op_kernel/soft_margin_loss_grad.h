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
 * \file soft_margin_loss_grad.h
 * \brief
 */
#ifndef SOFT_MARGIN_LOSS_GRAD_H
#define SOFT_MARGIN_LOSS_GRAD_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "soft_margin_loss_grad_tiling_data.h"
#include "soft_margin_loss_grad_tiling_key.h"

#include "kernel_operator.h"

namespace NsSoftMarginLossGrad {

using namespace AscendC;

constexpr int32_t QUEUE_DEPTH = 1;
constexpr int32_t BUFFER_NUM = 2;

template <typename TYPE_X>
class KernelSoftMarginLossGrad {
public:
    __aicore__ inline KernelSoftMarginLossGrad(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR grad_out, GM_ADDR out, uint64_t smallCoreDataNum, uint64_t bigCoreDataNum, uint64_t finalBigTileNum,
        uint64_t finalSmallTileNum, uint64_t tileDataNum, uint64_t smallTailDataNum, uint64_t bigTailDataNum, uint64_t tailBlockNum, float cof);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int32_t progress);
    __aicore__ inline void CopyOut(int32_t progress);
    __aicore__ inline void Compute(int32_t progress);

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, QUEUE_DEPTH> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECIN, QUEUE_DEPTH> inQueueY;
    AscendC::TQue<AscendC::TPosition::VECIN, QUEUE_DEPTH> inQueueGradOut;
    AscendC::TQue<AscendC::TPosition::VECOUT, QUEUE_DEPTH> outQueueOut;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpQueue0;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpQueue1;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpQueue2;
    AscendC::TBuf<AscendC::TPosition::VECCALC> castTmpQueue0;
    AscendC::TBuf<AscendC::TPosition::VECCALC> castTmpQueue1;
    AscendC::TBuf<AscendC::TPosition::VECCALC> castTmpQueue2;

    AscendC::GlobalTensor<TYPE_X> xGm;
    AscendC::GlobalTensor<TYPE_X> yGm;
    AscendC::GlobalTensor<TYPE_X> gradOutGm;
    AscendC::GlobalTensor<TYPE_X> outGm;
    uint64_t coreDataNum;
    uint64_t tileNum;
    uint64_t tileDataNum;
    uint64_t tailDataNum;
    uint64_t processDataNum;
    float cof;
};

template <typename TYPE_X>
__aicore__ inline void KernelSoftMarginLossGrad<TYPE_X>::Init(GM_ADDR x, GM_ADDR y, GM_ADDR grad_out, GM_ADDR out, uint64_t smallCoreDataNum, uint64_t bigCoreDataNum, uint64_t finalBigTileNum,
    uint64_t finalSmallTileNum, uint64_t tileDataNum, uint64_t smallTailDataNum, uint64_t bigTailDataNum, uint64_t tailBlockNum, float cof)
{
    uint64_t coreId = AscendC::GetBlockIdx();
    uint64_t globalBufferIndex = bigCoreDataNum * coreId;
    this->tileDataNum = tileDataNum;
    this->cof = cof;
    if (coreId < tailBlockNum) {
        this->coreDataNum = bigCoreDataNum;
        this->tileNum = finalBigTileNum;
        this->tailDataNum = bigTailDataNum;
    } else {
        this->coreDataNum = smallCoreDataNum;
        this->tileNum = finalSmallTileNum;
        this->tailDataNum = smallTailDataNum;
        globalBufferIndex -= (bigCoreDataNum - smallCoreDataNum) * (coreId - tailBlockNum);
    }
    xGm.SetGlobalBuffer((__gm__ TYPE_X *)x + globalBufferIndex, this->coreDataNum);
    yGm.SetGlobalBuffer((__gm__ TYPE_X *)y + globalBufferIndex, this->coreDataNum);
    gradOutGm.SetGlobalBuffer((__gm__ TYPE_X *)grad_out + globalBufferIndex, this->coreDataNum);
    outGm.SetGlobalBuffer((__gm__ TYPE_X *)out + globalBufferIndex, this->coreDataNum);
    pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_X));
    pipe.InitBuffer(inQueueY, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_X));
    pipe.InitBuffer(inQueueGradOut, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_X));
    pipe.InitBuffer(outQueueOut, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_X));
    pipe.InitBuffer(tmpQueue0, this->tileDataNum * sizeof(float));
    pipe.InitBuffer(tmpQueue1, this->tileDataNum * sizeof(float));
    pipe.InitBuffer(tmpQueue2, this->tileDataNum * sizeof(float));
    if constexpr (std::is_same_v<TYPE_X, half> || std::is_same_v<TYPE_X, bfloat16_t>) {
        pipe.InitBuffer(castTmpQueue0, this->tileDataNum * sizeof(float));
        pipe.InitBuffer(castTmpQueue1, this->tileDataNum * sizeof(float));
        pipe.InitBuffer(castTmpQueue2, this->tileDataNum * sizeof(float));
    }
}

template <typename TYPE_X>
__aicore__ inline void KernelSoftMarginLossGrad<TYPE_X>::CopyIn(int32_t progress)
{
    AscendC::LocalTensor<TYPE_X> xLocal = inQueueX.AllocTensor<TYPE_X>();
    AscendC::LocalTensor<TYPE_X> yLocal = inQueueY.AllocTensor<TYPE_X>();
    AscendC::LocalTensor<TYPE_X> gradOutLocal = inQueueGradOut.AllocTensor<TYPE_X>();
    AscendC::DataCopy(xLocal, xGm[progress * this->tileDataNum], this->processDataNum);
    AscendC::DataCopy(yLocal, yGm[progress * this->tileDataNum], this->processDataNum);
    AscendC::DataCopy(gradOutLocal, gradOutGm[progress * this->tileDataNum], this->processDataNum);
    inQueueX.EnQue(xLocal);
    inQueueY.EnQue(yLocal);
    inQueueGradOut.EnQue(gradOutLocal);
}

template <typename TYPE_X>
__aicore__ inline void KernelSoftMarginLossGrad<TYPE_X>::CopyOut(int32_t progress)
{
    AscendC::LocalTensor<TYPE_X> outLocal = outQueueOut.DeQue<TYPE_X>();
    AscendC::DataCopy(outGm[progress * this->tileDataNum], outLocal, this->processDataNum);
    outQueueOut.FreeTensor(outLocal);
}

template <typename TYPE_X>
__aicore__ inline void KernelSoftMarginLossGrad<TYPE_X>::Compute(int32_t progress)
{
    if constexpr (std::is_same_v<TYPE_X, half> || std::is_same_v<TYPE_X, bfloat16_t>) {
        AscendC::LocalTensor<TYPE_X> xLocal = inQueueX.DeQue<TYPE_X>();
        AscendC::LocalTensor<TYPE_X> yLocal = inQueueY.DeQue<TYPE_X>();
        AscendC::LocalTensor<TYPE_X> gradOutLocal = inQueueGradOut.DeQue<TYPE_X>();
        AscendC::LocalTensor<TYPE_X> outLocal = outQueueOut.AllocTensor<TYPE_X>();
        AscendC::LocalTensor<float> tmp0Local = tmpQueue0.AllocTensor<float>();
        AscendC::LocalTensor<float> tmp1Local = tmpQueue1.AllocTensor<float>();
        AscendC::LocalTensor<float> tmp2Local = tmpQueue2.AllocTensor<float>();
        AscendC::LocalTensor<float> cast0Local = castTmpQueue0.AllocTensor<float>();
        AscendC::LocalTensor<float> cast1Local = castTmpQueue1.AllocTensor<float>();
        AscendC::LocalTensor<float> cast2Local = castTmpQueue2.AllocTensor<float>();
        AscendC::Cast(cast0Local, xLocal, AscendC::RoundMode::CAST_NONE, this->processDataNum);
        AscendC::Cast(cast1Local, yLocal, AscendC::RoundMode::CAST_NONE, this->processDataNum);
        AscendC::Cast(cast2Local, gradOutLocal, AscendC::RoundMode::CAST_NONE, this->processDataNum);
        AscendC::Muls(tmp0Local, cast1Local, static_cast<float>(-1.0), this->processDataNum);
        AscendC::Mul(tmp1Local, cast0Local, tmp0Local, this->processDataNum);
        AscendC::Exp(tmp1Local, tmp1Local, this->processDataNum);
        AscendC::Adds(tmp2Local, tmp1Local, static_cast<float>(1.0), this->processDataNum);
        AscendC::Div(tmp2Local, tmp1Local, tmp2Local, this->processDataNum);
        AscendC::Mul(tmp0Local, tmp0Local, tmp2Local, this->processDataNum);
        AscendC::Mul(tmp1Local, tmp0Local, cast2Local, this->processDataNum);
        AscendC::Muls(tmp1Local, tmp1Local, static_cast<float>(this->cof), this->processDataNum);
        AscendC::Cast(outLocal, tmp1Local, AscendC::RoundMode::CAST_RINT, this->processDataNum);
        outQueueOut.EnQue<TYPE_X>(outLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
        inQueueGradOut.FreeTensor(gradOutLocal);
    } else {
        AscendC::LocalTensor<TYPE_X> xLocal = inQueueX.DeQue<TYPE_X>();
        AscendC::LocalTensor<TYPE_X> yLocal = inQueueY.DeQue<TYPE_X>();
        AscendC::LocalTensor<TYPE_X> gradOutLocal = inQueueGradOut.DeQue<TYPE_X>();
        AscendC::LocalTensor<TYPE_X> outLocal = outQueueOut.AllocTensor<TYPE_X>();
        AscendC::LocalTensor<float> tmp0Local = tmpQueue0.AllocTensor<float>();
        AscendC::LocalTensor<float> tmp1Local = tmpQueue1.AllocTensor<float>();
        AscendC::LocalTensor<float> tmp2Local = tmpQueue2.AllocTensor<float>();
        AscendC::Muls(tmp0Local, yLocal, static_cast<float>(-1.0), this->processDataNum);
        AscendC::Mul(tmp1Local, xLocal, tmp0Local, this->processDataNum);
        AscendC::Exp(tmp1Local, tmp1Local, this->processDataNum);
        AscendC::Adds(tmp2Local, tmp1Local, static_cast<float>(1.0), this->processDataNum);
        AscendC::Div(tmp2Local, tmp1Local, tmp2Local, this->processDataNum);
        AscendC::Mul(tmp0Local, tmp0Local, tmp2Local, this->processDataNum);
        AscendC::Mul(tmp1Local, tmp0Local, gradOutLocal, this->processDataNum);
        AscendC::Muls(outLocal, tmp1Local, static_cast<float>(this->cof), this->processDataNum);
        outQueueOut.EnQue<TYPE_X>(outLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
        inQueueGradOut.FreeTensor(gradOutLocal);
    }
}

template <typename TYPE_X>
__aicore__ inline void KernelSoftMarginLossGrad<TYPE_X>::Process()
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

} // namespace NsSoftMarginLossGrad
#endif // SOFT_MARGIN_LOSS_GRAD_H
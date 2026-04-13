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
 * \file elu_grad.h
 * \brief
 */
#ifndef ELU_GRAD_H
#define ELU_GRAD_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "elu_grad_tiling_data.h"
#include "elu_grad_tiling_key.h"

#include "kernel_operator.h"

namespace NsEluGrad {

using namespace AscendC;

constexpr int32_t SINGLE_BUFFER_NUM = 1;
constexpr int32_t DOUBLE_BUFFER_NUM = 2;

template <typename TYPE_GRADS>
class KernelEluGrad {
public:
    __aicore__ inline KernelEluGrad(){};

    __aicore__ inline void Init(GM_ADDR grads, GM_ADDR activations, GM_ADDR y, uint64_t smallCoreDataNum, uint64_t bigCoreDataNum, uint64_t finalBigTileNum,
        uint64_t finalSmallTileNum, uint64_t tileDataNum, uint64_t smallTailDataNum, uint64_t bigTailDataNum, uint64_t tailBlockNum, uint64_t bufferOpen);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int32_t progress);
    __aicore__ inline void CopyOut(int32_t progress);
    __aicore__ inline void Compute(int32_t progress);

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, DOUBLE_BUFFER_NUM> inQueueGrads, inQueueActivations;
    AscendC::TQue<AscendC::TPosition::VECOUT, DOUBLE_BUFFER_NUM> outQueueY;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpQueue0, tmpQueue1, tmpQueueMask1;

    AscendC::GlobalTensor<TYPE_GRADS> gradsGm, activationsGm, yGm;
    uint64_t coreDataNum;
    uint64_t tileNum;
    uint64_t tileDataNum;
    uint64_t tailDataNum;
    uint64_t processDataNum;
};

template <typename TYPE_GRADS>
__aicore__ inline void KernelEluGrad<TYPE_GRADS>::Init(GM_ADDR grads, GM_ADDR activations, GM_ADDR y, uint64_t smallCoreDataNum, uint64_t bigCoreDataNum, uint64_t finalBigTileNum,
    uint64_t finalSmallTileNum, uint64_t tileDataNum, uint64_t smallTailDataNum, uint64_t bigTailDataNum, uint64_t tailBlockNum, uint64_t bufferOpen)
{
    ASSERT(AscendC::GetBlockNum() != 0 && "block dim can not be zero!");
    uint64_t coreId = AscendC::GetBlockIdx();
    uint64_t globalBufferIndex = bigCoreDataNum * coreId;
    this->tileDataNum = tileDataNum;
    // DOUBLE_BUFFER default OPEN, but bufferOpen=0, DOUBLE_BUFFER will CLOSE.
    uint64_t BUFFER_NUM = DOUBLE_BUFFER_NUM;
    if (bufferOpen == 0) {
        BUFFER_NUM = SINGLE_BUFFER_NUM;
    }
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
    gradsGm.SetGlobalBuffer((__gm__ TYPE_GRADS *)grads + globalBufferIndex, this->coreDataNum);
    activationsGm.SetGlobalBuffer((__gm__ TYPE_GRADS *)activations + globalBufferIndex, this->coreDataNum);
    yGm.SetGlobalBuffer((__gm__ TYPE_GRADS *)y + globalBufferIndex, this->coreDataNum);

    pipe.InitBuffer(inQueueGrads, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_GRADS));
    pipe.InitBuffer(inQueueActivations, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_GRADS));
    pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_GRADS));
    if (std::is_same_v<TYPE_GRADS, half> || std::is_same_v<TYPE_GRADS, bfloat16_t>) {
        pipe.InitBuffer(tmpQueueMask1, this->tileDataNum * sizeof(uint8_t));
        pipe.InitBuffer(tmpQueue0, this->tileDataNum * sizeof(float));
        pipe.InitBuffer(tmpQueue1, this->tileDataNum * sizeof(float));
    } else {
        pipe.InitBuffer(tmpQueueMask1, this->tileDataNum * sizeof(uint8_t));
    }
}

template <typename TYPE_GRADS>
__aicore__ inline void KernelEluGrad<TYPE_GRADS>::CopyIn(int32_t progress)
{
    AscendC::LocalTensor<TYPE_GRADS> gradsLocal = inQueueGrads.AllocTensor<TYPE_GRADS>();
    AscendC::LocalTensor<TYPE_GRADS> activationsLocal = inQueueActivations.AllocTensor<TYPE_GRADS>();
    AscendC::DataCopy(gradsLocal, gradsGm[progress * this->tileDataNum], this->processDataNum);
    AscendC::DataCopy(activationsLocal, activationsGm[progress * this->tileDataNum], this->processDataNum);
    inQueueGrads.EnQue(gradsLocal);
    inQueueActivations.EnQue(activationsLocal);
}

template <typename TYPE_GRADS>
__aicore__ inline void KernelEluGrad<TYPE_GRADS>::CopyOut(int32_t progress)
{
    AscendC::LocalTensor<TYPE_GRADS> yLocal = outQueueY.DeQue<TYPE_GRADS>();
    AscendC::DataCopy(yGm[progress * this->tileDataNum], yLocal, this->processDataNum);
    outQueueY.FreeTensor(yLocal);
}

template <typename TYPE_GRADS>
__aicore__ inline void KernelEluGrad<TYPE_GRADS>::Compute(int32_t progress)
{
    if (std::is_same_v<TYPE_GRADS, half> || std::is_same_v<TYPE_GRADS, bfloat16_t>) {
        AscendC::LocalTensor<TYPE_GRADS> gradsLocal = inQueueGrads.DeQue<TYPE_GRADS>();
        AscendC::LocalTensor<TYPE_GRADS> activationsLocal = inQueueActivations.DeQue<TYPE_GRADS>();
        AscendC::LocalTensor<TYPE_GRADS> yLocal = outQueueY.AllocTensor<TYPE_GRADS>();
        AscendC::LocalTensor<float> tmp0Local = tmpQueue0.AllocTensor<float>();
        AscendC::LocalTensor<float> tmp1Local = tmpQueue1.AllocTensor<float>();
        AscendC::LocalTensor<uint8_t> mask1Local = tmpQueueMask1.AllocTensor<uint8_t>();
        AscendC::Cast(tmp0Local, activationsLocal, AscendC::RoundMode::CAST_NONE, this->processDataNum);
        AscendC::Adds(tmp1Local, tmp0Local, static_cast<float>(1.0), this->processDataNum);
        AscendC::CompareScalar(mask1Local, tmp0Local, static_cast<float>(0.0), AscendC::CMPMODE::LE, this->processDataNum);
        AscendC::Select(tmp1Local, mask1Local, tmp1Local, static_cast<float>(1.0), AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, this->processDataNum);
        AscendC::Cast(tmp0Local, gradsLocal, AscendC::RoundMode::CAST_NONE, this->processDataNum);
        AscendC::Mul(tmp1Local, tmp0Local, tmp1Local, this->processDataNum);
        AscendC::Cast(yLocal, tmp1Local, AscendC::RoundMode::CAST_RINT, this->processDataNum);
        outQueueY.EnQue<TYPE_GRADS>(yLocal);
        inQueueGrads.FreeTensor(gradsLocal);
        inQueueActivations.FreeTensor(activationsLocal);
    } else {
        AscendC::LocalTensor<float> gradsLocal = inQueueGrads.DeQue<float>();
        AscendC::LocalTensor<float> activationsLocal = inQueueActivations.DeQue<float>();
        AscendC::LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
        AscendC::LocalTensor<uint8_t> mask1Local = tmpQueueMask1.AllocTensor<uint8_t>();
        AscendC::Adds(yLocal, activationsLocal, static_cast<float>(1.0), this->processDataNum);
        AscendC::CompareScalar(mask1Local, activationsLocal, static_cast<float>(0.0), AscendC::CMPMODE::LE, this->processDataNum);
        AscendC::Select(yLocal, mask1Local, yLocal, static_cast<float>(1.0), AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, this->processDataNum);
        AscendC::Mul(yLocal, yLocal, gradsLocal, this->processDataNum);
        outQueueY.EnQue<float>(yLocal);
        inQueueGrads.FreeTensor(gradsLocal);
        inQueueActivations.FreeTensor(activationsLocal);
    }
}

template <typename TYPE_GRADS>
__aicore__ inline void KernelEluGrad<TYPE_GRADS>::Process()
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

} // namespace NsEluGrad
#endif // ELU_GRAD_H
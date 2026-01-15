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
 * \file swish_grad.h
 * \brief
 */
#ifndef SWISH_GRAD_H
#define SWISH_GRAD_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "swish_grad_tiling_data.h"
#include "swish_grad_tiling_key.h"

#include "kernel_operator.h"

namespace NsSwishGrad {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

template <typename TYPE_X>
class KernelSwishGrad {
public:
    __aicore__ inline KernelSwishGrad(){};

    __aicore__ inline void Init(GM_ADDR grad, GM_ADDR x, GM_ADDR y, GM_ADDR grad_x, uint64_t smallCoreDataNum, uint64_t bigCoreDataNum, uint64_t finalBigTileNum,
        uint64_t finalSmallTileNum, uint64_t tileDataNum, uint64_t smallTailDataNum, uint64_t bigTailDataNum, uint64_t tailBlockNum, float sconf);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int32_t progress);
    __aicore__ inline void CopyOut(int32_t progress);
    __aicore__ inline void Compute(int32_t progress);

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueGrad, inQueueX, inQueueY;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueGrad;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpQueue0, tmpQueue1, tmpQueue2;

    AscendC::GlobalTensor<TYPE_X> gradGm, xGm, yGm, outputGm;
    AscendC::GlobalTensor<float> sconfGm;
    uint64_t coreDataNum;
    uint64_t tileNum;
    uint64_t tileDataNum;
    uint64_t tailDataNum;
    uint64_t processDataNum;
    float sconf;
    float value;
};

template <typename TYPE_X>
__aicore__ inline void KernelSwishGrad<TYPE_X>::Init(GM_ADDR grad, GM_ADDR x, GM_ADDR y, GM_ADDR grad_x, uint64_t smallCoreDataNum, uint64_t bigCoreDataNum, uint64_t finalBigTileNum,
    uint64_t finalSmallTileNum, uint64_t tileDataNum, uint64_t smallTailDataNum, uint64_t bigTailDataNum, uint64_t tailBlockNum, float sconf)
{
    ASSERT(AscendC::GetBlockNum() != 0 && "block dim can not be zero!");
    uint64_t coreId = AscendC::GetBlockIdx();
    uint64_t globalBufferIndex = bigCoreDataNum * coreId;
    this->tileDataNum = tileDataNum;
    this->sconf = sconf;
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
    gradGm.SetGlobalBuffer((__gm__ TYPE_X *)grad + globalBufferIndex, this->coreDataNum);
    xGm.SetGlobalBuffer((__gm__ TYPE_X *)x + globalBufferIndex, this->coreDataNum);
    outputGm.SetGlobalBuffer((__gm__ TYPE_X *)grad_x + globalBufferIndex, this->coreDataNum);

    pipe.InitBuffer(inQueueGrad, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_X));
    pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_X));
    pipe.InitBuffer(outQueueGrad, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_X));
    pipe.InitBuffer(tmpQueue0, this->tileDataNum * sizeof(float));
    if (std::is_same_v<TYPE_X, float16_t> || std::is_same_v<TYPE_X, bfloat16_t>) {
        pipe.InitBuffer(tmpQueue1, this->tileDataNum * sizeof(float));
        pipe.InitBuffer(tmpQueue2, this->tileDataNum * sizeof(float));
    }
}

template <typename TYPE_X>
__aicore__ inline void KernelSwishGrad<TYPE_X>::CopyIn(int32_t progress)
{
    AscendC::LocalTensor<TYPE_X> gradLocal = inQueueGrad.AllocTensor<TYPE_X>();
    AscendC::LocalTensor<TYPE_X> xLocal = inQueueX.AllocTensor<TYPE_X>();
    AscendC::DataCopy(gradLocal, gradGm[progress * this->tileDataNum], this->processDataNum);
    AscendC::DataCopy(xLocal, xGm[progress * this->tileDataNum], this->processDataNum);
    inQueueGrad.EnQue(gradLocal);
    inQueueX.EnQue(xLocal);
}

template <typename TYPE_X>
__aicore__ inline void KernelSwishGrad<TYPE_X>::CopyOut(int32_t progress)
{
    AscendC::LocalTensor<TYPE_X> outLocal = outQueueGrad.DeQue<TYPE_X>();
    AscendC::DataCopy(outputGm[progress * this->tileDataNum], outLocal, this->processDataNum);
    outQueueGrad.FreeTensor(outLocal);
}

template <typename TYPE_X>
__aicore__ inline void KernelSwishGrad<TYPE_X>::Compute(int32_t progress)
{
    if (std::is_same_v<TYPE_X, float16_t> || std::is_same_v<TYPE_X, bfloat16_t>) {
        AscendC::LocalTensor<TYPE_X> gradLocal = inQueueGrad.DeQue<TYPE_X>();
        AscendC::LocalTensor<TYPE_X> xLocal = inQueueX.DeQue<TYPE_X>();
        AscendC::LocalTensor<TYPE_X> outLocal = outQueueGrad.AllocTensor<TYPE_X>();
        AscendC::LocalTensor<float> tmp0Local = tmpQueue0.AllocTensor<float>();
        AscendC::LocalTensor<float> tmp1Local = tmpQueue1.AllocTensor<float>();
        AscendC::LocalTensor<float> tmp2Local = tmpQueue2.AllocTensor<float>();
        AscendC::Cast(tmp0Local, xLocal, AscendC::RoundMode::CAST_NONE, this->processDataNum);
        AscendC::Muls(tmp0Local, tmp0Local, static_cast<float>(this->value), this->processDataNum);
        AscendC::Muls(tmp1Local, tmp0Local, static_cast<float>(-1.0), this->processDataNum);
        AscendC::Exp(tmp1Local, tmp1Local, this->processDataNum);
        AscendC::Adds(tmp1Local, tmp1Local, static_cast<float>(1.0), this->processDataNum);
        AscendC::Duplicate(tmp2Local, static_cast<float>(1.0), this->processDataNum);
        AscendC::Div(tmp1Local, tmp2Local, tmp1Local, this->processDataNum);
        AscendC::Sub(tmp2Local, tmp2Local, tmp1Local, this->processDataNum);
        AscendC::Mul(tmp0Local, tmp0Local, tmp2Local, this->processDataNum);
        AscendC::Duplicate(tmp2Local, static_cast<float>(1.0), this->processDataNum);
        AscendC::Add(tmp0Local, tmp2Local, tmp0Local, this->processDataNum);
        AscendC::Mul(tmp0Local, tmp1Local, tmp0Local, this->processDataNum);
        AscendC::Cast(tmp1Local, gradLocal, AscendC::RoundMode::CAST_NONE, this->processDataNum);
        AscendC::Mul(tmp0Local, tmp1Local, tmp0Local, this->processDataNum);
        AscendC::Cast(outLocal, tmp0Local, AscendC::RoundMode::CAST_RINT, this->processDataNum);
        outQueueGrad.EnQue<TYPE_X>(outLocal);
        inQueueGrad.FreeTensor(gradLocal);
        inQueueX.FreeTensor(xLocal);
    } else {
        AscendC::LocalTensor<float> tmp0Local = tmpQueue0.AllocTensor<float>();
        AscendC::LocalTensor<float> gradLocal = inQueueGrad.DeQue<float>();
        AscendC::LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        AscendC::LocalTensor<float> outLocal = outQueueGrad.AllocTensor<float>();
        AscendC::Muls(xLocal, xLocal, static_cast<float>(this->value), this->processDataNum);
        AscendC::Muls(tmp0Local, xLocal, static_cast<float>(-1.0), this->processDataNum);
        AscendC::Exp(tmp0Local, tmp0Local, this->processDataNum);
        AscendC::Adds(tmp0Local, tmp0Local, static_cast<float>(1.0), this->processDataNum);
        AscendC::Duplicate(outLocal, static_cast<float>(1.0), this->processDataNum);
        AscendC::Div(outLocal, outLocal, tmp0Local, this->processDataNum);
        AscendC::Duplicate(tmp0Local, static_cast<float>(1.0), this->processDataNum);
        AscendC::Sub(tmp0Local, tmp0Local, outLocal, this->processDataNum);
        AscendC::Mul(xLocal, xLocal, tmp0Local, this->processDataNum);
        AscendC::Duplicate(tmp0Local, static_cast<float>(1.0), this->processDataNum);
        AscendC::Add(xLocal, tmp0Local, xLocal, this->processDataNum);
        AscendC::Mul(outLocal, outLocal, xLocal, this->processDataNum);
        AscendC::Mul(outLocal, gradLocal, outLocal, this->processDataNum);
        outQueueGrad.EnQue<float>(outLocal);
        inQueueGrad.FreeTensor(gradLocal);
        inQueueX.FreeTensor(xLocal);
    }
}

template <typename TYPE_X>
__aicore__ inline void KernelSwishGrad<TYPE_X>::Process()
{
    int32_t loopCount = this->tileNum;
    this->processDataNum = this->tileDataNum;
    this->value = this->sconf;
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

} // namespace NsSwishGrad
#endif // SWISH_GRAD_H
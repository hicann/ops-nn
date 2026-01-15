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
 * \file mish_grad.h
 * \brief
 */
#ifndef MISH_GRAD_H
#define MISH_GRAD_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "mish_grad_tiling_data.h"
#include "mish_grad_tiling_key.h"

#include "kernel_operator.h"

namespace NsMishGrad {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

template <typename TYPE_X>
class KernelMishGrad {
public:
    __aicore__ inline KernelMishGrad(){};

    __aicore__ inline void Init(GM_ADDR grad, GM_ADDR x, GM_ADDR tanhx, GM_ADDR x_grad, uint64_t smallCoreDataNum, uint64_t bigCoreDataNum, uint64_t finalBigTileNum,
        uint64_t finalSmallTileNum, uint64_t tileDataNum, uint64_t smallTailDataNum, uint64_t bigTailDataNum, uint64_t tailBlockNum, uint64_t haveTanhx);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int32_t progress);
    __aicore__ inline void CopyOut(int32_t progress);
    __aicore__ inline void Compute(int32_t progress);

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueGrad, inQueueX, inQueueTanhx;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueGrad;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpQueue0, tmpQueue1, tmpQueue2, tmpQueue3, tmpQueue4, tmpQueue5, tmpQueueMask1, tmpQueueMask2, tmpQueueMask3, tmpQueueMask4;

    AscendC::GlobalTensor<TYPE_X> gradGm, xGm, tanhxGm, outputGm;
    uint64_t coreDataNum;
    uint64_t tileNum;
    uint64_t tileDataNum;
    uint64_t tailDataNum;
    uint64_t processDataNum;
    uint64_t haveTanhx;
};

template <typename TYPE_X>
__aicore__ inline void KernelMishGrad<TYPE_X>::Init(GM_ADDR grad, GM_ADDR x, GM_ADDR tanhx, GM_ADDR x_grad, uint64_t smallCoreDataNum, uint64_t bigCoreDataNum, uint64_t finalBigTileNum,
    uint64_t finalSmallTileNum, uint64_t tileDataNum, uint64_t smallTailDataNum, uint64_t bigTailDataNum, uint64_t tailBlockNum, uint64_t haveTanhx)
{
    ASSERT(AscendC::GetBlockNum() != 0 && "block dim can not be zero!");
    this->haveTanhx = haveTanhx;
    uint64_t coreId = AscendC::GetBlockIdx();
    uint64_t globalBufferIndex = bigCoreDataNum * coreId;
    this->tileDataNum = tileDataNum;
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
    outputGm.SetGlobalBuffer((__gm__ TYPE_X *)x_grad + globalBufferIndex, this->coreDataNum);
    if (this->haveTanhx == 1) {
        tanhxGm.SetGlobalBuffer((__gm__ TYPE_X *)tanhx + globalBufferIndex, this->coreDataNum);
        pipe.InitBuffer(inQueueTanhx, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_X));
        if constexpr (!std::is_same_v<TYPE_X, float32_t>) {
            pipe.InitBuffer(tmpQueue2, this->tileDataNum * sizeof(float));
            pipe.InitBuffer(tmpQueue3, this->tileDataNum * sizeof(float));
        }
    } else {
        pipe.InitBuffer(tmpQueue2, this->tileDataNum * sizeof(float));
        pipe.InitBuffer(tmpQueue3, this->tileDataNum * sizeof(float));
        pipe.InitBuffer(tmpQueueMask1, this->tileDataNum * sizeof(uint8_t));
        pipe.InitBuffer(tmpQueueMask2, this->tileDataNum * sizeof(uint8_t));
        pipe.InitBuffer(tmpQueueMask3, this->tileDataNum * sizeof(half));
        pipe.InitBuffer(tmpQueueMask4, this->tileDataNum * sizeof(half));
        if constexpr (!std::is_same_v<TYPE_X, float32_t>) {
            pipe.InitBuffer(tmpQueue4, this->tileDataNum * sizeof(float));
            pipe.InitBuffer(tmpQueue5, this->tileDataNum * sizeof(float));
        }
    }
    pipe.InitBuffer(inQueueGrad, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_X));
    pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_X));
    pipe.InitBuffer(outQueueGrad, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_X));
    pipe.InitBuffer(tmpQueue0, this->tileDataNum * sizeof(float));
    pipe.InitBuffer(tmpQueue1, this->tileDataNum * sizeof(float));
}

template <typename TYPE_X>
__aicore__ inline void KernelMishGrad<TYPE_X>::CopyIn(int32_t progress)
{
    AscendC::LocalTensor<TYPE_X> gradLocal = inQueueGrad.AllocTensor<TYPE_X>();
    AscendC::LocalTensor<TYPE_X> xLocal = inQueueX.AllocTensor<TYPE_X>();
    if (this->haveTanhx == 1){
        AscendC::LocalTensor<TYPE_X> tanhxLocal = inQueueTanhx.AllocTensor<TYPE_X>();
        AscendC::DataCopy(tanhxLocal, tanhxGm[progress * this->tileDataNum], this->processDataNum);
        inQueueTanhx.EnQue(tanhxLocal);
    }
    AscendC::DataCopy(gradLocal, gradGm[progress * this->tileDataNum], this->processDataNum);
    AscendC::DataCopy(xLocal, xGm[progress * this->tileDataNum], this->processDataNum);
    inQueueGrad.EnQue(gradLocal);
    inQueueX.EnQue(xLocal);
}

template <typename TYPE_X>
__aicore__ inline void KernelMishGrad<TYPE_X>::CopyOut(int32_t progress)
{
    AscendC::LocalTensor<TYPE_X> outLocal = outQueueGrad.DeQue<TYPE_X>();
    AscendC::DataCopy(outputGm[progress * this->tileDataNum], outLocal, this->processDataNum);
    outQueueGrad.FreeTensor(outLocal);
}

template <typename TYPE_X>
__aicore__ inline void KernelMishGrad<TYPE_X>::Compute(int32_t progress)
{
    AscendC::LocalTensor<float> tmp0Local = tmpQueue0.AllocTensor<float>();
    AscendC::LocalTensor<float> tmp1Local = tmpQueue1.AllocTensor<float>();
    if constexpr (std::is_same_v<TYPE_X, float16_t> || std::is_same_v<TYPE_X, bfloat16_t>) {
        AscendC::LocalTensor<TYPE_X> gradLocal = inQueueGrad.DeQue<TYPE_X>();
        AscendC::LocalTensor<TYPE_X> xLocal = inQueueX.DeQue<TYPE_X>();
        AscendC::LocalTensor<TYPE_X> outLocal = outQueueGrad.AllocTensor<TYPE_X>();
        AscendC::LocalTensor<float> tmp2Local = tmpQueue2.AllocTensor<float>();
        AscendC::LocalTensor<float> tmp3Local = tmpQueue3.AllocTensor<float>();
        if (this->haveTanhx == 0) {
            AscendC::LocalTensor<float> tmp4Local = tmpQueue4.AllocTensor<float>();
            AscendC::LocalTensor<float> tmp5Local = tmpQueue5.AllocTensor<float>();
            AscendC::LocalTensor<uint8_t> mask1Local = tmpQueueMask1.AllocTensor<uint8_t>();
            AscendC::LocalTensor<uint8_t> mask2Local = tmpQueueMask2.AllocTensor<uint8_t>();
            AscendC::LocalTensor<half> tmpMask1Local = tmpQueueMask3.AllocTensor<half>();
            AscendC::LocalTensor<half> tmpMask2Local = tmpQueueMask4.AllocTensor<half>();
            AscendC::Cast(tmp4Local, xLocal, AscendC::RoundMode::CAST_NONE, this->processDataNum);
            AscendC::CompareScalar(mask1Local, tmp4Local, static_cast<float>(0), AscendC::CMPMODE::GT, this->processDataNum);
            AscendC::Muls(tmp1Local, tmp4Local, static_cast<float>(-1.0), this->processDataNum);
            AscendC::Muls(tmp2Local, tmp4Local, static_cast<float>(-2.0), this->processDataNum);
            AscendC::Exp(tmp1Local, tmp1Local, this->processDataNum);
            AscendC::Exp(tmp2Local, tmp2Local, this->processDataNum);
            AscendC::Muls(tmp0Local, tmp1Local, static_cast<float>(2.0), this->processDataNum);
            AscendC::Muls(tmp2Local, tmp2Local, static_cast<float>(2.0), this->processDataNum);
            AscendC::Adds(tmp5Local, tmp0Local, static_cast<float>(1.0), this->processDataNum);
            AscendC::Add(tmp0Local, tmp0Local, tmp2Local, this->processDataNum);
            AscendC::Adds(tmp0Local, tmp0Local, static_cast<float>(1.0), this->processDataNum);
            AscendC::Div(tmp5Local, tmp5Local, tmp0Local, this->processDataNum);
            AscendC::Adds(tmp1Local, tmp1Local, static_cast<float>(1.0), this->processDataNum);
            AscendC::Div(tmp1Local, tmp4Local, tmp1Local, this->processDataNum);
            AscendC::Mul(tmp0Local, tmp5Local, tmp5Local, this->processDataNum);
            AscendC::Duplicate(tmp3Local, static_cast<float>(1.0), this->processDataNum);
            AscendC::Sub(tmp0Local, tmp3Local, tmp0Local, this->processDataNum);
            AscendC::Mul(tmp0Local, tmp1Local, tmp0Local, this->processDataNum);
            AscendC::Add(tmp0Local, tmp5Local, tmp0Local, this->processDataNum);
            AscendC::Select(tmp5Local, mask1Local, tmp0Local, static_cast<float>(0), AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, this->processDataNum);
            AscendC::CompareScalar(mask2Local, tmp4Local, static_cast<float>(0), AscendC::CMPMODE::LE, this->processDataNum);
            AscendC::Muls(tmp0Local, tmp4Local, static_cast<float>(2.0), this->processDataNum);
            AscendC::Exp(tmp1Local, tmp4Local, this->processDataNum);
            AscendC::Muls(tmp2Local, tmp1Local, static_cast<float>(2.0), this->processDataNum);
            AscendC::Exp(tmp0Local, tmp0Local, this->processDataNum);
            AscendC::Add(tmp0Local, tmp2Local, tmp0Local, this->processDataNum);
            AscendC::Adds(tmp2Local, tmp0Local, static_cast<float>(2.0), this->processDataNum);
            AscendC::Div(tmp2Local, tmp0Local, tmp2Local, this->processDataNum);
            AscendC::Mul(tmp0Local, tmp4Local, tmp1Local, this->processDataNum);
            AscendC::Adds(tmp1Local, tmp1Local, static_cast<float>(1.0), this->processDataNum);
            AscendC::Div(tmp0Local, tmp0Local, tmp1Local, this->processDataNum);
            AscendC::Mul(tmp1Local, tmp2Local, tmp2Local, this->processDataNum);
            AscendC::Sub(tmp1Local, tmp3Local, tmp1Local, this->processDataNum);
            AscendC::Mul(tmp1Local, tmp0Local, tmp1Local, this->processDataNum);
            AscendC::Add(tmp1Local, tmp2Local, tmp1Local, this->processDataNum);
            AscendC::Select(tmp1Local, mask2Local, tmp1Local, static_cast<float>(0), AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, this->processDataNum);
            AscendC::Add(tmp5Local, tmp5Local, tmp1Local, this->processDataNum);
            AscendC::Cast(tmpMask1Local, mask1Local, AscendC::RoundMode::CAST_NONE, this->processDataNum);
            AscendC::Cast(tmpMask2Local, mask2Local, AscendC::RoundMode::CAST_NONE, this->processDataNum);
            AscendC::Add(tmpMask2Local, tmpMask1Local, tmpMask2Local, this->processDataNum);
            AscendC::Cast(mask2Local, tmpMask2Local, AscendC::RoundMode::CAST_RINT, this->processDataNum);
            AscendC::Select(tmp5Local, mask2Local, tmp5Local, tmp4Local, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, this->processDataNum);
            AscendC::Cast(tmp4Local, gradLocal, AscendC::RoundMode::CAST_NONE, this->processDataNum);
            AscendC::Mul(tmp5Local, tmp5Local, tmp4Local, this->processDataNum);
            AscendC::Cast(outLocal, tmp5Local, AscendC::RoundMode::CAST_RINT, this->processDataNum);
        } else {
            AscendC::LocalTensor<TYPE_X> tanhxLocal = inQueueTanhx.DeQue<TYPE_X>();
            AscendC::Cast(tmp0Local, xLocal, AscendC::RoundMode::CAST_NONE, this->processDataNum);
            AscendC::Exp(tmp1Local, tmp0Local, this->processDataNum);
            AscendC::Duplicate(tmp2Local, static_cast<float>(1.0), this->processDataNum);
            AscendC::Adds(tmp3Local, tmp1Local, static_cast<float>(1.0), this->processDataNum);
            AscendC::Div(tmp2Local, tmp2Local, tmp3Local, this->processDataNum);
            AscendC::Mul(tmp1Local, tmp0Local, tmp1Local, this->processDataNum);
            AscendC::Cast(tmp0Local, tanhxLocal, AscendC::RoundMode::CAST_NONE, this->processDataNum);
            AscendC::Mul(tmp3Local, tmp0Local, tmp0Local, this->processDataNum);
            AscendC::Adds(tmp3Local, tmp3Local, static_cast<float>(-1.0), this->processDataNum);
            AscendC::Mul(tmp1Local, tmp1Local, tmp3Local, this->processDataNum);
            AscendC::Mul(tmp1Local, tmp1Local, tmp2Local, this->processDataNum);
            AscendC::Sub(tmp1Local, tmp0Local, tmp1Local, this->processDataNum);
            AscendC::Cast(tmp0Local, gradLocal, AscendC::RoundMode::CAST_NONE, this->processDataNum);
            AscendC::Mul(tmp1Local, tmp1Local, tmp0Local, this->processDataNum);
            AscendC::Cast(outLocal, tmp1Local, AscendC::RoundMode::CAST_RINT, this->processDataNum);
            inQueueTanhx.FreeTensor(tanhxLocal);
        }
        outQueueGrad.EnQue<TYPE_X>(outLocal);
        inQueueGrad.FreeTensor(gradLocal);
        inQueueX.FreeTensor(xLocal);
    } else {
        AscendC::LocalTensor<float> gradLocal = inQueueGrad.DeQue<float>();
        AscendC::LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        AscendC::LocalTensor<float> outLocal = outQueueGrad.AllocTensor<float>();
        if (this->haveTanhx == 0) {
            AscendC::LocalTensor<float> tmp2Local = tmpQueue2.AllocTensor<float>();
            AscendC::LocalTensor<float> tmp3Local = tmpQueue3.AllocTensor<float>();
            AscendC::LocalTensor<uint8_t> mask1Local = tmpQueueMask1.AllocTensor<uint8_t>();
            AscendC::LocalTensor<uint8_t> mask2Local = tmpQueueMask2.AllocTensor<uint8_t>();
            AscendC::LocalTensor<half> tmpMask1Local = tmpQueueMask3.AllocTensor<half>();
            AscendC::LocalTensor<half> tmpMask2Local = tmpQueueMask4.AllocTensor<half>();
            AscendC::CompareScalar(mask1Local, xLocal, static_cast<float>(0), AscendC::CMPMODE::GT, this->processDataNum);
            AscendC::Muls(tmp1Local, xLocal, static_cast<float>(-1.0), this->processDataNum);
            AscendC::Muls(tmp2Local, xLocal, static_cast<float>(-2.0), this->processDataNum);
            AscendC::Exp(tmp1Local, tmp1Local, this->processDataNum);
            AscendC::Exp(tmp2Local, tmp2Local, this->processDataNum);
            AscendC::Muls(tmp0Local, tmp1Local, static_cast<float>(2.0), this->processDataNum);
            AscendC::Muls(tmp2Local, tmp2Local, static_cast<float>(2.0), this->processDataNum);
            AscendC::Adds(outLocal, tmp0Local, static_cast<float>(1.0), this->processDataNum);
            AscendC::Add(tmp0Local, tmp0Local, tmp2Local, this->processDataNum);
            AscendC::Adds(tmp0Local, tmp0Local, static_cast<float>(1.0), this->processDataNum);
            AscendC::Div(outLocal, outLocal, tmp0Local, this->processDataNum);
            AscendC::Adds(tmp1Local, tmp1Local, static_cast<float>(1.0), this->processDataNum);
            AscendC::Div(tmp1Local, xLocal, tmp1Local, this->processDataNum);
            AscendC::Mul(tmp0Local, outLocal, outLocal, this->processDataNum);
            AscendC::Duplicate(tmp3Local, static_cast<float>(1.0), this->processDataNum);
            AscendC::Sub(tmp0Local, tmp3Local, tmp0Local, this->processDataNum);
            AscendC::Mul(tmp0Local, tmp1Local, tmp0Local, this->processDataNum);
            AscendC::Add(tmp0Local, outLocal, tmp0Local, this->processDataNum);
            AscendC::Select(outLocal, mask1Local, tmp0Local, static_cast<float>(0), AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, this->processDataNum);
            AscendC::CompareScalar(mask2Local, xLocal, static_cast<float>(0), AscendC::CMPMODE::LE, this->processDataNum);
            AscendC::Muls(tmp0Local, xLocal, static_cast<float>(2.0), this->processDataNum);
            AscendC::Exp(tmp1Local, xLocal, this->processDataNum);
            AscendC::Muls(tmp2Local, tmp1Local, static_cast<float>(2.0), this->processDataNum);
            AscendC::Exp(tmp0Local, tmp0Local, this->processDataNum);
            AscendC::Add(tmp0Local, tmp2Local, tmp0Local, this->processDataNum);
            AscendC::Adds(tmp2Local, tmp0Local, static_cast<float>(2.0), this->processDataNum);
            AscendC::Div(tmp2Local, tmp0Local, tmp2Local, this->processDataNum);
            AscendC::Mul(tmp0Local, xLocal, tmp1Local, this->processDataNum);
            AscendC::Adds(tmp1Local, tmp1Local, static_cast<float>(1.0), this->processDataNum);
            AscendC::Div(tmp0Local, tmp0Local, tmp1Local, this->processDataNum);
            AscendC::Mul(tmp1Local, tmp2Local, tmp2Local, this->processDataNum);
            AscendC::Sub(tmp1Local, tmp3Local, tmp1Local, this->processDataNum);
            AscendC::Mul(tmp1Local, tmp0Local, tmp1Local, this->processDataNum);
            AscendC::Add(tmp1Local, tmp2Local, tmp1Local, this->processDataNum);
            AscendC::Select(tmp1Local, mask2Local, tmp1Local, static_cast<float>(0), AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, this->processDataNum);
            AscendC::Add(outLocal, outLocal, tmp1Local, this->processDataNum);
            AscendC::Cast(tmpMask1Local, mask1Local, AscendC::RoundMode::CAST_NONE, this->processDataNum);
            AscendC::Cast(tmpMask2Local, mask2Local, AscendC::RoundMode::CAST_NONE, this->processDataNum);
            AscendC::Add(tmpMask2Local, tmpMask1Local, tmpMask2Local, this->processDataNum);
            AscendC::Cast(mask2Local, tmpMask2Local, AscendC::RoundMode::CAST_RINT, this->processDataNum);
            AscendC::Select(outLocal, mask2Local, outLocal, xLocal, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, this->processDataNum);
            AscendC::Mul(outLocal, outLocal, gradLocal, this->processDataNum);
        } else {
            AscendC::LocalTensor<float> tanhxLocal = inQueueTanhx.DeQue<float>();
            AscendC::Exp(tmp0Local, xLocal, this->processDataNum);
            AscendC::Duplicate(tmp1Local, static_cast<float>(1.0), this->processDataNum);
            AscendC::Adds(outLocal, tmp0Local, static_cast<float>(1.0), this->processDataNum);
            AscendC::Div(tmp1Local, tmp1Local, outLocal, this->processDataNum);
            AscendC::Mul(outLocal, tanhxLocal, tanhxLocal, this->processDataNum);
            AscendC::Adds(outLocal, outLocal, static_cast<float>(-1.0), this->processDataNum);
            AscendC::Mul(tmp0Local, xLocal, tmp0Local, this->processDataNum);
            AscendC::Mul(outLocal, tmp0Local, outLocal, this->processDataNum);
            AscendC::Mul(outLocal, outLocal, tmp1Local, this->processDataNum);
            AscendC::Sub(outLocal, tanhxLocal, outLocal, this->processDataNum);
            AscendC::Mul(outLocal, outLocal, gradLocal, this->processDataNum);
            inQueueTanhx.FreeTensor(tanhxLocal);
        }
        outQueueGrad.EnQue<float>(outLocal);
        inQueueGrad.FreeTensor(gradLocal);
        inQueueX.FreeTensor(xLocal);
    }
}

template <typename TYPE_X>
__aicore__ inline void KernelMishGrad<TYPE_X>::Process()
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

} // namespace NsMishGrad
#endif // MISH_GRAD_H
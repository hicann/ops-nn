/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2026 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Zhou Jianhua <@LePenseur>
 * - Su Tonghua <@sutonghua>
 *
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file tanh_grad.h
 * \brief
 */
#ifndef TANH_GRAD_H
#define TANH_GRAD_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "tanh_grad_tiling_data.h"
#include "tanh_grad_tiling_key.h"

namespace NsTanhGrad {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

template <typename TYPE_GRADIENTS>
class KernelTanhGrad {
public:
    __aicore__ inline KernelTanhGrad(){};

    __aicore__ inline void Init(GM_ADDR y, GM_ADDR dy, GM_ADDR dx, uint64_t smallCoreDataNum, uint64_t bigCoreDataNum,
                                uint64_t finalBigTileNum, uint64_t finalSmallTileNum, uint64_t tileDataNum,
                                uint64_t smallTailDataNum, uint64_t bigTailDataNum, uint64_t tailBlockNum);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int32_t progress);
    __aicore__ inline void CopyOut(int32_t progress);
    __aicore__ inline void Compute(int32_t progress);

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueY, inQueueDY;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueDX;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf1, tmpBuf2;

    AscendC::GlobalTensor<TYPE_GRADIENTS> yGm, dyGm, dxGm;
    uint64_t coreDataNum;
    uint64_t tileNum;
    uint64_t tileDataNum;
    uint64_t tailDataNum;
    uint64_t processDataNum;
};

template <typename TYPE_GRADIENTS>
__aicore__ inline void KernelTanhGrad<TYPE_GRADIENTS>::Init(GM_ADDR y, GM_ADDR dy, GM_ADDR dx,
                                                            uint64_t smallCoreDataNum, uint64_t bigCoreDataNum,
                                                            uint64_t finalBigTileNum, uint64_t finalSmallTileNum,
                                                            uint64_t tileDataNum, uint64_t smallTailDataNum,
                                                            uint64_t bigTailDataNum, uint64_t tailBlockNum)
{
    ASSERT(AscendC::GetBlockNum() != 0 && "block dim can not be zero!");
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
    yGm.SetGlobalBuffer((__gm__ TYPE_GRADIENTS*)y + globalBufferIndex, this->coreDataNum);
    dyGm.SetGlobalBuffer((__gm__ TYPE_GRADIENTS*)dy + globalBufferIndex, this->coreDataNum);
    dxGm.SetGlobalBuffer((__gm__ TYPE_GRADIENTS*)dx + globalBufferIndex, this->coreDataNum);

    if constexpr (std::is_same_v<TYPE_GRADIENTS, half> || std::is_same_v<TYPE_GRADIENTS, bfloat16_t>) {
        pipe.InitBuffer(tmpBuf1, this->tileDataNum * sizeof(float));
        pipe.InitBuffer(tmpBuf2, this->tileDataNum * sizeof(float));
    }
    pipe.InitBuffer(inQueueY, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_GRADIENTS));
    pipe.InitBuffer(inQueueDY, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_GRADIENTS));
    pipe.InitBuffer(outQueueDX, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_GRADIENTS));
}

template <typename TYPE_GRADIENTS>
__aicore__ inline void KernelTanhGrad<TYPE_GRADIENTS>::CopyIn(int32_t progress)
{
    AscendC::LocalTensor<TYPE_GRADIENTS> yLocal = inQueueY.AllocTensor<TYPE_GRADIENTS>();
    AscendC::LocalTensor<TYPE_GRADIENTS> dyLocal = inQueueDY.AllocTensor<TYPE_GRADIENTS>();
    AscendC::DataCopy(yLocal, yGm[progress * this->tileDataNum], this->processDataNum);
    AscendC::DataCopy(dyLocal, dyGm[progress * this->tileDataNum], this->processDataNum);
    inQueueY.EnQue(yLocal);
    inQueueDY.EnQue(dyLocal);
}

template <typename TYPE_GRADIENTS>
__aicore__ inline void KernelTanhGrad<TYPE_GRADIENTS>::CopyOut(int32_t progress)
{
    AscendC::LocalTensor<TYPE_GRADIENTS> dxLocal = outQueueDX.DeQue<TYPE_GRADIENTS>();
    AscendC::DataCopy(dxGm[progress * this->tileDataNum], dxLocal, this->processDataNum);
    outQueueDX.FreeTensor(dxLocal);
}

template <typename TYPE_GRADIENTS>
__aicore__ inline void KernelTanhGrad<TYPE_GRADIENTS>::Compute(int32_t progress)
{
    AscendC::LocalTensor<TYPE_GRADIENTS> yLocal = inQueueY.DeQue<TYPE_GRADIENTS>();
    AscendC::LocalTensor<TYPE_GRADIENTS> dyLocal = inQueueDY.DeQue<TYPE_GRADIENTS>();
    AscendC::LocalTensor<TYPE_GRADIENTS> dxLocal = outQueueDX.AllocTensor<TYPE_GRADIENTS>();

    // half/bfloat16 提升到 fp32 计算并 CAST_RINT 回写（精度）；fp32 原生计算
    if constexpr (std::is_same_v<TYPE_GRADIENTS, half> || std::is_same_v<TYPE_GRADIENTS, bfloat16_t>) {
        AscendC::LocalTensor<float> tmp1Local = tmpBuf1.Get<float>();
        AscendC::LocalTensor<float> tmp2Local = tmpBuf2.Get<float>();
        AscendC::Cast(tmp1Local, yLocal, AscendC::RoundMode::CAST_RINT, this->processDataNum);
        AscendC::Cast(tmp2Local, dyLocal, AscendC::RoundMode::CAST_RINT, this->processDataNum);
        AscendC::Mul(tmp1Local, tmp1Local, tmp1Local, this->processDataNum); // y*y
        AscendC::Muls(tmp1Local, tmp1Local, -1.0f, this->processDataNum);    // -(y*y)
        AscendC::Adds(tmp1Local, tmp1Local, 1.0f, this->processDataNum);     // 1 - y*y
        AscendC::Mul(tmp1Local, tmp2Local, tmp1Local, this->processDataNum); // dy*(1-y*y)
        AscendC::Cast(dxLocal, tmp1Local, AscendC::RoundMode::CAST_RINT, this->processDataNum);
    } else {
        // float32: compute directly in native type
        AscendC::Mul(dxLocal, yLocal, yLocal, this->processDataNum);
        AscendC::Muls(dxLocal, dxLocal, static_cast<TYPE_GRADIENTS>(-1), this->processDataNum);
        AscendC::Adds(dxLocal, dxLocal, static_cast<TYPE_GRADIENTS>(1), this->processDataNum);
        AscendC::Mul(dxLocal, dyLocal, dxLocal, this->processDataNum);
    }

    outQueueDX.EnQue<TYPE_GRADIENTS>(dxLocal);
    inQueueY.FreeTensor(yLocal);
    inQueueDY.FreeTensor(dyLocal);
}

template <typename TYPE_GRADIENTS>
__aicore__ inline void KernelTanhGrad<TYPE_GRADIENTS>::Process()
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

} // namespace NsTanhGrad
#endif // TANH_GRAD_H

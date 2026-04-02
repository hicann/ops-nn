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
 * \file softplus.h
 * \brief
 */
#ifndef SOFTPLUS_H
#define SOFTPLUS_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "softplus_tiling_data.h"
#include "softplus_tiling_key.h"

#include "kernel_operator.h"

namespace NsSoftplus {

using namespace AscendC;
constexpr float MAGIC_NUM = -0.69314718055994530941723212145818f;
constexpr int32_t SINGLE_BUFFER_NUM = 1;
constexpr int32_t DOUBLE_BUFFER_NUM = 2;

template <typename TYPE_X>
class KernelSoftplus {
public:
    __aicore__ inline KernelSoftplus(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint64_t smallCoreDataNum, uint64_t bigCoreDataNum, uint64_t finalBigTileNum,
        uint64_t finalSmallTileNum, uint64_t tileDataNum, uint64_t smallTailDataNum, uint64_t bigTailDataNum, uint64_t tailBlockNum, uint64_t bufferOpen);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int32_t progress);
    __aicore__ inline void CopyOut(int32_t progress);
    __aicore__ inline void Compute(int32_t progress);

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, DOUBLE_BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, DOUBLE_BUFFER_NUM> outQueueY;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpQueue0, tmpQueue1, tmpQueue2;

    AscendC::GlobalTensor<TYPE_X> xGm, yGm;
    uint64_t coreDataNum;
    uint64_t tileNum;
    uint64_t tileDataNum;
    uint64_t tailDataNum;
    uint64_t processDataNum;
};

template <typename TYPE_X>
__aicore__ inline void KernelSoftplus<TYPE_X>::Init(GM_ADDR x, GM_ADDR y, uint64_t smallCoreDataNum, uint64_t bigCoreDataNum, uint64_t finalBigTileNum,
    uint64_t finalSmallTileNum, uint64_t tileDataNum, uint64_t smallTailDataNum, uint64_t bigTailDataNum, uint64_t tailBlockNum, uint64_t bufferOpen)
{
    ASSERT(AscendC::GetBlockNum() != 0 && "block dim can not be zero!");
    uint64_t coreId = AscendC::GetBlockIdx();
    uint64_t globalBufferIndex = bigCoreDataNum * coreId;
    this->tileDataNum = tileDataNum;
    // default open DOUBLE BUFFER
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
    xGm.SetGlobalBuffer((__gm__ TYPE_X *)x + globalBufferIndex, this->coreDataNum);
    yGm.SetGlobalBuffer((__gm__ TYPE_X *)y + globalBufferIndex, this->coreDataNum);

    pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_X));
    pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_X));
    if (std::is_same_v<TYPE_X, float>) {
        pipe.InitBuffer(tmpQueue0, this->tileDataNum * sizeof(float));
    } else {
        pipe.InitBuffer(tmpQueue0, this->tileDataNum * sizeof(float));
        pipe.InitBuffer(tmpQueue1, this->tileDataNum * sizeof(float));
        pipe.InitBuffer(tmpQueue2, this->tileDataNum * sizeof(float));
    }
}

template <typename TYPE_X>
__aicore__ inline void KernelSoftplus<TYPE_X>::CopyIn(int32_t progress)
{
    AscendC::LocalTensor<TYPE_X> xLocal = inQueueX.AllocTensor<TYPE_X>();
    AscendC::DataCopy(xLocal, xGm[progress * this->tileDataNum], this->processDataNum);
    inQueueX.EnQue(xLocal);
}

template <typename TYPE_X>
__aicore__ inline void KernelSoftplus<TYPE_X>::CopyOut(int32_t progress)
{
    AscendC::LocalTensor<TYPE_X> yLocal = outQueueY.DeQue<TYPE_X>();
    AscendC::DataCopy(yGm[progress * this->tileDataNum], yLocal, this->processDataNum);
    outQueueY.FreeTensor(yLocal);
}

template <typename TYPE_X>
__aicore__ inline void KernelSoftplus<TYPE_X>::Compute(int32_t progress)
{
    if constexpr (std::is_same_v<TYPE_X, float>) {
        //float32
        AscendC::LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        AscendC::LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
        AscendC::LocalTensor<float> tmp0Local = tmpQueue0.AllocTensor<float>();
        AscendC::Maxs(tmp0Local, xLocal, static_cast<float>(0.0), this->processDataNum);
        AscendC::Muls(yLocal, tmp0Local, static_cast<float>(-1.0), this->processDataNum);
        AscendC::Exp(yLocal, yLocal, this->processDataNum);
        AscendC::Adds(yLocal, yLocal, static_cast<float>(1.0), this->processDataNum);
        AscendC::Ln(yLocal, yLocal, this->processDataNum);
        AscendC::Add(yLocal, yLocal, tmp0Local, this->processDataNum);
        AscendC::Mins(tmp0Local, xLocal, static_cast<float>(0.0), this->processDataNum);
        AscendC::Exp(tmp0Local, tmp0Local, this->processDataNum);
        AscendC::Adds(tmp0Local, tmp0Local, static_cast<float>(1.0), this->processDataNum);
        AscendC::Ln(tmp0Local, tmp0Local, this->processDataNum);
        AscendC::Add(yLocal, yLocal, tmp0Local, this->processDataNum);
        AscendC::Adds(yLocal, yLocal, static_cast<float>(MAGIC_NUM), this->processDataNum);
        outQueueY.EnQue<float>(yLocal);
        inQueueX.FreeTensor(xLocal);
    } else {
        // float16 or bfloat16
        // xLocal---tmp1Local tmp2Local---yLocal
        AscendC::LocalTensor<TYPE_X> xLocal = inQueueX.DeQue<TYPE_X>();
        AscendC::LocalTensor<TYPE_X> yLocal = outQueueY.AllocTensor<TYPE_X>();
        AscendC::LocalTensor<float> tmp0Local = tmpQueue0.AllocTensor<float>();
        AscendC::LocalTensor<float> tmp1Local = tmpQueue1.AllocTensor<float>();
        AscendC::LocalTensor<float> tmp2Local = tmpQueue2.AllocTensor<float>();
        AscendC::Cast(tmp1Local, xLocal, AscendC::RoundMode::CAST_NONE, this->processDataNum);
        AscendC::Maxs(tmp0Local, tmp1Local, static_cast<float>(0.0), this->processDataNum);
        AscendC::Muls(tmp2Local, tmp0Local, static_cast<float>(-1.0), this->processDataNum);
        AscendC::Exp(tmp2Local, tmp2Local, this->processDataNum);
        AscendC::Adds(tmp2Local, tmp2Local, static_cast<float>(1.0), this->processDataNum);
        AscendC::Ln(tmp2Local, tmp2Local, this->processDataNum);
        AscendC::Add(tmp2Local, tmp2Local, tmp0Local, this->processDataNum);
        AscendC::Mins(tmp0Local, tmp1Local, static_cast<float>(0.0), this->processDataNum);
        AscendC::Exp(tmp0Local, tmp0Local, this->processDataNum);
        AscendC::Adds(tmp0Local, tmp0Local, static_cast<float>(1.0), this->processDataNum);
        AscendC::Ln(tmp0Local, tmp0Local, this->processDataNum);
        AscendC::Add(tmp2Local, tmp2Local, tmp0Local, this->processDataNum);
        AscendC::Adds(tmp2Local, tmp2Local, static_cast<float>(MAGIC_NUM), this->processDataNum);
        AscendC::Cast(yLocal, tmp2Local, AscendC::RoundMode::CAST_RINT, this->processDataNum);
        outQueueY.EnQue<TYPE_X>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }
}

template <typename TYPE_X>
__aicore__ inline void KernelSoftplus<TYPE_X>::Process()
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

} // namespace NsSoftplus
#endif // SOFTPLUS_H
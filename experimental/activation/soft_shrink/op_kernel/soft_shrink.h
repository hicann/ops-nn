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
* \file soft_shrink.h
* \brief
*/
#ifndef SOFT_SHRINK_H
#define SOFT_SHRINK_H
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "soft_shrink_tiling_data.h"
#include "soft_shrink_tiling_key.h"
#include "kernel_operator.h"
namespace NsSoftShrink {
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;
template <typename TYPE_X>
class KernelSoftShrink {
public:
    __aicore__ inline KernelSoftShrink(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint64_t smallCoreDataNum, uint64_t bigCoreDataNum, uint64_t finalBigTileNum,
        uint64_t finalSmallTileNum, uint64_t tileDataNum, uint64_t smallTailDataNum, uint64_t bigTailDataNum, uint64_t tailBlockNum, float lambd);
    __aicore__ inline void Process();
private:
    __aicore__ inline void CopyIn(int32_t progress);
    __aicore__ inline void CopyOut(int32_t progress);
    __aicore__ inline void Compute(int32_t progress);
private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpQueue0;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpQueue1;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpQueue2;
    AscendC::TBuf<AscendC::TPosition::VECCALC> castTemp;
    AscendC::TBuf<AscendC::TPosition::VECCALC> lambdQueue;
    AscendC::TBuf<AscendC::TPosition::VECCALC> maskQueue;
    AscendC::GlobalTensor<TYPE_X> xGm;
    AscendC::GlobalTensor<TYPE_X> yGm;
    uint64_t coreDataNum = 0;
    uint64_t tileNum = 0;
    uint64_t tileDataNum = 0;
    uint64_t tailDataNum = 0;
    uint64_t processDataNum = 0;
    float lambd = 0.5f;
};
template <typename TYPE_X>
__aicore__ inline void KernelSoftShrink<TYPE_X>::Init(GM_ADDR x, GM_ADDR y, uint64_t smallCoreDataNum, uint64_t bigCoreDataNum, uint64_t finalBigTileNum,
    uint64_t finalSmallTileNum, uint64_t tileDataNum, uint64_t smallTailDataNum, uint64_t bigTailDataNum, uint64_t tailBlockNum, float lambd)
{
    ASSERT(AscendC::GetBlockNum() != 0 && "block dim can not be zero!");
    uint64_t coreId = AscendC::GetBlockIdx();
    uint64_t globalBufferIndex = bigCoreDataNum * coreId;
    this->tileDataNum = tileDataNum;
    this->lambd = lambd;
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
    if constexpr (std::is_same_v<TYPE_X, float32_t>) {
        pipe.InitBuffer(tmpQueue0, this->tileDataNum * sizeof(float));
        pipe.InitBuffer(tmpQueue1, this->tileDataNum * sizeof(float));
        pipe.InitBuffer(tmpQueue2, this->tileDataNum * sizeof(float));
        pipe.InitBuffer(maskQueue, (this->tileDataNum * sizeof(uint8_t) + 255) / 256 * 256 / 8);
        pipe.InitBuffer(lambdQueue, this->tileDataNum * sizeof(float));
    } else if constexpr (std::is_same_v<TYPE_X, bfloat16_t>) {
        pipe.InitBuffer(castTemp, this->tileDataNum * sizeof(float));
        pipe.InitBuffer(tmpQueue0, this->tileDataNum * sizeof(float));
        pipe.InitBuffer(tmpQueue1, this->tileDataNum * sizeof(float));
        pipe.InitBuffer(tmpQueue2, this->tileDataNum * sizeof(float));
        pipe.InitBuffer(maskQueue, (this->tileDataNum * sizeof(uint8_t) + 255) / 256 * 256 / 8);
        pipe.InitBuffer(lambdQueue, this->tileDataNum * sizeof(float));
    } else if constexpr (std::is_same_v<TYPE_X, half>) {
    #if defined(HIGH_PRECISION) && HIGH_PRECISION == 1
        pipe.InitBuffer(castTemp, this->tileDataNum * sizeof(float));
        pipe.InitBuffer(tmpQueue0, this->tileDataNum * sizeof(float));
        pipe.InitBuffer(tmpQueue1, this->tileDataNum * sizeof(float));
        pipe.InitBuffer(tmpQueue2, this->tileDataNum * sizeof(float));
        pipe.InitBuffer(maskQueue, (this->tileDataNum * sizeof(uint8_t) + 255) / 256 * 256 / 8);
        pipe.InitBuffer(lambdQueue, this->tileDataNum * sizeof(float));
    #else
        pipe.InitBuffer(tmpQueue0, this->tileDataNum * sizeof(TYPE_X));
        pipe.InitBuffer(tmpQueue1, this->tileDataNum * sizeof(TYPE_X));
        pipe.InitBuffer(tmpQueue2, this->tileDataNum * sizeof(TYPE_X));
        pipe.InitBuffer(maskQueue, (this->tileDataNum * sizeof(uint8_t) + 255) / 256 * 256 / 8);
        pipe.InitBuffer(lambdQueue, this->tileDataNum * sizeof(TYPE_X));
    #endif
    }
}
template <typename TYPE_X>
__aicore__ inline void KernelSoftShrink<TYPE_X>::CopyIn(int32_t progress)
{
    AscendC::LocalTensor<TYPE_X> xLocal = inQueueX.AllocTensor<TYPE_X>();
    AscendC::DataCopy(xLocal, xGm[progress * this->tileDataNum], this->processDataNum);
    inQueueX.EnQue(xLocal);
}
template <typename TYPE_X>
__aicore__ inline void KernelSoftShrink<TYPE_X>::CopyOut(int32_t progress)
{
    AscendC::LocalTensor<TYPE_X> yLocal = outQueueY.DeQue<TYPE_X>();
    AscendC::DataCopy(yGm[progress * this->tileDataNum], yLocal, this->processDataNum);
    outQueueY.FreeTensor(yLocal);
}
template <typename TYPE_X>
__aicore__ inline void KernelSoftShrink<TYPE_X>::Compute(int32_t progress)
{
    if constexpr (std::is_same_v<TYPE_X, bfloat16_t>) {
        AscendC::LocalTensor<TYPE_X> xLocal = inQueueX.DeQue<TYPE_X>();
        AscendC::LocalTensor<TYPE_X> yLocal = outQueueY.AllocTensor<TYPE_X>();
        AscendC::LocalTensor<float> castLocal = castTemp.AllocTensor<float>();
        AscendC::LocalTensor<float> lambdTensor = lambdQueue.AllocTensor<float>();
        AscendC::LocalTensor<float> tmp0Local = tmpQueue0.AllocTensor<float>();
        AscendC::LocalTensor<float> tmp1Local = tmpQueue1.AllocTensor<float>();
        AscendC::LocalTensor<float> tmp2Local = tmpQueue2.AllocTensor<float>();
        AscendC::LocalTensor<uint8_t> maskLocal = maskQueue.AllocTensor<uint8_t>();
        AscendC::Cast(castLocal, xLocal, AscendC::RoundMode::CAST_NONE, this->processDataNum);
        AscendC::Abs(tmp0Local, castLocal, this->processDataNum);
        AscendC::Duplicate(lambdTensor, static_cast<float>(this->lambd), this->processDataNum);
        AscendC::CompareScalar(maskLocal, tmp0Local, static_cast<float>(this->lambd), AscendC::CMPMODE::GT, (this->processDataNum+255)/256*256);
        AscendC::Select(tmp1Local, maskLocal, castLocal, static_cast<float>(0.0), AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, this->processDataNum);
        AscendC::Sub(tmp0Local, tmp1Local, lambdTensor, this->processDataNum);
        AscendC::CompareScalar(maskLocal, tmp1Local, static_cast<float>(this->lambd), AscendC::CMPMODE::GT, (this->processDataNum+255)/256*256);
        AscendC::Select(tmp1Local, maskLocal, tmp0Local, tmp1Local, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, this->processDataNum);
        AscendC::Add(tmp2Local, tmp1Local, lambdTensor, this->processDataNum);
        AscendC::Muls(lambdTensor, lambdTensor, static_cast<float>(-1.0), this->processDataNum);
        AscendC::Compare(maskLocal, tmp1Local, lambdTensor, AscendC::CMPMODE::LT, (this->processDataNum+255)/256*256);
        AscendC::Select(tmp0Local, maskLocal, tmp2Local, tmp1Local, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, this->processDataNum);
        AscendC::Cast(yLocal, tmp0Local, AscendC::RoundMode::CAST_RINT, this->processDataNum);
        outQueueY.EnQue<TYPE_X>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }else if constexpr (std::is_same_v<TYPE_X, half>) {
        #if defined(HIGH_PERFORMANCE) && HIGH_PERFORMANCE == 1
            AscendC::LocalTensor<TYPE_X> xLocal = inQueueX.DeQue<TYPE_X>();
            AscendC::LocalTensor<TYPE_X> yLocal = outQueueY.AllocTensor<TYPE_X>();
            AscendC::LocalTensor<TYPE_X> lambdTensor = lambdQueue.AllocTensor<TYPE_X>();
            AscendC::LocalTensor<TYPE_X> tmp0Local = tmpQueue0.AllocTensor<TYPE_X>();
            AscendC::LocalTensor<TYPE_X> tmp1Local = tmpQueue1.AllocTensor<TYPE_X>();
            AscendC::LocalTensor<TYPE_X> tmp2Local = tmpQueue2.AllocTensor<TYPE_X>();
            AscendC::LocalTensor<uint8_t> maskLocal = maskQueue.AllocTensor<uint8_t>();
            AscendC::Abs(tmp0Local, xLocal, this->processDataNum);
            AscendC::Duplicate(lambdTensor, static_cast<half>(this->lambd), this->processDataNum);
            AscendC::CompareScalar(maskLocal, tmp0Local, static_cast<half>(this->lambd), AscendC::CMPMODE::GT, (this->processDataNum+255)/256*256);
            AscendC::Select(tmp1Local, maskLocal, xLocal, static_cast<half>(0.0), AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, this->processDataNum);
            AscendC::Sub(tmp0Local, tmp1Local, lambdTensor, this->processDataNum);
            AscendC::CompareScalar(maskLocal, tmp1Local, static_cast<half>(this->lambd), AscendC::CMPMODE::GT, (this->processDataNum+255)/256*256);
            AscendC::Select(tmp1Local, maskLocal, tmp0Local, tmp1Local, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, this->processDataNum);
            AscendC::Add(tmp2Local, tmp1Local, lambdTensor, this->processDataNum);
            AscendC::Muls(lambdTensor, lambdTensor, static_cast<half>(-1.0), this->processDataNum);
            AscendC::Compare(maskLocal, tmp1Local, lambdTensor, AscendC::CMPMODE::LT, (this->processDataNum+255)/256*256);
            AscendC::Select(yLocal, maskLocal, tmp2Local, tmp1Local, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, this->processDataNum);
            outQueueY.EnQue<TYPE_X>(yLocal);
            inQueueX.FreeTensor(xLocal);
        #else
            AscendC::LocalTensor<TYPE_X> xLocal = inQueueX.DeQue<TYPE_X>();
            AscendC::LocalTensor<TYPE_X> yLocal = outQueueY.AllocTensor<TYPE_X>();
            AscendC::LocalTensor<float> castLocal = castTemp.AllocTensor<float>();
            AscendC::LocalTensor<float> lambdTensor = lambdQueue.AllocTensor<float>();
            AscendC::LocalTensor<float> tmp0Local = tmpQueue0.AllocTensor<float>();
            AscendC::LocalTensor<float> tmp1Local = tmpQueue1.AllocTensor<float>();
            AscendC::LocalTensor<float> tmp2Local = tmpQueue2.AllocTensor<float>();
            AscendC::LocalTensor<uint8_t> maskLocal = maskQueue.AllocTensor<uint8_t>();
            AscendC::Cast(castLocal, xLocal, AscendC::RoundMode::CAST_NONE, this->processDataNum);
            AscendC::Abs(tmp0Local, castLocal, this->processDataNum);
            AscendC::Duplicate(lambdTensor, static_cast<float>(this->lambd), this->processDataNum);
            AscendC::CompareScalar(maskLocal, tmp0Local, static_cast<float>(this->lambd), AscendC::CMPMODE::GT, (this->processDataNum+255)/256*256);
            AscendC::Select(tmp1Local, maskLocal, castLocal, static_cast<float>(0.0), AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, this->processDataNum);
            AscendC::Sub(tmp0Local, tmp1Local, lambdTensor, this->processDataNum);
            AscendC::CompareScalar(maskLocal, tmp1Local, static_cast<float>(this->lambd), AscendC::CMPMODE::GT, (this->processDataNum+255)/256*256);
            AscendC::Select(tmp1Local, maskLocal, tmp0Local, tmp1Local, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, this->processDataNum);
            AscendC::Add(tmp2Local, tmp1Local, lambdTensor, this->processDataNum);
            AscendC::Muls(lambdTensor, lambdTensor, static_cast<float>(-1.0), this->processDataNum);
            AscendC::Compare(maskLocal, tmp1Local, lambdTensor, AscendC::CMPMODE::LT, (this->processDataNum+255)/256*256);
            AscendC::Select(tmp0Local, maskLocal, tmp2Local, tmp1Local, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, this->processDataNum);
            AscendC::Cast(yLocal, tmp0Local, AscendC::RoundMode::CAST_RINT, this->processDataNum);
            outQueueY.EnQue<TYPE_X>(yLocal);
            inQueueX.FreeTensor(xLocal);
        #endif
        
    }else {
        AscendC::LocalTensor<TYPE_X> xLocal = inQueueX.DeQue<TYPE_X>();
        AscendC::LocalTensor<TYPE_X> yLocal = outQueueY.AllocTensor<TYPE_X>();
        AscendC::LocalTensor<TYPE_X> lambdTensor = lambdQueue.AllocTensor<TYPE_X>();
        AscendC::LocalTensor<TYPE_X> tmp0Local = tmpQueue0.AllocTensor<TYPE_X>();
        AscendC::LocalTensor<TYPE_X> tmp1Local = tmpQueue1.AllocTensor<TYPE_X>();
        AscendC::LocalTensor<TYPE_X> tmp2Local = tmpQueue2.AllocTensor<TYPE_X>();
        AscendC::LocalTensor<uint8_t> maskLocal = maskQueue.AllocTensor<uint8_t>();
        AscendC::Abs(tmp0Local, xLocal, this->processDataNum);
        AscendC::Duplicate(lambdTensor, static_cast<float>(this->lambd), this->processDataNum);
        AscendC::CompareScalar(maskLocal, tmp0Local, static_cast<float>(this->lambd), AscendC::CMPMODE::GT, (this->processDataNum+255)/256*256);
        AscendC::Select(tmp1Local, maskLocal, xLocal, static_cast<float>(0.0), AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, this->processDataNum);
        AscendC::Sub(tmp0Local, tmp1Local, lambdTensor, this->processDataNum);
        AscendC::CompareScalar(maskLocal, tmp1Local, static_cast<float>(this->lambd), AscendC::CMPMODE::GT, (this->processDataNum+255)/256*256);
        AscendC::Select(tmp1Local, maskLocal, tmp0Local, tmp1Local, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, this->processDataNum);
        AscendC::Add(tmp2Local, tmp1Local, lambdTensor, this->processDataNum);
        AscendC::Muls(lambdTensor, lambdTensor, static_cast<float>(-1.0), this->processDataNum);
        AscendC::Compare(maskLocal, tmp1Local, lambdTensor, AscendC::CMPMODE::LT, (this->processDataNum+255)/256*256);
        AscendC::Select(yLocal, maskLocal, tmp2Local, tmp1Local, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, this->processDataNum);
        outQueueY.EnQue<TYPE_X>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }
}
template <typename TYPE_X>
__aicore__ inline void KernelSoftShrink<TYPE_X>::Process()
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
} // namespace NsSoftShrink
#endif // SOFT_SHRINK_H

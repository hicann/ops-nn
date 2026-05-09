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
 * \file gelu_grad_v2.h
 * \brief
 */
#ifndef GELU_GRAD_V2_H
#define GELU_GRAD_V2_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "gelu_grad_v2_tiling_data.h"
#include "gelu_grad_v2_tiling_key.h"

namespace NsGeluGradV2 {

using namespace AscendC;

constexpr int32_t DOUBLE_BUFFER_NUM = 2;
constexpr int32_t SINGLE_BUFFER_NUM = 1;

template <typename TYPE_DY>
class KernelGeluGradV2 {
public:
    __aicore__ inline KernelGeluGradV2(){};

    __aicore__ inline void Init(GM_ADDR dy, GM_ADDR x, GM_ADDR z, uint64_t smallCoreDataNum, uint64_t bigCoreDataNum, uint64_t finalBigTileNum,
        uint64_t finalSmallTileNum, uint64_t tileDataNum, uint64_t smallTailDataNum, uint64_t bigTailDataNum, uint64_t tailBlockNum, uint64_t bufferOpen);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int32_t progress);
    __aicore__ inline void CopyOut(int32_t progress);
    __aicore__ inline void Compute(int32_t progress);

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, DOUBLE_BUFFER_NUM> inQueueX, inQueueDY;
    AscendC::TQue<AscendC::TPosition::VECOUT, DOUBLE_BUFFER_NUM> outQueueZ;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpQueue0, tmpQueue1, tmpQueue2, tmpQueue3, tmpQueue4, tmpQueueMask;

    AscendC::GlobalTensor<TYPE_DY> xGm, dyGm, zGm;
    uint64_t coreDataNum;
    uint64_t tileNum;
    uint64_t tileDataNum;
    uint64_t tailDataNum;
    uint64_t processDataNum;
};

template <typename TYPE_DY>
__aicore__ inline void KernelGeluGradV2<TYPE_DY>::Init(GM_ADDR dy, GM_ADDR x, GM_ADDR z, uint64_t smallCoreDataNum, uint64_t bigCoreDataNum, uint64_t finalBigTileNum,
    uint64_t finalSmallTileNum, uint64_t tileDataNum, uint64_t smallTailDataNum, uint64_t bigTailDataNum, uint64_t tailBlockNum, uint64_t bufferOpen)
{
    ASSERT(AscendC::GetBlockNum() != 0 && "block dim can not be zero!");
    uint64_t coreId = AscendC::GetBlockIdx();
    uint64_t globalBufferIndex = bigCoreDataNum * coreId;
    this->tileDataNum = tileDataNum;
    // default open double buffer
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
    xGm.SetGlobalBuffer((__gm__ TYPE_DY *)x + globalBufferIndex, this->coreDataNum);
    dyGm.SetGlobalBuffer((__gm__ TYPE_DY *)dy + globalBufferIndex, this->coreDataNum);
    zGm.SetGlobalBuffer((__gm__ TYPE_DY *)z + globalBufferIndex, this->coreDataNum);

    pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_DY));
    pipe.InitBuffer(inQueueDY, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_DY));
    pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_DY));
    
    pipe.InitBuffer(tmpQueue0, this->tileDataNum * sizeof(float));
    pipe.InitBuffer(tmpQueue1, this->tileDataNum * sizeof(float));
    if constexpr (!std::is_same_v<TYPE_DY, float>) {
        pipe.InitBuffer(tmpQueue2, this->tileDataNum * sizeof(float));
        pipe.InitBuffer(tmpQueue3, this->tileDataNum * sizeof(float));
    }
}

template <typename TYPE_DY>
__aicore__ inline void KernelGeluGradV2<TYPE_DY>::CopyIn(int32_t progress)
{
    AscendC::LocalTensor<TYPE_DY> xLocal = inQueueX.AllocTensor<TYPE_DY>();
    AscendC::LocalTensor<TYPE_DY> dyLocal = inQueueDY.AllocTensor<TYPE_DY>();
    AscendC::DataCopy(xLocal, xGm[progress * this->tileDataNum], this->processDataNum);
    AscendC::DataCopy(dyLocal, dyGm[progress * this->tileDataNum], this->processDataNum);
    inQueueX.EnQue(xLocal);
    inQueueDY.EnQue(dyLocal);
}

template <typename TYPE_DY>
__aicore__ inline void KernelGeluGradV2<TYPE_DY>::CopyOut(int32_t progress)
{
    AscendC::LocalTensor<TYPE_DY> zLocal = outQueueZ.DeQue<TYPE_DY>();
    AscendC::DataCopy(zGm[progress * this->tileDataNum], zLocal, this->processDataNum);
    outQueueZ.FreeTensor(zLocal);
}

template <typename TYPE_DY>
__aicore__ inline void KernelGeluGradV2<TYPE_DY>::Compute(int32_t progress)
{
    if constexpr (std::is_same_v<TYPE_DY, float>) {
        AscendC::LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        AscendC::LocalTensor<float> dyLocal = inQueueDY.DeQue<float>();
        AscendC::LocalTensor<float> zLocal = outQueueZ.AllocTensor<float>();
        AscendC::LocalTensor<float> tmp0Local = tmpQueue0.AllocTensor<float>();
        AscendC::LocalTensor<float> tmp1Local = tmpQueue1.AllocTensor<float>();
        AscendC::ShiftRight(tmp0Local.ReinterpretCast<uint32_t>(), xLocal.ReinterpretCast<uint32_t>(), static_cast<uint32_t>(31), this->processDataNum);
        AscendC::Cast(tmp0Local, tmp0Local.ReinterpretCast<int32_t>(), AscendC::RoundMode::CAST_NONE, this->processDataNum);
        AscendC::Adds(tmp0Local, tmp0Local, static_cast<float>(-0.5), this->processDataNum);

        AscendC::Abs(xLocal, xLocal, this->processDataNum);
        AscendC::Mins(xLocal, xLocal, static_cast<float>(30.0), this->processDataNum);
        AscendC::Muls(zLocal, xLocal, static_cast<float>(-0.7978849236182753), this->processDataNum);
        AscendC::Adds(zLocal, zLocal, static_cast<float>(-6.223065142195392), this->processDataNum);
        AscendC::Mul(zLocal, zLocal, xLocal, this->processDataNum);
        AscendC::Adds(zLocal, zLocal, static_cast<float>(-20.451359645895863), this->processDataNum);
        AscendC::Mul(zLocal, zLocal, xLocal, this->processDataNum);
        AscendC::Adds(zLocal, zLocal, static_cast<float>(-29.633095101286557), this->processDataNum);
        AscendC::Mul(zLocal, zLocal, xLocal, this->processDataNum);
        AscendC::Adds(zLocal, zLocal, static_cast<float>(-4.474629239730099), this->processDataNum);
        AscendC::Mul(zLocal, zLocal, xLocal, this->processDataNum);
        AscendC::Adds(zLocal, zLocal, static_cast<float>(30.9817411426), this->processDataNum);
        
        AscendC::Adds(tmp1Local, xLocal, static_cast<float>(7.79950327625), this->processDataNum);
        AscendC::Mul(tmp1Local, tmp1Local, xLocal, this->processDataNum);
        AscendC::Adds(tmp1Local, tmp1Local, static_cast<float>(26.6304065815), this->processDataNum);
        AscendC::Mul(tmp1Local, tmp1Local, xLocal, this->processDataNum);
        AscendC::Adds(tmp1Local, tmp1Local, static_cast<float>(44.965039885), this->processDataNum);
        AscendC::Mul(tmp1Local, tmp1Local, xLocal, this->processDataNum);
        AscendC::Adds(tmp1Local, tmp1Local, static_cast<float>(30.9817411527), this->processDataNum);
        
        AscendC::Div(zLocal, zLocal, tmp1Local, this->processDataNum);
        AscendC::Mul(xLocal, xLocal, xLocal, this->processDataNum);
        AscendC::Muls(xLocal, xLocal, static_cast<float>(-0.5), this->processDataNum);
        AscendC::Exp(xLocal, xLocal, this->processDataNum);
        AscendC::Mul(zLocal, zLocal, xLocal,  this->processDataNum);

        AscendC::Mul(zLocal, tmp0Local, zLocal, this->processDataNum);
        AscendC::Adds(tmp1Local, tmp0Local, static_cast<float>(-0.5), this->processDataNum);
        AscendC::Sub(zLocal, zLocal, tmp1Local, this->processDataNum);
        AscendC::Mul(zLocal, zLocal, dyLocal, this->processDataNum);

        outQueueZ.EnQue<float>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueDY.FreeTensor(dyLocal);
    } else {
        AscendC::LocalTensor<TYPE_DY> xLocal = inQueueX.DeQue<TYPE_DY>();
        AscendC::LocalTensor<TYPE_DY> dyLocal = inQueueDY.DeQue<TYPE_DY>();
        AscendC::LocalTensor<TYPE_DY> zLocal = outQueueZ.AllocTensor<TYPE_DY>();
        AscendC::LocalTensor<float> tmp0Local = tmpQueue0.AllocTensor<float>();
        AscendC::LocalTensor<float> tmp1Local = tmpQueue1.AllocTensor<float>();
        
        AscendC::LocalTensor<float> tmp2Local = tmpQueue2.AllocTensor<float>();
        AscendC::LocalTensor<float> tmp3Local = tmpQueue3.AllocTensor<float>();

        AscendC::Cast(tmp2Local, xLocal, AscendC::RoundMode::CAST_NONE, this->processDataNum);
        AscendC::ShiftRight(tmp0Local.ReinterpretCast<uint32_t>(), tmp2Local.ReinterpretCast<uint32_t>(), static_cast<uint32_t>(31), this->processDataNum);
        AscendC::Cast(tmp0Local, tmp0Local.ReinterpretCast<int32_t>(), AscendC::RoundMode::CAST_NONE, this->processDataNum);
        AscendC::Adds(tmp0Local, tmp0Local, static_cast<float>(-0.5), this->processDataNum);

        AscendC::Abs(tmp2Local, tmp2Local, this->processDataNum);
        AscendC::Mins(tmp2Local, tmp2Local, static_cast<float>(30.0), this->processDataNum);
        AscendC::Muls(tmp3Local, tmp2Local, static_cast<float>(-0.7978849236182753), this->processDataNum);
        AscendC::Adds(tmp3Local, tmp3Local, static_cast<float>(-6.223065142195392), this->processDataNum);
        AscendC::Mul(tmp3Local, tmp3Local, tmp2Local, this->processDataNum);
        AscendC::Adds(tmp3Local, tmp3Local, static_cast<float>(-20.451359645895863), this->processDataNum);
        AscendC::Mul(tmp3Local, tmp3Local, tmp2Local, this->processDataNum);
        AscendC::Adds(tmp3Local, tmp3Local, static_cast<float>(-29.633095101286557), this->processDataNum);
        AscendC::Mul(tmp3Local, tmp3Local, tmp2Local, this->processDataNum);
        AscendC::Adds(tmp3Local, tmp3Local, static_cast<float>(-4.474629239730099), this->processDataNum);
        AscendC::Mul(tmp3Local, tmp3Local, tmp2Local, this->processDataNum);
        AscendC::Adds(tmp3Local, tmp3Local, static_cast<float>(30.9817411426), this->processDataNum);
        
        AscendC::Adds(tmp1Local, tmp2Local, static_cast<float>(7.79950327625), this->processDataNum);
        AscendC::Mul(tmp1Local, tmp1Local, tmp2Local, this->processDataNum);
        AscendC::Adds(tmp1Local, tmp1Local, static_cast<float>(26.6304065815), this->processDataNum);
        AscendC::Mul(tmp1Local, tmp1Local, tmp2Local, this->processDataNum);
        AscendC::Adds(tmp1Local, tmp1Local, static_cast<float>(44.965039885), this->processDataNum);
        AscendC::Mul(tmp1Local, tmp1Local, tmp2Local, this->processDataNum);
        AscendC::Adds(tmp1Local, tmp1Local, static_cast<float>(30.9817411527), this->processDataNum);
        
        AscendC::Div(tmp3Local, tmp3Local, tmp1Local, this->processDataNum);
        AscendC::Mul(tmp2Local, tmp2Local, tmp2Local, this->processDataNum);
        AscendC::Muls(tmp2Local, tmp2Local, static_cast<float>(-0.5), this->processDataNum);
        AscendC::Exp(tmp2Local, tmp2Local, this->processDataNum);
        AscendC::Mul(tmp3Local, tmp3Local, tmp2Local,  this->processDataNum);

        AscendC::Mul(tmp3Local, tmp0Local, tmp3Local, this->processDataNum);
        AscendC::Adds(tmp1Local, tmp0Local, static_cast<float>(-0.5), this->processDataNum);
        AscendC::Sub(tmp3Local, tmp3Local, tmp1Local, this->processDataNum);
        AscendC::Cast(tmp2Local, dyLocal, AscendC::RoundMode::CAST_NONE, this->processDataNum);
        AscendC::Mul(tmp3Local, tmp3Local, tmp2Local, this->processDataNum);
        if constexpr (std::is_same_v<TYPE_DY, half>) {
            AscendC::Cast(zLocal, tmp3Local, AscendC::RoundMode::CAST_NONE, this->processDataNum);
        } else {
            AscendC::Cast(zLocal, tmp3Local, AscendC::RoundMode::CAST_RINT, this->processDataNum);
        }
        outQueueZ.EnQue<TYPE_DY>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueDY.FreeTensor(dyLocal);
    }
}

template <typename TYPE_DY>
__aicore__ inline void KernelGeluGradV2<TYPE_DY>::Process()
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

} // namespace NsGeluGradV2

namespace NsGeluGradV2Tanh {

using namespace AscendC;

constexpr int32_t DOUBLE_BUFFER_NUM = 2;
constexpr int32_t SINGLE_BUFFER_NUM = 1;

template <typename TYPE_DY>
class KernelGeluGradV2Tanh {
public:
    __aicore__ inline KernelGeluGradV2Tanh(){};

    __aicore__ inline void Init(GM_ADDR dy, GM_ADDR x, GM_ADDR z, uint64_t smallCoreDataNum, uint64_t bigCoreDataNum, uint64_t finalBigTileNum,
        uint64_t finalSmallTileNum, uint64_t tileDataNum, uint64_t smallTailDataNum, uint64_t bigTailDataNum, uint64_t tailBlockNum, uint64_t bufferOpen);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int32_t progress);
    __aicore__ inline void CopyOut(int32_t progress);
    __aicore__ inline void Compute(int32_t progress);

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, DOUBLE_BUFFER_NUM> inQueueX, inQueueDY;
    AscendC::TQue<AscendC::TPosition::VECOUT, DOUBLE_BUFFER_NUM> outQueueZ;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpQueue0, tmpQueue1, tmpQueue2, tmpQueue3, tmpQueue4, tmpQueueMask;

    AscendC::GlobalTensor<TYPE_DY> xGm, dyGm, zGm;
    uint64_t coreDataNum;
    uint64_t tileNum;
    uint64_t tileDataNum;
    uint64_t tailDataNum;
    uint64_t processDataNum;
};

template <typename TYPE_DY>
__aicore__ inline void KernelGeluGradV2Tanh<TYPE_DY>::Init(GM_ADDR dy, GM_ADDR x, GM_ADDR z, uint64_t smallCoreDataNum, uint64_t bigCoreDataNum, uint64_t finalBigTileNum,
    uint64_t finalSmallTileNum, uint64_t tileDataNum, uint64_t smallTailDataNum, uint64_t bigTailDataNum, uint64_t tailBlockNum, uint64_t bufferOpen)
{
    ASSERT(AscendC::GetBlockNum() != 0 && "block dim can not be zero!");
    uint64_t coreId = AscendC::GetBlockIdx();
    uint64_t globalBufferIndex = bigCoreDataNum * coreId;
    this->tileDataNum = tileDataNum;
    // default open double buffer
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
    xGm.SetGlobalBuffer((__gm__ TYPE_DY *)x + globalBufferIndex, this->coreDataNum);
    dyGm.SetGlobalBuffer((__gm__ TYPE_DY *)dy + globalBufferIndex, this->coreDataNum);
    zGm.SetGlobalBuffer((__gm__ TYPE_DY *)z + globalBufferIndex, this->coreDataNum);

    pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_DY));
    pipe.InitBuffer(inQueueDY, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_DY));
    pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_DY));
    
    pipe.InitBuffer(tmpQueue0, this->tileDataNum * sizeof(float));
    pipe.InitBuffer(tmpQueue1, this->tileDataNum * sizeof(float));
    if constexpr (std::is_same_v<TYPE_DY, float>) {
        pipe.InitBuffer(tmpQueueMask, this->tileDataNum * sizeof(uint8_t));
        pipe.InitBuffer(tmpQueue2, this->tileDataNum * sizeof(float));
    } else if constexpr (!std::is_same_v<TYPE_DY, float>) {
        pipe.InitBuffer(tmpQueueMask, this->tileDataNum * sizeof(uint8_t));
        pipe.InitBuffer(tmpQueue2, this->tileDataNum * sizeof(float));
        pipe.InitBuffer(tmpQueue3, this->tileDataNum * sizeof(float));
        pipe.InitBuffer(tmpQueue4, this->tileDataNum * sizeof(float));
    }
}

template <typename TYPE_DY>
__aicore__ inline void KernelGeluGradV2Tanh<TYPE_DY>::CopyIn(int32_t progress)
{
    AscendC::LocalTensor<TYPE_DY> xLocal = inQueueX.AllocTensor<TYPE_DY>();
    AscendC::LocalTensor<TYPE_DY> dyLocal = inQueueDY.AllocTensor<TYPE_DY>();
    AscendC::DataCopy(xLocal, xGm[progress * this->tileDataNum], this->processDataNum);
    AscendC::DataCopy(dyLocal, dyGm[progress * this->tileDataNum], this->processDataNum);
    inQueueX.EnQue(xLocal);
    inQueueDY.EnQue(dyLocal);
}

template <typename TYPE_DY>
__aicore__ inline void KernelGeluGradV2Tanh<TYPE_DY>::CopyOut(int32_t progress)
{
    AscendC::LocalTensor<TYPE_DY> zLocal = outQueueZ.DeQue<TYPE_DY>();
    AscendC::DataCopy(zGm[progress * this->tileDataNum], zLocal, this->processDataNum);
    outQueueZ.FreeTensor(zLocal);
}

template <typename TYPE_DY>
__aicore__ inline void KernelGeluGradV2Tanh<TYPE_DY>::Compute(int32_t progress)
{
    if constexpr (std::is_same_v<TYPE_DY, float>) {
        AscendC::LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        AscendC::LocalTensor<float> dyLocal = inQueueDY.DeQue<float>();
        AscendC::LocalTensor<float> zLocal = outQueueZ.AllocTensor<float>();
        AscendC::LocalTensor<float> tmp0Local = tmpQueue0.AllocTensor<float>();
        AscendC::LocalTensor<float> tmp1Local = tmpQueue1.AllocTensor<float>();
        AscendC::LocalTensor<float> tmp2Local = tmpQueue2.AllocTensor<float>();
        AscendC::LocalTensor<uint8_t> maskLocal = tmpQueueMask.AllocTensor<uint8_t>();

        AscendC::Mul(tmp0Local, xLocal, xLocal, this->processDataNum);
        AscendC::Muls(zLocal, tmp0Local, static_cast<float>(-0.0713548162726002527220), this->processDataNum);
        AscendC::Adds(zLocal, zLocal, static_cast<float>(-1.595769121605730711759), this->processDataNum);
        AscendC::Mul(zLocal, zLocal, xLocal, this->processDataNum);
        AscendC::Exp(zLocal, zLocal, this->processDataNum);

        AscendC::Adds(tmp1Local, zLocal, static_cast<float>(1.0), this->processDataNum);
        AscendC::Duplicate(tmp2Local,  static_cast<float>(1.0), this->processDataNum);
        AscendC::Div(tmp1Local, tmp2Local, tmp1Local, this->processDataNum);
        AscendC::Mul(zLocal, zLocal, tmp1Local, this->processDataNum);

        AscendC::Muls(tmp0Local, tmp0Local, static_cast<float>(0.2140644488178007), this->processDataNum);
        AscendC::Adds(tmp0Local, tmp0Local, static_cast<float>(1.595769121605730711759), this->processDataNum);
        AscendC::Mul(tmp0Local, tmp0Local, xLocal, this->processDataNum);

        AscendC::Mul(zLocal, zLocal, tmp0Local, this->processDataNum);
        AscendC::Mul(zLocal, zLocal, tmp1Local, this->processDataNum);
        AscendC::Compare(maskLocal, zLocal, zLocal, AscendC::CMPMODE::EQ, this->processDataNum);
        AscendC::Duplicate(tmp2Local[0].ReinterpretCast<half>(),  static_cast<half>(1.0), this->processDataNum);
        AscendC::Select(tmp2Local[0].ReinterpretCast<half>(), maskLocal, tmp2Local[0].ReinterpretCast<half>(), static_cast<half>(0.0), AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, this->processDataNum);
        AscendC::Cast(tmp0Local[0].ReinterpretCast<int8_t>(), tmp2Local[0].ReinterpretCast<half>(), AscendC::RoundMode::CAST_NONE, this->processDataNum);
        AscendC::Cast(tmp2Local[0].ReinterpretCast<half>(), tmp0Local[0].ReinterpretCast<int8_t>(), AscendC::RoundMode::CAST_NONE, this->processDataNum);
        AscendC::Duplicate(tmp0Local[0].ReinterpretCast<half>(),  static_cast<half>(0.0), this->processDataNum);
        AscendC::Compare(maskLocal, tmp2Local[0].ReinterpretCast<half>(), tmp0Local[0].ReinterpretCast<half>(), AscendC::CMPMODE::GT, this->processDataNum);
        AscendC::Select(zLocal, maskLocal, zLocal, static_cast<float>(0), AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, this->processDataNum);
        AscendC::Add(zLocal, zLocal, tmp1Local, this->processDataNum);
        AscendC::Mul(zLocal, zLocal, dyLocal, this->processDataNum);

        outQueueZ.EnQue<float>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueDY.FreeTensor(dyLocal);
    } else {
        AscendC::LocalTensor<TYPE_DY> xLocal = inQueueX.DeQue<TYPE_DY>();
        AscendC::LocalTensor<TYPE_DY> dyLocal = inQueueDY.DeQue<TYPE_DY>();
        AscendC::LocalTensor<TYPE_DY> zLocal = outQueueZ.AllocTensor<TYPE_DY>();
        AscendC::LocalTensor<float> tmp0Local = tmpQueue0.AllocTensor<float>();
        AscendC::LocalTensor<float> tmp1Local = tmpQueue1.AllocTensor<float>();
        AscendC::LocalTensor<float> tmp2Local = tmpQueue2.AllocTensor<float>();
        AscendC::LocalTensor<float> tmp3Local = tmpQueue3.AllocTensor<float>();
        AscendC::LocalTensor<float> tmp4Local = tmpQueue4.AllocTensor<float>();
        AscendC::LocalTensor<uint8_t> maskLocal = tmpQueueMask.AllocTensor<uint8_t>();
        
        AscendC::Cast(tmp3Local, xLocal, AscendC::RoundMode::CAST_NONE, this->processDataNum);

        AscendC::Mul(tmp0Local, tmp3Local, tmp3Local, this->processDataNum);
        AscendC::Muls(tmp4Local, tmp0Local, static_cast<float>(-0.0713548162726002527220), this->processDataNum);
        AscendC::Adds(tmp4Local, tmp4Local, static_cast<float>(-1.595769121605730711759), this->processDataNum);
        AscendC::Mul(tmp4Local, tmp4Local, tmp3Local, this->processDataNum);
        AscendC::Exp(tmp4Local, tmp4Local, this->processDataNum);

        AscendC::Adds(tmp1Local, tmp4Local, static_cast<float>(1.0), this->processDataNum);
        AscendC::Duplicate(tmp2Local,  static_cast<float>(1.0), this->processDataNum);
        AscendC::Div(tmp1Local, tmp2Local, tmp1Local, this->processDataNum);
        AscendC::Mul(tmp4Local, tmp4Local, tmp1Local, this->processDataNum);

        AscendC::Muls(tmp0Local, tmp0Local, static_cast<float>(0.2140644488178007), this->processDataNum);
        AscendC::Adds(tmp0Local, tmp0Local, static_cast<float>(1.595769121605730711759), this->processDataNum);
        AscendC::Mul(tmp0Local, tmp0Local, tmp3Local, this->processDataNum);

        AscendC::Mul(tmp4Local, tmp4Local, tmp0Local, this->processDataNum);
        AscendC::Mul(tmp4Local, tmp4Local, tmp1Local, this->processDataNum);
        AscendC::Compare(maskLocal, tmp4Local, tmp4Local, AscendC::CMPMODE::EQ, this->processDataNum);
        AscendC::Duplicate(tmp0Local[0].ReinterpretCast<half>(),  static_cast<half>(1.0), this->processDataNum);
        AscendC::Select(tmp0Local[0].ReinterpretCast<half>(), maskLocal, tmp0Local[0].ReinterpretCast<half>(), static_cast<half>(0.0), AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, this->processDataNum);
        AscendC::Cast(tmp2Local[0].ReinterpretCast<int8_t>(), tmp0Local[0].ReinterpretCast<half>(), AscendC::RoundMode::CAST_NONE, this->processDataNum);
        AscendC::Cast(tmp0Local[0].ReinterpretCast<half>(), tmp2Local[0].ReinterpretCast<int8_t>(), AscendC::RoundMode::CAST_NONE, this->processDataNum);
        AscendC::Duplicate(tmp2Local[0].ReinterpretCast<half>(),  static_cast<half>(0.0), this->processDataNum);
        AscendC::Compare(maskLocal, tmp0Local[0].ReinterpretCast<half>(), tmp2Local[0].ReinterpretCast<half>(), AscendC::CMPMODE::GT, this->processDataNum);
        AscendC::Select(tmp4Local, maskLocal, tmp4Local, static_cast<float>(0.0), AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, this->processDataNum);
        AscendC::Add(tmp4Local, tmp4Local, tmp1Local, this->processDataNum);
        AscendC::Cast(tmp3Local, dyLocal, AscendC::RoundMode::CAST_NONE, this->processDataNum);
        AscendC::Mul(tmp4Local, tmp4Local, tmp3Local, this->processDataNum);
        if constexpr (std::is_same_v<TYPE_DY, half>) {
            AscendC::Cast(zLocal, tmp4Local, AscendC::RoundMode::CAST_NONE, this->processDataNum);
        } else {
            AscendC::Cast(zLocal, tmp4Local, AscendC::RoundMode::CAST_RINT, this->processDataNum);
        }
        outQueueZ.EnQue<TYPE_DY>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueDY.FreeTensor(dyLocal);
    }
}

template <typename TYPE_DY>
__aicore__ inline void KernelGeluGradV2Tanh<TYPE_DY>::Process()
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

}

#endif // GELU_GRAD_V2_H
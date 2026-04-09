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
 * \file hardtanh_grad.h
 * \brief
 */
#ifndef __HARDTANH_GRAD_H__
#define __HARDTANH_GRAD_H__

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "hardtanh_grad_tiling_data.h"
#include "hardtanh_grad_tiling_key.h"

namespace MyHardtanhGrad {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
constexpr uint32_t BITS_PER_INT16 = 16;
constexpr uint32_t MAX_TILEDATA = 6144;

template <typename TYPE_X, typename TYPE_Y>
class KernelHardtanhGrad {
public:
    __aicore__ inline KernelHardtanhGrad(){};

    __aicore__ inline void Init(
        GM_ADDR gradOutput, GM_ADDR self, GM_ADDR out, 
        const HardtanhGradTilingData* tilingData, TPipe* pipeIn);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int32_t progress);
    __aicore__ inline void CopyOut(int32_t progress);
    __aicore__ inline void Compute(int32_t progress);
    __aicore__ inline void CalculateMask(AscendC::LocalTensor<uint8_t> &maskLocal, 
                                         AscendC::LocalTensor<float> &xLocal, 
                                         uint32_t length);    

private:
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueC, inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> maskBuf, tmpBuf, calcBuf;

    AscendC::GlobalTensor<TYPE_X> gradOutputGm, selfGm;
    AscendC::GlobalTensor<TYPE_Y> outGm;
    uint64_t coreDataNum;
    uint64_t tileNum;
    uint64_t tileDataNum;
    uint64_t tailDataNum;
    uint64_t processDataNum;
    uint64_t maskTileDataNum;
    float min;
    float max;
};

template <typename TYPE_X, typename TYPE_Y>
__aicore__ inline void KernelHardtanhGrad<TYPE_X, TYPE_Y>::Init(
    GM_ADDR gradOutput, GM_ADDR self, GM_ADDR out, const HardtanhGradTilingData* tilingData, TPipe* pipeIn)
{
    ASSERT(AscendC::GetBlockNum() != 0 && "block dim can not be zero!");
    uint64_t coreId = AscendC::GetBlockIdx();
    uint64_t globalBufferIndex = tilingData->bigCoreDataNum * AscendC::GetBlockIdx();
    this->tileDataNum = tilingData->tileDataNum;
    if (coreId < tilingData->tailBlockNum) {
        this->coreDataNum = tilingData->bigCoreDataNum;
        this->tileNum = tilingData->finalBigTileNum;
        this->tailDataNum = tilingData->bigTailDataNum;
    } else {
        this->coreDataNum = tilingData->smallCoreDataNum;
        this->tileNum = tilingData->finalSmallTileNum;
        this->tailDataNum = tilingData->smallTailDataNum;
        globalBufferIndex -= (tilingData->bigCoreDataNum - tilingData->smallCoreDataNum) * (AscendC::GetBlockIdx() - tilingData->tailBlockNum);
    }
    this->maskTileDataNum = tilingData->maskTileDataNum;
    this->min = tilingData->min;
    this->max = tilingData->max;

    gradOutputGm.SetGlobalBuffer((__gm__ TYPE_Y*)gradOutput + globalBufferIndex, this->coreDataNum);
    selfGm.SetGlobalBuffer((__gm__ TYPE_Y*)self + globalBufferIndex, this->coreDataNum);
    outGm.SetGlobalBuffer((__gm__ TYPE_Y*)out + globalBufferIndex, this->coreDataNum);

    pipeIn->InitBuffer(inQueueX, BUFFER_NUM, 2 * this->tileDataNum * sizeof(TYPE_Y));
    pipeIn->InitBuffer(outQueueY, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_Y));
    pipeIn->InitBuffer(maskBuf, 3 * this->maskTileDataNum * sizeof(uint8_t));

    if constexpr((std::is_same_v<TYPE_Y, half>) || (std::is_same_v<TYPE_Y, bfloat16_t>)) {
        pipeIn->InitBuffer(tmpBuf, this->tileDataNum * sizeof(float));
    }
}

template <typename TYPE_X, typename TYPE_Y>
__aicore__ inline void KernelHardtanhGrad<TYPE_X, TYPE_Y>::CopyIn(int32_t progress)
{
    LocalTensor<TYPE_Y> xLocal = inQueueX.AllocTensor<TYPE_Y>();

    DataCopy(xLocal, gradOutputGm[progress * this->tileDataNum], this->processDataNum);
    DataCopy(xLocal[this->tileDataNum], selfGm[progress * this->tileDataNum], this->processDataNum);

    inQueueX.EnQue(xLocal);
}

template <typename TYPE_X, typename TYPE_Y>
__aicore__ inline void KernelHardtanhGrad<TYPE_X, TYPE_Y>::CopyOut(int32_t progress)
{
    LocalTensor<TYPE_Y> yLocal = outQueueY.DeQue<TYPE_Y>();

    DataCopy(outGm[progress * this->tileDataNum], yLocal, this->processDataNum);

    outQueueY.FreeTensor(yLocal);
}

template <typename TYPE_X, typename TYPE_Y>
__aicore__ inline void KernelHardtanhGrad<TYPE_X, TYPE_Y>::CalculateMask(AscendC::LocalTensor<uint8_t> &maskLocal, 
                                                                         AscendC::LocalTensor<float> &xLocal, 
                                                                         uint32_t length)
{
    LocalTensor<uint8_t> maskTmp1 = maskLocal;
    LocalTensor<uint8_t> maskTmp2 = maskLocal[this->maskTileDataNum];
    LocalTensor<uint8_t> maskTmp3 = maskLocal[2 * this->maskTileDataNum];   
    CompareScalar(maskTmp1, xLocal, (float)(1.0), CMPMODE::LE, length);
    CompareScalar(maskTmp2, xLocal, (float)(1.0), CMPMODE::GT, length);
    AscendC::PipeBarrier<PIPE_V>();
    Or(maskTmp1.ReinterpretCast<uint16_t>(), maskTmp1.ReinterpretCast<uint16_t>(), maskTmp2.ReinterpretCast<uint16_t>(), (length / BITS_PER_INT16));
    AscendC::PipeBarrier<PIPE_V>();
    Not(maskTmp1.ReinterpretCast<uint16_t>(), maskTmp1.ReinterpretCast<uint16_t>(), (length / BITS_PER_INT16));
    CompareScalar(maskTmp3, xLocal, (float)(this->max), CMPMODE::LE, length);
    CompareScalar(maskTmp2, xLocal, (float)(this->min), CMPMODE::GE, length);
    AscendC::PipeBarrier<PIPE_V>();
    And(maskTmp2.ReinterpretCast<uint16_t>(), maskTmp2.ReinterpretCast<uint16_t>(), maskTmp3.ReinterpretCast<uint16_t>(), (length / BITS_PER_INT16));
    AscendC::PipeBarrier<PIPE_V>();
    Or(maskTmp1.ReinterpretCast<uint16_t>(), maskTmp1.ReinterpretCast<uint16_t>(), maskTmp2.ReinterpretCast<uint16_t>(), (length / BITS_PER_INT16));   
}


template <typename TYPE_X, typename TYPE_Y>
__aicore__ inline void KernelHardtanhGrad<TYPE_X, TYPE_Y>::Compute(int32_t progress)
{
    LocalTensor<uint8_t> maskTmp = maskBuf.Get<uint8_t>();
    LocalTensor<TYPE_X> xLocal = inQueueX.DeQue<TYPE_X>();
    LocalTensor<TYPE_X> yLocal = outQueueY.AllocTensor<TYPE_X>();
    LocalTensor<TYPE_X> x2Local = xLocal;
    LocalTensor<TYPE_X> x1Local = xLocal[this->tileDataNum];    
    if constexpr (std::is_same_v<TYPE_X, float>) 
    {
        CalculateMask(maskTmp, x2Local, this->processDataNum);
        Select(yLocal, maskTmp, x1Local, static_cast<float>(0), SELMODE::VSEL_TENSOR_SCALAR_MODE, this->processDataNum);        
    }
    if constexpr (std::is_same_v<TYPE_X, half>) 
    {
        LocalTensor<float> p1 = tmpBuf.Get<float>();
        Cast(p1, x2Local, RoundMode::CAST_NONE, this->processDataNum);
        CalculateMask(maskTmp, p1, this->processDataNum);
        Select(yLocal, maskTmp, x1Local, static_cast<half>(0), SELMODE::VSEL_TENSOR_SCALAR_MODE, this->processDataNum);     
    }    
    if constexpr (std::is_same_v<TYPE_X, bfloat16_t>) 
    {
        LocalTensor<float> p1 = tmpBuf.Get<float>();
        Cast(p1, x2Local, RoundMode::CAST_NONE, this->processDataNum);
        CalculateMask(maskTmp, p1, this->processDataNum);
        Select(yLocal.template ReinterpretCast<half>(), maskTmp, x1Local.template ReinterpretCast<half>(), static_cast<half>(0), SELMODE::VSEL_TENSOR_SCALAR_MODE, this->processDataNum);      
    }     

    outQueueY.EnQue<TYPE_X>(yLocal);
    inQueueX.FreeTensor(xLocal);
}

template <typename TYPE_X, typename TYPE_Y>
__aicore__ inline void KernelHardtanhGrad<TYPE_X, TYPE_Y>::Process()
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
    
template <typename TYPE_X, typename TYPE_Y>
class KernelHardtanhGrad_NoDB {
public:
    __aicore__ inline KernelHardtanhGrad_NoDB(){};
    __aicore__ inline void Init(
        GM_ADDR gradOutput, GM_ADDR self, GM_ADDR out, 
        const HardtanhGradTilingData* tilingData, 
        TPipe* pipeIn);
private:
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> inQueueC, inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> outQueueY;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> maskBuf, tmpBuf, calcBuf;
    AscendC::GlobalTensor<TYPE_X> gradOutputGm, selfGm;
    AscendC::GlobalTensor<TYPE_Y> outGm;
    uint64_t processDataNum;
    uint32_t maskTileDataNum;
    float min;
    float max;
};

template <typename TYPE_X, typename TYPE_Y>
__aicore__ inline void KernelHardtanhGrad_NoDB<TYPE_X, TYPE_Y>::Init(
        GM_ADDR gradOutput, GM_ADDR self, GM_ADDR out, 
        const HardtanhGradTilingData* tilingData, 
        TPipe* pipeIn)
{
    ASSERT(AscendC::GetBlockNum() != 0 && "block dim can not be zero!");
    uint64_t coreId = AscendC::GetBlockIdx();
    uint64_t coreNum = AscendC::GetBlockNum();
    if (coreId != coreNum - 1)
    {
        this->processDataNum = tilingData->tileDataNum;
    }
    else
    {
        this->processDataNum = tilingData->smallTailDataNum;
    }
    uint64_t globalBufferIndex = tilingData->tileDataNum * coreId;
    this->maskTileDataNum = tilingData->maskTileDataNum;
    this->min = tilingData->min;
    this->max = tilingData->max;

    gradOutputGm.SetGlobalBuffer((__gm__ TYPE_Y*)gradOutput + globalBufferIndex, tilingData->tileDataNum);
    selfGm.SetGlobalBuffer((__gm__ TYPE_Y*)self + globalBufferIndex, tilingData->tileDataNum);
    outGm.SetGlobalBuffer((__gm__ TYPE_Y*)out + globalBufferIndex, tilingData->tileDataNum);

    pipeIn->InitBuffer(inQueueX, BUFFER_NUM, 2 * MAX_TILEDATA * sizeof(TYPE_Y));
    pipeIn->InitBuffer(outQueueY, BUFFER_NUM, MAX_TILEDATA * sizeof(TYPE_Y));
    pipeIn->InitBuffer(maskBuf, 3 * MAX_TILEDATA * sizeof(uint8_t));

    if constexpr((std::is_same_v<TYPE_Y, half>) || (std::is_same_v<TYPE_Y, bfloat16_t>)) {
        pipeIn->InitBuffer(tmpBuf, MAX_TILEDATA * sizeof(float));
    }
    LocalTensor<TYPE_Y> xLocal = inQueueX.AllocTensor<TYPE_Y>();

    DataCopy(xLocal, gradOutputGm, this->processDataNum);
    DataCopy(xLocal[MAX_TILEDATA], selfGm, this->processDataNum);

    event_t eventId1 = static_cast<event_t>(pipeIn->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventId1);

    LocalTensor<uint8_t> maskTmp = maskBuf.Get<uint8_t>();
    LocalTensor<uint8_t> maskTmp1 = maskTmp;
    LocalTensor<uint8_t> maskTmp2 = maskTmp[MAX_TILEDATA];
    LocalTensor<uint8_t> maskTmp3 = maskTmp[2 * MAX_TILEDATA];   
    
    WaitFlag<HardEvent::MTE2_V>(eventId1);

    if constexpr (std::is_same_v<TYPE_X, float>) 
    {
        LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
        LocalTensor<float> x2Local = xLocal;
        LocalTensor<float> x1Local = xLocal[MAX_TILEDATA];
  
        CompareScalar(maskTmp1, x2Local, (float)(1.0), CMPMODE::LE, this->processDataNum);
        CompareScalar(maskTmp2, x2Local, (float)(1.0), CMPMODE::GT, this->processDataNum);
        AscendC::PipeBarrier<PIPE_V>();
        Or(maskTmp1.ReinterpretCast<uint16_t>(), maskTmp1.ReinterpretCast<uint16_t>(), maskTmp2.ReinterpretCast<uint16_t>(), (this->processDataNum / BITS_PER_INT16));
        AscendC::PipeBarrier<PIPE_V>();
        Not(maskTmp1.ReinterpretCast<uint16_t>(), maskTmp1.ReinterpretCast<uint16_t>(), (this->processDataNum / BITS_PER_INT16));
        CompareScalar(maskTmp3, x2Local, (float)(this->max), CMPMODE::LE, this->processDataNum);
        CompareScalar(maskTmp2, x2Local, (float)(this->min), CMPMODE::GE, this->processDataNum);
        AscendC::PipeBarrier<PIPE_V>();
        And(maskTmp2.ReinterpretCast<uint16_t>(), maskTmp2.ReinterpretCast<uint16_t>(), maskTmp3.ReinterpretCast<uint16_t>(), (this->processDataNum / BITS_PER_INT16));
        AscendC::PipeBarrier<PIPE_V>();
        Or(maskTmp1.ReinterpretCast<uint16_t>(), maskTmp1.ReinterpretCast<uint16_t>(), maskTmp2.ReinterpretCast<uint16_t>(), (this->processDataNum / BITS_PER_INT16));  
        AscendC::PipeBarrier<PIPE_V>();
        Select(yLocal, maskTmp1, x1Local, static_cast<float>(0), SELMODE::VSEL_TENSOR_SCALAR_MODE, this->processDataNum);
        
        event_t eventId2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventId2);
        inQueueX.FreeTensor(xLocal);
        WaitFlag<HardEvent::V_MTE3>(eventId2);
        DataCopy(outGm, yLocal, this->processDataNum);
        outQueueY.FreeTensor(yLocal);            
    }

    if constexpr (std::is_same_v<TYPE_X, half>) 
    {
        LocalTensor<half> yLocal = outQueueY.AllocTensor<half>();
        LocalTensor<half> x2Local = xLocal;
        LocalTensor<half> x1Local = xLocal[MAX_TILEDATA];

        LocalTensor<float> p1 = tmpBuf.Get<float>();
        Cast(p1, x2Local, RoundMode::CAST_NONE, this->processDataNum);
        AscendC::PipeBarrier<PIPE_V>();
        CompareScalar(maskTmp1, p1, (float)(1.0), CMPMODE::LE, this->processDataNum);
        CompareScalar(maskTmp2, p1, (float)(1.0), CMPMODE::GT, this->processDataNum);
        AscendC::PipeBarrier<PIPE_V>();
        Or(maskTmp1.ReinterpretCast<uint16_t>(), maskTmp1.ReinterpretCast<uint16_t>(), maskTmp2.ReinterpretCast<uint16_t>(), (this->processDataNum / BITS_PER_INT16));
        AscendC::PipeBarrier<PIPE_V>();
        Not(maskTmp1.ReinterpretCast<uint16_t>(), maskTmp1.ReinterpretCast<uint16_t>(), (this->processDataNum / BITS_PER_INT16));
        CompareScalar(maskTmp3, p1, (float)(this->max), CMPMODE::LE, this->processDataNum);
        CompareScalar(maskTmp2, p1, (float)(this->min), CMPMODE::GE, this->processDataNum);
        AscendC::PipeBarrier<PIPE_V>();
        And(maskTmp2.ReinterpretCast<uint16_t>(), maskTmp2.ReinterpretCast<uint16_t>(), maskTmp3.ReinterpretCast<uint16_t>(), (this->processDataNum / BITS_PER_INT16));
        AscendC::PipeBarrier<PIPE_V>();
        Or(maskTmp1.ReinterpretCast<uint16_t>(), maskTmp1.ReinterpretCast<uint16_t>(), maskTmp2.ReinterpretCast<uint16_t>(), (this->processDataNum / BITS_PER_INT16));   
        AscendC::PipeBarrier<PIPE_V>();
        Select(yLocal, maskTmp1, x1Local, static_cast<half>(0), SELMODE::VSEL_TENSOR_SCALAR_MODE, this->processDataNum);

        event_t eventId2 = static_cast<event_t>(pipeIn->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventId2);
        inQueueX.FreeTensor(xLocal);
        WaitFlag<HardEvent::V_MTE3>(eventId2);

        DataCopy(outGm, yLocal, this->processDataNum);
        outQueueY.FreeTensor(yLocal);            
    }

    if constexpr (std::is_same_v<TYPE_X, bfloat16_t>) 
    {
        LocalTensor<bfloat16_t> yLocal = outQueueY.AllocTensor<bfloat16_t>();
        LocalTensor<bfloat16_t> x2Local = xLocal;
        LocalTensor<bfloat16_t> x1Local = xLocal[MAX_TILEDATA];
        LocalTensor<float> p1 = tmpBuf.Get<float>();
        Cast(p1, x2Local, RoundMode::CAST_NONE, this->processDataNum);
        AscendC::PipeBarrier<PIPE_V>();
        CompareScalar(maskTmp1, p1, (float)(1.0f), CMPMODE::LE, this->processDataNum);
        CompareScalar(maskTmp2, p1, (float)(1.0f), CMPMODE::GT, this->processDataNum);
        AscendC::PipeBarrier<PIPE_V>();
        Or(maskTmp1.ReinterpretCast<uint16_t>(), maskTmp1.ReinterpretCast<uint16_t>(), maskTmp2.ReinterpretCast<uint16_t>(), (this->processDataNum / BITS_PER_INT16));
        AscendC::PipeBarrier<PIPE_V>();
        Not(maskTmp1.ReinterpretCast<uint16_t>(), maskTmp1.ReinterpretCast<uint16_t>(), (this->processDataNum / BITS_PER_INT16));
        CompareScalar(maskTmp3, p1, this->max, CMPMODE::LE, this->processDataNum);
        CompareScalar(maskTmp2, p1, this->min, CMPMODE::GE, this->processDataNum);
        AscendC::PipeBarrier<PIPE_V>();
        And(maskTmp2.ReinterpretCast<uint16_t>(), maskTmp2.ReinterpretCast<uint16_t>(), maskTmp3.ReinterpretCast<uint16_t>(), (this->processDataNum / BITS_PER_INT16));
        AscendC::PipeBarrier<PIPE_V>();
        Or(maskTmp1.ReinterpretCast<uint16_t>(), maskTmp1.ReinterpretCast<uint16_t>(), maskTmp2.ReinterpretCast<uint16_t>(), (this->processDataNum / BITS_PER_INT16));  
        AscendC::PipeBarrier<PIPE_V>();
        Select(yLocal.ReinterpretCast<half>(), maskTmp1, x1Local.ReinterpretCast<half>(), static_cast<half>(0), SELMODE::VSEL_TENSOR_SCALAR_MODE, this->processDataNum);

        event_t eventId2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventId2);
        inQueueX.FreeTensor(xLocal);
        WaitFlag<HardEvent::V_MTE3>(eventId2);

        DataCopy(outGm, yLocal, this->processDataNum);
        outQueueY.FreeTensor(yLocal);        
    }
}    
    
} // namespace Myhardtanh_grad
#endif // __HARDTANH_GRAD_H__
/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/

/*!
 * \file mish.h
 * \brief
 */
#ifndef MISH_H
#define MISH_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "mish_tiling_data.h"
#include "mish_tiling_key.h"


namespace MyMish {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
constexpr uint32_t COMPARE_ALIGN = 64;    

template <typename TYPE_X, typename TYPE_Y>
class KernelMish {
public:
    __aicore__ inline KernelMish(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const MishTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int32_t progress);
    __aicore__ inline void CopyOut(int32_t progress);
    __aicore__ inline void Compute(int32_t progress);
    __aicore__ inline void ComputeSupPerfBf16(LocalTensor<TYPE_X> &xLocal, LocalTensor<TYPE_Y> &yLocal);
    __aicore__ inline void ComputeSupPerf(LocalTensor<TYPE_X> &xLocal, LocalTensor<TYPE_Y> &yLocal);
    __aicore__ inline void ComputeHighPerf16(LocalTensor<TYPE_X> &xLocal, LocalTensor<TYPE_Y> &yLocal);
    __aicore__ inline void ComputeHighPerf(LocalTensor<TYPE_X> &xLocal, LocalTensor<TYPE_Y> &yLocal);

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::TBuf<QuePosition::VECCALC> tmpBuffer1, tmpBuffer2, tmpBuffer3; 
    AscendC::TBuf<QuePosition::VECCALC> QueueTmpX, QueueTmpY;
    AscendC::GlobalTensor<TYPE_X> xGm;
    AscendC::GlobalTensor<TYPE_Y> yGm;
    uint64_t coreDataNum;
    uint64_t tileNum;
    uint64_t tileDataNum;
    uint64_t tailDataNum;
    uint64_t processDataNum;
};

template <typename TYPE_X, typename TYPE_Y>
__aicore__ inline void KernelMish<TYPE_X, TYPE_Y>::Init(GM_ADDR x, GM_ADDR y, const MishTilingData* tilingData)
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
    yGm.SetGlobalBuffer((__gm__ TYPE_Y*)y + globalBufferIndex, this->coreDataNum);
    pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_X));
    pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_Y));
    
    pipe.InitBuffer(tmpBuffer1, this->tileDataNum * sizeof(float));
    pipe.InitBuffer(tmpBuffer2, this->tileDataNum * sizeof(float));
    pipe.InitBuffer(tmpBuffer3, this->tileDataNum * sizeof(uint8_t));
    pipe.InitBuffer(QueueTmpX, this->tileDataNum * sizeof(float));
    pipe.InitBuffer(QueueTmpY, this->tileDataNum * sizeof(float));
}

template <typename TYPE_X, typename TYPE_Y>
__aicore__ inline void KernelMish<TYPE_X, TYPE_Y>::CopyIn(int32_t progress)
{
    AscendC::LocalTensor<TYPE_X> xLocal = inQueueX.AllocTensor<TYPE_X>();
    AscendC::DataCopy(xLocal, xGm[progress * this->tileDataNum], this->processDataNum);
    inQueueX.EnQue(xLocal);
}

template <typename TYPE_X, typename TYPE_Y>
__aicore__ inline void KernelMish<TYPE_X, TYPE_Y>::CopyOut(int32_t progress)
{
    AscendC::LocalTensor<TYPE_Y> yLocal = outQueueY.DeQue<TYPE_Y>();
    AscendC::DataCopy(yGm[progress * this->tileDataNum], yLocal, this->processDataNum);
    outQueueY.FreeTensor(yLocal);
}

template <typename TYPE_X, typename TYPE_Y>
__aicore__ inline void KernelMish<TYPE_X, TYPE_Y>::ComputeSupPerfBf16(LocalTensor<TYPE_X> &xLocal, LocalTensor<TYPE_Y> &yLocal)
{
    AscendC::LocalTensor<float> tmp1Local = tmpBuffer1.Get<float>();
    AscendC::LocalTensor<float> xLocalfp32 = QueueTmpX.Get<float>();
    AscendC::LocalTensor<float> yLocalfp32 = QueueTmpY.Get<float>();
    AscendC::Cast(xLocalfp32, xLocal, RoundMode::CAST_NONE, this->processDataNum);
    AscendC::PipeBarrier<PIPE_V>();
    
    AscendC::Muls(yLocalfp32, xLocalfp32, float(0.015625), this->processDataNum);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Adds(yLocalfp32, yLocalfp32, float(1), this->processDataNum);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Mul(yLocalfp32, yLocalfp32, yLocalfp32, this->processDataNum);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Mul(yLocalfp32, yLocalfp32, yLocalfp32, this->processDataNum);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Mul(yLocalfp32, yLocalfp32, yLocalfp32, this->processDataNum);
    AscendC::Mul(yLocalfp32, yLocalfp32, yLocalfp32, this->processDataNum);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Mul(yLocalfp32, yLocalfp32, yLocalfp32, this->processDataNum);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Mul(yLocalfp32, yLocalfp32, yLocalfp32, this->processDataNum);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Adds(yLocalfp32, yLocalfp32, float(1), this->processDataNum);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Mul(yLocalfp32, yLocalfp32, yLocalfp32, this->processDataNum);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Adds(yLocalfp32, yLocalfp32, float(1), this->processDataNum);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Duplicate(tmp1Local, float(1), this->processDataNum);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Div(yLocalfp32, tmp1Local, yLocalfp32, this->processDataNum);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Muls(yLocalfp32, yLocalfp32, float(-2), this->processDataNum);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Adds(yLocalfp32, yLocalfp32, float(1), this->processDataNum);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Mul(yLocalfp32, xLocalfp32, yLocalfp32, this->processDataNum);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Cast(yLocal, yLocalfp32, RoundMode::CAST_RINT, this->processDataNum);
    AscendC::PipeBarrier<PIPE_V>();
}

template <typename TYPE_X, typename TYPE_Y>
__aicore__ inline void KernelMish<TYPE_X, TYPE_Y>::ComputeSupPerf(LocalTensor<TYPE_X> &xLocal, LocalTensor<TYPE_Y> &yLocal)
{
    AscendC::LocalTensor<TYPE_X> tmp1Local = tmpBuffer1.Get<TYPE_X>();
    AscendC::Muls(yLocal, xLocal, TYPE_X(0.015625), this->processDataNum);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Adds(yLocal, yLocal, TYPE_X(1), this->processDataNum);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Mul(yLocal, yLocal, yLocal, this->processDataNum);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Mul(yLocal, yLocal, yLocal, this->processDataNum);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Mul(yLocal, yLocal, yLocal, this->processDataNum);
    AscendC::Mul(yLocal, yLocal, yLocal, this->processDataNum);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Mul(yLocal, yLocal, yLocal, this->processDataNum);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Mul(yLocal, yLocal, yLocal, this->processDataNum);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Adds(yLocal, yLocal, TYPE_X(1), this->processDataNum);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Mul(yLocal, yLocal, yLocal, this->processDataNum);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Adds(yLocal, yLocal, TYPE_X(1), this->processDataNum);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Duplicate(tmp1Local, TYPE_X(-2), this->processDataNum);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Div(yLocal, tmp1Local, yLocal, this->processDataNum);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Adds(yLocal, yLocal, TYPE_X(1), this->processDataNum);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Mul(yLocal, xLocal, yLocal, this->processDataNum);
    AscendC::PipeBarrier<PIPE_V>();
}

template <typename TYPE_X, typename TYPE_Y>
__aicore__ inline void KernelMish<TYPE_X, TYPE_Y>::ComputeHighPerf16(LocalTensor<TYPE_X> &xLocal, LocalTensor<TYPE_Y> &yLocal)
{
    AscendC::LocalTensor<float> xLocalfp32 = QueueTmpX.Get<float>();
    AscendC::LocalTensor<float> yLocalfp32 = QueueTmpY.Get<float>();        
    AscendC::LocalTensor<float> tmp1Local = tmpBuffer1.Get<float>();
    AscendC::LocalTensor<float> tmp2Local = tmpBuffer2.Get<float>();
    AscendC::LocalTensor<uint8_t> cmp1 = tmpBuffer3.Get<uint8_t>();

    AscendC::Cast(xLocalfp32, xLocal, RoundMode::CAST_NONE, this->processDataNum);
    AscendC::PipeBarrier<PIPE_V>();
    uint32_t comparelength = (this->processDataNum + COMPARE_ALIGN - 1) / COMPARE_ALIGN * COMPARE_ALIGN;
    AscendC::CompareScalar(cmp1, xLocalfp32, float(0), CMPMODE::GT, comparelength);   
    AscendC::PipeBarrier<PIPE_V>();
    
    AscendC::Muls(tmp1Local , xLocalfp32, float(-1), this->processDataNum);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Exp(tmp1Local, tmp1Local, this->processDataNum);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Muls(tmp1Local, tmp1Local, float(2), this->processDataNum);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Adds(tmp1Local, tmp1Local, float(1), this->processDataNum);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Muls(tmp2Local , xLocalfp32, float(-2), this->processDataNum);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Exp(tmp2Local, tmp2Local, this->processDataNum);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Muls(tmp2Local, tmp2Local, float(2), this->processDataNum);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Add(tmp2Local, tmp1Local, tmp2Local, this->processDataNum);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Div(tmp1Local, tmp1Local, tmp2Local, this->processDataNum);    // res for x > 0
    AscendC::PipeBarrier<PIPE_V>();
    
    AscendC::Muls(tmp2Local , xLocalfp32, float(2), this->processDataNum);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Exp(tmp2Local, tmp2Local, this->processDataNum);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Exp(yLocalfp32, xLocalfp32, this->processDataNum);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Muls(yLocalfp32, yLocalfp32, float(2), this->processDataNum);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Add(yLocalfp32, yLocalfp32, tmp2Local, this->processDataNum);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Adds(tmp2Local, yLocalfp32, float(2), this->processDataNum);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Div(yLocalfp32, yLocalfp32, tmp2Local, this->processDataNum);    //res for x <= 0
    AscendC::PipeBarrier<PIPE_V>();
    
    AscendC::Select(yLocalfp32, cmp1, tmp1Local, yLocalfp32, SELMODE::VSEL_TENSOR_TENSOR_MODE, this->processDataNum);
    AscendC::Mul(yLocalfp32, yLocalfp32, xLocalfp32, this->processDataNum);
    AscendC::PipeBarrier<PIPE_V>();

    AscendC::Cast(yLocal, yLocalfp32, RoundMode::CAST_RINT, this->processDataNum);
    AscendC::PipeBarrier<PIPE_V>();
}

template <typename TYPE_X, typename TYPE_Y>
__aicore__ inline void KernelMish<TYPE_X, TYPE_Y>::ComputeHighPerf(LocalTensor<TYPE_X> &xLocal, LocalTensor<TYPE_Y> &yLocal)
{
    AscendC::LocalTensor<TYPE_X> tmp1Local = tmpBuffer1.Get<TYPE_X>();
    AscendC::LocalTensor<TYPE_X> tmp2Local = tmpBuffer2.Get<TYPE_X>();
    AscendC::LocalTensor<uint8_t> cmp1 = tmpBuffer3.Get<uint8_t>();

    uint32_t comparelength = (this->processDataNum + COMPARE_ALIGN - 1) / COMPARE_ALIGN * COMPARE_ALIGN;
    AscendC::CompareScalar(cmp1, xLocal, TYPE_X(0), CMPMODE::GT, comparelength);   
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Muls(tmp1Local , xLocal, TYPE_X(-1), this->processDataNum);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Exp(tmp1Local, tmp1Local, this->processDataNum);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Muls(tmp1Local, tmp1Local, TYPE_X(2), this->processDataNum);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Adds(tmp1Local, tmp1Local, TYPE_X(1), this->processDataNum);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Muls(tmp2Local , xLocal, TYPE_X(-2), this->processDataNum);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Exp(tmp2Local, tmp2Local, this->processDataNum);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Muls(tmp2Local, tmp2Local, TYPE_X(2), this->processDataNum);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Add(tmp2Local, tmp1Local, tmp2Local, this->processDataNum);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Div(tmp1Local, tmp1Local, tmp2Local, this->processDataNum);    // res for x > 0
    AscendC::PipeBarrier<PIPE_V>();
    
    AscendC::Muls(tmp2Local , xLocal, TYPE_X(2), this->processDataNum);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Exp(tmp2Local, tmp2Local, this->processDataNum);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Exp(yLocal, xLocal, this->processDataNum);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Muls(yLocal, yLocal, TYPE_X(2), this->processDataNum);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Add(yLocal, yLocal, tmp2Local, this->processDataNum);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Adds(tmp2Local, yLocal, TYPE_X(2), this->processDataNum);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Div(yLocal, yLocal, tmp2Local, this->processDataNum);    //res for x <= 0
    AscendC::PipeBarrier<PIPE_V>();
    
    AscendC::Select(yLocal, cmp1, tmp1Local, yLocal, SELMODE::VSEL_TENSOR_TENSOR_MODE, this->processDataNum);
    AscendC::Mul(yLocal, yLocal, xLocal, this->processDataNum);
    AscendC::PipeBarrier<PIPE_V>();
}

template <typename TYPE_X, typename TYPE_Y>
__aicore__ inline void KernelMish<TYPE_X, TYPE_Y>::Compute(int32_t progress)
{
    AscendC::LocalTensor<TYPE_X> xLocal = inQueueX.DeQue<TYPE_X>();
    AscendC::LocalTensor<TYPE_Y> yLocal = outQueueY.AllocTensor<TYPE_Y>();

    #if defined(SUPER_PERFORMANCE) && SUPER_PERFORMANCE == 1 
        if constexpr (std::is_same_v<TYPE_X, __bf16>) {
            ComputeSupPerfBf16(xLocal, yLocal);
        } else if constexpr (std::is_same_v<TYPE_X, float> || std::is_same_v<TYPE_X, half>){
            ComputeSupPerf(xLocal, yLocal);
        }
    #else
        if constexpr (std::is_same_v<TYPE_X, __bf16> || std::is_same_v<TYPE_X, half>) { 
            ComputeHighPerf16(xLocal, yLocal);
        } else if constexpr (std::is_same_v<TYPE_X, float>) {
            ComputeHighPerf(xLocal, yLocal);
        }    
    #endif 
    
    outQueueY.EnQue<TYPE_Y>(yLocal);
    inQueueX.FreeTensor(xLocal);
}

template <typename TYPE_X, typename TYPE_Y>
__aicore__ inline void KernelMish<TYPE_X, TYPE_Y>::Process()
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

} // namespace MyMish
#endif // MISH_H

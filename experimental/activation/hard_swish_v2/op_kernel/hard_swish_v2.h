/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2025 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Pei Haobo<@xiaopei-1>
 * - Su Tonghua <@sutonghua>
 *
 * This program is free software: you can redistribute it and/or modify it.
 * Licensed under the CANN Open Software License Agreement Version 2.0 (the "License").
 * You may not use this file except in compliance with the License.
 * See the LICENSE file at the root of the repository for the full text of the License.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTIES OF ANY KIND, EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file hard_swish_v2.h
 * \brief
 */
#ifndef HARD_SWISH_V2_H
#define HARD_SWISH_V2_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "hard_swish_v2_tiling_data.h"
#include "hard_swish_v2_tiling_key.h"

namespace NsHardSwishV2 {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

template <typename T>
class HardSwishV2
{
public:
    __aicore__ inline HardSwishV2(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR z, const HardSwishV2TilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int32_t progress);
    __aicore__ inline void CopyOut(int32_t progress);
    __aicore__ inline void Compute(int32_t progress);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueZ;
    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMZ;

    TBuf<QuePosition::VECCALC> tmpBuf0;
    TBuf<QuePosition::VECCALC> tmpBuf1;
    uint32_t coreDataNum;
    uint32_t tileNum;
    uint32_t tileDataNum;
    uint32_t tailDataNum;
    uint32_t processDataNum;
};

template <typename T>
__aicore__ inline void HardSwishV2<T>::Init(GM_ADDR x, GM_ADDR z, const HardSwishV2TilingData* tilingData)
{
    ASSERT(AscendC::GetBlockNum() != 0 && "block dim can not be zero!");
    uint32_t coreNum = AscendC::GetBlockIdx();
    uint32_t globalBufferIndex = tilingData->bigCoreDataNum * AscendC::GetBlockIdx();
    this->tileDataNum = tilingData->tileDataNum;
    if (coreNum < tilingData->tailBlockNum) { 
        this->coreDataNum = tilingData->bigCoreDataNum;
        this->tileNum = tilingData->finalBigTileNum;
        this->tailDataNum = tilingData->bigTailDataNum;
    }
    else { 
        this->coreDataNum = tilingData->smallCoreDataNum;
        this->tileNum = tilingData->finalSmallTileNum;
        this->tailDataNum = tilingData->smallTailDataNum;
        globalBufferIndex -= (tilingData->bigCoreDataNum - tilingData->smallCoreDataNum) * (AscendC::GetBlockIdx() - tilingData->tailBlockNum);
    }
    inputGMX.SetGlobalBuffer((__gm__ T*)x + globalBufferIndex, this->coreDataNum);
    outputGMZ.SetGlobalBuffer((__gm__ T*)z + globalBufferIndex, this->coreDataNum);

    pipe.InitBuffer(inputQueueX, BUFFER_NUM, this->tileDataNum * sizeof(T));
    pipe.InitBuffer(outputQueueZ, BUFFER_NUM, this->tileDataNum * sizeof(T));
    if constexpr (AscendC::Std::is_same<T, bfloat16_t>::value){
        pipe.InitBuffer(tmpBuf0, this->tileDataNum * sizeof(float));
        pipe.InitBuffer(tmpBuf1, this->tileDataNum * sizeof(float));
    }else{
        pipe.InitBuffer(tmpBuf0, this->tileDataNum * sizeof(T));
    }
}

template <typename T>
__aicore__ inline void HardSwishV2<T>::CopyIn(int32_t progress)
{
    AscendC::LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    AscendC::DataCopy(xLocal, inputGMX[progress * this->tileDataNum], this->processDataNum);
    inputQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void HardSwishV2<T>::CopyOut(int32_t progress)
{
    AscendC::LocalTensor<T> zLocal = outputQueueZ.DeQue<T>();
    AscendC::DataCopy(outputGMZ[progress * this->tileDataNum], zLocal, this->processDataNum);
    outputQueueZ.FreeTensor(zLocal);
}

template <typename T>
__aicore__ inline void HardSwishV2<T>::Compute(int32_t progress)
{
    AscendC::LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    AscendC::LocalTensor<T> zLocal = outputQueueZ.AllocTensor<T>();
    if constexpr (AscendC::Std::is_same<T, bfloat16_t>::value || AscendC::Std::is_same<T, float16_t>::value) {
        AscendC::LocalTensor<float> tmp0 = tmpBuf0.Get<float>();
        AscendC::LocalTensor<float> tmp1 = tmpBuf1.Get<float>(); 
        // 将输入从bfloat16转换为float
        AscendC::Cast(tmp0, xLocal, AscendC::RoundMode::CAST_NONE, this->processDataNum);
        AscendC::Cast(tmp1, zLocal, AscendC::RoundMode::CAST_NONE, this->processDataNum);
        PipeBarrier<PIPE_V>();
        // 执行计算 (使用float类型)
        AscendC::Adds(tmp1, tmp0, 3.0f, this->processDataNum);
        PipeBarrier<PIPE_V>();
        AscendC::Maxs(tmp1, tmp1, 0.0f, this->processDataNum);
        PipeBarrier<PIPE_V>();
        AscendC::Mins(tmp1, tmp1, 6.0f, this->processDataNum);
        PipeBarrier<PIPE_V>();
        AscendC::Duplicate<float>(tmp0, 6.0f, this->processDataNum);
        PipeBarrier<PIPE_V>();
        AscendC::Div(tmp1, tmp1, tmp0, this->processDataNum);
        PipeBarrier<PIPE_V>();
        AscendC::Mul(tmp1, tmp0, tmp1, this->processDataNum); // 使用原始x值
        PipeBarrier<PIPE_V>();
        // 将结果从float转换回bfloat16或float16
        if(AscendC::Std::is_same<T, bfloat16_t>::value){
            AscendC::Cast(zLocal, tmp1,AscendC::RoundMode::CAST_RINT, this->processDataNum);
        }else{
            AscendC::Cast(zLocal, tmp1, AscendC::RoundMode::CAST_NONE, this->processDataNum);     
        }
    }else{
        AscendC::LocalTensor<T> tmp0 = tmpBuf0.Get<T>();
        AscendC::Adds(zLocal, xLocal, (T)3.0f,this->processDataNum);
        PipeBarrier<PIPE_V>();
        AscendC::Maxs(zLocal, zLocal, (T)0.0f, this->processDataNum);
        PipeBarrier<PIPE_V>();
        AscendC::Mins(zLocal, zLocal, (T)6.0f,this->processDataNum);
        PipeBarrier<PIPE_V>();
        AscendC::Duplicate<T>(tmp0,(T)6.0, this->processDataNum);
        PipeBarrier<PIPE_V>();
        AscendC::Div(zLocal, zLocal,tmp0,this->processDataNum);
        PipeBarrier<PIPE_V>();
        AscendC::Mul(zLocal,xLocal, zLocal,this->processDataNum);
    }    
    outputQueueZ.EnQue<T>(zLocal);
    inputQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void HardSwishV2<T>::Process()
{
    int32_t loopCount = this->tileNum;
    this->processDataNum = this->tileDataNum;
    for (int32_t i = 0; i < loopCount; i++) {
        if (i == this->tileNum - 1) {
            this->processDataNum = this->tailDataNum;
        }
        CopyIn(i);
        Compute(i);
        CopyOut(i);
    }
}

} // namespace NsHardSwishV2
#endif // HARD_SWISH_V2_H
/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2025 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Zhou Jianhua <@LePenseur>
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
 * \file celu_v2.h
 * \brief
 */
#ifndef CELU_V2_H
#define CELU_V2_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "celu_v2_tiling_data.h"
#include "celu_v2_tiling_key.h"

namespace NsCeluV2 {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

template <typename T>
class CeluV2 {
public:
    __aicore__ inline CeluV2(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const CeluV2TilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int32_t progress);
    __aicore__ inline void CopyOut(int32_t progress);
    __aicore__ inline void Compute(int32_t progress);

private:
    AscendC::TPipe pipe;
    AscendC::TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    AscendC::TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;
    AscendC::TBuf<QuePosition::VECCALC> tmpBuffer1, tmpBuffer2;
    AscendC::GlobalTensor<T> inputGMX;
    AscendC::GlobalTensor<T> outputGMY;

    uint32_t coreDataNum;
    uint32_t tileNum;
    uint32_t tileDataNum;
    uint32_t tailDataNum;
    uint32_t processDataNum;
    float alpha;
    float invAlpha;
};

template <typename T>
__aicore__ inline void CeluV2<T>::Init(GM_ADDR x, GM_ADDR y, const CeluV2TilingData* tilingData)
{
    ASSERT(AscendC::GetBlockNum() != 0 && "block dim can not be zero!");
    uint32_t coreNum = AscendC::GetBlockIdx();
    uint32_t globalBufferIndex = tilingData->bigCoreDataNum * AscendC::GetBlockIdx();
    this->tileDataNum = tilingData->tileDataNum;
    this->alpha = tilingData->alpha;
    this->invAlpha = tilingData->invAlpha;
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
    outputGMY.SetGlobalBuffer((__gm__ T*)y + globalBufferIndex, this->coreDataNum);
    pipe.InitBuffer(inputQueueX, BUFFER_NUM, this->tileDataNum * sizeof(T));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, this->tileDataNum * sizeof(T));
    pipe.InitBuffer(tmpBuffer1, this->tileDataNum * sizeof(T));
    pipe.InitBuffer(tmpBuffer2, this->tileDataNum * sizeof(T));
}

template <typename T>
__aicore__ inline void CeluV2<T>::CopyIn(int32_t progress)
{
    AscendC::LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    AscendC::DataCopy(xLocal, inputGMX[progress * this->tileDataNum], this->processDataNum);
    inputQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void CeluV2<T>::CopyOut(int32_t progress)
{
    AscendC::LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
    AscendC::DataCopy(outputGMY[progress * this->tileDataNum], yLocal, this->processDataNum);
    outputQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void CeluV2<T>::Compute(int32_t progress)
{
    AscendC::LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    AscendC::LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();
    LocalTensor<T> tmpTensor1 = tmpBuffer1.Get<T>();
    LocalTensor<T> tmpTensor2 = tmpBuffer2.Get<T>();
    AscendC::Muls(tmpTensor1, xLocal, static_cast<T>(invAlpha), this->tileDataNum);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Exp(tmpTensor1, tmpTensor1, this->tileDataNum);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Adds(tmpTensor1, tmpTensor1, static_cast<T>(-1), this->tileDataNum);
    AscendC::Muls(tmpTensor1, tmpTensor1, static_cast<T>(alpha), this->tileDataNum);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Mins(tmpTensor1, tmpTensor1, static_cast<T>(0), this->tileDataNum);
    AscendC::Maxs(tmpTensor2, xLocal, static_cast<T>(0), this->tileDataNum);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Add(yLocal, tmpTensor2, tmpTensor1, this->tileDataNum);
    outputQueueY.EnQue<T>(yLocal);
    inputQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void CeluV2<T>::Process()
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

} // namespace NsCeluV2
#endif // LEAKYRELU_V2_H

/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2025 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Li Wen <@liwenkkklll>
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
 * \file leaky_relu_grad_v2.h
 * \brief
*/
#ifndef LEAKYRELUGRADV2_H
#define LEAKYRELUGRADV2_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "leaky_relu_grad_v2_tiling_data.h"
#include "leaky_relu_grad_v2_tiling_key.h"

namespace NsLeakyReluGradV2 {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

template <typename T>
class LeakyReluGradV2 {
public:
    __aicore__ inline LeakyReluGradV2(){};

    __aicore__ inline void Init(GM_ADDR dy, GM_ADDR x, GM_ADDR dx, const LeakyReluGradV2TilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int32_t progress);
    __aicore__ inline void CopyOut(int32_t progress);
    __aicore__ inline void Compute(int32_t progress);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueY;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueZ;
    TBuf<TPosition::VECCALC> tmpBuffer;            // 临时计算缓冲区
    TBuf<TPosition::VECCALC> tmp1Buffer;
    GlobalTensor<T> inputGMX;
    GlobalTensor<T> inputGMY;
    GlobalTensor<T> outputGMZ;

    uint32_t coreDataNum;
    uint32_t tileNum;
    uint32_t tileDataNum;
    uint32_t tailDataNum;
    uint32_t processDataNum;
    float negativeSlope;
};

template <typename T>
__aicore__ inline void LeakyReluGradV2<T>::Init(GM_ADDR dy, GM_ADDR x, GM_ADDR dx, const LeakyReluGradV2TilingData* tilingData)
{
        ASSERT(AscendC::GetBlockNum() != 0 && "block dim can not be zero!");
        uint32_t coreNum = AscendC::GetBlockIdx();
        uint32_t globalBufferIndex = tilingData->bigCoreDataNum * AscendC::GetBlockIdx();
        this->tileDataNum = tilingData->tileDataNum;
        this->negativeSlope=tilingData->negativeSlope;
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
        inputGMX.SetGlobalBuffer((__gm__ T*)dy + globalBufferIndex, this->coreDataNum);
        inputGMY.SetGlobalBuffer((__gm__ T*)x + globalBufferIndex, this->coreDataNum);
        outputGMZ.SetGlobalBuffer((__gm__ T*)dx + globalBufferIndex, this->coreDataNum);
        pipe.InitBuffer(inputQueueX, BUFFER_NUM, this->tileDataNum * sizeof(T));
        pipe.InitBuffer(inputQueueY, BUFFER_NUM, this->tileDataNum * sizeof(T));
        pipe.InitBuffer(outputQueueZ, BUFFER_NUM, this->tileDataNum * sizeof(T));
        pipe.InitBuffer(tmpBuffer, this->tileDataNum * sizeof(T));
        pipe.InitBuffer(tmp1Buffer, this->tileDataNum * sizeof(T));
    }

template <typename T>
__aicore__ inline void LeakyReluGradV2<T>::CopyIn(int32_t progress)
{
    AscendC::LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    AscendC::LocalTensor<T> yLocal = inputQueueY.AllocTensor<T>();
    AscendC::DataCopy(xLocal, inputGMX[progress * this->tileDataNum], this->processDataNum);
    AscendC::DataCopy(yLocal, inputGMY[progress * this->tileDataNum], this->processDataNum);
    inputQueueX.EnQue(xLocal);
    inputQueueY.EnQue(yLocal);
}

template <typename T>
__aicore__ inline void LeakyReluGradV2<T>::CopyOut(int32_t progress)
{
    AscendC::LocalTensor<T> zLocal = outputQueueZ.DeQue<T>();
    AscendC::DataCopy(outputGMZ[progress * this->tileDataNum], zLocal, this->processDataNum);
    outputQueueZ.FreeTensor(zLocal);
}

template <typename T>
__aicore__ inline void LeakyReluGradV2<T>::Compute(int32_t progress)
{
    AscendC::LocalTensor<T> dyLocal = inputQueueX.DeQue<T>();
    AscendC::LocalTensor<T> xLocal = inputQueueY.DeQue<T>();
    AscendC::LocalTensor<T> dxLocal = outputQueueZ.AllocTensor<T>();
    
    
    AscendC::LocalTensor<uint8_t> temp = tmpBuffer.Get<uint8_t>();
    AscendC::LocalTensor<T> negativePart = tmp1Buffer.Get<T>();
    // LeakyReLU梯度计算: dx = (x > 0) ? dy : negativeSlope * dy
    AscendC::CompareScalar(temp, xLocal, static_cast<T>(0.0f), AscendC::CMPMODE::GT, this->processDataNum);
    AscendC::DataCopy(negativePart, dyLocal, this->processDataNum);
    AscendC::Muls(negativePart, negativePart, static_cast<T>(this->negativeSlope), this->processDataNum);
    AscendC::Select(dxLocal, temp, dyLocal, negativePart, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, this->processDataNum);
    outputQueueZ.EnQue<T>(dxLocal);
    inputQueueX.FreeTensor(dyLocal);
    inputQueueY.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void LeakyReluGradV2<T>::Process()
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

} // namespace NsLeakyReluGradV2
#endif // LeakyReluGradV2_H
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
 * \file swish.h
 * \brief
 */
#ifndef __SWISH_H__
#define __SWISH_H__

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "swish_tiling_data.h"
#include "swish_tiling_key.h"

namespace NsSwish {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

template <typename TYPE_X, typename TYPE_Y>
class KernelSwish
{
public:
    __aicore__ inline KernelSwish() {}
    __aicore__ inline void Init(GM_ADDR x, float scale, GM_ADDR y, uint32_t smallCoreDataNum,
                                uint32_t bigCoreDataNum, uint32_t finalBigTileNum,
                                uint32_t finalSmallTileNum, uint32_t tileDataNum,
                                uint32_t smallTailDataNum, uint32_t bigTailDataNum,
                                uint32_t tailBlockNum)
    {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        uint32_t coreNum = GetBlockIdx();
        uint32_t globalBufferIndex = bigCoreDataNum * GetBlockIdx();
        this->tileDataNum = tileDataNum;
        if (coreNum < tailBlockNum)
        {
            this->coreDataNum = bigCoreDataNum;
            this->tileNum = finalBigTileNum;
            this->tailDataNum = bigTailDataNum;
        }
        else
        {
            this->coreDataNum = smallCoreDataNum;
            this->tileNum = finalSmallTileNum;
            this->tailDataNum = smallTailDataNum;
            globalBufferIndex -= (bigCoreDataNum - smallCoreDataNum) * (GetBlockIdx() - tailBlockNum);
        }
        xGm.SetGlobalBuffer((__gm__ TYPE_X *)x + globalBufferIndex, this->coreDataNum);
        yGm.SetGlobalBuffer((__gm__ TYPE_Y *)y + globalBufferIndex, this->coreDataNum);

        this->scale = -1.0f * scale;
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_X));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_Y));
        if constexpr (!std::is_same_v<TYPE_X, float>)
        {
            pipe.InitBuffer(calcBuf1, this->tileDataNum * sizeof(float));
            pipe.InitBuffer(calcBuf2, this->tileDataNum * sizeof(float));
        }
    }
    __aicore__ inline void Process()
    {
        int32_t loopCount = this->tileNum;
        this->processDataNum = this->tileDataNum;
        for (int32_t i = 0; i < loopCount-1; i++)
        {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
        this->processDataNum = this->tailDataNum;
        CopyIn(loopCount-1);
        Compute(loopCount-1);
        CopyOut(loopCount-1);
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        LocalTensor<TYPE_X> xLocal = inQueueX.AllocTensor<TYPE_X>();
        DataCopy(xLocal, xGm[progress * this->tileDataNum], this->processDataNum);
        inQueueX.EnQue(xLocal);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        LocalTensor<TYPE_X> xLocal = inQueueX.DeQue<TYPE_X>();
        LocalTensor<TYPE_Y> yLocal = outQueueY.AllocTensor<TYPE_Y>();
        if constexpr (std::is_same_v<TYPE_X, float>){
            Muls(yLocal, xLocal, this->scale, this->processDataNum);
            Exp(yLocal, yLocal, this->processDataNum);
            Adds(yLocal, yLocal, (TYPE_X)1.0f, this->processDataNum);
            Div(yLocal, xLocal, yLocal, this->processDataNum);
        }else if constexpr (!std::is_same_v<TYPE_X, float>){
            LocalTensor<float> xLocalFp32 = calcBuf1.Get<float>();
            LocalTensor<float> yLocalFp32 = calcBuf2.Get<float>();
            
            Cast(xLocalFp32, xLocal, RoundMode::CAST_NONE, this->processDataNum);
            Muls(yLocalFp32, xLocalFp32, this->scale, this->processDataNum);
            Exp(yLocalFp32, yLocalFp32, this->processDataNum);
            Adds(yLocalFp32, yLocalFp32, 1.0f, this->processDataNum);
            Div(yLocalFp32, xLocalFp32, yLocalFp32, this->processDataNum);
            Cast(yLocal, yLocalFp32, RoundMode::CAST_ROUND, this->processDataNum);
        }
        outQueueY.EnQue<TYPE_Y>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        LocalTensor<TYPE_Y> yLocal = outQueueY.DeQue<TYPE_Y>();
        DataCopy(yGm[progress * this->tileDataNum], yLocal, this->processDataNum);
        outQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    TBuf<QuePosition::VECCALC> calcBuf1;
    TBuf<QuePosition::VECCALC> calcBuf2;
    GlobalTensor<TYPE_X> xGm;
    GlobalTensor<TYPE_Y> yGm;
    uint32_t coreDataNum;
    uint32_t tileNum;
    uint32_t tileDataNum;
    uint32_t tailDataNum;
    uint32_t processDataNum;
    float scale;
};

} // namespace NsSwish
#endif // SWISH_H
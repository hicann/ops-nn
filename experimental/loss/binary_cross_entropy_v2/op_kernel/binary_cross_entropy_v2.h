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
 * \file binary_cross_entropy_v2.h
 * \brief
 */
#ifndef __BINARY_CROSS_ENTROPY_V2_H__
#define __BINARY_CROSS_ENTROPY_V2_H__

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "binary_cross_entropy_v2_tiling_data.h"
#include "binary_cross_entropy_v2_tiling_key.h"

namespace NsBinaryCrossEntropyV2 {

    using namespace AscendC;

    constexpr int32_t BUFFER_NUM = 2;
    
    template <typename T>
    class BinaryCrossEntropyV2 {
    public:
        __aicore__ inline BinaryCrossEntropyV2(){};
    
        __aicore__ inline void Init(GM_ADDR x,GM_ADDR y, GM_ADDR z, const BinaryCrossEntropyV2TilingData* tilingData);
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
        TBuf<AscendC::TPosition::VECCALC> tmpBuf0;
        GlobalTensor<T> inputGMX;
        GlobalTensor<T> inputGMY;
        GlobalTensor<T> outputGMZ;
    
        int64_t coreDataNum;
        int64_t tileNum;
        int64_t tileDataNum;
        int64_t tailDataNum;
        int64_t processDataNum;
    };
    
    template <typename T>
    __aicore__ inline void BinaryCrossEntropyV2<T>::Init(GM_ADDR x,GM_ADDR y, GM_ADDR z, const BinaryCrossEntropyV2TilingData* tilingData)
    {
            int64_t coreNum = AscendC::GetBlockIdx();
            int64_t globalBufferIndex = tilingData->bigCoreDataNum * AscendC::GetBlockIdx();
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
            inputGMY.SetGlobalBuffer((__gm__ T*)y + globalBufferIndex, this->coreDataNum);
            outputGMZ.SetGlobalBuffer((__gm__ T*)z + globalBufferIndex, this->coreDataNum);
            pipe.InitBuffer(inputQueueX, BUFFER_NUM, this->tileDataNum * sizeof(T));
            pipe.InitBuffer(inputQueueY, BUFFER_NUM, this->tileDataNum * sizeof(T));
            pipe.InitBuffer(outputQueueZ, BUFFER_NUM, this->tileDataNum * sizeof(T));
            pipe.InitBuffer(tmpBuf0, this->tileDataNum * sizeof(T));
        }
    
    template <typename T>
    __aicore__ inline void BinaryCrossEntropyV2<T>::CopyIn(int32_t progress)
    {
        AscendC::LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
        AscendC::LocalTensor<T> yLocal = inputQueueY.AllocTensor<T>();
        AscendC::DataCopy(xLocal, inputGMX[progress * this->tileDataNum], this->processDataNum);
        AscendC::DataCopy(yLocal, inputGMY[progress * this->tileDataNum], this->processDataNum);
        inputQueueX.EnQue(xLocal);
        inputQueueY.EnQue(yLocal);
    }
    
    template <typename T>
    __aicore__ inline void BinaryCrossEntropyV2<T>::CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<T> zLocal = outputQueueZ.DeQue<T>();
        AscendC::DataCopy(outputGMZ[progress * this->tileDataNum], zLocal, this->processDataNum);
        outputQueueZ.FreeTensor(zLocal);
    }
    
    template <typename T>
    __aicore__ inline void BinaryCrossEntropyV2<T>::Compute(int32_t progress)
    {
        AscendC::LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
        AscendC::LocalTensor<T> yLocal = inputQueueY.DeQue<T>();
        AscendC::LocalTensor<T> zLocal = outputQueueZ.AllocTensor<T>();
        AscendC::LocalTensor<T> tmpTensor0 = tmpBuf0.Get<T>();

        AscendC::Ln(zLocal, xLocal, this->processDataNum);
        AscendC::Mul(zLocal, yLocal, zLocal, this->processDataNum);
        AscendC::Duplicate(tmpTensor0, static_cast<T>(1.0f), this->processDataNum);
        AscendC::Sub(xLocal, tmpTensor0, xLocal, this->processDataNum);
        AscendC::Ln(xLocal, xLocal, this->processDataNum);
        AscendC::Sub(yLocal, tmpTensor0, yLocal, this->processDataNum);
        AscendC::MulAddDst(zLocal, yLocal, xLocal, this->processDataNum);
        AscendC::Duplicate(tmpTensor0, static_cast<T>(0.0f), this->processDataNum);
        AscendC::Sub(zLocal, tmpTensor0, zLocal, this->processDataNum);

        outputQueueZ.EnQue<T>(zLocal);
        inputQueueX.FreeTensor(xLocal);
        inputQueueX.FreeTensor(yLocal);
    }
    
    template <typename T>
    __aicore__ inline void BinaryCrossEntropyV2<T>::Process()
    {
        int32_t loopCount = this->tileNum;
        this->processDataNum = this->tileDataNum;
        for (int32_t i = 0; i < loopCount-1; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
        this->processDataNum = this->tailDataNum;
        CopyIn(loopCount-1);
        Compute(loopCount-1);
        CopyOut(loopCount-1);
    }

} // namespace NsBinaryCrossEntropyV2
#endif // BINARY_CROSS_ENTROPY_V2_H
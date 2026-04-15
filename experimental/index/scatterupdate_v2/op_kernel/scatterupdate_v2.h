/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2025 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Liang Yanglin <@liang-yanglin>
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
 * \file scatterupdate_v2.h
 * \brief
 */
#ifndef __SCATTERUPDATE_V2_H__
#define __SCATTERUPDATE_V2_H__

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "scatterupdate_v2_tiling_data.h"
#include "scatterupdate_v2_tiling_key.h"

namespace NsScatterupdateV2 {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

template <typename TYPE_X, typename TYPE_Y, typename TYPE_Z>
class ScatterupdateV2 {
public:
    __aicore__ inline ScatterupdateV2(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR xout, const ScatterupdateV2TilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t inbegin,uint32_t num);
    __aicore__ inline void CopyOut(int64_t outbegin,uint32_t num);
    __aicore__ inline void Compute(uint32_t i, int64_t x1, int64_t x2, uint8_t& isinside, int64_t& inbegin, int64_t& outbegin, int64_t& num);

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueY;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> inQueueZ;
    AscendC::GlobalTensor<TYPE_X> xGm;
    AscendC::GlobalTensor<TYPE_Y> yGm;
    AscendC::GlobalTensor<TYPE_Z> zGm;
    int64_t coreDataNum;
    int64_t tileNum;
    int64_t tileDataNum;
    int64_t tailDataNum;
    int64_t processDataNum;
    int64_t stride;
    int64_t low;
    int64_t high;
    int64_t globalBufferIndex;
    int64_t totalIndicesNum;
    int32_t eventIDMTE2ToMTE3 ;
};

template <typename TYPE_X, typename TYPE_Y, typename TYPE_Z>
__aicore__ inline void ScatterupdateV2<TYPE_X, TYPE_Y, TYPE_Z>::Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR xout,const ScatterupdateV2TilingData* tilingData)
{
        ASSERT(AscendC::GetBlockNum() != 0 && "block dim can not be zero!");
        uint32_t coreNum = AscendC::GetBlockIdx();
        this->globalBufferIndex = tilingData->bigCoreDataNum * AscendC::GetBlockIdx();
        this->tileDataNum = tilingData->tileDataNum;
        this->totalIndicesNum = tilingData->totalIndicesNum;
        if (coreNum < tilingData->tailBlockNum) { 
          this->coreDataNum = tilingData->bigCoreDataNum;
          this->tileNum = tilingData->finalBigTileNum;
          this->tailDataNum = tilingData->bigTailDataNum;
        }
        else { 
          this->coreDataNum = tilingData->smallCoreDataNum;
          this->tileNum = tilingData->finalSmallTileNum;
          this->tailDataNum = tilingData->smallTailDataNum;
          this->globalBufferIndex -= (tilingData->bigCoreDataNum - tilingData->smallCoreDataNum) * (AscendC::GetBlockIdx() - tilingData->tailBlockNum);
        }
        this->low = globalBufferIndex;
        this->high = globalBufferIndex + this->coreDataNum;
        this->stride = tilingData->stride;
        xGm.SetGlobalBuffer((__gm__ TYPE_X*)xout + globalBufferIndex, this->coreDataNum);
        yGm.SetGlobalBuffer((__gm__ TYPE_Y*)y , tilingData->totalIndicesNum);
        zGm.SetGlobalBuffer((__gm__ TYPE_Z*)z , tilingData->totalUpdatesNum);
        pipe.InitBuffer(inQueueY, BUFFER_NUM, tilingData->totalIndicesNum * sizeof(TYPE_Y));
        pipe.InitBuffer(inQueueZ, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_Z));
}

template <typename TYPE_X, typename TYPE_Y, typename TYPE_Z>
__aicore__ inline void ScatterupdateV2<TYPE_X, TYPE_Y, TYPE_Z>::CopyIn(int64_t inbegin,uint32_t num)
{
    AscendC::LocalTensor<TYPE_Z> zLocal = inQueueZ.AllocTensor<TYPE_Z>();
    AscendC::DataCopyExtParams copyParams1{1, static_cast<uint32_t>(num * sizeof(TYPE_Z)), 0, 0, 0};
    AscendC::DataCopyPadExtParams<TYPE_Z> padParams1{true, 0, 0, 0};
    AscendC::DataCopyPad(zLocal, zGm[inbegin], copyParams1, padParams1);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(eventIDMTE2ToMTE3);
    inQueueZ.EnQue<TYPE_Z>(zLocal);
}

template <typename TYPE_X, typename TYPE_Y, typename TYPE_Z>
__aicore__ inline void ScatterupdateV2<TYPE_X, TYPE_Y, TYPE_Z>::CopyOut(int64_t outbegin,uint32_t num)
{
    AscendC::LocalTensor<TYPE_Z> zLocal = inQueueZ.DeQue<TYPE_Z>();  
    AscendC::DataCopyExtParams copyParams2{1, static_cast<uint32_t>(num * sizeof(TYPE_Z)), 0, 0, 0};
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(eventIDMTE2ToMTE3);
    AscendC::DataCopyPad(xGm[outbegin], zLocal, copyParams2);
    inQueueZ.FreeTensor(zLocal);
}

template <typename TYPE_X, typename TYPE_Y, typename TYPE_Z>
__aicore__ inline void ScatterupdateV2<TYPE_X, TYPE_Y, TYPE_Z>::Compute(uint32_t i, int64_t x1, int64_t x2, uint8_t& isinside, int64_t& inbegin, int64_t& outbegin, int64_t& num)
{
    //不在区间内的两种情况
        isinside = 1;
        if(x1 >= this->high||x2 < this->low) isinside = 0;
        else isinside = 1;
        if(isinside == 1){
            int64_t s = x1 > this->low  ? x1 : this->low;   // max
            int64_t e = x2 < this->high ? x2 : this->high;  // min
            int64_t len = e - s;
            if (len <= 0) {              // 再次保护
                isinside = 0;
                return;
            }
            isinside = 1;
            outbegin = s - this->low;
            inbegin  = i * this->stride + s - x1;
            num      = len;
        }
}

template <typename TYPE_X, typename TYPE_Y, typename TYPE_Z>
__aicore__ inline void ScatterupdateV2<TYPE_X, TYPE_Y, TYPE_Z>::Process()
{
    uint8_t flag;
    uint32_t loopCount;
    int64_t inbegin;
    int64_t outbegin;
    int64_t totalnum;
    uint32_t realnum;
    int64_t x1;
    int64_t x2;
    uint32_t copyloopcount;
    realnum = this->tileDataNum;
    loopCount = this->totalIndicesNum;
    flag = 0;
    eventIDMTE2ToMTE3 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_MTE3));
    AscendC::LocalTensor<TYPE_Y> yLocal = inQueueY.AllocTensor<TYPE_Y>();
    AscendC::DataCopyExtParams copyParams0{1, static_cast<uint32_t>(loopCount * sizeof(TYPE_Y)), 0, 0, 0};
    AscendC::DataCopyPadExtParams<TYPE_Y> padParams0{true, 0, 0, 0};
    AscendC::DataCopyPad(yLocal, yGm, copyParams0, padParams0);
    for (uint32_t i = 0; i < loopCount; i++) {
        x1 = yLocal.GetValue(i) * this->stride;
        x2 = x1 + this->stride; 
        Compute(i, x1, x2, flag, inbegin, outbegin, totalnum);
        if(flag==1){
            copyloopcount = (totalnum + this->tileDataNum - 1)/this->tileDataNum;
            for(uint32_t j = 0; j < copyloopcount; j++){
                if(j == copyloopcount - 1) {
                    realnum = totalnum % this->tileDataNum;
                    if (realnum == 0) realnum = this->tileDataNum;
                } else realnum = this->tileDataNum;
                CopyIn(inbegin, realnum);
                inbegin = inbegin + realnum;
                CopyOut(outbegin, realnum);
                outbegin = outbegin + realnum;
            }
        }   
    }
    inQueueY.FreeTensor(yLocal);
}

} // namespace NsScatterupdateV2
#endif // SCATTERUPDATE_V2_H
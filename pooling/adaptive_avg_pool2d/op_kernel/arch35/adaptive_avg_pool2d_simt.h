/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file adaptive_avg_pool2d_simt.h
 * \brief adaptive_avg_pool2d implied by simt
 */

#ifndef ADAPTIVE_AVG_POOL2D_SIMT_H
#define ADAPTIVE_AVG_POOL2D_SIMT_H

#include "kernel_operator.h"
#include "../inc/load_store_utils.h"
#include "../inc/platform.h"
#include "../inc/kernel_utils.h"
#include "adaptive_avg_pool2d_struct.h"

using namespace AscendC;

namespace AdaptiveAvgPool2dOp {
constexpr static uint32_t THREAD_DIM = 512;
constexpr static uint32_t TILING_DATA_NUM = 8;
constexpr static uint32_t SIMT_PARAMS_NUM = 32;
constexpr static uint32_t IDX0 = 0;
constexpr static uint32_t IDX1 = 1;
constexpr static uint32_t IDX2 = 2;
constexpr static uint32_t IDX3 = 3;

template <typename VALUE_T, typename OFFSET_T>
class AdaptiveAvgPool2dSimt {
public:
    __aicore__ inline AdaptiveAvgPool2dSimt(TPipe *pipe,
                                                  const AdaptivePool2DSimtTilingData *__restrict__ tilingData)
        : pipe_(pipe), tilingData_(tilingData)
    {
    }

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y);
    __aicore__ inline void Process();

private:
    TPipe *pipe_;
    AscendC::GlobalTensor<VALUE_T> x_;
    AscendC::GlobalTensor<VALUE_T> y_;
    const AdaptivePool2DSimtTilingData *tilingData_;
    TBuf<TPosition::VECCALC> simtTilingDataBuf_;
    TBuf<TPosition::VECCALC> paramBuf_;
};

template <typename DIV_T>
__simt_callee__ __aicore__ __attribute__((always_inline)) inline static DIV_T CalStartIdx(DIV_T outIdx,  DIV_T inLen, DIV_T magicOutLen, DIV_T shiftOutLen)
{
    DIV_T pStart = outIdx * inLen;
    return Simt::UintDiv<DIV_T>(pStart, magicOutLen, shiftOutLen);
}

template <typename DIV_T>
__simt_callee__ __aicore__ __attribute__((always_inline)) inline static DIV_T CalEndIdx(DIV_T outIdx,  DIV_T inLen, DIV_T outLen, DIV_T magicOutLen, DIV_T shiftOutLen)
{
    // wOutIndex=0, inW=109, outW=46
    DIV_T pEnd = ((outIdx + 1) * inLen + outLen - 1);
    return Simt::UintDiv<DIV_T>(pEnd, magicOutLen, shiftOutLen);
}

template <typename VALUE_T, typename OFFSET_T>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_DIM) inline void AdaptiveAvgPool2dNchw(
    const __gm__ VALUE_T* x,  __gm__ VALUE_T* y,
    const OFFSET_T magicOH, const OFFSET_T shiftOH,
    const OFFSET_T magicOW, const OFFSET_T shiftOW,
    const OFFSET_T nDims, const OFFSET_T cDims,
    const OFFSET_T inH, const OFFSET_T inW,
    const OFFSET_T outH, const OFFSET_T outW
   )
{
    OFFSET_T count = nDims * cDims * outH * outW;
    for (OFFSET_T index = Simt::GetBlockIdx() * Simt::GetThreadNum() + Simt::GetThreadIdx(); index < count;
         index += Simt::GetBlockNum() * Simt::GetThreadNum()) {
        //强转去掉
        OFFSET_T wDiv = Simt::UintDiv(index, static_cast<OFFSET_T>(magicOW), static_cast<OFFSET_T>(shiftOW));
        OFFSET_T wOutIndex = index - wDiv * static_cast<OFFSET_T>(outW);      //w= index % outW
        OFFSET_T hDiv = Simt::UintDiv(wDiv, static_cast<OFFSET_T>(magicOH), static_cast<OFFSET_T>(shiftOH));
        OFFSET_T hOutIndex = wDiv - hDiv * static_cast<OFFSET_T>(outH);      //h = (index / inw) % inH

        OFFSET_T hStarts = CalStartIdx<OFFSET_T>(hOutIndex, inH, magicOH, shiftOH); // oy*isize/osize
        OFFSET_T hEnds = CalEndIdx<OFFSET_T>(hOutIndex, inH, outH, magicOH, shiftOH);// ((oy+1)* isize + osize - 1)/osize
        OFFSET_T wStarts = CalStartIdx<OFFSET_T>(wOutIndex, inW, magicOW, shiftOW);
        OFFSET_T wEnds = CalEndIdx<OFFSET_T>(wOutIndex, inW, outW, magicOW, shiftOW);

        //input
        OFFSET_T kH = hEnds - hStarts;
        OFFSET_T kW = wEnds - wStarts;
        OFFSET_T div = kH * kW;
        float gradient = 0.0f;

        OFFSET_T curX = hDiv * inH * inW;
        for (OFFSET_T hIndex = hStarts; hIndex < hEnds; ++hIndex) {
            for (OFFSET_T wIndex = wStarts; wIndex < wEnds; ++wIndex) {
                OFFSET_T outputIdx = curX + hIndex * inW + wIndex;
                gradient += static_cast<float>(x[outputIdx]);
            }
        }

        gradient = gradient / static_cast<float>(div);
        y[index] = static_cast<VALUE_T>(gradient);
    }
}

template <typename VALUE_T, typename OFFSET_T>
__aicore__ inline void AdaptiveAvgPool2dSimt<VALUE_T, OFFSET_T>::Init(GM_ADDR x, GM_ADDR y)
{
    x_.SetGlobalBuffer((__gm__ VALUE_T *)(x));
    y_.SetGlobalBuffer((__gm__ VALUE_T *)(y));
    pipe_->InitBuffer(simtTilingDataBuf_, TILING_DATA_NUM * sizeof(int64_t));
    pipe_->InitBuffer(paramBuf_, SIMT_PARAMS_NUM * sizeof(OFFSET_T));
}

template <typename VALUE_T, typename OFFSET_T>
__aicore__ inline void AdaptiveAvgPool2dSimt<VALUE_T, OFFSET_T>::Process()
{
    OFFSET_T magicOsizeH = 0, shiftOsizeH = 0;
    OFFSET_T magicOsizeW = 0, shiftOsizeW = 0;

    GetUintDivMagicAndShift<OFFSET_T>(magicOsizeH, shiftOsizeH, tilingData_->hOutDim);
    GetUintDivMagicAndShift<OFFSET_T>(magicOsizeW, shiftOsizeW, tilingData_->wOutDim);

    DataSyncBarrier<MemDsbT::UB>();
    auto xData = (__gm__ VALUE_T *)x_.GetPhyAddr();
    auto yData = (__gm__ VALUE_T *)y_.GetPhyAddr();

    Simt::VF_CALL<AdaptiveAvgPool2dNchw<VALUE_T, OFFSET_T>>(
        Simt::Dim3(THREAD_DIM),
        xData, yData,
        magicOsizeH, shiftOsizeH,
        magicOsizeW, shiftOsizeW,
        tilingData_->nDim, tilingData_->cDim,
        tilingData_->hInDim, tilingData_->wInDim,
        tilingData_->hOutDim, tilingData_->wOutDim);
}

}

#endif  //ADAPTIVE_AVG_POOL2D_SIMT_H

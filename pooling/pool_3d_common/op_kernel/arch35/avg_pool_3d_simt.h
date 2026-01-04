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
 * \file avg_pool_3d_simt.h
 * \brief avg_pool_3d implied by simt
 */

#ifndef CANN_AVG_POOL_3D_SIMT_H
#define CANN_AVG_POOL_3D_SIMT_H
#include "../inc/platform.h"
#include "../inc/kernel_utils.h"
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"

namespace AvgPool3DSimt {
using namespace AscendC;

constexpr size_t PARAM_NUM = 8;
constexpr size_t TILING_DATA_NUM = 25;
constexpr size_t TILING_DATA_UB_NUM = 32;

#ifdef __DAV_FPGA__
constexpr uint32_t THREAD_NUM = 128;
#else
constexpr uint32_t THREAD_NUM = 512;
#endif

struct AvgPool3DSimtTilingData {
    int64_t nDim;
    int64_t cDim;
    int64_t dInDim;
    int64_t hInDim;
    int64_t wInDim;
    int64_t dOutDim;
    int64_t hOutDim;
    int64_t wOutDim;
    int64_t kD;
    int64_t kH;
    int64_t kW;
    int64_t sD;
    int64_t sH;
    int64_t sW;
    int64_t dD;
    int64_t dH;
    int64_t dW;
    int64_t fPad;
    int64_t bkPad;
    int64_t tPad;
    int64_t bPad;
    int64_t lPad;
    int64_t rPad;
    int64_t divisorOverride;
    int64_t countIncludePad;
};

template <typename X_T, typename TYPE_T, int32_t FORMAT_TYPE>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void AvgPool3dNcSimtCompute(
    __gm__ X_T* x, __gm__ X_T* y, __ubuf__ AvgPool3DSimtTilingData* SimtTilingData, __ubuf__ TYPE_T* AvgPool3DSimtParam);

template <typename X_T, typename TYPE_T, int32_t FORMAT_TYPE>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void AvgPool3dNdSimtCompute(
    __gm__ X_T* x, __gm__ X_T* y, __ubuf__ AvgPool3DSimtTilingData* SimtTilingData, __ubuf__ TYPE_T* AvgPool3DSimtParam);

template <typename X_T, typename TYPE_T, int32_t FORMAT_TYPE>
class AvgPool3DSimtImpl {
public:
__aicore__ inline AvgPool3DSimtImpl(TPipe *pipe, const Pool3DSimtTilingData* __restrict tilingData)
    : pipe_(pipe), tilingData_(tilingData), blockIdx_(GetBlockIdx()), blockNum_(GetBlockNum()) {
}

__aicore__ inline void Init(GM_ADDR x, GM_ADDR y);
__aicore__ inline void Process();

private:
    TPipe *pipe_;
    AscendC::GlobalTensor<X_T> x_;
    AscendC::GlobalTensor<X_T> y_;
    const Pool3DSimtTilingData* tilingData_;
    TBuf<TPosition::VECCALC> simtTilingDataBuf_;
    TBuf<TPosition::VECCALC> paramBuf_;
    uint32_t blockIdx_ = 0;
    uint32_t blockNum_ = 0;
    const uint32_t F32_NEG_INF = 0xff800000;
};

template <typename X_T, typename TYPE_T, int32_t FORMAT_TYPE>
__aicore__ inline void AvgPool3DSimtImpl<X_T, TYPE_T, FORMAT_TYPE>::Init(GM_ADDR x, GM_ADDR y)
{
    x_.SetGlobalBuffer((__gm__ X_T*)(x));
    y_.SetGlobalBuffer((__gm__ X_T*)(y));

    pipe_->InitBuffer(simtTilingDataBuf_, TILING_DATA_UB_NUM * sizeof(int64_t));
    pipe_->InitBuffer(paramBuf_, PARAM_NUM * sizeof(TYPE_T));
}

template <typename X_T, typename TYPE_T, int32_t FORMAT_TYPE>
__aicore__ inline void AvgPool3DSimtImpl<X_T, TYPE_T, FORMAT_TYPE>::Process()
{
    LocalTensor<int64_t> SimtTilingData = simtTilingDataBuf_.Get<int64_t>();
    LocalTensor<TYPE_T> AvgPool3DSimtParam = paramBuf_.Get<TYPE_T>();
    const int64_t* tilingP = reinterpret_cast<const int64_t*>(tilingData_);
    for (uint32_t i = 0; i < TILING_DATA_NUM; i++) {
        SimtTilingData.SetValue(i, tilingP[i]);
    }

    using DIV_T = typename std::conditional<std::is_same<TYPE_T, int32_t>::value, uint32_t, uint64_t>::type;
    DIV_T magicD = 0;
    DIV_T shiftD = 0;
    DIV_T magicH = 0;
    DIV_T shiftH = 0;
    DIV_T magicW = 0;
    DIV_T shiftW = 0;
    DIV_T magicC = 0;
    DIV_T shiftC = 0;
    GetUintDivMagicAndShift<DIV_T>(magicD, shiftD, SimtTilingData(5));
    GetUintDivMagicAndShift<DIV_T>(magicH, shiftH, SimtTilingData(6));
    GetUintDivMagicAndShift<DIV_T>(magicW, shiftW, SimtTilingData(7));
    GetUintDivMagicAndShift<DIV_T>(magicC, shiftC, SimtTilingData(1));

    AvgPool3DSimtParam.SetValue(0, static_cast<TYPE_T>(magicD));
    AvgPool3DSimtParam.SetValue(1, static_cast<TYPE_T>(shiftD));
    AvgPool3DSimtParam.SetValue(2, static_cast<TYPE_T>(magicH));
    AvgPool3DSimtParam.SetValue(3, static_cast<TYPE_T>(shiftH));
    AvgPool3DSimtParam.SetValue(4, static_cast<TYPE_T>(magicW));
    AvgPool3DSimtParam.SetValue(5, static_cast<TYPE_T>(shiftW));
    AvgPool3DSimtParam.SetValue(6, static_cast<TYPE_T>(magicC));
    AvgPool3DSimtParam.SetValue(7, static_cast<TYPE_T>(shiftC));

    DataSyncBarrier<MemDsbT::UB>();
    if constexpr (FORMAT_TYPE == 0) {
        Simt::VF_CALL<AvgPool3dNcSimtCompute<X_T, TYPE_T, FORMAT_TYPE>>(Simt::Dim3(THREAD_NUM), 
            (__gm__ X_T*)x_.GetPhyAddr(), (__gm__ X_T*)y_.GetPhyAddr(), (__ubuf__ AvgPool3DSimtTilingData*)(SimtTilingData.GetPhyAddr()),
            (__ubuf__ TYPE_T*)(AvgPool3DSimtParam.GetPhyAddr()));
    } else if constexpr (FORMAT_TYPE == 1) {
        Simt::VF_CALL<AvgPool3dNdSimtCompute<X_T, TYPE_T, FORMAT_TYPE>>(Simt::Dim3(THREAD_NUM), 
            (__gm__ X_T*)x_.GetPhyAddr(), (__gm__ X_T*)y_.GetPhyAddr(), (__ubuf__ AvgPool3DSimtTilingData*)(SimtTilingData.GetPhyAddr()),
            (__ubuf__ TYPE_T*)(AvgPool3DSimtParam.GetPhyAddr()));
    }
}
 
template <typename X_T, typename TYPE_T, int32_t FORMAT_TYPE>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void AvgPool3dNcSimtCompute(
    __gm__ X_T* x, __gm__ X_T* y, __ubuf__ AvgPool3DSimtTilingData* SimtTilingData, __ubuf__ TYPE_T* AvgPool3DSimtParam)
{
    TYPE_T magicD = AvgPool3DSimtParam[0];
    TYPE_T shiftD = AvgPool3DSimtParam[1];
    TYPE_T magicH = AvgPool3DSimtParam[2];
    TYPE_T shiftH = AvgPool3DSimtParam[3];
    TYPE_T magicW = AvgPool3DSimtParam[4];
    TYPE_T shiftW = AvgPool3DSimtParam[5];

    using DIV_T = typename std::conditional<std::is_same<TYPE_T, int32_t>::value, uint32_t, uint64_t>::type;
    TYPE_T outSize = SimtTilingData->nDim * SimtTilingData->cDim * SimtTilingData->dOutDim * SimtTilingData->hOutDim * SimtTilingData->wOutDim;
    for (DIV_T i = Simt::GetBlockIdx() * Simt::GetThreadNum() + Simt::GetThreadIdx(); i < outSize;
        i += Simt::GetBlockNum() * Simt::GetThreadNum()) {
        DIV_T quotientW = Simt::UintDiv<DIV_T>(i, magicW, shiftW);
        DIV_T quotientH = Simt::UintDiv<DIV_T>(quotientW, magicH, shiftH);
        DIV_T quotientD = Simt::UintDiv<DIV_T>(quotientH, magicD, shiftD);
        TYPE_T pw = i  - quotientW * SimtTilingData->wOutDim;
        TYPE_T ph = quotientW - quotientH * SimtTilingData->hOutDim;
        TYPE_T pd = quotientH  - quotientD * SimtTilingData->dOutDim;
        TYPE_T pnc = quotientD;
        TYPE_T dStart = pd * SimtTilingData->sD - SimtTilingData->fPad;
        TYPE_T hStart = ph * SimtTilingData->sH - SimtTilingData->tPad;
        TYPE_T wStart = pw * SimtTilingData->sW - SimtTilingData->lPad;
        TYPE_T dEnd = Simt::Min(dStart + (TYPE_T)SimtTilingData->kD, (TYPE_T)SimtTilingData->dInDim + (TYPE_T)SimtTilingData->bkPad);
        TYPE_T hEnd = Simt::Min(hStart + (TYPE_T)SimtTilingData->kH, (TYPE_T)SimtTilingData->hInDim + (TYPE_T)SimtTilingData->bPad);
        TYPE_T wEnd = Simt::Min(wStart + (TYPE_T)SimtTilingData->kW, (TYPE_T)SimtTilingData->wInDim + (TYPE_T)SimtTilingData->rPad);
        TYPE_T poolSize = (dEnd - dStart) * (hEnd - hStart) * (wEnd - wStart);
        dStart = Simt::Max(dStart, (TYPE_T)0);
        hStart = Simt::Max(hStart, (TYPE_T)0);
        wStart = Simt::Max(wStart, (TYPE_T)0);
        dEnd = Simt::Min(dEnd, (TYPE_T)SimtTilingData->dInDim);
        hEnd = Simt::Min(hEnd, (TYPE_T)SimtTilingData->hInDim);
        wEnd = Simt::Min(wEnd, (TYPE_T)SimtTilingData->wInDim);
        if(dStart >= dEnd || hStart >= hEnd || wStart >= wEnd) {
            y[i] = 0;
            continue;
        }

        TYPE_T divisorFactor;
        if (SimtTilingData->divisorOverride)  {
            divisorFactor = SimtTilingData->divisorOverride;
        } else {
            if (SimtTilingData->countIncludePad) {
                divisorFactor = poolSize;
            } else {
                divisorFactor = (dEnd - dStart) * (hEnd - hStart) * (wEnd - wStart);
            }
        }
        float sum = 0;
        auto xData = x + pnc * SimtTilingData->dInDim * SimtTilingData->hInDim * SimtTilingData->wInDim;
        for (TYPE_T d = dStart; d < dEnd; d++) {
            for (TYPE_T h = hStart; h < hEnd; h++) {
                for (TYPE_T w = wStart; w < wEnd; w++) {
                    TYPE_T idxOffset = d * SimtTilingData->hInDim * SimtTilingData->wInDim  + h * SimtTilingData->wInDim + w;
                    sum += static_cast<float>(xData[idxOffset]);
                }
            }
        }
        y[i] = static_cast<X_T>(sum / static_cast<float>(divisorFactor));
    }
}

template <typename X_T, typename TYPE_T, int32_t FORMAT_TYPE>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void AvgPool3dNdSimtCompute(
    __gm__ X_T* x, __gm__ X_T* y, __ubuf__ AvgPool3DSimtTilingData* SimtTilingData, __ubuf__ TYPE_T* AvgPool3DSimtParam)
{
    TYPE_T magicD = AvgPool3DSimtParam[0];
    TYPE_T shiftD = AvgPool3DSimtParam[1];
    TYPE_T magicH = AvgPool3DSimtParam[2];
    TYPE_T shiftH = AvgPool3DSimtParam[3];
    TYPE_T magicW = AvgPool3DSimtParam[4];
    TYPE_T shiftW = AvgPool3DSimtParam[5];
    TYPE_T magicC = AvgPool3DSimtParam[6];
    TYPE_T shiftC = AvgPool3DSimtParam[7];

    using DIV_T = typename std::conditional<std::is_same<TYPE_T, int32_t>::value, uint32_t, uint64_t>::type;
    TYPE_T outSize = SimtTilingData->nDim * SimtTilingData->cDim * SimtTilingData->dOutDim * SimtTilingData->hOutDim * SimtTilingData->wOutDim;
    for (DIV_T i = Simt::GetBlockIdx() * Simt::GetThreadNum() + Simt::GetThreadIdx(); i < outSize;
        i += Simt::GetBlockNum() * Simt::GetThreadNum()) {
        DIV_T quotientC = Simt::UintDiv<DIV_T>(i, magicC, shiftC);
        DIV_T quotientW = Simt::UintDiv<DIV_T>(quotientC, magicW, shiftW);
        DIV_T quotientH = Simt::UintDiv<DIV_T>(quotientW, magicH, shiftH);
        DIV_T quotientD = Simt::UintDiv<DIV_T>(quotientH, magicD, shiftD);
        TYPE_T pc = i  - quotientC * SimtTilingData->cDim;
        TYPE_T pw = quotientC - quotientW * SimtTilingData->wOutDim;
        TYPE_T ph = quotientW - quotientH * SimtTilingData->hOutDim;
        TYPE_T pd = quotientH - quotientD * SimtTilingData->dOutDim;
        TYPE_T pn = quotientD;
        TYPE_T dStart = pd * SimtTilingData->sD - SimtTilingData->fPad;
        TYPE_T hStart = ph * SimtTilingData->sH - SimtTilingData->tPad;
        TYPE_T wStart = pw * SimtTilingData->sW - SimtTilingData->lPad;
        TYPE_T dEnd = Simt::Min(dStart + (TYPE_T)SimtTilingData->kD, (TYPE_T)SimtTilingData->dInDim + (TYPE_T)SimtTilingData->bkPad);
        TYPE_T hEnd = Simt::Min(hStart + (TYPE_T)SimtTilingData->kH, (TYPE_T)SimtTilingData->hInDim + (TYPE_T)SimtTilingData->bPad);
        TYPE_T wEnd = Simt::Min(wStart + (TYPE_T)SimtTilingData->kW, (TYPE_T)SimtTilingData->wInDim + (TYPE_T)SimtTilingData->rPad);
        TYPE_T poolSize = (dEnd - dStart) * (hEnd - hStart) * (wEnd - wStart);
        dStart = Simt::Max(dStart, (TYPE_T)0);
        hStart = Simt::Max(hStart, (TYPE_T)0);
        wStart = Simt::Max(wStart, (TYPE_T)0);
        dEnd = Simt::Min(dEnd, (TYPE_T)SimtTilingData->dInDim);
        hEnd = Simt::Min(hEnd, (TYPE_T)SimtTilingData->hInDim);
        wEnd = Simt::Min(wEnd, (TYPE_T)SimtTilingData->wInDim);
        if(dStart >= dEnd || hStart >= hEnd || wStart >= wEnd) {
            y[i] = 0;
            continue;
        }

        TYPE_T divisorFactor;
        if (SimtTilingData->divisorOverride)  {
            divisorFactor = SimtTilingData->divisorOverride;
        } else {
            if (SimtTilingData->countIncludePad) {
                divisorFactor = poolSize;
            } else {
                divisorFactor = (dEnd - dStart) * (hEnd - hStart) * (wEnd - wStart);
            }
        }
        float sum = 0;
        auto xData = x + pn * SimtTilingData->dInDim * SimtTilingData->hInDim * SimtTilingData->wInDim * SimtTilingData->cDim;
        for (TYPE_T d = dStart; d < dEnd; d++) {
            for (TYPE_T h = hStart; h < hEnd; h++) {
                for (TYPE_T w = wStart; w < wEnd; w++) {
                    TYPE_T idxOffset = d * SimtTilingData->hInDim * SimtTilingData->wInDim + h * SimtTilingData->wInDim + w;
                    sum += static_cast<float>(xData[idxOffset * SimtTilingData->cDim + pc]);
                }
            }
        }
        y[i] = static_cast<X_T>(sum / static_cast<float>(divisorFactor));
    }
}
} // namespace AvgPool3DSimt

#endif  // CANN_AVG_POOL_WITH_ARGAVG_V3_SIMT_H
 
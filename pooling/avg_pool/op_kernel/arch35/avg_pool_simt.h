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
 * \file avg_pool__simt.h
 * \brief avg_pool_ implied by simt
 */

#ifndef CANN_AVG_POOL_SIMT_H
#define CANN_AVG_POOL_SIMT_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "avg_pool_struct.h"

namespace AvgPoolSimt {
using namespace AscendC;

constexpr size_t PARAM_NUM = 8;
constexpr size_t TILING_DATA_NUM = 18;
constexpr size_t TILING_DATA_UB_NUM = 32;

#ifdef __DAV_FPGA__
constexpr uint32_t THREAD_NUM = 128;
#else
constexpr uint32_t THREAD_NUM = 512;
#endif

template <typename X_T, typename TYPE_T, int32_t FORMAT_TYPE>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void AvgPoolNcSimtCompute(
    __gm__ X_T* x, __gm__ X_T* y, __ubuf__ AvgPool::AvgPoolSimtTilingData* SimtTilingData, __ubuf__ TYPE_T* AvgPoolSimtParam, int64_t hInDim, int64_t wInDim, int64_t hOutDim, int64_t wOutDim, 
    int64_t sH, int64_t sW, int64_t tPad, int64_t lPad, int64_t bPad, int64_t rPad, int64_t divisorOverride, int64_t countIncludePad);

template <typename X_T, typename TYPE_T, int32_t FORMAT_TYPE>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void AvgPoolNdSimtCompute(
    __gm__ X_T* x, __gm__ X_T* y, __ubuf__ AvgPool::AvgPoolSimtTilingData* SimtTilingData, __ubuf__ TYPE_T* AvgPoolSimtParam, int64_t hInDim, int64_t wInDim, int64_t hOutDim, int64_t wOutDim, 
    int64_t sH, int64_t sW, int64_t tPad, int64_t lPad, int64_t bPad, int64_t rPad, int64_t divisorOverride, int64_t cDim);

template <typename X_T, typename TYPE_T, int32_t FORMAT_TYPE>
class AvgPoolSimtImpl {
public:
__aicore__ inline AvgPoolSimtImpl(TPipe *pipe, const AvgPool::AvgPoolSimtTilingData* __restrict tilingData)
    : pipe_(pipe), tilingData_(tilingData), blockIdx_(GetBlockIdx()), blockNum_(GetBlockNum()) {
}

__aicore__ inline void Init(GM_ADDR x, GM_ADDR y);
__aicore__ inline void Process();

private:
    TPipe *pipe_;
    AscendC::GlobalTensor<X_T> x_;
    AscendC::GlobalTensor<X_T> y_;
    const AvgPool::AvgPoolSimtTilingData* tilingData_;
    TBuf<TPosition::VECCALC> simtTilingDataBuf_;
    TBuf<TPosition::VECCALC> paramBuf_;
    uint32_t blockIdx_ = 0;
    uint32_t blockNum_ = 0;
    const uint32_t F32_NEG_INF = 0xff800000;
};

template <typename X_T, typename TYPE_T, int32_t FORMAT_TYPE>
__aicore__ inline void AvgPoolSimtImpl<X_T, TYPE_T, FORMAT_TYPE>::Init(GM_ADDR x, GM_ADDR y)
{
    x_.SetGlobalBuffer((__gm__ X_T*)(x));
    y_.SetGlobalBuffer((__gm__ X_T*)(y));

    pipe_->InitBuffer(simtTilingDataBuf_, TILING_DATA_UB_NUM * sizeof(int64_t));
    pipe_->InitBuffer(paramBuf_, PARAM_NUM * sizeof(TYPE_T));
}

template <typename X_T, typename TYPE_T, int32_t FORMAT_TYPE>
__aicore__ inline void AvgPoolSimtImpl<X_T, TYPE_T, FORMAT_TYPE>::Process()
{
    LocalTensor<int64_t> SimtTilingData = simtTilingDataBuf_.Get<int64_t>();
    LocalTensor<TYPE_T> AvgPoolSimtParam = paramBuf_.Get<TYPE_T>();
    const int64_t* tilingP = reinterpret_cast<const int64_t*>(tilingData_);
    for (uint32_t i = 0; i < TILING_DATA_NUM; i++) {
        SimtTilingData.SetValue(i, tilingP[i]);
    }

    using DIV_T = typename std::conditional<std::is_same<TYPE_T, int32_t>::value, uint32_t, uint64_t>::type;
    DIV_T magicH = 0;
    DIV_T shiftH = 0;
    DIV_T magicW = 0;
    DIV_T shiftW = 0;
    DIV_T magicC = 0;
    DIV_T shiftC = 0;
    GetUintDivMagicAndShift<DIV_T>(magicH, shiftH, SimtTilingData(4));
    GetUintDivMagicAndShift<DIV_T>(magicW, shiftW, SimtTilingData(5));
    GetUintDivMagicAndShift<DIV_T>(magicC, shiftC, SimtTilingData(1));

    AvgPoolSimtParam.SetValue(0, static_cast<TYPE_T>(magicH));
    AvgPoolSimtParam.SetValue(1, static_cast<TYPE_T>(shiftH));
    AvgPoolSimtParam.SetValue(2, static_cast<TYPE_T>(magicW));
    AvgPoolSimtParam.SetValue(3, static_cast<TYPE_T>(shiftW));
    AvgPoolSimtParam.SetValue(4, static_cast<TYPE_T>(magicC));
    AvgPoolSimtParam.SetValue(5, static_cast<TYPE_T>(shiftC));

    DataSyncBarrier<MemDsbT::UB>();
    if constexpr (FORMAT_TYPE == 0) {
        Simt::VF_CALL<AvgPoolNcSimtCompute<X_T, TYPE_T, FORMAT_TYPE>>(Simt::Dim3(THREAD_NUM), 
            (__gm__ X_T*)x_.GetPhyAddr(), (__gm__ X_T*)y_.GetPhyAddr(), (__ubuf__ AvgPool::AvgPoolSimtTilingData*)(SimtTilingData.GetPhyAddr()),
            (__ubuf__ TYPE_T*)(AvgPoolSimtParam.GetPhyAddr()), SimtTilingData(2), SimtTilingData(3), SimtTilingData(4), SimtTilingData(5), 
            SimtTilingData(8), SimtTilingData(9), SimtTilingData(12), SimtTilingData(14), SimtTilingData(13), SimtTilingData(15), 
            SimtTilingData(16), SimtTilingData(17));
    } else if constexpr (FORMAT_TYPE == 1) {
        Simt::VF_CALL<AvgPoolNdSimtCompute<X_T, TYPE_T, FORMAT_TYPE>>(Simt::Dim3(THREAD_NUM), 
            (__gm__ X_T*)x_.GetPhyAddr(), (__gm__ X_T*)y_.GetPhyAddr(), (__ubuf__ AvgPool::AvgPoolSimtTilingData*)(SimtTilingData.GetPhyAddr()),
            (__ubuf__ TYPE_T*)(AvgPoolSimtParam.GetPhyAddr()),SimtTilingData(2), SimtTilingData(3), SimtTilingData(4), SimtTilingData(5), 
            SimtTilingData(8), SimtTilingData(9), SimtTilingData(12), SimtTilingData(14), SimtTilingData(13), SimtTilingData(15), 
            SimtTilingData(16), SimtTilingData(1));
    }
}
 
template <typename X_T, typename TYPE_T, int32_t FORMAT_TYPE>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void AvgPoolNcSimtCompute(
    __gm__ X_T* x, __gm__ X_T* y, __ubuf__ AvgPool::AvgPoolSimtTilingData* SimtTilingData, __ubuf__ TYPE_T* AvgPoolSimtParam, int64_t hInDim, int64_t wInDim, int64_t hOutDim, int64_t wOutDim, 
    int64_t sH, int64_t sW, int64_t tPad, int64_t lPad, int64_t bPad, int64_t rPad, int64_t divisorOverride, int64_t countIncludePad)
{
    TYPE_T magicH = AvgPoolSimtParam[0];
    TYPE_T shiftH = AvgPoolSimtParam[1];
    TYPE_T magicW = AvgPoolSimtParam[2];
    TYPE_T shiftW = AvgPoolSimtParam[3];

    TYPE_T divisorFactor = divisorOverride;
    TYPE_T cOffset = hInDim * wInDim;
    using DIV_T = typename std::conditional<std::is_same<TYPE_T, int32_t>::value, uint32_t, uint64_t>::type;
    TYPE_T outSize = SimtTilingData->nDim * SimtTilingData->cDim * hOutDim * wOutDim;
    for (DIV_T i = Simt::GetBlockIdx() * Simt::GetThreadNum() + Simt::GetThreadIdx(); i < outSize;
        i += Simt::GetBlockNum() * Simt::GetThreadNum()) {
        DIV_T quotientW = Simt::UintDiv<DIV_T>(i, magicW, shiftW);
        DIV_T quotientH = Simt::UintDiv<DIV_T>(quotientW, magicH, shiftH);
        TYPE_T pnc = quotientH;
        // ph = quotientW - quotientH * hOutDim
        TYPE_T hStart = (quotientW - quotientH * hOutDim) * sH - tPad;
        // pw = i - quotientW * wOutDim
        TYPE_T wStart = (i - quotientW * wOutDim) * sW - lPad;
        TYPE_T hEnd = Simt::Min(hStart + (TYPE_T)SimtTilingData->kH, (TYPE_T)hInDim + (TYPE_T)bPad);
        TYPE_T wEnd = Simt::Min(wStart + (TYPE_T)SimtTilingData->kW, (TYPE_T)wInDim + (TYPE_T)rPad);
        TYPE_T poolSize = (hEnd - hStart) * (wEnd - wStart);
        hStart = Simt::Max(hStart, (TYPE_T)0);
        wStart = Simt::Max(wStart, (TYPE_T)0);
        hEnd = Simt::Min(hEnd, (TYPE_T)hInDim);
        wEnd = Simt::Min(wEnd, (TYPE_T)wInDim);
        if(hStart >= hEnd || wStart >= wEnd) {
            y[i] = 0;
            continue;
        }

        if (!divisorOverride) {
            divisorFactor = countIncludePad ? poolSize : ((hEnd - hStart) * (wEnd - wStart));
        }

        float sum = 0;
        auto xData = x + pnc * hInDim * wInDim;
        for (TYPE_T h = hStart; h < hEnd; h++) {
            TYPE_T hOffset = h * wInDim;
            for (TYPE_T w = wStart; w < wEnd; w++) {
                sum += static_cast<float>(xData[hOffset + w]);
            }
        }
        y[i] = static_cast<X_T>(sum / static_cast<float>(divisorFactor));
    }
}

template <typename X_T, typename TYPE_T, int32_t FORMAT_TYPE>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void AvgPoolNdSimtCompute(
    __gm__ X_T* x, __gm__ X_T* y, __ubuf__ AvgPool::AvgPoolSimtTilingData* SimtTilingData, __ubuf__ TYPE_T* AvgPoolSimtParam, int64_t hInDim, int64_t wInDim, int64_t hOutDim, int64_t wOutDim, 
    int64_t sH, int64_t sW, int64_t tPad, int64_t lPad, int64_t bPad, int64_t rPad, int64_t divisorOverride, int64_t cDim)
{
    TYPE_T magicH = AvgPoolSimtParam[0];
    TYPE_T shiftH = AvgPoolSimtParam[1];
    TYPE_T magicW = AvgPoolSimtParam[2];
    TYPE_T shiftW = AvgPoolSimtParam[3];
    TYPE_T magicC = AvgPoolSimtParam[4];
    TYPE_T shiftC = AvgPoolSimtParam[5];

    TYPE_T divisorFactor = divisorOverride;
    using DIV_T = typename std::conditional<std::is_same<TYPE_T, int32_t>::value, uint32_t, uint64_t>::type;
    TYPE_T outSize = SimtTilingData->nDim * cDim * hOutDim * wOutDim;
    for (DIV_T i = Simt::GetBlockIdx() * Simt::GetThreadNum() + Simt::GetThreadIdx(); i < outSize;
        i += Simt::GetBlockNum() * Simt::GetThreadNum()) {
        DIV_T quotientC = Simt::UintDiv<DIV_T>(i, magicC, shiftC);
        DIV_T quotientW = Simt::UintDiv<DIV_T>(quotientC, magicW, shiftW);
        DIV_T quotientH = Simt::UintDiv<DIV_T>(quotientW, magicH, shiftH);
        TYPE_T pn = quotientH;
        // ph = quotientW - quotientH * hOutDim
        TYPE_T hStart = (quotientW - quotientH * hOutDim) * sH - tPad;
        // pw = quotientC - quotientW * wOutDim
        TYPE_T wStart = (quotientC - quotientW * wOutDim) * sW - lPad;
        TYPE_T hEnd = Simt::Min(hStart + (TYPE_T)SimtTilingData->kH, (TYPE_T)hInDim + (TYPE_T)bPad);
        TYPE_T wEnd = Simt::Min(wStart + (TYPE_T)SimtTilingData->kW, (TYPE_T)wInDim + (TYPE_T)rPad);
        TYPE_T poolSize = (hEnd - hStart) * (wEnd - wStart);
        hStart = Simt::Max(hStart, (TYPE_T)0);
        wStart = Simt::Max(wStart, (TYPE_T)0);
        hEnd = Simt::Min(hEnd, (TYPE_T)hInDim);
        wEnd = Simt::Min(wEnd, (TYPE_T)wInDim);
        if(hStart >= hEnd || wStart >= wEnd) {
            y[i] = 0;
            continue;
        }

        if (!divisorOverride) {
            divisorFactor = SimtTilingData->countIncludePad ? poolSize : ((hEnd - hStart) * (wEnd - wStart));
        }

        TYPE_T pc = i - quotientC * cDim;
        float sum = 0;
        auto xData = x + pn * hInDim * wInDim * cDim;
        for (TYPE_T h = hStart; h < hEnd; h++) {
            TYPE_T hOffset = h * wInDim;
            for (TYPE_T w = wStart; w < wEnd; w++) {
                sum += static_cast<float>(xData[(hOffset + w) * cDim + pc]);
            }
        }
        y[i] = static_cast<X_T>(sum / static_cast<float>(divisorFactor));
    }
}
} // namespace AvgPoolSimt

#endif // CANN_AVG_POOL_WITH_ARGAVG_V3_SIMT_H
 
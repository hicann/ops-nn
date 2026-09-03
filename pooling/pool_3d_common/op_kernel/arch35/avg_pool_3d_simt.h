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
#include "simt_api/asc_simt.h"

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
    __gm__ X_T* x, __gm__ X_T* y, __ubuf__ AvgPool3DSimtTilingData* SimtTilingData,
    __ubuf__ TYPE_T* AvgPool3DSimtParam);

template <typename X_T, typename TYPE_T, int32_t FORMAT_TYPE>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void AvgPool3dNdSimtCompute(
    __gm__ X_T* x, __gm__ X_T* y, __ubuf__ AvgPool3DSimtTilingData* SimtTilingData,
    __ubuf__ TYPE_T* AvgPool3DSimtParam);

template <typename X_T, typename TYPE_T, int32_t FORMAT_TYPE>
class AvgPool3DSimtImpl {
public:
    __aicore__ inline AvgPool3DSimtImpl(TPipe* pipe, const Pool3DSimtTilingData* __restrict tilingData)
        : pipe_(pipe), tilingData_(tilingData), blockIdx_(GetBlockIdx()), blockNum_(GetBlockNum())
    {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y);
    __aicore__ inline void Process();

private:
    TPipe* pipe_;
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
        asc_vf_call<AvgPool3dNcSimtCompute<X_T, TYPE_T, FORMAT_TYPE>>(
            dim3(THREAD_NUM), (__gm__ X_T*)x_.GetPhyAddr(), (__gm__ X_T*)y_.GetPhyAddr(),
            (__ubuf__ AvgPool3DSimtTilingData*)(SimtTilingData.GetPhyAddr()),
            (__ubuf__ TYPE_T*)(AvgPool3DSimtParam.GetPhyAddr()));
    } else if constexpr (FORMAT_TYPE == 1) {
        asc_vf_call<AvgPool3dNdSimtCompute<X_T, TYPE_T, FORMAT_TYPE>>(
            dim3(THREAD_NUM), (__gm__ X_T*)x_.GetPhyAddr(), (__gm__ X_T*)y_.GetPhyAddr(),
            (__ubuf__ AvgPool3DSimtTilingData*)(SimtTilingData.GetPhyAddr()),
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

    // 循环不变量外提
    const TYPE_T dInDim = static_cast<TYPE_T>(SimtTilingData->dInDim);
    const TYPE_T hInDim = static_cast<TYPE_T>(SimtTilingData->hInDim);
    const TYPE_T wInDim = static_cast<TYPE_T>(SimtTilingData->wInDim);
    const TYPE_T dOutDim = static_cast<TYPE_T>(SimtTilingData->dOutDim);
    const TYPE_T hOutDim = static_cast<TYPE_T>(SimtTilingData->hOutDim);
    const TYPE_T wOutDim = static_cast<TYPE_T>(SimtTilingData->wOutDim);
    const TYPE_T kD = static_cast<TYPE_T>(SimtTilingData->kD);
    const TYPE_T kH = static_cast<TYPE_T>(SimtTilingData->kH);
    const TYPE_T kW = static_cast<TYPE_T>(SimtTilingData->kW);
    const TYPE_T sD = static_cast<TYPE_T>(SimtTilingData->sD);
    const TYPE_T sH = static_cast<TYPE_T>(SimtTilingData->sH);
    const TYPE_T sW = static_cast<TYPE_T>(SimtTilingData->sW);
    const TYPE_T fPad = static_cast<TYPE_T>(SimtTilingData->fPad);
    const TYPE_T bkPad = static_cast<TYPE_T>(SimtTilingData->bkPad);
    const TYPE_T tPad = static_cast<TYPE_T>(SimtTilingData->tPad);
    const TYPE_T bPad = static_cast<TYPE_T>(SimtTilingData->bPad);
    const TYPE_T lPad = static_cast<TYPE_T>(SimtTilingData->lPad);
    const TYPE_T rPad = static_cast<TYPE_T>(SimtTilingData->rPad);
    const TYPE_T divisorOverride = static_cast<TYPE_T>(SimtTilingData->divisorOverride);
    const TYPE_T countIncludePad = static_cast<TYPE_T>(SimtTilingData->countIncludePad);
    const TYPE_T outSize = SimtTilingData->nDim * SimtTilingData->cDim * dOutDim * hOutDim * wOutDim;
    const TYPE_T planeSize = hInDim * wInDim;
    const TYPE_T rowSize = wInDim;
    const bool hasOverride = divisorOverride != 0;

    using DIV_T = typename std::conditional<std::is_same<TYPE_T, int32_t>::value, uint32_t, uint64_t>::type;
    for (DIV_T i = blockIdx.x * blockDim.x + threadIdx.x; i < outSize; i += gridDim.x * blockDim.x) {
        DIV_T quotientW = Simt::UintDiv<DIV_T>(i, magicW, shiftW);
        DIV_T quotientH = Simt::UintDiv<DIV_T>(quotientW, magicH, shiftH);
        DIV_T quotientD = Simt::UintDiv<DIV_T>(quotientH, magicD, shiftD);
        TYPE_T pw = i - quotientW * wOutDim;
        TYPE_T ph = quotientW - quotientH * hOutDim;
        TYPE_T pd = quotientH - quotientD * dOutDim;
        TYPE_T pnc = quotientD;
        TYPE_T dStart = pd * sD - fPad;
        TYPE_T hStart = ph * sH - tPad;
        TYPE_T wStart = pw * sW - lPad;
        TYPE_T dEnd = min(dStart + kD, dInDim + bkPad);
        TYPE_T hEnd = min(hStart + kH, hInDim + bPad);
        TYPE_T wEnd = min(wStart + kW, wInDim + rPad);
        TYPE_T poolSize = (dEnd - dStart) * (hEnd - hStart) * (wEnd - wStart);
        dStart = max(dStart, (TYPE_T)0);
        hStart = max(hStart, (TYPE_T)0);
        wStart = max(wStart, (TYPE_T)0);
        dEnd = min(dEnd, dInDim);
        hEnd = min(hEnd, hInDim);
        wEnd = min(wEnd, wInDim);
        if (dStart >= dEnd || hStart >= hEnd || wStart >= wEnd) {
            y[i] = 0;
            continue;
        }

        TYPE_T divisorFactor;
        if (hasOverride) {
            divisorFactor = divisorOverride;
        } else {
            if (countIncludePad != 0) {
                divisorFactor = poolSize;
            } else {
                divisorFactor = (dEnd - dStart) * (hEnd - hStart) * (wEnd - wStart);
            }
        }
        // 行指针递增代替每元素全量寻址
        __gm__ X_T* dPtr = x + pnc * (dInDim * planeSize) + dStart * planeSize + hStart * rowSize;
        float s0 = 0;
        float s1 = 0;
        float s2 = 0;
        float s3 = 0;
        for (TYPE_T d = dStart; d < dEnd; d++) {
            __gm__ X_T* rowPtr = dPtr;
            for (TYPE_T h = hStart; h < hEnd; h++) {
                TYPE_T len = wEnd - wStart;
                TYPE_T len4 = len / 4 * 4;
                for (TYPE_T j = 0; j < len4; j += 4) {
                    s0 += static_cast<float>(rowPtr[wStart + j]);
                    s1 += static_cast<float>(rowPtr[wStart + j + 1]);
                    s2 += static_cast<float>(rowPtr[wStart + j + 2]);
                    s3 += static_cast<float>(rowPtr[wStart + j + 3]);
                }
                for (TYPE_T j = len4; j < len; j++) {
                    s0 += static_cast<float>(rowPtr[wStart + j]);
                }
                rowPtr += rowSize;
            }
            dPtr += planeSize;
        }
        float sum = (s0 + s1) + (s2 + s3);
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

    // 循环不变量外提
    const TYPE_T cDim = static_cast<TYPE_T>(SimtTilingData->cDim);
    const TYPE_T dInDim = static_cast<TYPE_T>(SimtTilingData->dInDim);
    const TYPE_T hInDim = static_cast<TYPE_T>(SimtTilingData->hInDim);
    const TYPE_T wInDim = static_cast<TYPE_T>(SimtTilingData->wInDim);
    const TYPE_T dOutDim = static_cast<TYPE_T>(SimtTilingData->dOutDim);
    const TYPE_T hOutDim = static_cast<TYPE_T>(SimtTilingData->hOutDim);
    const TYPE_T wOutDim = static_cast<TYPE_T>(SimtTilingData->wOutDim);
    const TYPE_T kD = static_cast<TYPE_T>(SimtTilingData->kD);
    const TYPE_T kH = static_cast<TYPE_T>(SimtTilingData->kH);
    const TYPE_T kW = static_cast<TYPE_T>(SimtTilingData->kW);
    const TYPE_T sD = static_cast<TYPE_T>(SimtTilingData->sD);
    const TYPE_T sH = static_cast<TYPE_T>(SimtTilingData->sH);
    const TYPE_T sW = static_cast<TYPE_T>(SimtTilingData->sW);
    const TYPE_T fPad = static_cast<TYPE_T>(SimtTilingData->fPad);
    const TYPE_T bkPad = static_cast<TYPE_T>(SimtTilingData->bkPad);
    const TYPE_T tPad = static_cast<TYPE_T>(SimtTilingData->tPad);
    const TYPE_T bPad = static_cast<TYPE_T>(SimtTilingData->bPad);
    const TYPE_T lPad = static_cast<TYPE_T>(SimtTilingData->lPad);
    const TYPE_T rPad = static_cast<TYPE_T>(SimtTilingData->rPad);
    const TYPE_T divisorOverride = static_cast<TYPE_T>(SimtTilingData->divisorOverride);
    const TYPE_T countIncludePad = static_cast<TYPE_T>(SimtTilingData->countIncludePad);
    const TYPE_T outSize = SimtTilingData->nDim * SimtTilingData->cDim * dOutDim * hOutDim * wOutDim;
    const TYPE_T planeSize = hInDim * wInDim * cDim;
    const TYPE_T rowSize = wInDim * cDim;
    const TYPE_T cStep = cDim;
    const bool hasOverride = divisorOverride != 0;

    using DIV_T = typename std::conditional<std::is_same<TYPE_T, int32_t>::value, uint32_t, uint64_t>::type;
    for (DIV_T i = blockIdx.x * blockDim.x + threadIdx.x; i < outSize; i += gridDim.x * blockDim.x) {
        DIV_T quotientC = Simt::UintDiv<DIV_T>(i, magicC, shiftC);
        DIV_T quotientW = Simt::UintDiv<DIV_T>(quotientC, magicW, shiftW);
        DIV_T quotientH = Simt::UintDiv<DIV_T>(quotientW, magicH, shiftH);
        DIV_T quotientD = Simt::UintDiv<DIV_T>(quotientH, magicD, shiftD);
        TYPE_T pc = i - quotientC * cDim;
        TYPE_T pw = quotientC - quotientW * wOutDim;
        TYPE_T ph = quotientW - quotientH * hOutDim;
        TYPE_T pd = quotientH - quotientD * dOutDim;
        TYPE_T pn = quotientD;
        TYPE_T dStart = pd * sD - fPad;
        TYPE_T hStart = ph * sH - tPad;
        TYPE_T wStart = pw * sW - lPad;
        TYPE_T dEnd = min(dStart + kD, dInDim + bkPad);
        TYPE_T hEnd = min(hStart + kH, hInDim + bPad);
        TYPE_T wEnd = min(wStart + kW, wInDim + rPad);
        TYPE_T poolSize = (dEnd - dStart) * (hEnd - hStart) * (wEnd - wStart);
        dStart = max(dStart, (TYPE_T)0);
        hStart = max(hStart, (TYPE_T)0);
        wStart = max(wStart, (TYPE_T)0);
        dEnd = min(dEnd, dInDim);
        hEnd = min(hEnd, hInDim);
        wEnd = min(wEnd, wInDim);
        if (dStart >= dEnd || hStart >= hEnd || wStart >= wEnd) {
            y[i] = 0;
            continue;
        }

        TYPE_T divisorFactor;
        if (hasOverride) {
            divisorFactor = divisorOverride;
        } else {
            if (countIncludePad != 0) {
                divisorFactor = poolSize;
            } else {
                divisorFactor = (dEnd - dStart) * (hEnd - hStart) * (wEnd - wStart);
            }
        }
        // 指针递增代替每元素全量寻址
        __gm__ X_T* dPtr = x + pn * (dInDim * planeSize) + dStart * planeSize + hStart * rowSize + wStart * cStep + pc;
        float s0 = 0;
        float s1 = 0;
        float s2 = 0;
        float s3 = 0;
        for (TYPE_T d = dStart; d < dEnd; d++) {
            __gm__ X_T* rowPtr = dPtr;
            for (TYPE_T h = hStart; h < hEnd; h++) {
                __gm__ X_T* wPtr = rowPtr;
                TYPE_T len = wEnd - wStart;
                TYPE_T len4 = len / 4 * 4;
                for (TYPE_T j = 0; j < len4; j += 4) {
                    s0 += static_cast<float>(wPtr[0]);
                    s1 += static_cast<float>(wPtr[cStep]);
                    s2 += static_cast<float>(wPtr[2 * cStep]);
                    s3 += static_cast<float>(wPtr[3 * cStep]);
                    wPtr += 4 * cStep;
                }
                for (TYPE_T j = len4; j < len; j++) {
                    s0 += static_cast<float>(*wPtr);
                    wPtr += cStep;
                }
                rowPtr += rowSize;
            }
            dPtr += planeSize;
        }
        float sum = (s0 + s1) + (s2 + s3);
        y[i] = static_cast<X_T>(sum / static_cast<float>(divisorFactor));
    }
}
} // namespace AvgPool3DSimt

#endif // CANN_AVG_POOL_WITH_ARGAVG_V3_SIMT_H

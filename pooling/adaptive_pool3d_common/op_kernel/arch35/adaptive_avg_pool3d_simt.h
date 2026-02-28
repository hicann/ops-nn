/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file adaptive_avg_pool3d_simt.h
 * \brief adaptive_avg_pool3d implied by simt
 */

#ifndef ADAPTIVE_AVG_POOL_3D_SIMT_H
#define ADAPTIVE_AVG_POOL_3D_SIMT_H

#include "kernel_operator.h"
#include "../inc/load_store_utils.h"
#include "../inc/platform.h"
#include "../inc/kernel_utils.h"
#include "adaptive_pool3d_tiling_struct.h"

#ifdef __CCE_KT_TEST__
#define LAUNCH_BOUND(threads)
#endif

namespace AdaptivePool3DWithSimt{
    using namespace AscendC;

    constexpr uint32_t USE_THREAD_NUM = 1024;
    constexpr size_t UINT_DIV_NUM = 8;
    constexpr size_t TILING_DATA_NUM = 8;
    constexpr static uint32_t IDX0 = 0;
    constexpr static uint32_t IDX1 = 1;
    constexpr static uint32_t IDX2 = 2;
    constexpr static uint32_t IDX3 = 3;
    constexpr static uint32_t IDX4 = 4;
    constexpr static uint32_t IDX5 = 5;
    constexpr static uint32_t IDX6 = 6;
    constexpr static uint32_t IDX7 = 7;

template <typename DIV_T>
__aicore__ __attribute__((always_inline)) inline static DIV_T CalStartIdx(DIV_T outIdx, DIV_T magicOutLen, DIV_T shiftOutLen,  DIV_T inLen)
{
    DIV_T pStart = outIdx * inLen;
    return Simt::UintDiv<DIV_T>(pStart, magicOutLen, shiftOutLen);
}

template <typename DIV_T>
__aicore__ __attribute__((always_inline)) inline static DIV_T CalEndIdx(DIV_T outIdx, DIV_T magicOutLen, DIV_T shiftOutLen,  DIV_T inLen, DIV_T outLen)
{
    DIV_T pEnd = ((outIdx + 1) * inLen + outLen - 1);
    return Simt::UintDiv<DIV_T>(pEnd, magicOutLen, shiftOutLen);
}

template <typename X_T, typename DIV_T>
__simt_vf__ __aicore__ LAUNCH_BOUND(USE_THREAD_NUM) inline void AdaptiveAvgPool3DNcSimtCompute(__gm__ X_T* x, __gm__ X_T* y,
    __ubuf__ AdaptivePool3DTiling::AdaptivePool3DSimtTilingData* tilingData, __ubuf__ DIV_T* uintDivData)
{
    DIV_T outputSize = tilingData->nDim * tilingData->cDim * tilingData->dOutDim * tilingData->hOutDim * tilingData->wOutDim;
    for (DIV_T idxOut = Simt::GetBlockIdx() * Simt::GetThreadNum() + Simt::GetThreadIdx(); idxOut < outputSize;
        idxOut += Simt::GetBlockNum() * Simt::GetThreadNum()) {
        DIV_T divW = Simt::UintDiv<DIV_T>(idxOut, uintDivData[IDX0], uintDivData[IDX1]);
        DIV_T divH = Simt::UintDiv<DIV_T>(divW, uintDivData[IDX2], uintDivData[IDX3]);
        DIV_T divD = Simt::UintDiv<DIV_T>(divH, uintDivData[IDX4], uintDivData[IDX5]);
        DIV_T curWOutIdx = idxOut - divW * tilingData->wOutDim;
        DIV_T curHOutIdx = divW - divH * tilingData->hOutDim;
        DIV_T curDOutIdx = divH - divD * tilingData->dOutDim;

        DIV_T startInD = CalStartIdx<DIV_T>(curDOutIdx, uintDivData[IDX4], uintDivData[IDX5], tilingData->dInDim);
        DIV_T endInD = CalEndIdx<DIV_T>(curDOutIdx, uintDivData[IDX4], uintDivData[IDX5], tilingData->dInDim, tilingData->dOutDim);
        DIV_T startInH = CalStartIdx<DIV_T>(curHOutIdx, uintDivData[IDX2], uintDivData[IDX3], tilingData->hInDim);
        DIV_T endInH = CalEndIdx<DIV_T>(curHOutIdx, uintDivData[IDX2], uintDivData[IDX3], tilingData->hInDim, tilingData->hOutDim);
        DIV_T startInW = CalStartIdx<DIV_T>(curWOutIdx, uintDivData[IDX0], uintDivData[IDX1], tilingData->wInDim);
        DIV_T endInW = CalEndIdx<DIV_T>(curWOutIdx, uintDivData[IDX0], uintDivData[IDX1], tilingData->wInDim, tilingData->wOutDim);

        DIV_T windowsSize = (endInD - startInD) * (endInH - startInH) * (endInW - startInW);
        auto curX = x + divD * tilingData->dInDim * tilingData->hInDim * tilingData->wInDim;
        float sum = 0;
        for (DIV_T dIdx = startInD; dIdx < endInD; ++dIdx) {
            for (DIV_T hIdx = startInH; hIdx < endInH; ++hIdx) {
                for (DIV_T wIdx = startInW; wIdx < endInW; ++wIdx) {
                    DIV_T idxOffset = dIdx * tilingData->hInDim * tilingData->wInDim + hIdx * tilingData->wInDim + wIdx;
                    sum += static_cast<float>(curX[idxOffset]);
                }
            }
        }
        y[idxOut] = static_cast<X_T>(sum / static_cast<float>(windowsSize));
    }
}

template <typename X_T, typename DIV_T>
__simt_vf__ __aicore__ LAUNCH_BOUND(USE_THREAD_NUM) inline void AdaptiveAvgPool3DNdSimtCompute(__gm__ X_T* x, __gm__ X_T* y,
    __ubuf__ AdaptivePool3DTiling::AdaptivePool3DSimtTilingData* tilingData, __ubuf__ DIV_T* uintDivData)
{
    DIV_T outputSize = tilingData->nDim * tilingData->cDim * tilingData->dOutDim * tilingData->hOutDim * tilingData->wOutDim;
    for (DIV_T idxOut = Simt::GetBlockIdx() * Simt::GetThreadNum() + Simt::GetThreadIdx(); idxOut < outputSize;
        idxOut += Simt::GetBlockNum() * Simt::GetThreadNum()) {
        DIV_T divC = Simt::UintDiv<DIV_T>(idxOut, uintDivData[IDX6], uintDivData[IDX7]);
        DIV_T divW = Simt::UintDiv<DIV_T>(divC, uintDivData[IDX0], uintDivData[IDX1]);
        DIV_T divH = Simt::UintDiv<DIV_T>(divW, uintDivData[IDX2], uintDivData[IDX3]);
        DIV_T divD = Simt::UintDiv<DIV_T>(divH, uintDivData[IDX4], uintDivData[IDX5]);
        DIV_T curCOutIdx = idxOut - divC * tilingData->cDim;
        DIV_T curWOutIdx = divC - divW * tilingData->wOutDim;
        DIV_T curHOutIdx = divW - divH * tilingData->hOutDim;
        DIV_T curDOutIdx = divH - divD * tilingData->dOutDim;

        DIV_T startInD = CalStartIdx<DIV_T>(curDOutIdx, uintDivData[IDX4], uintDivData[IDX5], tilingData->dInDim);
        DIV_T endInD = CalEndIdx<DIV_T>(curDOutIdx, uintDivData[IDX4], uintDivData[IDX5], tilingData->dInDim, tilingData->dOutDim);
        DIV_T startInH = CalStartIdx<DIV_T>(curHOutIdx, uintDivData[IDX2], uintDivData[IDX3], tilingData->hInDim);
        DIV_T endInH = CalEndIdx<DIV_T>(curHOutIdx, uintDivData[IDX2], uintDivData[IDX3], tilingData->hInDim, tilingData->hOutDim);
        DIV_T startInW = CalStartIdx<DIV_T>(curWOutIdx, uintDivData[IDX0], uintDivData[IDX1], tilingData->wInDim);
        DIV_T endInW = CalEndIdx<DIV_T>(curWOutIdx, uintDivData[IDX0], uintDivData[IDX1], tilingData->wInDim, tilingData->wOutDim);

        DIV_T windowsSize = (endInD - startInD) * (endInH - startInH) * (endInW - startInW);
        auto curX = x + divD * tilingData->dInDim * tilingData->hInDim * tilingData->wInDim * tilingData->cDim;
        float sum = 0;
        for (DIV_T dIdx = startInD; dIdx < endInD; ++dIdx) {
            for (DIV_T hIdx = startInH; hIdx < endInH; ++hIdx) {
                for (DIV_T wIdx = startInW; wIdx < endInW; ++wIdx) {
                    DIV_T idxOffset = dIdx * tilingData->hInDim * tilingData->wInDim + hIdx * tilingData->wInDim + wIdx;
                    sum += static_cast<float>(curX[idxOffset * tilingData->cDim + curCOutIdx]);
                }
            }
        }
        y[idxOut] = static_cast<X_T>(sum / static_cast<float>(windowsSize));
    }
}

template <typename X_T, typename DIV_T, uint64_t FORMAT_TYPE>
class AdaptiveAvgPool3DSimt
{
public:
    __aicore__ inline AdaptiveAvgPool3DSimt(TPipe *pipe, const AdaptivePool3DTiling::AdaptivePool3DSimtTilingData* __restrict tilingData) 
        : pipe_(pipe), tilingData_(tilingData) {}
    
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y);
    __aicore__ inline void Process();

private:
    TPipe *pipe_;
    AscendC::GlobalTensor<X_T> x_;
    AscendC::GlobalTensor<X_T> y_;

    const AdaptivePool3DTiling::AdaptivePool3DSimtTilingData* tilingData_;
    TBuf<TPosition::VECCALC> uintDivBuf_;
    TBuf<TPosition::VECCALC> tilingDataBuf_;
};

template <typename X_T, typename DIV_T, uint64_t FORMAT_TYPE>
__aicore__ inline void AdaptiveAvgPool3DSimt<X_T, DIV_T, FORMAT_TYPE>::Init(GM_ADDR x, GM_ADDR y)
{
    x_.SetGlobalBuffer((__gm__ X_T*)(x));
    y_.SetGlobalBuffer((__gm__ X_T*)(y));

    pipe_->InitBuffer(uintDivBuf_, UINT_DIV_NUM * sizeof(DIV_T));
    pipe_->InitBuffer(tilingDataBuf_, TILING_DATA_NUM * sizeof(int64_t));
}

template <typename X_T, typename DIV_T, uint64_t FORMAT_TYPE>
__aicore__ inline void AdaptiveAvgPool3DSimt<X_T, DIV_T, FORMAT_TYPE>::Process()
{
    LocalTensor<int64_t> tilingDataLocal = tilingDataBuf_.Get<int64_t>();
    LocalTensor<DIV_T> uintDivLocal = uintDivBuf_.Get<DIV_T>();
    const int64_t* tiling = reinterpret_cast<const int64_t*>(tilingData_);
    for (uint32_t i = 0; i < TILING_DATA_NUM; i++) {
        tilingDataLocal.SetValue(i, tiling[i]);
    }

    DIV_T magicD = 0;
    DIV_T shiftD = 0;
    DIV_T magicH = 0;
    DIV_T shiftH = 0;
    DIV_T magicW = 0;
    DIV_T shiftW = 0;
    DIV_T magicC = 0;
    DIV_T shiftC = 0;
    GetUintDivMagicAndShift<DIV_T>(magicD, shiftD, tilingDataLocal(IDX5));
    GetUintDivMagicAndShift<DIV_T>(magicH, shiftH, tilingDataLocal(IDX6));
    GetUintDivMagicAndShift<DIV_T>(magicW, shiftW, tilingDataLocal(IDX7));
    uintDivLocal.SetValue(IDX0, static_cast<DIV_T>(magicW));
    uintDivLocal.SetValue(IDX1, static_cast<DIV_T>(shiftW));
    uintDivLocal.SetValue(IDX2, static_cast<DIV_T>(magicH));
    uintDivLocal.SetValue(IDX3, static_cast<DIV_T>(shiftH));
    uintDivLocal.SetValue(IDX4, static_cast<DIV_T>(magicD));
    uintDivLocal.SetValue(IDX5, static_cast<DIV_T>(shiftD));

    DataSyncBarrier<MemDsbT::UB>();
    if constexpr (FORMAT_TYPE == 0) {
        GetUintDivMagicAndShift<DIV_T>(magicC, shiftC, tilingDataLocal(IDX1));
        uintDivLocal.SetValue(IDX6, static_cast<DIV_T>(magicC));
        uintDivLocal.SetValue(IDX7, static_cast<DIV_T>(shiftC));
        Simt::VF_CALL<AdaptiveAvgPool3DNdSimtCompute<X_T, DIV_T>>(Simt::Dim3(USE_THREAD_NUM), 
            (__gm__ X_T*)x_.GetPhyAddr(), (__gm__ X_T*)y_.GetPhyAddr(), 
            (__ubuf__ AdaptivePool3DTiling::AdaptivePool3DSimtTilingData*)(tilingDataLocal.GetPhyAddr()),
            (__ubuf__ DIV_T*)(uintDivLocal.GetPhyAddr()));
    } else if constexpr (FORMAT_TYPE == 1) {
        Simt::VF_CALL<AdaptiveAvgPool3DNcSimtCompute<X_T, DIV_T>>(Simt::Dim3(USE_THREAD_NUM), 
            (__gm__ X_T*)x_.GetPhyAddr(), (__gm__ X_T*)y_.GetPhyAddr(), 
            (__ubuf__ AdaptivePool3DTiling::AdaptivePool3DSimtTilingData*)(tilingDataLocal.GetPhyAddr()),
            (__ubuf__ DIV_T*)(uintDivLocal.GetPhyAddr()));
    }
}
} // namespace AdaptivePool3DWithSimt
#endif //ADAPTIVE_AVG_POOL_3D_SIMT_H
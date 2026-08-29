/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file avg_pool3_d_grad_simt.h
 * \brief 3D average pooling backward SIMT kernel (arch35).
 *        Modeled on avg_pool_v2_grad_simt.h, extended to D/H/W.
 */

#ifndef CANN_AVG_POOL3_D_GRAD_SIMT_H
#define CANN_AVG_POOL3_D_GRAD_SIMT_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "avg_pool3_d_grad_tiling_data.h"
#include "simt_api/asc_simt.h"

#ifdef __CCE_KT_TEST__
#define LAUNCH_BOUND(threads)
#endif

using namespace AscendC;

namespace AvgPool3DGrad {
constexpr size_t PARAM_NUM = 8; // 8 * 4 = 32B align
constexpr size_t TILING_DATA_NUM = 22;

constexpr size_t MAGIC_W_IDX = 0;
constexpr size_t SHIFT_W_IDX = 1;
constexpr size_t MAGIC_H_IDX = 2;
constexpr size_t SHIFT_H_IDX = 3;
constexpr size_t MAGIC_D_IDX = 4;
constexpr size_t SHIFT_D_IDX = 5;
constexpr size_t MAGIC_N_IDX = 6;
constexpr size_t SHIFT_N_IDX = 7;

constexpr size_t KERNEL_D_IDX = 0;
constexpr size_t KERNEL_H_IDX = 1;
constexpr size_t KERNEL_W_IDX = 2;
constexpr size_t STRIDE_D_IDX = 3;
constexpr size_t STRIDE_H_IDX = 4;
constexpr size_t STRIDE_W_IDX = 5;
constexpr size_t PAD_DL_IDX = 6;
constexpr size_t PAD_DR_IDX = 7;
constexpr size_t PAD_HL_IDX = 8;
constexpr size_t PAD_HR_IDX = 9;
constexpr size_t PAD_WL_IDX = 10;
constexpr size_t PAD_WR_IDX = 11;
constexpr size_t DIV_IDX = 12;

constexpr uint32_t FORMAT_NCDHW_TYPE = 0;
constexpr uint32_t FORMAT_NDHWC_TYPE = 1;

constexpr uint32_t THREAD_DIM = 1024;

template <typename VALUE_T, typename IDX_T, uint32_t FORMAT_T, uint32_t COUNTPAD_T, uint32_t DIV_T>
class AvgPool3DGradSimt {
public:
    __aicore__ inline AvgPool3DGradSimt(TPipe* pipe, const AvgPool3DGradSimtTilingData* __restrict tilingData)
        : pipe_(pipe), tilingData_(tilingData)
    {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y);
    __aicore__ inline void Process();
    __aicore__ inline void Compute();

private:
    TPipe* pipe_;
    AscendC::GlobalTensor<VALUE_T> x_;
    AscendC::GlobalTensor<VALUE_T> y_;
    TBuf<TPosition::VECCALC> paramBuf_;
    TBuf<TPosition::VECCALC> tilingDataBuf_;
    const AvgPool3DGradSimtTilingData* tilingData_;
};

template <typename VALUE_T, typename IDX_T, uint32_t FORMAT_T, uint32_t COUNTPAD_T, uint32_t DIV_T>
__aicore__ inline void AvgPool3DGradSimt<VALUE_T, IDX_T, FORMAT_T, COUNTPAD_T, DIV_T>::Init(GM_ADDR x, GM_ADDR y)
{
    x_.SetGlobalBuffer((__gm__ VALUE_T*)(x));
    y_.SetGlobalBuffer((__gm__ VALUE_T*)(y));
    pipe_->InitBuffer(paramBuf_, PARAM_NUM * sizeof(IDX_T));
    pipe_->InitBuffer(tilingDataBuf_, TILING_DATA_NUM * sizeof(int32_t));
}

template <typename VALUE_T, typename IDX_T, uint32_t FORMAT_T, uint32_t COUNTPAD_T, uint32_t DIV_T>
__aicore__ inline void AvgPool3DGradSimt<VALUE_T, IDX_T, FORMAT_T, COUNTPAD_T, DIV_T>::Process()
{
    Compute();
}

template <typename VALUE_T, typename IDX_T, typename ACC_VALUE_T, uint32_t FORMAT_T, uint32_t COUNTPAD_T,
          uint32_t DIV_T>
__simt_callee__ __aicore__ inline static void CycleUpdateGradValue(
    IDX_T channels, IDX_T depth, IDX_T height, IDX_T width, int32_t pooledDepth, int32_t pooledHeight,
    int32_t pooledWidth, IDX_T pdStart, IDX_T pdEnd, IDX_T phStart, IDX_T phEnd, IDX_T pwStart, IDX_T pwEnd,
    int32_t strideD, int32_t strideH, int32_t strideW, int32_t padDL, int32_t padDR, int32_t padHL, int32_t padHR,
    int32_t padWL, int32_t padWR, int32_t kernelD, int32_t kernelH, int32_t kernelW, int32_t divisorOverride,
    const __gm__ VALUE_T* xDataSlice, ACC_VALUE_T* gradient)
{
    for (IDX_T i = pdStart; i < pdEnd; ++i) {
        for (IDX_T j = phStart; j < phEnd; ++j) {
            for (IDX_T k = pwStart; k < pwEnd; ++k) {
                IDX_T dStart = i * strideD - padDL;
                IDX_T hStart = j * strideH - padHL;
                IDX_T wStart = k * strideW - padWL;
                IDX_T dEnd = min(dStart + kernelD, depth + padDR);
                IDX_T hEnd = min(hStart + kernelH, height + padHR);
                IDX_T wEnd = min(wStart + kernelW, width + padWR);
                IDX_T poolSize = (dEnd - dStart) * (hEnd - hStart) * (wEnd - wStart);
                dStart = max(dStart, static_cast<IDX_T>(0));
                hStart = max(hStart, static_cast<IDX_T>(0));
                wStart = max(wStart, static_cast<IDX_T>(0));
                dEnd = min(dEnd, depth);
                hEnd = min(hEnd, height);
                wEnd = min(wEnd, width);

                if (dStart >= dEnd || hStart >= hEnd || wStart >= wEnd) {
                    continue;
                }

                int32_t divideFactor;
                if constexpr (DIV_T != 0) {
                    divideFactor = divisorOverride;
                } else {
                    if constexpr (COUNTPAD_T != 0) {
                        divideFactor = poolSize;
                    } else {
                        divideFactor = (dEnd - dStart) * (hEnd - hStart) * (wEnd - wStart);
                    }
                }
                IDX_T pwIdx = (i * pooledHeight + j) * pooledWidth + k;
                if constexpr (FORMAT_T == FORMAT_NCDHW_TYPE) {
                    *gradient += static_cast<ACC_VALUE_T>(xDataSlice[pwIdx]) / static_cast<ACC_VALUE_T>(divideFactor);
                } else {
                    *gradient += static_cast<ACC_VALUE_T>(xDataSlice[pwIdx * channels]) /
                                 static_cast<ACC_VALUE_T>(divideFactor);
                }
            }
        }
    }
}

template <typename VALUE_T, typename IDX_T, typename UIDX_T, typename ACC_VALUE_T, uint32_t COUNTPAD_T, uint32_t DIV_T>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_DIM) inline void AvgPool3DGradSimtNcdhwKernel(
    const int64_t count, __ubuf__ UIDX_T* simtParam, __ubuf__ int32_t* tilingDataParam, const __gm__ VALUE_T* xData,
    const IDX_T channels, const IDX_T depth, const IDX_T height, const IDX_T width, const IDX_T pooledDepth,
    const IDX_T pooledHeight, const IDX_T pooledWidth, __gm__ VALUE_T* yData)
{
    // NCDHW layout: x[nc, d, h, w]; magicW=width, magicH=height, magicD=depth, magicN=merged nc.
    const auto& magicW = simtParam[MAGIC_W_IDX];
    const auto& shiftW = simtParam[SHIFT_W_IDX];
    const auto& magicH = simtParam[MAGIC_H_IDX];
    const auto& shiftH = simtParam[SHIFT_H_IDX];
    const auto& magicD = simtParam[MAGIC_D_IDX];
    const auto& shiftD = simtParam[SHIFT_D_IDX];
    const auto& magicN = simtParam[MAGIC_N_IDX];
    const auto& shiftN = simtParam[SHIFT_N_IDX];

    const auto& kernelD = tilingDataParam[KERNEL_D_IDX];
    const auto& kernelH = tilingDataParam[KERNEL_H_IDX];
    const auto& kernelW = tilingDataParam[KERNEL_W_IDX];
    const auto& strideD = tilingDataParam[STRIDE_D_IDX];
    const auto& strideH = tilingDataParam[STRIDE_H_IDX];
    const auto& strideW = tilingDataParam[STRIDE_W_IDX];
    const auto& padDL = tilingDataParam[PAD_DL_IDX];
    const auto& padDR = tilingDataParam[PAD_DR_IDX];
    const auto& padHL = tilingDataParam[PAD_HL_IDX];
    const auto& padHR = tilingDataParam[PAD_HR_IDX];
    const auto& padWL = tilingDataParam[PAD_WL_IDX];
    const auto& padWR = tilingDataParam[PAD_WR_IDX];
    const auto& divisorOverride = tilingDataParam[DIV_IDX];

    for (IDX_T index = blockIdx.x * blockDim.x + threadIdx.x; index < count; index = index + gridDim.x * blockDim.x) {
        UIDX_T dim0Idx = Simt::UintDiv(static_cast<UIDX_T>(index), magicW, shiftW);
        IDX_T w = index - dim0Idx * static_cast<UIDX_T>(width);
        UIDX_T dim1Idx = Simt::UintDiv(dim0Idx, magicH, shiftH);
        IDX_T h = dim0Idx - dim1Idx * static_cast<UIDX_T>(height);
        UIDX_T dim2Idx = Simt::UintDiv(dim1Idx, magicD, shiftD);
        IDX_T d = dim1Idx - dim2Idx * static_cast<UIDX_T>(depth);
        IDX_T nc = Simt::UintDiv(dim2Idx, magicN, shiftN);

        d += padDL;
        h += padHL;
        w += padWL;

        IDX_T pdStart = (d < kernelD) ? 0 : (d - kernelD) / strideD + 1;
        IDX_T pdEnd = min(d / strideD + 1, static_cast<IDX_T>(pooledDepth));
        IDX_T phStart = (h < kernelH) ? 0 : (h - kernelH) / strideH + 1;
        IDX_T phEnd = min(h / strideH + 1, static_cast<IDX_T>(pooledHeight));
        IDX_T pwStart = (w < kernelW) ? 0 : (w - kernelW) / strideW + 1;
        IDX_T pwEnd = min(w / strideW + 1, static_cast<IDX_T>(pooledWidth));

        ACC_VALUE_T gradient = 0;
        const __gm__ VALUE_T* xDataSlice = xData + nc * pooledDepth * pooledHeight * pooledWidth;
        CycleUpdateGradValue<VALUE_T, IDX_T, ACC_VALUE_T, FORMAT_NCDHW_TYPE, COUNTPAD_T, DIV_T>(
            channels, depth, height, width, pooledDepth, pooledHeight, pooledWidth, pdStart, pdEnd, phStart, phEnd,
            pwStart, pwEnd, strideD, strideH, strideW, padDL, padDR, padHL, padHR, padWL, padWR, kernelD, kernelH,
            kernelW, divisorOverride, xDataSlice, &gradient);
        yData[index] = static_cast<VALUE_T>(gradient);
    }
}

template <typename VALUE_T, typename IDX_T, typename UIDX_T, typename ACC_VALUE_T, uint32_t COUNTPAD_T, uint32_t DIV_T>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_DIM) inline void AvgPool3DGradSimtNdhwcKernel(
    const int64_t count, __ubuf__ UIDX_T* simtParam, __ubuf__ int32_t* tilingDataParam, const __gm__ VALUE_T* xData,
    const IDX_T channels, const IDX_T depth, const IDX_T height, const IDX_T width, const IDX_T pooledDepth,
    const IDX_T pooledHeight, const IDX_T pooledWidth, __gm__ VALUE_T* yData)
{
    // NDHWC layout: x[n, d, h, w, c]; magicW=channels, magicH=width, magicD=height, magicN=depth.
    const auto& magicW = simtParam[MAGIC_W_IDX];
    const auto& shiftW = simtParam[SHIFT_W_IDX];
    const auto& magicH = simtParam[MAGIC_H_IDX];
    const auto& shiftH = simtParam[SHIFT_H_IDX];
    const auto& magicD = simtParam[MAGIC_D_IDX];
    const auto& shiftD = simtParam[SHIFT_D_IDX];
    const auto& magicN = simtParam[MAGIC_N_IDX];
    const auto& shiftN = simtParam[SHIFT_N_IDX];

    const auto& kernelD = tilingDataParam[KERNEL_D_IDX];
    const auto& kernelH = tilingDataParam[KERNEL_H_IDX];
    const auto& kernelW = tilingDataParam[KERNEL_W_IDX];
    const auto& strideD = tilingDataParam[STRIDE_D_IDX];
    const auto& strideH = tilingDataParam[STRIDE_H_IDX];
    const auto& strideW = tilingDataParam[STRIDE_W_IDX];
    const auto& padDL = tilingDataParam[PAD_DL_IDX];
    const auto& padDR = tilingDataParam[PAD_DR_IDX];
    const auto& padHL = tilingDataParam[PAD_HL_IDX];
    const auto& padHR = tilingDataParam[PAD_HR_IDX];
    const auto& padWL = tilingDataParam[PAD_WL_IDX];
    const auto& padWR = tilingDataParam[PAD_WR_IDX];
    const auto& divisorOverride = tilingDataParam[DIV_IDX];

    for (IDX_T index = blockIdx.x * blockDim.x + threadIdx.x; index < count; index = index + gridDim.x * blockDim.x) {
        UIDX_T dim0Idx = Simt::UintDiv(static_cast<UIDX_T>(index), magicW, shiftW);
        IDX_T c = index - dim0Idx * static_cast<UIDX_T>(channels);
        UIDX_T dim1Idx = Simt::UintDiv(dim0Idx, magicH, shiftH);
        IDX_T w = dim0Idx - dim1Idx * static_cast<UIDX_T>(width);
        UIDX_T dim2Idx = Simt::UintDiv(dim1Idx, magicD, shiftD);
        IDX_T h = dim1Idx - dim2Idx * static_cast<UIDX_T>(height);
        IDX_T n = Simt::UintDiv(dim2Idx, magicN, shiftN);
        IDX_T d = dim2Idx - n * static_cast<UIDX_T>(depth);

        d += padDL;
        h += padHL;
        w += padWL;

        IDX_T pdStart = (d < kernelD) ? 0 : (d - kernelD) / strideD + 1;
        IDX_T pdEnd = min(d / strideD + 1, static_cast<IDX_T>(pooledDepth));
        IDX_T phStart = (h < kernelH) ? 0 : (h - kernelH) / strideH + 1;
        IDX_T phEnd = min(h / strideH + 1, static_cast<IDX_T>(pooledHeight));
        IDX_T pwStart = (w < kernelW) ? 0 : (w - kernelW) / strideW + 1;
        IDX_T pwEnd = min(w / strideW + 1, static_cast<IDX_T>(pooledWidth));

        ACC_VALUE_T gradient = 0;
        const __gm__ VALUE_T* xDataSlice = xData + n * pooledDepth * pooledHeight * pooledWidth * channels + c;
        CycleUpdateGradValue<VALUE_T, IDX_T, ACC_VALUE_T, FORMAT_NDHWC_TYPE, COUNTPAD_T, DIV_T>(
            channels, depth, height, width, pooledDepth, pooledHeight, pooledWidth, pdStart, pdEnd, phStart, phEnd,
            pwStart, pwEnd, strideD, strideH, strideW, padDL, padDR, padHL, padHR, padWL, padWR, kernelD, kernelH,
            kernelW, divisorOverride, xDataSlice, &gradient);
        yData[index] = static_cast<VALUE_T>(gradient);
    }
}

template <typename VALUE_T, typename IDX_T, uint32_t FORMAT_T, uint32_t COUNTPAD_T, uint32_t DIV_T>
__aicore__ inline void AvgPool3DGradSimt<VALUE_T, IDX_T, FORMAT_T, COUNTPAD_T, DIV_T>::Compute()
{
    auto xAddr = (__gm__ VALUE_T*)x_.GetPhyAddr();
    auto yAddr = (__gm__ VALUE_T*)y_.GetPhyAddr();

    using UIDX_T = std::conditional_t<std::is_same_v<IDX_T, int32_t>, uint32_t, uint64_t>;
    int64_t count = tilingData_->nDim * tilingData_->cDim * tilingData_->dInDim * tilingData_->hInDim *
                    tilingData_->wInDim;

    UIDX_T magicW = 0;
    UIDX_T shiftW = 0;
    UIDX_T magicH = 0;
    UIDX_T shiftH = 0;
    UIDX_T magicD = 0;
    UIDX_T shiftD = 0;
    UIDX_T magicN = 0;
    UIDX_T shiftN = 0;
    if constexpr (FORMAT_T == FORMAT_NCDHW_TYPE) {
        // x[nc, d, h, w]: decode order w -> h -> d -> nc.
        // nc already merges N*C (nbatch = N*C, cDim = 1); dim2Idx = index/(D*H*W) is nc,
        // so no further division on the merged nc dimension is needed.
        GetUintDivMagicAndShift<UIDX_T>(magicW, shiftW, tilingData_->wInDim);
        GetUintDivMagicAndShift<UIDX_T>(magicH, shiftH, tilingData_->hInDim);
        GetUintDivMagicAndShift<UIDX_T>(magicD, shiftD, tilingData_->dInDim);
        GetUintDivMagicAndShift<UIDX_T>(magicN, shiftN, 1);
    } else {
        // x[n, d, h, w, c]: decode order c -> w -> h -> d -> n.
        GetUintDivMagicAndShift<UIDX_T>(magicW, shiftW, tilingData_->cDim);
        GetUintDivMagicAndShift<UIDX_T>(magicH, shiftH, tilingData_->wInDim);
        GetUintDivMagicAndShift<UIDX_T>(magicD, shiftD, tilingData_->hInDim);
        GetUintDivMagicAndShift<UIDX_T>(magicN, shiftN, tilingData_->dInDim);
    }
    LocalTensor<UIDX_T> simtParam = paramBuf_.Get<UIDX_T>();
    simtParam.SetValue(MAGIC_W_IDX, static_cast<UIDX_T>(magicW));
    simtParam.SetValue(SHIFT_W_IDX, static_cast<UIDX_T>(shiftW));
    simtParam.SetValue(MAGIC_H_IDX, static_cast<UIDX_T>(magicH));
    simtParam.SetValue(SHIFT_H_IDX, static_cast<UIDX_T>(shiftH));
    simtParam.SetValue(MAGIC_D_IDX, static_cast<UIDX_T>(magicD));
    simtParam.SetValue(SHIFT_D_IDX, static_cast<UIDX_T>(shiftD));
    simtParam.SetValue(MAGIC_N_IDX, static_cast<UIDX_T>(magicN));
    simtParam.SetValue(SHIFT_N_IDX, static_cast<UIDX_T>(shiftN));

    LocalTensor<int32_t> tilingDataParam = tilingDataBuf_.Get<int32_t>();
    tilingDataParam.SetValue(KERNEL_D_IDX, static_cast<int32_t>(tilingData_->kSizeD));
    tilingDataParam.SetValue(KERNEL_H_IDX, static_cast<int32_t>(tilingData_->kSizeH));
    tilingDataParam.SetValue(KERNEL_W_IDX, static_cast<int32_t>(tilingData_->kSizeW));
    tilingDataParam.SetValue(STRIDE_D_IDX, static_cast<int32_t>(tilingData_->stridesD));
    tilingDataParam.SetValue(STRIDE_H_IDX, static_cast<int32_t>(tilingData_->stridesH));
    tilingDataParam.SetValue(STRIDE_W_IDX, static_cast<int32_t>(tilingData_->stridesW));
    tilingDataParam.SetValue(PAD_DL_IDX, static_cast<int32_t>(tilingData_->padDLeft));
    tilingDataParam.SetValue(PAD_DR_IDX, static_cast<int32_t>(tilingData_->padDRight));
    tilingDataParam.SetValue(PAD_HL_IDX, static_cast<int32_t>(tilingData_->padHLeft));
    tilingDataParam.SetValue(PAD_HR_IDX, static_cast<int32_t>(tilingData_->padHRight));
    tilingDataParam.SetValue(PAD_WL_IDX, static_cast<int32_t>(tilingData_->padWLeft));
    tilingDataParam.SetValue(PAD_WR_IDX, static_cast<int32_t>(tilingData_->padWRight));
    tilingDataParam.SetValue(DIV_IDX, static_cast<int32_t>(tilingData_->divisorOverride));
    DataSyncBarrier<MemDsbT::UB>();

    if constexpr (FORMAT_T == FORMAT_NCDHW_TYPE) {
        asc_vf_call<AvgPool3DGradSimtNcdhwKernel<VALUE_T, IDX_T, UIDX_T, float, COUNTPAD_T, DIV_T>>(
            dim3(THREAD_DIM), count, (__ubuf__ UIDX_T*)simtParam.GetPhyAddr(),
            (__ubuf__ int32_t*)tilingDataParam.GetPhyAddr(), xAddr, tilingData_->cDim, tilingData_->dInDim,
            tilingData_->hInDim, tilingData_->wInDim, tilingData_->dOutDim, tilingData_->hOutDim, tilingData_->wOutDim,
            yAddr);
    } else {
        asc_vf_call<AvgPool3DGradSimtNdhwcKernel<VALUE_T, IDX_T, UIDX_T, float, COUNTPAD_T, DIV_T>>(
            dim3(THREAD_DIM), count, (__ubuf__ UIDX_T*)simtParam.GetPhyAddr(),
            (__ubuf__ int32_t*)tilingDataParam.GetPhyAddr(), xAddr, tilingData_->cDim, tilingData_->dInDim,
            tilingData_->hInDim, tilingData_->wInDim, tilingData_->dOutDim, tilingData_->hOutDim, tilingData_->wOutDim,
            yAddr);
    }
}

} // namespace AvgPool3DGrad
#endif // CANN_AVG_POOL3_D_GRAD_SIMT_H

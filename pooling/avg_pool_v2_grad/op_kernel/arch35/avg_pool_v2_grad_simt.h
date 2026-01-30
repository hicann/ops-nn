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
 * \file avg_pool_v2_grad_simt.h
 * \brief avg_pool_v2_grad_simt implied by simt
 */

#ifndef CANN_AVG_POOL_V2_GRAD_SIMT_H
#define CANN_AVG_POOL_V2_GRAD_SIMT_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "avg_pool_v2_grad_tiling_data.h"

#ifdef __CCE_KT_TEST__
#define LAUNCH_BOUND(threads)
#endif

using namespace AscendC;

namespace AvgPoolV2GradSimtNamespace {
using namespace AvgPoolV2Grad;

constexpr size_t PARAM_NUM = 8; // 8 * 4 = 32B align
constexpr size_t TILING_DATA_NUM = 16; // 16 * 4 = 64B align

constexpr size_t MAGIC_W_IDX = 0;
constexpr size_t SHIFT_W_IDX = 1;
constexpr size_t MAGIC_H_IDX = 2;
constexpr size_t SHIFT_H_IDX = 3;
constexpr size_t MAGIC_C_IDX = 4;
constexpr size_t SHIFT_C_IDX = 5;

constexpr size_t KERNEL_H_IDX = 0;
constexpr size_t KERNEL_W_IDX = 1;
constexpr size_t STRIDE_H_IDX = 2;
constexpr size_t STRIDE_W_IDX = 3;
constexpr size_t PAD_HL_IDX = 4;
constexpr size_t PAD_HR_IDX = 5;
constexpr size_t PAD_WL_IDX = 6;
constexpr size_t PAD_WR_IDX = 7;
constexpr size_t DIV_IDX = 8;

constexpr uint32_t FORMAT_NCHW_TYPE = 0;
constexpr uint32_t FORMAT_NHWC_TYPE = 1;

constexpr uint32_t THREAD_DIM = 1024;

template <typename VALUE_T, typename IDX_T, uint32_t FORMAT_T, uint32_t COUNTPAD_T, uint32_t DIV_T>
class AvgPoolV2GradSimt {
public:
    __aicore__ inline AvgPoolV2GradSimt(TPipe* pipe, const AvgPoolV2GradSimtTilingData* __restrict tilingData)
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
    const AvgPoolV2GradSimtTilingData* tilingData_;
};

template <typename VALUE_T, typename IDX_T, uint32_t FORMAT_T, uint32_t COUNTPAD_T, uint32_t DIV_T>
__aicore__ inline void AvgPoolV2GradSimt<VALUE_T, IDX_T, FORMAT_T, COUNTPAD_T, DIV_T>::Init(
    GM_ADDR x, GM_ADDR y)
{
    x_.SetGlobalBuffer((__gm__ VALUE_T*)(x));
    y_.SetGlobalBuffer((__gm__ VALUE_T*)(y));

    pipe_->InitBuffer(paramBuf_, PARAM_NUM * sizeof(IDX_T));
    pipe_->InitBuffer(tilingDataBuf_, TILING_DATA_NUM * sizeof(int32_t));
}

template <typename VALUE_T, typename IDX_T, uint32_t FORMAT_T, uint32_t COUNTPAD_T, uint32_t DIV_T>
__aicore__ inline void AvgPoolV2GradSimt<VALUE_T, IDX_T, FORMAT_T, COUNTPAD_T, DIV_T>::Process()
{
    Compute();
}

template <typename VALUE_T, typename IDX_T, typename ACC_VALUE_T, uint32_t FORMAT_T, uint32_t COUNTPAD_T, uint32_t DIV_T>
__aicore__ inline static void CycleUpdateGradValue(
    IDX_T channels, IDX_T height, IDX_T width, int32_t pooledWidth,
    IDX_T phStart, IDX_T phEnd, IDX_T pwStart, IDX_T pwEnd,
    int32_t strideH, int32_t strideW, int32_t padHL, int32_t padWL, int32_t padHR, int32_t padWR,
    int32_t kernelH, int32_t kernelW, int32_t divisorOverride, const __gm__ VALUE_T* xDataSlice,
    ACC_VALUE_T* gradient)
{
    for (IDX_T i = phStart; i < phEnd ; ++i) {
        for (IDX_T j = pwStart; j < pwEnd ; ++j) {
            IDX_T hStart = i * strideH - padHL;
            IDX_T wStart = j * strideW - padWL;
            IDX_T hEnd = Simt::Min(hStart + kernelH, height + padHR);
            IDX_T wEnd = Simt::Min(wStart + kernelW, width + padWR);
            IDX_T poolSize = (hEnd - hStart) * (wEnd - wStart);
            hStart = Simt::Max(hStart, static_cast<IDX_T>(0));
            wStart = Simt::Max(wStart, static_cast<IDX_T>(0));
            hEnd = Simt::Min(hEnd, height);
            wEnd = Simt::Min(wEnd, width);

            if (hStart >= hEnd || wStart >= wEnd) {
                continue;
            }

            int32_t divideFactor;
            if constexpr (DIV_T != 0) {
                divideFactor = divisorOverride;
            } else {
                if constexpr (COUNTPAD_T != 0) {
                    divideFactor = poolSize;
                } else {
                    divideFactor = (hEnd - hStart) * (wEnd - wStart);
                }
            }
            if constexpr (FORMAT_T == FORMAT_NCHW_TYPE) {
                *gradient += static_cast<ACC_VALUE_T>(xDataSlice[i * pooledWidth + j]) / static_cast<ACC_VALUE_T>(divideFactor);
            } else {
                *gradient += static_cast<ACC_VALUE_T>(xDataSlice[(i * pooledWidth + j) * channels]) / static_cast<ACC_VALUE_T>(divideFactor);
            }
        }
    }
}

template <typename VALUE_T, typename IDX_T, typename UIDX_T, typename ACC_VALUE_T,
          uint32_t COUNTPAD_T, uint32_t DIV_T>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_DIM) inline void AvgPoolV2GradSimtNchwKernel(
    const int64_t count, __ubuf__ UIDX_T* simtParam,
    __ubuf__ int32_t* tilingDataParam, const __gm__ VALUE_T* xData,
    const IDX_T channels, const IDX_T height, const IDX_T width,
    const IDX_T pooledHeight, const IDX_T pooledWidth, __gm__ VALUE_T* yData)
{
    const auto& magicW = simtParam[MAGIC_W_IDX];
    const auto& shiftW = simtParam[SHIFT_W_IDX];
    const auto& magicH = simtParam[MAGIC_H_IDX];
    const auto& shiftH = simtParam[SHIFT_H_IDX];
    const auto& magicC = simtParam[MAGIC_C_IDX];
    const auto& shiftC = simtParam[SHIFT_C_IDX];

    const auto& kernelH = tilingDataParam[KERNEL_H_IDX];
    const auto& kernelW = tilingDataParam[KERNEL_W_IDX];
    const auto& strideH = tilingDataParam[STRIDE_H_IDX];
    const auto& strideW = tilingDataParam[STRIDE_W_IDX];
    const auto& padHL = tilingDataParam[PAD_HL_IDX];
    const auto& padHR = tilingDataParam[PAD_HR_IDX];
    const auto& padWL = tilingDataParam[PAD_WL_IDX];
    const auto& padWR = tilingDataParam[PAD_WR_IDX];
    const auto& divisorOverride = tilingDataParam[DIV_IDX];

    for (IDX_T index = Simt::GetBlockIdx() * Simt::GetThreadNum() + Simt::GetThreadIdx(); index < count;
         index = index + Simt::GetBlockNum() * Simt::GetThreadNum()) {
        UIDX_T dim0Idx = Simt::UintDiv(static_cast<UIDX_T>(index), magicW, shiftW);
        IDX_T w = index - dim0Idx * static_cast<UIDX_T>(width);
        UIDX_T dim1Idx = Simt::UintDiv(dim0Idx, magicH, shiftH);
        IDX_T h = dim0Idx - dim1Idx * static_cast<UIDX_T>(height);
        IDX_T n = Simt::UintDiv(dim1Idx, magicC, shiftC);

        h += padHL;
        w += padWL;

        IDX_T phStart = (h < kernelH) ? 0 : (h - kernelH) / strideH + 1;
        IDX_T phEnd = Simt::Min(h / strideH + 1, static_cast<IDX_T>(pooledHeight));
        IDX_T pwStart = (w < kernelW) ? 0 : (w - kernelW) / strideW + 1;
        IDX_T pwEnd = Simt::Min(w / strideW + 1, static_cast<IDX_T>(pooledWidth));

        ACC_VALUE_T gradient = 0;
        const __gm__ VALUE_T* xDataSlice = xData + n * pooledHeight * pooledWidth;
        CycleUpdateGradValue<VALUE_T, IDX_T, ACC_VALUE_T, FORMAT_NCHW_TYPE, COUNTPAD_T, DIV_T>(
                            channels, height, width, pooledWidth, phStart, phEnd, pwStart, pwEnd,
                            strideH, strideW, padHL, padWL, padHR, padWR,
                            kernelH, kernelW, divisorOverride, xDataSlice, &gradient);
        yData[index] = static_cast<VALUE_T>(gradient);
    }
}

template <typename VALUE_T,typename IDX_T, typename UIDX_T, typename ACC_VALUE_T,
          uint32_t COUNTPAD_T, uint32_t DIV_T>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_DIM) inline void AvgPoolV2GradSimtNhwcKernel(
    const int64_t count, __ubuf__ UIDX_T* simtParam,
    __ubuf__ int32_t* tilingDataParam, const __gm__ VALUE_T* xData,
    const IDX_T channels, const IDX_T height, const IDX_T width,
    const IDX_T pooledHeight, const IDX_T pooledWidth, __gm__ VALUE_T* yData)
{
    const auto& magicW = simtParam[MAGIC_W_IDX];
    const auto& shiftW = simtParam[SHIFT_W_IDX];
    const auto& magicH = simtParam[MAGIC_H_IDX];
    const auto& shiftH = simtParam[SHIFT_H_IDX];
    const auto& magicC = simtParam[MAGIC_C_IDX];
    const auto& shiftC = simtParam[SHIFT_C_IDX];

    const auto& kernelH = tilingDataParam[KERNEL_H_IDX];
    const auto& kernelW = tilingDataParam[KERNEL_W_IDX];
    const auto& strideH = tilingDataParam[STRIDE_H_IDX];
    const auto& strideW = tilingDataParam[STRIDE_W_IDX];
    const auto& padHL = tilingDataParam[PAD_HL_IDX];
    const auto& padHR = tilingDataParam[PAD_HR_IDX];
    const auto& padWL = tilingDataParam[PAD_WL_IDX];
    const auto& padWR = tilingDataParam[PAD_WR_IDX];
    const auto& divisorOverride = tilingDataParam[DIV_IDX];

    for (IDX_T index = Simt::GetBlockIdx() * Simt::GetThreadNum() + Simt::GetThreadIdx(); index < count;
         index = index + Simt::GetBlockNum() * Simt::GetThreadNum()) {
        UIDX_T dim0Idx = Simt::UintDiv(static_cast<UIDX_T>(index), magicC, shiftC);
        IDX_T c = index - dim0Idx * static_cast<UIDX_T>(channels);
        UIDX_T dim1Idx = Simt::UintDiv(dim0Idx, magicW, shiftW);
        IDX_T w = dim0Idx - dim1Idx * static_cast<UIDX_T>(width);
        IDX_T n = Simt::UintDiv(dim1Idx, magicH, shiftH);
        IDX_T h = dim1Idx - n * static_cast<UIDX_T>(height);

        h += padHL;
        w += padWL;

        IDX_T phStart = (h < kernelH) ? 0 : (h - kernelH) / strideH + 1;
        IDX_T phEnd = Simt::Min(h / strideH + 1, static_cast<IDX_T>(pooledHeight));
        IDX_T pwStart = (w < kernelW) ? 0 : (w - kernelW) / strideW + 1;
        IDX_T pwEnd = Simt::Min(w / strideW + 1, static_cast<IDX_T>(pooledWidth));

        ACC_VALUE_T gradient = 0;
        const __gm__ VALUE_T* xDataSlice = xData + n * channels * pooledHeight * pooledWidth + c;
        CycleUpdateGradValue<VALUE_T, IDX_T, ACC_VALUE_T, FORMAT_NHWC_TYPE, COUNTPAD_T, DIV_T>(
                            channels, height, width, pooledWidth, phStart, phEnd, pwStart, pwEnd,
                            strideH, strideW, padHL, padWL, padHR, padWR,
                            kernelH, kernelW, divisorOverride, xDataSlice, &gradient);
        yData[index] = static_cast<VALUE_T>(gradient);
    }
}

template <typename VALUE_T, typename IDX_T, uint32_t FORMAT_T, uint32_t COUNTPAD_T, uint32_t DIV_T>
__aicore__ inline void AvgPoolV2GradSimt<VALUE_T, IDX_T, FORMAT_T, COUNTPAD_T, DIV_T>::Compute()
{
    const int32_t kH = tilingData_->kSizeH;
    const int32_t kW = tilingData_->kSizeW;

    const int32_t dH = tilingData_->stridesH;
    const int32_t dW = tilingData_->stridesW;

    const int32_t padHL = tilingData_->padHLeft;
    const int32_t padHR = tilingData_->padHRight;
    const int32_t padWL = tilingData_->padWLeft;
    const int32_t padWR = tilingData_->padWRight;

    const int64_t nbatch = tilingData_->nDim;
    const int64_t inputChannel = tilingData_->cDim;
    const int64_t inputHeight = tilingData_->hInDim;
    const int64_t inputWidth = tilingData_->wInDim;

    const int64_t pooledHeight = tilingData_->hPooledDim;
    const int64_t pooledWidth = tilingData_->wPooledDim;

    const bool countIncludePad = tilingData_->countIncludePad;
    const int32_t divisorOverride = tilingData_->divisorOverride;

    auto xAddr = (__gm__ VALUE_T*)x_.GetPhyAddr();
    auto yAddr = (__gm__ VALUE_T*)y_.GetPhyAddr();

    using UIDX_T = std::conditional_t<std::is_same_v<IDX_T, int32_t>, uint32_t, uint64_t>;
    int64_t count = nbatch * inputChannel * inputHeight * inputWidth;

    UIDX_T magicW = 0;
    UIDX_T shiftW = 0;
    UIDX_T magicH = 0;
    UIDX_T shiftH = 0;
    UIDX_T magicC = 0;
    UIDX_T shiftC = 0;
    GetUintDivMagicAndShift<UIDX_T>(magicC, shiftC, inputChannel);
    GetUintDivMagicAndShift<UIDX_T>(magicH, shiftH, inputHeight);
    GetUintDivMagicAndShift<UIDX_T>(magicW, shiftW, inputWidth);
    LocalTensor<UIDX_T> simtParam = paramBuf_.Get<UIDX_T>();
    simtParam.SetValue(MAGIC_W_IDX, static_cast<UIDX_T>(magicW));
    simtParam.SetValue(SHIFT_W_IDX, static_cast<UIDX_T>(shiftW));
    simtParam.SetValue(MAGIC_H_IDX, static_cast<UIDX_T>(magicH));
    simtParam.SetValue(SHIFT_H_IDX, static_cast<UIDX_T>(shiftH));
    simtParam.SetValue(MAGIC_C_IDX, static_cast<UIDX_T>(magicC));
    simtParam.SetValue(SHIFT_C_IDX, static_cast<UIDX_T>(shiftC));
    
    LocalTensor<int32_t> tilingDataParam = tilingDataBuf_.Get<int32_t>();
    tilingDataParam.SetValue(KERNEL_H_IDX, static_cast<int32_t>(kH));
    tilingDataParam.SetValue(KERNEL_W_IDX, static_cast<int32_t>(kW));
    tilingDataParam.SetValue(STRIDE_H_IDX, static_cast<int32_t>(dH));
    tilingDataParam.SetValue(STRIDE_W_IDX, static_cast<int32_t>(dW));
    tilingDataParam.SetValue(PAD_HL_IDX, static_cast<int32_t>(padHL));
    tilingDataParam.SetValue(PAD_HR_IDX, static_cast<int32_t>(padHR));
    tilingDataParam.SetValue(PAD_WL_IDX, static_cast<int32_t>(padWL));
    tilingDataParam.SetValue(PAD_WR_IDX, static_cast<int32_t>(padWR));
    tilingDataParam.SetValue(DIV_IDX, static_cast<int32_t>(divisorOverride));
    DataSyncBarrier<MemDsbT::UB>();

    if constexpr (FORMAT_T == FORMAT_NCHW_TYPE) {
        Simt::VF_CALL<AvgPoolV2GradSimtNchwKernel<VALUE_T, IDX_T, UIDX_T, float, COUNTPAD_T, DIV_T>>(
            Simt::Dim3(THREAD_DIM), count,
            (__ubuf__ UIDX_T*)simtParam.GetPhyAddr(),
            (__ubuf__ int32_t*)tilingDataParam.GetPhyAddr(),
            xAddr, inputChannel, inputHeight, inputWidth,
            pooledHeight, pooledWidth, yAddr);
    } else {
        Simt::VF_CALL<AvgPoolV2GradSimtNhwcKernel<VALUE_T, IDX_T, UIDX_T, float, COUNTPAD_T, DIV_T>>(
            Simt::Dim3(THREAD_DIM), count,
            (__ubuf__ UIDX_T*)simtParam.GetPhyAddr(),
            (__ubuf__ int32_t*)tilingDataParam.GetPhyAddr(),
            xAddr, inputChannel, inputHeight, inputWidth,
            pooledHeight, pooledWidth, yAddr);
    }
}

}
#endif // CANN_AVG_POOL_V2_GRAD_SIMT_H

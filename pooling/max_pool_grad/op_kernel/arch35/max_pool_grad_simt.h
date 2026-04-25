/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MAX_POOL_GRAD_KERNEL_H
#define MAX_POOL_GRAD_KERNEL_H

#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "../inc/kernel_utils.h"
#include "../inc/platform.h"
#include "../pool_grad_common/arch35/max_pool_grad_with_argmax_struct_common.h"
#include "max_pool_grad_struct.h"

#ifdef __CCE_KT_TEST__
#define LAUNCH_BOUND(threads)
#endif

namespace MaxPoolGrad {

using namespace AscendC;
constexpr uint32_t THREAD_DIM = 256;

enum SimtParamIndex
{
    MAGIC_C_IDX = 0,
    MAGIC_H_IDX = 2,
    MAGIC_W_IDX = 4,
    MAGIC_STRIDE_H_IDX = 6,
    MAGIC_STRIDE_W_IDX = 8,
    SIMT_PARAMS_NUM = 10
};

template <typename VALUE_T, typename PROCESS_T>
__simt_callee__ __aicore__ inline static void CycleUpdate(
    VALUE_T val, PROCESS_T idxOffset, VALUE_T* maxVal, PROCESS_T* maxIdx)
{
    if ((static_cast<VALUE_T>(val) > *maxVal) || isnan(static_cast<float>(val))) {
        *maxIdx = idxOffset;
        *maxVal = val;
    }
}

template <typename PROCESS_T>
__simt_callee__ __aicore__ inline static void CalcPoolWindowBounds(
    PROCESS_T ph, PROCESS_T pw, int32_t strideH, int32_t strideW, int32_t padH, int32_t padW, int32_t kernelH,
    int32_t kernelW, int32_t dilationH, int32_t dilationW, int64_t height, int64_t width, PROCESS_T& hStart,
    PROCESS_T& wStart, PROCESS_T& hEnd, PROCESS_T& wEnd, PROCESS_T& maxIdx)
{
    hStart = ph * strideH - padH;
    wStart = pw * strideW - padW;
    hEnd = (hStart + (kernelH - 1) * dilationH + 1) < height ? (hStart + (kernelH - 1) * dilationH + 1) : height;
    wEnd = (wStart + (kernelW - 1) * dilationW + 1) < width ? (wStart + (kernelW - 1) * dilationW + 1) : width;
    while (hStart < 0) {
        hStart += dilationH;
    }
    while (wStart < 0) {
        wStart += dilationW;
    }
    maxIdx = hStart * width + wStart;
}

template <typename VALUE_T, typename INDICES_T, int64_t Format_T>
class MaxPoolGradSIMT {
public:
    using TilingData = MaxPoolGradWithArgmaxNHWCNameSpace::MaxPoolGradWithArgmaxSimtTilingCommonData;

    __aicore__ inline MaxPoolGradSIMT(TPipe* pipe, const TilingData* __restrict__ tilingData)
        : pipe_(pipe), tilingData_(tilingData)
    {}
    __aicore__ inline void Init(GM_ADDR orig_x, GM_ADDR orig_y, GM_ADDR grad, GM_ADDR y, GM_ADDR workspace);
    __aicore__ inline void Process();
    __aicore__ inline void ComputePos() const;
    __aicore__ inline void ComputeBack();

private:
    AscendC::TPipe* pipe_;
    AscendC::GlobalTensor<VALUE_T> origx_;
    AscendC::GlobalTensor<VALUE_T> grad_;
    AscendC::GlobalTensor<INDICES_T> argmax_;
    AscendC::GlobalTensor<VALUE_T> y_;
    const TilingData* __restrict__ tilingData_;
    uint32_t blockIdx_ = 0;
    uint32_t blockNum_ = 1;
    TBuf<TPosition::VECCALC> paramBuf_;
};

template <typename VALUE_T, typename INDICES_T, int64_t Format_T>
__aicore__ inline void MaxPoolGradSIMT<VALUE_T, INDICES_T, Format_T>::Init(
    GM_ADDR orig_x, GM_ADDR orig_y, GM_ADDR grad, GM_ADDR y, GM_ADDR workspace)
{
    origx_.SetGlobalBuffer((__gm__ VALUE_T*)(orig_x));
    grad_.SetGlobalBuffer((__gm__ VALUE_T*)(grad));
    y_.SetGlobalBuffer((__gm__ VALUE_T*)(y));
    argmax_.SetGlobalBuffer((__gm__ INDICES_T*)(workspace));
    pipe_->InitBuffer(paramBuf_, SIMT_PARAMS_NUM * sizeof(INDICES_T));
}

template <typename VALUE_T, typename INDICES_T, int64_t Format_T>
__aicore__ inline void MaxPoolGradSIMT<VALUE_T, INDICES_T, Format_T>::Process()
{
    ComputePos();
    AscendC::SyncAll();
    ComputeBack();
}

template <typename VAL_T, typename IDX_T, typename DIV_T>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_DIM) inline void MaxPoolNchw(
    const int64_t count, const __gm__ VAL_T* bottomData, const int64_t height, const int64_t width,
    const int32_t outputHeight, const int32_t outputWidth, const int32_t kernelH, const int32_t kernelW,
    const int32_t strideH, const int32_t strideW, const int32_t padH, const int32_t padW, const int32_t dilationH,
    const int32_t dilationW, __gm__ volatile IDX_T* topMask, int32_t blockIdx, int32_t blockNum, DIV_T m0, DIV_T shift0,
    DIV_T m1, DIV_T shift1)
{
    using PROCESS_T = typename std::conditional<std::is_same<DIV_T, uint32_t>::value, int32_t, int64_t>::type;
    for (DIV_T index = blockIdx * blockDim.x + threadIdx.x; index < count; index = index + blockNum * blockDim.x) {
        DIV_T dim0Idx = Simt::UintDiv(index, m0, shift0);
        DIV_T pw = index - dim0Idx * outputWidth;
        DIV_T nxc = Simt::UintDiv(dim0Idx, m1, shift1);
        DIV_T ph = dim0Idx - nxc * outputHeight;

        PROCESS_T hStart, wStart, hEnd, wEnd, maxIdx;
        CalcPoolWindowBounds<PROCESS_T>(
            ph, pw, strideH, strideW, padH, padW, kernelH, kernelW, dilationH, dilationW, height, width, hStart, wStart,
            hEnd, wEnd, maxIdx);
        VAL_T maxVal = AscendC::NumericLimits<VAL_T>::NegativeInfinity();
        auto firData = bottomData + nxc * height * width;
        for (PROCESS_T h = hStart; h < hEnd; h += dilationH) {
            for (PROCESS_T w = wStart; w < wEnd; w += dilationW) {
                PROCESS_T idxOffset = h * width + w;
                VAL_T val = static_cast<VAL_T>(firData[idxOffset]);
                CycleUpdate<VAL_T, PROCESS_T>(val, idxOffset, &maxVal, &maxIdx);
            }
        }
        topMask[index] = static_cast<IDX_T>(maxIdx);
    }
}

template <typename VAL_T, typename IDX_T, typename DIV_T>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_DIM) inline void MaxPoolNhwc(
    const int64_t count, const __gm__ VAL_T* bottomData, const int64_t channels, const int64_t height,
    const int64_t width, const int32_t outputHeight, const int32_t outputWidth, const int32_t kernelH,
    const int32_t kernelW, const int32_t strideH, const int32_t strideW, const int32_t padH, const int32_t padW,
    const int32_t dilationH, const int32_t dilationW, __gm__ volatile IDX_T* topMask, int32_t blockIdx,
    int32_t blockNum, DIV_T m0, DIV_T shift0, DIV_T m1, DIV_T shift1, DIV_T m2, DIV_T shift2)
{
    using PROCESS_T = typename std::conditional<std::is_same<DIV_T, uint32_t>::value, int32_t, int64_t>::type;
    for (DIV_T index = blockIdx * blockDim.x + threadIdx.x; index < count; index = index + blockNum * blockDim.x) {
        DIV_T dim0Idx = Simt::UintDiv(index, m0, shift0);
        DIV_T c = index - dim0Idx * channels;
        DIV_T dim1Idx = Simt::UintDiv(dim0Idx, m1, shift1);
        DIV_T pw = dim0Idx - dim1Idx * outputWidth;
        DIV_T n = Simt::UintDiv(dim1Idx, m2, shift2);
        DIV_T ph = dim1Idx - n * outputHeight;

        PROCESS_T hStart, wStart, hEnd, wEnd, maxIdx;
        CalcPoolWindowBounds<PROCESS_T>(
            ph, pw, strideH, strideW, padH, padW, kernelH, kernelW, dilationH, dilationW, height, width, hStart, wStart,
            hEnd, wEnd, maxIdx);
        VAL_T maxVal = AscendC::NumericLimits<VAL_T>::NegativeInfinity();
        auto firData = bottomData + (n * height * width * channels) + c;
        for (PROCESS_T h = hStart; h < hEnd; h += dilationH) {
            for (PROCESS_T w = wStart; w < wEnd; w += dilationW) {
                PROCESS_T idxOffset = h * width + w;
                VAL_T val = static_cast<VAL_T>(firData[idxOffset * channels]);
                CycleUpdate<VAL_T, PROCESS_T>(val, idxOffset, &maxVal, &maxIdx);
            }
        }
        topMask[index] = static_cast<IDX_T>(maxIdx);
    }
}

template <typename IDX_T>
__simt_callee__ __aicore__ inline static IDX_T PStart(
    int64_t size, int64_t pad, int64_t kernel, int64_t dilation, IDX_T magicStride, IDX_T shiftStride)
{
    if (size + pad < ((kernel - 1) * dilation + 1)) {
        return 0;
    } else {
        using DIV_T = typename std::conditional<std::is_same<IDX_T, int32_t>::value, uint32_t, uint64_t>::type;
        IDX_T phStart = size + pad - ((kernel - 1) * dilation + 1);
        phStart = Simt::UintDiv<DIV_T>(phStart, magicStride, shiftStride);
        phStart += 1;
        return phStart;
    }
}

template <typename IDX_T>
__simt_callee__ __aicore__ inline static IDX_T PEnd(
    int64_t size, int64_t pad, int64_t poolSize, IDX_T magicStride, IDX_T shiftStride)
{
    using DIV_T = typename std::conditional<std::is_same<IDX_T, int32_t>::value, uint32_t, uint64_t>::type;
    IDX_T pEnd = size + pad;
    pEnd = Simt::UintDiv<DIV_T>(pEnd, magicStride, shiftStride);
    pEnd += 1;
    return (pEnd > poolSize) ? static_cast<IDX_T>(poolSize) : pEnd;
}

template <typename VALUE_T, typename IDX_T, typename DIV_T>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_DIM) inline void MaxPoolGradNchw(
    __ubuf__ IDX_T* simtParams, const __gm__ VALUE_T* gradY, const int64_t nDims, const int64_t cDims,
    const int64_t hDims, const int64_t wDims, const int64_t hOutDim, const int64_t wOutDim, const int64_t kernelH,
    const int64_t kernelW, const int64_t padH, const int64_t padW, const int64_t dilationH, const int64_t dilationW,
    __gm__ VALUE_T* gradX, __gm__ IDX_T* argmax)
{
    DIV_T magicC = simtParams[MAGIC_C_IDX];
    DIV_T shiftC = simtParams[MAGIC_C_IDX + 1];
    DIV_T magicH = simtParams[MAGIC_H_IDX];
    DIV_T shiftH = simtParams[MAGIC_H_IDX + 1];
    DIV_T magicW = simtParams[MAGIC_W_IDX];
    DIV_T shiftW = simtParams[MAGIC_W_IDX + 1];
    DIV_T magicStrideH = simtParams[MAGIC_STRIDE_H_IDX];
    DIV_T shiftStrideH = simtParams[MAGIC_STRIDE_H_IDX + 1];
    DIV_T magicStrideW = simtParams[MAGIC_STRIDE_W_IDX];
    DIV_T shiftStrideW = simtParams[MAGIC_STRIDE_W_IDX + 1];

    DIV_T count = nDims * cDims * hDims * wDims;
    for (DIV_T index = blockIdx.x * blockDim.x + threadIdx.x; index < count; index += gridDim.x * blockDim.x) {
        DIV_T temp1 = Simt::UintDiv(index, magicW, shiftW);
        DIV_T w = index - temp1 * wDims;
        DIV_T temp2 = Simt::UintDiv(temp1, magicH, shiftH);
        DIV_T h = temp1 - temp2 * hDims;
        DIV_T n = Simt::UintDiv(temp2, magicC, shiftC);
        DIV_T c = temp2 - n * cDims;

        DIV_T inputIdx = h * wDims + w;
        IDX_T phStarts = PStart<IDX_T>(h, padH, kernelH, dilationH, magicStrideH, shiftStrideH);
        IDX_T phEnds = PEnd<IDX_T>(h, padH, hOutDim, magicStrideH, shiftStrideH);
        IDX_T pwStarts = PStart<IDX_T>(w, padW, kernelW, dilationW, magicStrideW, shiftStrideW);
        IDX_T pwEnds = PEnd<IDX_T>(w, padW, wOutDim, magicStrideW, shiftStrideW);

        float gradient = 0.0f;
        for (IDX_T ph = phStarts; ph < phEnds; ++ph) {
            for (IDX_T pw = pwStarts; pw < pwEnds; ++pw) {
                DIV_T outputIdx = n * cDims * hOutDim * wOutDim + c * hOutDim * wOutDim + ph * wOutDim + pw;
                if (argmax[outputIdx] == inputIdx) {
                    gradient += static_cast<float>(gradY[outputIdx]);
                }
            }
        }
        gradX[index] = static_cast<VALUE_T>(gradient);
    }
}

template <typename VALUE_T, typename IDX_T, typename DIV_T>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_DIM) inline void MaxPoolGradNhwc(
    __ubuf__ IDX_T* simtParam, const __gm__ VALUE_T* gradY, const int64_t nDims, const int64_t cDims,
    const int64_t hDims, const int64_t wDims, const int64_t hOutDim, const int64_t wOutDim, const int64_t kernelH,
    const int64_t kernelW, const int64_t padH, const int64_t padW, const int64_t dilationH, const int64_t dilationW,
    __gm__ VALUE_T* gradX, __gm__ IDX_T* argmax)
{
    DIV_T magicC = simtParam[MAGIC_C_IDX];
    DIV_T shiftC = simtParam[MAGIC_C_IDX + 1];
    DIV_T magicH = simtParam[MAGIC_H_IDX];
    DIV_T shiftH = simtParam[MAGIC_H_IDX + 1];
    DIV_T magicW = simtParam[MAGIC_W_IDX];
    DIV_T shiftW = simtParam[MAGIC_W_IDX + 1];
    DIV_T magicStrideH = simtParam[MAGIC_STRIDE_H_IDX];
    DIV_T shiftStrideH = simtParam[MAGIC_STRIDE_H_IDX + 1];
    DIV_T magicStrideW = simtParam[MAGIC_STRIDE_W_IDX];
    DIV_T shiftStrideW = simtParam[MAGIC_STRIDE_W_IDX + 1];

    DIV_T count = nDims * cDims * hDims * wDims;
    for (DIV_T index = blockIdx.x * blockDim.x + threadIdx.x; index < count; index += gridDim.x * blockDim.x) {
        DIV_T temp1 = Simt::UintDiv(index, magicC, shiftC);
        DIV_T c = index - temp1 * cDims;
        DIV_T temp2 = Simt::UintDiv(temp1, magicW, shiftW);
        DIV_T w = temp1 - temp2 * wDims;
        DIV_T n = Simt::UintDiv(temp2, magicH, shiftH);
        DIV_T h = temp2 - n * hDims;

        DIV_T inputIdx = h * wDims + w;
        IDX_T phStart = PStart<IDX_T>(h, padH, kernelH, dilationH, magicStrideH, shiftStrideH);
        IDX_T phEnd = PEnd<IDX_T>(h, padH, hOutDim, magicStrideH, shiftStrideH);
        IDX_T pwStart = PStart<IDX_T>(w, padW, kernelW, dilationW, magicStrideW, shiftStrideW);
        IDX_T pwEnd = PEnd<IDX_T>(w, padW, wOutDim, magicStrideW, shiftStrideW);

        float gradient = 0.0f;
        for (IDX_T ph = phStart; ph < phEnd; ++ph) {
            for (IDX_T pw = pwStart; pw < pwEnd; ++pw) {
                DIV_T outputIdx = n * hOutDim * wOutDim * cDims + ph * wOutDim * cDims + pw * cDims + c;
                if (argmax[outputIdx] == inputIdx) {
                    gradient += static_cast<float>(gradY[outputIdx]);
                }
            }
        }
        gradX[index] = static_cast<VALUE_T>(gradient);
    }
}

template <typename VALUE_T, typename INDICES_T, int64_t Format_T>
__aicore__ inline void MaxPoolGradSIMT<VALUE_T, INDICES_T, Format_T>::ComputePos() const
{
    auto inputData = (__gm__ VALUE_T*)origx_.GetPhyAddr();
    auto indicesData = (__gm__ volatile INDICES_T*)argmax_.GetPhyAddr();
    int64_t totalSize = tilingData_->nDim * tilingData_->cDim * tilingData_->hOutDim * tilingData_->wOutDim;

    if constexpr (std::is_same<INDICES_T, int32_t>::value) {
        uint32_t m0 = 1, shift0 = 1, m1 = 1, shift1 = 1;
        if constexpr (Format_T == 0) {
            GetUintDivMagicAndShift(m0, shift0, static_cast<uint32_t>(tilingData_->wOutDim));
            GetUintDivMagicAndShift(m1, shift1, static_cast<uint32_t>(tilingData_->hOutDim));
            asc_vf_call<MaxPoolNchw<VALUE_T, INDICES_T, uint32_t>>(
                dim3(THREAD_DIM), totalSize, inputData, tilingData_->hInDim, tilingData_->wInDim, tilingData_->hOutDim,
                tilingData_->wOutDim, tilingData_->kSizeH, tilingData_->kSizeW, tilingData_->stridesH,
                tilingData_->stridesW, tilingData_->padH, tilingData_->padW, tilingData_->dilationH,
                tilingData_->dilationW, indicesData, blockIdx_, blockNum_, m0, shift0, m1, shift1);
        } else {
            uint32_t m2 = 1, shift2 = 1;
            GetUintDivMagicAndShift(m0, shift0, static_cast<uint32_t>(tilingData_->cDim));
            GetUintDivMagicAndShift(m1, shift1, static_cast<uint32_t>(tilingData_->wOutDim));
            GetUintDivMagicAndShift(m2, shift2, static_cast<uint32_t>(tilingData_->hOutDim));
            asc_vf_call<MaxPoolNhwc<VALUE_T, INDICES_T, uint32_t>>(
                dim3(THREAD_DIM), totalSize, inputData, tilingData_->cDim, tilingData_->hInDim, tilingData_->wInDim,
                tilingData_->hOutDim, tilingData_->wOutDim, tilingData_->kSizeH, tilingData_->kSizeW,
                tilingData_->stridesH, tilingData_->stridesW, tilingData_->padH, tilingData_->padW,
                tilingData_->dilationH, tilingData_->dilationW, indicesData, blockIdx_, blockNum_, m0, shift0, m1,
                shift1, m2, shift2);
        }
    } else {
        uint64_t m0 = 1, shift0 = 1, m1 = 1, shift1 = 1;
        if constexpr (Format_T == 0) {
            GetUintDivMagicAndShift(m0, shift0, static_cast<uint64_t>(tilingData_->wOutDim));
            GetUintDivMagicAndShift(m1, shift1, static_cast<uint64_t>(tilingData_->hOutDim));
            asc_vf_call<MaxPoolNchw<VALUE_T, INDICES_T, uint64_t>>(
                dim3(THREAD_DIM), totalSize, inputData, tilingData_->hInDim, tilingData_->wInDim, tilingData_->hOutDim,
                tilingData_->wOutDim, tilingData_->kSizeH, tilingData_->kSizeW, tilingData_->stridesH,
                tilingData_->stridesW, tilingData_->padH, tilingData_->padW, tilingData_->dilationH,
                tilingData_->dilationW, indicesData, blockIdx_, blockNum_, m0, shift0, m1, shift1);
        } else {
            uint64_t m2 = 1, shift2 = 1;
            GetUintDivMagicAndShift(m0, shift0, static_cast<uint64_t>(tilingData_->cDim));
            GetUintDivMagicAndShift(m1, shift1, static_cast<uint64_t>(tilingData_->wOutDim));
            GetUintDivMagicAndShift(m2, shift2, static_cast<uint64_t>(tilingData_->hOutDim));
            asc_vf_call<MaxPoolNhwc<VALUE_T, INDICES_T, uint64_t>>(
                dim3(THREAD_DIM), totalSize, inputData, tilingData_->cDim, tilingData_->hInDim, tilingData_->wInDim,
                tilingData_->hOutDim, tilingData_->wOutDim, tilingData_->kSizeH, tilingData_->kSizeW,
                tilingData_->stridesH, tilingData_->stridesW, tilingData_->padH, tilingData_->padW,
                tilingData_->dilationH, tilingData_->dilationW, indicesData, blockIdx_, blockNum_, m0, shift0, m1,
                shift1, m2, shift2);
        }
    }
}

template <typename VALUE_T, typename INDICES_T, int64_t Format_T>
__aicore__ inline void MaxPoolGradSIMT<VALUE_T, INDICES_T, Format_T>::ComputeBack()
{
    LocalTensor<INDICES_T> simtParam = paramBuf_.Get<INDICES_T>();
    using DIV_T = typename std::conditional<std::is_same<INDICES_T, int32_t>::value, uint32_t, uint64_t>::type;

    DIV_T magicC = 0, shiftC = 0;
    DIV_T magicH = 0, shiftH = 0;
    DIV_T magicW = 0, shiftW = 0;
    DIV_T magicStrideH = 0, shiftStrideH = 0;
    DIV_T magicStrideW = 0, shiftStrideW = 0;

    GetUintDivMagicAndShift<DIV_T>(magicC, shiftC, tilingData_->cDim);
    GetUintDivMagicAndShift<DIV_T>(magicH, shiftH, tilingData_->hInDim);
    GetUintDivMagicAndShift<DIV_T>(magicW, shiftW, tilingData_->wInDim);
    GetUintDivMagicAndShift<DIV_T>(magicStrideH, shiftStrideH, tilingData_->stridesH);
    GetUintDivMagicAndShift<DIV_T>(magicStrideW, shiftStrideW, tilingData_->stridesW);

    simtParam.SetValue(MAGIC_C_IDX, static_cast<INDICES_T>(magicC));
    simtParam.SetValue(MAGIC_C_IDX + 1, static_cast<INDICES_T>(shiftC));
    simtParam.SetValue(MAGIC_H_IDX, static_cast<INDICES_T>(magicH));
    simtParam.SetValue(MAGIC_H_IDX + 1, static_cast<INDICES_T>(shiftH));
    simtParam.SetValue(MAGIC_W_IDX, static_cast<INDICES_T>(magicW));
    simtParam.SetValue(MAGIC_W_IDX + 1, static_cast<INDICES_T>(shiftW));
    simtParam.SetValue(MAGIC_STRIDE_H_IDX, static_cast<INDICES_T>(magicStrideH));
    simtParam.SetValue(MAGIC_STRIDE_H_IDX + 1, static_cast<INDICES_T>(shiftStrideH));
    simtParam.SetValue(MAGIC_STRIDE_W_IDX, static_cast<INDICES_T>(magicStrideW));
    simtParam.SetValue(MAGIC_STRIDE_W_IDX + 1, static_cast<INDICES_T>(shiftStrideW));

    DataSyncBarrier<MemDsbT::UB>();

    auto gradData = (__gm__ VALUE_T*)grad_.GetPhyAddr();
    auto outputData = (__gm__ VALUE_T*)y_.GetPhyAddr();
    auto indicesData = (__gm__ INDICES_T*)argmax_.GetPhyAddr();

    if constexpr (std::is_same<INDICES_T, int32_t>::value) {
        if constexpr (Format_T == 1) {
            asc_vf_call<MaxPoolGradNhwc<VALUE_T, INDICES_T, uint32_t>>(
                dim3(THREAD_DIM), (__ubuf__ INDICES_T*)simtParam.GetPhyAddr(), gradData, tilingData_->nDim,
                tilingData_->cDim, tilingData_->hInDim, tilingData_->wInDim, tilingData_->hOutDim, tilingData_->wOutDim,
                tilingData_->kSizeH, tilingData_->kSizeW, tilingData_->padH, tilingData_->padW, tilingData_->dilationH,
                tilingData_->dilationW, outputData, indicesData);
        } else {
            asc_vf_call<MaxPoolGradNchw<VALUE_T, INDICES_T, uint32_t>>(
                dim3(THREAD_DIM), (__ubuf__ INDICES_T*)simtParam.GetPhyAddr(), gradData, tilingData_->nDim,
                tilingData_->cDim, tilingData_->hInDim, tilingData_->wInDim, tilingData_->hOutDim, tilingData_->wOutDim,
                tilingData_->kSizeH, tilingData_->kSizeW, tilingData_->padH, tilingData_->padW, tilingData_->dilationH,
                tilingData_->dilationW, outputData, indicesData);
        }
    } else {
        if constexpr (Format_T == 1) {
            asc_vf_call<MaxPoolGradNhwc<VALUE_T, INDICES_T, uint64_t>>(
                dim3(THREAD_DIM), (__ubuf__ INDICES_T*)simtParam.GetPhyAddr(), gradData, tilingData_->nDim,
                tilingData_->cDim, tilingData_->hInDim, tilingData_->wInDim, tilingData_->hOutDim, tilingData_->wOutDim,
                tilingData_->kSizeH, tilingData_->kSizeW, tilingData_->padH, tilingData_->padW, tilingData_->dilationH,
                tilingData_->dilationW, outputData, indicesData);
        } else {
            asc_vf_call<MaxPoolGradNchw<VALUE_T, INDICES_T, uint64_t>>(
                dim3(THREAD_DIM), (__ubuf__ INDICES_T*)simtParam.GetPhyAddr(), gradData, tilingData_->nDim,
                tilingData_->cDim, tilingData_->hInDim, tilingData_->wInDim, tilingData_->hOutDim, tilingData_->wOutDim,
                tilingData_->kSizeH, tilingData_->kSizeW, tilingData_->padH, tilingData_->padW, tilingData_->dilationH,
                tilingData_->dilationW, outputData, indicesData);
        }
    }
}

} // namespace MaxPoolGrad

#endif
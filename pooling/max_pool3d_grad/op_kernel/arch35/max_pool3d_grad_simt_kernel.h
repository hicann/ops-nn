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
 * \file max_pool3d_grad_simt_kernel.h
 * \brief
 */
#ifndef MAX_POOL3D_GRAD_SIMT_KERNEL_H_
#define MAX_POOL3D_GRAD_SIMT_KERNEL_H_

#include "kernel_operator.h"
#include "../inc/kernel_utils.h"
#include "../inc/platform.h"
#include "max_pool3d_grad_struct.h"

#ifdef __CCE_KT_TEST__
#define LAUNCH_BOUND(threads)
#endif
namespace MaxPool3DGrad {

    using namespace AscendC;
    constexpr uint32_t THREAD_DIM = 256;
    constexpr uint32_t DIM_0 = 0;
    constexpr uint32_t DIM_1 = 1; 
    constexpr uint32_t DIM_2 = 2;
    constexpr uint32_t DIM_3 = 3;
    constexpr uint32_t SHAPE_DIM_THERR = 3;
    constexpr uint32_t SHAPE_DIM_FOUR = 4;
    constexpr uint32_t TILING_DATA_NUM = 23;
    constexpr static uint32_t SIMT_PARAMS_NUM = 32;
    constexpr static uint32_t MAGIC_C_IDX = 0;
    constexpr static uint32_t MAGIC_D_IDX = 2;
    constexpr static uint32_t MAGIC_H_IDX = 4;
    constexpr static uint32_t MAGIC_W_IDX = 6;
    constexpr static uint32_t MAGIC_STRIDE_D_IDX = 8;
    constexpr static uint32_t MAGIC_STRIDE_H_IDX = 10;
    constexpr static uint32_t MAGIC_STRIDE_W_IDX = 12;

    template <typename VALUE_T, typename PROCESS_T>
    __simt_callee__ __aicore__ inline static void CycleUpdate(VALUE_T val, PROCESS_T idxOffset, VALUE_T *maxVal, PROCESS_T *maxIdx)
    {
        if ((static_cast<VALUE_T>(val) > *maxVal) || Simt::IsNan(static_cast<float>(val))) {
            *maxIdx = idxOffset;
            *maxVal = val;
        }
    }
template <typename VALUE_T, typename INDICES_T, int64_t Format_T, bool useINT64Index, typename OFFSET_T>
class MaxPool3DGradSimtKernel
{
public:
    __aicore__ inline MaxPool3DGradSimtKernel(TPipe *pipe, const Pool3DGradNameSpace::MaxPool3DGradSimtTilingData* __restrict__ tilingData) 
        : pipe_(pipe), tilingData_(tilingData) {};
    __aicore__ inline void Init(GM_ADDR orig_x, GM_ADDR orig_y, GM_ADDR grad, GM_ADDR y, GM_ADDR workspace);
    __aicore__ inline void Process();
    __aicore__ inline void ComputePos() const;
    __aicore__ inline void ComputeBack();

private:
    AscendC::TPipe *pipe_;
    AscendC::GlobalTensor<VALUE_T> origx_;
    AscendC::GlobalTensor<VALUE_T> origy_;
    AscendC::GlobalTensor<VALUE_T> grad_;
    AscendC::GlobalTensor<INDICES_T> argmax_;
    AscendC::GlobalTensor<VALUE_T> y_;
    const Pool3DGradNameSpace::MaxPool3DGradSimtTilingData* tilingData_;
    uint32_t blockIdx_ = 0;
    uint32_t blockNum_ = 1;
    TBuf<TPosition::VECCALC> simtTilingDataBuf_;
    TBuf<TPosition::VECCALC> paramBuf_;
};

template <typename VALUE_T, typename INDICES_T, int64_t Format_T, bool useINT64Index, typename OFFSET_T>
__aicore__ inline void MaxPool3DGradSimtKernel<VALUE_T, INDICES_T, Format_T, useINT64Index, OFFSET_T>::Init(GM_ADDR orig_x, GM_ADDR orig_y, GM_ADDR grad, GM_ADDR y, GM_ADDR workspace)
{
    origx_.SetGlobalBuffer((__gm__ VALUE_T*)(orig_x));
    origy_.SetGlobalBuffer((__gm__ VALUE_T*)(orig_y));
    grad_.SetGlobalBuffer((__gm__ VALUE_T*)(grad));
    y_.SetGlobalBuffer((__gm__ VALUE_T*)(y));
    argmax_.SetGlobalBuffer((__gm__ INDICES_T*)(workspace));
    pipe_->InitBuffer(simtTilingDataBuf_, TILING_DATA_NUM * sizeof(int64_t));
    pipe_->InitBuffer(paramBuf_, SIMT_PARAMS_NUM * sizeof(OFFSET_T));
}

template <typename VALUE_T, typename INDICES_T, int64_t Format_T, bool useINT64Index, typename OFFSET_T>
__aicore__ inline void MaxPool3DGradSimtKernel<VALUE_T, INDICES_T, Format_T, useINT64Index, OFFSET_T>::Process()
{
    ComputePos();
    AscendC::SyncAll();
    ComputeBack();
}

template <typename VAL_T, typename IDX_T, typename PROCESS_T, typename FASTDIV_T>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_DIM) inline void MaxPool3DNcdhw(const int64_t count, const __gm__ VAL_T* bottomData, const int64_t dSize,
                                                                               const int64_t height, const int64_t width, const int32_t outputDi, const int32_t outputHeight, const int32_t outputWidth, 
                                                                               const int32_t kernelD, const int32_t kernelH, const int32_t kernelW, const int32_t strideD, const int32_t strideH, const int32_t strideW, 
                                                                               const int32_t padD, const int32_t padH, const int32_t padW, const int32_t dilationD, const int32_t dilationH, const int32_t dilationW,
                                                                               __gm__ VAL_T* topData, __gm__ IDX_T* topMask, int32_t blockIdx, int32_t blockNum, FASTDIV_T m0, FASTDIV_T shift0,
                                                                               FASTDIV_T m1, FASTDIV_T shift1, FASTDIV_T m2, FASTDIV_T shift2)
{
    for (FASTDIV_T index = blockIdx * Simt::GetThreadNum() + Simt::GetThreadIdx(); index < count;
         index = index + blockNum * Simt::GetThreadNum()) {
        FASTDIV_T dim0Idx = Simt::UintDiv(index, m0, shift0);
        FASTDIV_T pw = index - dim0Idx * outputWidth;
        FASTDIV_T dim1Idx = Simt::UintDiv(dim0Idx, m1, shift1);
        FASTDIV_T ph = dim0Idx - dim1Idx * outputHeight;
        FASTDIV_T nxc = Simt::UintDiv(dim1Idx, m2, shift2);
        FASTDIV_T pd = dim1Idx - nxc * outputDi;
        PROCESS_T dStart = pd * strideD - padD;
        PROCESS_T hStart = ph * strideH - padH;
        PROCESS_T wStart = pw * strideW - padW;
        PROCESS_T dEnd = (dStart + (kernelD - 1) * dilationD + 1) < dSize ? (dStart + (kernelD - 1) * dilationD + 1) : dSize;
        PROCESS_T hEnd = (hStart + (kernelH - 1) * dilationH + 1) < height ? (hStart + (kernelH - 1) * dilationH + 1) : height;
        PROCESS_T wEnd = (wStart + (kernelW - 1) * dilationW + 1) < width ? (wStart + (kernelW - 1) * dilationW + 1) : width;
        while (dStart < 0) {
            dStart += dilationD;
        }
        while (hStart < 0) {
            hStart += dilationH;
        }
        while (wStart < 0) {
            wStart += dilationW;
        }
        
        VAL_T maxVal = AscendC::NumericLimits<VAL_T>::NegativeInfinity();
        PROCESS_T maxIdx = dStart * height * width + hStart * width + wStart;
        auto firData = bottomData + nxc * dSize * height * width;
        for (PROCESS_T d = dStart; d < dEnd; d += dilationD) {
            for (PROCESS_T h = hStart; h < hEnd; h += dilationH) {
                for (PROCESS_T w = wStart; w < wEnd; w += dilationW) {
                    PROCESS_T idxOffset = d * height * width + h * width + w;
                    VAL_T val = static_cast<VAL_T>(firData[idxOffset]);
                    CycleUpdate<VAL_T, PROCESS_T>(val, idxOffset, &maxVal, &maxIdx);
                }
            }
        }
        topMask[index] = static_cast<IDX_T>(maxIdx);
    }
}

template <typename VAL_T, typename IDX_T, typename PROCESS_T, typename FASTDIV_T>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_DIM) inline void MaxPool3DNdhwc(const int64_t count, const __gm__ VAL_T* bottomData, const int64_t channels, const int64_t dSize,
                                                                               const int64_t height, const int64_t width, const int32_t outputDi, const int32_t outputHeight, const int32_t outputWidth, 
                                                                               const int32_t kernelD, const int32_t kernelH, const int32_t kernelW, const int32_t strideD, const int32_t strideH, const int32_t strideW, 
                                                                               const int32_t padD, const int32_t padH, const int32_t padW, const int32_t dilationD, const int32_t dilationH, const int32_t dilationW,
                                                                               __gm__ VAL_T* topData, __gm__ IDX_T* topMask, int32_t blockIdx, int32_t blockNum, FASTDIV_T m0, FASTDIV_T shift0,
                                                                               FASTDIV_T m1, FASTDIV_T shift1, FASTDIV_T m2, FASTDIV_T shift2, FASTDIV_T m3, FASTDIV_T shift3)
{
    for (FASTDIV_T index = blockIdx * Simt::GetThreadNum() + Simt::GetThreadIdx(); index < count;
         index = index + blockNum * Simt::GetThreadNum()) {
        FASTDIV_T dim0Idx = Simt::UintDiv(index, m0, shift0);
        FASTDIV_T c = index - dim0Idx * channels;
        FASTDIV_T dim1Idx = Simt::UintDiv(dim0Idx, m1, shift1);
        FASTDIV_T pw = dim0Idx - dim1Idx * outputWidth;
        FASTDIV_T dim2Idx = Simt::UintDiv(dim1Idx, m2, shift2);
        FASTDIV_T ph = dim1Idx - dim2Idx * outputHeight;
        FASTDIV_T n = Simt::UintDiv(dim2Idx, m3, shift3);
        FASTDIV_T pd = dim2Idx - n * outputDi;
        PROCESS_T dStart = pd * strideD - padD;
        PROCESS_T hStart = ph * strideH - padH;
        PROCESS_T wStart = pw * strideW - padW;
        PROCESS_T dEnd = (dStart + (kernelD - 1) * dilationD + 1) < dSize ? (dStart + (kernelD - 1) * dilationD + 1) : dSize;
        PROCESS_T hEnd = (hStart + (kernelH - 1) * dilationH + 1) < height ? (hStart + (kernelH - 1) * dilationH + 1) : height;
        PROCESS_T wEnd = (wStart + (kernelW - 1) * dilationW + 1) < width ? (wStart + (kernelW - 1) * dilationW + 1) : width;
        while (dStart < 0) {
            dStart += dilationD;
        }
        while (hStart < 0) {
            hStart += dilationH;
        }
        while (wStart < 0) {
            wStart += dilationW;
        }
        VAL_T maxVal = AscendC::NumericLimits<VAL_T>::NegativeInfinity();
        PROCESS_T maxIdx = dStart * height * width + hStart * width + wStart;
        auto firData = bottomData + (n * dSize * height * width * channels) + c;
        for (PROCESS_T d = dStart; d < dEnd; d += dilationD) {
            for (PROCESS_T h = hStart; h < hEnd; h += dilationH) {
                for (PROCESS_T w = wStart; w < wEnd; w += dilationW) {
                    PROCESS_T idxOffset = d * height * width + h * width + w;
                    VAL_T val = static_cast<VAL_T>(firData[idxOffset * channels]);
                    CycleUpdate<VAL_T, PROCESS_T>(val, idxOffset, &maxVal, &maxIdx);
                }
            }
        }
        topMask[index] = static_cast<IDX_T>(maxIdx);
    }
}

template <typename OFFSET_T>
__simt_callee__ __aicore__ inline static OFFSET_T PStart(int64_t size, int64_t pad, int64_t kernel, int64_t dilation,
                                         OFFSET_T magicStride, OFFSET_T shiftStride)
{
    if (size + pad < ((kernel - 1) * dilation + 1)) {
        return 0;
    } else {
        using DIV_T = typename std::conditional<std::is_same<OFFSET_T, int32_t>::value, uint32_t, uint64_t>::type;
        OFFSET_T phStart = size + pad - ((kernel - 1) * dilation + 1);
        phStart = Simt::UintDiv<DIV_T>(phStart, magicStride, shiftStride);
        phStart += 1;
        return phStart;
    }
}

template <typename OFFSET_T>
__simt_callee__ __aicore__ inline static OFFSET_T PEnd(int64_t size, int64_t pad, int64_t poolSize, OFFSET_T magicStride,
                                       OFFSET_T shiftStride)
{
    using DIV_T = typename std::conditional<std::is_same<OFFSET_T, int32_t>::value, uint32_t, uint64_t>::type;
    OFFSET_T pEnd = size + pad;
    pEnd = Simt::UintDiv<DIV_T>(pEnd, magicStride, shiftStride);
    pEnd += 1;
    return (pEnd > poolSize) ? static_cast<OFFSET_T>(poolSize) : pEnd;
}

template <typename VALUE_T, typename INDICES_T, typename OFFSET_T>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_DIM) inline void MaxPool3DGradWithArgmaxNcdhw(
    __ubuf__ OFFSET_T* simtParams, const __gm__ VALUE_T* gradY, __ubuf__ Pool3DGradNameSpace::MaxPool3DGradSimtTilingData* tilingData_,
    __gm__ VALUE_T* gradX, __gm__ INDICES_T* argmax)
{
    OFFSET_T magicC = simtParams[MAGIC_C_IDX];
    OFFSET_T shiftC = simtParams[MAGIC_C_IDX + 1];
    OFFSET_T magicD = simtParams[MAGIC_D_IDX];
    OFFSET_T shiftD = simtParams[MAGIC_D_IDX + 1];
    OFFSET_T magicH = simtParams[MAGIC_H_IDX];
    OFFSET_T shiftH = simtParams[MAGIC_H_IDX + 1];
    OFFSET_T magicW = simtParams[MAGIC_W_IDX];
    OFFSET_T shiftW = simtParams[MAGIC_W_IDX + 1];
    OFFSET_T magicStrideD = simtParams[MAGIC_STRIDE_D_IDX];
    OFFSET_T shiftStrideD = simtParams[MAGIC_STRIDE_D_IDX + 1];
    OFFSET_T magicStrideH = simtParams[MAGIC_STRIDE_H_IDX];
    OFFSET_T shiftStrideH = simtParams[MAGIC_STRIDE_H_IDX + 1];
    OFFSET_T magicStrideW = simtParams[MAGIC_STRIDE_W_IDX];
    OFFSET_T shiftStrideW = simtParams[MAGIC_STRIDE_W_IDX + 1];
    using DIV_T = typename std::conditional<std::is_same<OFFSET_T, int32_t>::value, uint32_t, uint64_t>::type;
    DIV_T count = tilingData_->nDim * tilingData_->cDim * tilingData_->dInDim * tilingData_->hInDim * tilingData_->wInDim;
    for (DIV_T index = Simt::GetBlockIdx() * Simt::GetThreadNum() + Simt::GetThreadIdx(); index < count;
         index += Simt::GetBlockNum() * Simt::GetThreadNum()) {
        DIV_T temp1 = Simt::UintDiv(index, static_cast<DIV_T>(magicW), static_cast<DIV_T>(shiftW));
        DIV_T w = index - temp1 * static_cast<DIV_T>(tilingData_->wInDim);
        DIV_T temp2 = Simt::UintDiv(temp1, static_cast<DIV_T>(magicH), static_cast<DIV_T>(shiftH));
        DIV_T h = temp1 - temp2 * static_cast<DIV_T>(tilingData_->hInDim);
        DIV_T temp3 = Simt::UintDiv(temp2, static_cast<DIV_T>(magicD), static_cast<DIV_T>(shiftD));
        DIV_T d = temp2 - temp3 * static_cast<DIV_T>(tilingData_->dInDim);
        DIV_T n = Simt::UintDiv(temp3, static_cast<DIV_T>(magicC), static_cast<DIV_T>(shiftC));
        DIV_T c = temp3 - n * static_cast<DIV_T>(tilingData_->cDim);

        DIV_T inputIdx = d * tilingData_->hInDim * tilingData_->wInDim + h * tilingData_->wInDim + w;
        OFFSET_T pdStarts = PStart<OFFSET_T>(d, tilingData_->padD, tilingData_->kSizeD, tilingData_->dilationD, magicStrideD, shiftStrideD);
        OFFSET_T pdEnds = PEnd<OFFSET_T>(d, tilingData_->padD, tilingData_->dOutDim, magicStrideD, shiftStrideD);
        OFFSET_T phStarts = PStart<OFFSET_T>(h, tilingData_->padH, tilingData_->kSizeH, tilingData_->dilationH, magicStrideH, shiftStrideH);
        OFFSET_T phEnds = PEnd<OFFSET_T>(h, tilingData_->padH, tilingData_->hOutDim, magicStrideH, shiftStrideH);
        OFFSET_T pwStarts = PStart<OFFSET_T>(w, tilingData_->padW, tilingData_->kSizeW, tilingData_->dilationW, magicStrideW, shiftStrideW);
        OFFSET_T pwEnds = PEnd<OFFSET_T>(w, tilingData_->padW, tilingData_->wOutDim, magicStrideW, shiftStrideW);

        float gradient = 0.0f;
        for (OFFSET_T pd = pdStarts; pd < pdEnds; ++pd) {
            for (OFFSET_T ph = phStarts; ph < phEnds; ++ph) {
                for (OFFSET_T pw = pwStarts; pw < pwEnds; ++pw) {
                    DIV_T outputIdx = n * tilingData_->cDim * tilingData_->dOutDim * tilingData_->hOutDim * tilingData_->wOutDim + c * tilingData_->dOutDim * tilingData_->hOutDim * tilingData_->wOutDim +
                                      pd * tilingData_->hOutDim * tilingData_->wOutDim + ph * tilingData_->wOutDim + pw;
                    if (static_cast<DIV_T>(argmax[outputIdx]) == inputIdx) {
                        gradient += static_cast<float>(gradY[outputIdx]);
                    }
                }
            }
        }
        gradX[index] = static_cast<VALUE_T>(gradient);
    }
}

template <typename VALUE_T, typename INDICES_T, typename OFFSET_T>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_DIM) inline void MaxPool3DGradWithArgmaxNdhwc(
    __ubuf__ OFFSET_T* simtParam, const __gm__ VALUE_T* gradY, __ubuf__ Pool3DGradNameSpace::MaxPool3DGradSimtTilingData* tilingData_,
    __gm__ VALUE_T* gradX, __gm__ INDICES_T* argmax)
{
    OFFSET_T magicC = simtParam[MAGIC_C_IDX];
    OFFSET_T shiftC = simtParam[MAGIC_C_IDX + 1];
    OFFSET_T magicD = simtParam[MAGIC_D_IDX];
    OFFSET_T shiftD = simtParam[MAGIC_D_IDX + 1];
    OFFSET_T magicH = simtParam[MAGIC_H_IDX];
    OFFSET_T shiftH = simtParam[MAGIC_H_IDX + 1];
    OFFSET_T magicW = simtParam[MAGIC_W_IDX];
    OFFSET_T shiftW = simtParam[MAGIC_W_IDX + 1];
    OFFSET_T magicStrideD = simtParam[MAGIC_STRIDE_D_IDX];
    OFFSET_T shiftStrideD = simtParam[MAGIC_STRIDE_D_IDX + 1];
    OFFSET_T magicStrideH = simtParam[MAGIC_STRIDE_H_IDX];
    OFFSET_T shiftStrideH = simtParam[MAGIC_STRIDE_H_IDX + 1];
    OFFSET_T magicStrideW = simtParam[MAGIC_STRIDE_W_IDX];
    OFFSET_T shiftStrideW = simtParam[MAGIC_STRIDE_W_IDX + 1];
    using DIV_T = typename std::conditional<std::is_same<OFFSET_T, int32_t>::value, uint32_t, uint64_t>::type;
    DIV_T count = tilingData_->nDim * tilingData_->cDim * tilingData_->dInDim * tilingData_->hInDim * tilingData_->wInDim;
    for (DIV_T index = Simt::GetBlockIdx() * Simt::GetThreadNum() + Simt::GetThreadIdx(); index < count;
         index += Simt::GetBlockNum() * Simt::GetThreadNum()) {
        DIV_T temp1 = Simt::UintDiv(index, static_cast<DIV_T>(magicC), static_cast<DIV_T>(shiftC));
        DIV_T c = index - temp1 * static_cast<DIV_T>(tilingData_->cDim);
        DIV_T temp2 = Simt::UintDiv(temp1, static_cast<DIV_T>(magicW), static_cast<DIV_T>(shiftW));
        DIV_T w = temp1 - temp2 * static_cast<DIV_T>(tilingData_->wInDim);
        DIV_T temp3 = Simt::UintDiv(temp2, static_cast<DIV_T>(magicH), static_cast<DIV_T>(shiftH));
        DIV_T h = temp2 - temp3 * static_cast<DIV_T>(tilingData_->hInDim);
        DIV_T n = Simt::UintDiv(temp3, static_cast<DIV_T>(magicD), static_cast<DIV_T>(shiftD));
        DIV_T d = temp3 - n * static_cast<DIV_T>(tilingData_->dInDim);

        DIV_T inputIdx = d * tilingData_->hInDim * tilingData_->wInDim + h * tilingData_->wInDim + w;
        OFFSET_T pdStart = PStart<OFFSET_T>(d, tilingData_->padD, tilingData_->kSizeD, tilingData_->dilationD, magicStrideD, shiftStrideD);
        OFFSET_T pdEnd = PEnd<OFFSET_T>(d, tilingData_->padD, tilingData_->dOutDim, magicStrideD, shiftStrideD);
        OFFSET_T phStart = PStart<OFFSET_T>(h, tilingData_->padH, tilingData_->kSizeH, tilingData_->dilationH, magicStrideH, shiftStrideH);
        OFFSET_T phEnd = PEnd<OFFSET_T>(h, tilingData_->padH, tilingData_->hOutDim, magicStrideH, shiftStrideH);
        OFFSET_T pwStart = PStart<OFFSET_T>(w, tilingData_->padW, tilingData_->kSizeW, tilingData_->dilationW, magicStrideW, shiftStrideW);
        OFFSET_T pwEnd = PEnd<OFFSET_T>(w, tilingData_->padW, tilingData_->wOutDim, magicStrideW, shiftStrideW);

        float gradient = 0.0f;
        for (OFFSET_T pd = pdStart; pd < pdEnd; ++pd) {
            for (OFFSET_T ph = phStart; ph < phEnd; ++ph) {
                for (OFFSET_T pw = pwStart; pw < pwEnd; ++pw) {
                    DIV_T outputIdx = n * tilingData_->dOutDim * tilingData_->hOutDim * tilingData_->wOutDim * tilingData_->cDim + pd * tilingData_->hOutDim * tilingData_->wOutDim * tilingData_->cDim +
                                      ph * tilingData_->wOutDim * tilingData_->cDim + pw * tilingData_->cDim + c;
                    if (static_cast<DIV_T>(argmax[outputIdx]) == inputIdx) {
                        gradient += static_cast<float>(gradY[outputIdx]);
                    }
                }
            }
        }
        gradX[index] = static_cast<VALUE_T>(gradient);
    }
}

template <typename VALUE_T, typename INDICES_T, int64_t Format_T, bool useINT64Index, typename OFFSET_T>
__aicore__ inline void MaxPool3DGradSimtKernel<VALUE_T, INDICES_T, Format_T, useINT64Index, OFFSET_T>::ComputePos() const
{

    auto inputData = (__gm__ VALUE_T*)origx_.GetPhyAddr();
    auto outputData = (__gm__ VALUE_T*)y_.GetPhyAddr();
    auto indicesData = (__gm__ INDICES_T*)argmax_.GetPhyAddr();
    int64_t totalSize = tilingData_->nDim * tilingData_->cDim * tilingData_->dOutDim * tilingData_->hOutDim * tilingData_->wOutDim;
    if constexpr (Format_T == 0 && !useINT64Index) {
        uint32_t m_[SHAPE_DIM_THERR] = {1, 1, 1};
        uint32_t shift_[SHAPE_DIM_THERR] = {1, 1, 1};
        GetUintDivMagicAndShift(m_[DIM_0], shift_[DIM_0], static_cast<uint32_t>(tilingData_->wOutDim));
        GetUintDivMagicAndShift(m_[DIM_1], shift_[DIM_1], static_cast<uint32_t>(tilingData_->hOutDim));
        GetUintDivMagicAndShift(m_[DIM_2], shift_[DIM_2], static_cast<uint32_t>(tilingData_->dOutDim));
        Simt::VF_CALL<MaxPool3DNcdhw<VALUE_T, INDICES_T, int32_t, uint32_t>>(Simt::Dim3(THREAD_DIM),
                                                                        totalSize, inputData, 
                                                                        tilingData_->dInDim, tilingData_->hInDim, tilingData_->wInDim, 
                                                                        tilingData_->dOutDim, tilingData_->hOutDim, tilingData_->wOutDim, 
                                                                        tilingData_->kSizeD, tilingData_->kSizeH, tilingData_->kSizeW, 
                                                                        tilingData_->stridesD, tilingData_->stridesH, tilingData_->stridesW, 
                                                                        tilingData_->padD, tilingData_->padH, tilingData_->padW, 
                                                                        tilingData_->dilationD, tilingData_->dilationH, tilingData_->dilationW, 
                                                                        outputData, indicesData, blockIdx_, blockNum_,
                                                                        m_[DIM_0], shift_[DIM_0], m_[DIM_1], shift_[DIM_1], m_[DIM_2], shift_[DIM_2]);
    } else if constexpr (Format_T == 1 && !useINT64Index) {
        uint32_t m_[SHAPE_DIM_FOUR] = {1, 1, 1, 1};
        uint32_t shift_[SHAPE_DIM_FOUR] = {1, 1, 1, 1};
        GetUintDivMagicAndShift(m_[DIM_0], shift_[DIM_0], static_cast<uint32_t>(tilingData_->cDim));
        GetUintDivMagicAndShift(m_[DIM_1], shift_[DIM_1], static_cast<uint32_t>(tilingData_->wOutDim));
        GetUintDivMagicAndShift(m_[DIM_2], shift_[DIM_2], static_cast<uint32_t>(tilingData_->hOutDim));
        GetUintDivMagicAndShift(m_[DIM_3], shift_[DIM_3], static_cast<uint32_t>(tilingData_->dOutDim));
        Simt::VF_CALL<MaxPool3DNdhwc<VALUE_T, INDICES_T, int32_t, uint32_t>>(Simt::Dim3(THREAD_DIM),
                                                                        totalSize, inputData, tilingData_->cDim, 
                                                                        tilingData_->dInDim, tilingData_->hInDim, tilingData_->wInDim, 
                                                                        tilingData_->dOutDim, tilingData_->hOutDim, tilingData_->wOutDim, 
                                                                        tilingData_->kSizeD, tilingData_->kSizeH, tilingData_->kSizeW, 
                                                                        tilingData_->stridesD, tilingData_->stridesH, tilingData_->stridesW, 
                                                                        tilingData_->padD, tilingData_->padH, tilingData_->padW, 
                                                                        tilingData_->dilationD, tilingData_->dilationH, tilingData_->dilationW, 
                                                                        outputData, indicesData, blockIdx_, blockNum_, m_[DIM_0], shift_[DIM_0],
                                                                        m_[DIM_1], shift_[DIM_1], m_[DIM_2], shift_[DIM_2], m_[DIM_3], shift_[DIM_3]);
    } else if constexpr (Format_T == 0 && useINT64Index) {
        uint64_t m_[SHAPE_DIM_THERR] = {1, 1, 1};
        uint64_t shift_[SHAPE_DIM_THERR] = {1, 1, 1};
        GetUintDivMagicAndShift(m_[DIM_0], shift_[DIM_0], static_cast<uint64_t>(tilingData_->wOutDim));
        GetUintDivMagicAndShift(m_[DIM_1], shift_[DIM_1], static_cast<uint64_t>(tilingData_->hOutDim));
        GetUintDivMagicAndShift(m_[DIM_2], shift_[DIM_2], static_cast<uint64_t>(tilingData_->dOutDim));
        Simt::VF_CALL<MaxPool3DNcdhw<VALUE_T, INDICES_T, int64_t, uint64_t>>(Simt::Dim3(THREAD_DIM),
                                                                        totalSize, inputData, 
                                                                        tilingData_->dInDim, tilingData_->hInDim, tilingData_->wInDim, 
                                                                        tilingData_->dOutDim, tilingData_->hOutDim, tilingData_->wOutDim, 
                                                                        tilingData_->kSizeD, tilingData_->kSizeH, tilingData_->kSizeW, 
                                                                        tilingData_->stridesD, tilingData_->stridesH, tilingData_->stridesW, 
                                                                        tilingData_->padD, tilingData_->padH, tilingData_->padW, 
                                                                        tilingData_->dilationD, tilingData_->dilationH, tilingData_->dilationW, 
                                                                        outputData, indicesData, blockIdx_, blockNum_,
                                                                        m_[DIM_0], shift_[DIM_0], m_[DIM_1], shift_[DIM_1], m_[DIM_2], shift_[DIM_2]);
    } else if constexpr (Format_T == 1 && useINT64Index) {
        uint64_t m_[SHAPE_DIM_FOUR] = {1, 1, 1, 1};
        uint64_t shift_[SHAPE_DIM_FOUR] = {1, 1, 1, 1};
        GetUintDivMagicAndShift(m_[DIM_0], shift_[DIM_0], static_cast<uint64_t>(tilingData_->cDim));
        GetUintDivMagicAndShift(m_[DIM_1], shift_[DIM_1], static_cast<uint64_t>(tilingData_->wOutDim));
        GetUintDivMagicAndShift(m_[DIM_2], shift_[DIM_2], static_cast<uint64_t>(tilingData_->hOutDim));
        GetUintDivMagicAndShift(m_[DIM_3], shift_[DIM_3], static_cast<uint64_t>(tilingData_->dOutDim));
        Simt::VF_CALL<MaxPool3DNdhwc<VALUE_T, INDICES_T, int64_t, uint64_t>>(Simt::Dim3(THREAD_DIM),
                                                                        totalSize, inputData, tilingData_->cDim, 
                                                                        tilingData_->dInDim, tilingData_->hInDim, tilingData_->wInDim, 
                                                                        tilingData_->dOutDim, tilingData_->hOutDim, tilingData_->wOutDim, 
                                                                        tilingData_->kSizeD, tilingData_->kSizeH, tilingData_->kSizeW, 
                                                                        tilingData_->stridesD, tilingData_->stridesH, tilingData_->stridesW, 
                                                                        tilingData_->padD, tilingData_->padH, tilingData_->padW, 
                                                                        tilingData_->dilationD, tilingData_->dilationH, tilingData_->dilationW, 
                                                                        outputData, indicesData, blockIdx_, blockNum_, m_[DIM_0], shift_[DIM_0],
                                                                        m_[DIM_1], shift_[DIM_1], m_[DIM_2], shift_[DIM_2], m_[DIM_3], shift_[DIM_3]);
    }
}

template <typename VALUE_T, typename INDICES_T, int64_t Format_T, bool useINT64Index, typename OFFSET_T>
__aicore__ inline void MaxPool3DGradSimtKernel<VALUE_T, INDICES_T, Format_T, useINT64Index, OFFSET_T>::ComputeBack()
{
    LocalTensor<OFFSET_T> simtParam = paramBuf_.Get<OFFSET_T>();
    LocalTensor<int64_t> simtTilingData = simtTilingDataBuf_.Get<int64_t>();
    const int64_t *tilingPtr = reinterpret_cast<const int64_t *>(tilingData_);
    for (uint32_t i = 0; i < TILING_DATA_NUM; ++i) {
        simtTilingData.SetValue(i, tilingPtr[i]);
    }

    using DIV_T = typename std::conditional<std::is_same<OFFSET_T, int32_t>::value, uint32_t, uint64_t>::type;
    DIV_T magicC = 0, shiftC = 0;
    DIV_T magicD = 0, shiftD = 0;
    DIV_T magicH = 0, shiftH = 0;
    DIV_T magicW = 0, shiftW = 0;
    DIV_T magicStrideD = 0, shiftStrideD = 0;
    DIV_T magicStrideH = 0, shiftStrideH = 0;
    DIV_T magicStrideW = 0, shiftStrideW = 0;

    GetUintDivMagicAndShift<DIV_T>(magicC, shiftC, tilingData_->cDim);
    GetUintDivMagicAndShift<DIV_T>(magicD, shiftD, tilingData_->dInDim);
    GetUintDivMagicAndShift<DIV_T>(magicH, shiftH, tilingData_->hInDim);
    GetUintDivMagicAndShift<DIV_T>(magicW, shiftW, tilingData_->wInDim);
    GetUintDivMagicAndShift<DIV_T>(magicStrideD, shiftStrideD, tilingData_->stridesD);
    GetUintDivMagicAndShift<DIV_T>(magicStrideH, shiftStrideH, tilingData_->stridesH);
    GetUintDivMagicAndShift<DIV_T>(magicStrideW, shiftStrideW, tilingData_->stridesW);

    simtParam.SetValue(MAGIC_C_IDX, static_cast<OFFSET_T>(magicC));
    simtParam.SetValue(MAGIC_C_IDX + 1, static_cast<OFFSET_T>(shiftC));
    simtParam.SetValue(MAGIC_D_IDX, static_cast<OFFSET_T>(magicD));
    simtParam.SetValue(MAGIC_D_IDX + 1, static_cast<OFFSET_T>(shiftD));
    simtParam.SetValue(MAGIC_H_IDX, static_cast<OFFSET_T>(magicH));
    simtParam.SetValue(MAGIC_H_IDX + 1, static_cast<OFFSET_T>(shiftH));
    simtParam.SetValue(MAGIC_W_IDX, static_cast<OFFSET_T>(magicW));
    simtParam.SetValue(MAGIC_W_IDX + 1, static_cast<OFFSET_T>(shiftW));
    simtParam.SetValue(MAGIC_STRIDE_D_IDX, static_cast<OFFSET_T>(magicStrideD));
    simtParam.SetValue(MAGIC_STRIDE_D_IDX + 1, static_cast<OFFSET_T>(shiftStrideD));
    simtParam.SetValue(MAGIC_STRIDE_H_IDX, static_cast<OFFSET_T>(magicStrideH));
    simtParam.SetValue(MAGIC_STRIDE_H_IDX + 1, static_cast<OFFSET_T>(shiftStrideH));
    simtParam.SetValue(MAGIC_STRIDE_W_IDX, static_cast<OFFSET_T>(magicStrideW));
    simtParam.SetValue(MAGIC_STRIDE_W_IDX + 1, static_cast<OFFSET_T>(shiftStrideW));
    DataSyncBarrier<MemDsbT::UB>();
    auto gradData = (__gm__ VALUE_T *)grad_.GetPhyAddr();
    auto outputData = (__gm__ VALUE_T *)y_.GetPhyAddr();
    auto indicesData = (__gm__ INDICES_T *)argmax_.GetPhyAddr();

    if constexpr (Format_T == 1) {
        Simt::VF_CALL<MaxPool3DGradWithArgmaxNdhwc<VALUE_T, INDICES_T, OFFSET_T>>(
            Simt::Dim3(THREAD_DIM), (__ubuf__ OFFSET_T *)simtParam.GetPhyAddr(), gradData, (__ubuf__ Pool3DGradNameSpace::MaxPool3DGradSimtTilingData*)(simtTilingData.GetPhyAddr()), outputData, indicesData);
    } else if (Format_T == 0) {
        Simt::VF_CALL<MaxPool3DGradWithArgmaxNcdhw<VALUE_T, INDICES_T, OFFSET_T>>(
            Simt::Dim3(THREAD_DIM), (__ubuf__ OFFSET_T *)simtParam.GetPhyAddr(), gradData, (__ubuf__ Pool3DGradNameSpace::MaxPool3DGradSimtTilingData*)(simtTilingData.GetPhyAddr()), outputData, indicesData);
    }
}

}

#endif
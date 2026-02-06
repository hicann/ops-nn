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
 * \file max_pool3d_with_argmax_v2_simt.h
 * \brief max_pool3d_with_argmax_v2 implied by simt
 */

#ifndef MAX_POOL3D_WITH_ARGMAX_V2_SIMT_H
#define MAX_POOL3D_WITH_ARGMAX_V2_SIMT_H

#include "kernel_operator.h"
#include "../inc/load_store_utils.h"
#include "../inc/platform.h"
#include "../inc/kernel_utils.h"
#include "max_pool3d_with_argmax_v2_tiling_struct.h"

#ifdef __CCE_KT_TEST__
#define LAUNCH_BOUND(threads)
#endif

namespace MaxPool3DWithArgmaxV2WithSimt{
    using namespace AscendC;

    constexpr uint32_t THREAD_DIM = 256; 
    constexpr uint32_t DIM_0 = 0;
    constexpr uint32_t DIM_1 = 1;
    constexpr uint32_t DIM_2 = 2;
    constexpr uint32_t DIM_3 = 3;

    template <typename VALUE_T, typename PROCESS_T>
    __aicore__ inline static void CycleUpdate(VALUE_T val, PROCESS_T idxOffset, VALUE_T *maxVal, PROCESS_T *maxIdx)
    {
        if ((static_cast<VALUE_T>(val) > *maxVal) || Simt::IsNan(static_cast<float>(val))) {
            *maxIdx = idxOffset;
            *maxVal = val;
        }
    }

template <typename VALUE_T, typename INDICES_T, int64_t Format_T, bool useINT64Index>
class MaxPool3DWithArgmaxV2Simt
{
public:
    __aicore__ inline MaxPool3DWithArgmaxV2Simt(const MaxPool3DWithArgmaxV2Tiling::MaxPool3DWithArgmaxV2SimtTilingData* __restrict tilingData) 
        : tilingData_(tilingData), blockIdx_(GetBlockIdx()), blockNum_(GetBlockNum())
    {
    }

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR indices);
    __aicore__ inline void Process();
    __aicore__ inline void Compute() const;

private:
    AscendC::GlobalTensor<VALUE_T> x_;
    AscendC::GlobalTensor<VALUE_T> y_;
    AscendC::GlobalTensor<INDICES_T> indices_;
    const MaxPool3DWithArgmaxV2Tiling::MaxPool3DWithArgmaxV2SimtTilingData* tilingData_;
    uint32_t blockIdx_ = 0;
    uint32_t blockNum_ = 1;    
};

template <typename VALUE_T, typename INDICES_T, int64_t Format_T, bool useINT64Index>
__aicore__ inline void MaxPool3DWithArgmaxV2Simt<VALUE_T, INDICES_T, Format_T, useINT64Index>::Init(GM_ADDR x, GM_ADDR y, GM_ADDR indices)
{
    x_.SetGlobalBuffer((__gm__ VALUE_T*)(x));
    y_.SetGlobalBuffer((__gm__ VALUE_T*)(y));
    indices_.SetGlobalBuffer((__gm__ INDICES_T*)(indices));
}

template <typename VALUE_T, typename INDICES_T, int64_t Format_T, bool useINT64Index>
__aicore__ inline void MaxPool3DWithArgmaxV2Simt<VALUE_T, INDICES_T, Format_T, useINT64Index>::Process()
{
    Compute();
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
        topData[index] = static_cast<VAL_T>(maxVal);
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
        topData[index] = static_cast<VAL_T>(maxVal);
        topMask[index] = static_cast<IDX_T>(maxIdx);
    }
}

template <typename VALUE_T, typename INDICES_T, int64_t Format_T, bool useINT64Index>
__aicore__ inline void MaxPool3DWithArgmaxV2Simt<VALUE_T, INDICES_T, Format_T, useINT64Index>::Compute() const
{
    const int32_t kD = tilingData_->kSizeD;
    const int32_t kH = tilingData_->kSizeH;
    const int32_t kW = tilingData_->kSizeW;

    const int32_t dD = tilingData_->stridesD;
    const int32_t dH = tilingData_->stridesH;
    const int32_t dW =  tilingData_->stridesW;

    const int32_t padD = tilingData_->padD;
    const int32_t padH = tilingData_->padH;
    const int32_t padW = tilingData_->padW;

    const int32_t dilationD = tilingData_->dilationD;
    const int32_t dilationH = tilingData_->dilationH;
    const int32_t dilationW = tilingData_->dilationW;

    const int64_t nbatch = tilingData_->nDim;
    const int64_t inputChannel = tilingData_->cDim;
    const int64_t inputDi = tilingData_->dInDim;
    const int64_t inputHeight = tilingData_->hInDim;
    const int64_t inputWidth = tilingData_->wInDim;

    const int64_t outputDi = tilingData_->dOutDim;
    const int64_t outputHeight = tilingData_->hOutDim;
    const int64_t outputWidth = tilingData_->wOutDim;

    auto inputData = (__gm__ VALUE_T*)x_.GetPhyAddr();
    auto outputData = (__gm__ VALUE_T*)y_.GetPhyAddr();
    auto indicesData = (__gm__ INDICES_T*)indices_.GetPhyAddr();
    int64_t totalSize = nbatch * inputChannel * outputDi * outputHeight * outputWidth;
    if constexpr (Format_T == 0 && !useINT64Index) {
        uint32_t m_[3] = {1, 1, 1};
        uint32_t shift_[3] = {1, 1, 1};
        GetUintDivMagicAndShift(m_[DIM_0], shift_[DIM_0], static_cast<uint32_t>(outputWidth));
        GetUintDivMagicAndShift(m_[DIM_1], shift_[DIM_1], static_cast<uint32_t>(outputHeight));
        GetUintDivMagicAndShift(m_[DIM_2], shift_[DIM_2], static_cast<uint32_t>(outputDi));
        Simt::VF_CALL<MaxPool3DNcdhw<VALUE_T, INDICES_T, int32_t, uint32_t>>(Simt::Dim3(THREAD_DIM),
                                                                        totalSize, inputData, inputDi, inputHeight, inputWidth, outputDi, outputHeight,
                                                                        outputWidth, kD, kH, kW, dD, dH, dW, padD, padH, padW, dilationD, 
                                                                        dilationH, dilationW, outputData, indicesData, blockIdx_, blockNum_,
                                                                        m_[DIM_0], shift_[DIM_0], m_[DIM_1], shift_[DIM_1], m_[DIM_2], shift_[DIM_2]);
    } else if constexpr (Format_T == 1 && !useINT64Index) {
        uint32_t m_[4] = {1, 1, 1, 1};
        uint32_t shift_[4] = {1, 1, 1, 1};
        GetUintDivMagicAndShift(m_[DIM_0], shift_[DIM_0], static_cast<uint32_t>(inputChannel));
        GetUintDivMagicAndShift(m_[DIM_1], shift_[DIM_1], static_cast<uint32_t>(outputWidth));
        GetUintDivMagicAndShift(m_[DIM_2], shift_[DIM_2], static_cast<uint32_t>(outputHeight));
        GetUintDivMagicAndShift(m_[DIM_3], shift_[DIM_3], static_cast<uint32_t>(outputDi));
        Simt::VF_CALL<MaxPool3DNdhwc<VALUE_T, INDICES_T, int32_t, uint32_t>>(Simt::Dim3(THREAD_DIM),
                                                                        totalSize, inputData, inputChannel, inputDi, inputHeight, inputWidth, outputDi, outputHeight,
                                                                        outputWidth, kD, kH, kW, dD, dH, dW, padD, padH, padW, dilationD, 
                                                                        dilationH, dilationW, outputData, indicesData, blockIdx_, blockNum_, m_[DIM_0], shift_[DIM_0],
                                                                        m_[DIM_1], shift_[DIM_1], m_[DIM_2], shift_[DIM_2], m_[DIM_3], shift_[DIM_3]);
    } else if constexpr (Format_T == 0 && useINT64Index) {
        uint64_t m_[3] = {1, 1, 1};
        uint64_t shift_[3] = {1, 1, 1};
        GetUintDivMagicAndShift(m_[DIM_0], shift_[DIM_0], static_cast<uint64_t>(outputWidth));
        GetUintDivMagicAndShift(m_[DIM_1], shift_[DIM_1], static_cast<uint64_t>(outputHeight));
        GetUintDivMagicAndShift(m_[DIM_2], shift_[DIM_2], static_cast<uint64_t>(outputDi));
        Simt::VF_CALL<MaxPool3DNcdhw<VALUE_T, INDICES_T, int64_t, uint64_t>>(Simt::Dim3(THREAD_DIM),
                                                                        totalSize, inputData, inputDi, inputHeight, inputWidth, outputDi, outputHeight,
                                                                        outputWidth, kD, kH, kW, dD, dH, dW, padD, padH, padW, dilationD, 
                                                                        dilationH, dilationW, outputData, indicesData, blockIdx_, blockNum_,
                                                                        m_[DIM_0], shift_[DIM_0], m_[DIM_1], shift_[DIM_1], m_[DIM_2], shift_[DIM_2]);
    } else if constexpr (Format_T == 1 && useINT64Index) {
        uint64_t m_[4] = {1, 1, 1, 1};
        uint64_t shift_[4] = {1, 1, 1, 1};
        GetUintDivMagicAndShift(m_[DIM_0], shift_[DIM_0], static_cast<uint64_t>(inputChannel));
        GetUintDivMagicAndShift(m_[DIM_1], shift_[DIM_1], static_cast<uint64_t>(outputWidth));
        GetUintDivMagicAndShift(m_[DIM_2], shift_[DIM_2], static_cast<uint64_t>(outputHeight));
        GetUintDivMagicAndShift(m_[DIM_3], shift_[DIM_3], static_cast<uint64_t>(outputDi));
        Simt::VF_CALL<MaxPool3DNdhwc<VALUE_T, INDICES_T, int64_t, uint64_t>>(Simt::Dim3(THREAD_DIM),
                                                                        totalSize, inputData, inputChannel, inputDi, inputHeight, inputWidth, outputDi, outputHeight,
                                                                        outputWidth, kD, kH, kW, dD, dH, dW, padD, padH, padW, dilationD, 
                                                                        dilationH, dilationW, outputData, indicesData, blockIdx_, blockNum_, m_[DIM_0], shift_[DIM_0],
                                                                        m_[DIM_1], shift_[DIM_1], m_[DIM_2], shift_[DIM_2], m_[DIM_3], shift_[DIM_3]);
    }
}
}

#endif //MAX_POOL3D_WITH_ARGMAX_V2_SIMT_H
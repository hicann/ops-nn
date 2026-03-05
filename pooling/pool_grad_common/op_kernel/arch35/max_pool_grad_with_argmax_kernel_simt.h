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
 * \file max_pool_grad_with_argmax_simt.h
 * \brief
 */

#ifndef MAX_POOL_GRAD_WITH_ARGMAX_SIMT_H
#define MAX_POOL_GRAD_WITH_ARGMAX_SIMT_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#ifdef __CCE_KT_TEST__
#define LAUNCH_BOUND(threads)
#endif

using namespace AscendC;

constexpr size_t PARAM_NUM = 8;
constexpr size_t TILING_DATA_NUM = 16;      
constexpr size_t TILING_DATA_UB_NUM = 32;

#ifndef NCHW_FORMAT
constexpr int NCHW_FORMAT = 0;
#endif
#ifndef NHWC_FORMAT
constexpr int NHWC_FORMAT = 1;
#endif

namespace SimtProc {
    constexpr static uint32_t THREAD_DIM = 1024;

    template<typename FORMAT_TYPE>
    __aicore__ inline static FORMAT_TYPE PStart(int64_t size, int64_t pad, int64_t kernel, FORMAT_TYPE magicStride, FORMAT_TYPE shiftStride) {
        if (size + pad < kernel) {
            return 0;
        } else {
            using DIV_T = typename std::conditional<std::is_same<FORMAT_TYPE, int32_t>::value, uint32_t, uint64_t>::type;
            FORMAT_TYPE phStart = size + pad - kernel;
            phStart = Simt::UintDiv<DIV_T>(phStart, magicStride, shiftStride);
            phStart += 1;
            return phStart;
        }
    }

    template<typename FORMAT_TYPE>
    __aicore__ inline static FORMAT_TYPE PEnd(int64_t size, int64_t pad, int64_t poolSize, FORMAT_TYPE magicStride, FORMAT_TYPE shiftStride) {
        using DIV_T = typename std::conditional<std::is_same<FORMAT_TYPE, int32_t>::value, uint32_t, uint64_t>::type;
        FORMAT_TYPE pEnd = size + pad;
        pEnd = Simt::UintDiv<DIV_T>(pEnd, magicStride, shiftStride);
        pEnd += 1;
        return (pEnd > poolSize) ? static_cast<FORMAT_TYPE>(poolSize) : pEnd;
    }
}

template <typename VALUE_T, typename INDICES_T, typename FORMAT_TYPE>
__simt_vf__ __aicore__ LAUNCH_BOUND(SimtProc::THREAD_DIM) inline void MaxPoolGradWithArgmaxNchw(
    __ubuf__ FORMAT_TYPE* SimtParam,
    const __gm__ VALUE_T* bottomData,
    const int64_t nD, const int64_t cD,
    const int hD, const int wD,
    const int hOutDim, const int wOutDim,
    const int kernelH, const int kernelW,
    const int strideH, const int strideW,
    const int padH, const int padW,
    __gm__ VALUE_T* topData,
    __gm__ INDICES_T* topMask);

template <typename VALUE_T, typename INDICES_T, typename FORMAT_TYPE>
__simt_vf__ __aicore__ LAUNCH_BOUND(SimtProc::THREAD_DIM) inline void MaxPoolGradWithArgmaxNhwc(
    __ubuf__ FORMAT_TYPE* SimtParam,
    const __gm__ VALUE_T* bottomData,
    const int64_t nD, const int64_t cD,
    const int hDim, const int wDim,
    const int hOutDim, const int wOutDim,
    const int kernelH, const int kernelW,
    const int strideH, const int strideW,
    const int padH, const int padW,
    __gm__ VALUE_T* topData,
    __gm__ INDICES_T* topMask);

template <typename VALUE_T, typename INDICES_T, int Format_T, typename FORMAT_TYPE>
class MaxPoolGradWithArgmaxSimt {
public:
    __aicore__ inline MaxPoolGradWithArgmaxSimt(TPipe *pipe, const MaxPoolGradWithArgmaxNHWCNameSpace::MaxPoolGradWithArgmaxSimtTilingCommonData* __restrict tilingData)
        : pipe_(pipe), tilingData_(tilingData) {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR grad, GM_ADDR argmax, GM_ADDR y);
    __aicore__ inline void Process();

private:
    TPipe *pipe_;
    AscendC::GlobalTensor<VALUE_T> x_;          
    AscendC::GlobalTensor<VALUE_T> grad_;       // input: grad_output
    AscendC::GlobalTensor<INDICES_T> argmax_;   // input: argmax
    AscendC::GlobalTensor<VALUE_T> y_;          // output: grad_input
    const MaxPoolGradWithArgmaxNHWCNameSpace::MaxPoolGradWithArgmaxSimtTilingCommonData* tilingData_;
    TBuf<TPosition::VECCALC> simtTilingDataBuf_;
    TBuf<TPosition::VECCALC> paramBuf_;
};

template <typename VALUE_T, typename INDICES_T, int Format_T, typename FORMAT_TYPE>
__aicore__ inline void MaxPoolGradWithArgmaxSimt<VALUE_T, INDICES_T, Format_T, FORMAT_TYPE>::Init(
    GM_ADDR x, GM_ADDR grad, GM_ADDR argmax, GM_ADDR y) {
    x_.SetGlobalBuffer((__gm__ VALUE_T*)(x));
    grad_.SetGlobalBuffer((__gm__ VALUE_T*)(grad));
    argmax_.SetGlobalBuffer((__gm__ INDICES_T*)(argmax));
    y_.SetGlobalBuffer((__gm__ VALUE_T*)(y));
    pipe_->InitBuffer(simtTilingDataBuf_, TILING_DATA_UB_NUM * sizeof(int64_t));
    pipe_->InitBuffer(paramBuf_, PARAM_NUM * sizeof(FORMAT_TYPE));
}

template <typename VALUE_T, typename INDICES_T, int Format_T, typename FORMAT_TYPE>
__aicore__ inline void MaxPoolGradWithArgmaxSimt<VALUE_T, INDICES_T, Format_T, FORMAT_TYPE>::Process() {
    LocalTensor<int64_t> SimtTilingData = simtTilingDataBuf_.Get<int64_t>();
    LocalTensor<FORMAT_TYPE> SimtParam = paramBuf_.Get<FORMAT_TYPE>();
    const int64_t* tilingP = reinterpret_cast<const int64_t*>(tilingData_);
    for (uint32_t i = 0; i < TILING_DATA_NUM; ++i) {
        SimtTilingData.SetValue(i, tilingP[i]);
    }

    using DIV_T = typename std::conditional<std::is_same<FORMAT_TYPE, int32_t>::value, uint32_t, uint64_t>::type;
    DIV_T magicC = 0, shiftC = 0;
    DIV_T magicH = 0, shiftH = 0;
    DIV_T magicW = 0, shiftW = 0;
    DIV_T magicStrideH = 0, shiftStrideH = 0;
    DIV_T magicStrideW = 0, shiftStrideW = 0;

    GetUintDivMagicAndShift<DIV_T>(magicC, shiftC, SimtTilingData(1)); 
    GetUintDivMagicAndShift<DIV_T>(magicH, shiftH, SimtTilingData(2)); 
    GetUintDivMagicAndShift<DIV_T>(magicW, shiftW, SimtTilingData(3)); 
    GetUintDivMagicAndShift<DIV_T>(magicStrideH, shiftStrideH, SimtTilingData(8)); 
    GetUintDivMagicAndShift<DIV_T>(magicStrideW, shiftStrideW, SimtTilingData(9));

    SimtParam.SetValue(0, static_cast<FORMAT_TYPE>(magicH));
    SimtParam.SetValue(1, static_cast<FORMAT_TYPE>(shiftH));
    SimtParam.SetValue(2, static_cast<FORMAT_TYPE>(magicW));
    SimtParam.SetValue(3, static_cast<FORMAT_TYPE>(shiftW));
    SimtParam.SetValue(4, static_cast<FORMAT_TYPE>(magicStrideH));
    SimtParam.SetValue(5, static_cast<FORMAT_TYPE>(shiftStrideH));
    SimtParam.SetValue(6, static_cast<FORMAT_TYPE>(magicStrideW));
    SimtParam.SetValue(7, static_cast<FORMAT_TYPE>(shiftStrideW));
    SimtParam.SetValue(8, static_cast<FORMAT_TYPE>(magicC));
    SimtParam.SetValue(9, static_cast<FORMAT_TYPE>(shiftC));
    DataSyncBarrier<MemDsbT::UB>();
    auto gradData = (__gm__ VALUE_T*)grad_.GetPhyAddr();
    auto outputData = (__gm__ VALUE_T*)y_.GetPhyAddr();
    auto indicesData = (__gm__ INDICES_T*)argmax_.GetPhyAddr();

    if constexpr (Format_T == NCHW_FORMAT) {
        Simt::VF_CALL<MaxPoolGradWithArgmaxNchw<VALUE_T, INDICES_T, FORMAT_TYPE>>(
            Simt::Dim3(SimtProc::THREAD_DIM),
            (__ubuf__ FORMAT_TYPE*)SimtParam.GetPhyAddr(), gradData, SimtTilingData(0), SimtTilingData(1), SimtTilingData(2), SimtTilingData(3),  
            SimtTilingData(4), SimtTilingData(5), SimtTilingData(6), SimtTilingData(7), SimtTilingData(8), SimtTilingData(9), SimtTilingData(10), 
            SimtTilingData(11), outputData, indicesData);
    } else if constexpr (Format_T == NHWC_FORMAT) {
        Simt::VF_CALL<MaxPoolGradWithArgmaxNhwc<VALUE_T, INDICES_T, FORMAT_TYPE>>(
            Simt::Dim3(SimtProc::THREAD_DIM),
            (__ubuf__ FORMAT_TYPE*)SimtParam.GetPhyAddr(), gradData, SimtTilingData(0), SimtTilingData(1), SimtTilingData(2), SimtTilingData(3),  
            SimtTilingData(4), SimtTilingData(5), SimtTilingData(6), SimtTilingData(7), SimtTilingData(8), SimtTilingData(9), SimtTilingData(10), 
            SimtTilingData(11), outputData, indicesData);
    }
}

template <typename VALUE_T, typename INDICES_T, typename FORMAT_TYPE>
__simt_vf__ __aicore__ LAUNCH_BOUND(SimtProc::THREAD_DIM) inline void MaxPoolGradWithArgmaxNchw(
    __ubuf__ FORMAT_TYPE* SimtParam,
    const __gm__ VALUE_T* bottomData,   
    const int64_t nDims, const int64_t cDims,
    const int hDims, const int wDims,
    const int hOutDim, const int wOutDim,
    const int kernelH, const int kernelW,
    const int strideH, const int strideW,
    const int padH, const int padW,
    __gm__ VALUE_T* topData,           
    __gm__ INDICES_T* topMask) {       

    FORMAT_TYPE magicH       = SimtParam[0];
    FORMAT_TYPE shiftH       = SimtParam[1];
    FORMAT_TYPE magicW       = SimtParam[2];
    FORMAT_TYPE shiftW       = SimtParam[3];
    FORMAT_TYPE magicStrideH = SimtParam[4];
    FORMAT_TYPE shiftStrideH = SimtParam[5];
    FORMAT_TYPE magicStrideW = SimtParam[6];
    FORMAT_TYPE shiftStrideW = SimtParam[7];
    FORMAT_TYPE magicC       = SimtParam[8];
    FORMAT_TYPE shiftC       = SimtParam[9];
    using DIV_T = typename std::conditional<std::is_same<FORMAT_TYPE, int32_t>::value, uint32_t, uint64_t>::type;
    DIV_T count = nDims * cDims * hDims * wDims;
    for (DIV_T index = Simt::GetBlockIdx() * Simt::GetThreadNum() + Simt::GetThreadIdx();
        index < count;
        index += Simt::GetBlockNum() * Simt::GetThreadNum()) {

        DIV_T temp1 = Simt::UintDiv(index, static_cast<DIV_T>(magicW), static_cast<DIV_T>(shiftW));
        DIV_T w = index - temp1 * static_cast<DIV_T>(wDims);
        DIV_T temp2 = Simt::UintDiv(temp1, static_cast<DIV_T>(magicH), static_cast<DIV_T>(shiftH));
        DIV_T h = temp1 - temp2 * static_cast<DIV_T>(hDims);
        DIV_T n = Simt::UintDiv(temp2, static_cast<DIV_T>(magicC), static_cast<DIV_T>(shiftC));
        DIV_T c = temp2 - n * static_cast<DIV_T>(cDims);

        DIV_T local_input_index = (c * hDims + h) * wDims + w;
        FORMAT_TYPE phstart = SimtProc::PStart<FORMAT_TYPE>(h, padH, kernelH, magicStrideH, shiftStrideH);
        FORMAT_TYPE phend   = SimtProc::PEnd<FORMAT_TYPE>(h, padH, hOutDim, magicStrideH, shiftStrideH);
        FORMAT_TYPE pwstart = SimtProc::PStart<FORMAT_TYPE>(w, padW, kernelW, magicStrideW, shiftStrideW);
        FORMAT_TYPE pwend   = SimtProc::PEnd<FORMAT_TYPE>(w, padW, wOutDim, magicStrideW, shiftStrideW);

        float gradient = 0.0f;
        for (FORMAT_TYPE ph = phstart; ph < phend; ++ph) {
            for (FORMAT_TYPE pw = pwstart; pw < pwend; ++pw) {
                DIV_T output_idx = ((n * cDims + c) * hOutDim + ph) * wOutDim + pw;
                if (static_cast<DIV_T>(topMask[output_idx]) == local_input_index) {
                    gradient += static_cast<float>(bottomData[output_idx]);
                }
            }
        }
        topData[index] = static_cast<VALUE_T>(gradient);
    }
}

template <typename VALUE_T, typename INDICES_T, typename FORMAT_TYPE>
__simt_vf__ __aicore__ LAUNCH_BOUND(SimtProc::THREAD_DIM) inline void MaxPoolGradWithArgmaxNhwc(
    __ubuf__ FORMAT_TYPE* SimtParam,
    const __gm__ VALUE_T* bottomData,
    const int64_t nDim, const int64_t cDim,
    const int hDim, const int wDim,
    const int hOutDim, const int wOutDim,
    const int kernelH, const int kernelW,
    const int strideH, const int strideW,
    const int padH, const int padW,
    __gm__ VALUE_T* topData,
    __gm__ INDICES_T* topMask) {

    using DIV_T = typename std::conditional<std::is_same<FORMAT_TYPE, int32_t>::value, uint32_t, uint64_t>::type;
    FORMAT_TYPE magicH = SimtParam[0];
    FORMAT_TYPE shiftH = SimtParam[1];
    FORMAT_TYPE magicW = SimtParam[2];
    FORMAT_TYPE shiftW = SimtParam[3];
    FORMAT_TYPE magicC = SimtParam[8];  
    FORMAT_TYPE shiftC = SimtParam[9];
    FORMAT_TYPE magicStrideH = SimtParam[4];
    FORMAT_TYPE shiftStrideH = SimtParam[5];
    FORMAT_TYPE magicStrideW = SimtParam[6];
    FORMAT_TYPE shiftStrideW = SimtParam[7];

    DIV_T count = nDim * hDim * cDim * wDim;
    for (DIV_T index = Simt::GetBlockIdx() * Simt::GetThreadNum() + Simt::GetThreadIdx(); index < count;
        index += Simt::GetBlockNum() * Simt::GetThreadNum()) {
                
        DIV_T spatial_idx = Simt::UintDiv(index, static_cast<DIV_T>(magicC), static_cast<DIV_T>(shiftC));
        DIV_T c = index - spatial_idx * static_cast<DIV_T>(cDim);
        DIV_T nh_idx = Simt::UintDiv(spatial_idx, static_cast<DIV_T>(magicW), static_cast<DIV_T>(shiftW));
        DIV_T w = spatial_idx - nh_idx * static_cast<DIV_T>(wDim);
        DIV_T n = Simt::UintDiv(nh_idx, static_cast<DIV_T>(magicH), static_cast<DIV_T>(shiftH));
        DIV_T h = nh_idx - n * static_cast<DIV_T>(hDim);

        DIV_T full_input_index = (h * wDim + w) *cDim + c;
        FORMAT_TYPE phstart = SimtProc::PStart<FORMAT_TYPE>(h, padH, kernelH, magicStrideH, shiftStrideH);
        FORMAT_TYPE phend = SimtProc::PEnd<FORMAT_TYPE>(h, padH, hOutDim, magicStrideH, shiftStrideH);
        FORMAT_TYPE pwstart = SimtProc::PStart<FORMAT_TYPE>(w, padW, kernelW, magicStrideW, shiftStrideW);
        FORMAT_TYPE pwend = SimtProc::PEnd<FORMAT_TYPE>(w, padW, wOutDim, magicStrideW, shiftStrideW);

        float gradient = 0.0f;
        for (FORMAT_TYPE ph = phstart; ph < phend; ++ph) {
            for (FORMAT_TYPE pw = pwstart; pw < pwend; ++pw) {
                DIV_T output_idx = ((n * hOutDim + ph) * wOutDim + pw) * cDim + c;
                if (static_cast<DIV_T>(topMask[output_idx]) == full_input_index) {
                    gradient += static_cast<float>(bottomData[output_idx]);
                }
            }
        }
        topData[index] = static_cast<VALUE_T>(gradient);
    }
}

#endif // MAX_POOL_GRAD_WITH_ARGMAX_SIMT_H
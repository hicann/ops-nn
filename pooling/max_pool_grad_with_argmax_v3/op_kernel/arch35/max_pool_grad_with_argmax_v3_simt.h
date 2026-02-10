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
 * \file max_pool_grad_with_argmax_v3_simt.h
 * \brief max_pool_grad_with_argmax_v3 implied by simt
 */

#ifndef MAX_POOL_GRAD_WITH_ARGMAX_V3_SIMT_H
#define MAX_POOL_GRAD_WITH_ARGMAX_V3_SIMT_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"

#ifdef __CCE_KT_TEST__
#define LAUNCH_BOUND(threads)
#endif

using namespace AscendC;

constexpr size_t PARAM_NUM = 8;

namespace SimtProc {
    constexpr static uint32_t THREAD_DIM = 1024;

    template<typename FORMAT_TYPE, typename DIV_T>
    __aicore__ __attribute__((always_inline)) inline static FORMAT_TYPE PStart(FORMAT_TYPE size, FORMAT_TYPE pad, FORMAT_TYPE kernel, FORMAT_TYPE dilation, DIV_T magicStride, DIV_T shiftStride) {
        if (size + pad < ((kernel -1) * dilation + 1)) {
            return 0;
        } else {
            FORMAT_TYPE phStart = size + pad - ((kernel - 1) * dilation + 1);
            phStart = Simt::UintDiv<DIV_T>(phStart, magicStride, shiftStride);
            phStart += 1;
            return phStart;
        }
    }

    template<typename FORMAT_TYPE, typename DIV_T>
     __aicore__ __attribute__((always_inline)) inline static FORMAT_TYPE PEnd(FORMAT_TYPE size, FORMAT_TYPE pad, FORMAT_TYPE poolSize, DIV_T magicStride, DIV_T shiftStride) {
        FORMAT_TYPE pEnd = size + pad;
        pEnd = Simt::UintDiv<DIV_T>(pEnd, magicStride, shiftStride);
        pEnd += 1;
        return pEnd > poolSize ? poolSize : pEnd;
    }
}

template <typename VALUE_T, typename INDICES_T, typename FORMAT_TYPE, typename DIV_T>
__simt_vf__ __aicore__ LAUNCH_BOUND(SimtProc::THREAD_DIM) inline void MaxPoolGradWithArgmaxNchw(__ubuf__ DIV_T* SimtParam,
    const __gm__ VALUE_T* bottomData, FORMAT_TYPE count, FORMAT_TYPE hDim, FORMAT_TYPE wDim,
    FORMAT_TYPE hOutDim, FORMAT_TYPE wOutDim, FORMAT_TYPE kernelH, FORMAT_TYPE kernelW, FORMAT_TYPE padH,
    FORMAT_TYPE padW, FORMAT_TYPE dilationH, FORMAT_TYPE dilationW, __gm__ VALUE_T* topData, __gm__ INDICES_T* topMask);

template <typename VALUE_T, typename INDICES_T, int Format_T, typename FORMAT_TYPE, typename DIV_T>
class MaxPoolGradWithArgmaxV3Simt
{
public:
    __aicore__ inline MaxPoolGradWithArgmaxV3Simt(TPipe *pipe, const MaxPoolGradWithArgmaxNHWCNameSpace::MaxPoolGradWithArgmaxSimtTilingCommonData* __restrict tilingData)
        : pipe_(pipe), tilingData_(tilingData)
    {
    }

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR grad, GM_ADDR argmax, GM_ADDR y);
    __aicore__ inline void Process();

private:
    TPipe *pipe_;
    AscendC::GlobalTensor<VALUE_T> x_;
    AscendC::GlobalTensor<VALUE_T> grad_;
    AscendC::GlobalTensor<INDICES_T> argmax_;
    AscendC::GlobalTensor<VALUE_T> y_;
    const MaxPoolGradWithArgmaxNHWCNameSpace::MaxPoolGradWithArgmaxSimtTilingCommonData* tilingData_;
    TBuf<TPosition::VECCALC> paramBuf_;
};

template <typename VALUE_T, typename INDICES_T, int Format_T, typename FORMAT_TYPE, typename DIV_T>
__aicore__ inline void MaxPoolGradWithArgmaxV3Simt<VALUE_T, INDICES_T, Format_T, FORMAT_TYPE, DIV_T>::Init(GM_ADDR x, GM_ADDR grad, GM_ADDR argmax, GM_ADDR y)
{
    x_.SetGlobalBuffer((__gm__ VALUE_T*)(x));
    grad_.SetGlobalBuffer((__gm__ VALUE_T*)(grad));
    argmax_.SetGlobalBuffer((__gm__ INDICES_T*)(argmax));
    y_.SetGlobalBuffer((__gm__ VALUE_T*)(y));
    pipe_->InitBuffer(paramBuf_, PARAM_NUM * sizeof(DIV_T));
}

template <typename VALUE_T, typename INDICES_T, int Format_T, typename FORMAT_TYPE, typename DIV_T>
__aicore__ inline void MaxPoolGradWithArgmaxV3Simt<VALUE_T, INDICES_T, Format_T, FORMAT_TYPE, DIV_T>::Process()
{
    LocalTensor<DIV_T> SimtParam = paramBuf_.Get<DIV_T>();
    FORMAT_TYPE count = tilingData_->hInDim * tilingData_->wInDim * tilingData_->nDim * tilingData_->cDim;
    DIV_T hW = tilingData_->hInDim * tilingData_->wInDim;
    DIV_T magicHW = 0;
    DIV_T shiftHW = 0;
    DIV_T magicW = 0;
    DIV_T shiftW = 0;
    DIV_T magicStrideH = 0;
    DIV_T shiftStrideH = 0;
    DIV_T magicStrideW = 0;
    DIV_T shiftStrideW = 0;
    GetUintDivMagicAndShift<DIV_T>(magicHW, shiftHW, hW);
    GetUintDivMagicAndShift<DIV_T>(magicW, shiftW, tilingData_->wInDim);
    GetUintDivMagicAndShift<DIV_T>(magicStrideH, shiftStrideH, tilingData_->stridesH);
    GetUintDivMagicAndShift<DIV_T>(magicStrideW, shiftStrideW, tilingData_->stridesW);

    SimtParam.SetValue(0, static_cast<DIV_T>(magicHW));
    SimtParam.SetValue(1, static_cast<DIV_T>(shiftHW));
    SimtParam.SetValue(2, static_cast<DIV_T>(magicW));
    SimtParam.SetValue(3, static_cast<DIV_T>(shiftW));
    SimtParam.SetValue(4, static_cast<DIV_T>(magicStrideH));
    SimtParam.SetValue(5, static_cast<DIV_T>(shiftStrideH));
    SimtParam.SetValue(6, static_cast<DIV_T>(magicStrideW));
    SimtParam.SetValue(7, static_cast<DIV_T>(shiftStrideW));
    
    DataSyncBarrier<MemDsbT::UB>();
    auto gradData = (__gm__ VALUE_T*)grad_.GetPhyAddr();
    auto outputData = (__gm__ VALUE_T*)y_.GetPhyAddr();
    auto indicesData = (__gm__ INDICES_T*)argmax_.GetPhyAddr();
    Simt::VF_CALL<MaxPoolGradWithArgmaxNchw<VALUE_T, INDICES_T, FORMAT_TYPE, DIV_T>>(
        Simt::Dim3(SimtProc::THREAD_DIM), (__ubuf__ DIV_T*)SimtParam.GetPhyAddr(), gradData, count,
        static_cast<FORMAT_TYPE>(tilingData_->hInDim), static_cast<FORMAT_TYPE>(tilingData_->wInDim),
        static_cast<FORMAT_TYPE>(tilingData_->hOutDim), static_cast<FORMAT_TYPE>(tilingData_->wOutDim),
        static_cast<FORMAT_TYPE>(tilingData_->kSizeH), static_cast<FORMAT_TYPE>(tilingData_->kSizeW),
        static_cast<FORMAT_TYPE>(tilingData_->padH), static_cast<FORMAT_TYPE>(tilingData_->padW),
        static_cast<FORMAT_TYPE>(tilingData_->dilationH), static_cast<FORMAT_TYPE>(tilingData_->dilationW), outputData,
        indicesData);
}

template <typename VALUE_T, typename INDICES_T, typename FORMAT_TYPE, typename DIV_T>
__simt_vf__ __aicore__ LAUNCH_BOUND(SimtProc::THREAD_DIM) inline void MaxPoolGradWithArgmaxNchw(__ubuf__ DIV_T* SimtParam,
    const __gm__ VALUE_T* bottomData, FORMAT_TYPE count, FORMAT_TYPE hDim, FORMAT_TYPE wDim,
    FORMAT_TYPE hOutDim, FORMAT_TYPE wOutDim, FORMAT_TYPE kernelH, FORMAT_TYPE kernelW, FORMAT_TYPE padH,
    FORMAT_TYPE padW, FORMAT_TYPE dilationH, FORMAT_TYPE dilationW, __gm__ VALUE_T* topData, __gm__ INDICES_T* topMask)
{   
    DIV_T magicHW = SimtParam[0];
    DIV_T shiftHW = SimtParam[1];
    DIV_T magicW = SimtParam[2];
    DIV_T shiftW = SimtParam[3];
    DIV_T magicStrideH = SimtParam[4];
    DIV_T shiftStrideH = SimtParam[5];
    DIV_T magicStrideW = SimtParam[6];
    DIV_T shiftStrideW = SimtParam[7];

    for (FORMAT_TYPE index = Simt::GetBlockIdx() * Simt::GetThreadNum() + Simt::GetThreadIdx(); index < count;
        index = index + Simt::GetBlockNum() * Simt::GetThreadNum()) {
        DIV_T nc = Simt::UintDiv<DIV_T>(index, magicHW, shiftHW);
        DIV_T tempH = index - nc * hDim * wDim;
        DIV_T h = Simt::UintDiv<DIV_T>(tempH, magicW, shiftW);
        DIV_T w = tempH - h * wDim;

        FORMAT_TYPE phstart = SimtProc::PStart<FORMAT_TYPE, DIV_T>(h, padH, kernelH, dilationH, magicStrideH, shiftStrideH);
        FORMAT_TYPE phend = SimtProc::PEnd<FORMAT_TYPE, DIV_T>(h, padH, hOutDim, magicStrideH, shiftStrideH);
        FORMAT_TYPE pwstart = SimtProc::PStart<FORMAT_TYPE, DIV_T>(w, padW, kernelW, dilationW, magicStrideW, shiftStrideW);
        FORMAT_TYPE pwend = SimtProc::PEnd<FORMAT_TYPE, DIV_T>(w, padW, wOutDim, magicStrideW, shiftStrideW);

        DIV_T offset = nc * hOutDim * wOutDim;
        float gradient = 0;
        for (FORMAT_TYPE ph = phstart; ph < phend; ++ph) {
            for (FORMAT_TYPE pw = pwstart; pw < pwend; ++pw) {
                DIV_T idx = ph * wOutDim + pw + offset;
                if (topMask[idx] == h * wDim + w) {
                    gradient += static_cast<float>(bottomData[idx]);
                }
            }
        }
        topData[index] = static_cast<VALUE_T>(gradient);
    }
}

#endif  // MAX_POOL_GRAD_WITH_ARGMAX_V3_SIMT_H

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
 * \file adaptive_avg_pool2d_grad_simt.h
 * \brief adaptive_avg_pool2d_grad implied by simt
 */

#ifndef ADAPTIVE_AVG_POOL2D_GRAD_SIMT_H
#define ADAPTIVE_AVG_POOL2D_GRAD_SIMT_H

#include "kernel_operator.h"
#include "../inc/load_store_utils.h"
#include "../inc/platform.h"
#include "../inc/kernel_utils.h"
#include "adaptive_avg_pool2d_grad_struct.h"
#include "simt_api/asc_simt.h"
using namespace AscendC;

namespace AdaptiveAvgPool2dGradOp {
constexpr static uint32_t THREAD_DIM = 1024;
constexpr static uint32_t TILING_DATA_NUM = 6;
constexpr static uint32_t SIMT_PARAMS_NUM = 32;
constexpr static uint32_t MAGIC_C_IDX = 0;
constexpr static uint32_t MAGIC_IN_H_IDX = 2;
constexpr static uint32_t MAGIC_IN_W_IDX = 4;
constexpr static uint32_t MAGIC_OSIZE_H_IDX = 6;
constexpr static uint32_t MAGIC_OSIZE_W_IDX = 8;

template <typename VALUE_T, typename OFFSET_T, int64_t CHANNEL_LAST>
class AdaptiveAvgPool2dGradSimt {
public:
    __aicore__ inline AdaptiveAvgPool2dGradSimt(
        TPipe* pipe, const AdaptiveAvgPool2dGradSimtTiling* __restrict__ tilingData)
        : pipe_(pipe), tilingData_(tilingData)
    {}
    __aicore__ inline void Init(GM_ADDR yGrad, GM_ADDR xGrad);
    __aicore__ inline void Process();

private:
    TPipe* pipe_;
    AscendC::GlobalTensor<VALUE_T> yGrad_;
    AscendC::GlobalTensor<VALUE_T> xGrad_;
    const AdaptiveAvgPool2dGradSimtTiling* tilingData_;
    TBuf<TPosition::VECCALC> paramBuf_;
};

template <typename OFFSET_T, typename DIV_T>
__simt_callee__ __aicore__ inline static OFFSET_T FloorDivMul(
    OFFSET_T numerator, OFFSET_T mulFactor, DIV_T divisorMagic, DIV_T divisorShift)
{
    DIV_T wideNumerator = static_cast<DIV_T>(numerator) * static_cast<DIV_T>(mulFactor);
    DIV_T quotient = Simt::UintDiv<DIV_T>(wideNumerator, divisorMagic, divisorShift);
    return static_cast<OFFSET_T>(quotient);
}

template <typename OFFSET_T, typename DIV_T>
__simt_callee__ __aicore__ inline static OFFSET_T CeilDivMul(
    OFFSET_T numerator, OFFSET_T mulFactor, OFFSET_T ceilAddend, DIV_T divisorMagic, DIV_T divisorShift)
{
    DIV_T wideNumerator =
        static_cast<DIV_T>(numerator) * static_cast<DIV_T>(mulFactor) + static_cast<DIV_T>(ceilAddend);
    DIV_T quotient = Simt::UintDiv<DIV_T>(wideNumerator, divisorMagic, divisorShift);
    return static_cast<OFFSET_T>(quotient);
}

template <typename OFFSET_T, typename DIV_T>
__simt_callee__ __aicore__ inline static OFFSET_T StartIndexIn2Out(
    OFFSET_T inIdx, OFFSET_T osize, DIV_T magicIsize, DIV_T shiftIsize)
{
    return FloorDivMul<OFFSET_T, DIV_T>(inIdx, osize, magicIsize, shiftIsize);
}

template <typename OFFSET_T, typename DIV_T>
__simt_callee__ __aicore__ inline static OFFSET_T EndIndexIn2Out(
    OFFSET_T inIdx, OFFSET_T isize, OFFSET_T osize, DIV_T magicIsize, DIV_T shiftIsize)
{
    return CeilDivMul<OFFSET_T, DIV_T>(inIdx + 1, osize, isize - 1, magicIsize, shiftIsize);
}

template <typename OFFSET_T, typename DIV_T>
__simt_callee__ __aicore__ inline static OFFSET_T StartIndexOut2In(
    OFFSET_T outIdx, OFFSET_T isize, DIV_T magicOsize, DIV_T shiftOsize)
{
    return FloorDivMul<OFFSET_T, DIV_T>(outIdx, isize, magicOsize, shiftOsize);
}

template <typename OFFSET_T, typename DIV_T>
__simt_callee__ __aicore__ inline static OFFSET_T EndIndexOut2In(
    OFFSET_T outIdx, OFFSET_T osize, OFFSET_T isize, DIV_T magicOsize, DIV_T shiftOsize)
{
    return CeilDivMul<OFFSET_T, DIV_T>(outIdx + 1, isize, osize - 1, magicOsize, shiftOsize);
}

template <typename VALUE_T, typename OFFSET_T>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_DIM) inline void AdaptiveAvgPool2dGradNchw(
    __ubuf__ OFFSET_T* simtParams, const __gm__ VALUE_T* gradY, const OFFSET_T nDims, const OFFSET_T cDims,
    const OFFSET_T inH, const OFFSET_T inW, const OFFSET_T outH, const OFFSET_T outW, __gm__ VALUE_T* gradX)
{
    using DIV_T = typename std::conditional<std::is_same<OFFSET_T, int32_t>::value, uint32_t, uint64_t>::type;
    DIV_T magicC = simtParams[MAGIC_C_IDX];
    DIV_T shiftC = simtParams[MAGIC_C_IDX + 1];
    DIV_T magicInH = simtParams[MAGIC_IN_H_IDX];
    DIV_T shiftInH = simtParams[MAGIC_IN_H_IDX + 1];
    DIV_T magicInW = simtParams[MAGIC_IN_W_IDX];
    DIV_T shiftInW = simtParams[MAGIC_IN_W_IDX + 1];
    DIV_T magicOsizeH = simtParams[MAGIC_OSIZE_H_IDX];
    DIV_T shiftOsizeH = simtParams[MAGIC_OSIZE_H_IDX + 1];
    DIV_T magicOsizeW = simtParams[MAGIC_OSIZE_W_IDX];
    DIV_T shiftOsizeW = simtParams[MAGIC_OSIZE_W_IDX + 1];

    DIV_T count = nDims * cDims * inH * inW;
    for (DIV_T index = blockIdx.x * blockDim.x + threadIdx.x; index < count;
         index += gridDim.x * blockDim.x) {
        DIV_T temp1 = Simt::UintDiv(index, magicInW, shiftInW);
        DIV_T w = index - temp1 * static_cast<DIV_T>(inW);
        DIV_T temp2 = Simt::UintDiv(temp1, magicInH, shiftInH);
        DIV_T h = temp1 - temp2 * static_cast<DIV_T>(inH);
        DIV_T n = Simt::UintDiv(temp2, magicC, shiftC);
        DIV_T c = temp2 - n * static_cast<DIV_T>(cDims);

        OFFSET_T ohStarts = StartIndexIn2Out<OFFSET_T, DIV_T>(h, outH, magicInH, shiftInH);
        OFFSET_T ohEnds = EndIndexIn2Out<OFFSET_T, DIV_T>(h, inH, outH, magicInH, shiftInH);
        OFFSET_T owStarts = StartIndexIn2Out<OFFSET_T, DIV_T>(w, outW, magicInW, shiftInW);
        OFFSET_T owEnds = EndIndexIn2Out<OFFSET_T, DIV_T>(w, inW, outW, magicInW, shiftInW);
        // 遍历所有可能覆盖这个input点的输出窗口
        float gradient = 0.0f;

        for (OFFSET_T oh = ohStarts; oh < ohEnds; ++oh) {
            OFFSET_T ih0 = StartIndexOut2In<OFFSET_T, DIV_T>(oh, inH, magicOsizeH, shiftOsizeH);
            OFFSET_T ih1 = EndIndexOut2In<OFFSET_T, DIV_T>(oh, outH, inH, magicOsizeH, shiftOsizeH);
            OFFSET_T kH = ih1 - ih0;
            for (OFFSET_T ow = owStarts; ow < owEnds; ++ow) {
                OFFSET_T iw0 = StartIndexOut2In<OFFSET_T, DIV_T>(ow, inW, magicOsizeW, shiftOsizeW);
                OFFSET_T iw1 = EndIndexOut2In<OFFSET_T, DIV_T>(ow, outW, inW, magicOsizeW, shiftOsizeW);
                OFFSET_T kW = iw1 - iw0;
                OFFSET_T div = kH * kW;
                DIV_T outputIdx = n * cDims * outH * outW + c * outH * outW + oh * outW + ow;
                gradient += static_cast<float>(gradY[outputIdx]) / static_cast<float>(div);
            }
        }
        gradX[index] = static_cast<VALUE_T>(gradient);
    }
}

template <typename VALUE_T, typename OFFSET_T, int64_t CHANNEL_LAST>
__aicore__ inline void AdaptiveAvgPool2dGradSimt<VALUE_T, OFFSET_T, CHANNEL_LAST>::Init(GM_ADDR yGrad, GM_ADDR xGrad)
{
    yGrad_.SetGlobalBuffer((__gm__ VALUE_T*)(yGrad));
    xGrad_.SetGlobalBuffer((__gm__ VALUE_T*)(xGrad));
    pipe_->InitBuffer(paramBuf_, SIMT_PARAMS_NUM * sizeof(OFFSET_T));
}

template <typename VALUE_T, typename OFFSET_T, int64_t CHANNEL_LAST>
__aicore__ inline void AdaptiveAvgPool2dGradSimt<VALUE_T, OFFSET_T, CHANNEL_LAST>::Process()
{
    using DIV_T = typename std::conditional<std::is_same<OFFSET_T, int32_t>::value, uint32_t, uint64_t>::type;
    LocalTensor<DIV_T> simtParam = paramBuf_.Get<DIV_T>();
    DIV_T magicC = 0;
    DIV_T shiftC = 0;
    DIV_T magicInH = 0;
    DIV_T shiftInH = 0;
    DIV_T magicInW = 0;
    DIV_T shiftInW = 0;
    DIV_T magicOsizeH = 0;
    DIV_T shiftOsizeH = 0;
    DIV_T magicOsizeW = 0;
    DIV_T shiftOsizeW = 0;

    GetUintDivMagicAndShift<DIV_T>(magicC, shiftC, static_cast<DIV_T>(tilingData_->cDim));
    GetUintDivMagicAndShift<DIV_T>(magicInH, shiftInH, static_cast<DIV_T>(tilingData_->hInDim));
    GetUintDivMagicAndShift<DIV_T>(magicInW, shiftInW, static_cast<DIV_T>(tilingData_->wInDim));
    GetUintDivMagicAndShift<DIV_T>(magicOsizeH, shiftOsizeH, static_cast<DIV_T>(tilingData_->hOutDim));
    GetUintDivMagicAndShift<DIV_T>(magicOsizeW, shiftOsizeW, static_cast<DIV_T>(tilingData_->wOutDim));

    simtParam.SetValue(MAGIC_C_IDX, magicC);
    simtParam.SetValue(MAGIC_C_IDX + 1, shiftC);
    simtParam.SetValue(MAGIC_IN_H_IDX, magicInH);
    simtParam.SetValue(MAGIC_IN_H_IDX + 1, shiftInH);
    simtParam.SetValue(MAGIC_IN_W_IDX, magicInW);
    simtParam.SetValue(MAGIC_IN_W_IDX + 1, shiftInW);
    simtParam.SetValue(MAGIC_OSIZE_H_IDX, magicOsizeH);
    simtParam.SetValue(MAGIC_OSIZE_H_IDX + 1, shiftOsizeH);
    simtParam.SetValue(MAGIC_OSIZE_W_IDX, magicOsizeW);
    simtParam.SetValue(MAGIC_OSIZE_W_IDX + 1, shiftOsizeW);

    DataSyncBarrier<MemDsbT::UB>();

    auto gradData = (__gm__ VALUE_T*)yGrad_.GetPhyAddr();
    auto outputData = (__gm__ VALUE_T*)xGrad_.GetPhyAddr();

    asc_vf_call<AdaptiveAvgPool2dGradNchw<VALUE_T, OFFSET_T>>(
        dim3(THREAD_DIM), (__ubuf__ OFFSET_T*)simtParam.GetPhyAddr(), gradData,
        static_cast<OFFSET_T>(tilingData_->nDim), static_cast<OFFSET_T>(tilingData_->cDim),
        static_cast<OFFSET_T>(tilingData_->hInDim), static_cast<OFFSET_T>(tilingData_->wInDim),
        static_cast<OFFSET_T>(tilingData_->hOutDim), static_cast<OFFSET_T>(tilingData_->wOutDim), outputData);
}

} // namespace AdaptiveAvgPool2dGradOp

#endif // ADAPTIVE_AVG_POOL2D_GRAD_SIMT_H
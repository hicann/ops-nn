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
 * \file max_pool3d_grad_with_argmax_simt.h
 * \brief max_pool3d_grad_with_argmax implied by simt
 */

#ifndef MAX_POOL3D_GRAD_WITH_ARGMAX_SIMT_H
#define MAX_POOL3D_GRAD_WITH_ARGMAX_SIMT_H

#include "kernel_operator.h"
#include "../inc/load_store_utils.h"
#include "../inc/platform.h"
#include "../inc/kernel_utils.h"
#include "max_pool3d_grad_with_argmax_struct.h"

using namespace AscendC;

namespace MaxPool3DGradWithArgmaxOp {
constexpr static uint32_t THREAD_DIM = 1024;
constexpr static uint32_t TILING_DATA_NUM = 20;
constexpr static uint32_t SIMT_PARAMS_NUM = 32;
constexpr static uint32_t MAGIC_C_IDX = 0;
constexpr static uint32_t MAGIC_D_IDX = 2;
constexpr static uint32_t MAGIC_H_IDX = 4;
constexpr static uint32_t MAGIC_W_IDX = 6;
constexpr static uint32_t MAGIC_STRIDE_D_IDX = 8;
constexpr static uint32_t MAGIC_STRIDE_H_IDX = 10;
constexpr static uint32_t MAGIC_STRIDE_W_IDX = 12;

template <typename VALUE_T, typename INDICES_T, typename OFFSET_T, int64_t CHANNEL_LAST>
class MaxPool3DGradWithArgmaxSimt {
public:
    __aicore__ inline MaxPool3DGradWithArgmaxSimt(TPipe *pipe,
                                                  const MaxPool3DGradWithArgmaxTilingDataV35 *__restrict__ tilingData)
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
    const MaxPool3DGradWithArgmaxTilingDataV35 *tilingData_;
    TBuf<TPosition::VECCALC> simtTilingDataBuf_;
    TBuf<TPosition::VECCALC> paramBuf_;
};

template <typename OFFSET_T>
__aicore__ inline static OFFSET_T PStart(int64_t size, int64_t pad, int64_t kernel, int64_t dilation,
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
__aicore__ inline static OFFSET_T PEnd(int64_t size, int64_t pad, int64_t poolSize, OFFSET_T magicStride,
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
    __ubuf__ OFFSET_T* simtParams, const __gm__ VALUE_T* gradY, const int64_t nDims, const int64_t cDims,
    const int64_t dDims, const int64_t hDims, const int64_t wDims,
    const int64_t dOutDim, const int64_t hOutDim, const int64_t wOutDim,
    const int64_t kernelD, const int64_t kernelH, const int64_t kernelW,
    const int64_t padD, const int64_t padH, const int64_t padW,
    const int64_t dilationD, const int64_t dilationH, const int64_t dilationW,
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
    DIV_T count = nDims * cDims * dDims * hDims * wDims;
    for (DIV_T index = Simt::GetBlockIdx() * Simt::GetThreadNum() + Simt::GetThreadIdx(); index < count;
         index += Simt::GetBlockNum() * Simt::GetThreadNum()) {
        DIV_T temp1 = Simt::UintDiv(index, static_cast<DIV_T>(magicW), static_cast<DIV_T>(shiftW));
        DIV_T w = index - temp1 * static_cast<DIV_T>(wDims);
        DIV_T temp2 = Simt::UintDiv(temp1, static_cast<DIV_T>(magicH), static_cast<DIV_T>(shiftH));
        DIV_T h = temp1 - temp2 * static_cast<DIV_T>(hDims);
        DIV_T temp3 = Simt::UintDiv(temp2, static_cast<DIV_T>(magicD), static_cast<DIV_T>(shiftD));
        DIV_T d = temp2 - temp3 * static_cast<DIV_T>(dDims);
        DIV_T n = Simt::UintDiv(temp3, static_cast<DIV_T>(magicC), static_cast<DIV_T>(shiftC));
        DIV_T c = temp3 - n * static_cast<DIV_T>(cDims);

        DIV_T inputIdx = d * hDims * wDims + h * wDims + w;
        OFFSET_T pdStarts = PStart<OFFSET_T>(d, padD, kernelD, dilationD, magicStrideD, shiftStrideD);
        OFFSET_T pdEnds = PEnd<OFFSET_T>(d, padD, dOutDim, magicStrideD, shiftStrideD);
        OFFSET_T phStarts = PStart<OFFSET_T>(h, padH, kernelH, dilationH, magicStrideH, shiftStrideH);
        OFFSET_T phEnds = PEnd<OFFSET_T>(h, padH, hOutDim, magicStrideH, shiftStrideH);
        OFFSET_T pwStarts = PStart<OFFSET_T>(w, padW, kernelW, dilationW, magicStrideW, shiftStrideW);
        OFFSET_T pwEnds = PEnd<OFFSET_T>(w, padW, wOutDim, magicStrideW, shiftStrideW);

        float gradient = 0.0f;
        for (OFFSET_T pd = pdStarts; pd < pdEnds; ++pd) {
            for (OFFSET_T ph = phStarts; ph < phEnds; ++ph) {
                for (OFFSET_T pw = pwStarts; pw < pwEnds; ++pw) {
                    DIV_T outputIdx = n * cDims * dOutDim * hOutDim * wOutDim + c * dOutDim * hOutDim * wOutDim +
                                      pd * hOutDim * wOutDim + ph * wOutDim + pw;
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
    __ubuf__ OFFSET_T* simtParam, const __gm__ VALUE_T* gradY, const int64_t nDims, const int64_t cDims,
    const int64_t dDims, const int64_t hDims, const int64_t wDims,
    const int64_t dOutDim, const int64_t hOutDim, const int64_t wOutDim,
    const int64_t kernelD, const int64_t kernelH, const int64_t kernelW,
    const int64_t padD, const int64_t padH, const int64_t padW,
    const int64_t dilationD, const int64_t dilationH, const int64_t dilationW,
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
    DIV_T count = nDims * cDims * dDims * hDims * wDims;
    for (DIV_T index = Simt::GetBlockIdx() * Simt::GetThreadNum() + Simt::GetThreadIdx(); index < count;
         index += Simt::GetBlockNum() * Simt::GetThreadNum()) {
        DIV_T temp1 = Simt::UintDiv(index, static_cast<DIV_T>(magicC), static_cast<DIV_T>(shiftC));
        DIV_T c = index - temp1 * static_cast<DIV_T>(cDims);
        DIV_T temp2 = Simt::UintDiv(temp1, static_cast<DIV_T>(magicW), static_cast<DIV_T>(shiftW));
        DIV_T w = temp1 - temp2 * static_cast<DIV_T>(wDims);
        DIV_T temp3 = Simt::UintDiv(temp2, static_cast<DIV_T>(magicH), static_cast<DIV_T>(shiftH));
        DIV_T h = temp2 - temp3 * static_cast<DIV_T>(hDims);
        DIV_T n = Simt::UintDiv(temp3, static_cast<DIV_T>(magicD), static_cast<DIV_T>(shiftD));
        DIV_T d = temp3 - n * static_cast<DIV_T>(dDims);

        DIV_T inputIdx = d * hDims * wDims + h * wDims + w;
        OFFSET_T pdStart = PStart<OFFSET_T>(d, padD, kernelD, dilationD, magicStrideD, shiftStrideD);
        OFFSET_T pdEnd = PEnd<OFFSET_T>(d, padD, dOutDim, magicStrideD, shiftStrideD);
        OFFSET_T phStart = PStart<OFFSET_T>(h, padH, kernelH, dilationH, magicStrideH, shiftStrideH);
        OFFSET_T phEnd = PEnd<OFFSET_T>(h, padH, hOutDim, magicStrideH, shiftStrideH);
        OFFSET_T pwStart = PStart<OFFSET_T>(w, padW, kernelW, dilationW, magicStrideW, shiftStrideW);
        OFFSET_T pwEnd = PEnd<OFFSET_T>(w, padW, wOutDim, magicStrideW, shiftStrideW);

        float gradient = 0.0f;
        for (OFFSET_T pd = pdStart; pd < pdEnd; ++pd) {
            for (OFFSET_T ph = phStart; ph < phEnd; ++ph) {
                for (OFFSET_T pw = pwStart; pw < pwEnd; ++pw) {
                    DIV_T outputIdx = n * dOutDim * hOutDim * wOutDim * cDims + pd * hOutDim * wOutDim * cDims +
                                      ph * wOutDim * cDims + pw * cDims + c;
                    if (static_cast<DIV_T>(argmax[outputIdx]) == inputIdx) {
                        gradient += static_cast<float>(gradY[outputIdx]);
                    }
                }
            }
        }
        gradX[index] = static_cast<VALUE_T>(gradient);
    }
}

template <typename VALUE_T, typename INDICES_T, typename OFFSET_T, int64_t CHANNEL_LAST>
__aicore__ inline void MaxPool3DGradWithArgmaxSimt<VALUE_T, INDICES_T, OFFSET_T, CHANNEL_LAST>::Init(GM_ADDR x,
                                                                                                     GM_ADDR grad,
                                                                                                     GM_ADDR argmax,
                                                                                                     GM_ADDR y)
{
    x_.SetGlobalBuffer((__gm__ VALUE_T *)(x));
    grad_.SetGlobalBuffer((__gm__ VALUE_T *)(grad));
    argmax_.SetGlobalBuffer((__gm__ INDICES_T *)(argmax));
    y_.SetGlobalBuffer((__gm__ VALUE_T *)(y));
    pipe_->InitBuffer(simtTilingDataBuf_, TILING_DATA_NUM * sizeof(int64_t));
    pipe_->InitBuffer(paramBuf_, SIMT_PARAMS_NUM * sizeof(OFFSET_T));
}

template <typename VALUE_T, typename INDICES_T, typename OFFSET_T, int64_t CHANNEL_LAST>
__aicore__ inline void MaxPool3DGradWithArgmaxSimt<VALUE_T, INDICES_T, OFFSET_T, CHANNEL_LAST>::Process()
{
    LocalTensor<int64_t> simtTilingData = simtTilingDataBuf_.Get<int64_t>();
    LocalTensor<OFFSET_T> simtParam = paramBuf_.Get<OFFSET_T>();
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

    GetUintDivMagicAndShift<DIV_T>(magicC, shiftC, simtTilingData(1));
    GetUintDivMagicAndShift<DIV_T>(magicD, shiftD, simtTilingData(2));
    GetUintDivMagicAndShift<DIV_T>(magicH, shiftH, simtTilingData(3));
    GetUintDivMagicAndShift<DIV_T>(magicW, shiftW, simtTilingData(4));
    GetUintDivMagicAndShift<DIV_T>(magicStrideD, shiftStrideD, simtTilingData(11));
    GetUintDivMagicAndShift<DIV_T>(magicStrideH, shiftStrideH, simtTilingData(12));
    GetUintDivMagicAndShift<DIV_T>(magicStrideW, shiftStrideW, simtTilingData(13));

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

    if constexpr (CHANNEL_LAST == 1) {
        Simt::VF_CALL<MaxPool3DGradWithArgmaxNdhwc<VALUE_T, INDICES_T, OFFSET_T>>(
            Simt::Dim3(THREAD_DIM), (__ubuf__ OFFSET_T *)simtParam.GetPhyAddr(), gradData, simtTilingData(0),
            simtTilingData(1), simtTilingData(2), simtTilingData(3), simtTilingData(4), simtTilingData(5),
            simtTilingData(6), simtTilingData(7), simtTilingData(8), simtTilingData(9), simtTilingData(10),
            simtTilingData(14), simtTilingData(15), simtTilingData(16), simtTilingData(17), simtTilingData(18),
            simtTilingData(19), outputData, indicesData);
    } else {
        Simt::VF_CALL<MaxPool3DGradWithArgmaxNcdhw<VALUE_T, INDICES_T, OFFSET_T>>(
            Simt::Dim3(THREAD_DIM), (__ubuf__ OFFSET_T *)simtParam.GetPhyAddr(), gradData, simtTilingData(0),
            simtTilingData(1), simtTilingData(2), simtTilingData(3), simtTilingData(4), simtTilingData(5),
            simtTilingData(6), simtTilingData(7), simtTilingData(8), simtTilingData(9), simtTilingData(10),
            simtTilingData(14), simtTilingData(15), simtTilingData(16), simtTilingData(17), simtTilingData(18),
            simtTilingData(19), outputData, indicesData);
    }
}

}

#endif  //MAX_POOL3D_GRAD_WITH_ARGMAX_SIMT_H
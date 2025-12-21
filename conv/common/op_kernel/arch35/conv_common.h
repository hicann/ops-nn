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
 * \file conv_common.h
 * \brief
 */

#ifndef CONV_COMMON_H
#define CONV_COMMON_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"

#if defined(FORMAT_X)
#if FORMAT_X == FORMAT_NCHW || FORMAT_X == FORMAT_NHWC
#include "../../conv2d_v2/arch35/conv2d_v2_api.h"
using namespace conv2d;
#elif FORMAT_X == FORMAT_NCDHW || FORMAT_X == FORMAT_NDHWC
#include "../../conv3d_v2_apt/arch35/conv3d_v2_api.h"
using namespace conv3d;
#endif
#endif
 
constexpr uint8_t SINGLE_BLOCK_SIZE = 32;
constexpr uint8_t M0 = 16;
constexpr uint8_t N0 = 16;

struct DimDataToFill {
    __aicore__ __forceinline__ DimDataToFill(uint64_t& singleCoreDim_, uint64_t& dimIdxStart_, bool& isDimTail_) :
        singleCoreDim(singleCoreDim_), dimIdxStart(dimIdxStart_), isDimTail(isDimTail_) {}

    uint64_t& singleCoreDim;
    uint64_t& dimIdxStart;
    bool& isDimTail;
};

struct ExtendParams {
    __aicore__ inline ExtendParams(){};

    __aicore__ inline ExtendParams(GM_ADDR scale0_, GM_ADDR reluWeight0_, GM_ADDR clipValue0_, GM_ADDR scale1_,
        GM_ADDR reluWeight1_, GM_ADDR clipValue1_, GM_ADDR y1_)
        : scale0(scale0_), reluWeight0(reluWeight0_), clipValue0(clipValue0_), scale1(scale1_),
          reluWeight1(reluWeight1_), clipValue1(clipValue1_), y1(y1_)
    {};

    GM_ADDR scale0 = nullptr;
    GM_ADDR reluWeight0 = nullptr;
    GM_ADDR clipValue0 = nullptr;
    GM_ADDR scale1 = nullptr;
    GM_ADDR reluWeight1 = nullptr;
    GM_ADDR clipValue1 = nullptr;
    GM_ADDR y1 = nullptr;
};

template <class CONV, class RUN_INFO>
class ConvCommon {
public:
    __aicore__ __forceinline__ void Init(CONV *convPtr, const RUN_INFO* tilingPtr, bool hasScaleFlag);

    // Calc singleCore dim data
    __aicore__ __forceinline__ bool CalcDimData(const uint32_t& blockPerDim, const uint32_t& dim,
                                                const uint64_t& wholeDim, const uint64_t &realWholeDim,
                                                DimDataToFill& curStruct);

    // Calc singleCore N dim data for weight NZ
    __aicore__ __forceinline__ bool CalcNDimDataWeightNZ(const uint64_t& blockPerDim, const uint64_t& dim,
                                                         const uint64_t& wholeDim, const uint64_t& realWholeDim,
                                                         DimDataToFill& curStruct);

    __aicore__ __forceinline__ void CalcStartAddrCommon(const uint32_t din, const uint32_t dout);

    __aicore__ __forceinline__ void CalcStartAddrMMode(const uint32_t din, const uint32_t dout, const uint32_t kd,
                                                       const uint64_t doIdxStart = 0, const int64_t diIdxStart = 0);

    __aicore__ __forceinline__ void CalcStartAddrHWode(const uint32_t din, const uint32_t dout, const uint32_t kd,
                                                       const uint64_t doIdxStart = 0, const int64_t diIdxStart = 0);

    __aicore__ __forceinline__ void CalcStartAddrMModeHWC(const uint32_t din, const uint32_t dout,
                                                          const uint64_t doIdxStart = 0, const int64_t diIdxStart = 0);

    __aicore__ __forceinline__ void CalcStartAddrHWodeHWC(const uint32_t din, const uint32_t dout,
                                                          const uint64_t doIdxStart = 0, const int64_t diIdxStart = 0);

    __aicore__ __forceinline__ void InitBufferCommon(GM_ADDR x, GM_ADDR filter, GM_ADDR bias, GM_ADDR y,
                                                     const ExtendParams* extendParams);

    __aicore__ __forceinline__ void InitFixpipeBuffer(const ExtendParams* extendParams);

    __aicore__ __forceinline__ uint64_t AlignB(const uint64_t numA, const uint64_t numB);

    __aicore__ __forceinline__ uint64_t CeilDiv(const uint64_t numA, const uint64_t numB);

    __aicore__ __forceinline__ int64_t Max(const int64_t numA, const int64_t numB);

public:
    uint32_t blockIdx = get_block_idx();

protected:
    CONV *convOps;
    const RUN_INFO *convRunInfo;
    bool hasScale = true;
    uint64_t fmStartAddr = 0;
    uint64_t weightStartAddr = 0;
    uint64_t biasStartAddr = 0;
    uint64_t scaleStartAddr = 0;
    uint64_t outputStartAddr = 0;

    uint64_t hwIn = 0;
    uint64_t hwOut = 0;
};

template <class CONV, class RUN_INFO>
__aicore__ __forceinline__ void ConvCommon<CONV, RUN_INFO>::
    Init(CONV *convPtr, const RUN_INFO* tilingPtr, bool hasScaleFlag)
{
    convOps = convPtr;
    convRunInfo = tilingPtr;
    hasScale = hasScaleFlag;
}

template <class CONV, class RUN_INFO>
__aicore__ __forceinline__ bool ConvCommon<CONV, RUN_INFO>::
    CalcDimData(const uint32_t& blockPerDim, const uint32_t& dim, const uint64_t& wholeDim,
                const uint64_t &realWholeDim, DimDataToFill& curStruct)
{
    const uint32_t dimIdx = (blockIdx / blockPerDim) % dim;
    const uint64_t maxDimPerCore = CeilDiv(wholeDim, dim);
    const uint64_t realDim = CeilDiv(realWholeDim, maxDimPerCore);

    if (unlikely(dimIdx >= realDim)) {
        return false;
    }

    curStruct.isDimTail = (dimIdx == (realDim - 1));
    curStruct.singleCoreDim = !curStruct.isDimTail ? maxDimPerCore : realWholeDim - (realDim - 1) * maxDimPerCore;
    curStruct.dimIdxStart = dimIdx * maxDimPerCore;
    return true;
}

template <class CONV, class RUN_INFO>
__aicore__ __forceinline__ bool ConvCommon<CONV, RUN_INFO>::CalcNDimDataWeightNZ(const uint64_t& blockPerDim,
                                                                                 const uint64_t& dim,
                                                                                 const uint64_t& wholeDim,
                                                                                 const uint64_t& realWholeDim,
                                                                                 DimDataToFill& curStruct)
{
    const uint64_t dimIdx = (blockIdx / blockPerDim) % dim;
    const uint64_t maxDimPerCore = AlignB(CeilDiv(wholeDim, dim), N0);
    const uint64_t realDim = CeilDiv(realWholeDim, maxDimPerCore);

    if (unlikely(dimIdx >= realDim)) {
        return false;
    }

    curStruct.isDimTail = (dimIdx == (realDim - 1));
    curStruct.singleCoreDim = !curStruct.isDimTail ? maxDimPerCore : realWholeDim - (realDim - 1) * maxDimPerCore;
    curStruct.dimIdxStart = dimIdx * maxDimPerCore;
    return true;
}

template <class CONV, class RUN_INFO>
__aicore__ __forceinline__ void ConvCommon<CONV, RUN_INFO>::
    CalcStartAddrCommon(const uint32_t din, const uint32_t dout)
{
    hwIn = convRunInfo->hin * convRunInfo->win;
    hwOut = convRunInfo->hout * convRunInfo->wout;

    convOps->fmapOneBatchSize = convRunInfo->cin * din * hwIn;
    convOps->outputOneBatchSize = convRunInfo->cout * dout * hwOut;

    if (convRunInfo->hasBias) {
        biasStartAddr = convOps->nIdxStart;
    }

    if constexpr (CONV::isQuant) {
        if (hasScale) {
            scaleStartAddr = convOps->nIdxStart;
        }
    }
}

template <class CONV, class RUN_INFO>
__aicore__ __forceinline__ void ConvCommon<CONV, RUN_INFO>::
    CalcStartAddrMMode(const uint32_t din, const uint32_t dout, const uint32_t kd, const uint64_t doIdxStart,
                       const int64_t diIdxStart)
{
    CalcStartAddrCommon(din, dout);

    fmStartAddr = convOps->batchIdxStart * convOps->fmapOneBatchSize;
    if constexpr (CONV::B_FORMAT == ConvFormat::FRACTAL_Z || CONV::B_FORMAT == ConvFormat::FRACTAL_Z_C04) {
        weightStartAddr = convOps->nIdxStart * convOps->k0;
    } else {
        weightStartAddr = convOps->nIdxStart * convRunInfo->cin * kd * convRunInfo->kh * convRunInfo->kw;
    }
    outputStartAddr = convOps->batchIdxStart * convOps->outputOneBatchSize +
                      convOps->nIdxStart * dout * hwOut +
                      convOps->mIdxStart;

    if constexpr (CONV::A_FORMAT == ConvFormat::NCDHW) {
        fmStartAddr += diIdxStart * hwIn;
        outputStartAddr += doIdxStart * hwOut;
    }

    if constexpr (CONV::IS_EXTEND_CONV2D) {
        if (convOps->conv2dApiTiling->quantMode0 == static_cast<uint8_t>(QuantModeType::VECTOR_QUANT) ||
            convOps->conv2dApiTiling->quantMode1 == static_cast<uint8_t>(QuantModeType::VECTOR_QUANT)) {
            scaleStartAddr = convOps->nIdxStart;
        }
    }
}

template <class CONV, class RUN_INFO>
__aicore__ __forceinline__ void ConvCommon<CONV, RUN_INFO>::
    CalcStartAddrHWode(const uint32_t din, const uint32_t dout, const uint32_t kd, const uint64_t doIdxStart,
                       const int64_t diIdxStart)
{
    CalcStartAddrCommon(din, dout);

    int64_t hiStartPosTmp = static_cast<int64_t>(convOps->hoIdxStart * convRunInfo->strideH) -
        static_cast<int64_t>(convRunInfo->padTop);
    convOps->singleCoreHiStartPos = hiStartPosTmp < 0 ? 0 : hiStartPosTmp;
    fmStartAddr = convOps->batchIdxStart * convOps->fmapOneBatchSize + convOps->singleCoreHiStartPos *
        convRunInfo->win;
    convOps->singleCoreHiStartPos = hiStartPosTmp;
    if constexpr (CONV::A_FORMAT == ConvFormat::NCHW) {
        int64_t wiStartPosTmp = static_cast<int64_t>(convOps->woIdxStart * convRunInfo->strideW) -
                                static_cast<int64_t>(convRunInfo->padLeft);
        convOps->singleCoreWiStartPos = wiStartPosTmp < 0 ? 0 : wiStartPosTmp;
        fmStartAddr += convOps->singleCoreWiStartPos;
        convOps->singleCoreWiStartPos = wiStartPosTmp;
    }

    if constexpr (CONV::B_FORMAT == ConvFormat::FRACTAL_Z || CONV::B_FORMAT == ConvFormat::FRACTAL_Z_C04) {
        weightStartAddr = convOps->nIdxStart * convOps->k0;
    } else {
        weightStartAddr = convOps->nIdxStart * convRunInfo->cin * kd * convRunInfo->kh * convRunInfo->kw;
    }

    outputStartAddr = convOps->batchIdxStart * convOps->outputOneBatchSize +
                      convOps->nIdxStart * dout * hwOut +
                      convOps->hoIdxStart * convRunInfo->wout;
    if constexpr (CONV::A_FORMAT == ConvFormat::NCHW) {
        outputStartAddr += convOps->woIdxStart;
    }

    if constexpr (CONV::A_FORMAT == ConvFormat::NCDHW) {
        fmStartAddr += diIdxStart * hwIn;
        outputStartAddr += doIdxStart * hwOut;
    }
}

template <class CONV, class RUN_INFO>
__aicore__ __forceinline__ void ConvCommon<CONV, RUN_INFO>::
    CalcStartAddrMModeHWC(const uint32_t din, const uint32_t dout, const uint64_t doIdxStart, const int64_t diIdxStart)
{
    CalcStartAddrCommon(din, dout);

    fmStartAddr = convOps->batchIdxStart * convOps->fmapOneBatchSize;
    weightStartAddr = convOps->nIdxStart;
    outputStartAddr = convOps->batchIdxStart * convOps->outputOneBatchSize +
                      convOps->mIdxStart * convRunInfo->cout +
                      convOps->nIdxStart;

    if constexpr (CONV::A_FORMAT == ConvFormat::NDHWC) {
        fmStartAddr += diIdxStart * hwIn * convRunInfo->cin;
        outputStartAddr += doIdxStart * hwOut * convRunInfo->cout;
    }
}

template <class CONV, class RUN_INFO>
__aicore__ __forceinline__ void ConvCommon<CONV, RUN_INFO>::
    CalcStartAddrHWodeHWC(const uint32_t din, const uint32_t dout, const uint64_t doIdxStart, const int64_t diIdxStart)
{
    CalcStartAddrCommon(din, dout);

    int64_t hiStartPosTmp = static_cast<int64_t>(convOps->hoIdxStart * convRunInfo->strideH) -
        static_cast<int64_t>(convRunInfo->padTop);
    convOps->singleCoreHiStartPos = hiStartPosTmp < 0 ? 0 : hiStartPosTmp;
    fmStartAddr = convOps->batchIdxStart * convOps->fmapOneBatchSize + convOps->singleCoreHiStartPos *
        convRunInfo->win * convRunInfo->cin;
    convOps->singleCoreHiStartPos = hiStartPosTmp;
    if constexpr (CONV::A_FORMAT == ConvFormat::NHWC) {
        int64_t wiStartPosTmp = static_cast<int64_t>(convOps->woIdxStart * convRunInfo->strideW) -
                                static_cast<int64_t>(convRunInfo->padLeft);
        convOps->singleCoreWiStartPos = wiStartPosTmp;
        fmStartAddr += Max(convOps->singleCoreWiStartPos, 0) * convRunInfo->cin;
    }

    weightStartAddr = convOps->nIdxStart;

    outputStartAddr = convOps->batchIdxStart * convOps->outputOneBatchSize + convOps->hoIdxStart *
                      convRunInfo->wout * convRunInfo->cout + convOps->nIdxStart;

    if constexpr (CONV::A_FORMAT == ConvFormat::NHWC) {
        outputStartAddr += convOps->woIdxStart * convRunInfo->cout;
    }
    if constexpr (CONV::A_FORMAT == ConvFormat::NDHWC) {
        fmStartAddr += diIdxStart * hwIn * convRunInfo->cin;
        outputStartAddr += doIdxStart * hwOut * convRunInfo->cout;
    }
}

template <class CONV, class RUN_INFO>
__aicore__ __forceinline__ void ConvCommon<CONV, RUN_INFO>::
     InitFixpipeBuffer(const ExtendParams* extendParams)
{
    if constexpr (CONV::isQuant || CONV::IS_EXTEND_CONV2D) {
        if (hasScale) {
            if constexpr (CONV::A_FORMAT == ConvFormat::NCHW || CONV::A_FORMAT == ConvFormat::NHWC) {
                if (convOps->conv2dApiTiling->quantMode0 == static_cast<uint8_t>(QuantModeType::VECTOR_QUANT)) {
                    convOps->fixpipeParams.scale0.SetGlobalBuffer(
                        reinterpret_cast<__gm__ typename CONV::SCALE_T*>(extendParams->scale0 + scaleStartAddr *
                        sizeof(typename CONV::SCALE_T)));
                } else if (convOps->conv2dApiTiling->quantMode0 == static_cast<uint8_t>(QuantModeType::SCALAR_QUANT)) {
                    convOps->fixpipeParams.scale0.SetGlobalBuffer(
                        reinterpret_cast<__gm__ typename CONV::SCALE_T*>(extendParams->scale0));
                }
                if constexpr (CONV::IS_EXTEND_CONV2D) {
                    if (convOps->dualOutput) {
                        if (convOps->conv2dApiTiling->quantMode1 == static_cast<uint8_t>(QuantModeType::VECTOR_QUANT)) {
                            convOps->fixpipeParams.scale1.SetGlobalBuffer(
                                reinterpret_cast<__gm__ typename CONV::SCALE_T*>(extendParams->scale1 + scaleStartAddr *
                                sizeof(typename CONV::SCALE_T)));
                        } else if (convOps->conv2dApiTiling->quantMode1 == static_cast<uint8_t>(QuantModeType::SCALAR_QUANT)) {
                            convOps->fixpipeParams.scale1.SetGlobalBuffer(
                                reinterpret_cast<__gm__ typename CONV::SCALE_T*>(extendParams->scale1));
                        }
                    }
                }
            } else {
                convOps->scaleGm.SetGlobalBuffer(
                    reinterpret_cast<__gm__ typename CONV::SCALE_T*>(extendParams->scale0 + scaleStartAddr *
                    sizeof(typename CONV::SCALE_T)));
            }
        }
    }
}

template <class CONV, class RUN_INFO>
__aicore__ __forceinline__ void ConvCommon<CONV, RUN_INFO>::
    InitBufferCommon(GM_ADDR x, GM_ADDR filter, GM_ADDR bias, GM_ADDR y, const ExtendParams* extendParams)
{
    convOps->fmapGm.SetGlobalBuffer(
        reinterpret_cast<__gm__ typename CONV::FMAP_T*>(x + fmStartAddr * sizeof(typename CONV::FMAP_T)));
    convOps->filterGm.SetGlobalBuffer(
        reinterpret_cast<__gm__ typename CONV::WEIGHT_T*>(filter + weightStartAddr * sizeof(typename CONV::WEIGHT_T)));
    convOps->outputGm.SetGlobalBuffer(
        reinterpret_cast<__gm__ typename CONV::OUTPUT_T*>(y + outputStartAddr * sizeof(typename CONV::OUTPUT_T)));
    if constexpr (CONV::IS_EXTEND_CONV2D) {
        if (convOps->dualOutput) {
            convOps->output1Gm.SetGlobalBuffer(
                reinterpret_cast<__gm__ typename CONV::OUTPUT1_T*>(
                    extendParams->y1 + outputStartAddr * sizeof(typename CONV::OUTPUT1_T)));
        }
    }
    if (convRunInfo->hasBias) {
        convOps->biasGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ typename CONV::BIAS_T*>(bias + biasStartAddr * sizeof(typename CONV::BIAS_T)));
    }
 
    InitFixpipeBuffer(extendParams);
}

template <class CONV, class RUN_INFO>
__aicore__ __forceinline__ uint64_t ConvCommon<CONV, RUN_INFO>::
    AlignB(const uint64_t numA, const uint64_t numB)
{
    return ((numA + numB - 1) / numB) * numB;
}

template <class CONV, class RUN_INFO>
__aicore__ __forceinline__ uint64_t ConvCommon<CONV, RUN_INFO>::
    CeilDiv(const uint64_t numA, const uint64_t numB)
{
    return (numA + numB - 1) / numB;
}

template <class CONV, class RUN_INFO>
__aicore__ __forceinline__ int64_t ConvCommon<CONV, RUN_INFO>::
    Max(const int64_t numA, const int64_t numB)
{
    return numA > numB ? numA : numB;
}

#endif // CONV_COMMON_H
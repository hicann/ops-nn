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

#include "kernel_basic_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "conv_config.h"

using namespace conv;

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

template <class CONV, class CONV_TILING>
class ConvCommon {
public:
    __aicore__ __forceinline__ void Init(CONV *convPtr, const CONV_TILING* tilingPtr, bool hasScaleFlag);

    // Calc singleCore dim data
    __aicore__ __forceinline__ bool CalcDimData(const uint32_t& blockPerDim, const uint32_t& dim,
                                                const uint64_t& wholeDim, const uint64_t &realWholeDim,
                                                DimDataToFill& curStruct);

    // Calc singleCore N dim data with align by N0
    __aicore__ __forceinline__ bool CalcNDimDataAlign(const uint64_t& blockPerDim, const uint64_t& dim,
                                                      const uint64_t& wholeDim, DimDataToFill& curStruct);

    __aicore__ __forceinline__ void CalcStartAddrCommon(const uint32_t din, const uint32_t dout);

    __aicore__ __forceinline__ void CalcStartAddrMMode(const uint32_t din, const uint32_t dout, const uint32_t kd,
                                                       const uint64_t doIdxStart = 0, const int64_t diIdxStart = 0);

    __aicore__ __forceinline__ void CalcStartAddrHWode(const uint32_t din, const uint32_t dout, const uint32_t kd,
                                                       const uint64_t doIdxStart = 0, const int64_t diIdxStart = 0);

    __aicore__ __forceinline__ void CalcStartAddrMModeHWC(const uint32_t din, const uint32_t dout,
                                                          const uint64_t doIdxStart = 0, const int64_t diIdxStart = 0);

    __aicore__ __forceinline__ void CalcStartAddrHWodeHWC(const uint32_t din, const uint32_t dout,
                                                          const uint64_t doIdxStart = 0, const int64_t diIdxStart = 0);

    __aicore__ __forceinline__ void CalcStartAddrDeQuant(const uint32_t din, const uint32_t dout, const uint32_t kd,
                                                         const uint64_t doIdxStart, const int64_t diIdxStart);

    __aicore__ __forceinline__ void InitBufferCommon(GM_ADDR x, GM_ADDR filter, GM_ADDR bias, GM_ADDR y,
                                                     const ExtendParams* extendParams);

    __aicore__ __forceinline__ void InitFixpipeBuffer(const ExtendParams* extendParams);

    __aicore__ __forceinline__ uint64_t AlignB(const uint64_t numA, const uint64_t numB);

    __aicore__ __forceinline__ uint64_t CeilDiv(const uint64_t numA, const uint64_t numB);

    __aicore__ __forceinline__ int64_t Max(const int64_t numA, const int64_t numB);

public:
    uint32_t blockIdx = 0;

protected:
    CONV *convOps;
    const CONV_TILING *convTilingData;
    bool hasScale = true;
    uint64_t fmStartAddr = 0;
    uint64_t weightStartAddr = 0;
    uint64_t biasStartAddr = 0;
    uint64_t scaleStartAddr = 0;
    uint64_t reluWeightStartAddr = 0;
    uint64_t outputStartAddr = 0;

    uint64_t hwIn = 0;
    uint64_t hwOut = 0;
};

template <class CONV, class CONV_TILING>
__aicore__ __forceinline__ void ConvCommon<CONV, CONV_TILING>::
    Init(CONV *convPtr, const CONV_TILING* tilingPtr, bool hasScaleFlag)
{
    if ASCEND_IS_AIV {
        blockIdx = GetBlockIdx();
        uint32_t subblocknum = GetSubBlockNum();
        blockIdx = blockIdx / subblocknum;
    } else {
        blockIdx = GetBlockIdx();
    }
    convOps = convPtr;
    convTilingData = tilingPtr;
    hasScale = hasScaleFlag;
}

template <class CONV, class CONV_TILING>
__aicore__ __forceinline__ bool ConvCommon<CONV, CONV_TILING>::
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

template <class CONV, class CONV_TILING>
__aicore__ __forceinline__ bool ConvCommon<CONV, CONV_TILING>::CalcNDimDataAlign(const uint64_t& blockPerDim,
                                                                              const uint64_t& dim,
                                                                              const uint64_t& wholeDim,
                                                                              DimDataToFill& curStruct)
{
    const uint64_t dimIdx = (blockIdx / blockPerDim) % dim;
    const uint64_t maxDimPerCore = AlignB(CeilDiv(AlignB(wholeDim, N0), dim), N0);
    const uint64_t realDim = CeilDiv(wholeDim, maxDimPerCore);

    if (unlikely(dimIdx >= realDim)) {
        return false;
    }

    curStruct.isDimTail = (dimIdx == (realDim - 1));
    curStruct.singleCoreDim = !curStruct.isDimTail ? maxDimPerCore : wholeDim - (realDim - 1) * maxDimPerCore;
    curStruct.dimIdxStart = dimIdx * maxDimPerCore;
    return true;
}

template <class CONV, class CONV_TILING>
__aicore__ __forceinline__ void ConvCommon<CONV, CONV_TILING>::
    CalcStartAddrCommon(const uint32_t din, const uint32_t dout)
{
    hwIn = convTilingData->convRunInfo.hin * convTilingData->convRunInfo.win;
    hwOut = convTilingData->convRunInfo.hout * convTilingData->convRunInfo.wout;

    convOps->fmapOneBatchSize = convTilingData->convRunInfo.cin * din * hwIn;
    convOps->outputOneBatchSize = convTilingData->convRunInfo.cout * dout * hwOut;

    if (convTilingData->convRunInfo.hasBias) {
        biasStartAddr = convOps->nIdxStart;
    }

    if constexpr (CONV::isQuant) {
        if (hasScale) {
            scaleStartAddr = convOps->nIdxStart;
        }
    }
}

template <class CONV, class CONV_TILING>
__aicore__ __forceinline__ void ConvCommon<CONV, CONV_TILING>::
    CalcStartAddrMMode(const uint32_t din, const uint32_t dout, const uint32_t kd, const uint64_t doIdxStart,
                       const int64_t diIdxStart)
{
    CalcStartAddrCommon(din, dout);

    if constexpr (CONV::DIS_CONTINUOUS) {
        fmStartAddr = convOps->batchIdxStart * convTilingData->convRunInfo.cin;
    } else {
        fmStartAddr = convOps->batchIdxStart * convOps->fmapOneBatchSize;
    }
    if constexpr (CONV::B_FORMAT == ConvFormat::FRACTAL_Z || CONV::B_FORMAT == ConvFormat::FRACTAL_Z_C04) {
        weightStartAddr = convOps->nIdxStart * convOps->k0;
    } else {
        weightStartAddr = convOps->nIdxStart * convTilingData->convRunInfo.cin * kd * convTilingData->convRunInfo.kh * convTilingData->convRunInfo.kw;
    }
    outputStartAddr = convOps->batchIdxStart * convOps->outputOneBatchSize +
                      convOps->nIdxStart * dout * hwOut +
                      convOps->mIdxStart;

    if constexpr (CONV::A_FORMAT == ConvFormat::NCDHW) {
        fmStartAddr += diIdxStart * hwIn;
        outputStartAddr += doIdxStart * hwOut;
    }

    if constexpr (CONV::IS_EXTEND_CONV2D) {
        if (convOps->convTilingData->convApiTiling.quantMode0 == static_cast<uint8_t>(QuantModeType::VECTOR_QUANT) ||
            convOps->convTilingData->convApiTiling.quantMode1 == static_cast<uint8_t>(QuantModeType::VECTOR_QUANT)) {
            scaleStartAddr = convOps->nIdxStart;
        }
        if (convTilingData->convApiTiling.reluMode0 == static_cast<uint8_t>(ReluMode::VECTOR_RELU) ||
            convTilingData->convApiTiling.reluMode1 == static_cast<uint8_t>(ReluMode::VECTOR_RELU)) {
            reluWeightStartAddr = convOps->nIdxStart;
        }
    }
}

template <class CONV, class CONV_TILING>
__aicore__ __forceinline__ void ConvCommon<CONV, CONV_TILING>::
    CalcStartAddrHWode(const uint32_t din, const uint32_t dout, const uint32_t kd, const uint64_t doIdxStart,
                       const int64_t diIdxStart)
{
    CalcStartAddrCommon(din, dout);

    int64_t hiStartPosTmp = static_cast<int64_t>(convOps->hoIdxStart * convTilingData->convRunInfo.strideH) -
        static_cast<int64_t>(convTilingData->convRunInfo.padTop);
    convOps->singleCoreHiStartPos = hiStartPosTmp < 0 ? 0 : hiStartPosTmp;
    if constexpr (CONV::DIS_CONTINUOUS) {
        fmStartAddr = convOps->singleCoreHiStartPos * convTilingData->convRunInfo.win *
                      convTilingData->convRunInfo.batch * convTilingData->convRunInfo.cin +
                      convOps->batchIdxStart * convTilingData->convRunInfo.cin;
    } else {
        fmStartAddr = convOps->batchIdxStart * convOps->fmapOneBatchSize + convOps->singleCoreHiStartPos *
            convTilingData->convRunInfo.win;
    }
    convOps->singleCoreHiStartPos = hiStartPosTmp;
    if constexpr (CONV::A_FORMAT == ConvFormat::NCHW) {
        int64_t wiStartPosTmp = static_cast<int64_t>(convOps->woIdxStart * convTilingData->convRunInfo.strideW) -
                                static_cast<int64_t>(convTilingData->convRunInfo.padLeft);
        convOps->singleCoreWiStartPos = wiStartPosTmp < 0 ? 0 : wiStartPosTmp;
        if constexpr (CONV::DIS_CONTINUOUS) {
            fmStartAddr += convOps->singleCoreWiStartPos * convTilingData->convRunInfo.batch * convTilingData->convRunInfo.cin;
        } else {
            fmStartAddr += convOps->singleCoreWiStartPos;
        }
        convOps->singleCoreWiStartPos = wiStartPosTmp;
    }

    if constexpr (CONV::B_FORMAT == ConvFormat::FRACTAL_Z || CONV::B_FORMAT == ConvFormat::FRACTAL_Z_C04) {
        weightStartAddr = convOps->nIdxStart * convOps->k0;
    } else {
        weightStartAddr = convOps->nIdxStart * convTilingData->convRunInfo.cin * kd * convTilingData->convRunInfo.kh * convTilingData->convRunInfo.kw;
    }

    outputStartAddr = convOps->batchIdxStart * convOps->outputOneBatchSize +
                      convOps->nIdxStart * dout * hwOut +
                      convOps->hoIdxStart * convTilingData->convRunInfo.wout;
    if constexpr (CONV::A_FORMAT == ConvFormat::NCHW) {
        outputStartAddr += convOps->woIdxStart;
    }

    if constexpr (CONV::A_FORMAT == ConvFormat::NCDHW) {
        fmStartAddr += diIdxStart * hwIn;
        outputStartAddr += doIdxStart * hwOut;
    }
}

template <class CONV, class CONV_TILING>
__aicore__ __forceinline__ void ConvCommon<CONV, CONV_TILING>::
    CalcStartAddrMModeHWC(const uint32_t din, const uint32_t dout, const uint64_t doIdxStart, const int64_t diIdxStart)
{
    CalcStartAddrCommon(din, dout);

    fmStartAddr = convOps->batchIdxStart * convOps->fmapOneBatchSize;
    weightStartAddr = convOps->nIdxStart;
    outputStartAddr = convOps->batchIdxStart * convOps->outputOneBatchSize +
                      convOps->mIdxStart * convTilingData->convRunInfo.cout +
                      convOps->nIdxStart;

    if constexpr (CONV::A_FORMAT == ConvFormat::NDHWC) {
        fmStartAddr += diIdxStart * hwIn * convTilingData->convRunInfo.cin;
        outputStartAddr += doIdxStart * hwOut * convTilingData->convRunInfo.cout;
    }
}

template <class CONV, class CONV_TILING>
__aicore__ __forceinline__ void ConvCommon<CONV, CONV_TILING>::
    CalcStartAddrHWodeHWC(const uint32_t din, const uint32_t dout, const uint64_t doIdxStart, const int64_t diIdxStart)
{
    CalcStartAddrCommon(din, dout);

    int64_t hiStartPosTmp = static_cast<int64_t>(convOps->hoIdxStart * convTilingData->convRunInfo.strideH) -
        static_cast<int64_t>(convTilingData->convRunInfo.padTop);
    convOps->singleCoreHiStartPos = hiStartPosTmp < 0 ? 0 : hiStartPosTmp;
    fmStartAddr = convOps->batchIdxStart * convOps->fmapOneBatchSize + convOps->singleCoreHiStartPos *
        convTilingData->convRunInfo.win * convTilingData->convRunInfo.cin;
    convOps->singleCoreHiStartPos = hiStartPosTmp;
    if constexpr (CONV::A_FORMAT == ConvFormat::NHWC) {
        int64_t wiStartPosTmp = static_cast<int64_t>(convOps->woIdxStart * convTilingData->convRunInfo.strideW) -
                                static_cast<int64_t>(convTilingData->convRunInfo.padLeft);
        convOps->singleCoreWiStartPos = wiStartPosTmp;
        fmStartAddr += Max(convOps->singleCoreWiStartPos, 0) * convTilingData->convRunInfo.cin;
    }

    weightStartAddr = convOps->nIdxStart;

    outputStartAddr = convOps->batchIdxStart * convOps->outputOneBatchSize + convOps->hoIdxStart *
                      convTilingData->convRunInfo.wout * convTilingData->convRunInfo.cout + convOps->nIdxStart;

    if constexpr (CONV::A_FORMAT == ConvFormat::NHWC) {
        outputStartAddr += convOps->woIdxStart * convTilingData->convRunInfo.cout;
    }
    if constexpr (CONV::A_FORMAT == ConvFormat::NDHWC) {
        fmStartAddr += diIdxStart * hwIn * convTilingData->convRunInfo.cin;
        outputStartAddr += doIdxStart * hwOut * convTilingData->convRunInfo.cout;
    }
}

template <class CONV, class CONV_TILING>
__aicore__ __forceinline__ void ConvCommon<CONV, CONV_TILING>::
    CalcStartAddrDeQuant(const uint32_t din, const uint32_t dout, const uint32_t kd,
                         const uint64_t doIdxStart, const int64_t diIdxStart)
{
    CalcStartAddrCommon(din, dout);

    if constexpr (CONV::isMMode) {
        fmStartAddr = convOps->batchIdxStart * convOps->fmapOneBatchSize + diIdxStart * hwIn;

        outputStartAddr = convOps->batchIdxStart * convOps->outputOneBatchSize +
                          convOps->mIdxStart * convTilingData->convRunInfo.cout + convOps->nIdxStart +
                          doIdxStart * hwOut * convTilingData->convRunInfo.cout;
    } else {
        int64_t hiStartPosTmp = static_cast<int64_t>(convOps->hoIdxStart * convTilingData->convRunInfo.strideH) -
        static_cast<int64_t>(convTilingData->convRunInfo.padTop);
        convOps->singleCoreHiStartPos = hiStartPosTmp < 0 ? 0 : hiStartPosTmp;
        fmStartAddr = convOps->batchIdxStart * convOps->fmapOneBatchSize + convOps->singleCoreHiStartPos *
            convTilingData->convRunInfo.win + diIdxStart * hwIn;
        convOps->singleCoreHiStartPos = hiStartPosTmp;

        outputStartAddr = convOps->batchIdxStart * convOps->outputOneBatchSize + convOps->hoIdxStart *
                          convTilingData->convRunInfo.wout * convTilingData->convRunInfo.cout + convOps->nIdxStart +
                          doIdxStart * hwOut * convTilingData->convRunInfo.cout;
    }
    weightStartAddr = convOps->nIdxStart * convTilingData->convRunInfo.cin * kd * convTilingData->convRunInfo.kh * convTilingData->convRunInfo.kw;
}

template <class CONV, class CONV_TILING>
__aicore__ __forceinline__ void ConvCommon<CONV, CONV_TILING>::
     InitFixpipeBuffer(const ExtendParams* extendParams)
{
    if constexpr (CONV::isQuant || CONV::IS_EXTEND_CONV2D) {
        if (hasScale) {
            if constexpr (CONV::A_FORMAT == ConvFormat::NCHW || CONV::A_FORMAT == ConvFormat::NHWC) {
                if (convOps->convTilingData->convApiTiling.quantMode0 == static_cast<uint8_t>(QuantModeType::VECTOR_QUANT)) {
                    convOps->fixpipeParams.scale0.SetGlobalBuffer(
                        reinterpret_cast<__gm__ typename CONV::SCALE_T*>(extendParams->scale0 + scaleStartAddr *
                        sizeof(typename CONV::SCALE_T)));
                } else if (convOps->convTilingData->convApiTiling.quantMode0 == static_cast<uint8_t>(QuantModeType::SCALAR_QUANT)) {
                    convOps->fixpipeParams.scale0.SetGlobalBuffer(
                        reinterpret_cast<__gm__ typename CONV::SCALE_T*>(extendParams->scale0));
                }
                if constexpr (CONV::IS_EXTEND_CONV2D) {
                    if (convTilingData->convApiTiling.reluMode0 == static_cast<uint8_t>(ReluMode::SCALAR_RELU)) {
                        convOps->fixpipeParams.reluWeight0.SetGlobalBuffer(
                            reinterpret_cast<__gm__ typename CONV::RELU_WEIGHT_T*>(extendParams->reluWeight0));
                    } else if (convTilingData->convApiTiling.reluMode0 == static_cast<uint8_t>(ReluMode::VECTOR_RELU)) {
                        // copy relu_weight from om to gm
                        convOps->fixpipeParams.reluWeight0.SetGlobalBuffer(
                            reinterpret_cast<__gm__ typename CONV::RELU_WEIGHT_T*>(extendParams->reluWeight0 +
                            reluWeightStartAddr * sizeof(typename CONV::RELU_WEIGHT_T)));
                    }
                    if (convTilingData->convApiTiling.dualOutput) {
                        if (convTilingData->convApiTiling.quantMode1 == static_cast<uint8_t>(QuantModeType::VECTOR_QUANT)) {
                            convOps->fixpipeParams.scale1.SetGlobalBuffer(
                                reinterpret_cast<__gm__ typename CONV::SCALE_T*>(extendParams->scale1 + scaleStartAddr *
                                sizeof(typename CONV::SCALE_T)));
                        } else if (convOps->convTilingData->convApiTiling.quantMode1 == static_cast<uint8_t>(QuantModeType::SCALAR_QUANT)) {
                            convOps->fixpipeParams.scale1.SetGlobalBuffer(
                                reinterpret_cast<__gm__ typename CONV::SCALE_T*>(extendParams->scale1));
                        }
                        if (convTilingData->convApiTiling.reluMode1 == static_cast<uint8_t>(ReluMode::SCALAR_RELU)) {
                            convOps->fixpipeParams.reluWeight1.SetGlobalBuffer(
                                reinterpret_cast<__gm__ typename CONV::RELU_WEIGHT_T*>(extendParams->reluWeight1));
                        } else if (convTilingData->convApiTiling.reluMode1 == static_cast<uint8_t>(ReluMode::VECTOR_RELU)) {
                            convOps->fixpipeParams.reluWeight1.SetGlobalBuffer(
                                reinterpret_cast<__gm__ typename CONV::RELU_WEIGHT_T*>(extendParams->reluWeight1 +
                                reluWeightStartAddr * sizeof(typename CONV::RELU_WEIGHT_T)));
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

template <class CONV, class CONV_TILING>
__aicore__ __forceinline__ void ConvCommon<CONV, CONV_TILING>::
    InitBufferCommon(GM_ADDR x, GM_ADDR filter, GM_ADDR bias, GM_ADDR y, const ExtendParams* extendParams)
{
    convOps->fmapGm.SetGlobalBuffer(
        reinterpret_cast<__gm__ typename CONV::FMAP_T*>(x + fmStartAddr * sizeof(typename CONV::FMAP_T)));
    convOps->filterGm.SetGlobalBuffer(
        reinterpret_cast<__gm__ typename CONV::WEIGHT_T*>(filter + weightStartAddr * sizeof(typename CONV::WEIGHT_T)));
    convOps->outputGm.SetGlobalBuffer(
        reinterpret_cast<__gm__ typename CONV::OUTPUT_T*>(y + outputStartAddr * sizeof(typename CONV::OUTPUT_T)));
    if constexpr (CONV::IS_EXTEND_CONV2D) {
        if (convTilingData->convApiTiling.dualOutput) {
            convOps->output1Gm.SetGlobalBuffer(
                reinterpret_cast<__gm__ typename CONV::OUTPUT1_T*>(
                    extendParams->y1 + outputStartAddr * sizeof(typename CONV::OUTPUT1_T)));
        }
    }
    if (convTilingData->convRunInfo.hasBias) {
        convOps->biasGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ typename CONV::BIAS_T*>(bias + biasStartAddr * sizeof(typename CONV::BIAS_T)));
    }

    InitFixpipeBuffer(extendParams);
}

template <class CONV, class CONV_TILING>
__aicore__ __forceinline__ uint64_t ConvCommon<CONV, CONV_TILING>::
    AlignB(const uint64_t numA, const uint64_t numB)
{
    return ((numA + numB - 1) / numB) * numB;
}

template <class CONV, class CONV_TILING>
__aicore__ __forceinline__ uint64_t ConvCommon<CONV, CONV_TILING>::
    CeilDiv(const uint64_t numA, const uint64_t numB)
{
    return (numA + numB - 1) / numB;
}

template <class CONV, class CONV_TILING>
__aicore__ __forceinline__ int64_t ConvCommon<CONV, CONV_TILING>::
    Max(const int64_t numA, const int64_t numB)
{
    return numA > numB ? numA : numB;
}

#endif // CONV_COMMON_H
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
 * \file conv3d_v2.h
 * \brief
 */

#ifndef CONV3D_V2_H
#define CONV3D_V2_H

#include "kernel_tiling/kernel_tiling.h"
#include "../../common/arch35/conv_common.h"
#include "conv3d_v2_api.h"
#include "../conv3d_v2_tiling_data.h"

using namespace AscendC;
using namespace conv3d;
#define mDim hoDim

template<int8_t FmapTiling, int8_t WeightTiling,int8_t L1PingPong, int8_t L0PingPong, int8_t OutputOrder,
         int8_t IterOrder, int8_t GroupType>
struct Conv3DV2Param : public Conv3dParam {
    __aicore__ inline Conv3DV2Param() {}
    constexpr static int8_t fmapTiling = FmapTiling;
    constexpr static int8_t weightTiling = WeightTiling;
    constexpr static int8_t l1PingPong = L1PingPong;
    constexpr static int8_t l0PingPong = L0PingPong;
    constexpr static int8_t outputOrder = OutputOrder;
    constexpr static int8_t iterOrder = IterOrder;
    constexpr static int8_t groupType = GroupType;
    constexpr static int8_t isExtendConv2d = false;
    using Output1Dtype = half;
};

template <class FMAP_TYPE, class WEIGHT_TYPE, class OUTPUT_TYPE, class BIAS_TYPE, class SCALE_TYPE, class CONV_CFG>
class Conv3dV2Base {
public:
    __aicore__ inline Conv3dV2Base() {}

    __aicore__ inline void RunConv3dV2Kernel(GM_ADDR x, GM_ADDR filter, GM_ADDR bias, GM_ADDR y,
                                             const Ops::NN::Conv3dV2::Conv3DV2TilingData& convTilingData,
                                             const ExtendParams* extendParams = nullptr);
protected:
    __aicore__ inline bool Conv3dV2KernelInit(GM_ADDR x, GM_ADDR filter, GM_ADDR bias, GM_ADDR y,
                                              const ExtendParams* extendParams,
                                              const Ops::NN::Conv3dV2::Conv3DV2TilingData& tilingData);

    __aicore__ inline bool InitSingleCoreData(uint32_t blockPerNDim, uint32_t blockPerHoDim, uint32_t blockPerMDim);

    __aicore__ inline void InitBuffer(GM_ADDR x, GM_ADDR filter, GM_ADDR bias, GM_ADDR y,
                                      const ExtendParams* extendParams);

    __aicore__ inline void Conv3dV2KernelImpl();

public:
    // Get dtype
    using FMAP_T = typename FMAP_TYPE::T;
    using WEIGHT_T = typename WEIGHT_TYPE::T;
    using OUTPUT_T = typename OUTPUT_TYPE::T;
    using BIAS_T = typename BIAS_TYPE::T;
    using SCALE_T = typename SCALE_TYPE::T;

    // Conv3D API
    Conv3d<FMAP_TYPE, WEIGHT_TYPE, OUTPUT_TYPE, BIAS_TYPE, SCALE_TYPE, CONV_CFG> conv;
    ConvCommon<Conv3dV2Base, Ops::NN::Conv3dV2::Conv3DV2TilingData> convCommon;

    // Tiling data
    const Ops::NN::Conv3dV2::Conv3DV2TilingData* convTilingData;
    // Input and output tensor declare
    GlobalTensor<FMAP_T> fmapGm;
    GlobalTensor<WEIGHT_T> filterGm;
    GlobalTensor<OUTPUT_T> outputGm;
    GlobalTensor<BIAS_T> biasGm;
    GlobalTensor<SCALE_T> scaleGm;

    uint64_t batchIdxStart = 0;
    uint64_t doIdxStart = 0;
    uint64_t nIdxStart = 0;
    uint64_t mIdxStart = 0;
    uint64_t hoIdxStart = 0;

    // Single core data
    uint64_t singleCoreBatch = 0;
    uint64_t singleCoreCi = 0;
    uint64_t singleCoreDout = 0;
    uint64_t singleCoreN = 0;
    uint64_t singleCoreM = 0;
    uint64_t singleCoreHo = 0;
    int64_t singleCoreHiStartPos = 0;

    bool isBatchDimTail = false;
    bool isNDimTail = false;
    bool isDoDimTail = false;
    bool isMDimTail = false;
    bool isHoDimTail = false;
    bool hasScale = false;
    uint64_t fmapOneBatchSize = 0;
    uint64_t outputOneBatchSize = 0;

    constexpr static ConvFormat A_FORMAT = FMAP_TYPE::format;
    constexpr static ConvFormat B_FORMAT = WEIGHT_TYPE::format;
    constexpr static ConvFormat C_FORMAT = OUTPUT_TYPE::format;
    constexpr static bool isMMode = CONV_CFG::outputOrder == static_cast<int8_t>(ConvOutputOrder::M_MODE);
    constexpr static bool isQuant = (IsSameType<FMAP_T, int8_t>::value) ||
                                    (IsSameType<FMAP_T, hifloat8_t>::value) ||
                                    (IsSameType<FMAP_T, fp8_e4m3fn_t>::value);
    constexpr static bool IS_OPTGROUP_PRELOAD = false;
    constexpr static bool IS_EXTEND_CONV2D = false;
    constexpr static bool DIS_CONTINUOUS = false;
};

template <class FMAP_TYPE, class WEIGHT_TYPE, class OUTPUT_TYPE, class BIAS_TYPE, class SCALE_TYPE, class CONV_CFG>
__aicore__ inline void Conv3dV2Base<FMAP_TYPE, WEIGHT_TYPE, OUTPUT_TYPE, BIAS_TYPE, SCALE_TYPE, CONV_CFG>::
    RunConv3dV2Kernel(GM_ADDR x, GM_ADDR filter, GM_ADDR bias, GM_ADDR y,
                      const Ops::NN::Conv3dV2::Conv3DV2TilingData& convTilingData, const ExtendParams* extendParams)
{
    if (Conv3dV2KernelInit(x, filter, bias, y, extendParams, convTilingData)) {
        Conv3dV2KernelImpl();
    }
}

template <class FMAP_TYPE, class WEIGHT_TYPE, class OUTPUT_TYPE, class BIAS_TYPE, class SCALE_TYPE, class CONV_CFG>
__aicore__ inline bool Conv3dV2Base<FMAP_TYPE, WEIGHT_TYPE, OUTPUT_TYPE, BIAS_TYPE, SCALE_TYPE, CONV_CFG>::
    Conv3dV2KernelInit(GM_ADDR x, GM_ADDR filter, GM_ADDR bias, GM_ADDR y, const ExtendParams* extendParams,
                       const Ops::NN::Conv3dV2::Conv3DV2TilingData& tilingData)
{
    hasScale = (extendParams != nullptr);
    convTilingData = &tilingData;
    convCommon.Init(this, convTilingData, hasScale);

    conv.Init(convTilingData);

    if constexpr (C_FORMAT == ConvFormat::NCDHW) {
        if constexpr (isMMode) {
            if (!InitSingleCoreData(convTilingData->convRunInfo.mDim, 1, 0)) {
                return false;
            }
        } else {
            if (!InitSingleCoreData(convTilingData->convRunInfo.hoDim, 0, 1)) {
                return false;
            }
        }
    } else {
        if constexpr (isMMode) {
            if (!InitSingleCoreData(1, convTilingData->convRunInfo.nDim, 0)) {
                return false;
            }
        } else {
            if (!InitSingleCoreData(1, 0, convTilingData->convRunInfo.nDim)) {
                return false;
            }
        }
    }

    InitBuffer(x, filter, bias, y, extendParams);

    return true;
}

template <class FMAP_TYPE, class WEIGHT_TYPE, class OUTPUT_TYPE, class BIAS_TYPE, class SCALE_TYPE, class CONV_CFG>
__aicore__ inline bool Conv3dV2Base<FMAP_TYPE, WEIGHT_TYPE, OUTPUT_TYPE, BIAS_TYPE, SCALE_TYPE, CONV_CFG>::
    InitSingleCoreData(uint32_t blockPerNDim, uint32_t blockPerMDim, uint32_t blockPerHoDim)
{
    const uint32_t dataPerBatchDim = convTilingData->convRunInfo.hoDim * convTilingData->convRunInfo.nDim * convTilingData->convRunInfo.doDim;
    DimDataToFill batchStruct(singleCoreBatch, batchIdxStart, isBatchDimTail);
    bool isRealDim = convCommon.CalcDimData(dataPerBatchDim, convTilingData->convRunInfo.batchDim, convTilingData->convRunInfo.batch,
                                            convTilingData->convRunInfo.batch, batchStruct);
    if (unlikely(!isRealDim)) {
        return false;
    }

    const uint32_t dataPerDoDim = convTilingData->convRunInfo.nDim * convTilingData->convRunInfo.hoDim;
    DimDataToFill doStruct(singleCoreDout, doIdxStart, isDoDimTail);
    isRealDim = convCommon.CalcDimData(dataPerDoDim, convTilingData->convRunInfo.doDim, convTilingData->convRunInfo.dout, convTilingData->convRunInfo.dout,
                                       doStruct);
    if (unlikely(!isRealDim)) {
        return false;
    }

    DimDataToFill nStruct(singleCoreN, nIdxStart, isNDimTail);
    isRealDim = convCommon.CalcNDimDataAlign(blockPerNDim, convTilingData->convRunInfo.nDim, convTilingData->convRunInfo.cout, nStruct);
    if (unlikely(!isRealDim)) {
        return false;
    }

    if constexpr (isMMode) {
        DimDataToFill mStruct(singleCoreM, mIdxStart, isMDimTail);
        uint64_t totalM = convTilingData->convRunInfo.hout * convTilingData->convRunInfo.wout;
        isRealDim = convCommon.CalcDimData(blockPerMDim, convTilingData->convRunInfo.mDim,
            convCommon.AlignB(totalM, M0), totalM, mStruct);
    } else {
        DimDataToFill hoToFill(singleCoreHo, hoIdxStart, isHoDimTail);
        isRealDim = convCommon.CalcDimData(blockPerHoDim, convTilingData->convRunInfo.hoDim, convTilingData->convRunInfo.hout,
            convTilingData->convRunInfo.hout, hoToFill);
    }
    if (unlikely(!isRealDim)) {
        return false;
    }

    return true;
}

template <class FMAP_TYPE, class WEIGHT_TYPE, class OUTPUT_TYPE, class BIAS_TYPE, class SCALE_TYPE, class CONV_CFG>
__aicore__ inline void Conv3dV2Base<FMAP_TYPE, WEIGHT_TYPE, OUTPUT_TYPE, BIAS_TYPE, SCALE_TYPE, CONV_CFG>::
    InitBuffer(GM_ADDR x, GM_ADDR filter, GM_ADDR bias, GM_ADDR y, const ExtendParams* extendParams)
{
    int64_t diIdxStart = doIdxStart * convTilingData->convRunInfo.strideD - convTilingData->convRunInfo.padHead;
    diIdxStart = convCommon.Max(diIdxStart, 0);
    if constexpr (A_FORMAT == ConvFormat::NCDHW && C_FORMAT == ConvFormat::NDHWC) {
        convCommon.CalcStartAddrDeQuant(convTilingData->convRunInfo.din, convTilingData->convRunInfo.dout, convTilingData->convRunInfo.kd,
                                        doIdxStart, diIdxStart);
    } else if constexpr (A_FORMAT == ConvFormat::NCDHW) {
        if constexpr (isMMode) {
            convCommon.CalcStartAddrMMode(convTilingData->convRunInfo.din, convTilingData->convRunInfo.dout, convTilingData->convRunInfo.kd,
                doIdxStart, diIdxStart);
        } else {
            convCommon.CalcStartAddrHWode(convTilingData->convRunInfo.din, convTilingData->convRunInfo.dout, convTilingData->convRunInfo.kd,
                doIdxStart, diIdxStart);
        }
    } else {
        if constexpr (isMMode) {
            convCommon.CalcStartAddrMModeHWC(convTilingData->convRunInfo.din, convTilingData->convRunInfo.dout, doIdxStart, diIdxStart);
        } else {
            convCommon.CalcStartAddrHWodeHWC(convTilingData->convRunInfo.din, convTilingData->convRunInfo.dout, doIdxStart, diIdxStart);
        }
    }
    convCommon.InitBufferCommon(x, filter, bias, y, extendParams);
}

template <class FMAP_TYPE, class WEIGHT_TYPE, class OUTPUT_TYPE, class BIAS_TYPE, class SCALE_TYPE, class CONV_CFG>
__aicore__ inline void Conv3dV2Base<FMAP_TYPE, WEIGHT_TYPE, OUTPUT_TYPE, BIAS_TYPE, SCALE_TYPE, CONV_CFG>::
    Conv3dV2KernelImpl()
{
    int64_t diIdxStart = doIdxStart * convTilingData->convRunInfo.strideD;
    if constexpr (isMMode) {
        if (unlikely(isDoDimTail || isNDimTail || isMDimTail || isBatchDimTail)) {
            conv.SetSingleOutputShape(singleCoreN, singleCoreDout, singleCoreM, singleCoreBatch);
        }
        conv.SetFmapStartPosition(diIdxStart, mIdxStart, 0);
    } else {
        if (unlikely(this->isDoDimTail || this->isNDimTail || this->isHoDimTail || isBatchDimTail)) {
            conv.SetSingleOutputShape(this->singleCoreN, this->singleCoreDout, this->singleCoreHo,
                convTilingData->convRunInfo.wout, singleCoreBatch);
        }
        conv.SetFmapStartPosition(diIdxStart, this->singleCoreHiStartPos, 0, 0);
    }

    conv.SetWeight(filterGm);
    if (convTilingData->convRunInfo.hasBias) {
        conv.SetBias(biasGm);
    }
    if constexpr (isQuant) {
        if (hasScale) {
            conv.SetScale(scaleGm);
        }
    }
    conv.SetFmap(fmapGm);
    conv.IterateAll(outputGm);
    conv.End();
}

#endif // CONV3D_V2_H
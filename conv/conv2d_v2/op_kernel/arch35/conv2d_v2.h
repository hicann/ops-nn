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
 * \file conv2_dv2.h
 * \brief
 */

#ifndef CONV_2D_H
#define CONV_2D_H

#include "kernel_tiling/kernel_tiling.h"
#include "../../common/arch35/conv_common.h"
#include "conv2d_v2_api.h"

#define mDim hoDim

using namespace AscendC;
using namespace conv2d;

template<int8_t FmapTiling, int8_t WeightTiling, int8_t L1PingPong, int8_t L0PingPong, int8_t OutputOrder,
        int8_t IterOrder, int8_t GroupType, int8_t EnableSmallChannel, int8_t WeightUbTrans, int8_t FmapCopyMode,
        int8_t InnerBatch, IsExtendConv2D ExtendConv2DFlag = IsExtendConv2D::FALSE, typename Output1T = half>
struct Conv2DV1Param : public Conv2dParam {
    __aicore__ inline Conv2DV1Param() {}
    constexpr static int8_t fmapTiling = FmapTiling;
    constexpr static int8_t weightTiling = WeightTiling;
    constexpr static int8_t l1PingPong = L1PingPong;
    constexpr static int8_t l0PingPong = L0PingPong;
    constexpr static int8_t outputOrder = OutputOrder;
    constexpr static int8_t iterOrder = IterOrder;
    constexpr static int8_t groupType = GroupType;
    constexpr static int8_t enableSmallChannel = EnableSmallChannel;
    constexpr static int8_t weightUbTrans = WeightUbTrans;
    constexpr static int8_t fmapCopyMode = FmapCopyMode;
    constexpr static int8_t innerBatch = InnerBatch;
    constexpr static int8_t isExtendConv2d = static_cast<int8_t>(ExtendConv2DFlag);
    using Output1Dtype = Output1T;
};

template <class FMAP_TYPE, class WEIGHT_TYPE, class OUTPUT_TYPE, class BIAS_TYPE, class SCALE_TYPE, class CONV_CFG>
class Conv2dBase {
public:
    __aicore__ inline Conv2dBase() {}

    __aicore__ inline void RunConv2dKernel(GM_ADDR x, GM_ADDR filter, GM_ADDR bias, GM_ADDR y,
                                           const Conv2DTilingData& conv2dTilingData,
                                           const ExtendParams* extendParams = nullptr);

protected:
    // Conv2d op kernel init intf
    __aicore__ inline bool Conv2dKernelInit(GM_ADDR x, GM_ADDR filter, GM_ADDR bias, GM_ADDR y,
                                            const ExtendParams* extendParams, const Conv2DTilingData& conv2dTilingData);

    __aicore__ inline void InitTilingData(const Conv2DTilingData& conv2dTilingData);

    __aicore__ inline bool InitSingleCoreData(uint32_t blockPerNDim, uint32_t blockPerMDim, uint32_t blockPerHoDim,
                                              uint32_t blockPerWoDim);

    __aicore__ inline void InitBuffer(GM_ADDR x, GM_ADDR filter, GM_ADDR bias, GM_ADDR y,
                                      const ExtendParams* extendParams);

    // Conv2d op kerenle impl intf
    __aicore__ inline void Conv2dKernelImpl();

public:
    // Get dtype
    using FMAP_T = typename FMAP_TYPE::T;
    using WEIGHT_T = typename WEIGHT_TYPE::T;
    using OUTPUT_T = typename OUTPUT_TYPE::T;
    using OUTPUT1_T = typename CONV_CFG::Output1Dtype;
    using BIAS_T = typename BIAS_TYPE::T;
    using SCALE_T = typename SCALE_TYPE::T;
    using RELU_WEIGHT_T = float;
    using CLIP_VALUE_0_T = typename OUTPUT_TYPE::T;
    using CLIP_VALUE_1_T = typename CONV_CFG::Output1Dtype;

    // Conv2d API
    Conv2d<FMAP_TYPE, WEIGHT_TYPE, OUTPUT_TYPE, BIAS_TYPE, SCALE_TYPE, CONV_CFG> conv;
    ConvCommon<Conv2dBase, Conv2DRunInfo> convCommon;

    const TConv2DTiling* conv2dApiTiling;
    const Conv2DRunInfo* conv2dRunInfo;

    // Input and output tensor declare
    GlobalTensor<FMAP_T> fmapGm;
    GlobalTensor<WEIGHT_T> filterGm;
    GlobalTensor<OUTPUT_T> outputGm;
    GlobalTensor<OUTPUT1_T> output1Gm; // extend conv2d dural ouput
    GlobalTensor<BIAS_T> biasGm;
    GlobalTensor<SCALE_T> scaleGm;
    Extendconv2dFixpipeParams<SCALE_T, RELU_WEIGHT_T, CLIP_VALUE_0_T, CLIP_VALUE_1_T> fixpipeParams;

     uint64_t batchIdxStart = 0;
    uint64_t nIdxStart = 0;
    uint64_t mIdxStart = 0;
    uint64_t hoIdxStart = 0;
    uint64_t woIdxStart = 0;

    // Single core data
    uint64_t singleCoreBatch = 0;
    uint64_t singleCoreCi = 0;
    uint64_t singleCoreN = 0;
    uint64_t singleCoreM = 0;
    uint64_t singleCoreHo = 0;
    uint64_t singleCoreWo = 0;
    int64_t singleCoreHiStartPos = 0;
    int64_t singleCoreWiStartPos = 0;

    // block num
    uint32_t batchBlockNums = 0;
    uint32_t nBlockNums = 0;
    uint32_t hoBlockNums = 0; // also mBlockNums
    uint32_t woBlockNums = 0;

    bool isBatchDimTail = false;
    bool isNDimTail = false;
    bool isMDimTail = false;
    bool isHoDimTail = false;
    bool isWoDimTail = false;
    bool hasScale = false;
    bool dualOutput = false;
    uint64_t fmapOneBatchSize = 0;
    uint64_t outputOneBatchSize = 0;

    constexpr static uint32_t din = 1;
    constexpr static uint32_t dout = 1;
    constexpr static uint32_t kd = 1;

    constexpr static ConvFormat A_FORMAT = FMAP_TYPE::format;
    constexpr static ConvFormat B_FORMAT = WEIGHT_TYPE::format;
    constexpr static bool isMMode = CONV_CFG::outputOrder == static_cast<int8_t>(ConvOutputOrder::M_MODE);
    constexpr static bool isQuant = (IsSameType<FMAP_T, int8_t>::value && IsSameType<OUTPUT_T, half>::value) ||
                                    (IsSameType<FMAP_T, int8_t>::value && IsSameType<OUTPUT_T, int8_t>::value) ||
                                    (IsSameType<FMAP_T, hifloat8_t>::value) ||
                                    (IsSameType<FMAP_T, fp8_e4m3fn_t>::value);
    constexpr static int8_t IS_EXTEND_CONV2D = CONV_CFG::isExtendConv2d;

    constexpr static uint8_t k0 = SINGLE_BLOCK_SIZE / sizeof(WEIGHT_T);
};

template <class FMAP_TYPE, class WEIGHT_TYPE, class OUTPUT_TYPE, class BIAS_TYPE, class SCALE_TYPE, class CONV_CFG>
__aicore__ inline void Conv2dBase<FMAP_TYPE, WEIGHT_TYPE, OUTPUT_TYPE, BIAS_TYPE, SCALE_TYPE, CONV_CFG>::
    RunConv2dKernel(GM_ADDR x, GM_ADDR filter, GM_ADDR bias, GM_ADDR y, const Conv2DTilingData& conv2dTilingData,
                    const ExtendParams* extendParams)
{
    if (Conv2dKernelInit(x, filter, bias, y, extendParams, conv2dTilingData)) {
        Conv2dKernelImpl();
    }
}

template <class FMAP_TYPE, class WEIGHT_TYPE, class OUTPUT_TYPE, class BIAS_TYPE, class SCALE_TYPE, class CONV_CFG>
__aicore__ inline bool Conv2dBase<FMAP_TYPE, WEIGHT_TYPE, OUTPUT_TYPE, BIAS_TYPE, SCALE_TYPE, CONV_CFG>::
    Conv2dKernelInit(GM_ADDR x, GM_ADDR filter, GM_ADDR bias, GM_ADDR y, const ExtendParams* extendParams,
                     const Conv2DTilingData& conv2dTilingData)
{
    hasScale = (extendParams != nullptr);
    InitTilingData(conv2dTilingData);
    if constexpr (CONV_CFG::isExtendConv2d) {
        dualOutput = conv2dApiTiling->dualOutput;
    }
    convCommon.Init(this, conv2dRunInfo, hasScale);

    conv.Init(conv2dApiTiling);

    if constexpr (A_FORMAT == ConvFormat::NCHW) {
        if constexpr (isMMode) {
            if (!InitSingleCoreData(conv2dRunInfo->mDim, 1, 0, 0)) {
                return false;
            }
        } else {
            if (!InitSingleCoreData(conv2dRunInfo->hoDim * conv2dRunInfo->woDim, 0, conv2dRunInfo->woDim, 1)) {
                return false;
            }
        }
    } else {
        if constexpr (isMMode) {
            if (!InitSingleCoreData(1, conv2dRunInfo->nDim, 0, 0)) {
                return false;
            }
        } else {
            if (!InitSingleCoreData(1, 0, conv2dRunInfo->woDim * conv2dRunInfo->nDim, conv2dRunInfo->nDim)) {
                return false;
            }
        }
    }

    InitBuffer(x, filter, bias, y, extendParams);

    return true;
}

template <class FMAP_TYPE, class WEIGHT_TYPE, class OUTPUT_TYPE, class BIAS_TYPE, class SCALE_TYPE, class CONV_CFG>
__aicore__ inline void Conv2dBase<FMAP_TYPE, WEIGHT_TYPE, OUTPUT_TYPE, BIAS_TYPE, SCALE_TYPE, CONV_CFG>::
    InitTilingData(const Conv2DTilingData& conv2dTilingData)
{
    conv2dApiTiling = &(conv2dTilingData.conv2dApiTiling);
    conv2dRunInfo = &(conv2dTilingData.conv2dRunInfo);
}

template <class FMAP_TYPE, class WEIGHT_TYPE, class OUTPUT_TYPE, class BIAS_TYPE, class SCALE_TYPE, class CONV_CFG>
__aicore__ inline bool Conv2dBase<FMAP_TYPE, WEIGHT_TYPE, OUTPUT_TYPE, BIAS_TYPE, SCALE_TYPE, CONV_CFG>::
    InitSingleCoreData(uint32_t blockPerNDim, uint32_t blockPerMDim, uint32_t blockPerHoDim, uint32_t blockPerWoDim)
{
    DimDataToFill batchToFill(singleCoreBatch, batchIdxStart, isBatchDimTail);
    bool isRealDim =
        convCommon.CalcDimData(conv2dRunInfo->hoDim * conv2dRunInfo->nDim * conv2dRunInfo->woDim,
                               conv2dRunInfo->batchDim, conv2dRunInfo->batch, conv2dRunInfo->batch, batchToFill);
    if (unlikely(!isRealDim)) {
        return false;
    }

    DimDataToFill nToFill(singleCoreN, nIdxStart, isNDimTail);
    if constexpr (B_FORMAT == ConvFormat::FRACTAL_Z || B_FORMAT == ConvFormat::FRACTAL_Z_C04) {
        isRealDim =
            convCommon.CalcNDimDataWeightNZ(blockPerNDim, conv2dRunInfo->nDim,
                                            convCommon.AlignB(conv2dRunInfo->cout, N0), conv2dRunInfo->cout, nToFill);
    } else {
        isRealDim = convCommon.CalcDimData(blockPerNDim, conv2dRunInfo->nDim,
                                           convCommon.AlignB(conv2dRunInfo->cout, N0), conv2dRunInfo->cout, nToFill);
    }
    if (unlikely(!isRealDim)) {
        return false;
    }

    if constexpr (isMMode) {
        DimDataToFill mToFill(singleCoreM, mIdxStart, isMDimTail);
        uint64_t totalM = conv2dRunInfo->hout * conv2dRunInfo->wout;
        isRealDim = convCommon.CalcDimData(blockPerMDim, conv2dRunInfo->mDim, convCommon.AlignB(totalM, M0),
                                           totalM, mToFill);
        if (unlikely(!isRealDim)) {
            return false;
        }
    } else {
        DimDataToFill hoToFill(singleCoreHo, hoIdxStart, isHoDimTail);
        isRealDim = convCommon.CalcDimData(blockPerHoDim, conv2dRunInfo->hoDim, conv2dRunInfo->hout,
                                           conv2dRunInfo->hout, hoToFill);
        if (unlikely(!isRealDim)) {
            return false;
        }

        DimDataToFill woToFill(singleCoreWo, woIdxStart, isWoDimTail);
        isRealDim = convCommon.CalcDimData(blockPerWoDim, conv2dRunInfo->woDim, conv2dRunInfo->wout,
                                           conv2dRunInfo->wout, woToFill);
        if (unlikely(!isRealDim)) {
            return false;
        }
    }

    return true;
}

template <class FMAP_TYPE, class WEIGHT_TYPE, class OUTPUT_TYPE, class BIAS_TYPE, class SCALE_TYPE, class CONV_CFG>
__aicore__ inline void Conv2dBase<FMAP_TYPE, WEIGHT_TYPE, OUTPUT_TYPE, BIAS_TYPE, SCALE_TYPE, CONV_CFG>::
    InitBuffer(GM_ADDR x, GM_ADDR filter, GM_ADDR bias, GM_ADDR y, const ExtendParams* extendParams)
{
    if constexpr (A_FORMAT == ConvFormat::NCHW) {
        if constexpr (isMMode) {
            convCommon.CalcStartAddrMMode(din, dout, kd);
        } else {
            convCommon.CalcStartAddrHWode(din, dout, kd);
        }
    } else {
        if constexpr (isMMode) {
            convCommon.CalcStartAddrMModeHWC(din, dout);
        } else {
            convCommon.CalcStartAddrHWodeHWC(din, dout);
        }
    }

    convCommon.InitBufferCommon(x, filter, bias, y, extendParams);
}

template <class FMAP_TYPE, class WEIGHT_TYPE, class OUTPUT_TYPE, class BIAS_TYPE, class SCALE_TYPE, class CONV_CFG>
__aicore__ inline void Conv2dBase<FMAP_TYPE, WEIGHT_TYPE, OUTPUT_TYPE, BIAS_TYPE, SCALE_TYPE, CONV_CFG>::
    Conv2dKernelImpl()
{
    if constexpr (isMMode) {
        if (unlikely(isNDimTail || isMDimTail || isBatchDimTail)) {
            conv.SetSingleOutputShape(singleCoreN, singleCoreM, singleCoreBatch);
        }
 
        conv.SetFmapStartPosition(mIdxStart);
    } else {
        if (unlikely(isNDimTail || isHoDimTail || isWoDimTail || isBatchDimTail)) {
            conv.SetSingleOutputShape(singleCoreN, singleCoreHo, singleCoreWo, singleCoreBatch);
        }

        conv.SetFmapStartPosition(singleCoreHiStartPos, singleCoreWiStartPos);
    }

    conv.SetWeight(filterGm);
    if (conv2dRunInfo->hasBias) {
        conv.SetBias(biasGm);
    }

    if constexpr (isQuant || CONV_CFG::isExtendConv2d) {
        if (hasScale) {
            conv.SetFixpipeParams(fixpipeParams);
        }
    }

    conv.SetFmap(fmapGm);
    if constexpr (CONV_CFG::isExtendConv2d) {
        if (dualOutput) {
            conv.IterateAll(outputGm, output1Gm);
        } else {
            conv.IterateAll(outputGm);
        }
    } else {
        conv.IterateAll(outputGm);
    }

    conv.End();
}

#endif // CONV_2D_H
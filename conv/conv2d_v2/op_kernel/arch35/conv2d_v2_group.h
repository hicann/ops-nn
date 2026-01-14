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
 * \file conv2d_v2_group.h
 * \brief
 */

#ifndef CONV2D_V2_GROUP_H
#define CONV2D_V2_GROUP_H

#include "conv2d_v2.h"
#include "../../common/arch35/conv_group_common.h"

template <class FMAP_TYPE, class WEIGHT_TYPE, class OUTPUT_TYPE, class BIAS_TYPE, class SCALE_TYPE, class CONV_CFG>
class GroupConv2d : public Conv2dBase<FMAP_TYPE, WEIGHT_TYPE, OUTPUT_TYPE, BIAS_TYPE, SCALE_TYPE, CONV_CFG> {
public:
    __aicore__ inline GroupConv2d() {}

    __aicore__ inline void RunConv2dKernel(GM_ADDR x, GM_ADDR filter, GM_ADDR bias, GM_ADDR y,
                                           const Conv2DTilingData& conv2dTilingData,
                                           const ExtendParams* extendParams = nullptr);
private:
    __aicore__ inline bool Conv2dKernelInit(GM_ADDR x, GM_ADDR filter, GM_ADDR bias, GM_ADDR y,
                                            const ExtendParams* extendParams, const Conv2DTilingData& conv2dTilingData);

    __aicore__ inline bool InitSingleCoreData();
    __aicore__ inline bool InitSingleCoreDataOriGroup();
    __aicore__ inline bool InitSingleCoreDataOptGroup();
    __aicore__ inline void InitBlockNums();

    __aicore__ inline void InitBuffer(GM_ADDR x, GM_ADDR filter, GM_ADDR bias, GM_ADDR y,
                                      const ExtendParams* extendParams);

    __aicore__ inline void Conv2dKernelImpl();

public:
    using FMAP_T = typename FMAP_TYPE::T;
    using WEIGHT_T = typename WEIGHT_TYPE::T;
    using OUTPUT_T = typename OUTPUT_TYPE::T;
    using OUTPUT1_T = typename CONV_CFG::Output1Dtype;
    using BIAS_T = typename BIAS_TYPE::T;
    using SCALE_T = uint64_t;
    using RELU_WEIGHT_T = float;
    using CLIP_VALUE_0_T = typename OUTPUT_TYPE::T;
    using CLIP_VALUE_1_T = typename CONV_CFG::Output1Dtype;

    using Conv2dBase<FMAP_TYPE, WEIGHT_TYPE, OUTPUT_TYPE, BIAS_TYPE, SCALE_TYPE, CONV_CFG>::conv;
    using Conv2dBase<FMAP_TYPE, WEIGHT_TYPE, OUTPUT_TYPE, BIAS_TYPE, SCALE_TYPE, CONV_CFG>::conv2dRunInfo;
    using Conv2dBase<FMAP_TYPE, WEIGHT_TYPE, OUTPUT_TYPE, BIAS_TYPE, SCALE_TYPE, CONV_CFG>::isMMode;
    using Conv2dBase<FMAP_TYPE, WEIGHT_TYPE, OUTPUT_TYPE, BIAS_TYPE, SCALE_TYPE, CONV_CFG>::din;
    using Conv2dBase<FMAP_TYPE, WEIGHT_TYPE, OUTPUT_TYPE, BIAS_TYPE, SCALE_TYPE, CONV_CFG>::dout;
    using Conv2dBase<FMAP_TYPE, WEIGHT_TYPE, OUTPUT_TYPE, BIAS_TYPE, SCALE_TYPE, CONV_CFG>::kd;
    using Conv2dBase<FMAP_TYPE, WEIGHT_TYPE, OUTPUT_TYPE, BIAS_TYPE, SCALE_TYPE, CONV_CFG>::A_FORMAT;
    using Conv2dBase<FMAP_TYPE, WEIGHT_TYPE, OUTPUT_TYPE, BIAS_TYPE, SCALE_TYPE, CONV_CFG>::fixpipeParams;
    using Conv2dBase<FMAP_TYPE, WEIGHT_TYPE, OUTPUT_TYPE, BIAS_TYPE, SCALE_TYPE, CONV_CFG>::IS_EXTEND_CONV2D;

    ConvGroupCommon<GroupConv2d, Conv2DRunInfo> convCommon;

    uint32_t ciPerGroup = 0;
    uint32_t coPerGroup = 0;

    uint64_t groupIdxStart = 0;
    uint64_t singleGroups = 0;
    uint64_t singleGroupOpt = 0;
    uint64_t singleCoOpt = 0;
    bool isGroupDimTail = false;
    bool hasScale = false;
    bool dualOutput = false;

    uint32_t groupBlockNums;

    constexpr static bool isOptGroup = CONV_CFG::groupType == static_cast<int8_t>(ConvGroupType::OPT_GROUP_CONV);
};

template <class FMAP_TYPE, class WEIGHT_TYPE, class OUTPUT_TYPE, class BIAS_TYPE, class SCALE_TYPE, class CONV_CFG>
__aicore__ inline void GroupConv2d<FMAP_TYPE, WEIGHT_TYPE, OUTPUT_TYPE, BIAS_TYPE, SCALE_TYPE, CONV_CFG>::
    RunConv2dKernel(GM_ADDR x, GM_ADDR filter, GM_ADDR bias, GM_ADDR y, const Conv2DTilingData& conv2dTilingData,
                    const ExtendParams* extendParams)
{
    if (Conv2dKernelInit(x, filter, bias, y, extendParams, conv2dTilingData)) {
        Conv2dKernelImpl();
    }
}

template <class FMAP_TYPE, class WEIGHT_TYPE, class OUTPUT_TYPE, class BIAS_TYPE, class SCALE_TYPE, class CONV_CFG>
__aicore__ inline bool GroupConv2d<FMAP_TYPE, WEIGHT_TYPE, OUTPUT_TYPE, BIAS_TYPE, SCALE_TYPE, CONV_CFG>::
    Conv2dKernelInit(GM_ADDR x, GM_ADDR filter, GM_ADDR bias, GM_ADDR y, const ExtendParams* extendParams,
                     const Conv2DTilingData& conv2dTilingData)
{
    hasScale = (extendParams != nullptr);
    this->InitTilingData(conv2dTilingData);
    if constexpr (CONV_CFG::isExtendConv2d) {
        dualOutput = this->conv2dApiTiling->dualOutput;
    }
    convCommon.Init(this, conv2dRunInfo, hasScale);

    conv.Init(this->conv2dApiTiling);

    ciPerGroup = conv2dRunInfo->cin / conv2dRunInfo->groups;
    coPerGroup = conv2dRunInfo->cout / conv2dRunInfo->groups;

    if (!InitSingleCoreData()) {
        return false;
    }

    InitBuffer(x, filter, bias, y, extendParams);

    return true;
}

template <class FMAP_TYPE, class WEIGHT_TYPE, class OUTPUT_TYPE, class BIAS_TYPE, class SCALE_TYPE, class CONV_CFG>
__aicore__ inline bool GroupConv2d<FMAP_TYPE, WEIGHT_TYPE, OUTPUT_TYPE, BIAS_TYPE, SCALE_TYPE, CONV_CFG>::
    InitSingleCoreData()
{
    InitBlockNums();

    DimDataToFill batchToFill(this->singleCoreBatch, this->batchIdxStart, this->isBatchDimTail);
    bool isRealDim = convCommon.CalcDimData(this->batchBlockNums, conv2dRunInfo->batchDim, conv2dRunInfo->batch,
                                            conv2dRunInfo->batch, batchToFill);
    if (unlikely(!isRealDim)) {
        return false;
    }

    if constexpr (isMMode) {
        DimDataToFill mToFill(this->singleCoreM, this->mIdxStart, this->isMDimTail);
        uint64_t totalM = conv2dRunInfo->hout * conv2dRunInfo->wout;
        isRealDim = convCommon.CalcDimData(this->hoBlockNums, conv2dRunInfo->hoDim, convCommon.AlignB(totalM, M0),
                                           totalM, mToFill);
        if (unlikely(!isRealDim)) {
            return false;
        }
    } else {
        DimDataToFill hoToFill(this->singleCoreHo, this->hoIdxStart, this->isHoDimTail);
        isRealDim = convCommon.CalcDimData(this->hoBlockNums, conv2dRunInfo->hoDim, conv2dRunInfo->hout,
                                            conv2dRunInfo->hout, hoToFill);
        if (unlikely(!isRealDim)) {
            return false;
        }
        DimDataToFill woToFill(this->singleCoreWo, this->woIdxStart, this->isWoDimTail);
        isRealDim = convCommon.CalcDimData(this->woBlockNums, conv2dRunInfo->woDim, conv2dRunInfo->wout,
                                           conv2dRunInfo->wout, woToFill);
        if (unlikely(!isRealDim)) {
            return false;
        }
    }

    if constexpr (isOptGroup) {
        return InitSingleCoreDataOptGroup();
    } else {
        return InitSingleCoreDataOriGroup();
    }

    return true;
}

template <class FMAP_TYPE, class WEIGHT_TYPE, class OUTPUT_TYPE, class BIAS_TYPE, class SCALE_TYPE, class CONV_CFG>
__aicore__ inline void GroupConv2d<FMAP_TYPE, WEIGHT_TYPE, OUTPUT_TYPE, BIAS_TYPE, SCALE_TYPE, CONV_CFG>::
    InitBlockNums()
{
    // group can be seem as divided axis of n
    // NCHW: batchDim -> 'groupDim -> nDim' -> hoDim -> woDim / batchDim -> 'groupDim -> nDim' -> mDim
    // NHWC: batchDim -> hoDim -> woDim -> 'groupDim -> nDim' / batchDim -> mDim -> 'groupDim -> nDim'
    if constexpr (isMMode) {
        this->batchBlockNums = conv2dRunInfo->groupDim * conv2dRunInfo->nDim * conv2dRunInfo->hoDim;
        if constexpr (A_FORMAT == ConvFormat::NCHW) {
            this->groupBlockNums = conv2dRunInfo->nDim * conv2dRunInfo->hoDim;
            this->nBlockNums = conv2dRunInfo->hoDim;
            this->hoBlockNums = 1;
        } else {
            this->hoBlockNums = conv2dRunInfo->groupDim * conv2dRunInfo->nDim;
            this->groupBlockNums = conv2dRunInfo->nDim;
            this->nBlockNums = 1;
        }
    } else {
        this->batchBlockNums = conv2dRunInfo->groupDim * conv2dRunInfo->nDim * conv2dRunInfo->hoDim *
                               conv2dRunInfo->woDim;
        if constexpr (A_FORMAT == ConvFormat::NCHW) {
            this->groupBlockNums = conv2dRunInfo->nDim * conv2dRunInfo->hoDim * conv2dRunInfo->woDim;
            this->nBlockNums = conv2dRunInfo->hoDim * conv2dRunInfo->woDim;
            this->hoBlockNums = conv2dRunInfo->woDim;
            this->woBlockNums = 1;
        } else {
            this->hoBlockNums = conv2dRunInfo->woDim * conv2dRunInfo->groupDim * conv2dRunInfo->nDim;
            this->woBlockNums = conv2dRunInfo->groupDim * conv2dRunInfo->nDim;
            this->groupBlockNums = conv2dRunInfo->nDim;
            this->nBlockNums = 1;
        }
    }
}

template <class FMAP_TYPE, class WEIGHT_TYPE, class OUTPUT_TYPE, class BIAS_TYPE, class SCALE_TYPE, class CONV_CFG>
__aicore__ inline bool GroupConv2d<FMAP_TYPE, WEIGHT_TYPE, OUTPUT_TYPE, BIAS_TYPE, SCALE_TYPE, CONV_CFG>::
    InitSingleCoreDataOriGroup()
{
    DimDataToFill groupToFill(singleGroups, groupIdxStart, isGroupDimTail);
    bool isRealDim = convCommon.CalcDimData(this->groupBlockNums, conv2dRunInfo->groupDim,
                                            conv2dRunInfo->groups, conv2dRunInfo->groups, groupToFill);
    if (unlikely(!isRealDim)) {
        return false;
    }

    DimDataToFill nToFill(this->singleCoreN, this->nIdxStart, this->isNDimTail);
    isRealDim = convCommon.CalcDimData(this->nBlockNums, conv2dRunInfo->nDim, convCommon.AlignB(coPerGroup, N0),
                                        coPerGroup, nToFill);
    if (unlikely(!isRealDim)) {
        return false;
    }

    return true;
}

template <class FMAP_TYPE, class WEIGHT_TYPE, class OUTPUT_TYPE, class BIAS_TYPE, class SCALE_TYPE, class CONV_CFG>
__aicore__ inline bool GroupConv2d<FMAP_TYPE, WEIGHT_TYPE, OUTPUT_TYPE, BIAS_TYPE, SCALE_TYPE, CONV_CFG>::
    InitSingleCoreDataOptGroup()
{
    DimDataToFill groupToFill(singleGroupOpt, groupIdxStart, isGroupDimTail);
    bool isRealDim = convCommon.CalcDimData(this->groupBlockNums, conv2dRunInfo->groupDim,
                                            conv2dRunInfo->groupOpt, conv2dRunInfo->groupOpt, groupToFill);
    if (unlikely(!isRealDim)) {
        return false;
    }

    DimDataToFill nToFill(singleCoOpt, this->nIdxStart, this->isNDimTail);
    isRealDim = convCommon.CalcDimData(this->nBlockNums, conv2dRunInfo->nDim,
                                        convCommon.AlignB(conv2dRunInfo->coutOpt, N0),
                                        conv2dRunInfo->coutOpt, nToFill);
    if (unlikely(!isRealDim)) {
        return false;
    }

    convCommon.UpdateRealCoutOptGroup();

    return true;
}

template <class FMAP_TYPE, class WEIGHT_TYPE, class OUTPUT_TYPE, class BIAS_TYPE, class SCALE_TYPE, class CONV_CFG>
__aicore__ inline void GroupConv2d<FMAP_TYPE, WEIGHT_TYPE, OUTPUT_TYPE, BIAS_TYPE, SCALE_TYPE, CONV_CFG>::
    InitBuffer(GM_ADDR x, GM_ADDR filter, GM_ADDR bias, GM_ADDR y, const ExtendParams* extendParams)
{
    if constexpr (isOptGroup) {
        convCommon.CalcStartAddrOptGroup(din, dout, kd);
    } else {
        convCommon.CalcStartAddrOriGroup(din, dout, kd);
    }

    convCommon.InitBufferCommon(x, filter, bias, y, extendParams);
}

template <class FMAP_TYPE, class WEIGHT_TYPE, class OUTPUT_TYPE, class BIAS_TYPE, class SCALE_TYPE, class CONV_CFG>
__aicore__ inline void GroupConv2d<FMAP_TYPE, WEIGHT_TYPE, OUTPUT_TYPE, BIAS_TYPE, SCALE_TYPE, CONV_CFG>::
    Conv2dKernelImpl()
{
    if constexpr (isMMode) {
        if (unlikely(this->isNDimTail || this->isMDimTail || this->isBatchDimTail)) {
            conv.SetSingleOutputShape(this->singleCoreN, this->singleCoreM, this->singleCoreBatch);
        }
 
        conv.SetFmapStartPosition(this->mIdxStart);
    } else {
        if (unlikely(this->isNDimTail || this->isHoDimTail || this->isWoDimTail || this->isBatchDimTail)) {
            conv.SetSingleOutputShape(this->singleCoreN, this->singleCoreHo, this->singleCoreWo, this->singleCoreBatch);
        }

        conv.SetFmapStartPosition(this->singleCoreHiStartPos, this->singleCoreWiStartPos);
    }

    if constexpr (CONV_CFG::isExtendConv2d) {
        conv.SetFixpipeParams(fixpipeParams);
    }

    if constexpr (isOptGroup) {
        convCommon.ConvKernelImplOptGroup();
    } else {
        convCommon.ConvKernelImplOriGroup();
    }
}

#endif // CONV2D_V2_GROUP_H
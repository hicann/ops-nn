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
 * \file conv3d_v2_group.h
 * \brief
 */
 
#ifndef CONV3D_V2_GROUP_H
#define CONV3D_V2_GROUP_H

#include "conv3d_v2.h"
#include "../../common/arch35/conv_group_common.h"
#include "../conv3d_v2_tiling_data.h"

template <class FMAP_TYPE, class WEIGHT_TYPE, class OUTPUT_TYPE, class BIAS_TYPE, class SCALE_TYPE, class CONV_CFG>
class GroupConv3dV2 : public Conv3dV2Base<FMAP_TYPE, WEIGHT_TYPE, OUTPUT_TYPE, BIAS_TYPE, SCALE_TYPE, CONV_CFG> {
public:
    __aicore__ inline GroupConv3dV2() {}

    __aicore__ inline void RunConv3dV2Kernel(GM_ADDR x, GM_ADDR filter, GM_ADDR bias, GM_ADDR y,
                                             const Ops::NN::Conv3dV2::Conv3DV2TilingData& conv3dTilingData,
                                             const ExtendParams* extendParams = nullptr);

private:
    __aicore__ inline bool Conv3dV2KernelInit(GM_ADDR x, GM_ADDR filter, GM_ADDR bias, GM_ADDR y,
                                              const ExtendParams* extendParams,
                                              const Ops::NN::Conv3dV2::Conv3DV2TilingData& conv3dTilingData);

    __aicore__ inline bool InitSingleCoreData();
    __aicore__ inline bool InitSingleCoreDataOriGroup();
    __aicore__ inline bool InitSingleCoreDataOptGroup();
    __aicore__ inline void InitBlockNums();
    __aicore__ inline void InitBuffer(GM_ADDR x, GM_ADDR filter, GM_ADDR bias, GM_ADDR y,
                                      const ExtendParams* extendParams);

    __aicore__ inline void Conv3dV2KernelImpl();

public:
    using FMAP_T = typename FMAP_TYPE::T;
    using WEIGHT_T = typename WEIGHT_TYPE::T;
    using OUTPUT_T = typename OUTPUT_TYPE::T;
    using BIAS_T = typename BIAS_TYPE::T;
    using SCALE_T = typename SCALE_TYPE::T;

    using Conv3dV2Base<FMAP_TYPE, WEIGHT_TYPE, OUTPUT_TYPE, BIAS_TYPE, SCALE_TYPE, CONV_CFG>::conv;
    using Conv3dV2Base<FMAP_TYPE, WEIGHT_TYPE, OUTPUT_TYPE, BIAS_TYPE, SCALE_TYPE, CONV_CFG>::conv3dRunInfo;
    using Conv3dV2Base<FMAP_TYPE, WEIGHT_TYPE, OUTPUT_TYPE, BIAS_TYPE, SCALE_TYPE, CONV_CFG>::isMMode;
    using Conv3dV2Base<FMAP_TYPE, WEIGHT_TYPE, OUTPUT_TYPE, BIAS_TYPE, SCALE_TYPE, CONV_CFG>::A_FORMAT;
    uint32_t ciPerGroup = 0;
    uint32_t coPerGroup = 0;

    uint64_t groupIdxStart = 0;
    uint64_t singleGroups = 0;
    uint64_t singleGroupOpt = 0;
    uint64_t singleCoOpt = 0;
    uint64_t singleCi = 0;
    bool isGroupDimTail = false;
    bool hasScale =false;
    ConvGroupCommon<GroupConv3dV2, Ops::NN::Conv3dV2::Conv3DRunInfo> convCommon;

    constexpr static bool isOptGroup = CONV_CFG::groupType == static_cast<int8_t>(ConvGroupType::OPT_GROUP_CONV);
    uint32_t batchBlockNums = 0;
    uint32_t groupBlockNums = 0;
    uint32_t nBlockNums = 0;
    uint32_t hoBlockNums = 0;
    uint32_t doBlockNums = 0;
};

template <class FMAP_TYPE, class WEIGHT_TYPE, class OUTPUT_TYPE, class BIAS_TYPE, class SCALE_TYPE, class CONV_CFG>
__aicore__ inline void GroupConv3dV2<FMAP_TYPE, WEIGHT_TYPE, OUTPUT_TYPE, BIAS_TYPE, SCALE_TYPE, CONV_CFG>::
    RunConv3dV2Kernel(GM_ADDR x, GM_ADDR filter, GM_ADDR bias, GM_ADDR y,
                      const Ops::NN::Conv3dV2::Conv3DV2TilingData& conv3dTilingData, const ExtendParams* extendParams)
{
    if (Conv3dV2KernelInit(x, filter, bias, y, extendParams, conv3dTilingData)) {
        Conv3dV2KernelImpl();
    }
}

template <class FMAP_TYPE, class WEIGHT_TYPE, class OUTPUT_TYPE, class BIAS_TYPE, class SCALE_TYPE, class CONV_CFG>
__aicore__ inline bool GroupConv3dV2<FMAP_TYPE, WEIGHT_TYPE, OUTPUT_TYPE, BIAS_TYPE, SCALE_TYPE, CONV_CFG>::
    Conv3dV2KernelInit(GM_ADDR x, GM_ADDR filter, GM_ADDR bias, GM_ADDR y, const ExtendParams* extendParams,
                       const Ops::NN::Conv3dV2::Conv3DV2TilingData& conv3dTilingData)
{
    hasScale = (extendParams != nullptr);
    this->InitTilingData(conv3dTilingData);

    convCommon.Init(this, conv3dRunInfo, hasScale);

    conv.Init(this->conv3dApiTiling);

    ciPerGroup = conv3dRunInfo->cin / conv3dRunInfo->groups;
    coPerGroup = conv3dRunInfo->cout / conv3dRunInfo->groups;

    if (!InitSingleCoreData()) {
        return false;
    }

    InitBuffer(x, filter, bias, y, extendParams);

    return true;
}

template <class FMAP_TYPE, class WEIGHT_TYPE, class OUTPUT_TYPE, class BIAS_TYPE, class SCALE_TYPE, class CONV_CFG>
__aicore__ inline void GroupConv3dV2<FMAP_TYPE, WEIGHT_TYPE, OUTPUT_TYPE, BIAS_TYPE, SCALE_TYPE, CONV_CFG>::
    InitBlockNums()
{
    // NCDHW: batchDim -> groupDim -> nDim -> doDim -> hoDim
    // NDHWC: batchDim -> doDim -> hoDim -> groupDim -> nDim
    this->batchBlockNums = conv3dRunInfo->doDim * conv3dRunInfo->groupDim * conv3dRunInfo->nDim * conv3dRunInfo->hoDim;
    if constexpr (A_FORMAT == ConvFormat::NCDHW) {
        this->doBlockNums = conv3dRunInfo->hoDim;
        this->hoBlockNums = 1;
        this->groupBlockNums = conv3dRunInfo->nDim * conv3dRunInfo->hoDim * conv3dRunInfo->doDim;
        this->nBlockNums = conv3dRunInfo->doDim * conv3dRunInfo->hoDim;
    } else {
        this->doBlockNums = conv3dRunInfo->nDim * conv3dRunInfo->hoDim * conv3dRunInfo->groupDim;
        this->hoBlockNums = conv3dRunInfo->nDim * conv3dRunInfo->groupDim;
        this->groupBlockNums = conv3dRunInfo->nDim;
        this->nBlockNums = 1;
    }
}

template <class FMAP_TYPE, class WEIGHT_TYPE, class OUTPUT_TYPE, class BIAS_TYPE, class SCALE_TYPE, class CONV_CFG>
__aicore__ inline bool GroupConv3dV2<FMAP_TYPE, WEIGHT_TYPE, OUTPUT_TYPE, BIAS_TYPE, SCALE_TYPE, CONV_CFG>::
    InitSingleCoreData()
{
    InitBlockNums();
    DimDataToFill batchToFill(this->singleCoreBatch, this->batchIdxStart, this->isBatchDimTail);
    bool isRealDim = convCommon.CalcDimData(this->batchBlockNums, conv3dRunInfo->batchDim, conv3dRunInfo->batch,
                                            conv3dRunInfo->batch, batchToFill);
    if (unlikely(!isRealDim)) {
        return false;
    }

    DimDataToFill doToFill(this->singleCoreDout, this->doIdxStart, this->isDoDimTail);
    isRealDim = convCommon.CalcDimData(this->doBlockNums, conv3dRunInfo->doDim, conv3dRunInfo->dout, conv3dRunInfo->dout,
                                       doToFill);
    if (unlikely(!isRealDim)) {
        return false;
    }

    if constexpr (isMMode) {
        DimDataToFill mToFill(this->singleCoreM, this->mIdxStart, this->isMDimTail);
        uint64_t totalM = conv3dRunInfo->hout * conv3dRunInfo->wout;
        isRealDim = convCommon.CalcDimData(this->hoBlockNums, conv3dRunInfo->hoDim, convCommon.AlignB(totalM, M0),
                                           totalM, mToFill);
    } else {
        DimDataToFill hoToFill(this->singleCoreHo, this->hoIdxStart, this->isHoDimTail);
        isRealDim = convCommon.CalcDimData(this->hoBlockNums, conv3dRunInfo->hoDim, conv3dRunInfo->hout,
                                           conv3dRunInfo->hout, hoToFill);
    }
    if (unlikely(!isRealDim)) {
        return false;
    }

    if constexpr (isOptGroup) {
        return InitSingleCoreDataOptGroup();
    } else {
        return InitSingleCoreDataOriGroup();
    }
}

template <class FMAP_TYPE, class WEIGHT_TYPE, class OUTPUT_TYPE, class BIAS_TYPE, class SCALE_TYPE, class CONV_CFG>
__aicore__ inline bool GroupConv3dV2<FMAP_TYPE, WEIGHT_TYPE, OUTPUT_TYPE, BIAS_TYPE, SCALE_TYPE, CONV_CFG>::
    InitSingleCoreDataOriGroup()
{
    DimDataToFill groupToFill(singleGroups, groupIdxStart, isGroupDimTail);
    bool isRealDim = convCommon.CalcDimData(this->groupBlockNums, conv3dRunInfo->groupDim, conv3dRunInfo->groups,
                                            conv3dRunInfo->groups, groupToFill);
    if (unlikely(!isRealDim)) {
        return false;
    }

    DimDataToFill nToFill(this->singleCoreN, this->nIdxStart, this->isNDimTail);
    isRealDim = convCommon.CalcDimData(this->nBlockNums, conv3dRunInfo->nDim, convCommon.AlignB(coPerGroup, N0),
                                       coPerGroup, nToFill);
    if (unlikely(!isRealDim)) {
        return false;
    }

    return true;
}

template <class FMAP_TYPE, class WEIGHT_TYPE, class OUTPUT_TYPE, class BIAS_TYPE, class SCALE_TYPE, class CONV_CFG>
__aicore__ inline bool GroupConv3dV2<FMAP_TYPE, WEIGHT_TYPE, OUTPUT_TYPE, BIAS_TYPE, SCALE_TYPE, CONV_CFG>::
    InitSingleCoreDataOptGroup()
{
    DimDataToFill groupToFill(singleGroupOpt, groupIdxStart, isGroupDimTail);
    bool isRealDim = convCommon.CalcDimData(this->groupBlockNums, conv3dRunInfo->groupDim, conv3dRunInfo->groupOpt,
                                            conv3dRunInfo->groupOpt, groupToFill);
    if (unlikely(!isRealDim)) {
        return false;
    }

    DimDataToFill nToFill(singleCoOpt, this->nIdxStart, this->isNDimTail);
    isRealDim = convCommon.CalcDimData(this->nBlockNums, conv3dRunInfo->nDim,
                                       convCommon.AlignB(conv3dRunInfo->coutOpt, N0), conv3dRunInfo->coutOpt, nToFill);
    if (unlikely(!isRealDim)) {
        return false;
    }

    convCommon.UpdateRealCoutOptGroup();

    return true;
}

template <class FMAP_TYPE, class WEIGHT_TYPE, class OUTPUT_TYPE, class BIAS_TYPE, class SCALE_TYPE, class CONV_CFG>
__aicore__ inline void GroupConv3dV2<FMAP_TYPE, WEIGHT_TYPE, OUTPUT_TYPE, BIAS_TYPE, SCALE_TYPE, CONV_CFG>::
    InitBuffer(GM_ADDR x, GM_ADDR filter, GM_ADDR bias, GM_ADDR y, const ExtendParams* extendParams)
{
    int64_t diIdxStart = this->doIdxStart * conv3dRunInfo->strideD - conv3dRunInfo->padHead;
    diIdxStart = convCommon.Max(diIdxStart, 0);
    if constexpr (isOptGroup) {
        convCommon.CalcStartAddrOptGroup(conv3dRunInfo->din, conv3dRunInfo->dout, conv3dRunInfo->kd, this->doIdxStart,
                                         diIdxStart);
    } else {
        convCommon.CalcStartAddrOriGroup(conv3dRunInfo->din, conv3dRunInfo->dout, conv3dRunInfo->kd, this->doIdxStart,
                                         diIdxStart);
    }

    convCommon.InitBufferCommon(x, filter, bias, y, extendParams);
}

template <class FMAP_TYPE, class WEIGHT_TYPE, class OUTPUT_TYPE, class BIAS_TYPE, class SCALE_TYPE, class CONV_CFG>
__aicore__ inline void GroupConv3dV2<FMAP_TYPE, WEIGHT_TYPE, OUTPUT_TYPE, BIAS_TYPE, SCALE_TYPE, CONV_CFG>::
    Conv3dV2KernelImpl()
{
    int64_t diIdxStart = this->doIdxStart * conv3dRunInfo->strideD;
    if constexpr (isMMode) {
        if (unlikely(this->isDoDimTail || this->isNDimTail || this->isMDimTail || this->isBatchDimTail)) {
            conv.SetSingleOutputShape(this->singleCoreN, this->singleCoreDout, this->singleCoreM,
                this->singleCoreBatch);
        }
        conv.SetFmapStartPosition(diIdxStart, this->mIdxStart, 0);
    } else {
        if (unlikely(this->isDoDimTail || this->isNDimTail || this->isHoDimTail || this->isBatchDimTail)) {
            conv.SetSingleOutputShape(this->singleCoreN, this->singleCoreDout, this->singleCoreHo,
                conv3dRunInfo->wout, this->singleCoreBatch);
        }
        conv.SetFmapStartPosition(diIdxStart, this->singleCoreHiStartPos, 0, 0);
    }

    if constexpr (isOptGroup) {
        convCommon.ConvKernelImplOptGroup();
    } else {
        convCommon.ConvKernelImplOriGroup();
    }
}

#endif // CONV3D_V2_GROUP_H
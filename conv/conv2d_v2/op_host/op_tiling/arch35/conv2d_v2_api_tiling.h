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
 * \file conv2d_v2_api_tiling.h
 * \brief
 */

#ifndef ASCENDC_TILING_CONV2D_V2_API_TILING_H
#define ASCENDC_TILING_CONV2D_V2_API_TILING_H

#include "../../../../common/op_host/op_tiling/arch35/conv_api_tiling_base.h"
#include "../../../../common/op_host/op_tiling/arch35/conv_api_tiling_algorithm_base.h"
#include "conv/common/op_host/op_tiling/arch35/conv_api_strategy.h"

namespace optiling {
class Conv2DTilingData;
}

namespace conv_tiling {
using optiling::conv_ops_tiling::ConvOptGroupInfo;
using optiling::conv_ops_tiling::ConvOriGroupInfo;
using optiling::conv_ops_tiling::FixpipeInfo;

using Conv2DBasicBlockTilingParams = BasicBlockTilingParams;

using Conv2DBasicBlockMinLoadStrategy = BasicBlockMinLoadStrategy;

using Conv2DBasicBlockInfo = BasicBlockInfo;

class Conv2dTiling : public ConvTilingBase {
public:
    Conv2dTiling() {};
    explicit Conv2dTiling(const PlatformInfo& platform) : ConvTilingBase(platform) {};
    ~Conv2dTiling() override {};
    int64_t GetTiling(optiling::Conv2DTilingData& tiling);
    bool GetTiling(Conv2DBasicBlockInfo& conv2DBasicBlockInfo, optiling::Conv2DTilingData& tiling,
                   std::shared_ptr<conv_tiling::ConvAPIStrategy> apiStrategy);
    int64_t Compute() override;

    void SetOrgWeightShape(int64_t orgCo, int64_t orgkH, int64_t orgkW);
    void SetOrgFmapShape(int64_t orgCi, int64_t orgHi, int64_t orgWi);
    void SetSingleOutputShape(int64_t singleCo, int64_t singleHo, int64_t singleWo, int64_t singleBatch);
    void SetSingleOutputShape(int64_t singleCo, int64_t singleM, int64_t singleBatch);
    void SetSingleWeightShape(int64_t singleCi, int64_t singlekH, int64_t singlekW);
    void SetWeightType(TPosition pos, ConvFormat format, ConvDtype dtype);
    void SetFmapType(TPosition pos, ConvFormat format, ConvDtype dtype);
    void SetBiasType(TPosition pos, ConvFormat format, ConvDtype dtype);
    void SetOutputType(TPosition pos, ConvFormat format, ConvDtype dtype);
    void SetPadding(int32_t padTop, int32_t padBottom, int32_t padLeft, int32_t padRight);
    void SetDilation(int32_t dilationH, int32_t dilationW);
    void SetStride(int32_t strideH, int32_t strideW);
    void SetHF32(bool hf32EnableFlag, bool hf32TransModeFlag);
    void SetQuantScale(bool hasScaleFlag);
    void SetExtendConvFlag(bool extendConvEnable);
    void SetQuantConvFlag(bool quantConvEnable);
    void SetFixpipeParams(FixpipeInfo& fixpipeInfo);
    void SetOffsetx(int8_t offsetx);
    void SetC04Flag(bool isC04Enable);
    void SetRoundMode(int8_t roundMode);
    void SetDisContinuousFlag(bool disContinuous);
    void SetOutputOrder(int8_t outOrder);
    void SetTilingAlgorithmType(int8_t tilingAlgorithmType);
    void SetGroups(int32_t groups);
    void SetOptGroupParams(int32_t enlarge, int64_t singleGroups, int64_t singleGroupOpt);
    void CalcOptGroupParams(const ConvOriGroupInfo& oriGroupInfo, ConvOptGroupInfo& optGroupInfo) const;
    uint64_t CalcC04UbLoadNsize(const ConvC04Info& c04Info) const;
    uint64_t CalcC04UbLoadMaxNsize(const ConvC04Info& c04Info) const;
    void GetWeightUBTiling(ConvWeightUbTransParams& params);
    void GetDmaUbTiling(ConvDmaParams& params);
    bool GetIsWeightBypassL1() const { return isWeightBypassL1; }
    bool GetIsTilingKh() const { return isTilingKh; }
    void SetIsWeightBypassL1(bool flag) { isWeightBypassL1 = flag; }
    void SetIsTilingKh(bool flag) { isTilingKh = flag; }

private:
    std::shared_ptr<ConvTilingAlgorithmBase> algoPtr;
    void SetTilingData(optiling::Conv2DTilingData& tiling);
    void SetAttrsTilingData(optiling::Conv2DTilingData& tiling);
    void SetScalarParams(optiling::Conv2DTilingData& tiling);
    void SetUbTiling(optiling::Conv2DTilingData& tiling);
    void SetExtendConv2DParams(optiling::Conv2DTilingData& tiling);
    uint64_t CalcWeightUBSize(ConvWeightUbTransParams& params, uint64_t ci1Ub, uint64_t co1Ub) const;
    uint64_t CalcDmaUBSize(ConvDmaParams& params, uint64_t khUb, uint64_t kwUb) const;
    uint32_t CalcAL1SpaceSize(optiling::Conv2DTilingData& tiling);
    void SetDefaultDdim();
    void Infer5hdShape() override;
    bool CheckBBmodeLimits(BasicBlockInfo& basicBlockInfo) override;
    bool CheckShapeBBModePhase1() override;
    bool CheckAlgorithmLimit() override;
    bool CheckAttr() override;
    bool CheckAttrBBModePhase1() override;
    bool CheckShape() override;
    bool CheckFormat() override;
    bool CheckInstructionLimits() override;
    bool CheckDataCopyLimits();
    bool CheckFixpipeLimits();
    bool CheckFeaMapShape();
    bool CheckWeightShape();
    bool CheckFmapShapeBeforeCoreBind();
    bool CheckWeightShapeBeforeCoreBind();
    bool CheckAttrBeforeCoreBind();
    bool CheckGroupsBBModePhase1();
    void PrintConv2DBasicBlockInfoPhase1(const Conv2DBasicBlockInfo& conv2DBasicBlockInfo) const;
    void PrintConv2DBasicBlockInfoPhase2(const Conv2DBasicBlockInfo& conv2DBasicBlockInfo) const;
    bool CheckBasicBlockInfoPhase1(Conv2DBasicBlockInfo& conv2DBasicBlockInfo) const override;
    bool CheckBasicBlockInfoPhase2(Conv2DBasicBlockInfo& conv2DBasicBlockInfo) override;
    bool CheckL1SizeLimitsKernelFullLoad(bool isC04);
    void SetIsWeightBypassL1();
    bool isWeightBypassL1 = false;
    bool isTilingKh = false;
};
} // namespace conv_tiling

#endif // ASCENDC_TILING_CONV2D_V2_API_TILING_H

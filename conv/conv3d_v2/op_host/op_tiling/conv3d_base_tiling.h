/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file conv3d_base_tiling.h
 * \brief
 */

#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_CONV3D_BASE_TILING_H
#define OPS_BUILT_IN_OP_TILING_RUNTIME_CONV3D_BASE_TILING_H

#include "tiling_base/tiling_base.h"
#include "conv3d_tuning_tiling.h"
#include "tiling/platform/platform_ascendc.h"
#include "conv3d_v2_tiling.h"
#include "conv3d_api_tiling.h"
#include "conv3d_tiling_utils.h"
#include "conv/common/op_host/op_tiling/cube_tiling.h"

namespace optiling {

struct Conv3DTilingParseInfo: CubeTilingCommonParseInfo {
        uint32_t aicoreNum = Conv3dOpsTiling::INITIAL_AICORE_ZERO;
        uint64_t l2Size = Conv3dApiTiling::INITIAL_SIZE;
        uint64_t l1Size = Conv3dApiTiling::INITIAL_SIZE;
        uint64_t l0aSize = Conv3dApiTiling::INITIAL_SIZE;
        uint64_t l0bSize = Conv3dApiTiling::INITIAL_SIZE;
        uint64_t l0cSize = Conv3dApiTiling::INITIAL_SIZE;
        uint64_t ubSize = Conv3dApiTiling::INITIAL_SIZE;
        uint64_t btSize = Conv3dApiTiling::INITIAL_SIZE;
        uint64_t l2Rate = Conv3dOpsTiling::INITIAL_L2_RATE_ZERO;
        std::string socVersion = "";
        std::string shortSocVersion = "";
    };

struct Conv3DAttrInfo {
    uint32_t dilationH = 1;
    uint32_t dilationW = 1;
    uint32_t dilationD = 1;
    uint32_t strideH = 1;
    uint32_t strideW = 1;
    uint32_t strideD = 1;
    uint32_t padh = 0;
    uint32_t padt = 0;
    uint32_t padu = 0;
    uint32_t padd = 0;
    uint32_t padl = 0;
    uint32_t padr = 0;
    uint32_t groups = 0;
    uint64_t groupOpt = 0;
    uint32_t hf32Mode = 0;
};

struct Conv3DOrignalFormat {
    // for fmap
    uint32_t FORMAT_FMAP_N_INDEX = Conv3dApiTiling::INITIAL_INDEX;
    uint32_t FORMAT_FMAP_C_INDEX = Conv3dApiTiling::INITIAL_INDEX;
    uint32_t FORMAT_FMAP_D_INDEX = Conv3dApiTiling::INITIAL_INDEX;
    uint32_t FORMAT_FMAP_H_INDEX = Conv3dApiTiling::INITIAL_INDEX;
    uint32_t FORMAT_FMAP_W_INDEX = Conv3dApiTiling::INITIAL_INDEX;
    // for weight
    uint32_t FORMAT_WEIGHT_N_INDEX = Conv3dApiTiling::INITIAL_INDEX;
    uint32_t FORMAT_WEIGHT_C_INDEX = Conv3dApiTiling::INITIAL_INDEX;
    uint32_t FORMAT_WEIGHT_D_INDEX = Conv3dApiTiling::INITIAL_INDEX;
    uint32_t FORMAT_WEIGHT_H_INDEX = Conv3dApiTiling::INITIAL_INDEX;
    uint32_t FORMAT_WEIGHT_W_INDEX = Conv3dApiTiling::INITIAL_INDEX;
    // for stride and dilation
    uint32_t FORMAT_DATA_D_INDEX = Conv3dApiTiling::INITIAL_INDEX;
    uint32_t FORMAT_DATA_H_INDEX = Conv3dApiTiling::INITIAL_INDEX;
    uint32_t FORMAT_DATA_W_INDEX = Conv3dApiTiling::INITIAL_INDEX;
};

using Ops::NN::Optiling::TilingBaseClass;
namespace Conv3dOpsTiling {

class Conv3dBaseTiling : public TilingBaseClass {
public:
    explicit Conv3dBaseTiling(gert::TilingContext* context) : TilingBaseClass(context) {};
    ~Conv3dBaseTiling() override = default;

protected:
    bool IsCapable() override {
        return true;
    };
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    [[nodiscard]] uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;

    bool GetTilingFromRepo();
    bool TranslateAoeTiling(tuningtiling::TuningTilingDefPtr &tuningTiling);
    void TranslateApiTiling(std::shared_ptr<tuningtiling::Conv3DTunnerTiling> aoeTiling);
    void TranslateRunInfo(std::shared_ptr<tuningtiling::Conv3DTunnerTiling> aoeTiling);
    bool GetTilingInputArgs(std::shared_ptr<void> &inputArgs, size_t &size);
    void SetAdditionalTilingInfo();

private:
    Conv3dApiTiling::Conv3dTiling conv3dApiTiling_;
    Conv3DTilingParseInfo opInfo_;
    Conv3DTilingParseInfo opRunInfo_;
    Conv3DAscendcShapesInfo shapeInfo_;
    Conv3DAttrInfo attrInfo_;
    Conv3DTilingData tilingData_;
    Conv3DDescInfo descInfo_;
    Conv3DTilingFlag flagInfo_;
    Conv3DOrignalFormat originalFormat_;

    // blockdim decision
    BlockDimRange blockDimRanges;
    BlockDimConstParas blockDimConst;
    std::vector<uint32_t> blockDimInit;
    BlockDimRes blockDimRes;

    bool useTilingRepo_ = false;
    bool isPointWise = false;
    int8_t outputOrder_ = M_Mode;

private:
    bool CheckDims(const gert::Shape& inputShape);
    ge::graphStatus CheckStrideLegal();
    ge::graphStatus CheckDilationLegal();
    ge::graphStatus CheckPadLegal();
    ge::graphStatus CheckFmapShape();
    ge::graphStatus CheckFmapNCDHWShape();
    ge::graphStatus CheckWeightShape();
    ge::graphStatus CheckWeightNCDHWShape();
    ge::graphStatus CheckInputShapeWithPad();
    ge::graphStatus CheckScaleShape();
    ge::graphStatus CheckBiasShape();
    ge::graphStatus CheckOutputShape();
    ge::graphStatus CheckOutputNCDHWShape();
    ge::graphStatus CheckInputDesc();
    ge::graphStatus CheckParamsDtype();
    ge::graphStatus CheckLoad3DLimits();
    ge::graphStatus CheckInstructionLimits();
    ge::graphStatus InitConv3dApiTiling();
    ge::graphStatus GetConv3dApiTiling();
    ge::graphStatus CheckInputLimitsHwMode();
    ge::graphStatus SetTilingKey();
    ge::graphStatus GetGroupConvOpt();
    ge::graphStatus CheckGroupOpt();
    ge::graphStatus CheckParamsOverflow();
    ge::graphStatus CheckPointWise();
    ge::graphStatus InitOutputOrder();
    uint64_t CalcMinL1LoadSize(int8_t outputOrder);
    void SetSingleOutputShapeByMode();
    void InitConv3dOriginFormat();
    void InitPointWiseFlag();
    void GetConv3dApiTilingPartSetAttrAndShape();
    void GetConv3dApiTilingSetGroupsInfo();
    void GetShapeInfo();
    void GetAttrsInfo();
    void GetDescInfo();
    void PrintTilingInfo();
    void PrintOpTilingData();
    void PrintApiTilingDataShapeInfo();
    void PrintApiTilingDataDecisionInfo();
    void PrintApiTilingDataScalarInfo();
    void PrintLibApiTilingData();
    void GetConv3DParasHf32Mode(const uint32_t enableHf32Idx, uint32_t& hf32Mode);
    bool Is3DFp32InputFp32Output();

    // blockdim decision
    bool IsExceedMinBurstNum(uint64_t input);
    uint64_t GetMinBurstNum();
    uint64_t CalcFixParamSize() const;
    uint64_t CalcTotalCost(uint32_t batchDim, uint32_t mDim, uint32_t nDim, uint32_t doDim, uint32_t groupDim);
    void BlockDimDecision();
    void GetBlockDimRange();
    void GetBlockDimInit();
    void BlockDimDecisionBackTrack(const std::vector<std::vector<uint32_t>> &inputRanges, uint32_t rangeIdx,
                                   std::vector<uint32_t> &record);
    void CoreBlockDimDecision();
    void BlockDimFactorMix(uint32_t orgDim, std::vector<uint32_t> &inputRange, const std::vector<uint32_t> &mixRange);
    void GetBlockDimRangeforGroupRange(std::vector<uint32_t> &groupRange);
};

} // namespace Conv3dOpsTiling

} // namespace optiling

#endif // OPS_BUILT_IN_OP_TILING_RUNTIME_CONV3D_BASE_TILING_H
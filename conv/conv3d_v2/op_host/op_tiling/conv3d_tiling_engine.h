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
 * \file conv3d_tiling_engine.h
 * \brief Conv3D Tiling Engine - Decoupled tiling decision and check module
 */

#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_CONV3D_TILING_ENGINE_H
#define OPS_BUILT_IN_OP_TILING_RUNTIME_CONV3D_TILING_ENGINE_H

#include <vector>
#include <string>
#include <cstdint>
#include "tiling/platform/platform_ascendc.h"
#include "conv3d_api_tiling.h"
#include "conv3d_tiling_utils.h"
#include "conv3d_common_utils.h"
#include "log/log.h"
#include "graph/types.h"
#include "../../op_kernel/conv3d_v2_tiling_data.h"

namespace Ops {
namespace NN {
namespace Conv3dTilingEngineApi {

constexpr uint8_t ATTRS_D_DIM_IDX_NCDHW = 0;
constexpr uint8_t ATTRS_H_DIM_IDX_NCDHW = 1;
constexpr uint8_t ATTRS_W_DIM_IDX_NCDHW = 2;

constexpr uint64_t MAX_64_BIT_NUM = 0xFFFFFFFFFFFFFFFFU;
constexpr uint8_t CONV_ATTRS_DIM = 3;

struct Conv3dPlatformInfo {
    uint64_t l1Size = 0;
    uint64_t l0aSize = 0;
    uint64_t l0bSize = 0;
    uint64_t l0cSize = 0;
    uint64_t ubSize = 0;
    uint64_t btSize = 0;
    uint64_t l2Rate = 0;
    uint32_t aicoreNum = 0;
    std::string socVersion = "";
};

struct Conv3DEngineDescInfo {
    Conv3dApiTiling::ConvDtype weightDtype = Conv3dApiTiling::ConvDtype::BF16;
    Conv3dApiTiling::ConvDtype fMapDtype = Conv3dApiTiling::ConvDtype::BF16;
    Conv3dApiTiling::ConvDtype biasDtype = Conv3dApiTiling::ConvDtype::FLOAT32;
    Conv3dApiTiling::ConvDtype scaleDtype = Conv3dApiTiling::ConvDtype::FLOAT32;
    Conv3dApiTiling::ConvDtype outDtype = Conv3dApiTiling::ConvDtype::BF16;

    Conv3dApiTiling::ConvFormat weightFormat = Conv3dApiTiling::ConvFormat::FRACTAL_Z_3D;
    Conv3dApiTiling::ConvFormat fMapFormat = Conv3dApiTiling::ConvFormat::NDC1HWC0;
    Conv3dApiTiling::ConvFormat biasFormat = Conv3dApiTiling::ConvFormat::ND;
    Conv3dApiTiling::ConvFormat scaleFormat = Conv3dApiTiling::ConvFormat::ND;
    Conv3dApiTiling::ConvFormat outFormat = Conv3dApiTiling::ConvFormat::NDC1HWC0;
};

/*!
 * \brief Conv3D Tiling Engine - Decoupled tiling decision module
 */
class Conv3dTilingEngine {
public:
    explicit Conv3dTilingEngine(const std::string &logTag = "Conv3DV2");

    Conv3dPlatformInfo platformInfo_;

    optiling::Conv3dOpsTiling::Conv3DAscendcShapesInfo shapeInfo_;
    Conv3dApiTiling::Conv3DInputAttr attrInfo_;
    Conv3DEngineDescInfo descInfo_;
    optiling::Conv3dOpsTiling::Conv3DTilingFlag flagInfo_;

    bool isPointWise = false;

    optiling::Conv3dOpsTiling::BlockDimRange blockDimRanges_;
    optiling::Conv3dOpsTiling::BlockDimConstParas blockDimConst_;
    std::vector<uint32_t> blockDimInit_;
    optiling::Conv3dOpsTiling::BlockDimRes blockDimRes_;

    Conv3dApiTiling::Conv3dTiling conv3dApiTiling_;

    int8_t outputOrder_ = Conv3dApiTiling::M_Mode;

    void SetOrgWeightShape(const std::vector<int64_t> &orgWeightShapeList);
    void SetOrgFmapShape(const std::vector<int64_t> &orgFmapShapeList);
    void SetOrgOutputShape(const std::vector<int64_t> &orgOutputShapeList);
    void SetPadding(const std::vector<int64_t> &padList);
    // Stride and dilation are passed in DHW order: [D, H, W]
    void SetStride(const std::vector<int64_t> &strideList);
    void SetDilation(const std::vector<int64_t> &dilationList);
    void SetGroups(int64_t groups);
    void SetDataType(Conv3dApiTiling::ConvDtype fmapDtype,
                     Conv3dApiTiling::ConvDtype weightDtype,
                     Conv3dApiTiling::ConvDtype outDtype);
    void SetBias(bool hasBias, Conv3dApiTiling::ConvDtype biasDtype);
    void SetScale(bool hasScale, Conv3dApiTiling::ConvDtype scaleDtype);
    void SetHF32(bool enable);

    bool GetConv3DV2TilingData(Ops::NN::Conv3dV2::Conv3DV2TilingData &tilingData);
    uint32_t GetBlockDim() const;
    void GetBlockDimDetail(uint32_t &batchDim, uint32_t &mDim, uint32_t &nDim,
                           uint32_t &doDim, uint32_t &groupDim) const;

    void PrintOpTilingData(const Ops::NN::Conv3dV2::Conv3DV2TilingData &tilingData) const;
    void PrintApiTilingDataShapeInfo(const Ops::NN::Conv3dV2::Conv3DV2TilingData &tilingData) const;
    void PrintApiTilingDataDecisionInfo(const Ops::NN::Conv3dV2::Conv3DV2TilingData &tilingData) const;
    void PrintApiTilingDataScalarInfo(const Ops::NN::Conv3dV2::Conv3DV2TilingData &tilingData) const;

    // Decision pipeline pieces
    void GetBlockDimRange();
    void GetBlockDimInit();
    void BlockDimDecision();
    void CoreBlockDimDecision();
    void BlockDimDecisionBackTrack(const std::vector<std::vector<uint32_t>> &inputRanges,
                                   uint32_t rangeIdx,
                                   std::vector<uint32_t> &record);
    uint64_t CalcTotalCost(uint32_t batchDim, uint32_t mDim, uint32_t nDim,
                           uint32_t doDim, uint32_t groupDim);
    void BlockDimFactorMix(uint32_t orgDim, std::vector<uint32_t> &inputRange,
                           const std::vector<uint32_t> &mixRange);
    void GetBlockDimRangeforGroupRange(std::vector<uint32_t> &groupRange);
    void GetConv3DRunInfo(Ops::NN::Conv3dV2::Conv3DV2TilingData &tilingdata);

    // Checks and computations (return bool for success/failure)
    bool GetGroupConvOpt();
    bool ComputeBlockDim();
    bool ComputeApiTiling(Ops::NN::Conv3dV2::Conv3DV2TilingData &tilingdata);
    bool GetConv3dApiTiling(Ops::NN::Conv3dV2::Conv3DV2TilingData &tilingdata);
    void SetSingleOutputShapeByMode();
    void GetConv3dApiTilingPartSetAttrAndShape();
    void GetConv3dApiTilingSetGroupsInfo();
    bool InitOutputOrder();
    uint64_t CalcMinL1LoadSize(int8_t outputOrder);
    bool CheckInputLimitsHwMode();
    bool CheckDims(const std::vector<int64_t>& shape);

private:
    std::string logTag_ {"Conv3DV2"};

    bool InitPlatformInfoFromAscendC();

public:
    bool CheckStrideLegal();
    bool CheckDilationLegal();
    bool CheckPadLegal();
    bool CheckFmapShape();
    bool CheckWeightShape();
    bool CheckParamsDtype();
    bool CheckLoad3DLimits();
    bool CheckInputShapeWithPad();
    bool CheckInputShapeWithPadDetail(int64_t &idPad, int64_t &ihPad, int64_t &iwPad);
    bool CheckGroupOptAgainstWeightShape(uint64_t weightD, uint64_t weightN1);
    bool CheckParamsOverflow();
    bool CheckShapeSizeLimits();
    void CheckFmapShapeSizeLimits();
    void CheckWeightShapeSizeLimits();
    void CheckOutputShapeSizeLimits();
    void CheckAttrShapeSizeLimits();
    bool CheckAllParams();
};

} // namespace Conv3dTilingEngineApi
} // namespace NN
} // namespace Ops

#endif // OPS_BUILT_IN_OP_TILING_RUNTIME_CONV3D_TILING_ENGINE_H

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
 * \file conv3d_base_tiling.h
 * \brief
 */

#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_CONV3D_BASE_TILING_H
#define OPS_BUILT_IN_OP_TILING_RUNTIME_CONV3D_BASE_TILING_H

#include "tiling_base/tiling_base.h"
#include "conv3d_tuning_tiling.h"
#include "tiling/platform/platform_ascendc.h"
#include "../../op_kernel/conv3d_v2_tiling_data.h"
#include "conv3d_api_tiling.h"
#include "tiling/tiling_api.h"
#include "conv3d_tiling_utils.h"
#include "conv3d_tiling_engine.h"
#include "conv/common/op_host/op_tiling/cube_tiling.h"
#include <memory>

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

struct Conv3DDescInfo {
    ge::DataType weightDtype = ge::DT_BF16;
    ge::DataType fMapDtype = ge::DT_BF16;
    ge::DataType biasDtype = ge::DT_FLOAT;
    ge::DataType scaleDtype = ge::DT_FLOAT;
    ge::DataType outDtype = ge::DT_BF16;

    ge::Format weightFormat = ge::FORMAT_FRACTAL_Z_3D;
    ge::Format fMapFormat = ge::FORMAT_NDC1HWC0;
    ge::Format biasFormat = ge::FORMAT_ND;
    ge::Format scaleFormat = ge::FORMAT_ND;
    ge::Format outFormat = ge::FORMAT_NDC1HWC0;
};

static std::map<ge::DataType, uint32_t> g_dataTypeSizeTab = {
    {ge::DataType::DT_FLOAT16, 2}, {ge::DataType::DT_FLOAT, 4}, {ge::DataType::DT_BF16, 2}, {ge::DataType::DT_INT8, 1},
    {ge::DataType::DT_UINT8, 1}, {ge::DataType::DT_INT64, 8}, {ge::DataType::DT_UINT64, 8}, {ge::DataType::DT_INT32, 4}};

static std::map<ge::DataType, Conv3dApiTiling::ConvDtype> g_dtypeMap = {
    {ge::DT_FLOAT16, Conv3dApiTiling::ConvDtype::FLOAT16},
    {ge::DT_FLOAT, Conv3dApiTiling::ConvDtype::FLOAT32},
    {ge::DT_BF16, Conv3dApiTiling::ConvDtype::BF16},
    {ge::DT_INT8, Conv3dApiTiling::ConvDtype::INT8},
    {ge::DT_UINT8, Conv3dApiTiling::ConvDtype::UINT8},
    {ge::DT_INT64, Conv3dApiTiling::ConvDtype::INT64},
    {ge::DT_UINT64, Conv3dApiTiling::ConvDtype::UINT64},
    {ge::DT_INT32, Conv3dApiTiling::ConvDtype::INT32}
};

static std::map<ge::Format, std::string> g_formatToStrTab = {
    {ge::FORMAT_NCHW, "NCHW"}, {ge::FORMAT_NHWC, "NHWC"}, {ge::FORMAT_HWCN, "HWCN"}, {ge::FORMAT_DHWNC, "DHWNC"},
    {ge::FORMAT_DHWCN, "DHWCN"}, {ge::FORMAT_NDHWC, "NDHWC"}, {ge::FORMAT_NCDHW, "NCDHW"},
    {ge::FORMAT_NC1HWC0, "NC1HWC0"}, {ge::FORMAT_ND, "ND"}, {ge::FORMAT_NDC1HWC0, "NDC1HWC0"},
    {ge::FORMAT_FRACTAL_Z_3D, "FRACTAL_Z_3D"}};

static std::map<ge::Format, Conv3dApiTiling::ConvFormat> g_formatMap = {
    {ge::FORMAT_ND, Conv3dApiTiling::ConvFormat::ND},
    {ge::FORMAT_NCHW, Conv3dApiTiling::ConvFormat::NCHW},
    {ge::FORMAT_NHWC, Conv3dApiTiling::ConvFormat::NHWC},
    {ge::FORMAT_HWCN, Conv3dApiTiling::ConvFormat::HWCN},
    {ge::FORMAT_DHWNC, Conv3dApiTiling::ConvFormat::DHWNC},
    {ge::FORMAT_DHWCN, Conv3dApiTiling::ConvFormat::DHWCN},
    {ge::FORMAT_NDHWC, Conv3dApiTiling::ConvFormat::NDHWC},
    {ge::FORMAT_NCDHW, Conv3dApiTiling::ConvFormat::NCDHW},
    {ge::FORMAT_NC1HWC0, Conv3dApiTiling::ConvFormat::NC1HWC0},
    {ge::FORMAT_NDC1HWC0, Conv3dApiTiling::ConvFormat::NDC1HWC0},
    {ge::FORMAT_FRACTAL_Z_3D, Conv3dApiTiling::ConvFormat::FRACTAL_Z_3D}
};

static std::map<ge::DataType, std::string> g_dtypeToStrTab = {
    {ge::DataType::DT_FLOAT16, "float16"}, {ge::DataType::DT_FLOAT, "float32"}, {ge::DataType::DT_BF16, "bfloat16"},
    {ge::DataType::DT_INT8, "int8"}, {ge::DataType::DT_UINT8, "uint8"}, {ge::DataType::DT_INT64, "int64"},
    {ge::DataType::DT_UINT64, "uint64"}, {ge::DataType::DT_INT32, "int32"}};

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
    std::unique_ptr<Ops::NN::Conv3dTilingEngineApi::Conv3dTilingEngine> engine_;
    Conv3DTilingParseInfo opInfo_;
    Conv3DTilingParseInfo opRunInfo_;
    Conv3DAscendcShapesInfo shapeInfo_;
    Conv3DAttrInfo attrInfo_;
    Ops::NN::Conv3dV2::Conv3DV2TilingData tilingData_;
    Conv3DDescInfo descInfo_;
    Conv3DTilingFlag flagInfo_;
    Conv3DOrignalFormat originalFormat_;

    // blockdim decision
    BlockDimRes blockDimRes;

    bool useTilingRepo_ = false;
    bool isPointWise = false;
    int8_t outputOrder_ = 0;

    private:
      ge::graphStatus SetTilingKey();
      void InitConv3dOriginFormat();
      void InitPointWiseFlag();
      void GetShapeInfo();
      void GetAttrsInfo();
      void GetDescInfo();
      void PrintTilingInfo();
      void GetConv3DParasHf32Mode(const uint32_t enableHf32Idx, uint32_t& hf32Mode);
      bool Is3DFp32InputFp32Output();

      std::vector<int64_t> ExtractOriginFmapShape();
      std::vector<int64_t> ExtractOriginWeightShape();
      std::vector<int64_t> ExtractOriginOutputShape();
      bool ExtractPadList(std::vector<int64_t> &padList);
      bool ExtractStrideList(std::vector<int64_t> &strideList);
      bool ExtractDilationList(std::vector<int64_t> &dilationList);
      bool ExtractBiasShape(std::vector<int64_t> &biasShape);
      bool ExtractScaleShape(std::vector<int64_t> &scaleShape);
      int64_t ExtractGroups();
      void ExtractAndSetDataTypes();
      void ExtractAndSetFormats();
      void SetHF32Mode();
      bool ExtractAndSetBiasScale();
      bool ExtractAndPassParamsToEngine();
};

} // namespace Conv3dOpsTiling

} // namespace optiling

#endif // OPS_BUILT_IN_OP_TILING_RUNTIME_CONV3D_BASE_TILING_H

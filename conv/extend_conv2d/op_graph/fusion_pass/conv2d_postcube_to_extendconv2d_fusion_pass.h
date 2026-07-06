/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CONV2D_POSTCUBE_TO_EXTENDCONV2D_FUSION_PASS_H
#define CONV2D_POSTCUBE_TO_EXTENDCONV2D_FUSION_PASS_H

#include <map>
#include <memory>
#include <sstream>
#include <vector>

#include "../../common/graph_fusion/cube_utils/cube_utils.h"
#include "../../conv/common/op_graph/fusion_pass/conv_fusion_base_pass.h"
#include "ge/fusion/subgraph_boundary.h"
#include "platform/soc_spec.h"

namespace Ops {
namespace NN {
namespace Conv {
namespace Conv2DPostCubeToExtendConv2DFusion {

const ge::AscendString ASCEND_REQUANT = "AscendRequant";
const ge::AscendString DUAL_OUTPUT = "dual_output";
const ge::AscendString ENABLE_RELU_0 = "enable_relu0";
const ge::AscendString ENABLE_RELU_1 = "enable_relu1";
const ge::AscendString FUSION_OP_LIST = "fusion_op_list";
const ge::AscendString LEAKY_RELU = "LeakyRelu";
const ge::AscendString RELU = "Relu";
const ge::AscendString RINT = "rint";
const ge::AscendString SCALE_0 = "scale0";
const ge::AscendString SCALE_1 = "scale1";
const ge::AscendString RELU_WEIGHT_0 = "relu_weight0";
const ge::AscendString RELU_WEIGHT_1 = "relu_weight1";

const std::map<std::string, NpuArch> SUPPORT_SOC_LIST = {{"Ascend950", NpuArch::DAV_3510}, {"MC62", NpuArch::DAV_5102}};
const std::string FUSION_NAME = "Conv2DPostCubeToExtendConv2DFusionPass";

constexpr int32_t EXTENDCONV2D_QUANT_SCALE_0_INDEX = 4;
constexpr int32_t EXTENDCONV2D_QUANT_SCALE_1_INDEX = 7;
constexpr int32_t EXTENDCONV2D_RELU_WEIGHT_0_INDEX = 5;
constexpr int32_t EXTENDCONV2D_RELU_WEIGHT_1_INDEX = 8;
constexpr int32_t POST_CUBE_INPUT_QUANT_SCALE_0_INDEX = 2;
constexpr int32_t POST_CUBE_INPUT_RELU_WEIGHT_0_INDEX = 3;
constexpr int32_t OUTPUT_0_INDEX = 0;
constexpr int32_t OUTPUT_1_INDEX = 1;
constexpr size_t DUAL_OUTPUTNUM = 2;

// Fmap Filter Output Bias
const std::vector<std::vector<ge::DataType>> CONV_SUPPORT_DTYPES = {
    {ge::DataType::DT_INT8, ge::DataType::DT_INT8, ge::DataType::DT_INT32, ge::DataType::DT_INT32},
    {ge::DataType::DT_FLOAT16, ge::DataType::DT_FLOAT16, ge::DataType::DT_INT32, ge::DataType::DT_FLOAT16},
    {ge::DataType::DT_FLOAT16, ge::DataType::DT_INT8, ge::DataType::DT_INT32, ge::DataType::DT_INT32},
    {ge::DataType::DT_FLOAT16, ge::DataType::DT_FLOAT16, ge::DataType::DT_FLOAT16, ge::DataType::DT_FLOAT16}};

// Fmap Filter Output
const std::vector<std::vector<ge::Format>> CONV_SUPPORT_FORMATS_DAV_3510 = {
    {ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW}, {ge::FORMAT_NHWC, ge::FORMAT_HWCN, ge::FORMAT_NHWC}};

// Fmap Filter Output
const std::vector<std::vector<ge::Format>> CONV_SUPPORT_FORMATS_DAV_5102 = {
    {ge::FORMAT_NCHW, ge::FORMAT_FRACTAL_Z, ge::FORMAT_NCHW},
    {ge::FORMAT_NCHW, ge::FORMAT_FRACTAL_Z_C04, ge::FORMAT_NCHW},
    {ge::FORMAT_NHWC, ge::FORMAT_FRACTAL_Z, ge::FORMAT_NHWC},
    {ge::FORMAT_NHWC, ge::FORMAT_FRACTAL_Z_C04, ge::FORMAT_NHWC}};

// Fmap Filter PostCubeIn PostCubeOut
const std::vector<std::vector<ge::DataType>> SUPPORTED_DTYPES_WITH_POST_CUBE_DAV_3510 = {
    {ge::DataType::DT_FLOAT16, ge::DataType::DT_FLOAT16, ge::DataType::DT_FLOAT16, ge::DataType::DT_FLOAT16},
    {ge::DataType::DT_FLOAT16, ge::DataType::DT_FLOAT16, ge::DataType::DT_FLOAT16, ge::DataType::DT_INT8},
    {ge::DataType::DT_INT8, ge::DataType::DT_INT8, ge::DataType::DT_INT32, ge::DataType::DT_FLOAT16},
    {ge::DataType::DT_INT8, ge::DataType::DT_INT8, ge::DataType::DT_INT32, ge::DataType::DT_INT8}};

// Fmap Filter PostCubeIn PostCubeOut
const std::vector<std::vector<ge::DataType>> SUPPORTED_DTYPES_WITH_POST_CUBE_DAV_5102 = {
    {ge::DataType::DT_FLOAT16, ge::DataType::DT_FLOAT16, ge::DataType::DT_FLOAT16, ge::DataType::DT_FLOAT16},
    {ge::DataType::DT_FLOAT16, ge::DataType::DT_FLOAT16, ge::DataType::DT_FLOAT16, ge::DataType::DT_INT8},
    {ge::DataType::DT_INT8, ge::DataType::DT_INT8, ge::DataType::DT_INT32, ge::DataType::DT_FLOAT16},
    {ge::DataType::DT_INT8, ge::DataType::DT_INT8, ge::DataType::DT_INT32, ge::DataType::DT_INT8},
    {ge::DataType::DT_FLOAT16, ge::DataType::DT_INT8, ge::DataType::DT_INT32, ge::DataType::DT_FLOAT16},
    {ge::DataType::DT_FLOAT16, ge::DataType::DT_INT8, ge::DataType::DT_INT32, ge::DataType::DT_INT8}};

const std::vector<ge::AscendString> SUPPORTED_NODE_TYPES = {"Conv2D",      "AscendDequant", "AscendRequant",
                                                            "AscendQuant", "Relu",          "LeakyRelu"};
const std::vector<ge::AscendString> POST_CUBE_NODE_TYPES = {"AscendDequant", "AscendRequant", "AscendQuant", "Relu",
                                                            "LeakyRelu"};
enum class OutputCase : std::uint8_t { SINGLE, DUAL_POST_CUBE, POST_CUBE_OTHER, OTHER_POST_CUBE };
} // namespace Conv2DPostCubeToExtendConv2DFusion

class __attribute__((visibility("default"))) Conv2DPostCubeToExtendConv2DFusionPass : public ConvFusionBasePass {
protected:
    void InitMember() override;
    bool MeetRequirements(const ge::GNode& convNode) override;
    ge::AscendString GetNodeType() const override;
    void PrintGraphStructure() const override;
    ge::Status ConvFusionPreImpl(ge::GraphPtr& graph, ge::GNode& convNode,
                                 const ge::CustomPassContext& pass_context) override;
    bool ConvFusionReplaceImpl(ge::GraphPtr& graph, const ge::GNode& convNode) override;
    std::unique_ptr<ge::fusion::SubgraphBoundary> ConstructBoundary(const ge::GNode& convNode) override;
    ge::fusion::GraphUniqPtr Replacement(const ge::GNode& convNode) override;

private:
    bool AddScaleReluToBoundAry(std::unique_ptr<ge::fusion::SubgraphBoundary>& boundary);
    bool CheckConvPostCubeDtype(const ge::GNodePtr postCubeNode) const;
    bool CheckDescInfo();
    bool CheckSupportPostCubeCase(const ge::GNodePtr postCubeNode);
    bool GetPostCubeNodes(const ge::GNode& convNode);
    bool IsReluEnable(const std::vector<ge::AscendString>& postCubeFusionOp,
                      const ge::AscendString& opType = "default") const;
    bool IsScaleEnable(const std::vector<ge::AscendString>& postCubeFusionOp) const;
    void SelectPostCubePassByWhiteList(std::vector<ops::PostCubePassInfo>& matchLists) const;
    bool UpdateExtendConv2DDesc(ge::GNode* extendConv2D) const;
    bool UpdateScaleReluDesc(ge::GNodePtr postCube, ge::GNode* extendConv2D, const int32_t getIndex,
                             const int32_t updateIndex, const ge::AscendString& name) const;

    std::vector<std::vector<ge::AscendString>> postCubeFusionOps = {};
    std::vector<ge::GNodePtr> postCubeNodes = {};
    std::vector<std::pair<ge::GNodePtr, int32_t>> otherNodes = {};

    Conv2DPostCubeToExtendConv2DFusion::OutputCase outputCase = Conv2DPostCubeToExtendConv2DFusion::OutputCase::SINGLE;

    bool hasScale0 = false;
    bool hasScale1 = false;
    bool hasRelu0 = false;
    bool hasRelu1 = false;
    int64_t graphIndex = ConvFusionUtils::REQUIRED_INPUT_NUMS;
};

} // namespace Conv
} // namespace NN
} // namespace Ops
#endif // CONV2D_POSTCUBE_TO_EXTENDCONV2D_FUSION_PASS_H
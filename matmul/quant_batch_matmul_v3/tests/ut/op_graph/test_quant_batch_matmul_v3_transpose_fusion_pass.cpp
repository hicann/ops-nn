/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "ge/compliant_node_builder.h"
#include "ge/es_graph_builder.h"
#include "graph/graph.h"
#define private public
#include "platform/platform_info.h"
#undef private
#include "register/register_custom_pass.h"
#include "../../../op_graph/fusion_pass/quant_batch_matmul_v3_transpose_fusion_pass.h"
#if __has_include("aclnn/aclnn_base.h")
#include "aclnn/aclnn_base.h"
#ifndef ACLNN_SUCCESS
#define ACLNN_SUCCESS static_cast<aclnnStatus>(0)
#define QUANT_BATCH_MATMUL_V3_UT_DEFINED_ACLNN_SUCCESS
#endif
#ifndef ACLNN_ERR_PARAM_INVALID
#define ACLNN_ERR_PARAM_INVALID static_cast<aclnnStatus>(-1)
#define QUANT_BATCH_MATMUL_V3_UT_DEFINED_ACLNN_ERR_PARAM_INVALID
#endif
#ifndef ACLNN_ERR_PARAM_NULLPTR
#define ACLNN_ERR_PARAM_NULLPTR static_cast<aclnnStatus>(-2)
#define QUANT_BATCH_MATMUL_V3_UT_DEFINED_ACLNN_ERR_PARAM_NULLPTR
#endif
#endif
#include "../../../../../tests/ut/common/ut_string_utils.h"
#ifdef QUANT_BATCH_MATMUL_V3_UT_DEFINED_ACLNN_SUCCESS
#undef ACLNN_SUCCESS
#undef QUANT_BATCH_MATMUL_V3_UT_DEFINED_ACLNN_SUCCESS
#endif
#ifdef QUANT_BATCH_MATMUL_V3_UT_DEFINED_ACLNN_ERR_PARAM_INVALID
#undef ACLNN_ERR_PARAM_INVALID
#undef QUANT_BATCH_MATMUL_V3_UT_DEFINED_ACLNN_ERR_PARAM_INVALID
#endif
#ifdef QUANT_BATCH_MATMUL_V3_UT_DEFINED_ACLNN_ERR_PARAM_NULLPTR
#undef ACLNN_ERR_PARAM_NULLPTR
#undef QUANT_BATCH_MATMUL_V3_UT_DEFINED_ACLNN_ERR_PARAM_NULLPTR
#endif

using namespace ge;
using namespace ge::fusion;
using namespace ut_str;

namespace {
constexpr char QUANT_BATCH_MATMUL_V3_TRANSPOSE_FUSION_PASS_NAME[] = "QuantBatchMatmulV3TransposeFusionPass";
constexpr char QUANT_BATCH_MATMUL_V3_TRANSPOSE_LIMIT_FUSION_PASS_NAME[] = "QuantBatchMatmulV3TransposeLimitFusionPass";
constexpr char TEST_CSV_FILE[] = "test_quant_batch_matmul_v3_transpose_fusion_pass.csv";
#ifndef QUANT_BATCH_MATMUL_V3_TRANSPOSE_FUSION_PASS_CSV_PATH
#define QUANT_BATCH_MATMUL_V3_TRANSPOSE_FUSION_PASS_CSV_PATH ""
#endif
constexpr size_t CASE_FIELD_NUM = 26;
constexpr int64_t X1_INDEX = 0;
constexpr int64_t X2_INDEX = 1;
constexpr int64_t SCALE_INDEX = 2;
constexpr int64_t PERTOKEN_SCALE_INDEX = 5;

struct QuantBatchMatmulV3TransposeFusionPassParam {
    std::string caseName;
    std::string passName;
    std::vector<int64_t> x1Shape;
    DataType x1Dtype;
    std::vector<int64_t> x2Shape;
    DataType x2Dtype;
    std::vector<int64_t> scaleShape;
    DataType scaleDtype;
    std::vector<int64_t> x1ScaleShape;
    DataType x1ScaleDtype;
    std::vector<int64_t> outShape;
    DataType outDtype;
    bool transposeX1;
    bool transposeX2;
    bool bitcastX1;
    bool bitcastX2;
    bool reshapeScale;
    std::vector<int64_t> reshapeScaleShape;
    std::string reshapeScaleFrom;
    DataType dtypeAttr;
    int64_t groupSize;
    Status expectedStatus;
    int expectedTransposeCount;
    int expectedReshapeCount;
    bool expectedTransposeX1;
    bool expectedTransposeX2;
};

struct TensorWithDesc {
    ge::es::EsTensorHolder tensor;
    TensorDesc desc;
};

class TestQuantBatchMatmulV3TransposeFusionPass : public ops::QuantBatchMatmulV3TransposeFusionPass {
public:
    Status RunForTest(GraphPtr& graph, CustomPassContext& passContext) { return Run(graph, passContext); }
};

class TestQuantBatchMatmulV3TransposeLimitFusionPass : public ops::QuantBatchMatmulV3TransposeLimitFusionPass {
public:
    Status RunForTest(GraphPtr& graph, CustomPassContext& passContext) { return Run(graph, passContext); }
};

static std::string GetDirName(const std::string& path)
{
    auto pos = path.find_last_of("/\\");
    return pos == std::string::npos ? "." : path.substr(0, pos);
}

static std::vector<std::string> SplitCsvLine(const std::string& line)
{
    std::vector<std::string> fields;
    SplitStr2Vec(line, ",", fields);
    for (auto& field : fields) {
        field = Trim(field);
    }
    return fields;
}

static std::ifstream OpenCsvData()
{
    const std::vector<std::string> casePaths = {
        QUANT_BATCH_MATMUL_V3_TRANSPOSE_FUSION_PASS_CSV_PATH,
        GetDirName(__FILE__) + "/" + TEST_CSV_FILE,
        TEST_CSV_FILE,
        "../../../../matmul/quant_batch_matmul_v3/tests/ut/op_graph/" + std::string(TEST_CSV_FILE),
    };
    std::string triedPaths;
    for (const auto& casePath : casePaths) {
        if (casePath.empty()) {
            continue;
        }
        std::ifstream csvData(casePath, std::ios::in);
        if (csvData.is_open()) {
            return csvData;
        }
        if (!triedPaths.empty()) {
            triedPaths += ", ";
        }
        triedPaths += casePath;
    }
    throw std::runtime_error("Open csv file failed, tried: " + triedPaths);
}

static Status ToStatus(const std::string& value)
{
    if (value == "SUCCESS") {
        return SUCCESS;
    }
    if (value == "GRAPH_NOT_CHANGED") {
        return GRAPH_NOT_CHANGED;
    }
    if (value == "FAILED") {
        return FAILED;
    }
    throw std::runtime_error("Unsupported status: " + value);
}

static void SetPlatformSupport()
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optionalInfo;
    platformInfo.ai_core_intrinsic_dtype_map["Intrinsic_data_move_l12bt"] = {"float16"};
    platformInfo.ai_core_intrinsic_dtype_map["Intrinsic_mmad"] = {"s8s8"};
    optionalInfo.soc_version = "soc_version";
    fe::PlatformInfoManager::Instance().platform_info_map_.clear();
    fe::PlatformInfoManager::Instance().platform_info_map_["soc_version"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optionalInfo);
}

static void ClearPlatformSupport() { fe::PlatformInfoManager::Instance().platform_info_map_.clear(); }

static QuantBatchMatmulV3TransposeFusionPassParam ParseParam(const std::vector<std::string>& fields)
{
    if (fields.size() != CASE_FIELD_NUM) {
        throw std::runtime_error("Invalid csv column size.");
    }
    size_t index = 0;
    QuantBatchMatmulV3TransposeFusionPassParam param;
    param.caseName = fields[index++];
    param.passName = fields[index++];
    param.x1Shape = ParseInt64Vec(fields[index++]);
    param.x1Dtype = ParseDtype(fields[index++], DT_UNDEFINED);
    param.x2Shape = ParseInt64Vec(fields[index++]);
    param.x2Dtype = ParseDtype(fields[index++], DT_UNDEFINED);
    param.scaleShape = ParseInt64Vec(fields[index++]);
    param.scaleDtype = ParseDtype(fields[index++], DT_UNDEFINED);
    param.x1ScaleShape = ParseInt64Vec(fields[index++]);
    param.x1ScaleDtype = ParseDtype(fields[index++], DT_UNDEFINED);
    param.outShape = ParseInt64Vec(fields[index++]);
    param.outDtype = ParseDtype(fields[index++], DT_UNDEFINED);
    param.transposeX1 = ParseBool(fields[index++]);
    param.transposeX2 = ParseBool(fields[index++]);
    param.bitcastX1 = ParseBool(fields[index++]);
    param.bitcastX2 = ParseBool(fields[index++]);
    param.reshapeScale = ParseBool(fields[index++]);
    param.reshapeScaleShape = ParseInt64Vec(fields[index++]);
    param.reshapeScaleFrom = fields[index++];
    param.dtypeAttr = ParseDtype(fields[index++], DT_UNDEFINED);
    param.groupSize = ParseInt64OrDefault(fields[index++], -1);
    param.expectedStatus = ToStatus(fields[index++]);
    param.expectedTransposeCount = ParseIntOrDefault(fields[index++], 0);
    param.expectedReshapeCount = ParseIntOrDefault(fields[index++], 0);
    param.expectedTransposeX1 = ParseBool(fields[index++]);
    param.expectedTransposeX2 = ParseBool(fields[index++]);
    return param;
}

static std::vector<QuantBatchMatmulV3TransposeFusionPassParam> GetParams()
{
    std::ifstream csvData = OpenCsvData();

    std::vector<QuantBatchMatmulV3TransposeFusionPassParam> params;
    std::string line;
    while (std::getline(csvData, line)) {
        if (Trim(line).empty()) {
            continue;
        }
        auto fields = SplitCsvLine(line);
        if (!fields.empty() && fields[0] == "case_name") {
            continue;
        }
        params.emplace_back(ParseParam(fields));
    }
    return params;
}

static std::vector<std::pair<int64_t, int64_t>> MakeShapeRange(const std::vector<int64_t>& shape)
{
    std::vector<std::pair<int64_t, int64_t>> range;
    range.reserve(shape.size());
    for (auto dim : shape) {
        if (dim < 0) {
            range.emplace_back(1, -1);
        } else {
            range.emplace_back(dim, dim);
        }
    }
    return range;
}

static TensorDesc MakeTensorDesc(const std::vector<int64_t>& shape, DataType dtype, Format format = FORMAT_ND)
{
    TensorDesc desc(Shape(shape), format, dtype);
    desc.SetOriginFormat(format);
    desc.SetOriginShape(Shape(shape));
    desc.SetShapeRange(MakeShapeRange(shape));
    return desc;
}

static std::vector<int64_t> TransposeLastTwoDims(const std::vector<int64_t>& shape)
{
    auto transposedShape = shape;
    if (transposedShape.size() >= 2) {
        std::swap(transposedShape[transposedShape.size() - 1], transposedShape[transposedShape.size() - 2]);
    }
    return transposedShape;
}

static GNode BuildUnaryNode(Graph* graph, const char* opType, const char* nodeName, const TensorDesc& outputDesc)
{
    return ge::es::CompliantNodeBuilder(graph)
        .OpType(opType)
        .Name(nodeName)
        .IrDefInputs({{"x", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
        .IrDefOutputs({{"y", ge::es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
        .InstanceOutputDataType("y", outputDesc.GetDataType())
        .InstanceOutputShape("y", outputDesc.GetShape().GetDims())
        .InstanceOutputFormat("y", FORMAT_ND)
        .Build();
}

static void LinkUnaryNode(Graph* graph, ge::es::EsGraphBuilder& graphBuilder, TensorWithDesc& input, GNode& node,
                          const TensorDesc& outputDesc)
{
    ge::es::AddEdgeAndUpdatePeerDesc(*graph, *input.tensor.GetProducer(), input.tensor.GetProducerOutIndex(), node, 0);
    node.UpdateInputDesc(0, input.desc);
    node.UpdateOutputDesc(0, outputDesc);
    input.tensor = ge::es::EsTensorHolder(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(node, 0));
    input.desc = outputDesc;
}

static void AddTransposeIfNeeded(Graph* graph, ge::es::EsGraphBuilder& graphBuilder, TensorWithDesc& input,
                                 bool enabled, const char* nodeName)
{
    if (!enabled) {
        return;
    }
    auto outputDesc = MakeTensorDesc(TransposeLastTwoDims(input.desc.GetShape().GetDims()), input.desc.GetDataType());
    auto transposeNode = BuildUnaryNode(graph, "Transpose", nodeName, outputDesc);
    LinkUnaryNode(graph, graphBuilder, input, transposeNode, outputDesc);
}

static void AddBitcastIfNeeded(Graph* graph, ge::es::EsGraphBuilder& graphBuilder, TensorWithDesc& input, bool enabled,
                               const char* nodeName)
{
    if (!enabled) {
        return;
    }
    auto outputDesc = input.desc;
    outputDesc.SetDataType(DT_HIFLOAT8);
    auto bitcastNode = BuildUnaryNode(graph, "Bitcast", nodeName, outputDesc);
    LinkUnaryNode(graph, graphBuilder, input, bitcastNode, outputDesc);
    DataType bitcastType = DT_HIFLOAT8;
    bitcastNode.SetAttr("type", bitcastType);
}

static void AddReshapeIfNeeded(Graph* graph, ge::es::EsGraphBuilder& graphBuilder, TensorWithDesc& scaleInput,
                               const TensorWithDesc& reshapeInput, bool enabled,
                               const std::vector<int64_t>& outputShape)
{
    if (!enabled) {
        return;
    }
    TensorWithDesc reshapeOutput = reshapeInput;
    auto outputDesc = MakeTensorDesc(outputShape, reshapeInput.desc.GetDataType());
    auto reshapeNode = BuildUnaryNode(graph, "Reshape", "reshape_scale", outputDesc);
    LinkUnaryNode(graph, graphBuilder, reshapeOutput, reshapeNode, outputDesc);
    scaleInput = reshapeOutput;
}

static void LinkQuantInput(Graph* graph, GNode& quantBmmV3, const TensorWithDesc& input, int64_t inputIndex)
{
    ge::es::AddEdgeAndUpdatePeerDesc(*graph, *input.tensor.GetProducer(), input.tensor.GetProducerOutIndex(),
                                     quantBmmV3, inputIndex);
    quantBmmV3.UpdateInputDesc(inputIndex, input.desc);
}

static GraphPtr BuildGraph(const QuantBatchMatmulV3TransposeFusionPassParam& param)
{
    auto graphBuilder = ge::es::EsGraphBuilder(param.caseName.c_str());
    auto* graph = graphBuilder.GetCGraphBuilder()->GetGraph();

    auto x1Desc = MakeTensorDesc(param.x1Shape, param.x1Dtype);
    auto x2Desc = MakeTensorDesc(param.x2Shape, param.x2Dtype);
    auto scaleDesc = MakeTensorDesc(param.scaleShape, param.scaleDtype);
    auto outDesc = MakeTensorDesc(param.outShape, param.outDtype);

    auto dataX1 = graphBuilder.CreateInput(0, "dataX1", param.x1Dtype, FORMAT_ND, param.x1Shape);
    auto dataX2 = graphBuilder.CreateInput(1, "dataX2", param.x2Dtype, FORMAT_ND, param.x2Shape);
    auto dataScale = graphBuilder.CreateInput(2, "dataScale", param.scaleDtype, FORMAT_ND, param.scaleShape);
    dataX1.GetProducer()->UpdateOutputDesc(0, x1Desc);
    dataX2.GetProducer()->UpdateOutputDesc(0, x2Desc);
    dataScale.GetProducer()->UpdateOutputDesc(0, scaleDesc);

    TensorWithDesc rawX1 = {dataX1, x1Desc};
    TensorWithDesc rawX2 = {dataX2, x2Desc};
    TensorWithDesc rawScale = {dataScale, scaleDesc};

    auto x1Input = rawX1;
    AddTransposeIfNeeded(graph, graphBuilder, x1Input, param.transposeX1, "transpose_x1");
    AddBitcastIfNeeded(graph, graphBuilder, x1Input, param.bitcastX1, "bitcast_x1");

    auto x2Input = rawX2;
    AddTransposeIfNeeded(graph, graphBuilder, x2Input, param.transposeX2, "transpose_x2");
    AddBitcastIfNeeded(graph, graphBuilder, x2Input, param.bitcastX2, "bitcast_x2");

    auto scaleInput = rawScale;
    const auto& reshapeInput = param.reshapeScaleFrom == "x2" ? rawX2 : rawScale;
    AddReshapeIfNeeded(graph, graphBuilder, scaleInput, reshapeInput, param.reshapeScale, param.reshapeScaleShape);

    const bool hasX1Scale = !param.x1ScaleShape.empty();
    TensorWithDesc x1ScaleInput;
    if (hasX1Scale) {
        x1ScaleInput.desc = MakeTensorDesc(param.x1ScaleShape, param.x1ScaleDtype);
        x1ScaleInput.tensor = graphBuilder.CreateInput(3, "dataX1Scale", param.x1ScaleDtype, FORMAT_ND,
                                                       param.x1ScaleShape);
        x1ScaleInput.tensor.GetProducer()->UpdateOutputDesc(0, x1ScaleInput.desc);
    }

    auto quantBmmV3 = ge::es::CompliantNodeBuilder(graph)
                          .OpType("QuantBatchMatmulV3")
                          .Name("QuantBatchMatmulV3")
                          .IrDefInputs({
                              {"x1", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                              {"x2", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                              {"scale", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                              {"offset", ge::es::CompliantNodeBuilder::kEsIrInputOptional, ""},
                              {"bias", ge::es::CompliantNodeBuilder::kEsIrInputOptional, ""},
                              {"pertoken_scale", ge::es::CompliantNodeBuilder::kEsIrInputOptional, ""},
                          })
                          .IrDefOutputs({{"y", ge::es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                          .InstanceOutputDataType("y", param.outDtype)
                          .InstanceOutputShape("y", param.outShape)
                          .InstanceOutputFormat("y", FORMAT_ND)
                          .Build();
    LinkQuantInput(graph, quantBmmV3, x1Input, X1_INDEX);
    LinkQuantInput(graph, quantBmmV3, x2Input, X2_INDEX);
    LinkQuantInput(graph, quantBmmV3, scaleInput, SCALE_INDEX);
    if (hasX1Scale) {
        LinkQuantInput(graph, quantBmmV3, x1ScaleInput, PERTOKEN_SCALE_INDEX);
    }
    quantBmmV3.UpdateOutputDesc(0, outDesc);
    int32_t dtypeAttr = static_cast<int32_t>(param.dtypeAttr);
    quantBmmV3.SetAttr("dtype", dtypeAttr);
    bool transposeX1 = false;
    bool transposeX2 = false;
    quantBmmV3.SetAttr("transpose_x1", transposeX1);
    quantBmmV3.SetAttr("transpose_x2", transposeX2);
    if (param.groupSize >= 0) {
        int64_t groupSize = param.groupSize;
        quantBmmV3.SetAttr("group_size", groupSize);
    }

    auto output = ge::es::EsTensorHolder(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(quantBmmV3, 0));
    return graphBuilder.BuildAndReset({output});
}

static int CountNodes(const GraphPtr& graph, const char* nodeType)
{
    int count = 0;
    for (auto node : graph->GetAllNodes()) {
        AscendString type;
        if (node.GetType(type) == GRAPH_SUCCESS && type == nodeType) {
            count++;
        }
    }
    return count;
}

static GNode FindQuantBatchMatmulV3(const GraphPtr& graph)
{
    for (auto node : graph->GetAllNodes()) {
        AscendString type;
        if (node.GetType(type) == GRAPH_SUCCESS && type == "QuantBatchMatmulV3") {
            return node;
        }
    }
    throw std::runtime_error("QuantBatchMatmulV3 node is not found.");
}

static void ExpectTransposeAttrs(const QuantBatchMatmulV3TransposeFusionPassParam& param, GNode& node)
{
    bool transposeX1 = false;
    bool transposeX2 = false;
    ASSERT_EQ(node.GetAttr("transpose_x1", transposeX1), GRAPH_SUCCESS);
    ASSERT_EQ(node.GetAttr("transpose_x2", transposeX2), GRAPH_SUCCESS);
    EXPECT_EQ(transposeX1, param.expectedTransposeX1);
    EXPECT_EQ(transposeX2, param.expectedTransposeX2);
}

static std::string TestName(const testing::TestParamInfo<QuantBatchMatmulV3TransposeFusionPassParam>& info)
{
    std::string name = info.param.caseName;
    for (auto& ch : name) {
        if (!std::isalnum(static_cast<unsigned char>(ch))) {
            ch = '_';
        }
    }
    return name;
}
} // namespace

class QuantBatchMatmulV3TransposeFusionPassTest
    : public testing::TestWithParam<QuantBatchMatmulV3TransposeFusionPassParam> {};

TEST_P(QuantBatchMatmulV3TransposeFusionPassTest, RunFusionPass)
{
    const auto& param = GetParam();
    auto graph = BuildGraph(param);

    CustomPassContext passContext;
    Status status = SUCCESS;
    SetPlatformSupport();
    if (param.passName == QUANT_BATCH_MATMUL_V3_TRANSPOSE_LIMIT_FUSION_PASS_NAME) {
        passContext.SetPassName(QUANT_BATCH_MATMUL_V3_TRANSPOSE_LIMIT_FUSION_PASS_NAME);
        TestQuantBatchMatmulV3TransposeLimitFusionPass pass;
        status = pass.RunForTest(graph, passContext);
    } else {
        passContext.SetPassName(QUANT_BATCH_MATMUL_V3_TRANSPOSE_FUSION_PASS_NAME);
        TestQuantBatchMatmulV3TransposeFusionPass pass;
        status = pass.RunForTest(graph, passContext);
    }
    ClearPlatformSupport();

    ASSERT_EQ(status, param.expectedStatus);
    EXPECT_EQ(CountNodes(graph, "Transpose") + CountNodes(graph, "TransposeD"), param.expectedTransposeCount);
    EXPECT_EQ(CountNodes(graph, "Reshape"), param.expectedReshapeCount);

    auto quantNode = FindQuantBatchMatmulV3(graph);
    ExpectTransposeAttrs(param, quantNode);
}

INSTANTIATE_TEST_CASE_P(QuantBatchMatmulV3TransposeFusionPass, QuantBatchMatmulV3TransposeFusionPassTest,
                        testing::ValuesIn(GetParams()), TestName);

/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <array>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "es_nn_ops.h"
#include "ge/es_graph_builder.h"
#include "register/register_custom_pass.h"
#include "../../../op_graph/fusion_pass/quant_batch_matmul_v4_to_v3_fusion_pass.h"
#include "../../../../../tests/ut/common/ut_string_utils.h"

using namespace ge;
using namespace ge::fusion;
using namespace ut_str;

namespace {
constexpr char QUANT_BATCH_MATMUL_V4_TO_V3_FUSION_PASS_NAME[] = "QuantBatchMatmulV4ToV3FusionPass";
constexpr char TEST_CSV_FILE[] = "test_quant_batch_matmul_v4_to_v3_fusion_pass.csv";
#ifndef QUANT_BATCH_MATMUL_V4_TO_V3_FUSION_PASS_CSV_PATH
#define QUANT_BATCH_MATMUL_V4_TO_V3_FUSION_PASS_CSV_PATH ""
#endif
constexpr size_t CASE_FIELD_NUM = 23;
constexpr size_t V4_INPUT_NUM = 10;
constexpr size_t V4_X1_INDEX = 0;
constexpr size_t V4_X2_INDEX = 1;
constexpr size_t V4_BIAS_INDEX = 2;
constexpr size_t V4_X1SCALE_INDEX = 3;
constexpr size_t V4_X2SCALE_INDEX = 4;
constexpr size_t V4_YSCALE_INDEX = 5;
constexpr size_t V4_X1OFFSET_INDEX = 6;
constexpr size_t V4_X2OFFSET_INDEX = 7;
constexpr size_t V4_YOFFSET_INDEX = 8;
constexpr size_t V4_X2TABLE_INDEX = 9;
constexpr size_t V3_X1_INDEX = 0;
constexpr size_t V3_X2_INDEX = 1;
constexpr size_t V3_SCALE_INDEX = 2;
constexpr size_t V3_OFFSET_INDEX = 3;
constexpr size_t V3_BIAS_INDEX = 4;
constexpr size_t V3_PERTOKENSCALE_INDEX = 5;

struct QuantBatchMatmulV4ToV3FusionPassParam {
    std::string caseName;
    DataType x1Dtype;
    DataType x2Dtype;
    DataType biasDtype;
    DataType x1ScaleDtype;
    DataType x2ScaleDtype;
    DataType x2OffsetDtype;
    DataType yDtype;
    int dtypeAttr;
    int64_t groupSize;
    bool transposeX1;
    bool transposeX2;
    bool hasBias;
    bool hasX1Scale;
    bool hasX2Scale;
    bool hasX2Offset;
    int64_t x1DimNum;
    int64_t x2DimNum;
    int64_t x1ScaleDimNum;
    int64_t x2ScaleDimNum;
    Status expectedStatus;
    bool expectFusion;
    int64_t expectedGroupSize;
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
        QUANT_BATCH_MATMUL_V4_TO_V3_FUSION_PASS_CSV_PATH,
        GetDirName(__FILE__) + "/" + TEST_CSV_FILE,
        TEST_CSV_FILE,
        "../../../../matmul/quant_batch_matmul_v4/tests/ut/op_graph/" + std::string(TEST_CSV_FILE),
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

static QuantBatchMatmulV4ToV3FusionPassParam ParseParam(const std::vector<std::string>& fields)
{
    if (fields.size() != CASE_FIELD_NUM) {
        throw std::runtime_error("Invalid csv column size.");
    }
    size_t index = 0;
    QuantBatchMatmulV4ToV3FusionPassParam param;
    param.caseName = fields[index++];
    param.x1Dtype = ParseDtype(fields[index++], DT_UNDEFINED);
    param.x2Dtype = ParseDtype(fields[index++], DT_UNDEFINED);
    param.biasDtype = ParseDtype(fields[index++], DT_UNDEFINED);
    param.x1ScaleDtype = ParseDtype(fields[index++], DT_UNDEFINED);
    param.x2ScaleDtype = ParseDtype(fields[index++], DT_UNDEFINED);
    param.x2OffsetDtype = ParseDtype(fields[index++], DT_UNDEFINED);
    param.yDtype = ParseDtype(fields[index++], DT_UNDEFINED);
    param.dtypeAttr = ParseIntOrDefault(fields[index++], -1);
    param.groupSize = ParseInt64OrDefault(fields[index++], -1);
    param.transposeX1 = ParseBool(fields[index++]);
    param.transposeX2 = ParseBool(fields[index++]);
    param.hasBias = ParseBool(fields[index++]);
    param.hasX1Scale = ParseBool(fields[index++]);
    param.hasX2Scale = ParseBool(fields[index++]);
    param.hasX2Offset = ParseBool(fields[index++]);
    param.x1DimNum = ParseInt64OrDefault(fields[index++], 0);
    param.x2DimNum = ParseInt64OrDefault(fields[index++], 0);
    param.x1ScaleDimNum = ParseInt64OrDefault(fields[index++], 0);
    param.x2ScaleDimNum = ParseInt64OrDefault(fields[index++], 0);
    param.expectedStatus = ToStatus(fields[index++]);
    param.expectFusion = ParseBool(fields[index++]);
    param.expectedGroupSize = ParseInt64OrDefault(fields[index++], 0);
    return param;
}

static std::vector<QuantBatchMatmulV4ToV3FusionPassParam> GetParams()
{
    std::ifstream csvData = OpenCsvData();

    std::vector<QuantBatchMatmulV4ToV3FusionPassParam> params;
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

static std::vector<int64_t> MakeDims(int64_t dimNum)
{
    if (dimNum <= 0) {
        throw std::invalid_argument("dimNum must be positive");
    }
    std::vector<int64_t> dims(static_cast<size_t>(dimNum), 16);
    if (dims.size() > 2) {
        dims[0] = 2;
    }
    return dims;
}

static TensorDesc MakeTensorDesc(DataType dtype, int64_t dimNum)
{
    Shape shape(MakeDims(dimNum));
    TensorDesc desc;
    desc.SetDataType(dtype);
    desc.SetFormat(FORMAT_ND);
    desc.SetOriginFormat(FORMAT_ND);
    desc.SetShape(shape);
    desc.SetOriginShape(shape);
    return desc;
}

static ge::es::EsTensorHolder CreateInput(ge::es::EsGraphBuilder& graphBuilder, size_t index, const std::string& name,
                                          DataType dtype, int64_t dimNum)
{
    auto input = graphBuilder.CreateInput(static_cast<int64_t>(index), name.c_str(), dtype, FORMAT_ND,
                                          MakeDims(dimNum));
    input.GetProducer()->UpdateOutputDesc(0, MakeTensorDesc(dtype, dimNum));
    return input;
}

template <typename NodePtr>
static void UpdateInputDesc(const NodePtr& node, size_t inputIndex, DataType dtype, int64_t dimNum)
{
    node->UpdateInputDesc(inputIndex, MakeTensorDesc(dtype, dimNum));
}

static std::shared_ptr<Graph> BuildGraph(const QuantBatchMatmulV4ToV3FusionPassParam& param)
{
    auto graphBuilder = ge::es::EsGraphBuilder(param.caseName.c_str());
    std::array<ge::es::EsTensorHolder, V4_INPUT_NUM> inputs;
    size_t graphInputIndex = 0;

    inputs[V4_X1_INDEX] = CreateInput(graphBuilder, graphInputIndex++, "x1", param.x1Dtype, param.x1DimNum);
    inputs[V4_X2_INDEX] = CreateInput(graphBuilder, graphInputIndex++, "x2", param.x2Dtype, param.x2DimNum);
    if (param.hasBias) {
        inputs[V4_BIAS_INDEX] = CreateInput(graphBuilder, graphInputIndex++, "bias", param.biasDtype, 1);
    } else {
        inputs[V4_BIAS_INDEX] = nullptr;
    }
    if (param.hasX1Scale) {
        inputs[V4_X1SCALE_INDEX] = CreateInput(graphBuilder, graphInputIndex++, "x1_scale", param.x1ScaleDtype,
                                               param.x1ScaleDimNum);
    } else {
        inputs[V4_X1SCALE_INDEX] = nullptr;
    }
    if (param.hasX2Scale) {
        inputs[V4_X2SCALE_INDEX] = CreateInput(graphBuilder, graphInputIndex++, "x2_scale", param.x2ScaleDtype,
                                               param.x2ScaleDimNum);
    } else {
        inputs[V4_X2SCALE_INDEX] = nullptr;
    }
    inputs[V4_YSCALE_INDEX] = nullptr;
    inputs[V4_X1OFFSET_INDEX] = nullptr;
    if (param.hasX2Offset) {
        inputs[V4_X2OFFSET_INDEX] = CreateInput(graphBuilder, graphInputIndex++, "x2_offset", param.x2OffsetDtype, 1);
    } else {
        inputs[V4_X2OFFSET_INDEX] = nullptr;
    }
    inputs[V4_YOFFSET_INDEX] = nullptr;
    inputs[V4_X2TABLE_INDEX] = nullptr;

    auto y = ge::es::QuantBatchMatmulV4(inputs[V4_X1_INDEX], inputs[V4_X2_INDEX], inputs[V4_BIAS_INDEX],
                                        inputs[V4_X1SCALE_INDEX], inputs[V4_X2SCALE_INDEX], inputs[V4_YSCALE_INDEX],
                                        inputs[V4_X1OFFSET_INDEX], inputs[V4_X2OFFSET_INDEX], inputs[V4_YOFFSET_INDEX],
                                        inputs[V4_X2TABLE_INDEX], param.dtypeAttr);
    auto qmmNode = y.GetProducer();
    int32_t dtypeAttr = param.dtypeAttr;
    bool transposeX1 = param.transposeX1;
    bool transposeX2 = param.transposeX2;
    int64_t groupSize = param.groupSize;
    qmmNode->SetAttr("dtype", dtypeAttr);
    qmmNode->SetAttr("transpose_x1", transposeX1);
    qmmNode->SetAttr("transpose_x2", transposeX2);
    qmmNode->SetAttr("group_size", groupSize);

    UpdateInputDesc(qmmNode, V4_X1_INDEX, param.x1Dtype, param.x1DimNum);
    UpdateInputDesc(qmmNode, V4_X2_INDEX, param.x2Dtype, param.x2DimNum);
    if (param.hasBias) {
        UpdateInputDesc(qmmNode, V4_BIAS_INDEX, param.biasDtype, 1);
    }
    if (param.hasX1Scale) {
        UpdateInputDesc(qmmNode, V4_X1SCALE_INDEX, param.x1ScaleDtype, param.x1ScaleDimNum);
    }
    if (param.hasX2Scale) {
        UpdateInputDesc(qmmNode, V4_X2SCALE_INDEX, param.x2ScaleDtype, param.x2ScaleDimNum);
    }
    if (param.hasX2Offset) {
        UpdateInputDesc(qmmNode, V4_X2OFFSET_INDEX, param.x2OffsetDtype, 1);
    }
    qmmNode->UpdateOutputDesc(0, MakeTensorDesc(param.yDtype, 2));

    return graphBuilder.BuildAndReset({y});
}

static int CountNodes(const std::shared_ptr<Graph>& graph, const char* nodeType)
{
    int count = 0;
    for (auto node : graph->GetAllNodes()) {
        AscendString type;
        node.GetType(type);
        if (type == nodeType) {
            count++;
        }
    }
    return count;
}

static void ExpectInputDtype(GNode& node, size_t inputIndex, DataType expectedDtype)
{
    TensorDesc desc;
    ASSERT_EQ(node.GetInputDesc(inputIndex, desc), GRAPH_SUCCESS);
    EXPECT_EQ(desc.GetDataType(), expectedDtype);
}

static void CheckFusedNode(const QuantBatchMatmulV4ToV3FusionPassParam& param, GNode& node)
{
    ExpectInputDtype(node, V3_X1_INDEX, param.x1Dtype);
    ExpectInputDtype(node, V3_X2_INDEX, param.x2Dtype);
    if (param.hasX2Scale) {
        ExpectInputDtype(node, V3_SCALE_INDEX, param.x2ScaleDtype);
    }
    if (param.hasX2Offset) {
        ExpectInputDtype(node, V3_OFFSET_INDEX, param.x2OffsetDtype);
    }
    if (param.hasBias) {
        ExpectInputDtype(node, V3_BIAS_INDEX, param.biasDtype);
    }
    if (param.hasX1Scale) {
        ExpectInputDtype(node, V3_PERTOKENSCALE_INDEX, param.x1ScaleDtype);
    }

    TensorDesc outputDesc;
    ASSERT_EQ(node.GetOutputDesc(0, outputDesc), GRAPH_SUCCESS);
    EXPECT_EQ(outputDesc.GetDataType(), param.yDtype);

    int dtype = -1;
    bool transposeX1 = false;
    bool transposeX2 = false;
    int64_t groupSize = 0;
    node.GetAttr("dtype", dtype);
    node.GetAttr("transpose_x1", transposeX1);
    node.GetAttr("transpose_x2", transposeX2);
    node.GetAttr("group_size", groupSize);
    EXPECT_EQ(dtype, param.dtypeAttr);
    EXPECT_EQ(transposeX1, param.transposeX1);
    EXPECT_EQ(transposeX2, param.transposeX2);
    EXPECT_EQ(groupSize, param.expectedGroupSize);
}

static std::string TestName(const testing::TestParamInfo<QuantBatchMatmulV4ToV3FusionPassParam>& info)
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

class QuantBatchMatmulV4ToV3FusionPassTest : public testing::TestWithParam<QuantBatchMatmulV4ToV3FusionPassParam> {};

TEST_P(QuantBatchMatmulV4ToV3FusionPassTest, RunFusionPass)
{
    const auto& param = GetParam();
    auto graph = BuildGraph(param);

    CustomPassContext passContext;
    passContext.SetPassName(QUANT_BATCH_MATMUL_V4_TO_V3_FUSION_PASS_NAME);
    ops::QuantBatchMatmulV4ToV3FusionPass pass;
    Status status = pass.Run(graph, passContext);
    ASSERT_EQ(status, param.expectedStatus);

    EXPECT_EQ(CountNodes(graph, "QuantBatchMatmulV4"), param.expectFusion ? 0 : 1);
    EXPECT_EQ(CountNodes(graph, "QuantBatchMatmulV3"), param.expectFusion ? 1 : 0);

    if (!param.expectFusion) {
        return;
    }
    for (auto node : graph->GetAllNodes()) {
        AscendString type;
        node.GetType(type);
        if (type == "QuantBatchMatmulV3") {
            CheckFusedNode(param, node);
            return;
        }
    }
    FAIL() << "QuantBatchMatmulV3 node is not found.";
}

INSTANTIATE_TEST_CASE_P(QuantBatchMatmulV4ToV3FusionPass, QuantBatchMatmulV4ToV3FusionPassTest,
                        testing::ValuesIn(GetParams()), TestName);

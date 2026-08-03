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
#include "../../../op_graph/fusion_pass/z_z_matmul_t_qbmmv3_fusion_pass.h"
#if __has_include("aclnn/aclnn_base.h")
#include "aclnn/aclnn_base.h"
#ifndef ACLNN_SUCCESS
#define ACLNN_SUCCESS static_cast<aclnnStatus>(0)
#define ZZ_MATMUL_T_QBMMV3_UT_DEFINED_ACLNN_SUCCESS
#endif
#ifndef ACLNN_ERR_PARAM_INVALID
#define ACLNN_ERR_PARAM_INVALID static_cast<aclnnStatus>(-1)
#define ZZ_MATMUL_T_QBMMV3_UT_DEFINED_ACLNN_ERR_PARAM_INVALID
#endif
#ifndef ACLNN_ERR_PARAM_NULLPTR
#define ACLNN_ERR_PARAM_NULLPTR static_cast<aclnnStatus>(-2)
#define ZZ_MATMUL_T_QBMMV3_UT_DEFINED_ACLNN_ERR_PARAM_NULLPTR
#endif
#endif
#include "../../../../../tests/ut/common/ut_string_utils.h"
#ifdef ZZ_MATMUL_T_QBMMV3_UT_DEFINED_ACLNN_SUCCESS
#undef ACLNN_SUCCESS
#undef ZZ_MATMUL_T_QBMMV3_UT_DEFINED_ACLNN_SUCCESS
#endif
#ifdef ZZ_MATMUL_T_QBMMV3_UT_DEFINED_ACLNN_ERR_PARAM_INVALID
#undef ACLNN_ERR_PARAM_INVALID
#undef ZZ_MATMUL_T_QBMMV3_UT_DEFINED_ACLNN_ERR_PARAM_INVALID
#endif
#ifdef ZZ_MATMUL_T_QBMMV3_UT_DEFINED_ACLNN_ERR_PARAM_NULLPTR
#undef ACLNN_ERR_PARAM_NULLPTR
#undef ZZ_MATMUL_T_QBMMV3_UT_DEFINED_ACLNN_ERR_PARAM_NULLPTR
#endif

using namespace ge;
using namespace ge::es;
using namespace ge::fusion;
using namespace ut_str;

namespace {

constexpr char ZZ_MATMUL_T_QBMMV3_FUSION_PASS_NAME[] = "ZZMatMulTOQBMMV3FusionPass";
constexpr char TEST_CSV_FILE[] = "test_z_z_matmul_t_qbmmv3_fusion_pass.csv";
#ifndef ZZ_MATMUL_T_QBMMV3_FUSION_PASS_CSV_PATH
#define ZZ_MATMUL_T_QBMMV3_FUSION_PASS_CSV_PATH ""
#endif
constexpr size_t CASE_FIELD_NUM = 18;
constexpr int64_t kV3BiasIdx = 4;

struct ZZMatMulTOQBMMV3FusionPassParam {
    std::string caseName;
    std::string opType;
    std::vector<int64_t> x1Shape;
    DataType x1Dtype;
    std::vector<int64_t> x2Shape;
    DataType x2Dtype;
    std::vector<int64_t> outShape;
    DataType outDtype;
    bool transposeX1;
    bool transposeX2;
    std::vector<int64_t> biasShape;
    std::string socVersion;
    Status expectedStatus;
    int expectedOldOpCount;
    int expectedQbmmV3Count;
    bool expectedTransposeX1;
    bool expectedTransposeX2;
    int64_t expectedDtypeAttr;
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
        ZZ_MATMUL_T_QBMMV3_FUSION_PASS_CSV_PATH,
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

static std::vector<int64_t> ParseShape(const std::string& value)
{
    if (value.empty()) {
        return {};
    }
    return ParseInt64Vec(value);
}

static int64_t DtypeToInt64(const std::string& value) { return static_cast<int64_t>(ParseDtype(value, DT_UNDEFINED)); }

static void SetPlatformInfo(const std::string& socVersion)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optionalInfo;
    platformInfo.soc_info.ai_core_cnt = 24;
    platformInfo.ai_core_spec.l1_size = 512 * 1024;
    platformInfo.soc_info.l2_size = 192 * 1024 * 1024;
    optionalInfo.soc_version = socVersion;
    platformInfo.str_info.short_soc_version = socVersion;
    fe::PlatformInfoManager::Instance().platform_info_map_.clear();
    fe::PlatformInfoManager::Instance().platform_info_map_[socVersion] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optionalInfo);
}

static void ClearPlatformSupport() { fe::PlatformInfoManager::Instance().platform_info_map_.clear(); }

static ZZMatMulTOQBMMV3FusionPassParam ParseParam(const std::vector<std::string>& fields)
{
    if (fields.size() != CASE_FIELD_NUM) {
        throw std::runtime_error("Invalid csv column size.");
    }
    size_t index = 0;
    ZZMatMulTOQBMMV3FusionPassParam param;
    param.caseName = fields[index++];
    param.opType = fields[index++];
    param.x1Shape = ParseShape(fields[index++]);
    param.x1Dtype = ParseDtype(fields[index++], DT_UNDEFINED);
    param.x2Shape = ParseShape(fields[index++]);
    param.x2Dtype = ParseDtype(fields[index++], DT_UNDEFINED);
    param.outShape = ParseShape(fields[index++]);
    param.outDtype = ParseDtype(fields[index++], DT_UNDEFINED);
    param.transposeX1 = ParseBool(fields[index++]);
    param.transposeX2 = ParseBool(fields[index++]);
    param.biasShape = ParseShape(fields[index++]);
    param.socVersion = fields[index++];
    param.expectedStatus = ToStatus(fields[index++]);
    param.expectedOldOpCount = ParseIntOrDefault(fields[index++], 0);
    param.expectedQbmmV3Count = ParseIntOrDefault(fields[index++], 0);
    param.expectedTransposeX1 = ParseBool(fields[index++]);
    param.expectedTransposeX2 = ParseBool(fields[index++]);
    param.expectedDtypeAttr = DtypeToInt64(fields[index++]);
    return param;
}

static std::vector<ZZMatMulTOQBMMV3FusionPassParam> GetParams()
{
    std::ifstream csvData = OpenCsvData();

    std::vector<ZZMatMulTOQBMMV3FusionPassParam> params;
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

static TensorDesc MakeTensorDesc(const std::vector<int64_t>& shape, DataType dtype, Format format = FORMAT_ND)
{
    TensorDesc desc(ge::Shape(shape), format, dtype);
    desc.SetOriginFormat(format);
    desc.SetOriginShape(ge::Shape(shape));
    return desc;
}

static bool IsBatchOp(const std::string& opType)
{
    return opType == "BatchMatMul" || opType == "BatchMatMulV2" || opType == "BatchMatMulV3";
}

static bool IsV2OrV3Op(const std::string& opType)
{
    return opType == "MatMulV2" || opType == "BatchMatMulV2" || opType == "MatMulV3" || opType == "BatchMatMulV3";
}

static GraphPtr BuildGraph(const ZZMatMulTOQBMMV3FusionPassParam& param)
{
    auto graphBuilder = EsGraphBuilder(param.caseName.c_str());
    auto* graph = graphBuilder.GetCGraphBuilder()->GetGraph();

    auto x1Desc = MakeTensorDesc(param.x1Shape, param.x1Dtype);
    auto x2Desc = MakeTensorDesc(param.x2Shape, param.x2Dtype);
    auto outDesc = MakeTensorDesc(param.outShape, param.outDtype);

    auto dataX1 = graphBuilder.CreateInput(0, "dataX1", param.x1Dtype, FORMAT_ND, param.x1Shape);
    auto dataX2 = graphBuilder.CreateInput(1, "dataX2", param.x2Dtype, FORMAT_ND, param.x2Shape);
    dataX1.GetProducer()->UpdateOutputDesc(0, x1Desc);
    dataX2.GetProducer()->UpdateOutputDesc(0, x2Desc);

    bool hasBias = !param.biasShape.empty();
    EsTensorHolder dataBias = nullptr;
    if (hasBias) {
        auto biasDesc = MakeTensorDesc(param.biasShape, param.outDtype);
        dataBias = graphBuilder.CreateInput(2, "dataBias", param.outDtype, FORMAT_ND, param.biasShape);
        dataBias.GetProducer()->UpdateOutputDesc(0, biasDesc);
    }

    bool isBatch = IsBatchOp(param.opType);
    const char* transAttr1 = isBatch ? "adj_x1" : "transpose_x1";
    const char* transAttr2 = isBatch ? "adj_x2" : "transpose_x2";

    std::vector<CompliantNodeBuilder::IrInputDef> irInputs = {
        {"x1", CompliantNodeBuilder::kEsIrInputRequired, ""},
        {"x2", CompliantNodeBuilder::kEsIrInputRequired, ""},
    };
    if (hasBias) {
        irInputs.push_back({"bias", CompliantNodeBuilder::kEsIrInputOptional, ""});
    }
    if (IsV2OrV3Op(param.opType) && !hasBias) {
        irInputs.push_back({"offset_w", CompliantNodeBuilder::kEsIrInputOptional, ""});
    }

    auto node = CompliantNodeBuilder(graph)
                    .OpType(param.opType.c_str())
                    .Name(param.caseName.c_str())
                    .IrDefInputs(irInputs)
                    .IrDefOutputs({{"y", CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                    .IrDefAttrs({
                        {transAttr1, CompliantNodeBuilder::kEsAttrRequired, "Bool", CreateFrom(false)},
                        {transAttr2, CompliantNodeBuilder::kEsAttrRequired, "Bool", CreateFrom(false)},
                    })
                    .Build();

    AddEdgeAndUpdatePeerDesc(*graph, *dataX1.GetProducer(), dataX1.GetProducerOutIndex(), node, 0);
    AddEdgeAndUpdatePeerDesc(*graph, *dataX2.GetProducer(), dataX2.GetProducerOutIndex(), node, 1);
    node.UpdateInputDesc(0, x1Desc);
    node.UpdateInputDesc(1, x2Desc);
    if (hasBias) {
        AddEdgeAndUpdatePeerDesc(*graph, *dataBias.GetProducer(), dataBias.GetProducerOutIndex(), node, 2);
        TensorDesc biasDesc = MakeTensorDesc(param.biasShape, param.outDtype);
        node.UpdateInputDesc(2, biasDesc);
    }
    node.UpdateOutputDesc(0, outDesc);
    bool transX1 = param.transposeX1;
    bool transX2 = param.transposeX2;
    node.SetAttr(transAttr1, transX1);
    node.SetAttr(transAttr2, transX2);

    auto output = EsTensorHolder(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(node, 0));
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

static GNode FindNodeByOpType(const GraphPtr& graph, const char* opType)
{
    for (auto node : graph->GetAllNodes()) {
        AscendString type;
        if (node.GetType(type) == GRAPH_SUCCESS && type == opType) {
            return node;
        }
    }
    throw std::runtime_error(std::string(opType) + " node is not found.");
}

static std::string TestName(const testing::TestParamInfo<ZZMatMulTOQBMMV3FusionPassParam>& info)
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

class ZZMatMulTOQBMMV3FusionPassTest : public testing::TestWithParam<ZZMatMulTOQBMMV3FusionPassParam> {};

TEST_P(ZZMatMulTOQBMMV3FusionPassTest, RunFusionPass)
{
    const auto& param = GetParam();
    auto graph = BuildGraph(param);

    SetPlatformInfo(param.socVersion);

    CustomPassContext passContext;
    passContext.SetPassName(ZZ_MATMUL_T_QBMMV3_FUSION_PASS_NAME);
    ops::ZZMatMulTOQBMMV3FusionPass pass;
    Status status = pass.Run(graph, passContext);

    ClearPlatformSupport();

    ASSERT_EQ(status, param.expectedStatus);
    EXPECT_EQ(CountNodes(graph, param.opType.c_str()), param.expectedOldOpCount);
    EXPECT_EQ(CountNodes(graph, "QuantBatchMatmulV3"), param.expectedQbmmV3Count);

    if (param.expectedStatus == SUCCESS) {
        auto qbmmV3Node = FindNodeByOpType(graph, "QuantBatchMatmulV3");
        bool transposeX1 = false;
        bool transposeX2 = false;
        int64_t dtype = 0;
        qbmmV3Node.GetAttr("transpose_x1", transposeX1);
        qbmmV3Node.GetAttr("transpose_x2", transposeX2);
        qbmmV3Node.GetAttr("dtype", dtype);
        EXPECT_EQ(transposeX1, param.expectedTransposeX1);
        EXPECT_EQ(transposeX2, param.expectedTransposeX2);
        EXPECT_EQ(dtype, param.expectedDtypeAttr);
        if (!param.biasShape.empty()) {
            TensorDesc biasDesc;
            ASSERT_EQ(qbmmV3Node.GetInputDesc(kV3BiasIdx, biasDesc), GRAPH_SUCCESS);
            EXPECT_EQ(biasDesc.GetDataType(), param.outDtype);
        }
    }
}

INSTANTIATE_TEST_CASE_P(ZZMatMulTOQBMMV3FusionPass, ZZMatMulTOQBMMV3FusionPassTest, testing::ValuesIn(GetParams()),
                        TestName);

class ZZMatMulTOQBMMV3FusionPassPatternTest : public testing::Test {};

TEST_F(ZZMatMulTOQBMMV3FusionPassPatternTest, patternTest)
{
    ops::ZZMatMulTOQBMMV3FusionPass pass;
    std::vector<PatternUniqPtr> patterns = pass.Patterns();
    EXPECT_GT(patterns.size(), 0);
}

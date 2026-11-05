/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the License).
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "ge/compliant_node_builder.h"
#include "ge/es_graph_builder.h"
#include "platform/platform_info.h"
#include "register/register_custom_pass.h"
#include "es_nn_ops.h"
#include "../../../op_graph/fusion_pass/fused_mat_mul_fusion_pass.h"

using namespace ge;
using namespace ge::es;
using namespace ge::fusion;
using namespace fe;
using namespace ops;

namespace ge {
namespace ut_detail {
// The option overrides installed by the current test (see ScopedGeOptions below).
std::map<std::string, std::string>& GetOptionsMap()
{
    static std::map<std::string, std::string> options;
    return options;
}
} // namespace ut_detail

// UT interposition of the option lookup: the real implementation reads the global GEContext
// options which have no UT injection path (the internal setters use the old C++ ABI). The UT
// binary links the option consumers (the common option bridges) statically from
// libnn_op_graph_ut_static_lib.a with --whole-archive, so this definition binds first.
graphStatus CustomPassContext::GetOptionValue(const AscendString& optionKey, AscendString& optionValue) const
{
    const auto& options = ut_detail::GetOptionsMap();
    const auto it = options.find(std::string(optionKey.GetString()));
    if (it == options.end()) {
        return GRAPH_FAILED; // an absent option fails, same as the real lookup
    }
    optionValue = AscendString(it->second.c_str());
    return GRAPH_SUCCESS;
}
} // namespace ge

namespace fe {
// UT interposition of the FE resolved precision mode: the pass resolves the mode via the weak
// fe::Configuration::GetPrecisionModeStr symbol (the exact call the canndev pass makes), the
// UT defines it strongly so the weak reference binds here (the same interposition pattern as
// the CustomPassContext::GetOptionValue above) reading the same injected option map - the real
// libaicore_utils is not loaded in the UT, and this keeps the precision mode scenarios
// controllable (an empty map means "unset", the same precondition as the device run).
class Configuration {
public:
    static std::string GetPrecisionModeStr();
};

std::string Configuration::GetPrecisionModeStr()
{
    const auto& options = ge::ut_detail::GetOptionsMap();
    for (const char* key : {"ge.exec.precision_mode", "precision_mode"}) {
        const auto it = options.find(key);
        if (it != options.end()) {
            return it->second;
        }
    }
    return ""; // unset: the real FE would run its default branches here
}
} // namespace fe

namespace {

constexpr char kOpTypeFusedMatMul[] = "FusedMatMul";
constexpr char kOpTypeMatMulV3[] = "MatMulV3";
constexpr char kOpTypeBatchMatMulV3[] = "BatchMatMulV3";
constexpr char kOpTypeRelu[] = "Relu";
constexpr char kOpTypeAscendQuant[] = "AscendQuant";
constexpr char kFusedOpTypeRelu[] = "relu";
constexpr int64_t kHf32OpImplMode = 0x40;
constexpr int64_t kDefaultOpImplMode = 0x1;
constexpr uint32_t kNpuArch3510 = 3510;

struct MatmulReluGraphParam {
    DataType dtype = DT_FLOAT16;
    std::vector<int64_t> x1Dims = {256, 128};
    std::vector<int64_t> x2Dims = {128, 256};
    std::vector<int64_t> yDims = {256, 256};
    bool isBatch = false;
    bool hasBias = false;
    std::vector<int64_t> biasDims = {256};
    bool enableHf32 = false;
    bool transposeX1 = false;
    bool transposeX2 = false;
};

void SetNodeOutputDesc(GNode& node, const std::vector<int64_t>& dims, DataType dtype)
{
    TensorDesc desc(Shape(dims), FORMAT_ND, dtype);
    desc.SetOriginShape(Shape(dims));
    desc.SetOriginFormat(FORMAT_ND);
    node.UpdateOutputDesc(0, desc);
}

void SetNodeInputDesc(GNode& node, int64_t inputIdx, const std::vector<int64_t>& dims, DataType dtype)
{
    TensorDesc desc;
    node.GetInputDesc(inputIdx, desc);
    desc.SetShape(Shape(dims));
    desc.SetOriginShape(Shape(dims));
    desc.SetDataType(dtype);
    desc.SetFormat(FORMAT_ND);
    desc.SetOriginFormat(FORMAT_ND);
    node.UpdateInputDesc(inputIdx, desc);
}

// Create a Relu node with CompliantNodeBuilder because Relu has no generated ES API in this repo.
es::EsTensorHolder MakeReluNode(es::EsGraphBuilder& graphBuilder, const es::EsTensorHolder& x)
{
    auto* graph = graphBuilder.GetCGraphBuilder()->GetGraph();
    auto reluNode = es::CompliantNodeBuilder(graph)
                        .OpType(kOpTypeRelu)
                        .Name(kOpTypeRelu)
                        .IrDefInputs({{"x", es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
                        .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                        .Build();
    if (AddEdgeAndUpdatePeerDesc(*graph, *x.GetProducer(), x.GetProducerOutIndex(), reluNode, 0) != GRAPH_SUCCESS) {
        return nullptr;
    }
    auto* yHolder = graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(reluNode, 0);
    return es::EsTensorHolder(yHolder);
}

// Create an AscendQuant node with CompliantNodeBuilder.
es::EsTensorHolder MakeAscendQuantNode(es::EsGraphBuilder& graphBuilder, const es::EsTensorHolder& x)
{
    auto* graph = graphBuilder.GetCGraphBuilder()->GetGraph();
    auto quantNode = es::CompliantNodeBuilder(graph)
                         .OpType(kOpTypeAscendQuant)
                         .Name(kOpTypeAscendQuant)
                         .IrDefInputs({{"x", es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
                         .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                         .IrDefAttrs({
                             {"scale", es::CompliantNodeBuilder::kEsAttrRequired, "Float", CreateFrom(1.0F)},
                             {"offset", es::CompliantNodeBuilder::kEsAttrRequired, "Float", CreateFrom(0.0F)},
                             {"sqrt_mode", es::CompliantNodeBuilder::kEsAttrOptional, "Bool", CreateFrom(false)},
                             {"round_mode", es::CompliantNodeBuilder::kEsAttrOptional, "String",
                              CreateFrom(AscendString("Round"))},
                             {"dst_type", es::CompliantNodeBuilder::kEsAttrOptional, "Int",
                              CreateFrom(static_cast<int64_t>(2))},
                         })
                         .Build();
    if (AddEdgeAndUpdatePeerDesc(*graph, *x.GetProducer(), x.GetProducerOutIndex(), quantNode, 0) != GRAPH_SUCCESS) {
        return nullptr;
    }
    auto* yHolder = graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(quantNode, 0);
    return es::EsTensorHolder(yHolder);
}

es::EsTensorHolder MakeMatmulNode(es::EsGraphBuilder& graphBuilder, const MatmulReluGraphParam& param,
                                  const es::EsTensorHolder& x1, const es::EsTensorHolder& x2,
                                  const es::EsTensorHolder& bias)
{
    if (param.isBatch) {
        return es::BatchMatMulV3(x1, x2, bias, nullptr, param.transposeX1, param.transposeX2, 0, param.enableHf32);
    }
    const int64_t opImplMode = param.enableHf32 ? kHf32OpImplMode : kDefaultOpImplMode;
    return es::MatMulV3(x1, x2, bias, nullptr, param.transposeX1, param.transposeX2, 0, opImplMode);
}

void SetMatmulNodeDescs(GNode& matmulNode, const MatmulReluGraphParam& param)
{
    SetNodeInputDesc(matmulNode, 0, param.x1Dims, param.dtype);
    SetNodeInputDesc(matmulNode, 1, param.x2Dims, param.dtype);
    if (param.hasBias) {
        SetNodeInputDesc(matmulNode, 2, param.biasDims, param.dtype);
    }
    SetNodeOutputDesc(matmulNode, param.yDims, param.dtype);
}

// Build graph: x1/x2(/bias) -> MatMulV3/BatchMatMulV3 -> Relu, all descs are set.
std::shared_ptr<Graph> BuildMatmulReluGraph(const MatmulReluGraphParam& param)
{
    auto graphBuilder = es::EsGraphBuilder("fused_mat_mul_fusion_test_graph");
    auto x1 = graphBuilder.CreateInput(0, "x1", param.dtype, FORMAT_ND, param.x1Dims);
    auto x2 = graphBuilder.CreateInput(1, "x2", param.dtype, FORMAT_ND, param.x2Dims);
    es::EsTensorHolder bias = nullptr;
    if (param.hasBias) {
        bias = graphBuilder.CreateInput(2, "bias", param.dtype, FORMAT_ND, param.biasDims);
    }

    auto matmulOut = MakeMatmulNode(graphBuilder, param, x1, x2, bias);
    auto reluOut = MakeReluNode(graphBuilder, matmulOut);

    SetNodeOutputDesc(*x1.GetProducer(), param.x1Dims, param.dtype);
    SetNodeOutputDesc(*x2.GetProducer(), param.x2Dims, param.dtype);
    if (param.hasBias) {
        SetNodeOutputDesc(*bias.GetProducer(), param.biasDims, param.dtype);
    }

    GNode matmulNode = *matmulOut.GetProducer();
    SetMatmulNodeDescs(matmulNode, param);

    GNode reluNode = *reluOut.GetProducer();
    SetNodeInputDesc(reluNode, 0, param.yDims, param.dtype);
    SetNodeOutputDesc(reluNode, param.yDims, param.dtype);

    return graphBuilder.BuildAndReset({reluOut});
}

// Build graph: x1/x2 -> MatMulV3 -> Relu -> AscendQuant, only relu part should be fused.
std::shared_ptr<Graph> BuildMatmulReluQuantGraph(const MatmulReluGraphParam& param)
{
    auto graphBuilder = es::EsGraphBuilder("fused_mat_mul_relu_quant_test_graph");
    auto x1 = graphBuilder.CreateInput(0, "x1", param.dtype, FORMAT_ND, param.x1Dims);
    auto x2 = graphBuilder.CreateInput(1, "x2", param.dtype, FORMAT_ND, param.x2Dims);

    auto matmulOut = MakeMatmulNode(graphBuilder, param, x1, x2, nullptr);
    auto reluOut = MakeReluNode(graphBuilder, matmulOut);
    auto quantOut = MakeAscendQuantNode(graphBuilder, reluOut);

    SetNodeOutputDesc(*x1.GetProducer(), param.x1Dims, param.dtype);
    SetNodeOutputDesc(*x2.GetProducer(), param.x2Dims, param.dtype);

    GNode matmulNode = *matmulOut.GetProducer();
    SetMatmulNodeDescs(matmulNode, param);

    GNode reluNode = *reluOut.GetProducer();
    SetNodeInputDesc(reluNode, 0, param.yDims, param.dtype);
    SetNodeOutputDesc(reluNode, param.yDims, param.dtype);

    GNode quantNode = *quantOut.GetProducer();
    SetNodeInputDesc(quantNode, 0, param.yDims, param.dtype);
    SetNodeOutputDesc(quantNode, param.yDims, DT_INT8);

    return graphBuilder.BuildAndReset({quantOut});
}

// Build graph: x1/x2 -> MatMulV3 -> AscendQuant, no pattern matches.
std::shared_ptr<Graph> BuildMatmulQuantGraph(const MatmulReluGraphParam& param)
{
    auto graphBuilder = es::EsGraphBuilder("fused_mat_mul_quant_test_graph");
    auto x1 = graphBuilder.CreateInput(0, "x1", param.dtype, FORMAT_ND, param.x1Dims);
    auto x2 = graphBuilder.CreateInput(1, "x2", param.dtype, FORMAT_ND, param.x2Dims);

    auto matmulOut = MakeMatmulNode(graphBuilder, param, x1, x2, nullptr);
    auto quantOut = MakeAscendQuantNode(graphBuilder, matmulOut);

    SetNodeOutputDesc(*x1.GetProducer(), param.x1Dims, param.dtype);
    SetNodeOutputDesc(*x2.GetProducer(), param.x2Dims, param.dtype);

    GNode matmulNode = *matmulOut.GetProducer();
    SetMatmulNodeDescs(matmulNode, param);

    GNode quantNode = *quantOut.GetProducer();
    SetNodeInputDesc(quantNode, 0, param.yDims, param.dtype);
    SetNodeOutputDesc(quantNode, param.yDims, DT_INT8);

    return graphBuilder.BuildAndReset({quantOut});
}

// Build graph: x1/x2 -> MatMulV3 -> {Relu1, Relu2}, matmul has two data outputs.
std::shared_ptr<Graph> BuildMatmulMultiOutputGraph(const MatmulReluGraphParam& param)
{
    auto graphBuilder = es::EsGraphBuilder("fused_mat_mul_multi_output_test_graph");
    auto x1 = graphBuilder.CreateInput(0, "x1", param.dtype, FORMAT_ND, param.x1Dims);
    auto x2 = graphBuilder.CreateInput(1, "x2", param.dtype, FORMAT_ND, param.x2Dims);

    auto matmulOut = MakeMatmulNode(graphBuilder, param, x1, x2, nullptr);
    auto reluOut1 = MakeReluNode(graphBuilder, matmulOut);
    auto reluOut2 = MakeReluNode(graphBuilder, matmulOut);

    SetNodeOutputDesc(*x1.GetProducer(), param.x1Dims, param.dtype);
    SetNodeOutputDesc(*x2.GetProducer(), param.x2Dims, param.dtype);

    GNode matmulNode = *matmulOut.GetProducer();
    SetMatmulNodeDescs(matmulNode, param);

    GNode reluNode1 = *reluOut1.GetProducer();
    SetNodeInputDesc(reluNode1, 0, param.yDims, param.dtype);
    SetNodeOutputDesc(reluNode1, param.yDims, param.dtype);
    GNode reluNode2 = *reluOut2.GetProducer();
    SetNodeInputDesc(reluNode2, 0, param.yDims, param.dtype);
    SetNodeOutputDesc(reluNode2, param.yDims, param.dtype);

    return graphBuilder.BuildAndReset({reluOut1, reluOut2});
}

Status RunFusedMatMulFusionPass(std::shared_ptr<Graph>& graph)
{
    CustomPassContext passContext;
    ops::FusedMatMulFusionPass pass;
    return pass.Run(graph, passContext);
}

int CountNodesByType(const std::shared_ptr<Graph>& graph, const char* nodeType)
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

GNodePtr FindNodeByType(const std::shared_ptr<Graph>& graph, const char* nodeType)
{
    for (auto node : graph->GetAllNodes()) {
        AscendString type;
        node.GetType(type);
        if (type == nodeType) {
            return std::make_shared<GNode>(node);
        }
    }
    return nullptr;
}

std::string GetFusedOpTypeAttr(const GNodePtr& node)
{
    AscendString fusedOpType;
    if (node->GetAttr("fused_op_type", fusedOpType) != GRAPH_SUCCESS) {
        return "";
    }
    return fusedOpType.GetString();
}

void SetDav3510Platform()
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.soc_info.arch_type = static_cast<int32_t>(kNpuArch3510);
    platformInfo.str_info.short_soc_version = "Ascend950";
    optiCompilationInfo.soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);
}

void SetNonDav3510Platform()
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.soc_info.arch_type = 2201;
    platformInfo.str_info.short_soc_version = "Ascend910_93";
    optiCompilationInfo.soc_version = "Ascend910_93";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend910_93"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);
}

} // namespace

class FusedMatMulFusionPassTest : public testing::Test {
protected:
    static void SetUpTestCase() { SetDav3510Platform(); }

    void SetUp() override { SetDav3510Platform(); }
};

// DAV_3510 fp16 matmul + relu fuses to FusedMatMul directly.
TEST_F(FusedMatMulFusionPassTest, matmulFp16ReluToFmmFusionSuccess)
{
    MatmulReluGraphParam param;
    param.dtype = DT_FLOAT16;
    // Non-default transpose attrs verify the attr transfer from matmul to FusedMatMul.
    // Note: the ES builder skips setting an attr equal to the IR default, so a failed
    // GetAttr after fusion means the default value (false) applies.
    param.transposeX1 = true;
    std::shared_ptr<Graph> graph = BuildMatmulReluGraph(param);

    Status status = RunFusedMatMulFusionPass(graph);
    EXPECT_EQ(status, SUCCESS);
    EXPECT_EQ(CountNodesByType(graph, kOpTypeFusedMatMul), 1);
    EXPECT_EQ(CountNodesByType(graph, kOpTypeMatMulV3), 0);
    EXPECT_EQ(CountNodesByType(graph, kOpTypeRelu), 0);

    GNodePtr fusedMatMul = FindNodeByType(graph, kOpTypeFusedMatMul);
    ASSERT_NE(fusedMatMul, nullptr);
    EXPECT_EQ(GetFusedOpTypeAttr(fusedMatMul), kFusedOpTypeRelu);
    bool transX1 = false;
    bool transX2 = false;
    if (fusedMatMul->GetAttr("transpose_x1", transX1) != GRAPH_SUCCESS) {
        transX1 = false; // attr absent means the IR default value false
    }
    if (fusedMatMul->GetAttr("transpose_x2", transX2) != GRAPH_SUCCESS) {
        transX2 = false;
    }
    EXPECT_TRUE(transX1);
    EXPECT_FALSE(transX2);

    TensorDesc outputDesc;
    EXPECT_EQ(fusedMatMul->GetOutputDesc(0, outputDesc), GRAPH_SUCCESS);
    EXPECT_EQ(outputDesc.GetDataType(), DT_FLOAT16);
    EXPECT_EQ(outputDesc.GetShape().GetDims(), param.yDims);
}

// DAV_3510 bf16 matmul + relu fuses to FusedMatMul directly.
TEST_F(FusedMatMulFusionPassTest, matmulBf16ReluToFmmFusionSuccess)
{
    MatmulReluGraphParam param;
    param.dtype = DT_BF16;
    std::shared_ptr<Graph> graph = BuildMatmulReluGraph(param);

    Status status = RunFusedMatMulFusionPass(graph);
    EXPECT_EQ(status, SUCCESS);
    EXPECT_EQ(CountNodesByType(graph, kOpTypeFusedMatMul), 1);
    EXPECT_EQ(CountNodesByType(graph, kOpTypeMatMulV3), 0);
    EXPECT_EQ(CountNodesByType(graph, kOpTypeRelu), 0);
}

// DAV_3510 fp32 matmul with HF32 mode (opImplMode & 0x40) fuses to FusedMatMul.
TEST_F(FusedMatMulFusionPassTest, matmulFp32Hf32ReluToFmmFusionSuccess)
{
    MatmulReluGraphParam param;
    param.dtype = DT_FLOAT;
    param.enableHf32 = true;
    std::shared_ptr<Graph> graph = BuildMatmulReluGraph(param);

    Status status = RunFusedMatMulFusionPass(graph);
    EXPECT_EQ(status, SUCCESS);
    GNodePtr fusedMatMul = FindNodeByType(graph, kOpTypeFusedMatMul);
    ASSERT_NE(fusedMatMul, nullptr);
    bool enableHf32 = false;
    EXPECT_EQ(fusedMatMul->GetAttr("enable_hf32", enableHf32), GRAPH_SUCCESS);
    EXPECT_TRUE(enableHf32);
}

// DAV_3510 fp32 batchmatmul with enable_hf32 fuses to FusedMatMul.
TEST_F(FusedMatMulFusionPassTest, batchMatmulFp32Hf32ReluToFmmFusionSuccess)
{
    MatmulReluGraphParam param;
    param.dtype = DT_FLOAT;
    param.isBatch = true;
    param.enableHf32 = true;
    std::shared_ptr<Graph> graph = BuildMatmulReluGraph(param);

    Status status = RunFusedMatMulFusionPass(graph);
    EXPECT_EQ(status, SUCCESS);
    EXPECT_EQ(CountNodesByType(graph, kOpTypeFusedMatMul), 1);
    EXPECT_EQ(CountNodesByType(graph, kOpTypeBatchMatMulV3), 0);
    GNodePtr fusedMatMul = FindNodeByType(graph, kOpTypeFusedMatMul);
    ASSERT_NE(fusedMatMul, nullptr);
    bool enableHf32 = false;
    EXPECT_EQ(fusedMatMul->GetAttr("enable_hf32", enableHf32), GRAPH_SUCCESS);
    EXPECT_TRUE(enableHf32);
}

// fp32 without HF32 mode does not fuse.
TEST_F(FusedMatMulFusionPassTest, matmulFp32WithoutHf32NotFused)
{
    MatmulReluGraphParam param;
    param.dtype = DT_FLOAT;
    param.enableHf32 = false;
    std::shared_ptr<Graph> graph = BuildMatmulReluGraph(param);

    Status status = RunFusedMatMulFusionPass(graph);
    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodesByType(graph, kOpTypeFusedMatMul), 0);
}

// matmul + quant does not fuse: DAV_3510 only supports the Relu fusion mode.
TEST_F(FusedMatMulFusionPassTest, matmulQuantNotFused)
{
    MatmulReluGraphParam param;
    param.dtype = DT_FLOAT16;
    std::shared_ptr<Graph> graph = BuildMatmulQuantGraph(param);

    Status status = RunFusedMatMulFusionPass(graph);
    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodesByType(graph, kOpTypeFusedMatMul), 0);
    EXPECT_EQ(CountNodesByType(graph, kOpTypeAscendQuant), 1);
}

// matmul + relu + quant fuses relu only, quant node is kept.
TEST_F(FusedMatMulFusionPassTest, matmulReluQuantFusesReluOnly)
{
    MatmulReluGraphParam param;
    param.dtype = DT_FLOAT16;
    std::shared_ptr<Graph> graph = BuildMatmulReluQuantGraph(param);

    Status status = RunFusedMatMulFusionPass(graph);
    EXPECT_EQ(status, SUCCESS);
    EXPECT_EQ(CountNodesByType(graph, kOpTypeFusedMatMul), 1);
    EXPECT_EQ(CountNodesByType(graph, kOpTypeAscendQuant), 1);
    EXPECT_EQ(CountNodesByType(graph, kOpTypeMatMulV3), 0);
    EXPECT_EQ(CountNodesByType(graph, kOpTypeRelu), 0);

    GNodePtr fusedMatMul = FindNodeByType(graph, kOpTypeFusedMatMul);
    ASSERT_NE(fusedMatMul, nullptr);
    EXPECT_EQ(GetFusedOpTypeAttr(fusedMatMul), kFusedOpTypeRelu);
}

// matmul with multiple outputs does not fuse.
TEST_F(FusedMatMulFusionPassTest, matmulMultipleOutputsNotFused)
{
    MatmulReluGraphParam param;
    param.dtype = DT_FLOAT16;
    std::shared_ptr<Graph> graph = BuildMatmulMultiOutputGraph(param);

    Status status = RunFusedMatMulFusionPass(graph);
    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodesByType(graph, kOpTypeFusedMatMul), 0);
    EXPECT_EQ(CountNodesByType(graph, kOpTypeMatMulV3), 1);
    EXPECT_EQ(CountNodesByType(graph, kOpTypeRelu), 2);
}

// matmul with 1D bias [N] fuses to FusedMatMul.
TEST_F(FusedMatMulFusionPassTest, matmulWithBiasReluFusionSuccess)
{
    MatmulReluGraphParam param;
    param.dtype = DT_FLOAT16;
    param.hasBias = true;
    param.biasDims = {256};
    std::shared_ptr<Graph> graph = BuildMatmulReluGraph(param);

    Status status = RunFusedMatMulFusionPass(graph);
    EXPECT_EQ(status, SUCCESS);
    GNodePtr fusedMatMul = FindNodeByType(graph, kOpTypeFusedMatMul);
    ASSERT_NE(fusedMatMul, nullptr);
    TensorDesc biasDesc;
    EXPECT_EQ(fusedMatMul->GetInputDesc(2, biasDesc), GRAPH_SUCCESS);
    EXPECT_EQ(biasDesc.GetShape().GetDims(), param.biasDims);
    EXPECT_EQ(biasDesc.GetDataType(), DT_FLOAT16);
}

// matmul with 2D bias [1, N] fuses to FusedMatMul.
TEST_F(FusedMatMulFusionPassTest, matmulWithBiasOneNReluFusionSuccess)
{
    MatmulReluGraphParam param;
    param.dtype = DT_FLOAT16;
    param.hasBias = true;
    param.biasDims = {1, 256};
    std::shared_ptr<Graph> graph = BuildMatmulReluGraph(param);

    Status status = RunFusedMatMulFusionPass(graph);
    EXPECT_EQ(status, SUCCESS);
    EXPECT_EQ(CountNodesByType(graph, kOpTypeFusedMatMul), 1);
}

// matmul with bias [2, N] does not fuse: DAV_3510 bias only supports [N], [1, N] or empty.
TEST_F(FusedMatMulFusionPassTest, matmulWithBiasTwoNNotFused)
{
    MatmulReluGraphParam param;
    param.dtype = DT_FLOAT16;
    param.hasBias = true;
    param.biasDims = {2, 256};
    std::shared_ptr<Graph> graph = BuildMatmulReluGraph(param);

    Status status = RunFusedMatMulFusionPass(graph);
    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodesByType(graph, kOpTypeFusedMatMul), 0);
}

// matmul with a 0-dim (scalar) bias STILL FUSES: aligned with the canndev
// CheckMatmulBiasShape, where the 0-dim bias passes the shape gate and the last-dim match is
// short-circuited by biasN == 0, so the fusion continues with the scalar bias kept.
TEST_F(FusedMatMulFusionPassTest, matmulWithBiasZeroDimStillFuses)
{
    MatmulReluGraphParam param;
    param.dtype = DT_FLOAT16;
    param.hasBias = true;
    param.biasDims = {};
    std::shared_ptr<Graph> graph = BuildMatmulReluGraph(param);

    Status status = RunFusedMatMulFusionPass(graph);
    EXPECT_EQ(status, SUCCESS);
    EXPECT_EQ(CountNodesByType(graph, kOpTypeFusedMatMul), 1);
    GNodePtr fusedMatMul = FindNodeByType(graph, kOpTypeFusedMatMul);
    ASSERT_NE(fusedMatMul, nullptr);
    TensorDesc biasDesc;
    EXPECT_EQ(fusedMatMul->GetInputDesc(2, biasDesc), GRAPH_SUCCESS);
    EXPECT_EQ(biasDesc.GetShape().GetDims(), std::vector<int64_t>{});
}

// matmul with bias dim mismatch does not fuse.
TEST_F(FusedMatMulFusionPassTest, matmulWithBiasDimMismatchNotFused)
{
    MatmulReluGraphParam param;
    param.dtype = DT_FLOAT16;
    param.hasBias = true;
    param.biasDims = {128};
    std::shared_ptr<Graph> graph = BuildMatmulReluGraph(param);

    Status status = RunFusedMatMulFusionPass(graph);
    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodesByType(graph, kOpTypeFusedMatMul), 0);
}

// unsupported platform (neither DAV_3510 nor s8s4 mmad) does not fuse. The pass dispatches by
// the short_soc_version against the 3510-series soc list, so the non-3510 platform is fully
// simulatable in the UT.
TEST_F(FusedMatMulFusionPassTest, unsupportedPlatformNotFused)
{
    SetNonDav3510Platform();
    MatmulReluGraphParam param;
    param.dtype = DT_FLOAT16;
    std::shared_ptr<Graph> graph = BuildMatmulReluGraph(param);

    Status status = RunFusedMatMulFusionPass(graph);
    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodesByType(graph, kOpTypeFusedMatMul), 0);
}

// unsupported dtype does not fuse.
TEST_F(FusedMatMulFusionPassTest, unsupportedDtypeNotFused)
{
    MatmulReluGraphParam param;
    param.dtype = DT_INT8;
    std::shared_ptr<Graph> graph = BuildMatmulReluGraph(param);

    Status status = RunFusedMatMulFusionPass(graph);
    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodesByType(graph, kOpTypeFusedMatMul), 0);
}

// BatchMatMulV3 3D x1 with 2D x2 (total batch of x2 is 1) fuses via MergeBatchAndMAxis.
TEST_F(FusedMatMulFusionPassTest, batchMatmul3d2dReluFusionSuccess)
{
    MatmulReluGraphParam param;
    param.dtype = DT_FLOAT;
    param.enableHf32 = true;
    param.isBatch = true;
    param.x1Dims = {50, 1, 448};
    param.x2Dims = {448, 896};
    param.yDims = {50, 1, 896};
    std::shared_ptr<Graph> graph = BuildMatmulReluGraph(param);

    Status status = RunFusedMatMulFusionPass(graph);
    EXPECT_EQ(status, SUCCESS);
    EXPECT_EQ(CountNodesByType(graph, kOpTypeFusedMatMul), 1);
}

// BatchMatMulV3 dynamic 3D x1 with 2D x2 fuses: dynamic dims are checked during runtime tiling.
TEST_F(FusedMatMulFusionPassTest, batchMatmulDynamic3d2dReluFusionSuccess)
{
    MatmulReluGraphParam param;
    param.dtype = DT_FLOAT;
    param.enableHf32 = true;
    param.isBatch = true;
    param.x1Dims = {-1, 1, 448};
    param.x2Dims = {448, 896};
    param.yDims = {-1, 1, 896};
    std::shared_ptr<Graph> graph = BuildMatmulReluGraph(param);

    Status status = RunFusedMatMulFusionPass(graph);
    EXPECT_EQ(status, SUCCESS);
    EXPECT_EQ(CountNodesByType(graph, kOpTypeFusedMatMul), 1);
}

// BatchMatMulV3 with the same batch dims (6D) fuses.
TEST_F(FusedMatMulFusionPassTest, batchMatmul6dSameBatchReluFusionSuccess)
{
    MatmulReluGraphParam param;
    param.dtype = DT_FLOAT16;
    param.isBatch = true;
    param.x1Dims = {2, 3, 4, 5, 8, 16};
    param.x2Dims = {2, 3, 4, 5, 16, 32};
    param.yDims = {2, 3, 4, 5, 8, 32};
    std::shared_ptr<Graph> graph = BuildMatmulReluGraph(param);

    Status status = RunFusedMatMulFusionPass(graph);
    EXPECT_EQ(status, SUCCESS);
    EXPECT_EQ(CountNodesByType(graph, kOpTypeFusedMatMul), 1);
}

// BatchMatMulV3 with broadcast batch dims (x2 batch is neither 1 nor equal to x1) does not fuse.
// Note: when the total batch of x2 is 1 (e.g. x2 = {1, 8, 16}), the fusion is allowed via
// MergeBatchAndMAxis, so a real broadcast case here requires a non-1 and unequal x2 batch dim.
TEST_F(FusedMatMulFusionPassTest, batchMatmulBroadcastReluNotFused)
{
    MatmulReluGraphParam param;
    param.dtype = DT_FLOAT16;
    param.isBatch = true;
    param.x1Dims = {2, 4, 8};
    param.x2Dims = {3, 8, 16};
    param.yDims = {2, 4, 16};
    std::shared_ptr<Graph> graph = BuildMatmulReluGraph(param);

    Status status = RunFusedMatMulFusionPass(graph);
    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodesByType(graph, kOpTypeFusedMatMul), 0);
}

// BatchMatMulV3 with 3D x1 and 3D x2 whose total batch is 1 fuses via MergeBatchAndMAxis.
TEST_F(FusedMatMulFusionPassTest, batchMatmul3dX2BatchOneReluFusionSuccess)
{
    MatmulReluGraphParam param;
    param.dtype = DT_FLOAT16;
    param.isBatch = true;
    param.x1Dims = {2, 4, 8};
    param.x2Dims = {1, 8, 16};
    param.yDims = {2, 4, 16};
    std::shared_ptr<Graph> graph = BuildMatmulReluGraph(param);

    Status status = RunFusedMatMulFusionPass(graph);
    EXPECT_EQ(status, SUCCESS);
    EXPECT_EQ(CountNodesByType(graph, kOpTypeFusedMatMul), 1);
}

// BatchMatMulV3 with different dim nums (2D x1, 3D x2) does not fuse.
TEST_F(FusedMatMulFusionPassTest, batchMatmul2d3dReluNotFused)
{
    MatmulReluGraphParam param;
    param.dtype = DT_FLOAT16;
    param.isBatch = true;
    param.x1Dims = {4, 8};
    param.x2Dims = {2, 8, 16};
    param.yDims = {2, 4, 16};
    std::shared_ptr<Graph> graph = BuildMatmulReluGraph(param);

    Status status = RunFusedMatMulFusionPass(graph);
    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodesByType(graph, kOpTypeFusedMatMul), 0);
}

// BatchMatMulV3 with transposed x1 and broadcast batch dims does not fuse.
TEST_F(FusedMatMulFusionPassTest, batchMatmulTransposedX13d2dReluNotFused)
{
    MatmulReluGraphParam param;
    param.dtype = DT_FLOAT16;
    param.isBatch = true;
    param.transposeX1 = true;
    param.x1Dims = {50, 448, 1};
    param.x2Dims = {448, 896};
    param.yDims = {50, 1, 896};
    std::shared_ptr<Graph> graph = BuildMatmulReluGraph(param);

    Status status = RunFusedMatMulFusionPass(graph);
    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodesByType(graph, kOpTypeFusedMatMul), 0);
}

// ============================================================================
// s8s4 mmad PostCube path tests (single pass, platform dispatch inside)
// ============================================================================
namespace pc_test {

constexpr char kOpTypeLeakyRelu[] = "LeakyRelu";
constexpr char kOpTypeFixPipe[] = "FixPipe";
constexpr char kOpTypeSetQuantOffsetScale[] = "SetQuantOffsetScale";
constexpr char kFusedOpTypeQuant[] = "quant";
constexpr char kFusedOpTypeReluQuant[] = "relu_quant";

// The post op chain appended after the matmul node. RELU_QUANT keeps the original Relu->Quant
// order, the production form (the canndev FixPipeReluQuantPass swap never applies to a matmul
// head, its white list is Conv2D/Conv2DTranspose only); RELU_QUANT_SWAPPED builds the
// Quant->Relu order for a producer that already emits it.
enum class PostOp { NONE, RELU, QUANT, RELU_QUANT, RELU_QUANT_SWAPPED, LEAKY_RELU };

struct MatmulPostCubeParam {
    DataType dtype = DT_FLOAT16;
    DataType quantOutDtype = DT_INT8; // the output dtype of AscendQuant
    float quantScale = 1.0F;          // the AscendQuant scale attr
    float quantOffset = 0.0F;         // the AscendQuant offset attr
    std::vector<int64_t> x1Dims = {256, 128};
    std::vector<int64_t> x2Dims = {128, 256};
    std::vector<int64_t> yDims = {256, 256};
    bool isBatch = false;
    bool hasBias = false;
    std::vector<int64_t> biasDims = {256};
    bool transposeX1 = false;
    PostOp postOp = PostOp::RELU;
    bool withUncache = false;
    // Aligned with the production semantics: the forbidden shapes (ND input / ND-class output
    // NCHW/NHWC/HWCN/NCDHW/ND) are gated by the ability flag set by the canndev
    // FixPipeAbilityProcessPass, so the normal scenario keeps x1/x2 in the pass format list
    // (e.g. NCHW) and the output in the fractal NZ.
    Format x1x2Format = FORMAT_NCHW;
    Format yFormat = FORMAT_FRACTAL_NZ;
};

void SetNodeOutputDescWithFormat(GNode& node, const std::vector<int64_t>& dims, DataType dtype, Format format)
{
    TensorDesc desc(Shape(dims), format, dtype);
    desc.SetOriginShape(Shape(dims));
    desc.SetOriginFormat(format);
    node.UpdateOutputDesc(0, desc);
}

void SetNodeInputDescWithFormat(GNode& node, int64_t inputIdx, const std::vector<int64_t>& dims, DataType dtype,
                                Format format)
{
    TensorDesc desc;
    node.GetInputDesc(inputIdx, desc);
    desc.SetShape(Shape(dims));
    desc.SetOriginShape(Shape(dims));
    desc.SetDataType(dtype);
    desc.SetFormat(format);
    desc.SetOriginFormat(format);
    node.UpdateInputDesc(inputIdx, desc);
}

es::EsTensorHolder MakeLeakyReluNode(es::EsGraphBuilder& graphBuilder, const es::EsTensorHolder& x)
{
    auto* graph = graphBuilder.GetCGraphBuilder()->GetGraph();
    auto leakyReluNode = es::CompliantNodeBuilder(graph)
                             .OpType(kOpTypeLeakyRelu)
                             .Name(kOpTypeLeakyRelu)
                             .IrDefInputs({{"x", es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
                             .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                             .IrDefAttrs({{"negative_slope", es::CompliantNodeBuilder::kEsAttrRequired, "Float",
                                           CreateFrom(0.1F)}})
                             .Build();
    if (AddEdgeAndUpdatePeerDesc(*graph, *x.GetProducer(), x.GetProducerOutIndex(), leakyReluNode, 0) !=
        GRAPH_SUCCESS) {
        return nullptr;
    }
    auto* yHolder = graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(leakyReluNode, 0);
    return es::EsTensorHolder(yHolder);
}

// AscendQuant: x(fp16) -> y(int8), scale/offset are attrs.
es::EsTensorHolder MakeAscendQuantNode(es::EsGraphBuilder& graphBuilder, const es::EsTensorHolder& x, float scale,
                                       float offset)
{
    auto* graph = graphBuilder.GetCGraphBuilder()->GetGraph();
    auto quantNode = es::CompliantNodeBuilder(graph)
                         .OpType(kOpTypeAscendQuant)
                         .Name(kOpTypeAscendQuant)
                         .IrDefInputs({{"x", es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
                         .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                         .IrDefAttrs({
                             {"scale", es::CompliantNodeBuilder::kEsAttrRequired, "Float", CreateFrom(scale)},
                             {"offset", es::CompliantNodeBuilder::kEsAttrRequired, "Float", CreateFrom(offset)},
                             {"sqrt_mode", es::CompliantNodeBuilder::kEsAttrOptional, "Bool", CreateFrom(false)},
                             {"round_mode", es::CompliantNodeBuilder::kEsAttrOptional, "String",
                              CreateFrom(AscendString("Round"))},
                             {"dst_type", es::CompliantNodeBuilder::kEsAttrOptional, "Int",
                              CreateFrom(static_cast<int64_t>(DT_INT8))},
                         })
                         .Build();
    if (AddEdgeAndUpdatePeerDesc(*graph, *x.GetProducer(), x.GetProducerOutIndex(), quantNode, 0) != GRAPH_SUCCESS) {
        return nullptr;
    }
    auto* yHolder = graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(quantNode, 0);
    return es::EsTensorHolder(yHolder);
}

// The float payload of a scalar Const node created by the utils' CreateScalarConstNode (the
// "value" Tensor attr).
float GetScalarConstValue(const GNode& constNode)
{
    Tensor weight;
    EXPECT_EQ(constNode.GetAttr("value", weight), GRAPH_SUCCESS);
    EXPECT_EQ(weight.GetSize(), sizeof(float));
    float value = 0.0F;
    (void)memcpy(&value, weight.GetData(), sizeof(float));
    return value;
}

// Verify the whole x3 scalar chain of a quant-fused FusedMatMul: the SetQuantOffsetScale
// producer on x3, its const inputs carrying the AscendQuant scale/offset, and the
// _quant_output_data_type/_input_strategy attrs the runtime unpacks the scalar with. Shared by
// the folded-quant path (the postcube utils' Strategy21) and the trailing-quant absorption: both
// must produce the byte-for-byte identical x3 payload.
void VerifyFusedMatMulX3(const GNodePtr& fusedMatMul, float expectedScale, float expectedOffset,
                         DataType expectedQuantDtype)
{
    auto x3Node = fusedMatMul->GetInDataNodesAndPortIndexs(3).first;
    ASSERT_NE(x3Node, nullptr);
    AscendString x3Type;
    ASSERT_EQ(x3Node->GetType(x3Type), GRAPH_SUCCESS);
    EXPECT_EQ(x3Type, kOpTypeSetQuantOffsetScale);

    // input 0 = the offset const, input 1 = the scale const (the CreateScalarInputNode wiring).
    auto offsetNode = x3Node->GetInDataNodesAndPortIndexs(0).first;
    auto scaleNode = x3Node->GetInDataNodesAndPortIndexs(1).first;
    ASSERT_NE(offsetNode, nullptr);
    ASSERT_NE(scaleNode, nullptr);
    EXPECT_FLOAT_EQ(GetScalarConstValue(*offsetNode), expectedOffset);
    EXPECT_FLOAT_EQ(GetScalarConstValue(*scaleNode), expectedScale);

    int64_t quantOutputDtype = -1;
    EXPECT_EQ(x3Node->GetAttr("_quant_output_data_type", quantOutputDtype), GRAPH_SUCCESS);
    EXPECT_EQ(quantOutputDtype, static_cast<int64_t>(expectedQuantDtype));
    int64_t inputStrategy = -1;
    EXPECT_EQ(x3Node->GetAttr("_input_strategy", inputStrategy), GRAPH_SUCCESS);
    EXPECT_EQ(inputStrategy, 21); // the canndev FMM_QUANT_INPUT_STRATEGY
}

std::shared_ptr<Graph> BuildMatmulPostOpGraph(const MatmulPostCubeParam& param)
{
    auto graphBuilder = es::EsGraphBuilder("fused_mat_mul_postcube_test_graph");
    auto x1 = graphBuilder.CreateInput(0, "x1", param.dtype, param.x1x2Format, param.x1Dims);
    auto x2 = graphBuilder.CreateInput(1, "x2", param.dtype, param.x1x2Format, param.x2Dims);
    es::EsTensorHolder bias = nullptr;
    if (param.hasBias) {
        bias = graphBuilder.CreateInput(2, "bias", param.dtype, FORMAT_ND, param.biasDims);
    }

    auto matmulOut = param.isBatch ? es::BatchMatMulV3(x1, x2, bias, nullptr, param.transposeX1, false, 0, 1) :
                                     es::MatMulV3(x1, x2, bias, nullptr, param.transposeX1, false, 0, 1);

    SetNodeOutputDescWithFormat(*x1.GetProducer(), param.x1Dims, param.dtype, param.x1x2Format);
    SetNodeOutputDescWithFormat(*x2.GetProducer(), param.x2Dims, param.dtype, param.x1x2Format);
    if (param.hasBias) {
        SetNodeOutputDesc(*bias.GetProducer(), param.biasDims, param.dtype);
    }

    GNode matmulNode = *matmulOut.GetProducer();
    SetNodeInputDescWithFormat(matmulNode, 0, param.x1Dims, param.dtype, param.x1x2Format);
    SetNodeInputDescWithFormat(matmulNode, 1, param.x2Dims, param.dtype, param.x1x2Format);
    if (param.hasBias) {
        SetNodeInputDesc(matmulNode, 2, param.biasDims, param.dtype);
    }
    SetNodeOutputDescWithFormat(matmulNode, param.yDims, param.dtype, param.yFormat);
    if (param.withUncache) {
        int64_t enableUncache = 1;
        matmulNode.SetAttr("enable_uncache", enableUncache);
    }

    // Build the post op chain.
    es::EsTensorHolder tail = matmulOut;
    DataType tailDtype = param.dtype;
    std::vector<int64_t> tailDims = param.yDims;
    if (param.postOp == PostOp::RELU || param.postOp == PostOp::RELU_QUANT) {
        tail = MakeReluNode(graphBuilder, tail);
        SetNodeInputDesc(*tail.GetProducer(), 0, tailDims, tailDtype);
        SetNodeOutputDesc(*tail.GetProducer(), tailDims, tailDtype);
    }
    if (param.postOp == PostOp::QUANT || param.postOp == PostOp::RELU_QUANT) {
        tail = MakeAscendQuantNode(graphBuilder, tail, param.quantScale, param.quantOffset);
        SetNodeInputDesc(*tail.GetProducer(), 0, tailDims, tailDtype);
        tailDtype = param.quantOutDtype;
        SetNodeOutputDesc(*tail.GetProducer(), tailDims, tailDtype);
    }
    if (param.postOp == PostOp::LEAKY_RELU) {
        tail = MakeLeakyReluNode(graphBuilder, tail);
        SetNodeInputDesc(*tail.GetProducer(), 0, tailDims, tailDtype);
        SetNodeOutputDesc(*tail.GetProducer(), tailDims, tailDtype);
    }
    if (param.postOp == PostOp::RELU_QUANT_SWAPPED) {
        // The quant-first order produced by the canndev FixPipeReluQuantPass: AscendQuant first,
        // then the Relu re-created on the int8 output.
        tail = MakeAscendQuantNode(graphBuilder, tail, param.quantScale, param.quantOffset);
        SetNodeInputDesc(*tail.GetProducer(), 0, tailDims, tailDtype);
        tailDtype = param.quantOutDtype;
        SetNodeOutputDesc(*tail.GetProducer(), tailDims, tailDtype);
        tail = MakeReluNode(graphBuilder, tail);
        SetNodeInputDesc(*tail.GetProducer(), 0, tailDims, tailDtype);
        SetNodeOutputDesc(*tail.GetProducer(), tailDims, tailDtype);
    }

    if (param.postOp == PostOp::NONE) {
        return graphBuilder.BuildAndReset({matmulOut});
    }
    return graphBuilder.BuildAndReset({tail});
}

Status RunPostCubePass(std::shared_ptr<Graph>& graph)
{
    CustomPassContext passContext;
    ops::FusedMatMulFusionPass pass;
    return pass.Run(graph, passContext);
}

// Install the ge option overrides (precision_mode / customize_dtypes ...) for one test: the
// map is process state shared by every test, cleared on scope exit.
class ScopedGeOptions {
public:
    explicit ScopedGeOptions(const std::map<std::string, std::string>& options)
    {
        ge::ut_detail::GetOptionsMap() = options;
    }
    ~ScopedGeOptions() { ge::ut_detail::GetOptionsMap().clear(); }
};

// Prepare the s8s4 mmad platform for the postcube pass: use the real Ascend910B1 platform
// (the target s8s4 platform). Its config (installed under data/platform_config) natively
// contains the Intrinsic_mmad s8s4 support and the fixpipe unit/func list, and its soc name
// (Ascend910B) is not in the 3510-series soc list, so no manual injection is needed at all.
void SetS8s4Platform()
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 24;
    platformInfo.str_info.short_soc_version = "Ascend910B";
    optiCompilationInfo.soc_version = "Ascend910B1";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend910B1"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    // Load the real platform config: fixpipe unit/func list, AIC version.
    fe::PlatformInfoManager::Instance().InitializePlatformInfo();
    fe::PlatformInfoManager::Instance().opti_compilation_infos_.SetSocVersion("Ascend910B1");

    fe::PlatformInfo& info = fe::PlatformInfoManager::Instance().platform_info_map_["Ascend910B1"];
    info.str_info.short_soc_version = "Ascend910B";
    // The s8s4 mmad capability is injected manually (aligned with the canndev UT SetSocMC62):
    // the real ini describes it as "s32s8s8" while the pass checks the "s8s4" literal.
    info.ai_core_intrinsic_dtype_map["Intrinsic_mmad"] = {"s8s4"};
}

void SetNonS8s4Platform()
{
    SetS8s4Platform();
    fe::PlatformInfo& info = fe::PlatformInfoManager::Instance().platform_info_map_["Ascend910B1"];
    info.ai_core_intrinsic_dtype_map["Intrinsic_mmad"] = {"f16f16"};
}

// DAV_3510 platform: the direct fusion pass owns the matmul fusion, the postcube pass must
// yield. The dispatch matches the short_soc_version against the 3510-series soc list (the
// common IsNpuArch3510Series), so the 3510 platform is simulated by the Ascend950 soc name.
void SetDav3510ArchPlatform()
{
    SetS8s4Platform();
    fe::PlatformInfo& info = fe::PlatformInfoManager::Instance().platform_info_map_["Ascend910B1"];
    info.str_info.short_soc_version = "Ascend950";
}

void SetPcS8s4Platform() { SetS8s4Platform(); }

} // namespace pc_test

using namespace pc_test;

class FusedMatMulPostCubePathTest : public testing::Test {
protected:
    static void SetUpTestCase() { SetS8s4Platform(); }

    void SetUp() override { SetS8s4Platform(); }
};

// matmul + relu fuses to FusedMatMul via the PostCube chain on the s8s4 platform.
TEST_F(FusedMatMulPostCubePathTest, postcubeReluToFmmFusionSuccess)
{
    MatmulPostCubeParam param;
    param.postOp = PostOp::RELU;
    std::shared_ptr<Graph> graph = BuildMatmulPostOpGraph(param);

    Status status = RunPostCubePass(graph);
    EXPECT_EQ(status, SUCCESS);
    EXPECT_EQ(CountNodesByType(graph, kOpTypeFusedMatMul), 1);
    EXPECT_EQ(CountNodesByType(graph, kOpTypeMatMulV3), 0);
    EXPECT_EQ(CountNodesByType(graph, kOpTypeRelu), 0);
    EXPECT_EQ(CountNodesByType(graph, kOpTypeFixPipe), 0);

    GNodePtr fusedMatMul = FindNodeByType(graph, kOpTypeFusedMatMul);
    ASSERT_NE(fusedMatMul, nullptr);
    EXPECT_EQ(GetFusedOpTypeAttr(fusedMatMul), kFusedOpTypeRelu);
}

// matmul + ascendquant fuses to FusedMatMul, the quant scale/offset is folded to the x3 input.
TEST_F(FusedMatMulPostCubePathTest, postcubeQuantToFmmFusionSuccess)
{
    MatmulPostCubeParam param;
    param.postOp = PostOp::QUANT;
    std::shared_ptr<Graph> graph = BuildMatmulPostOpGraph(param);

    Status status = RunPostCubePass(graph);
    EXPECT_EQ(status, SUCCESS);
    EXPECT_EQ(CountNodesByType(graph, kOpTypeFusedMatMul), 1);
    EXPECT_EQ(CountNodesByType(graph, kOpTypeAscendQuant), 0);
    EXPECT_EQ(CountNodesByType(graph, kOpTypeFixPipe), 0);

    GNodePtr fusedMatMul = FindNodeByType(graph, kOpTypeFusedMatMul);
    ASSERT_NE(fusedMatMul, nullptr);
    EXPECT_EQ(GetFusedOpTypeAttr(fusedMatMul), kFusedOpTypeQuant);
    // The x3 input must be connected (the SetQuantOffsetScale scalar generated by the postcube utils).
    VerifyFusedMatMulX3(fusedMatMul, param.quantScale, param.quantOffset, param.quantOutDtype);
    EXPECT_EQ(CountNodesByType(graph, kOpTypeSetQuantOffsetScale), 1);
}

// matmul + quant + relu (the order the canndev FixPipeReluQuantPass produces) fuses to
// FusedMatMul with fused_op_type = relu_quant: the pass consumes the already-swapped graph,
// the same way as the canndev FusedMatMulFusionPass relies on the canndev swap pass. The
// distinct scale/offset values make the folded x3 payload verified directionally.
TEST_F(FusedMatMulPostCubePathTest, postcubeReluQuantToFmmFusionSuccess)
{
    MatmulPostCubeParam param;
    param.quantScale = 2.5F;
    param.quantOffset = -1.5F;
    param.postOp = PostOp::RELU_QUANT_SWAPPED;
    std::shared_ptr<Graph> graph = BuildMatmulPostOpGraph(param);

    Status status = RunPostCubePass(graph);
    EXPECT_EQ(status, SUCCESS);
    EXPECT_EQ(CountNodesByType(graph, kOpTypeFusedMatMul), 1);
    EXPECT_EQ(CountNodesByType(graph, kOpTypeMatMulV3), 0);
    EXPECT_EQ(CountNodesByType(graph, kOpTypeRelu), 0);
    EXPECT_EQ(CountNodesByType(graph, kOpTypeAscendQuant), 0);
    EXPECT_EQ(CountNodesByType(graph, kOpTypeFixPipe), 0);

    GNodePtr fusedMatMul = FindNodeByType(graph, kOpTypeFusedMatMul);
    ASSERT_NE(fusedMatMul, nullptr);
    EXPECT_EQ(GetFusedOpTypeAttr(fusedMatMul), kFusedOpTypeReluQuant);
    VerifyFusedMatMulX3(fusedMatMul, param.quantScale, param.quantOffset, param.quantOutDtype);
}

// matmul + relu + ascendquant in the ORIGINAL graph order (the production form, the swap never
// applies to a matmul head) fuses to a single FusedMatMul(relu_quant): the postcube units require
// the quant before the activation, so the fold covers the Relu only and the trailing AscendQuant
// is absorbed into the replace (the output takes over the quant output and the x3 scalar is built
// from the quant scale/offset attrs), the same canndev TransferOpToFusedMatMul semantics.
TEST_F(FusedMatMulPostCubePathTest, postcubeReluQuantOriginalOrderFullFusion)
{
    MatmulPostCubeParam param;
    param.quantScale = 2.5F;
    param.quantOffset = -1.5F;
    param.postOp = PostOp::RELU_QUANT;
    std::shared_ptr<Graph> graph = BuildMatmulPostOpGraph(param);

    Status status = RunPostCubePass(graph);
    EXPECT_EQ(status, SUCCESS);
    EXPECT_EQ(CountNodesByType(graph, kOpTypeFusedMatMul), 1);
    EXPECT_EQ(CountNodesByType(graph, kOpTypeMatMulV3), 0);
    EXPECT_EQ(CountNodesByType(graph, kOpTypeRelu), 0);
    EXPECT_EQ(CountNodesByType(graph, kOpTypeAscendQuant), 0);
    EXPECT_EQ(CountNodesByType(graph, kOpTypeFixPipe), 0);

    GNodePtr fusedMatMul = FindNodeByType(graph, kOpTypeFusedMatMul);
    ASSERT_NE(fusedMatMul, nullptr);
    EXPECT_EQ(GetFusedOpTypeAttr(fusedMatMul), kFusedOpTypeReluQuant);

    // The absorbed x3 payload must be byte-for-byte identical to the folded one (the same
    // VerifyFusedMatMulX3 asserts both): const scale/offset + _quant_output_data_type +
    // _input_strategy=21.
    VerifyFusedMatMulX3(fusedMatMul, param.quantScale, param.quantOffset, param.quantOutDtype);

    // The fused output inherits the quant output desc (int8), not the FixPipe fp16 one.
    TensorDesc outputDesc;
    EXPECT_EQ(fusedMatMul->GetOutputDesc(0, outputDesc), GRAPH_SUCCESS);
    EXPECT_EQ(outputDesc.GetDataType(), DT_INT8);
}

// The quant absorption is gated by the canndev pattern requirement that the intermediate Relu is
// exclusively consumed by the quant: with an extra consumer of the Relu (the FixPipe output has
// two consumers after the fold) the chain keeps the partial relu-only fusion and both the quant
// and the extra consumer stay in the graph.
TEST_F(FusedMatMulPostCubePathTest, postcubeReluQuantWithExtraReluConsumerFusesReluOnly)
{
    MatmulPostCubeParam param;
    param.postOp = PostOp::RELU_QUANT;
    std::shared_ptr<Graph> graph = BuildMatmulPostOpGraph(param);

    // Attach a second consumer (another Relu) to the original Relu output.
    GNodePtr reluNode = FindNodeByType(graph, kOpTypeRelu);
    ASSERT_NE(reluNode, nullptr);
    auto extraRelu = es::CompliantNodeBuilder(graph.get())
                         .OpType(kOpTypeRelu)
                         .Name("ExtraRelu")
                         .IrDefInputs({{"x", es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
                         .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                         .Build();
    ASSERT_EQ(AddEdgeAndUpdatePeerDesc(*graph, *reluNode, 0, extraRelu, 0), GRAPH_SUCCESS);
    SetNodeInputDesc(extraRelu, 0, param.yDims, param.dtype);
    SetNodeOutputDesc(extraRelu, param.yDims, param.dtype);

    Status status = RunPostCubePass(graph);
    EXPECT_EQ(status, SUCCESS);
    EXPECT_EQ(CountNodesByType(graph, kOpTypeFusedMatMul), 1);
    EXPECT_EQ(CountNodesByType(graph, kOpTypeAscendQuant), 1);
    EXPECT_EQ(CountNodesByType(graph, kOpTypeRelu), 1); // the extra consumer stays
    GNodePtr fusedMatMul = FindNodeByType(graph, kOpTypeFusedMatMul);
    ASSERT_NE(fusedMatMul, nullptr);
    EXPECT_EQ(GetFusedOpTypeAttr(fusedMatMul), kFusedOpTypeRelu);
}

// batchmatmul + relu fuses to FusedMatMul.
TEST_F(FusedMatMulPostCubePathTest, postcubeBmmv3ReluFusionSuccess)
{
    MatmulPostCubeParam param;
    param.isBatch = true;
    param.postOp = PostOp::RELU;
    std::shared_ptr<Graph> graph = BuildMatmulPostOpGraph(param);

    Status status = RunPostCubePass(graph);
    EXPECT_EQ(status, SUCCESS);
    EXPECT_EQ(CountNodesByType(graph, kOpTypeFusedMatMul), 1);
    EXPECT_EQ(CountNodesByType(graph, kOpTypeBatchMatMulV3), 0);
}

// matmul + bias + relu fuses to FusedMatMul.
TEST_F(FusedMatMulPostCubePathTest, postcubeReluWithBiasFusionSuccess)
{
    MatmulPostCubeParam param;
    param.hasBias = true;
    param.biasDims = {256};
    param.postOp = PostOp::RELU;
    std::shared_ptr<Graph> graph = BuildMatmulPostOpGraph(param);

    Status status = RunPostCubePass(graph);
    EXPECT_EQ(status, SUCCESS);
    GNodePtr fusedMatMul = FindNodeByType(graph, kOpTypeFusedMatMul);
    ASSERT_NE(fusedMatMul, nullptr);
    TensorDesc biasDesc;
    EXPECT_EQ(fusedMatMul->GetInputDesc(2, biasDesc), GRAPH_SUCCESS);
    EXPECT_EQ(biasDesc.GetShape().GetDims(), param.biasDims);
}

// transpose attrs are transferred from the matmul node to FusedMatMul.
TEST_F(FusedMatMulPostCubePathTest, postcubeReluTransposeAttrTransfer)
{
    MatmulPostCubeParam param;
    param.transposeX1 = true;
    param.postOp = PostOp::RELU;
    std::shared_ptr<Graph> graph = BuildMatmulPostOpGraph(param);

    Status status = RunPostCubePass(graph);
    EXPECT_EQ(status, SUCCESS);
    GNodePtr fusedMatMul = FindNodeByType(graph, kOpTypeFusedMatMul);
    ASSERT_NE(fusedMatMul, nullptr);
    bool transX1 = false;
    if (fusedMatMul->GetAttr("transpose_x1", transX1) != GRAPH_SUCCESS) {
        transX1 = false; // attr absent means the IR default value false
    }
    EXPECT_TRUE(transX1);
}

// the quant output dtype (int8) is kept by FusedMatMul.
TEST_F(FusedMatMulPostCubePathTest, postcubeQuantOutputDtypeInt8)
{
    MatmulPostCubeParam param;
    param.postOp = PostOp::QUANT;
    param.quantOutDtype = DT_INT8;
    std::shared_ptr<Graph> graph = BuildMatmulPostOpGraph(param);

    Status status = RunPostCubePass(graph);
    EXPECT_EQ(status, SUCCESS);
    GNodePtr fusedMatMul = FindNodeByType(graph, kOpTypeFusedMatMul);
    ASSERT_NE(fusedMatMul, nullptr);
    TensorDesc outputDesc;
    EXPECT_EQ(fusedMatMul->GetOutputDesc(0, outputDesc), GRAPH_SUCCESS);
    EXPECT_EQ(outputDesc.GetDataType(), DT_INT8);
}

// the enable_uncache attr is transferred by CopyOtherAttrs.
TEST_F(FusedMatMulPostCubePathTest, postcubeUncacheAttrTransfer)
{
    MatmulPostCubeParam param;
    param.withUncache = true;
    param.postOp = PostOp::RELU;
    std::shared_ptr<Graph> graph = BuildMatmulPostOpGraph(param);

    Status status = RunPostCubePass(graph);
    EXPECT_EQ(status, SUCCESS);
    GNodePtr fusedMatMul = FindNodeByType(graph, kOpTypeFusedMatMul);
    ASSERT_NE(fusedMatMul, nullptr);
    int64_t enableUncache = 0;
    EXPECT_EQ(fusedMatMul->GetAttr("enable_uncache", enableUncache), GRAPH_SUCCESS);
    EXPECT_EQ(enableUncache, 1);
}

// platform without s8s4 mmad does not fuse.
TEST_F(FusedMatMulPostCubePathTest, postcubeNoS8s4MmadNotFused)
{
    SetNonS8s4Platform();
    MatmulPostCubeParam param;
    param.postOp = PostOp::RELU;
    std::shared_ptr<Graph> graph = BuildMatmulPostOpGraph(param);

    Status status = RunPostCubePass(graph);
    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodesByType(graph, kOpTypeFusedMatMul), 0);
}

// bf16 is not supported on the s8s4 path (fp16 only).
TEST_F(FusedMatMulPostCubePathTest, postcubeDtypeBf16NotFused)
{
    MatmulPostCubeParam param;
    param.dtype = DT_BF16;
    param.postOp = PostOp::RELU;
    std::shared_ptr<Graph> graph = BuildMatmulPostOpGraph(param);

    Status status = RunPostCubePass(graph);
    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodesByType(graph, kOpTypeFusedMatMul), 0);
    EXPECT_EQ(CountNodesByType(graph, kOpTypeMatMulV3), 1);
}

// bias [1, N] is not supported on the s8s4 path (DAV_5102: [N] or empty only).
TEST_F(FusedMatMulPostCubePathTest, postcubeBiasOneNNotFused)
{
    MatmulPostCubeParam param;
    param.hasBias = true;
    param.biasDims = {1, 256};
    param.postOp = PostOp::RELU;
    std::shared_ptr<Graph> graph = BuildMatmulPostOpGraph(param);

    Status status = RunPostCubePass(graph);
    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodesByType(graph, kOpTypeFusedMatMul), 0);
}

// A 0-dim (scalar) bias does not fuse on the s8s4 path: aligned with the canndev
// CheckDav5102BiasShape, which rejects any bias whose dim num is not exactly 1 ([N]).
TEST_F(FusedMatMulPostCubePathTest, postcubeBiasZeroDimNotFused)
{
    MatmulPostCubeParam param;
    param.hasBias = true;
    param.biasDims = {};
    param.postOp = PostOp::RELU;
    std::shared_ptr<Graph> graph = BuildMatmulPostOpGraph(param);

    Status status = RunPostCubePass(graph);
    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodesByType(graph, kOpTypeFusedMatMul), 0);
    EXPECT_EQ(CountNodesByType(graph, kOpTypeMatMulV3), 1);
}

// a post cube chain out of the FusedMatMul white list (LeakyRelu) does not fuse.
TEST_F(FusedMatMulPostCubePathTest, postcubeChainNotMatchedNotFused)
{
    MatmulPostCubeParam param;
    param.postOp = PostOp::LEAKY_RELU;
    std::shared_ptr<Graph> graph = BuildMatmulPostOpGraph(param);

    Status status = RunPostCubePass(graph);
    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodesByType(graph, kOpTypeFusedMatMul), 0);
    EXPECT_EQ(CountNodesByType(graph, kOpTypeMatMulV3), 1);
    EXPECT_EQ(CountNodesByType(graph, kOpTypeLeakyRelu), 1);
}

// force_fp16 + whole-FP32 chain: the Fp16FusionPreparation lowers the descs to FP16 before the
// checks (canndev order: Prepare -> ScenarioCheck), the chain fuses into FusedMatMul and the
// fused node keeps the lowered FP16 target dtypes (UpdateFusedNodeDtype).
TEST_F(FusedMatMulPostCubePathTest, postcubeForceFp16Fp32ChainFusesFp16)
{
    MatmulPostCubeParam param;
    param.dtype = DT_FLOAT;
    param.postOp = PostOp::RELU;
    std::shared_ptr<Graph> graph = BuildMatmulPostOpGraph(param);

    const std::map<std::string, std::string> options = {{"ge.exec.precision_mode", "force_fp16"},
                                                        {"precision_mode", "force_fp16"}};
    ScopedGeOptions optionsGuard(options);

    Status status = RunPostCubePass(graph);
    EXPECT_EQ(status, SUCCESS);
    EXPECT_EQ(CountNodesByType(graph, kOpTypeFusedMatMul), 1);
    EXPECT_EQ(CountNodesByType(graph, kOpTypeMatMulV3), 0);
    EXPECT_EQ(CountNodesByType(graph, kOpTypeRelu), 0);

    GNodePtr fusedMatMul = FindNodeByType(graph, kOpTypeFusedMatMul);
    ASSERT_NE(fusedMatMul, nullptr);
    EXPECT_EQ(GetFusedOpTypeAttr(fusedMatMul), kFusedOpTypeRelu);
    // The original chain was FP32: the FP16 dtypes prove the preparation and the target-dtype
    // writeback both ran.
    TensorDesc inputDesc;
    ASSERT_EQ(fusedMatMul->GetInputDesc(0, inputDesc), GRAPH_SUCCESS);
    EXPECT_EQ(inputDesc.GetDataType(), DT_FLOAT16);
    TensorDesc outputDesc;
    ASSERT_EQ(fusedMatMul->GetOutputDesc(0, outputDesc), GRAPH_SUCCESS);
    EXPECT_EQ(outputDesc.GetDataType(), DT_FLOAT16);
}

// The mode resolution is the FE interposition: no precision_mode entry in the injected option
// map (the "unset" precondition, the same as the device run without any setting) resolves to
// an empty mode, the FP16 preparation skips and the FP32 chain hits the s8s4 dtype table. On
// the device the real fe::Configuration::GetPrecisionModeStr() would resolve the FE default
// (force_fp16 for inference) instead - this UT pins the resolution-empty behavior only.
TEST_F(FusedMatMulPostCubePathTest, postcubeFp32ChainNoModeNotFusedInUt)
{
    MatmulPostCubeParam param;
    param.dtype = DT_FLOAT;
    param.postOp = PostOp::RELU;
    std::shared_ptr<Graph> graph = BuildMatmulPostOpGraph(param);

    Status status = RunPostCubePass(graph);
    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodesByType(graph, kOpTypeFusedMatMul), 0);
    EXPECT_EQ(CountNodesByType(graph, kOpTypeMatMulV3), 1);
    EXPECT_EQ(CountNodesByType(graph, kOpTypeRelu), 1);
}

// A customize_dtypes entry on the FusedMatMul op type forbids the precision reduction: the FP32
// chain is not lowered, the s8s4 dtype check rejects it, no fusion, descs untouched.
TEST_F(FusedMatMulPostCubePathTest, postcubeForceFp16CustomizeDtypeNotFused)
{
    MatmulPostCubeParam param;
    param.dtype = DT_FLOAT;
    param.postOp = PostOp::RELU;
    std::shared_ptr<Graph> graph = BuildMatmulPostOpGraph(param);

    const std::map<std::string, std::string> options = {{"ge.exec.precision_mode", "force_fp16"},
                                                        {"precision_mode", "force_fp16"},
                                                        {"ge.exec.customize_dtypes", "FusedMatMul:0:fp16"},
                                                        {"customize_dtypes", "FusedMatMul:0:fp16"}};
    ScopedGeOptions optionsGuard(options);

    Status status = RunPostCubePass(graph);
    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodesByType(graph, kOpTypeFusedMatMul), 0);
    EXPECT_EQ(CountNodesByType(graph, kOpTypeMatMulV3), 1);

    GNodePtr matmul = FindNodeByType(graph, kOpTypeMatMulV3);
    ASSERT_NE(matmul, nullptr);
    TensorDesc inputDesc;
    ASSERT_EQ(matmul->GetInputDesc(0, inputDesc), GRAPH_SUCCESS);
    EXPECT_EQ(inputDesc.GetDataType(), DT_FLOAT);
}

// force_fp16 + whole-FP32 chain rejected by the postcube ability gate (dynamic shape flag): the
// temporarily lowered descs must be restored by the Fp16FusionPreparation destructor (RAII),
// the matmul keeps the original FP32 descs.
TEST_F(FusedMatMulPostCubePathTest, postcubeForceFp16RejectedChainRestoresDescs)
{
    MatmulPostCubeParam param;
    param.dtype = DT_FLOAT;
    param.x1Dims = {-1, 128};
    param.yDims = {-1, 256};
    param.postOp = PostOp::RELU;
    std::shared_ptr<Graph> graph = BuildMatmulPostOpGraph(param);

    GNodePtr matmulNode = FindNodeByType(graph, kOpTypeMatMulV3);
    ASSERT_NE(matmulNode, nullptr);
    // The int64 SetAttr overload takes a non-const lvalue reference.
    int64_t abilityFlag = 0x08; // the NodeCantAccess bit
    matmulNode->SetAttr("support_fixpipe_ability", abilityFlag);

    const std::map<std::string, std::string> options = {{"ge.exec.precision_mode", "force_fp16"},
                                                        {"precision_mode", "force_fp16"}};
    ScopedGeOptions optionsGuard(options);

    Status status = RunPostCubePass(graph);
    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodesByType(graph, kOpTypeFusedMatMul), 0);
    EXPECT_EQ(CountNodesByType(graph, kOpTypeMatMulV3), 1);
    EXPECT_EQ(CountNodesByType(graph, kOpTypeRelu), 1);

    // Restored by the RAII destructor, not left as the temporarily lowered FP16.
    TensorDesc inputDesc;
    ASSERT_EQ(matmulNode->GetInputDesc(0, inputDesc), GRAPH_SUCCESS);
    EXPECT_EQ(inputDesc.GetDataType(), DT_FLOAT);
    TensorDesc outputDesc;
    ASSERT_EQ(matmulNode->GetOutputDesc(0, outputDesc), GRAPH_SUCCESS);
    EXPECT_EQ(outputDesc.GetDataType(), DT_FLOAT);
}

// DAV_3510 platform: the single pass dispatches to the direct fusion path, the matmul + relu is
// fused into FusedMatMul directly (no FixPipe/PostCube involved).
TEST_F(FusedMatMulPostCubePathTest, postcubeDav3510DispatchesToDirectPath)
{
    SetDav3510ArchPlatform();
    MatmulPostCubeParam param;
    param.postOp = PostOp::RELU;
    std::shared_ptr<Graph> graph = BuildMatmulPostOpGraph(param);

    Status status = RunPostCubePass(graph);
    EXPECT_EQ(status, SUCCESS);
    EXPECT_EQ(CountNodesByType(graph, kOpTypeFusedMatMul), 1);
    EXPECT_EQ(CountNodesByType(graph, kOpTypeMatMulV3), 0);
    EXPECT_EQ(CountNodesByType(graph, kOpTypeRelu), 0);
    EXPECT_EQ(CountNodesByType(graph, kOpTypeFixPipe), 0);

    GNodePtr fusedMatMul = FindNodeByType(graph, kOpTypeFusedMatMul);
    ASSERT_NE(fusedMatMul, nullptr);
    EXPECT_EQ(GetFusedOpTypeAttr(fusedMatMul), kFusedOpTypeRelu);
}

// ND input is forbidden by the postcube ability gate: the "support_fixpipe_ability" int64 attr
// on the matmul node carries the NodeCantAccess bit (0x08). In production the flag is set by
// the canndev FixPipeAbilityProcessPass (REG_PASS with FORBIDDEN_CLOSE, it always runs) and
// consumed by the PostCubeUtils::GetPostCubeNodeList entry; the UT sets it directly.
TEST_F(FusedMatMulPostCubePathTest, postcubeNdInputNotFused)
{
    MatmulPostCubeParam param;
    param.x1x2Format = FORMAT_ND;
    param.postOp = PostOp::RELU;
    std::shared_ptr<Graph> graph = BuildMatmulPostOpGraph(param);

    GNodePtr matmulNode = FindNodeByType(graph, kOpTypeMatMulV3);
    ASSERT_NE(matmulNode, nullptr);
    // The int64 SetAttr overload takes a non-const lvalue reference.
    int64_t abilityFlag = 0x08; // the NodeCantAccess bit
    matmulNode->SetAttr("support_fixpipe_ability", abilityFlag);

    Status status = RunPostCubePass(graph);
    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodesByType(graph, kOpTypeFusedMatMul), 0);
    EXPECT_EQ(CountNodesByType(graph, kOpTypeMatMulV3), 1);
    EXPECT_EQ(CountNodesByType(graph, kOpTypeRelu), 1);
}

// Dynamic shape is forbidden by the postcube ability gate (the same NodeCantAccess flag set by
// the canndev FixPipeAbilityProcessPass in production).
TEST_F(FusedMatMulPostCubePathTest, postcubeDynamicShapeNotFused)
{
    MatmulPostCubeParam param;
    param.x1Dims = {-1, 128};
    param.yDims = {-1, 256};
    param.postOp = PostOp::RELU;
    std::shared_ptr<Graph> graph = BuildMatmulPostOpGraph(param);

    GNodePtr matmulNode = FindNodeByType(graph, kOpTypeMatMulV3);
    ASSERT_NE(matmulNode, nullptr);
    // The int64 SetAttr overload takes a non-const lvalue reference.
    int64_t abilityFlag = 0x08; // the NodeCantAccess bit
    matmulNode->SetAttr("support_fixpipe_ability", abilityFlag);

    Status status = RunPostCubePass(graph);
    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodesByType(graph, kOpTypeFusedMatMul), 0);
    EXPECT_EQ(CountNodesByType(graph, kOpTypeMatMulV3), 1);
    EXPECT_EQ(CountNodesByType(graph, kOpTypeRelu), 1);
}

// Without the ability flag (the ability pass absent), the forbidden-shape node is no longer
// gated here and the fusion proceeds - the same behavior of the canndev FusedMatMulFusionPass
// in the same environment (the unsupported cases are then rejected by the FusedMatMul runtime
// tiling).
TEST_F(FusedMatMulPostCubePathTest, postcubeDynamicShapeWithoutAbilityFlagStillFuses)
{
    MatmulPostCubeParam param;
    param.x1Dims = {-1, 128};
    param.yDims = {-1, 256};
    param.postOp = PostOp::RELU;
    std::shared_ptr<Graph> graph = BuildMatmulPostOpGraph(param);

    Status status = RunPostCubePass(graph);
    EXPECT_EQ(status, SUCCESS);
    EXPECT_EQ(CountNodesByType(graph, kOpTypeFusedMatMul), 1);
}

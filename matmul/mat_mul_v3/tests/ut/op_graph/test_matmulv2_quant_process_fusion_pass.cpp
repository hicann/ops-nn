/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <gtest/gtest.h>

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "../../../op_graph/fusion_pass/matmulv2_quant_process_fusion_pass.h"
#include "ge/compliant_node_builder.h"
#include "ge/es_graph_builder.h"
#include "platform/platform_info.h"

namespace {
using namespace ge;
using namespace ge::es;

constexpr int32_t kQuantInputPort = 0;
constexpr int32_t kCubeXPort = 0;
constexpr int32_t kCubeWeightPort = 1;
constexpr int32_t kCubeBiasPort = 2;
constexpr int32_t kDequantXPort = 0;
constexpr int32_t kDequantScalePort = 1;
constexpr int32_t kOutPort = 0;

void SetPlatform(const std::string& soc)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optionalInfo;
    platformInfo.str_info.short_soc_version = soc;
    optionalInfo.soc_version = soc;
    fe::PlatformInfoManager::Instance().platform_info_map_[soc] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optionalInfo);
}

class MatMulV2QuantTestGraph {
public:
    explicit MatMulV2QuantTestGraph(const std::string& name) : builder_(name.c_str())
    {
        rawGraph_ = builder_.GetCGraphBuilder()->GetGraph();
    }

    GNode& AddNode(const std::string& nodeName, const std::string& opType,
                   const std::vector<std::pair<std::string, bool>>& inputs)
    {
        std::vector<CompliantNodeBuilder::IrInputDef> inputDefs;
        for (const auto& in : inputs) {
            inputDefs.push_back(
                {in.first,
                 in.second ? CompliantNodeBuilder::kEsIrInputRequired : CompliantNodeBuilder::kEsIrInputOptional, ""});
        }
        GNode node = CompliantNodeBuilder(rawGraph_)
                         .OpType(opType.c_str())
                         .Name(nodeName.c_str())
                         .IrDefInputs(inputDefs)
                         .IrDefOutputs({{"y", CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                         .Build();
        nodes_[nodeName] = node;
        return nodes_[nodeName];
    }

    GNode& CreateConst(const std::string& nodeName, const std::vector<int64_t>& dims, DataType dtype,
                       const std::vector<uint8_t>& data)
    {
        TensorDesc desc(Shape(dims), FORMAT_ND, dtype);
        Tensor tensor(desc, data);
        AttrValue attrValue;
        (void)attrValue.SetAttrValue(tensor);
        GNode node = CompliantNodeBuilder(rawGraph_)
                         .OpType("Const")
                         .Name(nodeName.c_str())
                         .IrDefOutputs({{"y", CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                         .IrDefAttrs({{"value", CompliantNodeBuilder::kEsAttrOptional, "VT_TENSOR", attrValue}})
                         .InstanceOutputShape("y", dims)
                         .InstanceOutputDataType("y", dtype)
                         .InstanceOutputFormat("y", FORMAT_ND)
                         .Build();
        nodes_[nodeName] = node;
        return nodes_[nodeName];
    }

    void Connect(const std::string& from, int32_t fromPort, const std::string& to, int32_t toPort)
    {
        (void)AddEdgeAndUpdatePeerDesc(*rawGraph_, nodes_[from], fromPort, nodes_[to], toPort);
    }

    void ConnectInput(const EsTensorHolder& input, const std::string& to, int32_t toPort)
    {
        (void)AddEdgeAndUpdatePeerDesc(*rawGraph_, *input.GetProducer(), kOutPort, nodes_[to], toPort);
    }

    void SetInDesc(const std::string& nodeName, int32_t idx, DataType dtype, Format format,
                   const std::vector<int64_t>& dims)
    {
        TensorDesc desc;
        (void)nodes_[nodeName].GetInputDesc(idx, desc);
        desc.SetDataType(dtype);
        desc.SetFormat(format);
        desc.SetShape(Shape(dims));
        desc.SetOriginFormat(format);
        desc.SetOriginShape(Shape(dims));
        (void)nodes_[nodeName].UpdateInputDesc(idx, desc);
    }

    void SetOutDesc(const std::string& nodeName, int32_t idx, DataType dtype, Format format,
                    const std::vector<int64_t>& dims)
    {
        TensorDesc desc;
        (void)nodes_[nodeName].GetOutputDesc(idx, desc);
        desc.SetDataType(dtype);
        desc.SetFormat(format);
        desc.SetShape(Shape(dims));
        desc.SetOriginFormat(format);
        desc.SetOriginShape(Shape(dims));
        (void)nodes_[nodeName].UpdateOutputDesc(idx, desc);
    }

    std::shared_ptr<Graph> Build(const std::string& outputNode)
    {
        auto holder = builder_.GetCGraphBuilder()->GetTensorHolderFromNode(nodes_[outputNode], kOutPort);
        geGraph_ = builder_.BuildAndReset({holder});
        if (geGraph_ != nullptr) {
            rawGraph_ = geGraph_.get();
        }
        return geGraph_;
    }

    EsTensorHolder CreateInput(const std::string& name, DataType dtype, Format format, const std::vector<int64_t>& dims)
    {
        return builder_.CreateInput(inputIdx_++, name.c_str(), dtype, format, dims);
    }

    GNode& Node(const std::string& nodeName) { return nodes_[nodeName]; }

private:
    EsGraphBuilder builder_;
    Graph* rawGraph_ = nullptr;
    std::map<std::string, GNode> nodes_;
    int64_t inputIdx_ = 0;
    std::shared_ptr<Graph> geGraph_ = nullptr;
};

struct MatMulV2QuantGraphOptions {
    bool withWeightQuant = true; // weight 走 AscendWeightQuant（quant 场景常态）
    bool withBias = false;
    std::vector<int64_t> xDims = {16, 16};
    std::vector<int64_t> weightDims = {16, 16};
    bool weightTranspose = false; // cube 的 transpose_x2 属性
    std::string cubeType = "MatMulV2";
};

std::shared_ptr<Graph> BuildMatMulV2QuantGraph(MatMulV2QuantTestGraph& tg, const MatMulV2QuantGraphOptions& opt)
{
    auto xInput = tg.CreateInput("x", DT_FLOAT16, FORMAT_ND, opt.xDims);

    GNode& quant = tg.AddNode("quant", "AscendQuant", {{"x", true}});
    float scaleAttr = 1.2F;
    float offsetAttr = 1.1F;
    (void)quant.SetAttr(AscendString("scale"), scaleAttr);
    (void)quant.SetAttr(AscendString("offset"), offsetAttr);

    GNode& cube = tg.AddNode("cube", opt.cubeType, {{"x1", true}, {"x2", true}, {"bias", false}, {"offset_w", false}});
    bool transposeX2 = opt.weightTranspose;
    (void)cube.SetAttr(AscendString("transpose_x2"), transposeX2);

    const size_t weightElemNum = static_cast<size_t>(Shape(opt.weightDims).GetShapeSize());
    if (opt.withWeightQuant) {
        std::vector<uint8_t> weightData(weightElemNum * sizeof(uint16_t), 1U);
        (void)tg.CreateConst("weight_const", opt.weightDims, DT_FLOAT16, weightData);
        std::vector<uint8_t> offsetData(4, 0U);
        (void)tg.CreateConst("offset_const", {1}, DT_FLOAT16, offsetData);
        GNode& awq = tg.AddNode("awq", "AscendWeightQuant", {{"x", true}, {"offset", false}});
        int32_t dstType = static_cast<int32_t>(DT_INT8);
        (void)awq.SetAttr(AscendString("dst_type"), dstType);
        tg.Connect("weight_const", kOutPort, "awq", kQuantInputPort);
        tg.Connect("offset_const", kOutPort, "awq", 1);
        tg.Connect("awq", kOutPort, "cube", kCubeWeightPort);
        tg.SetInDesc("awq", 0, DT_FLOAT16, FORMAT_ND, opt.weightDims);
        tg.SetInDesc("awq", 1, DT_FLOAT16, FORMAT_ND, {1});
        tg.SetOutDesc("awq", kOutPort, DT_INT8, FORMAT_ND, opt.weightDims);
    } else {
        std::vector<uint8_t> weightData(weightElemNum * sizeof(uint16_t), 1U);
        (void)tg.CreateConst("weight_const", opt.weightDims, DT_FLOAT16, weightData);
        tg.Connect("weight_const", kOutPort, "cube", kCubeWeightPort);
    }

    if (opt.withBias) {
        std::vector<uint8_t> biasData(16 * sizeof(int32_t), 0U);
        (void)tg.CreateConst("bias_const", {16}, DT_INT32, biasData);
        tg.Connect("bias_const", kOutPort, "cube", kCubeBiasPort);
    }

    std::vector<uint8_t> scaleData(16 * sizeof(uint64_t), 0U);
    (void)tg.CreateConst("deq_scale", {16}, DT_UINT64, scaleData);

    GNode& dequant = tg.AddNode("dequant", "AscendDequant", {{"x", true}, {"deq_scale", true}});
    GNode& relu = tg.AddNode("relu", "Relu", {{"x", true}});

    tg.ConnectInput(xInput, "quant", kQuantInputPort);
    tg.Connect("quant", kOutPort, "cube", kCubeXPort);
    tg.Connect("deq_scale", kOutPort, "dequant", kDequantScalePort);
    tg.Connect("cube", kOutPort, "dequant", kDequantXPort);
    tg.Connect("dequant", kOutPort, "relu", kQuantInputPort);

    const std::vector<int64_t> cubeOutDims = opt.weightDims;
    tg.SetInDesc("quant", 0, DT_FLOAT16, FORMAT_ND, opt.xDims);
    tg.SetOutDesc("quant", kOutPort, DT_INT8, FORMAT_ND, opt.xDims);
    tg.SetInDesc("cube", kCubeXPort, DT_INT8, FORMAT_ND, opt.xDims);
    tg.SetInDesc("cube", kCubeWeightPort, DT_INT8, FORMAT_ND, opt.weightDims);
    tg.SetOutDesc("cube", kOutPort, DT_INT32, FORMAT_ND, cubeOutDims);
    tg.SetInDesc("dequant", kDequantXPort, DT_INT32, FORMAT_ND, cubeOutDims);
    tg.SetInDesc("dequant", kDequantScalePort, DT_UINT64, FORMAT_ND, {16});
    tg.SetOutDesc("dequant", kOutPort, DT_FLOAT16, FORMAT_ND, cubeOutDims);
    tg.SetInDesc("relu", kQuantInputPort, DT_FLOAT16, FORMAT_ND, cubeOutDims);
    tg.SetOutDesc("relu", kOutPort, DT_FLOAT16, FORMAT_ND, cubeOutDims);
    return tg.Build("relu");
}

int32_t CountByType(const std::shared_ptr<Graph>& graph, const std::string& opType)
{
    int32_t count = 0;
    for (const auto& node : graph->GetAllNodes()) {
        AscendString type;
        if (node.GetType(type) == GRAPH_SUCCESS && type.GetString() != nullptr &&
            std::string(type.GetString()) == opType) {
            count++;
        }
    }
    return count;
}

int32_t CountByName(const std::shared_ptr<Graph>& graph, const std::string& nodeName)
{
    int32_t count = 0;
    for (const auto& node : graph->GetAllNodes()) {
        AscendString name;
        if (node.GetName(name) == GRAPH_SUCCESS && name.GetString() != nullptr &&
            std::string(name.GetString()) == nodeName) {
            count++;
        }
    }
    return count;
}

GNodePtr FindByName(const std::shared_ptr<Graph>& graph, const std::string& nodeName)
{
    for (const auto& node : graph->GetAllNodes()) {
        AscendString name;
        if (node.GetName(name) == GRAPH_SUCCESS && name.GetString() != nullptr &&
            std::string(name.GetString()) == nodeName) {
            return std::make_shared<GNode>(node);
        }
    }
    return nullptr;
}

GNodePtr FindByType(const std::shared_ptr<Graph>& graph, const std::string& opType)
{
    for (const auto& node : graph->GetAllNodes()) {
        AscendString type;
        if (node.GetType(type) == GRAPH_SUCCESS && type.GetString() != nullptr &&
            std::string(type.GetString()) == opType) {
            return std::make_shared<GNode>(node);
        }
    }
    return nullptr;
}

std::string PeerInType(const GNodePtr& node, int32_t inIdx)
{
    auto peer = node->GetInDataNodesAndPortIndexs(inIdx);
    if (peer.first == nullptr) {
        return "";
    }
    AscendString type;
    if (peer.first->GetType(type) != GRAPH_SUCCESS || type.GetString() == nullptr) {
        return "";
    }
    return type.GetString();
}

std::string PeerInName(const GNodePtr& node, int32_t inIdx)
{
    auto peer = node->GetInDataNodesAndPortIndexs(inIdx);
    if (peer.first == nullptr) {
        return "";
    }
    AscendString name;
    if (peer.first->GetName(name) != GRAPH_SUCCESS || name.GetString() == nullptr) {
        return "";
    }
    return name.GetString();
}

// Probe 子类覆写 CheckOpSupported 注入支持性：基类默认当前恒 true（GE 接口兼容性问题），
// 覆写用于独立控制 IsCubeNodeSupported 分支走向（验证 ROLLBACK 分支逻辑——生产当前不可达）
class ProbeMatmulV2QuantProcessFusionPass : public ops::MatmulV2QuantProcessFusionPass {
public:
    bool cubeSupported = true;

    int64_t CoValueOf(const GNodePtr& cubeNode)
    {
        int64_t co = -1;
        (void)GetCoValue(cubeNode, co);
        return co;
    }

protected:
    bool CheckOpSupported(const GNode& node) const override
    {
        (void)node;
        return cubeSupported;
    }
};
} // namespace

TEST(MatmulV2QuantProcessFusionPassTest, biasOptimizeSuccess)
{
    SetPlatform("Ascend950");
    MatMulV2QuantTestGraph tg("mmv2_bias_optimize");
    MatMulV2QuantGraphOptions opt;
    auto graph = BuildMatMulV2QuantGraph(tg, opt);
    ASSERT_NE(graph, nullptr);
    ProbeMatmulV2QuantProcessFusionPass pass;
    pass.cubeSupported = true;
    CustomPassContext passContext;
    EXPECT_EQ(pass.Run(graph, passContext), SUCCESS);
    EXPECT_EQ(CountByType(graph, "QuantBiasOptimization"), 1);
    EXPECT_EQ(CountByType(graph, "QuantWeightRollBack"), 0);
    // bias optimize 场景保留 quant/dequant 两个量化节点
    EXPECT_EQ(CountByName(graph, "quant"), 1);
    EXPECT_EQ(CountByName(graph, "dequant"), 1);
    const GNodePtr cube = FindByName(graph, "cube");
    ASSERT_NE(cube, nullptr);
    EXPECT_EQ(PeerInType(cube, kCubeBiasPort), "QuantBiasOptimization");
    EXPECT_EQ(PeerInType(cube, kCubeWeightPort), "AscendWeightQuant");
    int64_t offsetX = -1;
    EXPECT_EQ(cube->GetAttr(AscendString("offset_x"), offsetX), GRAPH_SUCCESS);
    EXPECT_EQ(offsetX, 1); // quant 节点 offset=1.1 取整后写入 cube 的 offset_x
    const GNodePtr hostOp = FindByType(graph, "QuantBiasOptimization");
    ASSERT_NE(hostOp, nullptr);
    Tensor biasTensor;
    EXPECT_EQ(hostOp->GetInputConstData(0, biasTensor), GRAPH_SUCCESS);
    ASSERT_NE(biasTensor.GetData(), nullptr);
    // 无 bias 输入时按 co=weight dims[1]=16 补零 INT32 bias
    EXPECT_EQ(biasTensor.GetSize(), 16U * sizeof(int32_t));
    for (size_t i = 0; i < biasTensor.GetSize(); ++i) {
        EXPECT_EQ(biasTensor.GetData()[i], 0U);
    }
    EXPECT_EQ(biasTensor.GetTensorDesc().GetDataType(), DT_INT32);
    EXPECT_EQ(hostOp->GetInputsSize(), 3U);
    EXPECT_EQ(PeerInType(hostOp, 0), "Const");
    EXPECT_EQ(PeerInName(hostOp, 1), "weight_const");
    EXPECT_EQ(PeerInType(hostOp, 2), "Const");
}

TEST(MatmulV2QuantProcessFusionPassTest, biasOptimizeWithBiasSuccess)
{
    SetPlatform("Ascend950");
    MatMulV2QuantTestGraph tg("mmv2_bias_optimize_with_bias");
    MatMulV2QuantGraphOptions opt;
    opt.withBias = true;
    auto graph = BuildMatMulV2QuantGraph(tg, opt);
    ASSERT_NE(graph, nullptr);
    ProbeMatmulV2QuantProcessFusionPass pass;
    pass.cubeSupported = true;
    CustomPassContext passContext;
    EXPECT_EQ(pass.Run(graph, passContext), SUCCESS);
    EXPECT_EQ(CountByType(graph, "QuantBiasOptimization"), 1);
    const GNodePtr hostOp = FindByType(graph, "QuantBiasOptimization");
    ASSERT_NE(hostOp, nullptr);
    // 已有 bias 输入时 host op 的 bias 入参来自既有 bias_const
    EXPECT_EQ(PeerInName(hostOp, 0), "bias_const");
    EXPECT_EQ(PeerInName(hostOp, 1), "weight_const");
}

TEST(MatmulV2QuantProcessFusionPassTest, quantRollbackSuccess)
{
    SetPlatform("Ascend950");
    MatMulV2QuantTestGraph tg("mmv2_rollback");
    MatMulV2QuantGraphOptions opt;
    auto graph = BuildMatMulV2QuantGraph(tg, opt);
    ASSERT_NE(graph, nullptr);
    ProbeMatmulV2QuantProcessFusionPass pass;
    pass.cubeSupported = false;
    CustomPassContext passContext;
    EXPECT_EQ(pass.Run(graph, passContext), SUCCESS);
    EXPECT_EQ(CountByType(graph, "QuantWeightRollBack"), 1);
    EXPECT_EQ(CountByType(graph, "QuantBiasRollBack"), 0); // 无 bias 输入时不创建
    EXPECT_EQ(CountByType(graph, "AscendWeightQuant"), 0);
    EXPECT_EQ(CountByName(graph, "offset_const"), 0);
    // rollback 场景拆除 quant/dequant 两个量化节点
    EXPECT_EQ(CountByName(graph, "quant"), 0);
    EXPECT_EQ(CountByName(graph, "dequant"), 0);
    const GNodePtr cube = FindByName(graph, "cube");
    ASSERT_NE(cube, nullptr);
    EXPECT_EQ(PeerInType(cube, kCubeXPort), "Data");
    EXPECT_EQ(PeerInType(cube, kCubeWeightPort), "QuantWeightRollBack");
    const GNodePtr relu = FindByName(graph, "relu");
    ASSERT_NE(relu, nullptr);
    EXPECT_EQ(PeerInType(relu, kQuantInputPort), "MatMulV2");
    TensorDesc cubeOut;
    (void)cube->GetOutputDesc(kOutPort, cubeOut);
    EXPECT_EQ(cubeOut.GetDataType(), DT_FLOAT16);
    TensorDesc cubeIn;
    (void)cube->GetInputDesc(kCubeXPort, cubeIn);
    EXPECT_EQ(cubeIn.GetDataType(), DT_FLOAT16);
    const GNodePtr weightRb = FindByType(graph, "QuantWeightRollBack");
    ASSERT_NE(weightRb, nullptr);
    AscendString cubeOpType;
    EXPECT_EQ(weightRb->GetAttr(AscendString("cube_op_type"), cubeOpType), GRAPH_SUCCESS);
    EXPECT_STREQ(cubeOpType.GetString(), "MatMulV2");
    float scale = 0.0F;
    EXPECT_EQ(weightRb->GetAttr(AscendString("scale"), scale), GRAPH_SUCCESS);
    EXPECT_FLOAT_EQ(scale, 1.2F);
}

TEST(MatmulV2QuantProcessFusionPassTest, quantRollbackWithBiasSuccess)
{
    SetPlatform("Ascend950");
    MatMulV2QuantTestGraph tg("mmv2_rollback_with_bias");
    MatMulV2QuantGraphOptions opt;
    opt.withBias = true;
    auto graph = BuildMatMulV2QuantGraph(tg, opt);
    ASSERT_NE(graph, nullptr);
    ProbeMatmulV2QuantProcessFusionPass pass;
    pass.cubeSupported = false;
    CustomPassContext passContext;
    EXPECT_EQ(pass.Run(graph, passContext), SUCCESS);
    EXPECT_EQ(CountByType(graph, "QuantWeightRollBack"), 1);
    EXPECT_EQ(CountByType(graph, "QuantBiasRollBack"), 1);
    const GNodePtr cube = FindByName(graph, "cube");
    ASSERT_NE(cube, nullptr);
    EXPECT_EQ(PeerInType(cube, kCubeBiasPort), "QuantBiasRollBack");
    EXPECT_EQ(PeerInType(cube, kCubeWeightPort), "QuantWeightRollBack");
}

TEST(MatmulV2QuantProcessFusionPassTest, unknownShapeSkip)
{
    SetPlatform("Ascend950");
    MatMulV2QuantTestGraph tg("mmv2_unknown_shape");
    MatMulV2QuantGraphOptions opt;
    opt.xDims = {-1, 16};
    auto graph = BuildMatMulV2QuantGraph(tg, opt);
    ASSERT_NE(graph, nullptr);
    ProbeMatmulV2QuantProcessFusionPass pass;
    pass.cubeSupported = true;
    CustomPassContext passContext;
    // UNDEFINED 跳过场景：原版 Fusion 返回 SUCCESS（旧框架记 effect=1）, Run 同构返回 SUCCESS
    // （NOT_CHANGED 仅表示未匹配——见 skipsOtherCubeType 等用例）
    EXPECT_EQ(pass.Run(graph, passContext), SUCCESS);
    EXPECT_EQ(CountByType(graph, "QuantBiasOptimization"), 0);
    EXPECT_EQ(CountByType(graph, "QuantWeightRollBack"), 0);
    EXPECT_EQ(CountByName(graph, "quant"), 1);
    EXPECT_EQ(CountByName(graph, "dequant"), 1);
}

TEST(MatmulV2QuantProcessFusionPassTest, skipsOtherCubeType)
{
    SetPlatform("Ascend950");
    MatMulV2QuantTestGraph tg("mmv2_other_type");
    MatMulV2QuantGraphOptions opt;
    opt.cubeType = "MatMul";
    auto graph = BuildMatMulV2QuantGraph(tg, opt);
    ASSERT_NE(graph, nullptr);
    ProbeMatmulV2QuantProcessFusionPass pass;
    pass.cubeSupported = true;
    CustomPassContext passContext;
    EXPECT_EQ(pass.Run(graph, passContext), GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountByType(graph, "QuantBiasOptimization"), 0);
    EXPECT_EQ(CountByType(graph, "QuantWeightRollBack"), 0);
}

TEST(MatmulV2QuantProcessFusionPassTest, skipsNonQuantProducer)
{
    SetPlatform("Ascend950");
    MatMulV2QuantTestGraph tg("mmv2_non_quant_producer");
    MatMulV2QuantGraphOptions opt;
    auto graph = BuildMatMulV2QuantGraph(tg, opt);
    ASSERT_NE(graph, nullptr);
    // 将 x1 口的输入从 quant 改接为 relu，模拟非 AscendQuant 生产者
    const GNodePtr cube = FindByName(graph, "cube");
    ASSERT_NE(cube, nullptr);
    const auto quantPeer = cube->GetInDataNodesAndPortIndexs(kCubeXPort);
    ASSERT_NE(quantPeer.first, nullptr);
    const GNodePtr relu = FindByName(graph, "relu");
    ASSERT_NE(relu, nullptr);
    const auto xPeer = quantPeer.first->GetInDataNodesAndPortIndexs(kQuantInputPort);
    ASSERT_NE(xPeer.first, nullptr);
    ASSERT_EQ(graph->RemoveEdge(*quantPeer.first, quantPeer.second, *cube, kCubeXPort), GRAPH_SUCCESS);
    ASSERT_EQ(graph->AddDataEdge(*xPeer.first, xPeer.second, *cube, kCubeXPort), GRAPH_SUCCESS);
    ProbeMatmulV2QuantProcessFusionPass pass;
    pass.cubeSupported = true;
    CustomPassContext passContext;
    EXPECT_EQ(pass.Run(graph, passContext), GRAPH_NOT_CHANGED);
}

TEST(MatmulV2QuantProcessFusionPassTest, skipsWhenNoDequantConsumer)
{
    SetPlatform("Ascend950");
    MatMulV2QuantTestGraph tg("mmv2_no_dequant");
    MatMulV2QuantGraphOptions opt;
    auto graph = BuildMatMulV2QuantGraph(tg, opt);
    ASSERT_NE(graph, nullptr);
    // 将 dequant 从 cube 输出摘除，cube 直连 relu
    const GNodePtr cube = FindByName(graph, "cube");
    ASSERT_NE(cube, nullptr);
    const GNodePtr dequant = FindByName(graph, "dequant");
    ASSERT_NE(dequant, nullptr);
    const GNodePtr relu = FindByName(graph, "relu");
    ASSERT_NE(relu, nullptr);
    ASSERT_EQ(graph->RemoveEdge(*cube, kOutPort, *dequant, kDequantXPort), GRAPH_SUCCESS);
    ASSERT_EQ(graph->RemoveEdge(*dequant, kOutPort, *relu, kQuantInputPort), GRAPH_SUCCESS);
    ASSERT_EQ(graph->AddDataEdge(*cube, kOutPort, *relu, kQuantInputPort), GRAPH_SUCCESS);
    ProbeMatmulV2QuantProcessFusionPass pass;
    pass.cubeSupported = true;
    CustomPassContext passContext;
    EXPECT_EQ(pass.Run(graph, passContext), GRAPH_NOT_CHANGED);
}

TEST(MatmulV2QuantProcessFusionPassTest, getCoValueTransposeBranch)
{
    SetPlatform("Ascend950");
    ProbeMatmulV2QuantProcessFusionPass pass;
    MatMulV2QuantTestGraph tgNoTrans("mmv2_co_no_trans");
    MatMulV2QuantGraphOptions optNoTrans;
    optNoTrans.weightDims = {8, 16};
    auto graphNoTrans = BuildMatMulV2QuantGraph(tgNoTrans, optNoTrans);
    ASSERT_NE(graphNoTrans, nullptr);
    const GNodePtr cubeNoTrans = FindByName(graphNoTrans, "cube");
    ASSERT_NE(cubeNoTrans, nullptr);
    // 非转置：co 取 dims[1]
    EXPECT_EQ(pass.CoValueOf(cubeNoTrans), 16);

    MatMulV2QuantTestGraph tgTrans("mmv2_co_trans");
    MatMulV2QuantGraphOptions optTrans;
    optTrans.weightDims = {8, 16};
    optTrans.weightTranspose = true;
    auto graphTrans = BuildMatMulV2QuantGraph(tgTrans, optTrans);
    ASSERT_NE(graphTrans, nullptr);
    const GNodePtr cubeTrans = FindByName(graphTrans, "cube");
    ASSERT_NE(cubeTrans, nullptr);
    // transpose_x2=true：co 取 dims[0]
    EXPECT_EQ(pass.CoValueOf(cubeTrans), 8);
}

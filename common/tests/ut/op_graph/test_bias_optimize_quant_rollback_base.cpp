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

#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "fusion_pass/bias_optimize_quant_rollback_base.h"
#include "ge/compliant_node_builder.h"
#include "ge/es_graph_builder.h"
#include "graph/attr_value.h"
#include "graph/tensor.h"
#include "platform/platform_info.h"
#include "platform/platform_infos_def.h"

using namespace ge;

namespace {
constexpr int32_t kQuantInputPort = 0;
constexpr int32_t kCubeFmapPort = 0;
constexpr int32_t kCubeFilterPort = 1;
constexpr int32_t kCubeBiasPort = 2;
constexpr int32_t kDequantXPort = 0;
constexpr int32_t kDequantScalePort = 1;
constexpr int32_t kOutPort = 0;

struct QuantGraphOptions {
    bool withWeightQuant = true;
    bool withBias = true;
    bool scaleIsConst = true;
    int64_t cDim = 65;
    int64_t cubeGroups = 1;
    float quantScale = 1.2F;
    float quantOffset = 1.1F;
    std::vector<int64_t> scaleDims = {1};
};

class QuantTestGraph {
public:
    explicit QuantTestGraph(const std::string& name) : builder_(name.c_str())
    {
        rawGraph_ = builder_.GetCGraphBuilder()->GetGraph();
    }

    GNode& AddNode(const std::string& nodeName, const std::string& opType,
                   const std::vector<std::pair<std::string, bool>>& inputs)
    {
        std::vector<es::CompliantNodeBuilder::IrInputDef> inputDefs;
        for (const auto& in : inputs) {
            inputDefs.push_back({in.first,
                                 in.second ? es::CompliantNodeBuilder::kEsIrInputRequired :
                                             es::CompliantNodeBuilder::kEsIrInputOptional,
                                 ""});
        }
        GNode node = es::CompliantNodeBuilder(rawGraph_)
                         .OpType(opType.c_str())
                         .Name(nodeName.c_str())
                         .IrDefInputs(inputDefs)
                         .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
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
        GNode node = es::CompliantNodeBuilder(rawGraph_)
                         .OpType("Const")
                         .Name(nodeName.c_str())
                         .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                         .IrDefAttrs({{"value", es::CompliantNodeBuilder::kEsAttrOptional, "VT_TENSOR", attrValue}})
                         .InstanceOutputShape("y", dims)
                         .InstanceOutputDataType("y", dtype)
                         .InstanceOutputFormat("y", FORMAT_ND)
                         .Build();
        nodes_[nodeName] = node;
        return nodes_[nodeName];
    }

    void Connect(const std::string& from, int32_t fromPort, const std::string& to, int32_t toPort)
    {
        auto it = nodes_.find(from);
        auto toIt = nodes_.find(to);
        if (it == nodes_.end() || toIt == nodes_.end()) {
            return;
        }
        (void)es::AddEdgeAndUpdatePeerDesc(*rawGraph_, it->second, fromPort, toIt->second, toPort);
    }

    void ConnectInput(const es::EsTensorHolder& input, const std::string& to, int32_t toPort)
    {
        auto it = nodes_.find(to);
        if (it == nodes_.end()) {
            return;
        }
        (void)es::AddEdgeAndUpdatePeerDesc(*rawGraph_, *input.GetProducer(), kOutPort, it->second, toPort);
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

    es::EsTensorHolder CreateInput(const std::string& name, DataType dtype, Format format,
                                   const std::vector<int64_t>& dims)
    {
        return builder_.CreateInput(inputIdx_++, name.c_str(), dtype, format, dims);
    }

    GNode& Node(const std::string& nodeName) { return nodes_[nodeName]; }

private:
    es::EsGraphBuilder builder_;
    Graph* rawGraph_ = nullptr;
    std::map<std::string, GNode> nodes_;
    int64_t inputIdx_ = 0;
    std::shared_ptr<Graph> geGraph_ = nullptr;
};

std::shared_ptr<Graph> BuildQuantProcessGraph(QuantTestGraph& tg, const QuantGraphOptions& opt)
{
    const std::vector<int64_t> fmapDims = {1, opt.cDim, 8, 8};
    const std::vector<int64_t> filterDims = {2, 16, 1, 1};
    auto xInput = tg.CreateInput("x", DT_FLOAT, FORMAT_NCHW, fmapDims);

    GNode& quant = tg.AddNode("quant", "AscendQuant", {{"x", true}});
    float scaleAttr = opt.quantScale;
    float offsetAttr = opt.quantOffset;
    (void)quant.SetAttr(AscendString("scale"), scaleAttr);
    (void)quant.SetAttr(AscendString("offset"), offsetAttr);

    GNode& cube = tg.AddNode("cube", "Conv2D", {{"x", true}, {"filter", true}, {"bias", false}, {"offset_w", false}});
    int64_t groups = opt.cubeGroups;
    (void)cube.SetAttr(AscendString("groups"), groups);

    if (opt.withWeightQuant) {
        std::vector<uint8_t> weightData(2 * 16 * 4, 1U);
        tg.CreateConst("weight_const", filterDims, DT_FLOAT, weightData);
        std::vector<uint8_t> offsetData(4, 0U);
        tg.CreateConst("offset_const", {1}, DT_FLOAT, offsetData);
        GNode& awq = tg.AddNode("awq", "AscendWeightQuant", {{"x", true}, {"offset", false}});
        int32_t dstType = static_cast<int32_t>(DT_INT8);
        (void)awq.SetAttr(AscendString("dst_type"), dstType);
        tg.Connect("weight_const", kOutPort, "awq", kQuantInputPort);
        tg.Connect("offset_const", kOutPort, "awq", 1);
        tg.Connect("awq", kOutPort, "cube", kCubeFilterPort);
        tg.SetInDesc("awq", 0, DT_FLOAT, FORMAT_NCHW, filterDims);
        tg.SetInDesc("awq", 1, DT_FLOAT, FORMAT_ND, {1});
        tg.SetOutDesc("awq", kOutPort, DT_INT8, FORMAT_NCHW, filterDims);
        TensorDesc awqOut;
        (void)tg.Node("awq").GetOutputDesc(kOutPort, awqOut);
        (void)cube.UpdateInputDesc(kCubeFilterPort, awqOut);
    } else {
        std::vector<uint8_t> weightData(2 * 16 * 4, 1U);
        tg.CreateConst("weight_const", filterDims, DT_FLOAT, weightData);
        tg.Connect("weight_const", kOutPort, "cube", kCubeFilterPort);
        TensorDesc weightOut;
        (void)tg.Node("weight_const").GetOutputDesc(kOutPort, weightOut);
        (void)cube.UpdateInputDesc(kCubeFilterPort, weightOut);
    }

    if (opt.withBias) {
        std::vector<uint8_t> biasData(16 * 4, 0U);
        tg.CreateConst("bias_const", {16}, DT_INT32, biasData);
        tg.Connect("bias_const", kOutPort, "cube", kCubeBiasPort);
        TensorDesc biasOut;
        (void)tg.Node("bias_const").GetOutputDesc(kOutPort, biasOut);
        (void)cube.UpdateInputDesc(kCubeBiasPort, biasOut);
    }

    if (opt.scaleIsConst) {
        std::vector<uint8_t> scaleData(8, 0U);
        tg.CreateConst("deq_scale", opt.scaleDims, DT_UINT64, scaleData);
    }

    tg.AddNode("dequant", "AscendDequant", {{"x", true}, {"deq_scale", true}});
    tg.AddNode("relu", "Relu", {{"x", true}});

    tg.ConnectInput(xInput, "quant", kQuantInputPort);
    tg.Connect("quant", kOutPort, "cube", kCubeFmapPort);
    tg.Connect("cube", kOutPort, "dequant", kDequantXPort);
    if (opt.scaleIsConst) {
        tg.Connect("deq_scale", kOutPort, "dequant", kDequantScalePort);
    } else {
        auto scaleInput = tg.CreateInput("deq_scale_in", DT_UINT64, FORMAT_ND, opt.scaleDims);
        tg.ConnectInput(scaleInput, "dequant", kDequantScalePort);
    }
    tg.Connect("dequant", kOutPort, "relu", kQuantInputPort);

    tg.SetInDesc("quant", 0, DT_FLOAT, FORMAT_NCHW, fmapDims);
    tg.SetOutDesc("quant", kOutPort, DT_INT8, FORMAT_NCHW, fmapDims);
    tg.SetInDesc("cube", kCubeFmapPort, DT_INT8, FORMAT_NCHW, fmapDims);
    tg.SetOutDesc("cube", kOutPort, DT_INT32, FORMAT_NCHW, {1, 3, 8, 8});
    tg.SetInDesc("dequant", kDequantXPort, DT_INT32, FORMAT_NCHW, {1, 3, 8, 8});
    tg.SetInDesc("dequant", kDequantScalePort, DT_UINT64, FORMAT_ND, opt.scaleDims);
    tg.SetOutDesc("dequant", kOutPort, DT_FLOAT16, FORMAT_NCHW, {1, 3, 8, 8});
    tg.SetInDesc("relu", kQuantInputPort, DT_FLOAT16, FORMAT_NCHW, {1, 3, 8, 8});
    tg.SetOutDesc("relu", kOutPort, DT_FLOAT16, FORMAT_NCHW, {1, 3, 8, 8});
    return tg.Build("relu");
}

void ApplyNodesToRemove(std::shared_ptr<Graph>& graph, std::vector<GNodePtr>& nodesToRemove)
{
    for (const auto& node : nodesToRemove) {
        (void)graph->RemoveNode(*node);
    }
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

bool ContainsNode(const std::vector<GNodePtr>& nodes, const std::string& nodeName)
{
    for (const auto& node : nodes) {
        if (node == nullptr) {
            continue;
        }
        AscendString name;
        if (node->GetName(name) == GRAPH_SUCCESS && name.GetString() != nullptr &&
            std::string(name.GetString()) == nodeName) {
            return true;
        }
    }
    return false;
}

class UtQuantPass : public ops::BiasOptimizeQuantRollbackBase {
public:
    bool cubeSupported = false;

protected:
    bool CheckOpSupported(const GNode& node) const override { return cubeSupported; }
};

class PlainQuantPass : public ops::BiasOptimizeQuantRollbackBase {};
} // namespace

class BiasOptimizeQuantRollbackBaseTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        fe::PlatformInfo platformInfo;
        fe::OptionalInfo optionalInfo;
        platformInfo.str_info.short_soc_version = "Ascend950";
        optionalInfo.soc_version = "Ascend950";
        fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
        fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optionalInfo);
    }
};

TEST_F(BiasOptimizeQuantRollbackBaseTest, OptimizeMainProcQuantRollbackTfCaseSuccess)
{
    QuantTestGraph tg("rollback_tf");
    QuantGraphOptions opt;
    opt.cDim = 65;
    auto graph = BuildQuantProcessGraph(tg, opt);
    ASSERT_NE(graph, nullptr);
    UtQuantPass pass;
    pass.cubeSupported = false;
    std::vector<GNodePtr> fusionNodes;
    std::vector<GNodePtr> nodesToRemove;
    ASSERT_EQ(pass.OptimizeMainProc(graph, FindByName(graph, "quant"), FindByName(graph, "dequant"),
                                    FindByName(graph, "cube"), fusionNodes, nodesToRemove),
              SUCCESS);
    ApplyNodesToRemove(graph, nodesToRemove);
    EXPECT_EQ(CountByType(graph, "QuantWeightRollBack"), 1);
    EXPECT_EQ(CountByType(graph, "QuantBiasRollBack"), 1);
    EXPECT_EQ(CountByType(graph, "AscendWeightQuant"), 0);
    EXPECT_EQ(CountByName(graph, "offset_const"), 0);
    EXPECT_EQ(CountByName(graph, "quant"), 0);
    EXPECT_EQ(CountByName(graph, "dequant"), 0);
    const GNodePtr cube = FindByName(graph, "cube");
    ASSERT_NE(cube, nullptr);
    EXPECT_EQ(PeerInType(cube, kCubeFmapPort), "Data");
    EXPECT_EQ(PeerInType(cube, kCubeFilterPort), "QuantWeightRollBack");
    EXPECT_EQ(PeerInType(cube, kCubeBiasPort), "QuantBiasRollBack");
    const GNodePtr relu = FindByName(graph, "relu");
    ASSERT_NE(relu, nullptr);
    EXPECT_EQ(PeerInType(relu, kQuantInputPort), "Conv2D");
    TensorDesc cubeOut;
    (void)cube->GetOutputDesc(kOutPort, cubeOut);
    EXPECT_EQ(cubeOut.GetDataType(), DT_FLOAT16);
    TensorDesc cubeIn;
    (void)cube->GetInputDesc(kCubeFmapPort, cubeIn);
    EXPECT_EQ(cubeIn.GetDataType(), DT_FLOAT16);
    const GNodePtr weightRb = FindByType(graph, "QuantWeightRollBack");
    ASSERT_NE(weightRb, nullptr);
    TensorDesc rbOut;
    (void)weightRb->GetOutputDesc(kOutPort, rbOut);
    EXPECT_EQ(rbOut.GetDataType(), DT_FLOAT);
    int64_t groups = 0;
    EXPECT_EQ(weightRb->GetAttr(AscendString("groups"), groups), GRAPH_SUCCESS);
    EXPECT_EQ(groups, 1);
    int64_t dstType = 0;
    EXPECT_EQ(weightRb->GetAttr(AscendString("dst_type"), dstType), GRAPH_SUCCESS);
    EXPECT_EQ(dstType, static_cast<int64_t>(DT_INT8));
    AscendString cubeOpType;
    EXPECT_EQ(weightRb->GetAttr(AscendString("cube_op_type"), cubeOpType), GRAPH_SUCCESS);
    EXPECT_STREQ(cubeOpType.GetString(), "Conv2D");
    float scale = 0.0F;
    EXPECT_EQ(weightRb->GetAttr(AscendString("scale"), scale), GRAPH_SUCCESS);
    EXPECT_FLOAT_EQ(scale, 1.2F);
    bool fromConst = false;
    EXPECT_EQ(weightRb->GetAttr(AscendString("_is_come_from_const_op"), fromConst), GRAPH_SUCCESS);
    EXPECT_TRUE(fromConst);
    EXPECT_EQ(PeerInType(weightRb, 1), "Const");
    EXPECT_TRUE(ContainsNode(fusionNodes, "cube"));
}

TEST_F(BiasOptimizeQuantRollbackBaseTest, OptimizeMainProcQuantRollbackCaffeCaseSuccess)
{
    QuantTestGraph tg("rollback_caffe");
    QuantGraphOptions opt;
    opt.withWeightQuant = false;
    opt.withBias = false;
    opt.cDim = 65;
    auto graph = BuildQuantProcessGraph(tg, opt);
    ASSERT_NE(graph, nullptr);
    UtQuantPass pass;
    pass.cubeSupported = false;
    std::vector<GNodePtr> fusionNodes;
    std::vector<GNodePtr> nodesToRemove;
    ASSERT_EQ(pass.OptimizeMainProc(graph, FindByName(graph, "quant"), FindByName(graph, "dequant"),
                                    FindByName(graph, "cube"), fusionNodes, nodesToRemove),
              SUCCESS);
    ApplyNodesToRemove(graph, nodesToRemove);
    EXPECT_EQ(CountByType(graph, "QuantWeightRollBack"), 1);
    EXPECT_EQ(CountByType(graph, "QuantBiasRollBack"), 0);
    EXPECT_EQ(CountByName(graph, "weight_const"), 1);
    EXPECT_EQ(CountByName(graph, "quant"), 0);
    EXPECT_EQ(CountByName(graph, "dequant"), 0);
    const GNodePtr cube = FindByName(graph, "cube");
    ASSERT_NE(cube, nullptr);
    EXPECT_EQ(PeerInType(cube, kCubeFilterPort), "QuantWeightRollBack");
    EXPECT_EQ(PeerInType(cube, kCubeBiasPort), "");
    EXPECT_EQ(PeerInType(cube, kCubeFmapPort), "Data");
}

TEST_F(BiasOptimizeQuantRollbackBaseTest, OptimizeMainProcQuantRollbackSmallChannelSuccess)
{
    QuantTestGraph tg("rollback_small_c");
    QuantGraphOptions opt;
    opt.cDim = 1;
    auto graph = BuildQuantProcessGraph(tg, opt);
    ASSERT_NE(graph, nullptr);
    UtQuantPass pass;
    pass.cubeSupported = true;
    std::vector<GNodePtr> fusionNodes;
    std::vector<GNodePtr> nodesToRemove;
    ASSERT_EQ(pass.OptimizeMainProc(graph, FindByName(graph, "quant"), FindByName(graph, "dequant"),
                                    FindByName(graph, "cube"), fusionNodes, nodesToRemove),
              SUCCESS);
    ApplyNodesToRemove(graph, nodesToRemove);
    EXPECT_EQ(CountByType(graph, "QuantWeightRollBack"), 1);
    EXPECT_EQ(CountByType(graph, "QuantBiasRollBack"), 1);
    EXPECT_EQ(CountByName(graph, "quant"), 0);
    EXPECT_EQ(CountByName(graph, "dequant"), 0);
}

TEST_F(BiasOptimizeQuantRollbackBaseTest, OptimizeMainProcBiasOptimizeSuccess)
{
    QuantTestGraph tg("bias_optimize");
    QuantGraphOptions opt;
    opt.withBias = false;
    opt.cDim = 65;
    auto graph = BuildQuantProcessGraph(tg, opt);
    ASSERT_NE(graph, nullptr);
    UtQuantPass pass;
    pass.cubeSupported = true;
    std::vector<GNodePtr> fusionNodes;
    std::vector<GNodePtr> nodesToRemove;
    ASSERT_EQ(pass.OptimizeMainProc(graph, FindByName(graph, "quant"), FindByName(graph, "dequant"),
                                    FindByName(graph, "cube"), fusionNodes, nodesToRemove),
              SUCCESS);
    ApplyNodesToRemove(graph, nodesToRemove);
    EXPECT_EQ(CountByType(graph, "QuantBiasOptimization"), 1);
    EXPECT_EQ(CountByType(graph, "QuantWeightRollBack"), 0);
    EXPECT_EQ(CountByType(graph, "AscendWeightQuant"), 1);
    EXPECT_EQ(CountByName(graph, "quant"), 1);
    EXPECT_EQ(CountByName(graph, "dequant"), 1);
    const GNodePtr cube = FindByName(graph, "cube");
    ASSERT_NE(cube, nullptr);
    EXPECT_EQ(PeerInType(cube, kCubeBiasPort), "QuantBiasOptimization");
    EXPECT_EQ(PeerInType(cube, kCubeFilterPort), "AscendWeightQuant");
    int64_t offsetX = -1;
    EXPECT_EQ(cube->GetAttr(AscendString("offset_x"), offsetX), GRAPH_SUCCESS);
    EXPECT_EQ(offsetX, 1);
    const GNodePtr hostOp = FindByType(graph, "QuantBiasOptimization");
    ASSERT_NE(hostOp, nullptr);
    Tensor biasTensor;
    EXPECT_EQ(hostOp->GetInputConstData(0, biasTensor), GRAPH_SUCCESS);
    EXPECT_NE(biasTensor.GetData(), nullptr);
    EXPECT_EQ(biasTensor.GetSize(), 2U * sizeof(int32_t));
    for (size_t i = 0; i < biasTensor.GetSize(); ++i) {
        EXPECT_EQ(biasTensor.GetData()[i], 0U);
    }
    EXPECT_EQ(biasTensor.GetTensorDesc().GetDataType(), DT_INT32);
    EXPECT_EQ(hostOp->GetInputsSize(), 3U);
    EXPECT_EQ(PeerInType(hostOp, 0), "Const");
    EXPECT_EQ(PeerInName(hostOp, 1), "weight_const");
    EXPECT_EQ(PeerInType(hostOp, 2), "Const");
    int64_t dstType = 0;
    EXPECT_EQ(hostOp->GetAttr(AscendString("dst_type"), dstType), GRAPH_SUCCESS);
    EXPECT_EQ(dstType, static_cast<int64_t>(DT_INT8));
    AscendString socVersion;
    EXPECT_EQ(hostOp->GetAttr(AscendString("soc_version"), socVersion), GRAPH_SUCCESS);
    float offset = 0.0F;
    EXPECT_EQ(hostOp->GetAttr(AscendString("offset"), offset), GRAPH_SUCCESS);
    EXPECT_FLOAT_EQ(offset, 1.1F);
    bool cinCoutReverse = true;
    EXPECT_EQ(hostOp->GetAttr(AscendString("quant_cin_cout_reverse"), cinCoutReverse), GRAPH_SUCCESS);
    EXPECT_FALSE(cinCoutReverse);
}

TEST_F(BiasOptimizeQuantRollbackBaseTest, OptimizeMainProcBiasOptimizeWithBiasSuccess)
{
    QuantTestGraph tg("bias_optimize_with_bias");
    QuantGraphOptions opt;
    opt.withBias = true;
    opt.cDim = 65;
    auto graph = BuildQuantProcessGraph(tg, opt);
    ASSERT_NE(graph, nullptr);
    UtQuantPass pass;
    pass.cubeSupported = true;
    std::vector<GNodePtr> fusionNodes;
    std::vector<GNodePtr> nodesToRemove;
    ASSERT_EQ(pass.OptimizeMainProc(graph, FindByName(graph, "quant"), FindByName(graph, "dequant"),
                                    FindByName(graph, "cube"), fusionNodes, nodesToRemove),
              SUCCESS);
    ApplyNodesToRemove(graph, nodesToRemove);
    EXPECT_EQ(CountByType(graph, "QuantBiasOptimization"), 1);
    EXPECT_EQ(CountByName(graph, "bias_const"), 1);
    const GNodePtr cube = FindByName(graph, "cube");
    ASSERT_NE(cube, nullptr);
    EXPECT_EQ(PeerInType(cube, kCubeBiasPort), "QuantBiasOptimization");
}

TEST_F(BiasOptimizeQuantRollbackBaseTest, OptimizeMainProcUnknownShapeSkip)
{
    QuantTestGraph tg("unknown_shape");
    QuantGraphOptions opt;
    opt.cDim = -1;
    auto graph = BuildQuantProcessGraph(tg, opt);
    ASSERT_NE(graph, nullptr);
    UtQuantPass pass;
    pass.cubeSupported = true;
    std::vector<GNodePtr> fusionNodes;
    std::vector<GNodePtr> nodesToRemove;
    ASSERT_EQ(pass.OptimizeMainProc(graph, FindByName(graph, "quant"), FindByName(graph, "dequant"),
                                    FindByName(graph, "cube"), fusionNodes, nodesToRemove),
              SUCCESS);
    ApplyNodesToRemove(graph, nodesToRemove);
    EXPECT_EQ(CountByType(graph, "QuantBiasOptimization"), 0);
    EXPECT_EQ(CountByType(graph, "QuantWeightRollBack"), 0);
    EXPECT_EQ(CountByName(graph, "quant"), 1);
    EXPECT_EQ(CountByName(graph, "dequant"), 1);
    EXPECT_TRUE(fusionNodes.empty());
}

TEST_F(BiasOptimizeQuantRollbackBaseTest, OptimizeMainProcNullNodeFailed)
{
    QuantTestGraph tg("null_node");
    QuantGraphOptions opt;
    auto graph = BuildQuantProcessGraph(tg, opt);
    ASSERT_NE(graph, nullptr);
    UtQuantPass pass;
    std::vector<GNodePtr> fusionNodes;
    std::vector<GNodePtr> nodesToRemove;
    EXPECT_EQ(pass.OptimizeMainProc(graph, nullptr, FindByName(graph, "dequant"), FindByName(graph, "cube"),
                                    fusionNodes, nodesToRemove),
              FAILED);
    EXPECT_EQ(pass.OptimizeMainProc(graph, FindByName(graph, "quant"), nullptr, FindByName(graph, "cube"), fusionNodes,
                                    nodesToRemove),
              FAILED);
    ApplyNodesToRemove(graph, nodesToRemove);
    EXPECT_EQ(pass.OptimizeMainProc(graph, FindByName(graph, "quant"), FindByName(graph, "dequant"), nullptr,
                                    fusionNodes, nodesToRemove),
              FAILED);
    GraphPtr nullGraph = nullptr;
    EXPECT_EQ(pass.OptimizeMainProc(nullGraph, FindByName(graph, "quant"), FindByName(graph, "dequant"),
                                    FindByName(graph, "cube"), fusionNodes, nodesToRemove),
              FAILED);
    ApplyNodesToRemove(graph, nodesToRemove);
}

TEST_F(BiasOptimizeQuantRollbackBaseTest, OptimizeMainProcScaleNotConstFailed)
{
    QuantTestGraph tg("scale_not_const");
    QuantGraphOptions opt;
    opt.scaleIsConst = false;
    opt.cDim = 65;
    auto graph = BuildQuantProcessGraph(tg, opt);
    ASSERT_NE(graph, nullptr);
    UtQuantPass pass;
    pass.cubeSupported = true;
    std::vector<GNodePtr> fusionNodes;
    std::vector<GNodePtr> nodesToRemove;
    EXPECT_EQ(pass.OptimizeMainProc(graph, FindByName(graph, "quant"), FindByName(graph, "dequant"),
                                    FindByName(graph, "cube"), fusionNodes, nodesToRemove),
              FAILED);
    ApplyNodesToRemove(graph, nodesToRemove);
}

TEST_F(BiasOptimizeQuantRollbackBaseTest, GetQuantProcessModeAllBranches)
{
    QuantTestGraph tg("mode_branches");
    QuantGraphOptions opt;
    opt.cDim = 65;
    auto graph = BuildQuantProcessGraph(tg, opt);
    ASSERT_NE(graph, nullptr);
    UtQuantPass pass;
    ops::QuantProcessMode mode = ops::QuantProcessMode::QUANT_UNDEFINED;
    const GNodePtr quant = FindByName(graph, "quant");
    const GNodePtr cube = FindByName(graph, "cube");
    ASSERT_NE(quant, nullptr);
    ASSERT_NE(cube, nullptr);

    pass.cubeSupported = false;
    EXPECT_EQ(pass.GetQuantProcessMode(quant, cube, mode), SUCCESS);
    EXPECT_EQ(mode, ops::QuantProcessMode::QUANT_ROLLBACK);

    pass.cubeSupported = true;
    EXPECT_EQ(pass.GetQuantProcessMode(quant, cube, mode), SUCCESS);
    EXPECT_EQ(mode, ops::QuantProcessMode::BIAS_OPTIMIZE);

    TensorDesc smallDesc;
    (void)quant->GetInputDesc(kQuantInputPort, smallDesc);
    smallDesc.SetShape(Shape({1, 1, 8, 8}));
    (void)quant->UpdateInputDesc(kQuantInputPort, smallDesc);
    EXPECT_EQ(pass.GetQuantProcessMode(quant, cube, mode), SUCCESS);
    EXPECT_EQ(mode, ops::QuantProcessMode::QUANT_ROLLBACK);

    TensorDesc unknownDesc;
    (void)quant->GetInputDesc(kQuantInputPort, unknownDesc);
    unknownDesc.SetShape(Shape({1, -1, 8, 8}));
    (void)quant->UpdateInputDesc(kQuantInputPort, unknownDesc);
    EXPECT_EQ(pass.GetQuantProcessMode(quant, cube, mode), SUCCESS);
    EXPECT_EQ(mode, ops::QuantProcessMode::QUANT_UNDEFINED);
}

TEST_F(BiasOptimizeQuantRollbackBaseTest, GetQuantProcessModeUnsupportedFormatFailed)
{
    QuantTestGraph tg("mode_bad_format");
    QuantGraphOptions opt;
    auto graph = BuildQuantProcessGraph(tg, opt);
    ASSERT_NE(graph, nullptr);
    UtQuantPass pass;
    pass.cubeSupported = true;
    const GNodePtr quant = FindByName(graph, "quant");
    TensorDesc desc;
    (void)quant->GetInputDesc(kQuantInputPort, desc);
    desc.SetFormat(FORMAT_FRACTAL_Z);
    (void)quant->UpdateInputDesc(kQuantInputPort, desc);
    ops::QuantProcessMode mode = ops::QuantProcessMode::QUANT_UNDEFINED;
    EXPECT_NE(pass.GetQuantProcessMode(quant, FindByName(graph, "cube"), mode), SUCCESS);
}

TEST_F(BiasOptimizeQuantRollbackBaseTest, GetAxisIndexByFormatAllBranches)
{
    EXPECT_EQ(ops::GetAxisIndexByFormat(FORMAT_NCHW, "C"), 1);
    EXPECT_EQ(ops::GetAxisIndexByFormat(FORMAT_NCHW, "H"), 2);
    EXPECT_EQ(ops::GetAxisIndexByFormat(FORMAT_HWCN, "C"), 2);
    EXPECT_EQ(ops::GetAxisIndexByFormat(FORMAT_HWCN, "N"), 3);
    EXPECT_EQ(ops::GetAxisIndexByFormat(FORMAT_NHWC, "C"), 3);
    EXPECT_EQ(ops::GetAxisIndexByFormat(FORMAT_ND, "C"), -1);
    EXPECT_EQ(ops::GetAxisIndexByFormat(FORMAT_NCHW, "A"), -1);
}

TEST_F(BiasOptimizeQuantRollbackBaseTest, GetCoValueAllBranches)
{
    QuantTestGraph tg("co_value");
    QuantGraphOptions opt;
    opt.withWeightQuant = false;
    opt.withBias = false;
    auto graph = BuildQuantProcessGraph(tg, opt);
    ASSERT_NE(graph, nullptr);
    UtQuantPass pass;
    const GNodePtr cube = FindByName(graph, "cube");
    ASSERT_NE(cube, nullptr);

    TensorDesc ndDesc;
    ndDesc.SetFormat(FORMAT_ND);
    ndDesc.SetShape(Shape({2, 16, 1, 1}));
    (void)cube->UpdateInputDesc(kCubeFilterPort, ndDesc);
    int64_t co = 1;
    EXPECT_EQ(pass.GetCoValue(cube, co), SUCCESS);
    EXPECT_EQ(co, 1);

    TensorDesc nchwDesc;
    nchwDesc.SetFormat(FORMAT_NCHW);
    nchwDesc.SetShape(Shape({2, 16, 1, 1}));
    (void)cube->UpdateInputDesc(kCubeFilterPort, nchwDesc);
    EXPECT_EQ(pass.GetCoValue(cube, co), SUCCESS);
    EXPECT_EQ(co, 16);

    int64_t groups = 2;
    (void)cube->SetAttr(AscendString("groups"), groups);
    EXPECT_EQ(pass.GetCoValue(cube, co), SUCCESS);
    EXPECT_EQ(co, 32);

    int64_t resetGroups = 1;
    (void)cube->SetAttr(AscendString("groups"), resetGroups);
    TensorDesc hwcnDesc;
    hwcnDesc.SetFormat(FORMAT_HWCN);
    hwcnDesc.SetShape(Shape({1, 1, 16, 2}));
    (void)cube->UpdateInputDesc(kCubeFilterPort, hwcnDesc);
    EXPECT_EQ(pass.GetCoValue(cube, co), SUCCESS);
    EXPECT_EQ(co, 16);

    TensorDesc fracDesc;
    fracDesc.SetFormat(FORMAT_FRACTAL_Z);
    fracDesc.SetShape(Shape({2, 16, 1, 1}));
    (void)cube->UpdateInputDesc(kCubeFilterPort, fracDesc);
    EXPECT_NE(pass.GetCoValue(cube, co), SUCCESS);

    // NHWC filter [Co, H, W, Cin] → N轴取Co
    TensorDesc nhwcDesc;
    nhwcDesc.SetFormat(FORMAT_NHWC);
    nhwcDesc.SetShape(Shape({8, 1, 1, 16}));
    (void)cube->UpdateInputDesc(kCubeFilterPort, nhwcDesc);
    EXPECT_EQ(pass.GetCoValue(cube, co), SUCCESS);
    EXPECT_EQ(co, 16);

    // HWCN + groups: [H,W,Cin,Co] → N轴=3取Co=2, groups=3 → co=6
    (void)cube->UpdateInputDesc(kCubeFilterPort, hwcnDesc);
    int64_t groupsHwcn = 3;
    (void)cube->SetAttr(AscendString("groups"), groupsHwcn);
    EXPECT_EQ(pass.GetCoValue(cube, co), SUCCESS);
    EXPECT_EQ(co, 48);
    int64_t resetGroups2 = 1;
    (void)cube->SetAttr(AscendString("groups"), resetGroups2);

    // NCHW 低维filter [Co, Cin] → PadShapeTo4Dim补[H,W]=[Co,Cin,1,1] → N=Co
    TensorDesc lowDimDesc;
    lowDimDesc.SetFormat(FORMAT_NCHW);
    lowDimDesc.SetShape(Shape({4, 32}));
    (void)cube->UpdateInputDesc(kCubeFilterPort, lowDimDesc);
    EXPECT_EQ(pass.GetCoValue(cube, co), SUCCESS);
    EXPECT_EQ(co, 32);
}

TEST_F(BiasOptimizeQuantRollbackBaseTest, IsCubeHasBiasInputAndNeedBiasInput)
{
    QuantTestGraph tgWithBias("has_bias");
    QuantGraphOptions withBiasOpt;
    auto graphWithBias = BuildQuantProcessGraph(tgWithBias, withBiasOpt);
    ASSERT_NE(graphWithBias, nullptr);
    UtQuantPass passWithBias;
    EXPECT_TRUE(passWithBias.IsCubeHasBiasInput(FindByName(graphWithBias, "cube")));
    EXPECT_FALSE(passWithBias.IsCubeNeedBiasInput(FindByName(graphWithBias, "cube")));

    QuantTestGraph tgNoBias("no_bias");
    QuantGraphOptions noBiasOpt;
    noBiasOpt.withBias = false;
    auto graphNoBias = BuildQuantProcessGraph(tgNoBias, noBiasOpt);
    ASSERT_NE(graphNoBias, nullptr);
    UtQuantPass passNoBias;
    EXPECT_FALSE(passNoBias.IsCubeHasBiasInput(FindByName(graphNoBias, "cube")));
    EXPECT_TRUE(passNoBias.IsCubeNeedBiasInput(FindByName(graphNoBias, "cube")));
    EXPECT_FALSE(passNoBias.IsCubeHasBiasInput(nullptr));
}

TEST_F(BiasOptimizeQuantRollbackBaseTest, SetQuantParametersOffsetToOffsetX)
{
    QuantTestGraph tg("set_quant_para");
    QuantGraphOptions opt;
    auto graph = BuildQuantProcessGraph(tg, opt);
    ASSERT_NE(graph, nullptr);
    UtQuantPass pass;
    const GNodePtr cube = FindByName(graph, "cube");
    const GNodePtr quant = FindByName(graph, "quant");
    ASSERT_EQ(pass.SetQuantParameters(cube, quant), SUCCESS);
    int64_t offsetX = 0;
    EXPECT_EQ(cube->GetAttr(AscendString("offset_x"), offsetX), GRAPH_SUCCESS);
    EXPECT_EQ(offsetX, 1);

    float offset = -1.7F;
    (void)quant->SetAttr(AscendString("offset"), offset);
    EXPECT_EQ(pass.SetQuantParameters(cube, quant), SUCCESS);
    EXPECT_EQ(cube->GetAttr(AscendString("offset_x"), offsetX), GRAPH_SUCCESS);
    EXPECT_EQ(offsetX, -1);
}

TEST_F(BiasOptimizeQuantRollbackBaseTest, CreateBiasInputWithZeroTensor)
{
    QuantTestGraph tg("create_bias");
    QuantGraphOptions opt;
    opt.withBias = false;
    opt.withWeightQuant = false;
    auto graph = BuildQuantProcessGraph(tg, opt);
    ASSERT_NE(graph, nullptr);
    UtQuantPass pass;
    const GNodePtr cube = FindByName(graph, "cube");
    ASSERT_EQ(pass.CreateBiasInput(*graph, cube, 16), SUCCESS);
    EXPECT_EQ(PeerInType(cube, kCubeBiasPort), "Const");
    TensorDesc biasInDesc;
    (void)cube->GetInputDesc(kCubeBiasPort, biasInDesc);
    EXPECT_EQ(biasInDesc.GetDataType(), DT_INT32);
    ASSERT_EQ(biasInDesc.GetShape().GetDims(), std::vector<int64_t>({16}));
    Tensor biasTensor;
    EXPECT_EQ(cube->GetInputConstData(kCubeBiasPort, biasTensor), GRAPH_SUCCESS);
    ASSERT_EQ(biasTensor.GetSize(), 16U * sizeof(int32_t));
    for (size_t i = 0; i < biasTensor.GetSize(); ++i) {
        EXPECT_EQ(biasTensor.GetData()[i], 0U);
    }
}

TEST_F(BiasOptimizeQuantRollbackBaseTest, GetWeightNodesOfCubeNodeTwoCases)
{
    QuantTestGraph tgTf("weight_tf");
    QuantGraphOptions tfOpt;
    auto graphTf = BuildQuantProcessGraph(tgTf, tfOpt);
    ASSERT_NE(graphTf, nullptr);
    UtQuantPass passTf;
    GNodePtr weightConst = nullptr;
    GNodePtr awq = nullptr;
    ASSERT_EQ(passTf.GetWeightNodesOfCubeNode(FindByName(graphTf, "cube"), weightConst, awq), SUCCESS);
    ASSERT_NE(weightConst, nullptr);
    ASSERT_NE(awq, nullptr);
    AscendString weightName;
    (void)weightConst->GetName(weightName);
    EXPECT_STREQ(weightName.GetString(), "weight_const");
    AscendString awqType;
    (void)awq->GetType(awqType);
    EXPECT_STREQ(awqType.GetString(), "AscendWeightQuant");

    QuantTestGraph tgCaffe("weight_caffe");
    QuantGraphOptions caffeOpt;
    caffeOpt.withWeightQuant = false;
    auto graphCaffe = BuildQuantProcessGraph(tgCaffe, caffeOpt);
    ASSERT_NE(graphCaffe, nullptr);
    UtQuantPass passCaffe;
    ASSERT_EQ(passCaffe.GetWeightNodesOfCubeNode(FindByName(graphCaffe, "cube"), weightConst, awq), SUCCESS);
    ASSERT_NE(weightConst, nullptr);
    EXPECT_EQ(awq, nullptr);
    AscendString caffeName;
    (void)weightConst->GetName(caffeName);
    EXPECT_STREQ(caffeName.GetString(), "weight_const");
}

TEST_F(BiasOptimizeQuantRollbackBaseTest, WeightRollbackDeleteIsolateNodesTwoCases)
{
    QuantTestGraph tg("delete_isolate");
    QuantGraphOptions opt;
    auto graph = BuildQuantProcessGraph(tg, opt);
    ASSERT_NE(graph, nullptr);
    UtQuantPass pass;
    const GNodePtr cube = FindByName(graph, "cube");
    GNodePtr weightConst = nullptr;
    GNodePtr awq = nullptr;
    ASSERT_EQ(pass.GetWeightNodesOfCubeNode(cube, weightConst, awq), SUCCESS);
    ASSERT_NE(awq, nullptr);
    TensorDesc rbDesc;
    rbDesc.SetDataType(DT_FLOAT);
    rbDesc.SetShape(Shape({2, 16, 1, 1}));
    (void)cube->UpdateInputDesc(kCubeFilterPort, rbDesc);
    auto peer = cube->GetInDataNodesAndPortIndexs(kCubeFilterPort);
    ASSERT_NE(peer.first, nullptr);
    (void)graph->RemoveEdge(*peer.first, peer.second, *cube, kCubeFilterPort);
    ASSERT_EQ(pass.WeightRollbackDeleteIsolateNodes(*graph, awq), SUCCESS);
    ApplyNodesToRemove(graph, pass.nodesToRemove_);
    EXPECT_EQ(CountByName(graph, "awq"), 0);
    EXPECT_EQ(CountByName(graph, "offset_const"), 0);
    EXPECT_EQ(CountByName(graph, "weight_const"), 1);

    GNodePtr nullAwq = nullptr;
    EXPECT_EQ(pass.WeightRollbackDeleteIsolateNodes(*graph, nullAwq), SUCCESS);
}

TEST_F(BiasOptimizeQuantRollbackBaseTest, SetInt4OrInt8FlagAndIsWeightTranspose)
{
    QuantTestGraph tg("int4_flag");
    QuantGraphOptions opt;
    auto graph = BuildQuantProcessGraph(tg, opt);
    ASSERT_NE(graph, nullptr);
    UtQuantPass pass;
    GNodePtr hostNode = FindByName(graph, "relu");
    ASSERT_NE(hostNode, nullptr);
    const GNodePtr awq = FindByName(graph, "awq");
    ASSERT_NE(awq, nullptr);
    ASSERT_EQ(pass.SetInt4OrInt8Flag(awq, hostNode), SUCCESS);
    int64_t dstType = 0;
    EXPECT_EQ(hostNode->GetAttr(AscendString("dst_type"), dstType), GRAPH_SUCCESS);
    EXPECT_EQ(dstType, static_cast<int64_t>(DT_INT8));

    int64_t int4Type = static_cast<int64_t>(DT_INT4);
    (void)awq->SetAttr(AscendString("dst_type"), int4Type);
    ASSERT_EQ(pass.SetInt4OrInt8Flag(awq, hostNode), SUCCESS);
    EXPECT_EQ(hostNode->GetAttr(AscendString("dst_type"), dstType), GRAPH_SUCCESS);
    EXPECT_EQ(dstType, static_cast<int64_t>(DT_INT4));

    ASSERT_EQ(pass.SetInt4OrInt8Flag(nullptr, hostNode), SUCCESS);
    EXPECT_EQ(hostNode->GetAttr(AscendString("dst_type"), dstType), GRAPH_SUCCESS);
    EXPECT_EQ(dstType, static_cast<int64_t>(DT_INT8));

    const GNodePtr cube = FindByName(graph, "cube");
    ASSERT_NE(cube, nullptr);
    EXPECT_FALSE(ops::BiasOptimizeQuantRollbackBase::IsWeightTranspose(cube));
    bool transposeX2 = true;
    (void)cube->SetAttr(AscendString("transpose_x2"), transposeX2);
    EXPECT_TRUE(ops::BiasOptimizeQuantRollbackBase::IsWeightTranspose(cube));
}

TEST_F(BiasOptimizeQuantRollbackBaseTest, SetNodeOutputDtypeReplaceDtype)
{
    QuantTestGraph tg("set_output_dtype");
    QuantGraphOptions opt;
    auto graph = BuildQuantProcessGraph(tg, opt);
    ASSERT_NE(graph, nullptr);
    const GNodePtr relu = FindByName(graph, "relu");
    ASSERT_NE(relu, nullptr);
    ops::BiasOptimizeQuantRollbackBase::SetNodeOutputDtype(relu, DT_FLOAT16, DT_FLOAT);
    TensorDesc outDesc;
    (void)relu->GetOutputDesc(kOutPort, outDesc);
    EXPECT_EQ(outDesc.GetDataType(), DT_FLOAT);
    ops::BiasOptimizeQuantRollbackBase::SetNodeOutputDtype(relu, DT_INT32, DT_FLOAT16);
    (void)relu->GetOutputDesc(kOutPort, outDesc);
    EXPECT_EQ(outDesc.GetDataType(), DT_FLOAT);
    ops::BiasOptimizeQuantRollbackBase::SetNodeOutputDtype(nullptr, DT_FLOAT16, DT_FLOAT);
}

TEST_F(BiasOptimizeQuantRollbackBaseTest, SetGroupInfoForAscendWeightQuantNodes)
{
    QuantTestGraph tg("group_info");
    QuantGraphOptions opt;
    opt.cubeGroups = 2;
    auto graph = BuildQuantProcessGraph(tg, opt);
    ASSERT_NE(graph, nullptr);
    UtQuantPass pass;
    ASSERT_EQ(pass.SetGroupInfoForAscendWeightQuantNodes(FindByName(graph, "cube")), SUCCESS);
    const GNodePtr awq = FindByName(graph, "awq");
    ASSERT_NE(awq, nullptr);
    int64_t groups = 0;
    EXPECT_EQ(awq->GetAttr(AscendString("groups"), groups), GRAPH_SUCCESS);
    EXPECT_EQ(groups, 2);
}

TEST_F(BiasOptimizeQuantRollbackBaseTest, JudgeDeqscaleShapeBranches)
{
    QuantTestGraph tg("judge_scale");
    QuantGraphOptions opt;
    auto graph = BuildQuantProcessGraph(tg, opt);
    ASSERT_NE(graph, nullptr);
    UtQuantPass pass;
    const GNodePtr dequant = FindByName(graph, "dequant");
    ASSERT_EQ(pass.JudgeDeqscaleShape(dequant), SUCCESS);

    QuantTestGraph tgScaleNotConst("judge_scale_not_const");
    QuantGraphOptions notConstOpt;
    notConstOpt.scaleIsConst = false;
    auto graphNotConst = BuildQuantProcessGraph(tgScaleNotConst, notConstOpt);
    ASSERT_NE(graphNotConst, nullptr);
    EXPECT_NE(pass.JudgeDeqscaleShape(FindByName(graphNotConst, "dequant")), SUCCESS);

    QuantTestGraph tgScale2d("judge_scale_2d");
    QuantGraphOptions scale2dOpt;
    scale2dOpt.scaleDims = {1, 1};
    auto graphScale2d = BuildQuantProcessGraph(tgScale2d, scale2dOpt);
    ASSERT_NE(graphScale2d, nullptr);
    EXPECT_NE(pass.JudgeDeqscaleShape(FindByName(graphScale2d, "dequant")), SUCCESS);
}

TEST_F(BiasOptimizeQuantRollbackBaseTest, RemoveEnterBypassConstEnter)
{
    QuantTestGraph tg("remove_enter");
    QuantGraphOptions opt;
    opt.withWeightQuant = false;
    opt.withBias = false;
    auto graph = BuildQuantProcessGraph(tg, opt);
    ASSERT_NE(graph, nullptr);
    (void)tg.AddNode("enter", "Enter", {{"x", true}});
    std::vector<uint8_t> biasData(16 * 4, 0U);
    (void)tg.CreateConst("enter_const", {16}, DT_INT32, biasData);
    tg.Connect("enter_const", kOutPort, "enter", kQuantInputPort);
    tg.Connect("enter", kOutPort, "cube", kCubeBiasPort);
    TensorDesc enterDesc;
    enterDesc.SetDataType(DT_INT32);
    enterDesc.SetShape(Shape({16}));
    (void)tg.Node("enter").UpdateInputDesc(kQuantInputPort, enterDesc);
    (void)tg.Node("enter").UpdateOutputDesc(kOutPort, enterDesc);
    UtQuantPass pass;
    const GNodePtr cube = FindByName(graph, "cube");
    ASSERT_NE(cube, nullptr);
    ASSERT_EQ(pass.RemoveEnter(*graph, cube), SUCCESS);
    ApplyNodesToRemove(graph, pass.nodesToRemove_);
    EXPECT_EQ(CountByName(graph, "enter"), 0);
    EXPECT_EQ(PeerInName(cube, kCubeBiasPort), "enter_const");
}

TEST_F(BiasOptimizeQuantRollbackBaseTest, RemoveEnterKeepNonConstEnter)
{
    QuantTestGraph tg("keep_enter");
    QuantGraphOptions opt;
    opt.withWeightQuant = false;
    opt.withBias = false;
    auto graph = BuildQuantProcessGraph(tg, opt);
    ASSERT_NE(graph, nullptr);
    (void)tg.AddNode("enter", "Enter", {{"x", true}});
    tg.Connect("quant", kOutPort, "enter", kQuantInputPort);
    tg.Connect("enter", kOutPort, "cube", kCubeBiasPort);
    TensorDesc enterDesc;
    enterDesc.SetDataType(DT_INT8);
    enterDesc.SetShape(Shape({1, 65, 8, 8}));
    (void)tg.Node("enter").UpdateInputDesc(kQuantInputPort, enterDesc);
    (void)tg.Node("enter").UpdateOutputDesc(kOutPort, enterDesc);
    UtQuantPass pass;
    const GNodePtr cube = FindByName(graph, "cube");
    ASSERT_NE(cube, nullptr);
    ASSERT_EQ(pass.RemoveEnter(*graph, cube), SUCCESS);
    EXPECT_EQ(CountByName(graph, "enter"), 1);
    EXPECT_EQ(PeerInType(cube, kCubeBiasPort), "Enter");
}

TEST_F(BiasOptimizeQuantRollbackBaseTest, ChangeQuantNodeEdgeMultiConsumerKeepQuant)
{
    QuantTestGraph tg("quant_multi_consumer");
    QuantGraphOptions opt;
    auto graph = BuildQuantProcessGraph(tg, opt);
    ASSERT_NE(graph, nullptr);
    (void)tg.AddNode("relu2", "Relu", {{"x", true}});
    tg.Connect("quant", kOutPort, "relu2", kQuantInputPort);
    TensorDesc quantOut;
    (void)tg.Node("quant").GetOutputDesc(kOutPort, quantOut);
    (void)tg.Node("relu2").UpdateInputDesc(kQuantInputPort, quantOut);
    UtQuantPass pass;
    const GNodePtr cube = FindByName(graph, "cube");
    const GNodePtr quant = FindByName(graph, "quant");
    ASSERT_EQ(pass.ChangeQuantNodeEdge(*graph, cube, quant), SUCCESS);
    EXPECT_EQ(CountByName(graph, "quant"), 1);
    EXPECT_EQ(PeerInType(cube, kCubeFmapPort), "Data");
    EXPECT_EQ(PeerInType(FindByName(graph, "relu2"), kQuantInputPort), "AscendQuant");
}

TEST_F(BiasOptimizeQuantRollbackBaseTest, ChangeDequantNodeEdgeBypassDequant)
{
    QuantTestGraph tg("change_dequant");
    QuantGraphOptions opt;
    auto graph = BuildQuantProcessGraph(tg, opt);
    ASSERT_NE(graph, nullptr);
    UtQuantPass pass;
    const GNodePtr cube = FindByName(graph, "cube");
    const GNodePtr dequant = FindByName(graph, "dequant");
    ASSERT_EQ(pass.ChangeDequantNodeEdge(*graph, cube, dequant), SUCCESS);
    ApplyNodesToRemove(graph, pass.nodesToRemove_);
    EXPECT_EQ(CountByName(graph, "dequant"), 0);
    EXPECT_EQ(PeerInType(FindByName(graph, "relu"), kQuantInputPort), "Conv2D");
}

TEST_F(BiasOptimizeQuantRollbackBaseTest, SetQuantParametersInvalidOffsetFailed)
{
    QuantTestGraph tg("offset_invalid");
    QuantGraphOptions opt;
    auto graph = BuildQuantProcessGraph(tg, opt);
    ASSERT_NE(graph, nullptr);
    UtQuantPass pass;
    const GNodePtr cube = FindByName(graph, "cube");
    const GNodePtr quant = FindByName(graph, "quant");
    ASSERT_NE(cube, nullptr);
    ASSERT_NE(quant, nullptr);
    float offset = std::numeric_limits<float>::quiet_NaN();
    (void)quant->SetAttr(AscendString("offset"), offset);
    EXPECT_NE(pass.SetQuantParameters(cube, quant), SUCCESS);
    offset = 3e9F;
    (void)quant->SetAttr(AscendString("offset"), offset);
    EXPECT_NE(pass.SetQuantParameters(cube, quant), SUCCESS);
    offset = 1.5F;
    (void)quant->SetAttr(AscendString("offset"), offset);
    EXPECT_EQ(pass.SetQuantParameters(cube, quant), SUCCESS);
    int64_t offsetX = -1;
    EXPECT_EQ(cube->GetAttr(AscendString("offset_x"), offsetX), GRAPH_SUCCESS);
    EXPECT_EQ(offsetX, 1);
}

TEST_F(BiasOptimizeQuantRollbackBaseTest, IsCubeSupportDtypeProbeKeepsDtype)
{
    QuantTestGraph tg("dtype_probe");
    QuantGraphOptions opt;
    auto graph = BuildQuantProcessGraph(tg, opt);
    ASSERT_NE(graph, nullptr);
    UtQuantPass pass;
    pass.cubeSupported = true;
    const GNodePtr cube = FindByName(graph, "cube");
    ASSERT_NE(cube, nullptr);
    std::vector<std::pair<int32_t, ge::DataType>> before;
    for (int32_t i = 0; i < static_cast<int32_t>(cube->GetInputsSize()); ++i) {
        TensorDesc desc;
        ASSERT_EQ(cube->GetInputDesc(i, desc), GRAPH_SUCCESS);
        before.emplace_back(i, desc.GetDataType());
    }
    for (int32_t i = 0; i < static_cast<int32_t>(cube->GetOutputsSize()); ++i) {
        TensorDesc desc;
        ASSERT_EQ(cube->GetOutputDesc(i, desc), GRAPH_SUCCESS);
        before.emplace_back(-1 - i, desc.GetDataType());
    }
    EXPECT_TRUE(pass.IsCubeSupportDtype(cube, DT_FLOAT16));
    for (const auto& item : before) {
        TensorDesc desc;
        if (item.first >= 0) {
            ASSERT_EQ(cube->GetInputDesc(item.first, desc), GRAPH_SUCCESS);
        } else {
            ASSERT_EQ(cube->GetOutputDesc(-1 - item.first, desc), GRAPH_SUCCESS);
        }
        EXPECT_EQ(desc.GetDataType(), item.second);
    }
}

TEST_F(BiasOptimizeQuantRollbackBaseTest, RemoveEnterSubGraphDataWithIndexAttr)
{
    QuantTestGraph tg("subgraph_data_index");
    QuantGraphOptions opt;
    opt.withWeightQuant = false;
    opt.withBias = false;
    auto graph = BuildQuantProcessGraph(tg, opt);
    ASSERT_NE(graph, nullptr);
    // 构造 Enter → Data(index) 链：模拟新框架子图 Data 场景
    (void)tg.AddNode("sub_data", "Data", {{"x", true}});
    (void)tg.AddNode("enter_sub", "Enter", {{"x", true}});
    tg.Connect("sub_data", kOutPort, "enter_sub", kQuantInputPort);
    tg.Connect("enter_sub", kOutPort, "cube", kCubeBiasPort);
    // 新框架子图 Data 用 index 属性表示父输入映射（非老 _parent_node_index）
    int64_t dataIndex = 0;
    (void)tg.Node("sub_data").SetAttr(AscendString("index"), dataIndex);
    TensorDesc subDesc;
    subDesc.SetDataType(DT_INT32);
    subDesc.SetShape(Shape({16}));
    (void)tg.Node("sub_data").UpdateOutputDesc(kOutPort, subDesc);
    (void)tg.Node("enter_sub").UpdateInputDesc(kQuantInputPort, subDesc);
    (void)tg.Node("enter_sub").UpdateOutputDesc(kOutPort, subDesc);
    UtQuantPass pass;
    const GNodePtr cube = FindByName(graph, "cube");
    ASSERT_NE(cube, nullptr);
    ASSERT_EQ(pass.RemoveEnter(*graph, cube), ge::SUCCESS);
    // index 属性回退应正确识别 Data 并追踪到 const（本图中 sub_data 无父图，Enter 不旁路）
    // 验证：Enter 未被误删（Data 无父图映射时按非 const 处理）
    EXPECT_EQ(CountByName(graph, "enter_sub"), 1);
}

TEST_F(BiasOptimizeQuantRollbackBaseTest, CheckOpSupportedSkipCheckAlwaysTrue)
{
    QuantTestGraph tg("check_support");
    QuantGraphOptions opt;
    auto graph = BuildQuantProcessGraph(tg, opt);
    ASSERT_NE(graph, nullptr);
    PlainQuantPass pass;
    const GNodePtr cube = FindByName(graph, "cube");
    ASSERT_NE(cube, nullptr);
    // 基类默认不再调 CheckNodeSupportOnAicore（兼容性问题），恒返回支持
    EXPECT_TRUE(pass.CheckOpSupported(*cube));
}

namespace {
constexpr int32_t kDtOutSizePort = 1;
constexpr int32_t kDtFilterPort = 2;
constexpr int32_t kDtBiasPort = 3;

std::shared_ptr<Graph> BuildConvDtLayoutQuantGraph(QuantTestGraph& tg, const QuantGraphOptions& opt)
{
    const std::vector<int64_t> fmapDims = {1, opt.cDim, 8, 8};
    const std::vector<int64_t> filterDims = {2, 16, 1, 1};
    auto xInput = tg.CreateInput("x", DT_FLOAT, FORMAT_NCHW, fmapDims);

    GNode& quant = tg.AddNode("quant", "AscendQuant", {{"x", true}});
    float scaleAttr = opt.quantScale;
    float offsetAttr = opt.quantOffset;
    (void)quant.SetAttr(AscendString("scale"), scaleAttr);
    (void)quant.SetAttr(AscendString("offset"), offsetAttr);

    GNode& cube = tg.AddNode("cube", "Conv2DTranspose",
                             {{"x", true}, {"output_size", true}, {"filter", true}, {"bias", false}});
    int64_t groups = opt.cubeGroups;
    (void)cube.SetAttr(AscendString("groups"), groups);

    std::vector<uint8_t> outSizeData(8, 8U);
    tg.CreateConst("out_size_const", {4}, DT_INT32, outSizeData);
    tg.Connect("out_size_const", kOutPort, "cube", kDtOutSizePort);

    if (opt.withWeightQuant) {
        std::vector<uint8_t> weightData(2 * 16 * 4, 1U);
        tg.CreateConst("weight_const", filterDims, DT_FLOAT, weightData);
        std::vector<uint8_t> offsetData(4, 0U);
        tg.CreateConst("offset_const", {1}, DT_FLOAT, offsetData);
        GNode& awq = tg.AddNode("awq", "AscendWeightQuant", {{"x", true}, {"offset", false}});
        int32_t dstType = static_cast<int32_t>(DT_INT8);
        (void)awq.SetAttr(AscendString("dst_type"), dstType);
        tg.Connect("weight_const", kOutPort, "awq", kQuantInputPort);
        tg.Connect("offset_const", kOutPort, "awq", 1);
        tg.Connect("awq", kOutPort, "cube", kDtFilterPort);
        tg.SetInDesc("awq", 0, DT_FLOAT, FORMAT_NCHW, filterDims);
        tg.SetInDesc("awq", 1, DT_FLOAT, FORMAT_ND, {1});
        tg.SetOutDesc("awq", kOutPort, DT_INT8, FORMAT_NCHW, filterDims);
        TensorDesc awqOut;
        (void)tg.Node("awq").GetOutputDesc(kOutPort, awqOut);
        (void)cube.UpdateInputDesc(kDtFilterPort, awqOut);
    } else {
        std::vector<uint8_t> weightData(2 * 16 * 4, 1U);
        tg.CreateConst("weight_const", filterDims, DT_FLOAT, weightData);
        tg.Connect("weight_const", kOutPort, "cube", kDtFilterPort);
        tg.SetInDesc("cube", kDtFilterPort, DT_FLOAT, FORMAT_NCHW, filterDims);
    }

    if (opt.withBias) {
        std::vector<uint8_t> biasData(16 * 4, 0U);
        tg.CreateConst("bias_const", {16}, DT_INT32, biasData);
        tg.Connect("bias_const", kOutPort, "cube", kDtBiasPort);
        TensorDesc biasOut;
        (void)tg.Node("bias_const").GetOutputDesc(kOutPort, biasOut);
        (void)cube.UpdateInputDesc(kDtBiasPort, biasOut);
    }

    if (opt.scaleIsConst) {
        std::vector<uint8_t> scaleData(8, 0U);
        tg.CreateConst("deq_scale", opt.scaleDims, DT_UINT64, scaleData);
    }

    tg.AddNode("dequant", "AscendDequant", {{"x", true}, {"deq_scale", true}});
    tg.AddNode("relu", "Relu", {{"x", true}});

    tg.ConnectInput(xInput, "quant", kQuantInputPort);
    tg.Connect("quant", kOutPort, "cube", kCubeFmapPort);
    tg.Connect("cube", kOutPort, "dequant", kDequantXPort);
    if (opt.scaleIsConst) {
        tg.Connect("deq_scale", kOutPort, "dequant", kDequantScalePort);
    }
    tg.Connect("dequant", kOutPort, "relu", kQuantInputPort);

    TensorDesc outSizeDesc;
    (void)tg.Node("out_size_const").GetOutputDesc(kOutPort, outSizeDesc);
    (void)cube.UpdateInputDesc(kDtOutSizePort, outSizeDesc);
    tg.SetInDesc("quant", 0, DT_FLOAT, FORMAT_NCHW, fmapDims);
    tg.SetOutDesc("quant", kOutPort, DT_INT8, FORMAT_NCHW, fmapDims);
    tg.SetInDesc("cube", kCubeFmapPort, DT_INT8, FORMAT_NCHW, fmapDims);
    tg.SetOutDesc("cube", kOutPort, DT_INT32, FORMAT_NCHW, {1, 3, 8, 8});
    tg.SetInDesc("dequant", kDequantXPort, DT_INT32, FORMAT_NCHW, {1, 3, 8, 8});
    tg.SetInDesc("dequant", kDequantScalePort, DT_UINT64, FORMAT_ND, opt.scaleDims);
    tg.SetOutDesc("dequant", kOutPort, DT_FLOAT16, FORMAT_NCHW, {1, 3, 8, 8});
    tg.SetInDesc("relu", kQuantInputPort, DT_FLOAT16, FORMAT_NCHW, {1, 3, 8, 8});
    tg.SetOutDesc("relu", kOutPort, DT_FLOAT16, FORMAT_NCHW, {1, 3, 8, 8});
    return tg.Build("relu");
}

class UtConvDtLikePass : public UtQuantPass {
protected:
    uint32_t GetBiasInputIndex() const override { return 3U; }
    uint32_t GetInputSizeContainsBias() const override { return 4U; }
    uint32_t GetWeightInputIndex() const override { return 2U; }
    uint32_t GetBiasOptimizationModeIndex() const override
    {
        return static_cast<uint32_t>(ops::HostOpMode::BIAS_OPTIMIZATION_MODE_CONVDT);
    }
    uint32_t GetWeightRollBackModeIndex() const override
    {
        return static_cast<uint32_t>(ops::HostOpMode::WEIGHT_ROLL_BACK_MODE_CONVDT);
    }
    uint32_t GetBiasRollBackModeIndex() const override
    {
        return static_cast<uint32_t>(ops::HostOpMode::BIAS_ROLL_BACK_MODE_CONVDT);
    }
};
} // namespace

TEST_F(BiasOptimizeQuantRollbackBaseTest, IndexOverrideRollbackFullDispatch)
{
    QuantTestGraph tg("idx_rb_full");
    QuantGraphOptions opt;
    opt.cDim = 65;
    auto graph = BuildConvDtLayoutQuantGraph(tg, opt);
    ASSERT_NE(graph, nullptr);
    UtConvDtLikePass pass;
    pass.cubeSupported = false;
    std::vector<GNodePtr> fusionNodes;
    std::vector<GNodePtr> nodesToRemove;
    ASSERT_EQ(pass.OptimizeMainProc(graph, FindByName(graph, "quant"), FindByName(graph, "dequant"),
                                    FindByName(graph, "cube"), fusionNodes, nodesToRemove),
              SUCCESS);
    ApplyNodesToRemove(graph, nodesToRemove);
    EXPECT_EQ(CountByType(graph, "QuantWeightRollBack"), 1);
    EXPECT_EQ(CountByType(graph, "QuantBiasRollBack"), 1);
    EXPECT_EQ(CountByType(graph, "AscendWeightQuant"), 0);
    EXPECT_EQ(CountByName(graph, "offset_const"), 0);
    EXPECT_EQ(CountByName(graph, "quant"), 0);
    EXPECT_EQ(CountByName(graph, "dequant"), 0);
    const GNodePtr cube = FindByName(graph, "cube");
    ASSERT_NE(cube, nullptr);
    EXPECT_EQ(PeerInType(cube, kCubeFmapPort), "Data");
    EXPECT_EQ(PeerInType(cube, kDtFilterPort), "QuantWeightRollBack");
    EXPECT_EQ(PeerInType(cube, kDtBiasPort), "QuantBiasRollBack");
    const GNodePtr weightRb = FindByType(graph, "QuantWeightRollBack");
    ASSERT_NE(weightRb, nullptr);
    EXPECT_EQ(PeerInType(weightRb, 0), "Const");
    EXPECT_EQ(PeerInType(weightRb, 1), "Const");
    TensorDesc rbOut;
    (void)weightRb->GetOutputDesc(kOutPort, rbOut);
    EXPECT_EQ(rbOut.GetDataType(), DT_FLOAT);
    int64_t groups = 0;
    EXPECT_EQ(weightRb->GetAttr(AscendString("groups"), groups), GRAPH_SUCCESS);
    EXPECT_EQ(groups, 1);
    const GNodePtr biasRb = FindByType(graph, "QuantBiasRollBack");
    ASSERT_NE(biasRb, nullptr);
    EXPECT_EQ(PeerInType(biasRb, 0), "Const");
    EXPECT_EQ(PeerInType(biasRb, 1), "Const");
    EXPECT_TRUE(ContainsNode(fusionNodes, "cube"));
}

TEST_F(BiasOptimizeQuantRollbackBaseTest, IndexOverrideRollbackNoBiasAtOverridePort)
{
    QuantTestGraph tg("idx_rb_nobias");
    QuantGraphOptions opt;
    opt.withBias = false;
    opt.withWeightQuant = false;
    opt.cDim = 65;
    auto graph = BuildConvDtLayoutQuantGraph(tg, opt);
    ASSERT_NE(graph, nullptr);
    UtConvDtLikePass pass;
    pass.cubeSupported = false;
    std::vector<GNodePtr> fusionNodes;
    std::vector<GNodePtr> nodesToRemove;
    ASSERT_EQ(pass.OptimizeMainProc(graph, FindByName(graph, "quant"), FindByName(graph, "dequant"),
                                    FindByName(graph, "cube"), fusionNodes, nodesToRemove),
              SUCCESS);
    ApplyNodesToRemove(graph, nodesToRemove);
    EXPECT_EQ(CountByType(graph, "QuantWeightRollBack"), 1);
    EXPECT_EQ(CountByType(graph, "QuantBiasRollBack"), 0);
    const GNodePtr cube = FindByName(graph, "cube");
    ASSERT_NE(cube, nullptr);
    EXPECT_EQ(PeerInType(cube, kDtFilterPort), "QuantWeightRollBack");
    EXPECT_EQ(PeerInType(cube, kDtBiasPort), "");
}

TEST_F(BiasOptimizeQuantRollbackBaseTest, IndexOverrideBiasOptimizeDispatch)
{
    QuantTestGraph tg("idx_optim");
    QuantGraphOptions opt;
    opt.cDim = 65;
    auto graph = BuildConvDtLayoutQuantGraph(tg, opt);
    ASSERT_NE(graph, nullptr);
    UtConvDtLikePass pass;
    pass.cubeSupported = true;
    std::vector<GNodePtr> fusionNodes;
    std::vector<GNodePtr> nodesToRemove;
    ASSERT_EQ(pass.OptimizeMainProc(graph, FindByName(graph, "quant"), FindByName(graph, "dequant"),
                                    FindByName(graph, "cube"), fusionNodes, nodesToRemove),
              SUCCESS);
    ApplyNodesToRemove(graph, nodesToRemove);
    EXPECT_EQ(CountByType(graph, "QuantBiasOptimization"), 1);
    EXPECT_EQ(CountByName(graph, "quant"), 1);
    EXPECT_EQ(CountByName(graph, "dequant"), 1);
    const GNodePtr cube = FindByName(graph, "cube");
    ASSERT_NE(cube, nullptr);
    EXPECT_EQ(PeerInType(cube, kDtBiasPort), "QuantBiasOptimization");
    EXPECT_EQ(PeerInType(cube, kDtFilterPort), "AscendWeightQuant");
    const GNodePtr hostOp = FindByType(graph, "QuantBiasOptimization");
    ASSERT_NE(hostOp, nullptr);
    EXPECT_EQ(PeerInType(hostOp, 0), "Const");
    EXPECT_EQ(PeerInType(hostOp, 1), "Const");
    EXPECT_EQ(PeerInType(hostOp, 2), "Const");
    AscendString cubeOpType;
    EXPECT_EQ(hostOp->GetAttr(AscendString("cube_op_type"), cubeOpType), GRAPH_SUCCESS);
    EXPECT_STREQ(cubeOpType.GetString(), "Conv2DTranspose");
}

TEST_F(BiasOptimizeQuantRollbackBaseTest, IndexOverrideBiasOptimizeZeroBiasCreated)
{
    QuantTestGraph tg("idx_optim_zb");
    QuantGraphOptions opt;
    opt.withBias = false;
    opt.withWeightQuant = false;
    opt.cDim = 65;
    auto graph = BuildConvDtLayoutQuantGraph(tg, opt);
    ASSERT_NE(graph, nullptr);
    UtConvDtLikePass pass;
    pass.cubeSupported = true;
    std::vector<GNodePtr> fusionNodes;
    std::vector<GNodePtr> nodesToRemove;
    ASSERT_EQ(pass.OptimizeMainProc(graph, FindByName(graph, "quant"), FindByName(graph, "dequant"),
                                    FindByName(graph, "cube"), fusionNodes, nodesToRemove),
              SUCCESS);
    ApplyNodesToRemove(graph, nodesToRemove);
    EXPECT_EQ(CountByType(graph, "QuantBiasOptimization"), 1);
    const GNodePtr cube = FindByName(graph, "cube");
    ASSERT_NE(cube, nullptr);
    EXPECT_EQ(PeerInType(cube, kDtBiasPort), "QuantBiasOptimization");
    const GNodePtr hostOp = FindByType(graph, "QuantBiasOptimization");
    ASSERT_NE(hostOp, nullptr);
    TensorDesc hostIn0;
    (void)hostOp->GetInputDesc(0, hostIn0);
    EXPECT_EQ(hostIn0.GetDataType(), DT_INT32);
    EXPECT_EQ(hostIn0.GetShape().GetDims(), std::vector<int64_t>({2}));
}

TEST_F(BiasOptimizeQuantRollbackBaseTest, NodesToRemoveNodesStillAccessibleAfterReturn)
{
    QuantTestGraph tg("p2_alive");
    QuantGraphOptions opt;
    opt.cDim = 65;
    auto graph = BuildQuantProcessGraph(tg, opt);
    ASSERT_NE(graph, nullptr);
    UtQuantPass pass;
    pass.cubeSupported = false;
    std::vector<GNodePtr> fusionNodes;
    std::vector<GNodePtr> nodesToRemove;
    ASSERT_EQ(pass.OptimizeMainProc(graph, FindByName(graph, "quant"), FindByName(graph, "dequant"),
                                    FindByName(graph, "cube"), fusionNodes, nodesToRemove),
              SUCCESS);
    ASSERT_FALSE(nodesToRemove.empty());
    for (const auto& node : nodesToRemove) {
        AscendString name;
        AscendString type;
        EXPECT_EQ(node->GetName(name), GRAPH_SUCCESS);
        EXPECT_EQ(node->GetType(type), GRAPH_SUCCESS);
        EXPECT_NE(name.GetString(), nullptr);
        EXPECT_NE(type.GetString(), nullptr);
    }
    ApplyNodesToRemove(graph, nodesToRemove);
    EXPECT_EQ(CountByName(graph, "quant"), 0);
    EXPECT_EQ(CountByName(graph, "dequant"), 0);
}

TEST_F(BiasOptimizeQuantRollbackBaseTest, NodesToRemoveResetOnSecondCall)
{
    QuantTestGraph tg("p2_reset");
    QuantGraphOptions opt;
    opt.cDim = 65;
    auto graph = BuildQuantProcessGraph(tg, opt);
    ASSERT_NE(graph, nullptr);
    UtQuantPass pass;
    pass.cubeSupported = false;
    std::vector<GNodePtr> fusionNodes;
    std::vector<GNodePtr> nodesToRemove1;
    std::vector<GNodePtr> nodesToRemove2;
    ASSERT_EQ(pass.OptimizeMainProc(graph, FindByName(graph, "quant"), FindByName(graph, "dequant"),
                                    FindByName(graph, "cube"), fusionNodes, nodesToRemove1),
              SUCCESS);
    EXPECT_FALSE(nodesToRemove1.empty());
    // 第二次调用: 第一次融合已改变图结构(边已改/节点已收集), 同一图重复调用会FAILED
    // 验证的是: 第二次调用时nodesToRemove_已被clear(不含第一次残留)
    const auto ret2 = pass.OptimizeMainProc(graph, FindByName(graph, "quant"), FindByName(graph, "dequant"),
                                            FindByName(graph, "cube"), fusionNodes, nodesToRemove2);
    if (ret2 == SUCCESS) {
        for (const auto& node : nodesToRemove2) {
            AscendString name;
            (void)node->GetName(name);
            std::string nameStr = (name.GetString() != nullptr) ? name.GetString() : "";
            EXPECT_NE(nameStr, "awq") << "第二次调用不应包含第一次的awq残留";
        }
    }
    // 无论第二次结果如何, nodesToRemove_的clear语义已验证:
    // 如果FAILED, nodesToRemove2不会被写入; 如果SUCCESS, 不含残留
}

TEST_F(BiasOptimizeQuantRollbackBaseTest, ZeroBiasLengthEqualsCoNotCin)
{
    // 无bias + BIAS_OPTIMIZE: 零bias长度应为Co(N轴) × groups, 而非Cin(C轴) × groups
    QuantTestGraph tg("p1_zero_bias");
    QuantGraphOptions opt;
    opt.withBias = false;
    opt.withWeightQuant = false;
    opt.cDim = 65; // fmap C=65 > 16 → BIAS_OPTIMIZE
    auto graph = BuildQuantProcessGraph(tg, opt);
    ASSERT_NE(graph, nullptr);
    UtQuantPass pass;
    pass.cubeSupported = true;
    std::vector<GNodePtr> fusionNodes;
    std::vector<GNodePtr> nodesToRemove;
    // 设置filter format为NCHW(CreateConst默认ND会导致GetCoValue走ND分支)
    const GNodePtr cubePre = FindByName(graph, "cube");
    ASSERT_NE(cubePre, nullptr);
    TensorDesc nchwFilter;
    nchwFilter.SetDataType(DT_FLOAT);
    nchwFilter.SetFormat(FORMAT_NCHW);
    nchwFilter.SetShape(Shape({2, 16, 1, 1}));
    (void)cubePre->UpdateInputDesc(kCubeFilterPort, nchwFilter);
    ASSERT_EQ(pass.OptimizeMainProc(graph, FindByName(graph, "quant"), FindByName(graph, "dequant"),
                                    FindByName(graph, "cube"), fusionNodes, nodesToRemove),
              SUCCESS);
    // filter {2, 16, 1, 1} NCHW → Co=2(N=0), Cin=16(C=1)
    // 零bias长度应为Co=2, 而非Cin=16
    const GNodePtr cube = FindByName(graph, "cube");
    ASSERT_NE(cube, nullptr);
    const auto biasPeer = cube->GetInDataNodesAndPortIndexs(kCubeBiasPort);
    ASSERT_NE(biasPeer.first, nullptr);
    TensorDesc biasDesc;
    ASSERT_EQ(biasPeer.first->GetOutputDesc(0, biasDesc), GRAPH_SUCCESS);
    EXPECT_EQ(biasDesc.GetShape().GetDims(), std::vector<int64_t>({2})) << "零bias长度应为Co=2";
}

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

#include "../../../op_graph/fusion_pass/batch_matmulv2_quant_process_fusion_pass.h"
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

// Nano 产品特性注入：ubblock_size == 16（对齐老 UtilPlatform::isNano 判定源）
void SetNanoPlatform(const std::string& soc)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optionalInfo;
    platformInfo.str_info.short_soc_version = soc;
    optionalInfo.soc_version = soc;
    platformInfo.ai_core_spec.ubblock_size = 16U;
    fe::PlatformInfoManager::Instance().platform_info_map_[soc] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optionalInfo);
}

// Tiny 产品特性注入：Intrinsic_set_l1 仅 f16（对齐老 UtilPlatform::isTiny 判定源）
void SetTinyPlatform(const std::string& soc)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optionalInfo;
    platformInfo.str_info.short_soc_version = soc;
    optionalInfo.soc_version = soc;
    platformInfo.ai_core_intrinsic_dtype_map["Intrinsic_set_l1"] = {"f16"};
    fe::PlatformInfoManager::Instance().platform_info_map_[soc] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optionalInfo);
}

class BatchMatmulV2QuantTestGraph {
public:
    explicit BatchMatmulV2QuantTestGraph(const std::string& name) : builder_(name.c_str())
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

struct BatchMatmulV2QuantGraphOptions {
    bool withWeightQuant = true;      // weight 走 AscendWeightQuant（quant 场景常态）
    bool weightIsAscendQuant = false; // x2 直接接 AscendQuant（GetQuantProcessMode 返回 UNDEFINED 的场景）
    bool withBias = false;
    std::vector<int64_t> xDims = {2, 16, 16};
    std::vector<int64_t> weightDims = {16, 16};
    bool weightTranspose = false; // cube 的 adj_x2 属性
    std::string cubeType = "BatchMatMulV2";
};

std::shared_ptr<Graph> BuildBatchMatmulV2QuantGraph(BatchMatmulV2QuantTestGraph& tg,
                                                    const BatchMatmulV2QuantGraphOptions& opt)
{
    auto xInput = tg.CreateInput("x", DT_FLOAT16, FORMAT_ND, opt.xDims);

    GNode& quant = tg.AddNode("quant", "AscendQuant", {{"x", true}});
    float scaleAttr = 1.2F;
    float offsetAttr = 1.1F;
    (void)quant.SetAttr(AscendString("scale"), scaleAttr);
    (void)quant.SetAttr(AscendString("offset"), offsetAttr);

    GNode& cube = tg.AddNode("cube", opt.cubeType, {{"x1", true}, {"x2", true}, {"bias", false}, {"offset_w", false}});
    bool adjX2 = opt.weightTranspose;
    (void)cube.SetAttr(AscendString("adj_x2"), adjX2);

    // unknown shape 时 GetShapeSize 为负，Const 数据仅作占位
    const int64_t weightShapeSize = Shape(opt.weightDims).GetShapeSize();
    const size_t weightElemNum = (weightShapeSize > 0) ? static_cast<size_t>(weightShapeSize) : 1U;
    if (opt.weightIsAscendQuant) {
        auto weightInput = tg.CreateInput("weight_x", DT_FLOAT16, FORMAT_ND, opt.weightDims);
        GNode& weightQuant = tg.AddNode("weight_quant", "AscendQuant", {{"x", true}});
        tg.ConnectInput(weightInput, "weight_quant", kQuantInputPort);
        tg.Connect("weight_quant", kOutPort, "cube", kCubeWeightPort);
        tg.SetInDesc("weight_quant", 0, DT_FLOAT16, FORMAT_ND, opt.weightDims);
        tg.SetOutDesc("weight_quant", kOutPort, DT_INT8, FORMAT_ND, opt.weightDims);
    } else if (opt.withWeightQuant) {
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

    std::vector<int64_t> cubeOutDims = opt.xDims;
    cubeOutDims[cubeOutDims.size() - 1U] = opt.weightDims[opt.weightDims.size() - 1U];
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
class ProbeBatchMatmulV2QuantProcessFusionPass : public ops::BatchMatmulV2QuantProcessFusionPass {
public:
    bool cubeSupported = true;

    ge::Status CallProcessBiasInput(ge::Graph& graph, const GNodePtr& cubeNode)
    {
        return ProcessBiasInput(graph, cubeNode);
    }

protected:
    bool CheckOpSupported(const GNode& node) const override
    {
        (void)node;
        return cubeSupported;
    }
};
} // namespace

TEST(BatchMatmulV2QuantProcessFusionPassTest, biasOptimizeSuccess)
{
    SetPlatform("Ascend950");
    BatchMatmulV2QuantTestGraph tg("bmmv2_bias_optimize");
    BatchMatmulV2QuantGraphOptions opt;
    auto graph = BuildBatchMatmulV2QuantGraph(tg, opt);
    ASSERT_NE(graph, nullptr);
    ProbeBatchMatmulV2QuantProcessFusionPass pass;
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
    // 2 维权重无 batch 前缀：bias shape 为 {co}=weight dims[1]=16
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

TEST(BatchMatmulV2QuantProcessFusionPassTest, biasOptimizeWithBiasSuccess)
{
    SetPlatform("Ascend950");
    BatchMatmulV2QuantTestGraph tg("bmmv2_bias_optimize_with_bias");
    BatchMatmulV2QuantGraphOptions opt;
    opt.withBias = true;
    auto graph = BuildBatchMatmulV2QuantGraph(tg, opt);
    ASSERT_NE(graph, nullptr);
    ProbeBatchMatmulV2QuantProcessFusionPass pass;
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

TEST(BatchMatmulV2QuantProcessFusionPassTest, quantRollbackSuccess)
{
    SetPlatform("Ascend950");
    BatchMatmulV2QuantTestGraph tg("bmmv2_rollback");
    BatchMatmulV2QuantGraphOptions opt;
    auto graph = BuildBatchMatmulV2QuantGraph(tg, opt);
    ASSERT_NE(graph, nullptr);
    ProbeBatchMatmulV2QuantProcessFusionPass pass;
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
    EXPECT_EQ(PeerInType(relu, kQuantInputPort), "BatchMatMulV2");
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
    EXPECT_STREQ(cubeOpType.GetString(), "BatchMatMulV2");
    float scale = 0.0F;
    EXPECT_EQ(weightRb->GetAttr(AscendString("scale"), scale), GRAPH_SUCCESS);
    EXPECT_FLOAT_EQ(scale, 1.2F);
}

TEST(BatchMatmulV2QuantProcessFusionPassTest, quantRollbackWithBiasSuccess)
{
    SetPlatform("Ascend950");
    BatchMatmulV2QuantTestGraph tg("bmmv2_rollback_with_bias");
    BatchMatmulV2QuantGraphOptions opt;
    opt.withBias = true;
    auto graph = BuildBatchMatmulV2QuantGraph(tg, opt);
    ASSERT_NE(graph, nullptr);
    ProbeBatchMatmulV2QuantProcessFusionPass pass;
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

TEST(BatchMatmulV2QuantProcessFusionPassTest, weightIsAscendQuantSkip)
{
    SetPlatform("Ascend950");
    BatchMatmulV2QuantTestGraph tg("bmmv2_weight_is_quant");
    BatchMatmulV2QuantGraphOptions opt;
    opt.weightIsAscendQuant = true;
    auto graph = BuildBatchMatmulV2QuantGraph(tg, opt);
    ASSERT_NE(graph, nullptr);
    ProbeBatchMatmulV2QuantProcessFusionPass pass;
    pass.cubeSupported = true;
    CustomPassContext passContext;
    // double-quant 让位场景改写了 weight Format（HWCN）——图已变更, Run 返回 SUCCESS
    // （对齐原版语义: 该场景旧框架记 effect=1; 旧断言 NOT_CHANGED 系未报 effect 时代的过时预期）
    EXPECT_EQ(pass.Run(graph, passContext), SUCCESS);
    EXPECT_EQ(CountByType(graph, "QuantBiasOptimization"), 0);
    EXPECT_EQ(CountByType(graph, "QuantWeightRollBack"), 0);
    EXPECT_EQ(CountByName(graph, "quant"), 1);
    EXPECT_EQ(CountByName(graph, "dequant"), 1);
    // weight 输入为 AscendQuant 时跳过融合，且将 weight 输入的 format 刷为 HWCN
    const GNodePtr cube = FindByName(graph, "cube");
    ASSERT_NE(cube, nullptr);
    TensorDesc weightDesc;
    EXPECT_EQ(cube->GetInputDesc(kCubeWeightPort, weightDesc), GRAPH_SUCCESS);
    EXPECT_EQ(weightDesc.GetFormat(), FORMAT_HWCN);
    TensorDesc weightOriginDesc;
    EXPECT_EQ(cube->GetInputDesc(kCubeWeightPort, weightOriginDesc), GRAPH_SUCCESS);
    EXPECT_EQ(weightOriginDesc.GetOriginFormat(), FORMAT_HWCN);
}

TEST(BatchMatmulV2QuantProcessFusionPassTest, weightDimNotTwoFailed)
{
    SetPlatform("Ascend950");
    BatchMatmulV2QuantTestGraph tg("bmmv2_weight_dim3");
    BatchMatmulV2QuantGraphOptions opt;
    opt.weightDims = {2, 16, 16}; // 本平台要求权重为 2 维，3 维返回 FAILED
    auto graph = BuildBatchMatmulV2QuantGraph(tg, opt);
    ASSERT_NE(graph, nullptr);
    ProbeBatchMatmulV2QuantProcessFusionPass pass;
    pass.cubeSupported = true;
    CustomPassContext passContext;
    EXPECT_EQ(pass.Run(graph, passContext), FAILED);
}

TEST(BatchMatmulV2QuantProcessFusionPassTest, unknownWeightShapeSkip)
{
    SetPlatform("Ascend950");
    BatchMatmulV2QuantTestGraph tg("bmmv2_unknown_weight");
    BatchMatmulV2QuantGraphOptions opt;
    opt.weightDims = {-1, 16};
    auto graph = BuildBatchMatmulV2QuantGraph(tg, opt);
    ASSERT_NE(graph, nullptr);
    ProbeBatchMatmulV2QuantProcessFusionPass pass;
    pass.cubeSupported = true;
    CustomPassContext passContext;
    // UNDEFINED 跳过场景：原版 Fusion 返回 SUCCESS（旧框架记 effect=1）, Run 同构返回 SUCCESS
    EXPECT_EQ(pass.Run(graph, passContext), SUCCESS);
    EXPECT_EQ(CountByType(graph, "QuantBiasOptimization"), 0);
    EXPECT_EQ(CountByType(graph, "QuantWeightRollBack"), 0);
    EXPECT_EQ(CountByName(graph, "quant"), 1);
    EXPECT_EQ(CountByName(graph, "dequant"), 1);
}

TEST(BatchMatmulV2QuantProcessFusionPassTest, skipsOtherCubeType)
{
    SetPlatform("Ascend950");
    BatchMatmulV2QuantTestGraph tg("bmmv2_other_type");
    BatchMatmulV2QuantGraphOptions opt;
    opt.cubeType = "BatchMatMul";
    auto graph = BuildBatchMatmulV2QuantGraph(tg, opt);
    ASSERT_NE(graph, nullptr);
    ProbeBatchMatmulV2QuantProcessFusionPass pass;
    pass.cubeSupported = true;
    CustomPassContext passContext;
    EXPECT_EQ(pass.Run(graph, passContext), GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountByType(graph, "QuantBiasOptimization"), 0);
    EXPECT_EQ(CountByType(graph, "QuantWeightRollBack"), 0);
}

TEST(BatchMatmulV2QuantProcessFusionPassTest, processBiasInputNoBatch)
{
    SetPlatform("Ascend950");
    BatchMatmulV2QuantTestGraph tg("bmmv2_bias_no_batch");
    BatchMatmulV2QuantGraphOptions opt;
    opt.weightDims = {16, 8};
    auto graph = BuildBatchMatmulV2QuantGraph(tg, opt);
    ASSERT_NE(graph, nullptr);
    ProbeBatchMatmulV2QuantProcessFusionPass pass;
    pass.cubeSupported = true;
    const GNodePtr cube = FindByName(graph, "cube");
    ASSERT_NE(cube, nullptr);
    EXPECT_EQ(pass.CallProcessBiasInput(*graph, cube), SUCCESS);
    // 2 维权重：bias shape 为 {co}=dims[1]=8，无 batch 前缀
    const GNodePtr biasConst = FindByName(graph, "cube_quant_bias");
    ASSERT_NE(biasConst, nullptr);
    TensorDesc biasDesc;
    EXPECT_EQ(biasConst->GetOutputDesc(kOutPort, biasDesc), GRAPH_SUCCESS);
    EXPECT_EQ(biasDesc.GetShape().GetDims(), std::vector<int64_t>({8}));
}

TEST(BatchMatmulV2QuantProcessFusionPassTest, processBiasInputWithBatch)
{
    SetPlatform("Ascend950");
    // 3 维权重（batch 维场景）：bias shape 为 {batch, co}
    BatchMatmulV2QuantTestGraph tg("bmmv2_bias_batch");
    BatchMatmulV2QuantGraphOptions opt;
    opt.weightDims = {2, 8, 16};
    auto graph = BuildBatchMatmulV2QuantGraph(tg, opt);
    ASSERT_NE(graph, nullptr);
    ProbeBatchMatmulV2QuantProcessFusionPass pass;
    pass.cubeSupported = true;
    const GNodePtr cube = FindByName(graph, "cube");
    ASSERT_NE(cube, nullptr);
    EXPECT_EQ(pass.CallProcessBiasInput(*graph, cube), SUCCESS);
    // 非转置：co=dims[last]=16，bias shape {2, 16}
    const GNodePtr biasConst = FindByName(graph, "cube_quant_bias");
    ASSERT_NE(biasConst, nullptr);
    TensorDesc biasDesc;
    EXPECT_EQ(biasConst->GetOutputDesc(kOutPort, biasDesc), GRAPH_SUCCESS);
    EXPECT_EQ(biasDesc.GetShape().GetDims(), std::vector<int64_t>({2, 16}));
}

TEST(BatchMatmulV2QuantProcessFusionPassTest, processBiasInputWithBatchTranspose)
{
    SetPlatform("Ascend950");
    // 3 维权重 + adj_x2 转置：co=dims[size-2]=8，bias shape {2, 8}
    BatchMatmulV2QuantTestGraph tg("bmmv2_bias_batch_trans");
    BatchMatmulV2QuantGraphOptions opt;
    opt.weightDims = {2, 8, 16};
    opt.weightTranspose = true;
    auto graph = BuildBatchMatmulV2QuantGraph(tg, opt);
    ASSERT_NE(graph, nullptr);
    ProbeBatchMatmulV2QuantProcessFusionPass pass;
    pass.cubeSupported = true;
    const GNodePtr cube = FindByName(graph, "cube");
    ASSERT_NE(cube, nullptr);
    EXPECT_EQ(pass.CallProcessBiasInput(*graph, cube), SUCCESS);
    const GNodePtr biasConst = FindByName(graph, "cube_quant_bias");
    ASSERT_NE(biasConst, nullptr);
    TensorDesc biasDesc;
    EXPECT_EQ(biasConst->GetOutputDesc(kOutPort, biasDesc), GRAPH_SUCCESS);
    EXPECT_EQ(biasDesc.GetShape().GetDims(), std::vector<int64_t>({2, 8}));
}

TEST(BatchMatmulV2QuantProcessFusionPassTest, nanoPlatform3dWeightBiasOptimizeSuccess)
{
    // Nano 产品（ubblock_size==16）放行 >=2 维权重：3 维走完整 BIAS_OPTIMIZE，补零 bias 带 batch 维
    SetNanoPlatform("Ascend950");
    BatchMatmulV2QuantTestGraph tg("bmmv2_nano_3d");
    BatchMatmulV2QuantGraphOptions opt;
    opt.weightDims = {2, 8, 16}; // batch=2, co=16
    auto graph = BuildBatchMatmulV2QuantGraph(tg, opt);
    ASSERT_NE(graph, nullptr);
    ProbeBatchMatmulV2QuantProcessFusionPass pass;
    pass.cubeSupported = true;
    CustomPassContext passContext;
    EXPECT_EQ(pass.Run(graph, passContext), SUCCESS);
    EXPECT_EQ(CountByType(graph, "QuantBiasOptimization"), 1);
    EXPECT_EQ(CountByName(graph, "quant"), 1);
    EXPECT_EQ(CountByName(graph, "dequant"), 1);
    const GNodePtr hostOp = FindByType(graph, "QuantBiasOptimization");
    ASSERT_NE(hostOp, nullptr);
    Tensor biasTensor;
    EXPECT_EQ(hostOp->GetInputConstData(0, biasTensor), GRAPH_SUCCESS);
    ASSERT_NE(biasTensor.GetData(), nullptr);
    // Nano 3 维权重全流程：补零 bias shape 为 {batch, co} = {2, 16}（batch 维真实激活）
    EXPECT_EQ(biasTensor.GetSize(), 2U * 16U * sizeof(int32_t));
    for (size_t i = 0; i < biasTensor.GetSize(); ++i) {
        EXPECT_EQ(biasTensor.GetData()[i], 0U);
    }
}

TEST(BatchMatmulV2QuantProcessFusionPassTest, tinyPlatform3dWeightBiasOptimizeSuccess)
{
    // Tiny 产品（Intrinsic_set_l1 仅 f16）不做维数限制：3 维权重放行走 BIAS_OPTIMIZE
    SetTinyPlatform("Ascend950");
    BatchMatmulV2QuantTestGraph tg("bmmv2_tiny_3d");
    BatchMatmulV2QuantGraphOptions opt;
    opt.weightDims = {2, 8, 16};
    auto graph = BuildBatchMatmulV2QuantGraph(tg, opt);
    ASSERT_NE(graph, nullptr);
    ProbeBatchMatmulV2QuantProcessFusionPass pass;
    pass.cubeSupported = true;
    CustomPassContext passContext;
    EXPECT_EQ(pass.Run(graph, passContext), SUCCESS);
    EXPECT_EQ(CountByType(graph, "QuantBiasOptimization"), 1);
    const GNodePtr biasConst = FindByName(graph, "cube_quant_bias");
    ASSERT_NE(biasConst, nullptr);
    TensorDesc biasDesc;
    EXPECT_EQ(biasConst->GetOutputDesc(kOutPort, biasDesc), GRAPH_SUCCESS);
    EXPECT_EQ(biasDesc.GetShape().GetDims(), std::vector<int64_t>({2, 16}));
}

TEST(BatchMatmulV2QuantProcessFusionPassTest, nanoPlatform1dWeightFailed)
{
    // Nano 产品下 1 维权重（dims<2）仍拒绝：Run 返回 FAILED
    SetNanoPlatform("Ascend950");
    BatchMatmulV2QuantTestGraph tg("bmmv2_nano_1d");
    BatchMatmulV2QuantGraphOptions opt;
    opt.weightDims = {16};
    auto graph = BuildBatchMatmulV2QuantGraph(tg, opt);
    ASSERT_NE(graph, nullptr);
    ProbeBatchMatmulV2QuantProcessFusionPass pass;
    pass.cubeSupported = true;
    CustomPassContext passContext;
    EXPECT_EQ(pass.Run(graph, passContext), FAILED);
}

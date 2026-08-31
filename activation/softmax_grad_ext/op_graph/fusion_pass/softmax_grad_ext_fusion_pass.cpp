/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "securec.h"
#include "common/inc/error_util.h"
#include "softmax_grad_ext_fusion_pass.h"
#include "es_nn_ops.h"
#include "ge/compliant_node_builder.h"
#include "platform/platform_info.h"
#include "ge/ge_utils.h"
#include "version/cann_version.h"

using namespace ge;
using namespace fe;
using namespace fusion;

namespace ops {
namespace {
const std::string kPassName = "SoftmaxGradExtFusionPass";
const std::string kPassNameV2 = "SoftmaxGradExtV2FusionPass";

const int64_t kCaptureSumIdx = 0;
const int64_t kSubgraphInputGrad = 0;
const int64_t kSubgraphInputX1 = 1;
const int64_t kSubgraphInputX2 = 2;
const int32_t kReduceSumAxesInputIdx = 1;
const size_t kMinSubgraphInputCount = 3;

const int64_t kUnknownShapeDim = -1;
const int64_t kReduceLastAxis = -1;
const int64_t kAxesShapeDim = 1;
const int32_t kPatternV2VariantCount = 4;
const int32_t kBinaryInputX1Idx = 0;
const int32_t kBinaryInputX2Idx = 1;
const int32_t kReduceSumInputXIdx = 0;
const int32_t kNodeOutputIdx = 0;
const std::string kTargetSocVersion = "Ascend950";

bool IsUnknownShape(const std::vector<int64_t>& dims)
{
    for (auto dim : dims) {
        if (dim == kUnknownShapeDim) {
            return true;
        }
    }
    return false;
}

bool IsTargetPlatform()
{
    PlatformInfo platform_info;
    OptionalInfo optional_info;
    OP_LOGE_IF(
        PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platform_info, optional_info) != SUCCESS,
        false, kPassName.c_str(), "Get platform_info failed.");
    const std::string soc = platform_info.str_info.short_soc_version;
    OPS_LOG_D(kPassName.c_str(), "Platform short soc: %s", soc.c_str());
    if (soc != kTargetSocVersion) {
        OPS_LOG_D(kPassName.c_str(), "Platform is not support, only support %s.", kTargetSocVersion.c_str());
        return false;
    }
    return true;
}

bool CheckInputsShapeValid(const std::unique_ptr<MatchResult>& match_result)
{
    std::vector<SubgraphInput> subgraph_inputs;
    match_result->ToSubgraphBoundary()->GetAllInputs(subgraph_inputs);
    for (const auto& subgraph_input : subgraph_inputs) {
        const auto all_inputs = subgraph_input.GetAllInputs();
        if (all_inputs.empty()) {
            return false;
        }
        auto match_node = all_inputs.at(0);
        TensorDesc tensor_desc;
        if (match_node.node.GetInputDesc(match_node.index, tensor_desc) != GRAPH_SUCCESS) {
            return false;
        }
        if (IsUnknownShape(tensor_desc.GetShape().GetDims())) {
            OPS_LOG_D(kPassName.c_str(), "Input has unknown shape, skip fusion.");
            return false;
        }
    }
    return true;
}

bool GetCapturedSumNode(const std::unique_ptr<MatchResult>& match_result, GNode& sum_node)
{
    NodeIo node_io;
    OP_LOGE_IF(match_result->GetCapturedTensor(kCaptureSumIdx, node_io) != SUCCESS, false, kPassName.c_str(),
               "Failed to get captured sum node.");
    sum_node = node_io.node;
    return true;
}

// ReduceSum in the new IR carries `axes` as a const input (index 1) and `keep_dims` as an attribute.
// SoftmaxGradExt takes a single `axes` (Int) attribute and `keep_dims` (Bool) attribute.
bool GetAxisFromReduceSum(const GNode& sum_node, int64_t& axis_value, bool& keep_dims)
{
    Tensor axes_tensor;
    if (sum_node.GetInputConstData(kReduceSumAxesInputIdx, axes_tensor) != GRAPH_SUCCESS) {
        OPS_LOG_D(kPassName.c_str(), "Failed to get axes const input from ReduceSum.");
        return false;
    }
    const DataType dtype = axes_tensor.GetDataType();
    const int32_t elem_size = GetSizeByDataType(dtype);
    if (elem_size <= 0 || axes_tensor.GetSize() != static_cast<size_t>(elem_size)) {
        OPS_LOG_D(kPassName.c_str(), "ReduceSum axes must be a single int, but size is %zu.", axes_tensor.GetSize());
        return false;
    }
    const auto* data = axes_tensor.GetData();
    if (data == nullptr) {
        OPS_LOG_D(kPassName.c_str(), "ReduceSum axes const data is nullptr.");
        return false;
    }
    if (dtype == DT_INT64) {
        int64_t tmp = 0;
        auto memRet = memcpy_s(&tmp, sizeof(int64_t), data, sizeof(int64_t));
        OP_LOGE_IF(memRet != EOK, false, kPassName.c_str(), "memcpy_s for int64 axes failed, ret=%d.", memRet);
        axis_value = tmp;
    } else if (dtype == DT_INT32) {
        int32_t tmp = 0;
        auto memRet = memcpy_s(&tmp, sizeof(int32_t), data, sizeof(int32_t));
        OP_LOGE_IF(memRet != EOK, false, kPassName.c_str(), "memcpy_s for int32 axes failed, ret=%d.", memRet);
        axis_value = tmp;
    } else {
        OPS_LOG_D(kPassName.c_str(), "ReduceSum axes dtype %d is not supported.", static_cast<int32_t>(dtype));
        return false;
    }

    if (sum_node.GetAttr(AscendString("keep_dims"), keep_dims) != GRAPH_SUCCESS) {
        OPS_LOG_D(kPassName.c_str(), "Failed to get keep_dims attr from ReduceSum.");
        return false;
    }
    OPS_LOG_D(kPassName.c_str(), "ReduceSum axis=%lld, keep_dims=%d.", axis_value, static_cast<int32_t>(keep_dims));
    return true;
}

std::vector<es::EsTensorHolder> CreateReplacementInputs(es::EsGraphBuilder& graph_builder,
                                                        const std::vector<SubgraphInput>& subgraph_inputs)
{
    std::vector<es::EsTensorHolder> inputs;
    for (size_t i = 0; i < subgraph_inputs.size(); ++i) {
        const auto all_inputs = subgraph_inputs[i].GetAllInputs();
        if (all_inputs.empty()) {
            OPS_LOG_E(kPassName.c_str(), "Subgraph input %zu is empty.", i);
            return {};
        }
        TensorDesc tensor_desc;
        const auto match_node = all_inputs.at(0);
        if (match_node.node.GetInputDesc(match_node.index, tensor_desc) != GRAPH_SUCCESS) {
            OPS_LOG_E(kPassName.c_str(), "Get subgraph input %zu desc failed.", i);
            return {};
        }
        auto data = graph_builder.CreateInput(
            static_cast<int64_t>(i), ("replacement_input_" + std::to_string(i)).c_str(), tensor_desc.GetDataType(),
            tensor_desc.GetFormat(), tensor_desc.GetShape().GetDims());
        inputs.emplace_back(data);
    }
    return inputs;
}

Status InferShape(const GraphUniqPtr& replace_graph, const std::vector<SubgraphInput>& subgraph_inputs)
{
    std::vector<Shape> input_shapes;
    for (const auto& subgraph_input : subgraph_inputs) {
        const auto all_inputs = subgraph_input.GetAllInputs();
        if (all_inputs.empty()) {
            return FAILED;
        }
        TensorDesc tensor_desc;
        const auto match_node = all_inputs.at(0);
        if (match_node.node.GetInputDesc(match_node.index, tensor_desc) != GRAPH_SUCCESS) {
            return FAILED;
        }
        input_shapes.emplace_back(tensor_desc.GetShape());
    }
    return GeUtils::InferShape(*replace_graph, input_shapes);
}

// V2 IR definition APIs (IrDefInputsV2/IrDefOutputsV2/IrDefAttrsV2) use pimpl (IrInputDefV2)
// with strings constructed inside the GE library, avoiding ABI mismatch issues that V1 APIs
// (IrDefInputs/IrDefOutputs/IrDefAttrs) have due to std::string layout differences across
// _GLIBCXX_USE_CXX11_ABI settings. V2 is available since CANN 9.2.0.
#if defined(CANN_MAJOR) && defined(CANN_MINOR)
#define NN_HAS_V2_IR_API ((CANN_MAJOR > 9) || (CANN_MAJOR == 9 && CANN_MINOR >= 1))
#else
#define NN_HAS_V2_IR_API 0
#endif

// Build a two-input one-output element-wise node (Mul/Sub) with CompliantNodeBuilder.
es::EsTensorHolder BuildBinaryNode(es::EsGraphBuilder& graph_builder, const es::EsTensorHolder& input0,
                                   const es::EsTensorHolder& input1, const char* op_type)
{
    auto* c_builder = graph_builder.GetCGraphBuilder();
    auto* graph = c_builder->GetGraph();
#if NN_HAS_V2_IR_API
    GNode node = es::CompliantNodeBuilder(graph)
                     .OpType(op_type)
                     .Name(c_builder->GenerateNodeName(op_type).GetString())
                     .IrDefInputsV2({{"x1", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                                     {"x2", es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
                     .IrDefOutputsV2({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                     .Build();
#else
    GNode node = es::CompliantNodeBuilder(graph)
                     .OpType(op_type)
                     .Name(c_builder->GenerateNodeName(op_type).GetString())
                     .IrDefInputs({{"x1", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                                   {"x2", es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
                     .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                     .Build();
#endif
    ES_ASSERT_GRAPH_SUCCESS(es::AddEdgeAndUpdatePeerDesc(*graph, *input0.GetProducer(), input0.GetProducerOutIndex(),
                                                         node, kBinaryInputX1Idx));
    ES_ASSERT_GRAPH_SUCCESS(es::AddEdgeAndUpdatePeerDesc(*graph, *input1.GetProducer(), input1.GetProducerOutIndex(),
                                                         node, kBinaryInputX2Idx));
    return es::EsTensorHolder(c_builder->GetTensorHolderFromNode(node, kNodeOutputIdx));
}

// Build a ReduceSum node used inside a pattern. axes is an internal Const node (CreateConst).
es::EsTensorHolder BuildPatternReduceSum(es::EsGraphBuilder& graph_builder, const es::EsTensorHolder& input)
{
    auto axes = graph_builder.CreateConst(std::vector<int64_t>{kReduceLastAxis}, std::vector<int64_t>{kAxesShapeDim});
    auto* c_builder = graph_builder.GetCGraphBuilder();
    auto* graph = c_builder->GetGraph();
#if NN_HAS_V2_IR_API
    GNode node = es::CompliantNodeBuilder(graph)
                     .OpType("ReduceSum")
                     .Name(c_builder->GenerateNodeName("ReduceSum").GetString())
                     .IrDefInputsV2({{"x", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                                     {"axes", es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
                     .IrDefOutputsV2({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                     .IrDefAttrsV2(
                         {{"keep_dims", es::CompliantNodeBuilder::kEsAttrOptional, "Bool", es::CreateFrom(true)},
                          {"noop_with_empty_axes", es::CompliantNodeBuilder::kEsAttrOptional, "Bool",
                           es::CreateFrom(true)}})
                     .Build();
#else
    GNode node = es::CompliantNodeBuilder(graph)
                     .OpType("ReduceSum")
                     .Name(c_builder->GenerateNodeName("ReduceSum").GetString())
                     .IrDefInputs({{"x", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                                   {"axes", es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
                     .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                     .IrDefAttrs(
                         {{"keep_dims", es::CompliantNodeBuilder::kEsAttrOptional, "Bool", es::CreateFrom(true)},
                          {"noop_with_empty_axes", es::CompliantNodeBuilder::kEsAttrOptional, "Bool",
                           es::CreateFrom(true)}})
                     .Build();
#endif
    ES_ASSERT_GRAPH_SUCCESS(es::AddEdgeAndUpdatePeerDesc(*graph, *input.GetProducer(), input.GetProducerOutIndex(),
                                                         node, kReduceSumInputXIdx));
    ES_ASSERT_GRAPH_SUCCESS(es::AddEdgeAndUpdatePeerDesc(*graph, *axes.GetProducer(), axes.GetProducerOutIndex(), node,
                                                         kReduceSumAxesInputIdx));
    return es::EsTensorHolder(c_builder->GetTensorHolderFromNode(node, kNodeOutputIdx));
}

// Pattern: output = x2 * x1 * (grad - ReduceSum(grad * x1))
PatternUniqPtr MakePatternSoftmaxGradExt(const std::string& pass_name)
{
    auto graph_builder = es::EsGraphBuilder(pass_name.c_str());
    auto input0 = graph_builder.CreateInput(kSubgraphInputGrad, "grad");
    auto input1 = graph_builder.CreateInput(kSubgraphInputX1, "x1");
    auto input2 = graph_builder.CreateInput(kSubgraphInputX2, "x2");

    auto mul = BuildBinaryNode(graph_builder, input0, input1, "Mul");
    auto sum = BuildPatternReduceSum(graph_builder, mul);
    auto sub = BuildBinaryNode(graph_builder, input0, sum, "Sub");
    auto mul1 = BuildBinaryNode(graph_builder, input2, input1, "Mul");
    auto mul_grad = BuildBinaryNode(graph_builder, mul1, sub, "Mul");

    auto graph = graph_builder.BuildAndReset({mul_grad});
    auto pattern = std::make_unique<Pattern>(std::move(*graph));
    pattern->CaptureTensor({*sum.GetProducer(), kNodeOutputIdx});
    return pattern;
}

// v2 patterns (4 variants), differing only in the input order of mul1 and mulGrad:
//   variant 0: mul1 = Mul(input1, sub); mulGrad = Mul(mul1, input2)
//   variant 1: mul1 = Mul(sub, input1); mulGrad = Mul(mul1, input2)
//   variant 2: mul1 = Mul(input1, sub); mulGrad = Mul(input2, mul1)
//   variant 3: mul1 = Mul(sub, input1); mulGrad = Mul(input2, mul1)
PatternUniqPtr MakePatternSoftmaxGradExtV2(const std::string& pass_name, int32_t variant)
{
    std::string builder_name = pass_name + "_" + std::to_string(variant);
    auto graph_builder = es::EsGraphBuilder(builder_name.c_str());
    auto input0 = graph_builder.CreateInput(kSubgraphInputGrad, "grad");
    auto input1 = graph_builder.CreateInput(kSubgraphInputX1, "x1");
    auto input2 = graph_builder.CreateInput(kSubgraphInputX2, "x2");

    auto mul = BuildBinaryNode(graph_builder, input0, input1, "Mul");
    auto sum = BuildPatternReduceSum(graph_builder, mul);
    auto sub = BuildBinaryNode(graph_builder, input0, sum, "Sub");

    es::EsTensorHolder mul1;
    es::EsTensorHolder mul_grad;
    switch (variant) {
        case 0:
            mul1 = BuildBinaryNode(graph_builder, input1, sub, "Mul");
            mul_grad = BuildBinaryNode(graph_builder, mul1, input2, "Mul");
            break;
        case 1:
            mul1 = BuildBinaryNode(graph_builder, sub, input1, "Mul");
            mul_grad = BuildBinaryNode(graph_builder, mul1, input2, "Mul");
            break;
        case 2:
            mul1 = BuildBinaryNode(graph_builder, input1, sub, "Mul");
            mul_grad = BuildBinaryNode(graph_builder, input2, mul1, "Mul");
            break;
        default:
            mul1 = BuildBinaryNode(graph_builder, sub, input1, "Mul");
            mul_grad = BuildBinaryNode(graph_builder, input2, mul1, "Mul");
            break;
    }

    auto graph = graph_builder.BuildAndReset({mul_grad});
    auto pattern = std::make_unique<Pattern>(std::move(*graph));
    pattern->CaptureTensor({*sum.GetProducer(), kNodeOutputIdx});
    return pattern;
}

GraphUniqPtr SoftmaxGradExtReplacementCommon(const std::unique_ptr<MatchResult>& match_result,
                                             const std::string& pass_name)
{
    OPS_LOG_D(pass_name.c_str(), "Enter Replacement for %s.", pass_name.c_str());

    GNode sum_node;
    OP_LOGE_IF(!GetCapturedSumNode(match_result, sum_node), nullptr, pass_name.c_str(),
               "Get captured ReduceSum node failed.");

    int64_t axis_value = 0;
    bool keep_dims = true;
    OP_LOGE_IF(!GetAxisFromReduceSum(sum_node, axis_value, keep_dims), nullptr, pass_name.c_str(),
               "Failed to get axis/keep_dims from ReduceSum.");

    std::vector<SubgraphInput> subgraph_inputs;
    match_result->ToSubgraphBoundary()->GetAllInputs(subgraph_inputs);
    OP_LOGE_IF(subgraph_inputs.size() < kMinSubgraphInputCount, nullptr, pass_name.c_str(),
               "Subgraph inputs size %zu is less than %zu.", subgraph_inputs.size(), kMinSubgraphInputCount);

    auto graph_builder = es::EsGraphBuilder("replacement");
    auto replacement_inputs = CreateReplacementInputs(graph_builder, subgraph_inputs);
    OP_LOGE_IF(replacement_inputs.size() < kMinSubgraphInputCount, nullptr, pass_name.c_str(),
               "Create replacement inputs failed.");

    // SoftmaxGradExt(grad, x1, x2): grad=input0, x1=input1, x2=input2.
    auto softmax_grad_ext = es::SoftmaxGradExt(replacement_inputs[kSubgraphInputGrad],
                                               replacement_inputs[kSubgraphInputX1],
                                               replacement_inputs[kSubgraphInputX2], axis_value, keep_dims);

    GraphUniqPtr replace_graph = graph_builder.BuildAndReset({softmax_grad_ext});
    if (InferShape(replace_graph, subgraph_inputs) != SUCCESS) {
        OPS_LOG_E(pass_name.c_str(), "InferShape for replacement failed.");
        return nullptr;
    }
    return replace_graph;
}
} // namespace

// ==================== SoftmaxGradExtFusionPass ====================

std::vector<PatternUniqPtr> SoftmaxGradExtFusionPass::Patterns()
{
    OPS_LOG_D(kPassName.c_str(), "Enter Patterns for SoftmaxGradExtFusionPass.");
    std::vector<PatternUniqPtr> patterns;
    patterns.emplace_back(MakePatternSoftmaxGradExt(kPassName));
    return patterns;
}

bool SoftmaxGradExtFusionPass::MeetRequirements(const std::unique_ptr<MatchResult>& match_result)
{
    OPS_LOG_D(kPassName.c_str(), "Enter MeetRequirements for SoftmaxGradExtFusionPass.");
    if (!IsTargetPlatform()) {
        return false;
    }
    if (!CheckInputsShapeValid(match_result)) {
        return false;
    }
    return true;
}

GraphUniqPtr SoftmaxGradExtFusionPass::Replacement(const std::unique_ptr<MatchResult>& match_result)
{
    return SoftmaxGradExtReplacementCommon(match_result, kPassName);
}

// ==================== SoftmaxGradExtV2FusionPass ====================

std::vector<PatternUniqPtr> SoftmaxGradExtV2FusionPass::Patterns()
{
    OPS_LOG_D(kPassNameV2.c_str(), "Enter Patterns for SoftmaxGradExtV2FusionPass.");
    std::vector<PatternUniqPtr> patterns;
    for (int32_t i = 0; i < kPatternV2VariantCount; ++i) {
        patterns.emplace_back(MakePatternSoftmaxGradExtV2(kPassNameV2, i));
    }
    return patterns;
}

bool SoftmaxGradExtV2FusionPass::MeetRequirements(const std::unique_ptr<MatchResult>& match_result)
{
    OPS_LOG_D(kPassNameV2.c_str(), "Enter MeetRequirements for SoftmaxGradExtV2FusionPass.");
    if (!IsTargetPlatform()) {
        return false;
    }
    if (!CheckInputsShapeValid(match_result)) {
        return false;
    }
    return true;
}

GraphUniqPtr SoftmaxGradExtV2FusionPass::Replacement(const std::unique_ptr<MatchResult>& match_result)
{
    return SoftmaxGradExtReplacementCommon(match_result, kPassNameV2);
}

REG_FUSION_PASS(SoftmaxGradExtFusionPass).Stage(CustomPassStage::kAfterInferShape);
REG_FUSION_PASS(SoftmaxGradExtV2FusionPass).Stage(CustomPassStage::kAfterInferShape);

} // namespace ops

/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "layer_norm_remove_broadcast_fusion_pass.h"

#include <algorithm>
#include <set>
#include <string>
#include <vector>

#include "common/inc/error_util.h"
#include "es_math_ops.h"
#include "es_nn_ops.h"
#include "ge/ge_utils.h"
#include "platform/platform_info.h"

namespace ops {
namespace {
const std::string kPassName = "LayerNormRemoveBroadcastFusionPass";

const int64_t kLNCaptureIdx = 0l;
const int64_t kGammaBrcCaptureIdx = 1l;
const int64_t kBetaBrcCaptureIdx = 2l;

const int64_t kSubgraphInputX = 0l;
const int64_t kSubgraphInputGamma = 1l;
const int64_t kSubgraphInputBeta = 2l;
const int64_t kSubgraphInputShape = 3l;
const float32_t EPSILON = 0.00001f;

bool GetCapturedNode(const std::unique_ptr<MatchResult>& match_result, int64_t index, GNode& node)
{
    NodeIo node_io;
    OP_LOGE_IF(
        match_result->GetCapturedTensor(index, node_io) != SUCCESS, false, kPassName.c_str(),
        "get captured node failed, index is %ld.", index);
    node = node_io.node;
    return true;
}

bool IsDynamicShape(const Shape& shape)
{
    const auto dims = shape.GetDims();
    return std::any_of(dims.begin(), dims.end(), [](int64_t dim) { return dim < 0; });
}

bool IsScalar(const Shape& shape)
{
    return shape.GetDimNum() == 0;
}

Status InferShape(const std::unique_ptr<Graph>& replace_graph, const std::vector<SubgraphInput>& subgraph_inputs)
{
    std::vector<Shape> input_shapes;
    for (const auto& subgraph_input : subgraph_inputs) {
        const auto all_inputs = subgraph_input.GetAllInputs();
        if (all_inputs.empty()) {
            OPS_LOG_E(kPassName.c_str(), "subgraph input is empty.");
            return FAILED;
        }
        TensorDesc tensor_desc;
        const auto match_node = all_inputs.at(0);
        if (match_node.node.GetInputDesc(match_node.index, tensor_desc) != GRAPH_SUCCESS) {
            OPS_LOG_E(kPassName.c_str(), "get subgraph input desc failed.");
            return FAILED;
        }
        input_shapes.emplace_back(tensor_desc.GetShape());
    }
    return GeUtils::InferShape(*replace_graph, input_shapes);
}

PatternUniqPtr MakePattern()
{
    auto graph_builder = es::EsGraphBuilder(kPassName.c_str());
    auto [x, gamma, beta, shape] = graph_builder.CreateInputs<4>();
    auto gamma_brc = es::BroadcastTo(gamma, shape);
    auto beta_brc = es::BroadcastTo(beta, shape);
    auto layer_norm = es::LayerNorm(x, gamma_brc, beta_brc, 0, 0, EPSILON);

    std::vector<es::EsTensorHolder> outputs = {layer_norm.y, layer_norm.mean, layer_norm.variance};
    auto graph = graph_builder.BuildAndReset(outputs);
    auto pattern = std::make_unique<Pattern>(std::move(*graph));
    pattern->CaptureTensor({*layer_norm.y.GetProducer(), 0})
        .CaptureTensor({*gamma_brc.GetProducer(), 0})
        .CaptureTensor({*beta_brc.GetProducer(), 0});
    return pattern;
}

bool IsSupportedPlatform()
{
    fe::PlatformInfo platform_info;
    fe::OptionalInfo optional_info;
    OP_LOGE_IF(
        fe::PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platform_info, optional_info) != SUCCESS,
        false, kPassName.c_str(), "Get platform_info failed.");

    const std::string soc = platform_info.str_info.short_soc_version;
    OPS_LOG_D(kPassName.c_str(), "Platform short soc: %s", soc.c_str());
    const static std::set<std::string> support_soc = {"Ascend950", "MC62"};
    if (support_soc.count(soc) == 0) {
        OPS_LOG_D(kPassName.c_str(), "Platform is not support!");
        return false;
    }
    return true;
}

bool CheckNormAxis(const GNode& ln_node, const Shape& x_shape, size_t norm_axis)
{
    int64_t begin_norm_axis = 0;
    OP_LOGE_IF(
        ln_node.GetAttr("begin_norm_axis", begin_norm_axis) != GRAPH_SUCCESS, false, kPassName.c_str(),
        "get LayerNorm begin_norm_axis attr failed.");
    const int64_t x_rank = static_cast<int64_t>(x_shape.GetDimNum());
    const int64_t real_norm_axis = begin_norm_axis < 0 ? begin_norm_axis + x_rank : begin_norm_axis;
    if (real_norm_axis < 0 || real_norm_axis >= x_rank) {
        OPS_LOG_D(kPassName.c_str(), "LayerNorm begin_norm_axis is out of range.");
        return false;
    }
    if (static_cast<size_t>(real_norm_axis) != norm_axis) {
        OPS_LOG_D(kPassName.c_str(), "LayerNorm begin_norm_axis does not match gamma shape.");
        return false;
    }
    return true;
}

bool CheckRemoveBrc(
    const GNode& ln_node, const GNode& gamma_brc_node, const GNode& beta_brc_node)
{
    TensorDesc x_desc;
    TensorDesc gamma_desc;
    TensorDesc beta_desc;
    OP_LOGE_IF(
        ln_node.GetInputDesc(0, x_desc) != GRAPH_SUCCESS, false, kPassName.c_str(),
        "get LayerNorm x input desc failed.");
    OP_LOGE_IF(
        gamma_brc_node.GetInputDesc(0, gamma_desc) != GRAPH_SUCCESS, false, kPassName.c_str(),
        "get gamma input desc failed.");
    OP_LOGE_IF(
        beta_brc_node.GetInputDesc(0, beta_desc) != GRAPH_SUCCESS, false, kPassName.c_str(),
        "get beta input desc failed.");

    const Shape x_shape = x_desc.GetShape();
    const Shape gamma_shape = gamma_desc.GetShape();
    const Shape beta_shape = beta_desc.GetShape();
    if (IsDynamicShape(x_shape) || IsDynamicShape(gamma_shape) || IsDynamicShape(beta_shape)) {
        OPS_LOG_D(kPassName.c_str(), "Only support static x/gamma/beta shape.");
        return false;
    }
    if (IsScalar(gamma_shape) || IsScalar(beta_shape)) {
        OPS_LOG_D(kPassName.c_str(), "Do not remove BroadcastTo for scalar gamma/beta.");
        return false;
    }
    if (gamma_shape.GetDims() != beta_shape.GetDims()) {
        OPS_LOG_D(kPassName.c_str(), "Only remove BroadcastTo when gamma shape equals beta shape.");
        return false;
    }
    if (gamma_desc.GetDataType() != beta_desc.GetDataType()) {
        OPS_LOG_D(kPassName.c_str(), "Only remove BroadcastTo when gamma dtype equals beta dtype.");
        return false;
    }
    if (gamma_shape.GetDimNum() > x_shape.GetDimNum()) {
        OPS_LOG_D(kPassName.c_str(), "Gamma rank cannot be larger than x rank.");
        return false;
    }
    const size_t norm_axis = x_shape.GetDimNum() - gamma_shape.GetDimNum();
    if (!CheckNormAxis(ln_node, x_shape, norm_axis)) {
        return false;
    }
    for (size_t i = 0; i < gamma_shape.GetDimNum(); ++i) {
        if (gamma_shape.GetDim(i) != x_shape.GetDim(norm_axis + i)) {
            OPS_LOG_D(kPassName.c_str(), "Gamma shape must equal x suffix shape.");
            return false;
        }
    }
    return true;
}

Status CalcNormAxis(const GNode& ln_node, const GNode& gamma_brc_node, int64_t& norm_axis)
{
    TensorDesc x_desc;
    TensorDesc gamma_desc;
    OP_LOGE_IF(
        ln_node.GetInputDesc(0, x_desc) != GRAPH_SUCCESS, FAILED, kPassName.c_str(),
        "get LayerNorm x input desc failed.");
    OP_LOGE_IF(
        gamma_brc_node.GetInputDesc(0, gamma_desc) != GRAPH_SUCCESS, FAILED, kPassName.c_str(),
        "get gamma input desc failed.");

    norm_axis = static_cast<int64_t>(x_desc.GetShape().GetDimNum() - gamma_desc.GetShape().GetDimNum());
    return SUCCESS;
}

std::vector<es::EsTensorHolder> CreateReplacementInputs(
    es::EsGraphBuilder& graph_builder, const std::vector<SubgraphInput>& subgraph_inputs)
{
    std::vector<es::EsTensorHolder> inputs;
    for (size_t i = 0; i < subgraph_inputs.size(); ++i) {
        const auto all_inputs = subgraph_inputs[i].GetAllInputs();
        if (all_inputs.empty()) {
            OPS_LOG_E(kPassName.c_str(), "subgraph input is empty.");
            return {};
        }
        TensorDesc tensor_desc;
        const auto match_node = all_inputs.at(0);
        if (match_node.node.GetInputDesc(match_node.index, tensor_desc) != GRAPH_SUCCESS) {
            OPS_LOG_E(kPassName.c_str(), "get subgraph input desc failed.");
            return {};
        }
        auto data = graph_builder.CreateInput(
            static_cast<int64_t>(i), ("replacement_input_" + std::to_string(i)).c_str(), tensor_desc.GetDataType(),
            tensor_desc.GetFormat(), tensor_desc.GetShape().GetDims());
        inputs.emplace_back(data);
    }
    return inputs;
}

}

std::vector<PatternUniqPtr> LayerNormRemoveBroadcastFusionPass::Patterns()
{
    std::vector<PatternUniqPtr> patterns;
    patterns.emplace_back(MakePattern());
    return patterns;
}

bool LayerNormRemoveBroadcastFusionPass::MeetRequirements(const std::unique_ptr<MatchResult>& match_result)
{
    OPS_LOG_D(kPassName.c_str(), "Enter LayerNormRemoveBroadcastFusionPass MeetRequirements.");
    if (!IsSupportedPlatform()) {
        OPS_LOG_D(kPassName.c_str(), "Platform is not support.");
        return false;
    }

    GNode ln_node;
    if (!GetCapturedNode(match_result, kLNCaptureIdx, ln_node)) {
        OPS_LOG_D(kPassName.c_str(), "Get captured LayerNorm node failed.");
        return false;
    }

    GNode gamma_brc_node;
    if (!GetCapturedNode(match_result, kGammaBrcCaptureIdx, gamma_brc_node)) {
        OPS_LOG_D(kPassName.c_str(), "Get captured gamma BroadcastTo node failed.");
        return false;
    }

    GNode beta_brc_node;
    if (!GetCapturedNode(match_result, kBetaBrcCaptureIdx, beta_brc_node)) {
        OPS_LOG_D(kPassName.c_str(), "Get captured beta BroadcastTo node failed.");
        return false;
    }

    if (!CheckRemoveBrc(ln_node, gamma_brc_node, beta_brc_node)) {
        OPS_LOG_D(kPassName.c_str(), "Check remove BroadcastTo condition failed.");
        return false;
    }
    return true;
}

std::unique_ptr<Graph> LayerNormRemoveBroadcastFusionPass::Replacement(const std::unique_ptr<MatchResult>& match_result)
{
    OPS_LOG_D(kPassName.c_str(), "Enter Replacement for LayerNormRemoveBroadcastFusionPass.");
    GNode ln_node;
    OP_LOGE_IF(
        !GetCapturedNode(match_result, kLNCaptureIdx, ln_node), nullptr, kPassName.c_str(),
        "Get captured LayerNorm node failed.");

    GNode gamma_brc_node;
    OP_LOGE_IF(
        !GetCapturedNode(match_result, kGammaBrcCaptureIdx, gamma_brc_node), nullptr, kPassName.c_str(),
        "Get captured gamma BroadcastTo node failed.");

    std::vector<SubgraphInput> subgraph_inputs;
    match_result->ToSubgraphBoundary()->GetAllInputs(subgraph_inputs);
    if (subgraph_inputs.size() <= kSubgraphInputShape) {
        OPS_LOG_E(kPassName.c_str(), "Subgraph inputs are fewer than expected.");
        return nullptr;
    }

    int64_t norm_axis = 0;
    if (CalcNormAxis(ln_node, gamma_brc_node, norm_axis) != SUCCESS) {
        return nullptr;
    }

    float32_t epsilon = EPSILON;
    OP_LOGW_IF(ln_node.GetAttr("epsilon", epsilon) != SUCCESS, kPassName.c_str(), "Get epsilon attr failed.");

    auto graph_builder = es::EsGraphBuilder("replacement");
    auto replacement_inputs = CreateReplacementInputs(graph_builder, subgraph_inputs);
    if (replacement_inputs.size() != subgraph_inputs.size()) {
        return nullptr;
    }
    auto layer_norm = es::LayerNorm(
        replacement_inputs[kSubgraphInputX], replacement_inputs[kSubgraphInputGamma],
        replacement_inputs[kSubgraphInputBeta], norm_axis, norm_axis, epsilon);
    OP_LOGE_IF(
        layer_norm.y.AddControlEdge({replacement_inputs[kSubgraphInputShape]}) != SUCCESS, nullptr, kPassName.c_str(),
        "Add control edge from broadcast shape to replacement LayerNorm failed.");

    std::vector<es::EsTensorHolder> outputs = {layer_norm.y, layer_norm.mean, layer_norm.variance};
    auto graph = graph_builder.BuildAndReset(outputs);
    if (InferShape(graph, subgraph_inputs) != SUCCESS) {
        OPS_LOG_E(kPassName.c_str(), "Infershape for replacement failed.");
        return nullptr;
    }
    return graph;
}

REG_FUSION_PASS(LayerNormRemoveBroadcastFusionPass).Stage(CustomPassStage::kAfterInferShape);
}

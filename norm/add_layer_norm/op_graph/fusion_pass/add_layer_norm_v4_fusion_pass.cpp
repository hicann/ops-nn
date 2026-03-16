/**
* Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "add_layer_norm_v4_fusion_pass.h"
#include "common/inc/error_util.h"
#include "platform/platform_info.h"
#include "es_nn_ops.h"
#include "es_math_ops.h"
#include "add_layer_norm_utils.h"

namespace ops {
namespace {
const std::string kPassName = "AddLayerNormV4FusionPass";
const std::string kS2PatternName = "S2";
const int64_t kLayerNormV4CaptureIdx = 0l;
const int64_t kAdd1CaptureIdx = 1l;
const int64_t kAdd2CaptureIdx = 2l;
const int64_t kCast1CaptureIdx = 3l;

const int64_t kGammaInputIndex = 2;
const int64_t kBetaInputIndex = 3;
const bool KGammaExist = true;
const bool KGammaNotExist = false;
const bool KBetaExist = true;
const bool KBetaNotExist = false;
const bool KCast1AsOutput = true;
const bool KCast1NotAsOutput = false;

bool GetOptionalInputIfExist(const GNode& node, int32_t input_index, TensorDesc& input_desc)
{
    const auto node_2_port_index = node.GetInDataNodesAndPortIndexs(input_index);
    if (node_2_port_index.first == nullptr) {
        return false;
    }
    if (node.GetInputDesc(input_index, input_desc) != GRAPH_SUCCESS) {
        return false;
    }
    return true;
}

bool IsOptionalInputDescAsExpect(const TensorDesc& input_desc, const Shape& x1_shape)
{
    const auto shape = input_desc.GetShape();
    if (IsDynamicShape(shape)) {
        return false;
    }
    if (shape.GetDim(0) != x1_shape.GetDims().back()) {
        return false;
    }
    if (shape.GetDimNum() != 1) {
        return false;
    }
    if (input_desc.GetDataType() != DT_FLOAT) {
        return false;
    }
    return true;
}

PatternUniqPtr MakePatternForLayernormV4(bool is_gamma_exist, bool is_beta_exist, bool make_cast1_as_output)
{
    auto pattern_name = make_cast1_as_output ? "S2" : "S1";
    auto graph_builder = es::EsGraphBuilder(pattern_name);
    auto [x1, x2, bias, normalized_shape] = graph_builder.CreateInputs<4>();
    auto add1 = x2 + bias;
    auto add2 = x1 + add1;
    auto cast1 = es::Cast(add2, DT_FLOAT);
    int64_t cur_data_index = 3;
    es::EsTensorHolder gamma = is_gamma_exist ? graph_builder.CreateInput(++cur_data_index) : nullptr;
    es::EsTensorHolder beta = is_beta_exist ? graph_builder.CreateInput(++cur_data_index) : nullptr;

    auto [y, mean, rstd] = es::LayerNormV4(cast1, normalized_shape, gamma, beta);
    y = es::Cast(y, DT_FLOAT16);

    std::vector<es::EsTensorHolder> pattern_outputs = {y, mean, rstd};
    if (make_cast1_as_output) {
        pattern_outputs.emplace_back(cast1);
    }
    auto graph = graph_builder.BuildAndReset(pattern_outputs);
    auto pattern = std::make_unique<Pattern>(std::move(*graph));
    pattern->CaptureTensor({*mean.GetProducer(), 0})
        .CaptureTensor({*add1.GetProducer(), 0})
        .CaptureTensor({*add2.GetProducer(), 0})
        .CaptureTensor({*cast1.GetProducer(), 0});
    return pattern;
}

bool IsAscend950Platform()
{
    fe::PlatformInfo platform_info;
    fe::OptionalInfo optional_info;
    OP_LOGE_IF(
        fe::PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platform_info, optional_info) != SUCCESS,
        false, kPassName.c_str(), "Get platform_info failed.");

    const std::string soc = platform_info.str_info.short_soc_version;
    OPS_LOG_D(kPassName.c_str(), "Platform short soc: %s", soc.c_str());
    if (soc != "Ascend950") {
        OPS_LOG_D(kPassName.c_str(), "Platform is not support, layerNormV4 only work on Ascend950.");
        return false;
    }
    return true;
}

Status GetLNV4BeginNormAxis(const GNode& layer_norm_node, int32_t& begin_norm_axis)
{
    TensorDesc x_input_desc;
    layer_norm_node.GetInputDesc(0, x_input_desc);
    const size_t x_rank = x_input_desc.GetShape().GetDimNum();

    TensorDesc normalized_shape_input_desc;
    layer_norm_node.GetInputDesc(1, normalized_shape_input_desc);
    auto normalized_shape_shape = normalized_shape_input_desc.GetShape();
    const int64_t normalized_shape_rank = (normalized_shape_shape.GetDimNum() == 0) ?
                                              1 :
                                              normalized_shape_shape.GetDim(0);
    if (normalized_shape_rank <= 0) {
        OP_LOGE(kPassName.c_str(), "normalized_shape can not be -1");
        return FAILED;
    }
    begin_norm_axis = x_rank - normalized_shape_rank;
    return SUCCESS;
}

bool IsBeginNormAxisRight(const std::unique_ptr<MatchResult>& match_result)
{
    NodeIo layer_norm_v4_node;
    OP_LOGE_IF(
        match_result->GetCapturedTensor(kLayerNormV4CaptureIdx, layer_norm_v4_node) != SUCCESS, false,
        kPassName.c_str(),
        "get layernorm node failed.");
    NodeIo add2_node;
    OP_LOGE_IF(
        match_result->GetCapturedTensor(kAdd2CaptureIdx, add2_node) != SUCCESS, false, kPassName.c_str(),
        "get add2 node failed.");
    int32_t begin_norm_axis = -2;
    if (GetLNV4BeginNormAxis(layer_norm_v4_node.node, begin_norm_axis) != SUCCESS) {
        OPS_LOG_D(kPassName.c_str(), "Faile to get begin_norm_axis");
        return false;
    }
    TensorDesc add2_input0_tensor_desc;
    add2_node.node.GetInputDesc(0, add2_input0_tensor_desc);
    if (begin_norm_axis != -1 && begin_norm_axis != ((int64_t)add2_input0_tensor_desc.GetShape().GetDimNum() - 1)) {
        OPS_LOG_D(kPassName.c_str(), "Attr begin_norm_axis must be -1 or last axis, not change.");
        return false;
    }
    return true;
}

bool IsAdd1ShapeDataTypeValid(const GNode& add1_node, const AscendString& add1_node_name)
{
    TensorDesc add1_input_0_tensor_desc;
    TensorDesc add1_input_1_tensor_desc;
    add1_node.GetInputDesc(0, add1_input_0_tensor_desc);
    add1_node.GetInputDesc(1, add1_input_1_tensor_desc);

    const auto add1_input0_shape = add1_input_0_tensor_desc.GetShape();
    const auto add1_input1_shape = add1_input_1_tensor_desc.GetShape();

    if (add1_input0_shape.GetDimNum() < add1_input1_shape.GetDimNum()) {
        OPS_LOG_D(kPassName.c_str(), "%s only support dim of bias smaller than dim of x.", add1_node_name.GetString());
        return false;
    }
    if (IsScaler(add1_input0_shape) || IsScaler(add1_input1_shape)) {
        OPS_LOG_D(kPassName.c_str(), "%s inputs not support scaler.", add1_node_name.GetString());
        return false;
    }
    if (IsDynamicShape(add1_input0_shape) || IsDynamicShape(add1_input1_shape)) {
        OPS_LOG_D(kPassName.c_str(), "%s not support dynamic input.", add1_node_name.GetString());
        return false;
    }
    if (add1_input0_shape.GetDims().back() != add1_input1_shape.GetDims().back()) {
        OPS_LOG_D(kPassName.c_str(), "Only support %s inputs have same last dim.", add1_node_name.GetString());
        return false;
    }
    if (add1_input_0_tensor_desc.GetDataType() != add1_input_1_tensor_desc.GetDataType()) {
        OPS_LOG_D(kPassName.c_str(), "Only support %s inputs have same dtype.", add1_node_name.GetString());
        return false;
    }
    const std::set<DataType> supported_datatype = {DT_FLOAT16, DT_BF16};
    if (supported_datatype.count(add1_input_0_tensor_desc.GetDataType()) == 0) {
        OPS_LOG_D(
            kPassName.c_str(), "Only support %s inputs fp16/bf16, but now is %d", add1_node_name.GetString(),
            add1_input_0_tensor_desc.GetDataType());
        return false;
    }
    return true;
}

bool IsGammaBetaTensorDescAsExpect(const GNode& layer_norm_v4_node, const Shape& x1_shape)
{
    bool is_gamma_exist = false;
    TensorDesc gamma_tensor_desc;
    if (GetOptionalInputIfExist(layer_norm_v4_node, kGammaInputIndex, gamma_tensor_desc)) {
        is_gamma_exist = true;
        if (!IsOptionalInputDescAsExpect(gamma_tensor_desc, x1_shape)) {
            return false;
        }
    }
    bool is_beta_exist = false;
    TensorDesc beta_tensor_desc;
    if (GetOptionalInputIfExist(layer_norm_v4_node, kBetaInputIndex, beta_tensor_desc)) {
        is_beta_exist = true;
        if (!IsOptionalInputDescAsExpect(beta_tensor_desc, x1_shape)) {
            return false;
        }
    }
    if (is_gamma_exist && is_beta_exist) {
        OP_LOGE_IF(
            gamma_tensor_desc.GetShape().GetDims() != beta_tensor_desc.GetShape().GetDims(), false,
            kPassName.c_str(), "Only support gamma/beta have same shape.");
        OP_LOGE_IF(
            gamma_tensor_desc.GetDataType() != beta_tensor_desc.GetDataType(), false,
            kPassName.c_str(), "Only support gamma/beta have same dtype.");
    }
    return true;
}

bool IsAllInputShapeDtypeRight(const std::unique_ptr<MatchResult>& match_result)
{
    NodeIo add1OutNode;
    NodeIo add2OutNode;
    OP_LOGE_IF(
        match_result->GetCapturedTensor(kAdd1CaptureIdx, add1OutNode)!= SUCCESS, false, kPassName,
        "Failed to GetCaptrue tensor");
    OP_LOGE_IF(
        match_result->GetCapturedTensor(kAdd2CaptureIdx, add2OutNode) != SUCCESS, false, kPassName,
        "Failed to GetCaptrue tensor");
    auto add1_node = add1OutNode.node;
    auto add2_node = add2OutNode.node;
    AscendString add1_node_name;
    add1_node.GetName(add1_node_name);
    AscendString add2_node_name;
    add2_node.GetName(add2_node_name);

    if (!IsAdd1ShapeDataTypeValid(add1_node, add1_node_name)) {
        OPS_LOG_D(kPassName, "%s shape/dtype invalid", add1_node_name.GetString());
        return false;
    }
    if (!IsAdd2InputValid(add1_node, add2_node, add1_node_name, add2_node_name, kPassName)) {
        OPS_LOG_D(kPassName, "%s shape/dtype invalid", add2_node_name.GetString());
        return false;
    }

    NodeIo layer_norm_v4_output;
    OP_LOGE_IF(
        match_result->GetCapturedTensor(kLayerNormV4CaptureIdx, layer_norm_v4_output) != SUCCESS, false, kPassName,
        "Failed to get layernormv4");
    auto layer_norm_v4_node = layer_norm_v4_output.node;
    AscendString layer_norm_node_name;
    layer_norm_v4_node.GetName(layer_norm_node_name);

    TensorDesc add2_input_0_tensor_desc;
    add2_node.GetInputDesc(0, add2_input_0_tensor_desc);
    if (!IsGammaBetaTensorDescAsExpect(layer_norm_v4_node, add2_input_0_tensor_desc.GetShape())) {
        OPS_LOG_D(kPassName, "beta/gamma in %s is invalid", layer_norm_node_name.GetString());
        return false;
    }
    return true;
}

struct OptionalFillInfo {
    int32_t optional_input_index;
    float32_t fill_value;
};

es::EsTensorHolder GetOrFillOptionalInput(
    const GNode& layer_norm_node, const es::EsTensorHolder& normalize_shape, const OptionalFillInfo& fill_info,
    es::EsGraphBuilder& graph_builder, int64_t& current_data_index)
{
    TensorDesc optional_input_desc;
    if (GetOptionalInputIfExist(layer_norm_node, fill_info.optional_input_index, optional_input_desc)) {
        return graph_builder.CreateInput(++current_data_index);
    }
    auto fill_value = graph_builder.CreateScalar(fill_info.fill_value);
    return es::Fill(normalize_shape, fill_value);
}
} // namespace
std::vector<PatternUniqPtr> AddLayerNormV4FusionPass::Patterns()
{
    std::vector<PatternUniqPtr> pattern_graphs;
    pattern_graphs.emplace_back(MakePatternForLayernormV4(KGammaExist, KBetaExist, KCast1NotAsOutput));
    pattern_graphs.emplace_back(MakePatternForLayernormV4(KGammaNotExist, KBetaNotExist, KCast1NotAsOutput));
    pattern_graphs.emplace_back(MakePatternForLayernormV4(KGammaExist, KBetaNotExist, KCast1NotAsOutput));

    pattern_graphs.emplace_back(MakePatternForLayernormV4(KGammaExist, KBetaExist, KCast1AsOutput));
    pattern_graphs.emplace_back(MakePatternForLayernormV4(KGammaNotExist, KBetaNotExist, KCast1AsOutput));
    pattern_graphs.emplace_back(MakePatternForLayernormV4(KGammaExist, KBetaNotExist, KCast1AsOutput));
    return pattern_graphs;
}

bool AddLayerNormV4FusionPass::MeetRequirements(const std::unique_ptr<MatchResult>& match_result)
{
    OPS_LOG_D(FUSION_OP_TYPE.c_str(), "Enter AddLayerNormV4FusionPass MeetRequirements");
    if (!IsAscend950Platform()) {
        OPS_LOG_D(kPassName.c_str(), "Platform is not support, layerNormV4 only work on Ascend910_95.");
        return false;
    }
    if (!IsBeginNormAxisRight(match_result)) {
        OPS_LOG_D(kPassName.c_str(), "ChaeckAllAttr Failed.");
        return false;
    }
    if (!IsAllInputShapeDtypeRight(match_result)) {
        OPS_LOG_D(kPassName.c_str(), "CheckAllInputShape Failed.");
        return false;
    }
    if (IsCast1HasControlEdge(match_result, kCast1CaptureIdx, kPassName)) {
        OPS_LOG_D(kPassName.c_str(), "Not support cast_1 have control edge.");
        return false;
    }
    return true;
}

std::unique_ptr<Graph> AddLayerNormV4FusionPass::Replacement(const std::unique_ptr<MatchResult>& match_result)
{
    OPS_LOG_D(FUSION_OP_TYPE.c_str(), "Enter AddLayerNormV4FusionPass Replacement");
    auto replace_graph_builder = es::EsGraphBuilder("replacement");

    std::vector<SubgraphInput> subgraph_inputs;
    match_result->ToSubgraphBoundary()->GetAllInputs(subgraph_inputs);
    std::vector<Shape> input_shapes;
    std::vector<DataType> input_dtpyes;
    std::vector<Format> input_formats;
    GetInputsInfo(subgraph_inputs, input_shapes, input_dtpyes, input_formats);

    auto r_x1 = replace_graph_builder.CreateInput(0, "x1", input_dtpyes[0], input_formats[0], input_shapes[0].GetDims());
    auto r_x2 = replace_graph_builder.CreateInput(1, "x2", input_dtpyes[1], input_formats[1], input_shapes[1].GetDims());
    auto r_bias = replace_graph_builder.CreateInput(2, "bias", input_dtpyes[2], input_formats[2], input_shapes[2].GetDims());
    auto r_normalize_shape = replace_graph_builder.CreateInput(3, "normalize_shape", input_dtpyes[3], input_formats[3], input_shapes[3].GetDims());

    int64_t current_data_index = 3;
    NodeIo layer_norm_v4_node;
    if (match_result->GetCapturedTensor(kLayerNormV4CaptureIdx, layer_norm_v4_node) != SUCCESS) {
        OPS_LOG_E(kPassName.c_str(), "get layernorm node failed.");
    }
    auto r_gamma = GetOrFillOptionalInput(
        layer_norm_v4_node.node, r_normalize_shape, {kGammaInputIndex, 1.0f}, replace_graph_builder,
        current_data_index);
    auto r_beta = GetOrFillOptionalInput(
        layer_norm_v4_node.node, r_normalize_shape, {kBetaInputIndex, 0.0f}, replace_graph_builder, current_data_index);
    float32_t epsilon;
    layer_norm_v4_node.node.GetAttr("epsilon", epsilon);
    es::AddLayerNormOutput addlayernorm = es::AddLayerNorm(r_x1, r_x2, r_gamma, r_beta, r_bias, epsilon, true);

    GNode addlayernorm_node = *addlayernorm.y.GetProducer();
    auto layernorm_node_format = input_formats[3];
    UpdateAddLayerNormFormat(layernorm_node_format, addlayernorm_node);

    std::vector<es::EsTensorHolder> replace_outputs = {addlayernorm.y, addlayernorm.mean, addlayernorm.rstd};
    AscendString pattern_name;
    match_result->GetPatternGraph().GetName(pattern_name);
    if (pattern_name.GetString() == kS2PatternName) {
        replace_outputs.emplace_back(es::Cast(addlayernorm.x, DT_FLOAT));
    }
    GraphUniqPtr replaceGraph = replace_graph_builder.BuildAndReset(replace_outputs);

    OP_LOGW_IF(
        replaceGraph->AddControlEdge(*r_normalize_shape.GetProducer(),*addlayernorm.y.GetProducer()) != SUCCESS,
        kPassName.c_str(), "Add control edge failed.");
    OPS_LOG_I(kPassName.c_str(), "Add control edge from normalize_shape to addlayernorm.");

    if (InferShape(replaceGraph, subgraph_inputs) != SUCCESS) {
        OPS_LOG_E(kPassName.c_str(), "Infershape for replacement failed.");
        return nullptr;
    }
    return std::move(replaceGraph);
}

REG_FUSION_PASS(AddLayerNormV4FusionPass).Stage(CustomPassStage::kAfterInferShape);
}
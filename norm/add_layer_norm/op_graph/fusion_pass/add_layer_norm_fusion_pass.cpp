/**
* Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "common/inc/error_util.h"
#include "add_layer_norm_fusion_pass.h"
#include "es_nn_ops.h"
#include "es_math_ops.h"
#include "platform/platform_info.h"
#include "add_layer_norm_utils.h"

using namespace ge;
using namespace fe;
using namespace fusion;

/**
 * @brief Define fusion AddLayerNorm pattern (with bias)
 * @details Only support 910_93 and Ascend950
*              Add                                               Add
 *              |                                                 |
 *             Add                                               Add
 *              |                                                 |
 *             Cast      ==>    AddLayerNorm(with bias)          Cast    ====>  AddLayerNorm(with bias)
 *              |                                               /    \                   |
 *           LayerNorm                                     LayerNorm  op               cast
 *              |                                              |                         |
 *            Cast                                             Cast                     op
 *            场景一                                                          场景二
 */
namespace ops {
namespace {
const std::string kPassName = "AddLayerNormFusionPass";
const std::string kPatternOutputCast = "Cast1";
const std::string kPatternOutputAdd = "Add2";
const std::vector<std::string> kAddInputsName = {"x2", "bias"};
const int64_t kLayerNormCaptureIdx = 0l;
const int64_t kAdd1CaptureIdx = 1l;
const int64_t kAdd2CaptureIdx = 2l;
const int64_t kCast1CaptureIdx = 3l;

PatternUniqPtr MakePatternForLayernorm(bool make_cast1_as_output, bool make_add2_as_output, bool add_first)
{
    std::string pattern_name = make_cast1_as_output ? kPassName + "Cast1" : kPassName;
    pattern_name = make_add2_as_output ? pattern_name + "Add2" : pattern_name;
    pattern_name = add_first ? pattern_name + "AddFirst" : pattern_name + "AddSecond";
    auto graph_builder = es::EsGraphBuilder(pattern_name.c_str());
    auto [x1,x2_or_bias,gamma,beta,bias_or_x2] = graph_builder.CreateInputs<5>();
    auto add1 = x2_or_bias + bias_or_x2;
    auto add2 = add_first? add1 + x1 : x1 + add1;
    auto cast1 = es::Cast(add2, DT_FLOAT);
    auto [y, mean, variance] = es::LayerNorm(cast1, gamma, beta);
    y = es::Cast(y, DT_FLOAT16);
    std::vector outputs{y, mean, variance};
    if (make_cast1_as_output) outputs.emplace_back(cast1);
    if (make_add2_as_output) outputs.emplace_back(add2);
    auto graph = graph_builder.BuildAndReset(outputs);
    auto pattern = std::make_unique<Pattern>(std::move(*graph));
    pattern->CaptureTensor({*mean.GetProducer(), 0})
        .CaptureTensor({*add1.GetProducer(), 0})
        .CaptureTensor({*add2.GetProducer(), 0})
        .CaptureTensor({*cast1.GetProducer(), 0});
    return pattern;
}

PatternUniqPtr MakePatternForLayernormV3(bool make_cast1_as_output, bool make_add2_as_output, bool add_first)
{
    std::string pattern_name = make_cast1_as_output ? kPassName + "V3Cast1" : kPassName + "V3";
    pattern_name = make_add2_as_output ? pattern_name + "Add2" : pattern_name;
    pattern_name = add_first ? pattern_name + "AddFirst" : pattern_name + "AddSecond";
    auto graph_builder = es::EsGraphBuilder(pattern_name.c_str());
    auto [x1,x2_or_bias,gamma,beta,bias_or_x2] = graph_builder.CreateInputs<5>();
    auto add1 = x2_or_bias + bias_or_x2;
    auto add2 = add_first? add1 + x1 : x1 + add1;
    auto cast1 = es::Cast(add2, DT_FLOAT);
    auto [y, mean, rstd] = es::LayerNormV3(cast1, gamma, beta);
    y = es::Cast(y, DT_FLOAT16);
    std::vector outputs{y, mean, rstd};

    if (make_cast1_as_output) outputs.emplace_back(cast1);
    if (make_add2_as_output) outputs.emplace_back(add2);
    auto graph = graph_builder.BuildAndReset(outputs);
    auto pattern = std::make_unique<Pattern>(std::move(*graph));
    pattern->CaptureTensor({*mean.GetProducer(), 0})
        .CaptureTensor({*add1.GetProducer(), 0})
        .CaptureTensor({*add2.GetProducer(), 0})
        .CaptureTensor({*cast1.GetProducer(), 0});
    return pattern;
}

bool IsTargetPlatform()
{
    PlatformInfo platform_info;
    OptionalInfo optional_info;
    OP_LOGE_IF(
        PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platform_info, optional_info) != SUCCESS,
        false, kPassName.c_str(), "Get platform_info failed.");
    const std::string soc = platform_info.str_info.short_soc_version;
    bool is_platform910_93 = (soc == "Ascend910_93");
    bool is_platform950 = (soc == "Ascend950");
    OPS_LOG_D(kPassName.c_str(), "Platform short soc: %s", soc.c_str());
    if (!is_platform910_93 && !is_platform950) {
        OPS_LOG_D(kPassName.c_str(), "Platform is not support, only work on Ascend950 or Ascend910_93.");
        return false;
    }
    return true;
}

bool IsBeginNormAxisRight(const std::unique_ptr<MatchResult>& match_result)
{
    NodeIo layer_norm_node;
    OP_LOGE_IF(
        match_result->GetCapturedTensor(kLayerNormCaptureIdx, layer_norm_node) != SUCCESS, false, kPassName.c_str(),
        "get Layernorm node failed.");
    NodeIo add2_node;
    OP_LOGE_IF(
        match_result->GetCapturedTensor(kAdd2CaptureIdx, add2_node) != SUCCESS, false, kPassName.c_str(),
        "get Add2 node failed.");
    int64_t begin_norm_axis;
    layer_norm_node.node.GetAttr("begin_norm_axis", begin_norm_axis);
    TensorDesc add2_input0_tensor_desc;
    add2_node.node.GetInputDesc(0, add2_input0_tensor_desc);
    int64_t x1_last_dim = static_cast<int64_t>(add2_input0_tensor_desc.GetShape().GetDimNum()) - 1;
    bool is_axis_last_dim = (begin_norm_axis == -1) || (begin_norm_axis == x1_last_dim);
    if (!is_axis_last_dim) {
        OPS_LOG_D(
            kPassName.c_str(), "begin_norm_axis must be -1 or last axis, begin_norm_axis = %d, x1_last_dim = %d.",
            begin_norm_axis, x1_last_dim);
        return false;
    }
    return true;
}

bool IsAdd1InputValid(const GNode& add1_node, const AscendString& add1_node_name)
{
    TensorDesc add1_input0_desc;
    add1_node.GetInputDesc(0, add1_input0_desc);
    TensorDesc add1_input1_desc;
    add1_node.GetInputDesc(1, add1_input1_desc);
    //check shape
    Shape add1_input0_shape = add1_input0_desc.GetShape();
    Shape add1_input1_shape = add1_input1_desc.GetShape();
    if (add1_input0_shape.GetDims() == add1_input1_shape.GetDims()) {
        OPS_LOG_D(kPassName.c_str(), "%s not support element-wise add.", add1_node_name.GetString());
        return false;
    }
    if (IsScaler(add1_input0_shape) || IsScaler(add1_input1_shape)) {
        OPS_LOG_D(kPassName.c_str(), "%s inputs not support scaler", add1_node_name.GetString());
        return false;
    }
    if (IsDynamicShape(add1_input0_shape) || IsDynamicShape(add1_input1_shape)) {
        OPS_LOG_D(kPassName.c_str(), "%s not support dynamic input", add1_node_name.GetString());
        return false;
    }
    if (add1_input0_shape.GetDims().back() != add1_input1_shape.GetDims().back()) {
        OPS_LOG_D(kPassName.c_str(), "Only support %s inputs have same last dim", add1_node_name.GetString());
        return false;
    }
    //check dtype
    if (!((add1_input0_desc.GetDataType() == add1_input1_desc.GetDataType()) &&
          ((add1_input0_desc.GetDataType() == DT_FLOAT16) || (add1_input0_desc.GetDataType() == DT_BF16)))) {
        OPS_LOG_D(
            kPassName.c_str(),
            "Only support %s inputs have same dtype and must be fp16/bf16, but now input0_dtype is %d, input1_dtype is %d",
            add1_node_name.GetString(), add1_input0_desc.GetDataType(), add1_input1_desc.GetDataType());
        return false;
    }
    return true;
}

bool IsLayerNormInputValid(
    const GNode& add1_node, const GNode& layer_norm_node, const AscendString& layer_norm_node_name)
{
    TensorDesc add1_input0_desc;
    add1_node.GetInputDesc(0, add1_input0_desc);
    TensorDesc layer_norm_input1_desc;
    layer_norm_node.GetInputDesc(1, layer_norm_input1_desc);
    TensorDesc layer_norm_input2_desc;
    layer_norm_node.GetInputDesc(2, layer_norm_input2_desc);
    //check shape
    Shape layer_norm_input1_shape = layer_norm_input1_desc.GetShape();
    Shape layer_norm_input2_shape = layer_norm_input2_desc.GetShape();
    if ((layer_norm_input1_shape.GetDimNum() != 1) ||
        !(layer_norm_input1_shape.GetDimNum() == layer_norm_input2_shape.GetDimNum())) {
        OPS_LOG_D(
            kPassName.c_str(), "%s only support gamma/beta shape is one dim, and be equal.",
            layer_norm_node_name.GetString());
        return false;
    }
    if (IsDynamicShape(layer_norm_input1_shape)) {
        OPS_LOG_D(kPassName.c_str(), "%s only support static shape input.", layer_norm_node_name.GetString());
        return false;
    }
    if (layer_norm_input1_shape.GetDim(0) != add1_input0_desc.GetShape().GetDims().back()) {
        OPS_LOG_D(
            kPassName.c_str(), "%s only support gamma/beta last dim same with add_input.",
            layer_norm_node_name.GetString());
        return false;
    }
    //check dtype
    if (!(layer_norm_input1_desc.GetDataType() == layer_norm_input2_desc.GetDataType() && layer_norm_input1_desc.
          GetDataType() == DT_FLOAT)) {
        OPS_LOG_D(
            kPassName.c_str(), "%s only support gamma/beta fp32, but now gamma_dtype is %d, beta_dtype is %d",
            layer_norm_node_name.GetString(), layer_norm_input1_desc.GetDataType(),
            layer_norm_input2_desc.GetDataType());
        return false;
    }
    return true;
}

bool IsAllInputShapeAndDtypeValid(const std::unique_ptr<MatchResult>& match_result)
{
    NodeIo add1_output_0;
    NodeIo add2_output_0;
    OP_LOGE_IF(
        match_result->GetCapturedTensor(kAdd1CaptureIdx, add1_output_0)!= SUCCESS, false, kPassName,
        "Failed to GetCaptrue tensor");
    OP_LOGE_IF(
        match_result->GetCapturedTensor(kAdd2CaptureIdx, add2_output_0) != SUCCESS, false, kPassName,
        "Failed to GetCaptrue tensor");
    auto add1_node = add1_output_0.node;
    auto add2_node = add2_output_0.node;
    AscendString add1_node_name;
    add1_node.GetName(add1_node_name);
    AscendString add2_node_name;
    add2_node.GetName(add2_node_name);

    if (!IsAdd1InputValid(add1_node, add1_node_name)) {
        OPS_LOG_D(kPassName, "%s node input is invalid", add1_node_name.GetString());
        return false;
    }
    if (!IsAdd2InputValid(add1_node, add2_node, add1_node_name, add2_node_name, kPassName)) {
        OPS_LOG_D(kPassName, "%s node input is invalid", add2_node_name.GetString());
        return false;
    }

    NodeIo layer_norm_output;
    OP_LOGE_IF(
        match_result->GetCapturedTensor(kLayerNormCaptureIdx, layer_norm_output) != SUCCESS, false, kPassName,
        "Failed to get layernormv4");
    auto layer_norm_node = layer_norm_output.node;
    AscendString layer_norm_node_name;
    layer_norm_node.GetName(layer_norm_node_name);
    if (!IsLayerNormInputValid(add1_node, layer_norm_node, layer_norm_node_name)) {
        OPS_LOG_D(kPassName, "%s shape/dtype invalid", layer_norm_node_name.GetString());
        return false;
    }
    return true;
}

} //namespace

std::vector<PatternUniqPtr> AddLayerNormFusionPass::Patterns()
{
    std::vector<PatternUniqPtr> pattern_graphs;
    // 传入所有情况的bool值
    pattern_graphs.emplace_back(MakePatternForLayernorm(false,true,true));
    pattern_graphs.emplace_back(MakePatternForLayernorm(false,true,false));
    pattern_graphs.emplace_back(MakePatternForLayernorm(false,false,true));
    pattern_graphs.emplace_back(MakePatternForLayernorm(false,false,false));
    pattern_graphs.emplace_back(MakePatternForLayernorm(true,true,true));
    pattern_graphs.emplace_back(MakePatternForLayernorm(true,true,false));
    pattern_graphs.emplace_back(MakePatternForLayernorm(true,false,true));
    pattern_graphs.emplace_back(MakePatternForLayernorm(true,false,false));

    pattern_graphs.emplace_back(MakePatternForLayernormV3(false,true,true));
    pattern_graphs.emplace_back(MakePatternForLayernormV3(false,true,false));
    pattern_graphs.emplace_back(MakePatternForLayernormV3(false,false,true));
    pattern_graphs.emplace_back(MakePatternForLayernormV3(false,false,false));
    pattern_graphs.emplace_back(MakePatternForLayernormV3(true,true,true));
    pattern_graphs.emplace_back(MakePatternForLayernormV3(true,true,false));
    pattern_graphs.emplace_back(MakePatternForLayernormV3(true,false,true));
    pattern_graphs.emplace_back(MakePatternForLayernormV3(true,false,false));
    return pattern_graphs;
}

bool AddLayerNormFusionPass::MeetRequirements(const std::unique_ptr<MatchResult>& match_result)
{
    OPS_LOG_D(kPassName.c_str(), "Enter MeetRequirements for AddLayerNormFusionPass");

    if (!IsTargetPlatform()) {
        OPS_LOG_D(kPassName.c_str(), "Check platform fail");
        return false;
    }
    if (!IsBeginNormAxisRight(match_result)) {
        OPS_LOG_D(kPassName.c_str(), "Chaeck all attr fail");
        return false;
    }
    if (!IsAllInputShapeAndDtypeValid(match_result)) {
        OPS_LOG_D(kPassName.c_str(), "Check all input shape and dtype fail");
        return false;
    }
    if (IsCast1HasControlEdge(match_result, kCast1CaptureIdx, kPassName)) {
        return false;
    }
    NodeIo layer_norm_output;
    OP_LOGE_IF(
        match_result->GetCapturedTensor(kLayerNormCaptureIdx, layer_norm_output) != SUCCESS, false, kPassName,
        "Failed to get layernorm in meetrequirements");
    auto y_output_nodes = layer_norm_output.node.GetOutDataNodesAndPortIndexs(0);
    if (y_output_nodes.size() != 1) {
        return false;
    }
    return true;
}

GraphUniqPtr AddLayerNormFusionPass::Replacement(const std::unique_ptr<MatchResult>& match_result)
{
    OPS_LOG_D(kPassName.c_str(), "Enter Replacement for AddLayerNormFusionPass");
    std::vector<SubgraphInput> subgraph_inputs;
    match_result->ToSubgraphBoundary()->GetAllInputs(subgraph_inputs);

    std::vector<Shape> input_shapes;
    std::vector<DataType> input_dtypes;
    std::vector<Format> input_formats;
    GetInputsInfo(subgraph_inputs, input_shapes, input_dtypes, input_formats);
    if (input_shapes[1].GetDims().size() == input_shapes[4].GetDims().size()) {
        OPS_LOG_E(kPassName.c_str(), "not support element-wise add");
        return nullptr;
    }
    bool x2first = input_shapes[1].GetDims().size() > input_shapes[4].GetDims().size();

    auto replace_graph_builder = es::EsGraphBuilder("replacement");
    auto r_x1 = replace_graph_builder.CreateInput(
        0, "x1", input_dtypes[0], input_formats[0], input_shapes[0].GetDims());
    auto r_x2_or_bias = replace_graph_builder.CreateInput(
        1, x2first ? kAddInputsName[0].c_str():kAddInputsName[1].c_str(), input_dtypes[1], input_formats[1], input_shapes[1].GetDims());
    auto r_gamma = replace_graph_builder.CreateInput(
        2, "gamma", input_dtypes[2], input_formats[2], input_shapes[2].GetDims());
    auto r_beta = replace_graph_builder.CreateInput(
        3, "beta", input_dtypes[3], input_formats[3], input_shapes[3].GetDims());
    auto r_bias_or_x2 = replace_graph_builder.CreateInput(
        4, x2first ? kAddInputsName[1].c_str():kAddInputsName[0].c_str(), input_dtypes[4], input_formats[4], input_shapes[4].GetDims());

    NodeIo layer_norm_node;
    if (match_result->GetCapturedTensor(kLayerNormCaptureIdx, layer_norm_node) != SUCCESS) {
        OPS_LOG_E(kPassName.c_str(), "get layernorm node failed.");
        return nullptr;
    }
    float32_t attr_epsilion;
    layer_norm_node.node.GetAttr("epsilon", attr_epsilion);
    auto addlayernorm = es::AddLayerNorm(r_x1, x2first ? r_x2_or_bias:r_bias_or_x2, r_gamma,
        r_beta, x2first ? r_bias_or_x2:r_x2_or_bias, attr_epsilion, true);

    GNode addlayernorm_node = *addlayernorm.y.GetProducer();
    auto layernorm_node_format = input_formats[2];
    UpdateAddLayerNormFormat(layernorm_node_format, addlayernorm_node);

    std::vector<es::EsTensorHolder> replace_outputs = {addlayernorm.y, addlayernorm.mean, addlayernorm.rstd};
    std::vector<SubgraphOutput> subgraph_outputs;
    match_result->ToSubgraphBoundary()->GetAllOutputs(subgraph_outputs);
    AscendString pattern_name;
    match_result->GetPatternGraph().GetName(pattern_name);
    std::string pattern_name_str = pattern_name.GetString();
    if (pattern_name_str.find(kPatternOutputAdd) != std::string::npos) {
        replace_outputs.emplace_back(addlayernorm.x);
    }
    if (pattern_name_str.find(kPatternOutputCast) != std::string::npos) {
        replace_outputs.emplace_back(es::Cast(addlayernorm.x, DT_FLOAT));
    }
    GraphUniqPtr replaceGraph = replace_graph_builder.BuildAndReset(replace_outputs);
    //infershape
    if (InferShape(replaceGraph, subgraph_inputs) != SUCCESS) {
        OPS_LOG_E(kPassName.c_str(), "Infershape for replacement failed.");
        return nullptr;
    }
    return std::move(replaceGraph);
}

REG_FUSION_PASS(AddLayerNormFusionPass).Stage(CustomPassStage::kAfterInferShape);
} //namespace ops
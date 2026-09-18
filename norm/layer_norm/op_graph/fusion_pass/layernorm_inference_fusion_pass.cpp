/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file layernorm_inference_fusion_pass.cpp
 */
#include "layernorm_inference_fusion_pass.h"

#include <set>
#include <string>
#include <vector>

#include "common/inc/error_util.h"
#include "cube_utils/cube_fp16_t.h"
#include "es_math_ops.h"
#include "es_nn_ops.h"
#include "platform/platform_info.h"

namespace ops {
namespace {
const std::string kPassName = "LayerNormInferenceFusionPass";

constexpr size_t kCapMean1 = 0U;
constexpr size_t kCapMean2 = 1U;
constexpr size_t kCapAdd1 = 2U;
constexpr size_t kCapRsqrt = 3U;
constexpr size_t kCapAdd2 = 4U;

// 边界输入：[0]=x [1]=axes1 [2]=axes2 [3]=eps [4]=gamma [5]=beta
constexpr size_t kInputNum = 6U;
constexpr size_t kInputX = 0U;
constexpr size_t kInputGamma = 4U;
constexpr size_t kInputBeta = 5U;

// 交换律枚举位：b0=SquaredDifference, b1=Add(eps), b2=Mul(rsqrt,gamma),
//               b3=Mul(mean1,mul1), b4=Mul(mul1,x), b5=Add(mul3,sub)。
constexpr size_t kCommBitNum = 6U;
constexpr size_t kCommConfigNum = 1U << kCommBitNum;

constexpr int64_t kBeginParamsAxis = -1;
constexpr int64_t kLastAxis = -1;

const std::set<std::string> kSupportedNpuArch = {"3510"};

bool IsBitSet(size_t config, size_t bit) { return ((config >> bit) & 1U) != 0U; }

bool IsSupportedPlatform()
{
    fe::PlatFormInfos platform_infos;
    fe::OptionalInfos optional_infos;
    if (fe::PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platform_infos, optional_infos) !=
        SUCCESS) {
        OPS_LOG_D(kPassName.c_str(), "platform_rejected : get platform info failed.");
        return false;
    }
    std::string arch_str;
    if (!platform_infos.GetPlatformRes("version", "NpuArch", arch_str)) {
        OPS_LOG_D(kPassName.c_str(), "platform_rejected : NpuArch not found.");
        return false;
    }
    if (kSupportedNpuArch.count(arch_str) == 0U) {
        OPS_LOG_D(kPassName.c_str(), "platform_rejected : NpuArch=%s.", arch_str.c_str());
        return false;
    }
    OPS_LOG_D(kPassName.c_str(), "platform_accepted : NpuArch=%s.", arch_str.c_str());
    return true;
}

bool GetCapturedNode(const std::unique_ptr<MatchResult>& match_result, size_t index, GNode& node)
{
    NodeIo node_io;
    if (match_result->GetCapturedTensor(index, node_io) != SUCCESS) {
        OPS_LOG_D(kPassName.c_str(), "guard_rejected : get captured node failed.");
        return false;
    }
    node = node_io.node;
    return true;
}

bool ReadConstIntVec(const GNode& node, int32_t index, std::vector<int64_t>& values)
{
    Tensor tensor;
    if (node.GetInputConstData(index, tensor) != GRAPH_SUCCESS) {
        return false;
    }
    TensorDesc desc;
    if (node.GetInputDesc(index, desc) != GRAPH_SUCCESS) {
        return false;
    }
    const uint8_t* data = tensor.GetData();
    if (data == nullptr) {
        return false;
    }
    const DataType dtype = desc.GetDataType();
    if (dtype == DT_INT32) {
        const size_t count = tensor.GetSize() / sizeof(int32_t);
        const int32_t* typed = reinterpret_cast<const int32_t*>(data);
        for (size_t i = 0U; i < count; ++i) {
            values.push_back(static_cast<int64_t>(typed[i]));
        }
        return true;
    }
    if (dtype == DT_INT64) {
        const size_t count = tensor.GetSize() / sizeof(int64_t);
        const int64_t* typed = reinterpret_cast<const int64_t*>(data);
        for (size_t i = 0U; i < count; ++i) {
            values.push_back(typed[i]);
        }
        return true;
    }
    return false;
}

bool ReadConstFloatScalar(const GNode& node, int32_t index, float32_t& value)
{
    Tensor tensor;
    if (node.GetInputConstData(index, tensor) != GRAPH_SUCCESS) {
        return false;
    }
    TensorDesc desc;
    if (node.GetInputDesc(index, desc) != GRAPH_SUCCESS) {
        return false;
    }
    const uint8_t* data = tensor.GetData();
    if (data == nullptr) {
        return false;
    }
    const DataType dtype = desc.GetDataType();
    if (dtype == DT_FLOAT) {
        value = *(reinterpret_cast<const float32_t*>(data));
        return true;
    }
    if (dtype == DT_FLOAT16) {
        value = static_cast<float32_t>(fp16_t(*(reinterpret_cast<const uint16_t*>(data))));
        return true;
    }
    return false;
}

bool GetReduceOpAttr(const GNode& node, std::vector<int64_t>& axes, bool& keep_dims)
{
    if (!ReadConstIntVec(node, 1, axes)) {
        return false;
    }
    keep_dims = false;
    static_cast<void>(node.GetAttr("keep_dims", keep_dims));
    return true;
}

float32_t GetEpsilon(const GNode& add1)
{
    float32_t epsilon = 1e-7f;
    if (!ReadConstFloatScalar(add1, 0, epsilon)) {
        if (!ReadConstFloatScalar(add1, 1, epsilon)) {
            OPS_LOG_D(kPassName.c_str(), "guard_probe : set epsilon to default value.");
        }
    }
    return epsilon;
}

struct Matched {
    GNode mean1;
    GNode mean2;
    GNode add1;
    GNode rsqrt;
    GNode add2;
    std::vector<int64_t> axes;
};

bool CollectMatched(const std::unique_ptr<MatchResult>& match_result, Matched& matched)
{
    return GetCapturedNode(match_result, kCapMean1, matched.mean1) &&
           GetCapturedNode(match_result, kCapMean2, matched.mean2) &&
           GetCapturedNode(match_result, kCapAdd1, matched.add1) &&
           GetCapturedNode(match_result, kCapRsqrt, matched.rsqrt) &&
           GetCapturedNode(match_result, kCapAdd2, matched.add2);
}

bool CheckReduceOpAttr(const GNode& mean1, const GNode& mean2, std::vector<int64_t>& axes1)
{
    bool keep_dims1 = false;
    if (!GetReduceOpAttr(mean1, axes1, keep_dims1)) {
        OPS_LOG_D(kPassName.c_str(), "guard_rejected : unsuccess to get attribute of mean1.");
        return false;
    }

    std::vector<int64_t> axes2;
    bool keep_dims2 = false;
    if (!GetReduceOpAttr(mean2, axes2, keep_dims2)) {
        OPS_LOG_D(kPassName.c_str(), "guard_rejected : unsuccess to get attributes of mean2.");
        return false;
    }
    if ((axes1.size() != 1U) || (axes1 != axes2)) {
        OPS_LOG_D(kPassName.c_str(), "guard_rejected : the axes of mean nodes are not same.");
        return false;
    }
    if (!keep_dims1 || !keep_dims2) {
        OPS_LOG_D(kPassName.c_str(), "guard_rejected : the keep_dims of mean is false.");
        return false;
    }

    TensorDesc input_desc;
    if (mean1.GetInputDesc(0, input_desc) != GRAPH_SUCCESS) {
        OPS_LOG_D(kPassName.c_str(), "guard_rejected : mean1's input is nullptr.");
        return false;
    }
    const size_t dims_size = input_desc.GetShape().GetDims().size();
    if (dims_size < 1U) {
        OPS_LOG_D(kPassName.c_str(), "guard_rejected : input shape should be greater than 0.");
        return false;
    }
    if ((axes1[0] != kLastAxis) && (axes1[0] != static_cast<int64_t>(dims_size - 1U))) {
        OPS_LOG_D(kPassName.c_str(), "guard_rejected : the axes is not the last axis.");
        return false;
    }
    return true;
}

PatternUniqPtr MakePattern(size_t config)
{
    auto graph_builder = es::EsGraphBuilder("layernorm_inference_pattern");
    auto inputs = graph_builder.CreateInputs(kInputNum);

    const auto& x = inputs[kInputX];
    const auto& axes1 = inputs[1];
    const auto& axes2 = inputs[2];
    const auto& eps = inputs[3];
    const auto& gamma = inputs[kInputGamma];
    const auto& beta = inputs[kInputBeta];

    auto mean1 = es::ReduceMean(x, axes1, true);
    auto squared = IsBitSet(config, 0U) ? es::SquaredDifference(x, mean1) : es::SquaredDifference(mean1, x);
    auto mean2 = es::ReduceMean(squared, axes2, true);
    auto add1 = IsBitSet(config, 1U) ? es::Add(mean2, eps) : es::Add(eps, mean2);
    auto rsqrt = es::Rsqrt(add1);
    auto mul1 = IsBitSet(config, 2U) ? es::Mul(rsqrt, gamma) : es::Mul(gamma, rsqrt);
    auto mul2 = IsBitSet(config, 3U) ? es::Mul(mean1, mul1) : es::Mul(mul1, mean1);
    auto mul3 = IsBitSet(config, 4U) ? es::Mul(mul1, x) : es::Mul(x, mul1);
    auto sub = es::Sub(beta, mul2);
    auto add2 = IsBitSet(config, 5U) ? es::Add(mul3, sub) : es::Add(sub, mul3);

    auto graph = graph_builder.BuildAndReset({add2});
    if (graph == nullptr) {
        return nullptr;
    }
    auto pattern = std::make_unique<Pattern>(std::move(*graph));
    pattern->CaptureTensor({*mean1.GetProducer(), 0})
        .CaptureTensor({*mean2.GetProducer(), 0})
        .CaptureTensor({*add1.GetProducer(), 0})
        .CaptureTensor({*rsqrt.GetProducer(), 0})
        .CaptureTensor({*add2.GetProducer(), 0});
    return pattern;
}
} // namespace

std::vector<PatternUniqPtr> LayerNormInferenceFusionPass::Patterns()
{
    std::vector<PatternUniqPtr> patterns;
    patterns.reserve(kCommConfigNum);
    for (size_t config = 0U; config < kCommConfigNum; ++config) {
        auto pattern = MakePattern(config);
        if (pattern != nullptr) {
            patterns.emplace_back(std::move(pattern));
        }
    }
    OPS_LOG_D(kPassName.c_str(), "patterns_built : %zu.", patterns.size());
    return patterns;
}

bool LayerNormInferenceFusionPass::MeetRequirements(const std::unique_ptr<MatchResult>& match_result)
{
    OPS_LOG_D(kPassName.c_str(), "guard_begin.");
    if (!IsSupportedPlatform()) {
        return false;
    }
    Matched matched;
    if (!CollectMatched(match_result, matched)) {
        return false;
    }
    if (!CheckReduceOpAttr(matched.mean1, matched.mean2, matched.axes)) {
        return false;
    }
    OPS_LOG_D(kPassName.c_str(), "guard_passed.");
    return true;
}

std::unique_ptr<Graph> LayerNormInferenceFusionPass::Replacement(const std::unique_ptr<MatchResult>& match_result)
{
    OPS_LOG_D(kPassName.c_str(), "replacement_begin.");
    Matched matched;
    if (!CollectMatched(match_result, matched)) {
        return nullptr;
    }
    if (!CheckReduceOpAttr(matched.mean1, matched.mean2, matched.axes)) {
        return nullptr;
    }
    const float32_t epsilon = GetEpsilon(matched.add1);

    std::vector<SubgraphInput> subgraph_inputs;
    const auto boundary = match_result->ToSubgraphBoundary();
    if ((boundary == nullptr) || (boundary->GetAllInputs(subgraph_inputs) != SUCCESS)) {
        OPS_LOG_E(kPassName.c_str(), "replacement_rejected : cannot get subgraph boundary inputs.");
        return nullptr;
    }
    if (subgraph_inputs.size() != kInputNum) {
        OPS_LOG_E(kPassName.c_str(), "replacement_rejected : subgraph input num mismatch.");
        return nullptr;
    }

    auto graph_builder = es::EsGraphBuilder("replacement");
    std::vector<es::EsTensorHolder> replacement_inputs;
    std::vector<TensorDesc> boundary_descs;
    replacement_inputs.reserve(kInputNum);
    boundary_descs.reserve(kInputNum);
    for (size_t i = 0U; i < kInputNum; ++i) {
        const auto node_inputs = subgraph_inputs[i].GetAllInputs();
        if (node_inputs.empty()) {
            OPS_LOG_E(kPassName.c_str(), "replacement_rejected : subgraph input has no consumer.");
            return nullptr;
        }
        const auto& node_io = node_inputs.at(0);
        TensorDesc desc;
        if (node_io.node.GetInputDesc(static_cast<int32_t>(node_io.index), desc) != GRAPH_SUCCESS) {
            OPS_LOG_E(kPassName.c_str(), "replacement_rejected : cannot get boundary input desc.");
            return nullptr;
        }
        const std::string input_name = "input_" + std::to_string(i);
        replacement_inputs.emplace_back(graph_builder.CreateInput(static_cast<int64_t>(i), input_name.c_str(),
                                                                  desc.GetDataType(), desc.GetFormat(),
                                                                  desc.GetShape().GetDims()));
        boundary_descs.emplace_back(desc);
    }

    auto layer_norm = es::LayerNorm(replacement_inputs[kInputX], replacement_inputs[kInputGamma],
                                    replacement_inputs[kInputBeta], matched.axes[0], kBeginParamsAxis, epsilon);

    GNode* layer_norm_node = layer_norm.y.GetProducer();
    if (layer_norm_node == nullptr) {
        OPS_LOG_E(kPassName.c_str(), "replacement_rejected : cannot get producer of LayerNorm.");
        return nullptr;
    }
    if ((layer_norm_node->UpdateInputDesc(0, boundary_descs[kInputX]) != GRAPH_SUCCESS) ||
        (layer_norm_node->UpdateInputDesc(1, boundary_descs[kInputGamma]) != GRAPH_SUCCESS) ||
        (layer_norm_node->UpdateInputDesc(2, boundary_descs[kInputBeta]) != GRAPH_SUCCESS)) {
        OPS_LOG_E(kPassName.c_str(), "replacement_rejected : cannot update input desc of LayerNorm.");
        return nullptr;
    }

    TensorDesc y_desc;
    TensorDesc mean_desc;
    TensorDesc variance_desc;
    if ((matched.add2.GetOutputDesc(0, y_desc) != GRAPH_SUCCESS) ||
        (matched.mean1.GetOutputDesc(0, mean_desc) != GRAPH_SUCCESS) ||
        (matched.rsqrt.GetOutputDesc(0, variance_desc) != GRAPH_SUCCESS)) {
        OPS_LOG_E(kPassName.c_str(), "replacement_rejected : cannot get output desc from matched nodes.");
        return nullptr;
    }
    if ((layer_norm_node->UpdateOutputDesc(0, y_desc) != GRAPH_SUCCESS) ||
        (layer_norm_node->UpdateOutputDesc(1, mean_desc) != GRAPH_SUCCESS) ||
        (layer_norm_node->UpdateOutputDesc(2, variance_desc) != GRAPH_SUCCESS)) {
        OPS_LOG_E(kPassName.c_str(), "replacement_rejected : cannot update output desc of LayerNorm.");
        return nullptr;
    }

    OPS_LOG_D(kPassName.c_str(), "replacement_done.");
    return graph_builder.BuildAndReset({layer_norm.y});
}

REG_FUSION_PASS(LayerNormInferenceFusionPass).Stage(CustomPassStage::kAfterBuiltinFusionPass);
} // namespace ops

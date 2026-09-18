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
 * \file layer_norm_fusion_pass.cpp
 */
#include "layer_norm_fusion_pass.h"

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
const std::string kPassName = "LayerNormFusionPass";

constexpr size_t kCapMean0 = 0U;
constexpr size_t kCapMean1 = 1U;
constexpr size_t kCapAdd0 = 2U;
constexpr size_t kCapAdd1 = 3U;

// 边界输入：[0]=x [1]=axes0 [2]=axes1 [3]=eps [4]=gamma [5]=beta
constexpr size_t kInputNum = 6U;
constexpr size_t kInputX = 0U;
constexpr size_t kInputAxes0 = 1U;
constexpr size_t kInputAxes1 = 2U;
constexpr size_t kInputEps = 3U;
constexpr size_t kInputGamma = 4U;
constexpr size_t kInputBeta = 5U;

// 交换律枚举位：b0=SquaredDifference 操作数序, b1=Add(eps) 操作数序, b2=mul0 操作数序, b3=mul1 操作数序。
constexpr size_t kCommBitNum = 4U;
constexpr size_t kCommConfigNum = 1U << kCommBitNum;

constexpr int64_t kBeginParamsAxis = -1;
constexpr int64_t kLastAxis = -1;
constexpr size_t kScalarDimNum = 1U;
constexpr int64_t kScalarDimValue = 1L;

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

std::string TypeOf(const GNode& node)
{
    AscendString type;
    if (node.GetType(type) != GRAPH_SUCCESS || type.GetString() == nullptr) {
        return std::string();
    }
    return std::string(type.GetString());
}

bool IsUnknownDim(int64_t dim) { return dim < 0L; }

// 把某个输入端口上的常量按整型读出。ReduceMean 的 axes 用。
bool GetConstIntVec(const GNode& node, int32_t index, std::vector<int64_t>& values)
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

// 只接受 float / fp16，其余不融合。此处保持一致。
bool GetEpsilon(const GNode& add0, float32_t& epsilon)
{
    Tensor tensor;
    if (add0.GetInputConstData(0, tensor) != GRAPH_SUCCESS) {
        OPS_LOG_D(kPassName.c_str(), "guard_probe : add0 x1 is not const, check x2.");
        if (add0.GetInputConstData(1, tensor) != GRAPH_SUCCESS) {
            OPS_LOG_D(kPassName.c_str(), "guard_rejected : add0 has no const input.");
            return false;
        }
    }

    TensorDesc desc;
    if (add0.GetInputDesc(0, desc) != GRAPH_SUCCESS) {
        return false;
    }
    const uint8_t* data = tensor.GetData();
    if (data == nullptr) {
        OPS_LOG_D(kPassName.c_str(), "guard_rejected : add0 const data is null.");
        return false;
    }
    const DataType dtype = desc.GetDataType();
    if (dtype == DT_FLOAT) {
        epsilon = *(reinterpret_cast<const float32_t*>(data));
        return true;
    }
    if (dtype == DT_FLOAT16) {
        epsilon = static_cast<float32_t>(fp16_t(*(reinterpret_cast<const uint16_t*>(data))));
        return true;
    }
    OPS_LOG_D(kPassName.c_str(), "guard_rejected : add0 dtype is neither float nor fp16.");
    return false;
}

bool CheckAxesAndDim(const std::vector<int64_t>& axes0, const std::vector<int64_t>& axes1,
                     const std::vector<int64_t>& input_dims, const std::vector<int64_t>& add0_dims, bool keep_dims0,
                     bool keep_dims1)
{
    if ((axes0.size() != 1U) || (axes1.size() != 1U) || (axes0[0] != axes1[0])) {
        OPS_LOG_D(kPassName.c_str(), "guard_rejected : the axes of mean are not same.");
        return false;
    }
    if (!keep_dims0 || !keep_dims1) {
        OPS_LOG_D(kPassName.c_str(), "guard_rejected : the attr keep_dims of mean is not true.");
        return false;
    }
    const size_t dims_size = input_dims.size();
    if (dims_size < 1U) {
        OPS_LOG_D(kPassName.c_str(), "guard_rejected : input shape must be greater to one.");
        return false;
    }
    if ((axes0[0] != kLastAxis) && (axes0[0] != static_cast<int64_t>(dims_size - 1U))) {
        OPS_LOG_D(kPassName.c_str(), "guard_rejected : the axes of mean is not the last dim of input.");
        return false;
    }
    if (!add0_dims.empty()) {
        if (IsUnknownDim(add0_dims[0])) {
            OPS_LOG_D(kPassName.c_str(), "guard_rejected : cannot be applied for unknown shape.");
            return false;
        }
        if ((add0_dims.size() != kScalarDimNum) || (add0_dims[0] != kScalarDimValue)) {
            OPS_LOG_D(kPassName.c_str(), "guard_rejected : the const input of add0 must be scalar.");
            return false;
        }
    }
    return true;
}

bool CheckConstInput(const std::vector<int64_t>& add1_dims, const std::vector<int64_t>& input_dims)
{
    if (add1_dims.size() != kScalarDimNum) {
        OPS_LOG_D(kPassName.c_str(), "guard_rejected : the const input of add1 must be 1D.");
        return false;
    }
    if (IsUnknownDim(add1_dims[0])) {
        OPS_LOG_D(kPassName.c_str(), "guard_rejected : cannot be applied for unknown shape.");
        return false;
    }
    if (add1_dims[0] != input_dims[input_dims.size() - 1U]) {
        OPS_LOG_D(kPassName.c_str(), "guard_rejected : add1 const size must equal the last dim of input.");
        return false;
    }
    return true;
}

bool CheckAdd1Source(const GNode& add1)
{
    Tensor tensor;
    if (add1.GetInputConstData(0, tensor) == GRAPH_SUCCESS) {
        return true;
    }
    OPS_LOG_D(kPassName.c_str(), "guard_probe : add1 x1 is not const, check peer node.");
    const auto peer = add1.GetInDataNodesAndPortIndexs(0);
    if ((peer.first == nullptr) || (TypeOf(*peer.first) != "Enter")) {
        OPS_LOG_D(kPassName.c_str(), "guard_rejected : add1 x1 is neither const nor produced by Enter.");
        return false;
    }
    return true;
}

struct Matched {
    GNode mean0;
    GNode mean1;
    GNode add0;
    GNode add1;
    std::vector<int64_t> axes0;
    std::vector<int64_t> axes1;
    std::vector<int64_t> input_dims;
};

bool CollectMatched(const std::unique_ptr<MatchResult>& match_result, Matched& matched)
{
    if (!GetCapturedNode(match_result, kCapMean0, matched.mean0) ||
        !GetCapturedNode(match_result, kCapMean1, matched.mean1) ||
        !GetCapturedNode(match_result, kCapAdd0, matched.add0) ||
        !GetCapturedNode(match_result, kCapAdd1, matched.add1)) {
        return false;
    }

    if (!GetConstIntVec(matched.mean0, 1, matched.axes0)) {
        OPS_LOG_D(kPassName.c_str(), "guard_rejected : the first mean cannot get axis const input.");
        return false;
    }
    if (!GetConstIntVec(matched.mean1, 1, matched.axes1)) {
        OPS_LOG_D(kPassName.c_str(), "guard_rejected : the second mean cannot get axis const input.");
        return false;
    }

    TensorDesc input_desc;
    if (matched.mean0.GetInputDesc(0, input_desc) != GRAPH_SUCCESS) {
        OPS_LOG_D(kPassName.c_str(), "guard_rejected : cannot get input desc of mean0.");
        return false;
    }
    matched.input_dims = input_desc.GetShape().GetDims();
    return true;
}

PatternUniqPtr MakePattern(size_t config, bool form_b)
{
    auto graph_builder = es::EsGraphBuilder("layer_norm_fusion_pattern");
    auto inputs = graph_builder.CreateInputs(kInputNum);

    const auto& x = inputs[kInputX];
    auto mean0 = es::ReduceMean(x, inputs[kInputAxes0], true);
    auto squared0 = IsBitSet(config, 0U) ? es::SquaredDifference(x, mean0) : es::SquaredDifference(mean0, x);
    auto sub0 = es::Sub(x, mean0);
    auto mean1 = es::ReduceMean(squared0, inputs[kInputAxes1], true);
    auto add0 = IsBitSet(config, 1U) ? es::Add(mean1, inputs[kInputEps]) : es::Add(inputs[kInputEps], mean1);
    auto rsqrt0 = es::Rsqrt(add0);

    es::EsTensorHolder mul0;
    es::EsTensorHolder mul1;
    if (form_b) {
        mul0 = IsBitSet(config, 2U) ? es::Mul(rsqrt0, inputs[kInputGamma]) : es::Mul(inputs[kInputGamma], rsqrt0);
        mul1 = IsBitSet(config, 3U) ? es::Mul(mul0, sub0) : es::Mul(sub0, mul0);
    } else {
        mul0 = IsBitSet(config, 2U) ? es::Mul(sub0, rsqrt0) : es::Mul(rsqrt0, sub0);
        mul1 = IsBitSet(config, 3U) ? es::Mul(mul0, inputs[kInputGamma]) : es::Mul(inputs[kInputGamma], mul0);
    }
    auto add1 = es::Add(inputs[kInputBeta], mul1);

    auto graph = graph_builder.BuildAndReset({add1});
    if (graph == nullptr) {
        return nullptr;
    }
    auto pattern = std::make_unique<Pattern>(std::move(*graph));
    pattern->CaptureTensor({*mean0.GetProducer(), 0})
        .CaptureTensor({*mean1.GetProducer(), 0})
        .CaptureTensor({*add0.GetProducer(), 0})
        .CaptureTensor({*add1.GetProducer(), 0});
    return pattern;
}
} // namespace

std::vector<PatternUniqPtr> LayerNormFusionPass::Patterns()
{
    std::vector<PatternUniqPtr> patterns;
    patterns.reserve(kCommConfigNum * 2U);
    for (size_t config = 0U; config < kCommConfigNum; ++config) {
        for (const bool form_b : {false, true}) {
            auto pattern = MakePattern(config, form_b);
            if (pattern != nullptr) {
                patterns.emplace_back(std::move(pattern));
            }
        }
    }
    OPS_LOG_D(kPassName.c_str(), "patterns_built : %zu.", patterns.size());
    return patterns;
}

bool LayerNormFusionPass::MeetRequirements(const std::unique_ptr<MatchResult>& match_result)
{
    OPS_LOG_D(kPassName.c_str(), "guard_begin.");
    if (!IsSupportedPlatform()) {
        return false;
    }
    Matched matched;
    if (!CollectMatched(match_result, matched)) {
        return false;
    }

    bool keep_dims0 = false;
    bool keep_dims1 = false;
    if ((matched.mean0.GetAttr("keep_dims", keep_dims0) != GRAPH_SUCCESS) ||
        (matched.mean1.GetAttr("keep_dims", keep_dims1) != GRAPH_SUCCESS)) {
        OPS_LOG_D(kPassName.c_str(), "guard_rejected : get attr keep_dims failed.");
        return false;
    }

    TensorDesc add0_input0_desc;
    if (matched.add0.GetInputDesc(0, add0_input0_desc) != GRAPH_SUCCESS) {
        OPS_LOG_D(kPassName.c_str(), "guard_rejected : cannot get input desc of add0.");
        return false;
    }
    if (!CheckAxesAndDim(matched.axes0, matched.axes1, matched.input_dims, add0_input0_desc.GetShape().GetDims(),
                         keep_dims0, keep_dims1)) {
        return false;
    }

    float32_t epsilon = 0.0f;
    if (!GetEpsilon(matched.add0, epsilon)) {
        return false;
    }

    if (!CheckAdd1Source(matched.add1)) {
        return false;
    }

    TensorDesc add1_input0_desc;
    if (matched.add1.GetInputDesc(0, add1_input0_desc) != GRAPH_SUCCESS) {
        OPS_LOG_D(kPassName.c_str(), "guard_rejected : cannot get input desc of add1.");
        return false;
    }
    if (!CheckConstInput(add1_input0_desc.GetShape().GetDims(), matched.input_dims)) {
        return false;
    }

    OPS_LOG_D(kPassName.c_str(), "guard_passed.");
    return true;
}

std::unique_ptr<Graph> LayerNormFusionPass::Replacement(const std::unique_ptr<MatchResult>& match_result)
{
    OPS_LOG_D(kPassName.c_str(), "replacement_begin.");
    Matched matched;
    if (!CollectMatched(match_result, matched)) {
        return nullptr;
    }

    float32_t epsilon = 0.0f;
    if (!GetEpsilon(matched.add0, epsilon)) {
        return nullptr;
    }

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
                                    replacement_inputs[kInputBeta], matched.axes0[0], kBeginParamsAxis, epsilon);

    // 注册阶段在 InferShape 之后，三个输出的 desc 从被消除的节点直接拷贝，不做 shape 推导。
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
    if ((matched.add1.GetOutputDesc(0, y_desc) != GRAPH_SUCCESS) ||
        (matched.mean0.GetOutputDesc(0, mean_desc) != GRAPH_SUCCESS) ||
        (matched.mean1.GetOutputDesc(0, variance_desc) != GRAPH_SUCCESS)) {
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

REG_FUSION_PASS(LayerNormFusionPass).Stage(CustomPassStage::kAfterBuiltinFusionPass);
} // namespace ops

/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "add_rms_norm_fusion_pass.h"

#include <algorithm>
#include <array>
#include <set>
#include <string>
#include <utility>

#include "arch_utils.h"
#include "common/inc/error_util.h"
#include "es_math_ops.h"
#include "es_nn_ops.h"
#include "ge/ge_utils.h"

namespace ops {
namespace {
const std::string kPassName = "AddRmsNormFusionGraphPass";
const std::set<std::string> kSupportedNpuArchs = {"3510", "5102", "9201", "9202"};

constexpr float kDefaultEpsilon = 1e-5f;
constexpr size_t kInputCount = 3U;

bool SetOutputDesc(const ge::es::EsTensorHolder& tensor, const ge::TensorDesc& desc)
{
    auto* producer = tensor.GetProducer();
    return producer != nullptr && producer->UpdateOutputDesc(tensor.GetProducerOutIndex(), desc) == ge::GRAPH_SUCCESS;
}

bool SetDataDesc(const ge::es::EsTensorHolder& data, const ge::TensorDesc& desc)
{
    auto* node = data.GetProducer();
    if (node == nullptr) {
        return false;
    }
    if (node->UpdateInputDesc(0, desc) != ge::GRAPH_SUCCESS) {
        return false;
    }
    return node->UpdateOutputDesc(data.GetProducerOutIndex(), desc) == ge::GRAPH_SUCCESS;
}

ge::fusion::PatternUniqPtr MakePattern()
{
    ge::es::EsGraphBuilder builder(kPassName.c_str());
    const auto inputs = builder.CreateInputs(kInputCount);
    const auto add = ge::es::Add(inputs[0], inputs[1]);
    const auto norm = ge::es::RmsNorm(add, inputs[2]);
    if (norm.y.GetProducer() == nullptr || add.GetProducer() == nullptr) {
        OPS_LOG_E(kPassName.c_str(), "cannot create pattern operators.");
        return nullptr;
    }
    const ge::fusion::NodeIo norm_capture{*norm.y.GetProducer(), 0};
    const ge::fusion::NodeIo add_capture{*add.GetProducer(), 0};
    // Add.y can also feed nodes outside RmsNorm; expose it as a boundary output.
    auto graph = builder.BuildAndReset({norm.y, norm.rstd, add});
    if (graph == nullptr) {
        OPS_LOG_E(kPassName.c_str(), "pattern construction failed.");
        return nullptr;
    }
    auto pattern = std::make_unique<ge::fusion::Pattern>(std::move(*graph));
    pattern->CaptureTensor(norm_capture).CaptureTensor(add_capture);
    return pattern;
}
} // namespace

std::vector<ge::fusion::PatternUniqPtr> AddRmsNormFusionGraphPass::Patterns()
{
    if (!ops::CheckNpuArch(kPassName, kSupportedNpuArchs)) {
        OPS_LOG_D(kPassName.c_str(), "guard_rejected : unsupported/unavailable NpuArch.");
        return {};
    }
    std::vector<ge::fusion::PatternUniqPtr> patterns;
    auto pattern = MakePattern();
    if (pattern != nullptr) {
        patterns.emplace_back(std::move(pattern));
    }
    OPS_LOG_D(kPassName.c_str(), "patterns_built : %zu.", patterns.size());
    return patterns;
}

bool AddRmsNormFusionGraphPass::MeetRequirements(const std::unique_ptr<ge::fusion::MatchResult>& match_result)
{
    ge::fusion::NodeIo add;
    if (match_result == nullptr || match_result->GetCapturedTensor(1, add) != ge::SUCCESS) {
        OPS_LOG_D(kPassName.c_str(), "guard_rejected : cannot read captured Add node.");
        return false;
    }
    ge::TensorDesc x1_desc;
    ge::TensorDesc x2_desc;
    if (add.node.GetInputDesc(0, x1_desc) != ge::GRAPH_SUCCESS ||
        add.node.GetInputDesc(1, x2_desc) != ge::GRAPH_SUCCESS) {
        OPS_LOG_D(kPassName.c_str(), "guard_rejected : cannot read Add input descriptors.");
        return false;
    }
    const auto x1_dims = x1_desc.GetShape().GetDims();
    const auto x2_dims = x2_desc.GetShape().GetDims();
    if (x1_dims != x2_dims) {
        OPS_LOG_D(kPassName.c_str(), "guard_rejected : x1 and x2 shapes differ.");
        return false;
    }
    if (!x1_dims.empty() && std::all_of(x1_dims.begin(), x1_dims.end(), [](int64_t dim) { return dim == -1; })) {
        OPS_LOG_D(kPassName.c_str(), "guard_rejected : all input dimensions are unknown.");
        return false;
    }
    if (x1_desc.GetDataType() != x2_desc.GetDataType()) {
        OPS_LOG_D(kPassName.c_str(), "guard_rejected : x1 and x2 dtypes differ.");
        return false;
    }
    if (x1_desc.GetFormat() != ge::FORMAT_ND || x2_desc.GetFormat() != ge::FORMAT_ND) {
        OPS_LOG_D(kPassName.c_str(), "guard_rejected : x1 and x2 must both use ND format.");
        return false;
    }
    OPS_LOG_D(kPassName.c_str(), "guard_accepted : input requirements satisfied.");
    return true;
}

ge::fusion::GraphUniqPtr AddRmsNormFusionGraphPass::Replacement(
    const std::unique_ptr<ge::fusion::MatchResult>& match_result)
{
    ge::fusion::NodeIo norm;
    ge::fusion::NodeIo add;
    if (match_result == nullptr || match_result->GetCapturedTensor(0, norm) != ge::SUCCESS ||
        match_result->GetCapturedTensor(1, add) != ge::SUCCESS) {
        OPS_LOG_E(kPassName.c_str(), "cannot read captured nodes.");
        return nullptr;
    }
    // Pattern input order: Add.x1, Add.x2, RmsNorm.gamma.
    std::array<ge::TensorDesc, kInputCount> input_descs;
    if (add.node.GetInputDesc(0, input_descs[0]) != ge::GRAPH_SUCCESS ||
        add.node.GetInputDesc(1, input_descs[1]) != ge::GRAPH_SUCCESS ||
        norm.node.GetInputDesc(1, input_descs[2]) != ge::GRAPH_SUCCESS) {
        OPS_LOG_E(kPassName.c_str(), "Cannot read x1, x2 or gamma input descriptor.");
        return nullptr;
    }

    ge::es::EsGraphBuilder builder(kPassName.c_str());
    const auto inputs = builder.CreateInputs(kInputCount);
    std::vector<ge::Shape> input_shapes;
    input_shapes.reserve(kInputCount);
    for (size_t i = 0; i < kInputCount; ++i) {
        if (!SetDataDesc(inputs[i], input_descs[i])) {
            OPS_LOG_E(kPassName.c_str(), "Cannot initialize replacement Data %zu descriptors.", i);
            return nullptr;
        }
        input_shapes.emplace_back(input_descs[i].GetShape());
    }
    float epsilon = kDefaultEpsilon;
    (void)norm.node.GetAttr("epsilon", epsilon);
    const auto fused = ge::es::AddRmsNorm(inputs[0], inputs[1], inputs[2], epsilon);
    auto* fused_node = fused.y.GetProducer();
    if (fused_node == nullptr) {
        OPS_LOG_E(kPassName.c_str(), "cannot create AddRmsNorm.");
        return nullptr;
    }
    for (size_t i = 0; i < kInputCount; ++i) {
        if (fused_node->UpdateInputDesc(static_cast<int32_t>(i), input_descs[i]) != ge::GRAPH_SUCCESS) {
            OPS_LOG_E(kPassName.c_str(), "cannot initialize AddRmsNorm input descriptor.");
            return nullptr;
        }
    }
    const std::array<ge::fusion::NodeIo, 3> old_outputs{{{norm.node, 0}, {norm.node, 1}, {add.node, 0}}};
    const std::vector<ge::es::EsTensorHolder> outputs{fused.y, fused.rstd, fused.x};
    for (size_t i = 0; i < outputs.size(); ++i) {
        ge::TensorDesc desc;
        if (old_outputs[i].node.GetOutputDesc(old_outputs[i].index, desc) != ge::GRAPH_SUCCESS ||
            !SetOutputDesc(outputs[i], desc)) {
            OPS_LOG_E(kPassName.c_str(), "cannot initialize AddRmsNorm output descriptor.");
            return nullptr;
        }
    }
    auto graph = builder.BuildAndReset(outputs);
    if (graph == nullptr) {
        OPS_LOG_E(kPassName.c_str(), "Replacement graph construction failed.");
        return nullptr;
    }
    // Infer only after Data input/output and fused-node descriptors have been initialized.
    const auto status = ge::GeUtils::InferShape(*graph, input_shapes);
    if (status != ge::SUCCESS) {
        OPS_LOG_E(kPassName.c_str(), "Replacement InferShape failed, status=%u.", static_cast<unsigned int>(status));
        return nullptr;
    }
    OPS_LOG_D(kPassName.c_str(), "replacement_ready : InferShape succeeded.");
    return graph;
}

REG_FUSION_PASS(AddRmsNormFusionGraphPass).Stage(ge::CustomPassStage::kAfterBuiltinFusionPass);
} // namespace ops

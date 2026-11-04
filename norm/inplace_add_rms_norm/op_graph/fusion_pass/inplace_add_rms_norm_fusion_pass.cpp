/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "inplace_add_rms_norm_fusion_pass.h"

#include <array>
#include <cstdlib>
#include <set>
#include <string>
#include <vector>

#include "acl/acl_rt.h"
#include "arch_utils.h"
#include "common/inc/error_util.h"
#include "es_nn_ops.h"
#include "ge/ge_utils.h"
#include "norm/common/op_graph/fusion_report_utils.h"
#include "ge/fusion/pass/pattern_fusion_pass.h"
#include "version/ge-compiler_version.h"

namespace ops {
namespace {
const std::string kPassName = "InplaceAddRmsNormFusionPass";
constexpr int32_t kMinGeCompilerVersion = 90000000;
const std::set<std::string> kSupportedNpuArchs = {"3510", "5102", "9201", "9202"};

constexpr float kDefaultEpsilon = 1e-6f;
constexpr size_t kInputCount = 3U;
constexpr size_t kOutputCount = 3U;

bool HasType(const ge::GNode& node, const std::string& expected)
{
    ge::AscendString type;
    return node.GetType(type) == ge::GRAPH_SUCCESS && type.GetString() != nullptr &&
           std::string(type.GetString()) == expected;
}

bool IsInferenceMode(const ge::CustomPassContext& context)
{
#if defined(GE_COMPILER_VERSION_NUM) && (GE_COMPILER_VERSION_NUM >= 90000000)
    int32_t version = 0;
    std::string package_name = "ge_compiler";
    if (aclsysGetVersionNum(package_name.data(), &version) != ACL_SUCCESS || version < kMinGeCompilerVersion) {
        OPS_LOG_D(kPassName.c_str(), "guard_rejected : GE compiler version unavailable or below 9.0.0.");
        return false;
    }
    ge::AscendString mode;
    (void)context.GetOptionValue(ge::AscendString("ge.graphRunMode"), mode);
    if (std::strtol(mode.GetString() != nullptr ? mode.GetString() : "", nullptr, 10) == 1) {
        OPS_LOG_D(kPassName.c_str(), "guard_rejected : training mode.");
        return false;
    }
    return true;
#else
    (void)context;
    return false;
#endif
}

bool IsHalfType(ge::DataType dtype) { return dtype == ge::DT_FLOAT16 || dtype == ge::DT_BF16; }

bool FindCast(const ge::GNode& norm, ge::GNode& cast)
{
    const auto consumers = norm.GetOutDataNodesAndPortIndexs(0);
    // Require two consumers of y, with Cast as the first consumer.
    if (consumers.size() != 2U || consumers[0].first == nullptr || consumers[1].first == nullptr ||
        !HasType(*consumers[0].first, "Cast")) {
        return false;
    }
    cast = *consumers[0].first;
    ge::TensorDesc x1_desc;
    ge::TensorDesc cast_input_desc;
    ge::TensorDesc cast_output_desc;
    if (norm.GetInputDesc(0, x1_desc) != ge::GRAPH_SUCCESS ||
        cast.GetInputDesc(0, cast_input_desc) != ge::GRAPH_SUCCESS ||
        cast.GetOutputDesc(0, cast_output_desc) != ge::GRAPH_SUCCESS) {
        return false;
    }
    return IsHalfType(x1_desc.GetDataType()) && IsHalfType(cast_input_desc.GetDataType()) &&
           cast_output_desc.GetDataType() == ge::DT_FLOAT;
}

bool SetOutputDesc(const ge::es::EsTensorHolder& tensor, const ge::TensorDesc& desc)
{
    auto* producer = tensor.GetProducer();
    return producer != nullptr && producer->UpdateOutputDesc(tensor.GetProducerOutIndex(), desc) == ge::GRAPH_SUCCESS;
}

bool CheckInplaceInputs(const ge::GNode& norm)
{
    for (int32_t i = 0; i < 2; ++i) {
        const auto source = norm.GetInDataNodesAndPortIndexs(i);
        if (source.first == nullptr || source.second < 0) {
            OPS_LOG_D(kPassName.c_str(), "guard_rejected : cannot read source of input %d.", i);
            return false;
        }
        if (HasType(*source.first, "Data")) {
            OPS_LOG_D(kPassName.c_str(), "guard_rejected : input %d comes from graph input Data.", i);
            return false;
        }
        if (source.first->GetOutDataNodesAndPortIndexs(source.second).size() != 1U) {
            OPS_LOG_D(kPassName.c_str(), "guard_rejected : input %d source has multiple consumers.", i);
            return false;
        }
    }
    return true;
}

bool SetDataDesc(const ge::es::EsTensorHolder& data, const ge::TensorDesc& desc)
{
    auto* node = data.GetProducer();
    if (node == nullptr) {
        return false;
    }
    // Set the input and output descriptors of the temporary Data node.
    if (node->UpdateInputDesc(0, desc) != ge::GRAPH_SUCCESS) {
        return false;
    }
    return node->UpdateOutputDesc(data.GetProducerOutIndex(), desc) == ge::GRAPH_SUCCESS;
}

std::unique_ptr<ge::Graph> MakeReplacement(const ge::GNode& norm, bool with_cast,
                                           const std::vector<ge::fusion::NodeIo>& old_outputs)
{
    ge::es::EsGraphBuilder builder(kPassName.c_str());
    const auto inputs = builder.CreateInputs(kInputCount);
    std::array<ge::TensorDesc, kInputCount> input_descs;
    std::vector<ge::Shape> input_shapes;
    input_shapes.reserve(kInputCount);
    for (size_t i = 0; i < kInputCount; ++i) {
        if (norm.GetInputDesc(static_cast<int32_t>(i), input_descs[i]) != ge::GRAPH_SUCCESS) {
            OPS_LOG_E(kPassName.c_str(), "Cannot read descriptor for AddRmsNorm input %zu.", i);
            return nullptr;
        }
        if (!SetDataDesc(inputs[i], input_descs[i])) {
            OPS_LOG_E(kPassName.c_str(), "Cannot initialize replacement Data %zu descriptors.", i);
            return nullptr;
        }
        input_shapes.emplace_back(input_descs[i].GetShape());
    }

    float epsilon = kDefaultEpsilon;
    (void)norm.GetAttr("epsilon", epsilon);
    std::vector<ge::es::EsTensorHolder> outputs;
    if (with_cast) {
        const auto fused = ge::es::AddRmsNormCast(inputs[0], inputs[1], inputs[2], epsilon);
        outputs = {fused.y1, fused.y2, fused.rstd, fused.x};
    } else {
        const auto [x1, rstd, x2] = ge::es::InplaceAddRmsNorm(inputs[0], inputs[1], inputs[2], epsilon);
        outputs = {x1, rstd, x2};
    }
    auto* fused_node = outputs.front().GetProducer();
    if (fused_node == nullptr || outputs.size() != old_outputs.size()) {
        OPS_LOG_E(kPassName.c_str(), "Cannot create replacement operator or output mapping is invalid.");
        return nullptr;
    }
    for (size_t i = 0; i < kInputCount; ++i) {
        if (fused_node->UpdateInputDesc(static_cast<int32_t>(i), input_descs[i]) != ge::GRAPH_SUCCESS) {
            OPS_LOG_E(kPassName.c_str(), "Cannot initialize replacement input %zu descriptor.", i);
            return nullptr;
        }
    }
    for (size_t i = 0; i < outputs.size(); ++i) {
        ge::TensorDesc desc;
        if (old_outputs[i].node.GetOutputDesc(old_outputs[i].index, desc) != ge::GRAPH_SUCCESS ||
            !SetOutputDesc(outputs[i], desc)) {
            OPS_LOG_E(kPassName.c_str(), "Cannot initialize replacement output %zu descriptor.", i);
            return nullptr;
        }
    }
    auto graph = builder.BuildAndReset(outputs);
    if (graph == nullptr) {
        OPS_LOG_E(kPassName.c_str(), "Replacement graph construction failed.");
        return nullptr;
    }
    // Infer descriptors for the replacement graph.
    const auto status = ge::GeUtils::InferShape(*graph, input_shapes);
    if (status != ge::SUCCESS) {
        OPS_LOG_E(kPassName.c_str(), "Replacement InferShape failed, status=%u.", static_cast<unsigned int>(status));
        return nullptr;
    }
    OPS_LOG_D(kPassName.c_str(), "replacement_ready : InferShape succeeded.");
    return graph;
}

ge::Status Rewrite(const ge::GNode& norm, ge::CustomPassContext& context)
{
    if (norm.GetInputsSize() != kInputCount || norm.GetOutputsSize() != kOutputCount) {
        OPS_LOG_D(kPassName.c_str(), "guard_rejected : invalid AddRmsNorm arity.");
        return ge::SUCCESS;
    }
    ge::GNode cast;
    const bool with_cast = FindCast(norm, cast);
    OPS_LOG_D(kPassName.c_str(), "candidate : %s.", with_cast ? "AddRmsNormCast" : "InplaceAddRmsNorm");
    if (!with_cast && !CheckInplaceInputs(norm)) {
        return ge::SUCCESS;
    }

    ge::fusion::SubgraphBoundary boundary;
    for (size_t i = 0; i < kInputCount; ++i) {
        if (boundary.AddInput(static_cast<int64_t>(i), ge::fusion::SubgraphInput({{norm, static_cast<int64_t>(i)}})) !=
            ge::SUCCESS) {
            OPS_LOG_E(kPassName.c_str(), "Cannot create boundary input %zu.", i);
            return ge::FAILED;
        }
    }
    std::vector<ge::fusion::NodeIo> old_outputs;
    if (with_cast) {
        // AddRmsNormCast: y1 <- Cast.y, y2/rstd/x <- AddRmsNorm.y/rstd/x.
        old_outputs = {{cast, 0}, {norm, 0}, {norm, 1}, {norm, 2}};
    } else {
        // InplaceAddRmsNorm: x1/rstd/x2 <- AddRmsNorm.y/rstd/x.
        old_outputs = {{norm, 0}, {norm, 1}, {norm, 2}};
    }
    for (size_t i = 0; i < old_outputs.size(); ++i) {
        if (boundary.AddOutput(static_cast<int64_t>(i), ge::fusion::SubgraphOutput(old_outputs[i])) != ge::SUCCESS) {
            OPS_LOG_E(kPassName.c_str(), "Cannot create boundary output %zu.", i);
            return ge::FAILED;
        }
    }
    // Build the replacement graph and reconnect the subgraph boundaries.
    auto replacement = MakeReplacement(norm, with_cast, old_outputs);
    if (replacement == nullptr) {
        OPS_LOG_E(kPassName.c_str(), "Replacement construction failed.");
        return ge::FAILED;
    }
    const auto status = ops::fusion::ReplaceSubgraphWithReport(boundary, *replacement, context);
    if (status != ge::SUCCESS) {
        OPS_LOG_E(kPassName.c_str(), "Replacement failed, status=%u.", static_cast<unsigned int>(status));
        return status;
    }
    OPS_LOG_D(kPassName.c_str(), "replacement_done.");
    return status;
}
} // namespace

ge::Status InplaceAddRmsNormFusionPass::Run(ge::GraphPtr& graph, ge::CustomPassContext& context)
{
    if (graph == nullptr || !ops::CheckNpuArch(kPassName, kSupportedNpuArchs)) {
        OPS_LOG_D(kPassName.c_str(), "guard_rejected : null graph or unsupported/unavailable NpuArch.");
        return ge::SUCCESS;
    }
    if (!IsInferenceMode(context)) {
        OPS_LOG_D(kPassName.c_str(), "guard_rejected : runmode or compatibility check failed.");
        return ge::SUCCESS;
    }
    // Collect AddRmsNorm nodes, then rewrite each candidate.
    std::vector<ge::GNode> candidates;
    for (const auto& node : graph->GetAllNodes()) {
        if (HasType(node, "AddRmsNorm")) {
            candidates.push_back(node);
        }
    }
    for (const auto& norm : candidates) {
        if (Rewrite(norm, context) != ge::SUCCESS) {
            OPS_LOG_E(kPassName.c_str(), "Rewrite failed.");
            return ge::FAILED;
        }
    }
    return ge::SUCCESS;
}

#if defined(GE_COMPILER_VERSION_NUM) && (GE_COMPILER_VERSION_NUM >= 90000000)
REG_FUSION_PASS(InplaceAddRmsNormFusionPass).Stage(ge::CustomPassStage::kAfterBuiltinFusionPass);
#endif
} // namespace ops

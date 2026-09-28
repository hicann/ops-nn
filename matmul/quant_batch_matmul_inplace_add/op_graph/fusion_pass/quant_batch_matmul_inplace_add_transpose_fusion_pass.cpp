/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "quant_batch_matmul_inplace_add_transpose_fusion_pass.h"

#include <algorithm>
#include <array>
#include <set>
#include "acl/acl_rt.h"
#include "ge/fusion/pass/pattern_fusion_pass.h"
#include "platform/platform_info.h"
#include "version/ge-compiler_version.h"

namespace ops {
namespace {
using namespace ge;
constexpr char PASS_NAME[] = "QuantBatchMatmulInplaceAddTransposeFusionPass";
constexpr char OP_TYPE[] = "QuantBatchMatmulInplaceAdd";
constexpr int64_t X1_PORT = 0;
constexpr int64_t X2_PORT = 1;
constexpr int64_t X2_SCALE_PORT = 2;
constexpr int64_t X1_SCALE_PORT = 4;
constexpr size_t MATRIX_RANK = 2;
constexpr size_t MX_SCALE_RANK = 3;
constexpr size_t FIRST_DIM = 0;
constexpr size_t SECOND_DIM = 1;
constexpr int64_t SINGLETON_DIM = 1;
constexpr int32_t MAX_DETERMINISTIC = 1;
constexpr int32_t MAX_DETERMINISTIC_LEVEL = 3;
constexpr std::array<int64_t, 2> INPUT_PORTS = {X1_PORT, X1_SCALE_PORT};
constexpr int32_t TARGET_VERSION = 90100000;
using TransposeNodes = std::array<GNodePtr, 2>;

bool IsTargetVersion()
{
    int32_t version = 0;
    char component[] = "ge-compiler";
    return aclsysGetVersionNum(component, &version) == 0 && version >= TARGET_VERSION;
}

bool IsType(const GNodePtr& node, const char* type)
{
    AscendString actual;
    return node && node->GetType(actual) == GRAPH_SUCCESS && actual == type;
}

bool IsTranspose(const GNodePtr& node) { return IsType(node, "Transpose") || IsType(node, "TransposeD"); }

GNodePtr Input(const GNode& node, int64_t port)
{
    // Optional inputs (notably x1_scale) may be absent or disconnected.
    if (port < 0 || static_cast<size_t>(port) >= node.GetInputsSize()) {
        return nullptr;
    }
    return node.GetInDataNodesAndPortIndexs(port).first;
}

bool IsReshapeTrans(const TensorDesc& desc, const GNodePtr& node, bool isDynamic)
{
    if (!IsType(node, "Reshape")) {
        return false;
    }
    if (isDynamic) {
        return true;
    }
    const auto shape = desc.GetShape();
    if (shape.GetDimNum() < MATRIX_RANK) {
        return false;
    }
    TensorDesc input;
    TensorDesc output;
    if (node->GetInputDesc(0, input) != GRAPH_SUCCESS || node->GetOutputDesc(0, output) != GRAPH_SUCCESS) {
        return false;
    }
    const auto in = input.GetShape();
    const auto out = output.GetShape();
    if (desc.GetDataType() == DT_FLOAT8_E8M0 && in.GetDimNum() == MX_SCALE_RANK && out.GetDimNum() == MX_SCALE_RANK &&
        in.GetDim(FIRST_DIM) == out.GetDim(SECOND_DIM) && in.GetDim(SECOND_DIM) == out.GetDim(FIRST_DIM) &&
        (in.GetDim(FIRST_DIM) == SINGLETON_DIM || in.GetDim(SECOND_DIM) == SINGLETON_DIM)) {
        return true;
    }
    return shape.GetDim(shape.GetDimNum() - 1) == SINGLETON_DIM ||
           shape.GetDim(shape.GetDimNum() - MATRIX_RANK) == SINGLETON_DIM;
}

bool IsNodeEqualTrans(const TensorDesc& desc, const GNodePtr& node, bool isDynamic)
{
    return IsTranspose(node) || IsReshapeTrans(desc, node, isDynamic);
}

bool IsBitcastPattern(const GNode& node, bool isDynamic)
{
    for (const auto port : INPUT_PORTS) {
        auto cast = Input(node, port);
        if (!IsType(cast, "Bitcast")) {
            continue;
        }
        auto before = Input(*cast, 0);
        TensorDesc desc;
        if (port == X1_PORT ?
                IsTranspose(before) :
                (node.GetInputDesc(port, desc) == GRAPH_SUCCESS && IsNodeEqualTrans(desc, before, isDynamic))) {
            return true;
        }
    }
    return false;
}

bool PlatformSupportBitcastTransposeFusion()
{
    fe::PlatformInfo platform;
    fe::OptionalInfo optional;
    (void)fe::PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platform, optional);
    const auto found = platform.ai_core_intrinsic_dtype_map.find("Intrinsic_mmad");
    return found == platform.ai_core_intrinsic_dtype_map.end() ||
           std::find(found->second.begin(), found->second.end(), "s8s4") == found->second.end();
}

GNodePtr Candidate(const GNode& node, int64_t port, bool bitcast)
{
    auto input = Input(node, port);
    return bitcast && IsType(input, "Bitcast") ? Input(*input, 0) : input;
}

bool GetDynamicStatus(const GNode& node, bool& isDynamic)
{
    TensorDesc x1;
    TensorDesc x2;
    if (node.GetInputDesc(X1_PORT, x1) != GRAPH_SUCCESS || node.GetInputDesc(X2_PORT, x2) != GRAPH_SUCCESS) {
        return false;
    }
    const auto unknown = [](const TensorDesc& desc) {
        const auto dims = desc.GetShape().GetDims();
        return std::any_of(dims.begin(), dims.end(), [](int64_t dim) { return dim < 0; });
    };
    isDynamic = unknown(x1) || unknown(x2);
    return true;
}

bool CheckNode(const GNode& node, bool bitcast)
{
    TensorDesc x1;
    TensorDesc x2;
    TensorDesc output;
    if (node.GetInputDesc(X1_PORT, x1) != GRAPH_SUCCESS || node.GetInputDesc(X2_PORT, x2) != GRAPH_SUCCESS ||
        node.GetOutputDesc(0, output) != GRAPH_SUCCESS) {
        return false;
    }
    static const std::array<DataType, 6> inputs = {DT_INT8,        DT_INT4,     DT_FLOAT8_E4M3FN,
                                                   DT_FLOAT8_E5M2, DT_HIFLOAT8, DT_FLOAT4_E2M1};
    static const std::array<DataType, 5> outputs = {DT_INT8, DT_FLOAT16, DT_BF16, DT_INT32, DT_FLOAT};
    if (std::find(inputs.begin(), inputs.end(), x1.GetDataType()) == inputs.end() ||
        std::find(inputs.begin(), inputs.end(), x2.GetDataType()) == inputs.end() ||
        std::find(outputs.begin(), outputs.end(), output.GetDataType()) == outputs.end() ||
        x1.GetOriginShape().GetDimNum() < MATRIX_RANK || x2.GetOriginShape().GetDimNum() < MATRIX_RANK) {
        return false;
    }
    const auto a = Candidate(node, X1_PORT, bitcast);
    return IsTranspose(a) && Input(node, X2_PORT);
}

bool GetTransposeNodes(const GNode& node, bool bitcast, bool isDynamic, TransposeNodes& nodes)
{
    nodes[0] = Candidate(node, X1_PORT, bitcast);
    if (!Input(node, X2_SCALE_PORT)) {
        return false;
    }
    auto candidate = Candidate(node, X1_SCALE_PORT, bitcast);
    TensorDesc desc;
    if (candidate && node.GetInputDesc(X1_SCALE_PORT, desc) == GRAPH_SUCCESS &&
        IsNodeEqualTrans(desc, candidate, isDynamic)) {
        nodes[1] = candidate;
    }
    return true;
}

bool RelinkNode(const GraphPtr& graph, GNode& node, const GNodePtr& trans, int64_t port, bool bitcast)
{
    if (!trans) {
        return true;
    }
    auto source = trans->GetInDataNodesAndPortIndexs(0);
    auto before = Input(node, port);
    const bool hasCast = bitcast && IsType(before, "Bitcast");
    auto& destination = hasCast ? *before : node;
    const auto destinationPort = hasCast ? 0 : port;
    return source.first && graph->RemoveEdge(*trans, 0, destination, destinationPort) == GRAPH_SUCCESS &&
           graph->AddDataEdge(*source.first, source.second, destination, destinationPort) == GRAPH_SUCCESS;
}

bool UpdateBitcastAndInputDesc(GNode& node, const GNodePtr& trans, int64_t port, bool bitcast)
{
    if (!trans) {
        return true;
    }
    TensorDesc input;
    if (trans->GetInputDesc(0, input) != GRAPH_SUCCESS) {
        return false;
    }
    auto before = Input(node, port);
    if (bitcast && IsType(before, "Bitcast")) {
        TensorDesc targetInput;
        if (before->UpdateInputDesc(0, input) != GRAPH_SUCCESS ||
            node.GetInputDesc(port, targetInput) != GRAPH_SUCCESS) {
            return false;
        }
        // Preserve the legacy descriptor side effect, including shared transforms.
        input.SetDataType(targetInput.GetDataType());
        if (trans->UpdateInputDesc(0, input) != GRAPH_SUCCESS) {
            return false;
        }
    }
    return node.UpdateInputDesc(port, input) == GRAPH_SUCCESS;
}

bool RemoveUnusedNodes(const GraphPtr& graph, const TransposeNodes& nodes)
{
    std::set<std::string> removed;
    for (const auto& node : nodes) {
        if (!node) {
            continue;
        }
        AscendString name;
        if (node->GetName(name) != GRAPH_SUCCESS) {
            return false;
        }
        if (removed.count(name.GetString()) != 0) {
            // The legacy RemoveNode sequence fails if two ports identify the same removed node.
            return false;
        }
        if (!node->GetOutDataNodesAndPortIndexs(0).empty()) {
            continue;
        }
        GNodePtr controlSource;
        for (size_t port = 0; port < node->GetInputsSize(); ++port) {
            auto source = node->GetInDataNodesAndPortIndexs(port);
            if (source.first && graph->RemoveEdge(*source.first, source.second, *node, port) != GRAPH_SUCCESS) {
                return false;
            }
            if (!source.first) {
                continue;
            }
            if (IsType(source.first, "Const") || IsType(source.first, "Constant")) {
                bool used = !source.first->GetOutControlNodes().empty();
                for (size_t out = 0; out < source.first->GetOutputsSize(); ++out) {
                    used = used || !source.first->GetOutDataNodesAndPortIndexs(out).empty();
                }
                // ComputeGraph::RemoveNode also removes unused constant inputs.
                if (!used && graph->RemoveNode(*source.first) != GRAPH_SUCCESS) {
                    return false;
                }
            } else if (!controlSource) {
                controlSource = source.first;
            }
        }
        const auto incomingControl = node->GetInControlNodes();
        if (!controlSource && !incomingControl.empty()) {
            controlSource = incomingControl.front();
        }
        if (controlSource) {
            for (auto& successor : node->GetOutControlNodes()) {
                if (graph->AddControlEdge(*controlSource, *successor) != GRAPH_SUCCESS) {
                    return false;
                }
            }
        }
        if (graph->RemoveNode(*node) != GRAPH_SUCCESS) {
            return false;
        }
        removed.insert(name.GetString());
    }
    return true;
}

bool IsX2Dependency(const GNode& node, const GNodePtr& changed)
{
    AscendString changedName;
    changed->GetName(changedName);
    std::vector<GNodePtr> pending = {Input(node, X2_PORT), Input(node, X2_SCALE_PORT)};
    std::set<std::string> visited;
    while (!pending.empty()) {
        auto current = pending.back();
        pending.pop_back();
        if (!current) {
            continue;
        }
        AscendString name;
        current->GetName(name);
        if (name == changedName) {
            return true;
        }
        if (!visited.insert(name.GetString()).second) {
            continue;
        }
        for (size_t port = 0; port < current->GetInputsSize(); ++port) {
            pending.push_back(Input(*current, port));
        }
    }
    return false;
}

Status Fusion(const GraphPtr& graph, GNode& node, CustomPassContext& context)
{
    bool transA = false;
    bool transB = false;
    // Absorbing x1's transpose must produce the only supported backend layout: TN.
    if (node.GetAttr("transpose_x1", transA) != GRAPH_SUCCESS ||
        node.GetAttr("transpose_x2", transB) != GRAPH_SUCCESS || transA || transB) {
        return GRAPH_NOT_CHANGED;
    }
    // Pattern recognition must use this target's shape, independent of earlier targets or runs.
    bool isDynamic = false;
    if (!GetDynamicStatus(node, isDynamic)) {
        return GRAPH_NOT_CHANGED;
    }
    const bool bitcast = IsBitcastPattern(node, isDynamic);
    if ((!PlatformSupportBitcastTransposeFusion() && bitcast) || !CheckNode(node, bitcast)) {
        return GRAPH_NOT_CHANGED;
    }
    TransposeNodes nodes{};
    if (!GetTransposeNodes(node, bitcast, isDynamic, nodes)) {
        return GRAPH_NOT_CHANGED;
    }
    // Rewiring a shared Bitcast or changing its source descriptor must not alter an x2 branch.
    if (bitcast) {
        for (size_t i = 0; i < nodes.size(); ++i) {
            auto cast = Input(node, INPUT_PORTS[i]);
            if (nodes[i] && IsType(cast, "Bitcast") && (IsX2Dependency(node, cast) || IsX2Dependency(node, nodes[i]))) {
                return GRAPH_NOT_CHANGED;
            }
        }
    }
    // Validate transform inputs before changing attributes or reconnecting any edge.
    for (const auto& trans : nodes) {
        if (trans && !Input(*trans, 0)) {
            return GRAPH_NOT_CHANGED;
        }
    }
    transA = true;
    if (node.SetAttr("transpose_x1", transA) != GRAPH_SUCCESS) {
        return GRAPH_NOT_CHANGED;
    }
    for (size_t i = 0; i < nodes.size(); ++i) {
        if (!RelinkNode(graph, node, nodes[i], INPUT_PORTS[i], bitcast)) {
            return GRAPH_FAILED;
        }
    }
    // x2 and x2_scale are outside this pass; their branches must remain intact.
    if (!UpdateBitcastAndInputDesc(node, nodes[0], X1_PORT, bitcast) ||
        !UpdateBitcastAndInputDesc(node, nodes[1], X1_SCALE_PORT, bitcast)) {
        return GRAPH_FAILED;
    }
    if (ge::fusion::GraphFuseInspectorUtils::ReportFuse != nullptr) {
        std::vector<GNode> before = {node};
        for (const auto& trans : nodes) {
            if (trans) {
                before.push_back(*trans);
            }
        }
        // Public GE bookkeeping is not an additional fusion eligibility check.
        (void)ge::fusion::GraphFuseInspectorUtils::ReportFuse(before, {node}, context);
    }
    return RemoveUnusedNodes(graph, nodes) ? SUCCESS : GRAPH_FAILED;
}

std::vector<GNode> MatchPattern(const GNode& node, bool throughBitcast)
{
    // Only x1 can trigger a match; x1_scale is handled with its data transform.
    auto input = Input(node, X1_PORT);
    auto cast = input;
    if (throughBitcast) {
        input = IsType(input, "Bitcast") ? Input(*input, 0) : nullptr;
    }
    if (IsTranspose(input)) {
        // Legacy mappings are traversed in pattern-ID order: bitcast, target, transpose.
        return throughBitcast ? std::vector<GNode>{*cast, node, *input} : std::vector<GNode>{node, *input};
    }
    return {};
}

bool CheckIntAttribute(const std::vector<GNode>& nodes, const char* key, int32_t maximum)
{
    std::set<int32_t> values;
    for (auto node : nodes) {
        if (!node.HasAttr(key)) {
            continue;
        }
        AscendString text;
        if (node.GetAttr(key, text) != GRAPH_SUCCESS) {
            return false;
        }
        try {
            size_t end = 0;
            const std::string value(text.GetString());
            const auto number = std::stoi(value, &end);
            if (end != value.size() || number < 0 || number > maximum) {
                return false;
            }
            values.insert(number);
        } catch (const std::exception&) {
            return false;
        }
    }
    return values.size() <= 1;
}

bool SupportsFusionAttributes(const std::vector<GNode>& nodes)
{
    // Match ComputeGraph::IsSupportFuse. CanFuse additionally checks contraction into one node,
    // which would incorrectly reject a shared Transpose that this pass retains and only bypasses.
    for (const auto* key : {"_user_stream_label", "_super_kernel_scope", "_super_kernel_options", "_op_aicore_num",
                            "_op_vectorcore_num"}) {
        std::set<std::string> values;
        for (const auto& node : nodes) {
            AscendString value;
            if (node.GetAttr(key, value) == GRAPH_SUCCESS) {
                values.insert(value.GetString());
            }
        }
        if (values.size() > 1) {
            return false;
        }
    }
    return CheckIntAttribute(nodes, "_deterministic", MAX_DETERMINISTIC) &&
           CheckIntAttribute(nodes, "_deterministic_level", MAX_DETERMINISTIC_LEVEL);
}

std::vector<GNode> MatchAll(const GraphPtr& graph, bool throughBitcast)
{
    std::vector<GNode> matches;
    for (auto node : graph->GetDirectNode()) {
        AscendString type;
        if (node.GetType(type) != GRAPH_SUCCESS || type != OP_TYPE) {
            continue;
        }
        auto matched = MatchPattern(node, throughBitcast);
        if (matched.empty()) {
            continue;
        }
        // The legacy matcher aborts this pattern's entire match list on a stream-label mismatch.
        AscendString first("");
        for (auto& candidate : matched) {
            AscendString label("null");
            (void)candidate.GetAttr("_stream_label", label);
            if (first == "") {
                first = label;
            } else if (label != first) {
                return {};
            }
        }
        if (SupportsFusionAttributes(matched)) {
            matches.push_back(node);
        }
    }
    return matches;
}
} // namespace

ge::Status QuantBatchMatmulInplaceAddTransposeFusionPass::Run(ge::GraphPtr& graph, ge::CustomPassContext& context)
{
    if (!IsTargetVersion() || !graph || !graph->IsValid()) {
        return ge::GRAPH_NOT_CHANGED;
    }
    context.SetPassName(PASS_NAME);
    bool changed = false;
    for (const bool throughBitcast : {false, true}) {
        for (auto node : MatchAll(graph, throughBitcast)) {
            const auto status = Fusion(graph, node, context);
            if (status != ge::SUCCESS && status != ge::GRAPH_NOT_CHANGED) {
                return status;
            }
            changed = changed || status == ge::SUCCESS;
        }
    }
    return changed ? ge::SUCCESS : ge::GRAPH_NOT_CHANGED;
}

#if GE_COMPILER_VERSION_NUM >= 90100000
REG_FUSION_PASS(QuantBatchMatmulInplaceAddTransposeFusionPass)
    .Stage(IsTargetVersion() ? ge::CustomPassStage::kCompatibleInherited : ge::CustomPassStage::kAfterInferShape);
#endif
} // namespace ops

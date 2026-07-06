/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "matmul_fusion_utils_pass.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include "common/inc/error_util.h"
#include "graph/operator.h"
#include "ge/compliant_node_builder.h"
#include "runtime/base.h"

using namespace ge;
using namespace ge::es;
using namespace fe;

namespace ops {
namespace {

constexpr uint32_t kMaxNpuArchLen = 32U;

bool CopyAscendStringAttr(const GNode& matchedNode, GNode& v3Node, const char* attrName, const std::string& passName)
{
    AscendString attrValue;
    if (matchedNode.GetAttr(attrName, attrValue) == GRAPH_SUCCESS) {
        if (v3Node.SetAttr(attrName, attrValue) != GRAPH_SUCCESS) {
            OPS_LOG_E(passName.c_str(), "Set %s failed.", attrName);
            return false;
        }
    }
    return true;
}

void AddOptionalEdges(Graph& graph, const EsTensorHolder& bias, const EsTensorHolder& offsetW, GNode& node,
                      const char* opType)
{
    if (bias.GetCTensorHolder() != nullptr) {
        if (AddEdgeAndUpdatePeerDesc(graph, *bias.GetProducer(), bias.GetProducerOutIndex(), node, kBiasInputIdx) !=
            GRAPH_SUCCESS) {
            OPS_LOG_E(opType, "AddEdge for %s input bias failed.", opType);
        }
    }
    if (offsetW.GetCTensorHolder() != nullptr) {
        if (AddEdgeAndUpdatePeerDesc(graph, *offsetW.GetProducer(), offsetW.GetProducerOutIndex(), node,
                                     kOffsetWInputIdx) != GRAPH_SUCCESS) {
            OPS_LOG_E(opType, "AddEdge for %s input offset_w failed.", opType);
        }
    }
}

bool IsBatchOp(const char* opType)
{ return strcmp(opType, kOpTypeBatchMatMul) == 0 || strcmp(opType, kOpTypeBatchMatMulV2) == 0; }

bool IsV2Op(const char* opType)
{ return strcmp(opType, kOpTypeMatMulV2) == 0 || strcmp(opType, kOpTypeBatchMatMulV2) == 0; }

ge::fusion::PatternUniqPtr BuildPatternWithOptionalInputs(const std::string& patternName, const char* opType,
                                                          bool hasBias, bool hasOffsetW)
{
    auto graphBuilder = EsGraphBuilder(patternName.c_str());
    auto x1 = graphBuilder.CreateInput(kX1InputIdx);
    auto x2 = graphBuilder.CreateInput(kX2InputIdx);

    EsTensorHolder bias = nullptr;
    EsTensorHolder offsetW = nullptr;
    int64_t inputIdx = kBiasInputIdx;
    if (hasBias) {
        bias = graphBuilder.CreateInput(inputIdx++);
    }
    if (hasOffsetW) {
        offsetW = graphBuilder.CreateInput(inputIdx);
    }

    auto y = CreateMatMulLikeNode(graphBuilder, opType, x1, x2, bias, offsetW);

    auto graph = graphBuilder.BuildAndReset({y});
    auto pattern = std::make_unique<ge::fusion::Pattern>(std::move(*graph));
    pattern->CaptureTensor({*y.GetProducer(), kCaptureTensorIdx});
    return pattern;
}

} // namespace

bool IsSupportL12BtBf16(const PlatformInfo& platformInfo)
{
    auto iter = platformInfo.ai_core_intrinsic_dtype_map.find("Intrinsic_data_move_l12bt");
    if (iter == platformInfo.ai_core_intrinsic_dtype_map.end()) {
        return false;
    }
    return std::find(iter->second.begin(), iter->second.end(), "bf16") != iter->second.end();
}

bool IsDav3510Platform(const std::string& passName)
{
    char npuArch[kMaxNpuArchLen] = {0};
    const auto ret = rtGetSocSpec("version", "NpuArch", npuArch, sizeof(npuArch));
    if (ret != RT_ERROR_NONE) {
        OPS_LOG_W(passName.c_str(), "call rtGetSocSpec failed");
        return false;
    }
    return (std::string(npuArch) == "3510");
}

bool CopyOtherAttrs(const GNode& matchedNode, GNode& v3Node, const std::string& passName)
{
    int64_t opImplModeEnum = 0;
    if (matchedNode.GetAttr("_op_impl_mode_enum", opImplModeEnum) == GRAPH_SUCCESS) {
        if (v3Node.SetAttr("_op_impl_mode_enum", opImplModeEnum) != GRAPH_SUCCESS) {
            OPS_LOG_E(passName.c_str(), "Set _op_impl_mode_enum failed.");
            return false;
        }
    }

    if (!CopyAscendStringAttr(matchedNode, v3Node, "_user_stream_label", passName)) {
        return false;
    }
    if (!CopyAscendStringAttr(matchedNode, v3Node, "_user_stream_priority", passName)) {
        return false;
    }
    if (!CopyAscendStringAttr(matchedNode, v3Node, "_super_kernel_scope", passName)) {
        return false;
    }
    if (!CopyAscendStringAttr(matchedNode, v3Node, "_super_kernel_options", passName)) {
        return false;
    }
    if (!CopyAscendStringAttr(matchedNode, v3Node, "_op_aicore_num", passName)) {
        return false;
    }
    if (!CopyAscendStringAttr(matchedNode, v3Node, "_op_vectorcore_num", passName)) {
        return false;
    }

    return true;
}

EsTensorHolder CreateMatMulLikeNode(EsGraphBuilder& graphBuilder, const char* opType, const EsTensorHolder& x1,
                                    const EsTensorHolder& x2, const EsTensorHolder& bias, const EsTensorHolder& offsetW)
{
    auto* graph = graphBuilder.GetCGraphBuilder()->GetGraph();

    std::vector<CompliantNodeBuilder::IrInputDef> inputs = {
        {"x1", CompliantNodeBuilder::kEsIrInputRequired, ""},
        {"x2", CompliantNodeBuilder::kEsIrInputRequired, ""},
        {"bias", CompliantNodeBuilder::kEsIrInputOptional, ""},
    };
    if (IsV2Op(opType)) {
        inputs.push_back({"offset_w", CompliantNodeBuilder::kEsIrInputOptional, ""});
    }

    const char* transAttr1 = IsBatchOp(opType) ? "adj_x1" : "transpose_x1";
    const char* transAttr2 = IsBatchOp(opType) ? "adj_x2" : "transpose_x2";
    std::vector<CompliantNodeBuilder::IrAttrDef> attrs = {
        {transAttr1, CompliantNodeBuilder::kEsAttrRequired, "Bool", CreateFrom(false)},
        {transAttr2, CompliantNodeBuilder::kEsAttrRequired, "Bool", CreateFrom(false)},
    };
    if (IsV2Op(opType)) {
        attrs.push_back({"offset_x", CompliantNodeBuilder::kEsAttrOptional, "Int", CreateFrom(int64_t(0))});
    }

    auto node = CompliantNodeBuilder(graph)
                    .OpType(opType)
                    .Name(opType)
                    .IrDefInputs(inputs)
                    .IrDefOutputs({{"y", CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                    .IrDefAttrs(attrs)
                    .Build();
    OP_LOGE_IF(AddEdgeAndUpdatePeerDesc(*graph, *x1.GetProducer(), x1.GetProducerOutIndex(), node, kX1InputIdx) !=
                   GRAPH_SUCCESS,
               EsTensorHolder(), opType, "AddEdge for %s input x1 failed.", opType);
    OP_LOGE_IF(AddEdgeAndUpdatePeerDesc(*graph, *x2.GetProducer(), x2.GetProducerOutIndex(), node, kX2InputIdx) !=
                   GRAPH_SUCCESS,
               EsTensorHolder(), opType, "AddEdge for %s input x2 failed.", opType);
    AddOptionalEdges(*graph, bias, offsetW, node, opType);
    auto* yHolder = graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(node, 0);
    return EsTensorHolder(yHolder);
}

std::vector<ge::fusion::PatternUniqPtr> BuildMatMulPatterns(const std::string& prefix)
{
    std::vector<ge::fusion::PatternUniqPtr> patterns;
    patterns.emplace_back(BuildPatternWithOptionalInputs(prefix + "_matmul", kOpTypeMatMul, false, false));
    patterns.emplace_back(BuildPatternWithOptionalInputs(prefix + "_matmul_bias", kOpTypeMatMul, true, false));
    return patterns;
}

std::vector<ge::fusion::PatternUniqPtr> BuildMatMulV2Patterns(const std::string& prefix)
{
    std::vector<ge::fusion::PatternUniqPtr> patterns;
    patterns.emplace_back(BuildPatternWithOptionalInputs(prefix + "_matmulv2", kOpTypeMatMulV2, false, false));
    patterns.emplace_back(BuildPatternWithOptionalInputs(prefix + "_matmulv2_bias", kOpTypeMatMulV2, true, false));
    patterns.emplace_back(BuildPatternWithOptionalInputs(prefix + "_matmulv2_offsetw", kOpTypeMatMulV2, false, true));
    patterns.emplace_back(
        BuildPatternWithOptionalInputs(prefix + "_matmulv2_bias_offsetw", kOpTypeMatMulV2, true, true));
    return patterns;
}

std::vector<ge::fusion::PatternUniqPtr> BuildBatchMatMulPatterns(const std::string& prefix)
{
    std::vector<ge::fusion::PatternUniqPtr> patterns;
    patterns.emplace_back(BuildPatternWithOptionalInputs(prefix + "_batchmatmul", kOpTypeBatchMatMul, false, false));
    patterns.emplace_back(BuildPatternWithOptionalInputs(prefix + "_batchmatmul_bias", kOpTypeBatchMatMul, true, false));
    return patterns;
}

std::vector<ge::fusion::PatternUniqPtr> BuildBatchMatMulV2Patterns(const std::string& prefix)
{
    std::vector<ge::fusion::PatternUniqPtr> patterns;
    patterns.emplace_back(
        BuildPatternWithOptionalInputs(prefix + "_batchmatmulv2", kOpTypeBatchMatMulV2, false, false));
    patterns.emplace_back(
        BuildPatternWithOptionalInputs(prefix + "_batchmatmulv2_bias", kOpTypeBatchMatMulV2, true, false));
    patterns.emplace_back(
        BuildPatternWithOptionalInputs(prefix + "_batchmatmulv2_offsetw", kOpTypeBatchMatMulV2, false, true));
    patterns.emplace_back(
        BuildPatternWithOptionalInputs(prefix + "_batchmatmulv2_bias_offsetw", kOpTypeBatchMatMulV2, true, true));
    return patterns;
}

} // namespace ops

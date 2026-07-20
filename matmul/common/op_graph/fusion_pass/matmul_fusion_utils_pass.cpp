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

using namespace ge;
using namespace ge::es;
using namespace fe;

namespace ops {
namespace {

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
{
    return strcmp(opType, kOpTypeBatchMatMul) == 0 || strcmp(opType, kOpTypeBatchMatMulV2) == 0;
}

bool HasOffsetW(const char* opType)
{
    return strcmp(opType, kOpTypeMatMulV2) == 0 || strcmp(opType, kOpTypeBatchMatMulV2) == 0;
}

ge::fusion::PatternUniqPtr BuildPatternWithInputCount(const std::string& patternName, const char* opType,
                                                      int64_t inputCount)
{
    auto graphBuilder = EsGraphBuilder(patternName.c_str());
    auto x1 = graphBuilder.CreateInput(kX1InputIdx);
    auto x2 = graphBuilder.CreateInput(kX2InputIdx);

    EsTensorHolder bias = nullptr;
    EsTensorHolder offsetW = nullptr;
    int64_t inputIdx = kBiasInputIdx;
    if (inputCount >= kThreeInputNum) {
        bias = graphBuilder.CreateInput(inputIdx++);
    }
    if (inputCount >= kFourInputNum) {
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

static bool CopyInt64Attr(const GNode& matchedNode, GNode& v3Node, const char* attrName, const std::string& passName)
{
    int64_t attrValue = 0;
    if (matchedNode.GetAttr(attrName, attrValue) == GRAPH_SUCCESS) {
        OPS_LOG_D(passName.c_str(), "%s found, value: %ld.", attrName, attrValue);
        if (v3Node.SetAttr(attrName, attrValue) != GRAPH_SUCCESS) {
            OPS_LOG_E(passName.c_str(), "Set %s failed.", attrName);
            return false;
        }
    } else {
        OPS_LOG_D(passName.c_str(), "%s not found, skip.", attrName);
    }
    return true;
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

    // copy fixed_shift_value
    int64_t fixedShiftValue = 0;
    if (matchedNode.GetAttr("fixed_shift_value", fixedShiftValue) == GRAPH_SUCCESS) {
        OPS_LOG_I(passName.c_str(), "Get fixed_shift_value succeeded, value: %ld.", fixedShiftValue);
        if (v3Node.SetAttr("fixed_shift_value", fixedShiftValue) != GRAPH_SUCCESS) {
            OPS_LOG_E(passName.c_str(), "Set fixed_shift_value failed.");
            return false;
        }
    } else {
        OPS_LOG_I(passName.c_str(), "fixed_shift_value not found, skip.");
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
    if (!CopyInt64Attr(matchedNode, v3Node, "enable_uncache", passName)) {
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
    if (HasOffsetW(opType)) {
        inputs.push_back({"offset_w", CompliantNodeBuilder::kEsIrInputOptional, ""});
    }

    const char* transAttr1 = IsBatchOp(opType) ? "adj_x1" : "transpose_x1";
    const char* transAttr2 = IsBatchOp(opType) ? "adj_x2" : "transpose_x2";
    std::vector<CompliantNodeBuilder::IrAttrDef> attrs = {
        {transAttr1, CompliantNodeBuilder::kEsAttrRequired, "Bool", CreateFrom(false)},
        {transAttr2, CompliantNodeBuilder::kEsAttrRequired, "Bool", CreateFrom(false)},
    };
    if (HasOffsetW(opType)) {
        attrs.push_back({"offset_x", CompliantNodeBuilder::kEsAttrOptional, "Int", AttrValue()});
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
    patterns.emplace_back(BuildPatternWithInputCount(prefix + "_matmul_2in", kOpTypeMatMul, kBaseNodeNum));
    patterns.emplace_back(BuildPatternWithInputCount(prefix + "_matmul_3in", kOpTypeMatMul, kThreeInputNum));
    return patterns;
}

std::vector<ge::fusion::PatternUniqPtr> BuildMatMulV2Patterns(const std::string& prefix)
{
    std::vector<ge::fusion::PatternUniqPtr> patterns;
    patterns.emplace_back(BuildPatternWithInputCount(prefix + "_matmulv2_2in", kOpTypeMatMulV2, kBaseNodeNum));
    patterns.emplace_back(BuildPatternWithInputCount(prefix + "_matmulv2_3in", kOpTypeMatMulV2, kThreeInputNum));
    patterns.emplace_back(BuildPatternWithInputCount(prefix + "_matmulv2_4in", kOpTypeMatMulV2, kFourInputNum));
    return patterns;
}

std::vector<ge::fusion::PatternUniqPtr> BuildBatchMatMulPatterns(const std::string& prefix)
{
    std::vector<ge::fusion::PatternUniqPtr> patterns;
    patterns.emplace_back(BuildPatternWithInputCount(prefix + "_batchmatmul_2in", kOpTypeBatchMatMul, kBaseNodeNum));
    patterns.emplace_back(BuildPatternWithInputCount(prefix + "_batchmatmul_3in", kOpTypeBatchMatMul, kThreeInputNum));
    return patterns;
}

std::vector<ge::fusion::PatternUniqPtr> BuildBatchMatMulV2Patterns(const std::string& prefix)
{
    std::vector<ge::fusion::PatternUniqPtr> patterns;
    patterns.emplace_back(
        BuildPatternWithInputCount(prefix + "_batchmatmulv2_2in", kOpTypeBatchMatMulV2, kBaseNodeNum));
    patterns.emplace_back(
        BuildPatternWithInputCount(prefix + "_batchmatmulv2_3in", kOpTypeBatchMatMulV2, kThreeInputNum));
    patterns.emplace_back(
        BuildPatternWithInputCount(prefix + "_batchmatmulv2_4in", kOpTypeBatchMatMulV2, kFourInputNum));
    return patterns;
}

} // namespace ops

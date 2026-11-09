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
#include <limits>
#include <set>
#include <string>
#include <vector>

#include "securec.h"
#include "common/inc/error_util.h"
#include "version/ge-compiler_version.h"
#include "op_common/op_host/util/op_const_def.h"
#include "acl/acl_rt.h"
#include "acl/acl_platform.h"

using namespace ge;
using namespace ge::es;
using namespace fe;

namespace ge {
namespace fusion {
class GraphFuseInspectorUtils {
public:
    static Status ReportFuse(const std::vector<GNode>& nodesBeforeFuse, const std::vector<GNode>& nodesAfterFuse,
                             CustomPassContext& ctx) __attribute__((weak));
};
} // namespace fusion
} // namespace ge

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

// 判断当前ARCH是否为DAV_3510或后续兼容版本，后续新增兼容ARCH在此函数内追加
bool IsNpuArch3510Series()
{
    char buf[64] = {0};
    aclError ret = aclplatformGetDeviceInfo(ACL_PLATFORM_NPU_ARCH, buf, sizeof(buf));
    if (ret != ACL_SUCCESS) {
        OPS_LOG_E("IsNpuArch3510Series", "get npu arch failed.");
        return false;
    }

    return (std::atoi(buf) == static_cast<int32_t>(Ops::Base::DAV_3510) ||
            std::atoi(buf) == static_cast<int32_t>(Ops::Base::DAV_9201) ||
            std::atoi(buf) == static_cast<int32_t>(Ops::Base::DAV_9202));
}

constexpr int32_t kGeCompilerVersion900 = 90000000;

int32_t GetGeCompilerVersionNum()
{
    int32_t version = 0;
    if (aclsysGetVersionNum("ge-compiler", &version) != ACL_SUCCESS) {
        OPS_LOG_E("GetGeCompilerVersionNum", "aclsysGetVersionNum failed.");
        return 0;
    }
    return version;
}

ge::CustomPassStage GetCompatPassStage()
{
#if defined(GE_COMPILER_VERSION_NUM) && (GE_COMPILER_VERSION_NUM >= 90000000) // mirrors kGeCompilerVersion900
    if (GetGeCompilerVersionNum() >= kGeCompilerVersion900) {
        return ge::CustomPassStage::kCompatibleInherited;
    }
    return ge::CustomPassStage::kBeforeInferShape;
#else
    return ge::CustomPassStage::kBeforeInferShape;
#endif
}

static bool CopyInt64Attr(const GNode& matchedNode, GNode& v3Node, const char* attrName, const std::string& passName)
{
    int64_t attrValue = 0;
    if (matchedNode.GetAttr(attrName, attrValue) == GRAPH_SUCCESS) {
        OPS_LOG_D(passName.c_str(), "%s found, value: %lld.", attrName, static_cast<long long>(attrValue));
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
        OPS_LOG_I(passName.c_str(), "Get fixed_shift_value succeeded, value: %lld.",
                  static_cast<long long>(fixedShiftValue));
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

void ReportFusion(const std::vector<GNode>& nodesBeforeFuse, const std::vector<GNode>& nodesAfterFuse,
                  CustomPassContext& passContext, const std::string& passName)
{
    if (ge::fusion::GraphFuseInspectorUtils::ReportFuse == nullptr) {
        return;
    }
    if (ge::fusion::GraphFuseInspectorUtils::ReportFuse(nodesBeforeFuse, nodesAfterFuse, passContext) != SUCCESS) {
        OPS_LOG_W(passName.c_str(), "Failed to report fusion result.");
    }
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

    const char* transAttr1 = IsBatchOp(opType) ? kAttrAdjX1 : kAttrTransposeX1;
    const char* transAttr2 = IsBatchOp(opType) ? kAttrAdjX2 : kAttrTransposeX2;
    std::vector<CompliantNodeBuilder::IrAttrDef> attrs = {
        {transAttr1, CompliantNodeBuilder::kEsAttrRequired, "Bool", CreateFrom(false)},
        {transAttr2, CompliantNodeBuilder::kEsAttrRequired, "Bool", CreateFrom(false)},
    };
    if (HasOffsetW(opType)) {
        attrs.push_back(
            {kAttrOffsetX, CompliantNodeBuilder::kEsAttrOptional, "Int", CreateFrom(static_cast<int64_t>(0))});
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

bool GetTransposePermFromConst(const GNode& transposeNode, std::vector<int64_t>& permValue)
{
    ge::Tensor permConst;
    if (transposeNode.GetInputConstData(1, permConst) != GRAPH_SUCCESS) {
        OPS_LOG_W("GetTransposePerm", "Get perm const data failed.");
        return false;
    }
    TensorDesc permDesc;
    if (transposeNode.GetInputDesc(1, permDesc) != GRAPH_SUCCESS) {
        return false;
    }
    auto permShape = permDesc.GetShape().GetDims();
    if (permShape.size() != 1) {
        OPS_LOG_D("GetTransposePerm", "The perm dim size must be 1, but got %zu.", permShape.size());
        return false;
    }

    DataType permDtype = permDesc.GetDataType();
    const uint8_t* permData = permConst.GetData();
    if (permData == nullptr) {
        OPS_LOG_E("GetTransposePerm", "Get perm data failed.");
        return false;
    }

    if (permDtype == DT_INT32) {
        size_t cnt = permConst.GetSize() / sizeof(int32_t);
        for (size_t i = 0; i < cnt; ++i) {
            int32_t val = 0;
            errno_t ret = memcpy_s(&val, sizeof(int32_t), permData + i * sizeof(int32_t), sizeof(int32_t));
            if (ret != EOK) {
                OPS_LOG_E("GetTransposePerm", "memcpy_s failed, ret=%d.", ret);
                return false;
            }
            permValue.push_back(static_cast<int64_t>(val));
        }
    } else if (permDtype == DT_INT64) {
        size_t cnt = permConst.GetSize() / sizeof(int64_t);
        for (size_t i = 0; i < cnt; ++i) {
            int64_t val = 0;
            errno_t ret = memcpy_s(&val, sizeof(int64_t), permData + i * sizeof(int64_t), sizeof(int64_t));
            if (ret != EOK) {
                OPS_LOG_E("GetTransposePerm", "memcpy_s failed, ret=%d.", ret);
                return false;
            }
            permValue.push_back(val);
        }
    } else {
        OPS_LOG_D("GetTransposePerm", "Perm dtype is not int32 or int64.");
        return false;
    }
    return true;
}

// ============================================================================
// compile option helpers (bridged to CustomPassContext::GetOptionValue)
// ============================================================================

namespace {

// The option value keys tried in order: the full GE session option key first, then the short
// atc style key. An absent option returns an empty value and the caller degrades gracefully.
constexpr char kPrecisionModeOptionKey[] = "ge.exec.precision_mode";
constexpr char kPrecisionModeOptionKeyShort[] = "precision_mode";
constexpr char kCustomizeDtypesOptionKey[] = "ge.exec.customize_dtypes";
constexpr char kCustomizeDtypesOptionKeyShort[] = "customize_dtypes";
// Every option is queried with exactly one long key and one short key.
constexpr size_t kOptionKeyNum = 2;

std::string GetOptionValueStr(const ge::CustomPassContext& passContext, const char* const* keys, size_t keyNum)
{
    for (size_t index = 0; index < keyNum; index++) {
        ge::AscendString optionValue;
        if (passContext.GetOptionValue(ge::AscendString(keys[index]), optionValue) == ge::GRAPH_SUCCESS) {
            return std::string(optionValue.GetString());
        }
    }
    return "";
}

// The customize_dtypes option grammar (atc style):
//   "opType:inIdx0:dtype0,inIdx1:dtype1:outIdx0:dtypeOut0;opName:..."
// The input entries come before the first output entry whose index counts from the output side
// (the atc doc marks outputs with the "out" prefix), e.g. "Op:0:fp16,out0:fp16".
ge::DataType ParseDtypeToken(const std::string& token)
{
    if (token == "fp16" || token == "float16") {
        return ge::DT_FLOAT16;
    }
    if (token == "fp32" || token == "float32" || token == "float") {
        return ge::DT_FLOAT;
    }
    if (token == "bf16") {
        return ge::DT_BF16;
    }
    if (token == "int8" || token == "s8") {
        return ge::DT_INT8;
    }
    if (token == "uint8" || token == "u8") {
        return ge::DT_UINT8;
    }
    if (token == "int32" || token == "s32") {
        return ge::DT_INT32;
    }
    if (token == "int64") {
        return ge::DT_INT64;
    }
    return ge::DT_UNDEFINED;
}

// Parse one customize entry like "opSelector:0:fp16,out0:fp16" into the dtype lists.
bool ParseCustomizeEntry(const std::string& entry, std::string& opSelector, CustomizeDtype& customDtype)
{
    const size_t selectorEnd = entry.find(':');
    if (selectorEnd == std::string::npos) {
        return false;
    }
    opSelector = entry.substr(0, selectorEnd);
    std::string rest = entry.substr(selectorEnd + 1);

    size_t pos = 0;
    while (pos < rest.size()) {
        const size_t comma = rest.find(',', pos);
        const std::string item = (comma == std::string::npos) ? rest.substr(pos) : rest.substr(pos, comma - pos);
        // item looks like "0:fp16" or "out0:fp16"
        const size_t colon = item.find(':');
        if (colon == std::string::npos) {
            return false;
        }
        const std::string indexToken = item.substr(0, colon);
        const ge::DataType dtype = ParseDtypeToken(item.substr(colon + 1));
        if (dtype == ge::DT_UNDEFINED) {
            return false;
        }
        if (indexToken.rfind("out", 0) == 0) {
            customDtype.outputDtypes.emplace_back(dtype);
        } else {
            customDtype.inputDtypes.emplace_back(dtype);
        }
        if (comma == std::string::npos) {
            break;
        }
        pos = comma + 1;
    }
    return !opSelector.empty() && (!customDtype.inputDtypes.empty() || !customDtype.outputDtypes.empty());
}

} // namespace

std::string GetPrecisionModeStr(const ge::CustomPassContext& passContext)
{
    const char* keys[] = {kPrecisionModeOptionKey, kPrecisionModeOptionKeyShort};
    return GetOptionValueStr(passContext, keys, kOptionKeyNum);
}

// The customize_dtypes entries share one selector namespace (the atc grammar mixes the operator
// type and the operator instance name entries), so the query is one exact selector match.
bool GetCustomizeDtypeBySelector(const ge::CustomPassContext& passContext, const std::string& selector)
{
    const char* keys[] = {kCustomizeDtypesOptionKey, kCustomizeDtypesOptionKeyShort};
    const std::string optionValue = GetOptionValueStr(passContext, keys, kOptionKeyNum);
    if (optionValue.empty()) {
        return false;
    }
    // Entries are separated by ';', each entry is "selector:item,item...".
    size_t pos = 0;
    while (pos < optionValue.size()) {
        const size_t semicolon = optionValue.find(';', pos);
        const std::string entry = (semicolon == std::string::npos) ? optionValue.substr(pos) :
                                                                     optionValue.substr(pos, semicolon - pos);
        std::string entrySelector;
        CustomizeDtype entryDtypes;
        if (ParseCustomizeEntry(entry, entrySelector, entryDtypes) && entrySelector == selector) {
            return true; // the selector is overridden, the caller must respect the given dtypes
        }
        if (semicolon == std::string::npos) {
            break;
        }
        pos = semicolon + 1;
    }
    return false;
}

} // namespace ops

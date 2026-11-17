/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "matmul_quant_to_fixpipe_fusion_pass.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <set>
#include <type_traits>
#include <utility>
#include "acl/acl_rt.h"
#include "ge/compliant_node_builder.h"
#include "ge/es_graph_builder.h"
#include "ge/fusion/pass/pattern_fusion_pass.h"
#include "graph/utils/type_utils.h"
#include "platform/platform_info.h"

#include "version/ge-compiler_version.h"

// Weakly import the public TensorDesc methods so pre-API runtimes can load this plugin.
// CANN package and GE component versions are independent: the verified CANN
// 9.3.0 weekly.20261014.01 package reports GE 9.2.0 and provides these methods,
// while the tested older CANN 9.2.0 package also reports GE 9.2.0 but lacks them.
// The GE version alone therefore cannot establish public API availability.
namespace ops {
namespace fixpipe_public_api {
extern void GetOriginDataType() __asm__("_ZNK2ge10TensorDesc17GetOriginDataTypeEv") __attribute__((weak));
extern void SetOriginDataType() __asm__("_ZN2ge10TensorDesc17SetOriginDataTypeENS_8DataTypeE") __attribute__((weak));
} // namespace fixpipe_public_api
} // namespace ops

namespace ops {
namespace {
using namespace ge;
using Builder = ge::es::CompliantNodeBuilder;
constexpr char PASS_NAME[] = "MatmulQuantToFixpipeFusionPass";
// Minimum GE component version for "ge-compiler", not the CANN package version.
// The verified CANN 9.3 weekly package still reports GE 9.2; also check the APIs.
constexpr int32_t TARGET_VERSION = 90200000;
constexpr int64_t BIAS_INPUT = 2;
constexpr int64_t SCALE_INPUT = 1;
constexpr uint32_t SCALE_MASK = 0xffffe000U;
constexpr uint64_t OFFSET_MASK = 0x00001fe000000000ULL;
constexpr uint64_t OFFSET_SIGN = 0x0000200000000000ULL;
constexpr uint64_t SIGN_FLAG = 0x0000400000000000ULL;
constexpr uint32_t OFFSET_SHIFT = 37;

// Matched nodes and prepared descriptors belong to one candidate. Failed matches and subsequent graphs cannot reuse it.
struct FusionMatch {
    FusionMatch(const GNode& source, const GNode& quant, const GNode& last, bool three)
        : matmul(source), dequantOrRequant(quant), outputNode(last), isThreeNodePattern(three)
    {}
    // Two-node: outputNode == dequantOrRequant; three-node: outputNode is AscendQuant.
    GNode matmul;
    GNode dequantOrRequant;
    GNode outputNode;
    bool isThreeNodePattern = false;
    bool isWeightOnly = false;
    bool transposeA = false;
    bool transposeB = false;
    // Descriptors are normalized by PrepareFusionMatch before any graph mutation.
    TensorDesc x1Desc;
    TensorDesc x2Desc;
    TensorDesc scaleDesc;
    TensorDesc biasDesc;
    TensorDesc outputDesc;
    std::vector<uint64_t> encodedScale;
};

template <typename T, typename = void>
struct HasOriginDataType : std::false_type {};

template <typename T>
struct HasOriginDataType<T, std::void_t<decltype(std::declval<const T&>().GetOriginDataType()),
                                        decltype(std::declval<T&>().SetOriginDataType(DT_UNDEFINED))>>
    : std::true_type {};

bool PublicOriginApiAvailable()
{
    // Probe public declarations and runtime exports, not the CANN package version.
    return HasOriginDataType<TensorDesc>::value && fixpipe_public_api::GetOriginDataType != nullptr &&
           fixpipe_public_api::SetOriginDataType != nullptr;
}

template <typename T>
bool SetOriginDataType(T& desc, DataType type)
{
    if constexpr (HasOriginDataType<T>::value) {
        desc.SetOriginDataType(type);
        return true;
    }
    return false;
}

template <typename T>
DataType GetOriginDataType(const T& desc)
{
    if constexpr (HasOriginDataType<T>::value) {
        return desc.GetOriginDataType();
    }
    return DT_UNDEFINED;
}

bool IsTargetVersion()
{
    int32_t version = 0;
    char component[] = "ge-compiler";
    // Query the GE component; a CANN 9.3 package need not report GE 9.3 here.
    return aclsysGetVersionNum(component, &version) == 0 && version >= TARGET_VERSION && PublicOriginApiAvailable();
}

bool IsNodeType(const GNode& node, const char* type)
{
    AscendString actual;
    return node.GetType(actual) == GRAPH_SUCCESS && actual == type;
}

std::string GetNodeName(const GNode& node)
{
    AscendString name;
    (void)node.GetName(name);
    return name.GetString() == nullptr ? "" : name.GetString();
}

std::pair<GNodePtr, int32_t> GetInputSource(const GNode& node, int64_t port)
{
    if (port < 0) {
        return {nullptr, 0};
    }
    const auto input = node.GetInDataNodesAndPortIndexs(port);
    return {input.first, static_cast<int32_t>(input.second)};
}

bool PlatformSupportsFusion(bool requiresExtendedSupport)
{
    fe::PlatformInfo platform;
    fe::OptionalInfo optional;
    // Evaluate the returned capability entries even if the platform query reports failure.
    (void)fe::PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platform, optional);
    const auto& caps = platform.ai_core_intrinsic_dtype_map;
    const auto supportsDtype = [&caps](const char* key, const char* dtype) {
        const auto it = caps.find(key);
        return it != caps.end() && std::find(it->second.begin(), it->second.end(), dtype) != it->second.end();
    };
    return caps.count("Intrinsic_fix_pipe_l0c2out") != 0 && caps.count("Intrinsic_data_move_l0c2ub") == 0 &&
           (caps.count("Intrinsic_fix_pipe_l0c2ub") == 0 || supportsDtype("Intrinsic_data_move_l12bt", "bf16")) &&
           (!requiresExtendedSupport || supportsDtype("Intrinsic_mmad", "s8s4"));
}

bool IsSupportedDataFormat(Format format)
{
    const std::set<Format> formats = {FORMAT_ND,    FORMAT_NCHW,  FORMAT_NHWC,  FORMAT_HWCN, FORMAT_CHWN,
                                      FORMAT_NDHWC, FORMAT_NCDHW, FORMAT_DHWCN, FORMAT_DHWNC};
    return formats.count(format) != 0;
}

// Scale lookup and descriptor preparation.
std::set<int32_t> GetConnectedInputPorts(const GraphPtr& graph, const GNode& node)
{
    std::set<int32_t> ports;
    const auto inputCount = node.GetInputsSize();
    for (size_t port = 0; port < inputCount; ++port) {
        if (!GetInputSource(node, port).first) {
            break;
        }
        ports.insert(static_cast<int32_t>(port));
    }
    if (inputCount != 0 && ports.size() == inputCount) {
        return ports;
    }
    // GetInputsSize counts valid descriptors, not input slots. Recover sparse
    // ports from actual edges without probing absent slots or using private APIs.
    ports.clear();
    const auto nodeName = GetNodeName(node);
    for (const auto& producer : graph->GetDirectNode()) {
        for (size_t output = 0; output < producer.GetOutputsSize(); ++output) {
            for (const auto& consumer : producer.GetOutDataNodesAndPortIndexs(output)) {
                if (consumer.first && GetNodeName(*consumer.first) == nodeName) {
                    ports.insert(consumer.second);
                }
            }
        }
    }
    return ports;
}

// Read direct constant values or constants on the scale producer's input ports.
// Do not recursively evaluate an Identity chain.
bool ReadScaleWeights(const GraphPtr& graph, const GNode& dequantOrRequant, Tensor& weights)
{
    const auto source = GetInputSource(dequantOrRequant, SCALE_INPUT).first;
    if (!source) {
        return false;
    }
    if (IsNodeType(*source, "Const") || IsNodeType(*source, "Constant") || IsNodeType(*source, "PlaceHolder")) {
        return source->GetAttr("value", weights) == GRAPH_SUCCESS && weights.GetData() != nullptr;
    }
    const auto parent = graph->GetParentNode();
    const bool whileBody = parent && (IsNodeType(*parent, "While") || IsNodeType(*parent, "StatelessWhile") ||
                                      IsNodeType(*parent, "_While"));
    if (IsNodeType(*source, "Data")) {
        return !whileBody && dequantOrRequant.GetInputConstData(SCALE_INPUT, weights) == GRAPH_SUCCESS &&
               weights.GetData() != nullptr;
    }
    // Validate every direct Const input; retain the first resolved constant in input-port order.
    bool found = false;
    for (const auto port : GetConnectedInputPorts(graph, *source)) {
        const auto input = GetInputSource(*source, port).first;
        if (!input) {
            continue;
        }
        Tensor current;
        if (IsNodeType(*input, "Const")) {
            if (input->GetAttr("value", current) != GRAPH_SUCCESS || current.GetData() == nullptr) {
                return false;
            }
        } else if (!IsNodeType(*input, "Data") || source->GetInputConstData(port, current) != GRAPH_SUCCESS) {
            continue;
        }
        if (!found) {
            weights = current;
            found = true;
        }
    }
    return found && weights.GetData() != nullptr;
}

bool EncodeScale(const GraphPtr& graph, FusionMatch& match)
{
    Tensor weights;
    float factor = 0;
    float offset = 0;
    if (!ReadScaleWeights(graph, match.dequantOrRequant, weights) || weights.GetData() == nullptr ||
        match.outputNode.GetAttr("scale", factor) != GRAPH_SUCCESS ||
        match.outputNode.GetAttr("offset", offset) != GRAPH_SUCCESS) {
        return false;
    }
    // Reject offsets that cannot be rounded and converted to int64_t safely.
    const double rounded = std::round(static_cast<double>(offset));
    if (!std::isfinite(rounded) || rounded < -0x1p63 || rounded >= 0x1p63) {
        return false;
    }
    const auto integerOffset = static_cast<int64_t>(rounded);
    // Shift an unsigned offset to preserve the packed bits without signed-shift undefined behavior.
    const auto offsetBits = (static_cast<uint64_t>(integerOffset) << OFFSET_SHIFT) & OFFSET_MASK;
    const size_t count = weights.GetSize() / sizeof(uint64_t);
    match.encodedScale.resize(count);
    for (size_t i = 0; i < count; ++i) {
        uint64_t original = 0;
        std::memcpy(&original, weights.GetData() + i * sizeof(uint64_t), sizeof(original));
        uint32_t bits = static_cast<uint32_t>(original) & SCALE_MASK;
        float decoded = 0;
        std::memcpy(&decoded, &bits, sizeof(decoded));
        const float product = decoded * factor;
        std::memcpy(&bits, &product, sizeof(bits));
        match.encodedScale[i] = static_cast<uint64_t>(bits & SCALE_MASK) | offsetBits | (offset < 0 ? OFFSET_SIGN : 0) |
                                SIGN_FLAG;
    }
    match.scaleDesc = TensorDesc(Shape({static_cast<int64_t>(count)}), FORMAT_ND, DT_UINT64);
    match.scaleDesc.SetOriginShape(match.scaleDesc.GetShape());
    match.scaleDesc.SetOriginFormat(FORMAT_ND);
    return SetOriginDataType(match.scaleDesc, DT_UINT64);
}

bool ReadMatmulDescriptors(FusionMatch& match)
{
    if (match.matmul.GetInputsSize() > 3 || match.matmul.GetInputDesc(0, match.x1Desc) != GRAPH_SUCCESS ||
        match.matmul.GetInputDesc(1, match.x2Desc) != GRAPH_SUCCESS ||
        match.matmul.GetOutputDesc(0, match.outputDesc) != GRAPH_SUCCESS) {
        return false;
    }
    const auto x1Type = match.x1Desc.GetDataType();
    if ((x1Type != DT_INT8 && x1Type != DT_FLOAT16 && x1Type != DT_FLOAT) || match.x2Desc.GetDataType() != DT_INT8) {
        return false;
    }
    match.isWeightOnly = x1Type != DT_INT8;
    if (match.isWeightOnly) {
        return IsSupportedDataFormat(match.x1Desc.GetFormat()) && IsSupportedDataFormat(match.x2Desc.GetFormat());
    }
    return IsSupportedDataFormat(match.x1Desc.GetFormat()) &&
           (IsSupportedDataFormat(match.x2Desc.GetFormat()) || match.x2Desc.GetFormat() == FORMAT_FRACTAL_NZ);
}

bool ReadOutputDataType(const FusionMatch& match, DataType& outputType)
{
    TensorDesc outputDesc;
    if (match.outputNode.GetOutputDesc(0, outputDesc) != GRAPH_SUCCESS) {
        return false;
    }
    outputType = outputDesc.GetDataType();
    if (match.isThreeNodePattern) {
        return outputType == DT_INT8;
    }
    return outputType == DT_INT8 || outputType == DT_FLOAT16 || outputType == DT_FLOAT;
}

bool ReadTransposeAttributes(FusionMatch& match)
{
    const bool isBatchMatmul = IsNodeType(match.matmul, "BatchMatMul") || IsNodeType(match.matmul, "BatchMatMulV2");
    return match.matmul.GetAttr(isBatchMatmul ? "adj_x1" : "transpose_x1", match.transposeA) == GRAPH_SUCCESS &&
           match.matmul.GetAttr(isBatchMatmul ? "adj_x2" : "transpose_x2", match.transposeB) == GRAPH_SUCCESS;
}

bool PrepareFusionDescriptors(FusionMatch& match, DataType outputType)
{
    if (!match.isThreeNodePattern && IsNodeType(match.dequantOrRequant, "AscendDequant")) {
        // The fused operator uses FP16 output for the two-node AscendDequant pattern.
        outputType = DT_FLOAT16;
    }
    // Take the output shape from MatMul and the dtype from the selected quantization pattern.
    match.outputDesc = TensorDesc(match.outputDesc.GetShape(), FORMAT_ND, outputType);
    match.outputDesc.SetOriginShape(match.outputDesc.GetShape());
    match.outputDesc.SetOriginFormat(FORMAT_ND);
    if (!SetOriginDataType(match.outputDesc, outputType)) {
        return false;
    }
    match.x1Desc.SetFormat(FORMAT_ND);
    match.x1Desc.SetOriginFormat(FORMAT_ND);
    if (match.x2Desc.GetFormat() != FORMAT_FRACTAL_NZ) {
        match.x2Desc.SetFormat(FORMAT_ND);
        match.x2Desc.SetOriginFormat(FORMAT_ND);
    }
    return !GetInputSource(match.matmul, BIAS_INPUT).first ||
           match.matmul.GetInputDesc(BIAS_INPUT, match.biasDesc) == GRAPH_SUCCESS;
}

bool PrepareFusionMatch(const GraphPtr& graph, FusionMatch& match)
{
    if (!ReadMatmulDescriptors(match) || !PlatformSupportsFusion(match.isWeightOnly || match.isThreeNodePattern)) {
        return false;
    }
    DataType outputType = DT_UNDEFINED;
    if (!ReadOutputDataType(match, outputType) || !ReadTransposeAttributes(match)) {
        return false;
    }
    if (match.isThreeNodePattern) {
        if (!EncodeScale(graph, match)) {
            return false;
        }
    } else if (match.dequantOrRequant.GetInputDesc(SCALE_INPUT, match.scaleDesc) != GRAPH_SUCCESS) {
        return false;
    }
    return PrepareFusionDescriptors(match, outputType);
}

// Execution and compilation attributes are inherited by the public rewriter.
// Only the output-level dump mapping is handled here.
Status CopyDumpOutputMapping(const GNode& source, GNode& target)
{
    const auto readText = [&source](const char* key, AscendString& text) {
        AttrValue value;
        return source.GetOutputAttr(key, 0, value) == GRAPH_SUCCESS && value.GetAttrValue(text) == GRAPH_SUCCESS &&
               text.GetString() != nullptr;
    };
    const auto writeText = [&target](const char* key, const AscendString& text) {
        return target.SetOutputAttr(key, 0, ge::es::CreateFrom(text));
    };
    AscendString name;
    if (!readText("_datadump_origin_name", name) || name.GetString()[0] == '\0') {
        std::vector<AscendString> names;
        if (source.GetAttr("_datadump_original_op_names", names) == GRAPH_SUCCESS && !names.empty()) {
            name = names.front();
        } else if (source.GetName(name) != GRAPH_SUCCESS) {
            return GRAPH_FAILED;
        }
    }
    AttrValue value;
    int64_t index = 0;
    if (source.GetOutputAttr("_datadump_origin_output_index", 0, value) == GRAPH_SUCCESS) {
        (void)value.GetAttrValue(index);
    }
    TensorDesc desc;
    if (source.GetOutputDesc(0, desc) != GRAPH_SUCCESS) {
        return GRAPH_FAILED;
    }
    AscendString format;
    if (!readText("_datadump_origin_format", format) ||
        TypeUtils::SerialStringToFormat(format.GetString()) == FORMAT_RESERVED) {
        format = AscendString(TypeUtils::FormatToSerialString(desc.GetOriginFormat()).c_str());
    }
    AscendString dtype;
    if (!readText("_datadump_origin_data_type", dtype) ||
        TypeUtils::SerialStringToDataType(dtype.GetString()) == DT_UNDEFINED) {
        const auto originType = GetOriginDataType(desc);
        dtype = AscendString(originType == DT_UNDEFINED ? "RESERVED" :
                                                          TypeUtils::DataTypeToSerialString(originType).c_str());
    }
    if (writeText("_datadump_origin_name", name) != GRAPH_SUCCESS ||
        target.SetOutputAttr("_datadump_origin_output_index", 0, ge::es::CreateFrom(index)) != GRAPH_SUCCESS ||
        writeText("_datadump_origin_data_type", dtype) != GRAPH_SUCCESS ||
        writeText("_datadump_origin_format", format) != GRAPH_SUCCESS) {
        return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
}

// Graph mutation: build replacements, connect edges, report, then remove originals.
Status ConnectInput(const GraphPtr& graph, GNode& source, int64_t sourcePort, GNode& target, int64_t targetPort,
                    const TensorDesc& desc)
{
    if (target.UpdateInputDesc(targetPort, desc) != GRAPH_SUCCESS) {
        return GRAPH_FAILED;
    }
    return graph->AddDataEdge(source, sourcePort, target, targetPort);
}

struct FusionInputs {
    std::pair<GNodePtr, int32_t> x1;
    std::pair<GNodePtr, int32_t> x2;
    std::pair<GNodePtr, int32_t> scale;
    std::pair<GNodePtr, int32_t> bias;
};

bool ReadFusionInputs(FusionMatch& match, FusionInputs& inputs)
{
    inputs.x1 = GetInputSource(match.matmul, 0);
    inputs.x2 = GetInputSource(match.matmul, 1);
    inputs.bias = GetInputSource(match.matmul, BIAS_INPUT);
    inputs.scale = GetInputSource(match.dequantOrRequant, SCALE_INPUT);
    // Accept an unconnected optional bias only when the named bias IR slot exists at port 2.
    int32_t biasIndex = -1;
    const bool hasBiasSlot = match.matmul.GetInputIndexByName("bias", biasIndex) == GRAPH_SUCCESS &&
                             biasIndex == BIAS_INPUT;
    return inputs.x1.first && inputs.x2.first && inputs.scale.first && (hasBiasSlot || inputs.bias.first);
}

GNode CreateFusedMatmul(const GraphPtr& graph, const FusionMatch& match)
{
    Builder builder(graph.get());
    builder.OpType(match.isWeightOnly ? "WeightQuantBatchMatmulV2" : "QuantBatchMatmulV3")
        .Name((GetNodeName(match.matmul) + (match.isWeightOnly ? "weightQuantBatchmatmulv2_" : "quantBatchmatmulv3_"))
                  .c_str());
    // Declare the target IR exactly; the rewriter validates it against the runtime registry.
    if (match.isWeightOnly) {
        builder.IrDefInputs({{"x", Builder::kEsIrInputRequired, ""},
                             {"weight", Builder::kEsIrInputRequired, ""},
                             {"antiquant_scale", Builder::kEsIrInputRequired, ""},
                             {"antiquant_offset", Builder::kEsIrInputOptional, ""},
                             {"quant_scale", Builder::kEsIrInputOptional, ""},
                             {"quant_offset", Builder::kEsIrInputOptional, ""},
                             {"bias", Builder::kEsIrInputOptional, ""}});
    } else {
        builder.IrDefInputs({{"x1", Builder::kEsIrInputRequired, ""},
                             {"x2", Builder::kEsIrInputRequired, ""},
                             {"scale", Builder::kEsIrInputRequired, ""},
                             {"offset", Builder::kEsIrInputOptional, ""},
                             {"bias", Builder::kEsIrInputOptional, ""},
                             {"pertoken_scale", Builder::kEsIrInputOptional, ""}});
    }
    return builder.IrDefOutputs({{"y", Builder::kEsIrOutputRequired, ""}})
        .InstanceOutputDataType("y", match.outputDesc.GetDataType())
        .InstanceOutputShape("y", match.outputDesc.GetShape().GetDims())
        .InstanceOutputFormat("y", FORMAT_ND)
        .Build();
}

Status SetFusedMatmulAttributes(const FusionMatch& match, GNode& target)
{
    int64_t outputType = static_cast<int64_t>(match.outputDesc.GetDataType());
    int64_t groupSize = 0;
    // GNode::SetAttr takes mutable references in older public SDK headers.
    bool transposeA = match.transposeA;
    bool transposeB = match.transposeB;
    if (target.UpdateOutputDesc(0, match.outputDesc) != GRAPH_SUCCESS ||
        target.SetAttr(match.isWeightOnly ? "transpose_x" : "transpose_x1", transposeA) != GRAPH_SUCCESS ||
        target.SetAttr(match.isWeightOnly ? "transpose_weight" : "transpose_x2", transposeB) != GRAPH_SUCCESS ||
        target.SetAttr("dtype", outputType) != GRAPH_SUCCESS ||
        target.SetAttr("group_size", groupSize) != GRAPH_SUCCESS) {
        return GRAPH_FAILED;
    }
    int64_t enableUncache = 0;
    if (match.matmul.GetAttr("enable_uncache", enableUncache) == GRAPH_SUCCESS &&
        target.SetAttr("enable_uncache", enableUncache) != GRAPH_SUCCESS) {
        return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
}

Status CreateEncodedScale(const GraphPtr& graph, const FusionMatch& match, GNode& constant)
{
    constant = Builder(graph.get())
                   .OpType("Const")
                   .Name((GetNodeName(match.matmul) + "_qbmm_scale").c_str())
                   .IrDefOutputs({{"y", Builder::kEsIrOutputRequired, ""}})
                   .InstanceOutputDataType("y", DT_UINT64)
                   .InstanceOutputShape("y", match.scaleDesc.GetShape().GetDims())
                   .InstanceOutputFormat("y", FORMAT_ND)
                   .Build();
    Tensor value(match.scaleDesc, reinterpret_cast<const uint8_t*>(match.encodedScale.data()),
                 match.encodedScale.size() * sizeof(uint64_t));
    if (constant.SetAttr("value", value) != GRAPH_SUCCESS ||
        constant.UpdateOutputDesc(0, match.scaleDesc) != GRAPH_SUCCESS) {
        return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
}

} // namespace

// Only direct Const cleanup needs graph-run state; matching and all edge
// rewiring are owned by PatternFusionPass and SubgraphRewriter.
struct MatmulQuantToFixpipeFusionPass::RunState {
    GraphPtr graph;
    std::vector<std::pair<std::string, std::string>> constInputs;

    explicit RunState(const GraphPtr& source) : graph(source) {}

    void TrackConstInputs(const FusionMatch& match)
    {
        const auto owner = GetNodeName(match.matmul);
        std::vector<GNode> nodes = {match.matmul, match.dequantOrRequant};
        if (match.isThreeNodePattern) {
            nodes.push_back(match.outputNode);
        }
        for (const auto& node : nodes) {
            for (const auto port : GetConnectedInputPorts(graph, node)) {
                const auto source = GetInputSource(node, port).first;
                if (source && (IsNodeType(*source, "Const") || IsNodeType(*source, "Constant"))) {
                    constInputs.emplace_back(owner, GetNodeName(*source));
                }
            }
        }
    }

    Status RemoveUnusedConstInputs()
    {
        for (const auto& input : constInputs) {
            // A failed replacement leaves its MatMul in the graph.
            if (graph->FindNodeByName(input.first.c_str())) {
                continue;
            }
            auto node = graph->FindNodeByName(input.second.c_str());
            if (!node) {
                continue;
            }
            bool used = !node->GetOutControlNodes().empty();
            for (size_t output = 0; output < node->GetOutputsSize(); ++output) {
                used = used || !node->GetOutDataNodesAndPortIndexs(output).empty();
            }
            if (!used && graph->RemoveNode(*node) != GRAPH_SUCCESS) {
                return GRAPH_FAILED;
            }
        }
        return GRAPH_SUCCESS;
    }
};

bool MatmulQuantToFixpipeFusionPass::HasRequiredPublicApis() { return PublicOriginApiAvailable(); }

ge::Status MatmulQuantToFixpipeFusionPass::Run(ge::GraphPtr& graph, ge::CustomPassContext& context)
{
    if (!IsTargetVersion()) {
        return ge::GRAPH_NOT_CHANGED;
    }
    return Fuse(graph, context);
}

std::vector<ge::fusion::PatternUniqPtr> MatmulQuantToFixpipeFusionPass::Patterns()
{
    std::vector<ge::fusion::PatternUniqPtr> patterns;
    // The three-node family must run before the two-node fallback family.
    for (bool three : {true, false}) {
        for (const auto* type : {"MatMul", "BatchMatMul", "BatchMatMulV2", "MatMulV2"}) {
            for (const auto* quantType : {"AscendDequant", "AscendRequant"}) {
                if (three && std::strcmp(quantType, "AscendRequant") == 0) {
                    continue;
                }
                for (bool bias : {false, true}) {
                    ge::es::EsGraphBuilder builder(three ? "matmul_dequant_quant" : "matmul_dequant_requant");
                    auto* graph = builder.GetCGraphBuilder()->GetGraph();
                    auto a = builder.CreateInput(0);
                    auto b = builder.CreateInput(1);
                    auto optionalBias = bias ? builder.CreateInput(2) : ge::es::EsTensorHolder(nullptr);
                    auto scale = builder.CreateInput(bias ? 3 : 2);
                    auto matmul = Builder(graph)
                                      .Name("matmul")
                                      .OpType(type)
                                      .IrDefInputs({{"x1", Builder::kEsIrInputRequired, ""},
                                                    {"x2", Builder::kEsIrInputRequired, ""},
                                                    {"bias", Builder::kEsIrInputOptional, ""}})
                                      .IrDefOutputs({{"y", Builder::kEsIrOutputRequired, ""}})
                                      .Build();
                    graph->AddDataEdge(*a.GetProducer(), 0, matmul, 0);
                    graph->AddDataEdge(*b.GetProducer(), 0, matmul, 1);
                    if (bias) {
                        graph->AddDataEdge(*optionalBias.GetProducer(), 0, matmul, 2);
                    }
                    auto dequant = Builder(graph)
                                       .Name("dequant")
                                       .OpType(quantType)
                                       .IrDefInputs({{"x", Builder::kEsIrInputRequired, ""},
                                                     {"scale", Builder::kEsIrInputRequired, ""}})
                                       .IrDefOutputs({{"y", Builder::kEsIrOutputRequired, ""}})
                                       .Build();
                    graph->AddDataEdge(matmul, 0, dequant, 0);
                    graph->AddDataEdge(*scale.GetProducer(), 0, dequant, 1);
                    auto tail = dequant;
                    if (three) {
                        tail = Builder(graph)
                                   .Name("quant")
                                   .OpType("AscendQuant")
                                   .IrDefInputs({{"x", Builder::kEsIrInputRequired, ""}})
                                   .IrDefOutputs({{"y", Builder::kEsIrOutputRequired, ""}})
                                   .Build();
                        graph->AddDataEdge(dequant, 0, tail, 0);
                    }
                    auto output = ge::es::EsTensorHolder(builder.GetCGraphBuilder()->GetTensorHolderFromNode(tail, 0));
                    // Only the final tensor is a pattern output. A shared Dequant
                    // cannot be consumed by this three-node pattern and falls back to two nodes.
                    auto patternGraph = builder.BuildAndReset({output});
                    auto pattern = std::make_unique<ge::fusion::Pattern>(std::move(*patternGraph));
                    pattern->CaptureTensor({matmul, 0}).CaptureTensor({dequant, 0});
                    if (three) {
                        pattern->CaptureTensor({tail, 0});
                    }
                    patterns.push_back(std::move(pattern));
                }
            }
        }
    }
    return patterns;
}

namespace {
std::unique_ptr<FusionMatch> ReadPatternMatch(const std::unique_ptr<ge::fusion::MatchResult>& result)
{
    ge::fusion::NodeIo matmul, dequant, tail;
    if (!result || result->GetCapturedTensor(0, matmul) != GRAPH_SUCCESS ||
        result->GetCapturedTensor(1, dequant) != GRAPH_SUCCESS) {
        return nullptr;
    }
    AscendString name;
    result->GetPatternGraph().GetName(name);
    const bool three = name == "matmul_dequant_quant";
    tail = dequant;
    if (three && result->GetCapturedTensor(2, tail) != GRAPH_SUCCESS) {
        return nullptr;
    }
    return std::make_unique<FusionMatch>(matmul.node, dequant.node, tail.node, three);
}
} // namespace

bool MatmulQuantToFixpipeFusionPass::MeetRequirements(const std::unique_ptr<ge::fusion::MatchResult>& result)
{
    auto match = ReadPatternMatch(result);
    if (!match || !state_ || !state_->graph) {
        return false;
    }
    const auto users = match->matmul.GetOutDataNodesAndPortIndexs(0);
    FusionInputs inputs;
    return users.size() == 1 && ReadFusionInputs(*match, inputs) && PrepareFusionMatch(state_->graph, *match);
}

ge::fusion::GraphUniqPtr MatmulQuantToFixpipeFusionPass::Replacement(
    const std::unique_ptr<ge::fusion::MatchResult>& result)
{
    auto match = ReadPatternMatch(result);
    if (!match || !state_ || !PrepareFusionMatch(state_->graph, *match)) {
        return nullptr;
    }
    auto boundary = result->ToSubgraphBoundary();
    std::vector<ge::fusion::SubgraphInput> inputs;
    if (!boundary || boundary->GetAllInputs(inputs) != GRAPH_SUCCESS) {
        return nullptr;
    }
    ge::es::EsGraphBuilder builder("fixpipe_replacement");
    // The ES builder owns this graph until BuildAndReset. Helpers borrow it.
    auto graph = std::shared_ptr<Graph>(builder.GetCGraphBuilder()->GetGraph(), [](Graph*) {});
    auto target = CreateFusedMatmul(graph, *match);
    if (SetFusedMatmulAttributes(*match, target) != GRAPH_SUCCESS) {
        return nullptr;
    }
    FusionInputs sourceInputs;
    if (!ReadFusionInputs(*match, sourceInputs) || inputs.size() != (sourceInputs.bias.first ? 4U : 3U)) {
        return nullptr;
    }
    const bool hasBias = sourceInputs.bias.first != nullptr;
    int32_t biasPort = -1;
    if (hasBias && (target.GetInputIndexByName("bias", biasPort) != GRAPH_SUCCESS || biasPort < 0)) {
        return nullptr;
    }
    // Boundary indices come from Patterns: A, B, optional bias, scale. Do not
    // infer roles from producers: A and B may share the same source tensor.
    for (size_t index = 0; index < inputs.size(); ++index) {
        auto input = builder.CreateInput(index);
        const bool scaleInput = index == inputs.size() - 1;
        if (scaleInput && match->isThreeNodePattern) {
            continue;
        }
        // Preserve the bias role across the two target IRs, whose bias indices differ.
        const int64_t targetPort = scaleInput ? 2 : (hasBias && index == 2 ? biasPort : index);
        const TensorDesc& desc = scaleInput ? match->scaleDesc :
                                 index == 0 ? match->x1Desc :
                                 index == 1 ? match->x2Desc :
                                              match->biasDesc;
        if (ConnectInput(graph, *input.GetProducer(), 0, target, targetPort, desc) != GRAPH_SUCCESS) {
            return nullptr;
        }
    }
    if (match->isThreeNodePattern) {
        GNode scale;
        if (CreateEncodedScale(graph, *match, scale) != GRAPH_SUCCESS ||
            ConnectInput(graph, scale, 0, target, 2, match->scaleDesc) != GRAPH_SUCCESS) {
            return nullptr;
        }
        bool multi = true;
        for (auto node : {target, scale}) {
            node.SetAttr("_datadump_is_multiop", multi);
        }
    }
    if (!match->outputNode.GetOutDataNodesAndPortIndexs(0).empty() &&
        CopyDumpOutputMapping(match->outputNode, target) != GRAPH_SUCCESS) {
        return nullptr;
    }
    auto output = ge::es::EsTensorHolder(builder.GetCGraphBuilder()->GetTensorHolderFromNode(target, 0));
    auto replacement = builder.BuildAndReset({output});
    if (replacement) {
        state_->TrackConstInputs(*match);
    }
    return replacement;
}

ge::Status MatmulQuantToFixpipeFusionPass::Fuse(ge::GraphPtr& graph, ge::CustomPassContext& context)
{
    if (!graph || !graph->IsValid() || !PublicOriginApiAvailable()) {
        return ge::GRAPH_NOT_CHANGED;
    }
    state_ = std::make_shared<RunState>(graph);
    context.SetPassName(PASS_NAME);
    const auto status = ge::fusion::PatternFusionPass::Run(graph, context);
    const auto cleanup = state_->RemoveUnusedConstInputs();
    state_.reset();
    return cleanup == GRAPH_SUCCESS ? status : cleanup;
}

// This header macro is the GE component version, not the CANN package version.
// IsTargetVersion separately checks the runtime GE version and public APIs.
#if defined(GE_COMPILER_VERSION_NUM) && GE_COMPILER_VERSION_NUM >= 90200000
REG_FUSION_PASS(MatmulQuantToFixpipeFusionPass)
    .Stage(IsTargetVersion() ? ge::CustomPassStage::kCompatibleInherited : ge::CustomPassStage::kAfterInferShape);
#endif
} // namespace ops

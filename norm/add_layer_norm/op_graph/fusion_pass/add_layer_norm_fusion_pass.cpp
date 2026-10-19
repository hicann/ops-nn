/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "add_layer_norm_fusion_pass.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "acl/acl_rt.h"
#include "common/inc/error_util.h"
#include "ge/compliant_node_builder.h"
#include "ge/es_graph_builder.h"
#include "ge/ge_utils.h"
#include "graph/operator_factory.h"
#include "ge/fusion/graph_rewriter.h"
#include "ge/fusion/pass/pattern_fusion_pass.h"
#include "ge/fusion/subgraph_boundary.h"
#include "platform/platform_info.h"
#include "version/ge-compiler_version.h"

namespace ge {
namespace fusion {

// false means the API is unavailable; a missing option means an empty value.
bool ReadPassOption(const CustomPassContext& context, const std::string& key, std::string& value)
{
#if defined(GE_COMPILER_VERSION_NUM) && (GE_COMPILER_VERSION_NUM >= 90000000)
    AscendString optionValue;
    if (context.GetOptionValue(AscendString(key.c_str()), optionValue) != GRAPH_SUCCESS) {
        value.clear();
        return true;
    }
    value = optionValue.GetString() == nullptr ? "" : optionValue.GetString();
    return true;
#else
    (void)context;
    (void)key;
    (void)value;
    return false;
#endif
}

bool IsTrainGraphRunMode(const std::string& value)
{
    constexpr int32_t kDecimalBase = 10;
    return static_cast<int32_t>(std::strtol(value.c_str(), nullptr, kDecimalBase)) == 1;
}

bool IsAllowMixPrecision(const std::string& value)
{
    return value == "allow_mix_precision" || value == "allow_mix_precision_fp16";
}

int32_t GetRuntimeGeCompilerVersion()
{
#if defined(GE_COMPILER_VERSION_NUM) && (GE_COMPILER_VERSION_NUM >= 90000000)
    int32_t version = 0;
    char packageName[] = "ge_compiler";
    if (aclsysGetVersionNum(packageName, &version) != ACL_SUCCESS || version <= 0) {
        return 0;
    }
    return version;
#else
    return 0;
#endif
}

bool CanUsePassContextOptionsForVersion(const int32_t runtimeVersion)
{
    constexpr int32_t kGeCompilerVersion900 = 90000000;
    return runtimeVersion >= kGeCompilerVersion900;
}

namespace {

const std::string kPassName = "AddLayerNormFusionPass";
constexpr int32_t kX = 0;
constexpr int32_t kNormShape = 1;
constexpr int32_t kGamma = 2;
constexpr int32_t kBeta = 3;
constexpr int32_t kY = 0;
constexpr int32_t kMean = 1;
constexpr int32_t kRstd = 2;
constexpr int32_t kFusedX = 3;

const std::set<std::string> kSupportedNpuArch = {"3510", "5102"};

struct InputSpec {
    NodeIo inside;
    TensorDesc desc;
    std::string name;
};

enum class OutputKind { Y, MEAN, RSTD, X, CAST_X, GAMMA, BETA };

struct OutputSpec {
    NodeIo original;
    TensorDesc desc;
    OutputKind kind;
};

struct Candidate {
    GNode layerNorm;
    GNode add;
    GNode inputCast;
    GNode outputCast;
    GNode biasAdd;
    GNode reshape;
    bool hasInputCast = false;
    bool hasOutputCast = false;
    bool hasBias = false;
    bool hasReshape = false;
    bool isThird = false;
    bool isV4 = false;
    bool fillGamma = false;
    bool fillBeta = false;
    bool enableX = false;
    bool enableCastX = false;
    DataType fusedType = DT_UNDEFINED;
    float epsilon = 1.0e-5F;
    InputSpec x1;
    InputSpec x2;
    InputSpec gamma;
    InputSpec beta;
    TensorDesc gammaDesc;
    TensorDesc betaDesc;
    InputSpec bias;
    InputSpec normShape;
    InputSpec reshapeShape;
    std::vector<InputSpec> inputs;
    std::vector<OutputSpec> outputs;
    std::vector<GNode> internalNodes;
};

std::string ToString(const AscendString& value) { return value.GetString() == nullptr ? "" : value.GetString(); }

std::string NodeType(const GNode& node)
{
    AscendString type;
    return node.GetType(type) == GRAPH_SUCCESS ? ToString(type) : "";
}

std::string NodeName(const GNode& node)
{
    AscendString name;
    return node.GetName(name) == GRAPH_SUCCESS ? ToString(name) : "";
}

bool IsLayerNormType(const std::string& type)
{
    return type == "LayerNorm" || type == "LayerNormV3" || type == "LayerNormV4";
}

bool IsSameNode(const GNode& lhs, const GNode& rhs)
{
    const std::string lhsName = NodeName(lhs);
    return !lhsName.empty() && lhsName == NodeName(rhs);
}

GNodePtr Producer(const GNode& node, const int32_t index) { return node.GetInDataNodesAndPortIndexs(index).first; }

bool GetInputDesc(const GNode& node, const int32_t index, TensorDesc& desc)
{
    return node.GetInputDesc(index, desc) == GRAPH_SUCCESS;
}

bool GetOutputDesc(const GNode& node, const int32_t index, TensorDesc& desc)
{
    return node.GetOutputDesc(index, desc) == GRAPH_SUCCESS;
}

bool HasDynamicDim(const Shape& shape)
{
    const auto dims = shape.GetDims();
    return std::any_of(dims.begin(), dims.end(), [](const int64_t dim) { return dim < 0; });
}

bool HasScalarShape(const Shape& shape) { return shape.GetDimNum() == 0U; }

bool SameShape(const TensorDesc& lhs, const TensorDesc& rhs)
{
    return lhs.GetShape().GetDims() == rhs.GetShape().GetDims();
}

bool LastDim(const TensorDesc& desc, int64_t& last)
{
    const auto dims = desc.GetShape().GetDims();
    if (dims.empty()) {
        return false;
    }
    last = dims.back();
    return true;
}

bool MakeInputSpec(const GNode& insideNode, const int32_t insidePort, const std::string& name, InputSpec& spec)
{
    TensorDesc desc;
    if (!GetInputDesc(insideNode, insidePort, desc) || Producer(insideNode, insidePort) == nullptr) {
        return false;
    }
    spec = {{insideNode, insidePort}, desc, name};
    return true;
}

size_t ExternalConsumerCount(const GNode& node, const int32_t port, const std::vector<GNode>& internal)
{
    size_t count = 0U;
    for (const auto& consumer : node.GetOutDataNodesAndPortIndexs(port)) {
        if (consumer.first == nullptr) {
            continue;
        }
        const bool isInternal = std::any_of(internal.begin(), internal.end(),
                                            [&](const GNode& item) { return IsSameNode(*consumer.first, item); });
        if (!isInternal) {
            ++count;
        }
    }
    return count;
}

bool IsSingleConsumer(const GNode& node) { return node.GetOutDataNodesAndPortIndexs(0).size() == 1U; }

bool GetNpuArch(std::string& npuArch)
{
    fe::PlatFormInfos platformInfos;
    fe::OptionalInfos optionalInfos;
    if (fe::PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platformInfos, optionalInfos) != SUCCESS) {
        OPS_LOG_D(kPassName.c_str(), "Failed to get platform info.");
        return false;
    }
    if (!platformInfos.GetPlatformRes("version", "NpuArch", npuArch)) {
        OPS_LOG_D(kPassName.c_str(), "NpuArch is unavailable in platform info.");
        return false;
    }
    return !npuArch.empty();
}

bool IsSupportedPlatform(const std::string& npuArch) { return kSupportedNpuArch.count(npuArch) != 0U; }

bool CheckAxis(const GNode& layerNorm, const bool isV4, const TensorDesc& xDesc)
{
    const auto xDims = xDesc.GetShape().GetDims();
    if (xDims.empty()) {
        return false;
    }
    int64_t beginAxis = 0;
    if (isV4) {
        TensorDesc normXDesc;
        TensorDesc normalizedDesc;
        if (!GetInputDesc(layerNorm, kX, normXDesc) || !GetInputDesc(layerNorm, kNormShape, normalizedDesc)) {
            return false;
        }
        const auto normalizedDims = normalizedDesc.GetOriginShape().GetDims();
        const int64_t normalizedRank = normalizedDims.empty() ? 1 : normalizedDims.front();
        if (normalizedRank <= 0) {
            return false;
        }
        beginAxis = static_cast<int64_t>(normXDesc.GetOriginShape().GetDimNum()) - normalizedRank;
    } else if (layerNorm.GetAttr(AscendString("begin_norm_axis"), beginAxis) != GRAPH_SUCCESS) {
        return false;
    }
    return beginAxis == -1 || beginAxis == static_cast<int64_t>(xDims.size() - 1U);
}

bool CheckOptionalScaleShape(const GNode& layerNorm, const int64_t last)
{
    // Read normalized_shape to determine the missing parameter shape.
    Tensor value;
    if (layerNorm.GetInputConstData(kNormShape, value) != GRAPH_SUCCESS) {
        return false;
    }
    const TensorDesc desc = value.GetTensorDesc();
    if (value.GetData() == nullptr) {
        return false;
    }
    int64_t dim = 0;
    if (desc.GetDataType() == DT_INT64 && value.GetSize() == sizeof(int64_t)) {
        std::memcpy(&dim, value.GetData(), sizeof(dim));
    } else if (desc.GetDataType() == DT_INT32 && value.GetSize() == sizeof(int32_t)) {
        int32_t dim32 = 0;
        std::memcpy(&dim32, value.GetData(), sizeof(dim32));
        dim = dim32;
    } else {
        return false;
    }
    return dim == last;
}

bool PrepareScaleInputs(Candidate& candidate, const bool allowMixPrecision)
{
    const int32_t gammaPort = candidate.isV4 ? kGamma : 1;
    const int32_t betaPort = candidate.isV4 ? kBeta : 2;
    const GNodePtr gammaProducer = Producer(candidate.layerNorm, gammaPort);
    const GNodePtr betaProducer = Producer(candidate.layerNorm, betaPort);

    candidate.fillGamma = candidate.isV4 && gammaProducer == nullptr;
    candidate.fillBeta = candidate.isV4 && betaProducer == nullptr;
    if ((gammaProducer == nullptr && !candidate.fillGamma) || (betaProducer == nullptr && !candidate.fillBeta)) {
        return false;
    }

    TensorDesc xDesc;
    if (!GetInputDesc(candidate.layerNorm, kX, xDesc)) {
        return false;
    }
    int64_t last = 0;
    if (!LastDim(xDesc, last)) {
        return false;
    }
    if ((candidate.fillGamma || candidate.fillBeta) && !CheckOptionalScaleShape(candidate.layerNorm, last)) {
        return false;
    }

    TensorDesc gammaDesc;
    TensorDesc betaDesc;
    if (!candidate.fillGamma && !GetInputDesc(candidate.layerNorm, gammaPort, gammaDesc)) {
        return false;
    }
    if (!candidate.fillBeta && !GetInputDesc(candidate.layerNorm, betaPort, betaDesc)) {
        return false;
    }
    if (candidate.fillGamma) {
        const DataType dtype = candidate.fillBeta ? DT_FLOAT : betaDesc.GetDataType();
        const Format format = candidate.fillBeta ? xDesc.GetFormat() : betaDesc.GetFormat();
        gammaDesc = TensorDesc(Shape({last}), format, dtype);
        gammaDesc.SetOriginShape(Shape({last}));
    }
    if (candidate.fillBeta) {
        const DataType dtype = candidate.fillGamma ? DT_FLOAT : gammaDesc.GetDataType();
        const Format format = candidate.fillGamma ? xDesc.GetFormat() : gammaDesc.GetFormat();
        betaDesc = TensorDesc(Shape({last}), format, dtype);
        betaDesc.SetOriginShape(Shape({last}));
    }
    if (!SameShape(gammaDesc, betaDesc) || gammaDesc.GetShape().GetDimNum() != 1U ||
        gammaDesc.GetShape().GetDim(0) != last || gammaDesc.GetDataType() != betaDesc.GetDataType()) {
        return false;
    }
    candidate.gammaDesc = gammaDesc;
    candidate.betaDesc = betaDesc;
    if (candidate.isV4 && !MakeInputSpec(candidate.layerNorm, kNormShape, "normalized_shape", candidate.normShape)) {
        return false;
    }

    // Require FP32 gamma/beta and preserve their upstream Cast nodes.
    if (candidate.isThird) {
        if (gammaDesc.GetDataType() != DT_FLOAT) {
            return false;
        }
        if (!candidate.fillGamma && !MakeInputSpec(candidate.layerNorm, gammaPort, "gamma", candidate.gamma)) {
            return false;
        }
        if (!candidate.fillBeta && !MakeInputSpec(candidate.layerNorm, betaPort, "beta", candidate.beta)) {
            return false;
        }
        return true;
    }

    bool removeScaleCasts = false;
    if (!candidate.fillGamma && !candidate.fillBeta && gammaProducer != nullptr && betaProducer != nullptr &&
        NodeType(*gammaProducer) == "Cast" && NodeType(*betaProducer) == "Cast" &&
        (IsSingleConsumer(*gammaProducer) || IsSingleConsumer(*betaProducer)) && allowMixPrecision &&
        candidate.enableX) {
        TensorDesc gammaCastInput;
        TensorDesc betaCastInput;
        removeScaleCasts = GetInputDesc(*gammaProducer, 0, gammaCastInput) &&
                           GetInputDesc(*betaProducer, 0, betaCastInput) && gammaCastInput.GetDataType() == DT_FLOAT &&
                           betaCastInput.GetDataType() == DT_FLOAT;
        if (removeScaleCasts) {
            candidate.gamma = {{*gammaProducer, 0}, gammaCastInput, "gamma"};
            candidate.beta = {{*betaProducer, 0}, betaCastInput, "beta"};
        }
    }

    if (!removeScaleCasts) {
        if (!candidate.fillGamma && !MakeInputSpec(candidate.layerNorm, gammaPort, "gamma", candidate.gamma)) {
            return false;
        }
        if (!candidate.fillBeta && !MakeInputSpec(candidate.layerNorm, betaPort, "beta", candidate.beta)) {
            return false;
        }
    }

    return true;
}

bool AddOutput(Candidate& candidate, const GNode& node, const int32_t port, const OutputKind kind)
{
    TensorDesc desc;
    if (!GetOutputDesc(node, port, desc)) {
        return false;
    }
    candidate.outputs.push_back({{node, port}, desc, kind});
    return true;
}

bool CheckThirdShapesAndTypes(Candidate& candidate)
{
    TensorDesc bias0;
    TensorDesc bias1;
    if (!GetInputDesc(candidate.biasAdd, 0, bias0) || !GetInputDesc(candidate.biasAdd, 1, bias1) ||
        SameShape(bias0, bias1) || HasScalarShape(bias0.GetShape()) || HasScalarShape(bias1.GetShape()) ||
        HasDynamicDim(bias0.GetShape()) || HasDynamicDim(bias1.GetShape())) {
        return false;
    }
    int64_t last0 = 0;
    int64_t last1 = 0;
    if (!LastDim(bias0, last0) || !LastDim(bias1, last1) || last0 != last1) {
        return false;
    }
    const int64_t size0 = bias0.GetShape().GetShapeSize();
    const int64_t size1 = bias1.GetShape().GetShapeSize();
    if (size0 <= 0 || size1 <= 0) {
        return false;
    }
    const int32_t biasPort = size0 > size1 ? 1 : 0;
    const int32_t normalPort = 1 - biasPort;
    const GNodePtr addInput0 = Producer(candidate.add, 0);
    if (addInput0 == nullptr) {
        return false;
    }
    const int32_t x1Port = IsSameNode(*addInput0, candidate.biasAdd) ? 1 : 0;
    if (!MakeInputSpec(candidate.add, x1Port, "x1", candidate.x1) ||
        !MakeInputSpec(candidate.biasAdd, normalPort, "x2", candidate.x2) ||
        !MakeInputSpec(candidate.biasAdd, biasPort, "bias", candidate.bias)) {
        return false;
    }
    if (!SameShape(candidate.x1.desc, candidate.x2.desc)) {
        return false;
    }
    int64_t xLast = 0;
    int64_t biasLast = 0;
    if (!LastDim(candidate.x1.desc, xLast) || !LastDim(candidate.bias.desc, biasLast) || xLast != biasLast ||
        (candidate.isV4 && candidate.x1.desc.GetShape().GetDimNum() < candidate.bias.desc.GetShape().GetDimNum())) {
        return false;
    }
    const DataType dtype = candidate.x1.desc.GetDataType();
    if ((dtype != DT_FLOAT16 && dtype != DT_BF16) || candidate.x2.desc.GetDataType() != dtype ||
        candidate.bias.desc.GetDataType() != dtype) {
        return false;
    }
    candidate.fusedType = dtype;
    candidate.hasBias = true;
    return CheckAxis(candidate.layerNorm, candidate.isV4, candidate.x1.desc);
}

bool TryMakeThirdCandidate(const GNode& layerNorm, const bool isTrain, const bool allowMixPrecision,
                           Candidate& candidate)
{
    const GNodePtr castX = Producer(layerNorm, kX);
    if (castX == nullptr || NodeType(*castX) != "Cast") {
        return false;
    }
    const GNodePtr add = Producer(*castX, 0);
    if (add == nullptr || NodeType(*add) != "Add") {
        return false;
    }
    GNodePtr biasAdd;
    for (int32_t i = 0; i < 2; ++i) {
        const GNodePtr input = Producer(*add, i);
        if (input != nullptr && NodeType(*input) == "Add") {
            biasAdd = input;
            break;
        }
    }
    if (biasAdd == nullptr || !IsSingleConsumer(*biasAdd) || !castX->GetInControlNodes().empty() ||
        !castX->GetOutControlNodes().empty()) {
        return false;
    }
    const auto yConsumers = layerNorm.GetOutDataNodesAndPortIndexs(kY);
    if (yConsumers.size() != 1U || yConsumers[0].first == nullptr || NodeType(*yConsumers[0].first) != "Cast") {
        return false;
    }

    candidate.layerNorm = layerNorm;
    candidate.add = *add;
    candidate.inputCast = *castX;
    candidate.outputCast = *yConsumers[0].first;
    candidate.biasAdd = *biasAdd;
    candidate.hasInputCast = true;
    candidate.hasOutputCast = true;
    candidate.isThird = true;
    candidate.internalNodes = {*biasAdd, *add, *castX, layerNorm, *yConsumers[0].first};
    candidate.isV4 = NodeType(layerNorm) == "LayerNormV4";
    if (isTrain && NodeType(layerNorm) == "LayerNorm") {
        return false;
    }
    if (!CheckThirdShapesAndTypes(candidate)) {
        return false;
    }
    candidate.enableX = true;
    candidate.enableCastX = ExternalConsumerCount(candidate.inputCast, 0, {candidate.layerNorm}) > 0U;
    if (!PrepareScaleInputs(candidate, allowMixPrecision)) {
        return false;
    }
    candidate.inputs = {candidate.x1, candidate.x2};
    if (!candidate.fillGamma)
        candidate.inputs.push_back(candidate.gamma);
    if (!candidate.fillBeta)
        candidate.inputs.push_back(candidate.beta);
    candidate.inputs.push_back(candidate.bias);
    if (candidate.isV4)
        candidate.inputs.push_back(candidate.normShape);

    if (!AddOutput(candidate, candidate.outputCast, 0, OutputKind::Y) ||
        !AddOutput(candidate, candidate.layerNorm, kMean, OutputKind::MEAN) ||
        !AddOutput(candidate, candidate.layerNorm, kRstd, OutputKind::RSTD)) {
        return false;
    }
    // Remove the trailing Cast and output y with the fused computation dtype.
    candidate.outputs.front().desc.SetDataType(candidate.fusedType);
    if (ExternalConsumerCount(candidate.add, 0, {candidate.inputCast}) > 0U &&
        !AddOutput(candidate, candidate.add, 0, OutputKind::X)) {
        return false;
    }
    if (candidate.enableCastX && !AddOutput(candidate, candidate.inputCast, 0, OutputKind::CAST_X)) {
        return false;
    }
    (void)candidate.layerNorm.GetAttr(AscendString("epsilon"), candidate.epsilon);
    return true;
}

bool SelectBiasInput(const GNode& outerAdd, const bool isV4, Candidate& candidate)
{
    GNodePtr candidates[2] = {Producer(outerAdd, 0), Producer(outerAdd, 1)};
    bool valid[2] = {false, false};
    for (int32_t i = 0; i < 2; ++i) {
        if (candidates[i] == nullptr || !IsSingleConsumer(*candidates[i])) {
            continue;
        }
        const std::string type = NodeType(*candidates[i]);
        valid[i] = type == "Add" || type == "Reshape";
    }
    if (valid[0] == valid[1]) {
        return false;
    }
    const int32_t nestedOuterPort = valid[0] ? 0 : 1;
    const int32_t normalOuterPort = 1 - nestedOuterPort;
    GNodePtr nestedNode = candidates[nestedOuterPort];
    GNodePtr reshape;
    if (NodeType(*nestedNode) == "Reshape") {
        reshape = nestedNode;
        nestedNode = Producer(*reshape, 0);
        if (nestedNode == nullptr || NodeType(*nestedNode) != "Add" || !IsSingleConsumer(*nestedNode)) {
            return false;
        }
    }
    const GNode& nested = *nestedNode;
    TensorDesc desc0;
    TensorDesc desc1;
    if (!GetInputDesc(nested, 0, desc0) || !GetInputDesc(nested, 1, desc1)) {
        return false;
    }
    int64_t lastOuter = 0;
    int64_t last0 = 0;
    int64_t last1 = 0;
    TensorDesc outerDesc;
    if (!GetInputDesc(outerAdd, normalOuterPort, outerDesc) || !LastDim(outerDesc, lastOuter) ||
        !LastDim(desc0, last0) || !LastDim(desc1, last1) || (lastOuter != last0 && lastOuter != last1)) {
        return false;
    }
    if (isV4 && (outerDesc.GetShape().GetDimNum() < desc0.GetShape().GetDimNum() ||
                 outerDesc.GetShape().GetDimNum() < desc1.GetShape().GetDimNum())) {
        return false;
    }
    const bool dyn0 = HasDynamicDim(desc0.GetShape());
    const bool dyn1 = HasDynamicDim(desc1.GetShape());
    if ((dyn0 && dyn1) || (isV4 && (dyn0 || dyn1))) {
        return false;
    }
    int32_t biasPort = 0;
    if (!dyn0 && !dyn1) {
        biasPort = desc1.GetShape().GetShapeSize() == last1 ? 1 : 0;
    } else {
        biasPort = dyn0 ? 1 : 0;
    }
    const int32_t normalNestedPort = 1 - biasPort;
    if (!MakeInputSpec(outerAdd, normalOuterPort, "x1", candidate.x1) ||
        !MakeInputSpec(nested, normalNestedPort, "x2", candidate.x2) ||
        !MakeInputSpec(nested, biasPort, "bias", candidate.bias)) {
        return false;
    }
    candidate.biasAdd = nested;
    candidate.hasBias = true;
    if (reshape != nullptr) {
        candidate.reshape = *reshape;
        candidate.hasReshape = true;
        if (!MakeInputSpec(candidate.reshape, 1, "reshape_shape", candidate.reshapeShape)) {
            return false;
        }
    }
    return true;
}

bool TryMakeGeneralCandidate(const GNode& layerNorm, const bool isTrain, const bool allowMixPrecision,
                             Candidate& candidate)
{
    const bool isV4 = NodeType(layerNorm) == "LayerNormV4";
    GNodePtr before = Producer(layerNorm, kX);
    if (before == nullptr) {
        return false;
    }
    GNodePtr add = before;
    GNodePtr midCast;
    if (NodeType(*before) == "Cast") {
        midCast = before;
        add = Producer(*before, 0);
        if (add == nullptr || !IsSingleConsumer(*before)) {
            return false;
        }
    }
    if (add == nullptr || NodeType(*add) != "Add") {
        return false;
    }
    TensorDesc add0;
    TensorDesc add1;
    if (!GetInputDesc(*add, 0, add0) || !GetInputDesc(*add, 1, add1) || !SameShape(add0, add1) ||
        add0.GetDataType() != add1.GetDataType() ||
        (add0.GetDataType() != DT_FLOAT16 && add0.GetDataType() != DT_BF16 && add0.GetDataType() != DT_FLOAT)) {
        return false;
    }
    if (isV4 && (HasDynamicDim(add0.GetShape()) || HasDynamicDim(add1.GetShape()))) {
        return false;
    }

    candidate.layerNorm = layerNorm;
    candidate.add = *add;
    candidate.internalNodes = {*add, layerNorm};
    candidate.isV4 = isV4;
    candidate.hasInputCast = midCast != nullptr;
    if (midCast != nullptr) {
        candidate.inputCast = *midCast;
        candidate.internalNodes.push_back(*midCast);
    }
    candidate.enableX = ExternalConsumerCount(*add, 0, {midCast != nullptr ? *midCast : layerNorm}) > 0U;

    GNodePtr inputProducer0 = Producer(*add, 0);
    GNodePtr inputProducer1 = Producer(*add, 1);
    if (inputProducer0 == nullptr || inputProducer1 == nullptr) {
        return false;
    }
    bool removeInputCast0 = NodeType(*inputProducer0) == "Cast";
    bool removeInputCast1 = NodeType(*inputProducer1) == "Cast";
    if (removeInputCast0 && removeInputCast1) {
        removeInputCast0 = false;
        removeInputCast1 = false;
    }
    if ((removeInputCast0 && !IsSingleConsumer(*inputProducer0)) ||
        (removeInputCast1 && !IsSingleConsumer(*inputProducer1)) || !candidate.enableX) {
        removeInputCast0 = false;
        removeInputCast1 = false;
    }
    if (((add0.GetDataType() != DT_FLOAT) && removeInputCast1) ||
        ((add1.GetDataType() != DT_FLOAT) && removeInputCast0)) {
        removeInputCast0 = false;
        removeInputCast1 = false;
    }
    TensorDesc raw0;
    TensorDesc raw1;
    const bool input0IsCast = NodeType(*inputProducer0) == "Cast";
    const bool input1IsCast = NodeType(*inputProducer1) == "Cast";
    const bool hasRaw0 = GetInputDesc(*inputProducer0, 0, raw0);
    const bool hasRaw1 = GetInputDesc(*inputProducer1, 0, raw1);
    if (input0IsCast && !hasRaw0)
        return false;
    if (input1IsCast && !hasRaw1)
        return false;
    if (isTrain && (add0.GetDataType() == DT_FLOAT16 || add1.GetDataType() == DT_FLOAT16 ||
                    (hasRaw0 && raw0.GetDataType() == DT_FLOAT16) || (hasRaw1 && raw1.GetDataType() == DT_FLOAT16))) {
        return false;
    }
    if (!removeInputCast0 && !removeInputCast1 && add0.GetDataType() == DT_FLOAT) {
        return false;
    }
    candidate.fusedType = (removeInputCast0 || removeInputCast1) ? DT_FLOAT : add0.GetDataType();

    if (!CheckAxis(layerNorm, isV4, add0)) {
        return false;
    }
    if (!PrepareScaleInputs(candidate, allowMixPrecision)) {
        return false;
    }

    if (removeInputCast0) {
        candidate.x1 = {{*inputProducer0, 0}, raw0, "x1"};
        candidate.internalNodes.push_back(*inputProducer0);
    } else if (!MakeInputSpec(*add, 0, "x1", candidate.x1)) {
        return false;
    }
    if (removeInputCast1) {
        candidate.x2 = {{*inputProducer1, 0}, raw1, "x2"};
        candidate.internalNodes.push_back(*inputProducer1);
    } else if (!MakeInputSpec(*add, 1, "x2", candidate.x2)) {
        return false;
    }

    if (!removeInputCast0 && !removeInputCast1) {
        Candidate nested = candidate;
        if (SelectBiasInput(*add, isV4, nested)) {
            candidate.x1 = nested.x1;
            candidate.x2 = nested.x2;
            candidate.bias = nested.bias;
            candidate.biasAdd = nested.biasAdd;
            candidate.reshape = nested.reshape;
            candidate.reshapeShape = nested.reshapeShape;
            candidate.hasBias = true;
            candidate.hasReshape = nested.hasReshape;
            candidate.internalNodes.push_back(nested.biasAdd);
            if (nested.hasReshape)
                candidate.internalNodes.push_back(nested.reshape);
        }
    }

    if (NodeType(candidate.gamma.inside.node) == "Cast") {
        candidate.internalNodes.push_back(candidate.gamma.inside.node);
        if (ExternalConsumerCount(candidate.gamma.inside.node, 0, {layerNorm}) > 0U &&
            !AddOutput(candidate, candidate.gamma.inside.node, 0, OutputKind::GAMMA)) {
            return false;
        }
    }
    if (NodeType(candidate.beta.inside.node) == "Cast") {
        candidate.internalNodes.push_back(candidate.beta.inside.node);
        if (ExternalConsumerCount(candidate.beta.inside.node, 0, {layerNorm}) > 0U &&
            !AddOutput(candidate, candidate.beta.inside.node, 0, OutputKind::BETA)) {
            return false;
        }
    }

    candidate.inputs = {candidate.x1, candidate.x2};
    if (!candidate.fillGamma)
        candidate.inputs.push_back(candidate.gamma);
    if (!candidate.fillBeta)
        candidate.inputs.push_back(candidate.beta);
    if (candidate.hasBias)
        candidate.inputs.push_back(candidate.bias);
    if (candidate.isV4)
        candidate.inputs.push_back(candidate.normShape);
    if (candidate.hasReshape)
        candidate.inputs.push_back(candidate.reshapeShape);

    if (!AddOutput(candidate, layerNorm, kY, OutputKind::Y) ||
        !AddOutput(candidate, layerNorm, kMean, OutputKind::MEAN) ||
        !AddOutput(candidate, layerNorm, kRstd, OutputKind::RSTD)) {
        return false;
    }
    if (candidate.enableX && !AddOutput(candidate, *add, 0, OutputKind::X)) {
        return false;
    }
    (void)layerNorm.GetAttr(AscendString("epsilon"), candidate.epsilon);
    return true;
}

bool AddBoundaryInput(SubgraphBoundary& boundary, const int64_t index, const InputSpec& spec)
{
    SubgraphInput input;
    return input.AddInput(spec.inside) == SUCCESS && boundary.AddInput(index, std::move(input)) == SUCCESS;
}

std::unique_ptr<SubgraphBoundary> BuildBoundary(const Candidate& candidate)
{
    auto boundary = std::make_unique<SubgraphBoundary>();
    for (size_t i = 0U; i < candidate.inputs.size(); ++i) {
        if (!AddBoundaryInput(*boundary, static_cast<int64_t>(i), candidate.inputs[i])) {
            return nullptr;
        }
    }
    for (size_t i = 0U; i < candidate.outputs.size(); ++i) {
        SubgraphOutput output;
        if (output.SetOutput(candidate.outputs[i].original) != SUCCESS ||
            boundary->AddOutput(static_cast<int64_t>(i), std::move(output)) != SUCCESS) {
            return nullptr;
        }
    }
    return boundary;
}

bool SetOutputDesc(GNode& node, const int32_t index, const TensorDesc& desc)
{
    return node.UpdateOutputDesc(index, desc) == GRAPH_SUCCESS;
}

es::EsTensorHolder Holder(es::EsGraphBuilder& builder, const GNode& node, const int32_t port)
{
    return es::EsTensorHolder(builder.GetCGraphBuilder()->GetTensorHolderFromNode(node, port));
}

GNode BuildCast(es::EsGraphBuilder& builder, const es::EsTensorHolder& input, const TensorDesc& outputDesc,
                const std::string& name)
{
    Graph* graph = builder.GetCGraphBuilder()->GetGraph();
    GNode node = es::CompliantNodeBuilder(graph)
                     .OpType("Cast")
                     .Name(name.c_str())
                     .IrDefInputs({{"x", es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
                     .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                     .IrDefAttrs({{"dst_type", es::CompliantNodeBuilder::kEsAttrRequired, "Int",
                                   es::CreateFrom(static_cast<int64_t>(outputDesc.GetDataType()))}})
                     .Build();
    if (es::AddEdgeAndUpdatePeerDesc(*graph, *input.GetProducer(), input.GetProducerOutIndex(), node, 0) !=
        GRAPH_SUCCESS) {
        return GNode();
    }
    TensorDesc inputDesc;
    if (!GetOutputDesc(*input.GetProducer(), input.GetProducerOutIndex(), inputDesc) ||
        node.UpdateInputDesc(0, inputDesc) != GRAPH_SUCCESS || !SetOutputDesc(node, 0, outputDesc)) {
        return GNode();
    }
    return node;
}

GNode BuildFill(es::EsGraphBuilder& builder, const es::EsTensorHolder& dims, const float value,
                const TensorDesc& outputDesc, const std::string& name)
{
    Graph* graph = builder.GetCGraphBuilder()->GetGraph();
    es::EsTensorHolder scalar;
    if (outputDesc.GetDataType() == DT_FLOAT16) {
        const uint16_t raw = value == 0.0F ? 0U : 0x3c00U;
        scalar = builder.CreateConst<uint16_t>({raw}, {}, DT_FLOAT16);
    } else if (outputDesc.GetDataType() == DT_BF16) {
        const uint16_t raw = value == 0.0F ? 0U : 0x3f80U;
        scalar = builder.CreateConst<uint16_t>({raw}, {}, DT_BF16);
    } else {
        scalar = builder.CreateScalar(value);
    }
    GNode node = es::CompliantNodeBuilder(graph)
                     .OpType("Fill")
                     .Name(name.c_str())
                     .IrDefInputs({{"dims", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                                   {"value", es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
                     .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                     .Build();
    if (es::AddEdgeAndUpdatePeerDesc(*graph, *dims.GetProducer(), dims.GetProducerOutIndex(), node, 0) !=
            GRAPH_SUCCESS ||
        es::AddEdgeAndUpdatePeerDesc(*graph, *scalar.GetProducer(), scalar.GetProducerOutIndex(), node, 1) !=
            GRAPH_SUCCESS) {
        return GNode();
    }
    TensorDesc dimsDesc;
    TensorDesc valueDesc;
    if (!GetOutputDesc(*dims.GetProducer(), dims.GetProducerOutIndex(), dimsDesc) ||
        !GetOutputDesc(*scalar.GetProducer(), scalar.GetProducerOutIndex(), valueDesc) ||
        node.UpdateInputDesc(0, dimsDesc) != GRAPH_SUCCESS || node.UpdateInputDesc(1, valueDesc) != GRAPH_SUCCESS ||
        !SetOutputDesc(node, 0, outputDesc)) {
        return GNode();
    }
    return node;
}

GNode BuildReshape(es::EsGraphBuilder& builder, const es::EsTensorHolder& x, const es::EsTensorHolder& shape,
                   const TensorDesc& outputDesc, const std::string& name)
{
    Graph* graph = builder.GetCGraphBuilder()->GetGraph();
    GNode node = es::CompliantNodeBuilder(graph)
                     .OpType("Reshape")
                     .Name(name.c_str())
                     .IrDefInputs({{"x", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                                   {"shape", es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
                     .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                     .Build();
    if (es::AddEdgeAndUpdatePeerDesc(*graph, *x.GetProducer(), x.GetProducerOutIndex(), node, 0) != GRAPH_SUCCESS ||
        es::AddEdgeAndUpdatePeerDesc(*graph, *shape.GetProducer(), shape.GetProducerOutIndex(), node, 1) !=
            GRAPH_SUCCESS) {
        return GNode();
    }
    TensorDesc xDesc;
    TensorDesc shapeDesc;
    if (!GetOutputDesc(*x.GetProducer(), x.GetProducerOutIndex(), xDesc) ||
        !GetOutputDesc(*shape.GetProducer(), shape.GetProducerOutIndex(), shapeDesc) ||
        node.UpdateInputDesc(0, xDesc) != GRAPH_SUCCESS || node.UpdateInputDesc(1, shapeDesc) != GRAPH_SUCCESS ||
        !SetOutputDesc(node, 0, outputDesc)) {
        return GNode();
    }
    return node;
}

struct FusedOutputs {
    es::EsTensorHolder y;
    es::EsTensorHolder mean;
    es::EsTensorHolder rstd;
    es::EsTensorHolder x;
};

bool BuildAddLayerNorm(es::EsGraphBuilder& builder, const std::array<es::EsTensorHolder, 5>& inputs, const bool hasBias,
                       const Candidate& candidate, const std::array<TensorDesc, 4>& outputDescs, FusedOutputs& outputs)
{
    Graph* graph = builder.GetCGraphBuilder()->GetGraph();
    GNode node = es::CompliantNodeBuilder(graph)
                     .OpType("AddLayerNorm")
                     .Name((NodeName(candidate.layerNorm) + "/AddLayerNorm").c_str())
                     .IrDefInputs({{"x1", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                                   {"x2", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                                   {"gamma", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                                   {"beta", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                                   {"bias", es::CompliantNodeBuilder::kEsIrInputOptional, ""}})
                     .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""},
                                    {"mean", es::CompliantNodeBuilder::kEsIrOutputRequired, ""},
                                    {"rstd", es::CompliantNodeBuilder::kEsIrOutputRequired, ""},
                                    {"x", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                     .IrDefAttrs({{"epsilon", es::CompliantNodeBuilder::kEsAttrOptional, "Float",
                                   es::CreateFromIfNotEqual(candidate.epsilon, 1.0e-5F)},
                                  {"additional_output", es::CompliantNodeBuilder::kEsAttrOptional, "Bool",
                                   es::CreateFromIfNotEqual(candidate.enableX || candidate.isThird, false)}})
                     .Build();
    const int32_t inputCount = hasBias ? 5 : 4;
    for (int32_t i = 0; i < inputCount; ++i) {
        if (inputs[static_cast<size_t>(i)].GetProducer() == nullptr ||
            es::AddEdgeAndUpdatePeerDesc(*graph, *inputs[static_cast<size_t>(i)].GetProducer(),
                                         inputs[static_cast<size_t>(i)].GetProducerOutIndex(), node,
                                         i) != GRAPH_SUCCESS) {
            return false;
        }
        TensorDesc inputDesc;
        if (!GetOutputDesc(*inputs[static_cast<size_t>(i)].GetProducer(),
                           inputs[static_cast<size_t>(i)].GetProducerOutIndex(), inputDesc)) {
            return false;
        }
        if (node.UpdateInputDesc(i, inputDesc) != GRAPH_SUCCESS) {
            return false;
        }
    }
    for (int32_t i = 0; i < 4; ++i) {
        if (!SetOutputDesc(node, i, outputDescs[static_cast<size_t>(i)])) {
            return false;
        }
    }
    outputs = {Holder(builder, node, 0), Holder(builder, node, 1), Holder(builder, node, 2), Holder(builder, node, 3)};
    return outputs.y.GetProducer() != nullptr;
}

bool InferReplacement(Graph& graph, const Candidate& candidate, const std::vector<Shape>& inputShapes)
{
    std::vector<std::pair<GNode, GNode>> shapeConstants;
    // Supply shape values during inference, then restore the original Data edges.
    for (auto data : graph.GetDirectNode()) {
        int64_t index = -1;
        if (NodeType(data) != "Data" || data.GetAttr("index", index) != GRAPH_SUCCESS || index < 0 ||
            static_cast<size_t>(index) >= candidate.inputs.size()) {
            continue;
        }
        const auto& input = candidate.inputs[static_cast<size_t>(index)];
        if (input.name != "normalized_shape" && input.name != "reshape_shape") {
            continue;
        }
        Tensor value;
        if (input.inside.node.GetInputConstData(static_cast<int32_t>(input.inside.index), value) != GRAPH_SUCCESS) {
            continue;
        }
        const auto peers = data.GetOutDataNodesAndPortIndexs(0);
        if (peers.empty()) {
            continue;
        }
        const std::string name = "infer_shape_constant_" + std::to_string(index);
        auto op = OperatorFactory::CreateOperator(name.c_str(), "Const");
        op.SetAttr("value", value);
        if (op.UpdateOutputDesc("y", value.GetTensorDesc()) != GRAPH_SUCCESS) {
            return false;
        }
        auto constant = graph.AddNodeByOp(op);
        for (const auto& peer : peers) {
            if (peer.first == nullptr || graph.RemoveEdge(data, 0, *peer.first, peer.second) != GRAPH_SUCCESS ||
                graph.AddDataEdge(constant, 0, *peer.first, peer.second) != GRAPH_SUCCESS) {
                return false;
            }
        }
        shapeConstants.emplace_back(data, constant);
    }
    if (GeUtils::InferShape(graph, inputShapes) != SUCCESS) {
        return false;
    }
    for (auto& nodes : shapeConstants) {
        for (const auto& peer : nodes.second.GetOutDataNodesAndPortIndexs(0)) {
            if (peer.first == nullptr || graph.RemoveEdge(nodes.second, 0, *peer.first, peer.second) != GRAPH_SUCCESS ||
                graph.AddDataEdge(nodes.first, 0, *peer.first, peer.second) != GRAPH_SUCCESS) {
                return false;
            }
        }
        if (graph.RemoveNode(nodes.second) != GRAPH_SUCCESS) {
            return false;
        }
    }
    return true;
}

std::unique_ptr<Graph> BuildReplacement(const Candidate& candidate)
{
    es::EsGraphBuilder builder("add_layer_norm_fusion_replacement");
    std::vector<es::EsTensorHolder> graphInputs;
    graphInputs.reserve(candidate.inputs.size());
    std::vector<Shape> inputShapes;
    inputShapes.reserve(candidate.inputs.size());
    for (size_t i = 0U; i < candidate.inputs.size(); ++i) {
        TensorDesc desc = candidate.inputs[i].desc;
        if (!desc.GetShape().GetDims().empty() && desc.GetRealDimCnt() <= 0) {
            desc.SetRealDimCnt(static_cast<int64_t>(desc.GetShape().GetDims().size()));
        }
        graphInputs.emplace_back(builder.CreateInput(static_cast<int64_t>(i), candidate.inputs[i].name.c_str(),
                                                     desc.GetDataType(), desc.GetFormat(), desc.GetShape().GetDims()));
        if (graphInputs.back().GetProducer() == nullptr) {
            return nullptr;
        }
        if (graphInputs.back().GetProducer()->UpdateOutputDesc(0, desc) != GRAPH_SUCCESS ||
            graphInputs.back().GetProducer()->UpdateInputDesc(0, desc) != GRAPH_SUCCESS) {
            return nullptr;
        }
        inputShapes.emplace_back(desc.GetShape());
    }
    size_t cursor = 0U;
    es::EsTensorHolder x1 = graphInputs[cursor++];
    es::EsTensorHolder x2 = graphInputs[cursor++];
    es::EsTensorHolder gamma;
    es::EsTensorHolder beta;
    if (!candidate.fillGamma)
        gamma = graphInputs[cursor++];
    if (!candidate.fillBeta)
        beta = graphInputs[cursor++];
    es::EsTensorHolder bias;
    if (candidate.hasBias)
        bias = graphInputs[cursor++];
    es::EsTensorHolder normShape;
    if (candidate.isV4)
        normShape = graphInputs[cursor++];
    es::EsTensorHolder reshapeShape;
    if (candidate.hasReshape)
        reshapeShape = graphInputs[cursor++];

    if (candidate.hasReshape) {
        TensorDesc reshapeOut;
        if (!GetOutputDesc(candidate.reshape, 0, reshapeOut))
            return nullptr;
        GNode reshape = BuildReshape(builder, x2, reshapeShape, reshapeOut, NodeName(candidate.reshape) + "/preserved");
        if (NodeType(reshape) != "Reshape")
            return nullptr;
        x2 = Holder(builder, reshape, 0);
    }

    if (candidate.fillGamma) {
        GNode fill = BuildFill(builder, normShape, 1.0F, candidate.gammaDesc,
                               NodeName(candidate.layerNorm) + "/Gamma_Fill");
        if (NodeType(fill) != "Fill")
            return nullptr;
        gamma = Holder(builder, fill, 0);
    }

    if (candidate.fillBeta) {
        GNode fill = BuildFill(builder, normShape, 0.0F, candidate.betaDesc,
                               NodeName(candidate.layerNorm) + "/Beta_Fill");
        if (NodeType(fill) != "Fill")
            return nullptr;
        beta = Holder(builder, fill, 0);
    }

    TensorDesc yDesc;
    TensorDesc meanDesc;
    TensorDesc rstdDesc;
    TensorDesc xDesc;
    if (!GetOutputDesc(candidate.layerNorm, kY, yDesc) || !GetOutputDesc(candidate.layerNorm, kMean, meanDesc) ||
        !GetOutputDesc(candidate.layerNorm, kRstd, rstdDesc) || !GetOutputDesc(candidate.layerNorm, kY, xDesc)) {
        return nullptr;
    }
    for (TensorDesc* desc : {&yDesc, &meanDesc, &rstdDesc, &xDesc}) {
        if (!desc->GetShape().GetDims().empty() && desc->GetRealDimCnt() <= 0) {
            desc->SetRealDimCnt(static_cast<int64_t>(desc->GetShape().GetDims().size()));
        }
    }
    yDesc.SetDataType(candidate.fusedType);
    meanDesc.SetDataType(DT_FLOAT);
    rstdDesc.SetDataType(DT_FLOAT);
    xDesc.SetDataType(candidate.fusedType);
    std::array<TensorDesc, 4> fusedDescs = {yDesc, meanDesc, rstdDesc, xDesc};
    std::array<es::EsTensorHolder, 5> fusedInputs = {x1, x2, gamma, beta, bias};
    FusedOutputs fused;
    if (!BuildAddLayerNorm(builder, fusedInputs, candidate.hasBias, candidate, fusedDescs, fused)) {
        return nullptr;
    }

    es::EsTensorHolder yOut = fused.y;
    const auto yOutputIt = std::find_if(candidate.outputs.begin(), candidate.outputs.end(),
                                        [](const OutputSpec& output) { return output.kind == OutputKind::Y; });
    if (yOutputIt == candidate.outputs.end())
        return nullptr;
    if (yOutputIt->desc.GetDataType() != candidate.fusedType) {
        GNode cast = BuildCast(builder, fused.y, yOutputIt->desc, NodeName(candidate.layerNorm) + "/Y_Cast");
        if (NodeType(cast) != "Cast")
            return nullptr;
        yOut = Holder(builder, cast, 0);
    }

    es::EsTensorHolder castXOut;
    if (candidate.enableCastX) {
        const auto castOutputIt = std::find_if(
            candidate.outputs.begin(), candidate.outputs.end(),
            [](const OutputSpec& output) { return output.kind == OutputKind::CAST_X; });
        if (castOutputIt == candidate.outputs.end())
            return nullptr;
        TensorDesc castDesc = castOutputIt->desc;
        castDesc.SetDataType(DT_FLOAT);
        GNode cast = BuildCast(builder, fused.x, castDesc, NodeName(candidate.layerNorm) + "/X_Cast");
        if (NodeType(cast) != "Cast")
            return nullptr;
        castXOut = Holder(builder, cast, 0);
    }

    std::vector<es::EsTensorHolder> outputs;
    outputs.reserve(candidate.outputs.size());
    for (const auto& output : candidate.outputs) {
        switch (output.kind) {
            case OutputKind::Y:
                outputs.push_back(yOut);
                break;
            case OutputKind::MEAN:
                outputs.push_back(fused.mean);
                break;
            case OutputKind::RSTD:
                outputs.push_back(fused.rstd);
                break;
            case OutputKind::X:
                outputs.push_back(fused.x);
                break;
            case OutputKind::CAST_X:
                outputs.push_back(castXOut);
                break;
            // Expose the FP32 source after removing the gamma/beta Cast.
            case OutputKind::GAMMA:
                outputs.push_back(gamma);
                break;
            case OutputKind::BETA:
                outputs.push_back(beta);
                break;
        }
    }
    auto graph = builder.BuildAndReset(outputs);
    if (graph == nullptr || !InferReplacement(*graph, candidate, inputShapes)) {
        OPS_LOG_E("AddLayerNormFusionPass", "Cannot build or infer replacement graph.");
        return nullptr;
    }
    return graph;
}

Status ReplaceCandidate(Graph& graph, const Candidate& candidate)
{
    (void)graph;
    auto boundary = BuildBoundary(candidate);
    auto replacement = BuildReplacement(candidate);
    if (boundary == nullptr || replacement == nullptr) {
        return FAILED;
    }
    // Replace inherits external execution dependencies for the entire
    // replacement. Keep its control relays, including dependencies on Cast/Fill.
    return SubgraphRewriter::Replace(*boundary, *replacement);
}

} // namespace

Status AddLayerNormFusionPass::Run(GraphPtr& graph, CustomPassContext& passContext)
{
    if (graph == nullptr) {
        return GRAPH_NOT_CHANGED;
    }
    if (!CanUsePassContextOptionsForVersion(GetRuntimeGeCompilerVersion())) {
        OPS_LOG_D(kPassName.c_str(), "CustomPassContext options require GE compiler 9.0 or later; stay silent.");
        return GRAPH_NOT_CHANGED;
    }
    std::string graphRunMode;
    if (!ReadPassOption(passContext, "ge.graphRunMode", graphRunMode)) {
        OPS_LOG_D(kPassName.c_str(), "GetOptionValue is unavailable; leave the graph to the legacy pass.");
        return GRAPH_NOT_CHANGED;
    }
    std::string precisionMode;
    // Read ge.exec.precision_mode first, then fall back to precision_mode.
    if (!ReadPassOption(passContext, "ge.exec.precision_mode", precisionMode)) {
        return GRAPH_NOT_CHANGED;
    }
    if (precisionMode.empty()) {
        if (!ReadPassOption(passContext, "precision_mode", precisionMode)) {
            return GRAPH_NOT_CHANGED;
        }
    }
    const bool isTrain = IsTrainGraphRunMode(graphRunMode);
    const bool allowMixPrecision = IsAllowMixPrecision(precisionMode);
    std::string npuArch;
    if (!GetNpuArch(npuArch) || !IsSupportedPlatform(npuArch)) {
        OPS_LOG_D(kPassName.c_str(), "Unsupported NpuArch=%s.", npuArch.c_str());
        return GRAPH_NOT_CHANGED;
    }

    struct ScheduledNorm {
        GNode node;
        std::string type;
        std::string name;
        std::string outputCastName;
    };
    std::vector<ScheduledNorm> norms;
    for (const GNode& node : graph->GetDirectNode()) {
        const std::string type = NodeType(node);
        if (!IsLayerNormType(type)) {
            continue;
        }
        std::string outputCastName;
        const auto consumers = node.GetOutDataNodesAndPortIndexs(kY);
        if (consumers.size() == 1U && consumers[0].first != nullptr && NodeType(*consumers[0].first) == "Cast") {
            outputCastName = NodeName(*consumers[0].first);
        }
        norms.push_back({node, type, NodeName(node), outputCastName});
    }

    bool changed = false;
    // Process bias-and-cast fusion, direct fusion, then intermediate-cast fusion.
    enum class PatternKind { THIRD, DIRECT, MIDDLE_CAST };
    for (const PatternKind pattern : {PatternKind::THIRD, PatternKind::DIRECT, PatternKind::MIDDLE_CAST}) {
        std::sort(norms.begin(), norms.end(), [pattern](const ScheduledNorm& lhs, const ScheduledNorm& rhs) {
            if (pattern == PatternKind::THIRD) {
                return lhs.outputCastName < rhs.outputCastName;
            }
            return lhs.type != rhs.type ? lhs.type < rhs.type : lhs.name < rhs.name;
        });
        for (const ScheduledNorm& scheduled : norms) {
            const GNode& node = scheduled.node;
            // Earlier replacements may invalidate this node or change its
            // producers/consumers. Never reuse a Candidate captured beforehand.
            if (!IsLayerNormType(NodeType(node))) {
                continue;
            }
            Candidate candidate;
            bool matched = false;
            if (pattern == PatternKind::THIRD) {
                if (scheduled.outputCastName.empty()) {
                    continue;
                }
                matched = TryMakeThirdCandidate(node, isTrain, allowMixPrecision, candidate);
            } else {
                const GNodePtr producer = Producer(node, kX);
                const std::string expectedType = pattern == PatternKind::DIRECT ? "Add" : "Cast";
                if (producer == nullptr || NodeType(*producer) != expectedType) {
                    continue;
                }
                matched = TryMakeGeneralCandidate(node, isTrain, allowMixPrecision, candidate);
            }
            if (!matched) {
                continue;
            }
            const Status status = ReplaceCandidate(*graph, candidate);
            if (status != SUCCESS) {
                OPS_LOG_E(kPassName.c_str(), "Failed to replace candidate %s.", NodeName(node).c_str());
                return status;
            }
            changed = true;
        }
    }
    return changed ? SUCCESS : GRAPH_NOT_CHANGED;
}

// Register only when the build supports pass context options.
#if defined(GE_COMPILER_VERSION_NUM) && (GE_COMPILER_VERSION_NUM >= 90000000)
REG_FUSION_PASS(AddLayerNormFusionPass).Stage(CustomPassStage::kAfterBuiltinFusionPass);
#endif

} // namespace fusion
} // namespace ge

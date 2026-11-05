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
 * \file gelu_v2_onnx_fusion_pass.cpp
 * \brief GeluV2ONNXFusionPass
 *
 * Fusion pattern: fuse the onnx-style gelu formula into a single GeluV2 operator.
 *
 * erf scenario (approximate = "none"):
 *   gelu(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
 * tanh scenario (approximate = "tanh"), the full approximation matching the kernel:
 *   gelu(x) = 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))
 */

#include "securec.h"

#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "common/inc/error_util.h"
#include "arch_utils.h"
#include "cube_utils/cube_fp16_t.h"
#include "es_math_ops.h"
#include "es_nn_ops.h"
#include "ge/compliant_node_builder.h"
#include "ge/es_graph_builder.h"
#include "ge/ge_utils.h"

#include "gelu_v2_onnx_fusion_pass.h"

namespace ops {
namespace {
const std::string kPassName = "GeluV2ONNXFusionPass";
// NpuArch "3510" 覆盖 Ascend950/Ascend350
const std::set<std::string> kSupportedNpuArchs = {"3510"};

const char* const kOpTypeErf = "Erf";
const char* const kOpTypeTanh = "Tanh";
const char* const kOpTypeRealDiv = "RealDiv";
const char* const kOpTypeMul = "Mul";

const char* const kApproximateNone = "none";
const char* const kApproximateTanh = "tanh";

// fp32 representations of the gelu formula constants; the three math-derived values are the
// nearest fp32 of the exact constants (sqrt(2) = 1.414213562..., 1/sqrt(2) = 0.707106781...,
// sqrt(2/pi) = 0.797884560...), kappa is the standard gelu tanh approximation coefficient and is
// exact by definition. onnx exports may carry truncated or fp16-quantized values, both are
// accepted by the tolerance in CheckFormulaConsts
constexpr float kRealDivErfValue = 1.41421356F; // erf: inputX / sqrt(2)
constexpr float kTrueDivErfValue = 0.70710678F; // erf: inputX * (1 / sqrt(2))
constexpr float kTanhBetaValue = 0.79788456F;   // tanh: sqrt(2 / pi), the beta factor
constexpr float kTanhKappaValue = 0.044715F;    // tanh: the kappa cubic coefficient
constexpr float kAddConstValue = 1.0F;
constexpr float kMulConstValue = 0.5F;
constexpr float kDivToleranceFp32 = 0.00001F;
constexpr float kDivToleranceFp16 = 0.001F;
// covers fp32 truncation (~1e-6) and fp16 quantization (~1.5e-5) of the small kappa value
constexpr float kKappaTolerance = 0.0001F;

// capture indexes, consistent with the CaptureTensor order in the pattern builders:
// both families capture activation/add/mul0/mul1 at 0-3, family specific nodes follow at 4+
constexpr int64_t kCaptureActivationIdx = 0;
constexpr int64_t kCaptureAddIdx = 1;
constexpr int64_t kCaptureMul0Idx = 2;
constexpr int64_t kCaptureMul1Idx = 3;
constexpr int64_t kCaptureDivIdx = 4;       // erf family
constexpr int64_t kCaptureSquareMulIdx = 4; // tanh family
constexpr int64_t kCaptureCubeMulIdx = 5;
constexpr int64_t kCaptureKappaMulIdx = 6;
constexpr int64_t kCaptureInnerAddIdx = 7;
constexpr int64_t kCaptureBetaMulIdx = 8;

// inputX is the only boundary input, the scalar consts are pattern-internal Const nodes so the
// matched subgraph covers them and the replacement only needs one input for inputX
constexpr int64_t kPatternInputX = 0;
constexpr int64_t kScalarConstDim = 1;

constexpr int32_t kBinaryInputX1Idx = 0;
constexpr int32_t kBinaryInputX2Idx = 1;
constexpr int32_t kUnaryInputXIdx = 0;
constexpr int32_t kNodeOutputIdx = 0;

// erf input-order variant bits: add/mul0/mul1 may swap their two inputs independently (Add/Mul are commutative)
constexpr int32_t kAddConstFirstBit = 0x1;
constexpr int32_t kMul0FirstBit = 0x2;
constexpr int32_t kMul1FirstBit = 0x4;
constexpr int32_t kOrderVariantCount = 0x8;

// tanh variant bits: the kappa/beta/one const positions and the outer mul orders.
// cubeMul keeps the left-assoc order Mul(squareMul, inputX), innerAdd keeps the reading order
// Add(inputX, kappaMul) and the 0.5 const keeps the first-binding family (mul0): every known
// onnx exporter writes the formula as 0.5*x*(1+tanh(...)), no producer binds the 0.5 last.
// These are fixed to the dominant export forms to keep the pattern count at 32
constexpr int32_t kKappaConstFirstBit = 0x1;
constexpr int32_t kBetaConstFirstBit = 0x2;
constexpr int32_t kTanhAddConstFirstBit = 0x4;
constexpr int32_t kTanhMul0FirstBit = 0x8;
constexpr int32_t kTanhMul1FirstBit = 0x10;
constexpr int32_t kTanhVariantCount = 0x20;

// the outer mul chain variant: where the 0.5 const sits and the operand orders of both muls
struct OuterMulVariant {
    bool mul1ConstFamily = false; // 0.5 on mul1 (family 2) or mul0 (family 1/3)
    bool mul0First = false;       // the non-inputX operand of mul0 lands on x1
    bool mul1First = false;       // the non-mul0 operand of mul1 lands on x1
};

// wraps the pattern graph builder with the shared node construction helpers, so no helper
// signature needs to repeat the builder
struct GeluNodeFactory {
    ge::es::EsGraphBuilder& builder;

    // Erf/Tanh have no ES API in the ops-nn es packages, build them with CompliantNodeBuilder.
    // Build a one-input one-output element-wise node (Erf/Tanh); Mul/Add/RealDiv use es_math
    ge::es::EsTensorHolder BuildUnary(const ge::es::EsTensorHolder& input, const char* opType) const
    {
        auto* cBuilder = builder.GetCGraphBuilder();
        OPS_ERR_IF(cBuilder == nullptr, OPS_LOG_E(kPassName.c_str(), "Get graph builder failed."),
                   return ge::es::EsTensorHolder());
        auto* graph = cBuilder->GetGraph();
        OPS_ERR_IF(graph == nullptr, OPS_LOG_E(kPassName.c_str(), "Get graph from builder failed."),
                   return ge::es::EsTensorHolder());
        ge::GNode* producer = input.GetProducer();
        OPS_ERR_IF(producer == nullptr, OPS_LOG_E(kPassName.c_str(), "Get producer of unary node input failed."),
                   return ge::es::EsTensorHolder());
        ge::GNode node = ge::es::CompliantNodeBuilder(graph)
                             .OpType(opType)
                             .Name(cBuilder->GenerateNodeName(opType).GetString())
                             .IrDefInputs({{"x", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
                             .IrDefOutputs({{"y", ge::es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                             .Build();
        ES_ASSERT_GRAPH_SUCCESS(
            ge::es::AddEdgeAndUpdatePeerDesc(*graph, *producer, input.GetProducerOutIndex(), node, kUnaryInputXIdx));
        return ge::es::EsTensorHolder(cBuilder->GetTensorHolderFromNode(node, kNodeOutputIdx));
    }

    // build the outer two muls: mul0 = inputX * first, mul1 = mul0 * second, where {first, second} is
    // {mulConst, add} for family 1/3 (the 0.5 const on mul0) or {add, mulConst} for family 2
    // (the 0.5 const on mul1)
    std::pair<ge::es::EsTensorHolder, ge::es::EsTensorHolder> BuildOuterMulChain(const ge::es::EsTensorHolder& inputX,
                                                                                 const ge::es::EsTensorHolder& add,
                                                                                 const ge::es::EsTensorHolder& mulConst,
                                                                                 const OuterMulVariant& variant) const
    {
        const ge::es::EsTensorHolder& first = variant.mul1ConstFamily ? add : mulConst;
        const ge::es::EsTensorHolder& second = variant.mul1ConstFamily ? mulConst : add;
        ge::es::EsTensorHolder mul0 = variant.mul0First ? ge::es::Mul(first, inputX) : ge::es::Mul(inputX, first);
        ge::es::EsTensorHolder mul1 = variant.mul1First ? ge::es::Mul(second, mul0) : ge::es::Mul(mul0, second);
        return {std::move(mul0), std::move(mul1)};
    }
};

// read the raw const tensor fed to node input[inputIdx], only succeeds when the input is const
bool ReadConstTensor(const ge::GNode& node, const int32_t inputIdx, ge::Tensor& constTensor)
{
    return node.GetInputConstData(inputIdx, constTensor) == ge::GRAPH_SUCCESS;
}

// validate the const tensor is a fp32/fp16 scalar with a finite value and parse it; NaN must be
// rejected here because NaN compares false against every tolerance, it would otherwise pass all
// formula const checks and silently change the numerics of the fused graph
bool ParseScalarValue(const ge::Tensor& constTensor, float& value, ge::DataType& dtype)
{
    dtype = constTensor.GetTensorDesc().GetDataType();
    OPS_ERR_IF((dtype != ge::DT_FLOAT) && (dtype != ge::DT_FLOAT16),
               OPS_LOG_D(kPassName.c_str(), "Const dtype %d is not fp32/fp16.", static_cast<int32_t>(dtype)),
               return false);
    const std::vector<int64_t> dims = constTensor.GetTensorDesc().GetShape().GetDims();
    OPS_ERR_IF((!dims.empty()) && ((dims.size() != 1) || (dims[0] != 1)),
               OPS_LOG_D(kPassName.c_str(), "Const input must be scalar, but dims size is %zu.", dims.size()),
               return false);
    const uint8_t* dataPtr = constTensor.GetData();
    OPS_ERR_IF(dataPtr == nullptr, OPS_LOG_D(kPassName.c_str(), "Const tensor data is null."), return false);
    if (dtype == ge::DT_FLOAT) {
        OPS_ERR_IF(constTensor.GetSize() < sizeof(float),
                   OPS_LOG_D(kPassName.c_str(), "Const tensor size %zu is too small for fp32.", constTensor.GetSize()),
                   return false);
        float scalarValue = 0.0F;
        errno_t memRet = memcpy_s(&scalarValue, sizeof(float), dataPtr, sizeof(float));
        OP_LOGE_IF(memRet != EOK, false, kPassName.c_str(), "memcpy_s for fp32 const failed, ret=%d.", memRet);
        value = scalarValue;
    } else {
        OPS_ERR_IF(constTensor.GetSize() < sizeof(uint16_t),
                   OPS_LOG_D(kPassName.c_str(), "Const tensor size %zu is too small for fp16.", constTensor.GetSize()),
                   return false);
        uint16_t fp16Bits = 0;
        errno_t memRet = memcpy_s(&fp16Bits, sizeof(uint16_t), dataPtr, sizeof(uint16_t));
        OP_LOGE_IF(memRet != EOK, false, kPassName.c_str(), "memcpy_s for fp16 const failed, ret=%d.", memRet);
        value = fp16_t(fp16Bits).ToFloat();
    }
    OPS_ERR_IF(std::isnan(value), OPS_LOG_D(kPassName.c_str(), "Const value is nan, skip fusion."), return false);
    return true;
}

// the graph backend may return a fresh ge::GNode wrapper per query, so the boundary node is identified
// by its name (unique inside one graph) instead of pointer equality
bool IsBoundaryInputNode(const ge::GNode& node, const int32_t inputIdx, const std::string& boundaryNodeName)
{
    if (boundaryNodeName.empty()) {
        return false;
    }
    const ge::GNodePtr producer = node.GetInDataNodesAndPortIndexs(inputIdx).first;
    if (producer == nullptr) {
        return false;
    }
    ge::AscendString producerName;
    if (producer->GetName(producerName) != ge::SUCCESS) {
        return false;
    }
    return producerName == boundaryNodeName.c_str();
}

// the boundary inputX node feeds input 0 of the inputX carrier (div for erf, squareMul for tanh); return its name
// so const reads on the add/mul nodes can exclude inputX: when inputX itself is a Const node, its value
// must not be mistaken for the formula const. an empty name means unresolved and fails the fusion, so the
// const scans never run with the boundary exclusion silently disabled
std::string GetBoundaryInputName(const ge::GNode& xCarrierNode)
{
    const ge::GNodePtr xNode = xCarrierNode.GetInDataNodesAndPortIndexs(kBinaryInputX1Idx).first;
    ge::AscendString nameStr;
    OPS_ERR_IF(xNode == nullptr, OPS_LOG_D(kPassName.c_str(), "Get boundary input node failed."), return "");
    OPS_ERR_IF(xNode->GetName(nameStr) != ge::SUCCESS,
               OPS_LOG_D(kPassName.c_str(), "Get boundary input node name failed."), return "");
    const char* nameCStr = nameStr.GetString();
    OPS_ERR_IF(nameCStr == nullptr, OPS_LOG_D(kPassName.c_str(), "Boundary input node name is null, skip fusion."),
               return "");
    return std::string(nameCStr);
}

// legacy GetScalarFromOp behavior: prefer input x2, fall back to x1 only when x2 is not const.
// the boundary input must be excluded: when inputX itself is a Const node, its value must not be
// mistaken for the pattern-internal const (e.g. the 0.5 const on mul0/mul1)
bool GetConstScalarFromBinaryOp(const ge::GNode& node, const std::string& boundaryNodeName, float& value,
                                ge::DataType& dtype)
{
    ge::Tensor constTensor;
    const int32_t inputIndexes[] = {kBinaryInputX2Idx, kBinaryInputX1Idx};
    for (const int32_t inputIdx : inputIndexes) {
        if (ReadConstTensor(node, inputIdx, constTensor) && !IsBoundaryInputNode(node, inputIdx, boundaryNodeName)) {
            return ParseScalarValue(constTensor, value, dtype);
        }
    }
    return false;
}

float GetExpectedDivValue(const bool isTrueDiv) { return isTrueDiv ? kTrueDivErfValue : kRealDivErfValue; }

// check the node output is consumed exactly once and report a skip otherwise: an extra consumer
// outside the pattern would be broken by the replacement, as the whole matched subgraph is removed
bool CheckSingleConsumer(const ge::GNode& node, const char* nodeDesc)
{
    OPS_ERR_IF(node.GetOutDataNodesAndPortIndexs(kNodeOutputIdx).size() != 1,
               OPS_LOG_D(kPassName.c_str(), "%s node has more than one consumer, skip fusion.", nodeDesc),
               return false);
    return true;
}

// the three planned erf topologies, aligned with the legacy canndev pass:
// kRealDivMul0Const (family 1), kRealDivMul1Const (family 2), kTrueDivMul0Const (family 3)
enum class GeluTopology {
    kRealDivMul0Const,
    kRealDivMul1Const,
    kTrueDivMul0Const,
};

// build the pattern over the pattern graph and capture each holder's producer output in the given
// order; the capture indexes must line up with the kCapture* constants: activation/add/mul0/mul1 at
// 0-3 are shared by both families, the family specific nodes follow at 4+
ge::fusion::PatternUniqPtr CapturePatternOutputs(std::unique_ptr<ge::Graph> graph,
                                                 const std::vector<ge::es::EsTensorHolder>& holders)
{
    OPS_ERR_IF(graph == nullptr, OPS_LOG_E(kPassName.c_str(), "Build pattern graph failed."), return nullptr);
    auto pattern = std::make_unique<ge::fusion::Pattern>(std::move(*graph));
    for (const auto& holder : holders) {
        ge::GNode* producer = holder.GetProducer();
        OPS_ERR_IF(producer == nullptr, OPS_LOG_E(kPassName.c_str(), "Get pattern node producer failed."),
                   return nullptr);
        pattern->CaptureTensor({*producer, kNodeOutputIdx});
    }
    return pattern;
}

// build one erf pattern variant: gelu(inputX) = 0.5 * inputX * (1 + erf(inputX / sqrt(2))).
// div keeps inputX on input 0 and the const on input 1, so RealDiv is never commutatively swapped
ge::fusion::PatternUniqPtr MakeGeluErfPattern(const GeluTopology topology, const int32_t orderVariant)
{
    const bool isTrueDiv = (topology == GeluTopology::kTrueDivMul0Const);
    const bool addConstFirst = (orderVariant & kAddConstFirstBit) != 0;
    OuterMulVariant outerVariant;
    outerVariant.mul1ConstFamily = (topology == GeluTopology::kRealDivMul1Const);
    outerVariant.mul0First = (orderVariant & kMul0FirstBit) != 0;
    outerVariant.mul1First = (orderVariant & kMul1FirstBit) != 0;
    const char* divType = isTrueDiv ? kOpTypeMul : kOpTypeRealDiv;
    std::string patternName = kPassName + "_Erf_" + divType +
                              (outerVariant.mul1ConstFamily ? "_Mul1Const" : "_Mul0Const") +
                              (addConstFirst ? "_AddConstFirst" : "_AddConstSecond") +
                              (outerVariant.mul0First ? "_Mul0First" : "_Mul0Second") +
                              (outerVariant.mul1First ? "_Mul1First" : "_Mul1Second");
    auto graphBuilder = ge::es::EsGraphBuilder(patternName.c_str());
    const GeluNodeFactory factory{graphBuilder};
    auto inputX = graphBuilder.CreateInput(kPatternInputX);
    // the const values here only anchor the node type, the real values are validated in MeetRequirements
    auto divConst = graphBuilder.CreateConst(std::vector<float>{GetExpectedDivValue(isTrueDiv)},
                                             std::vector<int64_t>{kScalarConstDim});
    auto addConst = graphBuilder.CreateConst(std::vector<float>{kAddConstValue}, std::vector<int64_t>{kScalarConstDim});
    auto mulConst = graphBuilder.CreateConst(std::vector<float>{kMulConstValue}, std::vector<int64_t>{kScalarConstDim});

    auto div = isTrueDiv ? ge::es::Mul(inputX, divConst) : ge::es::RealDiv(inputX, divConst);
    auto activation = factory.BuildUnary(div, kOpTypeErf);
    auto add = addConstFirst ? ge::es::Add(addConst, activation) : ge::es::Add(activation, addConst);
    auto outer = factory.BuildOuterMulChain(inputX, add, mulConst, outerVariant);

    auto graph = graphBuilder.BuildAndReset({outer.second});
    // capture activation/add/mul0/mul1 at 0-3, then div at 4, matching the kCapture* constants
    return CapturePatternOutputs(std::move(graph), {activation, add, outer.first, outer.second, div});
}

// build one tanh pattern variant matching the kernel formula exactly:
// gelu(inputX) = 0.5 * inputX * (1 + tanh(beta * (inputX + kappa * inputX^3))), beta = sqrt(2/pi), kappa = 0.044715.
// cubeMul = Mul(squareMul, inputX) (left-assoc), innerAdd = Add(inputX, kappaMul) (reading order) and the
// 0.5 const family (mul0, first binding) are fixed to the dominant export forms to keep the pattern count
// at 32; kappaMul/betaMul/add const positions and the outer mul orders vary per tanhVariant
ge::fusion::PatternUniqPtr MakeGeluTanhPattern(const int32_t tanhVariant)
{
    const bool kappaConstFirst = (tanhVariant & kKappaConstFirstBit) != 0;
    const bool betaConstFirst = (tanhVariant & kBetaConstFirstBit) != 0;
    const bool addConstFirst = (tanhVariant & kTanhAddConstFirstBit) != 0;
    OuterMulVariant outerVariant;
    outerVariant.mul0First = (tanhVariant & kTanhMul0FirstBit) != 0;
    outerVariant.mul1First = (tanhVariant & kTanhMul1FirstBit) != 0;
    std::string patternName = kPassName + "_Tanh" + (kappaConstFirst ? "_KappaFirst" : "_KappaSecond") +
                              (betaConstFirst ? "_BetaFirst" : "_BetaSecond") +
                              (addConstFirst ? "_AddConstFirst" : "_AddConstSecond") +
                              (outerVariant.mul0First ? "_Mul0First" : "_Mul0Second") +
                              (outerVariant.mul1First ? "_Mul1First" : "_Mul1Second");
    auto graphBuilder = ge::es::EsGraphBuilder(patternName.c_str());
    const GeluNodeFactory factory{graphBuilder};
    auto inputX = graphBuilder.CreateInput(kPatternInputX);
    auto kappaConst = graphBuilder.CreateConst(std::vector<float>{kTanhKappaValue},
                                               std::vector<int64_t>{kScalarConstDim});
    auto betaConst = graphBuilder.CreateConst(std::vector<float>{kTanhBetaValue},
                                              std::vector<int64_t>{kScalarConstDim});
    auto addConst = graphBuilder.CreateConst(std::vector<float>{kAddConstValue}, std::vector<int64_t>{kScalarConstDim});
    auto mulConst = graphBuilder.CreateConst(std::vector<float>{kMulConstValue}, std::vector<int64_t>{kScalarConstDim});

    auto squareMul = ge::es::Mul(inputX, inputX);
    auto cubeMul = ge::es::Mul(squareMul, inputX);
    auto kappaMul = kappaConstFirst ? ge::es::Mul(kappaConst, cubeMul) : ge::es::Mul(cubeMul, kappaConst);
    auto innerAdd = ge::es::Add(inputX, kappaMul);
    auto betaMul = betaConstFirst ? ge::es::Mul(betaConst, innerAdd) : ge::es::Mul(innerAdd, betaConst);
    auto activation = factory.BuildUnary(betaMul, kOpTypeTanh);
    auto add = addConstFirst ? ge::es::Add(addConst, activation) : ge::es::Add(activation, addConst);
    auto outer = factory.BuildOuterMulChain(inputX, add, mulConst, outerVariant);

    auto graph = graphBuilder.BuildAndReset({outer.second});
    // capture activation/add/mul0/mul1 at 0-3, then the tanh inner chain at 4-8, matching the kCapture*
    // constants
    return CapturePatternOutputs(std::move(graph), {activation, add, outer.first, outer.second, squareMul, cubeMul,
                                                    kappaMul, innerAdd, betaMul});
}
// a failed pattern build must not break the whole pass, skip the null pattern and keep going
void AppendGeluPattern(std::vector<ge::fusion::PatternUniqPtr>& patterns, ge::fusion::PatternUniqPtr pattern)
{
    if (pattern != nullptr) {
        patterns.emplace_back(std::move(pattern));
    }
}

// matched nodes and derived variant flags extracted from a successful pattern match;
// both families capture activation/add/mul0/mul1, the erf family adds div, the tanh family adds
// squareMul/cubeMul/kappaMul/innerAdd/betaMul from the full-formula inner chain
struct MatchedNodes {
    ge::GNode activationNode;
    ge::GNode addNode;
    ge::GNode mul0Node;
    ge::GNode mul1Node;
    bool isTanh = false;
    ge::GNode divNode; // erf family only
    bool isTrueDiv = false;
    ge::GNode squareMulNode; // tanh family only
    ge::GNode cubeMulNode;
    ge::GNode kappaMulNode;
    ge::GNode innerAddNode;
    ge::GNode betaMulNode;
};

// fetch the node captured at captureIdx into node, shared by MeetRequirements and Replacement
bool GetCapturedNode(const std::unique_ptr<ge::fusion::MatchResult>& matchResult, const int64_t captureIdx,
                     ge::GNode& node)
{
    ge::fusion::NodeIo nodeIo;
    OP_LOGE_IF(matchResult->GetCapturedTensor(captureIdx, nodeIo) != ge::SUCCESS, false, kPassName.c_str(),
               "Get captured node at index %lld failed.", static_cast<long long>(captureIdx));
    node = nodeIo.node;
    return true;
}

// fetch the shared activation/add/mul0/mul1 nodes and dispatch on the activation type:
// Erf matches only erf patterns, Tanh matches only the full-formula tanh patterns
bool GetMatchedNodes(const std::unique_ptr<ge::fusion::MatchResult>& matchResult, MatchedNodes& nodes)
{
    if (!GetCapturedNode(matchResult, kCaptureActivationIdx, nodes.activationNode) ||
        !GetCapturedNode(matchResult, kCaptureAddIdx, nodes.addNode) ||
        !GetCapturedNode(matchResult, kCaptureMul0Idx, nodes.mul0Node) ||
        !GetCapturedNode(matchResult, kCaptureMul1Idx, nodes.mul1Node)) {
        return false;
    }

    ge::AscendString activationType;
    OP_LOGE_IF(nodes.activationNode.GetType(activationType) != ge::SUCCESS, false, kPassName.c_str(),
               "Get activation node type failed.");
    nodes.isTanh = (activationType == kOpTypeTanh);
    if (!nodes.isTanh) {
        if (!GetCapturedNode(matchResult, kCaptureDivIdx, nodes.divNode)) {
            return false;
        }
        ge::AscendString divNodeType;
        OP_LOGE_IF(nodes.divNode.GetType(divNodeType) != ge::SUCCESS, false, kPassName.c_str(),
                   "Get div node type failed.");
        nodes.isTrueDiv = (divNodeType == kOpTypeMul);
        return true;
    }
    return GetCapturedNode(matchResult, kCaptureSquareMulIdx, nodes.squareMulNode) &&
           GetCapturedNode(matchResult, kCaptureCubeMulIdx, nodes.cubeMulNode) &&
           GetCapturedNode(matchResult, kCaptureKappaMulIdx, nodes.kappaMulNode) &&
           GetCapturedNode(matchResult, kCaptureInnerAddIdx, nodes.innerAddNode) &&
           GetCapturedNode(matchResult, kCaptureBetaMulIdx, nodes.betaMulNode);
}

// input dtype must be fp32/fp16/bf16; legacy restriction: the TrueDiv(Mul) boundary input must
// not be a scalar. inputX is carried by div input 0 (erf) or squareMul input 0 (tanh,
// squareMul = Mul(inputX, inputX) so both ports are inputX)
bool CheckInputConstraints(const MatchedNodes& nodes)
{
    const ge::GNode& xCarrier = nodes.isTanh ? nodes.squareMulNode : nodes.divNode;
    ge::TensorDesc xDesc;
    OP_LOGE_IF(xCarrier.GetInputDesc(kBinaryInputX1Idx, xDesc) != ge::SUCCESS, false, kPassName.c_str(),
               "Get boundary input desc failed.");
    const ge::DataType xDtype = xDesc.GetDataType();
    OPS_ERR_IF(
        (xDtype != ge::DT_FLOAT) && (xDtype != ge::DT_FLOAT16) && (xDtype != ge::DT_BF16),
        OPS_LOG_D(kPassName.c_str(), "Input dtype %d is not supported by GeluV2, only fp32/fp16/bf16 are supported.",
                  static_cast<int32_t>(xDtype)),
        return false);
    if (!nodes.isTanh && nodes.isTrueDiv) {
        const std::vector<int64_t> xDims = xDesc.GetShape().GetDims();
        OPS_ERR_IF(xDims.empty() || ((xDims.size() == 1) && (xDims[0] == 1)),
                   OPS_LOG_D(kPassName.c_str(), "The TrueDiv boundary input must not be a scalar."), return false);
    }
    return true;
}

// each intermediate node must be consumed exactly once, otherwise fusion breaks other paths
bool CheckIntermediateConsumer(const MatchedNodes& nodes)
{
    if (nodes.isTanh) {
        return CheckSingleConsumer(nodes.squareMulNode, "square mul") &&
               CheckSingleConsumer(nodes.cubeMulNode, "cube mul") &&
               CheckSingleConsumer(nodes.kappaMulNode, "kappa mul") &&
               CheckSingleConsumer(nodes.innerAddNode, "inner add") &&
               CheckSingleConsumer(nodes.betaMulNode, "beta mul") &&
               CheckSingleConsumer(nodes.activationNode, "activation") && CheckSingleConsumer(nodes.addNode, "add") &&
               CheckSingleConsumer(nodes.mul0Node, "mul0");
    }
    return CheckSingleConsumer(nodes.divNode, "div") && CheckSingleConsumer(nodes.activationNode, "activation") &&
           CheckSingleConsumer(nodes.addNode, "add") && CheckSingleConsumer(nodes.mul0Node, "mul0");
}

// the add const must be 1.0 and the 0.5 const lives on mul0 (family 1/3) or mul1 (family 2),
// shared by both families; the boundary name excludes inputX when inputX itself is a Const node
bool CheckAddAndMulConsts(const MatchedNodes& nodes, const std::string& xNodeName)
{
    float addValue = 0.0F;
    ge::DataType addDtype = ge::DT_UNDEFINED;
    OPS_ERR_IF(!GetConstScalarFromBinaryOp(nodes.addNode, xNodeName, addValue, addDtype),
               OPS_LOG_D(kPassName.c_str(), "Cannot get const value from add node, skip fusion."), return false);
    OPS_ERR_IF(std::fabs(addValue - kAddConstValue) > std::numeric_limits<float>::epsilon(),
               OPS_LOG_D(kPassName.c_str(), "Add const value %f is not equal to 1.0, skip fusion.", addValue),
               return false);

    float mulValue = 0.0F;
    ge::DataType mulDtype = ge::DT_UNDEFINED;
    if (GetConstScalarFromBinaryOp(nodes.mul0Node, xNodeName, mulValue, mulDtype)) {
        OPS_ERR_IF(std::fabs(mulValue - kMulConstValue) > std::numeric_limits<float>::epsilon(),
                   OPS_LOG_D(kPassName.c_str(), "Mul0 const value %f is not equal to 0.5, skip fusion.", mulValue),
                   return false);
    } else if (GetConstScalarFromBinaryOp(nodes.mul1Node, xNodeName, mulValue, mulDtype)) {
        OPS_ERR_IF(std::fabs(mulValue - kMulConstValue) > std::numeric_limits<float>::epsilon(),
                   OPS_LOG_D(kPassName.c_str(), "Mul1 const value %f is not equal to 0.5, skip fusion.", mulValue),
                   return false);
    } else {
        OPS_LOG_D(kPassName.c_str(), "Cannot get 0.5 const value from mul0 or mul1, skip fusion.");
        return false;
    }
    return true;
}

// erf formula consts: div const + add 1.0 + mul 0.5
bool CheckErfFormulaConsts(const MatchedNodes& nodes)
{
    const std::string xNodeNameStr = GetBoundaryInputName(nodes.divNode);
    OPS_ERR_IF(xNodeNameStr.empty(),
               OPS_LOG_D(kPassName.c_str(), "Cannot resolve the erf boundary input name, skip fusion."), return false);

    // div const: inputX / sqrt(2) or inputX * (1 / sqrt(2))
    float divValue = 0.0F;
    ge::DataType divDtype = ge::DT_UNDEFINED;
    ge::Tensor divConstTensor;
    OPS_ERR_IF(!ReadConstTensor(nodes.divNode, kBinaryInputX2Idx, divConstTensor),
               OPS_LOG_D(kPassName.c_str(), "Cannot get div const input, skip fusion."), return false);
    OPS_ERR_IF(!ParseScalarValue(divConstTensor, divValue, divDtype),
               OPS_LOG_D(kPassName.c_str(), "Cannot parse div const scalar, skip fusion."), return false);
    const float divTolerance = (divDtype == ge::DT_FLOAT) ? kDivToleranceFp32 : kDivToleranceFp16;
    const float expectedDivValue = GetExpectedDivValue(nodes.isTrueDiv);
    OPS_ERR_IF(std::fabs(divValue - expectedDivValue) > divTolerance,
               OPS_LOG_D(kPassName.c_str(), "Div const value %f is not equal to expected %f, skip fusion.", divValue,
                         expectedDivValue),
               return false);
    return CheckAddAndMulConsts(nodes, xNodeNameStr);
}

// tanh formula consts: kappa + beta + add 1.0 + mul 0.5.
// kappaMul's other input is the cubeMul op output and betaMul's other input is the innerAdd op output, so the const
// scans on both nodes are unambiguous and need no boundary exclusion
bool CheckTanhFormulaConsts(const MatchedNodes& nodes)
{
    const std::string xNodeNameStr = GetBoundaryInputName(nodes.squareMulNode);
    OPS_ERR_IF(xNodeNameStr.empty(),
               OPS_LOG_D(kPassName.c_str(), "Cannot resolve the tanh boundary input name, skip fusion."), return false);

    float kappaValue = 0.0F;
    ge::DataType kappaDtype = ge::DT_UNDEFINED;
    OPS_ERR_IF(!GetConstScalarFromBinaryOp(nodes.kappaMulNode, "", kappaValue, kappaDtype),
               OPS_LOG_D(kPassName.c_str(), "Cannot get kappa const value, skip fusion."), return false);
    OPS_ERR_IF(std::fabs(kappaValue - kTanhKappaValue) > kKappaTolerance,
               OPS_LOG_D(kPassName.c_str(), "Kappa const value %f is not equal to expected %f, skip fusion.",
                         kappaValue, kTanhKappaValue),
               return false);

    float betaValue = 0.0F;
    ge::DataType betaDtype = ge::DT_UNDEFINED;
    OPS_ERR_IF(!GetConstScalarFromBinaryOp(nodes.betaMulNode, "", betaValue, betaDtype),
               OPS_LOG_D(kPassName.c_str(), "Cannot get beta const value, skip fusion."), return false);
    const float betaTolerance = (betaDtype == ge::DT_FLOAT) ? kDivToleranceFp32 : kDivToleranceFp16;
    OPS_ERR_IF(std::fabs(betaValue - kTanhBetaValue) > betaTolerance,
               OPS_LOG_D(kPassName.c_str(), "Beta const value %f is not equal to expected %f, skip fusion.", betaValue,
                         kTanhBetaValue),
               return false);
    return CheckAddAndMulConsts(nodes, xNodeNameStr);
}

// validate the formula consts against the gelu formula, dispatched by the activation family
bool CheckFormulaConsts(const MatchedNodes& nodes)
{
    if (nodes.isTanh) {
        return CheckTanhFormulaConsts(nodes);
    }
    return CheckErfFormulaConsts(nodes);
}
} // namespace

// The ES pattern matcher compares input ports strictly and provides no commutative-operand
// matching (PatternMatcherConfig only offers const-value and IR-attr matching, both
// binary-exact), so every export order of the commutative Add/Mul nodes must be enumerated
// as an explicit pattern variant. The erf formula keeps the three legacy topologies times 8
// input-order variants = 24 patterns; the tanh formula matches the kernel's full approximation
// 0.5*inputX*(1+tanh(beta*(inputX+kappa*inputX^3))) with 32 variants (kappa/beta/one const
// positions and the outer mul orders), 56 patterns in total. Const values are deliberately
// not matched at pattern level; they are validated with tolerance in MeetRequirements instead.
std::vector<ge::fusion::PatternUniqPtr> GeluV2ONNXFusionPass::Patterns()
{
    OPS_LOG_D(kPassName.c_str(), "Enter Patterns for GeluV2ONNXFusionPass");
    std::vector<ge::fusion::PatternUniqPtr> patterns;
    for (int32_t orderVariant = 0; orderVariant < kOrderVariantCount; ++orderVariant) {
        AppendGeluPattern(patterns, MakeGeluErfPattern(GeluTopology::kRealDivMul0Const, orderVariant));
        AppendGeluPattern(patterns, MakeGeluErfPattern(GeluTopology::kRealDivMul1Const, orderVariant));
        AppendGeluPattern(patterns, MakeGeluErfPattern(GeluTopology::kTrueDivMul0Const, orderVariant));
    }
    for (int32_t tanhVariant = 0; tanhVariant < kTanhVariantCount; ++tanhVariant) {
        AppendGeluPattern(patterns, MakeGeluTanhPattern(tanhVariant));
    }
    return patterns;
}

bool GeluV2ONNXFusionPass::MeetRequirements(const std::unique_ptr<ge::fusion::MatchResult>& match_result)
{
    OPS_LOG_D(kPassName.c_str(), "Enter MeetRequirements for GeluV2ONNXFusionPass");
    if (!CheckNpuArch(kPassName, kSupportedNpuArchs)) {
        OPS_LOG_D(kPassName.c_str(), "Current platform is not supported, skip fusion.");
        return false;
    }

    MatchedNodes nodes;
    OP_LOGE_IF(!GetMatchedNodes(match_result, nodes), false, kPassName.c_str(), "Get matched nodes failed.");

    if (!CheckInputConstraints(nodes)) {
        return false;
    }
    if (!CheckIntermediateConsumer(nodes)) {
        return false;
    }
    return CheckFormulaConsts(nodes);
}

ge::fusion::GraphUniqPtr GeluV2ONNXFusionPass::Replacement(const std::unique_ptr<ge::fusion::MatchResult>& match_result)
{
    OPS_LOG_D(kPassName.c_str(), "Enter Replacement for GeluV2ONNXFusionPass");
    ge::GNode xCarrierNode;
    ge::GNode activationNode;
    ge::GNode mul1Node;
    if (!GetCapturedNode(match_result, kCaptureDivIdx, xCarrierNode) ||
        !GetCapturedNode(match_result, kCaptureActivationIdx, activationNode) ||
        !GetCapturedNode(match_result, kCaptureMul1Idx, mul1Node)) {
        return nullptr;
    }

    ge::TensorDesc xDesc;
    OP_LOGE_IF(xCarrierNode.GetInputDesc(kBinaryInputX1Idx, xDesc) != ge::SUCCESS, nullptr, kPassName.c_str(),
               "Get boundary input desc failed.");
    ge::AscendString activationType;
    OP_LOGE_IF(activationNode.GetType(activationType) != ge::SUCCESS, nullptr, kPassName.c_str(),
               "Get activation node type failed.");
    const char* approximate = (activationType == kOpTypeTanh) ? kApproximateTanh : kApproximateNone;

    auto replaceGraphBuilder = ge::es::EsGraphBuilder("replacement");
    auto inputX = replaceGraphBuilder.CreateInput(kPatternInputX, "x", xDesc.GetDataType(), xDesc.GetFormat(),
                                                  xDesc.GetShape().GetDims());
    auto geluOutput = ge::es::GeluV2(inputX, approximate);

    ge::GNode* geluProducer = geluOutput.GetProducer();
    OP_LOGE_IF(geluProducer == nullptr, nullptr, kPassName.c_str(), "Get gelu node producer failed.");
    ge::GNode geluNode = *geluProducer;
    OP_LOGE_IF(geluNode.UpdateInputDesc(kUnaryInputXIdx, xDesc) != ge::GRAPH_SUCCESS, nullptr, kPassName.c_str(),
               "Update gelu input desc failed.");
    ge::TensorDesc outputDesc;
    OP_LOGE_IF(mul1Node.GetOutputDesc(kNodeOutputIdx, outputDesc) != ge::SUCCESS, nullptr, kPassName.c_str(),
               "Get mul1 output desc failed.");
    OP_LOGE_IF(geluNode.UpdateOutputDesc(kNodeOutputIdx, outputDesc) != ge::GRAPH_SUCCESS, nullptr, kPassName.c_str(),
               "Update gelu output desc failed.");

    ge::fusion::GraphUniqPtr replaceGraph = replaceGraphBuilder.BuildAndReset({geluOutput});
    OPS_ERR_IF(replaceGraph == nullptr, OPS_LOG_E(kPassName.c_str(), "Build replacement graph failed."),
               return nullptr);
    OPS_ERR_IF(ge::GeUtils::InferShape(*replaceGraph, {xDesc.GetShape()}) != ge::SUCCESS,
               OPS_LOG_E(kPassName.c_str(), "InferShape for replacement failed."), return nullptr);
    return replaceGraph;
}

REG_FUSION_PASS(GeluV2ONNXFusionPass).Stage(ge::CustomPassStage::kAfterInferShape);
} // namespace ops

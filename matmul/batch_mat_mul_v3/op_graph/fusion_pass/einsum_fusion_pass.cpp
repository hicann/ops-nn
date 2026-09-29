/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "einsum_fusion_pass.h"

#include <algorithm>
#include <cstdint>
#include <map>
#include <numeric>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include "es_nn_ops.h"
#include "es_math_ops.h"
#include "ge/es_graph_builder.h"
#include "ge/compliant_node_builder.h"
#include "ge/ge_utils.h"
#include "platform/platform_info.h"
#include "common/inc/error_util.h"
#include "common/op_graph/fusion_pass/matmul_fusion_utils_pass.h"

// Note: do NOT add "using namespace ge::es;" here. The es_math Shape op wrapper declares
// ge::es::Shape(...), which would clash with the ge::Shape class on every unqualified use.
// The ES names used unqualified in this file are imported explicitly below; all other ES
// calls go through the es:: qualified prefix (resolved via "using namespace ge").
using ge::Shape;
using ge::es::AddEdgeAndUpdatePeerDesc;
using ge::es::CompliantNodeBuilder;
using ge::es::CreateFrom;
using ge::es::EsGraphBuilder;
using ge::es::EsTensorHolder;
using namespace ge;
using namespace ge::fusion;
using namespace fe;

namespace ops {
namespace {

const std::string kPassName = "EinsumPass";
const std::string kFusedOpType = "Einsum";
const size_t kMaxInputNum = 2;
const size_t kBroadCastLabelLen = 3;
const size_t kEquationPartNum = 2;
const size_t kEllipsisTailLen = 2;
const size_t kArrowLen = 2;

static const uint8_t kNumOfLetters = 'z' - 'a' + 1;
static const uint8_t kTotalLabels = kNumOfLetters * 2;
static const uint8_t kEllipsis = kTotalLabels;
const size_t kMergedFreeDimNum = 2;
const size_t kDim1D = 1;
const size_t kDim2D = 2;
const size_t kDim3D = 3;
const size_t kDim4D = 4;
const int32_t kNumTwo = 2;
const int32_t kGeCompilerVersion900 = 90000000;

const std::string kFlatten = "FlattenV2";
const std::string kUnsqueeze = "Unsqueeze";
const std::string kReshape = "Reshape";
const std::string kTransposeD = "TransposeD";
const std::string kTranspose = "Transpose";
const std::string kMatMul = "MatMulV2";
const std::string kBatchMatMul = "BatchMatMul";
const std::string kBatchMatMulV2 = "BatchMatMulV2";
const std::string kGatherShapes = "GatherShapes";
const std::string kReduceSumD = "ReduceSumD";
const std::string kReduceSum = "ReduceSum";
const std::string kBroadCastLabel = "...";

int64_t GetDimMulValue(int64_t dimValue1, int64_t dimValue2)
{
    if (dimValue1 == -1 || dimValue2 == -1) {
        return -1;
    }
    return dimValue1 * dimValue2;
}

void EquationNormalization(std::string& equation)
{
    std::map<char, char> normalizeMap;
    size_t indice = 0;
    for (auto& dimShape : equation) {
        if (isalpha(dimShape)) {
            if (normalizeMap.find(dimShape) == normalizeMap.end()) {
                normalizeMap[dimShape] = 'a' + indice;
                indice++;
            }
            dimShape = normalizeMap[dimShape];
        }
    }
}

bool IsUnknownShape(const Shape& shape)
{
    const auto dims = shape.GetDims();
    return std::any_of(dims.begin(), dims.end(),
                       [](int64_t dim) { return dim == ge::UNKNOWN_DIM || dim == ge::UNKNOWN_DIM_NUM || dim < 0; });
}

Status InferShape(const GraphUniqPtr& replaceGraph, const std::vector<SubgraphInput>& subgraphInputs)
{
    std::vector<Shape> inputShapes;
    for (const auto& subgraphInput : subgraphInputs) {
        auto matchNode = subgraphInput.GetAllInputs().at(0);
        TensorDesc tensorDesc;
        FUSION_PASS_CHECK(matchNode.node.GetInputDesc(matchNode.index, tensorDesc) != GRAPH_SUCCESS,
                          OPS_LOG_E(kPassName.c_str(), "Failed to get input desc %zu.", matchNode.index),
                          return FAILED;);
        inputShapes.emplace_back(tensorDesc.GetShape());
    }
    return GeUtils::InferShape(*replaceGraph, inputShapes);
}

// Create a TransposeD node (perm as attribute) or Transpose node (perm as const input)
GNode CreateTransposeNode(EsGraphBuilder& builder, const std::string& name, const EsTensorHolder& x, bool isDynamic,
                          bool supportL12btBf16, const std::vector<int32_t>& perm)
{
    if (isDynamic || supportL12btBf16) {
        // Transpose via the ES API: perm as const input, edges wired automatically
        auto permConst = builder.CreateConst(perm, {static_cast<int64_t>(perm.size())});
        auto out = es::Transpose(x, permConst);
        return *out.GetProducer();
    }
    // TransposeD: perm is an attribute (no ES wrapper for the D variant)
    std::vector<int64_t> permInt64(perm.begin(), perm.end());
    ge::Graph* graphPtr = builder.GetCGraphBuilder()->GetGraph();
    GNode transNode = CompliantNodeBuilder(graphPtr)
                          .OpType(kTransposeD.c_str())
                          .Name(name.c_str())
                          .IrDefInputs({{"x", CompliantNodeBuilder::kEsIrInputRequired, ""}})
                          .IrDefOutputs({{"y", CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                          .IrDefAttrs({
                              {"perm", CompliantNodeBuilder::kEsAttrRequired, "ListInt", CreateFrom(permInt64)},
                          })
                          .Build();
    FUSION_PASS_CHECK(
        AddEdgeAndUpdatePeerDesc(*graphPtr, *x.GetProducer(), x.GetProducerOutIndex(), transNode, 0) != GRAPH_SUCCESS,
        OPS_LOG_W(kPassName.c_str(), "AddEdgeAndUpdatePeerDesc failed."), );
    return transNode;
}

void SetTransposeOutputDesc(GNode& transNode, const std::vector<int32_t>& perm)
{
    auto [inPtr, inPort] = transNode.GetInDataNodesAndPortIndexs(0);
    if (inPtr == nullptr) {
        return;
    }
    TensorDesc inDesc;
    FUSION_PASS_CHECK(inPtr->GetOutputDesc(inPort, inDesc) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName.c_str(), "Failed to get output desc."), return;);
    auto inDims = inDesc.GetShape().GetDims();
    std::vector<int64_t> outDims;
    outDims.reserve(perm.size());
    for (int32_t p : perm) {
        if (static_cast<size_t>(p) < inDims.size()) {
            outDims.push_back(inDims[p]);
        }
    }
    TensorDesc outDesc(Shape(outDims), inDesc.GetFormat(), inDesc.GetDataType());
    outDesc.SetOriginFormat(inDesc.GetFormat());
    outDesc.SetOriginShape(Shape(outDims));
    transNode.UpdateOutputDesc(0, outDesc);
}

void SetReshapeOutputDesc(GNode& reshapeNode, const std::vector<int64_t>& targetDims)
{
    auto [inPtr, inPort] = reshapeNode.GetInDataNodesAndPortIndexs(0);
    if (inPtr == nullptr) {
        return;
    }
    TensorDesc inDesc;
    FUSION_PASS_CHECK(inPtr->GetOutputDesc(inPort, inDesc) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName.c_str(), "Failed to get output desc."), return;);
    TensorDesc outDesc = inDesc;
    outDesc.SetShape(Shape(targetDims));
    outDesc.SetOriginShape(Shape(targetDims));
    reshapeNode.UpdateOutputDesc(0, outDesc);
}

void SetGatherShapesOutputDesc(GNode& gatherNode, size_t axesSize)
{
    int64_t sz = static_cast<int64_t>(axesSize);
    TensorDesc outDesc(Shape({sz}), FORMAT_ND, DT_INT64);
    outDesc.SetOriginFormat(FORMAT_ND);
    outDesc.SetOriginShape(Shape({sz}));
    gatherNode.UpdateOutputDesc(0, outDesc);
}

void SetUnsqueezeOutputDesc(GNode& unsqueezeNode, const std::vector<int64_t>& axes)
{
    auto [inPtr, inPort] = unsqueezeNode.GetInDataNodesAndPortIndexs(0);
    if (inPtr == nullptr) {
        return;
    }
    TensorDesc inDesc;
    FUSION_PASS_CHECK(inPtr->GetOutputDesc(inPort, inDesc) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName.c_str(), "Failed to get output desc."), return;);
    auto inDims = inDesc.GetShape().GetDims();
    std::set<int64_t> axisSet(axes.begin(), axes.end());
    std::vector<int64_t> outDims;
    size_t inIdx = 0;
    for (size_t i = 0; i < inDims.size() + axes.size(); ++i) {
        if (axisSet.find(static_cast<int64_t>(i)) != axisSet.end()) {
            outDims.push_back(1);
        } else if (inIdx < inDims.size()) {
            outDims.push_back(inDims[inIdx++]);
        }
    }
    TensorDesc outDesc(Shape(outDims), inDesc.GetFormat(), inDesc.GetDataType());
    outDesc.SetOriginFormat(inDesc.GetFormat());
    outDesc.SetOriginShape(Shape(outDims));
    unsqueezeNode.UpdateOutputDesc(0, outDesc);
}

void SetMulOutputDesc(GNode& mulNode)
{
    auto [in0Ptr, in0Port] = mulNode.GetInDataNodesAndPortIndexs(0);
    auto [in1Ptr, in1Port] = mulNode.GetInDataNodesAndPortIndexs(1);
    if (in0Ptr == nullptr) {
        return;
    }
    TensorDesc in0Desc;
    FUSION_PASS_CHECK(in0Ptr->GetOutputDesc(in0Port, in0Desc) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName.c_str(), "Failed to get output desc."), return;);
    auto in0Dims = in0Desc.GetShape().GetDims();
    if (in1Ptr != nullptr) {
        TensorDesc in1Desc;
        FUSION_PASS_CHECK(in1Ptr->GetOutputDesc(in1Port, in1Desc) != GRAPH_SUCCESS,
                          OPS_LOG_E(kPassName.c_str(), "Failed to get output desc."), return;);
        auto in1Dims = in1Desc.GetShape().GetDims();
        auto maxDimNum = std::max(in0Dims.size(), in1Dims.size());
        std::vector<int64_t> outDims;
        for (size_t i = 0; i < maxDimNum; ++i) {
            int64_t d0 = (i < in0Dims.size()) ? in0Dims[in0Dims.size() - 1 - i] : 1;
            int64_t d1 = (i < in1Dims.size()) ? in1Dims[in1Dims.size() - 1 - i] : 1;
            outDims.insert(outDims.begin(), (d0 == 1) ? d1 : d0);
        }
        TensorDesc outDesc(Shape(outDims), in0Desc.GetFormat(), in0Desc.GetDataType());
        outDesc.SetOriginFormat(in0Desc.GetFormat());
        outDesc.SetOriginShape(Shape(outDims));
        mulNode.UpdateOutputDesc(0, outDesc);
    } else {
        mulNode.UpdateOutputDesc(0, in0Desc);
    }
}

// Create a Reshape node (static) or FlattenV2 node (dynamic)
GNode CreateReshapeNode(EsGraphBuilder& builder, const std::string& name, const EsTensorHolder& x, bool isDynamic,
                        const std::vector<int64_t>& dims, int32_t axis = 0, int32_t endAxis = 0)
{
    ge::Graph* graphPtr = builder.GetCGraphBuilder()->GetGraph();
    if (isDynamic) {
        // FlattenV2: no ES wrapper in the local es_math package, keep CompliantNodeBuilder
        GNode reshapeNode = CompliantNodeBuilder(graphPtr)
                                .OpType(kFlatten.c_str())
                                .Name(name.c_str())
                                .IrDefInputs({{"x", CompliantNodeBuilder::kEsIrInputRequired, ""}})
                                .IrDefOutputs({{"y", CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                                .IrDefAttrs({
                                    {"axis", CompliantNodeBuilder::kEsAttrRequired, "Int",
                                     CreateFrom(static_cast<int64_t>(axis))},
                                    {"end_axis", CompliantNodeBuilder::kEsAttrRequired, "Int",
                                     CreateFrom(static_cast<int64_t>(endAxis))},
                                })
                                .Build();
        FUSION_PASS_CHECK(AddEdgeAndUpdatePeerDesc(*graphPtr, *x.GetProducer(), x.GetProducerOutIndex(), reshapeNode,
                                                   0) != GRAPH_SUCCESS,
                          OPS_LOG_W(kPassName.c_str(), "AddEdgeAndUpdatePeerDesc failed."), );
        return reshapeNode;
    }
    // Reshape via the ES API: shape as const input, edges wired automatically (axis=0, num_axes=-1)
    std::vector<int32_t> dimsInt32(dims.begin(), dims.end());
    auto shapeConst = builder.CreateConst(dimsInt32, {static_cast<int64_t>(dimsInt32.size())});
    auto out = es::Reshape(x, shapeConst);
    return *out.GetProducer();
}

// Create a Cast node via the ES API (edge wired automatically)
GNode CreateCastNode(EsGraphBuilder& builder, const EsTensorHolder& x, DataType dstType)
{
    (void)builder; // owner builder is resolved from the input tensor by the ES API
    auto out = es::Cast(x, static_cast<int64_t>(dstType));
    return *out.GetProducer();
}

// Create an Unsqueeze node
GNode CreateUnsqueezeNode(Graph* graphPtr, const std::string& name, const std::vector<int64_t>& axes)
{
    return CompliantNodeBuilder(graphPtr)
        .OpType(kUnsqueeze.c_str())
        .Name(name.c_str())
        .IrDefInputs({{"x", CompliantNodeBuilder::kEsIrInputRequired, ""}})
        .IrDefOutputs({{"y", CompliantNodeBuilder::kEsIrOutputRequired, ""}})
        .IrDefAttrs({
            {"axes", CompliantNodeBuilder::kEsAttrRequired, "ListInt", CreateFrom(axes)},
        })
        .Build();
}

// Create a Mul node via the ES API (edges wired automatically)
GNode CreateMulNode(EsGraphBuilder& builder, const EsTensorHolder& x1, const EsTensorHolder& x2)
{
    (void)builder; // owner builder is resolved from the input tensors by the ES API
    auto out = es::Mul(x1, x2);
    return *out.GetProducer();
}

// Create a ReduceSumD node (axes as attribute) or ReduceSum node (axes as const input)
GNode CreateReduceSumNode(Graph* graphPtr, EsGraphBuilder& builder, const std::string& name, bool supportL12btBf16,
                          const std::vector<int64_t>& axes, bool keepDims)
{
    if (supportL12btBf16) {
        // ReduceSum: axes as const input
        GNode reduceNode = CompliantNodeBuilder(graphPtr)
                               .OpType(kReduceSum.c_str())
                               .Name(name.c_str())
                               .IrDefInputs({
                                   {"x", CompliantNodeBuilder::kEsIrInputRequired, ""},
                                   {"axes", CompliantNodeBuilder::kEsIrInputRequired, ""},
                               })
                               .IrDefOutputs({{"y", CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                               .IrDefAttrs({
                                   {"keep_dims", CompliantNodeBuilder::kEsAttrRequired, "Bool", CreateFrom(keepDims)},
                               })
                               .Build();
        auto axesConst = builder.CreateConst(axes, {static_cast<int64_t>(axes.size())});
        GNode axesNode = *axesConst.GetProducer();
        FUSION_PASS_CHECK(AddEdgeAndUpdatePeerDesc(*graphPtr, axesNode, 0, reduceNode, 1) != GRAPH_SUCCESS,
                          OPS_LOG_W(kPassName.c_str(), "AddEdgeAndUpdatePeerDesc failed."), );
        return reduceNode;
    }
    // ReduceSumD: axes as attribute
    return CompliantNodeBuilder(graphPtr)
        .OpType(kReduceSumD.c_str())
        .Name(name.c_str())
        .IrDefInputs({{"x", CompliantNodeBuilder::kEsIrInputRequired, ""}})
        .IrDefOutputs({{"y", CompliantNodeBuilder::kEsIrOutputRequired, ""}})
        .IrDefAttrs({
            {"axes", CompliantNodeBuilder::kEsAttrRequired, "ListInt", CreateFrom(axes)},
            {"keep_dims", CompliantNodeBuilder::kEsAttrRequired, "Bool", CreateFrom(keepDims)},
        })
        .Build();
}

// Get EsTensorHolder from a GNode
EsTensorHolder GetTensorHolder(EsGraphBuilder& builder, const GNode& node, int32_t port)
{
    return EsTensorHolder(builder.GetCGraphBuilder()->GetTensorHolderFromNode(node, port));
}

// Set the output desc (batch dims + m + n) on the matmul node produced by the ES API
static void SetMatMulOutputDesc(const EsTensorHolder& out, const TensorDesc& x1Desc, const TensorDesc& x2Desc,
                                bool adjX1, bool adjX2)
{
    GNode node = *out.GetProducer();
    auto x1Dims = x1Desc.GetShape().GetDims();
    auto x2Dims = x2Desc.GetShape().GetDims();
    auto x1DimNum = x1Dims.size();
    auto x2DimNum = x2Dims.size();
    int64_t m = (x1DimNum >= kDim2D) ? (adjX1 ? x1Dims[x1DimNum - 1] : x1Dims[x1DimNum - 2]) : 1;
    int64_t n = (x2DimNum >= kDim2D) ? (adjX2 ? x2Dims[x2DimNum - 2] : x2Dims[x2DimNum - 1]) : 1;
    size_t batchNum = (x1DimNum >= kDim2D) ? (x1DimNum - kDim2D) : 0;
    std::vector<int64_t> outDims;
    for (size_t i = 0; i < batchNum; ++i) {
        outDims.push_back(x1Dims[i]);
    }
    outDims.push_back(m);
    outDims.push_back(n);
    TensorDesc outDesc(Shape(outDims), x1Desc.GetFormat(), x1Desc.GetDataType());
    outDesc.SetOriginFormat(x1Desc.GetFormat());
    outDesc.SetOriginShape(Shape(outDims));
    node.UpdateOutputDesc(0, outDesc);
}

// Build matmul-like node via the official ES API generated from the IR proto definition.
// The ES API (es::BatchMatMul / es::BatchMatMulV2 / es::MatMulV2) already declares the full IR
// (inputs / optional inputs / attrs) per the registered proto, so no manual IrDef is needed.
// It also wires the input edges automatically; the output desc (batch dims + m + n) is still
// computed manually here since the ES builder does not infer it.
EsTensorHolder CreateMatMulNode(EsGraphBuilder& builder, const std::string& opType, const EsTensorHolder& x1,
                                const EsTensorHolder& x2, bool adjX1, bool adjX2)
{
    (void)builder; // owner builder is resolved from the input tensors by the ES API
    GNode x1Producer = *x1.GetProducer();
    GNode x2Producer = *x2.GetProducer();
    TensorDesc x1Desc;
    if (x1Producer.GetOutputDesc(x1.GetProducerOutIndex(), x1Desc) != GRAPH_SUCCESS) {
        OPS_LOG_W(kPassName.c_str(), "Failed to get x1 output desc.");
    }
    TensorDesc x2Desc;
    if (x2Producer.GetOutputDesc(x2.GetProducerOutIndex(), x2Desc) != GRAPH_SUCCESS) {
        OPS_LOG_W(kPassName.c_str(), "Failed to get x2 output desc.");
    }

    // adj attrs keep the same semantics for all three types (adj_x1/adj_x2 for BatchMatMul(V2),
    // transpose_x1/transpose_x2 for MatMulV2); the ES API handles the naming internally.
    if (opType == kBatchMatMul) {
        auto out = es::BatchMatMul(x1, x2, adjX1, adjX2);
        SetMatMulOutputDesc(out, x1Desc, x2Desc, adjX1, adjX2);
        return out;
    }
    if (opType == kBatchMatMulV2) {
        auto out = es::BatchMatMulV2(x1, x2, nullptr, nullptr, adjX1, adjX2, 0);
        SetMatMulOutputDesc(out, x1Desc, x2Desc, adjX1, adjX2);
        return out;
    }
    auto out = es::MatMulV2(x1, x2, nullptr, nullptr, adjX1, adjX2, 0);
    SetMatMulOutputDesc(out, x1Desc, x2Desc, adjX1, adjX2);
    return out;
}

// Create a GatherShapes node via the ES API generated from the proto in op_nn_proto_extend.h;
// dtype is passed explicitly (DT_INT64) because the ES wrapper defaults to DT_INT32
GNode CreateGatherShapesNode(const std::vector<EsTensorHolder>& inputs, const std::vector<std::vector<int64_t>>& axes)
{
    auto out = es::GatherShapes(inputs, axes, static_cast<int64_t>(ge::DT_INT64));
    return *out.GetProducer();
}

struct InputInfo {
    bool valid = true; // false when any input desc is unavailable (CollectInputInfo failure)
    std::vector<Shape> shapes;
    std::vector<DataType> dtypes;
    std::vector<Format> formats;
    std::vector<bool> isDynamic;
    std::vector<std::vector<int64_t>> dims;
};

InputInfo CollectInputInfo(const std::vector<SubgraphInput>& subgraphInputs)
{
    InputInfo info;
    for (const auto& subgraphInput : subgraphInputs) {
        auto matchNode = subgraphInput.GetAllInputs().at(0);
        TensorDesc tensorDesc;
        if (matchNode.node.GetInputDesc(matchNode.index, tensorDesc) != GRAPH_SUCCESS) {
            OPS_LOG_W(kPassName.c_str(), "Failed to get input desc %zu.", matchNode.index);
            info.valid = false;
            return info;
        }
        info.shapes.emplace_back(tensorDesc.GetShape());
        info.dtypes.emplace_back(tensorDesc.GetDataType());
        info.formats.emplace_back(tensorDesc.GetFormat());
        info.isDynamic.emplace_back(IsUnknownShape(tensorDesc.GetShape()));
        info.dims.emplace_back(tensorDesc.GetShape().GetDims());
    }
    return info;
}

bool IsGNodeValid(const GNode& node)
{
    AscendString type;
    if (node.GetType(type) != GRAPH_SUCCESS) {
        return false;
    }
    return type.GetLength() > 0;
}

bool IsGNodeType(const GNode& node, const std::string& expectedType)
{
    AscendString type;
    if (node.GetType(type) != GRAPH_SUCCESS) {
        return false;
    }
    return type.GetString() == expectedType;
}

} // namespace

// Dispatch table for the 28 specific dynamic equations (same as legacy dynamicShapeProcs_).
// Also used by MeetRequirements to decide whether the dynamic fuzz fallback path
// (SplitDynamicFuzzScene) would be taken, where the legacy broadcast consistency check applies.
std::unordered_map<std::string, EinsumPass::ProcFunc> EinsumPass::dynamicShapeProcs_ = {
    {"abc,cde->abde", &EinsumPass::HandleDynamicABCxCDE2ABDE},
    {"abcd,aecd->aceb", &EinsumPass::HandleABCDxAECD2ACEB},
    {"abcd,adbe->acbe", &EinsumPass::HandleABCDxADBE2ACBE},
    {"abcd,cde->abe", &EinsumPass::HandleABCDxCDE2ABE},
    {"abc,cd->abd", &EinsumPass::HandleABCxCD2ABD},
    {"abc,dc->abd", &EinsumPass::HandleABCxDC2ABD},
    {"abc,abd->dc", &EinsumPass::HandleABCxABD2DC},
    {"abc,dec->abde", &EinsumPass::HandleDynamicABCxDEC2ABDE},
    {"abc,abde->dec", &EinsumPass::HandleDynamicABCxABDE2DEC},
    {"abcd,aecd->acbe", &EinsumPass::HandleABCDxAECD2ACBE},
    {"abcd,acbe->aecd", &EinsumPass::HandleABCDxACBE2AECD},
    {"abcd,ecd->abe", &EinsumPass::HandleABCDxECD2ABE},
    {"abcd,abe->ecd", &EinsumPass::HandleDynamicABCDxABE2ECD},
    {"abcd,acbe->adbe", &EinsumPass::HandleABCDxACBE2ADBE},
    {"abcd,abde->abce", &EinsumPass::HandleABCDxABDE2ABCE},
    {"abcd,abce->abde", &EinsumPass::HandleABCDxABCE2ABDE},
    {"abcd,aebd->aebc", &EinsumPass::HandleABCDxAEBD2AEBC},
    {"abcd,abce->acde", &EinsumPass::HandleABCDxABCE2ACDE},
    {"abc,abd->acd", &EinsumPass::HandleABCxABD2ACD},
    {"ab,cb->ac", &EinsumPass::HandleABxCB2AC},
    {"abc,acd->abd", &EinsumPass::HandleABCxACD2ABD},
    {"abc,adc->abd", &EinsumPass::HandleABCxADC2ABD},
    {"abcd,ced->abce", &EinsumPass::HandleABCDxCED2ABCE},
    {"abcd,ebcd->bcae", &EinsumPass::HandleABCDxEBCD2BCAE},
    {"abcd,dabe->cabe", &EinsumPass::HandleABCDxDABE2CABE},
    {"a,b->ab", &EinsumPass::HandleAxB2AB},
    {"abcd,aecd->eb", &EinsumPass::HandleABCDxAECD2EB},
    {"abcd,eb->aecd", &EinsumPass::HandleDynamicABCDxEB2AECD},
};

void EinsumPass::ResetState()
{
    // Note: supportL12btBf16_ is NOT reset here - it is a platform-level constant queried
    // once per fusion in MeetRequirements (which the framework always runs before Replacement).
    swapBmmInputs_ = false;
    mergeFreeLabels_ = true;
    castSeq_ = 1;
    unsqueezeSeq_ = 1;
    transposeSeq_ = 1;
    reduceSeq_ = 1;
    reshapeSeq_ = 1;
    batchmatmulSeq_ = 1;
    dimTypesMap_.clear();
    inputFreeContractOrders_.clear();
    outputLabelInfo_.clear();
    inputLabelInfos_.clear();
    builder_ = nullptr;
    graphPtr_ = nullptr;
    subgraphInputHolders_.clear();
    bmmInputNodes_.clear();
    batchmatmulNode_ = GNode();
    bmmOutputNodes_.clear();
    gathershapeNode_ = GNode();
    forcedInputDims_.clear();
}

// ============================================================================
// Patterns
// ============================================================================

// Build one Einsum match pattern: twoInput=true for 2-input Einsum, false for 1-input.
static PatternUniqPtr BuildEinsumPattern(bool twoInput)
{
    std::string builderName = twoInput ? kPassName : (kPassName + "_1input");
    auto graphBuilder = es::EsGraphBuilder(builderName.c_str());
    auto input0 = graphBuilder.CreateInput(0);
    auto input1 = twoInput ? std::make_optional(graphBuilder.CreateInput(1)) : std::nullopt;
    ge::Graph* graphPtr = graphBuilder.GetCGraphBuilder()->GetGraph();
    es::CompliantNodeBuilder nodeBuilder(graphPtr);
    nodeBuilder.OpType(kFusedOpType.c_str());
    if (twoInput) {
        nodeBuilder.IrDefInputs({{"x1", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                                 {"x2", es::CompliantNodeBuilder::kEsIrInputRequired, ""}});
    } else {
        nodeBuilder.IrDefInputs({{"x1", es::CompliantNodeBuilder::kEsIrInputRequired, ""}});
    }
    GNode einsumNode = nodeBuilder.IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                           .IrDefAttrs({
                               {"equation", es::CompliantNodeBuilder::kEsAttrRequired, "String",
                                es::CreateFrom(ge::AscendString(""))},
                               {"N", es::CompliantNodeBuilder::kEsAttrRequired, "Int",
                                es::CreateFrom(static_cast<int64_t>(0))},
                           })
                           .Build();
    FUSION_PASS_CHECK(AddEdgeAndUpdatePeerDesc(*graphPtr, *input0.GetProducer(), 0, einsumNode, 0) != GRAPH_SUCCESS,
                      OPS_LOG_W(kPassName.c_str(), "AddEdgeAndUpdatePeerDesc failed."), );
    if (input1.has_value()) {
        FUSION_PASS_CHECK(
            AddEdgeAndUpdatePeerDesc(*graphPtr, *input1->GetProducer(), 0, einsumNode, 1) != GRAPH_SUCCESS,
            OPS_LOG_W(kPassName.c_str(), "AddEdgeAndUpdatePeerDesc failed."), );
    }
    auto output = es::EsTensorHolder(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(einsumNode, 0));
    auto graph = graphBuilder.BuildAndReset({output});
    auto pattern = std::make_unique<Pattern>(std::move(*graph));
    pattern->CaptureTensor({einsumNode, 0});
    return pattern;
}

std::vector<PatternUniqPtr> EinsumPass::Patterns()
{
    OPS_LOG_D(kPassName.c_str(), "Enter Patterns for EinsumPass");
    std::vector<PatternUniqPtr> patternGraphs;
    patternGraphs.emplace_back(BuildEinsumPattern(true));
    patternGraphs.emplace_back(BuildEinsumPattern(false));
    return patternGraphs;
}

// ============================================================================
// MeetRequirements
// ============================================================================

// ============================================================================
// MeetRequirements helpers (split from the original oversized function)
// ============================================================================

// Check inflated output (repeated labels in output equation are not supported) and
// output label count consistency with output dim num
static bool CheckOutputLabelCount(const GNode& matchedNode, const std::string& equation, size_t arrowPos)
{
    std::string rhs = equation.substr(arrowPos + kArrowLen);
    std::set<char> outLabels;
    for (char label : rhs) {
        if (!isalpha(label)) {
            continue;
        }
        FUSION_PASS_CHECK(
            outLabels.count(label) > 0,
            OPS_LOG_D(kPassName.c_str(), "Inflated output not supported, equation: %s.", equation.c_str()),
            return false;);
        outLabels.insert(label);
    }
    TensorDesc outputDesc;
    if (matchedNode.GetOutputDesc(0, outputDesc) == GRAPH_SUCCESS) {
        size_t outDimNum = outputDesc.GetShape().GetDimNum();
        bool outHasEllipsis = rhs.find(kBroadCastLabel) != std::string::npos;
        if (outHasEllipsis) {
            FUSION_PASS_CHECK(outLabels.size() > outDimNum,
                              OPS_LOG_D(kPassName.c_str(), "Output label count %zu > output dim num %zu (ellipsis).",
                                        outLabels.size(), outDimNum),
                              return false;);
        } else if (outLabels.size() != outDimNum) {
            OPS_LOG_D(kPassName.c_str(), "Output label count %zu != output dim num %zu.", outLabels.size(), outDimNum);
            return false;
        }
    }
    return true;
}

bool EinsumPass::CheckEquationFormat(const GNode& matchedNode, std::string& equation,
                                     std::vector<std::string>& inEquations)
{
    FUSION_PASS_CHECK(equation.find("->") == std::string::npos,
                      OPS_LOG_D(kPassName.c_str(), "Equation[%s] has no arrow.", equation.c_str()), return false;);
    auto arrowPos = equation.find("->");
    std::string lhs = equation.substr(0, arrowPos);
    std::string current;
    for (char c : lhs) {
        if (c == ',') {
            inEquations.push_back(current);
            current.clear();
        } else {
            current += c;
        }
    }
    if (!current.empty()) {
        inEquations.push_back(current);
    }
    FUSION_PASS_CHECK(
        inEquations.size() > kMaxInputNum,
        OPS_LOG_D(kPassName.c_str(), "Equation has %zu inputs, max %zu.", inEquations.size(), kMaxInputNum),
        return false;);
    for (auto& inEq : inEquations) {
        for (size_t pos = 0; pos < inEq.size(); ++pos) {
            char label = inEq[pos];
            if (label == '.') {
                pos += (kBroadCastLabelLen - 1);
                continue;
            }
            FUSION_PASS_CHECK(inEq.find(label, pos + 1) != std::string::npos,
                              OPS_LOG_D(kPassName.c_str(), "Stride input not supported, equation: %s.", inEq.c_str()),
                              return false;);
        }
    }
    return CheckOutputLabelCount(matchedNode, equation, arrowPos);
}

bool EinsumPass::CheckLabelConsistency(const GNode& matchedNode, const std::string& equation,
                                       const std::vector<std::string>& inEquations, std::vector<DataType>& inputDtypes,
                                       std::vector<std::vector<int64_t>>& inputDims)
{
    for (size_t idx = 0; idx < matchedNode.GetInputsSize(); ++idx) {
        TensorDesc inputDesc;
        FUSION_PASS_CHECK(matchedNode.GetInputDesc(idx, inputDesc) != GRAPH_SUCCESS,
                          OPS_LOG_D(kPassName.c_str(), "Failed to get input desc %zu.", idx), return false;);
        inputDtypes.emplace_back(inputDesc.GetDataType());
        inputDims.emplace_back(inputDesc.GetShape().GetDims());
        auto dimNum = inputDesc.GetShape().GetDimNum();
        if (idx < inEquations.size()) {
            const auto& inEq = inEquations[idx];
            size_t labelCnt = 0;
            bool hasEllipsis = false;
            for (size_t p = 0; p < inEq.size(); ++p) {
                if (inEq[p] == '.') {
                    hasEllipsis = true;
                    p += (kBroadCastLabelLen - 1);
                } else if (isalpha(inEq[p])) {
                    labelCnt++;
                }
            }
            if (hasEllipsis) {
                FUSION_PASS_CHECK(labelCnt > dimNum,
                                  OPS_LOG_D(kPassName.c_str(), "Input %zu label count %zu > dim num %zu (ellipsis).",
                                            idx, labelCnt, dimNum),
                                  return false;);
            } else if (labelCnt != dimNum) {
                OPS_LOG_D(kPassName.c_str(), "Input %zu label count %zu != dim num %zu.", idx, labelCnt, dimNum);
                return false;
            }
        }
        int64_t product = 1;
        for (auto dim : inputDesc.GetShape().GetDims()) {
            if (dim > 0) {
                FUSION_PASS_CHECK(product > (INT64_MAX / dim),
                                  OPS_LOG_D(kPassName.c_str(), "Input shape product overflow."), return false;);
                product *= dim;
            }
        }
    }
    // NOTE: input dtype consistency is intentionally NOT checked here (aligned with the legacy
    // pass): the static fuzz path and the 28 pattern handlers let mixed-dtype inputs through,
    // while the dynamic fuzz path rejects raw input mismatch in ReduceAndSumPairForDynamic and
    // casts intermediate-result mismatches (fp32 promotion from ReduceSumInput) in SumproductPair.
    return true;
}

// Expand one input equation to a per-dim label sequence ("..." expands to actual dim num)
static std::string ExpandInputEquation(const std::string& inEq, size_t dimNum)
{
    size_t labelCnt = 0;
    bool hasEllipsis = false;
    for (size_t p = 0; p < inEq.size(); ++p) {
        if (inEq[p] == '.') {
            hasEllipsis = true;
            p += (kBroadCastLabelLen - 1);
        } else if (isalpha(inEq[p])) {
            labelCnt++;
        }
    }
    size_t ellNum = hasEllipsis ? (dimNum - labelCnt) : 0;
    std::string expanded;
    for (size_t p = 0; p < inEq.size(); ++p) {
        if (inEq[p] == '.') {
            expanded.append(ellNum, '.');
            p += (kBroadCastLabelLen - 1);
        } else if (isalpha(inEq[p])) {
            expanded += inEq[p];
        }
    }
    return expanded;
}

static bool IsDimBroadcastable(int64_t d0, int64_t d1) { return d0 == d1 || d0 == 1 || d1 == 1 || d0 < 0 || d1 < 0; }

// Check that labels appearing in both inputs are broadcastable (split from CheckBroadcastConsistency)
static bool CheckExplicitLabelBroadcast(const std::string& seq0, const std::string& seq1,
                                        const std::vector<std::vector<int64_t>>& checkDims, const std::string& equation)
{
    for (size_t p0 = 0; p0 < seq0.size(); ++p0) {
        char label = seq0[p0];
        if (label == '.') {
            continue;
        }
        size_t p1 = seq1.find(label);
        if (p1 == std::string::npos || p1 >= checkDims[1].size() || p0 >= checkDims[0].size()) {
            continue;
        }
        FUSION_PASS_CHECK(
            !IsDimBroadcastable(checkDims[0][p0], checkDims[1][p1]),
            OPS_LOG_D(kPassName.c_str(), "Label %c not broadcastable: %ld vs %ld, equation: %s.", label,
                      static_cast<long>(checkDims[0][p0]), static_cast<long>(checkDims[1][p1]), equation.c_str()),
            return false;);
    }
    return true;
}

// Check that ellipsis dims, right aligned across the two inputs, are broadcastable
// (split from CheckBroadcastConsistency)
static bool CheckEllipsisBroadcast(const std::string& seq0, const std::string& seq1,
                                   const std::vector<std::vector<int64_t>>& checkDims, const std::string& equation)
{
    std::vector<size_t> ellPos0;
    std::vector<size_t> ellPos1;
    for (size_t p = 0; p < seq0.size(); ++p) {
        if (seq0[p] == '.') {
            ellPos0.emplace_back(p);
        }
    }
    for (size_t p = 0; p < seq1.size(); ++p) {
        if (seq1[p] == '.') {
            ellPos1.emplace_back(p);
        }
    }
    size_t ellNum0 = ellPos0.size();
    size_t ellNum1 = ellPos1.size();
    for (size_t k = 0; k < std::max(ellNum0, ellNum1); ++k) {
        int64_t d0 = (k < ellNum0 && ellPos0[ellNum0 - 1 - k] < checkDims[0].size()) ?
                         checkDims[0][ellPos0[ellNum0 - 1 - k]] :
                         1;
        int64_t d1 = (k < ellNum1 && ellPos1[ellNum1 - 1 - k] < checkDims[1].size()) ?
                         checkDims[1][ellPos1[ellNum1 - 1 - k]] :
                         1;
        FUSION_PASS_CHECK(!IsDimBroadcastable(d0, d1),
                          OPS_LOG_D(kPassName.c_str(), "Ellipsis dim %zu not broadcastable: %ld vs %ld, equation: %s.",
                                    k, static_cast<long>(d0), static_cast<long>(d1), equation.c_str()),
                          return false;);
    }
    return true;
}

bool EinsumPass::CheckBroadcastConsistency(const std::string& equation, const std::vector<std::string>& inEquations,
                                           const std::vector<DataType>& inputDtypes,
                                           const std::vector<std::vector<int64_t>>& inputDims)
{
    // Only apply when the dynamic fuzz path would be taken (dynamic shape + equation not in 28 patterns)
    bool isDynamicShape = false;
    for (const auto& dims : inputDims) {
        if (IsUnknownShape(Shape(dims))) {
            isDynamicShape = true;
            break;
        }
    }
    if (!isDynamicShape || inEquations.size() != kMaxInputNum || inputDims.size() != kMaxInputNum) {
        return true;
    }
    std::string normalizedEquation = equation;
    EquationNormalization(normalizedEquation);
    if (dynamicShapeProcs_.find(normalizedEquation) != dynamicShapeProcs_.end()) {
        return true;
    }
    // Int32 inputs are forced to dynamic shape in the dynamic fuzz decomposition
    std::vector<std::vector<int64_t>> checkDims = inputDims;
    for (size_t idx = 0; idx < checkDims.size(); ++idx) {
        if (inputDtypes[idx] == ge::DT_INT32) {
            std::fill(checkDims[idx].begin(), checkDims[idx].end(), -1);
        }
    }
    // Expand each input equation to a per-dim label sequence ("..." expands to actual dim num)
    auto seq0 = ExpandInputEquation(inEquations[0], checkDims[0].size());
    auto seq1 = ExpandInputEquation(inEquations[1], checkDims[1].size());
    return CheckExplicitLabelBroadcast(seq0, seq1, checkDims, equation) &&
           CheckEllipsisBroadcast(seq0, seq1, checkDims, equation);
}

bool EinsumPass::MeetRequirements(const std::unique_ptr<MatchResult>& matchResult)
{
    OPS_LOG_D(kPassName.c_str(), "Begin to check EinsumPass requirements.");

    FUSION_PASS_CHECK(GetGeCompilerVersionNum() < kGeCompilerVersion900,
                      OPS_LOG_D(kPassName.c_str(), "GE runtime < 9.0.0, skip fusion."), return false);

    NodeIo nodeIo;
    FUSION_PASS_CHECK(matchResult->GetCapturedTensor(kCaptureTensorIdx, nodeIo) != SUCCESS,
                      OPS_LOG_E(kPassName.c_str(), "Failed to get captured tensor."), return false);
    GNode matchedNode = nodeIo.node;
    // Check input count (max 2)
    FUSION_PASS_CHECK(
        matchedNode.GetInputsSize() > kMaxInputNum,
        OPS_LOG_D(kPassName.c_str(), "Einsum input count %zu exceeds max 2.", matchedNode.GetInputsSize()),
        return false;);
    // Check platform info
    PlatformInfo platformInfo;
    OptionalInfo optionalInfo;
    FUSION_PASS_CHECK(
        PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platformInfo, optionalInfo) != SUCCESS,
        OPS_LOG_D(kPassName.c_str(), "Can't get platformInfo."), return false);
    supportL12btBf16_ = IsSupportL12BtBf16(platformInfo);
    // Check equation attribute
    std::string equation;
    AscendString equationStr;
    FUSION_PASS_CHECK(matchedNode.GetAttr("equation", equationStr) != GRAPH_SUCCESS,
                      OPS_LOG_W(kPassName.c_str(), "Get attr equation failed."), return false;);
    equation = equationStr.GetString();
    FUSION_PASS_CHECK(equation.empty(), OPS_LOG_W(kPassName.c_str(), "Equation is empty."), return false;);
    // Check equation format, stride, inflated output, output label count
    std::vector<std::string> inEquations;
    if (!CheckEquationFormat(matchedNode, equation, inEquations)) {
        return false;
    }
    // Check input label count, overflow, dtype consistency
    std::vector<DataType> inputDtypes;
    std::vector<std::vector<int64_t>> inputDims;
    if (!CheckLabelConsistency(matchedNode, equation, inEquations, inputDtypes, inputDims)) {
        return false;
    }
    // Check broadcast consistency (dynamic fuzz path only)
    if (!CheckBroadcastConsistency(equation, inEquations, inputDtypes, inputDims)) {
        return false;
    }
    // Aligned with the legacy dynamic fuzz path: mixed-dtype raw inputs are rejected on the
    // generic dynamic fuzz route only (2 inputs + dynamic shape + equation outside the
    // 28-pattern table, cf. the legacy dtype check placed before the reduce/sum pairing). The
    // static fuzz path and the 28 pattern handlers leave mixed dtypes to the downstream graph.
    if (inEquations.size() == kMaxInputNum) {
        bool dynamicShape = false;
        for (const auto& dims : inputDims) {
            if (IsUnknownShape(Shape(dims))) {
                dynamicShape = true;
                break;
            }
        }
        if (dynamicShape) {
            std::string normalizedEquation = equation;
            EquationNormalization(normalizedEquation);
            if (dynamicShapeProcs_.find(normalizedEquation) == dynamicShapeProcs_.end() &&
                inputDtypes[0] != inputDtypes[1]) {
                OPS_LOG_D(kPassName.c_str(), "Generic dynamic fuzz route requires same input dtypes, equation: %s.",
                          equation.c_str());
                return false;
            }
        }
    }
    return true;
}

// ============================================================================
// Replacement - dispatch logic
// ============================================================================

GraphUniqPtr EinsumPass::Replacement(const std::unique_ptr<MatchResult>& matchResult)
{
    OPS_LOG_D(kPassName.c_str(), "Enter Replacement for EinsumPass");
    ResetState();
    // supportL12btBf16_ has already been queried in MeetRequirements, which the framework
    // guarantees to run (and return true) before Replacement.

    NodeIo nodeIo;
    FUSION_PASS_CHECK(matchResult->GetCapturedTensor(kCaptureTensorIdx, nodeIo) != SUCCESS,
                      OPS_LOG_E(kPassName.c_str(), "Failed to get captured tensor in Replacement."), return nullptr);
    GNode matchedNode = nodeIo.node;

    // Get equation
    AscendString equationStr;
    FUSION_PASS_CHECK(matchedNode.GetAttr("equation", equationStr) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName.c_str(), "Get attr equation failed."), return nullptr);
    std::string equation = equationStr.GetString();

    // Get subgraph inputs
    std::vector<SubgraphInput> subgraphInputs;
    matchResult->ToSubgraphBoundary()->GetAllInputs(subgraphInputs);
    FUSION_PASS_CHECK(subgraphInputs.empty(), OPS_LOG_E(kPassName.c_str(), "Subgraph inputs is empty."),
                      return nullptr;);

    // Detect dynamic shape
    auto inputInfo = CollectInputInfo(subgraphInputs);
    FUSION_PASS_CHECK(!inputInfo.valid, OPS_LOG_E(kPassName.c_str(), "CollectInputInfo failed."), return nullptr);
    bool isDynamicShape = false;
    for (auto dynamic : inputInfo.isDynamic) {
        isDynamicShape = isDynamicShape || dynamic;
    }

    OPS_LOG_I(kPassName.c_str(), "EinsumPass equation=%s, dynamic=%d, inputs=%zu", equation.c_str(), isDynamicShape,
              subgraphInputs.size());

    if (isDynamicShape) {
        std::string normalizedEquation = equation;
        EquationNormalization(normalizedEquation);

        // Dispatch to specific handler via the lookup table (same as legacy dynamicShapeProcs_)
        auto procIter = dynamicShapeProcs_.find(normalizedEquation);
        if (procIter != dynamicShapeProcs_.end()) {
            return (this->*(procIter->second))(subgraphInputs, matchedNode);
        }
        return SplitDynamicFuzzScene(equation, subgraphInputs, matchedNode);
    } else {
        auto result = SplitOpInFuzzScene(equation, subgraphInputs, matchedNode);
        if (result != nullptr) {
            return result;
        }
        OPS_LOG_W(kPassName.c_str(), "equation[%s] is not support now.", equation.c_str());
        return nullptr;
    }
}

// ============================================================================
// BuildBatchMatmulReplacement - common helper for simple BatchMatMul patterns
// Patterns: 005, 006, 015, 016, 019, 020, 021, 022
// ============================================================================

GraphUniqPtr EinsumPass::BuildBatchMatmulReplacement(bool adjX1, bool adjX2,
                                                     const std::vector<SubgraphInput>& subgraphInputs,
                                                     const GNode& matchedNode)
{
    auto inputInfo = CollectInputInfo(subgraphInputs);
    FUSION_PASS_CHECK(!inputInfo.valid, OPS_LOG_E(kPassName.c_str(), "CollectInputInfo failed."), return nullptr);
    auto builder = es::EsGraphBuilder("replacement");
    auto input0 = builder.CreateInput(0, "x1", inputInfo.dtypes[0], inputInfo.formats[0], inputInfo.dims[0]);
    auto input1 = builder.CreateInput(1, "x2", inputInfo.dtypes[1], inputInfo.formats[1], inputInfo.dims[1]);

    auto bmmOutput = CreateMatMulNode(builder, kBatchMatMul, input0, input1, adjX1, adjX2);
    GNode bmmNode = *bmmOutput.GetProducer();

    // Copy output desc from matched node
    TensorDesc outputDesc;
    FUSION_PASS_CHECK(matchedNode.GetOutputDesc(0, outputDesc) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName.c_str(), "Failed to get output desc."), return nullptr);
    bmmNode.UpdateOutputDesc(0, outputDesc);

    FUSION_PASS_CHECK(!CopyOtherAttrs(matchedNode, bmmNode, kPassName),
                      OPS_LOG_E(kPassName.c_str(), "Copy other attrs failed."), return nullptr);

    auto replaceGraph = builder.BuildAndReset({bmmOutput});
    FUSION_PASS_CHECK(InferShape(replaceGraph, subgraphInputs) != SUCCESS,
                      OPS_LOG_E(kPassName.c_str(), "InferShape for replacement failed."), return nullptr;);
    return replaceGraph;
}

// ============================================================================
// Handle* functions - specific equation patterns
// ============================================================================

// 002: transpose+transpose+batchmatmul(swap input) - abcd,aecd->aceb
GraphUniqPtr EinsumPass::HandleABCDxAECD2ACEB(const std::vector<SubgraphInput>& subgraphInputs,
                                              const GNode& matchedNode)
{
    OPS_LOG_D(kPassName.c_str(), "HandleABCDxAECD2ACEB");
    auto inputInfo = CollectInputInfo(subgraphInputs);
    FUSION_PASS_CHECK(!inputInfo.valid, OPS_LOG_E(kPassName.c_str(), "CollectInputInfo failed."), return nullptr);
    auto& x0Dims = inputInfo.dims[0];
    auto& x1Dims = inputInfo.dims[1];
    FUSION_PASS_CHECK(x0Dims.size() != kDim4D || x1Dims.size() != kDim4D,
                      OPS_LOG_W(kPassName.c_str(), "Input dims size of x0 and x1 must be 4."), return nullptr);

    auto builder = es::EsGraphBuilder("replacement");
    ge::Graph* graphPtr = builder.GetCGraphBuilder()->GetGraph();
    auto input0 = builder.CreateInput(0, "x1", inputInfo.dtypes[0], inputInfo.formats[0], x0Dims);
    auto input1 = builder.CreateInput(1, "x2", inputInfo.dtypes[1], inputInfo.formats[1], x1Dims);

    // Transpose input1: perm {0, 2, 1, 3}
    std::vector<int32_t> perm{0, 2, 1, 3};
    GNode trans2Node = CreateTransposeNode(builder, "transpose_2", input1, inputInfo.isDynamic[1], supportL12btBf16_,
                                           perm);
    SetTransposeOutputDesc(trans2Node, perm);

    // Transpose input0: perm {0, 2, 1, 3}
    GNode trans1Node = CreateTransposeNode(builder, "transpose_1", input0, inputInfo.isDynamic[0], supportL12btBf16_,
                                           perm);
    SetTransposeOutputDesc(trans1Node, perm);

    // BatchMatMul(trans2, trans1) with adj_x1=false, adj_x2=true (swap input)
    auto trans2Output = GetTensorHolder(builder, trans2Node, 0);
    auto trans1Output = GetTensorHolder(builder, trans1Node, 0);
    auto bmmOutput = CreateMatMulNode(builder, kBatchMatMul, trans2Output, trans1Output, false, true);
    GNode bmmNode = *bmmOutput.GetProducer();

    TensorDesc outputDesc;
    FUSION_PASS_CHECK(matchedNode.GetOutputDesc(0, outputDesc) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName.c_str(), "Failed to get output desc."), return nullptr);
    bmmNode.UpdateOutputDesc(0, outputDesc);
    if (!CopyOtherAttrs(matchedNode, bmmNode, kPassName)) {
        return nullptr;
    }

    auto replaceGraph = builder.BuildAndReset({bmmOutput});
    if (InferShape(replaceGraph, subgraphInputs) != SUCCESS) {
        return nullptr;
    }
    return replaceGraph;
}

// 003: transpose+batchmatmul+transpose - abcd,adbe->acbe
GraphUniqPtr EinsumPass::HandleABCDxADBE2ACBE(const std::vector<SubgraphInput>& subgraphInputs,
                                              const GNode& matchedNode)
{
    OPS_LOG_D(kPassName.c_str(), "HandleABCDxADBE2ACBE");
    auto inputInfo = CollectInputInfo(subgraphInputs);
    FUSION_PASS_CHECK(!inputInfo.valid, OPS_LOG_E(kPassName.c_str(), "CollectInputInfo failed."), return nullptr);
    auto& x0Dims = inputInfo.dims[0];
    auto& x1Dims = inputInfo.dims[1];
    FUSION_PASS_CHECK(x0Dims.size() != kDim4D || x1Dims.size() != kDim4D,
                      OPS_LOG_W(kPassName.c_str(), "Input dims size check failed."), return nullptr);

    auto builder = es::EsGraphBuilder("replacement");
    ge::Graph* graphPtr = builder.GetCGraphBuilder()->GetGraph();
    auto input0 = builder.CreateInput(0, "x1", inputInfo.dtypes[0], inputInfo.formats[0], x0Dims);
    auto input1 = builder.CreateInput(1, "x2", inputInfo.dtypes[1], inputInfo.formats[1], x1Dims);

    // Transpose input1: perm {0, 2, 1, 3} -> abde
    std::vector<int32_t> perm{0, 2, 1, 3};
    GNode trans1Node = CreateTransposeNode(builder, "transpose_1", input1, inputInfo.isDynamic[1], supportL12btBf16_,
                                           perm);
    SetTransposeOutputDesc(trans1Node, perm);

    // BatchMatMul(input0, trans1, adj_x1=false, adj_x2=false)
    auto trans1Output = GetTensorHolder(builder, trans1Node, 0);
    auto bmmOutput = CreateMatMulNode(builder, kBatchMatMul, input0, trans1Output, false, false);
    GNode bmmNode = *bmmOutput.GetProducer();

    // Transpose output: perm {0, 2, 1, 3} -> acbe
    GNode trans2Node = CreateTransposeNode(builder, "transpose_2", GetTensorHolder(builder, bmmNode, 0),
                                           inputInfo.isDynamic[0], supportL12btBf16_, perm);
    SetTransposeOutputDesc(trans2Node, perm);

    TensorDesc outputDesc;
    FUSION_PASS_CHECK(matchedNode.GetOutputDesc(0, outputDesc) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName.c_str(), "Failed to get output desc."), return nullptr);
    trans2Node.UpdateOutputDesc(0, outputDesc);
    if (!CopyOtherAttrs(matchedNode, bmmNode, kPassName)) {
        return nullptr;
    }

    auto trans2Output = GetTensorHolder(builder, trans2Node, 0);
    auto replaceGraph = builder.BuildAndReset({trans2Output});
    if (InferShape(replaceGraph, subgraphInputs) != SUCCESS) {
        return nullptr;
    }
    return replaceGraph;
}

// 004: reshape+reshape+matmul+reshape->reshape+batchmatmul - abcd,cde->abe
GraphUniqPtr EinsumPass::HandleABCDxCDE2ABE(const std::vector<SubgraphInput>& subgraphInputs, const GNode& matchedNode)
{
    OPS_LOG_D(kPassName.c_str(), "HandleABCDxCDE2ABE");
    auto inputInfo = CollectInputInfo(subgraphInputs);
    FUSION_PASS_CHECK(!inputInfo.valid, OPS_LOG_E(kPassName.c_str(), "CollectInputInfo failed."), return nullptr);
    auto& x0Dims = inputInfo.dims[0];
    auto& x1Dims = inputInfo.dims[1];
    FUSION_PASS_CHECK(x0Dims.size() != kDim4D || x1Dims.size() != kDim3D,
                      OPS_LOG_W(kPassName.c_str(), "Input dims size check failed."), return nullptr);

    auto builder = es::EsGraphBuilder("replacement");
    ge::Graph* graphPtr = builder.GetCGraphBuilder()->GetGraph();
    auto input0 = builder.CreateInput(0, "x1", inputInfo.dtypes[0], inputInfo.formats[0], x0Dims);
    auto input1 = builder.CreateInput(1, "x2", inputInfo.dtypes[1], inputInfo.formats[1], x1Dims);

    // Reshape input0: {a, b, c*d}
    std::vector<int64_t> reshape1Dims{x0Dims[0], x0Dims[1], GetDimMulValue(x0Dims[2], x0Dims[3])};
    GNode reshape1Node = CreateReshapeNode(builder, "reshape_1", input0, inputInfo.isDynamic[0], reshape1Dims, 2, 3);
    SetReshapeOutputDesc(reshape1Node, reshape1Dims);

    // Reshape input1: {c*d, e}
    std::vector<int64_t> reshape2Dims{GetDimMulValue(x1Dims[0], x1Dims[1]), x1Dims[2]};
    GNode reshape2Node = CreateReshapeNode(builder, "reshape_2", input1, inputInfo.isDynamic[1], reshape2Dims, 0, 1);
    SetReshapeOutputDesc(reshape2Node, reshape2Dims);

    // BatchMatMul(reshape1, reshape2, adj_x1=false, adj_x2=false)
    auto reshape1Output = GetTensorHolder(builder, reshape1Node, 0);
    auto reshape2Output = GetTensorHolder(builder, reshape2Node, 0);
    auto bmmOutput = CreateMatMulNode(builder, kBatchMatMul, reshape1Output, reshape2Output, false, false);
    GNode bmmNode = *bmmOutput.GetProducer();

    TensorDesc outputDesc;
    FUSION_PASS_CHECK(matchedNode.GetOutputDesc(0, outputDesc) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName.c_str(), "Failed to get output desc."), return nullptr);
    bmmNode.UpdateOutputDesc(0, outputDesc);
    if (!CopyOtherAttrs(matchedNode, bmmNode, kPassName)) {
        return nullptr;
    }

    auto replaceGraph = builder.BuildAndReset({bmmOutput});
    if (InferShape(replaceGraph, subgraphInputs) != SUCCESS) {
        return nullptr;
    }
    return replaceGraph;
}

// 005: batchmatmul - abc,cd->abd
GraphUniqPtr EinsumPass::HandleABCxCD2ABD(const std::vector<SubgraphInput>& subgraphInputs, const GNode& matchedNode)
{
    OPS_LOG_D(kPassName.c_str(), "HandleABCxCD2ABD");
    return BuildBatchMatmulReplacement(false, false, subgraphInputs, matchedNode);
}

// 006: batchmatmul - abc,dc->abd
GraphUniqPtr EinsumPass::HandleABCxDC2ABD(const std::vector<SubgraphInput>& subgraphInputs, const GNode& matchedNode)
{
    OPS_LOG_D(kPassName.c_str(), "HandleABCxDC2ABD");
    return BuildBatchMatmulReplacement(false, true, subgraphInputs, matchedNode);
}

// 007: reshape+reshape+matmul(swap input) - abc,abd->dc
GraphUniqPtr EinsumPass::HandleABCxABD2DC(const std::vector<SubgraphInput>& subgraphInputs, const GNode& matchedNode)
{
    OPS_LOG_D(kPassName.c_str(), "HandleABCxABD2DC");
    auto inputInfo = CollectInputInfo(subgraphInputs);
    FUSION_PASS_CHECK(!inputInfo.valid, OPS_LOG_E(kPassName.c_str(), "CollectInputInfo failed."), return nullptr);
    auto& x0Dims = inputInfo.dims[0];
    auto& x1Dims = inputInfo.dims[1];
    FUSION_PASS_CHECK(x0Dims.size() != kDim3D || x1Dims.size() != kDim3D,
                      OPS_LOG_W(kPassName.c_str(), "Input dims size check failed."), return nullptr);

    auto builder = es::EsGraphBuilder("replacement");
    ge::Graph* graphPtr = builder.GetCGraphBuilder()->GetGraph();
    auto input0 = builder.CreateInput(0, "x1", inputInfo.dtypes[0], inputInfo.formats[0], x0Dims);
    auto input1 = builder.CreateInput(1, "x2", inputInfo.dtypes[1], inputInfo.formats[1], x1Dims);

    // Reshape input0: {a*b, c}
    std::vector<int64_t> reshape1Dims{GetDimMulValue(x0Dims[0], x0Dims[1]), x0Dims[2]};
    GNode reshape1Node = CreateReshapeNode(builder, "reshape_1", input0, inputInfo.isDynamic[0], reshape1Dims, 0, 1);
    SetReshapeOutputDesc(reshape1Node, reshape1Dims);

    // Reshape input1: {a*b, d}
    std::vector<int64_t> reshape2Dims{GetDimMulValue(x1Dims[0], x1Dims[1]), x1Dims[2]};
    GNode reshape2Node = CreateReshapeNode(builder, "reshape_2", input1, inputInfo.isDynamic[1], reshape2Dims, 0, 1);
    SetReshapeOutputDesc(reshape2Node, reshape2Dims);

    // MatMul(reshape2, reshape1, transpose_x1=true, transpose_x2=false) (swap input)
    auto reshape1Output = GetTensorHolder(builder, reshape1Node, 0);
    auto reshape2Output = GetTensorHolder(builder, reshape2Node, 0);
    auto matmulOutput = CreateMatMulNode(builder, kMatMul, reshape2Output, reshape1Output, true, false);
    GNode matmulNode = *matmulOutput.GetProducer();

    TensorDesc outputDesc;
    FUSION_PASS_CHECK(matchedNode.GetOutputDesc(0, outputDesc) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName.c_str(), "Failed to get output desc."), return nullptr);
    matmulNode.UpdateOutputDesc(0, outputDesc);
    if (!CopyOtherAttrs(matchedNode, matmulNode, kPassName)) {
        return nullptr;
    }

    auto replaceGraph = builder.BuildAndReset({matmulOutput});
    if (InferShape(replaceGraph, subgraphInputs) != SUCCESS) {
        return nullptr;
    }
    return replaceGraph;
}

// 001: reshape+reshape+matmul+reshape (dynamic) - abc,cde->abde
GraphUniqPtr EinsumPass::HandleDynamicABCxCDE2ABDE(const std::vector<SubgraphInput>& subgraphInputs,
                                                   const GNode& matchedNode)
{
    OPS_LOG_D(kPassName.c_str(), "HandleDynamicABCxCDE2ABDE");
    auto inputInfo = CollectInputInfo(subgraphInputs);
    FUSION_PASS_CHECK(!inputInfo.valid, OPS_LOG_E(kPassName.c_str(), "CollectInputInfo failed."), return nullptr);
    auto& x0Dims = inputInfo.dims[0];
    auto& x1Dims = inputInfo.dims[1];
    FUSION_PASS_CHECK(x0Dims.size() != kDim3D || x1Dims.size() != kDim3D,
                      OPS_LOG_W(kPassName.c_str(), "Input dims size check failed."), return nullptr);

    auto builder = es::EsGraphBuilder("replacement");
    ge::Graph* graphPtr = builder.GetCGraphBuilder()->GetGraph();
    auto input0 = builder.CreateInput(0, "x0", inputInfo.dtypes[0], inputInfo.formats[0], x0Dims);
    auto input1 = builder.CreateInput(1, "x1", inputInfo.dtypes[1], inputInfo.formats[1], x1Dims);

    // Reshape input0: {a*b, c}
    std::vector<int64_t> reshape1Dims{GetDimMulValue(x0Dims[0], x0Dims[1]), x0Dims[2]};
    GNode reshape1Node = CreateReshapeNode(builder, "reshape_1", input0, inputInfo.isDynamic[0], reshape1Dims, 0, 1);
    SetReshapeOutputDesc(reshape1Node, reshape1Dims);

    // Reshape input1: {c, d*e}
    std::vector<int64_t> reshape2Dims{x1Dims[0], GetDimMulValue(x1Dims[1], x1Dims[2])};
    GNode reshape2Node = CreateReshapeNode(builder, "reshape_2", input1, inputInfo.isDynamic[1], reshape2Dims, 1, 2);
    SetReshapeOutputDesc(reshape2Node, reshape2Dims);

    // MatMul(reshape1, reshape2)
    auto reshape1Output = GetTensorHolder(builder, reshape1Node, 0);
    auto reshape2Output = GetTensorHolder(builder, reshape2Node, 0);
    auto matmulOutput = CreateMatMulNode(builder, kMatMul, reshape1Output, reshape2Output, false, false);
    GNode matmulNode = *matmulOutput.GetProducer();

    // GatherShapes(input0, input1) for dynamic output shape
    std::vector<EsTensorHolder> gatherInputs{input0, input1};
    std::vector<std::vector<int64_t>> gatherAxes{{0, 0}, {0, 1}, {1, 1}, {1, 2}};
    GNode gatherNode = CreateGatherShapesNode(gatherInputs, gatherAxes);
    SetGatherShapesOutputDesc(gatherNode, gatherAxes.size());

    // Reshape output using GatherShapes result
    auto reshape3Output = es::Reshape(matmulOutput, GetTensorHolder(builder, gatherNode, 0));
    GNode reshape3Node = *reshape3Output.GetProducer();

    if (!CopyOtherAttrs(matchedNode, matmulNode, kPassName)) {
        return nullptr;
    }

    TensorDesc finalOutputDesc;
    FUSION_PASS_CHECK(matchedNode.GetOutputDesc(0, finalOutputDesc) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName.c_str(), "Failed to get output desc."), return nullptr);
    reshape3Node.UpdateOutputDesc(0, finalOutputDesc);
    auto replaceGraph = builder.BuildAndReset({reshape3Output});
    if (InferShape(replaceGraph, subgraphInputs) != SUCCESS) {
        return nullptr;
    }
    return replaceGraph;
}

// 008: reshape+reshape+matmul+reshape (dynamic) - abc,dec->abde
GraphUniqPtr EinsumPass::HandleDynamicABCxDEC2ABDE(const std::vector<SubgraphInput>& subgraphInputs,
                                                   const GNode& matchedNode)
{
    OPS_LOG_D(kPassName.c_str(), "HandleDynamicABCxDEC2ABDE");
    auto inputInfo = CollectInputInfo(subgraphInputs);
    FUSION_PASS_CHECK(!inputInfo.valid, OPS_LOG_E(kPassName.c_str(), "CollectInputInfo failed."), return nullptr);
    auto& x0Dims = inputInfo.dims[0];
    auto& x1Dims = inputInfo.dims[1];
    FUSION_PASS_CHECK(x0Dims.size() != kDim3D || x1Dims.size() != kDim3D,
                      OPS_LOG_W(kPassName.c_str(), "Input dims size check failed."), return nullptr);

    auto builder = es::EsGraphBuilder("replacement");
    ge::Graph* graphPtr = builder.GetCGraphBuilder()->GetGraph();
    auto input0 = builder.CreateInput(0, "x0", inputInfo.dtypes[0], inputInfo.formats[0], x0Dims);
    auto input1 = builder.CreateInput(1, "x1", inputInfo.dtypes[1], inputInfo.formats[1], x1Dims);

    // Reshape input0: {a*b, c}
    std::vector<int64_t> reshape1Dims{GetDimMulValue(x0Dims[0], x0Dims[1]), x0Dims[2]};
    GNode reshape1Node = CreateReshapeNode(builder, "reshape_1", input0, inputInfo.isDynamic[0], reshape1Dims, 0, 1);
    SetReshapeOutputDesc(reshape1Node, reshape1Dims);

    // Reshape input1: {d*e, c} (note: dec, so c is last)
    std::vector<int64_t> reshape2Dims{GetDimMulValue(x1Dims[0], x1Dims[1]), x1Dims[2]};
    GNode reshape2Node = CreateReshapeNode(builder, "reshape_2", input1, inputInfo.isDynamic[1], reshape2Dims, 0, 1);
    SetReshapeOutputDesc(reshape2Node, reshape2Dims);

    // MatMul(reshape1, reshape2, transpose_x1=false, transpose_x2=true)
    auto reshape1Output = GetTensorHolder(builder, reshape1Node, 0);
    auto reshape2Output = GetTensorHolder(builder, reshape2Node, 0);
    auto matmulOutput = CreateMatMulNode(builder, kMatMul, reshape1Output, reshape2Output, false, true);
    GNode matmulNode = *matmulOutput.GetProducer();

    // GatherShapes
    std::vector<EsTensorHolder> gatherInputs{input0, input1};
    std::vector<std::vector<int64_t>> gatherAxes{{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    GNode gatherNode = CreateGatherShapesNode(gatherInputs, gatherAxes);
    SetGatherShapesOutputDesc(gatherNode, gatherAxes.size());

    // Reshape output
    auto reshape3Output = es::Reshape(matmulOutput, GetTensorHolder(builder, gatherNode, 0));
    GNode reshape3Node = *reshape3Output.GetProducer();

    if (!CopyOtherAttrs(matchedNode, matmulNode, kPassName)) {
        return nullptr;
    }

    TensorDesc finalOutputDesc;
    FUSION_PASS_CHECK(matchedNode.GetOutputDesc(0, finalOutputDesc) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName.c_str(), "Failed to get output desc."), return nullptr);
    reshape3Node.UpdateOutputDesc(0, finalOutputDesc);
    auto replaceGraph = builder.BuildAndReset({reshape3Output});
    if (InferShape(replaceGraph, subgraphInputs) != SUCCESS) {
        return nullptr;
    }
    return replaceGraph;
}

// 009: reshape+reshape+matmul+reshape (swap input, dynamic) - abc,abde->dec
GraphUniqPtr EinsumPass::HandleDynamicABCxABDE2DEC(const std::vector<SubgraphInput>& subgraphInputs,
                                                   const GNode& matchedNode)
{
    OPS_LOG_D(kPassName.c_str(), "HandleDynamicABCxABDE2DEC");
    auto inputInfo = CollectInputInfo(subgraphInputs);
    FUSION_PASS_CHECK(!inputInfo.valid, OPS_LOG_E(kPassName.c_str(), "CollectInputInfo failed."), return nullptr);
    auto& x0Dims = inputInfo.dims[0];
    auto& x1Dims = inputInfo.dims[1];
    FUSION_PASS_CHECK(x0Dims.size() != kDim3D || x1Dims.size() != kDim4D,
                      OPS_LOG_W(kPassName.c_str(), "Input dims size check failed."), return nullptr);

    auto builder = es::EsGraphBuilder("replacement");
    ge::Graph* graphPtr = builder.GetCGraphBuilder()->GetGraph();
    auto input0 = builder.CreateInput(0, "x0", inputInfo.dtypes[0], inputInfo.formats[0], x0Dims);
    auto input1 = builder.CreateInput(1, "x1", inputInfo.dtypes[1], inputInfo.formats[1], x1Dims);

    // Reshape input0: {a*b, c}
    std::vector<int64_t> reshape1Dims{GetDimMulValue(x0Dims[0], x0Dims[1]), x0Dims[2]};
    GNode reshape1Node = CreateReshapeNode(builder, "reshape_1", input0, inputInfo.isDynamic[0], reshape1Dims, 0, 1);
    SetReshapeOutputDesc(reshape1Node, reshape1Dims);

    // Reshape input1: flatten dim 0-1 -> [a*b, c, d] 3D
    std::vector<int64_t> reshape2Dims{GetDimMulValue(x1Dims[0], x1Dims[1]), x1Dims[2], x1Dims[3]};
    GNode reshape2Node = CreateReshapeNode(builder, "reshape_2", input1, inputInfo.isDynamic[1], reshape2Dims, 0, 1);
    SetReshapeOutputDesc(reshape2Node, reshape2Dims);

    // Reshape input1 again: flatten dim 1-2 -> [a*b, c*d] 2D (only for dynamic shape)
    GNode reshape3Node = reshape2Node;
    if (inputInfo.isDynamic[1]) {
        std::vector<int64_t> reshape3Dims{GetDimMulValue(x1Dims[0], x1Dims[1]), GetDimMulValue(x1Dims[2], x1Dims[3])};
        reshape3Node = CreateReshapeNode(builder, "reshape_3", GetTensorHolder(builder, reshape2Node, 0), true,
                                         reshape3Dims, 1, kNumTwo);
        SetReshapeOutputDesc(reshape3Node, reshape3Dims);
    }

    // MatMul(reshape3, reshape1, transpose_x1=true, transpose_x2=false) (swap input)
    auto reshape1Output = GetTensorHolder(builder, reshape1Node, 0);
    auto reshape3Output = GetTensorHolder(builder, reshape3Node, 0);
    auto matmulOutput = CreateMatMulNode(builder, kMatMul, reshape3Output, reshape1Output, true, false);
    GNode matmulNode = *matmulOutput.GetProducer();

    // GatherShapes: axes {{1,2}, {1,3}, {0,2}} -> {d, e, c}
    std::vector<EsTensorHolder> gatherInputs{input0, input1};
    std::vector<std::vector<int64_t>> gatherAxes{{1, 2}, {1, 3}, {0, 2}};
    GNode gatherNode = CreateGatherShapesNode(gatherInputs, gatherAxes);
    SetGatherShapesOutputDesc(gatherNode, gatherAxes.size());

    // Reshape matmul output
    auto reshape4Output = es::Reshape(matmulOutput, GetTensorHolder(builder, gatherNode, 0));
    GNode reshape4Node = *reshape4Output.GetProducer();

    if (!CopyOtherAttrs(matchedNode, matmulNode, kPassName)) {
        return nullptr;
    }

    TensorDesc finalOutputDesc;
    FUSION_PASS_CHECK(matchedNode.GetOutputDesc(0, finalOutputDesc) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName.c_str(), "Failed to get output desc."), return nullptr);
    reshape4Node.UpdateOutputDesc(0, finalOutputDesc);
    auto replaceGraph = builder.BuildAndReset({reshape4Output});
    if (InferShape(replaceGraph, subgraphInputs) != SUCCESS) {
        return nullptr;
    }
    return replaceGraph;
}

// 010: transpose+batchmatmul+transpose - abcd,aecd->acbe
GraphUniqPtr EinsumPass::HandleABCDxAECD2ACBE(const std::vector<SubgraphInput>& subgraphInputs,
                                              const GNode& matchedNode)
{
    OPS_LOG_D(kPassName.c_str(), "HandleABCDxAECD2ACBE");
    auto inputInfo = CollectInputInfo(subgraphInputs);
    FUSION_PASS_CHECK(!inputInfo.valid, OPS_LOG_E(kPassName.c_str(), "CollectInputInfo failed."), return nullptr);
    auto& x0Dims = inputInfo.dims[0];
    auto& x1Dims = inputInfo.dims[1];
    FUSION_PASS_CHECK(x0Dims.size() != kDim4D || x1Dims.size() != kDim4D,
                      OPS_LOG_W(kPassName.c_str(), "Input dims size check failed."), return nullptr);

    auto builder = es::EsGraphBuilder("replacement");
    ge::Graph* graphPtr = builder.GetCGraphBuilder()->GetGraph();
    auto input0 = builder.CreateInput(0, "x1", inputInfo.dtypes[0], inputInfo.formats[0], x0Dims);
    auto input1 = builder.CreateInput(1, "x2", inputInfo.dtypes[1], inputInfo.formats[1], x1Dims);

    // Transpose input0: perm {0, 2, 1, 3} -> acbd
    std::vector<int32_t> perm{0, 2, 1, 3};
    GNode trans1Node = CreateTransposeNode(builder, "transpose_1", input0, inputInfo.isDynamic[0], supportL12btBf16_,
                                           perm);
    SetTransposeOutputDesc(trans1Node, perm);

    // Transpose input1: perm {0, 2, 1, 3} -> aced
    GNode trans2Node = CreateTransposeNode(builder, "transpose_2", input1, inputInfo.isDynamic[1], supportL12btBf16_,
                                           perm);
    SetTransposeOutputDesc(trans2Node, perm);

    // BatchMatMul(trans1, trans2, adj_x1=false, adj_x2=true)
    auto trans1Output = GetTensorHolder(builder, trans1Node, 0);
    auto trans2Output = GetTensorHolder(builder, trans2Node, 0);
    auto bmmOutput = CreateMatMulNode(builder, kBatchMatMul, trans1Output, trans2Output, false, true);
    GNode bmmNode = *bmmOutput.GetProducer();

    TensorDesc outputDesc;
    FUSION_PASS_CHECK(matchedNode.GetOutputDesc(0, outputDesc) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName.c_str(), "Failed to get output desc."), return nullptr);
    bmmNode.UpdateOutputDesc(0, outputDesc);
    if (!CopyOtherAttrs(matchedNode, bmmNode, kPassName)) {
        return nullptr;
    }

    auto replaceGraph = builder.BuildAndReset({bmmOutput});
    if (InferShape(replaceGraph, subgraphInputs) != SUCCESS) {
        return nullptr;
    }
    return replaceGraph;
}

// 011: transpose+batchmatmul+transpose(swap input) - abcd,acbe->aecd
GraphUniqPtr EinsumPass::HandleABCDxACBE2AECD(const std::vector<SubgraphInput>& subgraphInputs,
                                              const GNode& matchedNode)
{
    OPS_LOG_D(kPassName.c_str(), "HandleABCDxACBE2AECD");
    auto inputInfo = CollectInputInfo(subgraphInputs);
    FUSION_PASS_CHECK(!inputInfo.valid, OPS_LOG_E(kPassName.c_str(), "CollectInputInfo failed."), return nullptr);
    auto& x0Dims = inputInfo.dims[0];
    auto& x1Dims = inputInfo.dims[1];
    FUSION_PASS_CHECK(x0Dims.size() != kDim4D || x1Dims.size() != kDim4D,
                      OPS_LOG_W(kPassName.c_str(), "Input dims size check failed."), return nullptr);

    auto builder = es::EsGraphBuilder("replacement");
    ge::Graph* graphPtr = builder.GetCGraphBuilder()->GetGraph();
    auto input0 = builder.CreateInput(0, "x1", inputInfo.dtypes[0], inputInfo.formats[0], x0Dims);
    auto input1 = builder.CreateInput(1, "x2", inputInfo.dtypes[1], inputInfo.formats[1], x1Dims);

    // Transpose input0: perm {0, 2, 1, 3} -> acbd
    std::vector<int32_t> perm{0, 2, 1, 3};
    GNode trans1Node = CreateTransposeNode(builder, "transpose_1", input0, inputInfo.isDynamic[0], supportL12btBf16_,
                                           perm);
    SetTransposeOutputDesc(trans1Node, perm);

    // BatchMatMul(input1, trans1, adj_x1=true, adj_x2=false) (swap input)
    auto trans1Output = GetTensorHolder(builder, trans1Node, 0);
    auto bmmOutput = CreateMatMulNode(builder, kBatchMatMul, input1, trans1Output, true, false);
    GNode bmmNode = *bmmOutput.GetProducer();

    // Transpose output: perm {0, 2, 1, 3}
    GNode trans2Node = CreateTransposeNode(builder, "transpose_2", GetTensorHolder(builder, bmmNode, 0),
                                           inputInfo.isDynamic[0], supportL12btBf16_, perm);
    SetTransposeOutputDesc(trans2Node, perm);

    TensorDesc outputDesc;
    FUSION_PASS_CHECK(matchedNode.GetOutputDesc(0, outputDesc) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName.c_str(), "Failed to get output desc."), return nullptr);
    trans2Node.UpdateOutputDesc(0, outputDesc);
    if (!CopyOtherAttrs(matchedNode, bmmNode, kPassName)) {
        return nullptr;
    }

    auto trans2Output = GetTensorHolder(builder, trans2Node, 0);
    auto replaceGraph = builder.BuildAndReset({trans2Output});
    if (InferShape(replaceGraph, subgraphInputs) != SUCCESS) {
        return nullptr;
    }
    return replaceGraph;
}

// 012: reshape+reshape+matmul+reshape->reshape+batchmatmul - abcd,ecd->abe
GraphUniqPtr EinsumPass::HandleABCDxECD2ABE(const std::vector<SubgraphInput>& subgraphInputs, const GNode& matchedNode)
{
    OPS_LOG_D(kPassName.c_str(), "HandleABCDxECD2ABE");
    auto inputInfo = CollectInputInfo(subgraphInputs);
    FUSION_PASS_CHECK(!inputInfo.valid, OPS_LOG_E(kPassName.c_str(), "CollectInputInfo failed."), return nullptr);
    auto& x0Dims = inputInfo.dims[0];
    auto& x1Dims = inputInfo.dims[1];
    FUSION_PASS_CHECK(x0Dims.size() != kDim4D || x1Dims.size() != kDim3D,
                      OPS_LOG_W(kPassName.c_str(), "Input dims size check failed."), return nullptr);

    auto builder = es::EsGraphBuilder("replacement");
    ge::Graph* graphPtr = builder.GetCGraphBuilder()->GetGraph();
    auto input0 = builder.CreateInput(0, "x1", inputInfo.dtypes[0], inputInfo.formats[0], x0Dims);
    auto input1 = builder.CreateInput(1, "x2", inputInfo.dtypes[1], inputInfo.formats[1], x1Dims);

    // Reshape input0: {a, b, c*d}
    std::vector<int64_t> reshape1Dims{x0Dims[0], x0Dims[1], GetDimMulValue(x0Dims[2], x0Dims[3])};
    GNode reshape1Node = CreateReshapeNode(builder, "reshape_1", input0, inputInfo.isDynamic[0], reshape1Dims, 2, 3);
    SetReshapeOutputDesc(reshape1Node, reshape1Dims);

    // Reshape input1: {e, c*d}
    std::vector<int64_t> reshape2Dims{x1Dims[0], GetDimMulValue(x1Dims[1], x1Dims[2])};
    GNode reshape2Node = CreateReshapeNode(builder, "reshape_2", input1, inputInfo.isDynamic[1], reshape2Dims, 1, 2);
    SetReshapeOutputDesc(reshape2Node, reshape2Dims);

    // BatchMatMul(reshape1, reshape2, adj_x1=false, adj_x2=true)
    auto reshape1Output = GetTensorHolder(builder, reshape1Node, 0);
    auto reshape2Output = GetTensorHolder(builder, reshape2Node, 0);
    auto bmmOutput = CreateMatMulNode(builder, kBatchMatMul, reshape1Output, reshape2Output, false, true);
    GNode bmmNode = *bmmOutput.GetProducer();

    TensorDesc outputDesc;
    FUSION_PASS_CHECK(matchedNode.GetOutputDesc(0, outputDesc) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName.c_str(), "Failed to get output desc."), return nullptr);
    bmmNode.UpdateOutputDesc(0, outputDesc);
    if (!CopyOtherAttrs(matchedNode, bmmNode, kPassName)) {
        return nullptr;
    }

    auto replaceGraph = builder.BuildAndReset({bmmOutput});
    if (InferShape(replaceGraph, subgraphInputs) != SUCCESS) {
        return nullptr;
    }
    return replaceGraph;
}

// 013: reshape+reshape+matmul+reshape (swap input, dynamic) - abcd,abe->ecd
GraphUniqPtr EinsumPass::HandleDynamicABCDxABE2ECD(const std::vector<SubgraphInput>& subgraphInputs,
                                                   const GNode& matchedNode)
{
    OPS_LOG_D(kPassName.c_str(), "HandleDynamicABCDxABE2ECD");
    auto inputInfo = CollectInputInfo(subgraphInputs);
    FUSION_PASS_CHECK(!inputInfo.valid, OPS_LOG_E(kPassName.c_str(), "CollectInputInfo failed."), return nullptr);
    auto& x0Dims = inputInfo.dims[0];
    auto& x1Dims = inputInfo.dims[1];
    FUSION_PASS_CHECK(x0Dims.size() != kDim4D || x1Dims.size() != kDim3D,
                      OPS_LOG_W(kPassName.c_str(), "Input dims size check failed."), return nullptr);

    auto builder = es::EsGraphBuilder("replacement");
    ge::Graph* graphPtr = builder.GetCGraphBuilder()->GetGraph();
    auto input0 = builder.CreateInput(0, "x0", inputInfo.dtypes[0], inputInfo.formats[0], x0Dims);
    auto input1 = builder.CreateInput(1, "x1", inputInfo.dtypes[1], inputInfo.formats[1], x1Dims);

    // Reshape input0: flatten dim 0-1 -> [a*b, c, d] 3D
    std::vector<int64_t> reshape1Dims{GetDimMulValue(x0Dims[0], x0Dims[1]), x0Dims[2], x0Dims[3]};
    GNode reshape1Node = CreateReshapeNode(builder, "reshape_1", input0, inputInfo.isDynamic[0], reshape1Dims, 0, 1);
    SetReshapeOutputDesc(reshape1Node, reshape1Dims);

    // Reshape input0 again: flatten dim 1-2 -> [a*b, c*d] 2D (only for dynamic shape)
    GNode reshape3Node = reshape1Node;
    if (inputInfo.isDynamic[0]) {
        std::vector<int64_t> reshape3Dims{GetDimMulValue(x0Dims[0], x0Dims[1]), GetDimMulValue(x0Dims[2], x0Dims[3])};
        reshape3Node = CreateReshapeNode(builder, "reshape_3", GetTensorHolder(builder, reshape1Node, 0), true,
                                         reshape3Dims, 1, kNumTwo);
        SetReshapeOutputDesc(reshape3Node, reshape3Dims);
    }

    // Reshape input1: flatten dim 0-1 -> [a*b, e] 2D
    std::vector<int64_t> reshape2Dims{GetDimMulValue(x1Dims[0], x1Dims[1]), x1Dims[2]};
    GNode reshape2Node = CreateReshapeNode(builder, "reshape_2", input1, inputInfo.isDynamic[1], reshape2Dims, 0, 1);
    SetReshapeOutputDesc(reshape2Node, reshape2Dims);

    // MatMul(reshape2, reshape3, transpose_x1=true, transpose_x2=false) (swap input)
    auto reshape3Output = GetTensorHolder(builder, reshape3Node, 0);
    auto reshape2Output = GetTensorHolder(builder, reshape2Node, 0);
    auto matmulOutput = CreateMatMulNode(builder, kMatMul, reshape2Output, reshape3Output, true, false);
    GNode matmulNode = *matmulOutput.GetProducer();

    // GatherShapes: axes {{1,2}, {0,2}, {0,3}} -> {e, c, d}
    std::vector<EsTensorHolder> gatherInputs{input0, input1};
    std::vector<std::vector<int64_t>> gatherAxes{{1, 2}, {0, 2}, {0, 3}};
    GNode gatherNode = CreateGatherShapesNode(gatherInputs, gatherAxes);
    SetGatherShapesOutputDesc(gatherNode, gatherAxes.size());

    // Reshape matmul output
    auto reshape4Output = es::Reshape(matmulOutput, GetTensorHolder(builder, gatherNode, 0));
    GNode reshape4Node = *reshape4Output.GetProducer();

    if (!CopyOtherAttrs(matchedNode, matmulNode, kPassName)) {
        return nullptr;
    }

    TensorDesc finalOutputDesc;
    FUSION_PASS_CHECK(matchedNode.GetOutputDesc(0, finalOutputDesc) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName.c_str(), "Failed to get output desc."), return nullptr);
    reshape4Node.UpdateOutputDesc(0, finalOutputDesc);
    auto replaceGraph = builder.BuildAndReset({reshape4Output});
    if (InferShape(replaceGraph, subgraphInputs) != SUCCESS) {
        return nullptr;
    }
    return replaceGraph;
}

// 014: transpose+batchmatmul+transpose - abcd,acbe->adbe
GraphUniqPtr EinsumPass::HandleABCDxACBE2ADBE(const std::vector<SubgraphInput>& subgraphInputs,
                                              const GNode& matchedNode)
{
    OPS_LOG_D(kPassName.c_str(), "HandleABCDxACBE2ADBE");
    auto inputInfo = CollectInputInfo(subgraphInputs);
    FUSION_PASS_CHECK(!inputInfo.valid, OPS_LOG_E(kPassName.c_str(), "CollectInputInfo failed."), return nullptr);
    auto& x0Dims = inputInfo.dims[0];
    auto& x1Dims = inputInfo.dims[1];
    FUSION_PASS_CHECK(x0Dims.size() != kDim4D || x1Dims.size() != kDim4D,
                      OPS_LOG_W(kPassName.c_str(), "Input dims size check failed."), return nullptr);

    auto builder = es::EsGraphBuilder("replacement");
    ge::Graph* graphPtr = builder.GetCGraphBuilder()->GetGraph();
    auto input0 = builder.CreateInput(0, "x1", inputInfo.dtypes[0], inputInfo.formats[0], x0Dims);
    auto input1 = builder.CreateInput(1, "x2", inputInfo.dtypes[1], inputInfo.formats[1], x1Dims);

    // Transpose input1: perm {0, 2, 1, 3}
    std::vector<int32_t> perm{0, 2, 1, 3};
    GNode trans1Node = CreateTransposeNode(builder, "transpose_1", input1, inputInfo.isDynamic[1], supportL12btBf16_,
                                           perm);
    SetTransposeOutputDesc(trans1Node, perm);

    // BatchMatMul(input0, trans1, adj_x1=true, adj_x2=false)
    auto trans1Output = GetTensorHolder(builder, trans1Node, 0);
    auto bmmOutput = CreateMatMulNode(builder, kBatchMatMul, input0, trans1Output, true, false);
    GNode bmmNode = *bmmOutput.GetProducer();

    // Transpose output: perm {0, 2, 1, 3}
    GNode trans2Node = CreateTransposeNode(builder, "transpose_2", GetTensorHolder(builder, bmmNode, 0),
                                           inputInfo.isDynamic[0], supportL12btBf16_, perm);
    SetTransposeOutputDesc(trans2Node, perm);

    TensorDesc outputDesc;
    FUSION_PASS_CHECK(matchedNode.GetOutputDesc(0, outputDesc) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName.c_str(), "Failed to get output desc."), return nullptr);
    trans2Node.UpdateOutputDesc(0, outputDesc);
    if (!CopyOtherAttrs(matchedNode, bmmNode, kPassName)) {
        return nullptr;
    }

    auto trans2Output = GetTensorHolder(builder, trans2Node, 0);
    auto replaceGraph = builder.BuildAndReset({trans2Output});
    if (InferShape(replaceGraph, subgraphInputs) != SUCCESS) {
        return nullptr;
    }
    return replaceGraph;
}

// 015: batchmatmul - abcd,abde->abce
GraphUniqPtr EinsumPass::HandleABCDxABDE2ABCE(const std::vector<SubgraphInput>& subgraphInputs,
                                              const GNode& matchedNode)
{
    OPS_LOG_D(kPassName.c_str(), "HandleABCDxABDE2ABCE");
    return BuildBatchMatmulReplacement(false, false, subgraphInputs, matchedNode);
}

// 016: batchmatmul - abcd,abce->abde
GraphUniqPtr EinsumPass::HandleABCDxABCE2ABDE(const std::vector<SubgraphInput>& subgraphInputs,
                                              const GNode& matchedNode)
{
    OPS_LOG_D(kPassName.c_str(), "HandleABCDxABCE2ABDE");
    return BuildBatchMatmulReplacement(true, false, subgraphInputs, matchedNode);
}

// 017: transpose+batchmatmul+transpose - abcd,aebd->aebc
GraphUniqPtr EinsumPass::HandleABCDxAEBD2AEBC(const std::vector<SubgraphInput>& subgraphInputs,
                                              const GNode& matchedNode)
{
    OPS_LOG_D(kPassName.c_str(), "HandleABCDxAEBD2AEBC");
    auto inputInfo = CollectInputInfo(subgraphInputs);
    FUSION_PASS_CHECK(!inputInfo.valid, OPS_LOG_E(kPassName.c_str(), "CollectInputInfo failed."), return nullptr);
    auto& x0Dims = inputInfo.dims[0];
    auto& x1Dims = inputInfo.dims[1];
    FUSION_PASS_CHECK(x0Dims.size() != kDim4D || x1Dims.size() != kDim4D,
                      OPS_LOG_W(kPassName.c_str(), "Input dims size check failed."), return nullptr);

    auto builder = es::EsGraphBuilder("replacement");
    ge::Graph* graphPtr = builder.GetCGraphBuilder()->GetGraph();
    auto input0 = builder.CreateInput(0, "x1", inputInfo.dtypes[0], inputInfo.formats[0], x0Dims);
    auto input1 = builder.CreateInput(1, "x2", inputInfo.dtypes[1], inputInfo.formats[1], x1Dims);

    // Transpose input1: AEBD -> perm {0, 2, 3, 1} -> ABDE
    std::vector<int32_t> perm1{0, 2, 3, 1};
    GNode trans1Node = CreateTransposeNode(builder, "transpose_1", input1, inputInfo.isDynamic[1], supportL12btBf16_,
                                           perm1);
    SetTransposeOutputDesc(trans1Node, perm1);

    // BatchMatMul(input0, trans1)
    auto trans1Output = GetTensorHolder(builder, trans1Node, 0);
    auto bmmOutput = CreateMatMulNode(builder, kBatchMatMul, input0, trans1Output, false, false);
    GNode bmmNode = *bmmOutput.GetProducer();

    // Transpose output: ABCE -> perm {0, 3, 1, 2} -> AEBC
    std::vector<int32_t> perm2{0, 3, 1, 2};
    GNode trans2Node = CreateTransposeNode(builder, "transpose_2", GetTensorHolder(builder, bmmNode, 0),
                                           inputInfo.isDynamic[0], supportL12btBf16_, perm2);
    SetTransposeOutputDesc(trans2Node, perm2);

    TensorDesc outputDesc;
    FUSION_PASS_CHECK(matchedNode.GetOutputDesc(0, outputDesc) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName.c_str(), "Failed to get output desc."), return nullptr);
    trans2Node.UpdateOutputDesc(0, outputDesc);
    if (!CopyOtherAttrs(matchedNode, bmmNode, kPassName)) {
        return nullptr;
    }

    auto trans2Output = GetTensorHolder(builder, trans2Node, 0);
    auto replaceGraph = builder.BuildAndReset({trans2Output});
    if (InferShape(replaceGraph, subgraphInputs) != SUCCESS) {
        return nullptr;
    }
    return replaceGraph;
}

// 018: transpose+transpose+batchmatmul - abcd,abce->acde
GraphUniqPtr EinsumPass::HandleABCDxABCE2ACDE(const std::vector<SubgraphInput>& subgraphInputs,
                                              const GNode& matchedNode)
{
    OPS_LOG_D(kPassName.c_str(), "HandleABCDxABCE2ACDE");
    auto inputInfo = CollectInputInfo(subgraphInputs);
    FUSION_PASS_CHECK(!inputInfo.valid, OPS_LOG_E(kPassName.c_str(), "CollectInputInfo failed."), return nullptr);
    auto& x0Dims = inputInfo.dims[0];
    auto& x1Dims = inputInfo.dims[1];
    FUSION_PASS_CHECK(x0Dims.size() != kDim4D || x1Dims.size() != kDim4D,
                      OPS_LOG_W(kPassName.c_str(), "Input dims size check failed."), return nullptr);

    auto builder = es::EsGraphBuilder("replacement");
    ge::Graph* graphPtr = builder.GetCGraphBuilder()->GetGraph();
    auto input0 = builder.CreateInput(0, "x1", inputInfo.dtypes[0], inputInfo.formats[0], x0Dims);
    auto input1 = builder.CreateInput(1, "x2", inputInfo.dtypes[1], inputInfo.formats[1], x1Dims);

    // Transpose input0: ABCD -> ACDB, perm {0, 2, 3, 1}
    std::vector<int32_t> perm1{0, 2, 3, 1};
    GNode trans1Node = CreateTransposeNode(builder, "transpose_1", input0, inputInfo.isDynamic[0], supportL12btBf16_,
                                           perm1);
    SetTransposeOutputDesc(trans1Node, perm1);

    // Transpose input1: ABCE -> ACBE, perm {0, 2, 1, 3}
    std::vector<int32_t> perm2{0, 2, 1, 3};
    GNode trans2Node = CreateTransposeNode(builder, "transpose_2", input1, inputInfo.isDynamic[1], supportL12btBf16_,
                                           perm2);
    SetTransposeOutputDesc(trans2Node, perm2);

    // BatchMatMul(trans1, trans2)
    auto trans1Output = GetTensorHolder(builder, trans1Node, 0);
    auto trans2Output = GetTensorHolder(builder, trans2Node, 0);
    auto bmmOutput = CreateMatMulNode(builder, kBatchMatMul, trans1Output, trans2Output, false, false);
    GNode bmmNode = *bmmOutput.GetProducer();

    TensorDesc outputDesc;
    FUSION_PASS_CHECK(matchedNode.GetOutputDesc(0, outputDesc) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName.c_str(), "Failed to get output desc."), return nullptr);
    bmmNode.UpdateOutputDesc(0, outputDesc);
    if (!CopyOtherAttrs(matchedNode, bmmNode, kPassName)) {
        return nullptr;
    }

    auto replaceGraph = builder.BuildAndReset({bmmOutput});
    if (InferShape(replaceGraph, subgraphInputs) != SUCCESS) {
        return nullptr;
    }
    return replaceGraph;
}

// 019: batchmatmul - abc,abd->acd
GraphUniqPtr EinsumPass::HandleABCxABD2ACD(const std::vector<SubgraphInput>& subgraphInputs, const GNode& matchedNode)
{
    OPS_LOG_D(kPassName.c_str(), "HandleABCxABD2ACD");
    return BuildBatchMatmulReplacement(true, false, subgraphInputs, matchedNode);
}

// 020: batchmatmul - ab,cb->ac
GraphUniqPtr EinsumPass::HandleABxCB2AC(const std::vector<SubgraphInput>& subgraphInputs, const GNode& matchedNode)
{
    OPS_LOG_D(kPassName.c_str(), "HandleABxCB2AC");
    return BuildBatchMatmulReplacement(false, true, subgraphInputs, matchedNode);
}

// 021: batchmatmul - abc,acd->abd
GraphUniqPtr EinsumPass::HandleABCxACD2ABD(const std::vector<SubgraphInput>& subgraphInputs, const GNode& matchedNode)
{
    OPS_LOG_D(kPassName.c_str(), "HandleABCxACD2ABD");
    return BuildBatchMatmulReplacement(false, false, subgraphInputs, matchedNode);
}

// 022: batchmatmul - abc,adc->abd
GraphUniqPtr EinsumPass::HandleABCxADC2ABD(const std::vector<SubgraphInput>& subgraphInputs, const GNode& matchedNode)
{
    OPS_LOG_D(kPassName.c_str(), "HandleABCxADC2ABD");
    return BuildBatchMatmulReplacement(false, true, subgraphInputs, matchedNode);
}

// 023: transpose+batchmatmul+transpose - abcd,ced->abce
GraphUniqPtr EinsumPass::HandleABCDxCED2ABCE(const std::vector<SubgraphInput>& subgraphInputs, const GNode& matchedNode)
{
    OPS_LOG_D(kPassName.c_str(), "HandleABCDxCED2ABCE");
    auto inputInfo = CollectInputInfo(subgraphInputs);
    FUSION_PASS_CHECK(!inputInfo.valid, OPS_LOG_E(kPassName.c_str(), "CollectInputInfo failed."), return nullptr);
    auto& x0Dims = inputInfo.dims[0];
    auto& x1Dims = inputInfo.dims[1];
    FUSION_PASS_CHECK(x0Dims.size() != kDim4D || x1Dims.size() != kDim3D,
                      OPS_LOG_W(kPassName.c_str(), "Input dims size check failed."), return nullptr);

    auto builder = es::EsGraphBuilder("replacement");
    ge::Graph* graphPtr = builder.GetCGraphBuilder()->GetGraph();
    auto input0 = builder.CreateInput(0, "x1", inputInfo.dtypes[0], inputInfo.formats[0], x0Dims);
    auto input1 = builder.CreateInput(1, "x2", inputInfo.dtypes[1], inputInfo.formats[1], x1Dims);

    // Transpose input0: ABCD -> ACBD, perm {0, 2, 1, 3}
    std::vector<int32_t> perm1{0, 2, 1, 3};
    GNode trans1Node = CreateTransposeNode(builder, "transpose_1", input0, inputInfo.isDynamic[0], supportL12btBf16_,
                                           perm1);
    SetTransposeOutputDesc(trans1Node, perm1);

    // BatchMatMul(trans1, input1, adj_x1=false, adj_x2=true)
    auto trans1Output = GetTensorHolder(builder, trans1Node, 0);
    auto bmmOutput = CreateMatMulNode(builder, kBatchMatMul, trans1Output, input1, false, true);
    GNode bmmNode = *bmmOutput.GetProducer();

    // Transpose output: ACBE -> ABCE, perm {0, 2, 1, 3}
    std::vector<int32_t> perm2{0, 2, 1, 3};
    GNode trans2Node = CreateTransposeNode(builder, "transpose_2", GetTensorHolder(builder, bmmNode, 0),
                                           inputInfo.isDynamic[0], supportL12btBf16_, perm2);
    SetTransposeOutputDesc(trans2Node, perm2);

    TensorDesc outputDesc;
    FUSION_PASS_CHECK(matchedNode.GetOutputDesc(0, outputDesc) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName.c_str(), "Failed to get output desc."), return nullptr);
    trans2Node.UpdateOutputDesc(0, outputDesc);
    if (!CopyOtherAttrs(matchedNode, bmmNode, kPassName)) {
        return nullptr;
    }

    auto trans2Output = GetTensorHolder(builder, trans2Node, 0);
    auto replaceGraph = builder.BuildAndReset({trans2Output});
    if (InferShape(replaceGraph, subgraphInputs) != SUCCESS) {
        return nullptr;
    }
    return replaceGraph;
}

// 024: transpose+transpose+batchmatmul - abcd,ebcd->bcae
GraphUniqPtr EinsumPass::HandleABCDxEBCD2BCAE(const std::vector<SubgraphInput>& subgraphInputs,
                                              const GNode& matchedNode)
{
    OPS_LOG_D(kPassName.c_str(), "HandleABCDxEBCD2BCAE");
    auto inputInfo = CollectInputInfo(subgraphInputs);
    FUSION_PASS_CHECK(!inputInfo.valid, OPS_LOG_E(kPassName.c_str(), "CollectInputInfo failed."), return nullptr);
    auto& x0Dims = inputInfo.dims[0];
    auto& x1Dims = inputInfo.dims[1];
    FUSION_PASS_CHECK(x0Dims.size() != kDim4D || x1Dims.size() != kDim4D,
                      OPS_LOG_W(kPassName.c_str(), "Input dims size check failed."), return nullptr);

    auto builder = es::EsGraphBuilder("replacement");
    ge::Graph* graphPtr = builder.GetCGraphBuilder()->GetGraph();
    auto input0 = builder.CreateInput(0, "x1", inputInfo.dtypes[0], inputInfo.formats[0], x0Dims);
    auto input1 = builder.CreateInput(1, "x2", inputInfo.dtypes[1], inputInfo.formats[1], x1Dims);

    // Transpose input0: ABCD -> BCAD, perm {1, 2, 0, 3}
    std::vector<int32_t> perm1{1, 2, 0, 3};
    GNode trans1Node = CreateTransposeNode(builder, "transpose_1", input0, inputInfo.isDynamic[0], supportL12btBf16_,
                                           perm1);
    SetTransposeOutputDesc(trans1Node, perm1);

    // Transpose input1: EBCD -> BCED, perm {1, 2, 0, 3}
    std::vector<int32_t> perm2{1, 2, 0, 3};
    GNode trans2Node = CreateTransposeNode(builder, "transpose_2", input1, inputInfo.isDynamic[1], supportL12btBf16_,
                                           perm2);
    SetTransposeOutputDesc(trans2Node, perm2);

    // BatchMatMul(trans1, trans2, adj_x1=false, adj_x2=true)
    auto trans1Output = GetTensorHolder(builder, trans1Node, 0);
    auto trans2Output = GetTensorHolder(builder, trans2Node, 0);
    auto bmmOutput = CreateMatMulNode(builder, kBatchMatMul, trans1Output, trans2Output, false, true);
    GNode bmmNode = *bmmOutput.GetProducer();

    TensorDesc outputDesc;
    FUSION_PASS_CHECK(matchedNode.GetOutputDesc(0, outputDesc) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName.c_str(), "Failed to get output desc."), return nullptr);
    bmmNode.UpdateOutputDesc(0, outputDesc);
    if (!CopyOtherAttrs(matchedNode, bmmNode, kPassName)) {
        return nullptr;
    }

    auto replaceGraph = builder.BuildAndReset({bmmOutput});
    if (InferShape(replaceGraph, subgraphInputs) != SUCCESS) {
        return nullptr;
    }
    return replaceGraph;
}

// 025: transpose+batchmatmul+transpose - abcd,dabe->cabe
GraphUniqPtr EinsumPass::HandleABCDxDABE2CABE(const std::vector<SubgraphInput>& subgraphInputs,
                                              const GNode& matchedNode)
{
    OPS_LOG_D(kPassName.c_str(), "HandleABCDxDABE2CABE");
    auto inputInfo = CollectInputInfo(subgraphInputs);
    FUSION_PASS_CHECK(!inputInfo.valid, OPS_LOG_E(kPassName.c_str(), "CollectInputInfo failed."), return nullptr);
    auto& x0Dims = inputInfo.dims[0];
    auto& x1Dims = inputInfo.dims[1];
    FUSION_PASS_CHECK(x0Dims.size() != kDim4D || x1Dims.size() != kDim4D,
                      OPS_LOG_W(kPassName.c_str(), "Input dims size check failed."), return nullptr);

    auto builder = es::EsGraphBuilder("replacement");
    ge::Graph* graphPtr = builder.GetCGraphBuilder()->GetGraph();
    auto input0 = builder.CreateInput(0, "x1", inputInfo.dtypes[0], inputInfo.formats[0], x0Dims);
    auto input1 = builder.CreateInput(1, "x2", inputInfo.dtypes[1], inputInfo.formats[1], x1Dims);

    // Transpose input1: DABE -> ABDE, perm {1, 2, 0, 3}
    std::vector<int32_t> perm1{1, 2, 0, 3};
    GNode trans1Node = CreateTransposeNode(builder, "transpose_1", input1, inputInfo.isDynamic[1], supportL12btBf16_,
                                           perm1);
    SetTransposeOutputDesc(trans1Node, perm1);

    // BatchMatMul(input0, trans1)
    auto trans1Output = GetTensorHolder(builder, trans1Node, 0);
    auto bmmOutput = CreateMatMulNode(builder, kBatchMatMul, input0, trans1Output, false, false);
    GNode bmmNode = *bmmOutput.GetProducer();

    // Transpose output: ABCE -> CABE, perm {2, 0, 1, 3}
    std::vector<int32_t> perm2{2, 0, 1, 3};
    GNode trans2Node = CreateTransposeNode(builder, "transpose_2", GetTensorHolder(builder, bmmNode, 0),
                                           inputInfo.isDynamic[0], supportL12btBf16_, perm2);
    SetTransposeOutputDesc(trans2Node, perm2);

    TensorDesc outputDesc;
    FUSION_PASS_CHECK(matchedNode.GetOutputDesc(0, outputDesc) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName.c_str(), "Failed to get output desc."), return nullptr);
    trans2Node.UpdateOutputDesc(0, outputDesc);
    if (!CopyOtherAttrs(matchedNode, bmmNode, kPassName)) {
        return nullptr;
    }

    auto trans2Output = GetTensorHolder(builder, trans2Node, 0);
    auto replaceGraph = builder.BuildAndReset({trans2Output});
    if (InferShape(replaceGraph, subgraphInputs) != SUCCESS) {
        return nullptr;
    }
    return replaceGraph;
}

// 026: unsqueeze+unsqueeze+mul - a,b->ab
GraphUniqPtr EinsumPass::HandleAxB2AB(const std::vector<SubgraphInput>& subgraphInputs, const GNode& matchedNode)
{
    OPS_LOG_D(kPassName.c_str(), "HandleAxB2AB");
    auto inputInfo = CollectInputInfo(subgraphInputs);
    FUSION_PASS_CHECK(!inputInfo.valid, OPS_LOG_E(kPassName.c_str(), "CollectInputInfo failed."), return nullptr);
    auto& x0Dims = inputInfo.dims[0];
    auto& x1Dims = inputInfo.dims[1];
    FUSION_PASS_CHECK(x0Dims.size() != kDim1D || x1Dims.size() != kDim1D,
                      OPS_LOG_W(kPassName.c_str(), "Input dims size check failed."), return nullptr);

    auto builder = es::EsGraphBuilder("replacement");
    ge::Graph* graphPtr = builder.GetCGraphBuilder()->GetGraph();
    auto input0 = builder.CreateInput(0, "x1", inputInfo.dtypes[0], inputInfo.formats[0], x0Dims);
    auto input1 = builder.CreateInput(1, "x2", inputInfo.dtypes[1], inputInfo.formats[1], x1Dims);

    // Unsqueeze input0: axes {1} -> {a, 1}
    GNode unsqueeze1Node = CreateUnsqueezeNode(graphPtr, "unsqueeze_1", {1});
    FUSION_PASS_CHECK(AddEdgeAndUpdatePeerDesc(*graphPtr, *input0.GetProducer(), 0, unsqueeze1Node, 0) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName.c_str(), "AddEdgeAndUpdatePeerDesc failed."), return nullptr);
    SetUnsqueezeOutputDesc(unsqueeze1Node, {1});

    // Unsqueeze input1: axes {0} -> {1, b}
    GNode unsqueeze2Node = CreateUnsqueezeNode(graphPtr, "unsqueeze_2", {0});
    FUSION_PASS_CHECK(AddEdgeAndUpdatePeerDesc(*graphPtr, *input1.GetProducer(), 0, unsqueeze2Node, 0) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName.c_str(), "AddEdgeAndUpdatePeerDesc failed."), return nullptr);
    SetUnsqueezeOutputDesc(unsqueeze2Node, {0});

    // Mul(unsqueeze1, unsqueeze2)
    auto unsqueeze1Output = GetTensorHolder(builder, unsqueeze1Node, 0);
    auto unsqueeze2Output = GetTensorHolder(builder, unsqueeze2Node, 0);
    GNode mulNode = CreateMulNode(builder, unsqueeze1Output, unsqueeze2Output);

    TensorDesc outputDesc;
    FUSION_PASS_CHECK(matchedNode.GetOutputDesc(0, outputDesc) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName.c_str(), "Failed to get output desc."), return nullptr);
    mulNode.UpdateOutputDesc(0, outputDesc);

    auto mulOutput = GetTensorHolder(builder, mulNode, 0);
    auto replaceGraph = builder.BuildAndReset({mulOutput});
    if (InferShape(replaceGraph, subgraphInputs) != SUCCESS) {
        return nullptr;
    }
    return replaceGraph;
}

// 027: transpose+reshape+matmul - abcd,aecd->eb
GraphUniqPtr EinsumPass::HandleABCDxAECD2EB(const std::vector<SubgraphInput>& subgraphInputs, const GNode& matchedNode)
{
    OPS_LOG_D(kPassName.c_str(), "HandleABCDxAECD2EB");
    auto inputInfo = CollectInputInfo(subgraphInputs);
    FUSION_PASS_CHECK(!inputInfo.valid, OPS_LOG_E(kPassName.c_str(), "CollectInputInfo failed."), return nullptr);
    auto& x0Dims = inputInfo.dims[0];
    auto& x1Dims = inputInfo.dims[1];
    FUSION_PASS_CHECK(x0Dims.size() != kDim4D || x1Dims.size() != kDim4D,
                      OPS_LOG_W(kPassName.c_str(), "Input dims size check failed."), return nullptr);

    auto builder = es::EsGraphBuilder("replacement");
    ge::Graph* graphPtr = builder.GetCGraphBuilder()->GetGraph();
    auto input0 = builder.CreateInput(0, "x1", inputInfo.dtypes[0], inputInfo.formats[0], x0Dims);
    auto input1 = builder.CreateInput(1, "x2", inputInfo.dtypes[1], inputInfo.formats[1], x1Dims);

    // Transpose input0: ABCD -> BACD, perm {1, 0, 2, 3}
    std::vector<int32_t> perm{1, 0, 2, 3};
    GNode trans1Node = CreateTransposeNode(builder, "transpose_1", input0, inputInfo.isDynamic[0], supportL12btBf16_,
                                           perm);
    SetTransposeOutputDesc(trans1Node, perm);

    // Transpose input1: AECD -> EACD, perm {1, 0, 2, 3}
    GNode trans2Node = CreateTransposeNode(builder, "transpose_2", input1, inputInfo.isDynamic[1], supportL12btBf16_,
                                           perm);
    SetTransposeOutputDesc(trans2Node, perm);

    // Reshape trans1: {b, a*c*d}
    std::vector<int64_t> reshape1Dims{x0Dims[1], GetDimMulValue(GetDimMulValue(x0Dims[0], x0Dims[2]), x0Dims[3])};
    GNode reshape1Node = CreateReshapeNode(builder, "reshape_1", GetTensorHolder(builder, trans1Node, 0),
                                           inputInfo.isDynamic[0], reshape1Dims, 1, 3);
    SetReshapeOutputDesc(reshape1Node, reshape1Dims);

    // Reshape trans2: {e, a*c*d}
    std::vector<int64_t> reshape2Dims{x1Dims[1], GetDimMulValue(GetDimMulValue(x1Dims[0], x1Dims[2]), x1Dims[3])};
    GNode reshape2Node = CreateReshapeNode(builder, "reshape_2", GetTensorHolder(builder, trans2Node, 0),
                                           inputInfo.isDynamic[1], reshape2Dims, 1, 3);
    SetReshapeOutputDesc(reshape2Node, reshape2Dims);

    // MatMul(reshape2, reshape1, transpose_x1=false, transpose_x2=true) (swap input)
    auto reshape1Output = GetTensorHolder(builder, reshape1Node, 0);
    auto reshape2Output = GetTensorHolder(builder, reshape2Node, 0);
    auto matmulOutput = CreateMatMulNode(builder, kMatMul, reshape2Output, reshape1Output, false, true);
    GNode matmulNode = *matmulOutput.GetProducer();
    if (!CopyOtherAttrs(matchedNode, matmulNode, kPassName)) {
        return nullptr;
    }

    auto replaceGraph = builder.BuildAndReset({matmulOutput});
    if (InferShape(replaceGraph, subgraphInputs) != SUCCESS) {
        return nullptr;
    }
    return replaceGraph;
}

// 028: transpose+reshape+matmul+reshape+transpose (dynamic) - abcd,eb->aecd
GraphUniqPtr EinsumPass::HandleDynamicABCDxEB2AECD(const std::vector<SubgraphInput>& subgraphInputs,
                                                   const GNode& matchedNode)
{
    OPS_LOG_D(kPassName.c_str(), "HandleDynamicABCDxEB2AECD");
    auto inputInfo = CollectInputInfo(subgraphInputs);
    FUSION_PASS_CHECK(!inputInfo.valid, OPS_LOG_E(kPassName.c_str(), "CollectInputInfo failed."), return nullptr);
    auto& x0Dims = inputInfo.dims[0];
    auto& x1Dims = inputInfo.dims[1];
    FUSION_PASS_CHECK(x0Dims.size() != kDim4D || x1Dims.size() != kDim2D,
                      OPS_LOG_W(kPassName.c_str(), "Input dims size check failed."), return nullptr);

    auto builder = es::EsGraphBuilder("replacement");
    ge::Graph* graphPtr = builder.GetCGraphBuilder()->GetGraph();
    auto input0 = builder.CreateInput(0, "x0", inputInfo.dtypes[0], inputInfo.formats[0], x0Dims);
    auto input1 = builder.CreateInput(1, "x1", inputInfo.dtypes[1], inputInfo.formats[1], x1Dims);

    // Transpose input0: ABCD -> BACD, perm {1, 0, 2, 3}
    std::vector<int32_t> perm{1, 0, 2, 3};
    GNode trans1Node = CreateTransposeNode(builder, "transpose_1", input0, inputInfo.isDynamic[0], supportL12btBf16_,
                                           perm);
    SetTransposeOutputDesc(trans1Node, perm);

    // Reshape trans1: {b, a*c*d}
    std::vector<int64_t> reshape1Dims{x0Dims[1], GetDimMulValue(GetDimMulValue(x0Dims[0], x0Dims[2]), x0Dims[3])};
    GNode reshape1Node = CreateReshapeNode(builder, "reshape_1", GetTensorHolder(builder, trans1Node, 0),
                                           inputInfo.isDynamic[0], reshape1Dims, 1, 3);
    SetReshapeOutputDesc(reshape1Node, reshape1Dims);

    // Note: MatMul multiplies input1 with the reshaped trans1 result, producing a 2D output
    auto reshape1Output = GetTensorHolder(builder, reshape1Node, 0);
    auto matmulOutput = CreateMatMulNode(builder, kMatMul, input1, reshape1Output, false, false);
    GNode matmulNode = *matmulOutput.GetProducer();

    // GatherShapes for dynamic output: axes {{1,0},{0,0},{0,2},{0,3}} -> {e, a, c, d}
    std::vector<EsTensorHolder> gatherInputs{input0, input1};
    std::vector<std::vector<int64_t>> gatherAxes{{1, 0}, {0, 0}, {0, 2}, {0, 3}};
    GNode gatherNode = CreateGatherShapesNode(gatherInputs, gatherAxes);
    SetGatherShapesOutputDesc(gatherNode, gatherAxes.size());

    // Reshape matmul output to {e, a, c, d} using GatherShapes result
    auto reshape2Output = es::Reshape(matmulOutput, GetTensorHolder(builder, gatherNode, 0));
    GNode reshape2Node = *reshape2Output.GetProducer();

    // Transpose output: EACD -> AECD, perm {1, 0, 2, 3}
    GNode trans2Node = CreateTransposeNode(builder, "transpose_2", GetTensorHolder(builder, reshape2Node, 0),
                                           inputInfo.isDynamic[0], supportL12btBf16_, perm);
    SetTransposeOutputDesc(trans2Node, perm);

    if (!CopyOtherAttrs(matchedNode, matmulNode, kPassName)) {
        return nullptr;
    }

    auto trans2Output = GetTensorHolder(builder, trans2Node, 0);
    TensorDesc finalOutputDesc;
    FUSION_PASS_CHECK(matchedNode.GetOutputDesc(0, finalOutputDesc) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName.c_str(), "Failed to get output desc."), return nullptr);
    trans2Node.UpdateOutputDesc(0, finalOutputDesc);
    auto replaceGraph = builder.BuildAndReset({trans2Output});
    if (InferShape(replaceGraph, subgraphInputs) != SUCCESS) {
        return nullptr;
    }
    return replaceGraph;
}

// ============================================================================
// Generic decomposition - SplitOpInFuzzScene (static shape)
// ============================================================================

// Update the final node's output desc with the matched Einsum node's shape/format while keeping
// the replacement chain's computed dtype. The legacy fuzz paths leave the final output in the
// computed dtype (e.g. the static single-input reduction stays in the input dtype); overriding
// the dtype with the matched node's output desc would mislabel the data when input/output dtypes
// differ (the desc would claim fp16 while the producer actually computes fp32).
void KeepComputedDtypeAndUpdateOutputDesc(GNode& lastNode, const TensorDesc& outputDesc)
{
    TensorDesc mergedDesc = outputDesc;
    TensorDesc lastDesc;
    if (lastNode.GetOutputDesc(0, lastDesc) == GRAPH_SUCCESS) {
        mergedDesc.SetDataType(lastDesc.GetDataType());
    }
    lastNode.UpdateOutputDesc(0, mergedDesc);
}

// Resolve the final output node of the replacement chain (split from
// SplitOpInFuzzScene/SplitDynamicFuzzScene); returns invalid GNode on failure
GNode EinsumPass::ResolveLastOutputNode()
{
    if (!bmmOutputNodes_.empty()) {
        return bmmOutputNodes_.back();
    }
    if (IsGNodeValid(batchmatmulNode_)) {
        return batchmatmulNode_;
    }
    if (!bmmInputNodes_[0].empty()) {
        return bmmInputNodes_[0].back();
    }
    // Passthrough (identity equation like abc->abc): the input Data node cannot serve
    // as the replacement graph output; bridge with an identity Cast node (same dtype,
    // guaranteed by the Einsum op infershape which forces output dtype = input dtype).
    GNode inputProducer = *subgraphInputHolders_[0].GetProducer();
    TensorDesc inputDesc;
    FUSION_PASS_CHECK(inputProducer.GetOutputDesc(0, inputDesc) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName.c_str(), "Failed to get passthrough input desc."), return GNode(););
    GNode castNode = CreateCastNode(*builder_, subgraphInputHolders_[0], inputDesc.GetDataType());
    return castNode;
}

GraphUniqPtr EinsumPass::SplitOpInFuzzScene(const std::string& equation,
                                            const std::vector<SubgraphInput>& subgraphInputs, const GNode& matchedNode)
{
    OPS_LOG_D(kPassName.c_str(), "SplitOpInFuzzScene equation=%s", equation.c_str());

    std::vector<std::string> inEquations;
    std::string outEquation;
    ParseEquation(equation, inEquations, outEquation);
    FUSION_PASS_CHECK(inEquations.empty(), OPS_LOG_W(kPassName.c_str(), "Parse equation failed."), return nullptr;);

    auto inputInfo = CollectInputInfo(subgraphInputs);
    FUSION_PASS_CHECK(!inputInfo.valid, OPS_LOG_E(kPassName.c_str(), "CollectInputInfo failed."), return nullptr);
    es::EsGraphBuilder localBuilder("einsum_generic");
    builder_ = &localBuilder;
    graphPtr_ = localBuilder.GetCGraphBuilder()->GetGraph();
    subgraphInputHolders_.clear();
    for (size_t idx = 0; idx < subgraphInputs.size(); ++idx) {
        auto holder = localBuilder.CreateInput(static_cast<int64_t>(idx), ("x" + std::to_string(idx)).c_str(),
                                               inputInfo.dtypes[idx], inputInfo.formats[idx], inputInfo.dims[idx]);
        subgraphInputHolders_.push_back(holder);
    }

    std::string curOutEquation(outEquation);
    TensorDesc outputDesc;
    FUSION_PASS_CHECK(matchedNode.GetOutputDesc(0, outputDesc) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName.c_str(), "Failed to get output desc."), return nullptr);
    CollectDimensionType(outputDesc.GetShape().GetDimNum(), outEquation, outputLabelInfo_);
    ReorderAxes(outputLabelInfo_);

    bmmInputNodes_.resize(inEquations.size());
    FUSION_PASS_CHECK(TransposeInput(inEquations, matchedNode) != SUCCESS,
                      OPS_LOG_W(kPassName.c_str(), "failed to process input transpose"), return nullptr;);
    FUSION_PASS_CHECK(StrideInput(inEquations) != SUCCESS,
                      OPS_LOG_W(kPassName.c_str(), "failed to process input stride"), return nullptr;);
    FUSION_PASS_CHECK(ReduceInput(inEquations, matchedNode) != SUCCESS,
                      OPS_LOG_W(kPassName.c_str(), "failed to process input reduce"), return nullptr;);
    FUSION_PASS_CHECK(ReshapeInput(inEquations, outEquation, matchedNode) != SUCCESS,
                      OPS_LOG_W(kPassName.c_str(), "failed to process input reshape"), return nullptr;);
    FUSION_PASS_CHECK(DoBatchMatmul(inEquations, matchedNode) != SUCCESS,
                      OPS_LOG_W(kPassName.c_str(), "failed to process batchmatmul"), return nullptr;);
    FUSION_PASS_CHECK(ReshapeOutput(matchedNode, curOutEquation, inEquations) != SUCCESS,
                      OPS_LOG_W(kPassName.c_str(), "failed to process output reshape"), return nullptr;);
    FUSION_PASS_CHECK(InflatedOutput(outEquation) != SUCCESS,
                      OPS_LOG_W(kPassName.c_str(), "failed to process output inflated"), return nullptr;);
    FUSION_PASS_CHECK(TransposeOutput(outEquation, matchedNode, curOutEquation) != SUCCESS,
                      OPS_LOG_W(kPassName.c_str(), "failed to process output transpose"), return nullptr;);

    GNode lastNode = ResolveLastOutputNode();
    FUSION_PASS_CHECK(!IsGNodeValid(lastNode), OPS_LOG_E(kPassName.c_str(), "Failed to resolve last output node."),
                      return nullptr;);
    KeepComputedDtypeAndUpdateOutputDesc(lastNode, outputDesc);

    auto lastHolder = GetTensorHolder(localBuilder, lastNode, 0);
    auto replaceGraph = localBuilder.BuildAndReset({lastHolder});
    FUSION_PASS_CHECK(InferShape(replaceGraph, subgraphInputs) != SUCCESS,
                      OPS_LOG_E(kPassName.c_str(), "InferShape for SplitOpInFuzzScene failed."), return nullptr;);
    return replaceGraph;
}

// ============================================================================
// Generic decomposition - SplitDynamicFuzzScene (dynamic shape)
// ============================================================================

// ============================================================================
// SplitDynamicFuzzScene helpers (split from the original oversized function)
// ============================================================================

ge::Status EinsumPass::PrepareDynamicLabels(const std::string& equation, const GNode& matchedNode, size_t numOps,
                                            std::vector<std::vector<uint8_t>>& opLabels, int64_t& ellNumDim,
                                            int64_t& permIndex, int64_t& outNumDim, std::vector<int64_t>& dimCounts)
{
    const auto arrowPos = equation.find("->");
    const auto lhs = equation.substr(0, arrowPos);
    bool ellInInput = false;
    FUSION_PASS_CHECK(InputLabelProcess(lhs, numOps, ellInInput, opLabels) != SUCCESS,
                      OPS_LOG_E(kPassName.c_str(), "failed to process input labels"), return FAILED);
    std::vector<int64_t> labelCount(kTotalLabels, 0);
    std::vector<int64_t> labelPermIndex(kTotalLabels, -1);
    int64_t ellIndex = 0;
    bool ellInOutput = false;
    FUSION_PASS_CHECK(LabelCountMap(matchedNode, numOps, ellNumDim, labelCount, opLabels, arrowPos) != SUCCESS,
                      OPS_LOG_E(kPassName.c_str(), "failed to process label count map"), return FAILED);
    if (ParseOutputLabel(equation, arrowPos, ellNumDim, permIndex, ellInOutput, labelPermIndex, labelCount, outNumDim,
                         ellIndex) != SUCCESS) {
        OPS_LOG_E(kPassName.c_str(), "failed to parse output labels");
        return FAILED;
    }
    std::vector<int64_t> labelSize(kTotalLabels, 1);
    std::vector<int64_t> ellSizes(ellNumDim, 1);
    dimCounts.assign(permIndex, 0);
    if (AlignInputDimForOutLabel(matchedNode, numOps, permIndex, opLabels, dimCounts, ellSizes, labelSize,
                                 labelPermIndex, ellNumDim, ellIndex) != SUCCESS) {
        OPS_LOG_E(kPassName.c_str(), "failed to align input dimensions");
        return FAILED;
    }
    return SUCCESS;
}

ge::Status EinsumPass::ReduceAndSumPairForDynamic(const GNode& matchedNode, int64_t permIndex, int64_t outNumDim,
                                                  std::vector<int64_t>& dimCounts, std::vector<int64_t>& sumDims)
{
    std::vector<int64_t> aDimsToSum;
    std::vector<int64_t> bDimsToSum;
    auto aDesc = GetPrevOutputDesc(matchedNode, 0);
    auto bDesc = GetPrevOutputDesc(matchedNode, 1);
    auto a = aDesc->GetShape().GetDims();
    auto b = bDesc->GetShape().GetDims();
    // Raw input dtype consistency on this route is enforced in MeetRequirements (the legacy
    // NOT_CHANGED semantics cannot be expressed from inside the replacement chain here).
    for (auto dim = outNumDim; dim < permIndex; ++dim) {
        if (a[dim] != 1 && b[dim] != 1) {
            if (--dimCounts[dim] == 1) {
                sumDims.push_back(dim);
                dimCounts[dim] = 0;
            }
        } else if (dimCounts[dim] == 1) {
            if (a[dim] != 1) {
                aDimsToSum.push_back(dim);
                dimCounts[dim] = 0;
            } else if (b[dim] != 1) {
                bDimsToSum.push_back(dim);
                dimCounts[dim] = 0;
            }
        }
    }
    FUSION_PASS_CHECK(ReduceSumInput(matchedNode, aDimsToSum, 0, true) != SUCCESS,
                      OPS_LOG_W(kPassName.c_str(), "failed to process input A ReduceSum"), return FAILED);
    FUSION_PASS_CHECK(ReduceSumInput(matchedNode, bDimsToSum, 1, true) != SUCCESS,
                      OPS_LOG_W(kPassName.c_str(), "failed to process input B ReduceSum"), return FAILED);
    SumproductPair(matchedNode, sumDims, true);
    return SUCCESS;
}

ge::Status EinsumPass::ExecuteDynamicFuzz(const GNode& matchedNode, size_t numOps, int64_t permIndex, int64_t outNumDim,
                                          std::vector<int64_t>& dimCounts)
{
    std::vector<int64_t> sumDims;
    if (numOps == kMaxInputNum) {
        FUSION_PASS_CHECK(ReduceAndSumPairForDynamic(matchedNode, permIndex, outNumDim, dimCounts, sumDims) != SUCCESS,
                          OPS_LOG_W(kPassName.c_str(), "failed to reduce and sum pair inputs"), return FAILED);
    }
    if (permIndex - outNumDim > 0) {
        if (numOps == 1) {
            std::vector<int64_t> reduceDims(permIndex - outNumDim);
            std::iota(reduceDims.begin(), reduceDims.end(), outNumDim);
            FUSION_PASS_CHECK(ReduceSumInput(matchedNode, reduceDims, 0, false) != SUCCESS,
                              OPS_LOG_W(kPassName.c_str(), "failed to process input A ReduceSum"), return FAILED);
            auto xDesc = GetPrevOutputDesc(matchedNode, 0);
            TensorDesc outDesc;
            FUSION_PASS_CHECK(matchedNode.GetOutputDesc(0, outDesc) != GRAPH_SUCCESS,
                              OPS_LOG_E(kPassName.c_str(), "Failed to get output desc."), return FAILED);
            if (xDesc->GetDataType() != outDesc.GetDataType()) {
                FUSION_PASS_CHECK(CastNode(matchedNode, outDesc.GetDataType(), 0) != SUCCESS,
                                  OPS_LOG_E(kPassName.c_str(), "failed to cast to origin dtype"), return FAILED);
            }
        } else if (numOps == kMaxInputNum && (sumDims.empty() || IsGNodeType(gathershapeNode_, kGatherShapes))) {
            auto mmDesc = GetPrevOutputDescAfterBmm(matchedNode);
            auto outDims = mmDesc->GetShape().GetDims();
            size_t reshapeDim = (outNumDim > 0) ? static_cast<size_t>(outNumDim - 1) : 0;
            outDims.erase(outDims.begin() + outNumDim, outDims.begin() + permIndex);
            GNode prevNode = bmmOutputNodes_.empty() ? batchmatmulNode_ : bmmOutputNodes_.back();
            GNode reshapeNode = CreateReshapeNode(
                *builder_, "reshape_dyn_" + std::to_string(reshapeSeq_++), GetTensorHolder(*builder_, prevNode, 0),
                true, outDims, static_cast<int32_t>(reshapeDim), static_cast<int32_t>(permIndex - 1));
            SetReshapeOutputDesc(reshapeNode, outDims);
            bmmOutputNodes_.emplace_back(reshapeNode);
        }
    }
    return SUCCESS;
}

// Create subgraph input holders for the dynamic decomposition (split from
// SplitDynamicFuzzScene); int32 inputs are forced to dynamic dims because int32 static
// input fails to compile with static batchmatmul (aligned with legacy UpdateInputDimDynamic)
bool EinsumPass::CreateDynamicInputHolders(es::EsGraphBuilder& builder,
                                           const std::vector<SubgraphInput>& subgraphInputs)
{
    auto inputInfo = CollectInputInfo(subgraphInputs);
    if (!inputInfo.valid) {
        return false;
    }
    subgraphInputHolders_.clear();
    forcedInputDims_.assign(subgraphInputs.size(), {});
    if (subgraphInputs.size() == kMaxInputNum) {
        for (size_t idx = 0; idx < subgraphInputs.size(); ++idx) {
            if (inputInfo.dtypes[idx] == ge::DT_INT32) {
                forcedInputDims_[idx].assign(inputInfo.dims[idx].size(), -1);
            }
        }
    }
    for (size_t idx = 0; idx < subgraphInputs.size(); ++idx) {
        const auto& holderDims = forcedInputDims_[idx].empty() ? inputInfo.dims[idx] : forcedInputDims_[idx];
        auto holder = builder.CreateInput(static_cast<int64_t>(idx), ("x" + std::to_string(idx)).c_str(),
                                          inputInfo.dtypes[idx], inputInfo.formats[idx], holderDims);
        subgraphInputHolders_.push_back(holder);
    }
    return true;
}

GraphUniqPtr EinsumPass::SplitDynamicFuzzScene(const std::string& equation,
                                               const std::vector<SubgraphInput>& subgraphInputs,
                                               const GNode& matchedNode)
{
    OPS_LOG_D(kPassName.c_str(), "SplitDynamicFuzzScene equation=%s", equation.c_str());
    es::EsGraphBuilder localBuilder("einsum_dynamic_generic");
    builder_ = &localBuilder;
    graphPtr_ = localBuilder.GetCGraphBuilder()->GetGraph();
    FUSION_PASS_CHECK(!CreateDynamicInputHolders(localBuilder, subgraphInputs),
                      OPS_LOG_E(kPassName.c_str(), "CollectInputInfo failed."), return nullptr);
    const auto numOps = subgraphInputs.size();
    bmmInputNodes_.resize(numOps);
    std::vector<std::vector<uint8_t>> opLabels(numOps);
    int64_t ellNumDim = 0;
    int64_t permIndex = 0;
    int64_t outNumDim = 0;
    std::vector<int64_t> dimCounts;
    if (PrepareDynamicLabels(equation, matchedNode, numOps, opLabels, ellNumDim, permIndex, outNumDim, dimCounts) !=
        SUCCESS) {
        return nullptr;
    }
    if (ExecuteDynamicFuzz(matchedNode, numOps, permIndex, outNumDim, dimCounts) != SUCCESS) {
        return nullptr;
    }
    // Select final output node and build replacement graph
    GNode lastNode = ResolveLastOutputNode();
    FUSION_PASS_CHECK(!IsGNodeValid(lastNode), OPS_LOG_E(kPassName.c_str(), "Failed to resolve last output node."),
                      return nullptr;);
    TensorDesc outputDesc;
    FUSION_PASS_CHECK(matchedNode.GetOutputDesc(0, outputDesc) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName.c_str(), "Failed to get output desc."), return nullptr);
    KeepComputedDtypeAndUpdateOutputDesc(lastNode, outputDesc);
    auto lastHolder = GetTensorHolder(localBuilder, lastNode, 0);
    auto replaceGraph = localBuilder.BuildAndReset({lastHolder});
    FUSION_PASS_CHECK(InferShape(replaceGraph, subgraphInputs) != SUCCESS,
                      OPS_LOG_E(kPassName.c_str(), "InferShape for SplitDynamicFuzzScene failed."), return nullptr;);
    return replaceGraph;
}

// ============================================================================
// Equation parsing helpers
// ============================================================================

// ============================================================================
// Generic decomposition helpers - prev output desc/node access
// ============================================================================

std::shared_ptr<TensorDesc> EinsumPass::GetPrevOutputDesc(const GNode& matchedNode, size_t idx) const
{
    if (bmmInputNodes_[idx].empty()) {
        TensorDesc desc;
        if (matchedNode.GetInputDesc(idx, desc) != GRAPH_SUCCESS) {
            OPS_LOG_W(kPassName.c_str(), "Failed to get input desc %zu.", idx);
        }
        // Int32 input is forced to dynamic shape in dynamic decomposition (same as legacy)
        if (idx < forcedInputDims_.size() && !forcedInputDims_[idx].empty()) {
            desc.SetShape(Shape(forcedInputDims_[idx]));
            desc.SetOriginShape(Shape(forcedInputDims_[idx]));
        }
        return std::make_shared<TensorDesc>(desc);
    }
    TensorDesc desc;
    if (bmmInputNodes_[idx].back().GetOutputDesc(0, desc) != GRAPH_SUCCESS) {
        OPS_LOG_W(kPassName.c_str(), "Failed to get output desc %zu.", idx);
    }
    return std::make_shared<TensorDesc>(desc);
}

std::shared_ptr<TensorDesc> EinsumPass::GetPrevOutputDescAfterBmm(const GNode& matchedNode) const
{
    if (bmmOutputNodes_.empty()) {
        if (!IsGNodeValid(batchmatmulNode_)) {
            return GetPrevOutputDesc(matchedNode, 0);
        }
        TensorDesc desc;
        if (batchmatmulNode_.GetOutputDesc(0, desc) != GRAPH_SUCCESS) {
            OPS_LOG_W(kPassName.c_str(), "Failed to get batchmatmul output desc.");
        }
        return std::make_shared<TensorDesc>(desc);
    }
    TensorDesc desc;
    if (bmmOutputNodes_.back().GetOutputDesc(0, desc) != GRAPH_SUCCESS) {
        OPS_LOG_W(kPassName.c_str(), "Failed to get bmm output desc.");
    }
    return std::make_shared<TensorDesc>(desc);
}
void EinsumPass::SplitStr2Vector(const std::string& input, const std::string& delimiter,
                                 std::vector<std::string>& output) const
{
    auto delimiterLen = delimiter.size();
    std::string::size_type currPos = 0;
    std::string::size_type nextPos = input.find(delimiter, currPos);
    while (nextPos != std::string::npos) {
        output.emplace_back(std::move(input.substr(currPos, nextPos - currPos)));
        currPos = nextPos + delimiterLen;
        nextPos = input.find(delimiter, currPos);
    }
    if (currPos < input.size()) {
        output.emplace_back(std::move(input.substr(currPos)));
    }
}

std::string EinsumPass::FilterInvalidLabel(const std::string& equation) const
{
    std::string res;
    for (char label : equation) {
        if (!isalpha(label) && label != '.') {
            continue;
        }
        res += label;
    }
    return res;
}

void EinsumPass::ParseEquation(const std::string& equation, std::vector<std::string>& inEquations,
                               std::string& outEquation)
{
    std::vector<std::string> inputsAndOutput;
    SplitStr2Vector(equation, "->", inputsAndOutput);
    FUSION_PASS_CHECK(inputsAndOutput.size() != kEquationPartNum,
                      OPS_LOG_E(kPassName.c_str(), "Invalid equation format: %s", equation.c_str()), return;);

    outEquation = FilterInvalidLabel(inputsAndOutput[1]);
    std::vector<std::string> inputsOriginEquation;
    SplitStr2Vector(inputsAndOutput[0], ",", inputsOriginEquation);
    FUSION_PASS_CHECK(inputsOriginEquation.size() != 1 && inputsOriginEquation.size() != kMaxInputNum,
                      OPS_LOG_E(kPassName.c_str(), "equation[%s] only supports 1 or 2 inputs.", equation.c_str()),
                      return;);
    for (auto& inputEquation : inputsOriginEquation) {
        inEquations.emplace_back(FilterInvalidLabel(inputEquation));
    }

    std::set<char> labels;
    LabelCount input0LabelCount;
    LabelCount input1LabelCount;
    LabelCount outputLabelCount;
    CountLabels(inEquations[0], input0LabelCount, labels);
    if (inEquations.size() > 1) {
        CountLabels(inEquations[1], input1LabelCount, labels);
    }
    CountLabels(outEquation, outputLabelCount, labels);
    MapDimensionType(labels, input0LabelCount, input1LabelCount, outputLabelCount);
}

void EinsumPass::CountLabels(const std::string& equation, LabelCount& labelCount, std::set<char>& labels) const
{
    for (size_t pos = 0; pos < equation.size(); ++pos) {
        char label = equation[pos];
        (void)labels.insert(label);
        if (label == '.') {
            pos += (kBroadCastLabelLen - 1);
        }
        auto it = labelCount.find(label);
        if (it == labelCount.end()) {
            (void)labelCount.insert({label, 1});
        } else {
            it->second += 1;
        }
    }
}

void EinsumPass::MapDimensionType(const std::set<char>& labels, const LabelCount& input0LabelCount,
                                  const LabelCount& input1LabelCount, const LabelCount& outputLabelCount)
{
    for (auto label : labels) {
        EinsumDimensionType dimType = kBroadCast;
        if (label == '.') {
            (void)dimTypesMap_.insert({label, dimType});
            continue;
        }
        bool inInput0 = input0LabelCount.find(label) != input0LabelCount.end();
        bool inInput1 = input1LabelCount.find(label) != input1LabelCount.end();
        bool inOutput = outputLabelCount.find(label) != outputLabelCount.end();
        if (inInput0 && inInput1) {
            dimType = inOutput ? kBatch : kContract;
        } else if (inInput0 || inInput1) {
            dimType = inOutput ? kFree : kReduce;
        }
        (void)dimTypesMap_.insert({label, dimType});
    }
}

void EinsumPass::CollectDimensionType(size_t dimNum, const std::string& equation,
                                      DimensionType2LabelInfo& labelInfo) const
{
    labelInfo.resize(kDimTypeNum);
    std::string equationCopy(equation);
    auto pos = equationCopy.find(kBroadCastLabel);
    if (pos != std::string::npos) {
        FUSION_PASS_CHECK(
            dimNum < (equationCopy.size() - kBroadCastLabelLen),
            OPS_LOG_E(kPassName.c_str(), "equation[%s] is invalid, dim size[%zu].", equation.c_str(), dimNum), return;);
        size_t broadCastLen = dimNum - (equationCopy.size() - kBroadCastLabelLen);
        std::string targetBroadCastStr(broadCastLen, '.');
        equationCopy.replace(equationCopy.begin() + pos, equationCopy.begin() + pos + kBroadCastLabelLen,
                             targetBroadCastStr);
    } else if (equationCopy.size() != dimNum) {
        OPS_LOG_E(kPassName.c_str(), "equation[%s] label count[%zu] != dim size[%zu].", equation.c_str(),
                  equationCopy.size(), dimNum);
        return;
    }

    for (size_t idx = 0; idx < equationCopy.size(); ++idx) {
        char label = equationCopy[idx];
        auto it = dimTypesMap_.find(label);
        if (it != dimTypesMap_.end()) {
            labelInfo[it->second].labels.append({label});
            labelInfo[it->second].indices.push_back(static_cast<int32_t>(idx));
        }
    }
}

void EinsumPass::ReorderAxes(DimensionType2LabelInfo& labelInfos) const
{
    for (size_t dimType = 0; dimType < labelInfos.size(); ++dimType) {
        auto& labels = labelInfos[dimType].labels;
        auto& indices = labelInfos[dimType].indices;
        if (labels.size() != indices.size()) {
            continue;
        }
        size_t pos = 0;
        while (pos < labels.size()) {
            size_t currPos = pos + 1;
            size_t nextPos = labels.find(labels[pos], currPos);
            while (nextPos != std::string::npos) {
                labels.erase(nextPos, 1);
                labels.insert(currPos, 1, labels[pos]);
                int32_t index = indices[nextPos];
                indices.erase(indices.begin() + nextPos, indices.begin() + nextPos + 1);
                indices.insert(indices.begin() + currPos, index);
                ++currPos;
                nextPos = labels.find(labels[pos], currPos);
            }
            pos = currPos;
        }
    }
}

void EinsumPass::CompareAxes(EinsumDimensionType dimType, const DimensionType2LabelInfo& targetLabelInfo,
                             DimensionType2LabelInfo& inputLabelInfo) const
{
    auto& inputLabels = inputLabelInfo[dimType].labels;
    auto& inputIndices = inputLabelInfo[dimType].indices;
    const auto& outputLabels = targetLabelInfo[dimType].labels;
    if (inputLabels.empty() || outputLabels.empty()) {
        return;
    }

    std::string newLabels;
    std::vector<int32_t> newIndices;
    newLabels.reserve(inputLabels.size());
    newIndices.reserve(inputIndices.size());
    size_t pos = 0;
    while (pos < outputLabels.size()) {
        char label = outputLabels[pos];
        size_t firstPos = inputLabels.find_first_of(label);
        if (firstPos != std::string::npos) {
            size_t endPos = inputLabels.find_first_not_of(label, firstPos);
            size_t repeatNum = (endPos != std::string::npos) ? endPos - firstPos : inputLabels.size() - firstPos;
            newLabels.insert(newLabels.size(), repeatNum, label);
            newIndices.insert(newIndices.end(), inputIndices.begin() + firstPos,
                              inputIndices.begin() + firstPos + repeatNum);
        }
        do {
            ++pos;
        } while (pos < newLabels.size() && outputLabels[pos] == label);
    }

    if (inputLabels.size() != newLabels.size()) {
        pos = 0;
        while (pos < inputLabels.size()) {
            char label = inputLabels[pos];
            size_t newLabelsPos = newLabels.find_first_of(label);
            if (newLabelsPos == std::string::npos) {
                size_t endPos = inputLabels.find_first_not_of(label, pos);
                size_t repeatNum = (endPos != std::string::npos) ? endPos - pos : inputLabels.size() - pos;
                size_t prevLabelPos = (pos > 0) ? newLabels.rfind(inputLabels[pos - 1]) + 1 : 0;
                newLabels.insert(prevLabelPos, repeatNum, label);
                newIndices.insert(newIndices.begin() + prevLabelPos, inputIndices.begin() + pos,
                                  inputIndices.begin() + pos + repeatNum);
                pos += repeatNum;
                continue;
            }
            do {
                ++pos;
            } while (pos < inputLabels.size() && inputLabels[pos] == label);
        }
    }

    inputLabels.swap(newLabels);
    newIndices.swap(inputIndices);
}

bool EinsumPass::GetTransposeDstEquation(const std::string& oriEquation, const DimensionType2LabelInfo& labelInfos,
                                         std::vector<int32_t>& permList, bool& inputFreeContractOrder,
                                         std::string& dstEquation) const
{
    for (auto labelIter = oriEquation.rbegin(); labelIter != oriEquation.rend(); ++labelIter) {
        auto it = dimTypesMap_.find(*labelIter);
        if (it != dimTypesMap_.end()) {
            if (it->second == kFree) {
                inputFreeContractOrder = false;
                break;
            } else if (it->second == kContract) {
                inputFreeContractOrder = true;
                break;
            }
        }
    }

    std::string broadCastLabels;
    std::vector<int32_t> broadCastIndices;
    std::string oriEquationCopy(oriEquation);
    if (!labelInfos[kBroadCast].indices.empty()) {
        broadCastLabels.assign(kBroadCastLabel);
        broadCastIndices = labelInfos[kBroadCast].indices;
    } else {
        auto pos = oriEquationCopy.find(kBroadCastLabel);
        if (pos != std::string::npos) {
            oriEquationCopy.erase(pos, kBroadCastLabelLen);
        }
    }

    auto& batchLabels = labelInfos[kBatch].labels;
    auto& batchIndices = labelInfos[kBatch].indices;
    auto& freeLabels = labelInfos[kFree].labels;
    auto& freeIndices = labelInfos[kFree].indices;
    auto& contractLabels = labelInfos[kContract].labels;
    auto& contractIndices = labelInfos[kContract].indices;
    auto& reduceLabels = labelInfos[kReduce].labels;
    auto& reduceIndices = labelInfos[kReduce].indices;

    dstEquation.reserve(oriEquation.size());
    dstEquation.append(broadCastLabels).append(batchLabels);
    permList.insert(permList.end(), broadCastIndices.begin(), broadCastIndices.end());
    permList.insert(permList.end(), batchIndices.begin(), batchIndices.end());
    if (inputFreeContractOrder) {
        dstEquation.append(freeLabels).append(contractLabels).append(reduceLabels);
        permList.insert(permList.end(), freeIndices.begin(), freeIndices.end());
        permList.insert(permList.end(), contractIndices.begin(), contractIndices.end());
    } else {
        dstEquation.append(contractLabels).append(freeLabels).append(reduceLabels);
        permList.insert(permList.end(), contractIndices.begin(), contractIndices.end());
        permList.insert(permList.end(), freeIndices.begin(), freeIndices.end());
    }
    permList.insert(permList.end(), reduceIndices.begin(), reduceIndices.end());
    return dstEquation != oriEquationCopy;
}

void EinsumPass::CheckMergeFreeLabels(const std::string& outEquation)
{
    if (inputLabelInfos_.size() < kMaxInputNum) {
        return;
    }
    size_t input0BatchNum = inputLabelInfos_[0][kBatch].indices.size();
    size_t input1BatchNum = inputLabelInfos_[1][kBatch].indices.size();
    if (input0BatchNum > 0 || input1BatchNum > 0) {
        return;
    }

    std::string outEquationCopy(outEquation);
    auto pos = outEquationCopy.find(kBroadCastLabel);
    if (pos != std::string::npos) {
        outEquationCopy.erase(pos, kBroadCastLabelLen);
    }

    size_t input0FreeNum = inputLabelInfos_[0][kFree].indices.size();
    size_t input1FreeNum = inputLabelInfos_[1][kFree].indices.size();
    size_t input0BroadCastNum = inputLabelInfos_[0][kBroadCast].indices.size();
    size_t input1BroadCastNum = inputLabelInfos_[1][kBroadCast].indices.size();
    if (input0FreeNum == 1 && input1FreeNum > 1 && inputFreeContractOrders_[1]) {
        std::string freeLabels(inputLabelInfos_[1][kFree].labels);
        freeLabels.append(inputLabelInfos_[0][kFree].labels);
        mergeFreeLabels_ = !(input0BroadCastNum == 0 && freeLabels == outEquationCopy);
    } else if (input0FreeNum > 1 && input1FreeNum == 1 && inputFreeContractOrders_[0]) {
        std::string freeLabels(inputLabelInfos_[0][kFree].labels);
        freeLabels.append(inputLabelInfos_[1][kFree].labels);
        mergeFreeLabels_ = !(input1BroadCastNum == 0 && freeLabels == outEquationCopy);
    }
}

void EinsumPass::CheckBatchMatmulSwapInputs()
{
    if (inputLabelInfos_.size() < kMaxInputNum) {
        return;
    }
    auto& input1FreeLabelInfos = inputLabelInfos_[1][kFree];
    if (input1FreeLabelInfos.labels.empty()) {
        return;
    }

    const auto& outputFreeLabelInfos = outputLabelInfo_[kFree];
    bool freeLabelFromInput1 = true;
    for (char label : outputFreeLabelInfos.labels) {
        if (freeLabelFromInput1) {
            freeLabelFromInput1 = input1FreeLabelInfos.labels.find(label) != std::string::npos;
        } else {
            if (input1FreeLabelInfos.labels.find(label) != std::string::npos) {
                return;
            }
        }
    }
    swapBmmInputs_ = true;
}

// ============================================================================
// Static decomposition steps
// ============================================================================

Status EinsumPass::TransposeInput(const std::vector<std::string>& inEquations, const GNode& matchedNode)
{
    inputFreeContractOrders_.resize(inEquations.size());
    inputLabelInfos_.resize(inEquations.size());
    bmmInputNodes_.resize(inEquations.size());
    for (size_t idx = 0; idx < inEquations.size(); ++idx) {
        auto xDesc = GetPrevOutputDesc(matchedNode, idx);
        auto inputDims = xDesc->GetShape().GetDims();
        auto dimNum = xDesc->GetShape().GetDimNum();
        CollectDimensionType(dimNum, inEquations[idx], inputLabelInfos_[idx]);
        ReorderAxes(inputLabelInfos_[idx]);
        CompareAxes(kBatch, outputLabelInfo_, inputLabelInfos_[idx]);
        CompareAxes(kFree, outputLabelInfo_, inputLabelInfos_[idx]);
        if (idx > 0) {
            CompareAxes(kContract, inputLabelInfos_[0], inputLabelInfos_[idx]);
        }

        std::vector<int32_t> permList;
        bool inputFreeContractOrder = false;
        std::string dstEquation;
        if (GetTransposeDstEquation(inEquations[idx], inputLabelInfos_[idx], permList, inputFreeContractOrder,
                                    dstEquation)) {
            inputFreeContractOrders_[idx] = inputFreeContractOrder;
            bool isDynamic = IsUnknownShape(xDesc->GetShape());
            EsTensorHolder prevHolder = (bmmInputNodes_[idx].empty()) ?
                                            subgraphInputHolders_[idx] :

                                            GetTensorHolder(*builder_, bmmInputNodes_[idx].back(), 0);
            GNode transNode = CreateTransposeNode(*builder_, "transpose_" + std::to_string(transposeSeq_++), prevHolder,
                                                  isDynamic, supportL12btBf16_, permList);
            std::vector<int64_t> transOutDims;
            transOutDims.reserve(permList.size());
            for (int32_t p : permList) {
                if (static_cast<size_t>(p) < inputDims.size()) {
                    transOutDims.push_back(inputDims[p]);
                }
            }
            TensorDesc transOutDesc(Shape(transOutDims), xDesc->GetFormat(), xDesc->GetDataType());
            transOutDesc.SetOriginFormat(xDesc->GetFormat());
            transOutDesc.SetOriginShape(Shape(transOutDims));
            transNode.UpdateOutputDesc(0, transOutDesc);
            bmmInputNodes_[idx].emplace_back(transNode);
        } else {
            inputFreeContractOrders_[idx] = inputFreeContractOrder;
        }
    }
    return SUCCESS;
}

Status EinsumPass::StrideInput(const std::vector<std::string>& inEquations) const
{
    for (size_t idx = 0; idx < inEquations.size(); ++idx) {
        const std::string& inEquation = inEquations[idx];
        for (size_t pos = 0; pos < inEquation.size(); ++pos) {
            char label = inEquation[pos];
            if (label == '.') {
                pos += (kBroadCastLabelLen - 1);
                continue;
            }
            FUSION_PASS_CHECK(
                inEquation.find(label, pos + 1) != std::string::npos,
                OPS_LOG_E(kPassName.c_str(), "not support stride input now, equation: %s.", inEquation.c_str()),
                return FAILED;);
        }
    }
    return SUCCESS;
}

Status EinsumPass::ReduceInput(const std::vector<std::string>& inEquations, const GNode& matchedNode)
{
    for (size_t idx = 0; idx < inEquations.size(); ++idx) {
        auto& indices = inputLabelInfos_[idx][kReduce].indices;
        if (indices.empty()) {
            continue;
        }
        auto xDesc = GetPrevOutputDesc(matchedNode, idx);
        auto inputDims = xDesc->GetShape().GetDims();
        size_t dimNum = inputDims.size();
        FUSION_PASS_CHECK(dimNum < indices.size(),
                          OPS_LOG_E(kPassName.c_str(), "prev output dim number[%zu] less than reduce dim num[%zu].",
                                    dimNum, indices.size()),
                          return FAILED;);

        std::vector<int64_t> axes;
        axes.reserve(indices.size());
        for (size_t axis = dimNum - indices.size(); axis < dimNum; ++axis) {
            axes.push_back(static_cast<int64_t>(axis));
        }

        GNode reduceNode = CreateReduceSumNode(graphPtr_, *builder_, "reduce_" + std::to_string(reduceSeq_++),
                                               supportL12btBf16_, axes, false);
        GNode prevNode = (bmmInputNodes_[idx].empty()) ? *subgraphInputHolders_[idx].GetProducer() :
                                                         bmmInputNodes_[idx].back();
        FUSION_PASS_CHECK(AddEdgeAndUpdatePeerDesc(*graphPtr_, prevNode, 0, reduceNode, 0) != GRAPH_SUCCESS,
                          OPS_LOG_E(kPassName.c_str(), "AddEdgeAndUpdatePeerDesc failed."), return FAILED);
        std::vector<int64_t> reduceOutDims;
        for (size_t i = 0; i < dimNum; ++i) {
            if (std::find(axes.begin(), axes.end(), static_cast<int64_t>(i)) == axes.end()) {
                reduceOutDims.push_back(inputDims[i]);
            }
        }
        TensorDesc reduceOutDesc(Shape(reduceOutDims), xDesc->GetFormat(), xDesc->GetDataType());
        reduceOutDesc.SetOriginFormat(xDesc->GetFormat());
        reduceOutDesc.SetOriginShape(Shape(reduceOutDims));
        reduceNode.UpdateOutputDesc(0, reduceOutDesc);
        bmmInputNodes_[idx].emplace_back(reduceNode);
    }
    return SUCCESS;
}

// Append broadcast/batch and merged free/contract dims for one input (split from ReshapeInput)
void EinsumPass::AppendMergedInputDims(std::vector<int64_t>& newDims, const DimensionType2LabelInfo& inputLabelInfo,
                                       const std::vector<EinsumDimensionType>& checkOrders,
                                       const std::vector<int64_t>& oriDims) const
{
    for (EinsumDimensionType dimType : checkOrders) {
        auto& indices = inputLabelInfo[dimType].indices;
        if (mergeFreeLabels_ || dimType != kFree) {
            int64_t acc = 1;
            for (auto index : indices) {
                if (static_cast<size_t>(index) < oriDims.size()) {
                    acc = GetDimMulValue(acc, oriDims[index]);
                }
            }
            newDims.push_back(acc);
        } else {
            for (auto index : indices) {
                if (static_cast<size_t>(index) < oriDims.size()) {
                    newDims.push_back(oriDims[index]);
                }
            }
        }
    }
}

Status EinsumPass::ReshapeInput(const std::vector<std::string>& inEquations, const std::string& outEquation,
                                const GNode& matchedNode)
{
    if (inEquations.size() == 1) {
        return SUCCESS;
    }

    CheckMergeFreeLabels(outEquation);
    for (size_t idx = 0; idx < inEquations.size(); ++idx) {
        auto xDesc = GetPrevOutputDesc(matchedNode, idx);
        auto dims = xDesc->GetShape().GetDims();
        TensorDesc oriInputDesc;
        FUSION_PASS_CHECK(matchedNode.GetInputDesc(idx, oriInputDesc) != GRAPH_SUCCESS,
                          OPS_LOG_E(kPassName.c_str(), "Failed to get input desc %zu.", idx), return FAILED;);
        auto oriDims = oriInputDesc.GetShape().GetDims();
        auto& inputLabelInfo = inputLabelInfos_[idx];
        std::vector<EinsumDimensionType> checkOrders{kFree, kContract};
        if (!inputFreeContractOrders_[idx]) {
            checkOrders.assign({kContract, kFree});
        }

        std::vector<int64_t> newDims;
        newDims.reserve(dims.size());
        size_t broadcastBatchNum = inputLabelInfo[kBroadCast].indices.size() + inputLabelInfo[kBatch].indices.size();
        if (broadcastBatchNum <= dims.size()) {
            newDims.assign(dims.begin(), dims.begin() + broadcastBatchNum);
        }
        AppendMergedInputDims(newDims, inputLabelInfo, checkOrders, oriDims);

        if (newDims == dims) {
            continue;
        }

        bool isDynamic = IsUnknownShape(xDesc->GetShape());
        EsTensorHolder prevHolder = (bmmInputNodes_[idx].empty()) ?
                                        subgraphInputHolders_[idx] :

                                        GetTensorHolder(*builder_, bmmInputNodes_[idx].back(), 0);
        GNode reshapeNode = CreateReshapeNode(*builder_, "reshape_" + std::to_string(reshapeSeq_++), prevHolder,
                                              isDynamic, newDims);
        TensorDesc reshapeOutDesc(Shape(newDims), xDesc->GetFormat(), xDesc->GetDataType());
        reshapeOutDesc.SetOriginFormat(xDesc->GetFormat());
        reshapeOutDesc.SetOriginShape(Shape(newDims));
        reshapeNode.UpdateOutputDesc(0, reshapeOutDesc);
        bmmInputNodes_[idx].emplace_back(reshapeNode);
    }
    return SUCCESS;
}

Status EinsumPass::DoBatchMatmul(const std::vector<std::string>& inEquations, const GNode& matchedNode)
{
    if (inEquations.size() == 1) {
        return SUCCESS;
    }

    bool adjX1 = !(inputFreeContractOrders_[0]);
    bool adjX2 = inputFreeContractOrders_[1];
    auto x1Desc = GetPrevOutputDesc(matchedNode, 0);
    auto x2Desc = GetPrevOutputDesc(matchedNode, 1);
    bool useMatmul = (x1Desc->GetShape().GetDimNum() == kDim2D) && (x2Desc->GetShape().GetDimNum() == kDim2D);

    CheckBatchMatmulSwapInputs();

    EsTensorHolder x1Holder = bmmInputNodes_[0].empty() ? subgraphInputHolders_[0] :
                                                          GetTensorHolder(*builder_, bmmInputNodes_[0].back(), 0);
    EsTensorHolder x2Holder = bmmInputNodes_[1].empty() ? subgraphInputHolders_[1] :
                                                          GetTensorHolder(*builder_, bmmInputNodes_[1].back(), 0);

    if (swapBmmInputs_) {
        adjX1 = !(inputFreeContractOrders_[1]);
        adjX2 = inputFreeContractOrders_[0];
        const std::string& opType = useMatmul ? kMatMul : kBatchMatMulV2;
        auto bmmOutput = CreateMatMulNode(*builder_, opType, x2Holder, x1Holder, adjX1, adjX2);
        batchmatmulNode_ = *bmmOutput.GetProducer();
    } else {
        const std::string& opType = useMatmul ? kMatMul : kBatchMatMulV2;
        auto bmmOutput = CreateMatMulNode(*builder_, opType, x1Holder, x2Holder, adjX1, adjX2);
        batchmatmulNode_ = *bmmOutput.GetProducer();
    }

    FUSION_PASS_CHECK(!CopyOtherAttrs(matchedNode, batchmatmulNode_, kPassName),

                      OPS_LOG_E(kPassName.c_str(), "Copy other attrs failed."), return FAILED;);

    // Output desc (batch dims + m + n) has already been computed and set inside
    // CreateMatMulNode with the same formula, no need to recompute it here.

    return SUCCESS;
}

Status EinsumPass::ReshapeOutput(const GNode& matchedNode, std::string& curOutEquation,
                                 const std::vector<std::string>& inEquations)
{
    if (inEquations.size() == 1) {
        return SUCCESS;
    }

    bool needReshape = false;
    auto bmmDesc = GetPrevOutputDescAfterBmm(matchedNode);
    auto bmmDims = bmmDesc->GetShape().GetDims();
    auto bmmOriLen = bmmDims.size();
    CalcBatchMatmulOutput(inEquations.size(), matchedNode, needReshape, curOutEquation, bmmDims);
    if (!needReshape && bmmOriLen == bmmDims.size()) {
        return SUCCESS;
    }
    if (!mergeFreeLabels_) {
        return SUCCESS;
    }

    bool isDynamic = IsUnknownShape(bmmDesc->GetShape());
    GNode prevNode = bmmOutputNodes_.empty() ? batchmatmulNode_ : bmmOutputNodes_.back();
    GNode reshapeNode = CreateReshapeNode(*builder_, "reshape_out_" + std::to_string(reshapeSeq_++),
                                          GetTensorHolder(*builder_, prevNode, 0), isDynamic, bmmDims);

    TensorDesc reshapeOutDesc(Shape(bmmDims), bmmDesc->GetFormat(), bmmDesc->GetDataType());
    reshapeOutDesc.SetOriginFormat(bmmDesc->GetFormat());
    reshapeOutDesc.SetOriginShape(Shape(bmmDims));
    reshapeNode.UpdateOutputDesc(0, reshapeOutDesc);

    bmmOutputNodes_.emplace_back(reshapeNode);
    return SUCCESS;
}

Status EinsumPass::InflatedOutput(const std::string& outEquation) const
{
    std::string outEquationBak(outEquation);
    auto startIdx = outEquationBak.find(kBroadCastLabel);
    if (startIdx != std::string::npos) {
        outEquationBak.erase(startIdx, kBroadCastLabelLen);
    }
    std::set<char> outLabelSet(outEquationBak.begin(), outEquationBak.end());
    FUSION_PASS_CHECK(outEquationBak.size() != outLabelSet.size(),
                      OPS_LOG_E(kPassName.c_str(), "not support inflated now, equation: %s.", outEquation.c_str()),
                      return FAILED;);
    return SUCCESS;
}

Status EinsumPass::TransposeOutput(const std::string& outEquation, const GNode& matchedNode,
                                   std::string& curOutEquation)
{
    if (curOutEquation == outEquation) {
        return SUCCESS;
    }

    auto xDesc = GetPrevOutputDescAfterBmm(matchedNode);
    auto xDimSize = xDesc->GetShape().GetDimNum();
    bool ellDimFlag = outEquation.find(kBroadCastLabel) != std::string::npos;
    auto outLabelDim = ellDimFlag ? outEquation.size() - kBroadCastLabelLen : outEquation.size();
    auto ellDimSize = xDimSize - outLabelDim;
    auto ellDimOffset = ellDimFlag ? kBroadCastLabelLen : 0;
    std::vector<int32_t> permList;
    for (size_t i = 0; i < outEquation.size(); i++) {
        if (outEquation[i] == '.') {
            std::vector<int32_t> ellIndice(ellDimSize);
            std::iota(ellIndice.begin(), ellIndice.end(), 0);
            permList.insert(permList.end(), ellIndice.begin(), ellIndice.end());
            i += (kBroadCastLabelLen - 1);
        } else {
            auto curIdx = static_cast<int32_t>(curOutEquation.find_first_of(outEquation[i]) - ellDimOffset +
                                               ellDimSize);
            permList.insert(permList.end(), curIdx);
        }
    }

    std::vector<int32_t> noTransposeList(xDimSize);
    std::iota(noTransposeList.begin(), noTransposeList.end(), 0);
    if (permList == noTransposeList) {
        return SUCCESS;
    }

    bool isDynamic = IsUnknownShape(xDesc->GetShape());
    GNode prevNode = bmmOutputNodes_.empty() ? batchmatmulNode_ : bmmOutputNodes_.back();
    GNode transNode = CreateTransposeNode(*builder_, "transpose_out_" + std::to_string(transposeSeq_++),
                                          GetTensorHolder(*builder_, prevNode, 0), isDynamic, supportL12btBf16_,
                                          permList);

    auto xDims = xDesc->GetShape().GetDims();
    std::vector<int64_t> transOutDims;
    for (int32_t p : permList) {
        if (static_cast<size_t>(p) < xDims.size()) {
            transOutDims.push_back(xDims[p]);
        }
    }
    TensorDesc transOutDesc(Shape(transOutDims), xDesc->GetFormat(), xDesc->GetDataType());
    transOutDesc.SetOriginFormat(xDesc->GetFormat());
    transOutDesc.SetOriginShape(Shape(transOutDims));
    transNode.UpdateOutputDesc(0, transOutDesc);

    bmmOutputNodes_.emplace_back(transNode);
    return SUCCESS;
}

// Collect free labels/dims of all inputs (split from CalcBatchMatmulOutput)
void EinsumPass::CollectFreeLabelsAndDims(size_t inputNum, const GNode& matchedNode, bool& needReshape,
                                          std::string& outFreeLabels, std::vector<int64_t>& outFreeDim) const
{
    for (size_t idx = 0; idx < inputNum; ++idx) {
        auto& freeIndices = inputLabelInfos_[idx][kFree].indices;
        auto& freeLabels = inputLabelInfos_[idx][kFree].labels;
        if (freeIndices.empty()) {
            continue;
        }
        needReshape = needReshape || freeIndices.size() > 1;
        TensorDesc oriInputDesc;
        FUSION_PASS_CHECK(matchedNode.GetInputDesc(idx, oriInputDesc) != GRAPH_SUCCESS,
                          OPS_LOG_E(kPassName.c_str(), "Failed to get input desc %zu.", idx), return;);
        auto oriDims = oriInputDesc.GetShape().GetDims();
        if (swapBmmInputs_) {
            outFreeLabels.insert(outFreeLabels.begin(), freeLabels.begin(), freeLabels.end());
            for (size_t offset = 0; offset < freeIndices.size(); ++offset) {
                FUSION_PASS_CHECK(static_cast<size_t>(freeIndices[offset]) >= oriDims.size(),
                                  OPS_LOG_E(kPassName.c_str(), "freeIndices[%zu]=%d out of range[%zu].", offset,
                                            freeIndices[offset], oriDims.size()),
                                  return;);
                outFreeDim.insert(outFreeDim.begin() + offset, oriDims[freeIndices[offset]]);
            }
        } else {
            outFreeLabels.insert(outFreeLabels.end(), freeLabels.begin(), freeLabels.end());
            for (auto i : freeIndices) {
                FUSION_PASS_CHECK(static_cast<size_t>(i) >= oriDims.size(),
                                  OPS_LOG_E(kPassName.c_str(), "freeIndices=%d out of range[%zu].", i, oriDims.size()),
                                  return;);
                outFreeDim.push_back(oriDims[i]);
            }
        }
    }
}

void EinsumPass::CalcBatchMatmulOutput(size_t inputNum, const GNode& matchedNode, bool& needReshape,
                                       std::string& bmmOutEquation, std::vector<int64_t>& bmmDims) const
{
    std::string outFreeLabels;
    std::vector<int64_t> outFreeDim;

    CollectFreeLabelsAndDims(inputNum, matchedNode, needReshape, outFreeLabels, outFreeDim);

    std::vector<int64_t> outBatchDim;
    auto& batchIndices = inputLabelInfos_[0][kBatch].indices;
    size_t batchNum = batchIndices.empty() ? 0 : batchIndices.size();
    std::string batchLabels(inputLabelInfos_[0][kBatch].labels);
    TensorDesc oriInputDesc0;
    FUSION_PASS_CHECK(matchedNode.GetInputDesc(0, oriInputDesc0) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName.c_str(), "Failed to get input desc 0."), return;);
    auto oriDimsBatch = oriInputDesc0.GetShape().GetDims();
    for (auto i : batchIndices) {
        FUSION_PASS_CHECK(static_cast<size_t>(i) >= oriDimsBatch.size(),
                          OPS_LOG_E(kPassName.c_str(), "batchIndices=%d out of range[%zu].", i, oriDimsBatch.size()),
                          return;);
        outBatchDim.push_back(oriDimsBatch[i]);
    }

    size_t freeNum = mergeFreeLabels_ ?
                         kMergedFreeDimNum :
                         inputLabelInfos_[0][kFree].indices.size() + inputLabelInfos_[1][kFree].indices.size();
    if (bmmDims.size() >= batchNum + freeNum) {
        bmmDims.erase(bmmDims.end() - batchNum - freeNum, bmmDims.end());
    }
    bmmDims.insert(bmmDims.end(), outBatchDim.begin(), outBatchDim.end());
    bmmDims.insert(bmmDims.end(), outFreeDim.begin(), outFreeDim.end());

    if (bmmOutEquation.find(kBroadCastLabel) != std::string::npos) {
        bmmOutEquation.assign(kBroadCastLabel);
    } else {
        bmmOutEquation.clear();
    }
    bmmOutEquation.append(batchLabels).append(outFreeLabels);
}

// ============================================================================
// Dynamic decomposition steps
// ============================================================================

static uint8_t LabelToSubscript(unsigned char label)
{
    return std::isupper(label) ? label - 'A' : label - 'a' + kNumOfLetters;
}

#ifndef STRIP_ERROR_MESSAGES
static unsigned char SubscriptToLabel(uint8_t s) { return s < kNumOfLetters ? s + 'A' : s + 'a' - kNumOfLetters; }
#endif

Status EinsumPass::InputLabelProcess(const std::string& lhs, size_t numOps, bool& ellInInput,
                                     std::vector<std::vector<uint8_t>>& opLabels)
{
    std::size_t currOp = 0;
    for (std::size_t i = 0; i < lhs.length(); ++i) {
        const unsigned char label = lhs[i];
        switch (label) {
            case ' ':
                break;
            case '.':
                FUSION_PASS_CHECK(
                    ellInInput,
                    OPS_LOG_W(kPassName.c_str(), "found '.' for operand %zu for which an ellipsis was already found",
                              currOp),
                    return GRAPH_NOT_CHANGED;);
                FUSION_PASS_CHECK(
                    !(i + kEllipsisTailLen < lhs.length() && lhs[++i] == '.' && lhs[++i] == '.'),
                    OPS_LOG_W(kPassName.c_str(), "found '.' for operand %zu that is not part of any ellipsis", currOp),
                    return GRAPH_NOT_CHANGED;);
                opLabels[currOp].push_back(kEllipsis);
                ellInInput = true;
                break;
            case ',':
                ++currOp;
                FUSION_PASS_CHECK(currOp >= numOps,
                                  OPS_LOG_W(kPassName.c_str(), "fewer operands were provided than specified"),
                                  return GRAPH_NOT_CHANGED;);
                ellInInput = false;
                break;
            default:
                FUSION_PASS_CHECK(!std::isalpha(label),
                                  OPS_LOG_W(kPassName.c_str(), "invalid subscript at index %zu", i),
                                  return GRAPH_NOT_CHANGED;);
                opLabels[currOp].push_back(LabelToSubscript(label));
        }
    }
    FUSION_PASS_CHECK(currOp != numOps - 1,
                      OPS_LOG_W(kPassName.c_str(), "more operands were provided than specified in the equation"),
                      return GRAPH_NOT_CHANGED;);
    return SUCCESS;
}

Status EinsumPass::LabelCountMap(const GNode& matchedNode, size_t numOps, int64_t& ellNumDim,
                                 std::vector<int64_t>& labelCount, std::vector<std::vector<uint8_t>>& opLabels,
                                 size_t arrowPos)
{
    (void)arrowPos;
    for (size_t i = 0; i < numOps; ++i) {
        const auto labels = opLabels[i];
        TensorDesc inputDesc;
        FUSION_PASS_CHECK(matchedNode.GetInputDesc(i, inputDesc) != GRAPH_SUCCESS,
                          OPS_LOG_E(kPassName.c_str(), "Failed to get input desc %zu.", i), return GRAPH_NOT_CHANGED;);
        int ndims = inputDesc.GetShape().GetDims().size();
        int nlabels = static_cast<int>(labels.size());
        bool hasEllipsis = false;
        for (const auto& label : labels) {
            if (label == kEllipsis) {
                --nlabels;
                hasEllipsis = true;
                ellNumDim = std::max(ellNumDim, static_cast<int64_t>(ndims - nlabels));
            } else {
                ++labelCount[label];
            }
        }
        if (hasEllipsis) {
            FUSION_PASS_CHECK(
                nlabels > ndims,
                OPS_LOG_W(kPassName.c_str(), "subscripts count(%d) > dims(%d) for operand %zu", nlabels, ndims, i),
                return GRAPH_NOT_CHANGED;);
        } else {
            FUSION_PASS_CHECK(
                nlabels != ndims,
                OPS_LOG_W(kPassName.c_str(), "subscripts count(%d) != dims(%d) for operand %zu", nlabels, ndims, i),
                return GRAPH_NOT_CHANGED;);
        }
    }
    return SUCCESS;
}

// Parse the output labels on the right-hand side of the arrow (split from ParseOutputLabel)
static Status ParseOutputRhsLabel(const std::string& equation, size_t arrowPos, int64_t ellNumDim, int64_t& permIndex,
                                  bool& ellInOutput, std::vector<int64_t>& labelPermIndex,
                                  const std::vector<int64_t>& labelCount, int64_t& ellIndex)
{
    const auto rhs = equation.substr(arrowPos + 2);
    const auto lhs = equation.substr(0, arrowPos);
    for (std::size_t i = 0; i < rhs.length(); ++i) {
        const unsigned char label = rhs[i];
        switch (label) {
            case ' ':
                break;
            case '.':
                FUSION_PASS_CHECK(
                    ellInOutput, OPS_LOG_W(kPassName.c_str(), "found '.' for output but an ellipsis was already found"),
                    return GRAPH_NOT_CHANGED;);
                FUSION_PASS_CHECK(!(i + kEllipsisTailLen < rhs.length() && rhs[++i] == '.' && rhs[++i] == '.'),
                                  OPS_LOG_W(kPassName.c_str(), "found '.' for output that is not part of any ellipsis"),
                                  return GRAPH_NOT_CHANGED;);
                ellIndex = permIndex;
                permIndex += ellNumDim;
                ellInOutput = true;
                break;
            default:
                FUSION_PASS_CHECK(!std::isalpha(label),
                                  OPS_LOG_W(kPassName.c_str(), "invalid subscript at index %zu", lhs.size() + 2 + i),
                                  return GRAPH_NOT_CHANGED;);
                const auto index = LabelToSubscript(label);
                FUSION_PASS_CHECK(!(labelCount[index] > 0 && labelPermIndex[index] == -1),
                                  OPS_LOG_W(kPassName.c_str(), "output subscript %c invalid", label),
                                  return GRAPH_NOT_CHANGED;);
                labelPermIndex[index] = permIndex++;
        }
    }
    return SUCCESS;
}

Status EinsumPass::ParseOutputLabel(const std::string& equation, size_t arrowPos, int64_t& ellNumDim,
                                    int64_t& permIndex, bool& ellInOutput, std::vector<int64_t>& labelPermIndex,
                                    std::vector<int64_t>& labelCount, int64_t& outNumDim, int64_t& ellIndex)
{
    if (arrowPos == std::string::npos) {
        permIndex = ellNumDim;
        ellInOutput = true;
        for (int label = 0; label < static_cast<int>(kTotalLabels); ++label) {
            if (labelCount[label] == 1) {
                labelPermIndex[label] = permIndex++;
            }
        }
    } else {
        FUSION_PASS_CHECK(ParseOutputRhsLabel(equation, arrowPos, ellNumDim, permIndex, ellInOutput, labelPermIndex,
                                              labelCount, ellIndex) != SUCCESS,
                          OPS_LOG_W(kPassName.c_str(), "failed to parse output labels"), return GRAPH_NOT_CHANGED;);
    }
    outNumDim = permIndex;
    if (!ellInOutput) {
        ellIndex = permIndex;
        permIndex += ellNumDim;
    }
    for (int label = 0; label < static_cast<int>(kTotalLabels); ++label) {
        if (labelCount[label] > 0 && labelPermIndex[label] == -1) {
            labelPermIndex[label] = permIndex++;
        }
    }
    return SUCCESS;
}

Status EinsumPass::AlignInputDimForOutLabel(const GNode& matchedNode, size_t numOps, int64_t& permIndex,
                                            std::vector<std::vector<uint8_t>>& opLabels,
                                            std::vector<int64_t>& dimCounts, std::vector<int64_t>& ellSizes,
                                            std::vector<int64_t>& labelSize, std::vector<int64_t>& labelPermIndex,
                                            int64_t& ellNumDim, int64_t& ellIndex)
{
    for (size_t i = 0; i < numOps; ++i) {
        TensorDesc inputDesc;
        FUSION_PASS_CHECK(matchedNode.GetInputDesc(i, inputDesc) != GRAPH_SUCCESS,
                          OPS_LOG_E(kPassName.c_str(), "Failed to get input desc %zu.", i), return FAILED;);
        auto opShape = inputDesc.GetShape();
        std::vector<int> permutation(permIndex, -1);
        std::int32_t dim = 0;
        for (const auto s : opLabels[i]) {
            if (s == kEllipsis) {
                const auto ndim = opShape.GetDims().size() - (opLabels[i].size() - 1);
                for (int64_t j = ellNumDim - ndim; j < ellNumDim; ++j) {
                    if (opShape.GetDims()[dim] != 1) {
                        ellSizes[j] = opShape.GetDims()[dim];
                        ++dimCounts[ellIndex + j];
                    }
                    permutation[ellIndex + j] = dim++;
                }
            } else if (permutation[labelPermIndex[s]] == -1) {
                if (opShape.GetDims()[dim] != 1) {
                    labelSize[s] = opShape.GetDims()[dim];
                    ++dimCounts[labelPermIndex[s]];
                }
                permutation[labelPermIndex[s]] = dim++;
            } else {
                OPS_LOG_E(kPassName.c_str(), "diagonal operation is not supported");
                return FAILED;
            }
        }
        std::vector<int64_t> unsqueezeDim = opShape.GetDims();
        std::vector<int64_t> dims;
        for (auto& val : permutation) {
            if (val == -1) {
                unsqueezeDim.insert(unsqueezeDim.begin() + dim, 1);
                dims.emplace_back(dim);
                val = dim++;
            }
        }
        if (!dims.empty()) {
            FUSION_PASS_CHECK(UnsqueezeInput(matchedNode, dims, unsqueezeDim, static_cast<int32_t>(i)) != SUCCESS,
                              OPS_LOG_W(kPassName.c_str(), "failed to process UnsqueezeInput"),
                              return GRAPH_NOT_CHANGED);
        }
        FUSION_PASS_CHECK(PermuteInput(matchedNode, permutation, i) != SUCCESS,
                          OPS_LOG_W(kPassName.c_str(), "failed to process permutation"), return GRAPH_NOT_CHANGED);
    }
    return SUCCESS;
}

Status EinsumPass::CastNode(const GNode& matchedNode, const DataType& dtypeOut, int32_t idx)
{
    EsTensorHolder prevHolder;
    if (!IsGNodeValid(batchmatmulNode_)) {
        prevHolder = (bmmInputNodes_[idx].empty()) ? subgraphInputHolders_[idx] :
                                                     GetTensorHolder(*builder_, bmmInputNodes_[idx].back(), 0);
    } else {
        prevHolder = GetTensorHolder(*builder_, batchmatmulNode_, 0);
    }
    GNode castNode = CreateCastNode(*builder_, prevHolder, dtypeOut);

    auto prevDesc = GetPrevOutputDesc(matchedNode, static_cast<size_t>(idx));
    auto prevDims = prevDesc->GetShape().GetDims();
    TensorDesc castOutDesc(Shape(prevDims), prevDesc->GetFormat(), dtypeOut);
    castOutDesc.SetOriginFormat(prevDesc->GetFormat());
    castOutDesc.SetOriginShape(Shape(prevDims));
    castNode.UpdateOutputDesc(0, castOutDesc);

    if (!IsGNodeValid(batchmatmulNode_)) {
        bmmInputNodes_[idx].emplace_back(castNode);
    } else {
        bmmOutputNodes_.emplace_back(castNode);
    }
    return SUCCESS;
}

Status EinsumPass::UnsqueezeInput(const GNode& matchedNode, std::vector<int64_t> dims,
                                  const std::vector<int64_t>& unsqueezeDim, int32_t idx)
{
    if (dims.empty()) {
        return SUCCESS;
    }
    std::shared_ptr<TensorDesc> xDesc = (idx < static_cast<int32_t>(kMaxInputNum)) ?
                                            GetPrevOutputDesc(matchedNode, static_cast<size_t>(idx)) :
                                            GetPrevOutputDescAfterBmm(matchedNode);
    TensorDesc unsqueezeOutDesc(Shape(unsqueezeDim), xDesc->GetFormat(), xDesc->GetDataType());
    unsqueezeOutDesc.SetOriginFormat(xDesc->GetFormat());
    unsqueezeOutDesc.SetOriginShape(Shape(unsqueezeDim));
    if (idx < static_cast<int32_t>(kMaxInputNum)) {
        GNode unsqueezeNode = CreateUnsqueezeNode(graphPtr_, "unsqueeze_" + std::to_string(unsqueezeSeq_++), dims);
        GNode prevNode = (bmmInputNodes_[idx].empty()) ? *subgraphInputHolders_[idx].GetProducer() :
                                                         bmmInputNodes_[idx].back();
        FUSION_PASS_CHECK(AddEdgeAndUpdatePeerDesc(*graphPtr_, prevNode, 0, unsqueezeNode, 0) != GRAPH_SUCCESS,
                          OPS_LOG_E(kPassName.c_str(), "AddEdgeAndUpdatePeerDesc failed."), return FAILED);
        unsqueezeNode.UpdateOutputDesc(0, unsqueezeOutDesc);
        bmmInputNodes_[idx].emplace_back(unsqueezeNode);
    } else {
        GNode unsqueezeNode = CreateUnsqueezeNode(graphPtr_, "unsqueeze_out_" + std::to_string(unsqueezeSeq_++), dims);
        GNode prevNode = bmmOutputNodes_.empty() ? batchmatmulNode_ : bmmOutputNodes_.back();
        FUSION_PASS_CHECK(AddEdgeAndUpdatePeerDesc(*graphPtr_, prevNode, 0, unsqueezeNode, 0) != GRAPH_SUCCESS,
                          OPS_LOG_E(kPassName.c_str(), "AddEdgeAndUpdatePeerDesc failed."), return FAILED);
        unsqueezeNode.UpdateOutputDesc(0, unsqueezeOutDesc);
        bmmOutputNodes_.emplace_back(unsqueezeNode);
    }
    return SUCCESS;
}

Status EinsumPass::PermuteInput(const GNode& matchedNode, const std::vector<int>& permutation, size_t idx)
{
    auto xDesc = GetPrevOutputDesc(matchedNode, idx);
    auto ndim = xDesc->GetShape().GetDimNum();
    std::vector<int> vec(ndim);
    std::iota(vec.begin(), vec.end(), 0);
    if (permutation == vec) {
        return SUCCESS;
    }
    std::vector<int32_t> permInt32(permutation.begin(), permutation.end());
    EsTensorHolder prevHolder = (bmmInputNodes_[idx].empty()) ?
                                    subgraphInputHolders_[idx] :

                                    GetTensorHolder(*builder_, bmmInputNodes_[idx].back(), 0);
    GNode transNode = CreateTransposeNode(*builder_, "permute_" + std::to_string(transposeSeq_++), prevHolder, true,
                                          supportL12btBf16_, permInt32);

    auto xDims = xDesc->GetShape().GetDims();
    std::vector<int64_t> permOutDims;
    permOutDims.reserve(permInt32.size());
    for (int32_t p : permInt32) {
        if (static_cast<size_t>(p) < xDims.size()) {
            permOutDims.push_back(xDims[p]);
        }
    }
    TensorDesc permOutDesc(Shape(permOutDims), xDesc->GetFormat(), xDesc->GetDataType());
    permOutDesc.SetOriginFormat(xDesc->GetFormat());
    permOutDesc.SetOriginShape(Shape(permOutDims));
    transNode.UpdateOutputDesc(0, permOutDesc);

    bmmInputNodes_[idx].emplace_back(transNode);
    return SUCCESS;
}

Status EinsumPass::ReduceSumInput(const GNode& matchedNode, const std::vector<int64_t>& dimsToSum, size_t idx,
                                  bool keepDims)
{
    if (dimsToSum.empty()) {
        return SUCCESS;
    }
    auto xDesc = GetPrevOutputDesc(matchedNode, idx);
    if (xDesc->GetDataType() == ge::DT_FLOAT16 || xDesc->GetDataType() == ge::DT_BF16) {
        FUSION_PASS_CHECK(CastNode(matchedNode, ge::DT_FLOAT, static_cast<int32_t>(idx)) != SUCCESS,
                          OPS_LOG_E(kPassName.c_str(), "failed to cast to float32"), return GRAPH_NOT_CHANGED);
    }
    GNode reduceNode = CreateReduceSumNode(graphPtr_, *builder_, "reducesum_" + std::to_string(reduceSeq_++), true,
                                           dimsToSum, keepDims);
    GNode prevNode = (bmmInputNodes_[idx].empty()) ? *subgraphInputHolders_[idx].GetProducer() :
                                                     bmmInputNodes_[idx].back();
    FUSION_PASS_CHECK(AddEdgeAndUpdatePeerDesc(*graphPtr_, prevNode, 0, reduceNode, 0) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName.c_str(), "AddEdgeAndUpdatePeerDesc failed."), return FAILED);

    auto reduceDesc = GetPrevOutputDesc(matchedNode, idx);
    auto reduceDims = reduceDesc->GetShape().GetDims();
    if (keepDims) {
        for (auto d : dimsToSum) {
            if (static_cast<size_t>(d) < reduceDims.size()) {
                reduceDims[d] = 1;
            }
        }
    } else {
        std::set<int64_t> dimSet(dimsToSum.begin(), dimsToSum.end());
        std::vector<int64_t> newDims;
        for (size_t i = 0; i < reduceDims.size(); ++i) {
            if (dimSet.find(static_cast<int64_t>(i)) == dimSet.end()) {
                newDims.push_back(reduceDims[i]);
            }
        }
        reduceDims = newDims;
    }
    TensorDesc reduceOutDesc(Shape(reduceDims), reduceDesc->GetFormat(), reduceDesc->GetDataType());
    reduceOutDesc.SetOriginFormat(reduceDesc->GetFormat());
    reduceOutDesc.SetOriginShape(Shape(reduceDims));
    reduceNode.UpdateOutputDesc(0, reduceOutDesc);
    bmmInputNodes_[idx].emplace_back(reduceNode);
    return SUCCESS;
}

Status EinsumPass::BatchMatmul(const GNode& matchedNode)
{
    auto x1Desc = GetPrevOutputDesc(matchedNode, 0);
    auto x2Desc = GetPrevOutputDesc(matchedNode, 1);
    if (x1Desc->GetDataType() != x2Desc->GetDataType()) {
        if (x1Desc->GetDataType() < x2Desc->GetDataType()) {
            FUSION_PASS_CHECK(CastNode(matchedNode, ge::DT_FLOAT, 1) != SUCCESS,
                              OPS_LOG_E(kPassName.c_str(), "failed to cast to float32"), return FAILED);
            x2Desc = GetPrevOutputDesc(matchedNode, 1);
        } else {
            FUSION_PASS_CHECK(CastNode(matchedNode, ge::DT_FLOAT, 0) != SUCCESS,
                              OPS_LOG_E(kPassName.c_str(), "failed to cast to float32"), return FAILED);
            x1Desc = GetPrevOutputDesc(matchedNode, 0);
        }
    }

    EsTensorHolder x1Holder = bmmInputNodes_[0].empty() ? subgraphInputHolders_[0] :
                                                          GetTensorHolder(*builder_, bmmInputNodes_[0].back(), 0);
    EsTensorHolder x2Holder = bmmInputNodes_[1].empty() ? subgraphInputHolders_[1] :
                                                          GetTensorHolder(*builder_, bmmInputNodes_[1].back(), 0);

    bool useMatmul = (x1Desc->GetShape().GetDimNum() == kDim2D) && (x2Desc->GetShape().GetDimNum() == kDim2D);
    const std::string& bmmOpType = useMatmul ? kMatMul : kBatchMatMul;
    auto bmmOutput = CreateMatMulNode(*builder_, bmmOpType, x1Holder, x2Holder, false, false);
    batchmatmulNode_ = *bmmOutput.GetProducer();
    FUSION_PASS_CHECK(!CopyOtherAttrs(matchedNode, batchmatmulNode_, kPassName),
                      OPS_LOG_E(kPassName.c_str(), "Copy other attrs failed."), return FAILED;);

    TensorDesc inputDesc0;
    FUSION_PASS_CHECK(matchedNode.GetInputDesc(0, inputDesc0) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName.c_str(), "Failed to get input desc 0."), return FAILED;);
    auto outDtype = inputDesc0.GetDataType();
    if (x1Desc->GetDataType() != outDtype) {
        FUSION_PASS_CHECK(CastNode(matchedNode, outDtype, 0) != SUCCESS,
                          OPS_LOG_E(kPassName.c_str(), "failed to cast to origin dtype"), return FAILED);
    }
    return SUCCESS;
}

Status EinsumPass::SumDimEmptyMul(const GNode& matchedNode, std::shared_ptr<TensorDesc>& leftDesc,
                                  std::shared_ptr<TensorDesc>& rightDesc, const std::vector<int64_t>& sumDims)
{
    (void)sumDims;
    if (leftDesc->GetDataType() != rightDesc->GetDataType()) {
        if (leftDesc->GetDataType() < rightDesc->GetDataType()) {
            FUSION_PASS_CHECK(CastNode(matchedNode, ge::DT_FLOAT, 1) != SUCCESS,
                              OPS_LOG_E(kPassName.c_str(), "failed to cast to float32"), return FAILED);
            rightDesc = GetPrevOutputDesc(matchedNode, 1);
        } else {
            FUSION_PASS_CHECK(CastNode(matchedNode, ge::DT_FLOAT, 0) != SUCCESS,
                              OPS_LOG_E(kPassName.c_str(), "failed to cast to float32"), return FAILED);
            leftDesc = GetPrevOutputDesc(matchedNode, 0);
        }
    }
    EsTensorHolder leftHolder = bmmInputNodes_[0].empty() ? subgraphInputHolders_[0] :
                                                            GetTensorHolder(*builder_, bmmInputNodes_[0].back(), 0);
    EsTensorHolder rightHolder = bmmInputNodes_[1].empty() ? subgraphInputHolders_[1] :
                                                             GetTensorHolder(*builder_, bmmInputNodes_[1].back(), 0);

    GNode mulNode = CreateMulNode(*builder_, leftHolder, rightHolder);
    SetMulOutputDesc(mulNode);
    batchmatmulNode_ = mulNode;

    TensorDesc inputDesc0;
    FUSION_PASS_CHECK(matchedNode.GetInputDesc(0, inputDesc0) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName.c_str(), "Failed to get input desc 0."), return FAILED;);
    auto outDtype = inputDesc0.GetDataType();
    if (leftDesc->GetDataType() != outDtype) {
        FUSION_PASS_CHECK(CastNode(matchedNode, outDtype, 0) != SUCCESS,
                          OPS_LOG_E(kPassName.c_str(), "failed to cast to origin dtype"), return FAILED);
    }
    return SUCCESS;
}

Status EinsumPass::InsertGatherShape(std::shared_ptr<OpDesc>& gatherShapesDesc,
                                     const std::vector<std::vector<int64_t>>& axes, std::vector<int64_t>& tmpDims,
                                     std::shared_ptr<TensorDesc>& rightDesc)
{
    (void)gatherShapesDesc;
    (void)rightDesc;
    if (!axes.empty()) {
        std::vector<std::vector<int64_t>> axesAttr = axes;
        std::vector<EsTensorHolder> gatherInputs{
            (bmmInputNodes_[0].empty()) ? subgraphInputHolders_[0] :
                                          GetTensorHolder(*builder_, bmmInputNodes_[0].back(), 0),
            (bmmInputNodes_[1].empty()) ? subgraphInputHolders_[1] :
                                          GetTensorHolder(*builder_, bmmInputNodes_[1].back(), 0)};
        GNode gatherNode = CreateGatherShapesNode(gatherInputs, axesAttr);
        SetGatherShapesOutputDesc(gatherNode, axes.size());
        gathershapeNode_ = gatherNode;
    }
    return SUCCESS;
}

// Final flatten/unsqueeze after per-dim alignment (split from PermuteReshapeInput)
Status EinsumPass::FinalizePermuteReshape(const GNode& matchedNode, const std::vector<int64_t>& reshapeDim, size_t idx)
{
    auto prevDesc = GetPrevOutputDesc(matchedNode, idx);
    auto xDims = prevDesc->GetShape().GetDimNum();
    auto startDim = kDim2D;
    auto endDim = xDims - 1;
    if (startDim < static_cast<int>(endDim)) {
        // FlattenV2 (dynamic reshape) always: fold startDim..endDim into reshapeDim
        EsTensorHolder prevHolder = (bmmInputNodes_[idx].empty()) ?
                                        subgraphInputHolders_[idx] :

                                        GetTensorHolder(*builder_, bmmInputNodes_[idx].back(), 0);
        GNode reshapeNode = CreateReshapeNode(*builder_, "reshape_pr2_" + std::to_string(reshapeSeq_++), prevHolder,
                                              true, reshapeDim, static_cast<int32_t>(startDim),
                                              static_cast<int32_t>(endDim));
        SetReshapeOutputDesc(reshapeNode, reshapeDim);
        bmmInputNodes_[idx].emplace_back(reshapeNode);
    } else if (static_cast<int>(endDim) == 1) {
        auto prevDesc2 = GetPrevOutputDesc(matchedNode, idx);
        std::vector<int64_t> unsqueezeDim = prevDesc2->GetShape().GetDims();
        unsqueezeDim.insert(unsqueezeDim.begin() + startDim, 1);
        FUSION_PASS_CHECK(UnsqueezeInput(matchedNode, {static_cast<int64_t>(startDim)}, unsqueezeDim,
                                         static_cast<int32_t>(idx)) != SUCCESS,
                          OPS_LOG_W(kPassName.c_str(), "failed to process UnsqueezeInput"), return GRAPH_NOT_CHANGED);
    }
    return SUCCESS;
}

Status EinsumPass::PermuteReshapeInput(const GNode& matchedNode, const std::vector<int>& permuteDim,
                                       const std::vector<int64_t>& reshapeDim, const std::vector<size_t>& size,
                                       size_t idx)
{
    FUSION_PASS_CHECK(PermuteInput(matchedNode, permuteDim, idx) != SUCCESS,
                      OPS_LOG_W(kPassName.c_str(), "failed to process permute"), return GRAPH_NOT_CHANGED);
    for (size_t dim = 0; dim < size.size() - 1; ++dim) {
        if (size[dim] == 0) {
            auto xDesc = GetPrevOutputDesc(matchedNode, idx);
            std::vector<int64_t> unsqueezeDim = xDesc->GetShape().GetDims();
            unsqueezeDim.insert(unsqueezeDim.begin() + dim, 1);
            FUSION_PASS_CHECK(UnsqueezeInput(matchedNode, {static_cast<int64_t>(dim)}, unsqueezeDim,
                                             static_cast<int32_t>(idx)) != SUCCESS,
                              OPS_LOG_W(kPassName.c_str(), "failed to process UnsqueezeInput"),
                              return GRAPH_NOT_CHANGED);
        } else if (size[dim] > 1) {
            auto xDesc = GetPrevOutputDesc(matchedNode, idx);
            auto xDims = xDesc->GetShape().GetDims();
            // FlattenV2 (dynamic reshape) always: fold dim..dim+size[dim]-1 into one
            EsTensorHolder prevHolder = (bmmInputNodes_[idx].empty()) ?
                                            subgraphInputHolders_[idx] :

                                            GetTensorHolder(*builder_, bmmInputNodes_[idx].back(), 0);
            GNode reshapeNode = CreateReshapeNode(*builder_, "reshape_pr_" + std::to_string(reshapeSeq_++), prevHolder,
                                                  true, xDims, static_cast<int32_t>(dim),
                                                  static_cast<int32_t>(dim + size[dim] - 1));
            std::vector<int64_t> flatDims = xDims;
            if (flatDims.size() > static_cast<size_t>(dim) + size[dim] - 1) {
                int64_t mulVal = 1;
                for (size_t k = dim; k < dim + size[dim]; ++k) {
                    mulVal = (k < flatDims.size()) ? GetDimMulValue(mulVal, flatDims[k]) : mulVal;
                }
                flatDims.erase(flatDims.begin() + dim, flatDims.begin() + dim + size[dim]);
                flatDims.insert(flatDims.begin() + dim, mulVal);
            }
            SetReshapeOutputDesc(reshapeNode, flatDims);
            bmmInputNodes_[idx].emplace_back(reshapeNode);
        }
    }
    return FinalizePermuteReshape(matchedNode, reshapeDim, idx);
}

// ============================================================================
// SumproductPair helpers (split from the original oversized function)
// ============================================================================

EinsumPass::DimClassification EinsumPass::ClassifySumproductDims(const std::shared_ptr<TensorDesc>& leftDesc,
                                                                 const std::shared_ptr<TensorDesc>& rightDesc,
                                                                 const std::vector<int64_t>& sumDims)
{
    DimClassification cls;
    int64_t dim = leftDesc->GetShape().GetDims().size();
    auto sumDimsWrapped = MaybeWrapDim(sumDims, dim);
    auto leftSymSize = leftDesc->GetShape().GetDims();
    auto rightSymSize = rightDesc->GetShape().GetDims();
    for (int i = 0; i < dim; ++i) {
        auto sl = leftSymSize[i] != 1;
        auto sr = rightSymSize[i] != 1;
        if (sumDimsWrapped[i] != -1) {
            if (sl && sr) {
                cls.sumSize *= leftSymSize[i];
            } else if (sl) {
                cls.leftSumDims.push_back(i);
            } else if (sr) {
                cls.rightSumDims.push_back(i);
            }
        } else if (sl && sr) {
            cls.lro.push_back(i);
            cls.lroSize *= leftSymSize[i];
        } else if (sl) {
            cls.lo.push_back(i);
            cls.loSize *= leftSymSize[i];
        } else {
            cls.ro.push_back(i);
            cls.roSize *= rightSymSize[i];
        }
    }
    return cls;
}

// Build left/right/output permutations for sumproduct BMM (split from BuildSumproductBmm)
void EinsumPass::BuildSumproductPermutations(const DimClassification& cls, const std::vector<int64_t>& sumDims,
                                             int64_t outNumDim, std::vector<int>& lpermutation,
                                             std::vector<int>& rpermutation, std::vector<int>& opermutation) const
{
    lpermutation.assign(cls.lro.begin(), cls.lro.end());
    lpermutation.insert(lpermutation.end(), cls.lo.begin(), cls.lo.end());
    lpermutation.insert(lpermutation.end(), sumDims.begin(), sumDims.end());
    lpermutation.insert(lpermutation.end(), cls.ro.begin(), cls.ro.end());
    rpermutation.assign(cls.lro.begin(), cls.lro.end());
    rpermutation.insert(rpermutation.end(), sumDims.begin(), sumDims.end());
    rpermutation.insert(rpermutation.end(), cls.ro.begin(), cls.ro.end());
    rpermutation.insert(rpermutation.end(), cls.lo.begin(), cls.lo.end());
    opermutation.assign(outNumDim, -1);
    int64_t i = 0;
    for (auto it = cls.lro.cbegin(); it != cls.lro.cend(); i++, it++) {
        opermutation[*it] = i;
    }
    for (auto it = cls.lo.cbegin(); it != cls.lo.cend(); i++, it++) {
        opermutation[*it] = i;
    }
    for (auto it = sumDims.cbegin(); it != sumDims.cend(); i++, it++) {
        opermutation[*it] = i;
    }
    for (auto it = cls.ro.cbegin(); it != cls.ro.cend(); i++, it++) {
        opermutation[*it] = i;
    }
}

Status EinsumPass::BuildSumproductBmm(const GNode& matchedNode, const DimClassification& cls,
                                      const std::vector<int64_t>& sumDims, int64_t outNumDim,
                                      std::shared_ptr<TensorDesc>& rightDesc)
{
    std::vector<int64_t> unsqueezeDims;
    std::vector<int64_t> tmpDims;
    tmpDims.reserve(outNumDim);
    std::vector<std::vector<int64_t>> axes;
    axes.reserve(cls.lro.size() + cls.lo.size() + cls.ro.size());
    for (auto& d : cls.lro) {
        axes.push_back({0, d});
        tmpDims.push_back(rightDesc->GetShape().GetDims()[d]);
    }
    for (auto& d : cls.lo) {
        axes.push_back({0, d});
        tmpDims.push_back(rightDesc->GetShape().GetDims()[d]);
    }
    int64_t unsqueezeDim = cls.lro.size() + cls.lo.size();
    for (size_t idx = 0; idx < sumDims.size(); ++idx) {
        tmpDims.emplace_back(1);
        unsqueezeDims.emplace_back(unsqueezeDim++);
    }
    for (auto& d : cls.ro) {
        axes.push_back({1, d});
        tmpDims.emplace_back(rightDesc->GetShape().GetDims()[d]);
    }
    std::shared_ptr<OpDesc> dummyGatherDesc = nullptr;
    FUSION_PASS_CHECK(InsertGatherShape(dummyGatherDesc, axes, tmpDims, rightDesc) != SUCCESS,
                      OPS_LOG_E(kPassName.c_str(), "Insert gather shape node failed"), return GRAPH_NOT_CHANGED);
    std::vector<int> lpermutation;
    std::vector<int> rpermutation;
    std::vector<int> opermutation;
    BuildSumproductPermutations(cls, sumDims, outNumDim, lpermutation, rpermutation, opermutation);
    std::vector<size_t> lsize = {cls.lro.size(), cls.lo.size(), sumDims.size()};
    std::vector<size_t> rsize = {cls.lro.size(), sumDims.size(), cls.ro.size()};
    std::vector<int64_t> lreshape = {cls.lroSize, cls.loSize, cls.sumSize};
    std::vector<int64_t> rreshape = {cls.lroSize, cls.sumSize, cls.roSize};
    FUSION_PASS_CHECK(PermuteReshapeInput(matchedNode, lpermutation, lreshape, lsize, 0) != SUCCESS,
                      OPS_LOG_W(kPassName.c_str(), "failed to process transpose and reshape a"),
                      return GRAPH_NOT_CHANGED);
    FUSION_PASS_CHECK(PermuteReshapeInput(matchedNode, rpermutation, rreshape, rsize, 1) != SUCCESS,
                      OPS_LOG_W(kPassName.c_str(), "failed to process transpose and reshape b"),
                      return GRAPH_NOT_CHANGED);
    FUSION_PASS_CHECK(BatchMatmul(matchedNode) != SUCCESS,
                      OPS_LOG_W(kPassName.c_str(), "failed to process batchmatmul"), return GRAPH_NOT_CHANGED);
    FUSION_PASS_CHECK(
        ReshapePermuteInput(matchedNode, dummyGatherDesc, opermutation, tmpDims, unsqueezeDims) != SUCCESS,
        OPS_LOG_W(kPassName.c_str(), "failed to process reshape and permute"), return GRAPH_NOT_CHANGED);
    return SUCCESS;
}

Status EinsumPass::SumproductPair(const GNode& matchedNode, const std::vector<int64_t>& sumDims, bool keepdim)
{
    auto leftDesc = GetPrevOutputDesc(matchedNode, 0);
    auto rightDesc = GetPrevOutputDesc(matchedNode, 1);
    // NOTE: mixed-dtype inputs on this route come only from fp16/bf16->fp32 promotion inside
    // ReduceSumInput; the cast-up (and the cast back to input(0)'s dtype) lives in BatchMatmul
    // where the matmul is created, matching the legacy pass placement exactly.
    FUSION_PASS_CHECK(leftDesc->GetShape().GetDims().size() != rightDesc->GetShape().GetDims().size(),
                      OPS_LOG_W(kPassName.c_str(), "number of dimensions must match"), return GRAPH_NOT_CHANGED;);
    if (sumDims.empty()) {
        FUSION_PASS_CHECK(SumDimEmptyMul(matchedNode, leftDesc, rightDesc, sumDims) != SUCCESS,
                          OPS_LOG_E(kPassName.c_str(), "when sum_dim is empty, direct mul failed"),
                          return GRAPH_NOT_CHANGED);
        return SUCCESS;
    }
    auto cls = ClassifySumproductDims(leftDesc, rightDesc, sumDims);
    FUSION_PASS_CHECK(ReduceSumInput(matchedNode, cls.leftSumDims, 0, true) != SUCCESS,
                      OPS_LOG_W(kPassName.c_str(), "failed to process sum out a"), return GRAPH_NOT_CHANGED);
    FUSION_PASS_CHECK(ReduceSumInput(matchedNode, cls.rightSumDims, 1, true) != SUCCESS,
                      OPS_LOG_W(kPassName.c_str(), "failed to process sum out b"), return GRAPH_NOT_CHANGED);
    int64_t outNumDim = cls.lro.size() + cls.lo.size() + sumDims.size() + cls.ro.size();
    return BuildSumproductBmm(matchedNode, cls, sumDims, outNumDim, rightDesc);
}

Status EinsumPass::ReshapePermuteInput(const GNode& matchedNode, std::shared_ptr<OpDesc>& gatherShapesDesc,
                                       const std::vector<int>& opermuteDim, const std::vector<int64_t>& outDim,
                                       const std::vector<int64_t>& unsqueezeDim)
{
    auto xDesc = GetPrevOutputDescAfterBmm(matchedNode);
    if (!IsGNodeType(gathershapeNode_, kGatherShapes)) {
        GNode prevNode = bmmOutputNodes_.empty() ? batchmatmulNode_ : bmmOutputNodes_.back();
        GNode reshapeNode = CreateReshapeNode(*builder_, "reshape_rp_" + std::to_string(reshapeSeq_++),
                                              GetTensorHolder(*builder_, prevNode, 0), true, outDim, 0, 2);
        SetReshapeOutputDesc(reshapeNode, outDim);
        bmmOutputNodes_.emplace_back(reshapeNode);
    } else {
        GNode prevNode = bmmOutputNodes_.empty() ? batchmatmulNode_ : bmmOutputNodes_.back();
        GNode reshapeNode = *es::Reshape(GetTensorHolder(*builder_, prevNode, 0),
                                         GetTensorHolder(*builder_, gathershapeNode_, 0))
                                 .GetProducer();
        SetReshapeOutputDesc(reshapeNode, outDim);
        bmmOutputNodes_.emplace_back(reshapeNode);

        if (!unsqueezeDim.empty()) {
            FUSION_PASS_CHECK(UnsqueezeInput(matchedNode, unsqueezeDim, outDim, 2) != SUCCESS,
                              OPS_LOG_W(kPassName.c_str(), "failed to process Unsqueeze"), return GRAPH_NOT_CHANGED);
        }
        auto prevDesc = GetPrevOutputDescAfterBmm(matchedNode);
        std::vector<int32_t> opermuteInt32(opermuteDim.begin(), opermuteDim.end());
        GNode prevNode2 = bmmOutputNodes_.back();
        GNode transNode = CreateTransposeNode(*builder_, "transpose_rp_" + std::to_string(transposeSeq_++),
                                              GetTensorHolder(*builder_, prevNode2, 0), true, supportL12btBf16_,
                                              opermuteInt32);
        SetTransposeOutputDesc(transNode, opermuteInt32);
        bmmOutputNodes_.emplace_back(transNode);
    }
    return SUCCESS;
}

std::vector<int64_t> EinsumPass::MaybeWrapDim(const std::vector<int64_t>& sumDims, int64_t dim) const
{
    std::vector<int64_t> seen(dim, -1);
    for (int i : sumDims) {
        int minVal = -1 * static_cast<int>(dim);
        int maxVal = static_cast<int>(dim) - 1;
        if (i < minVal || i > maxVal) {
            OPS_LOG_E(kPassName.c_str(), "Dimension out of range sum_dim %d", i);
            continue;
        }
        if (i < 0) {
            seen[i + dim] = i + dim;
        } else {
            seen[i] = i;
        }
    }
    return seen;
}

// ============================================================================
// Registration
// ============================================================================

REG_FUSION_PASS(EinsumPass).Stage(CustomPassStage::kCompatibleInherited);

} // namespace ops

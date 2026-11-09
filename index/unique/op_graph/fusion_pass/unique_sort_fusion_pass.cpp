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
 * \file unique_sort_fusion_pass.cpp
 * \brief Unique|UniqueWithCounts --> (Sort + AdjDiff + Cumsum + Scatter + UniqueConsecutive))
 *               x                                  x
 *               |                                  |
 *               |                               Sort(axis=0, stable)
 *               |                             /        \
 *            Unique /                 UniqueConsecutive  AdjacentDifference
 *      UniqueWithCounts              (return_counts)         |
 *          |      |     |    ==>        |    |    |         Cumsum(axis=0)
 *          y     idx    cnt             y   (idx)  cnt        |         \
 *                                     (count仅在cnt输出被   ScatterElements
 *                                      下游使用时输出)          |
 *                                                             idx(逆索引)
 */

#include <algorithm>
#include <initializer_list>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "unique_sort_fusion_pass.h"
#include "acl/acl_rt.h"
#include "common/inc/error_util.h"
#include "common/inc/op_graph/arch_utils.h"
#include "es_math_ops.h"
#include "es_nn_ops.h"
#include "ge/compliant_node_builder.h"
#include "ge/es_graph_builder.h"
#include "ge/fusion/graph_rewriter.h"
#include "ge/fusion/pass/pattern_fusion_pass.h"
#include "ge/ge_utils.h"
#include "platform/platform_info.h"
#include "register/register_custom_pass.h"
#include "version/ge-compiler_version.h"

using namespace ge;
using namespace fe;

// GE 9.2+ 的 SubgraphRewriter::Replace 新增三参重载（携带 CustomPassContext），
// 低版本 CANN 的 libge_compiler.so 中不存在该符号，编译期按头文件直接调用会产生强引用，
// 导致 so 在旧版本环境加载时因 undefined symbol 失败。
// 这里通过 asm label 对该符号做弱镜像声明，运行时判空：
// 存在则走三参版本，否则回退到新老版本都有的两参版本。
extern ge::Status SubgraphRewriterReplaceWithCtx(
    const ge::fusion::SubgraphBoundary& subgraph, const ge::Graph& replacement,
    ge::CustomPassContext& ctx) __asm__("_ZN2ge6fusion16SubgraphRewriter7ReplaceERKNS0_16SubgraphBoundaryERKNS_"
                                        "5GraphERNS_17CustomPassContextE") __attribute__((weak));

namespace ops {
namespace {
const std::string kPassName = "UniqueSortFusionPass";
const char* kUniqueType = "Unique";
const char* kUniqueWithCountsType = "UniqueWithCounts";

// Unique/UniqueWithCounts IR input/output index
constexpr int64_t kUniqueValIn = 0;
constexpr int64_t kUniqueValOut = 0;
constexpr int64_t kUniqueIdxOut = 1;
constexpr int64_t kUniqueCntOut = 2;

// replacement node io index
constexpr int32_t kSortValueOut = 0;
constexpr int32_t kSortIdxOut = 1;
constexpr int32_t kCumsumXIn = 0;
constexpr int32_t kCumsumAxisIn = 1;
constexpr int32_t kAdjDiffXIn = 0;
constexpr int32_t kAdjDiffYOut = 0;
constexpr int32_t kCumsumYOut = 0;
constexpr int32_t kScatterYOut = 0;

// attributes, consistent with the original fusion pass
constexpr int64_t kSortAxis = 0;
constexpr bool kSortDescending = false;
constexpr bool kSortStable = true;
constexpr int64_t kUniqueConsecutiveAxis = 1000;
// Third-class dynamic shape flag, mirrors the legacy THRID_CLASS_DYN_TYPE in
// unique_sort_fusion_pass.cc (SetUniqueConsAttr: _unknown_shape_type = 3).
constexpr int64_t kThirdClassDynType = 3;
constexpr int64_t kCumsumAxis = 0;
constexpr int64_t kScatterAxis = 0;
const char* kScatterReduction = "none";

// same dtype whitelist as the original pass
static const std::initializer_list<DataType> kXYDtypeSupportList = {
    DT_INT64, DT_INT32, DT_INT16, DT_INT8, DT_UINT64, DT_UINT32, DT_UINT16, DT_UINT8, DT_BF16, DT_FLOAT16, DT_FLOAT};

// kCompatibleInherited 出于 9.0.0(GE_COMPILER_VERSION_NUM = 90000000)。
// 本 pass 基于 ES 框架(EsGraphBuilder/CompliantNodeBuilder/SubgraphRewriter)构建替换图,
// 注册 stage kCompatibleInherited 属兼容场景 D1(仅适用整体静默):
// 9.x 运行时正常融合,老运行时空跑。
const int32_t kGeCompilerVersion900 = 90000000;

// 查询运行时 GE compiler 版本号;查询失败按最低版本(0)处理,走兜底空跑分支
static int32_t GetRuntimeGeCompilerVersion()
{
    int32_t version = 0;
    char moduleName[] = "ge_compiler";
    aclsysGetVersionNum(moduleName, &version);
    return version;
}

// The original pass only fuses on regbase arch (IsRegBase). On the new framework the
// supported set below reuses the common ops::CheckNpuArch utility (common/inc/op_graph/
// arch_utils.h) for the unified NpuArch read, set judgement and failure logging, aligned
// with AddRmsNormFusionGraphPass etc. Scope note: the legacy canndev IsRegBase() covers
// DAV_3510 / DAV_5102 / DAV_5162, while this pass (and issue #7438) deliberately covers
// only 3510/5102 for now - DAV_5162 (ascend5162a) keeps being served by the legacy
// canndev pass: this pass registers at kCompatibleInherited only for the migrated scope
// and stays silent (GRAPH_NOT_CHANGED) on 5162, so the built-in slot still executes the
// legacy implementation there. Taking over 5162 later requires verifying the dependency
// ops (Sort/UniqueConsecutive/AdjacentDifference/Cumsum/ScatterElements) on that arch
// first and adjusting the set plus the UT platform resources accordingly; extending the
// set for other archs follows the same rule.
const std::set<std::string> kSupportedNpuArchs = {"3510", "5102", "9201", "9202"};

bool IsRegbasePlatform() { return ops::CheckNpuArch(kPassName, kSupportedNpuArchs); }

bool IsSupportedInputDtype(const DataType dtype)
{
    return std::find(kXYDtypeSupportList.begin(), kXYDtypeSupportList.end(), dtype) != kXYDtypeSupportList.end();
}

bool HasOutConsumer(const GNode& node, const int64_t outIndex)
{
    return !node.GetOutDataNodesAndPortIndexs(static_cast<int32_t>(outIndex)).empty();
}

// idx output is used by downstream nodes only (the return_inverse/return_idx attr check in the
// original pass is an unreachable branch for Unique/UniqueWithCounts).
bool IsIdxOut(const GNode& node) { return HasOutConsumer(node, kUniqueIdxOut); }

// count output exists only for UniqueWithCounts and only matters when it has downstream consumers.
bool IsCntOut(const GNode& node)
{
    AscendString nodeType;
    if (node.GetType(nodeType) != SUCCESS) {
        return false;
    }
    if (nodeType != kUniqueWithCountsType) {
        return false;
    }
    return HasOutConsumer(node, kUniqueCntOut);
}

// Migrated from the original CheckUniqueNode: input must be 1-D, dtype must be in the whitelist and
// the index output dtype (out_idx attr or output descs) must be int32/int64. Also resolves the
// outIdxDtype used to build the replacement graph.
bool CheckUniqueNode(const GNode& node, DataType& outIdxDtype)
{
    TensorDesc xDesc;
    if (node.GetInputDesc(kUniqueValIn, xDesc) != SUCCESS) {
        OPS_LOG_D(kPassName.c_str(), "Failed to get input x desc.");
        return false;
    }
    const size_t xDimNum = xDesc.GetShape().GetDimNum();
    if (xDimNum != 1) {
        OPS_LOG_D(kPassName.c_str(), "Unique input dim is %zu which is not support.", xDimNum);
        return false;
    }
    const DataType xDtype = xDesc.GetDataType();
    if (!IsSupportedInputDtype(xDtype)) {
        OPS_LOG_D(kPassName.c_str(), "Unique input dtype is %d which is not support.", static_cast<int32_t>(xDtype));
        return false;
    }

    // out_idx attr check: attr first (DataType or Int storage), then output descs.
    DataType attrDtype = DT_UNDEFINED;
    bool hasAttr = false;
    if (node.GetAttr("out_idx", attrDtype) == SUCCESS) {
        hasAttr = true;
    } else {
        int64_t attrDtypeInt = static_cast<int64_t>(DT_INT32);
        if (node.GetAttr("out_idx", attrDtypeInt) == SUCCESS) {
            attrDtype = static_cast<DataType>(attrDtypeInt);
            hasAttr = true;
        }
    }
    if (hasAttr) {
        OPS_LOG_D(kPassName.c_str(), "Get attr.out_idx = %d", static_cast<int32_t>(attrDtype));
        if (attrDtype != DT_INT32 && attrDtype != DT_INT64) {
            OPS_LOG_D(kPassName.c_str(), "Aicore does not support idxout's type except int64 and int32.");
            return false;
        }
        outIdxDtype = attrDtype;
        return true;
    }

    // attr absent: every index-like output desc must be int32/int64, outIdxDtype follows output 1.
    const size_t outputCount = node.GetOutputsSize();
    for (size_t i = 1; i < outputCount; i++) {
        TensorDesc outputDesc;
        if (node.GetOutputDesc(i, outputDesc) != SUCCESS) {
            OPS_LOG_D(kPassName.c_str(), "Failed to get output %zu desc.", i);
            return false;
        }
        const DataType outputDtype = outputDesc.GetDataType();
        if (outputDtype != DT_INT32 && outputDtype != DT_INT64) {
            OPS_LOG_D(kPassName.c_str(), "Aicore does not support idxout's type except int64 and int32.");
            return false;
        }
    }
    TensorDesc idxDesc;
    if (node.GetOutputDesc(kUniqueIdxOut, idxDesc) != SUCCESS) {
        OPS_LOG_D(kPassName.c_str(), "Failed to get idx output desc.");
        return false;
    }
    outIdxDtype = idxDesc.GetDataType();
    return true;
}

// Force the output dtype of a node port, keeping the (already inferred) shape untouched. Sort/Cumsum/
// AdjacentDifference have no registered InferDataType, so their explicit dtype settings must be
// re-applied after GeUtils::InferShape, mirroring the original pass which set them after
// NodeUtilsEx::InferShapeAndType.
void ForceOutputDtype(GNode& node, const int32_t outIndex, const DataType dtype)
{
    TensorDesc outputDesc;
    if (node.GetOutputDesc(outIndex, outputDesc) != SUCCESS) {
        OPS_LOG_E(kPassName.c_str(), "Failed to get output %d desc of node.", outIndex);
        return;
    }
    outputDesc.SetDataType(dtype);
    node.UpdateOutputDesc(outIndex, outputDesc);
}

// Cumsum and AdjacentDifference are canndev/ops-math operators, the ops-nn build environment does not
// generate ES APIs for them, so build them via CompliantNodeBuilder, consistent with the repo practice.
GNode CreateAdjacentDifferenceNode(es::EsGraphBuilder& graphBuilder, const std::string& nodeName,
                                   const std::vector<int64_t>& xDims, const DataType outIdxDtype, const Format xFormat)
{
    auto* graph = graphBuilder.GetCGraphBuilder()->GetGraph();
    auto adjDiffNode = es::CompliantNodeBuilder(graph)
                           .OpType("AdjacentDifference")
                           .Name(nodeName.c_str())
                           .IrDefInputs({{"x", es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
                           .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                           .IrDefAttrs({{"y_dtype", es::CompliantNodeBuilder::kEsAttrOptional, "Int",
                                         es::CreateFrom(static_cast<int64_t>(outIdxDtype))}})
                           .InstanceOutputShape("y", xDims)
                           .InstanceOutputDataType("y", outIdxDtype)
                           .InstanceOutputFormat("y", xFormat)
                           .Build();
    return adjDiffNode;
}

GNode CreateCumsumNode(es::EsGraphBuilder& graphBuilder, const std::string& nodeName, const std::vector<int64_t>& xDims,
                       const DataType outIdxDtype, const Format xFormat)
{
    auto* graph = graphBuilder.GetCGraphBuilder()->GetGraph();
    auto cumsumNode = es::CompliantNodeBuilder(graph)
                          .OpType("Cumsum")
                          .Name(nodeName.c_str())
                          .IrDefInputs({{"x", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                                        {"axis", es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
                          .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                          .IrDefAttrs(
                              {{"exclusive", es::CompliantNodeBuilder::kEsAttrOptional, "Bool", es::CreateFrom(false)},
                               {"reverse", es::CompliantNodeBuilder::kEsAttrOptional, "Bool", es::CreateFrom(false)}})
                          .InstanceOutputShape("y", xDims)
                          .InstanceOutputDataType("y", outIdxDtype)
                          .InstanceOutputFormat("y", xFormat)
                          .Build();
    // axis const: int64, shape {1}, value 0, same as the AddWeight in the original pass.
    auto cumsumAxis = graphBuilder.CreateConst(std::vector<int64_t>{kCumsumAxis}, {1});
    es::AddEdgeAndUpdatePeerDesc(*graph, *cumsumAxis.GetProducer(), cumsumAxis.GetProducerOutIndex(), cumsumNode,
                                 kCumsumAxisIn);
    return cumsumNode;
}

GraphUniqPtr CreateReplacement(const GNode& node, const bool idxOut, const bool cntOut, const DataType outIdxDtype)
{
    // 1. read the original node descs, used as input desc and output desc backfill source.
    TensorDesc xDesc;
    TensorDesc yDesc;
    TensorDesc idxDesc;
    TensorDesc cntDesc;
    if (node.GetInputDesc(kUniqueValIn, xDesc) != SUCCESS || node.GetOutputDesc(kUniqueValOut, yDesc) != SUCCESS ||
        node.GetOutputDesc(kUniqueIdxOut, idxDesc) != SUCCESS) {
        OPS_LOG_E(kPassName.c_str(), "Failed to get input/output desc of unique node.");
        return nullptr;
    }
    if (cntOut && node.GetOutputDesc(kUniqueCntOut, cntDesc) != SUCCESS) {
        OPS_LOG_E(kPassName.c_str(), "Failed to get count output desc of unique node.");
        return nullptr;
    }
    const std::vector<int64_t> xDims = xDesc.GetShape().GetDims();
    const Format xFormat = xDesc.GetFormat();

    AscendString nodeNameStr;
    if (node.GetName(nodeNameStr) != SUCCESS) {
        OPS_LOG_E(kPassName.c_str(), "Failed to get node name.");
        return nullptr;
    }
    const std::string baseName(nodeNameStr.GetString());

    auto graphBuilder = es::EsGraphBuilder("replacement");
    auto rX = graphBuilder.CreateInput(0, "x", xDesc.GetDataType(), xFormat, xDims);

    // 2. main structure: Sort(axis=0, stable) -> UniqueConsecutive.
    auto sortOut = es::Sort(rX, kSortAxis, kSortDescending, kSortStable, outIdxDtype);
    GNode sortNode = *sortOut.y1.GetProducer();
    // set Sort output descs explicitly, same as the original pass (infer funcs are shape-only).
    TensorDesc sortY1Desc(xDesc.GetShape(), xFormat, yDesc.GetDataType());
    sortNode.UpdateOutputDesc(kSortValueOut, sortY1Desc);
    TensorDesc sortY2Desc(xDesc.GetShape(), xFormat, outIdxDtype);
    sortNode.UpdateOutputDesc(kSortIdxOut, sortY2Desc);

    auto uniqueConsOut = es::UniqueConsecutive(sortOut.y1, /*returnIdx=*/false, cntOut, kUniqueConsecutiveAxis,
                                               outIdxDtype);
    GNode uniqueConsNode = *uniqueConsOut.y.GetProducer();
    // The legacy pass explicitly sets _unknown_shape_type = 3 (THRID_CLASS_DYN_TYPE, the
    // third-class dynamic shape flag) on the UniqueConsecutive node in SetUniqueConsAttr
    // and gives up the fusion when it fails; the ES interface only carries
    // return_idx/return_counts/axis/out_idx, so re-apply it here to keep the replacement
    // graph identical on this attribute. (GNode::SetAttr takes a non-const int64_t&, so
    // pass through a local value.)
    int64_t unknownShapeType = kThirdClassDynType;
    if (uniqueConsNode.SetAttr("_unknown_shape_type", unknownShapeType) != ge::GRAPH_SUCCESS) {
        OPS_LOG_E(kPassName.c_str(), "Failed to set _unknown_shape_type on UniqueConsecutive, do nothing.");
        return nullptr;
    }
    // backfill y/count output descs with the original unique node's descs, mirrors TransferOutputs.
    uniqueConsNode.UpdateOutputDesc(kUniqueValOut, yDesc);
    if (cntOut) {
        uniqueConsNode.UpdateOutputDesc(kUniqueCntOut, cntDesc);
    }

    std::vector<es::EsTensorHolder> outputs;
    outputs.push_back(uniqueConsOut.y); // boundary output 0: y

    // 3. inverse index structure: AdjacentDifference -> Cumsum -> ScatterElements (only when idx used).
    GNode adjDiffNode;
    GNode cumsumNode;
    GNode scatterNode;
    if (idxOut) {
        auto* graph = graphBuilder.GetCGraphBuilder()->GetGraph();

        adjDiffNode = CreateAdjacentDifferenceNode(graphBuilder, baseName + "/AdjacentDifference", xDims, outIdxDtype,
                                                   xFormat);
        es::AddEdgeAndUpdatePeerDesc(*graph, *sortOut.y1.GetProducer(), sortOut.y1.GetProducerOutIndex(), adjDiffNode,
                                     kAdjDiffXIn);

        cumsumNode = CreateCumsumNode(graphBuilder, baseName + "/Cumsum", xDims, outIdxDtype, xFormat);
        es::AddEdgeAndUpdatePeerDesc(*graph, adjDiffNode, kAdjDiffYOut, cumsumNode, kCumsumXIn);
        auto* cumsumYHolder = graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(cumsumNode, kCumsumYOut);
        auto cumsumY = es::EsTensorHolder(cumsumYHolder);

        auto scatterOut = es::ScatterElements(cumsumY, sortOut.y2, cumsumY, kScatterAxis, kScatterReduction);
        scatterNode = *scatterOut.GetProducer();
        // backfill the original idx output desc, mirrors TransferOutputs.
        scatterNode.UpdateOutputDesc(kScatterYOut, idxDesc);

        outputs.push_back(scatterOut); // boundary output: idx
    }
    if (cntOut) {
        outputs.push_back(uniqueConsOut.count); // boundary output: count
    }

    GraphUniqPtr replaceGraph = graphBuilder.BuildAndReset(outputs);
    if (replaceGraph == nullptr) {
        OPS_LOG_E(kPassName.c_str(), "Failed to build replacement graph.");
        return nullptr;
    }

    // 4. infer shape/dtype for the whole replacement graph; give up fusion on failure.
    if (GeUtils::InferShape(*replaceGraph, {xDesc.GetShape()}) != SUCCESS) {
        OPS_LOG_E(kPassName.c_str(), "InferShape for replacement failed.");
        return nullptr;
    }

    // 5. re-apply the explicit dtype settings of intermediate/boundary nodes after inference, same as
    // the original pass (Sort/AdjacentDifference/Cumsum have no InferDataType registration).
    ForceOutputDtype(sortNode, kSortValueOut, yDesc.GetDataType());
    ForceOutputDtype(sortNode, kSortIdxOut, outIdxDtype);
    if (idxOut) {
        ForceOutputDtype(adjDiffNode, kAdjDiffYOut, outIdxDtype);
        ForceOutputDtype(cumsumNode, kCumsumYOut, outIdxDtype);
        // scatter idx output keeps the original idx desc (shape + dtype + format).
        scatterNode.UpdateOutputDesc(kScatterYOut, idxDesc);
    }
    if (cntOut) {
        // the original pass forces the count dtype to follow the idx dtype.
        ForceOutputDtype(uniqueConsNode, kUniqueCntOut, outIdxDtype);
    }

    return replaceGraph;
}

// Construct the subgraph boundary of the original unique node. Boundary slot i maps to the i-th
// output of the replacement graph: y -> idx(when used) -> count(when used). Only outputs that have
// downstream consumers are declared, so unused outputs are neither declared nor computed.
std::unique_ptr<SubgraphBoundary> ConstructBoundary(const GNode& node, const bool idxOut, const bool cntOut)
{
    auto boundary = std::make_unique<SubgraphBoundary>();

    SubgraphInput subgraphInput;
    subgraphInput.AddInput({node, kUniqueValIn});
    if (boundary->AddInput(0, std::move(subgraphInput)) != SUCCESS) {
        OPS_LOG_E(kPassName.c_str(), "Failed to add input 0 to boundary.");
        return nullptr;
    }

    SubgraphOutput yOutput({node, kUniqueValOut});
    if (boundary->AddOutput(0, std::move(yOutput)) != SUCCESS) {
        OPS_LOG_E(kPassName.c_str(), "Failed to add output 0 to boundary.");
        return nullptr;
    }
    int64_t boundarySlot = 1;
    if (idxOut) {
        SubgraphOutput idxOutput({node, kUniqueIdxOut});
        if (boundary->AddOutput(boundarySlot, std::move(idxOutput)) != SUCCESS) {
            OPS_LOG_E(kPassName.c_str(), "Failed to add output %ld to boundary.", boundarySlot);
            return nullptr;
        }
        boundarySlot++;
    }
    if (cntOut) {
        SubgraphOutput cntOutput({node, kUniqueCntOut});
        if (boundary->AddOutput(boundarySlot, std::move(cntOutput)) != SUCCESS) {
            OPS_LOG_E(kPassName.c_str(), "Failed to add output %ld to boundary.", boundarySlot);
            return nullptr;
        }
    }
    return boundary;
}

// Replace a single unique node: create replacement graph -> construct boundary -> rewrite.
bool ReplaceUniqueNode(const GNode& node, const DataType outIdxDtype, CustomPassContext& passContext)
{
    const bool idxOut = IsIdxOut(node);
    const bool cntOut = IsCntOut(node);

    auto replacement = CreateReplacement(node, idxOut, cntOut, outIdxDtype);
    if (replacement == nullptr) {
        OPS_LOG_E(kPassName.c_str(), "Failed to create replacement graph.");
        return false;
    }

    auto boundary = ConstructBoundary(node, idxOut, cntOut);
    if (boundary == nullptr) {
        OPS_LOG_E(kPassName.c_str(), "Failed to construct subgraph boundary.");
        return false;
    }

    Status replaceStatus = FAILED;
    if (SubgraphRewriterReplaceWithCtx != nullptr) {
        replaceStatus = SubgraphRewriterReplaceWithCtx(*boundary, *replacement, passContext);
    } else {
        OPS_LOG_D(kPassName.c_str(), "Three-arg Replace is unavailable, fallback to two-arg version.");
        replaceStatus = SubgraphRewriter::Replace(*boundary, *replacement);
    }
    if (replaceStatus != SUCCESS) {
        OPS_LOG_E(kPassName.c_str(), "SubgraphRewriter::Replace failed, status=%d.", static_cast<int>(replaceStatus));
        return false;
    }

    OPS_LOG_D(kPassName.c_str(), "UniqueSortFusionPass replacement succeeded.");
    return true;
}
} // namespace

Status UniqueSortFusionPass::Run(GraphPtr& graph, CustomPassContext& passContext)
{
    OPS_LOG_D(kPassName.c_str(), "Enter UniqueSortFusionPass.");

    // 版本兼容(D1 整体静默): 9.x 编出的包跑在老 ge_compiler 运行时上时,
    // pass 降级注册到老 stage 占位,此处空跑,不发生融合。
    const int32_t geCompilerVersion = GetRuntimeGeCompilerVersion();
    if (geCompilerVersion < kGeCompilerVersion900) {
        OPS_LOG_D(kPassName.c_str(), "GE runtime version %d < 9.0.0, skip pass.", geCompilerVersion);
        return GRAPH_NOT_CHANGED;
    }

    if (!IsRegbasePlatform()) {
        return GRAPH_NOT_CHANGED;
    }

    // collect first, replace later: GetDirectNode returns a copy, keeps node handles valid while the
    // graph is being rewritten, and supports multiple unique nodes in one graph.
    std::vector<std::pair<GNode, DataType>> matchedNodes;
    for (auto& node : graph->GetDirectNode()) {
        AscendString nodeType;
        if (node.GetType(nodeType) != SUCCESS) {
            OPS_LOG_D(kPassName.c_str(), "GetType failed during node traversal.");
            continue;
        }
        if (nodeType != kUniqueType && nodeType != kUniqueWithCountsType) {
            continue;
        }
        DataType outIdxDtype = DT_INT32;
        if (!CheckUniqueNode(node, outIdxDtype)) {
            continue;
        }
        matchedNodes.emplace_back(node, outIdxDtype);
    }

    if (matchedNodes.empty()) {
        OPS_LOG_D(kPassName.c_str(), "No unique nodes to fuse.");
        return GRAPH_NOT_CHANGED;
    }

    // transaction: restore the origin graph on any failure so the compilation falls back to the
    // original AICPU unique implementation.
    Graph originGraph = *graph;
    for (auto& matched : matchedNodes) {
        if (!ReplaceUniqueNode(matched.first, matched.second, passContext)) {
            OPS_LOG_E(kPassName.c_str(), "ReplaceUniqueNode failed, rollback the whole graph.");
            *graph = originGraph;
            return GRAPH_NOT_CHANGED;
        }
    }

    OPS_LOG_D(kPassName.c_str(), "Fusion completed, %zu nodes fused.", matchedNodes.size());
    return SUCCESS;
}

// ===========================================================================
// 注册 —— 版本兼容(D1:仅使用新增 Stage 枚举 kCompatibleInherited @9.0.0)
// ===========================================================================
#if GE_COMPILER_VERSION_NUM >= 90000000
namespace {
// 9.x 编译产物的注册 stage 由运行时版本决定:
//   - 9.x 运行时:注册到期望的 kCompatibleInherited,正常融合
//   - 老运行时:kCompatibleInherited 调度点不存在,降级注册到老版本已有的
//     kBeforeInferShape 占位,并由 Run 入口的版本检查空跑(整体静默)
CustomPassStage GetRegisterStage()
{
    if (GetRuntimeGeCompilerVersion() >= kGeCompilerVersion900) {
        return CustomPassStage::kCompatibleInherited;
    }
    return CustomPassStage::kBeforeInferShape;
}
} // namespace
REG_FUSION_PASS(UniqueSortFusionPass).Stage(GetRegisterStage());
#else
// 老版本 toolkit 编译:kCompatibleInherited 枚举不存在,注册到已有 kBeforeInferShape;
// pass 逻辑依赖 ES 框架接口,老版本运行时由 Run 入口的版本检查空跑兜底。
REG_FUSION_PASS(UniqueSortFusionPass).Stage(CustomPassStage::kBeforeInferShape);
#endif

} // namespace ops

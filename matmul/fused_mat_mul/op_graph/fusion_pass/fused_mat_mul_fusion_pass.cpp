/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the License).
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fused_mat_mul_fusion_pass.cpp
 * \brief the pass will effect if the graph pattern is as follows (single pass with the platform
 *        dispatch inside, aligned with the canndev FusedMatMulFusionPass):
 *
 * [DAV_3510 direct path] (x1/x2/out/bias: fp16, bf16 or fp32-with-hf32)
 *   x1   x2   bias(optional)                     x1   x2   bias(optional)
 *    +    +        +                              +    +        +
 *    |    |        |                              |    |        |
 *    +-> MatMulV3/BatchMatMulV3 <-+      ===>     +-> FusedMatMul <-+
 *                |                                     |
 *                v                                     v
 *              Relu                                   out
 *
 * [s8s4 mmad PostCube path] (fp16 only)
 *   x1   x2   bias(optional)          x1   x2   bias(optional)  x3(quant, optional)
 *    +    +        +                   +    +        +             +
 *    |    |        |                   |    |        |             |
 *    +-> MatMulV3/BatchMatMulV3 <-+ => +-> FusedMatMul <-----------+
 *                |                        fused_op_type = relu / quant / relu_quant
 *                v
 *     Relu / AscendQuant / Relu->AscendQuant
 *
 * The s8s4 path reuses the open-sourced PostCube utils (common/op_graph/cube_utils, the renamed
 * FixPipe capability) to fold the post op chain into a FixPipe node and then replace
 * {matmul + FixPipe} with a single FusedMatMul node. The graph-order [Relu, AscendQuant] chain
 * folds the Relu only (the postcube units require the quant before the activation): the trailing
 * AscendQuant is then absorbed into the FusedMatMul(relu_quant) replace (the canndev
 * TransferOpToFusedMatMul semantics), while the forbidden-node ability flag stays provided by
 * the canndev FixPipeAbilityProcessPass (see the notes before the s8s4 section).
 */

#include "fused_mat_mul_fusion_pass.h"

#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "common/inc/error_util.h"
#include "common/op_graph/fusion_pass/matmul_fusion_utils_pass.h"
// weak-symbol declarations of SubgraphRewriter (the 3-param Replace reports the fusion result)
#include "common/op_graph/fusion_pass/weight_quant_fusion_utils.h"
#include "cube_utils/cube_addinputstrategy.h"
#include "cube_utils/cube_common.h"
#include "cube_utils/cube_utils.h"
#include "es_nn_ops.h"
#include "ge/compliant_node_builder.h"
#include "ge/es_graph_builder.h"
#include "ge/fusion/subgraph_boundary.h"
#include "platform/platform_info.h"
#include "platform/soc_spec.h"
#include "version/ge-compiler_version.h"
#include "acl/acl_rt.h"

// The preprocessor form of kTargetGeCompilerVersion (a #if guard cannot use a constexpr): the
// minimum ge-compiler version providing the CustomPassStage API and the 3-param Replace.
#if !defined(GE_COMPILER_TARGET_VERSION_NUM)
#define GE_COMPILER_TARGET_VERSION_NUM 90100000U
#endif

using namespace ge;
using namespace ge::es;
using namespace ge::fusion;
using namespace fe;

namespace ops {
namespace {

constexpr char kPassName[] = "FusedMatMulFusionPass";
constexpr char kOpTypeMatMulV3[] = "MatMulV3";
constexpr char kOpTypeBatchMatMulV3[] = "BatchMatMulV3";
constexpr char kOpTypeRelu[] = "Relu";
constexpr char kOpTypeAscendQuant[] = "AscendQuant";
constexpr char kOpTypeFixPipe[] = "FixPipe"; // the PostCube node keeps the original FixPipe type
constexpr char kOpTypeFusedMatMul[] = "FusedMatMul";
constexpr char kFusedOpTypeRelu[] = "relu";
constexpr char kFusedOpTypeQuant[] = "quant";
constexpr char kFusedOpTypeReluQuant[] = "relu_quant";
constexpr char kAttrEnableHf32[] = "enable_hf32";
constexpr char kAttrOpImplMode[] = "opImplMode";
constexpr char kAttrFusionOpList[] = "fusion_op_list";
constexpr char kFp16FusionPrecisionMode[] = "force_fp16";

constexpr uint64_t kHf32Flag = 0x40UL;
constexpr uint32_t kMatmulCheckNumNoBias = 3;
constexpr uint32_t kMatmulCheckNumBias = 4;
constexpr size_t kBiasDimNumMax3510 = 2;
constexpr size_t kBiasDimNumOne = 1;
constexpr size_t kMatmulDimNum = 2;
constexpr int64_t kMaxMergedMDim = INT32_MAX;
constexpr int64_t kFmmX3InputIdx = 3;          // FusedMatMul.x3 (quant scale/offset scalar)
constexpr int64_t kPostCubeQuantScale0Idx = 2; // FixPipe.quant_scale_0
constexpr int64_t kFmmQuantInputStrategy = 21; // canndev FMM_QUANT_INPUT_STRATEGY (SetQuantOffsetScale)
constexpr size_t kReluQuantChainOpNum = 2;     // op count of the [Relu, AscendQuant] post-cube chain

// x1/x2/out/bias dtype combinations: DAV_3510 supports fp16/bf16/fp32(hf32), s8s4 fp16 only
// (canndev FUSED_MATMUL_3510_RELU_SUPPORT_DTYPES / FUSED_MATMUL_FUSION_SUPPORT_DTYPES).
const std::vector<std::vector<DataType>> kDav3510SupportDtypes = {{DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16},
                                                                  {DT_BF16, DT_BF16, DT_BF16, DT_BF16},
                                                                  {DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT}};

const std::vector<std::vector<DataType>> kS8s4SupportDtypes = {{DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16}};

const std::vector<Format> kMatmulFormatSupportList = {FORMAT_ND,    FORMAT_NCHW,  FORMAT_NHWC,
                                                      FORMAT_HWCN,  FORMAT_CHWN,  FORMAT_NDHWC,
                                                      FORMAT_NCDHW, FORMAT_DHWCN, FORMAT_DHWNC};

// The runtime ge-compiler version gate for the registration stage (see the registration at the
// bottom): the kCompatibleInherited override (顶掉 the canndev builtin pass) requires the
// ge-compiler runtime of the target version.
bool IsTargetVersion()
{
    int32_t version = 0;
    if (aclsysGetVersionNum("ge-compiler", &version) != ACL_SUCCESS) {
        OPS_LOG_W(kPassName, "Failed to get ge-compiler version, skip fusion.");
        return false;
    }
    return version >= kTargetGeCompilerVersion;
}

// ==================== common helpers ====================

std::string NodeName(const GNode& node)
{
    AscendString name;
    if (node.GetName(name) != SUCCESS || name.GetString() == nullptr) {
        return "";
    }
    return std::string(name.GetString());
}

// GetOutDataNodesAndPortIndexs returns a fresh GNode instance per call, so the node identity
// must be compared by name (unique inside the graph), not by pointer.
bool IsSameNode(const GNodePtr& lhs, const GNodePtr& rhs)
{
    if (lhs == nullptr || rhs == nullptr) {
        return false;
    }
    const std::string lhsName = NodeName(*lhs);
    return !lhsName.empty() && lhsName == NodeName(*rhs);
}

// Prefer the GE 9.1.0+ 3-param Replace (auto fuseability check + fusion result report, e.g.
// fusion_result.json for the test frameworks); fall back to the 2-param version on old runtimes.
Status ReplaceSubgraphWithReport(const SubgraphBoundary& boundary, const Graph& replacement,
                                 CustomPassContext& passContext)
{
#if GE_COMPILER_VERSION_NUM >= GE_COMPILER_TARGET_VERSION_NUM
    using ReplaceWithCtxFn = Status (*)(const SubgraphBoundary&, const Graph&, CustomPassContext&);
    auto replaceWithCtx = static_cast<ReplaceWithCtxFn>(&SubgraphRewriter::Replace);
    if (replaceWithCtx != nullptr) {
        return replaceWithCtx(boundary, replacement, passContext);
    }
#endif
    return SubgraphRewriter::Replace(boundary, replacement);
}

// The x1/x2/bias inputs of the replacement graph, shared by both replace functions.
struct MatmulReplacementInputs {
    EsTensorHolder x1;
    EsTensorHolder x2;
    EsTensorHolder bias;
};

// Add the matmul x1/x2/bias boundary inputs shared by both replace functions.
Status AddMatmulBoundaryInputs(SubgraphBoundary& boundary, const GNode& matmulNode, bool hasBias, int64_t& graphIndex)
{
    {
        SubgraphInput subgraphInput;
        FUSION_PASS_CHECK(subgraphInput.AddInput({matmulNode, kX1InputIdx}) != GRAPH_SUCCESS ||
                              boundary.AddInput(graphIndex++, std::move(subgraphInput)) != GRAPH_SUCCESS,
                          OPS_LOG_E(kPassName, "Add x1 boundary input failed."), return FAILED);
    }
    {
        SubgraphInput subgraphInput;
        FUSION_PASS_CHECK(subgraphInput.AddInput({matmulNode, kX2InputIdx}) != GRAPH_SUCCESS ||
                              boundary.AddInput(graphIndex++, std::move(subgraphInput)) != GRAPH_SUCCESS,
                          OPS_LOG_E(kPassName, "Add x2 boundary input failed."), return FAILED);
    }
    if (hasBias) {
        SubgraphInput subgraphInput;
        FUSION_PASS_CHECK(subgraphInput.AddInput({matmulNode, kBiasInputIdx}) != GRAPH_SUCCESS ||
                              boundary.AddInput(graphIndex++, std::move(subgraphInput)) != GRAPH_SUCCESS,
                          OPS_LOG_E(kPassName, "Add bias boundary input failed."), return FAILED);
    }
    return SUCCESS;
}

// Create the x1/x2/bias inputs of the replacement graph shared by both replace functions.
MatmulReplacementInputs CreateMatmulReplacementInputs(EsGraphBuilder& builder, int64_t& replaceIndex,
                                                      const TensorDesc& x1Desc, const TensorDesc& x2Desc,
                                                      const TensorDesc& biasDesc, bool hasBias)
{
    MatmulReplacementInputs inputs;
    inputs.x1 = builder.CreateInput(replaceIndex++, "x1", x1Desc.GetDataType(), x1Desc.GetFormat(),
                                    x1Desc.GetShape().GetDims());
    inputs.x2 = builder.CreateInput(replaceIndex++, "x2", x2Desc.GetDataType(), x2Desc.GetFormat(),
                                    x2Desc.GetShape().GetDims());
    if (hasBias) {
        inputs.bias = builder.CreateInput(replaceIndex++, "bias", biasDesc.GetDataType(), biasDesc.GetFormat(),
                                          biasDesc.GetShape().GetDims());
    }
    return inputs;
}

// The fused node's inputs inherit the matmul input descs, shared by both replace functions.
void InheritMatmulInputDescs(GNode& fusedMatMulNode, const GNode& matmulNode)
{
    TensorDesc matmulInputDesc;
    for (int64_t inputIdx = 0; inputIdx <= kBiasInputIdx; inputIdx++) {
        if (matmulNode.GetInputDesc(inputIdx, matmulInputDesc) == GRAPH_SUCCESS) {
            fusedMatMulNode.UpdateInputDesc(inputIdx, matmulInputDesc);
        }
    }
}

bool IsMatmulType(const AscendString& nodeType)
{
    return nodeType == kOpTypeMatMulV3 || nodeType == kOpTypeBatchMatMulV3;
}

bool IsBatchMatMulV3Type(const GNode& node)
{
    AscendString nodeType;
    return node.GetType(nodeType) == GRAPH_SUCCESS && nodeType == kOpTypeBatchMatMulV3;
}

// A failed GetAttr means the attr is absent and the IR default (false) applies: the ES builder
// skips setting attrs equal to the IR default value, same as canndev.
void GetTransposeAttrs(const GNode& matmulNode, bool& transX1, bool& transX2)
{
    const char* transStrX1 = IsBatchMatMulV3Type(matmulNode) ? kAttrAdjX1 : kAttrTransposeX1;
    const char* transStrX2 = IsBatchMatMulV3Type(matmulNode) ? kAttrAdjX2 : kAttrTransposeX2;
    if (matmulNode.GetAttr(transStrX1, transX1) != SUCCESS) {
        transX1 = false;
    }
    if (matmulNode.GetAttr(transStrX2, transX2) != SUCCESS) {
        transX2 = false;
    }
}

// Collect the matmul nodes first: the graph is modified during the fusion loop.
std::vector<GNode> GetMatmulNodes(const GraphPtr& graph)
{
    std::vector<GNode> matmulNodes;
    for (auto& node : graph->GetDirectNode()) {
        AscendString nodeType;
        if (node.GetType(nodeType) != GRAPH_SUCCESS) {
            continue;
        }
        if (IsMatmulType(nodeType)) {
            matmulNodes.emplace_back(node);
        }
    }
    return matmulNodes;
}

// Shared prologue of both fusion loops: no output consumer, or the optional input at index 3
// (offset_w) present.
bool IsMatmulSkippable(const GNode& matmulNode)
{
    if (matmulNode.GetOutDataNodesAndPortIndexs(0).empty()) {
        OPS_LOG_D(kPassName, "Matmul out nodes is empty, don't need process.");
        return true;
    }
    if (matmulNode.GetInDataNodesAndPortIndexs(kOffsetWInputIdx).first != nullptr) {
        OPS_LOG_D(kPassName, "Matmul with offset_w is not supported, no fusion.");
        return true;
    }
    return false;
}

bool CheckMatmulFormat(const GNode& matmulNode)
{
    TensorDesc x1Desc;
    TensorDesc x2Desc;
    FUSION_PASS_CHECK(matmulNode.GetInputDesc(kX1InputIdx, x1Desc) != SUCCESS ||
                          matmulNode.GetInputDesc(kX2InputIdx, x2Desc) != SUCCESS,
                      OPS_LOG_W(kPassName, "Failed to get matmul input desc."), return false);

    FUSION_PASS_CHECK(std::find(kMatmulFormatSupportList.begin(), kMatmulFormatSupportList.end(), x1Desc.GetFormat()) ==
                          kMatmulFormatSupportList.end(),
                      OPS_LOG_D(kPassName, "X1's format is not supported, no fusion."), return false);
    FUSION_PASS_CHECK(std::find(kMatmulFormatSupportList.begin(), kMatmulFormatSupportList.end(), x2Desc.GetFormat()) ==
                          kMatmulFormatSupportList.end(),
                      OPS_LOG_D(kPassName, "X2's format is not supported, no fusion."), return false);
    return true;
}

// Whether the bias last dim matches the matmul output last dim (shared by both paths).
bool CheckBiasLastDimMatch(const GNode& matmulNode, const Shape& biasShape)
{
    TensorDesc outputDesc;
    FUSION_PASS_CHECK(matmulNode.GetOutputDesc(0, outputDesc) != SUCCESS,
                      OPS_LOG_D(kPassName, "Failed to get matmul output desc."), return false);
    const Shape outputShape = outputDesc.GetShape();
    FUSION_PASS_CHECK(outputShape.GetDimNum() == 0,
                      OPS_LOG_D(kPassName, "Matmul output shape is empty, cannot check bias shape."), return false);

    // The 0-dim bias takes 0 as its last dim: biasN > 0 is short-circuited and the fusion
    // continues, matching the canndev 0-dim GetDim(-1) behavior without relying on the UB.
    const int64_t biasN = (biasShape.GetDimNum() == 0) ? 0 : biasShape.GetDim(biasShape.GetDimNum() - 1);
    const int64_t outputN = outputShape.GetDim(outputShape.GetDimNum() - 1);
    FUSION_PASS_CHECK(
        biasN > 0 && outputN > 0 && biasN != outputN,
        OPS_LOG_D(kPassName, "Bias dim[%lld] must be equal to matmul output n dim[%lld].", biasN, outputN),
        return false);
    return true;
}

// FP32 FusedMatMul only supports the HF32 computation mode.
// MatMulV3 exposes opImplMode, while BatchMatMulV3 exposes enable_hf32 (aligned with canndev master).
bool GetEnableHf32FromMatmul(const GNode& matmulNode)
{
    int64_t opImplMode = 0;
    if (matmulNode.GetAttr(kAttrOpImplMode, opImplMode) == SUCCESS) {
        return (static_cast<uint64_t>(opImplMode) & kHf32Flag) != 0UL;
    }
    bool enableHf32 = false;
    (void)matmulNode.GetAttr(kAttrEnableHf32, enableHf32);
    return enableHf32;
}

// Dtype check shared by both paths (canndev CheckMatmulDtype): scan the platform support table
// - DAV_3510: fp16/bf16/fp32, s8s4: fp16 only. The FP32 HF32 gate below is dead code on s8s4
// (its table rejects FP32 first), the same as canndev.
bool CheckMatmulDtype(const GNode& matmulNode, bool hasBias, bool isDav3510)
{
    TensorDesc x1Desc;
    TensorDesc x2Desc;
    TensorDesc outputDesc;
    FUSION_PASS_CHECK(matmulNode.GetInputDesc(kX1InputIdx, x1Desc) != SUCCESS ||
                          matmulNode.GetInputDesc(kX2InputIdx, x2Desc) != SUCCESS ||
                          matmulNode.GetOutputDesc(0, outputDesc) != SUCCESS,
                      OPS_LOG_W(kPassName, "Failed to get matmul input/output desc."), return false);

    std::vector<DataType> matmulDtypes = {x1Desc.GetDataType(), x2Desc.GetDataType(), outputDesc.GetDataType()};
    if (hasBias) {
        TensorDesc biasDesc;
        FUSION_PASS_CHECK(matmulNode.GetInputDesc(kBiasInputIdx, biasDesc) != SUCCESS,
                          OPS_LOG_W(kPassName, "Failed to get bias input desc."), return false);
        matmulDtypes.emplace_back(biasDesc.GetDataType());
    }

    const std::vector<std::vector<DataType>>& supportDtypes = isDav3510 ? kDav3510SupportDtypes : kS8s4SupportDtypes;
    const uint32_t compareSize = hasBias ? kMatmulCheckNumBias : kMatmulCheckNumNoBias;
    bool dtypeCheck = false;
    for (const auto& supportDtype : supportDtypes) {
        if (std::equal(matmulDtypes.begin(), matmulDtypes.begin() + compareSize, supportDtype.begin())) {
            dtypeCheck = true;
            break;
        }
    }
    FUSION_PASS_CHECK(!dtypeCheck, OPS_LOG_D(kPassName, "Matmul dtype is not supported, no fusion."), return false);

    if (x1Desc.GetDataType() == DT_FLOAT) {
        FUSION_PASS_CHECK(!GetEnableHf32FromMatmul(matmulNode),
                          OPS_LOG_D(kPassName, "FP32 FusedMatMul only supports the HF32 computation mode."),
                          return false);
    }
    return true;
}

// Bias shape check shared by both paths (canndev CheckMatmulBiasShape dispatching to
// CheckDav3510BiasShape / CheckDav5102BiasShape): DAV_3510 allows [N], [1, N] or empty,
// s8s4 only [N], then the last dim must match the matmul output n dim.
bool CheckMatmulBiasShape(const GNode& matmulNode, bool isDav3510)
{
    TensorDesc biasDesc;
    FUSION_PASS_CHECK(matmulNode.GetInputDesc(kBiasInputIdx, biasDesc) != SUCCESS,
                      OPS_LOG_D(kPassName, "Failed to get bias input desc."), return false);
    const Shape biasShape = biasDesc.GetShape();
    const size_t biasDimNum = biasShape.GetDimNum();
    if (isDav3510) {
        FUSION_PASS_CHECK(
            biasDimNum > kBiasDimNumMax3510 || (biasDimNum == kBiasDimNumMax3510 && biasShape.GetDim(0) != 1),
            OPS_LOG_D(kPassName, "DAV_3510 bias only supports [N], [1, N] or empty."), return false);
    } else {
        FUSION_PASS_CHECK(biasDimNum != kBiasDimNumOne, OPS_LOG_D(kPassName, "Bias only supports [N] or empty."),
                          return false);
    }
    return CheckBiasLastDimMatch(matmulNode, biasShape);
}

// The shared gate of both fusion loops: bias shape / dtype / format. The skippable prologue is
// the already-shared IsMatmulSkippable called by both loops; hasBias is only consumed here, the
// replace functions re-resolve it from the graph.
bool CheckMatmulCommon(const GNode& matmulNode, bool isDav3510)
{
    const bool hasBias = matmulNode.GetInDataNodesAndPortIndexs(kBiasInputIdx).first != nullptr;
    if (hasBias && !CheckMatmulBiasShape(matmulNode, isDav3510)) {
        OPS_LOG_D(kPassName, "Matmul bias shape not supported, no fusion.");
        return false;
    }
    if (!CheckMatmulDtype(matmulNode, hasBias, isDav3510)) {
        OPS_LOG_D(kPassName, "Matmul dtype not supported, no fusion.");
        return false;
    }
    if (!CheckMatmulFormat(matmulNode)) {
        OPS_LOG_D(kPassName, "Matmul format not supported, no fusion.");
        return false;
    }
    return true;
}

// ==================== platform checks (aligned with canndev CheckSocSupport) ====================

// s8s4 platform check: Intrinsic_mmad supports s8s4 (aligned with canndev CheckSocSupport).
bool IsS8s4MmadPlatform()
{
    PlatformInfo platformInfo;
    OptionalInfo optionalInfo;
    FUSION_PASS_CHECK(
        PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platformInfo, optionalInfo) != SUCCESS,
        OPS_LOG_W(kPassName, "Failed to get platform info."), return false);
    auto iter = platformInfo.ai_core_intrinsic_dtype_map.find("Intrinsic_mmad");
    if (iter == platformInfo.ai_core_intrinsic_dtype_map.end()) {
        return false;
    }
    return std::find(iter->second.begin(), iter->second.end(), "s8s4") != iter->second.end();
}

// ==================== DAV_3510 direct path (aligned with canndev) ====================

// Whether the batch dims of x1/x2 are broadcast (different dim num or different batch dims).
bool IsBatchBroadcast(const Shape& x1Shape, const Shape& x2Shape)
{
    const size_t x1DimNum = x1Shape.GetDimNum();
    const size_t x2DimNum = x2Shape.GetDimNum();
    if (x1DimNum != x2DimNum) {
        return true;
    }
    for (size_t i = 0; i + kMatmulDimNum < x1DimNum; ++i) {
        if (x1Shape.GetDim(i) != x2Shape.GetDim(i)) {
            return true;
        }
    }
    return false;
}

bool IsUnknownShape(const Shape& shape)
{
    const auto dims = shape.GetDims();
    return std::any_of(dims.begin(), dims.end(),
                       [](const int64_t dim) { return dim == UNKNOWN_DIM || dim == UNKNOWN_DIM_NUM || dim < 0; });
}

// Whether the batch and m axis can be merged for the relu fusion on DAV_3510:
// x1 must not be transposed, the total batch of x2 must be 1, and the merged m
// must not exceed INT32_MAX. Dynamic dims are checked during FusedMatMul runtime tiling.
bool CanReluMergeBatchAndMAxis(const Shape& x1Shape, const Shape& x2Shape, bool transX1)
{
    const size_t x1DimNum = x1Shape.GetDimNum();
    const size_t x2DimNum = x2Shape.GetDimNum();
    if (transX1 || x1DimNum <= kMatmulDimNum || x2DimNum < kMatmulDimNum) {
        return false;
    }

    // MergeBatchAndMAxis requires the total batch of x2 to be 1.
    for (size_t i = 0; i + kMatmulDimNum < x2DimNum; ++i) {
        if (x2Shape.GetDim(i) != 1) {
            return false;
        }
    }

    if (IsUnknownShape(x1Shape)) {
        return true;
    }

    const int64_t mDim = x1Shape.GetDim(x1DimNum - kMatmulDimNum);
    if (mDim <= 0) {
        return false;
    }
    uint64_t mergedM = static_cast<uint64_t>(mDim);
    const uint64_t maxMergedM = static_cast<uint64_t>(kMaxMergedMDim);
    for (size_t i = 0; i + kMatmulDimNum < x1DimNum; ++i) {
        const int64_t batchDim = x1Shape.GetDim(i);
        if (batchDim <= 0 || static_cast<uint64_t>(batchDim) > maxMergedM / mergedM) {
            return false;
        }
        mergedM *= static_cast<uint64_t>(batchDim);
    }
    return true;
}

// DAV_3510: BatchMatMulV3 does not support batch broadcast fusion unless the batch and
// m axis can be merged. MatMulV3 has no batch dims and is not restricted (aligned with canndev).
bool CheckMatmulShape3510(const GNode& matmulNode, bool transX1)
{
    if (!IsBatchMatMulV3Type(matmulNode)) {
        return true;
    }

    TensorDesc x1Desc;
    TensorDesc x2Desc;
    FUSION_PASS_CHECK(matmulNode.GetInputDesc(kX1InputIdx, x1Desc) != SUCCESS ||
                          matmulNode.GetInputDesc(kX2InputIdx, x2Desc) != SUCCESS,
                      OPS_LOG_W(kPassName, "Failed to get matmul input desc."), return false);

    const Shape x1Shape = x1Desc.GetShape();
    const Shape x2Shape = x2Desc.GetShape();
    if (CanReluMergeBatchAndMAxis(x1Shape, x2Shape, transX1)) {
        return true;
    }

    FUSION_PASS_CHECK(IsBatchBroadcast(x1Shape, x2Shape),
                      OPS_LOG_D(kPassName, "DAV_3510 does not support batch broadcast fusion."), return false);
    return true;
}

// The matmul node must only output to the Relu node for the direct fusion (aligned with canndev:
// outDataNodes.size() == 1 && outDataNodes[0] == reluNode).
bool GetDirectReluNode(const GNode& matmulNode, GNodePtr& reluNode)
{
    auto outDataNodes = matmulNode.GetOutDataNodesAndPortIndexs(0);
    FUSION_PASS_CHECK(outDataNodes.size() != 1,
                      OPS_LOG_D(kPassName, "Matmul must only output to Relu for direct fusion."), return false);
    AscendString nodeType;
    FUSION_PASS_CHECK(outDataNodes[0].first == nullptr || outDataNodes[0].first->GetType(nodeType) != SUCCESS ||
                          nodeType != kOpTypeRelu,
                      OPS_LOG_D(kPassName, "Matmul must only output to Relu for direct fusion."), return false);
    reluNode = outDataNodes[0].first;
    return true;
}

// [DAV_3510 direct path] Replace {matmul + Relu} with a single FusedMatMul node.
Status ReplaceDirectWithFusedMatMul(GraphPtr& graph, GNode& matmulNode, const GNodePtr& reluNode,
                                    CustomPassContext& passContext)
{
    bool transX1 = false;
    bool transX2 = false;
    GetTransposeAttrs(matmulNode, transX1, transX2);
    // enable_hf32 is only read for the FP32 input, otherwise FusedMatMul keeps false (canndev).
    TensorDesc matmulX1Desc;
    FUSION_PASS_CHECK(matmulNode.GetInputDesc(kX1InputIdx, matmulX1Desc) != SUCCESS,
                      OPS_LOG_E(kPassName, "Failed to get matmul x1 desc."), return FAILED);
    const bool enableHf32 = (matmulX1Desc.GetDataType() == DT_FLOAT) ? GetEnableHf32FromMatmul(matmulNode) : false;

    const bool hasBias = matmulNode.GetInDataNodesAndPortIndexs(kBiasInputIdx).first != nullptr;

    TensorDesc x1Desc;
    TensorDesc x2Desc;
    TensorDesc biasDesc;
    FUSION_PASS_CHECK(matmulNode.GetInputDesc(kX1InputIdx, x1Desc) != SUCCESS ||
                          matmulNode.GetInputDesc(kX2InputIdx, x2Desc) != SUCCESS,
                      OPS_LOG_E(kPassName, "Failed to get matmul input desc."), return FAILED);
    if (hasBias) {
        FUSION_PASS_CHECK(matmulNode.GetInputDesc(kBiasInputIdx, biasDesc) != SUCCESS,
                          OPS_LOG_E(kPassName, "Failed to get bias input desc."), return FAILED);
    }

    // Boundary {matmul + Relu}: the input/output indexes must be continuous from 0 and map one
    // by one to the CreateInput indexes of the replacement.
    int64_t graphIndex = 0;
    SubgraphBoundary boundary;
    if (AddMatmulBoundaryInputs(boundary, matmulNode, hasBias, graphIndex) != SUCCESS) {
        return FAILED;
    }
    {
        SubgraphOutput subgraphOutput;
        FUSION_PASS_CHECK(subgraphOutput.SetOutput({*reluNode, 0}) != GRAPH_SUCCESS ||
                              boundary.AddOutput(0, std::move(subgraphOutput)) != GRAPH_SUCCESS,
                          OPS_LOG_E(kPassName, "Add boundary output failed."), return FAILED);
    }

    int64_t replaceIndex = 0;
    auto replaceGraphBuilder = EsGraphBuilder("replacement");
    auto inputs = CreateMatmulReplacementInputs(replaceGraphBuilder, replaceIndex, x1Desc, x2Desc, biasDesc, hasBias);

    auto rY = es::FusedMatMul(inputs.x1, inputs.x2, inputs.bias, nullptr, transX1, transX2, enableHf32,
                              kFusedOpTypeRelu);

    // Inputs inherit the matmul descs, the output inherits the Relu output desc (relu keeps shape/dtype).
    GNode fusedMatMulNode = *rY.GetProducer();
    InheritMatmulInputDescs(fusedMatMulNode, matmulNode);
    TensorDesc reluOutputDesc;
    FUSION_PASS_CHECK(reluNode->GetOutputDesc(0, reluOutputDesc) != SUCCESS,
                      OPS_LOG_E(kPassName, "Failed to get relu output desc."), return FAILED);
    fusedMatMulNode.UpdateOutputDesc(0, reluOutputDesc);

    FUSION_PASS_CHECK(!CopyOtherAttrs(matmulNode, fusedMatMulNode, kPassName),
                      OPS_LOG_E(kPassName, "Copy other attrs failed."), return FAILED);

    GraphUniqPtr replaceGraph = replaceGraphBuilder.BuildAndReset({rY});
    FUSION_PASS_CHECK(ReplaceSubgraphWithReport(boundary, *replaceGraph, passContext) != SUCCESS,
                      OPS_LOG_E(kPassName, "SubgraphRewriter replace failed."), return FAILED);
    OPS_LOG_I(kPassName, "FusedMatMul fusion success! matmul=%s, path=direct(3510).", NodeName(matmulNode).c_str());
    return SUCCESS;
}

// [DAV_3510 direct path] The direct fusion loop (aligned with the canndev isDav3510 branch).
Status RunDirectPath(GraphPtr& graph, CustomPassContext& passContext)
{
    std::vector<GNode> matmulNodes = GetMatmulNodes(graph);
    FUSION_PASS_CHECK(matmulNodes.empty(), OPS_LOG_D(kPassName, "No matmul node found, exit."),
                      return GRAPH_NOT_CHANGED);

    bool changed = false;
    for (auto& matmulNode : matmulNodes) {
        // NOTE: plain if + continue here: a FUSION_PASS_CHECK with `continue` would only leave
        // the do-while block of the macro, not the loop.
        if (IsMatmulSkippable(matmulNode)) {
            continue;
        }

        GNodePtr reluNode = nullptr;
        if (!GetDirectReluNode(matmulNode, reluNode)) {
            OPS_LOG_D(kPassName, "Matmul output check failed, no fusion.");
            continue;
        }

        bool transX1 = false;
        bool transX2 = false;
        GetTransposeAttrs(matmulNode, transX1, transX2);

        if (!CheckMatmulShape3510(matmulNode, transX1)) {
            OPS_LOG_D(kPassName, "Matmul shape not supported, no fusion.");
            continue;
        }

        if (!CheckMatmulCommon(matmulNode, true)) {
            continue;
        }

        Status replaceRes = ReplaceDirectWithFusedMatMul(graph, matmulNode, reluNode, passContext);
        if (replaceRes == FAILED) {
            OPS_LOG_E(kPassName, "Direct fusion failed.");
            return FAILED;
        }
        if (replaceRes == SUCCESS) {
            changed = true;
        }
    }
    return changed ? SUCCESS : GRAPH_NOT_CHANGED;
}

// ==================== s8s4 PostCube path (aligned with canndev) ====================

// The forbidden-node gate (dynamic shape / ND-class output / ND input / fp32 NZ c0=16 /
// all-int32) is NOT inlined: it is carried by the "support_fixpipe_ability" NodeCantAccess bit
// set by the canndev FixPipeAbilityProcessPass (FORBIDDEN_CLOSE, always runs) and consumed by
// the PostCubeUtils::GetPostCubeNodeList entry, the same collaboration the canndev pass relies on.

// Under force_fp16 on DAV_5102, the whole-FP32 chain is temporarily reduced to FP16 so the
// dtype check can pass (the FE inserts the conversion nodes later); the descs are restored on
// destruction when the fusion is rejected (RAII), aligned with canndev Fp16FusionPreparation.
class Fp16FusionPreparation {
public:
    Fp16FusionPreparation(const GNodePtr& matmulNode, const GNodePtr& reluNode, const GNodePtr& quantNode,
                          const CustomPassContext& passContext)
        : matmulNode_(matmulNode), reluNode_(reluNode), quantNode_(quantNode), passContext_(passContext)
    {}

    ~Fp16FusionPreparation()
    {
        if (committed_ || descBackups_.empty()) {
            return;
        }
        for (const auto& backup : descBackups_) {
            if (backup.isInput) {
                backup.node->UpdateInputDesc(backup.index, backup.desc);
            } else {
                backup.node->UpdateOutputDesc(backup.index, backup.desc);
            }
        }
    }

    void Prepare()
    {
        // Only a consistently FP32 chain under force_fp16 needs the preparation.
        if (GetPrecisionModeStr(passContext_) != kFp16FusionPrecisionMode || !CanReducePrecision(matmulNode_) ||
            !CanReducePrecision(reluNode_) || !CanReducePrecision(quantNode_)) {
            return;
        }
        if (GetCustomizeDtypeBySelector(passContext_, kOpTypeFusedMatMul) || !IsSingleConsumerChain() ||
            !CollectChainFloatDescs()) {
            return;
        }
        ApplyFp16ToBackups();
        OPS_LOG_I(kPassName, "Prepare FP16 fusion for the node under force_fp16.");
    }

    // Keep the FP16 dtype on the fused node's inputs (and the output when there is no quant
    // node): the boundary replacement may inherit the original FP32 peer descs.
    Status UpdateFusedNodeDtype(GNode& fusedNode) const
    {
        if (!IsPrepared()) {
            return SUCCESS;
        }
        for (const auto& backup : descBackups_) {
            if (backup.nodeName != NodeName(*matmulNode_) || !backup.isInput) {
                continue;
            }
            TensorDesc inputDesc;
            if (fusedNode.GetInputDesc(backup.index, inputDesc) != SUCCESS) {
                return FAILED;
            }
            inputDesc.SetDataType(DT_FLOAT16);
            fusedNode.UpdateInputDesc(backup.index, inputDesc);
        }
        if (quantNode_ == nullptr) {
            TensorDesc outputDesc;
            if (fusedNode.GetOutputDesc(0, outputDesc) != SUCCESS) {
                return FAILED;
            }
            outputDesc.SetDataType(DT_FLOAT16);
            fusedNode.UpdateOutputDesc(0, outputDesc);
        }
        return SUCCESS;
    }

    bool IsPrepared() const { return !descBackups_.empty(); }
    void Commit() { committed_ = true; }

private:
    struct DescBackup {
        GNodePtr node;
        std::string nodeName;
        int64_t index;
        bool isInput;
        TensorDesc desc;
    };

    // The customize dtype overrides and the _keep_dtype attr forbid the precision reduction.
    bool CanReducePrecision(const GNodePtr& node) const
    {
        if (node == nullptr) {
            return true;
        }
        int64_t keepDtype = 0;
        (void)node->GetAttr("_keep_dtype", keepDtype);
        return keepDtype == 0 && !GetCustomizeDtypeBySelector(passContext_, NodeName(*node)) &&
               !GetCustomizeDtypeBySelector(passContext_, GetNodeType(*node));
    }

    // The chain must be the exact single-consumer topology the fusion consumes. The input count
    // accepts the x1/x2-only and x1/x2/bias forms (canndev: MATMUL_INPUT_MAX - 1 <= n <= MATMUL_INPUT_MAX).
    bool IsSingleConsumerChain() const
    {
        const size_t inputNum = matmulNode_->GetInputsSize();
        const GNodePtr nextNode = (reluNode_ != nullptr) ? reluNode_ : quantNode_;
        auto consumers = matmulNode_->GetOutDataNodesAndPortIndexs(0);
        if (inputNum < kBaseNodeNum || inputNum > kBaseNodeNum + 1 || consumers.size() != 1 ||
            !IsSameNode(consumers[0].first, nextNode)) {
            return false;
        }
        if (reluNode_ != nullptr && quantNode_ != nullptr) {
            auto reluConsumers = reluNode_->GetOutDataNodesAndPortIndexs(0);
            if (reluConsumers.size() != 1 || !IsSameNode(reluConsumers[0].first, quantNode_)) {
                return false;
            }
        }
        return true;
    }

    // Save the descs of the whole-FP32 chain; any non-FP32 member cancels the collection.
    bool CollectChainFloatDescs()
    {
        const size_t inputNum = matmulNode_->GetInputsSize();
        for (size_t i = 0; i < inputNum; i++) {
            const auto anchor = matmulNode_->GetInDataNodesAndPortIndexs(static_cast<int64_t>(i)).first;
            if (static_cast<int64_t>(i) == kBiasInputIdx && anchor == nullptr) {
                continue;
            }
            if (anchor == nullptr || !SaveFloatDesc(matmulNode_, static_cast<int64_t>(i), true)) {
                descBackups_.clear();
                return false;
            }
        }
        if (!SaveFloatDesc(matmulNode_, 0, false) ||
            (reluNode_ != nullptr && (!SaveFloatDesc(reluNode_, 0, true) || !SaveFloatDesc(reluNode_, 0, false))) ||
            (quantNode_ != nullptr && !SaveFloatDesc(quantNode_, 0, true))) {
            descBackups_.clear();
            return false;
        }
        return true;
    }

    void ApplyFp16ToBackups()
    {
        for (const auto& backup : descBackups_) {
            TensorDesc desc = backup.desc;
            desc.SetDataType(DT_FLOAT16);
            if (backup.isInput) {
                backup.node->UpdateInputDesc(backup.index, desc);
            } else {
                backup.node->UpdateOutputDesc(backup.index, desc);
            }
        }
    }

    bool SaveFloatDesc(const GNodePtr& node, int64_t index, bool isInput)
    {
        if (node == nullptr) {
            return false;
        }
        TensorDesc desc;
        const Status ret = isInput ? node->GetInputDesc(index, desc) : node->GetOutputDesc(index, desc);
        if (ret != SUCCESS || desc.GetDataType() != DT_FLOAT) {
            return false;
        }
        descBackups_.push_back({node, NodeName(*node), index, isInput, desc});
        return true;
    }

    static std::string GetNodeType(const GNode& node)
    {
        AscendString type;
        if (node.GetType(type) != SUCCESS) {
            return "";
        }
        return std::string(type.GetString());
    }

    GNodePtr matmulNode_;
    GNodePtr reluNode_;
    GNodePtr quantNode_;
    const CustomPassContext& passContext_;
    std::vector<DescBackup> descBackups_;
    bool committed_ = false;
};

// The Relu->AscendQuant swap never applies to a matmul head: the canndev FixPipeReluQuantPass
// white list is Conv2D/Conv2DTranspose only, so the graph-order [Relu, AscendQuant] chain is the
// production form (the swapped [AscendQuant, Relu] order is also accepted when some producer
// already emits it). The postcube units require the quant (pre_conv) before the activation
// (pre_act), so the graph-order chain folds the Relu only and ReplacePostCubeWithFusedMatMul
// absorbs the trailing AscendQuant for the full relu_quant fusion.

// Get the in-graph GNodePtr of the matmul node: the PostCube utils works on GNodePtr, resolve it
// through the first output node and back to its input source (same way as the conv pass does).
GNodePtr GetMatmulNodePtr(const GNode& matmulNode)
{
    auto outDataNodes = matmulNode.GetOutDataNodesAndPortIndexs(0);
    FUSION_PASS_CHECK(outDataNodes.empty(), OPS_LOG_D(kPassName, "Matmul out nodes is empty."), return nullptr);
    FUSION_PASS_CHECK(outDataNodes[0].first == nullptr, OPS_LOG_D(kPassName, "Matmul out node is null."),
                      return nullptr);
    GNodePtr nodePtr = outDataNodes[0].first->GetInDataNodesAndPortIndexs(outDataNodes[0].second).first;
    FUSION_PASS_CHECK(nodePtr == nullptr, OPS_LOG_D(kPassName, "Failed to resolve matmul node ptr."), return nullptr);
    AscendString name;
    AscendString selfName;
    FUSION_PASS_CHECK(
        nodePtr->GetName(name) != GRAPH_SUCCESS || matmulNode.GetName(selfName) != GRAPH_SUCCESS || name != selfName,
        OPS_LOG_D(kPassName, "Resolved node ptr does not match the matmul node."), return nullptr);
    return nodePtr;
}

// Keep only the post-cube chains supported by FusedMatMul: [Relu], [AscendQuant],
// [Relu, AscendQuant] (graph order, canndev semantics) or [AscendQuant, Relu] (after the swap).
void SelectPostCubePassByWhiteList(std::vector<PostCubePassInfo>& matchLists)
{
    std::vector<PostCubePassInfo> tmpLists(matchLists);
    matchLists.clear();
    for (auto& postCubePass : tmpLists) {
        // m_opnodes[0] is the head (matmul) node itself, the post-cube op chain starts at index 1
        // (same as canndev FIRST_FUSED_NODE_INDEX).
        if (postCubePass.m_opnodes.empty()) {
            continue;
        }
        std::vector<AscendString> nodeTypes;
        bool isSupported = true;
        for (size_t index = 1; index < postCubePass.m_opnodes.size(); index++) {
            auto curNode = postCubePass.m_opnodes[index].GetNode();
            if (curNode == nullptr) {
                isSupported = false;
                break;
            }
            AscendString nodeType;
            if (curNode->GetType(nodeType) != GRAPH_SUCCESS) {
                isSupported = false;
                break;
            }
            nodeTypes.emplace_back(nodeType);
        }

        const bool isReluChain = nodeTypes.size() == 1 && nodeTypes[0] == kOpTypeRelu;
        const bool isQuantChain = nodeTypes.size() == 1 && nodeTypes[0] == kOpTypeAscendQuant;
        const bool isReluQuantChain = nodeTypes.size() == kReluQuantChainOpNum &&
                                      ((nodeTypes[0] == kOpTypeRelu && nodeTypes[1] == kOpTypeAscendQuant) ||
                                       (nodeTypes[0] == kOpTypeAscendQuant && nodeTypes[1] == kOpTypeRelu));
        if (isSupported && (isReluChain || isQuantChain || isReluQuantChain)) {
            matchLists.emplace_back(postCubePass);
        } else {
            OPS_LOG_D(kPassName, "PostCube chain is not supported by FusedMatMul, removed.");
        }
    }

    // Keep the longest chain only: the canndev pass filters by the expected fused op type
    // (a relu_quant graph expects the full [Relu, AscendQuant] chain), so an incomplete shorter
    // chain must not win the selection.
    if (matchLists.size() > 1) {
        size_t maxLen = 0;
        for (const auto& match : matchLists) {
            maxLen = std::max(maxLen, match.m_opnodes.size());
        }
        std::vector<PostCubePassInfo> longest;
        for (auto& match : matchLists) {
            if (match.m_opnodes.size() == maxLen) {
                longest.emplace_back(match);
            }
        }
        matchLists = std::move(longest);
    }
}

// Map the recognized chain to the fused_op_type of FusedMatMul (aligned with canndev
// GetFusedOpTypeFromFixpipePass, plus the swapped [AscendQuant, Relu] order).
std::string GetFusedOpTypeFromChain(const std::vector<AscendString>& fusionOpList)
{
    if (fusionOpList.size() == 1) {
        if (fusionOpList[0] == kOpTypeRelu) {
            return kFusedOpTypeRelu;
        }
        if (fusionOpList[0] == kOpTypeAscendQuant) {
            return kFusedOpTypeQuant;
        }
    }
    if (fusionOpList.size() == kReluQuantChainOpNum &&
        ((fusionOpList[0] == kOpTypeRelu && fusionOpList[1] == kOpTypeAscendQuant) ||
         (fusionOpList[0] == kOpTypeAscendQuant && fusionOpList[1] == kOpTypeRelu))) {
        return kFusedOpTypeReluQuant;
    }
    return "";
}

// Collect the FixPipe node created by the PostCube utils and the intermediate nodes left on the
// path (the swapped AscendQuant kept alive by Strategy21 to provide the scale/offset, and the
// folded Relu); they are removed by the caller before the replace (canndev oldNodes semantics).
bool GetPostCubeNodes(GNode& matmulNode, GNodePtr& postCubeNode, std::vector<GNodePtr>& intermediateNodes)
{
    intermediateNodes.clear();
    auto outDataNodes = matmulNode.GetOutDataNodesAndPortIndexs(0);
    FUSION_PASS_CHECK(outDataNodes.empty(), OPS_LOG_D(kPassName, "Matmul out nodes is empty."), return false);
    FUSION_PASS_CHECK(outDataNodes.size() != 1, OPS_LOG_D(kPassName, "Matmul must have exactly one chain output."),
                      return false);

    // Walk the single chain from the matmul output to the FixPipe node.
    GNodePtr chainNode = outDataNodes[0].first;
    AscendString nodeType;
    while (chainNode != nullptr && chainNode->GetType(nodeType) == GRAPH_SUCCESS && nodeType != kOpTypeFixPipe) {
        intermediateNodes.emplace_back(chainNode);
        auto nextNodes = chainNode->GetOutDataNodesAndPortIndexs(0);
        FUSION_PASS_CHECK(nextNodes.size() != 1 || nextNodes[0].first == nullptr,
                          OPS_LOG_D(kPassName, "Intermediate node must have exactly one output."), return false);
        chainNode = nextNodes[0].first;
    }

    FUSION_PASS_CHECK(
        chainNode == nullptr || chainNode->GetType(nodeType) != GRAPH_SUCCESS || nodeType != kOpTypeFixPipe,
        OPS_LOG_D(kPassName, "Matmul output is not covered by the PostCube chain, no fusion."), return false);
    postCubeNode = chainNode;
    return true;
}

// Remove the intermediate nodes between the matmul and the FixPipe node and link the matmul
// output directly to the FixPipe input.
bool RemoveIntermediateNodes(GraphPtr& graph, GNode& matmulNode, const std::vector<GNodePtr>& intermediateNodes,
                             const GNodePtr& postCubeNode)
{
    if (intermediateNodes.empty()) {
        return true;
    }
    if (graph->RemoveEdge(matmulNode, 0, *intermediateNodes.front(), 0) != GRAPH_SUCCESS) {
        OPS_LOG_E(kPassName, "Remove the matmul to intermediate edge failed.");
        return false;
    }
    for (size_t index = 0; index < intermediateNodes.size(); index++) {
        const GNodePtr& curNode = intermediateNodes[index];
        if (index + 1 < intermediateNodes.size()) {
            if (graph->RemoveEdge(*curNode, 0, *intermediateNodes[index + 1], 0) != GRAPH_SUCCESS) {
                OPS_LOG_E(kPassName, "Remove the intermediate chain edge failed.");
                return false;
            }
        } else if (graph->RemoveEdge(*curNode, 0, *postCubeNode, 0) != GRAPH_SUCCESS) {
            OPS_LOG_E(kPassName, "Remove the intermediate to FixPipe edge failed.");
            return false;
        }
        if (graph->RemoveNode(*curNode) != GRAPH_SUCCESS) {
            OPS_LOG_E(kPassName, "Remove the intermediate node failed.");
            return false;
        }
    }
    if (graph->AddDataEdge(matmulNode, 0, *postCubeNode, 0) != GRAPH_SUCCESS) {
        OPS_LOG_E(kPassName, "Link the matmul output to the FixPipe node failed.");
        return false;
    }
    return true;
}

// Remove the folded Relu nodes the postcube utils may leave as orphans after the absorption
// (no data input and no data output), aligned with the canndev oldNodes semantics.
Status RemoveOrphanReluNodes(const GraphPtr& graph)
{
    std::vector<GNode> orphanNodes;
    for (auto& node : graph->GetDirectNode()) {
        AscendString nodeType;
        if (node.GetType(nodeType) != GRAPH_SUCCESS || nodeType != kOpTypeRelu) {
            continue;
        }
        if (node.GetInDataNodesAndPortIndexs(0).first == nullptr && node.GetOutDataNodesAndPortIndexs(0).empty()) {
            orphanNodes.emplace_back(node);
        }
    }
    for (auto& orphan : orphanNodes) {
        FUSION_PASS_CHECK(graph->RemoveNode(orphan) != GRAPH_SUCCESS,
                          OPS_LOG_E(kPassName, "Remove the orphan Relu node failed."), return FAILED);
    }
    return SUCCESS;
}

// [s8s4 PostCube path] Fold the post-cube chain into a FixPipe node (aligned with the canndev
// FixpipeFusionImpl four steps).
Status PostCubeFusionImpl(GraphPtr& graph, const GNode& matmulNode, CustomPassContext& passContext)
{
    GNodePtr matmulNodePtr = GetMatmulNodePtr(matmulNode);
    FUSION_PASS_CHECK(matmulNodePtr == nullptr, OPS_LOG_D(kPassName, "Get matmul node ptr failed, no fusion."),
                      return GRAPH_NOT_CHANGED);

    PostCubeUtils postCubeUtils;
    const graphStatus getListDbg = postCubeUtils.GetPostCubeNodeList(matmulNodePtr, passContext);
    FUSION_PASS_CHECK(getListDbg != GRAPH_SUCCESS, OPS_LOG_D(kPassName, "GetPostCubeNodeList failed, no fusion."),
                      return GRAPH_NOT_CHANGED);

    SelectPostCubePassByWhiteList(postCubeUtils.m_matchpasses_);
    FUSION_PASS_CHECK(postCubeUtils.m_matchpasses_.empty(),
                      OPS_LOG_D(kPassName, "No PostCube chain matches the FusedMatMul white list."),
                      return GRAPH_NOT_CHANGED);

    const graphStatus selectDbg = postCubeUtils.SelectPostCubeNodeList(false);
    FUSION_PASS_CHECK(selectDbg != GRAPH_SUCCESS, OPS_LOG_D(kPassName, "SelectPostCubeNodeList failed, no fusion."),
                      return GRAPH_NOT_CHANGED);

    std::vector<GNodePtr> newNodes;
    const graphStatus createDbg = postCubeUtils.CreatePostCubeNode(kPassName, *graph, newNodes);
    FUSION_PASS_CHECK(createDbg != GRAPH_SUCCESS, OPS_LOG_D(kPassName, "CreatePostCubeNode failed, no fusion."),
                      return GRAPH_NOT_CHANGED);
    return SUCCESS;
}

// [s8s4 PostCube path] Replace {matmul + FixPipe} with a single FusedMatMul node.
Status ReplacePostCubeWithFusedMatMul(GraphPtr& graph, GNode& matmulNode, const Fp16FusionPreparation& fp16Preparation,
                                      CustomPassContext& passContext)
{
    // NOTE: the PostCube node has already been created in the graph at this point, any failure
    // below must abort the pass with FAILED (a half-done FixPipe node must not be left behind
    // silently), aligned with the canndev error semantics.
    GNodePtr postCubeNode;
    std::vector<GNodePtr> intermediateNodes;
    FUSION_PASS_CHECK(!GetPostCubeNodes(matmulNode, postCubeNode, intermediateNodes),
                      OPS_LOG_E(kPassName, "Get PostCube nodes failed."), return FAILED);
    FUSION_PASS_CHECK(!RemoveIntermediateNodes(graph, matmulNode, intermediateNodes, postCubeNode),
                      OPS_LOG_E(kPassName, "Remove the intermediate nodes failed."), return FAILED);
    if (RemoveOrphanReluNodes(graph) != SUCCESS) {
        return FAILED;
    }

    // The trailing AscendQuant of the graph-order [Relu, AscendQuant] chain: the postcube units
    // require the quant (pre_conv) before the activation (pre_act) and the unit walk is
    // forward-only, so the chain folds the Relu only and the FixPipe output feeds the original
    // AscendQuant. canndev never swaps a matmul chain (the FixPipeReluQuantPass white list is
    // Conv2D/Conv2DTranspose only); it derives relu_quant from the matched pattern instead and
    // absorbs this trailing quant in TransferOpToFusedMatMul (the FusedMatMul output takes over
    // the quant output and the x3 scalar is built from the quant scale/offset attrs with
    // FMM_QUANT_INPUT_STRATEGY=21). The single-consumer gate matches the canndev pattern
    // requirement that the intermediate Relu is exclusively consumed by the quant.
    GNodePtr trailingQuantNode = nullptr;
    auto fixPipeConsumers = postCubeNode->GetOutDataNodesAndPortIndexs(0);
    if (fixPipeConsumers.size() == 1 && fixPipeConsumers[0].first != nullptr) {
        AscendString consumerType;
        if (fixPipeConsumers[0].first->GetType(consumerType) == GRAPH_SUCCESS && consumerType == kOpTypeAscendQuant) {
            trailingQuantNode = fixPipeConsumers[0].first;
        }
    }

    std::vector<AscendString> fusionOpList;
    FUSION_PASS_CHECK(postCubeNode->GetAttr(kAttrFusionOpList, fusionOpList) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "Get fusion_op_list from FixPipe failed."), return FAILED);
    std::string fusedOpType = GetFusedOpTypeFromChain(fusionOpList);
    // The partial [Relu] fold of a graph-order [Relu, AscendQuant] chain fuses as relu_quant
    // (aligned with the canndev fused_op_type derived from the matched pattern, not from the
    // folded op list).
    if (trailingQuantNode != nullptr && fusedOpType == kFusedOpTypeRelu) {
        fusedOpType = kFusedOpTypeReluQuant;
    }
    FUSION_PASS_CHECK(fusedOpType.empty(), OPS_LOG_E(kPassName, "Fusion op list is not supported by FusedMatMul."),
                      return FAILED);

    const bool hasBias = matmulNode.GetInDataNodesAndPortIndexs(kBiasInputIdx).first != nullptr;
    const bool hasX3 = (fusedOpType == kFusedOpTypeQuant) || (fusedOpType == kFusedOpTypeReluQuant);

    bool transX1 = false;
    bool transX2 = false;
    GetTransposeAttrs(matmulNode, transX1, transX2);

    TensorDesc x1Desc;
    TensorDesc x2Desc;
    TensorDesc biasDesc;
    TensorDesc x3Desc;
    FUSION_PASS_CHECK(matmulNode.GetInputDesc(kX1InputIdx, x1Desc) != SUCCESS ||
                          matmulNode.GetInputDesc(kX2InputIdx, x2Desc) != SUCCESS,
                      OPS_LOG_E(kPassName, "Failed to get matmul input desc."), return FAILED);
    if (hasBias) {
        FUSION_PASS_CHECK(matmulNode.GetInputDesc(kBiasInputIdx, biasDesc) != SUCCESS,
                          OPS_LOG_E(kPassName, "Failed to get bias input desc."), return FAILED);
    }
    if (trailingQuantNode != nullptr) {
        // The trailing quant was not folded by the postcube utils: build the x3 scalar provider
        // (offset const + scale const + SetQuantOffsetScale) and wire it into the FixPipe
        // quant_scale_0 input, the same scalar the utils' Strategy21 creates for a folded
        // quant, so the boundary transfer below maps it onto FusedMatMul.x3.
        auto functionParam = std::make_shared<PostCubeFunctionParam>("quant_scale_0", postCubeNode,
                                                                     static_cast<uint32_t>(kPostCubeQuantScale0Idx));
        PostCubeAddInputBase addInput;
        FUSION_PASS_CHECK(addInput.CreateScalarInputNode(*graph, functionParam, trailingQuantNode,
                                                         kFmmQuantInputStrategy) != GRAPH_SUCCESS,
                          OPS_LOG_E(kPassName, "Create the x3 scalar input from the trailing quant failed."),
                          return FAILED);
    }
    if (hasX3) {
        FUSION_PASS_CHECK(postCubeNode->GetInputDesc(kPostCubeQuantScale0Idx, x3Desc) != SUCCESS,
                          OPS_LOG_E(kPassName, "Failed to get FixPipe quant_scale_0 desc."), return FAILED);
    }

    // Boundary inputs: matmul x1/x2/bias and FixPipe quant_scale_0 (-> FusedMatMul.x3), the
    // output is the FixPipe output 0, or the trailing quant output 0 when the [Relu, AscendQuant]
    // chain absorbs the quant (the quant node is then part of the replaced region, the same
    // canndev oldNodes semantics).
    int64_t graphIndex = 0;
    SubgraphBoundary boundary;
    if (AddMatmulBoundaryInputs(boundary, matmulNode, hasBias, graphIndex) != SUCCESS) {
        return FAILED;
    }
    if (hasX3) {
        SubgraphInput subgraphInput;
        FUSION_PASS_CHECK(subgraphInput.AddInput({*postCubeNode, kPostCubeQuantScale0Idx}) != GRAPH_SUCCESS ||
                              boundary.AddInput(graphIndex++, std::move(subgraphInput)) != GRAPH_SUCCESS,
                          OPS_LOG_E(kPassName, "Add x3 boundary input failed."), return FAILED);
    }
    {
        SubgraphOutput subgraphOutput;
        const GNode& boundaryOutNode = (trailingQuantNode != nullptr) ? *trailingQuantNode : *postCubeNode;
        FUSION_PASS_CHECK(subgraphOutput.SetOutput({boundaryOutNode, 0}) != GRAPH_SUCCESS ||
                              boundary.AddOutput(0, std::move(subgraphOutput)) != GRAPH_SUCCESS,
                          OPS_LOG_E(kPassName, "Add boundary output failed."), return FAILED);
    }

    int64_t replaceIndex = 0;
    auto replaceGraphBuilder = EsGraphBuilder("replacement");
    auto inputs = CreateMatmulReplacementInputs(replaceGraphBuilder, replaceIndex, x1Desc, x2Desc, biasDesc, hasBias);
    EsTensorHolder rX3 = nullptr;
    if (hasX3) {
        rX3 = replaceGraphBuilder.CreateInput(replaceIndex++, "x3", x3Desc.GetDataType(), x3Desc.GetFormat(),
                                              x3Desc.GetShape().GetDims());
    }

    // s8s4 only supports fp16, enable_hf32 keeps the default value (false).
    auto rY = es::FusedMatMul(inputs.x1, inputs.x2, inputs.bias, rX3, transX1, transX2, false, fusedOpType.c_str());

    // Inputs inherit the matmul/fixpipe descs, the output inherits the FixPipe output desc, or
    // the trailing quant output desc (int8) when the quant is absorbed.
    GNode fusedMatMulNode = *rY.GetProducer();
    InheritMatmulInputDescs(fusedMatMulNode, matmulNode);
    if (hasX3) {
        fusedMatMulNode.UpdateInputDesc(kFmmX3InputIdx, x3Desc);
    }
    const GNodePtr& fusedOutputSrcNode = (trailingQuantNode != nullptr) ? trailingQuantNode : postCubeNode;
    TensorDesc fusedOutputDesc;
    FUSION_PASS_CHECK(fusedOutputSrcNode->GetOutputDesc(0, fusedOutputDesc) != SUCCESS,
                      OPS_LOG_E(kPassName, "Failed to get fusion output desc."), return FAILED);
    fusedMatMulNode.UpdateOutputDesc(0, fusedOutputDesc);

    FUSION_PASS_CHECK(!CopyOtherAttrs(matmulNode, fusedMatMulNode, kPassName),
                      OPS_LOG_E(kPassName, "Copy other attrs failed."), return FAILED);

    // Under force_fp16 keep the FP16 dtype on the fused node (see UpdateFusedNodeDtype).
    FUSION_PASS_CHECK(fp16Preparation.UpdateFusedNodeDtype(fusedMatMulNode) != SUCCESS,
                      OPS_LOG_E(kPassName, "Failed to set FP16 fusion target dtype."), return FAILED);

    GraphUniqPtr replaceGraph = replaceGraphBuilder.BuildAndReset({rY});
    FUSION_PASS_CHECK(ReplaceSubgraphWithReport(boundary, *replaceGraph, passContext) != SUCCESS,
                      OPS_LOG_E(kPassName, "SubgraphRewriter replace failed."), return FAILED);
    OPS_LOG_I(kPassName, "FusedMatMul fusion success! matmul=%s, fused_op_type=%s, path=postcube(s8s4)%s.",
              NodeName(matmulNode).c_str(), fusedOpType.c_str(),
              (trailingQuantNode != nullptr) ? ", trailing quant absorbed" : "");
    return SUCCESS;
}

// Resolve the immediate post-op nodes of the matmul node (Relu and/or AscendQuant) for the
// FP16 preparation. The caller has checked the matmul node is not skippable.
void ResolvePostOpNodes(const GNode& matmulNode, GNodePtr& reluNode, GNodePtr& quantNode)
{
    reluNode = nullptr;
    quantNode = nullptr;
    auto outs = matmulNode.GetOutDataNodesAndPortIndexs(0);
    if (outs.size() != 1 || outs[0].first == nullptr) {
        return;
    }
    AscendString firstType;
    if (outs[0].first->GetType(firstType) != SUCCESS) {
        return;
    }
    if (firstType == kOpTypeRelu) {
        reluNode = outs[0].first;
        auto reluOuts = reluNode->GetOutDataNodesAndPortIndexs(0);
        if (reluOuts.size() == 1 && reluOuts[0].first != nullptr) {
            AscendString secondType;
            if (reluOuts[0].first->GetType(secondType) == SUCCESS && secondType == kOpTypeAscendQuant) {
                quantNode = reluOuts[0].first;
            }
        }
    } else if (firstType == kOpTypeAscendQuant) {
        quantNode = outs[0].first;
    }
}

// [s8s4 PostCube path] The postcube fusion loop (aligned with the canndev s8s4 branch).
Status RunPostCubePath(GraphPtr& graph, CustomPassContext& passContext)
{
    std::vector<GNode> matmulNodes = GetMatmulNodes(graph);
    FUSION_PASS_CHECK(matmulNodes.empty(), OPS_LOG_D(kPassName, "No matmul node found, exit."),
                      return GRAPH_NOT_CHANGED);

    bool changed = false;
    for (auto& matmulNode : matmulNodes) {
        if (IsMatmulSkippable(matmulNode)) {
            continue;
        }

        GNodePtr prepareReluNode = nullptr;
        GNodePtr prepareQuantNode = nullptr;
        ResolvePostOpNodes(matmulNode, prepareReluNode, prepareQuantNode);

        // Prepare the FP32 chain for the force_fp16 fusion BEFORE the checks (canndev order:
        // Prepare -> ScenarioCheck): the dtype check below must see the temporarily lowered
        // FP16 descs, restored by the RAII destructor on rejection.
        const GNodePtr matmulNodePtr = GetMatmulNodePtr(matmulNode);
        Fp16FusionPreparation fp16Preparation(
            (matmulNodePtr != nullptr) ? matmulNodePtr : std::make_shared<GNode>(matmulNode), prepareReluNode,
            prepareQuantNode, passContext);
        fp16Preparation.Prepare();

        if (!CheckMatmulCommon(matmulNode, false)) {
            continue;
        }

        // The graph-order [Relu, AscendQuant] chain folds the Relu only (see the note above);
        // ReplacePostCubeWithFusedMatMul absorbs the trailing AscendQuant into the
        // FusedMatMul(relu_quant) replace, the same way the canndev pass derives relu_quant from
        // the matched pattern and absorbs the quant node in TransferOpToFusedMatMul. The
        // swapped [AscendQuant, Relu] order (when a producer already emits it) folds fully.

        // Fold the post-cube chain into a FixPipe node.
        Status preRes = PostCubeFusionImpl(graph, matmulNode, passContext);
        if (preRes == FAILED) {
            // The graph may have been changed, abort without restoring the stale descs.
            fp16Preparation.Commit();
            OPS_LOG_E(kPassName, "PostCubeFusionImpl failed.");
            return FAILED;
        }
        if (preRes == GRAPH_NOT_CHANGED) {
            continue;
        }
        // The FixPipe node has consumed the original post-op nodes: further errors must abort.
        fp16Preparation.Commit();

        // Replace {matmul + FixPipe} with FusedMatMul.
        Status replaceRes = ReplacePostCubeWithFusedMatMul(graph, matmulNode, fp16Preparation, passContext);
        if (replaceRes == FAILED) {
            OPS_LOG_E(kPassName, "ReplacePostCubeWithFusedMatMul failed.");
            return FAILED;
        }
        if (replaceRes == SUCCESS) {
            changed = true;
        }
    }
    return changed ? SUCCESS : GRAPH_NOT_CHANGED;
}

} // namespace

// Single pass entry with the platform dispatch inside (canndev Fusion: isDav3510 -> direct
// path, s8s4 mmad -> fixpipe path).
Status FusedMatMulFusionPass::Run(GraphPtr& graph, CustomPassContext& passContext)
{
    OPS_LOG_D(kPassName, "Enter FusedMatMulFusionPass.");
    passContext.SetPassName(kPassName);

    // DAV_3510: matmul + relu is fused into FusedMatMul directly. canndev consults the runtime
    // soc spec (rtGetSocSpec, unavailable to a plugin pass) and soc_info.arch_type is unreliable
    // (the Ascend950 ini has no arch_type at all, which blocked the real 950 device); the common
    // IsNpuArch3510Series matches short_soc_version against the 3510-series soc list instead.
    const bool is3510Dbg = IsNpuArch3510Series();
    const bool isS8s4Dbg = IsS8s4MmadPlatform();
    OPS_LOG_D(kPassName, "Platform dispatch: isDav3510=%d, isS8s4Mmad=%d.", is3510Dbg ? 1 : 0, isS8s4Dbg ? 1 : 0);
    if (is3510Dbg) {
        return RunDirectPath(graph, passContext);
    }

    // s8s4 mmad: the post op chain is folded via the PostCube utils.
    if (isS8s4Dbg) {
        return RunPostCubePath(graph, passContext);
    }

    OPS_LOG_D(kPassName, "Platform is neither DAV_3510 nor s8s4 mmad, skip fusion.");
    return GRAPH_NOT_CHANGED;
}

#if GE_COMPILER_VERSION_NUM >= GE_COMPILER_TARGET_VERSION_NUM
REG_FUSION_PASS(FusedMatMulFusionPass)
    .Stage(IsTargetVersion() ? CustomPassStage::kCompatibleInherited : CustomPassStage::kAfterInferShape);
#endif

} // namespace ops

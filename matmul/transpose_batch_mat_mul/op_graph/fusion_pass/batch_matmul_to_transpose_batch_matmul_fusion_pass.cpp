/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file batch_matmul_to_transpose_batch_matmul_fusion_pass.cpp
 * \brief batch_matmul to transpose_batch_matmul fusion pass
 *        (batchmatmul (milan static whitelist) --> transpose_batch_matmul)
 *
 * pattern1:
 *         x                              weight
 *           \                            /
 *     transpose_1(optional)    transpose_2(optional)        x            weight
 *             \                  /                             \            /
 *              batchmatmul    ->                           transposebatchmatmul
 *                     |
 *                transpose_2
 *
 * pattern2:
 *         x1(B2,B1,1,K)         x2(1,B1,K,N)/(1,B1,N,K)        x1(B2,B1,1,K)             x2(1,B1,K,N)/(1,B1,N,K)
 *                    \         /                                      |                             |
 *                    batchmatmul                                    reshape1                       reshape2
 *                         |                                            |                             |
 *                   y(B2,B1,1,N)                                  x1(B2,B1,K)               x2(B1,K,N)/(B1,N,K)
 *                                                        -->               \                 /
 *                                                                          transposebatchmatmul
 *                                                                                   |
 *                                                                              y(B2,B1,N)
 *                                                                                   |
 *                                                                                reshape3
 *                                                                                   |
 *                                                                              y(B2,B1,1,N)
 */

#include "batch_matmul_to_transpose_batch_matmul_fusion_pass.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include "ge/compliant_node_builder.h"
#include "ge/es_graph_builder.h"
#include "ge/fusion/pass/pattern_fusion_pass.h"
#include "graph/graph.h"
#include "graph/utils/type_utils.h"
#include "log/log.h"
#include "platform/platform_info.h"
#include "acl/acl_rt.h"
#include "version/ge-compiler_version.h"
#include "common/inc/error_util.h"
#include "common/op_graph/fusion_pass/matmul_fusion_utils_pass.h"
#include "common/op_host/math_util.h"

using namespace ge;
using namespace ge::fusion;

namespace ops {
namespace {

constexpr char kPassName[] = "BatchMatMul2TransposeBatchMatMulFusionPass";
constexpr char kOpTypeBatchMatMul[] = "BatchMatMul";
constexpr char kOpTypeBatchMatMulV2[] = "BatchMatMulV2";
constexpr char kOpTypeTranspose[] = "Transpose";
constexpr char kOpTypeTransposeD[] = "TransposeD";
constexpr char kOpTypeReshape[] = "Reshape";
constexpr char kOpTypeTransposeBatchMatMul[] = "TransposeBatchMatMul";

constexpr int32_t kAllowDim = 3;
constexpr int32_t kPattern2AllowDim = 4;
constexpr int32_t kInputSizeWithoutBias = 2;
constexpr uint64_t kBlockByteSize = 32UL;
constexpr uint64_t kBasicBlockSize16 = 16UL;
constexpr uint64_t kBlockSize256 = 256UL;
constexpr uint64_t kFloat32Size = 4UL;
constexpr uint64_t kFloat16Size = 2UL;
constexpr uint64_t kDbSize = 2UL;
constexpr uint64_t kHf32EnableBit = 0x40UL;
constexpr int64_t kKNAlignValue = 128;
constexpr int64_t kMaxBatchKnThreshold = 65536;
constexpr int32_t kTargetVersionNum = 90100000;
constexpr int64_t kPattern2InnerAxisLimit = 65536;
constexpr char kPlatAscend910B[] = "Ascend910B";
constexpr char kPlatAscend910_93[] = "Ascend910_93";
constexpr char kPlatAscend950[] = "Ascend950";

// 3D shape 索引：x1=[batch, M, K], x2=[batch, K, N]，K 和 N 都在 index 2，
// kKDimIdx 用于 x1，kNDimIdx 用于 x2，语义不同但值相同。
constexpr int32_t kBatchDimIdx = 0;
constexpr int32_t kMDimIdx = 1;
constexpr int32_t kKDimIdx = 2;
constexpr int32_t kNDimIdx = 2;
// 4D shape 索引（Pattern2）：x1=(B2,B1,1,K), x2=(1,B1,K,N) 或 (1,B1,N,K)
constexpr int32_t kPattern2B2DimIdx = 0;
constexpr int32_t kPattern2B1DimIdx = 1;
constexpr int32_t kPattern2DummyDimIdx = 2;
constexpr int32_t kPattern2LastDimIdx = 3;

const std::vector<int64_t> kPerm012 = {0, 1, 2};
const std::vector<int64_t> kPerm102 = {1, 0, 2};
const std::vector<int64_t> kPerm021 = {0, 2, 1};

bool IsTargetVersion()
{
    int32_t version = 0;
    char geCompilerName[] = "ge-compiler";
    if (aclsysGetVersionNum(geCompilerName, &version) != ACL_SUCCESS) {
        OPS_LOG_W(kPassName, "Failed to get ge-compiler version, skip fusion.");
        return false;
    }
    return version >= kTargetVersionNum;
}

struct TbmmPerms {
    std::vector<int64_t> permX1;
    std::vector<int64_t> permX2;
    std::vector<int64_t> permY;
};

bool IsType(const GNode& node, const char* type)
{
    AscendString nodeType;
    return node.GetType(nodeType) == GRAPH_SUCCESS && nodeType == type;
}

bool IsBatchMatMulType(const GNode& node)
{
    return IsType(node, kOpTypeBatchMatMul) || IsType(node, kOpTypeBatchMatMulV2);
}

bool IsTransposeType(const GNode& node) { return IsType(node, kOpTypeTranspose) || IsType(node, kOpTypeTransposeD); }

GNodePtr GetInputNode(const GNode& dstNode, int64_t dstInputPort, int64_t* srcOutputPort = nullptr)
{
    auto [srcNode, resolvedSrcOutputPort] = dstNode.GetInDataNodesAndPortIndexs(dstInputPort);
    if (srcOutputPort != nullptr) {
        *srcOutputPort = resolvedSrcOutputPort;
    }
    return srcNode;
}

GNodePtr GetOutputNode(const GNode& srcNode, int64_t srcOutputPort, int64_t* dstInputPort = nullptr)
{
    auto consumers = srcNode.GetOutDataNodesAndPortIndexs(srcOutputPort);
    if (consumers.empty()) {
        return nullptr;
    }
    if (dstInputPort != nullptr) {
        *dstInputPort = consumers[0].second;
    }
    return consumers[0].first;
}

size_t GetOutputConsumerCount(const GNode& node, int64_t outputPort)
{
    return node.GetOutDataNodesAndPortIndexs(outputPort).size();
}

bool IsUnknownShape(const Shape& shape)
{
    const auto dims = shape.GetDims();
    return std::any_of(dims.begin(), dims.end(), [](const int64_t dim) {
        return dim == ge::UNKNOWN_DIM || dim == ge::UNKNOWN_DIM_NUM || dim < 0;
    });
}

std::string GetPlatform()
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optionalInfo;
    FUSION_PASS_CHECK(
        fe::PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platformInfo, optionalInfo) != SUCCESS,
        OPS_LOG_E(kPassName, "Get platform info failed."), return "");
    static const std::set<std::string> supportPlatList = {kPlatAscend910B, kPlatAscend910_93, kPlatAscend950};
    const std::string currentPlat = platformInfo.str_info.short_soc_version;
    FUSION_PASS_CHECK(supportPlatList.count(currentPlat) == 0,
                      OPS_LOG_W(kPassName, "Fusion TransposeBatchMatMul not supported on this platform."), return "");
    return currentPlat;
}

bool CheckTransposeNode(const GNodePtr& transNode, const std::vector<int64_t>& allowedPerm)
{
    FUSION_PASS_CHECK(transNode == nullptr, OPS_LOG_W(kPassName, "node is null, fusion is not supported"),
                      return false);
    FUSION_PASS_CHECK(!IsTransposeType(*transNode),
                      OPS_LOG_W(kPassName, "node is not transpose, fusion is not supported"), return false);
    std::vector<int64_t> permList;
    if (IsType(*transNode, kOpTypeTransposeD)) {
        FUSION_PASS_CHECK(transNode->GetAttr("perm", permList) != GRAPH_SUCCESS,
                          OPS_LOG_W(kPassName, "Get perm attr failed."), return false);
    } else {
        FUSION_PASS_CHECK(!GetTransposePermFromConst(*transNode, permList),
                          OPS_LOG_W(kPassName, "GetTransposePerm failed"), return false);
    }
    FUSION_PASS_CHECK(permList != allowedPerm, OPS_LOG_W(kPassName, "perm list not supported, fusion is not supported"),
                      return false);
    return true;
}

bool CheckReshapeNode(const GNodePtr& reshapeNode)
{
    FUSION_PASS_CHECK(reshapeNode == nullptr, OPS_LOG_W(kPassName, "reshape_node is null."), return false);
    FUSION_PASS_CHECK(!IsType(*reshapeNode, kOpTypeReshape), OPS_LOG_W(kPassName, "reshape_node type does not match."),
                      return false);
    FUSION_PASS_CHECK(GetOutputConsumerCount(*reshapeNode, 0) > 1,
                      OPS_LOG_W(kPassName, "reshape_node links to more than one node."), return false);
    TensorDesc reshapeInDesc;
    TensorDesc reshapeOutDesc;
    FUSION_PASS_CHECK(reshapeNode->GetInputDesc(0, reshapeInDesc) != GRAPH_SUCCESS ||
                          reshapeNode->GetOutputDesc(0, reshapeOutDesc) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "InputDesc/OutputDesc of reshape_node must not be null."), return false);
    bool validReshapePattern = reshapeInDesc.GetShape().GetDimNum() == kAllowDim &&
                               reshapeOutDesc.GetShape().GetDimNum() == kAllowDim;
    FUSION_PASS_CHECK(!validReshapePattern, OPS_LOG_W(kPassName, "reshape pattern of reshape_node is not as expected."),
                      return false);
    return true;
}

// 检测是否可以走入matmultomul模版
bool CheckShapeEqualToMul(const fe::PlatformInfo& platformInfo, uint64_t dtypeSize,
                          const std::vector<int64_t>& inputShapeX1, const std::vector<int64_t>& inputShapeX2)
{
    uint64_t inputM = static_cast<uint64_t>(inputShapeX1[kMDimIdx]);
    uint64_t inputK = static_cast<uint64_t>(inputShapeX1[kKDimIdx]);
    uint64_t inputN = static_cast<uint64_t>(inputShapeX2[kNDimIdx]);
    uint64_t batchNum = static_cast<uint64_t>(inputShapeX1[kBatchDimIdx]);
    constexpr uint64_t minBatchSize = 128UL;
    if (batchNum < minBatchSize) {
        return false;
    }
    if (inputK != 1UL) {
        return false;
    }
    if (inputN > kBlockByteSize / dtypeSize && inputN <= kBlockSize256 / dtypeSize) {
        return false;
    }
    if (inputN == 1UL) {
        return false;
    }
    uint64_t alignNum = kBlockByteSize / dtypeSize;
    uint64_t alignM = CeilAlign(inputM, alignNum);
    uint64_t alignN = CeilAlign(inputN, alignNum);
    if (alignM > platformInfo.ai_core_spec.ub_size / dtypeSize / (alignN + 1UL)) {
        return false;
    }
    if (static_cast<uint64_t>((alignM + alignN + alignM * alignN) * dtypeSize) > platformInfo.ai_core_spec.ub_size) {
        return false;
    }
    return inputN % (kBlockSize256 / dtypeSize) != 0UL;
}

// 检测是否可以走入iterbatch模版
bool CheckIterBatchMatmul(const fe::PlatformInfo& platformInfo, uint64_t dtypeSize,
                          const std::vector<int64_t>& inputShapeX1, const std::vector<int64_t>& inputShapeX2)
{
    uint64_t inputM = static_cast<uint64_t>(inputShapeX1[kMDimIdx]);
    uint64_t inputK = static_cast<uint64_t>(inputShapeX1[kKDimIdx]);
    uint64_t inputN = static_cast<uint64_t>(inputShapeX2[kNDimIdx]);
    uint64_t batchNum = static_cast<uint64_t>(inputShapeX1[kBatchDimIdx]);
    uint64_t aicNum = static_cast<uint64_t>(platformInfo.soc_info.ai_core_cnt);
    if (aicNum == 0UL) {
        return false;
    }
    uint64_t alignM = CeilAlign(inputM, kBasicBlockSize16);
    uint64_t alignK = CeilAlign(inputK, kBasicBlockSize16);
    uint64_t alignN = CeilAlign(inputN, kBasicBlockSize16);
    if (alignM == 0UL || alignK == 0UL || alignN == 0UL) {
        return false;
    }
    if (alignM > UINT64_MAX / alignK || alignK > UINT64_MAX / alignN) {
        return false;
    }
    uint64_t mkSize = alignM * alignK;
    uint64_t knSize = alignK * alignN;
    if (mkSize > UINT64_MAX - knSize || mkSize + knSize > UINT64_MAX / dtypeSize) {
        return false;
    }
    uint64_t inputSizeOneBatch = (mkSize + knSize) * dtypeSize;
    uint64_t iterBatch = FloorDiv(platformInfo.ai_core_spec.l1_size, inputSizeOneBatch);
    if (iterBatch > 1UL) {
        uint64_t preCoreBatch = CeilDiv(batchNum, aicNum);
        iterBatch = std::max(std::min(iterBatch, preCoreBatch), 1UL);
    }
    if (iterBatch <= 1UL) {
        return false;
    }
    uint64_t iterBatchL0A = FloorDiv(FloorDiv(platformInfo.ai_core_spec.l0_a_size, kDbSize),
                                     alignM * alignK * dtypeSize);
    uint64_t iterBatchL0B = FloorDiv(FloorDiv(platformInfo.ai_core_spec.l0_b_size, kDbSize),
                                     alignK * alignN * dtypeSize);
    uint64_t iterBatchL0C = FloorDiv(FloorDiv(platformInfo.ai_core_spec.l0_c_size, kDbSize),
                                     alignM * alignN * kFloat32Size);
    uint64_t iterBatchL1 = FloorDiv(FloorDiv(platformInfo.ai_core_spec.l1_size, kDbSize),
                                    (mkSize + knSize) * dtypeSize);
    constexpr double defaultBalanceOfBatch = 0.8;
    if (std::min({iterBatchL0A, iterBatchL0B, iterBatchL0C}) < 1UL) {
        double avgIterBatch = static_cast<double>(batchNum) / static_cast<double>(aicNum);
        double actualMaxIterBatch = static_cast<double>(CeilDiv(CeilDiv(batchNum, iterBatchL1), aicNum) * iterBatchL1);
        double balanceRateOfBatch = avgIterBatch / actualMaxIterBatch;
        if (balanceRateOfBatch < defaultBalanceOfBatch) {
            return false;
        }
    }
    return true;
}

// 检测是否可以走入优化模版
bool CheckOptimizedBatch(const GNode& bmmNode)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optionalInfo;
    FUSION_PASS_CHECK(
        fe::PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platformInfo, optionalInfo) != SUCCESS,
        OPS_LOG_E(kPassName, "Get platform info failed."), return false);
    TensorDesc x1Desc;
    TensorDesc x2Desc;
    FUSION_PASS_CHECK(bmmNode.GetInputDesc(kX1InputIdx, x1Desc) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "Get x1 input desc failed."), return false);
    FUSION_PASS_CHECK(bmmNode.GetInputDesc(kX2InputIdx, x2Desc) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "Get x2 input desc failed."), return false);
    auto inputShapeX1 = x1Desc.GetShape().GetDims();
    auto inputShapeX2 = x2Desc.GetShape().GetDims();
    uint64_t dtypeSize = x1Desc.GetDataType() == ge::DT_FLOAT ? kFloat32Size : kFloat16Size;
    bool shapeEqualToMul = CheckShapeEqualToMul(platformInfo, dtypeSize, inputShapeX1, inputShapeX2);
    bool iterBatchMatmul = CheckIterBatchMatmul(platformInfo, dtypeSize, inputShapeX1, inputShapeX2);
    if (shapeEqualToMul) {
        OPS_LOG_I(kPassName, "Enter matmultomul template (CheckShapeEqualToMul).");
    } else if (iterBatchMatmul) {
        OPS_LOG_I(kPassName, "Enter iterbatch matmul template (CheckIterBatchMatmul).");
    }
    return shapeEqualToMul || iterBatchMatmul;
}

void GetBatchSplitFactor(const GNode& bmmNode, int64_t batch, int64_t& batchSplitFactor)
{
    auto transNode3 = GetOutputNode(bmmNode, 0);
    if (transNode3 == nullptr) {
        OPS_LOG_W(kPassName, "trans_node_3 is null, skip batch split factor.");
        return;
    }
    FUSION_PASS_CHECK(GetOutputConsumerCount(*transNode3, 0) > 1,
                      OPS_LOG_W(kPassName, "trans_node_3 links to more than one node."), return);
    TensorDesc outputDescTrans3;
    FUSION_PASS_CHECK(transNode3->GetOutputDesc(0, outputDescTrans3) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "OutputDesc of trans_node_3 must not be null."), return);
    auto outShapeTrans3 = outputDescTrans3.GetShape().GetDims();

    auto reshapeNode1 = GetOutputNode(*transNode3, 0);
    FUSION_PASS_CHECK(!CheckReshapeNode(reshapeNode1), OPS_LOG_W(kPassName, "reshape_node_1 is not as expected."),
                      return);
    auto reshapeNode2 = GetOutputNode(*reshapeNode1, 0);
    FUSION_PASS_CHECK(!CheckReshapeNode(reshapeNode2), OPS_LOG_W(kPassName, "reshape_node_2 is not as expected."),
                      return);
    auto transNode4 = GetOutputNode(*reshapeNode2, 0);
    FUSION_PASS_CHECK(!CheckTransposeNode(transNode4, kPerm102),
                      OPS_LOG_W(kPassName, "trans_node_4 is not as expected."), return);
    TensorDesc outputDescTrans4;
    FUSION_PASS_CHECK(transNode4->GetOutputDesc(0, outputDescTrans4) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "OutputDesc of trans_node_4 must not be null."), return);
    auto outShapeTrans4 = outputDescTrans4.GetShape().GetDims();
    FUSION_PASS_CHECK(
        outputDescTrans3.GetShape().GetDimNum() != kAllowDim || outputDescTrans4.GetShape().GetDimNum() != kAllowDim,
        OPS_LOG_W(kPassName, "output dim num of trans_node should be 3"), return);
    FUSION_PASS_CHECK(outShapeTrans4[kBatchDimIdx] == 0 || batch % outShapeTrans4[kBatchDimIdx] != 0 ||
                          outShapeTrans3[0] != outShapeTrans4[kMDimIdx],
                      OPS_LOG_W(kPassName,
                                "the first dim of out_shape[%ld] should be a factor of batch[%ld], "
                                "and m_dim[%ld] should not be split.",
                                outShapeTrans4[kBatchDimIdx], batch, outShapeTrans4[kMDimIdx]),
                      return);
    batchSplitFactor = outShapeTrans4[kBatchDimIdx];
}

bool SetGeIrAttrs(GNode& tbmmNode, const GNode& bmmNode, const TbmmPerms& perms, int64_t batchSplitFactor)
{
    auto permX1 = perms.permX1;
    auto permX2 = perms.permX2;
    auto permY = perms.permY;
    FUSION_PASS_CHECK(tbmmNode.SetAttr("perm_x1", permX1) != GRAPH_SUCCESS, OPS_LOG_E(kPassName, "Set perm_x1 failed."),
                      return false);
    FUSION_PASS_CHECK(tbmmNode.SetAttr("perm_x2", permX2) != GRAPH_SUCCESS, OPS_LOG_E(kPassName, "Set perm_x2 failed."),
                      return false);
    FUSION_PASS_CHECK(tbmmNode.SetAttr("perm_y", permY) != GRAPH_SUCCESS, OPS_LOG_E(kPassName, "Set perm_y failed."),
                      return false);
    FUSION_PASS_CHECK(tbmmNode.SetAttr("batch_split_factor", batchSplitFactor) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "Set batch_split_factor failed."), return false);
    int64_t opImplModeEnum = 0;
    if (bmmNode.GetAttr("_op_impl_mode_enum", opImplModeEnum) == GRAPH_SUCCESS) {
        bool enableHf32 = (static_cast<uint64_t>(opImplModeEnum) & kHf32EnableBit) != 0UL;
        OPS_LOG_I(kPassName, "BMM enable_hf32: %d.", enableHf32);
        FUSION_PASS_CHECK(tbmmNode.SetAttr("enable_hf32", enableHf32) != GRAPH_SUCCESS,
                          OPS_LOG_E(kPassName, "Set enable_hf32 for TBMM failed."), return false);
    } else {
        OPS_LOG_W(kPassName, "Failed to get _op_impl_mode_enum from BMM.");
    }
    return true;
}

GNode CreateTransposeBatchMatMulNode(Graph* graph, const std::string& name)
{
    return es::CompliantNodeBuilder(graph)
        .OpType(kOpTypeTransposeBatchMatMul)
        .Name(name.c_str())
        .IrDefInputs({
            {"x1", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
            {"x2", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
            {"bias", es::CompliantNodeBuilder::kEsIrInputOptional, ""},
            {"scale", es::CompliantNodeBuilder::kEsIrInputOptional, ""},
        })
        .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
        .IrDefAttrs({
            {"perm_x1", es::CompliantNodeBuilder::kEsAttrRequired, "ListInt", es::CreateFrom(kPerm012)},
            {"perm_x2", es::CompliantNodeBuilder::kEsAttrRequired, "ListInt", es::CreateFrom(kPerm012)},
            {"perm_y", es::CompliantNodeBuilder::kEsAttrRequired, "ListInt", es::CreateFrom(kPerm102)},
            {"enable_hf32", es::CompliantNodeBuilder::kEsAttrRequired, "Bool", es::CreateFrom(false)},
            {"batch_split_factor", es::CompliantNodeBuilder::kEsAttrRequired, "Int",
             es::CreateFrom(static_cast<int64_t>(1))},
        })
        .Build();
}

GNode CreateReshapeNode(Graph* graph, const std::string& name, const TensorDesc& inDesc, const TensorDesc& outDesc,
                        const std::vector<int64_t>& shapeAttr)
{
    auto reshapeNode = es::CompliantNodeBuilder(graph)
                           .OpType(kOpTypeReshape)
                           .Name(name.c_str())
                           .IrDefInputs({{"x", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                                         {"shape", es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
                           .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                           .IrDefAttrs({{"axis", es::CompliantNodeBuilder::kEsAttrRequired, "Int",
                                         es::CreateFrom(static_cast<int64_t>(0))},
                                        {"num_axes", es::CompliantNodeBuilder::kEsAttrRequired, "Int",
                                         es::CreateFrom(static_cast<int64_t>(-1))}})
                           .Build();
    FUSION_PASS_CHECK(reshapeNode.UpdateInputDesc(0, inDesc) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "Update reshape input desc failed."), return reshapeNode);
    FUSION_PASS_CHECK(reshapeNode.UpdateOutputDesc(0, outDesc) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "Update reshape output desc failed."), return reshapeNode);
    TensorDesc shapeInputDesc(ge::Shape({static_cast<int64_t>(shapeAttr.size())}), FORMAT_ND, DT_INT64);
    FUSION_PASS_CHECK(reshapeNode.UpdateInputDesc(1, shapeInputDesc) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "Update reshape shape input desc failed."), return reshapeNode);
    auto shapeAttrCopy = shapeAttr;
    FUSION_PASS_CHECK(reshapeNode.SetAttr("shape", shapeAttrCopy) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "Set reshape shape attr failed."), return reshapeNode);
    return reshapeNode;
}

bool RemoveNodeFully(const GraphPtr& graph, GNode& node)
{
    auto inputSize = node.GetInputsSize();
    for (size_t port = 0; port < inputSize; ++port) {
        auto [srcNode, srcPort] = node.GetInDataNodesAndPortIndexs(static_cast<int64_t>(port));
        if (srcNode != nullptr) {
            FUSION_PASS_CHECK(graph->RemoveEdge(*srcNode, srcPort, node, port) != GRAPH_SUCCESS,
                              OPS_LOG_E(kPassName, "Failed to remove input edge."), return false);
        }
    }
    auto outputSize = node.GetOutputsSize();
    for (size_t port = 0; port < outputSize; ++port) {
        auto consumers = node.GetOutDataNodesAndPortIndexs(static_cast<int64_t>(port));
        for (auto& [consumer, inPort] : consumers) {
            if (consumer == nullptr) {
                OPS_LOG_W(kPassName, "Consumer node is null, skip.");
                continue;
            }
            FUSION_PASS_CHECK(graph->RemoveEdge(node, port, *consumer, inPort) != GRAPH_SUCCESS,
                              OPS_LOG_E(kPassName, "Failed to remove output edge."), return false);
        }
    }
    FUSION_PASS_CHECK(graph->RemoveNode(node) != GRAPH_SUCCESS, OPS_LOG_E(kPassName, "Failed to remove node."),
                      return false);
    return true;
}

bool RemoveNodeFully(const GraphPtr& graph, const GNodePtr& node)
{
    if (node == nullptr) {
        return true;
    }
    auto& ref = *node;
    return RemoveNodeFully(graph, ref);
}

// ==================== Pattern1 Check ====================

bool CheckBatchMatMulNodePattern1(const GNode& bmmNode, GNodePtr& transNode1, const std::string& currentPlat,
                                  int64_t& batchSplitFactor, int64_t& batch)
{
    bool x1TransFlag = false;
    bool x2TransFlag = false;
    FUSION_PASS_CHECK(bmmNode.GetAttr(kAttrAdjX1, x1TransFlag) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "Get adj_x1 failed."), return false);
    FUSION_PASS_CHECK(bmmNode.GetAttr(kAttrAdjX2, x2TransFlag) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "Get adj_x2 failed."), return false);
    FUSION_PASS_CHECK(x1TransFlag || x2TransFlag, OPS_LOG_W(kPassName, "bmm node's attr only supports no trans"),
                      return false);

    TensorDesc x1Desc;
    TensorDesc x2Desc;
    TensorDesc outputDesc;
    FUSION_PASS_CHECK(bmmNode.GetInputDesc(kX1InputIdx, x1Desc) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "Get x1 input desc failed."), return false);
    FUSION_PASS_CHECK(bmmNode.GetInputDesc(kX2InputIdx, x2Desc) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "Get x2 input desc failed."), return false);
    FUSION_PASS_CHECK(bmmNode.GetOutputDesc(0, outputDesc) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "Get output desc failed."), return false);

    bool isDynamic = IsUnknownShape(x1Desc.GetShape()) || IsUnknownShape(x2Desc.GetShape()) ||
                     IsUnknownShape(outputDesc.GetShape());
    FUSION_PASS_CHECK(isDynamic, OPS_LOG_W(kPassName, "only static shape is supported"), return false);

    bool isBf16Fp16Fp32 = (x1Desc.GetDataType() == ge::DT_FLOAT || x1Desc.GetDataType() == ge::DT_FLOAT16 ||
                           x1Desc.GetDataType() == ge::DT_BF16) &&
                          x1Desc.GetDataType() == x2Desc.GetDataType() &&
                          x1Desc.GetDataType() == outputDesc.GetDataType();
    FUSION_PASS_CHECK(!isBf16Fp16Fp32, OPS_LOG_W(kPassName, "only bf16/fp16/fp32 are supported"), return false);

    FUSION_PASS_CHECK(x1Desc.GetShape().GetDimNum() != kAllowDim || x2Desc.GetShape().GetDimNum() != kAllowDim,
                      OPS_LOG_W(kPassName, "input shape dim is not as expected"), return false);

    auto inputShapeX2 = x2Desc.GetShape().GetDims();
    batch = inputShapeX2[kBatchDimIdx];

    FUSION_PASS_CHECK(bmmNode.GetInputsSize() > kInputSizeWithoutBias,
                      OPS_LOG_W(kPassName, "tbmm does not support bias"), return false);

    if (currentPlat != kPlatAscend950) {
        int64_t inputK = inputShapeX2[1];
        int64_t inputN = inputShapeX2[kNDimIdx];
        bool isFp32 = x1Desc.GetDataType() == ge::DT_FLOAT && x1Desc.GetDataType() == x2Desc.GetDataType() &&
                      x1Desc.GetDataType() == outputDesc.GetDataType();
        // supportKN: K 和 N 都 128 对齐且 batch 小于阈值，满足基础条件
        // fp32 且有 transNode1（A 被 transpose）时强制放宽，因为 transpose 后的 A 走不同 TBE 模板
        bool supportKN = inputK % kKNAlignValue == 0 && inputN % kKNAlignValue == 0 && batch < kMaxBatchKnThreshold;
        if (isFp32 && transNode1 != nullptr) {
            supportKN = true;
        }
        // supportTransNode1: KN 对齐 + batch*K < 阈值 → 保留 transNode1
        // notSupportTransNode1: KN 对齐但 batch*K >= 阈值（L1 放不下）→ 丢弃 transNode1，让 A 不做 transpose
        bool supportTransNode1 = supportKN && batch * inputK < kMaxBatchKnThreshold && batch < kMaxBatchKnThreshold;
        bool notSupportTransNode1 = supportKN && batch * inputK >= kMaxBatchKnThreshold &&
                                    inputK < kMaxBatchKnThreshold;
        if (notSupportTransNode1) {
            transNode1 = nullptr;
        }
        FUSION_PASS_CHECK(!(supportTransNode1 || notSupportTransNode1),
                          OPS_LOG_W(kPassName, "bmm node's shape does not support fusion"), return false);
    } else {
        // 950 平台不需要 KN 对齐检查，而是走 CheckOptimizedBatch 判断是否可走 matmultomul 或 iterbatch 优化模板。
        // transNode1 必须为 null（950 上 A 的 transpose 由 TBE 内部处理，不需要外挂 transpose 节点）。
        FUSION_PASS_CHECK(transNode1 == nullptr && CheckOptimizedBatch(bmmNode),
                          OPS_LOG_W(kPassName, "bmm node's shape does not support fusion."), return false);
    }

    batchSplitFactor = 1;
    GetBatchSplitFactor(bmmNode, batch, batchSplitFactor);
    return true;
}

// ==================== Pattern2 Check ====================

bool CheckBatchMatMulNodePattern2(const GNode& bmmNode, TensorDesc& x1Desc, TensorDesc& x2Desc, TensorDesc& outputDesc)
{
    bool x1TransFlag = false;
    FUSION_PASS_CHECK(bmmNode.GetAttr(kAttrAdjX1, x1TransFlag) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "Get adj_x1 failed."), return false);
    FUSION_PASS_CHECK(x1TransFlag, OPS_LOG_W(kPassName, "bmm node's attr does not support A trans"), return false);

    FUSION_PASS_CHECK(bmmNode.GetInputDesc(kX1InputIdx, x1Desc) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "Get x1 input desc failed."), return false);
    FUSION_PASS_CHECK(bmmNode.GetInputDesc(kX2InputIdx, x2Desc) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "Get x2 input desc failed."), return false);
    FUSION_PASS_CHECK(bmmNode.GetOutputDesc(0, outputDesc) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "Get output desc failed."), return false);

    bool isDynamic = IsUnknownShape(x1Desc.GetShape()) || IsUnknownShape(x2Desc.GetShape()) ||
                     IsUnknownShape(outputDesc.GetShape());
    FUSION_PASS_CHECK(isDynamic, OPS_LOG_W(kPassName, "only static shape is supported"), return false);

    auto inputShapeX1 = x1Desc.GetShape().GetDims();
    auto inputShapeX2 = x2Desc.GetShape().GetDims();

    const std::string currentPlat = GetPlatform();
    bool supportInnerAxis = (currentPlat == kPlatAscend950 ||
                             inputShapeX1[kPattern2B1DimIdx] * inputShapeX1[kPattern2LastDimIdx] <
                                 kPattern2InnerAxisLimit);
    FUSION_PASS_CHECK(!supportInnerAxis, OPS_LOG_W(kPassName, "inner axis must be less than 65536"), return false);

    FUSION_PASS_CHECK(bmmNode.GetInputsSize() > kInputSizeWithoutBias,
                      OPS_LOG_W(kPassName, "tbmm does not support bias"), return false);

    return true;
}

// Pattern2 处理 4D 输入：x1=(B2,B1,1,K), x2=(1,B1,K,N) 或 (1,B1,N,K)。
// 校验 dummy 维（x1[2]==1, x2[0]==1）和 B1 一致性，以及 x1 最后一维 K 等于 x2 的 K 或 N 维。
// Pattern2 只支持 fp32，不支持 hf32（hf32 的精度特征与 Pattern2 的 TBE 模板不兼容）。
bool CheckPattern2Limit(const GNode& bmmNode)
{
    TensorDesc x1Desc;
    TensorDesc x2Desc;
    TensorDesc outputDesc;
    FUSION_PASS_CHECK(bmmNode.GetInputDesc(kX1InputIdx, x1Desc) != GRAPH_SUCCESS,
                      OPS_LOG_D(kPassName, "Get x1 input desc failed."), return false);
    FUSION_PASS_CHECK(bmmNode.GetInputDesc(kX2InputIdx, x2Desc) != GRAPH_SUCCESS,
                      OPS_LOG_D(kPassName, "Get x2 input desc failed."), return false);
    FUSION_PASS_CHECK(bmmNode.GetOutputDesc(0, outputDesc) != GRAPH_SUCCESS,
                      OPS_LOG_D(kPassName, "Get output desc failed."), return false);

    if (x1Desc.GetShape().GetDimNum() != kPattern2AllowDim || x2Desc.GetShape().GetDimNum() != kPattern2AllowDim) {
        OPS_LOG_D(kPassName, "Pattern2 input shape dim is not %d, x1:%zu, x2:%zu.", kPattern2AllowDim,
                  x1Desc.GetShape().GetDimNum(), x2Desc.GetShape().GetDimNum());
        return false;
    }
    auto inputShapeX1 = x1Desc.GetShape().GetDims();
    auto inputShapeX2 = x2Desc.GetShape().GetDims();
    if (inputShapeX1[kPattern2DummyDimIdx] != 1 || inputShapeX2[kPattern2B2DimIdx] != 1 ||
        inputShapeX1[kPattern2B1DimIdx] != inputShapeX2[kPattern2B1DimIdx] ||
        (inputShapeX1[kPattern2LastDimIdx] != inputShapeX2[kPattern2DummyDimIdx] &&
         inputShapeX1[kPattern2LastDimIdx] != inputShapeX2[kPattern2LastDimIdx])) {
        OPS_LOG_D(kPassName, "Pattern2 input shape does not match A(B2,B1,1,K) B(1,B1,K,N)/(1,B1,N,K).");
        return false;
    }
    if (x1Desc.GetDataType() != ge::DT_FLOAT || x1Desc.GetDataType() != x2Desc.GetDataType() ||
        x1Desc.GetDataType() != outputDesc.GetDataType()) {
        OPS_LOG_D(kPassName, "Pattern2 only supports fp32.");
        return false;
    }
    int64_t opImplModeEnum = 0;
    if (bmmNode.GetAttr("_op_impl_mode_enum", opImplModeEnum) == GRAPH_SUCCESS) {
        bool enableHf32 = (static_cast<uint64_t>(opImplModeEnum) & kHf32EnableBit) != 0UL;
        FUSION_PASS_CHECK(enableHf32, OPS_LOG_W(kPassName, "Pattern2 does not support hf32."), return false);
    }
    return true;
}

// ==================== Pattern1 Fusion ====================

// 返回 false = transNode1 的 perm 是 {0,2,1} 即 A trans，不支持此 pattern
// 返回 true = 可以继续，transNode2 不符合则丢弃
bool CheckBmmTransposePattern1(GNodePtr& transNode1, GNodePtr& transNode2)
{
    if (CheckTransposeNode(transNode1, kPerm021)) {
        OPS_LOG_W(kPassName, "bmm node's attr does not support A trans");
        return false;
    }
    if (!CheckTransposeNode(transNode2, kPerm021) || GetOutputConsumerCount(*transNode2, 0) != 1) {
        transNode2 = nullptr;
    }
    return true;
}

// 收集 Pattern1 周围的 Transpose 节点：
// transNode2 仅在 950 平台保留（B 的 transpose 在非 950 上不融合）；
// transNode1 输出有多个消费者时丢弃（不能安全移除）。
Status CollectTransposeNodesPattern1(const GNode& bmmNode, const std::string& currentPlat, GNodePtr& transNode1,
                                     GNodePtr& transNode2, GNodePtr& transNode3)
{
    transNode1 = GetInputNode(bmmNode, 0);
    transNode2 = GetInputNode(bmmNode, 1);
    if (!CheckTransposeNode(transNode1, kPerm102)) {
        transNode1 = nullptr;
    }
    if (currentPlat != kPlatAscend950) {
        transNode2 = nullptr;
    } else {
        FUSION_PASS_CHECK(!CheckBmmTransposePattern1(transNode1, transNode2),
                          OPS_LOG_W(kPassName, "bmm node's attr does not support trans"), return GRAPH_NOT_CHANGED);
    }
    transNode3 = GetOutputNode(bmmNode, 0);
    FUSION_PASS_CHECK(!CheckTransposeNode(transNode3, kPerm102),
                      OPS_LOG_W(kPassName, "bmm output node is invalid, fusion is not supported"),
                      return GRAPH_NOT_CHANGED);
    if (transNode1 != nullptr && GetOutputConsumerCount(*transNode1, 0) > 1) {
        transNode1 = nullptr;
    }
    return SUCCESS;
}

Status CreateAndSetupTbmmNodePattern1(const GraphPtr& graph, const GNode& bmmNode, const GNodePtr& transNode1,
                                      const GNodePtr& transNode2, const GNodePtr& transNode3, int64_t batchSplitFactor,
                                      GNode& tbmmNode)
{
    AscendString bmmName;
    FUSION_PASS_CHECK(bmmNode.GetName(bmmName) != GRAPH_SUCCESS, OPS_LOG_E(kPassName, "Get bmm node name failed."),
                      return FAILED);
    std::string tbmmName = std::string(bmmName.GetString()) + "_to_tranpose_batch_matmul";
    tbmmNode = CreateTransposeBatchMatMulNode(graph.get(), tbmmName);

    TensorDesc tbmmInput1Desc;
    if (transNode1 != nullptr) {
        FUSION_PASS_CHECK(transNode1->GetInputDesc(0, tbmmInput1Desc) != GRAPH_SUCCESS,
                          OPS_LOG_E(kPassName, "Get transNode1 input desc failed."), return FAILED);
    } else {
        FUSION_PASS_CHECK(bmmNode.GetInputDesc(0, tbmmInput1Desc) != GRAPH_SUCCESS,
                          OPS_LOG_E(kPassName, "Get bmm input desc 0 failed."), return FAILED);
    }
    FUSION_PASS_CHECK(tbmmNode.UpdateInputDesc(0, tbmmInput1Desc) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "Update tbmm input desc 0 failed."), return FAILED);

    TensorDesc tbmmInput2Desc;
    if (transNode2 != nullptr) {
        FUSION_PASS_CHECK(transNode2->GetInputDesc(0, tbmmInput2Desc) != GRAPH_SUCCESS,
                          OPS_LOG_E(kPassName, "Get transNode2 input desc failed."), return FAILED);
    } else {
        FUSION_PASS_CHECK(bmmNode.GetInputDesc(1, tbmmInput2Desc) != GRAPH_SUCCESS,
                          OPS_LOG_E(kPassName, "Get bmm input desc 1 failed."), return FAILED);
    }
    FUSION_PASS_CHECK(tbmmNode.UpdateInputDesc(1, tbmmInput2Desc) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "Update tbmm input desc 1 failed."), return FAILED);

    TensorDesc tbmmOutputDesc;
    FUSION_PASS_CHECK(transNode3->GetOutputDesc(0, tbmmOutputDesc) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "Get transNode3 output desc failed."), return FAILED);
    FUSION_PASS_CHECK(tbmmNode.UpdateOutputDesc(0, tbmmOutputDesc) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "Update tbmm output desc failed."), return FAILED);

    // 默认 perm: x1={0,1,2}(不 trans), x2={0,1,2}(不 trans), y={1,0,2}(输出 trans)。
    // 有 transNode1 → permX1={1,0,2}（A 的 transpose 合并到 TBMM 的 perm_x1）
    // 有 transNode2 → permX2={0,2,1}（B 的 transpose 合并到 TBMM 的 perm_x2）
    // 外部 Transpose 节点被消除，其语义转移到 TBMM 的 perm 属性中。
    TbmmPerms perms = {kPerm012, kPerm012, kPerm102};
    if (transNode1 != nullptr) {
        perms.permX1 = kPerm102;
    }
    if (transNode2 != nullptr) {
        perms.permX2 = kPerm021;
    }
    FUSION_PASS_CHECK(!SetGeIrAttrs(tbmmNode, bmmNode, perms, batchSplitFactor),
                      OPS_LOG_E(kPassName, "Set geir attrs failed."), return FAILED);
    FUSION_PASS_CHECK(!CopyOtherAttrs(bmmNode, tbmmNode, kPassName), OPS_LOG_E(kPassName, "Copy other attrs failed."),
                      return FAILED);
    return SUCCESS;
}

void ReportFusionPattern1(const GNode& bmmNode, const GNodePtr& transNode1, const GNodePtr& transNode2,
                          const GNodePtr& transNode3, const GNode& tbmmNode, CustomPassContext& passContext)
{
    std::vector<GNode> nodesBeforeFuse;
    nodesBeforeFuse.emplace_back(bmmNode);
    nodesBeforeFuse.emplace_back(*transNode3);
    if (transNode1 != nullptr) {
        nodesBeforeFuse.emplace_back(*transNode1);
    }
    if (transNode2 != nullptr) {
        nodesBeforeFuse.emplace_back(*transNode2);
    }
    if (ge::fusion::GraphFuseInspectorUtils::ReportFuse != nullptr) {
        if (ge::fusion::GraphFuseInspectorUtils::ReportFuse(nodesBeforeFuse, {tbmmNode}, passContext) != SUCCESS) {
            OPS_LOG_W(kPassName, "Failed to report fusion result.");
        }
    }
}

Status RelinkSingleInputEdge(const GraphPtr& graph, GNode& bmmNode, const GNodePtr& transNode, int64_t bmmInputPort,
                             GNode& tbmmNode, int64_t tbmmInputPort)
{
    int64_t srcOutPort = 0;
    auto srcNode = (transNode != nullptr) ? GetInputNode(*transNode, 0, &srcOutPort) :
                                            GetInputNode(bmmNode, bmmInputPort, &srcOutPort);
    if (srcNode == nullptr) {
        OPS_LOG_E(kPassName, "srcNode of input port %ld is null.", bmmInputPort);
        return FAILED;
    }
    if (transNode != nullptr) {
        FUSION_PASS_CHECK(graph->RemoveEdge(*srcNode, srcOutPort, *transNode, 0) != GRAPH_SUCCESS,
                          OPS_LOG_E(kPassName, "remove edge failed."), return FAILED);
        FUSION_PASS_CHECK(graph->RemoveEdge(*transNode, 0, bmmNode, bmmInputPort) != GRAPH_SUCCESS,
                          OPS_LOG_E(kPassName, "remove edge failed."), return FAILED);
    } else {
        FUSION_PASS_CHECK(graph->RemoveEdge(*srcNode, srcOutPort, bmmNode, bmmInputPort) != GRAPH_SUCCESS,
                          OPS_LOG_E(kPassName, "remove edge failed."), return FAILED);
    }
    FUSION_PASS_CHECK(graph->AddDataEdge(*srcNode, srcOutPort, tbmmNode, tbmmInputPort) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "add edge failed."), return FAILED);
    return SUCCESS;
}

Status RelinkTbmmInputEdgesPattern1(const GraphPtr& graph, GNode& bmmNode, const GNodePtr& transNode1,
                                    const GNodePtr& transNode2, GNode& tbmmNode)
{
    FUSION_PASS_CHECK(RelinkSingleInputEdge(graph, bmmNode, transNode1, 0, tbmmNode, 0) != SUCCESS,
                      OPS_LOG_E(kPassName, "relink input 0 failed."), return FAILED);
    FUSION_PASS_CHECK(RelinkSingleInputEdge(graph, bmmNode, transNode2, 1, tbmmNode, 1) != SUCCESS,
                      OPS_LOG_E(kPassName, "relink input 1 failed."), return FAILED);
    return SUCCESS;
}

// batch 拆分场景：transNode3 后面有 reshape→reshape→transNode4 链，
// 用于将 batch 拆分后的输出恢复为原始 shape。
// 用 transNode4 的输出 desc 更新 tbmm 输出，将 transNode4 的下游边迁移到 tbmm，
// 然后移除 transNode3、reshapeNode1、reshapeNode2、transNode4。
Status ProcessBatchSplitPattern1(const GraphPtr& graph, GNode& tbmmNode, const GNodePtr& transNode3)
{
    auto reshapeNode1 = GetOutputNode(*transNode3, 0);
    auto reshapeNode2 = (reshapeNode1 != nullptr) ? GetOutputNode(*reshapeNode1, 0) : nullptr;
    auto transNode4 = (reshapeNode2 != nullptr) ? GetOutputNode(*reshapeNode2, 0) : nullptr;

    if (transNode4 != nullptr) {
        TensorDesc transNode4OutDesc;
        FUSION_PASS_CHECK(transNode4->GetOutputDesc(0, transNode4OutDesc) != GRAPH_SUCCESS,
                          OPS_LOG_E(kPassName, "Get transNode4 output desc failed."), return FAILED);
        FUSION_PASS_CHECK(tbmmNode.UpdateOutputDesc(0, transNode4OutDesc) != GRAPH_SUCCESS,
                          OPS_LOG_E(kPassName, "Update tbmm output desc failed."), return FAILED);
        auto transNode4Consumers = transNode4->GetOutDataNodesAndPortIndexs(0);
        for (auto& [consumer, inPort] : transNode4Consumers) {
            if (consumer == nullptr) {
                OPS_LOG_W(kPassName, "Consumer node is null, skip.");
                continue;
            }
            FUSION_PASS_CHECK(graph->RemoveEdge(*transNode4, 0, *consumer, inPort) != GRAPH_SUCCESS,
                              OPS_LOG_E(kPassName, "remove edge failed."), return FAILED);
            FUSION_PASS_CHECK(graph->AddDataEdge(tbmmNode, 0, *consumer, inPort) != GRAPH_SUCCESS,
                              OPS_LOG_E(kPassName, "add edge failed."), return FAILED);
        }
    }
    FUSION_PASS_CHECK(!RemoveNodeFully(graph, transNode3), OPS_LOG_E(kPassName, "remove trans_node_3 node failed."),
                      return FAILED);
    if (reshapeNode1 != nullptr) {
        FUSION_PASS_CHECK(!RemoveNodeFully(graph, reshapeNode1),
                          OPS_LOG_E(kPassName, "remove reshape_node_1 node failed."), return FAILED);
    }
    if (reshapeNode2 != nullptr) {
        FUSION_PASS_CHECK(!RemoveNodeFully(graph, reshapeNode2),
                          OPS_LOG_E(kPassName, "remove reshape_node_2 node failed."), return FAILED);
    }
    if (transNode4 != nullptr) {
        FUSION_PASS_CHECK(!RemoveNodeFully(graph, transNode4), OPS_LOG_E(kPassName, "remove trans_node_4 node failed."),
                          return FAILED);
    }
    return SUCCESS;
}

Status RelinkTbmmOutputEdgesPattern1(const GraphPtr& graph, GNode& tbmmNode, const GNodePtr& transNode3)
{
    auto transNode3Consumers = transNode3->GetOutDataNodesAndPortIndexs(0);
    for (auto& [consumer, inPort] : transNode3Consumers) {
        if (consumer == nullptr) {
            OPS_LOG_W(kPassName, "Consumer node is null, skip.");
            continue;
        }
        FUSION_PASS_CHECK(graph->RemoveEdge(*transNode3, 0, *consumer, inPort) != GRAPH_SUCCESS,
                          OPS_LOG_E(kPassName, "remove edge failed."), return FAILED);
        FUSION_PASS_CHECK(graph->AddDataEdge(tbmmNode, 0, *consumer, inPort) != GRAPH_SUCCESS,
                          OPS_LOG_E(kPassName, "add edge failed."), return FAILED);
    }
    FUSION_PASS_CHECK(!RemoveNodeFully(graph, transNode3), OPS_LOG_E(kPassName, "remove trans_node_3 node failed."),
                      return FAILED);
    return SUCCESS;
}

Status RemoveOldNodesPattern1(const GraphPtr& graph, GNode& bmmNode, GNodePtr& transNode1, GNodePtr& transNode2)
{
    if (transNode1 != nullptr) {
        FUSION_PASS_CHECK(!RemoveNodeFully(graph, transNode1), OPS_LOG_E(kPassName, "remove trans_node_1 node failed."),
                          return FAILED);
    }
    if (transNode2 != nullptr) {
        FUSION_PASS_CHECK(!RemoveNodeFully(graph, transNode2), OPS_LOG_E(kPassName, "remove trans_node_2 node failed."),
                          return FAILED);
    }
    FUSION_PASS_CHECK(!RemoveNodeFully(graph, bmmNode), OPS_LOG_E(kPassName, "remove batch_matmul node failed."),
                      return FAILED);
    return SUCCESS;
}

Status DoFusionPattern1(const GraphPtr& graph, GNode& bmmNode, const std::string& currentPlat,
                        CustomPassContext& passContext)
{
    OPS_LOG_I(kPassName, "Begin DoFusionPattern1.");

    GNodePtr transNode1;
    GNodePtr transNode2;
    GNodePtr transNode3;
    FUSION_PASS_CHECK(
        CollectTransposeNodesPattern1(bmmNode, currentPlat, transNode1, transNode2, transNode3) != SUCCESS,
        OPS_LOG_W(kPassName, "Collect transpose nodes failed."), return GRAPH_NOT_CHANGED);

    int64_t batchSplitFactor = 1;
    int64_t batch = 1;
    FUSION_PASS_CHECK(!CheckBatchMatMulNodePattern1(bmmNode, transNode1, currentPlat, batchSplitFactor, batch),
                      OPS_LOG_W(kPassName, "Parameter[bmm_node] should not be changed."), return GRAPH_NOT_CHANGED);

    GNode tbmmNode;
    FUSION_PASS_CHECK(CreateAndSetupTbmmNodePattern1(graph, bmmNode, transNode1, transNode2, transNode3,
                                                     batchSplitFactor, tbmmNode) != SUCCESS,
                      OPS_LOG_E(kPassName, "CreateAndSetupTbmmNode failed."), return FAILED);

    // 必须在删除旧节点之前上报融合结果，因为 ReportFuse 要求 nodesBeforeFuse 中的节点仍属于当前图
    ReportFusionPattern1(bmmNode, transNode1, transNode2, transNode3, tbmmNode, passContext);

    FUSION_PASS_CHECK(RelinkTbmmInputEdgesPattern1(graph, bmmNode, transNode1, transNode2, tbmmNode) != SUCCESS,
                      OPS_LOG_E(kPassName, "RelinkTbmmInputEdges failed."), return FAILED);

    if (batchSplitFactor > 1) {
        FUSION_PASS_CHECK(ProcessBatchSplitPattern1(graph, tbmmNode, transNode3) != SUCCESS,
                          OPS_LOG_E(kPassName, "ProcessBatchSplit failed."), return FAILED);
    } else {
        FUSION_PASS_CHECK(RelinkTbmmOutputEdgesPattern1(graph, tbmmNode, transNode3) != SUCCESS,
                          OPS_LOG_E(kPassName, "RelinkTbmmOutputEdges failed."), return FAILED);
    }

    FUSION_PASS_CHECK(RemoveOldNodesPattern1(graph, bmmNode, transNode1, transNode2) != SUCCESS,
                      OPS_LOG_E(kPassName, "RemoveOldNodes failed."), return FAILED);

    OPS_LOG_I(kPassName, "batchmatmul to transposeBatchmatmul pattern1 fusion succeeded.");
    return SUCCESS;
}

// ==================== Pattern2 Fusion ====================

Status CreateInputReshapeNode(const GraphPtr& graph, GNode& bmmNode, const std::string& namePrefix, int64_t inputPort,
                              const std::vector<int64_t>& tbmmShape, GNode& reshapeNode)
{
    int64_t srcOutPort = 0;
    auto srcNode = GetInputNode(bmmNode, inputPort, &srcOutPort);
    FUSION_PASS_CHECK(srcNode == nullptr,
                      OPS_LOG_E(kPassName, "Failed to get x%d source node.", static_cast<int32_t>(inputPort + 1)),
                      return FAILED);
    TensorDesc srcOutDesc;
    FUSION_PASS_CHECK(srcNode->GetOutputDesc(srcOutPort, srcOutDesc) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "Get x%d source output desc failed.", static_cast<int32_t>(inputPort + 1)),
                      return FAILED);
    TensorDesc reshapeOutDesc = srcOutDesc;
    reshapeOutDesc.SetShape(ge::Shape(tbmmShape));
    reshapeOutDesc.SetOriginShape(ge::Shape(tbmmShape));
    reshapeNode = CreateReshapeNode(graph.get(), namePrefix + "_Reshape_" + std::to_string(inputPort), srcOutDesc,
                                    reshapeOutDesc, tbmmShape);
    FUSION_PASS_CHECK(graph->RemoveEdge(*srcNode, srcOutPort, bmmNode, inputPort) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "remove edge failed."), return FAILED);
    FUSION_PASS_CHECK(graph->AddDataEdge(*srcNode, srcOutPort, reshapeNode, 0) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "add edge failed."), return FAILED);
    return SUCCESS;
}

Status CreateInputReshapeNodes(const GraphPtr& graph, GNode& bmmNode, const std::string& namePrefix,
                               const std::vector<int64_t>& tbmmShapeX1, const std::vector<int64_t>& tbmmShapeX2,
                               GNode& reshape1Node, GNode& reshape2Node)
{
    FUSION_PASS_CHECK(CreateInputReshapeNode(graph, bmmNode, namePrefix, 0, tbmmShapeX1, reshape1Node) != SUCCESS,
                      OPS_LOG_E(kPassName, "CreateReshapeNode1 failed."), return FAILED);
    FUSION_PASS_CHECK(CreateInputReshapeNode(graph, bmmNode, namePrefix, 1, tbmmShapeX2, reshape2Node) != SUCCESS,
                      OPS_LOG_E(kPassName, "CreateReshapeNode2 failed."), return FAILED);
    return SUCCESS;
}

Status CreateTbmmNodePattern2(const GraphPtr& graph, const std::string& tbmmName, GNode& reshape1Node,
                              GNode& reshape2Node, const GNode& bmmNode, const TensorDesc& outputDesc,
                              const std::vector<int64_t>& tbmmShapeY, GNode& tbmmNode)
{
    TensorDesc reshape1OutDesc;
    TensorDesc reshape2OutDesc;
    FUSION_PASS_CHECK(reshape1Node.GetOutputDesc(0, reshape1OutDesc) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "Get reshape1 output desc failed."), return FAILED);
    FUSION_PASS_CHECK(reshape2Node.GetOutputDesc(0, reshape2OutDesc) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "Get reshape2 output desc failed."), return FAILED);

    tbmmNode = CreateTransposeBatchMatMulNode(graph.get(), tbmmName);
    FUSION_PASS_CHECK(tbmmNode.UpdateInputDesc(0, reshape1OutDesc) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "Update tbmm input desc 0 failed."), return FAILED);
    FUSION_PASS_CHECK(tbmmNode.UpdateInputDesc(1, reshape2OutDesc) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "Update tbmm input desc 1 failed."), return FAILED);
    TensorDesc tbmmOutDesc = outputDesc;
    tbmmOutDesc.SetShape(ge::Shape(tbmmShapeY));
    tbmmOutDesc.SetOriginShape(ge::Shape(tbmmShapeY));
    FUSION_PASS_CHECK(tbmmNode.UpdateOutputDesc(0, tbmmOutDesc) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "Update tbmm output desc failed."), return FAILED);
    FUSION_PASS_CHECK(graph->AddDataEdge(reshape1Node, 0, tbmmNode, 0) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "add edge failed."), return FAILED);
    FUSION_PASS_CHECK(graph->AddDataEdge(reshape2Node, 0, tbmmNode, 1) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "add edge failed."), return FAILED);

    bool x2TransFlag = false;
    FUSION_PASS_CHECK(bmmNode.GetAttr(kAttrAdjX2, x2TransFlag) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "Get adj_x2 failed."), return FAILED);
    TbmmPerms perms = {kPerm102, kPerm012, kPerm102};
    if (x2TransFlag) {
        perms.permX2 = kPerm021;
    }
    FUSION_PASS_CHECK(!SetGeIrAttrs(tbmmNode, bmmNode, perms, 1), OPS_LOG_E(kPassName, "Set geir attrs failed."),
                      return FAILED);
    FUSION_PASS_CHECK(!CopyOtherAttrs(bmmNode, tbmmNode, kPassName), OPS_LOG_E(kPassName, "Copy other attrs failed."),
                      return FAILED);
    return SUCCESS;
}

Status CreateOutputReshapeAndRelink(const GraphPtr& graph, GNode& bmmNode, GNode& tbmmNode,
                                    const std::string& namePrefix, const TensorDesc& tbmmOutDesc,
                                    const TensorDesc& outputDesc, const std::vector<int64_t>& shapeY,
                                    GNode& reshape3Node)
{
    reshape3Node = CreateReshapeNode(graph.get(), namePrefix + "_Reshape_3", tbmmOutDesc, outputDesc, shapeY);
    FUSION_PASS_CHECK(graph->AddDataEdge(tbmmNode, 0, reshape3Node, 0) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "add edge failed."), return FAILED);
    auto bmmConsumers = bmmNode.GetOutDataNodesAndPortIndexs(0);
    for (auto& [consumer, inPort] : bmmConsumers) {
        if (consumer == nullptr) {
            OPS_LOG_W(kPassName, "Consumer node is null, skip.");
            continue;
        }
        FUSION_PASS_CHECK(graph->RemoveEdge(bmmNode, 0, *consumer, inPort) != GRAPH_SUCCESS,
                          OPS_LOG_E(kPassName, "remove edge failed."), return FAILED);
        FUSION_PASS_CHECK(graph->AddDataEdge(reshape3Node, 0, *consumer, inPort) != GRAPH_SUCCESS,
                          OPS_LOG_E(kPassName, "add edge failed."), return FAILED);
    }
    return SUCCESS;
}

Status DoFusionPattern2(const GraphPtr& graph, GNode& bmmNode, CustomPassContext& passContext)
{
    OPS_LOG_I(kPassName, "Begin DoFusionPattern2.");

    TensorDesc x1Desc;
    TensorDesc x2Desc;
    TensorDesc outputDesc;
    FUSION_PASS_CHECK(!CheckBatchMatMulNodePattern2(bmmNode, x1Desc, x2Desc, outputDesc),
                      OPS_LOG_W(kPassName, "Parameter[bmm_node] should not be changed."), return GRAPH_NOT_CHANGED);

    auto shapeX1 = x1Desc.GetShape().GetDims();
    auto shapeX2 = x2Desc.GetShape().GetDims();
    auto shapeY = outputDesc.GetShape().GetDims();
    // 将 4D 输入 reshape 为 3D 送入 TBMM：
    // x1: (B2,B1,1,K) → (B2,B1,K)，去掉 dummy 维
    // x2: (1,B1,K,N) → (B1,K,N) 或 (1,B1,N,K) → (B1,N,K)，去掉 B2=1 维
    // y: (B2,B1,N) → 保持 3D，融合后通过 reshape3Node 恢复为原始 4D shape
    std::vector<int64_t> tbmmShapeX1 = {shapeX1[kPattern2B2DimIdx], shapeX1[kPattern2B1DimIdx],
                                        shapeX1[kPattern2LastDimIdx]};
    std::vector<int64_t> tbmmShapeX2 = {shapeX2[kPattern2B1DimIdx], shapeX2[kPattern2DummyDimIdx],
                                        shapeX2[kPattern2LastDimIdx]};
    std::vector<int64_t> tbmmShapeY = {shapeY[kPattern2B2DimIdx], shapeY[kPattern2B1DimIdx],
                                       shapeY[kPattern2LastDimIdx]};

    AscendString bmmName;
    FUSION_PASS_CHECK(bmmNode.GetName(bmmName) != GRAPH_SUCCESS, OPS_LOG_E(kPassName, "Get bmm node name failed."),
                      return FAILED);
    std::string namePrefix = std::string(bmmName.GetString()) + "_cann";

    GNode reshape1Node;
    GNode reshape2Node;
    FUSION_PASS_CHECK(CreateInputReshapeNodes(graph, bmmNode, namePrefix, tbmmShapeX1, tbmmShapeX2, reshape1Node,
                                              reshape2Node) != SUCCESS,
                      OPS_LOG_E(kPassName, "CreateInputReshapeNodes failed."), return FAILED);

    std::string tbmmName = std::string(bmmName.GetString()) + "_to_tranpose_batch_matmul";
    GNode tbmmNode;
    FUSION_PASS_CHECK(CreateTbmmNodePattern2(graph, tbmmName, reshape1Node, reshape2Node, bmmNode, outputDesc,
                                             tbmmShapeY, tbmmNode) != SUCCESS,
                      OPS_LOG_E(kPassName, "CreateTbmmNode failed."), return FAILED);

    GNode reshape3Node;
    TensorDesc tbmmOutDesc;
    FUSION_PASS_CHECK(tbmmNode.GetOutputDesc(0, tbmmOutDesc) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "Get tbmm output desc failed."), return FAILED);
    FUSION_PASS_CHECK(CreateOutputReshapeAndRelink(graph, bmmNode, tbmmNode, namePrefix, tbmmOutDesc, outputDesc,
                                                   shapeY, reshape3Node) != SUCCESS,
                      OPS_LOG_E(kPassName, "CreateOutputReshapeNode failed."), return FAILED);

    std::vector<GNode> nodesBeforeFuse = {bmmNode};
    if (ge::fusion::GraphFuseInspectorUtils::ReportFuse != nullptr) {
        if (ge::fusion::GraphFuseInspectorUtils::ReportFuse(
                nodesBeforeFuse, {tbmmNode, reshape1Node, reshape2Node, reshape3Node}, passContext) != SUCCESS) {
            OPS_LOG_W(kPassName, "Failed to report fusion result.");
        }
    }
    FUSION_PASS_CHECK(!RemoveNodeFully(graph, bmmNode), OPS_LOG_E(kPassName, "remove batch_matmul node failed."),
                      return FAILED);

    OPS_LOG_I(kPassName, "batchmatmul to transposeBatchmatmul pattern2 fusion succeeded.");
    return SUCCESS;
}

// ==================== Main Entry ====================

bool ValidateNodeInputs(const GNode& matchedNode)
{
    TensorDesc x1Desc;
    TensorDesc x2Desc;
    TensorDesc outputDesc;
    FUSION_PASS_CHECK(matchedNode.GetInputDesc(kX1InputIdx, x1Desc) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "Get x1 input desc failed."), return false);
    FUSION_PASS_CHECK(matchedNode.GetInputDesc(kX2InputIdx, x2Desc) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "Get x2 input desc failed."), return false);
    FUSION_PASS_CHECK(matchedNode.GetOutputDesc(0, outputDesc) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "Get output desc failed."), return false);
    return true;
}

Status ProcessBatchMatMulNode(const GraphPtr& graph, GNode& bmmNode, CustomPassContext& passContext)
{
    const std::string currentPlat = GetPlatform();
    FUSION_PASS_CHECK(currentPlat.empty(), OPS_LOG_W(kPassName, "platform limit."), return GRAPH_NOT_CHANGED);

    FUSION_PASS_CHECK(!ValidateNodeInputs(bmmNode), OPS_LOG_E(kPassName, "Validate node inputs failed."),
                      return GRAPH_NOT_CHANGED);

    if (CheckPattern2Limit(bmmNode)) {
        OPS_LOG_I(kPassName, "Match pattern2, begin DoFusionPattern2.");
        auto status = DoFusionPattern2(graph, bmmNode, passContext);
        if (status != SUCCESS) {
            OPS_LOG_W(kPassName, "Bmm Node in Pattern 2 does not satisfy condition.");
        }
        return status;
    }
    OPS_LOG_I(kPassName, "Match pattern1, begin DoFusionPattern1.");
    auto status = DoFusionPattern1(graph, bmmNode, currentPlat, passContext);
    if (status != SUCCESS) {
        OPS_LOG_W(kPassName, "Bmm Node in Pattern 1 does not satisfy condition.");
    }
    return status;
}

} // namespace

Status BatchMatMul2TransposeBatchMatMulFusionPass::Run(GraphPtr& graph, CustomPassContext& passContext)
{
    OPS_LOG_D(kPassName, "Enter BatchMatMul2TransposeBatchMatMulFusionPass.");
    if (!IsTargetVersion()) {
        return GRAPH_NOT_CHANGED;
    }
    FUSION_PASS_CHECK(graph == nullptr || !graph->IsValid(),
                      OPS_LOG_W(kPassName, "Graph is null or invalid, skip fusion pass."), return GRAPH_NOT_CHANGED);

    passContext.SetPassName(kPassName);

    std::vector<GNode> bmmNodeList;
    for (auto& node : graph->GetDirectNode()) {
        if (IsBatchMatMulType(node)) {
            bmmNodeList.emplace_back(node);
        }
    }
    if (bmmNodeList.empty()) {
        OPS_LOG_W(kPassName, "No BatchMatMul/BatchMatMulV2 node, skip fusion pass.");
        return GRAPH_NOT_CHANGED;
    }

    bool changed = false;
    for (auto& bmmNode : bmmNodeList) {
        auto status = ProcessBatchMatMulNode(graph, bmmNode, passContext);
        if (status == SUCCESS) {
            changed = true;
        } else if (status != GRAPH_NOT_CHANGED) {
            return status;
        }
    }

    OPS_LOG_D(kPassName, "Exit BatchMatMul2TransposeBatchMatMulFusionPass.");
    return changed ? SUCCESS : GRAPH_NOT_CHANGED;
}

// 满足目标版本时用 kCompatibleInherited（InferShape 前执行，与旧框架一致），
// 不满足时降级到 kAfterInferShape，保证旧版本 CANN 兼容性。
#if GE_COMPILER_VERSION_NUM >= 90100000
REG_FUSION_PASS(BatchMatMul2TransposeBatchMatMulFusionPass)
    .Stage(IsTargetVersion() ? CustomPassStage::kCompatibleInherited : CustomPassStage::kAfterInferShape);
#endif

} // namespace ops

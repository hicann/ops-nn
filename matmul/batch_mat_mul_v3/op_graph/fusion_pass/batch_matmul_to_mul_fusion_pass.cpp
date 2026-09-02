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
 * \file batch_matmul_to_mul_fusion_pass.cpp
 * \brief batch_matmul to mul fusion pass(batchmatmul K==1 --> mul)
 *
 * fusion rule like this:
 *    data      data                data         data
 *       \      /                     |            |
 *        \    /                   reshape      reshape
 *   matmul/batch_matmul               \          /
 *          |                              mul
 *          |           ---->               |
 *          v                               |
 *         out                              v
 *                                         out
 */

#include "batch_matmul_to_mul_fusion_pass.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include "es_math_ops.h"
#include "ge/compliant_node_builder.h"
#include "ge/es_graph_builder.h"
#include "platform/platform_info.h"
#include "common/inc/error_util.h"
#include "common/op_graph/fusion_pass/matmul_fusion_utils_pass.h"
#include "common/op_host/math_util_nn.h"

using namespace ge;
using namespace ge::es;
using namespace ge::fusion;
using namespace fe;

namespace ops {
namespace {

constexpr char kPassName[] = "BatchMatMul2MulFusionPass";
constexpr int32_t kGeCompilerVersion900 = 90000000;
constexpr int32_t kInputSizeWithoutBias = 2;
constexpr int32_t kLastOne = 1;
constexpr int32_t kSecondLast = 2;
constexpr int32_t kThirdLast = 3;
constexpr int32_t kFourthLast = 4;
constexpr int32_t kFifthLast = 5;
constexpr int32_t kSixthLast = 6;
constexpr int32_t kMaxBatchDimNum = 4;
constexpr int32_t kBatchIndex = 0;
constexpr int32_t kSecondIndex = 1;
constexpr int32_t kThirdIndex = 2;
constexpr uint32_t kLeftShapeDim = 3;
constexpr uint32_t kRightShapeDim = 3;
constexpr int64_t kBlockSize = 16L;
constexpr int64_t kBlockByteSize = 32L;
constexpr int64_t kBlockSize256 = 256L;
constexpr int64_t kFloatSize = 4L;
constexpr int64_t kFp16Size = 2L;
constexpr int64_t kPingpong = 2L;
constexpr int64_t kNRestrictValue = 4000L;
constexpr int64_t kMRestrictValue = 8166L;
constexpr int64_t kMnRestrictValue = 10000L;
constexpr int64_t kBatchThreshold = 128L;
constexpr char kOpTypeReduceSumD[] = "ReduceSumD";
constexpr char kOpTypeCast[] = "Cast";

struct BatchMatMulToMulArgs {
    int64_t m = 0;
    int64_t k = 0;
    int64_t n = 0;
    bool adjX1 = false;
    bool adjX2 = false;
    bool matmulFlag = false;
    std::vector<int64_t> batchX1;
    std::vector<int64_t> batchX2;
};

bool IsMatMulType(const GNode& node)
{
    AscendString opType;
    if (node.GetType(opType) != GRAPH_SUCCESS) {
        return false;
    }
    return opType == kOpTypeMatMul || opType == kOpTypeMatMulV2;
}

bool IsUnknownShape(const ge::Shape& shape)
{
    const auto dims = shape.GetDims();
    return std::any_of(dims.begin(), dims.end(), [](const int64_t dim) {
        return dim == ge::UNKNOWN_DIM || dim == ge::UNKNOWN_DIM_NUM || dim < 0;
    });
}

int64_t GetDtypeSize(DataType dtype)
{
    if (dtype == DT_FLOAT16 || dtype == DT_BF16) {
        return kFp16Size;
    }
    return kFloatSize;
}

bool CheckSocNeedBatchMatMulToMul910B(const PlatformInfo& platformInfo, const BatchMatMulToMulArgs& args,
                                      DataType x1Dtype, DataType x2Dtype, DataType outDtype)
{
    bool checkBf16Scene = (x1Dtype == DT_BF16) && (x2Dtype == DT_BF16) && (outDtype == DT_BF16) &&
                          (args.n < kNRestrictValue && args.m < kMRestrictValue &&
                           (!args.matmulFlag || args.m * args.n >= kMnRestrictValue));
    bool checkFp16Scene = (x1Dtype == DT_FLOAT16) && (x2Dtype == DT_FLOAT16) && (outDtype == DT_FLOAT16);
    bool checkFp32Scene = (x1Dtype == DT_FLOAT) && (x2Dtype == DT_FLOAT);
    return checkBf16Scene || checkFp16Scene || checkFp32Scene;
}

bool CheckSocNeedBatchMatMulToMulSupportS8S4(DataType x1Dtype, DataType x2Dtype, DataType outDtype)
{
    return (x1Dtype == DT_FLOAT16) && (x2Dtype == DT_FLOAT16) && (outDtype == DT_FLOAT16);
}

bool CheckShapeEqualToMul(const PlatformInfo& platformInfo, int64_t mDim, int64_t nDim, uint64_t batchNum,
                          int64_t dataSize)
{
    // if batch Num >= 128
    if (batchNum < static_cast<uint64_t>(kBatchThreshold)) {
        return false;
    }
    if (nDim > kBlockByteSize / dataSize && nDim <= kBlockSize256 / dataSize) {
        return false;
    }
    if (nDim == 1) {
        return false;
    }
    int64_t alignNum = kBlockByteSize / dataSize;
    int64_t alignM = ops::CeilAlign(mDim, alignNum);
    int64_t alignN = ops::CeilAlign(nDim, alignNum);
    if (static_cast<uint64_t>((alignM + alignN + alignM * alignN) * dataSize) > platformInfo.ai_core_spec.ub_size) {
        return false;
    }
    // if n align to 256B
    return nDim % (kBlockSize256 / dataSize) != 0;
}

std::vector<int64_t> GetBatchDim(const std::vector<int64_t>& inputShape)
{
    int64_t inputShapeSize = static_cast<int64_t>(inputShape.size());
    int64_t batchA3 = inputShapeSize > kSecondLast ? inputShape[inputShapeSize - kThirdLast] : 1L;
    int64_t batchA2 = inputShapeSize > kThirdLast ? inputShape[inputShapeSize - kFourthLast] : 1L;
    int64_t batchA1 = inputShapeSize > kFourthLast ? inputShape[inputShapeSize - kFifthLast] : 1L;
    int64_t batchA0 = inputShapeSize > kFifthLast ? inputShape[inputShapeSize - kSixthLast] : 1L;
    return {batchA0, batchA1, batchA2, batchA3};
}

uint64_t GetBatchDimAll(const std::vector<int64_t>& inputShape)
{
    const std::vector<int64_t> batchDims = GetBatchDim(inputShape);
    int64_t result = 1L;
    for (int64_t d : batchDims) {
        result *= d;
    }
    return static_cast<uint64_t>(result);
}

bool IsBatchEqual(const std::vector<int64_t>& batchDimForX1, const std::vector<int64_t>& batchDimForX2)
{
    const size_t dimNumA = batchDimForX1.size();
    const size_t dimNumB = batchDimForX2.size();
    if (dimNumA != dimNumB) {
        return false;
    }
    for (size_t i = 0; i < dimNumA; i++) {
        if (batchDimForX1[i] != batchDimForX2[i]) {
            return false;
        }
    }
    return true;
}

bool CheckSocNeedBatchMatMulToMul91095(const PlatformInfo& platformInfo, const BatchMatMulToMulArgs& args,
                                       DataType x1Dtype, DataType x2Dtype, DataType outDtype,
                                       const std::vector<int64_t>& shapeX1, const std::vector<int64_t>& shapeX2)
{
    bool checkBf16Scene = (x1Dtype == DT_BF16) && (x2Dtype == DT_BF16) && (outDtype == DT_BF16);
    bool checkFp16Scene = (x1Dtype == DT_FLOAT16) && (x2Dtype == DT_FLOAT16) && (outDtype == DT_FLOAT16);
    bool checkFp32Scene = (x1Dtype == DT_FLOAT) && (x2Dtype == DT_FLOAT) && (outDtype == DT_FLOAT);
    if (!checkBf16Scene && !checkFp16Scene && !checkFp32Scene) {
        return false;
    }
    // now only iterbatch module not need convert to mul
    int64_t dtypeSize = GetDtypeSize(x1Dtype);
    int64_t c0 = kBlockByteSize / dtypeSize;
    bool batchEqual = IsBatchEqual(GetBatchDim(shapeX1), GetBatchDim(shapeX2));
    bool batchLargerThanAicnum = GetBatchDimAll(shapeX1) > platformInfo.soc_info.ai_core_cnt;
    uint64_t alignMValue = ops::CeilAlign(args.m, kBlockSize);
    uint64_t alignKaValue = args.adjX1 ? ops::CeilAlign(args.k, kBlockSize) : ops::CeilAlign(args.k, c0);
    uint64_t alignKbValue = args.adjX2 ? ops::CeilAlign(args.k, c0) : ops::CeilAlign(args.k, kBlockSize);
    uint64_t alignNValue = ops::CeilAlign(args.n, kBlockSize);
    bool lessThanL0a = alignMValue * alignKaValue * dtypeSize * kPingpong <= platformInfo.ai_core_spec.l0_a_size;
    bool lessThanL0b = alignKbValue * alignNValue * dtypeSize * kPingpong <= platformInfo.ai_core_spec.l0_b_size;
    bool lessThanL0c = alignMValue * alignNValue * kFloatSize * kPingpong <= platformInfo.ai_core_spec.l0_c_size;
    bool lessThanL1 = (alignMValue * alignKaValue + alignKbValue * alignNValue) * dtypeSize * kPingpong <=
                      platformInfo.ai_core_spec.l1_size;
    bool fitIterBatch = batchEqual && batchLargerThanAicnum && lessThanL0a && lessThanL0b && lessThanL0c && lessThanL1;
    uint64_t batchNum = GetBatchDimAll(shapeX1);
    bool fitBatchMatMulToMul = CheckShapeEqualToMul(platformInfo, args.m, args.n, batchNum, dtypeSize);
    return args.matmulFlag || (!args.matmulFlag && !(fitIterBatch || fitBatchMatMulToMul));
}

bool CheckSocNeedBatchMatMulToMul(const PlatformInfo& platformInfo, const BatchMatMulToMulArgs& args, DataType x1Dtype,
                                  DataType x2Dtype, DataType outDtype, const std::vector<int64_t>& shapeX1,
                                  const std::vector<int64_t>& shapeX2)
{
    auto mmadIter = platformInfo.ai_core_intrinsic_dtype_map.find("Intrinsic_mmad");
    bool supportS8S4Mmad = mmadIter != platformInfo.ai_core_intrinsic_dtype_map.end() &&
                           std::find(mmadIter->second.begin(), mmadIter->second.end(), "s8s4") !=
                               mmadIter->second.end();
    if (supportS8S4Mmad) {
        return CheckSocNeedBatchMatMulToMulSupportS8S4(x1Dtype, x2Dtype, outDtype);
    }
    const std::string soc = platformInfo.str_info.short_soc_version;
    if (soc == "Ascend910B" || soc == "Ascend910_93") {
        return CheckSocNeedBatchMatMulToMul910B(platformInfo, args, x1Dtype, x2Dtype, outDtype);
    }
    if (soc == "Ascend950") {
        return CheckSocNeedBatchMatMulToMul91095(platformInfo, args, x1Dtype, x2Dtype, outDtype, shapeX1, shapeX2);
    }
    // default
    return true;
}

bool CheckProduct(const std::vector<int64_t>& shape, std::size_t len)
{
    if (len > shape.size()) {
        return false;
    }
    int64_t product = 1;
    for (std::size_t i = 0; i < len; i++) {
        if (shape[i] > 0) {
            if (product > (INT64_MAX / shape[i])) {
                return false;
            } else {
                product *= shape[i];
            }
        }
    }
    return true;
}

bool IsMatchReduceScenario(const GNode& bmmNode)
{
    // BatchMatMulV2 --> ReduceSumD --> Output
    // BatchMatMulV2 --> Cast32 --> ReduceSumD --> Output
    if (bmmNode.GetInputsSize() != 2) {
        return false;
    }
    auto outNodes = bmmNode.GetOutDataNodesAndPortIndexs(0);
    if (outNodes.size() != 1) {
        return false;
    }
    auto nextNode = outNodes[0].first;
    AscendString nextType;
    if (nextNode->GetType(nextType) != GRAPH_SUCCESS) {
        return false;
    }
    // Scenario1: direct ReduceSumD
    if (nextType == kOpTypeReduceSumD) {
        return true;
    }
    // Scenario2: Cast(fp16->fp32) -> ReduceSumD
    if (nextType != kOpTypeCast) {
        return false;
    }
    TensorDesc bmmOutDesc;
    TensorDesc castOutDesc;
    if (bmmNode.GetOutputDesc(0, bmmOutDesc) != GRAPH_SUCCESS ||
        nextNode->GetOutputDesc(0, castOutDesc) != GRAPH_SUCCESS) {
        return false;
    }
    if (bmmOutDesc.GetDataType() != DT_FLOAT16 || castOutDesc.GetDataType() != DT_FLOAT) {
        return false;
    }
    auto nextOutNodes = nextNode->GetOutDataNodesAndPortIndexs(0);
    if (nextOutNodes.size() != 1) {
        return false;
    }
    AscendString nextNextType;
    if (nextOutNodes[0].first->GetType(nextNextType) != GRAPH_SUCCESS) {
        return false;
    }
    return nextNextType == kOpTypeReduceSumD;
}

bool CheckNeedChange(const GNode& bmmNode, const std::vector<int64_t>& shapeX, const std::vector<int64_t>& shapeY,
                     const std::vector<int64_t>& productShapeX, const std::vector<int64_t>& productShapeY)
{
    auto xDims = shapeX.size();
    auto yDims = shapeY.size();
    auto productXDims = productShapeX.size();
    auto productYDims = productShapeY.size();
    if (xDims == 0 || yDims == 0 || productXDims == 0 || productYDims == 0) {
        return false;
    }

    if (xDims == kLeftShapeDim && yDims == kRightShapeDim && shapeX[kBatchIndex] > 1) {
        TensorDesc x0Desc;
        TensorDesc x1Desc;
        if (bmmNode.GetInputDesc(0, x0Desc) != GRAPH_SUCCESS || bmmNode.GetInputDesc(1, x1Desc) != GRAPH_SUCCESS) {
            return false;
        }
        if (IsUnknownShape(x0Desc.GetShape()) || IsUnknownShape(x1Desc.GetShape())) {
            return false;
        }
        return CheckProduct(productShapeX, productShapeX.size()) && CheckProduct(productShapeY, productShapeY.size()) &&
               IsMatchReduceScenario(bmmNode);
    }
    return false;
}

bool BatchMatMulReduceFusionCheck(const GNode& bmmNode, const BatchMatMulToMulArgs& args)
{
    TensorDesc input0desc;
    TensorDesc input1desc;
    if (bmmNode.GetInputDesc(0, input0desc) != GRAPH_SUCCESS || bmmNode.GetInputDesc(1, input1desc) != GRAPH_SUCCESS) {
        return false;
    }
    auto x1Shape = input0desc.GetOriginShape().GetDims();
    auto x2Shape = input1desc.GetOriginShape().GetDims();

    bool shapeValid = ((x1Shape.size() == kLeftShapeDim) && (x2Shape.size() == kRightShapeDim) &&
                       (x1Shape[kBatchIndex] > 1));
    if (!shapeValid) {
        return false;
    }

    std::vector<int64_t> productX1Shape;
    std::vector<int64_t> productX2Shape;
    if (args.adjX2) {
        // b,k,m adj_x2=true --> b*k,m
        productX1Shape.assign({x1Shape[kBatchIndex], x1Shape[kSecondIndex]});
    } else {
        // b,m,k adj_x2=false --> b*k,m
        productX1Shape.assign({x1Shape[kBatchIndex], x1Shape[kThirdIndex]});
    }
    if (args.adjX1) {
        // b,n,k adj_x1=true --> b*k,n
        productX2Shape.assign({x2Shape[kBatchIndex], x2Shape[kThirdIndex]});
    } else {
        // b,k,n adj_x1=false --> b*k,n
        productX2Shape.assign({x2Shape[kBatchIndex], x2Shape[kSecondIndex]});
    }

    if (!CheckNeedChange(bmmNode, x1Shape, x2Shape, productX1Shape, productX2Shape)) {
        return false;
    }
    return true;
}

bool CheckPlatformSupport(const TensorDesc& inputDescX1, const TensorDesc& inputDescX2, const TensorDesc& outputDesc,
                          const GNode& bmmNode)
{
    PlatformInfo platformInfo;
    OptionalInfo optionalInfo;
    FUSION_PASS_CHECK(
        PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platformInfo, optionalInfo) != SUCCESS,
        OPS_LOG_W(kPassName, "Failed to get platform info."), return false);
    FUSION_PASS_CHECK(platformInfo.ai_core_intrinsic_dtype_map.find("Intrinsic_fix_pipe_l0c2out") ==
                          platformInfo.ai_core_intrinsic_dtype_map.end(),
                      OPS_LOG_W(kPassName, "Platform does not support Intrinsic_fix_pipe_l0c2out, skip fusion."),
                      return false);
    FUSION_PASS_CHECK(IsUnknownShape(inputDescX1.GetShape()) || IsUnknownShape(inputDescX2.GetShape()) ||
                          IsUnknownShape(outputDesc.GetShape()),
                      OPS_LOG_D(kPassName, "Not support dynamic shape."), return false);
    FUSION_PASS_CHECK(
        bmmNode.GetInputsSize() > kInputSizeWithoutBias,
        OPS_LOG_D(kPassName, "Input node of bmm_node size is [%zu], which not equal to 2.", bmmNode.GetInputsSize()),
        return false);
    return true;
}

bool CheckSocCondition(const PlatformInfo& platformInfo, const BatchMatMulToMulArgs& args,
                       const TensorDesc& inputDescX1, const TensorDesc& inputDescX2, const TensorDesc& outputDesc)
{
    std::vector<int64_t> inputShapeX1 = inputDescX1.GetShape().GetDims();
    std::vector<int64_t> inputShapeX2 = inputDescX2.GetShape().GetDims();
    FUSION_PASS_CHECK(
        !CheckSocNeedBatchMatMulToMul(platformInfo, args, inputDescX1.GetDataType(), inputDescX2.GetDataType(),
                                      outputDesc.GetDataType(), inputShapeX1, inputShapeX2),
        OPS_LOG_I(kPassName, "Some conditions are not satisfied to convert bmm to mul in this socversion."),
        return false);
    FUSION_PASS_CHECK(args.k != 1, OPS_LOG_D(kPassName, "K != 1, no need to convert bmm to mul."), return false);
    return true;
}

bool NeedFusion(const GNode& bmmNode, BatchMatMulToMulArgs& args)
{
    TensorDesc inputDescX1;
    TensorDesc inputDescX2;
    TensorDesc outputDesc;
    FUSION_PASS_CHECK(bmmNode.GetInputDesc(kX1InputIdx, inputDescX1) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "Get x1 input desc failed."), return false);
    FUSION_PASS_CHECK(bmmNode.GetInputDesc(kX2InputIdx, inputDescX2) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "Get x2 input desc failed."), return false);
    FUSION_PASS_CHECK(bmmNode.GetOutputDesc(0, outputDesc) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "Get output desc failed."), return false);
    args.matmulFlag = IsMatMulType(bmmNode);
    const char* transStrX1 = args.matmulFlag ? "transpose_x1" : "adj_x1";
    const char* transStrX2 = args.matmulFlag ? "transpose_x2" : "adj_x2";
    FUSION_PASS_CHECK(bmmNode.GetAttr(transStrX1, args.adjX1) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "Get %s attr failed.", transStrX1), return false);
    FUSION_PASS_CHECK(bmmNode.GetAttr(transStrX2, args.adjX2) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "Get %s attr failed.", transStrX2), return false);
    if (!CheckPlatformSupport(inputDescX1, inputDescX2, outputDesc, bmmNode)) {
        return false;
    }
    PlatformInfo platformInfo;
    OptionalInfo optionalInfo;
    PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platformInfo, optionalInfo);
    std::vector<int64_t> inputShapeX1 = inputDescX1.GetShape().GetDims();
    std::vector<int64_t> inputShapeX2 = inputDescX2.GetShape().GetDims();
    int64_t inputShapeSizeX1 = static_cast<int64_t>(inputShapeX1.size());
    int64_t inputShapeSizeX2 = static_cast<int64_t>(inputShapeX2.size());
    args.m = args.adjX1 ? inputShapeX1[inputShapeSizeX1 - kLastOne] : inputShapeX1[inputShapeSizeX1 - kSecondLast];
    args.k = args.adjX1 ? inputShapeX1[inputShapeSizeX1 - kSecondLast] : inputShapeX1[inputShapeSizeX1 - kLastOne];
    args.n = args.adjX2 ? inputShapeX2[inputShapeSizeX2 - kSecondLast] : inputShapeX2[inputShapeSizeX2 - kLastOne];
    if (!CheckSocCondition(platformInfo, args, inputDescX1, inputDescX2, outputDesc)) {
        return false;
    }
    if (!args.matmulFlag) {
        args.batchX1.assign(inputShapeX1.begin(), inputShapeX1.end() - kSecondLast);
        args.batchX2.assign(inputShapeX2.begin(), inputShapeX2.end() - kSecondLast);
    }
    return true;
}

es::EsTensorHolder CreateReshapeNode(es::EsGraphBuilder& builder, const es::EsTensorHolder& input,
                                     const std::vector<int64_t>& shape, const TensorDesc& inputDesc)
{
    auto shapeConst = builder.CreateVector(shape);
    auto reshapeOutput = es::Reshape(input, shapeConst);
    GNode reshapeNode = *reshapeOutput.GetProducer();
    FUSION_PASS_CHECK(reshapeNode.UpdateInputDesc(0, inputDesc) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "Update reshape input desc failed."), return EsTensorHolder());
    TensorDesc outDesc = inputDesc;
    outDesc.SetShape(ge::Shape(shape));
    outDesc.SetOriginShape(ge::Shape(shape));
    FUSION_PASS_CHECK(reshapeNode.UpdateOutputDesc(0, outDesc) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "Update reshape output desc failed."), return EsTensorHolder());
    return reshapeOutput;
}

es::EsTensorHolder CreateMulNode(es::EsGraphBuilder& builder, const es::EsTensorHolder& x1,
                                 const es::EsTensorHolder& x2, const TensorDesc& x1Desc, const TensorDesc& x2Desc,
                                 const TensorDesc& outputDesc, const std::string& name)
{
    auto* graph = builder.GetCGraphBuilder()->GetGraph();
    auto mulNode = CompliantNodeBuilder(graph)
                       .OpType("Mul")
                       .Name(name.c_str())
                       .IrDefInputs({
                           {"x1", CompliantNodeBuilder::kEsIrInputRequired, ""},
                           {"x2", CompliantNodeBuilder::kEsIrInputRequired, ""},
                       })
                       .IrDefOutputs({{"y", CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                       .Build();
    FUSION_PASS_CHECK(
        AddEdgeAndUpdatePeerDesc(*graph, *x1.GetProducer(), x1.GetProducerOutIndex(), mulNode, 0) != GRAPH_SUCCESS,
        OPS_LOG_E(kPassName, "AddEdge for Mul input x1 failed."), return EsTensorHolder());
    FUSION_PASS_CHECK(
        AddEdgeAndUpdatePeerDesc(*graph, *x2.GetProducer(), x2.GetProducerOutIndex(), mulNode, 1) != GRAPH_SUCCESS,
        OPS_LOG_E(kPassName, "AddEdge for Mul input x2 failed."), return EsTensorHolder());
    FUSION_PASS_CHECK(mulNode.UpdateInputDesc(0, x1Desc) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "Update Mul input x1 desc failed."), return EsTensorHolder());
    FUSION_PASS_CHECK(mulNode.UpdateInputDesc(1, x2Desc) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "Update Mul input x2 desc failed."), return EsTensorHolder());
    FUSION_PASS_CHECK(mulNode.UpdateOutputDesc(0, outputDesc) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "Update Mul output desc failed."), return EsTensorHolder());
    auto* yHolder = builder.GetCGraphBuilder()->GetTensorHolderFromNode(mulNode, 0);
    return EsTensorHolder(yHolder);
}

} // namespace

std::vector<PatternUniqPtr> BatchMatMul2MulFusionPass::Patterns()
{
    std::vector<PatternUniqPtr> patternGraphs;
    // MatMul/MatMulV2 pattern (2 inputs)
    auto matMulPatterns = BuildMatMulPatterns("pattern");
    auto matMulV2Patterns = BuildMatMulV2Patterns("pattern");
    // BatchMatMul/BatchMatMulV2 pattern (2 inputs)
    auto batchMatMulPatterns = BuildBatchMatMulPatterns("pattern");
    auto batchMatMulV2Patterns = BuildBatchMatMulV2Patterns("pattern");
    patternGraphs.insert(patternGraphs.end(), std::make_move_iterator(matMulPatterns.begin()),
                         std::make_move_iterator(matMulPatterns.end()));
    patternGraphs.insert(patternGraphs.end(), std::make_move_iterator(matMulV2Patterns.begin()),
                         std::make_move_iterator(matMulV2Patterns.end()));
    patternGraphs.insert(patternGraphs.end(), std::make_move_iterator(batchMatMulPatterns.begin()),
                         std::make_move_iterator(batchMatMulPatterns.end()));
    patternGraphs.insert(patternGraphs.end(), std::make_move_iterator(batchMatMulV2Patterns.begin()),
                         std::make_move_iterator(batchMatMulV2Patterns.end()));
    return patternGraphs;
}

bool BatchMatMul2MulFusionPass::MeetRequirements(const std::unique_ptr<MatchResult>& matchResult)
{
    OPS_LOG_D(kPassName, "Begin to do BatchMatMul2MulFusionPass MeetRequirements.");

    FUSION_PASS_CHECK(GetGeCompilerVersionNum() < kGeCompilerVersion900,
                      OPS_LOG_D(kPassName, "GE runtime < 9.0.0, skip fusion."), return false);

    NodeIo nodeIo;
    FUSION_PASS_CHECK(matchResult->GetCapturedTensor(kCaptureTensorIdx, nodeIo) != SUCCESS,
                      OPS_LOG_E(kPassName, "Failed to get captured tensor."), return false);
    GNode matchedNode = nodeIo.node;

    BatchMatMulToMulArgs args;
    if (!NeedFusion(matchedNode, args)) {
        return false;
    }

    // BatchMatMulReduceFusion fusion b,k,m --> b*k,m, in those cases, BatchMatMul2MulFusionPass must not be trigged.
    FUSION_PASS_CHECK(!args.matmulFlag && BatchMatMulReduceFusionCheck(matchedNode, args),
                      OPS_LOG_W(kPassName, "Parameter[bmm_node] should not be changed."), return false);
    return true;
}

struct ReplacementInfo {
    BatchMatMulToMulArgs args;
    TensorDesc inputDescX1;
    TensorDesc inputDescX2;
    TensorDesc outputDesc;
    std::vector<int64_t> inputShapeX1;
    std::vector<int64_t> inputShapeX2;
    std::string baseName;
};

ReplacementInfo PrepareReplacementInfo(const GNode& matchedNode)
{
    ReplacementInfo info;

    info.args.matmulFlag = IsMatMulType(matchedNode);
    const char* transStrX1 = info.args.matmulFlag ? "transpose_x1" : "adj_x1";
    const char* transStrX2 = info.args.matmulFlag ? "transpose_x2" : "adj_x2";
    matchedNode.GetAttr(transStrX1, info.args.adjX1);
    matchedNode.GetAttr(transStrX2, info.args.adjX2);

    matchedNode.GetInputDesc(kX1InputIdx, info.inputDescX1);
    matchedNode.GetInputDesc(kX2InputIdx, info.inputDescX2);
    matchedNode.GetOutputDesc(0, info.outputDesc);

    info.inputShapeX1 = info.inputDescX1.GetShape().GetDims();
    info.inputShapeX2 = info.inputDescX2.GetShape().GetDims();
    int64_t inputShapeSizeX1 = static_cast<int64_t>(info.inputShapeX1.size());
    int64_t inputShapeSizeX2 = static_cast<int64_t>(info.inputShapeX2.size());
    info.args.m = info.args.adjX1 ? info.inputShapeX1[inputShapeSizeX1 - kLastOne] :
                                    info.inputShapeX1[inputShapeSizeX1 - kSecondLast];
    info.args.k = info.args.adjX1 ? info.inputShapeX1[inputShapeSizeX1 - kSecondLast] :
                                    info.inputShapeX1[inputShapeSizeX1 - kLastOne];
    info.args.n = info.args.adjX2 ? info.inputShapeX2[inputShapeSizeX2 - kSecondLast] :
                                    info.inputShapeX2[inputShapeSizeX2 - kLastOne];
    if (!info.args.matmulFlag) {
        info.args.batchX1.assign(info.inputShapeX1.begin(), info.inputShapeX1.end() - kSecondLast);
        info.args.batchX2.assign(info.inputShapeX2.begin(), info.inputShapeX2.end() - kSecondLast);
    }

    AscendString matchedNodeName;
    matchedNode.GetName(matchedNodeName);
    info.baseName = matchedNodeName.GetString();

    return info;
}

std::unique_ptr<Graph> BuildReplacementGraph(const ReplacementInfo& info)
{
    auto replaceGraphBuilder = es::EsGraphBuilder("replacement");
    auto rX1 = replaceGraphBuilder.CreateInput(0, "x1", info.inputDescX1.GetDataType(), info.inputDescX1.GetFormat(),
                                               info.inputShapeX1);
    auto rX2 = replaceGraphBuilder.CreateInput(1, "x2", info.inputDescX2.GetDataType(), info.inputDescX2.GetFormat(),
                                               info.inputShapeX2);
    FUSION_PASS_CHECK(rX1.GetProducer()->UpdateOutputDesc(0, info.inputDescX1) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "Update input x1 output desc failed."), return nullptr);
    FUSION_PASS_CHECK(rX2.GetProducer()->UpdateOutputDesc(0, info.inputDescX2) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "Update input x2 output desc failed."), return nullptr);
    es::EsTensorHolder mulInputX1 = rX1;
    TensorDesc mulInputDescX1 = info.inputDescX1;
    if (info.args.adjX1) {
        std::vector<int64_t> newShapeX1;
        if (!info.args.matmulFlag) {
            newShapeX1.assign(info.args.batchX1.begin(), info.args.batchX1.end());
        }
        newShapeX1.push_back(info.args.m);
        newShapeX1.push_back(1);
        mulInputX1 = CreateReshapeNode(replaceGraphBuilder, rX1, newShapeX1, info.inputDescX1);
        FUSION_PASS_CHECK(mulInputX1.GetProducer() == nullptr,
                          OPS_LOG_E(kPassName, "Create reshape node for x1 failed."), return nullptr);
        mulInputDescX1.SetShape(ge::Shape(newShapeX1));
        mulInputDescX1.SetOriginShape(ge::Shape(newShapeX1));
    }
    es::EsTensorHolder mulInputX2 = rX2;
    TensorDesc mulInputDescX2 = info.inputDescX2;
    if (info.args.adjX2) {
        std::vector<int64_t> newShapeX2;
        if (!info.args.matmulFlag) {
            newShapeX2.assign(info.args.batchX2.begin(), info.args.batchX2.end());
        }
        newShapeX2.push_back(1);
        newShapeX2.push_back(info.args.n);
        mulInputX2 = CreateReshapeNode(replaceGraphBuilder, rX2, newShapeX2, info.inputDescX2);
        FUSION_PASS_CHECK(mulInputX2.GetProducer() == nullptr,
                          OPS_LOG_E(kPassName, "Create reshape node for x2 failed."), return nullptr);
        mulInputDescX2.SetShape(ge::Shape(newShapeX2));
        mulInputDescX2.SetOriginShape(ge::Shape(newShapeX2));
    }
    auto rY = CreateMulNode(replaceGraphBuilder, mulInputX1, mulInputX2, mulInputDescX1, mulInputDescX2,
                            info.outputDesc, info.baseName + "_to_mul");
    FUSION_PASS_CHECK(rY.GetProducer() == nullptr, OPS_LOG_E(kPassName, "Create mul node failed."), return nullptr);
    auto result = replaceGraphBuilder.BuildAndReset({rY});
    FUSION_PASS_CHECK(result == nullptr, OPS_LOG_E(kPassName, "Build replacement graph failed."), return nullptr);
    OPS_LOG_D(kPassName, "BatchMatMul to Mul fusion success.");
    return result;
}

std::unique_ptr<Graph> BatchMatMul2MulFusionPass::Replacement(const std::unique_ptr<MatchResult>& matchResult)
{
    NodeIo nodeIo;
    FUSION_PASS_CHECK(matchResult->GetCapturedTensor(kCaptureTensorIdx, nodeIo) != SUCCESS,
                      OPS_LOG_E(kPassName, "Failed to get captured tensor in Replacement."), return nullptr);
    GNode matchedNode = nodeIo.node;

    // MeetRequirements 已校验过节点合法性，此处只获取信息并构建替换图
    auto info = PrepareReplacementInfo(matchedNode);
    return BuildReplacementGraph(info);
}

REG_FUSION_PASS(BatchMatMul2MulFusionPass).Stage(CustomPassStage::kCompatibleInherited);

} // namespace ops

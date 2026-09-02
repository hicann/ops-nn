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
 * \file matmul_to_gemm_op_fusion_pass.cpp
 * \brief matmul+cast+AssignAdd to GemmOp fusion pass
 *
 *                    x1    x2
 *                     \    /                 x1 x2   input0   1    1
 *      input0         matmul                  \  \     |     /    /
 *         \             /                      a  b    c  alpha beta
 *          \        (cast)         ===>        \  \  |  /     /
 *           \         /                              GemmOp
 *            AssignAdd                                 |
 *               |                                    input0
 *             input0
 *
 * - Milan (!supportL12btBf16): MatMul(+Cast)+AssignAdd → GemmV2 + alpha/beta Const
 * - David  ( supportL12btBf16): MatMul(+Cast)+AssignAdd → GemmV3 (alpha/beta as attrs)
 */

#include "matmul_to_gemm_op_fusion_pass.h"

#include <cstdint>
#include <string>
#include <vector>

#include "es_math_ops.h"
#include "platform/platform_info.h"
#include "common/inc/error_util.h"
#include "common/op_graph/fusion_pass/matmul_fusion_utils_pass.h"
#include "common/op_host/math_util_nn.h"
#include "ge/es_graph_builder.h"
#include "ge/compliant_node_builder.h"

using namespace ge;
using namespace ge::es;
using namespace ge::fusion;
using namespace fe;

namespace ops {
namespace {

constexpr char kPassName[] = "MatmulToGemmOpFusionPass";
constexpr int32_t kGeCompilerVersion900 = 90000000;

// Shape / index constants (from original code)
constexpr int64_t kShapeDim = 2;
constexpr int64_t kInnerAxis = 1;
constexpr int64_t kOuterAxis = 2;
constexpr int64_t kInputCIndex = 2;          // GemmV3 output reuse input index
constexpr int64_t kMatMulOutputIdx = 0;      // MatMul output index (for CaptureTensor)
constexpr int64_t kAssignAddOutputIdx = 0;   // AssignAdd output index (for CaptureTensor)
constexpr int64_t kCaptureMatMulSlot = 0;    // Capture slot for MatMul (for GetCapturedTensor)
constexpr int64_t kCaptureAssignAddSlot = 1; // Capture slot for AssignAdd (for GetCapturedTensor)

// Block / tiling constants (from original code)
constexpr uint64_t kBasicBlockSize256 = 256UL;
constexpr uint64_t kNumHalf = 2UL;
constexpr uint64_t kBasicBlockK256Byte = 256UL;
constexpr uint64_t kFp32Hf32DtypeSize = 4UL;
constexpr uint64_t kBasicBlockSize32 = 32UL;
constexpr uint64_t kCacheLine = 512UL;
constexpr uint64_t kNumTwo = 2UL;

constexpr uint64_t kHf32EnableBit = 0x40UL;

// Op type strings
constexpr char kOpTypeCast[] = "Cast";
constexpr char kOpTypeAssignAdd[] = "AssignAdd";
constexpr char kOpTypeGemmV2[] = "GemmV2";
constexpr char kOpTypeGemmV3[] = "GemmV3";
constexpr char kOpTypeMatMulV3[] = "MatMulV3";

// White-list shapes for Milan (supportL12btBf16 == false)
const std::vector<std::string> kWhiteList = {
    "4096_12288_4096_6144_1_0", "4096_1536_4096_12288_1_0",  "4096_16640_4096_12288_1_0", "4096_4608_4096_12288_1_0",
    "4096_6144_4096_12288_1_0", "4096_10240_5120_10240_0_1", "4096_3840_3840_10240_0_0",  "4096_3840_4096_10240_1_0",
    "4096_10240_4096_5120_1_0", "4096_5120_4096_10240_1_0",  "4096_1280_4096_10240_1_0"};

// Supported dtypes for MatMul output / AssignAdd input1
const std::vector<DataType> kDtypeSupport = {DT_FLOAT16, DT_BF16, DT_FLOAT};

// Struct to hold MatMul args (mirrors original GemmV3Args)
struct GemmOpArgs {
    bool isATrans = false;
    bool isBTrans = false;
    bool isHf32 = false;
    uint64_t mValue = 0UL;
    uint64_t kValue = 0UL;
    uint64_t nValue = 0UL;
    uint64_t aDtypeSize = 1UL;
};

// ---------------------------------------------------------------------------
// Pattern helpers
// ---------------------------------------------------------------------------

/// Create an AssignAdd node via CompliantNodeBuilder (es::AssignAdd not available)
EsTensorHolder CreateAssignAddNode(EsGraphBuilder& graphBuilder, const EsTensorHolder& ref, const EsTensorHolder& value)
{
    auto* graph = graphBuilder.GetCGraphBuilder()->GetGraph();
    auto node = CompliantNodeBuilder(graph)
                    .OpType(kOpTypeAssignAdd)
                    .Name(kOpTypeAssignAdd)
                    .IrDefInputs({{"ref", CompliantNodeBuilder::kEsIrInputRequired, ""},
                                  {"value", CompliantNodeBuilder::kEsIrInputRequired, ""}})
                    .IrDefOutputs({{"ref", CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                    .IrDefAttrs({{"use_locking", CompliantNodeBuilder::kEsAttrOptional, "Bool", CreateFrom(false)}})
                    .Build();
    AddEdgeAndUpdatePeerDesc(*graph, *ref.GetProducer(), ref.GetProducerOutIndex(), node, 0);
    AddEdgeAndUpdatePeerDesc(*graph, *value.GetProducer(), value.GetProducerOutIndex(), node, 1);
    auto* yHolder = graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(node, 0);
    return EsTensorHolder(yHolder);
}

/// Build a pattern: {opType} -> Cast -> AssignAdd  (with Cast)
PatternUniqPtr BuildPatternWithCast(const std::string& patternName, const char* opType)
{
    auto graphBuilder = EsGraphBuilder(patternName.c_str());
    auto x1 = graphBuilder.CreateInput(0);
    auto x2 = graphBuilder.CreateInput(1);
    auto input0 = graphBuilder.CreateInput(2);

    auto matmulOutput = CreateMatMulLikeNode(graphBuilder, opType, x1, x2, nullptr, nullptr);
    auto castOutput = es::Cast(matmulOutput, static_cast<int64_t>(DT_FLOAT));
    auto assignAddOutput = CreateAssignAddNode(graphBuilder, input0, castOutput);

    auto graph = graphBuilder.BuildAndReset({assignAddOutput});
    auto pattern = std::make_unique<Pattern>(std::move(*graph));
    pattern->CaptureTensor({*matmulOutput.GetProducer(), kMatMulOutputIdx});
    pattern->CaptureTensor({*assignAddOutput.GetProducer(), kAssignAddOutputIdx});
    return pattern;
}

/// Build a pattern: {opType} -> AssignAdd  (without Cast)
PatternUniqPtr BuildPatternWithoutCast(const std::string& patternName, const char* opType)
{
    auto graphBuilder = EsGraphBuilder(patternName.c_str());
    auto x1 = graphBuilder.CreateInput(0);
    auto x2 = graphBuilder.CreateInput(1);
    auto input0 = graphBuilder.CreateInput(2);

    auto matmulOutput = CreateMatMulLikeNode(graphBuilder, opType, x1, x2, nullptr, nullptr);
    auto assignAddOutput = CreateAssignAddNode(graphBuilder, input0, matmulOutput);

    auto graph = graphBuilder.BuildAndReset({assignAddOutput});
    auto pattern = std::make_unique<Pattern>(std::move(*graph));
    pattern->CaptureTensor({*matmulOutput.GetProducer(), kMatMulOutputIdx});
    pattern->CaptureTensor({*assignAddOutput.GetProducer(), kAssignAddOutputIdx});
    return pattern;
}

// ---------------------------------------------------------------------------
// Validation helpers (mirror original ScenarioCheck logic)
// ---------------------------------------------------------------------------

bool CheckPlatformIntrinsics(const PlatformInfo& platformInfo)
{
    const auto& intrinsicMap = platformInfo.ai_core_intrinsic_dtype_map;
    bool supportL0c2out = intrinsicMap.find("Intrinsic_fix_pipe_l0c2out") != intrinsicMap.end();
    bool supportOut2l1Nd2nz = intrinsicMap.find("Intrinsic_data_move_out2l1_nd2nz") != intrinsicMap.end();
    bool supportL0c2ub = intrinsicMap.find("Intrinsic_data_move_l0c2ub") != intrinsicMap.end();
    bool supportFixpipeL0c2ub = intrinsicMap.find("Intrinsic_fix_pipe_l0c2ub") != intrinsicMap.end();
    // Require: l0c2out && out2l1nd2nz && !l0c2ub && !fixpipeL0c2ub
    return supportL0c2out && supportOut2l1Nd2nz && !supportL0c2ub && !supportFixpipeL0c2ub;
}

bool IsUnknownShape(const ge::Shape& shape)
{
    const auto dims = shape.GetDims();
    return std::any_of(dims.begin(), dims.end(), [](const int64_t dim) {
        return dim == ge::UNKNOWN_DIM || dim == ge::UNKNOWN_DIM_NUM || dim < 0;
    });
}

bool CheckDtypeSupport(const TensorDesc& tensor)
{
    if (tensor.GetShape().GetDims().size() != static_cast<size_t>(kShapeDim)) {
        return false;
    }
    auto iter = std::find(kDtypeSupport.begin(), kDtypeSupport.end(), tensor.GetDataType());
    return iter != kDtypeSupport.end();
}

bool GetNodeMatMulAttrs(const GNode& nodeMatmul, bool supportL12btBf16, GemmOpArgs& args)
{
    FUSION_PASS_CHECK(nodeMatmul.GetAttr("transpose_x1", args.isATrans) != GRAPH_SUCCESS,
                      OPS_LOG_I(kPassName, "Get attr transpose_x1 failed."), return false);
    FUSION_PASS_CHECK(nodeMatmul.GetAttr("transpose_x2", args.isBTrans) != GRAPH_SUCCESS,
                      OPS_LOG_I(kPassName, "Get attr transpose_x2 failed."), return false);
    if (supportL12btBf16) {
        int64_t opImplMode = 0;
        FUSION_PASS_CHECK(nodeMatmul.GetAttr("_op_impl_mode_enum", opImplMode) != SUCCESS,
                          OPS_LOG_E(kPassName, "Get _op_impl_mode_enum attr failed."), return false);
        args.isHf32 = (static_cast<uint64_t>(opImplMode) & kHf32EnableBit) != 0UL;
    }
    return true;
}

bool CheckShapeInWhiteList(const std::vector<int64_t>& aShape, const std::vector<int64_t>& bShape, bool transA,
                           bool transB)
{
    uint64_t aKDim = transA ? aShape[kShapeDim - kOuterAxis] : aShape[kShapeDim - kInnerAxis];
    uint64_t bKDim = transB ? bShape[kShapeDim - kInnerAxis] : bShape[kShapeDim - kOuterAxis];
    FUSION_PASS_CHECK(aKDim != bKDim, OPS_LOG_I(kPassName, "mm input aK != bK can not support GemmOp."), return false);

    std::stringstream ss;
    for (size_t i = 0; i < aShape.size(); i++) {
        ss << aShape[i] << "_";
    }
    for (size_t i = 0; i < bShape.size(); i++) {
        ss << bShape[i] << "_";
    }
    ss << static_cast<int64_t>(transA) << "_" << static_cast<int64_t>(transB);
    std::string shapeStr = ss.str();

    auto iter = std::find(kWhiteList.begin(), kWhiteList.end(), shapeStr);
    if (iter != kWhiteList.end()) {
        OPS_LOG_D(kPassName, "mm shape info to string is: %s. Hit gemmOp case channel.", shapeStr.c_str());
        return true;
    }
    return false;
}

bool CheckStreamKSKTiling(const GemmOpArgs& args, uint32_t aiCoreCnt)
{
    uint64_t kAlign = ops::CeilAlign(args.kValue, kBasicBlockSize256);
    if (kAlign < aiCoreCnt * kNumHalf * kBasicBlockK256Byte / args.aDtypeSize) {
        return false;
    }

    uint64_t alignValue = kBasicBlockSize256;
    if (args.aDtypeSize == kFp32Hf32DtypeSize && !args.isHf32) {
        alignValue = kBasicBlockSize32;
    }
    uint64_t mCnt = ops::CeilDiv(args.mValue, alignValue);
    uint64_t nCnt = ops::CeilDiv(args.nValue, alignValue);
    return !(mCnt * nCnt > aiCoreCnt / kNumHalf);
}

bool CheckStreamKDPSKTiling(const GemmOpArgs& args, uint32_t aiCoreCnt)
{
    if (args.mValue % kBasicBlockSize256 != 0UL || args.nValue % kBasicBlockSize256 != 0UL ||
        args.kValue < aiCoreCnt * kBasicBlockK256Byte / args.aDtypeSize ||
        (args.aDtypeSize == kFp32Hf32DtypeSize && !args.isHf32)) {
        return false;
    }
    uint64_t mCnt = ops::CeilDiv(args.mValue, kBasicBlockSize256);
    uint64_t nCnt = ops::CeilDiv(args.nValue, kBasicBlockSize256);
    uint64_t totalMNCnt = mCnt * nCnt;
    if ((totalMNCnt < aiCoreCnt) || (totalMNCnt % aiCoreCnt == 0UL) || (totalMNCnt % aiCoreCnt > aiCoreCnt / kNumTwo)) {
        return false;
    }
    return true;
}

bool CheckShapeValid(const GemmOpArgs& args, uint32_t aiCoreCnt)
{
    bool notSkTiling = !CheckStreamKSKTiling(args, aiCoreCnt) && !CheckStreamKDPSKTiling(args, aiCoreCnt);
    bool mnValid = args.mValue >= kCacheLine && args.nValue >= kCacheLine;
    bool kValid = args.kValue > kBasicBlockSize256;
    bool shapeAswt = notSkTiling && mnValid && kValid;
    FUSION_PASS_CHECK(!shapeAswt, OPS_LOG_I(kPassName, "Not support this shape in gemmV3."), return false);
    return true;
}

bool ValidateMatMulDesc(const GNode& nodeMatmul, bool supportL12btBf16, TensorDesc& aTensor, TensorDesc& bTensor,
                        TensorDesc& mmOutTensor)
{
    FUSION_PASS_CHECK(
        nodeMatmul.GetInputsSize() != static_cast<size_t>(kBaseNodeNum),
        OPS_LOG_I(kPassName, "nodeMatmul should only have 2 input, actual %zu.", nodeMatmul.GetInputsSize()),
        return false);

    FUSION_PASS_CHECK(nodeMatmul.GetInputDesc(0, aTensor) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "Get matmul input 0 desc failed."), return false);
    FUSION_PASS_CHECK(nodeMatmul.GetInputDesc(1, bTensor) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "Get matmul input 1 desc failed."), return false);
    FUSION_PASS_CHECK(nodeMatmul.GetOutputDesc(0, mmOutTensor) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "Get matmul output 0 desc failed."), return false);

    auto outNodes = nodeMatmul.GetOutDataNodesAndPortIndexs(0);
    FUSION_PASS_CHECK(outNodes.size() != 1,
                      OPS_LOG_I(kPassName, "nodeMatmul should only have 1 output, actual %zu.", outNodes.size()),
                      return false);

    bool isDynamic = IsUnknownShape(aTensor.GetShape()) || IsUnknownShape(bTensor.GetShape()) ||
                     IsUnknownShape(mmOutTensor.GetShape());
    bool notSupportFormat = (aTensor.GetFormat() != FORMAT_ND) || (bTensor.GetFormat() != FORMAT_ND) ||
                            (mmOutTensor.GetFormat() != FORMAT_ND);
    bool inputDtypeFlag = (aTensor.GetDataType() != bTensor.GetDataType()) ||
                          ((aTensor.GetDataType() != DT_FLOAT16) && (aTensor.GetDataType() != DT_BF16) &&
                           (!supportL12btBf16 || aTensor.GetDataType() != DT_FLOAT));
    bool outDtypeFlag = !CheckDtypeSupport(mmOutTensor);
    FUSION_PASS_CHECK(isDynamic || notSupportFormat || inputDtypeFlag || outDtypeFlag,
                      OPS_LOG_I(kPassName,
                                "mm input info: isDynamic: %d, notSupportFormat: %d, inputDtypeFlag: %d, "
                                "outDtypeFlag: %d. can not support gemmOp.",
                                isDynamic, notSupportFormat, inputDtypeFlag, outDtypeFlag),
                      return false);
    return true;
}

bool CheckMatMulShape(const TensorDesc& aTensor, const TensorDesc& bTensor, bool supportL12btBf16, uint32_t aiCoreCnt,
                      GemmOpArgs& args)
{
    auto aShape = aTensor.GetShape().GetDims();
    auto bShape = bTensor.GetShape().GetDims();
    FUSION_PASS_CHECK(aShape.size() != static_cast<size_t>(kShapeDim),
                      OPS_LOG_E(kPassName, "mm input 0 dims must be 2."), return false);
    FUSION_PASS_CHECK(bShape.size() != static_cast<size_t>(kShapeDim),
                      OPS_LOG_E(kPassName, "mm input 1 dims must be 2."), return false);

    uint64_t aKDim = args.isATrans ? aShape[kShapeDim - kOuterAxis] : aShape[kShapeDim - kInnerAxis];
    uint64_t bKDim = args.isBTrans ? bShape[kShapeDim - kInnerAxis] : bShape[kShapeDim - kOuterAxis];
    FUSION_PASS_CHECK(aKDim != bKDim, OPS_LOG_I(kPassName, "mm input aK != bK can not support GemmOp."), return false);

    if (!supportL12btBf16) {
        if (!CheckShapeInWhiteList(aShape, bShape, args.isATrans, args.isBTrans)) {
            return false;
        }
    } else {
        args.aDtypeSize = GetSizeByDataType(aTensor.GetDataType());
        args.kValue = aKDim;
        args.mValue = args.isATrans ? aShape[kShapeDim - kInnerAxis] : aShape[kShapeDim - kOuterAxis];
        args.nValue = args.isBTrans ? bShape[kShapeDim - kOuterAxis] : bShape[kShapeDim - kInnerAxis];
        if (!CheckShapeValid(args, aiCoreCnt)) {
            return false;
        }
    }
    return true;
}

bool CheckNodeMatMul(const GNode& nodeMatmul, bool supportL12btBf16, uint32_t aiCoreCnt, GemmOpArgs& args)
{
    TensorDesc aTensor;
    TensorDesc bTensor;
    TensorDesc mmOutTensor;
    if (!ValidateMatMulDesc(nodeMatmul, supportL12btBf16, aTensor, bTensor, mmOutTensor)) {
        return false;
    }

    if (!GetNodeMatMulAttrs(nodeMatmul, supportL12btBf16, args)) {
        return false;
    }

    if (!CheckMatMulShape(aTensor, bTensor, supportL12btBf16, aiCoreCnt, args)) {
        return false;
    }
    return true;
}

bool CheckNodeCast(const GNode* nodeCast)
{
    if (nodeCast == nullptr) {
        return true; // no Cast node is valid
    }
    TensorDesc castIn;
    TensorDesc castOut;
    FUSION_PASS_CHECK(nodeCast->GetInputDesc(0, castIn) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "Get cast input desc failed."), return false);
    FUSION_PASS_CHECK(nodeCast->GetOutputDesc(0, castOut) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "Get cast output desc failed."), return false);

    FUSION_PASS_CHECK(castIn.GetFormat() != FORMAT_ND || castOut.GetFormat() != FORMAT_ND,
                      OPS_LOG_I(kPassName, "cast format can not support GemmOp."), return false);
    DataType castDtype = castIn.GetDataType();
    DataType castOutDtype = castOut.GetDataType();
    FUSION_PASS_CHECK((castDtype != DT_FLOAT16 && castDtype != DT_BF16) || (castOutDtype != DT_FLOAT),
                      OPS_LOG_I(kPassName, "cast dtype can not support GemmOp."), return false);
    FUSION_PASS_CHECK(castIn.GetShape().GetDims().size() != static_cast<size_t>(kShapeDim) ||
                          castOut.GetShape().GetDims().size() != static_cast<size_t>(kShapeDim),
                      OPS_LOG_I(kPassName, "cast shape dim can not support GemmOp."), return false);
    return true;
}

bool CheckNodeAssignAdd(const GNode& nodeAssignAdd)
{
    TensorDesc input0Desc;
    TensorDesc input1Desc;
    TensorDesc outputDesc;
    FUSION_PASS_CHECK(nodeAssignAdd.GetInputDesc(0, input0Desc) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "Get assignadd input0 desc failed."), return false);
    FUSION_PASS_CHECK(nodeAssignAdd.GetInputDesc(1, input1Desc) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "Get assignadd input1 desc failed."), return false);
    FUSION_PASS_CHECK(nodeAssignAdd.GetOutputDesc(0, outputDesc) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "Get assignadd output desc failed."), return false);

    FUSION_PASS_CHECK(input0Desc.GetFormat() != FORMAT_ND || input1Desc.GetFormat() != FORMAT_ND ||
                          outputDesc.GetFormat() != FORMAT_ND,
                      OPS_LOG_I(kPassName, "AssignAdd format can not support GemmOp."), return false);
    FUSION_PASS_CHECK(
        input0Desc.GetDataType() != DT_FLOAT || !CheckDtypeSupport(input1Desc) || outputDesc.GetDataType() != DT_FLOAT,
        OPS_LOG_I(kPassName, "AssignAdd dtype can not support GemmOp."), return false);
    auto input0Shape = input0Desc.GetShape().GetDims();
    auto input1Shape = input1Desc.GetShape().GetDims();
    auto outputShape = outputDesc.GetShape().GetDims();
    FUSION_PASS_CHECK(input0Shape.size() != static_cast<size_t>(kShapeDim) ||
                          input1Shape.size() != static_cast<size_t>(kShapeDim) ||
                          outputShape.size() != static_cast<size_t>(kShapeDim),
                      OPS_LOG_I(kPassName, "AssignAdd shape dim can not support GemmOp."), return false);
    int64_t input0M = input0Shape[kShapeDim - 2];
    int64_t input0N = input0Shape[kShapeDim - 1];
    int64_t input1M = input1Shape[kShapeDim - 2];
    int64_t input1N = input1Shape[kShapeDim - 1];
    FUSION_PASS_CHECK(input0M != input1M || input0N != input1N,
                      OPS_LOG_I(kPassName, "AssignAdd input0 != input1 can not support GemmOp."), return false);
    return true;
}

bool CheckAssignAddInputControl(const GNode& nodeAssignAdd)
{
    // AssignAdd must have NO input control edges
    auto inControlNodes = nodeAssignAdd.GetInControlNodes();
    for (auto& ctrlNode : inControlNodes) {
        if (ctrlNode != nullptr) {
            OPS_LOG_D(kPassName, "assignAdd have input control edge.");
            return false;
        }
    }
    return true;
}

/// Find the Cast node between MatMul output and AssignAdd input1 (if any).
/// Returns nullptr GNodePtr if there is no Cast (Pattern1: no Cast).
GNodePtr FindCastNode(const GNode& nodeAssignAdd)
{
    auto [srcNodePtr, srcPort] = nodeAssignAdd.GetInDataNodesAndPortIndexs(1);
    if (srcNodePtr == nullptr) {
        return nullptr;
    }
    AscendString opType;
    if (srcNodePtr->GetType(opType) != GRAPH_SUCCESS) {
        return nullptr;
    }
    if (opType == kOpTypeCast) {
        return srcNodePtr;
    }
    return nullptr;
}

// ---------------------------------------------------------------------------
// Replacement helpers
// ---------------------------------------------------------------------------

/// Create a zero-valued Const EsTensorHolder with the given dtype (for alpha/beta).
EsTensorHolder CreateZeroConst(EsGraphBuilder& builder, DataType dtype)
{
    auto constHolder = builder.CreateConst(std::vector<float>{0.0f}, {1}, dtype, FORMAT_ND);
    return constHolder;
}

/// Build GemmV2 replacement subgraph (Milan path: !supportL12btBf16)
bool ConnectGemmV2Edges(Graph* rawGraph, const EsTensorHolder& a, const EsTensorHolder& b, const EsTensorHolder& alpha,
                        const EsTensorHolder& beta, const EsTensorHolder& c, GNode& gemmNode)
{
    FUSION_PASS_CHECK(
        AddEdgeAndUpdatePeerDesc(*rawGraph, *a.GetProducer(), a.GetProducerOutIndex(), gemmNode, 0) != GRAPH_SUCCESS,
        OPS_LOG_E(kPassName, "AddEdge for GemmV2 input a failed."), return false);
    FUSION_PASS_CHECK(
        AddEdgeAndUpdatePeerDesc(*rawGraph, *b.GetProducer(), b.GetProducerOutIndex(), gemmNode, 1) != GRAPH_SUCCESS,
        OPS_LOG_E(kPassName, "AddEdge for GemmV2 input b failed."), return false);
    FUSION_PASS_CHECK(AddEdgeAndUpdatePeerDesc(*rawGraph, *alpha.GetProducer(), alpha.GetProducerOutIndex(), gemmNode,
                                               2) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "AddEdge for GemmV2 input alpha failed."), return false);
    FUSION_PASS_CHECK(AddEdgeAndUpdatePeerDesc(*rawGraph, *beta.GetProducer(), beta.GetProducerOutIndex(), gemmNode,
                                               3) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "AddEdge for GemmV2 input beta failed."), return false);
    FUSION_PASS_CHECK(
        AddEdgeAndUpdatePeerDesc(*rawGraph, *c.GetProducer(), c.GetProducerOutIndex(), gemmNode, 4) != GRAPH_SUCCESS,
        OPS_LOG_E(kPassName, "AddEdge for GemmV2 input c failed."), return false);
    return true;
}

bool SetGemmV2AttrsAndDescs(GNode& gemmNode, const GNode& nodeMatmul, const GNode& nodeAssignAdd,
                            const GemmOpArgs& args, DataType mmInputDtype)
{
    bool transA = args.isATrans;
    bool transB = args.isBTrans;
    FUSION_PASS_CHECK(gemmNode.SetAttr("transpose_a", transA) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "Set transpose_a attr failed."), return false);
    FUSION_PASS_CHECK(gemmNode.SetAttr("transpose_b", transB) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "Set transpose_b attr failed."), return false);

    TensorDesc desc;
    nodeMatmul.GetInputDesc(0, desc);
    FUSION_PASS_CHECK(gemmNode.UpdateInputDesc(0, desc) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "Update GemmV2 input a desc failed."), return false);
    nodeMatmul.GetInputDesc(1, desc);
    FUSION_PASS_CHECK(gemmNode.UpdateInputDesc(1, desc) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "Update GemmV2 input b desc failed."), return false);
    TensorDesc alphaDesc(ge::Shape({1}), FORMAT_ND, mmInputDtype);
    FUSION_PASS_CHECK(gemmNode.UpdateInputDesc(2, alphaDesc) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "Update GemmV2 input alpha desc failed."), return false);
    FUSION_PASS_CHECK(gemmNode.UpdateInputDesc(3, alphaDesc) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "Update GemmV2 input beta desc failed."), return false);
    nodeAssignAdd.GetInputDesc(0, desc);
    FUSION_PASS_CHECK(gemmNode.UpdateInputDesc(4, desc) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "Update GemmV2 input c desc failed."), return false);
    nodeAssignAdd.GetOutputDesc(0, desc);
    FUSION_PASS_CHECK(gemmNode.UpdateOutputDesc(0, desc) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "Update GemmV2 output desc failed."), return false);
    return true;
}

GraphUniqPtr BuildGemmV2Replacement(const GNode& nodeMatmul, const GNode& nodeAssignAdd, const GemmOpArgs& args)
{
    auto builder = es::EsGraphBuilder("replacement");

    auto a = builder.CreateInput(0);
    auto b = builder.CreateInput(1);
    auto c = builder.CreateInput(2);

    TensorDesc aDesc;
    nodeMatmul.GetInputDesc(0, aDesc);
    DataType mmInputDtype = aDesc.GetDataType();

    auto alpha = CreateZeroConst(builder, mmInputDtype);
    auto beta = CreateZeroConst(builder, mmInputDtype);

    auto* rawGraph = builder.GetCGraphBuilder()->GetGraph();
    AscendString matmulName;
    nodeMatmul.GetName(matmulName);
    std::string gemmName = std::string(matmulName.GetString()) + "_gemmV2";

    auto gemmNode = CompliantNodeBuilder(rawGraph)
                        .OpType(kOpTypeGemmV2)
                        .Name(gemmName.c_str())
                        .IrDefInputs({
                            {"a", CompliantNodeBuilder::kEsIrInputRequired, ""},
                            {"b", CompliantNodeBuilder::kEsIrInputRequired, ""},
                            {"alpha", CompliantNodeBuilder::kEsIrInputRequired, ""},
                            {"beta", CompliantNodeBuilder::kEsIrInputRequired, ""},
                            {"c", CompliantNodeBuilder::kEsIrInputRequired, ""},
                        })
                        .IrDefOutputs({{"c", CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                        .IrDefAttrs({
                            {"transpose_a", CompliantNodeBuilder::kEsAttrRequired, "Bool", CreateFrom(false)},
                            {"transpose_b", CompliantNodeBuilder::kEsAttrRequired, "Bool", CreateFrom(false)},
                        })
                        .Build();

    FUSION_PASS_CHECK(!ConnectGemmV2Edges(rawGraph, a, b, alpha, beta, c, gemmNode),
                      OPS_LOG_E(kPassName, "Connect GemmV2 edges failed."), return nullptr);
    FUSION_PASS_CHECK(!SetGemmV2AttrsAndDescs(gemmNode, nodeMatmul, nodeAssignAdd, args, mmInputDtype),
                      OPS_LOG_E(kPassName, "Set GemmV2 attrs and descs failed."), return nullptr);

    CopyOtherAttrs(nodeMatmul, gemmNode, kPassName);

    auto* yHolder = builder.GetCGraphBuilder()->GetTensorHolderFromNode(gemmNode, 0);
    auto y = EsTensorHolder(yHolder);

    return builder.BuildAndReset({y});
}

/// Build GemmV3 replacement subgraph (David path: supportL12btBf16)
bool ConnectGemmV3Edges(Graph* rawGraph, const EsTensorHolder& a, const EsTensorHolder& b, const EsTensorHolder& c,
                        GNode& gemmNode)
{
    FUSION_PASS_CHECK(
        AddEdgeAndUpdatePeerDesc(*rawGraph, *a.GetProducer(), a.GetProducerOutIndex(), gemmNode, 0) != GRAPH_SUCCESS,
        OPS_LOG_E(kPassName, "AddEdge for GemmV3 input a failed."), return false);
    FUSION_PASS_CHECK(
        AddEdgeAndUpdatePeerDesc(*rawGraph, *b.GetProducer(), b.GetProducerOutIndex(), gemmNode, 1) != GRAPH_SUCCESS,
        OPS_LOG_E(kPassName, "AddEdge for GemmV3 input b failed."), return false);
    FUSION_PASS_CHECK(
        AddEdgeAndUpdatePeerDesc(*rawGraph, *c.GetProducer(), c.GetProducerOutIndex(), gemmNode, 2) != GRAPH_SUCCESS,
        OPS_LOG_E(kPassName, "AddEdge for GemmV3 input c failed."), return false);
    return true;
}

bool SetGemmV3AttrsAndDescs(GNode& gemmNode, const GNode& nodeMatmul, const GNode& nodeAssignAdd,
                            const GemmOpArgs& args)
{
    bool transA = args.isATrans;
    bool transB = args.isBTrans;
    bool enableHf32 = args.isHf32;
    FUSION_PASS_CHECK(gemmNode.SetAttr("transpose_a", transA) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "Set transpose_a attr failed."), return false);
    FUSION_PASS_CHECK(gemmNode.SetAttr("transpose_b", transB) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "Set transpose_b attr failed."), return false);
    FUSION_PASS_CHECK(gemmNode.SetAttr("enable_hf32", enableHf32) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "Set enable_hf32 attr failed."), return false);

    TensorDesc desc;
    nodeMatmul.GetInputDesc(0, desc);
    FUSION_PASS_CHECK(gemmNode.UpdateInputDesc(0, desc) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "Update GemmV3 input a desc failed."), return false);
    nodeMatmul.GetInputDesc(1, desc);
    FUSION_PASS_CHECK(gemmNode.UpdateInputDesc(1, desc) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "Update GemmV3 input b desc failed."), return false);
    nodeAssignAdd.GetInputDesc(0, desc);
    FUSION_PASS_CHECK(gemmNode.UpdateInputDesc(2, desc) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "Update GemmV3 input c desc failed."), return false);
    nodeAssignAdd.GetOutputDesc(0, desc);
    desc.SetReuseInputIndex(static_cast<uint32_t>(kInputCIndex));
    FUSION_PASS_CHECK(gemmNode.UpdateOutputDesc(0, desc) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "Update GemmV3 output desc failed."), return false);
    return true;
}

GraphUniqPtr BuildGemmV3Replacement(const GNode& nodeMatmul, const GNode& nodeAssignAdd, const GemmOpArgs& args)
{
    auto builder = es::EsGraphBuilder("replacement");

    auto a = builder.CreateInput(0);
    auto b = builder.CreateInput(1);
    auto c = builder.CreateInput(2);

    auto* rawGraph = builder.GetCGraphBuilder()->GetGraph();
    AscendString matmulName;
    nodeMatmul.GetName(matmulName);
    std::string gemmName = std::string(matmulName.GetString()) + "_gemmV3";

    auto gemmNode = CompliantNodeBuilder(rawGraph)
                        .OpType(kOpTypeGemmV3)
                        .Name(gemmName.c_str())
                        .IrDefInputs({
                            {"a", CompliantNodeBuilder::kEsIrInputRequired, ""},
                            {"b", CompliantNodeBuilder::kEsIrInputRequired, ""},
                            {"c", CompliantNodeBuilder::kEsIrInputOptional, ""},
                        })
                        .IrDefOutputs({{"y", CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                        .IrDefAttrs({
                            {"alpha", CompliantNodeBuilder::kEsAttrRequired, "Float", CreateFrom(1.0f)},
                            {"beta", CompliantNodeBuilder::kEsAttrRequired, "Float", CreateFrom(1.0f)},
                            {"transpose_a", CompliantNodeBuilder::kEsAttrRequired, "Bool", CreateFrom(false)},
                            {"transpose_b", CompliantNodeBuilder::kEsAttrRequired, "Bool", CreateFrom(false)},
                            {"enable_hf32", CompliantNodeBuilder::kEsAttrRequired, "Bool", CreateFrom(false)},
                        })
                        .Build();

    FUSION_PASS_CHECK(!ConnectGemmV3Edges(rawGraph, a, b, c, gemmNode),
                      OPS_LOG_E(kPassName, "Connect GemmV3 edges failed."), return nullptr);
    FUSION_PASS_CHECK(!SetGemmV3AttrsAndDescs(gemmNode, nodeMatmul, nodeAssignAdd, args),
                      OPS_LOG_E(kPassName, "Set GemmV3 attrs and descs failed."), return nullptr);

    CopyOtherAttrs(nodeMatmul, gemmNode, kPassName);

    auto* yHolder = builder.GetCGraphBuilder()->GetTensorHolderFromNode(gemmNode, 0);
    auto y = EsTensorHolder(yHolder);

    return builder.BuildAndReset({y});
}

} // namespace

// ---------------------------------------------------------------------------
// PatternFusionPass implementation
// ---------------------------------------------------------------------------

std::vector<PatternUniqPtr> MatmulToGemmOpFusionPass::Patterns()
{
    std::vector<PatternUniqPtr> patterns;
    const char* opTypes[] = {kOpTypeMatMul, kOpTypeMatMulV2, kOpTypeMatMulV3};
    for (const char* opType : opTypes) {
        patterns.emplace_back(BuildPatternWithCast(std::string("MatmulToGemmOpFusionPass0_") + opType, opType));
        patterns.emplace_back(BuildPatternWithoutCast(std::string("MatmulToGemmOpFusionPass1_") + opType, opType));
    }
    return patterns;
}

bool MatmulToGemmOpFusionPass::MeetRequirements(const std::unique_ptr<MatchResult>& matchResult)
{
    OPS_LOG_D(kPassName, "Begin to do MatmulToGemmOpFusionPass MeetRequirements.");
    FUSION_PASS_CHECK(GetGeCompilerVersionNum() < kGeCompilerVersion900,
                      OPS_LOG_D(kPassName, "GE runtime < 9.0.0, skip fusion."), return false);
    PlatformInfo platformInfo;
    OptionalInfo optionalInfo;
    FUSION_PASS_CHECK(
        PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platformInfo, optionalInfo) != SUCCESS,
        OPS_LOG_I(kPassName, "Can't get platformInfo."), return false);
    bool supportL12btBf16 = IsSupportL12BtBf16(platformInfo);
    uint32_t aiCoreCnt = platformInfo.soc_info.ai_core_cnt;
    if (aiCoreCnt == 0) {
        aiCoreCnt = 1;
    }
    if (!supportL12btBf16) {
        FUSION_PASS_CHECK(!CheckPlatformIntrinsics(platformInfo),
                          OPS_LOG_I(kPassName, "matmul To gemmOp does not support in this platform."), return false);
    }
    NodeIo matmulIo;
    FUSION_PASS_CHECK(matchResult->GetCapturedTensor(kCaptureMatMulSlot, matmulIo) != SUCCESS,
                      OPS_LOG_E(kPassName, "Failed to get captured MatMul tensor."), return false);
    GNode nodeMatmul = matmulIo.node;
    NodeIo assignAddIo;
    FUSION_PASS_CHECK(matchResult->GetCapturedTensor(kCaptureAssignAddSlot, assignAddIo) != SUCCESS,
                      OPS_LOG_E(kPassName, "Failed to get captured AssignAdd tensor."), return false);
    GNode nodeAssignAdd = assignAddIo.node;
    GNodePtr nodeCastPtr = FindCastNode(nodeAssignAdd);
    GemmOpArgs args;
    FUSION_PASS_CHECK(!CheckNodeMatMul(nodeMatmul, supportL12btBf16, aiCoreCnt, args),
                      OPS_LOG_I(kPassName, "Node matmul check unsuccess, not support to do fusion."), return false);
    FUSION_PASS_CHECK(!CheckNodeCast(nodeCastPtr.get()),
                      OPS_LOG_I(kPassName, "Node cast check unsuccess, not support to do fusion."), return false);
    FUSION_PASS_CHECK(!CheckNodeAssignAdd(nodeAssignAdd),
                      OPS_LOG_I(kPassName, "Node AssignAdd check unsuccess, not support to do fusion."), return false);
    FUSION_PASS_CHECK(!CheckAssignAddInputControl(nodeAssignAdd),
                      OPS_LOG_I(kPassName, "Node AssignAdd have input control edge, not support to do fusion."),
                      return false);
    OPS_LOG_D(kPassName, "MatmulToGemmOpFusionPass requirements met.");
    return true;
}

std::unique_ptr<Graph> MatmulToGemmOpFusionPass::Replacement(const std::unique_ptr<MatchResult>& matchResult)
{
    OPS_LOG_I(kPassName, "Begin to do MatmulToGemmOpFusionPass Replacement.");

    // Get platform info (MeetRequirements already validated, just fetch here)
    PlatformInfo platformInfo;
    OptionalInfo optionalInfo;
    PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platformInfo, optionalInfo);
    bool supportL12btBf16 = IsSupportL12BtBf16(platformInfo);

    // Get captured nodes (MeetRequirements already validated, just fetch here)
    NodeIo matmulIo;
    matchResult->GetCapturedTensor(kCaptureMatMulSlot, matmulIo);
    GNode nodeMatmul = matmulIo.node;

    NodeIo assignAddIo;
    matchResult->GetCapturedTensor(kCaptureAssignAddSlot, assignAddIo);
    GNode nodeAssignAdd = assignAddIo.node;

    // Get attrs for building GemmOp (MeetRequirements already validated via CheckNodeMatMul)
    GemmOpArgs args;
    GetNodeMatMulAttrs(nodeMatmul, supportL12btBf16, args);

    // Build replacement
    if (!supportL12btBf16) {
        OPS_LOG_I(kPassName, "Building GemmV2 replacement (Milan path).");
        return BuildGemmV2Replacement(nodeMatmul, nodeAssignAdd, args);
    } else {
        OPS_LOG_I(kPassName, "Building GemmV3 replacement (David path).");
        return BuildGemmV3Replacement(nodeMatmul, nodeAssignAdd, args);
    }
}

REG_FUSION_PASS(MatmulToGemmOpFusionPass).Stage(CustomPassStage::kCompatibleInherited);

} // namespace ops

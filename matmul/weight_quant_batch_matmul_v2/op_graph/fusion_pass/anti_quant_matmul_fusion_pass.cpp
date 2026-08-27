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
 * \file anti_quant_matmul_fusion_pass.cpp
 * \brief AntiQuantMatMulFusionPass
 *
 * Fusion pattern: fuse AscendAntiQuant (+Add) (+Mul) + MatMul/BatchMatMul into WeightQuantBatchMatmulV2.
 *
 * Supported fusion scenarios (three cases):
 *   1. AscendAntiQuant + Mul + MatMul: only Mul exists (no Add).
 *   2. AscendAntiQuant + Add + Mul + MatMul: both Add and Mul exist (Add requires Mul).
 *   3. AscendAntiQuant + MatMul: neither Add nor Mul exists.
 *
 *                  int8_weight
 *                       |
 *               AscendAntiQuant  (attr: scale, offset)
 *                      |
 *                 (optional) Add  --- const_offset
 *                      |
 *                 (optional) Mul  --- const_scale
 *                      |
 *         fp16_input  |
 *              \    /
 *               MatMul/MatMulV2/BatchMatMul/BatchMatMulV2
 *                  |
 *                 out
 *
 *            -------->
 *
 * fp16_input   int8_weight   antiquant_scale(fp16)   antiquant_offset(fp16)   [bias]
 *     \            |              |                      |                     /
 *      WeightQuantBatchMatmulV2 (attr: transpose_x, transpose_weight)
 *                    |
 *                   out
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "ge/fusion/pass/pattern_fusion_pass.h"
#include "ge/fusion/graph_rewriter.h"
#include "ge/es_graph_builder.h"
#include "es_nn_ops.h"
#include "log/log.h"
#include "platform/platform_info.h"
#include "platform/soc_spec.h"
#include "runtime/runtime/base.h"
#include "version/ge-compiler_version.h"
#include "common/op_graph/fusion_pass/weight_quant_fusion_utils.h"
#include "common/inc/op_graph/cube_utils/cube_fp16_t.h"
#include "anti_quant_matmul_fusion_pass.h"

namespace ops {
using ge::fusion::GraphUniqPtr;
using ge::fusion::SubgraphBoundary;
using ge::fusion::SubgraphInput;
using ge::fusion::SubgraphOutput;
using ge::fusion::SubgraphRewriter;
using weight_quant::GetInputNode;

static const char* kFusedOpType = "AntiQuantMatMulFusionPass";
static const char* kOpTypeAntiQuant = "AscendAntiQuant";
static const char* kOpTypeAdd = "Add";
static const char* kOpTypeMul = "Mul";
static const char* kOpTypeMatMul = "MatMul";
static const char* kOpTypeMatMulV2 = "MatMulV2";
static const char* kOpTypeBatchMatMul = "BatchMatMul";
static const char* kOpTypeBatchMatMulV2 = "BatchMatMulV2";
static const char* kOpTypeWeightQuantBatchMatmulV2 = "WeightQuantBatchMatmulV2";
static const char* kOpTypeConst = "Const";

static const std::string kNpuArch2201 = "2201";
static const std::string kNpuArch3510 = "3510";

static const std::vector<std::string> kMatMulTypes = {kOpTypeMatMul, kOpTypeMatMulV2, kOpTypeBatchMatMul,
                                                      kOpTypeBatchMatMulV2};

constexpr int64_t kMatMulDimNum = 2;
constexpr int64_t kMdimMax = 64;
constexpr int64_t kDimMax = 10240;
constexpr int64_t kDimMin = 5120;
constexpr int64_t kBiasIndex = 3;
constexpr float kInitZero = 0.0F;
constexpr float kInitOne = 1.0F;
constexpr int64_t kXPort = 0;
constexpr int64_t kWeightPort = 1;
constexpr int64_t kAntiQuantScalePort = 2;
constexpr int64_t kAntiQuantOffsetPort = 3;
constexpr int64_t kBiasPort = 6;

struct FusionCtx {
    ge::GNodePtr nodeAntiquant;
    ge::GNodePtr nodeAdd;
    ge::GNodePtr nodeMul;
    ge::GNodePtr nodeMatmul;
    int64_t mDim = 1;
    int64_t nDim = 1;
    int64_t kDim = 1;
    int32_t addInputIndex = 1; // Const input port on Add (the other port receives AntiQuant output)
    int32_t mulInputIndex = 1; // Const input port on Mul (the other port receives Add/AntiQuant output)
    bool mmTransX1 = false;
    bool mmTransX2 = false;
    bool supportV1 = true; // true on 2201; false on 3510
};

static bool IsMatMulType(const ge::AscendString& opType)
{
    const std::string opTypeStr(opType.GetString());
    return std::find(kMatMulTypes.begin(), kMatMulTypes.end(), opTypeStr) != kMatMulTypes.end();
}

// Check NPU architecture: only 2201 and 3510 are supported.
// supportV1=true (2201) means shape准入 and blacklist checks apply;
// supportV1=false (3510) means WeightQuantBatchMatmulV2 has no shape limits, skip checks.
static bool CheckPlatform(bool& supportV1)
{
    constexpr uint32_t kMaxLen = 32;
    char npuArchStr[kMaxLen] = {};
    if (rtGetSocSpec("version", "NpuArch", npuArchStr, kMaxLen) == 0) {
        std::string npuArch(npuArchStr);
        const std::set<std::string> supportNpuArchList = {kNpuArch2201, kNpuArch3510};
        if (supportNpuArchList.count(npuArch) == 0) {
            OP_LOGW(kFusedOpType, "The fusion pass is not supported on NpuArch [%s]", npuArch.c_str());
            return false;
        }
        supportV1 = (npuArch != kNpuArch3510);
        return true;
    }
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optionalInfo;
    if (fe::PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platformInfo, optionalInfo) !=
        ge::SUCCESS) {
        OP_LOGW(kFusedOpType, "Failed to get platform info.");
        return false;
    }
    const int32_t npuArch = platformInfo.soc_info.arch_type;
    const std::set<int32_t> supportNpuArchList = {static_cast<int32_t>(NpuArch::DAV_2201),
                                                  static_cast<int32_t>(NpuArch::DAV_3510)};
    const std::string soc = platformInfo.str_info.short_soc_version;
    if (supportNpuArchList.count(npuArch) == 0) {
        OP_LOGW(kFusedOpType, "The fusion pass is not supported on SoC [%s]", soc.c_str());
        return false;
    }
    supportV1 = (npuArch != static_cast<int32_t>(NpuArch::DAV_3510));
    return true;
}

static bool CheckNodeDtype(const ge::GNode& mmNode, const ge::GNode& antiquantNode)
{
    ge::TensorDesc mmInX1Desc;
    if (mmNode.GetInputDesc(kXPort, mmInX1Desc) != ge::GRAPH_SUCCESS) {
        OP_LOGW(kFusedOpType, "Failed to get input x1 desc of matmul node");
        return false;
    }
    ge::TensorDesc mmInX2Desc;
    if (mmNode.GetInputDesc(kWeightPort, mmInX2Desc) != ge::GRAPH_SUCCESS) {
        OP_LOGW(kFusedOpType, "Failed to get input x2 desc of matmul node");
        return false;
    }
    ge::TensorDesc mmOutDesc;
    if (mmNode.GetOutputDesc(0, mmOutDesc) != ge::GRAPH_SUCCESS) {
        OP_LOGW(kFusedOpType, "Failed to get output desc of matmul node");
        return false;
    }
    ge::TensorDesc antiquantInDesc;
    if (antiquantNode.GetInputDesc(0, antiquantInDesc) != ge::GRAPH_SUCCESS) {
        OP_LOGW(kFusedOpType, "Failed to get input desc of AscendAntiQuant node");
        return false;
    }
    ge::TensorDesc antiquantOutDesc;
    if (antiquantNode.GetOutputDesc(0, antiquantOutDesc) != ge::GRAPH_SUCCESS) {
        OP_LOGW(kFusedOpType, "Failed to get output desc of AscendAntiQuant node");
        return false;
    }
    if (mmInX1Desc.GetDataType() != ge::DT_FLOAT16 || mmInX2Desc.GetDataType() != ge::DT_FLOAT16) {
        OP_LOGW(kFusedOpType, "The dtype of matmul input x and weight are %d and %d, which must be float16",
                static_cast<int32_t>(mmInX1Desc.GetDataType()), static_cast<int32_t>(mmInX2Desc.GetDataType()));
        return false;
    }
    if (mmOutDesc.GetDataType() != ge::DT_FLOAT16) {
        OP_LOGW(kFusedOpType, "The dtype of matmul output is %d, which must be float16",
                static_cast<int32_t>(mmOutDesc.GetDataType()));
        return false;
    }
    if (antiquantInDesc.GetDataType() != ge::DT_INT8) {
        OP_LOGW(kFusedOpType, "The dtype of AscendAntiQuant input is %d, which must be int8",
                static_cast<int32_t>(antiquantInDesc.GetDataType()));
        return false;
    }
    if (antiquantOutDesc.GetDataType() != ge::DT_FLOAT16) {
        OP_LOGW(kFusedOpType, "The dtype of AscendAntiQuant output is %d, which must be float16",
                static_cast<int32_t>(antiquantOutDesc.GetDataType()));
        return false;
    }
    return true;
}

static bool CheckNodeShape(const ge::GNode& mmNode, const ge::GNode& antiquantNode, FusionCtx& ctx)
{
    ge::TensorDesc mmInX1Desc;
    if (mmNode.GetInputDesc(kXPort, mmInX1Desc) != ge::GRAPH_SUCCESS) {
        return false;
    }
    ge::TensorDesc mmInX2Desc;
    if (mmNode.GetInputDesc(kWeightPort, mmInX2Desc) != ge::GRAPH_SUCCESS) {
        return false;
    }
    auto mmInX1Shape = mmInX1Desc.GetOriginShape().GetDims();
    auto mmInX2Shape = mmInX2Desc.GetOriginShape().GetDims();
    if (mmInX1Shape.size() != kMatMulDimNum || mmInX2Shape.size() != kMatMulDimNum) {
        OP_LOGW(kFusedOpType, "The shape length of matmul input x1 and x2 are %zu and %zu, which must be 2",
                mmInX1Shape.size(), mmInX2Shape.size());
        return false;
    }
    bool dynamicMode = false;
    for (auto dim : mmInX1Shape) {
        if (dim < 0) {
            dynamicMode = true;
            break;
        }
    }
    if (dynamicMode) {
        OP_LOGW(kFusedOpType, "Dynamic shape is not supported yet.");
        return false;
    }

    ge::TensorDesc antiquantInDesc;
    if (antiquantNode.GetInputDesc(0, antiquantInDesc) != ge::GRAPH_SUCCESS) {
        return false;
    }
    auto antiquantInShape = antiquantInDesc.GetOriginShape().GetDims();
    if (antiquantInShape.size() != kMatMulDimNum) {
        OP_LOGW(kFusedOpType, "The shape length of AscendAntiQuant input is %zu, which must be 2",
                antiquantInShape.size());
        return false;
    }

    ge::AscendString mmType;
    if (mmNode.GetType(mmType) != ge::GRAPH_SUCCESS) {
        return false;
    }
    std::string transX1Str("transpose_x1");
    std::string transX2Str("transpose_x2");
    const std::string mmTypeStr(mmType.GetString());
    if (mmTypeStr == kOpTypeBatchMatMul || mmTypeStr == kOpTypeBatchMatMulV2) {
        transX1Str = "adj_x1";
        transX2Str = "adj_x2";
    }
    if (mmNode.GetAttr(ge::AscendString(transX1Str.c_str()), ctx.mmTransX1) != ge::GRAPH_SUCCESS) {
        OP_LOGW(kFusedOpType, "Failed to get attr transpose_x1 of matmul node");
        return false;
    }
    if (mmNode.GetAttr(ge::AscendString(transX2Str.c_str()), ctx.mmTransX2) != ge::GRAPH_SUCCESS) {
        OP_LOGW(kFusedOpType, "Failed to get attr transpose_x2 of matmul node");
        return false;
    }
    // Read transpose attrs: MatMul uses "transpose_x1/x2", BatchMatMul uses "adj_x1/x2".
    // Compute M/K/N from shapes considering transpose: x2 is the weight, so when transposed N comes first.
    ctx.mDim = ctx.mmTransX1 ? mmInX1Shape[1] : mmInX1Shape[0];
    ctx.nDim = ctx.mmTransX2 ? mmInX2Shape[0] : mmInX2Shape[1];
    ctx.kDim = ctx.mmTransX1 ? mmInX1Shape[0] : mmInX1Shape[1];
    // Shape准入 (only on 2201): M<=64, K>=5120, N>=5120.
    bool shapeMeet = (ctx.mDim <= kMdimMax) && ctx.nDim >= kDimMin && ctx.kDim >= kDimMin;
    // Blacklist (only on 2201): (K,N)=(5120,10240) or (10240,5120) causes tiling issues.
    bool blackCase = (ctx.kDim == kDimMin && ctx.nDim == kDimMax) || (ctx.kDim == kDimMax && ctx.nDim == kDimMin);
    if (ctx.supportV1 && blackCase) {
        OP_LOGW(kFusedOpType, "K/N is in blacklist (5120/10240 or 10240/5120), but got K=%ld, N=%ld", ctx.kDim,
                ctx.nDim);
        return false;
    }
    if (ctx.supportV1 && !shapeMeet) {
        OP_LOGW(kFusedOpType, "Requires M<=64, K>=5120, N>=5120, but got M=%ld, K=%ld, N=%ld", ctx.mDim, ctx.kDim,
                ctx.nDim);
        return false;
    }
    return true;
}

// Verify AntiQuant/Add/Mul each have exactly one consumer (otherwise fusion would break other paths).
static bool CheckVectorNode(const FusionCtx& ctx)
{
    auto antiConsumers = ctx.nodeAntiquant->GetOutDataNodesAndPortIndexs(0);
    if (antiConsumers.size() != 1) {
        OP_LOGW(kFusedOpType, "The output of AscendAntiQuant has no consumer or more than one consumer");
        return false;
    }
    ge::GNodePtr nextNode = antiConsumers.front().first;
    if (nextNode == nullptr) {
        OP_LOGW(kFusedOpType, "Failed to get next node of AscendAntiQuant");
        return false;
    }
    if (ctx.nodeAdd != nullptr) {
        auto addConsumers = ctx.nodeAdd->GetOutDataNodesAndPortIndexs(0);
        if (addConsumers.size() != 1) {
            OP_LOGW(kFusedOpType, "The output of Add has no consumer or more than one consumer");
            return false;
        }
        nextNode = addConsumers.front().first;
        if (nextNode == nullptr) {
            OP_LOGW(kFusedOpType, "Failed to get next node of Add");
            return false;
        }
    }
    if (ctx.nodeMul != nullptr) {
        auto mulConsumers = ctx.nodeMul->GetOutDataNodesAndPortIndexs(0);
        if (mulConsumers.size() != 1) {
            OP_LOGW(kFusedOpType, "The output of Mul has no consumer or more than one consumer");
            return false;
        }
        nextNode = mulConsumers.front().first;
        if (nextNode == nullptr) {
            OP_LOGW(kFusedOpType, "Failed to get next node of Mul");
            return false;
        }
    }
    return true;
}

static bool GetConstValue(const ge::GNodePtr& constNode, float* dataValue, int64_t dataLen)
{
    ge::Tensor tensor;
    if (constNode->GetAttr(ge::AscendString("value"), tensor) != ge::GRAPH_SUCCESS) {
        OP_LOGW(kFusedOpType, "Failed to get const value");
        return false;
    }
    auto dtype = tensor.GetTensorDesc().GetDataType();
    const uint8_t* dataPtr = tensor.GetData();
    if (dataPtr == nullptr) {
        OP_LOGW(kFusedOpType, "Const tensor data is null");
        return false;
    }
    int64_t elemSize = (dtype == ge::DT_FLOAT16) ? sizeof(uint16_t) : sizeof(float);
    if (tensor.GetSize() < static_cast<size_t>(dataLen) * elemSize) {
        OP_LOGW(kFusedOpType, "Const tensor data size %zu less than required %ld", tensor.GetSize(),
                static_cast<long>(dataLen) * elemSize);
        return false;
    }
    if (dtype == ge::DT_FLOAT16) {
        const uint16_t* ptr = reinterpret_cast<const uint16_t*>(dataPtr);
        for (int64_t idx = 0; idx < dataLen; idx++) {
            fp16_t dataFp16(ptr[idx]);
            dataValue[idx] = dataFp16.ToFloat();
        }
    } else if (dtype == ge::DT_FLOAT) {
        const float* ptr = reinterpret_cast<const float*>(dataPtr);
        for (int64_t idx = 0; idx < dataLen; idx++) {
            dataValue[idx] = ptr[idx];
        }
    } else {
        OP_LOGW(kFusedOpType, "The dtype of const node only supports fp16 or fp32, but got %d",
                static_cast<int32_t>(dtype));
        return false;
    }
    return true;
}

// Constant folding: pre-compute fp16 antiquant_scale/offset at compile time, eliminating runtime
// AntiQuant+Add+Mul computation.
//
// Both AscendAntiQuant and WeightQuantBatchMatmulV2 use the same dequant formula: (weight + offset) * scale.
// The fusion folds AntiQuant's scale/offset attrs + Add's const offset + Mul's const scale into two
// equivalent fp16 constants:
//   antiquant_scale[i]  = scale_data[i] * anti_scale
//   antiquant_offset[i] = offset_data[i] / anti_scale + anti_offset
//
// Mathematical derivation (AntiQuant→Add→Mul→MatMul):
//   antiq = (weight + anti_offset) * anti_scale
//   mul_out = ((weight + anti_offset) * anti_scale + offset_data) * scale_data
//           = (weight + anti_offset) * anti_scale * scale_data + offset_data * scale_data
//
// Target WQBMMV2 form: (weight + antiquant_offset) * antiquant_scale, so:
//   antiquant_scale  = scale_data * anti_scale           (scales multiply directly)
//   antiquant_offset = offset_data / anti_scale + anti_offset  (offset_data must be divided by
//                       anti_scale to move it from the multiply position to the add position)
//
static bool CalAntiQuantPara(const FusionCtx& ctx, uint16_t* antiquantScale, uint16_t* antiquantOffset, int64_t scaleN,
                             int64_t offsetN)
{
    float scale = 0.0F;
    if (ctx.nodeAntiquant->GetAttr(ge::AscendString("scale"), scale) != ge::GRAPH_SUCCESS) {
        OP_LOGW(kFusedOpType, "Failed to get attr scale of AscendAntiQuant.");
        return false;
    }
    if (std::fabs(scale) < 1e-6F) {
        OP_LOGW(kFusedOpType, "The scale attr of AscendAntiQuant can not be 0.");
        return false;
    }
    float offset = 0.0F;
    if (ctx.nodeAntiquant->GetAttr(ge::AscendString("offset"), offset) != ge::GRAPH_SUCCESS) {
        OP_LOGW(kFusedOpType, "Failed to get attr offset of AscendAntiQuant.");
        return false;
    }
    std::unique_ptr<float[]> offsetData(new (std::nothrow) float[offsetN]());
    std::unique_ptr<float[]> scaleData(new (std::nothrow) float[scaleN]());
    if (offsetData == nullptr || scaleData == nullptr) {
        OP_LOGW(kFusedOpType, "Failed to allocate memory for offset_data and scale_data");
        return false;
    }
    // Read Add's const offset; if Add is absent, offset defaults to all zeros.
    if (ctx.nodeAdd != nullptr) {
        ge::GNodePtr addInputNode = ctx.nodeAdd->GetInDataNodesAndPortIndexs(ctx.addInputIndex).first;
        if (addInputNode == nullptr) {
            OP_LOGW(kFusedOpType, "The const input node of Add is null");
            return false;
        }
        ge::AscendString addInputType;
        if (addInputNode->GetType(addInputType) != ge::GRAPH_SUCCESS) {
            OP_LOGW(kFusedOpType, "Failed to get type of Add const input node");
            return false;
        }
        if (std::string(addInputType.GetString()).find(kOpTypeConst) == std::string::npos) {
            OP_LOGW(kFusedOpType, "The Add input node is not Const, but is %s", addInputType.GetString());
            return false;
        }
        if (!GetConstValue(addInputNode, offsetData.get(), offsetN)) {
            OP_LOGW(kFusedOpType, "The const value of Add is invalid");
            return false;
        }
    } else {
        for (int64_t idx = 0; idx < offsetN; idx++) {
            offsetData[idx] = kInitZero;
        }
    }
    // Read Mul's const scale; if Mul is absent, scale defaults to all ones.
    if (ctx.nodeMul != nullptr) {
        ge::GNodePtr mulInputNode = ctx.nodeMul->GetInDataNodesAndPortIndexs(ctx.mulInputIndex).first;
        if (mulInputNode == nullptr) {
            OP_LOGW(kFusedOpType, "The const input node of Mul is null");
            return false;
        }
        ge::AscendString mulInputType;
        if (mulInputNode->GetType(mulInputType) != ge::GRAPH_SUCCESS) {
            OP_LOGW(kFusedOpType, "Failed to get type of Mul const input node");
            return false;
        }
        if (std::string(mulInputType.GetString()).find(kOpTypeConst) == std::string::npos) {
            OP_LOGW(kFusedOpType, "The Mul input node is not Const, but is %s", mulInputType.GetString());
            return false;
        }
        if (!GetConstValue(mulInputNode, scaleData.get(), scaleN)) {
            OP_LOGW(kFusedOpType, "The const value of Mul is invalid");
            return false;
        }
    } else {
        for (int64_t idx = 0; idx < scaleN; idx++) {
            scaleData[idx] = kInitOne;
        }
    }

    // Fold constants into fp16 antiquant_scale and antiquant_offset per the derivation above.
    for (int64_t idx = 0; idx < scaleN; idx++) {
        fp16_t antiquantScaleValue(scaleData[idx] * scale);
        antiquantScale[idx] = antiquantScaleValue.val;
    }
    for (int64_t idx = 0; idx < offsetN; idx++) {
        fp16_t antiquantOffsetValue(offsetData[idx] / scale + offset);
        antiquantOffset[idx] = antiquantOffsetValue.val;
    }
    return true;
}

static void SetTensorDesc(ge::TensorDesc& desc, const ge::DataType& dtype, const ge::Shape& shape,
                          const ge::Format& format)
{
    desc.SetDataType(dtype);
    desc.SetShape(shape);
    desc.SetFormat(format);
    desc.SetOriginShape(shape);
    desc.SetOriginFormat(format);
}

static void SetProducerOutputDesc(ge::GNode* producer, const ge::DataType& dtype, const ge::Shape& shape,
                                  const ge::Format& format)
{
    if (producer == nullptr) {
        return;
    }
    ge::TensorDesc desc;
    producer->GetOutputDesc(0, desc);
    SetTensorDesc(desc, dtype, shape, format);
    producer->UpdateOutputDesc(0, desc);
}

static void SetConsumerInputDesc(ge::GNode* consumer, int64_t port, const ge::DataType& dtype, const ge::Shape& shape,
                                 const ge::Format& format)
{
    if (consumer == nullptr) {
        return;
    }
    ge::TensorDesc desc;
    consumer->GetInputDesc(port, desc);
    SetTensorDesc(desc, dtype, shape, format);
    consumer->UpdateInputDesc(port, desc);
}

// Match one of three patterns by backtracking from MatMul's weight (x2) input:
//   x2 is Mul → check Mul's chain input: Add or AntiQuant
//   x2 is AntiQuant
static bool MatchPattern(const ge::GNode& matmulNode, FusionCtx& ctx)
{
    ge::AscendString mmType;
    if (matmulNode.GetType(mmType) != ge::GRAPH_SUCCESS || !IsMatMulType(mmType)) {
        return false;
    }
    ctx.nodeMatmul = std::make_shared<ge::GNode>(matmulNode);
    auto x2Input = GetInputNode(matmulNode, kWeightPort);
    ctx.nodeAdd = nullptr;
    ctx.nodeMul = nullptr;
    ctx.nodeAntiquant = nullptr;
    if (x2Input == nullptr) {
        return false;
    }
    ge::AscendString x2Type;
    if (x2Input->GetType(x2Type) != ge::GRAPH_SUCCESS) {
        return false;
    }
    const std::string x2TypeStr(x2Type.GetString());
    if (x2TypeStr == kOpTypeMul) {
        ctx.nodeMul = x2Input;
        // Mul has two inputs: one from the AntiQuant(->Add) chain, the other is a const scale.
        ge::GNodePtr mulChainNode = nullptr;
        int32_t mulChainPort = -1;
        for (int32_t port = 0; port < 2; port++) {
            auto inNode = GetInputNode(*ctx.nodeMul, port);
            if (inNode == nullptr) {
                continue;
            }
            ge::AscendString inType;
            if (inNode->GetType(inType) != ge::GRAPH_SUCCESS) {
                continue;
            }
            const std::string inTypeStr(inType.GetString());
            if (inTypeStr == kOpTypeAdd || inTypeStr == kOpTypeAntiQuant) {
                mulChainNode = inNode;
                mulChainPort = port;
                break;
            }
        }
        if (mulChainNode == nullptr) {
            OP_LOGW(kFusedOpType, "No AntiQuant/Add chain input found on Mul node");
            return false;
        }
        ctx.mulInputIndex = 1 - mulChainPort;
        ge::AscendString mulPreType;
        if (mulChainNode->GetType(mulPreType) != ge::GRAPH_SUCCESS) {
            return false;
        }
        const std::string mulPreTypeStr(mulPreType.GetString());
        if (mulPreTypeStr == kOpTypeAdd) {
            ctx.nodeAdd = mulChainNode;
            // Add has two inputs: one from AntiQuant, the other is a const offset.
            ge::GNodePtr antiNode = nullptr;
            int32_t addChainPort = -1;
            for (int32_t port = 0; port < 2; port++) {
                auto inNode = GetInputNode(*ctx.nodeAdd, port);
                if (inNode == nullptr) {
                    continue;
                }
                ge::AscendString inType;
                if (inNode->GetType(inType) != ge::GRAPH_SUCCESS) {
                    continue;
                }
                if (std::string(inType.GetString()) == kOpTypeAntiQuant) {
                    antiNode = inNode;
                    addChainPort = port;
                    break;
                }
            }
            if (antiNode == nullptr) {
                OP_LOGW(kFusedOpType, "No AntiQuant input found on Add node");
                return false;
            }
            ctx.addInputIndex = 1 - addChainPort;
            ctx.nodeAntiquant = antiNode;
        } else if (mulPreTypeStr == kOpTypeAntiQuant) {
            ctx.nodeAntiquant = mulChainNode;
        } else {
            return false;
        }
    } else if (x2TypeStr == kOpTypeAntiQuant) {
        ctx.nodeAntiquant = x2Input;
    } else {
        return false;
    }
    if (ctx.nodeAntiquant == nullptr) {
        return false;
    }
    ge::AscendString antiType;
    if (ctx.nodeAntiquant->GetType(antiType) != ge::GRAPH_SUCCESS) {
        return false;
    }
    if (std::string(antiType.GetString()) != kOpTypeAntiQuant) {
        return false;
    }
    if (ctx.nodeAdd != nullptr && ctx.nodeMul == nullptr) {
        OP_LOGW(kFusedOpType, "Add node must exist with Mul node.");
        return false;
    }
    return true;
}

static bool MatchAndValidateFusion(const ge::GNode& matmulNode, bool supportV1, FusionCtx& ctx)
{
    ctx.supportV1 = supportV1;
    if (!MatchPattern(matmulNode, ctx)) {
        return false;
    }
    if (ctx.nodeAntiquant == nullptr) {
        OP_LOGW(kFusedOpType, "AscendAntiQuant node can not be null.");
        return false;
    }
    if (ctx.nodeMatmul == nullptr) {
        OP_LOGW(kFusedOpType, "MatMul node can not be null.");
        return false;
    }
    if (ctx.nodeAdd != nullptr && ctx.nodeMul == nullptr) {
        OP_LOGW(kFusedOpType, "Add node must exist with Mul node.");
        return false;
    }
    if (!CheckNodeDtype(*ctx.nodeMatmul, *ctx.nodeAntiquant)) {
        OP_LOGW(kFusedOpType, "The input or output dtype does not support fusion");
        return false;
    }
    if (!CheckNodeShape(*ctx.nodeMatmul, *ctx.nodeAntiquant, ctx)) {
        OP_LOGW(kFusedOpType, "The input or output shape does not support fusion");
        return false;
    }
    if (!CheckVectorNode(ctx)) {
        OP_LOGW(kFusedOpType, "The vector node does not support fusion");
        return false;
    }
    return true;
}

// Build the replacement subgraph: WeightQuantBatchMatmulV2 with inputs x, weight, pre-computed
// antiquant_scale/offset consts, and optional bias.
// Both SetProducerOutputDesc and SetConsumerInputDesc are needed because new opbase no longer
// auto-propagates output desc to consumer input desc.
static ge::fusion::GraphUniqPtr BuildReplacementGraph(const FusionCtx& ctx, const uint16_t* antiquantScale,
                                                      const uint16_t* antiquantOffset, int64_t scaleN, int64_t offsetN)
{
    ge::TensorDesc xDesc;
    if (ctx.nodeMatmul->GetInputDesc(kXPort, xDesc) != ge::GRAPH_SUCCESS) {
        OP_LOGW(kFusedOpType, "Failed to get input x desc of matmul node");
        return nullptr;
    }
    ge::TensorDesc wDesc;
    if (ctx.nodeAntiquant->GetInputDesc(0, wDesc) != ge::GRAPH_SUCCESS) {
        OP_LOGW(kFusedOpType, "Failed to get input desc of AscendAntiQuant node");
        return nullptr;
    }
    ge::TensorDesc outDesc;
    if (ctx.nodeMatmul->GetOutputDesc(0, outDesc) != ge::GRAPH_SUCCESS) {
        OP_LOGW(kFusedOpType, "Failed to get output desc of matmul node");
        return nullptr;
    }

    auto builder = ge::es::EsGraphBuilder("replacement");
    auto rX = builder.CreateInput(0, "x", xDesc.GetDataType(), xDesc.GetFormat(), xDesc.GetShape().GetDims());
    auto rW = builder.CreateInput(1, "weight", wDesc.GetDataType(), wDesc.GetFormat(), wDesc.GetShape().GetDims());

    std::vector<uint16_t> scaleVec(antiquantScale, antiquantScale + scaleN);
    std::vector<uint16_t> offsetVec(antiquantOffset, antiquantOffset + offsetN);
    auto rScale = builder.CreateConst(scaleVec, {scaleN}, ge::DT_FLOAT16, ge::FORMAT_ND);
    auto rOffset = builder.CreateConst(offsetVec, {offsetN}, ge::DT_FLOAT16, ge::FORMAT_ND);

    ge::es::EsTensorHolder rBias(nullptr);
    bool hasBias = static_cast<int64_t>(ctx.nodeMatmul->GetInputsSize()) == kBiasIndex;
    if (hasBias) {
        ge::TensorDesc biasDesc;
        if (ctx.nodeMatmul->GetInputDesc(kBiasIndex - 1, biasDesc) != ge::GRAPH_SUCCESS) {
            OP_LOGW(kFusedOpType, "Failed to get input bias desc of matmul node");
            return nullptr;
        }
        rBias = builder.CreateInput(2, "bias", biasDesc.GetDataType(), biasDesc.GetFormat(),
                                    biasDesc.GetShape().GetDims());
    }

    auto rOut = ge::es::WeightQuantBatchMatmulV2(rX, rW, rScale, rOffset, nullptr, nullptr, rBias, ctx.mmTransX1,
                                                 ctx.mmTransX2);

    SetProducerOutputDesc(rX.GetProducer(), xDesc.GetDataType(), xDesc.GetShape(), xDesc.GetFormat());
    SetProducerOutputDesc(rW.GetProducer(), wDesc.GetDataType(), wDesc.GetShape(), wDesc.GetFormat());
    SetProducerOutputDesc(rScale.GetProducer(), ge::DT_FLOAT16, ge::Shape({scaleN}), ge::FORMAT_ND);
    SetProducerOutputDesc(rOffset.GetProducer(), ge::DT_FLOAT16, ge::Shape({offsetN}), ge::FORMAT_ND);
    if (hasBias) {
        ge::TensorDesc biasDesc;
        if (ctx.nodeMatmul->GetInputDesc(kBiasIndex - 1, biasDesc) != ge::GRAPH_SUCCESS) {
            OP_LOGW(kFusedOpType, "Failed to get input bias desc of matmul node");
            return nullptr;
        }
        SetProducerOutputDesc(rBias.GetProducer(), biasDesc.GetDataType(), biasDesc.GetShape(), biasDesc.GetFormat());
    }
    SetProducerOutputDesc(rOut.GetProducer(), outDesc.GetDataType(), outDesc.GetShape(), outDesc.GetFormat());

    ge::GNode* fusedNode = rOut.GetProducer();
    SetConsumerInputDesc(fusedNode, kXPort, xDesc.GetDataType(), xDesc.GetShape(), xDesc.GetFormat());
    SetConsumerInputDesc(fusedNode, kWeightPort, wDesc.GetDataType(), wDesc.GetShape(), wDesc.GetFormat());
    SetConsumerInputDesc(fusedNode, kAntiQuantScalePort, ge::DT_FLOAT16, ge::Shape({scaleN}), ge::FORMAT_ND);
    SetConsumerInputDesc(fusedNode, kAntiQuantOffsetPort, ge::DT_FLOAT16, ge::Shape({offsetN}), ge::FORMAT_ND);
    if (hasBias) {
        ge::TensorDesc biasDesc;
        if (ctx.nodeMatmul->GetInputDesc(kBiasIndex - 1, biasDesc) != ge::GRAPH_SUCCESS) {
            OP_LOGW(kFusedOpType, "Failed to get input bias desc of matmul node");
            return nullptr;
        }
        SetConsumerInputDesc(fusedNode, kBiasPort, biasDesc.GetDataType(), biasDesc.GetShape(), biasDesc.GetFormat());
    }

    return builder.BuildAndReset({rOut});
}

static std::unique_ptr<ge::fusion::SubgraphBoundary> BuildBoundary(const FusionCtx& ctx)
{
    auto boundary = std::make_unique<ge::fusion::SubgraphBoundary>();

    ge::fusion::SubgraphInput input0;
    input0.AddInput({*ctx.nodeMatmul, kXPort});
    if (boundary->AddInput(0, std::move(input0)) != ge::SUCCESS) {
        OP_LOGW(kFusedOpType, "Failed to add subgraph input 0");
        return nullptr;
    }

    ge::fusion::SubgraphInput input1;
    input1.AddInput({*ctx.nodeAntiquant, 0});
    if (boundary->AddInput(1, std::move(input1)) != ge::SUCCESS) {
        OP_LOGW(kFusedOpType, "Failed to add subgraph input 1");
        return nullptr;
    }

    bool hasBias = static_cast<int64_t>(ctx.nodeMatmul->GetInputsSize()) == kBiasIndex;
    if (hasBias) {
        ge::fusion::SubgraphInput input2;
        input2.AddInput({*ctx.nodeMatmul, kBiasIndex - 1});
        if (boundary->AddInput(2, std::move(input2)) != ge::SUCCESS) {
            OP_LOGW(kFusedOpType, "Failed to add subgraph input 2");
            return nullptr;
        }
    }

    ge::fusion::SubgraphOutput output({*ctx.nodeMatmul, 0});
    if (boundary->AddOutput(0, std::move(output)) != ge::SUCCESS) {
        OP_LOGW(kFusedOpType, "Failed to add subgraph output 0");
        return nullptr;
    }

    return boundary;
}

// Collect old nodes to remove after fusion: matmul, antiquant, add, mul, and their const inputs.
// Const nodes are only removed if they have no other consumers.
static bool CollectNodesToRemove(const ge::GraphPtr& graph, const FusionCtx& ctx,
                                 std::vector<ge::GNode>& nodesBeforeFuse, std::vector<ge::GNodePtr>& nodesToRemove)
{
    auto addNode = [&](const ge::GNodePtr& node) {
        if (node == nullptr) {
            return;
        }
        if (std::find(nodesToRemove.begin(), nodesToRemove.end(), node) != nodesToRemove.end()) {
            return;
        }
        nodesToRemove.emplace_back(node);
        nodesBeforeFuse.emplace_back(*node);
    };

    addNode(ctx.nodeMatmul);
    addNode(ctx.nodeAntiquant);
    if (ctx.nodeMul != nullptr) {
        addNode(ctx.nodeMul);
        auto mulConstNode = GetInputNode(*ctx.nodeMul, ctx.mulInputIndex);
        if (mulConstNode != nullptr) {
            auto mulConsumers = mulConstNode->GetOutDataNodesAndPortIndexs(0);
            if (mulConsumers.size() <= 1) {
                addNode(mulConstNode);
            }
        }
    }
    if (ctx.nodeAdd != nullptr) {
        addNode(ctx.nodeAdd);
        auto addConstNode = GetInputNode(*ctx.nodeAdd, ctx.addInputIndex);
        if (addConstNode != nullptr) {
            auto addConsumers = addConstNode->GetOutDataNodesAndPortIndexs(0);
            if (addConsumers.size() <= 1) {
                addNode(addConstNode);
            }
        }
    }
    return true;
}

static ge::Status ProcessSingleNodeFusion(const ge::GraphPtr& graph, const ge::GNode& matmulNode, bool supportV1,
                                          ge::CustomPassContext& passContext)
{
    OP_LOGI(kFusedOpType, "Nodes matched the pattern antiquant_add_mul_matmul, begin fusing");
    FusionCtx ctx;
    if (!MatchAndValidateFusion(matmulNode, supportV1, ctx)) {
        OP_LOGW(kFusedOpType, "The node does not support fusion");
        return ge::GRAPH_NOT_CHANGED;
    }

    int64_t scaleN = 1;
    int64_t offsetN = 1;
    if (ctx.nodeMul != nullptr) {
        ge::TensorDesc mulInputDesc;
        if (ctx.nodeMul->GetInputDesc(ctx.mulInputIndex, mulInputDesc) != ge::GRAPH_SUCCESS) {
            OP_LOGW(kFusedOpType, "Failed to get input desc of Mul");
            return ge::GRAPH_NOT_CHANGED;
        }
        auto mulInputShape = mulInputDesc.GetOriginShape().GetDims();
        scaleN = std::accumulate(mulInputShape.cbegin(), mulInputShape.cend(), static_cast<int64_t>(1),
                                 std::multiplies<int64_t>());
        if (scaleN != 1 && scaleN != ctx.nDim) {
            OP_LOGW(kFusedOpType, "scale_n only supports 1 or n_dim while scale_n and n_dim are %ld and %ld", scaleN,
                    ctx.nDim);
            return ge::GRAPH_NOT_CHANGED;
        }
    }
    if (ctx.nodeAdd != nullptr) {
        ge::TensorDesc addInputDesc;
        if (ctx.nodeAdd->GetInputDesc(ctx.addInputIndex, addInputDesc) != ge::GRAPH_SUCCESS) {
            OP_LOGW(kFusedOpType, "Failed to get input desc of Add");
            return ge::GRAPH_NOT_CHANGED;
        }
        auto addInputShape = addInputDesc.GetOriginShape().GetDims();
        offsetN = std::accumulate(addInputShape.cbegin(), addInputShape.cend(), static_cast<int64_t>(1),
                                  std::multiplies<int64_t>());
        if (offsetN != 1 && offsetN != ctx.nDim) {
            OP_LOGW(kFusedOpType, "offset_n only supports 1 or n_dim while offset_n and n_dim are %ld and %ld", offsetN,
                    ctx.nDim);
            return ge::GRAPH_NOT_CHANGED;
        }
    }

    std::unique_ptr<uint16_t[]> antiquantScale(new (std::nothrow) uint16_t[scaleN]());
    std::unique_ptr<uint16_t[]> antiquantOffset(new (std::nothrow) uint16_t[offsetN]());
    if (antiquantScale == nullptr || antiquantOffset == nullptr) {
        OP_LOGW(kFusedOpType, "Failed to allocate memory for antiquant_scale/offset");
        return ge::GRAPH_NOT_CHANGED;
    }
    if (!CalAntiQuantPara(ctx, antiquantScale.get(), antiquantOffset.get(), scaleN, offsetN)) {
        OP_LOGW(kFusedOpType, "Failed to calculate antiquant parameters");
        return ge::GRAPH_NOT_CHANGED;
    }

    auto replacement = BuildReplacementGraph(ctx, antiquantScale.get(), antiquantOffset.get(), scaleN, offsetN);
    if (replacement == nullptr) {
        OP_LOGW(kFusedOpType, "Failed to build replacement graph");
        return ge::GRAPH_NOT_CHANGED;
    }

    std::vector<ge::GNode> nodesBeforeFuse;
    std::vector<ge::GNodePtr> nodesToRemove;
    if (!CollectNodesToRemove(graph, ctx, nodesBeforeFuse, nodesToRemove)) {
        OP_LOGW(kFusedOpType, "Failed to collect nodes to remove");
        return ge::GRAPH_NOT_CHANGED;
    }

    auto boundary = BuildBoundary(ctx);
    if (boundary == nullptr) {
        OP_LOGW(kFusedOpType, "Failed to build boundary");
        return ge::GRAPH_NOT_CHANGED;
    }

    passContext.SetPassName(ge::AscendString(kFusedOpType));
#if GE_COMPILER_VERSION_NUM >= 90100000
    auto replaceStatus = ge::fusion::SubgraphRewriter::Replace(*boundary, std::move(*replacement), passContext);
#else
    auto replaceStatus = ge::fusion::SubgraphRewriter::Replace(*boundary, std::move(*replacement));
#endif
    if (replaceStatus != ge::SUCCESS) {
        OP_LOGW(kFusedOpType, "SubgraphRewriter::Replace failed");
        return ge::GRAPH_NOT_CHANGED;
    }

    for (auto& node : nodesToRemove) {
        if (graph->RemoveNode(*node) != ge::GRAPH_SUCCESS) {
            OP_LOGW(kFusedOpType, "Failed to remove node after fusion");
        }
    }

    OP_LOGI(kFusedOpType, "Nodes matched the pattern, fusion succeeded");
    return ge::SUCCESS;
}

ge::Status AntiQuantMatMulFusionPass::Run(ge::GraphPtr& graph, ge::CustomPassContext& passContext)
{
    bool supportV1 = true;
    if (!CheckPlatform(supportV1)) {
        OP_LOGW(kFusedOpType, "The platform does not support fusion");
        return ge::GRAPH_NOT_CHANGED;
    }
    if (graph == nullptr || !graph->IsValid()) {
        OP_LOGW(kFusedOpType, "Graph is null or invalid, skip fusion pass.");
        return ge::GRAPH_NOT_CHANGED;
    }
    OP_LOGD(kFusedOpType, "Enter AntiQuantMatMulFusionPass");

    // Collect MatMul nodes first, then process separately
    std::vector<ge::GNode> matmulNodes;
    for (auto& node : graph->GetDirectNode()) {
        ge::AscendString opType;
        if (node.GetType(opType) != ge::GRAPH_SUCCESS) {
            continue;
        }
        if (!IsMatMulType(opType)) {
            continue;
        }
        matmulNodes.emplace_back(node);
    }
    if (matmulNodes.empty()) {
        OP_LOGD(kFusedOpType, "No matched MatMul node found, exit fusion pass");
        return ge::GRAPH_NOT_CHANGED;
    }
    OP_LOGD(kFusedOpType, "Found %zu MatMul nodes to check", matmulNodes.size());
    bool changed = false;
    for (auto& node : matmulNodes) {
        auto status = ProcessSingleNodeFusion(graph, node, supportV1, passContext);
        if (status == ge::SUCCESS) {
            changed = true;
            continue;
        }
        if (status != ge::GRAPH_NOT_CHANGED) {
            return status;
        }
    }
    OP_LOGD(kFusedOpType, "AntiQuantMatMulFusionPass completed");
    return changed ? ge::SUCCESS : ge::GRAPH_NOT_CHANGED;
}

#if GE_COMPILER_VERSION_NUM >= 90100000
REG_FUSION_PASS(AntiQuantMatMulFusionPass)
    .Stage(weight_quant::IsGeVersionSupported() ? ge::CustomPassStage::kCompatibleInherited :
                                                  ge::CustomPassStage::kAfterInferShape);
#endif

} // namespace ops

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
 * \file max_pool_fusion_pass.cpp
 * \brief MaxPool fusion into MaxPoolV3 pass
 *   (MaxPool --> MaxPoolV3)
 *
 *         x                            x
 *         |                            |
 *      MaxPool     ==>          MaxPoolV3
 *         |                            |
 *         y                            y
 */

#include "max_pool_fusion_pass.h"

#include <algorithm>
#include <array>
#include <string>
#include <vector>

#include "acl/acl_rt.h"
#include "common/inc/error_util.h"
#include "version/ge-compiler_version.h"
#include "es_nn_ops.h"
#include "ge/compliant_node_builder.h"
#include "ge/es_graph_builder.h"
#include "ge/ge_utils.h"
#include "platform/platform_info.h"

// D1 scenario: uses kCompatibleInherited stage (9.0.0+).
// Strategy: compile-time macro guard + runtime version check + overall silence.
#define GE_COMPILER_VERSION_900 90000000

using namespace ge;
using namespace fe;
using namespace fusion;

namespace ops {
namespace {
const std::string kPassName = "MaxPoolFusionPass";
const std::array<const char*, 1> kSourceOpTypes = {"MaxPool"};
const int64_t kCaptureInput = 0L;
const int64_t kCapturePool = 1L;
const size_t kShapeAttrSize = 4U;

inline static bool IsRegbasePlatform()
{
    PlatformInfo info;
    OptionalInfo optInfo;
    OP_LOGE_IF(PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(info, optInfo) != SUCCESS, false,
               kPassName.c_str(), "Get platform_info failed.");
    const std::string socVersion = info.str_info.short_soc_version;
    bool isRegbasePlatform = (socVersion == "Ascend950" || socVersion == "MC62");
    OPS_LOG_D(kPassName.c_str(), "Platform short soc: %s, is_regbase: %d", socVersion.c_str(), isRegbasePlatform);
    return isRegbasePlatform;
}

bool IsSupportedDtype(const DataType dtype)
{
    static const std::initializer_list<DataType> kSupportedDtypes = {DT_FLOAT, DT_FLOAT16, DT_BF16,  DT_INT8,  DT_INT16,
                                                                     DT_INT32, DT_INT64,   DT_UINT8, DT_UINT16};
    return std::find(kSupportedDtypes.begin(), kSupportedDtypes.end(), dtype) != kSupportedDtypes.end();
}

bool GetAttrVector4(const GNode& node, const char* attrName, std::vector<int64_t>& values)
{
    if (node.GetAttr(attrName, values) != SUCCESS) {
        OPS_LOG_D(kPassName.c_str(), "Get attr %s failed.", attrName);
        return false;
    }
    if (values.size() != kShapeAttrSize) {
        OPS_LOG_D(kPassName.c_str(), "Attr %s size %zu is invalid.", attrName, values.size());
        return false;
    }
    return true;
}

es::EsTensorHolder CreatePatternPool(es::EsGraphBuilder& graphBuilder, const char* opType, const std::string& nodeName,
                                     const es::EsTensorHolder& x)
{
    auto* graph = graphBuilder.GetCGraphBuilder()->GetGraph();
    auto pool = es::CompliantNodeBuilder(graph)
                    .OpType(opType)
                    .Name(nodeName.c_str())
                    .IrDefInputs({{"x", es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
                    .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                    .Build();

    OP_LOGE_IF(
        es::AddEdgeAndUpdatePeerDesc(*graph, *x.GetProducer(), x.GetProducerOutIndex(), pool, 0) != GRAPH_SUCCESS,
        es::EsTensorHolder(), kPassName.c_str(), "AddEdgeAndUpdatePeerDesc failed.");

    auto* yHolder = graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(pool, 0);
    return es::EsTensorHolder(yHolder);
}
} // namespace

std::vector<PatternUniqPtr> MaxPoolFusionPass::Patterns()
{
    std::vector<PatternUniqPtr> patterns;
    for (const char* opType : kSourceOpTypes) {
        auto graphBuilder = es::EsGraphBuilder(opType);
        auto x = graphBuilder.CreateInput(0, "x");
        auto y = CreatePatternPool(graphBuilder, opType, std::string(opType) + "Pattern", x);
        auto graph = graphBuilder.BuildAndReset({y});
        auto pattern = std::make_unique<Pattern>(std::move(*graph));
        pattern->CaptureTensor({*x.GetProducer(), 0}).CaptureTensor({*y.GetProducer(), 0});
        patterns.emplace_back(std::move(pattern));
    }
    return patterns;
}

bool MaxPoolFusionPass::MeetRequirements(const std::unique_ptr<MatchResult>& matchResult)
{
    int32_t version = 0;
    char geCompilerName[] = "ge_compiler";
    aclsysGetVersionNum(geCompilerName, &version);
    OPS_LOG_D(kPassName.c_str(), "GE compiler version num: %d", version);
    if (version < GE_COMPILER_VERSION_900) {
        return false;
    }
    if (!IsRegbasePlatform()) {
        return false;
    }

    NodeIo inputIo;
    OP_LOGE_IF(matchResult->GetCapturedTensor(kCaptureInput, inputIo) != SUCCESS, false, kPassName.c_str(),
               "Get captured input failed.");
    TensorDesc inputDesc;
    OP_LOGE_IF(inputIo.node.GetOutputDesc(inputIo.index, inputDesc) != SUCCESS, false, kPassName.c_str(),
               "Get input desc failed.");
    if (!IsSupportedDtype(inputDesc.GetDataType())) {
        return false;
    }

    NodeIo poolIo;
    OP_LOGE_IF(matchResult->GetCapturedTensor(kCapturePool, poolIo) != SUCCESS, false, kPassName.c_str(),
               "Get captured pool failed.");
    GNode sourceNode = poolIo.node;

    std::vector<int64_t> ksize;
    if (!GetAttrVector4(sourceNode, "ksize", ksize)) {
        return false;
    }
    std::vector<int64_t> strides;
    if (!GetAttrVector4(sourceNode, "strides", strides)) {
        return false;
    }
    AscendString paddingMode;
    if (sourceNode.GetAttr("padding", paddingMode) != SUCCESS) {
        return false;
    }
    const std::string padding = paddingMode.GetString();
    if (padding != "SAME" && padding != "VALID") {
        return false;
    }

    Format inputFormat = inputDesc.GetFormat();
    if (inputFormat != FORMAT_NCHW && inputFormat != FORMAT_NHWC) {
        return false;
    }
    return true;
}

GraphUniqPtr MaxPoolFusionPass::Replacement(const std::unique_ptr<MatchResult>& matchResult)
{
    OPS_LOG_D(kPassName.c_str(), "Enter Replacement for MaxPoolFusionPass");

    NodeIo inputIo;
    OP_LOGE_IF(matchResult->GetCapturedTensor(kCaptureInput, inputIo) != SUCCESS, nullptr, kPassName.c_str(),
               "Get captured input failed.");
    TensorDesc inputDesc;
    OP_LOGE_IF(inputIo.node.GetOutputDesc(inputIo.index, inputDesc) != SUCCESS, nullptr, kPassName.c_str(),
               "Get input desc failed.");

    NodeIo poolIo;
    OP_LOGE_IF(matchResult->GetCapturedTensor(kCapturePool, poolIo) != SUCCESS, nullptr, kPassName.c_str(),
               "Get captured pool failed.");
    GNode sourceNode = poolIo.node;

    std::vector<int64_t> ksize;
    OP_LOGE_IF(!GetAttrVector4(sourceNode, "ksize", ksize), nullptr, kPassName.c_str(), "Get ksize failed.");
    std::vector<int64_t> strides;
    OP_LOGE_IF(!GetAttrVector4(sourceNode, "strides", strides), nullptr, kPassName.c_str(), "Get strides failed.");

    AscendString paddingMode;
    OP_LOGE_IF(sourceNode.GetAttr("padding", paddingMode) != SUCCESS, nullptr, kPassName.c_str(),
               "Get padding failed.");

    std::vector<int64_t> pads = {0, 0, 0, 0};
    const char* dataFormat = "NCHW";
    if (inputDesc.GetFormat() == FORMAT_NHWC) {
        dataFormat = "NHWC";
    }

    auto graphBuilder = es::EsGraphBuilder("replacement");
    auto repX = graphBuilder.CreateInput(0, "x", inputDesc.GetDataType(), inputDesc.GetFormat(),
                                         inputDesc.GetShape().GetDims());
    auto repY = es::MaxPoolV3(repX, ksize, strides, paddingMode.GetString(), pads, dataFormat, false, false);

    TensorDesc outputDesc;
    OP_LOGE_IF(sourceNode.GetOutputDesc(0, outputDesc) != SUCCESS, nullptr, kPassName.c_str(),
               "Get output desc failed.");
    auto outNode = repY.GetProducer();
    outNode->UpdateOutputDesc(0, outputDesc);
    outNode->UpdateInputDesc(0, inputDesc);

    auto replaceGraph = graphBuilder.BuildAndReset({repY});
    return replaceGraph;
}

#if GE_COMPILER_VERSION_NUM >= GE_COMPILER_VERSION_900
namespace {
CustomPassStage GetMaxPoolFusionPassStage()
{
    int32_t version = 0;
    char geCompilerName[] = "ge_compiler";
    aclsysGetVersionNum(geCompilerName, &version);
    if (version >= GE_COMPILER_VERSION_900) {
        return CustomPassStage::kCompatibleInherited;
    }
    return CustomPassStage::kBeforeInferShape;
}
} // namespace
REG_FUSION_PASS(MaxPoolFusionPass).Stage(GetMaxPoolFusionPassStage());
#endif

} // namespace ops

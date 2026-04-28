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
 * \file max_pool_grad_with_argmax_v3_fusion_pass.cpp
 * \brief MaxPoolGradWithArgmax V0/V1/V2 fusion into MaxPoolGradWithArgmaxV3 pass
 *   (MaxPoolGradWithArgmax/MaxPoolGradWithArgmaxV1/MaxPoolGradWithArgmaxV2 --> MaxPoolGradWithArgmaxV3)
 *
 * case1:
 *    x    grad  argmax              x    grad  argmax
 *     \     |     /                  \     |     /
 *  MaxPoolGradWithArgmax   ==>  MaxPoolGradWithArgmaxV3
 *           |                              |
 *           y                              y
 *
 * case2:
 *    x    grad  argmax              x    grad  argmax
 *     \     |     /                  \     |     /
 *  MaxPoolGradWithArgmaxV1  ==>  MaxPoolGradWithArgmaxV3
 *           |                              |
 *           y                              y
 *
 * case3:
 *    x    grad  argmax              x    grad  argmax
 *     \     |     /                  \     |     /
 *  MaxPoolGradWithArgmaxV2  ==>  MaxPoolGradWithArgmaxV3
 *           |                              |
 *           y                              y
 */

#include "max_pool_grad_with_argmax_v3_fusion_pass.h"

#include <algorithm>
#include <array>
#include <string>
#include <vector>

#include "common/inc/error_util.h"
#include "es_nn_ops.h"
#include "ge/compliant_node_builder.h"
#include "ge/es_graph_builder.h"
#include "ge/ge_utils.h"
#include "platform/platform_info.h"

using namespace ge;
using namespace fe;
using namespace fusion;

namespace ops {
namespace {
const std::string kPassName = "MaxPoolGradWithArgmaxV3FusionPass";
const std::array<const char*, 3> kSourceOpTypes = {
    "MaxPoolGradWithArgmax", "MaxPoolGradWithArgmaxV1", "MaxPoolGradWithArgmaxV2"};
const int64_t kIdxInput = 0L;
const int64_t kIdxGrad = 1L;
const int64_t kIdxPool = 2L;
const size_t kListSize4 = 4U;

bool CheckPlatformSupport()
{
    PlatformInfo platformInfo;
    OptionalInfo optionalInfo;
    OP_LOGE_IF(
        PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platformInfo, optionalInfo) != SUCCESS, false,
        kPassName.c_str(), "Get platform_info failed.");
    const std::string soc = platformInfo.str_info.short_soc_version;
    const bool isPlatform91093 = (soc == "Ascend910_93");
    const bool isPlatform950 = (soc == "Ascend950");
    if (!isPlatform91093 && !isPlatform950) {
        OPS_LOG_D(kPassName.c_str(), "Platform %s is not supported.", soc.c_str());
        return false;
    }
    return true;
}

bool ValidateInputDtype(const DataType dtype)
{
    static const std::initializer_list<DataType> kSupportedDtypes = {DT_FLOAT, DT_FLOAT16, DT_BF16};
    return std::find(kSupportedDtypes.begin(), kSupportedDtypes.end(), dtype) != kSupportedDtypes.end();
}

bool GetAttrList4(const GNode& node, const char* attrName, std::vector<int64_t>& values)
{
    if (node.GetAttr(attrName, values) != SUCCESS) {
        OPS_LOG_D(kPassName.c_str(), "Get attr %s failed.", attrName);
        return false;
    }
    if (values.size() != kListSize4) {
        OPS_LOG_D(kPassName.c_str(), "Attr %s size %zu is invalid.", attrName, values.size());
        return false;
    }
    return true;
}

bool CalcPaddingSame(
    const int64_t inputSize, const int64_t ksize, const int64_t stride, int64_t& padBegin, int64_t& padEnd)
{
    if (stride == 0) {
        return false;
    }
    const int64_t outputSize = (inputSize + stride - 1) / stride;
    const int64_t totalPad = std::max(static_cast<int64_t>(0), (outputSize - 1) * stride + ksize - inputSize);
    padBegin = totalPad / 2;
    padEnd = totalPad - padBegin;
    return true;
}

void ExtractSpatialIdx(const Format inputFormat, const bool isV2, size_t& hIdx, size_t& wIdx)
{
    const bool isNchw = (inputFormat == FORMAT_NCHW);
    hIdx = (isNchw && isV2) ? 2U : 1U;
    wIdx = (isNchw && isV2) ? 3U : 2U;
}

bool ConvertPadsV0(
    const GNode& sourceNode, const bool isNchw, const std::vector<int64_t>& inputDims,
    const std::vector<int64_t>& ksizeV3, const std::vector<int64_t>& stridesV3, std::vector<int64_t>& padsV3)
{
    AscendString paddingMode;
    if (sourceNode.GetAttr("padding", paddingMode) != SUCCESS) {
        OPS_LOG_D(kPassName.c_str(), "Get attr padding failed.");
        return false;
    }
    const std::string padding = paddingMode.GetString();
    if (padding == "VALID") {
        padsV3 = {0, 0};
    } else if (padding == "SAME") {
        const size_t hDim = isNchw ? 2U : 1U;
        const size_t wDim = isNchw ? 3U : 2U;
        int64_t padHBegin = 0;
        int64_t padHEnd = 0;
        int64_t padWBegin = 0;
        int64_t padWEnd = 0;
        if (!CalcPaddingSame(inputDims[hDim], ksizeV3[0], stridesV3[0], padHBegin, padHEnd) ||
            !CalcPaddingSame(inputDims[wDim], ksizeV3[1], stridesV3[1], padWBegin, padWEnd)) {
            return false;
        }
        padsV3 = {std::max(padHBegin, padHEnd), std::max(padWBegin, padWEnd)};
    } else {
        OPS_LOG_D(kPassName.c_str(), "Padding mode %s is not supported.", padding.c_str());
        return false;
    }
    return true;
}

bool ConvertDilationV3(
    const GNode& sourceNode, const Format inputFormat, const bool isV2, std::vector<int64_t>& dilationV3,
    bool& hasDilation)
{
    std::vector<int64_t> dilation;
    hasDilation = (sourceNode.GetAttr("dilation", dilation) == SUCCESS);
    if (hasDilation) {
        if (dilation.size() != kListSize4) {
            OPS_LOG_D(kPassName.c_str(), "Attr dilation size %zu is invalid.", dilation.size());
            return false;
        }
        if ((inputFormat == FORMAT_NCHW) && isV2) {
            dilationV3 = {dilation[2], dilation[3]};
        } else {
            dilationV3 = {dilation[1], dilation[2]};
        }
    }
    return true;
}

bool ConvertAttrsToV3(
    const GNode& sourceNode, const Format inputFormat, const bool isV0, const bool isV2,
    const std::vector<int64_t>& inputDims, std::vector<int64_t>& ksizeV3, std::vector<int64_t>& stridesV3,
    std::vector<int64_t>& padsV3, std::vector<int64_t>& dilationV3, AscendString& dataFormat, bool& hasDilation)
{
    std::vector<int64_t> ksize;
    if (!GetAttrList4(sourceNode, "ksize", ksize)) {
        return false;
    }
    std::vector<int64_t> strides;
    if (!GetAttrList4(sourceNode, "strides", strides)) {
        return false;
    }

    size_t hIdx = 0;
    size_t wIdx = 0;
    ExtractSpatialIdx(inputFormat, isV2, hIdx, wIdx);
    ksizeV3 = {ksize[hIdx], ksize[wIdx]};
    stridesV3 = {strides[hIdx], strides[wIdx]};

    if (isV0) {
        if (!ConvertPadsV0(sourceNode, (inputFormat == FORMAT_NCHW), inputDims, ksizeV3, stridesV3, padsV3)) {
            return false;
        }
    } else {
        std::vector<int64_t> pads;
        if (!GetAttrList4(sourceNode, "pads", pads)) {
            return false;
        }
        padsV3 = {pads[1], pads[2]};
        if (pads[0] > 1 || pads[3] > 1) {
            OPS_LOG_D(
                kPassName.c_str(), "Pads {%ld, %ld, %ld, %ld} layout may be invalid, expected [N, pad_h, pad_w, C].",
                pads[0], pads[1], pads[2], pads[3]);
            return false;
        }
    }

    if (!ConvertDilationV3(sourceNode, inputFormat, isV2, dilationV3, hasDilation)) {
        return false;
    }

    if (inputFormat == FORMAT_NHWC) {
        dataFormat = "NHWC";
    } else if (inputFormat == FORMAT_NCHW) {
        dataFormat = "NCHW";
    } else {
        OPS_LOG_D(kPassName.c_str(), "Format %d is not supported.", inputFormat);
        return false;
    }
    return true;
}

es::EsTensorHolder CreatePatternPoolGrad(
    es::EsGraphBuilder& graphBuilder, const char* opType, const std::string& nodeName, const es::EsTensorHolder& x,
    const es::EsTensorHolder& grad, const es::EsTensorHolder& argmax)
{
    auto* graph = graphBuilder.GetCGraphBuilder()->GetGraph();
    auto pool = es::CompliantNodeBuilder(graph)
                    .OpType(opType)
                    .Name(nodeName.c_str())
                    .IrDefInputs(
                        {{"x", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                         {"grad", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                         {"argmax", es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
                    .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                    .Build();

    OP_LOGE_IF(
        es::AddEdgeAndUpdatePeerDesc(*graph, *x.GetProducer(), x.GetProducerOutIndex(), pool, 0) != GRAPH_SUCCESS,
        es::EsTensorHolder(), kPassName.c_str(), "AddEdgeAndUpdatePeerDesc for x failed.");
    OP_LOGE_IF(
        es::AddEdgeAndUpdatePeerDesc(*graph, *grad.GetProducer(), grad.GetProducerOutIndex(), pool, 1) != GRAPH_SUCCESS,
        es::EsTensorHolder(), kPassName.c_str(), "AddEdgeAndUpdatePeerDesc for grad failed.");
    OP_LOGE_IF(
        es::AddEdgeAndUpdatePeerDesc(*graph, *argmax.GetProducer(), argmax.GetProducerOutIndex(), pool, 2) !=
            GRAPH_SUCCESS,
        es::EsTensorHolder(), kPassName.c_str(), "AddEdgeAndUpdatePeerDesc for argmax failed.");

    auto* yHolder = graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(pool, 0);
    return es::EsTensorHolder(yHolder);
}
} // namespace

std::vector<PatternUniqPtr> MaxPoolGradWithArgmaxV3FusionPass::Patterns()
{
    std::vector<PatternUniqPtr> patterns;
    for (const char* opType : kSourceOpTypes) {
        auto graphBuilder = es::EsGraphBuilder(opType);
        auto x = graphBuilder.CreateInput(0, "x");
        auto grad = graphBuilder.CreateInput(1, "grad");
        auto argmax = graphBuilder.CreateInput(2, "argmax");
        es::EsTensorHolder y;

        if (std::string(opType) == "MaxPoolGradWithArgmax") {
            y = es::MaxPoolGradWithArgmax(x, grad, argmax, {1, 1, 1, 1}, {1, 1, 1, 1}, "VALID");
        } else {
            y = CreatePatternPoolGrad(graphBuilder, opType, std::string(opType) + "Pattern", x, grad, argmax);
        }

        auto graph = graphBuilder.BuildAndReset({y});
        auto pattern = std::make_unique<Pattern>(std::move(*graph));
        pattern->CaptureTensor({*x.GetProducer(), 0})
            .CaptureTensor({*grad.GetProducer(), 0})
            .CaptureTensor({*y.GetProducer(), 0});
        patterns.emplace_back(std::move(pattern));
    }
    return patterns;
}

bool MaxPoolGradWithArgmaxV3FusionPass::MeetRequirements(const std::unique_ptr<MatchResult>& matchResult)
{
    if (!CheckPlatformSupport()) {
        return false;
    }

    NodeIo inputIo;
    OP_LOGE_IF(
        matchResult->GetCapturedTensor(kIdxInput, inputIo) != SUCCESS, false, kPassName.c_str(),
        "Get captured input failed.");
    TensorDesc inputDesc;
    OP_LOGE_IF(
        inputIo.node.GetOutputDesc(inputIo.index, inputDesc) != SUCCESS, false, kPassName.c_str(),
        "Get input desc failed.");
    if (!ValidateInputDtype(inputDesc.GetDataType())) {
        return false;
    }

    NodeIo poolIo;
    OP_LOGE_IF(
        matchResult->GetCapturedTensor(kIdxPool, poolIo) != SUCCESS, false, kPassName.c_str(),
        "Get captured pool failed.");
    GNode sourceNode = poolIo.node;

    AscendString type;
    sourceNode.GetType(type);
    const bool isV0 = (type == "MaxPoolGradWithArgmax");
    const bool isV2 = (type == "MaxPoolGradWithArgmaxV2");
    if (isV0) {
        OPS_LOG_D(kPassName.c_str(), "V0 MaxPoolGradWithArgmax has different argmax format, skip fusion.");
        return false;
    }

    std::vector<int64_t> dummyKsize;
    std::vector<int64_t> dummyStrides;
    std::vector<int64_t> dummyPads;
    std::vector<int64_t> dummyDilation;
    AscendString dataFormat;
    bool hasDilation = false;
    if (!ConvertAttrsToV3(
            sourceNode, inputDesc.GetFormat(), isV0, isV2, inputDesc.GetShape().GetDims(), dummyKsize, dummyStrides,
            dummyPads, dummyDilation, dataFormat, hasDilation)) {
        return false;
    }
    return true;
}

GraphUniqPtr MaxPoolGradWithArgmaxV3FusionPass::Replacement(const std::unique_ptr<MatchResult>& matchResult)
{
    NodeIo inputIo;
    OP_LOGE_IF(matchResult->GetCapturedTensor(kIdxInput, inputIo) != SUCCESS, nullptr, kPassName.c_str(), "Get captured input failed.");
    TensorDesc inputDesc;
    OP_LOGE_IF(inputIo.node.GetOutputDesc(inputIo.index, inputDesc) != SUCCESS, nullptr, kPassName.c_str(), "Get input desc failed.");

    NodeIo gradIo;
    OP_LOGE_IF(matchResult->GetCapturedTensor(kIdxGrad, gradIo) != SUCCESS, nullptr, kPassName.c_str(), "Get captured grad failed.");
    TensorDesc gradDesc;
    OP_LOGE_IF(gradIo.node.GetOutputDesc(gradIo.index, gradDesc) != SUCCESS, nullptr, kPassName.c_str(), "Get grad desc failed.");

    NodeIo poolIo;
    OP_LOGE_IF(matchResult->GetCapturedTensor(kIdxPool, poolIo) != SUCCESS, nullptr, kPassName.c_str(), "Get captured pool failed.");
    GNode sourceNode = poolIo.node;

    AscendString sourceType;
    sourceNode.GetType(sourceType);
    const bool isV0 = (sourceType == "MaxPoolGradWithArgmax");
    const bool isV2 = (sourceType == "MaxPoolGradWithArgmaxV2");

    std::vector<int64_t> ksize;
    std::vector<int64_t> strides;
    std::vector<int64_t> pads;
    std::vector<int64_t> dilation;
    AscendString dataFormat;
    bool hasDilation = false;
    if (!ConvertAttrsToV3(sourceNode, inputDesc.GetFormat(), isV0, isV2, inputDesc.GetShape().GetDims(), ksize, strides, pads, dilation, dataFormat, hasDilation)) {
        return nullptr;
    }

    bool ceilMode = false;
    (void)sourceNode.GetAttr("ceil_mode", ceilMode);
    int32_t dtypeI32 = 3;
    if (sourceNode.GetAttr("dtype", dtypeI32) != SUCCESS) {
        dtypeI32 = 3;
    }
    int64_t dtype = static_cast<int64_t>(dtypeI32);

    auto graphBuilder = es::EsGraphBuilder("replacement");
    auto repX = graphBuilder.CreateInput(0, "x", inputDesc.GetDataType(), inputDesc.GetFormat(), inputDesc.GetShape().GetDims());
    auto repGrad = graphBuilder.CreateInput(1, "grad", gradDesc.GetDataType(), gradDesc.GetFormat(), gradDesc.GetShape().GetDims());
    auto repArgmax = graphBuilder.CreateInput(2, "argmax", DT_INT32, gradDesc.GetFormat(), gradDesc.GetShape().GetDims());
    auto repY = es::MaxPoolGradWithArgmaxV3(repX, repGrad, repArgmax, ksize, strides, pads, dtype, hasDilation ? dilation : std::vector<int64_t>{1, 1}, ceilMode, dataFormat.GetString());

    TensorDesc outputDesc;
    OP_LOGE_IF(sourceNode.GetOutputDesc(0, outputDesc) != SUCCESS, nullptr, kPassName.c_str(), "Get output desc failed.");
    TensorDesc argmaxInputDesc;
    repArgmax.GetProducer()->GetOutputDesc(0, argmaxInputDesc);
    auto outNode = repY.GetProducer();
    outNode->UpdateOutputDesc(0, outputDesc);
    outNode->UpdateInputDesc(0, inputDesc);
    outNode->UpdateInputDesc(1, gradDesc);
    outNode->UpdateInputDesc(2, argmaxInputDesc);

    auto replaceGraph = graphBuilder.BuildAndReset({repY});
    return replaceGraph;
}

REG_FUSION_PASS(MaxPoolGradWithArgmaxV3FusionPass).Stage(CustomPassStage::kCompatibleInherited);
} // namespace ops

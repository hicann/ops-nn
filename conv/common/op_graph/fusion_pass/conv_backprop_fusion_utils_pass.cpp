/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "conv_backprop_fusion_utils_pass.h"
#include "log/log.h"
#include "platform/platform_info.h"

namespace ops {
using namespace ge;
using namespace ge::es;
using namespace ConvBackpropFusionUtils;

std::vector<int64_t> ConvBackpropFusionUtilsPass::CalcTransposeShape(const std::vector<int64_t>& inputShape,
                                                                     const std::vector<int32_t>& perm)
{
    std::vector<int64_t> retShape;
    for (size_t i = 0; i < perm.size() && i < inputShape.size(); ++i) {
        if (perm[i] >= 0 && static_cast<size_t>(perm[i]) < inputShape.size()) {
            retShape.push_back(inputShape[perm[i]]);
        }
    }
    return retShape;
}

void ConvBackpropFusionUtilsPass::SetPlaceholderDesc(EsTensorHolder& tensorHolder, int64_t idx, const TensorDesc& desc)
{
    auto* producer = tensorHolder.GetProducer();
    if (producer == nullptr) {
        return;
    }
    producer->UpdateOutputDesc(static_cast<uint32_t>(idx), desc);
}

bool ConvBackpropFusionUtilsPass::InWhitelist(const std::vector<int64_t>& shape,
                                              const std::vector<std::vector<int64_t>>& whitelist)
{
    return std::find(whitelist.begin(), whitelist.end(), shape) != whitelist.end();
}

bool ConvBackpropFusionUtilsPass::CheckSocAndIntrinsic(const std::map<std::string, NpuArch>& supportSocList,
                                                       NpuArch& npuArch)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optionalInfo;
    if (fe::PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platformInfo, optionalInfo) != SUCCESS) {
        return false;
    }
    const std::string soc = platformInfo.str_info.short_soc_version;
    if (supportSocList.find(soc) == supportSocList.end()) {
        return false;
    }
    npuArch = supportSocList.at(soc);
    return true;
}

bool ConvBackpropFusionUtilsPass::GetNodeName(const GNode& node, std::string& nodeName)
{
    AscendString rawNodeName;
    if (node.GetName(rawNodeName) != GRAPH_SUCCESS) {
        OP_LOGE("ConvBackpropFusionUtilsPass", "Get node name failed.");
        return false;
    }
    nodeName = std::string(rawNodeName.GetString());
    return true;
}

int64_t ConvBackpropFusionUtilsPass::GetAiCoreCount()
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optionalInfo;
    if (fe::PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platformInfo, optionalInfo) != SUCCESS) {
        OP_LOGE("ConvBackpropFusionUtilsPass", "GetPlatformInfoWithOutSocVersion failed, use default core count 0");
        return 0;
    }
    return platformInfo.soc_info.ai_core_cnt;
}

bool ConvBackpropFusionUtilsPass::CreateTransposeNode(EsGraphBuilder& builder, const TransposeNodeConfig& config,
                                                      EsTensorHolder& output, TensorDesc& outDesc,
                                                      const AscendString& opType)
{
    auto* graph = builder.GetCGraphBuilder()->GetGraph();
    OP_CHECK_IF(graph == nullptr, OP_LOGE(opType.GetString(), "create transpose node failed"), return false);

    auto* producer = config.input.GetProducer();
    OP_CHECK_IF(producer == nullptr, OP_LOGE(opType.GetString(), "input producer is nullptr in CreateTransposeNode"),
                return false);

    TensorDesc inDesc;
    producer->GetOutputDesc(config.input.GetProducerOutIndex(), inDesc);
    auto transposeNode = ge::es::CompliantNodeBuilder(graph)
                             .OpType("Transpose")
                             .Name(config.name.c_str())
                             .IrDefInputs({{"x", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                                           {"perm", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
                             .IrDefOutputs({{"y", ge::es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                             .Build();
    OP_CHECK_IF(ge::es::AddEdgeAndUpdatePeerDesc(*graph, *producer, TENSOR_DEFAULT_OUTPUT_INDEX, transposeNode,
                                                 TRANSPOSE_INPUT_X_INDEX) != GRAPH_SUCCESS,
                OP_LOGE(opType.GetString(), "Add edge for transpose input failed"), return false);
    transposeNode.UpdateInputDesc(TRANSPOSE_INPUT_X_INDEX, inDesc);
    auto permTensorHolder = builder.CreateVector(config.perm);
    auto* permTensorProducer = permTensorHolder.GetProducer();
    OP_CHECK_IF(permTensorProducer == nullptr, OP_LOGE(opType.GetString(), "perm producer is nullptr"), return false);
    OP_CHECK_IF(ge::es::AddEdgeAndUpdatePeerDesc(*graph, *permTensorProducer, TENSOR_DEFAULT_OUTPUT_INDEX,
                                                 transposeNode, TRANSPOSE_INPUT_PERM_INDEX) != GRAPH_SUCCESS,
                OP_LOGE(opType.GetString(), "Add edge for transpose perm failed"), return false);

    TensorDesc permTensorDesc;
    permTensorProducer->GetOutputDesc(TENSOR_DEFAULT_OUTPUT_INDEX, permTensorDesc);
    transposeNode.UpdateInputDesc(TRANSPOSE_INPUT_PERM_INDEX, permTensorDesc);
    outDesc.SetDataType(inDesc.GetDataType());
    auto outShape = CalcTransposeShape(inDesc.GetShape().GetDims(), config.perm);
    outDesc.SetShape(Shape(outShape));
    outDesc.SetOriginShape(Shape(outShape));
    outDesc.SetFormat(config.format);
    outDesc.SetOriginFormat(config.format);
    transposeNode.UpdateOutputDesc(TRANSPOSE_OUTPUT_Y_INDEX, outDesc);
    output = EsTensorHolder(
        builder.GetCGraphBuilder()->GetTensorHolderFromNode(transposeNode, TRANSPOSE_OUTPUT_Y_INDEX));

    return true;
}

int32_t ConvBackpropFusionUtilsPass::GetExpandAxis(ge::Format format2D)
{
    if (format2D == ge::FORMAT_NCHW) {
        return D_DIM_NCDHW_INDEX;
    } else if (format2D == ge::FORMAT_NHWC) {
        return D_DIM_NDHWC_INDEX;
    }
    return 0;
}

ge::Format ConvBackpropFusionUtilsPass::Get3DFormat(ge::Format format2D)
{
    if (format2D == ge::FORMAT_NHWC) {
        return ge::FORMAT_NDHWC;
    } else if (format2D == ge::FORMAT_NCHW) {
        return ge::FORMAT_NCDHW;
    } else if (format2D == ge::FORMAT_HWCN) {
        return ge::FORMAT_DHWCN;
    }
    return ge::FORMAT_ND;
}

std::string ConvBackpropFusionUtilsPass::Get3DDataFormatStr(const std::string& format2D)
{
    if (format2D == "NCHW") {
        return "NCDHW";
    } else if (format2D == "NHWC") {
        return "NDHWC";
    } else if (format2D == "HWCN") {
        return "DHWCN";
    }
    return format2D;
}

ge::Shape ConvBackpropFusionUtilsPass::Get3DShape(const ge::Shape& shape2D, ge::Format format2D)
{
    auto dims = shape2D.GetDims();
    if (format2D == ge::FORMAT_NHWC) {
        dims.insert(dims.begin() + D_DIM_NDHWC_INDEX, EXPAND_AXIS_DEFAULT_VALUE);
    } else if (format2D == ge::FORMAT_HWCN) {
        dims.insert(dims.begin(), EXPAND_AXIS_DEFAULT_VALUE);
    } else if (format2D == ge::FORMAT_NCHW) {
        dims.insert(dims.begin() + D_DIM_NCDHW_INDEX, EXPAND_AXIS_DEFAULT_VALUE);
    }
    return ge::Shape(dims);
}

bool ConvBackpropFusionUtilsPass::CreateUnsqueezeNode(ge::es::EsGraphBuilder& builder,
                                                      ge::es::EsTensorHolder& inputHolder,
                                                      const ge::TensorDesc& inputDesc, const std::string& nodeName,
                                                      UnsqueezeNodeInfo& outInfo, const ge::AscendString& opType)
{
    auto* graph = builder.GetCGraphBuilder()->GetGraph();
    OP_CHECK_IF(graph == nullptr, OP_LOGE(opType.GetString(), "Get graph failed in CreateUnsqueezeNode"), return false);
    auto* producer = inputHolder.GetProducer();
    OP_CHECK_IF(producer == nullptr, OP_LOGE(opType.GetString(), "Producer is nullptr"), return false);

    int32_t expandAxis = GetExpandAxis(inputDesc.GetFormat());

    outInfo.outDesc.SetFormat(Get3DFormat(inputDesc.GetFormat()));
    outInfo.outDesc.SetOriginFormat(Get3DFormat(inputDesc.GetOriginFormat()));
    outInfo.outDesc.SetDataType(inputDesc.GetDataType());
    outInfo.outDesc.SetShape(Get3DShape(inputDesc.GetShape(), inputDesc.GetFormat()));
    outInfo.outDesc.SetOriginShape(Get3DShape(inputDesc.GetOriginShape(), inputDesc.GetOriginFormat()));

    outInfo.node = ge::es::CompliantNodeBuilder(graph)
                       .OpType("Unsqueeze")
                       .Name(nodeName.c_str())
                       .IrDefInputs({{"x", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
                       .IrDefOutputs({{"y", ge::es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                       .Build();

    OP_CHECK_IF(ge::es::AddEdgeAndUpdatePeerDesc(*graph, *producer, inputHolder.GetProducerOutIndex(), outInfo.node,
                                                 0) != ge::GRAPH_SUCCESS,
                OP_LOGE(opType.GetString(), "Add edge to Unsqueeze failed"), return false);
    outInfo.node.UpdateInputDesc(0, inputDesc);
    outInfo.node.UpdateOutputDesc(0, outInfo.outDesc);
    std::vector<int64_t> axes = {static_cast<int64_t>(expandAxis)};
    outInfo.node.SetAttr("axes", axes);

    return true;
}

bool ConvBackpropFusionUtilsPass::BuildSqueezeNode(ge::es::EsGraphBuilder& builder, ge::GNode& inputNode,
                                                   const ge::TensorDesc& output3DDesc,
                                                   const ge::TensorDesc& output2DDesc,
                                                   const std::string& nodeNamePrefix, ge::GNode& outNode,
                                                   const ge::AscendString& opType)
{
    auto* graph = builder.GetCGraphBuilder()->GetGraph();
    OP_CHECK_IF(graph == nullptr, OP_LOGE(opType.GetString(), "Get graph failed in BuildSqueezeNode"), return false);
    int32_t expandOutputAxis = GetExpandAxis(output2DDesc.GetFormat());

    std::string squeezeName = nodeNamePrefix + "_Squeeze_0";
    outNode = ge::es::CompliantNodeBuilder(graph)
                  .OpType("Squeeze")
                  .Name(squeezeName.c_str())
                  .IrDefInputs({{"x", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
                  .IrDefOutputs({{"y", ge::es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                  .Build();

    OP_CHECK_IF(ge::es::AddEdgeAndUpdatePeerDesc(*graph, inputNode, static_cast<int32_t>(OUTPUT_INDEX), outNode, 0) !=
                    ge::GRAPH_SUCCESS,
                OP_LOGE(opType.GetString(), "Add edge for Squeeze failed"), return false);

    outNode.UpdateInputDesc(0, output3DDesc);
    outNode.UpdateOutputDesc(0, output2DDesc);
    std::vector<int64_t> squeezeAxis = {static_cast<int64_t>(expandOutputAxis)};
    outNode.SetAttr("axis", squeezeAxis);

    return true;
}

void ConvBackpropFusionUtilsPass::ExpandAttrs(std::vector<int64_t>& strides, std::vector<int64_t>& pads,
                                              std::vector<int64_t>& dilations, std::string& dataFormat,
                                              std::vector<int64_t>* outputPadding)
{
    if (dataFormat == "NCHW") {
        strides.insert(strides.begin() + D_DIM_NCDHW_INDEX, EXPAND_AXIS_DEFAULT_VALUE);
        dilations.insert(dilations.begin() + D_DIM_NCDHW_INDEX, EXPAND_AXIS_DEFAULT_VALUE);
        if (outputPadding != nullptr) {
            outputPadding->insert(outputPadding->begin() + D_DIM_NCDHW_INDEX, EXPAND_PAD_DEFAULT_VALUE);
        }
    } else if (dataFormat == "NHWC") {
        strides.insert(strides.begin() + D_DIM_NDHWC_INDEX, EXPAND_AXIS_DEFAULT_VALUE);
        dilations.insert(dilations.begin() + D_DIM_NDHWC_INDEX, EXPAND_AXIS_DEFAULT_VALUE);
        if (outputPadding != nullptr) {
            outputPadding->insert(outputPadding->begin() + D_DIM_NDHWC_INDEX, EXPAND_PAD_DEFAULT_VALUE);
        }
    } else if (dataFormat == "HWCN") {
        strides.insert(strides.begin(), EXPAND_AXIS_DEFAULT_VALUE);
        dilations.insert(dilations.begin(), EXPAND_AXIS_DEFAULT_VALUE);
        if (outputPadding != nullptr) {
            outputPadding->insert(outputPadding->begin(), EXPAND_PAD_DEFAULT_VALUE);
        }
    }

    pads.insert(pads.begin(), EXPAND_PAD_INSERT_COUNT, EXPAND_PAD_DEFAULT_VALUE);
    dataFormat = Get3DDataFormatStr(dataFormat);
}

void ConvBackpropFusionUtilsPass::ExpandOutputDesc(const TensorDesc& output2DDesc, TensorDesc& output3DDesc)
{
    output3DDesc.SetFormat(Get3DFormat(output2DDesc.GetFormat()));
    output3DDesc.SetOriginFormat(Get3DFormat(output2DDesc.GetOriginFormat()));
    output3DDesc.SetDataType(output2DDesc.GetDataType());
    output3DDesc.SetShape(Get3DShape(output2DDesc.GetShape(), output2DDesc.GetFormat()));
    output3DDesc.SetOriginShape(Get3DShape(output2DDesc.GetOriginShape(), output2DDesc.GetOriginFormat()));
}

} // namespace ops

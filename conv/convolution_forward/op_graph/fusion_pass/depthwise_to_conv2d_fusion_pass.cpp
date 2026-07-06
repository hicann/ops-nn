/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "depthwise_to_conv2d_fusion_pass.h"

#include "graph/graph.h"
#include "register/register_custom_pass.h"
#include "version/ge-compiler_version.h"

namespace Ops {
using namespace NN;
using namespace Conv;
using namespace ConvFusionUtils;
using namespace DepthwiseToConv2dFusion;
using namespace ge;
using namespace fusion;

void DepthwiseToConv2dFusionPass::InitMember()
{
    npuArch = NpuArch::DAV_RESV;
    isNdSoc = false;
    fmapChannel = 0;
    convDescInfo = ConvDescInfo();
    depthwiseAttrs = ConvBaseAttrs();
}

bool DepthwiseToConv2dFusionPass::CheckDynamicShape() const
{
    if (isNdSoc) {
        return true;
    }

    FUSION_PASS_CHECK(!ConvFusionUtilsPass::IsUnknownShape(convDescInfo.fmapDesc),
        OP_LOGD(FUSION_NAME, "%s only support dynamic mode on non-ND soc.", convDescInfo.nodeNameStr.c_str()),
        return false);

    return true;
}

bool DepthwiseToConv2dFusionPass::GetFmapChannel()
{
    auto originFormat = convDescInfo.fmapDesc.GetOriginFormat();
    auto inputShape = convDescInfo.fmapDesc.GetOriginShape().GetDims();
    FUSION_PASS_CHECK(inputShape.size() != MAX_DIM_NUM,
        OP_LOGE(FUSION_NAME, "%s fmap origin shape dim not equal to 4.", convDescInfo.nodeNameStr.c_str()),
        return false);

    if (originFormat == Format::FORMAT_NCHW) {
        fmapChannel = inputShape[FMAP_CHANNEL_NCHW_INDEX];
    } else if (originFormat == Format::FORMAT_NHWC) {
        fmapChannel = inputShape[FMAP_CHANNEL_NHWC_INDEX];
    } else {
        OP_LOGE(FUSION_NAME, "%s not support format [%s].", convDescInfo.nodeNameStr.c_str(),
            TypeUtils::FormatToAscendString(originFormat).GetString());
        return false;
    }

    FUSION_PASS_CHECK(fmapChannel == -1,
        OP_LOGD(FUSION_NAME, "%s not support fmapChannel is -1.", convDescInfo.nodeNameStr.c_str()), return false);

    return true;
}

bool DepthwiseToConv2dFusionPass::GetDepthwiseConvAttrs(const GNode &depthwiseNode)
{
    depthwiseAttrs = ConvBaseAttrs();
    FUSION_PASS_CHECK(depthwiseNode.GetAttr(STRIDES, depthwiseAttrs.strides) != GRAPH_SUCCESS,
        OP_LOGE(FUSION_NAME, "%s get strides failed.", convDescInfo.nodeNameStr.c_str()), return false);
    FUSION_PASS_CHECK(depthwiseNode.GetAttr(PADS, depthwiseAttrs.pads) != GRAPH_SUCCESS,
        OP_LOGE(FUSION_NAME, "%s get pads failed.", convDescInfo.nodeNameStr.c_str()), return false);
    FUSION_PASS_CHECK(depthwiseNode.GetAttr(DILATIONS, depthwiseAttrs.dilations) != GRAPH_SUCCESS,
        OP_LOGE(FUSION_NAME, "%s get dilations failed.", convDescInfo.nodeNameStr.c_str()), return false);
    FUSION_PASS_CHECK(depthwiseNode.GetAttr(DATA_FORMAT, depthwiseAttrs.dataFormat) != GRAPH_SUCCESS,
        OP_LOGE(FUSION_NAME, "%s get data_format failed.", convDescInfo.nodeNameStr.c_str()), return false);

    if (isNdSoc) {
        FUSION_PASS_CHECK(depthwiseNode.GetAttr(OFFSET_X, depthwiseAttrs.offsetX) != GRAPH_SUCCESS,
            OP_LOGE(FUSION_NAME, "%s get offset_x failed.", convDescInfo.nodeNameStr.c_str()), return false);
    }

    depthwiseNode.GetAttr(PADDING, depthwiseAttrs.padding);
    return true;
}

bool DepthwiseToConv2dFusionPass::MeetRequirements(const GNode &depthwiseNode)
{
    InitMember();

    isNdSoc = ConvFusionUtilsPass::CheckSocList(ND_SOC_LIST, npuArch);
    FUSION_PASS_CHECK_NOLOG(!ConvFusionUtilsPass::GetConvDescInfo(depthwiseNode, convDescInfo), return false);
    OP_LOGD(convDescInfo.nodeNameStr, "Begin to do DepthwiseToConv2dFusionPass.");

    FUSION_PASS_CHECK_NOLOG(!CheckDynamicShape(), return false);
    FUSION_PASS_CHECK_NOLOG(!GetFmapChannel(), return false);

    return true;
}

GraphUniqPtr DepthwiseToConv2dFusionPass::Replacement(const GNode &depthwiseNode)
{
    FUSION_PASS_CHECK_NOLOG(!GetDepthwiseConvAttrs(depthwiseNode), return nullptr);

    auto graphBuilder = es::EsGraphBuilder("replacement");
    auto [fmap, filter] = graphBuilder.CreateInputs<REQUIRED_INPUT_NUMS>();
    std::vector<es::EsTensorHolder> inputs = {fmap, filter};
    if (convDescInfo.hasBias) {
        inputs.emplace_back(graphBuilder.CreateInput(static_cast<int64_t>(INPUT_BIAS_INDEX)));
    }

    auto *replaceGraph = graphBuilder.GetCGraphBuilder()->GetGraph();
    GNode conv2dNode;
    FUSION_PASS_CHECK(!ConvFusionUtilsPass::BuildConv2dNode(
        replaceGraph, convDescInfo.nodeNameStr + "_To_Conv2D", inputs, conv2dNode),
        OP_LOGE(FUSION_NAME, "%s build Conv2D node failed.", convDescInfo.nodeNameStr.c_str()), return nullptr);

    FUSION_PASS_CHECK_NOLOG(!SetConv2dAttrs(conv2dNode, depthwiseNode), return nullptr);
    FUSION_PASS_CHECK_NOLOG(!UpdateConv2dDesc(conv2dNode), return nullptr);

    auto *yHolder = graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(conv2dNode, OUTPUT_INDEX);
    FUSION_PASS_CHECK(yHolder == nullptr,
        OP_LOGE(FUSION_NAME, "%s get Conv2D output tensor holder failed.", convDescInfo.nodeNameStr.c_str()),
        return nullptr);

    return graphBuilder.BuildAndReset({es::EsTensorHolder(yHolder)});
}

bool DepthwiseToConv2dFusionPass::SetConv2dAttrs(GNode &conv2dNode, const GNode &depthwiseNode)
{
    FUSION_PASS_CHECK(conv2dNode.SetAttr(STRIDES, depthwiseAttrs.strides) != GRAPH_SUCCESS,
        OP_LOGE(FUSION_NAME, "%s set strides failed.", convDescInfo.nodeNameStr.c_str()), return false);
    FUSION_PASS_CHECK(conv2dNode.SetAttr(PADS, depthwiseAttrs.pads) != GRAPH_SUCCESS,
        OP_LOGE(FUSION_NAME, "%s set pads failed.", convDescInfo.nodeNameStr.c_str()), return false);
    FUSION_PASS_CHECK(conv2dNode.SetAttr(DILATIONS, depthwiseAttrs.dilations) != GRAPH_SUCCESS,
        OP_LOGE(FUSION_NAME, "%s set dilations failed.", convDescInfo.nodeNameStr.c_str()), return false);
    FUSION_PASS_CHECK(conv2dNode.SetAttr(DATA_FORMAT, depthwiseAttrs.dataFormat) != GRAPH_SUCCESS,
        OP_LOGE(FUSION_NAME, "%s set data_format failed.", convDescInfo.nodeNameStr.c_str()), return false);
    FUSION_PASS_CHECK(conv2dNode.SetAttr(GROUPS, fmapChannel) != GRAPH_SUCCESS,
        OP_LOGE(FUSION_NAME, "%s set groups failed.", convDescInfo.nodeNameStr.c_str()), return false);
    if (isNdSoc) {
        FUSION_PASS_CHECK(conv2dNode.SetAttr(OFFSET_X, depthwiseAttrs.offsetX) != GRAPH_SUCCESS,
            OP_LOGE(FUSION_NAME, "%s set offset_x failed.", convDescInfo.nodeNameStr.c_str()), return false);
    }
    FUSION_PASS_CHECK(conv2dNode.SetAttr(PADDING, depthwiseAttrs.padding) != GRAPH_SUCCESS,
        OP_LOGE(FUSION_NAME, "%s set padding failed.", convDescInfo.nodeNameStr.c_str()), return false);

    int64_t opImplModeEnum = 0;
    if (depthwiseNode.GetAttr(OP_IMPL_MODE_ENUM, opImplModeEnum) == GRAPH_SUCCESS) {
        (void)conv2dNode.SetAttr(OP_IMPL_MODE_ENUM, opImplModeEnum);
    }

    return true;
}

bool DepthwiseToConv2dFusionPass::UpdateConv2dDesc(GNode &conv2dNode)
{
    FUSION_PASS_CHECK_NOLOG(!ConvFusionUtilsPass::UpdateInputDesc(&conv2dNode, convDescInfo), return false);
    FUSION_PASS_CHECK(conv2dNode.UpdateOutputDesc(OUTPUT_INDEX, convDescInfo.outputDesc) != GRAPH_SUCCESS,
        OP_LOGE(FUSION_NAME, "%s update Conv2D output tensor desc failed.", convDescInfo.nodeNameStr.c_str()),
        return false);

    return true;
}

#if GE_COMPILER_VERSION_NUM >= 90000000U
REG_DECOMPOSE_PASS(DepthwiseToConv2dFusionPass, {DEPTHWISE_CONV2D})
    .Stage(CustomPassStage::kCompatibleInherited);
#endif

} // namespace Ops

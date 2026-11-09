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
    hasPrivatePads = false;
    convDescInfo = ConvDescInfo();
    depthwiseAttrs = ConvBaseAttrs();
}

bool DepthwiseToConv2dFusionPass::CheckDynamicShape() const
{
    if (isNdSoc) {
        return true;
    }

    FUSION_PASS_CHECK(
        !ConvFusionUtilsPass::IsUnknownShape(convDescInfo.fmapDesc),
        OP_LOGD(FUSION_NAME, "%s only supports dynamic mode on non-ND soc.", convDescInfo.nodeNameStr.c_str()),
        return false);

    return true;
}

bool DepthwiseToConv2dFusionPass::GetFmapChannel()
{
    auto originFormat = convDescInfo.fmapDesc.GetOriginFormat();
    auto inputShape = convDescInfo.fmapDesc.GetOriginShape().GetDims();
    FUSION_PASS_CHECK(
        inputShape.size() != MAX_DIM_NUM,
        OP_LOGD(FUSION_NAME, "%s fmap origin shape dim not equal to 4.", convDescInfo.nodeNameStr.c_str()),
        return false);

    if (originFormat == Format::FORMAT_NCHW) {
        fmapChannel = inputShape[FMAP_CHANNEL_NCHW_INDEX];
    } else if (originFormat == Format::FORMAT_NHWC) {
        fmapChannel = inputShape[FMAP_CHANNEL_NHWC_INDEX];
    } else {
        OP_LOGD(FUSION_NAME, "%s does not support format [%s].", convDescInfo.nodeNameStr.c_str(),
                TypeUtils::FormatToAscendString(originFormat).GetString());
        return false;
    }

    if (fmapChannel == -1 && isNdSoc) {
        const auto filterShape = convDescInfo.filterDesc.GetOriginShape().GetDims();
        const auto filterFormat = convDescInfo.filterDesc.GetOriginFormat();
        int64_t filterChannel = -1;
        if (filterShape.size() == static_cast<size_t>(MAX_DIM_NUM)) {
            if (filterFormat == Format::FORMAT_NCHW) {
                filterChannel = filterShape[FILTER_CHANNEL_NCHW_INDEX];
            } else if (filterFormat == Format::FORMAT_NHWC) {
                filterChannel = filterShape[FILTER_CHANNEL_NHWC_INDEX];
            } else if (filterFormat == Format::FORMAT_HWCN) {
                filterChannel = filterShape[FILTER_CHANNEL_HWCN_INDEX];
            }
        }
        if (filterChannel > 0) {
            fmapChannel = filterChannel;
            OP_LOGD(FUSION_NAME, "%s dynamic fmap channel completed by filter channel[%ld] on ND soc.",
                    convDescInfo.nodeNameStr.c_str(), static_cast<long>(fmapChannel));
        }
    }

    FUSION_PASS_CHECK(fmapChannel == -1,
                      OP_LOGD(FUSION_NAME, "%s does not support fmapChannel of -1.", convDescInfo.nodeNameStr.c_str()),
                      return false);

    return true;
}

bool DepthwiseToConv2dFusionPass::GetDepthwiseConvAttrs(const GNode& depthwiseNode)
{
    depthwiseAttrs = ConvBaseAttrs();
    FUSION_PASS_CHECK(depthwiseNode.GetAttr(STRIDES, depthwiseAttrs.strides) != GRAPH_SUCCESS,
                      OP_LOGE(FUSION_NAME, "%s get strides failed.", convDescInfo.nodeNameStr.c_str()), return false);
    FUSION_PASS_CHECK(depthwiseNode.GetAttr(PADS, depthwiseAttrs.pads) != GRAPH_SUCCESS,
                      OP_LOGE(FUSION_NAME, "%s get pads failed.", convDescInfo.nodeNameStr.c_str()), return false);
    FUSION_PASS_CHECK(depthwiseNode.GetAttr(DILATIONS, depthwiseAttrs.dilations) != GRAPH_SUCCESS,
                      OP_LOGE(FUSION_NAME, "%s get dilations failed.", convDescInfo.nodeNameStr.c_str()), return false);
    FUSION_PASS_CHECK(depthwiseNode.GetAttr(DATA_FORMAT, depthwiseAttrs.dataFormat) != GRAPH_SUCCESS,
                      OP_LOGE(FUSION_NAME, "%s get data_format failed.", convDescInfo.nodeNameStr.c_str()),
                      return false);

    if (isNdSoc) {
        FUSION_PASS_CHECK(depthwiseNode.GetAttr(OFFSET_X, depthwiseAttrs.offsetX) != GRAPH_SUCCESS,
                          OP_LOGE(FUSION_NAME, "%s get offset_x failed.", convDescInfo.nodeNameStr.c_str()),
                          return false);
    }

    std::vector<int64_t> privatePads;
    if (depthwiseNode.GetAttr(PRIVATE_PADDING, privatePads) == GRAPH_SUCCESS && privatePads.size() == PADS_ATTR_SIZE) {
        depthwiseAttrs.pads = privatePads;
        hasPrivatePads = true;
    }

    if (!hasPrivatePads) {
        AscendString paddingStr;
        if (depthwiseNode.GetAttr(PADDING, paddingStr) == GRAPH_SUCCESS) {
            const std::string paddingMode = paddingStr.GetString();
            if (paddingMode == "VALID") {
                depthwiseAttrs.pads = std::vector<int64_t>{0, 0, 0, 0};
            } else if (paddingMode == "SAME") {
                const auto fmapDims = convDescInfo.fmapDesc.GetOriginShape().GetDims();
                const auto filterDims = convDescInfo.filterDesc.GetOriginShape().GetDims();

                const auto filterFmt = convDescInfo.filterDesc.GetOriginFormat();
                int32_t ihIdx = -1, iwIdx = -1, khIdx = -1, kwIdx = -1;
                int32_t shIdx = -1, swIdx = -1, dhIdx = -1, dwIdx = -1;
                if (depthwiseAttrs.dataFormat == "NHWC") {
                    ihIdx = 1;
                    iwIdx = 2;
                    shIdx = 1;
                    swIdx = 2;
                    dhIdx = 1;
                    dwIdx = 2;
                } else {
                    ihIdx = 2;
                    iwIdx = 3;
                    shIdx = 2;
                    swIdx = 3;
                    dhIdx = 2;
                    dwIdx = 3;
                }
                if (filterFmt == Format::FORMAT_HWCN) {
                    khIdx = 0;
                    kwIdx = 1;
                } else if (filterFmt == Format::FORMAT_NHWC) {
                    khIdx = 1;
                    kwIdx = 2;
                } else {
                    khIdx = 2;
                    kwIdx = 3;
                }
                if (fmapDims.size() == static_cast<size_t>(MAX_DIM_NUM) &&
                    filterDims.size() == static_cast<size_t>(MAX_DIM_NUM) &&
                    depthwiseAttrs.strides.size() == PADS_ATTR_SIZE &&
                    depthwiseAttrs.dilations.size() == PADS_ATTR_SIZE) {
                    auto sameAxis = [](int64_t inSize, int64_t kSize, int64_t stride, int64_t dilation, int64_t& first,
                                       int64_t& second) {
                        const int64_t eff = (kSize - 1) * dilation + 1;
                        const int64_t tails = inSize % stride;
                        const int64_t pad = std::max((tails > 0 ? eff - tails : eff - stride), static_cast<int64_t>(0));
                        first = pad >> 1;
                        second = (pad >> 1) + (pad & 1);
                    };
                    int64_t pt = 0, pb = 0, pl = 0, pr = 0;
                    sameAxis(fmapDims[ihIdx], filterDims[khIdx], depthwiseAttrs.strides[shIdx],
                             depthwiseAttrs.dilations[dhIdx], pt, pb);
                    sameAxis(fmapDims[iwIdx], filterDims[kwIdx], depthwiseAttrs.strides[swIdx],
                             depthwiseAttrs.dilations[dwIdx], pl, pr);
                    depthwiseAttrs.pads = std::vector<int64_t>{pt, pb, pl, pr};
                }
            }
        }
    }

    return true;
}

bool DepthwiseToConv2dFusionPass::MeetRequirements(const GNode& depthwiseNode)
{
    InitMember();

    isNdSoc = ConvFusionUtilsPass::CheckSocList(ND_SOC_LIST, npuArch, true);
    FUSION_PASS_CHECK_NOLOG(!ConvFusionUtilsPass::GetConvDescInfo(depthwiseNode, convDescInfo), return false);
    OP_LOGD(convDescInfo.nodeNameStr, "Begin to do DepthwiseToConv2dFusionPass.");

    FUSION_PASS_CHECK_NOLOG(!CheckDynamicShape(), return false);
    FUSION_PASS_CHECK_NOLOG(!GetFmapChannel(), return false);

    return true;
}

GraphUniqPtr DepthwiseToConv2dFusionPass::Replacement(const GNode& depthwiseNode)
{
    FUSION_PASS_CHECK_NOLOG(!GetDepthwiseConvAttrs(depthwiseNode), return nullptr);

    auto graphBuilder = es::EsGraphBuilder("replacement");
    auto [fmap, filter] = graphBuilder.CreateInputs<REQUIRED_INPUT_NUMS>();
    std::vector<es::EsTensorHolder> inputs = {fmap, filter};
    if (convDescInfo.hasBias) {
        inputs.emplace_back(graphBuilder.CreateInput(static_cast<int64_t>(INPUT_BIAS_INDEX)));
    }

    auto* replaceGraph = graphBuilder.GetCGraphBuilder()->GetGraph();
    FUSION_PASS_CHECK(replaceGraph == nullptr, OP_LOGE_WITHOUT_REPORT(FUSION_NAME, "get replacement graph failed."),
                      return nullptr);
    GNode conv2dNode;
    FUSION_PASS_CHECK(
        !ConvFusionUtilsPass::BuildConv2dNode(replaceGraph, convDescInfo.nodeNameStr + "_To_Conv2D", inputs,
                                              conv2dNode),
        OP_LOGE_WITHOUT_REPORT(FUSION_NAME, "%s build Conv2D node failed.", convDescInfo.nodeNameStr.c_str()),
        return nullptr);

    FUSION_PASS_CHECK_NOLOG(!SetConv2dAttrs(conv2dNode, depthwiseNode), return nullptr);
    FUSION_PASS_CHECK_NOLOG(!UpdateConv2dDesc(conv2dNode), return nullptr);

    auto* yHolder = graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(conv2dNode, OUTPUT_INDEX);
    FUSION_PASS_CHECK(yHolder == nullptr,
                      OP_LOGE_WITHOUT_REPORT(FUSION_NAME, "%s get Conv2D output tensor holder failed.",
                                             convDescInfo.nodeNameStr.c_str()),
                      return nullptr);

    return graphBuilder.BuildAndReset({es::EsTensorHolder(yHolder)});
}

bool DepthwiseToConv2dFusionPass::SetConv2dAttrs(GNode& conv2dNode, const GNode& depthwiseNode)
{
    FUSION_PASS_CHECK(conv2dNode.SetAttr(STRIDES, depthwiseAttrs.strides) != GRAPH_SUCCESS,
                      OP_LOGE_WITHOUT_REPORT(FUSION_NAME, "%s set strides failed.", convDescInfo.nodeNameStr.c_str()),
                      return false);
    FUSION_PASS_CHECK(conv2dNode.SetAttr(PADS, depthwiseAttrs.pads) != GRAPH_SUCCESS,
                      OP_LOGE_WITHOUT_REPORT(FUSION_NAME, "%s set pads failed.", convDescInfo.nodeNameStr.c_str()),
                      return false);
    FUSION_PASS_CHECK(conv2dNode.SetAttr(DILATIONS, depthwiseAttrs.dilations) != GRAPH_SUCCESS,
                      OP_LOGE_WITHOUT_REPORT(FUSION_NAME, "%s set dilations failed.", convDescInfo.nodeNameStr.c_str()),
                      return false);
    FUSION_PASS_CHECK(
        conv2dNode.SetAttr(DATA_FORMAT, depthwiseAttrs.dataFormat) != GRAPH_SUCCESS,
        OP_LOGE_WITHOUT_REPORT(FUSION_NAME, "%s set data_format failed.", convDescInfo.nodeNameStr.c_str()),
        return false);
    FUSION_PASS_CHECK(conv2dNode.SetAttr(GROUPS, fmapChannel) != GRAPH_SUCCESS,
                      OP_LOGE_WITHOUT_REPORT(FUSION_NAME, "%s set groups failed.", convDescInfo.nodeNameStr.c_str()),
                      return false);
    if (isNdSoc) {
        FUSION_PASS_CHECK(
            conv2dNode.SetAttr(OFFSET_X, depthwiseAttrs.offsetX) != GRAPH_SUCCESS,
            OP_LOGE_WITHOUT_REPORT(FUSION_NAME, "%s set offset_x failed.", convDescInfo.nodeNameStr.c_str()),
            return false);
    }
    if (depthwiseNode.GetAttr(PADDING, depthwiseAttrs.padding) == GRAPH_SUCCESS) {
        FUSION_PASS_CHECK(
            conv2dNode.SetAttr(PADDING, depthwiseAttrs.padding) != GRAPH_SUCCESS,
            OP_LOGE_WITHOUT_REPORT(FUSION_NAME, "%s set padding failed.", convDescInfo.nodeNameStr.c_str()),
            return false);
    }
    int64_t opImplModeEnum = 0;
    if (depthwiseNode.GetAttr(OP_IMPL_MODE_ENUM, opImplModeEnum) == GRAPH_SUCCESS) {
        FUSION_PASS_CHECK(
            conv2dNode.SetAttr(OP_IMPL_MODE_ENUM, opImplModeEnum) != GRAPH_SUCCESS,
            OP_LOGE_WITHOUT_REPORT(FUSION_NAME, "%s set op_impl_mode_enum failed.", convDescInfo.nodeNameStr.c_str()),
            return false);
    }

    return true;
}

bool DepthwiseToConv2dFusionPass::UpdateConv2dDesc(GNode& conv2dNode)
{
    auto storageShape = convDescInfo.fmapDesc.GetShape();
    const auto storageFormat = convDescInfo.fmapDesc.GetFormat();
    int32_t storageCIdx = -1;
    if (storageFormat == Format::FORMAT_NCHW) {
        storageCIdx = FMAP_CHANNEL_NCHW_INDEX;
    } else if (storageFormat == Format::FORMAT_NHWC) {
        storageCIdx = FMAP_CHANNEL_NHWC_INDEX;
    }
    if (storageCIdx >= 0 && storageShape.GetDimNum() == static_cast<size_t>(MAX_DIM_NUM) &&
        storageShape.GetDim(storageCIdx) == -1 && fmapChannel > 0) {
        storageShape.SetDim(storageCIdx, fmapChannel);
        convDescInfo.fmapDesc.SetShape(storageShape);
        OP_LOGD(FUSION_NAME, "%s dynamic fmap storage channel completed to [%ld].", convDescInfo.nodeNameStr.c_str(),
                static_cast<long>(fmapChannel));
    }

    FUSION_PASS_CHECK_NOLOG(!ConvFusionUtilsPass::UpdateInputDesc(&conv2dNode, convDescInfo), return false);
    FUSION_PASS_CHECK(conv2dNode.UpdateOutputDesc(OUTPUT_INDEX, convDescInfo.outputDesc) != GRAPH_SUCCESS,
                      OP_LOGE_WITHOUT_REPORT(FUSION_NAME, "%s update Conv2D output tensor desc failed.",
                                             convDescInfo.nodeNameStr.c_str()),
                      return false);

    return true;
}

#if GE_COMPILER_VERSION_NUM >= 90000000U
REG_DECOMPOSE_PASS(DepthwiseToConv2dFusionPass, {DEPTHWISE_CONV2D}).Stage(CustomPassStage::kCompatibleInherited);
#endif

} // namespace Ops

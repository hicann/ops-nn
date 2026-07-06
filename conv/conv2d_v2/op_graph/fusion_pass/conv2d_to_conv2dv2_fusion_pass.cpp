/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "conv2d_to_conv2dv2_fusion_pass.h"

#include "conv/common/op_graph/fusion_pass/conv_fusion_utils_pass.h"
#include "es_nn_ops.h"
#include "ge/es_graph_builder.h"

namespace Ops {
using namespace NN;
using namespace Conv;
using namespace ConvFusionUtils;
using namespace Conv2dToConv2dV2Fusion;
using namespace ge;
using namespace fusion;

void Conv2dToConv2dV2FusionPass::InitMember()
{
    npuArch = NpuArch::DAV_RESV;
    convDescInfo = ConvDescInfo();
}

bool Conv2dToConv2dV2FusionPass::MeetRequirements(const GNode& convNode)
{
    InitMember();

    FUSION_PASS_CHECK(!ConvFusionUtilsPass::CheckSocList(SUPPORT_SOC_LIST, npuArch),
                      OP_LOGD(FUSION_NAME, "Current soc not supported, no fusion."), return false);

    FUSION_PASS_CHECK_NOLOG(!ConvFusionUtilsPass::GetConvDescInfo(convNode, convDescInfo), return false);
    OP_LOGD(convDescInfo.nodeNameStr, "Begin to do Conv2dToConv2dV2FusionPass.");

    std::vector<DataType> convDtypes = {convDescInfo.fmapDtype, convDescInfo.filterDtype, convDescInfo.outputDtype};
    if (convDescInfo.hasBias) {
        convDtypes.emplace_back(convDescInfo.biasDtype);
    }
    const auto& convSupportList = (npuArch == NpuArch::DAV_3510) ? CONV_SUPPORT_DTYPES_DAV_3510 :
                                                                   CONV_SUPPORT_DTYPES_DAV_5102;
    FUSION_PASS_CHECK(!ConvFusionUtilsPass::CheckSupportList<DataType>(convSupportList, convDtypes),
                      OP_LOGD(convDescInfo.nodeNameStr, "Conv2D dtype not supported, no fusion."), return false);

    return true;
}

GraphUniqPtr Conv2dToConv2dV2FusionPass::Replacement(const GNode& convNode)
{
    auto graphBuilder = es::EsGraphBuilder("replacement");

    ConvBaseAttrs baseAttrs;
    FUSION_PASS_CHECK_NOLOG(!ConvFusionUtilsPass::GetConvBaseAttr(convNode, baseAttrs, convDescInfo), return nullptr);

    auto [fmap, filter] = graphBuilder.CreateInputs<REQUIRED_INPUT_NUMS>();
    auto bias = convDescInfo.hasBias ? graphBuilder.CreateInput(static_cast<int64_t>(INPUT_BIAS_INDEX)) : nullptr;

    auto conv2dV2 = es::Conv2DV2(fmap, filter, bias, nullptr, baseAttrs.strides, baseAttrs.pads, baseAttrs.dilations,
                                 baseAttrs.groups, baseAttrs.dataFormat.GetString(), baseAttrs.offsetX,
                                 baseAttrs.padMode.GetString(), baseAttrs.enableHf32);

    auto conv2dV2Node = conv2dV2.GetProducer();
    FUSION_PASS_CHECK_NOLOG(!ConvFusionUtilsPass::UpdateInputDesc(conv2dV2Node, convDescInfo), return nullptr);
    FUSION_PASS_CHECK(conv2dV2Node->UpdateOutputDesc(OUTPUT_INDEX, convDescInfo.outputDesc) != GRAPH_SUCCESS,
                      OP_LOGE(convDescInfo.nodeNameStr, "Update Conv2DV2 output tensor desc failed."), return nullptr);
    FUSION_PASS_CHECK(conv2dV2Node->SetAttr(OP_IMPL_MODE_ENUM, baseAttrs.opImplModeEnum) != GRAPH_SUCCESS,
                      OP_LOGE(convDescInfo.nodeNameStr, "Set _op_impl_mode_enum for Conv2DV2 failed."), return nullptr);

    return graphBuilder.BuildAndReset({conv2dV2});
}

} // namespace Ops
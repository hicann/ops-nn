/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "conv3d_to_conv3dv2_fusion_pass.h"

#include "conv/common/op_graph/fusion_pass/conv_fusion_utils_pass.h"
#include "es_nn_ops.h"
#include "graph/utils/type_utils.h"
#include "log/log.h"
#include "platform/platform_info.h"
#include "version/ge-compiler_version.h"

namespace Ops {
using namespace NN;
using namespace Conv;
using namespace ConvFusionUtils;
using namespace Conv3dToConv3dV2Fusion;
using namespace ge;
using namespace fe;
using namespace fusion;

void Conv3dToConv3dV2FusionPass::InitMember()
{
    supportOut2L1Dn2Nz = false;
    convDescInfo = ConvDescInfo();
}

bool Conv3dToConv3dV2FusionPass::CheckSocCapability()
{
    PlatformInfo platformInfo;
    OptionalInfo optionalInfo;
    FUSION_PASS_CHECK(
        PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platformInfo, optionalInfo) != GRAPH_SUCCESS,
        OP_LOGE(FUSION_NAME, "Get PlatformInfo failed."), return false);

    supportOut2L1Dn2Nz = platformInfo.ai_core_intrinsic_dtype_map.find("Intrinsic_data_move_out2l1_dn2nz") !=
        platformInfo.ai_core_intrinsic_dtype_map.end();
    if (supportOut2L1Dn2Nz) {
        OP_LOGD(FUSION_NAME, "Current soc supported out2l1_dn2nz intrinsic.");
        return true;
    }

    bool supportL0c2Out = platformInfo.ai_core_intrinsic_dtype_map.find("Intrinsic_fix_pipe_l0c2out") !=
        platformInfo.ai_core_intrinsic_dtype_map.end();
    bool supportFixpipeL0c2Ub = platformInfo.ai_core_intrinsic_dtype_map.find("Intrinsic_fix_pipe_l0c2ub") !=
        platformInfo.ai_core_intrinsic_dtype_map.end();
    if (!supportL0c2Out || supportFixpipeL0c2Ub) {
        OP_LOGD(FUSION_NAME, "Current soc not supported, no fusion.");
        return false;
    }

    return true;
}

bool Conv3dToConv3dV2FusionPass::CheckPostCubeInOutNode(const GNode &convNode) const
{
    auto convOutputNodes = convNode.GetOutDataNodesAndPortIndexs(OUTPUT_INDEX);
    for (size_t i = 0; i < convOutputNodes.size(); ++i) {
        auto outNode = convOutputNodes[i].first;
        if (outNode == nullptr) {
            continue;
        }
        AscendString outType;
        outNode->GetType(outType);
        if (outType == POST_CUBE_OP) {
            return true;
        }
    }

    return false;
}

bool Conv3dToConv3dV2FusionPass::CheckTransDataInInputNode(const GNode &convNode) const
{
    auto nodePtr = convNode.GetInDataNodesAndPortIndexs(INPUT_FMAP_INDEX).first;
    FUSION_PASS_CHECK(nodePtr == nullptr,
        OP_LOGD(convDescInfo.nodeNameStr, "Get in data node failed."), return false);

    AscendString opType;
    nodePtr->GetType(opType);
    if (opType != TRANS_DATA_OP) {
        return false;
    }

    TensorDesc transInDesc;
    nodePtr->GetInputDesc(0, transInDesc);
    TensorDesc transOutDesc;
    nodePtr->GetOutputDesc(0, transOutDesc);
    if (transInDesc.GetFormat() == ge::FORMAT_NDHWC && transOutDesc.GetFormat() == ge::FORMAT_NDC1HWC0) {
        int64_t fmapC = transInDesc.GetOriginShape().GetDim(CIDX);
        if (fmapC % C0_SIZE == 0) {
            return true;
        }
    }

    return false;
}

bool Conv3dToConv3dV2FusionPass::CheckIFMRInSameOutputNode(const GNode &convNode) const
{
    auto convOutputNodes = convNode.GetOutDataNodesAndPortIndexs(OUTPUT_INDEX);
    for (size_t i = 0; i < convOutputNodes.size(); ++i) {
        auto outNode = convOutputNodes[i].first;
        if (outNode == nullptr) {
            continue;
        }
        for (int32_t idx = 0; idx < outNode->GetInputsSize(); ++idx) {
            auto inNode = outNode->GetInDataNodesAndPortIndexs(idx).first;
            if (inNode == nullptr) {
                continue;
            }
            AscendString nodeType;
            inNode->GetType(nodeType);
            if (nodeType == IFMR_OP) {
                return true;
            }
        }
    }

    return false;
}

bool Conv3dToConv3dV2FusionPass::MeetRequirements(const GNode &convNode)
{
    InitMember();

    FUSION_PASS_CHECK_NOLOG(!CheckSocCapability(), return false);

    FUSION_PASS_CHECK_NOLOG(!ConvFusionUtilsPass::GetConvDescInfo(convNode, convDescInfo), return false);
    OP_LOGD(convDescInfo.nodeNameStr, "Begin to do Conv3dToConv3dV2FusionPass.");

    std::vector<DataType> convDtypes = {convDescInfo.fmapDtype, convDescInfo.filterDtype, convDescInfo.outputDtype};
    if (convDescInfo.hasBias) {
        convDtypes.emplace_back(convDescInfo.biasDtype);
    }
    auto &convSupportList = supportOut2L1Dn2Nz ? CONV_SUPPORT_DTYPES_OUT2L1_DN2NZ : CONV_SUPPORT_DTYPES;
    FUSION_PASS_CHECK(!ConvFusionUtilsPass::CheckSupportList<DataType>(convSupportList, convDtypes),
        OP_LOGD(convDescInfo.nodeNameStr, "Conv3D dtype not supported, no fusion."), return false);

    if (supportOut2L1Dn2Nz) {
        return true;
    }

    bool isStatic = !ConvFusionUtilsPass::IsUnknownShape(convDescInfo.fmapDesc) &&
        !ConvFusionUtilsPass::IsUnknownShape(convDescInfo.filterDesc);
    if (convDescInfo.hasBias) {
        isStatic = isStatic && !ConvFusionUtilsPass::IsUnknownShape(convDescInfo.biasDesc);
    }
    FUSION_PASS_CHECK(!isStatic,
        OP_LOGD(convDescInfo.nodeNameStr, "Conv3dv2 only support static shape. in current soc, no fusion."),
        return false);

    FUSION_PASS_CHECK(CheckPostCubeInOutNode(convNode),
        OP_LOGD(convDescInfo.nodeNameStr,
            "Conv3d that contain FixPipe in out node is not supported."),
            return false);

    if (convDescInfo.fmapDtype == DataType::DT_FLOAT16) {
        FUSION_PASS_CHECK(CheckTransDataInInputNode(convNode),
            OP_LOGD(convDescInfo.nodeNameStr,
                "Extra situation: Conv3d that contain TransData in input node is not supported."),
                return false);
        FUSION_PASS_CHECK(CheckIFMRInSameOutputNode(convNode),
            OP_LOGD(convDescInfo.nodeNameStr,
                "Extra situation: Conv3d that contain IFMR in same output node is not supported."),
                return false);
    }

    return true;
}

GraphUniqPtr Conv3dToConv3dV2FusionPass::Replacement(const GNode &convNode)
{
    auto graphBuilder = es::EsGraphBuilder("replacement");

    ConvBaseAttrs baseAttrs = {};
    FUSION_PASS_CHECK_NOLOG(!ConvFusionUtilsPass::GetConvBaseAttr(convNode, baseAttrs, convDescInfo),
        return nullptr);

    if (!supportOut2L1Dn2Nz) {
        OP_LOGW(convDescInfo.nodeNameStr, "Current soc not supported Hf32, set enable_hf32 to false.");   
        baseAttrs.enableHf32 = false;
    }

    auto [fmap, filter] = graphBuilder.CreateInputs<REQUIRED_INPUT_NUMS>();
    auto bias = convDescInfo.hasBias ?
        graphBuilder.CreateInput(static_cast<int64_t>(INPUT_BIAS_INDEX)) : nullptr;

    auto conv3dV2 = es::Conv3DV2(fmap, filter, bias,
        nullptr, nullptr, nullptr,
        baseAttrs.strides, baseAttrs.pads, baseAttrs.dilations,
        baseAttrs.groups, baseAttrs.dataFormat.GetString(),
        baseAttrs.offsetX, baseAttrs.padMode.GetString(),
        baseAttrs.enableHf32);

    auto conv3dV2Node = conv3dV2.GetProducer();
    FUSION_PASS_CHECK_NOLOG(!ConvFusionUtilsPass::UpdateInputDesc(conv3dV2Node, convDescInfo), return nullptr);
    FUSION_PASS_CHECK(conv3dV2Node->UpdateOutputDesc(OUTPUT_INDEX, convDescInfo.outputDesc) != GRAPH_SUCCESS,
        OP_LOGE(convDescInfo.nodeNameStr, "Update Conv3DV2 output tensor desc failed."), return nullptr);
    FUSION_PASS_CHECK(conv3dV2Node->SetAttr(OP_IMPL_MODE_ENUM, baseAttrs.opImplModeEnum) != GRAPH_SUCCESS,
        OP_LOGE(convDescInfo.nodeNameStr, "Set _op_impl_mode_enum for Conv3DV2 failed."), return nullptr);
    OP_LOGD(convDescInfo.nodeNameStr, "Create Conv3DV2 node successfully.");

    return graphBuilder.BuildAndReset({conv3dV2});
}

#if GE_COMPILER_VERSION_NUM >= 90000000U
REG_DECOMPOSE_PASS(Conv3dToConv3dV2FusionPass, {ConvFusionUtils::CONV3D})
    .Stage(CustomPassStage::kCompatibleInherited);
#endif

} // namespace Ops
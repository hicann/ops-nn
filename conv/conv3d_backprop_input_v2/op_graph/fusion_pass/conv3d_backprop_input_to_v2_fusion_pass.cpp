/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "conv3d_backprop_input_to_v2_fusion_pass.h"

#include "es_nn_ops.h"
#include "log/log.h"
#include "register/register_custom_pass.h"

namespace ops {
using namespace ge;
using namespace ge::es;
using namespace fusion;
using namespace ConvBackpropFusionUtils;

AscendString Conv3DBackpropInputToV2FusionPass::GetNodeType() const {     return CONV_BACKPROP_INPUT_V2_PASS; }

bool Conv3DBackpropInputToV2FusionPass::CheckTransposeNeeded()
{
    if (input1Desc.GetOriginFormat() != FORMAT_DHWCN) {
        OP_LOGD(GetNodeType().GetString(), "Transpose not needed: filter format is not DHWCN");
        return false;
    }

    if (input2Desc.GetOriginFormat() != FORMAT_NDHWC) {
        OP_LOGD(GetNodeType().GetString(), "Transpose not needed: out_backprop format is not NDHWC");
        return false;
    }

    bool isDynamic = input1Desc.GetShape().GetShapeSize() == -1 || input2Desc.GetShape().GetShapeSize() == -1 ||
                outputDesc.GetShape().GetShapeSize() == -1;
    if (isDynamic) {
        OP_LOGD(GetNodeType().GetString(), "Transpose not needed: dynamic shape detected");
        return false;
    }

    auto outShape = outputDesc.GetShape().GetDims();
    auto fltShape = input1Desc.GetShape().GetDims();
    if (outShape.size() != CONV_DIM_LENGTH || fltShape.size() != CONV_DIM_LENGTH) {
        OP_LOGD(GetNodeType().GetString(), "Transpose not needed: filter or y shape dim != 5");
        return false;
    }

    if (!ConvBackpropFusionUtilsPass::InWhitelist(outShape, outputShapeWhitelist) ||
        !ConvBackpropFusionUtilsPass::InWhitelist(fltShape, filterShapeWhitelist) ||
        !ConvBackpropFusionUtilsPass::InWhitelist(input2Desc.GetShape().GetDims(), dedyShapeWhitelist)) {
        OP_LOGD(GetNodeType().GetString(), "Transpose not needed: shape not in whitelist");
        return false;
    }

    if (convBpAttr.strides.size() != CONV_DIM_LENGTH ||
        convBpAttr.dilations.size() != CONV_DIM_LENGTH || convBpAttr.groups != 1) {
        OP_LOGD(GetNodeType().GetString(), "Transpose not needed: attr check failed or groups != 1");
        return false;
    }

    if (convBpAttr.strides[H_DIM_NDHWC_INDEX] != 2 || convBpAttr.strides[W_DIM_NDHWC_INDEX] != 2 ||
        convBpAttr.dilations[H_DIM_NDHWC_INDEX] != 1 || convBpAttr.dilations[W_DIM_NDHWC_INDEX] != 1) {
        OP_LOGD(GetNodeType().GetString(), "Transpose not needed: strides or dilations constraint not met");
        return false;
    }

    OP_LOGD(GetNodeType().GetString(), "Transpose needed");
    return true;
}

bool Conv3DBackpropInputToV2FusionPass::CreateOutputWithTranspose(
    EsGraphBuilder& builder, const EsTensorHolder& conv3dBpInputV2, GNode* conv3dBpInputV2Node,
    EsTensorHolder& transOutput)
{
    TensorDesc outNcdhwDesc;
    outNcdhwDesc.SetDataType(outputDesc.GetDataType());
    auto ndhwc = outputDesc.GetShape().GetDims();
    auto ncdhw ={ndhwc[N_DIM_NDHWC_INDEX], ndhwc[C_DIM_NDHWC_INDEX],ndhwc[D_DIM_NDHWC_INDEX], ndhwc[H_DIM_NDHWC_INDEX], ndhwc[W_DIM_NDHWC_INDEX]};
    outNcdhwDesc.SetShape(Shape(ncdhw));
    outNcdhwDesc.SetOriginShape(Shape(ncdhw));
    outNcdhwDesc.SetFormat(Format::FORMAT_NCDHW);
    outNcdhwDesc.SetOriginFormat(Format::FORMAT_NCDHW);
    conv3dBpInputV2Node->UpdateOutputDesc(OUTPUT_INDEX, outNcdhwDesc);

    auto config = TransposeNodeConfig::Create(
        conv3dBpInputV2, OUTPUT_TRANSPOSE_PERM, "y_transpose", outputDesc.GetFormat());
    
    TensorDesc transOutDesc;
    OP_CHECK_IF(!ConvBackpropFusionUtilsPass::CreateTransposeNode(
                builder, config, transOutput, transOutDesc, GetNodeType()),
                OP_LOGE(GetNodeType().GetString(), "Create y transpose node failed"), return false);
    return true;
}

bool Conv3DBackpropInputToV2FusionPass::Createconv3dBpInputTransposeGraph(
    EsGraphBuilder& builder, EsTensorHolder& conv3dBpInputTrans, EsTensorHolder& inputSize, EsTensorHolder& filter,
    EsTensorHolder& dedy)
{
    EsTensorHolder transFlt;
    TensorDesc transFltDesc;
    auto fltConfig = TransposeNodeConfig::Create(
        filter, FILTER_TRANSPOSE_PERM, "filter_transpose", Format::FORMAT_NCDHW);
    OP_CHECK_IF(!ConvBackpropFusionUtilsPass::CreateTransposeNode(
                builder, fltConfig, transFlt, transFltDesc, GetNodeType()),
                OP_LOGE(GetNodeType().GetString(), "Create filter transpose node failed"), return false);

    auto dedyConfig = TransposeNodeConfig::Create(
        dedy, DEDY_TRANSPOSE_PERM, "dedy_transpose", Format::FORMAT_NCDHW);
    
    EsTensorHolder transDedy;
    TensorDesc transDedyDesc;
    OP_CHECK_IF(!ConvBackpropFusionUtilsPass::CreateTransposeNode(
                builder, dedyConfig, transDedy, transDedyDesc, GetNodeType()),
                OP_LOGE(GetNodeType().GetString(), "Create out_backprop transpose node failed"), return false);

    std::vector<int64_t> transStrides = convBpAttr.strides;
    std::vector<int64_t> transDils = convBpAttr.dilations;
    std::rotate(transStrides.begin() + 1, transStrides.begin() + 4, transStrides.end());
    std::rotate(transDils.begin() + 1, transDils.begin() + 4, transDils.end());
    std::string finalFmt = "NCDHW";

    auto conv3dBpInputV2 = Conv3DBackpropInputV2(
        inputSize, transFlt, transDedy, transStrides, convBpAttr.pads, transDils, convBpAttr.groups,
        finalFmt.c_str(), convBpAttr.hf32);

    auto* conv3dBpInputV2Node = conv3dBpInputV2.GetProducer();
    OP_CHECK_IF(conv3dBpInputV2Node == nullptr,
                OP_LOGE(GetNodeType().GetString(), "Create Conv3DBackpropInputV2 node failed"), return false);

    conv3dBpInputV2Node->SetAttr("_op_impl_mode_enum", convBpAttr.opImplModeEnum);
    conv3dBpInputV2Node->UpdateInputDesc(CONV_BP_V2_INPUT_INDEX, input0Desc);
    conv3dBpInputV2Node->UpdateInputDesc(CONV_BP_V2_FILTER_INDEX, transFltDesc);
    conv3dBpInputV2Node->UpdateInputDesc(CONV_BP_V2_OUT_BACKPROP_INDEX, transDedyDesc);
    conv3dBpInputV2Node->UpdateOutputDesc(CONV_BP_V2_OUTPUT_INDEX, outputDesc);

    OP_CHECK_IF(!CreateOutputWithTranspose(builder, conv3dBpInputV2, conv3dBpInputV2Node, conv3dBpInputTrans),
                OP_LOGE(GetNodeType().GetString(), "Create y with transpose failed"), return false);

    return true;
}

GraphUniqPtr Conv3DBackpropInputToV2FusionPass::Replacement(const GNode& convBpInputNode)
{
    OP_LOGD(GetNodeType().GetString(), "Replacement start");

    OP_CHECK_IF(!GetNodeDesc(convBpInputNode),
                OP_LOGE(GetNodeType().GetString(), "GetNodeDesc failed"), return nullptr);

    OP_CHECK_IF(!GetNodeAttrs(convBpInputNode),
                OP_LOGE(GetNodeType().GetString(), "GetNodeAttrs failed"), return nullptr);

    auto builder = EsGraphBuilder("replacement");
    auto [inputSize, filter, dedy] = builder.CreateInputs<3>();

    ConvBackpropFusionUtilsPass::SetPlaceholderDesc(inputSize, 0, input0Desc);
    ConvBackpropFusionUtilsPass::SetPlaceholderDesc(filter, 0, input1Desc);
    ConvBackpropFusionUtilsPass::SetPlaceholderDesc(dedy, 0, input2Desc);

    bool needTranspose = CheckTransposeNeeded();
    if (needTranspose) {
        EsTensorHolder conv3dBpInputTrans;
        if (!Createconv3dBpInputTransposeGraph(builder, conv3dBpInputTrans, inputSize, filter, dedy)) {
            return nullptr;
        }
        return builder.BuildAndReset(std::vector<EsTensorHolder>{conv3dBpInputTrans});
    }

    auto conv3dBpInputV2 = Conv3DBackpropInputV2(
        inputSize, filter, dedy, convBpAttr.strides, convBpAttr.pads, convBpAttr.dilations,
        convBpAttr.groups, convBpAttr.dataFormat.c_str(), convBpAttr.hf32);

    auto* conv3dBpInputV2Node = conv3dBpInputV2.GetProducer();
    OP_CHECK_IF(conv3dBpInputV2Node == nullptr,
                OP_LOGE(GetNodeType().GetString(), "Create Conv3DBackpropInputV2 node failed"), return nullptr);

    conv3dBpInputV2Node->SetAttr("_op_impl_mode_enum", convBpAttr.opImplModeEnum);
    OP_CHECK_IF(!UpdateNodeInputDescInfo(conv3dBpInputV2Node),
                OP_LOGE(GetNodeType().GetString(), "Update conv3dBpInputV2Node DescInfo failed"), return nullptr);

    return builder.BuildAndReset(std::vector<EsTensorHolder>{conv3dBpInputV2});
}

REG_DECOMPOSE_PASS(Conv3DBackpropInputToV2FusionPass, {CONV_BACKPROP_INPUT})
    .Stage(CustomPassStage::kCompatibleInherited);

} // namespace ops

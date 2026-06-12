/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "conv_backprop_fusion_base_pass.h"
#include "log/log.h"

namespace ops {
using namespace ge;
using namespace ConvBackpropFusionUtils;

void ConvBackpropFusionBasePass::InitMember()
{
    npuArch = NpuArch::DAV_RESV;
    input0Desc = TensorDesc();
    input1Desc = TensorDesc();
    input2Desc = TensorDesc();
    outputDesc = TensorDesc();
    convBpAttr.Reset();
}

bool ConvBackpropFusionBasePass::CheckSocAndIntrinsic()
{
    return ConvBackpropFusionUtilsPass::CheckSocAndIntrinsic(SUPPORT_SOC_LIST, npuArch);
}


bool ConvBackpropFusionBasePass::MeetRequirements(const GNode& convBpInputNode)
{
    if (!CheckSocAndIntrinsic()) {
        OP_LOGD(GetNodeType().GetString(), "SOC check failed");
        return false;
    }
    OP_LOGD(GetNodeType().GetString(), "SOC check passed");
    return true;
}

bool ConvBackpropFusionBasePass::GetNodeDesc(const GNode& node)
{
    InitMember();
    // 获取输入索引
    int32_t inputIdx0 = 0;
    int32_t inputIdx1 = 1;
    int32_t outBackpropIdx = 2;
    OP_CHECK_IF(node.GetInputDesc(inputIdx0, input0Desc) != GRAPH_SUCCESS ||
        node.GetInputDesc(inputIdx1, input1Desc) != GRAPH_SUCCESS ||
        node.GetInputDesc(outBackpropIdx, input2Desc) != GRAPH_SUCCESS ||
        node.GetOutputDesc(OUTPUT_INDEX, outputDesc) != GRAPH_SUCCESS,
        OP_LOGE(GetNodeType().GetString(), "Get input/output desc failed"), return false);
    return true;
}

bool ConvBackpropFusionBasePass::GetNodeAttrs(const GNode& node)
{
    AscendString name;
    node.GetName(name);

    AscendString format;
    OP_CHECK_IF(node.GetAttr("strides", convBpAttr.strides) != GRAPH_SUCCESS ||
        node.GetAttr("pads", convBpAttr.pads) != GRAPH_SUCCESS ||
        node.GetAttr("dilations", convBpAttr.dilations) != GRAPH_SUCCESS ||
        node.GetAttr("groups", convBpAttr.groups) != GRAPH_SUCCESS ||
        node.GetAttr("data_format", format) != GRAPH_SUCCESS,
        OP_LOGE(GetNodeType().GetString(), "Get attrs from %s failed", name.GetString()), return false);
    
    if(node.GetAttr("_op_impl_mode_enum", convBpAttr.opImplModeEnum) != GRAPH_SUCCESS){
        OP_LOGD(GetNodeType().GetString(), "Get _op_impl_mode_enum attrs from %s failed, set default value", name.GetString());
    }

    convBpAttr.dataFormat = std::string(format.GetString());

    convBpAttr.hf32 =
        input2Desc.GetDataType() == DataType::DT_FLOAT && 
        convBpAttr.opImplModeEnum == HF32_PRECISION_MODE_INT;

    convBpAttr.opImplModeEnum = convBpAttr.hf32 ? convBpAttr.opImplModeEnum : 0x1;

    return true;
}

bool ConvBackpropFusionBasePass::UpdateNodeInputDescInfo(ge::GNode *node)
{
    OP_CHECK_IF(node->UpdateInputDesc(CONV_BP_V2_INPUT_INDEX, input0Desc) != GRAPH_SUCCESS ||
        node->UpdateInputDesc(CONV_BP_V2_FILTER_INDEX, input1Desc) != GRAPH_SUCCESS ||
        node->UpdateInputDesc(CONV_BP_V2_OUT_BACKPROP_INDEX, input2Desc) != GRAPH_SUCCESS ||
        node->UpdateOutputDesc(CONV_BP_V2_OUTPUT_INDEX, outputDesc) != GRAPH_SUCCESS,
        OP_LOGE(GetNodeType().GetString(), "Update NodeInputDescInfo failed"), return false);
    return true;
}

} // namespace ops

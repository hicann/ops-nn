/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "conv3d_backprop_filter_to_v2_fusion_pass.h"

#include "es_nn_ops.h"
#include "log/log.h"
#include "register/register_custom_pass.h"

namespace ops {
using namespace ge;
using namespace ge::es;
using namespace fusion;
using namespace ConvBackpropFusionUtils;

// FilterV2特有的常量
constexpr int64_t CONDICTION_DIVIDE_K = 16 * 32 * 32;
const std::vector<int32_t> TRANSPOSE_PERM_NDHWC = {0, 2, 3, 4, 1};   // NCDHW -> NDHWC
const std::vector<int32_t> TRANSPOSE_PERM_DHWCN = {2, 3, 4, 1, 0};   // NCDHW -> DHWCN

AscendString Conv3DBackpropFilterToV2FusionPass::GetNodeType() const
{
    return CONV_BACKPROP_FILTER_V2_PASS;
}

bool Conv3DBackpropFilterToV2FusionPass::CheckTransposeNeeded()
{
    auto outputOriginFormat = outputDesc.GetOriginFormat();
    if (outputOriginFormat == Format::FORMAT_NCDHW) {
        OP_LOGD(GetNodeType().GetString(), "y format is NCDHW, need not insert transpose node.");
        return false;
    }

    bool isDynamic = input0Desc.GetShape().GetShapeSize() == -1 || 
                input2Desc.GetShape().GetShapeSize() == -1 ||
                outputDesc.GetShape().GetShapeSize() == -1;
    if (isDynamic) {
        OP_LOGD(GetNodeType().GetString(), "all shape must be specify.");
        return false;
    }

    auto yShapeVec = outputDesc.GetShape().GetDims();
    OP_CHECK_IF(yShapeVec.size() != CONV_DIM_LENGTH,
                OP_LOGE(GetNodeType().GetString(), "y shape size %zu != %d", yShapeVec.size(), CONV_DIM_LENGTH),
                return false);

    int64_t cin = 0, cout = 0, di = 0;
    if (outputOriginFormat == Format::FORMAT_NDHWC) {
        cin = yShapeVec[C_DIM_NDHWC_INDEX];
        cout = yShapeVec[N_DIM_NDHWC_INDEX];
        auto xShapeVec = input0Desc.GetShape().GetDims();
        OP_CHECK_IF(xShapeVec.size() != CONV_DIM_LENGTH,
                    OP_LOGE(GetNodeType().GetString(), "x shape size %zu != %d for NDHWC format",
                            xShapeVec.size(), CONV_DIM_LENGTH), return false);
        di = xShapeVec[D_DIM_NDHWC_INDEX];
    } else {
        cin = yShapeVec[C_DIM_DHWCN_INDEX];
        cout = yShapeVec[N_DIM_DHWCN_INDEX];
        auto xShapeVec = input0Desc.GetShape().GetDims();
        OP_CHECK_IF(xShapeVec.size() != CONV_DIM_LENGTH,
                    OP_LOGE(GetNodeType().GetString(), "x shape size %zu != %d for DHWCN format",
                            xShapeVec.size(), CONV_DIM_LENGTH), return false);
        di = xShapeVec[D_DIM_NCDHW_INDEX];
    } 
    
    if (di * cin * cout >= CONDICTION_DIVIDE_K) {
        OP_LOGD(GetNodeType().GetString(), "need satisfy divide K condition.");
        return false;
    }

    OP_LOGD(GetNodeType().GetString(), "Transpose needed");
    return true;
}

bool Conv3DBackpropFilterToV2FusionPass::CreateOutputWithTranspose(
    EsGraphBuilder& builder, 
    const EsTensorHolder& conv3dBackpropFilterV2, 
    GNode* conv3dBackpropFilterV2Node,
    EsTensorHolder& transOutput)
{
    TensorDesc ncdhwDesc;
    ncdhwDesc.SetDataType(outputDesc.GetDataType());
    auto yShapeVec = outputDesc.GetShape().GetDims();
    std::vector<int64_t> yShapeNcdhw;
    auto outputOriginFormat = outputDesc.GetOriginFormat();
    if (outputOriginFormat == Format::FORMAT_NDHWC) {
        yShapeNcdhw = {
            yShapeVec[N_DIM_NDHWC_INDEX], yShapeVec[C_DIM_NDHWC_INDEX], 
            yShapeVec[D_DIM_NDHWC_INDEX], yShapeVec[H_DIM_NDHWC_INDEX], yShapeVec[W_DIM_NDHWC_INDEX]
        };
    } else {
        yShapeNcdhw = {
            yShapeVec[N_DIM_DHWCN_INDEX], yShapeVec[C_DIM_DHWCN_INDEX], 
            yShapeVec[D_DIM_DHWCN_INDEX], yShapeVec[H_DIM_DHWCN_INDEX], yShapeVec[W_DIM_DHWCN_INDEX]
        };
    } 
    
    ncdhwDesc.SetShape(Shape(yShapeNcdhw));
    ncdhwDesc.SetOriginShape(Shape(yShapeNcdhw));
    ncdhwDesc.SetFormat(Format::FORMAT_NCDHW);
    ncdhwDesc.SetOriginFormat(Format::FORMAT_NCDHW);
    conv3dBackpropFilterV2Node->UpdateOutputDesc(OUTPUT_INDEX, ncdhwDesc);

    std::vector<int32_t> transposePerm = TRANSPOSE_PERM_DHWCN;
    if (outputOriginFormat == Format::FORMAT_NDHWC) {
        transposePerm = TRANSPOSE_PERM_NDHWC;
    }
    
    auto config = TransposeNodeConfig::Create(
        conv3dBackpropFilterV2, transposePerm, "y_transpose", outputOriginFormat);
    
    TensorDesc transOutDesc;
    OP_CHECK_IF(!ConvBackpropFusionUtilsPass::CreateTransposeNode(
                builder, config, transOutput, transOutDesc, GetNodeType()),
                OP_LOGE(GetNodeType().GetString(), "Create y transpose node failed"), return false);
    
    return true;
}

GraphUniqPtr Conv3DBackpropFilterToV2FusionPass::Replacement(const GNode& convBpFilterNode)
{
    OP_LOGD(GetNodeType().GetString(), "Replacement start");

    OP_CHECK_IF(!GetNodeDesc(convBpFilterNode),
                OP_LOGE(GetNodeType().GetString(), "GetNodeDesc failed"), return nullptr);
    OP_CHECK_IF(!GetNodeAttrs(convBpFilterNode),
                OP_LOGE(GetNodeType().GetString(), "GetNodeAttrs failed"), return nullptr);

    auto builder = EsGraphBuilder("replacement");
    auto [x, filterSize, outBackprop] = builder.CreateInputs<3>();
        
    auto conv3dBackpropFilterV2 = Conv3DBackpropFilterV2(
        x, filterSize, outBackprop, convBpAttr.strides, convBpAttr.pads, 
        convBpAttr.dilations, convBpAttr.groups,
        convBpAttr.dataFormat.c_str(), convBpAttr.hf32);

    auto* conv3dBackpropFilterV2Node = conv3dBackpropFilterV2.GetProducer();
    OP_CHECK_IF(conv3dBackpropFilterV2Node == nullptr,
                OP_LOGE(GetNodeType().GetString(), "Create Conv3DBackpropFilterV2 node failed"), return nullptr);

    conv3dBackpropFilterV2Node->SetAttr("_op_impl_mode_enum", convBpAttr.opImplModeEnum);
    OP_CHECK_IF(!UpdateNodeInputDescInfo(conv3dBackpropFilterV2Node),
                OP_LOGE(GetNodeType().GetString(), "Update conv3dBackpropFilterV2Node DescInfo failed"),
                return nullptr);
    
    EsTensorHolder finalY = conv3dBackpropFilterV2;
    bool needTranspose = CheckTransposeNeeded();
    if (needTranspose) {
        OP_CHECK_IF(!CreateOutputWithTranspose(builder, conv3dBackpropFilterV2, conv3dBackpropFilterV2Node, finalY),
                    OP_LOGE(GetNodeType().GetString(), "Create y with transpose failed"), return nullptr);
    } 
    
    return builder.BuildAndReset(std::vector<EsTensorHolder>{finalY});
}

REG_DECOMPOSE_PASS(Conv3DBackpropFilterToV2FusionPass, {CONV_BACKPROP_FILTER})
    .Stage(CustomPassStage::kCompatibleInherited);

} // namespace ops

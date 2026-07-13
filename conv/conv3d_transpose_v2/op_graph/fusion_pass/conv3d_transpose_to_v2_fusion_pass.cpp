/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "conv3d_transpose_to_v2_fusion_pass.h"

#include "es_nn_ops.h"
#include "log/log.h"
#include "register/register_custom_pass.h"

namespace ops {
using namespace ge;
using namespace ge::es;
using namespace fusion;
using namespace ConvBackpropFusionUtils;

namespace {
const AscendString PASS_NAME = "Conv3DTransposeToV2FusionPass";
const AscendString CONV3D_TRANSPOSE = "Conv3DTranspose";

constexpr size_t INPUT_SIZE_INDEX = 0;
constexpr size_t X_INDEX = 1;
constexpr size_t FILTER_INDEX = 2;
constexpr size_t BIAS_INDEX = 3;
constexpr size_t OFFSET_W_INDEX = 4;
constexpr size_t CONV3D_TRANSPOSE_INPUT_NUM = 3;

constexpr int64_t GROUPS_SINGLE = 1;
constexpr uint64_t MIN_KERNEL_VOLUME = 1;
constexpr uint64_t CIN_SPECIAL_VALUE = 16;
constexpr uint64_t CIN_MIN_THRESHOLD = 32;
constexpr float COUT_CIN_RATIO_THRESHOLD = 1.5f;
constexpr int64_t DEFAULT_OFFSET_X = 0;
const std::vector<int64_t> DEFAULT_OUTPUT_PADDING = {0, 0, 0, 0, 0};
const std::vector<int64_t> DEFAULT_OFFSET_W_VALUE = {0};
const std::vector<int32_t> FILTER_TRANSPOSE_PERM_NCDHW_TO_NDHWC = {0, 2, 3, 4, 1};
} // namespace

AscendString Conv3DTransposeToV2FusionPass::GetNodeType() const { return PASS_NAME; }

bool Conv3DTransposeToV2FusionPass::GetNodeDesc(const GNode& node)
{
    OP_CHECK_IF(!ConvBackpropFusionBasePass::GetNodeDesc(node),
                OP_LOGE(GetNodeType().GetString(), "Base GetNodeDesc failed"), return false);

    hasBias = false;
    hasOffsetW = false;
    biasDesc = TensorDesc();
    offsetWDesc = TensorDesc();

    if (node.GetInputsSize() > BIAS_INDEX) {
        node.GetInputDesc(BIAS_INDEX, biasDesc);
        hasBias = true;
    }
    if (node.GetInputsSize() > OFFSET_W_INDEX) {
        node.GetInputDesc(OFFSET_W_INDEX, offsetWDesc);
        hasOffsetW = true;
    }

    return true;
}

bool Conv3DTransposeToV2FusionPass::GetNodeAttrs(const GNode& node)
{
    OP_CHECK_IF(!ConvBackpropFusionBasePass::GetNodeAttrs(node),
                OP_LOGE(GetNodeType().GetString(), "Base GetNodeAttrs failed"), return false);

    outputPadding = DEFAULT_OUTPUT_PADDING;
    offsetX = DEFAULT_OFFSET_X;
    node.GetAttr("output_padding", outputPadding);
    node.GetAttr("offset_x", offsetX);

    return true;
}

bool Conv3DTransposeToV2FusionPass::CheckDtypeSupported()
{
    auto xDtype = input1Desc.GetDataType();
    auto filterDtype = input2Desc.GetDataType();
    auto yDtype = outputDesc.GetDataType();

    OP_CHECK_IF(xDtype == DT_HIFLOAT8 || xDtype == DT_FLOAT8_E4M3FN,
                OP_LOGE(GetNodeType().GetString(), "x dtype is not supported"), return false);
    OP_CHECK_IF(filterDtype == DT_HIFLOAT8 || filterDtype == DT_FLOAT8_E4M3FN,
                OP_LOGE(GetNodeType().GetString(), "filter dtype is not supported"), return false);
    OP_CHECK_IF(yDtype == DT_HIFLOAT8 || yDtype == DT_FLOAT8_E4M3FN,
                OP_LOGE(GetNodeType().GetString(), "y dtype is not supported"), return false);

    if (hasBias) {
        auto biasDtype = biasDesc.GetDataType();
        OP_CHECK_IF(biasDtype != DT_FLOAT16 && biasDtype != DT_BF16 && biasDtype != DT_FLOAT,
                    OP_LOGE(GetNodeType().GetString(), "bias dtype is not supported"), return false);
        OP_CHECK_IF(biasDtype != filterDtype && biasDtype != DT_FLOAT,
                    OP_LOGE(GetNodeType().GetString(), "bias dtype must match filter dtype or be float32"),
                    return false);
    }

    return true;
}

bool Conv3DTransposeToV2FusionPass::CheckTransposeNeeded()
{
    auto filterFormat = input2Desc.GetOriginFormat();
    OP_CHECK_IF(filterFormat != FORMAT_NCDHW,
                OP_LOGD(GetNodeType().GetString(), "filter format is not NCDHW, no transpose needed"), return false);

    OP_CHECK_IF(convBpAttr.groups != GROUPS_SINGLE,
                OP_LOGD(GetNodeType().GetString(), "groups=%lld, should be %lld", convBpAttr.groups, GROUPS_SINGLE),
                return false);

    auto filterDtype = input2Desc.GetDataType();
    OP_CHECK_IF(filterDtype != DT_FLOAT && filterDtype != DT_FLOAT16,
                OP_LOGD(GetNodeType().GetString(), "filter dtype must be float16 or float32"), return false);

    auto filterShape = input2Desc.GetShape().GetDims();
    for (size_t i = 0; i < filterShape.size(); ++i) {
        OP_CHECK_IF(filterShape[i] < 0, OP_LOGD(GetNodeType().GetString(), "filter shape must be specified"),
                    return false);
    }

    uint64_t cout = filterShape[N_DIM_NCDHW_INDEX];
    uint64_t cin = filterShape[C_DIM_NCDHW_INDEX];
    uint64_t dk = filterShape[D_DIM_NCDHW_INDEX];
    uint64_t hk = filterShape[H_DIM_NCDHW_INDEX];
    uint64_t wk = filterShape[W_DIM_NCDHW_INDEX];

    OP_CHECK_IF(dk * hk * wk <= MIN_KERNEL_VOLUME,
                OP_LOGD(GetNodeType().GetString(), "dk*hk*wk=%lu, too small", dk * hk * wk), return false);

    OP_CHECK_IF((cin != CIN_SPECIAL_VALUE && cin < CIN_MIN_THRESHOLD) || cin <= hk * wk,
                OP_LOGD(GetNodeType().GetString(), "cin=%lu, hk*wk=%lu, condition not met", cin, hk * wk),
                return false);

    bool ratioOk = (cout > cin) ? (cout < COUT_CIN_RATIO_THRESHOLD * cin) : (cin < COUT_CIN_RATIO_THRESHOLD * cout);
    OP_CHECK_IF(!ratioOk, OP_LOGD(GetNodeType().GetString(), "cout/cin ratio not in range"), return false);

    return true;
}

bool Conv3DTransposeToV2FusionPass::CreateFilterTranspose(EsGraphBuilder& builder, const EsTensorHolder& filter,
                                                          EsTensorHolder& transFilter, TensorDesc& transFilterDesc)
{
    auto config = TransposeNodeConfig::Create(filter, FILTER_TRANSPOSE_PERM_NCDHW_TO_NDHWC, "filter_transpose",
                                              Format::FORMAT_NDHWC);

    OP_CHECK_IF(
        !ConvBackpropFusionUtilsPass::CreateTransposeNode(builder, config, transFilter, transFilterDesc, GetNodeType()),
        OP_LOGE(GetNodeType().GetString(), "Create filter transpose node failed"), return false);

    return true;
}

GraphUniqPtr Conv3DTransposeToV2FusionPass::Replacement(const GNode& convTransposeNode)
{
    OP_LOGD(GetNodeType().GetString(), "Replacement start");

    OP_CHECK_IF(!GetNodeDesc(convTransposeNode), OP_LOGE(GetNodeType().GetString(), "GetNodeDesc failed"),
                return nullptr);

    OP_CHECK_IF(!GetNodeAttrs(convTransposeNode), OP_LOGE(GetNodeType().GetString(), "GetNodeAttrs failed"),
                return nullptr);

    OP_CHECK_IF(!CheckDtypeSupported(), OP_LOGE(GetNodeType().GetString(), "Dtype check failed"), return nullptr);

    auto builder = EsGraphBuilder("replacement");

    auto [inputSize, x, filter] = builder.CreateInputs<CONV3D_TRANSPOSE_INPUT_NUM>();
    ConvBackpropFusionUtilsPass::SetPlaceholderDesc(filter, 0, input2Desc);

    bool needTranspose = CheckTransposeNeeded();
    EsTensorHolder finalFilter = filter;
    TensorDesc transFilterDesc;

    if (needTranspose) {
        EsTensorHolder transFilter;
        OP_CHECK_IF(!CreateFilterTranspose(builder, filter, transFilter, transFilterDesc),
                    OP_LOGE(GetNodeType().GetString(), "Create filter transpose failed"), return nullptr);
        finalFilter = transFilter;
    }
    EsTensorHolder bias;
    EsTensorHolder defaultOffsetW;
    if (hasBias) {
        // 创建bias的输入节点
        bias = builder.CreateInput(static_cast<int64_t>(BIAS_INDEX));
    }
    if (hasOffsetW) {
        // 创建offset_w的输入节点
        defaultOffsetW = builder.CreateInput(static_cast<int64_t>(OFFSET_W_INDEX));
    }
    auto conv3dTransposeV2 = Conv3DTransposeV2(inputSize, x, finalFilter, bias, defaultOffsetW, convBpAttr.strides,
                                               convBpAttr.pads, convBpAttr.dilations, convBpAttr.groups,
                                               convBpAttr.dataFormat.c_str(), outputPadding, offsetX, convBpAttr.hf32);

    auto* v2Node = conv3dTransposeV2.GetProducer();
    OP_CHECK_IF(v2Node == nullptr, OP_LOGE(GetNodeType().GetString(), "Create Conv3DTransposeV2 node failed"),
                return nullptr);

    v2Node->SetAttr("_op_impl_mode_enum", convBpAttr.opImplModeEnum);
    v2Node->SetAttr("enable_hf32", convBpAttr.hf32);

    v2Node->UpdateInputDesc(INPUT_SIZE_INDEX, input0Desc);
    v2Node->UpdateInputDesc(X_INDEX, input1Desc);
    v2Node->UpdateInputDesc(FILTER_INDEX, needTranspose ? transFilterDesc : input2Desc);
    if (hasBias) {
        v2Node->UpdateInputDesc(BIAS_INDEX, biasDesc);
    }
    if (hasOffsetW) {
        v2Node->UpdateInputDesc(OFFSET_W_INDEX, offsetWDesc);
    }
    v2Node->UpdateOutputDesc(OUTPUT_INDEX, outputDesc);

    std::vector<EsTensorHolder> outputs;
    outputs.push_back(conv3dTransposeV2);
    return builder.BuildAndReset(outputs);
}

REG_DECOMPOSE_PASS(Conv3DTransposeToV2FusionPass, {CONV3D_TRANSPOSE}).Stage(CustomPassStage::kCompatibleInherited);

} // namespace ops

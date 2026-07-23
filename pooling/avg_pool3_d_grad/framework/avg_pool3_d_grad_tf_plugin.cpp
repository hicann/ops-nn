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
 * \file avg_pool3_d_grad_tf_plugin.cpp
 * \brief
 */
#include "register/register.h"
#include "framework/plugin_util.h"

namespace domi {
// TF AvgPool3DGrad: auto mapping plus data_format on input/output and default pads/ceil_mode attrs.
static Status ParseParamsAvgPool3DGrad(const google::protobuf::Message* op_src, ge::Operator& op)
{
    if (AutoMappingFn(op_src, op) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "auto mapping failed.");
        return FAILED;
    }

    ge::Format data_format = ge::FORMAT_NDHWC;
    std::string data_format_attr;
    if (op.GetAttr("data_format", data_format_attr) == ge::GRAPH_SUCCESS) {
        if (data_format_attr != "NCDHW" && data_format_attr != "NDHWC") {
            OP_LOGE(GetOpName(op).c_str(), "data_format only support NCDHW and NDHWC.");
            return FAILED;
        }
        data_format = data_format_attr == "NCDHW" ? ge::FORMAT_NCDHW : ge::FORMAT_NDHWC;
    }

    auto tensor_grads = op.GetInputDesc(1);
    tensor_grads.SetOriginFormat(data_format);
    tensor_grads.SetFormat(data_format);
    if (op.UpdateInputDesc(static_cast<uint32_t>(0), tensor_grads) != ge::GRAPH_SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "update input:grads desc failed.");
        return FAILED;
    }

    auto tensor_out = op.GetOutputDesc(0);
    tensor_out.SetOriginFormat(data_format);
    tensor_out.SetFormat(data_format);
    if (op.UpdateOutputDesc(static_cast<uint32_t>(0), tensor_out) != ge::GRAPH_SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "update output desc failed.");
        return FAILED;
    }

    // Escape GE require attr [pads] check here, and set default attrs for avg_pool3d_grad.
    std::vector<int32_t> pad_list = {0, 0, 0, 0, 0, 0};
    (void)op.SetAttr("pads", pad_list);
    (void)op.SetAttr("ceil_mode", false);
    (void)op.SetAttr("count_include_pad", false);
    (void)op.SetAttr("divisor_override", 0);
    return SUCCESS;
}

REGISTER_CUSTOM_OP("AvgPool3DGrad")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("AvgPool3DGrad")
    .ParseParamsFn(ParseParamsAvgPool3DGrad)
    .ImplyType(ImplyType::TVM);
} // namespace domi

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
 * \file dilation2d_tf_plugin.cpp
 * \brief
 */
#include "register/register.h"
#include "framework/plugin_util.h"
#include "graph/operator.h"

#include "log/log.h"

namespace domi {
static Status ParseDilation2D(const ge::Operator& op_src, ge::Operator& op)
{
    if (AutoMappingByOpFn(op_src, op) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "auto mapping failed.");
        return FAILED;
    }

    ge::TensorDesc input_tensor = op.GetInputDescByName("x");
    input_tensor.SetOriginFormat(ge::FORMAT_NHWC);
    input_tensor.SetFormat(ge::FORMAT_NHWC);
    auto ret = op.UpdateInputDesc("x", input_tensor);
    if (ret != ge::GRAPH_SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "Update input format failed.");
        return FAILED;
    }
    ge::TensorDesc output_tensor = op.GetOutputDescByName("y");
    output_tensor.SetOriginFormat(ge::FORMAT_NHWC);
    output_tensor.SetFormat(ge::FORMAT_NHWC);
    auto ret_output = op.UpdateOutputDesc("y", output_tensor);
    if (ret_output != ge::GRAPH_SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "Update output format failed.");
        return FAILED;
    }
    ge::TensorDesc filter_tensor = op.GetInputDescByName("filter");
    filter_tensor.SetOriginFormat(ge::FORMAT_NHWC);
    filter_tensor.SetFormat(ge::FORMAT_NHWC);
    auto filter_ret = op.UpdateInputDesc("filter", filter_tensor);
    if (filter_ret != ge::GRAPH_SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "Update filter format failed.");
        return FAILED;
    }
    std::string padding;
    if (op.GetAttr("padding", padding) == ge::GRAPH_SUCCESS) {
        (void)op.SetAttr("padding_mode", padding);
    }
    return SUCCESS;
}
// register Dilation2D op to GE
REGISTER_CUSTOM_OP("Dilation2D")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("Dilation2D")
    .ParseParamsByOperatorFn(ParseDilation2D)
    .ImplyType(ImplyType::TVM);
} // namespace domi

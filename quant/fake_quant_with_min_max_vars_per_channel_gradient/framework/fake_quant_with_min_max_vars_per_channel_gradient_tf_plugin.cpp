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
 * \file fake_quant_with_min_max_vars_per_channel_gradient_tf_plugin.cpp
 * \brief
 */
#include "register/register.h"
#include "framework/plugin_util.h"

namespace domi {
// TensorFlow FakeQuantWithMinMaxVarsPerChannelGradient: auto mapping plus NHWC format setup on inputs/output.
static Status ParseParamsFakeQuantWithMinMaxVarsPerChannelGradient(const google::protobuf::Message* op_src,
                                                                   ge::Operator& op)
{
    if (AutoMappingFn(op_src, op) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "auto mapping failed.");
        return FAILED;
    }
    for (size_t i = 0; i < op.GetInputsSize(); i++) {
        auto tensor = op.GetInputDesc(i);
        tensor.SetOriginFormat(ge::FORMAT_NHWC);
        tensor.SetFormat(ge::FORMAT_NHWC);
        if (op.UpdateInputDesc(i, tensor) != ge::GRAPH_SUCCESS) {
            OP_LOGE(GetOpName(op).c_str(), "update input format failed.");
            return FAILED;
        }
    }
    for (size_t i = 0; i < op.GetOutputsSize(); i++) {
        auto tensor = op.GetOutputDesc(i);
        tensor.SetOriginFormat(ge::FORMAT_NHWC);
        tensor.SetFormat(ge::FORMAT_NHWC);
        if (op.UpdateOutputDesc(i, tensor) != ge::GRAPH_SUCCESS) {
            OP_LOGE(GetOpName(op).c_str(), "update output format failed.");
            return FAILED;
        }
    }
    return SUCCESS;
}

REGISTER_CUSTOM_OP("FakeQuantWithMinMaxVarsPerChannelGradient")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("FakeQuantWithMinMaxVarsPerChannelGradient")
    .ParseParamsFn(ParseParamsFakeQuantWithMinMaxVarsPerChannelGradient)
    .ImplyType(ImplyType::TVM);
} // namespace domi

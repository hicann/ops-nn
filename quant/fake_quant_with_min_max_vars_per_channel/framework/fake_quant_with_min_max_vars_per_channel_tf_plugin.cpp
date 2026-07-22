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
 * \file fake_quant_with_min_max_vars_per_channel_tf_plugin.cpp
 * \brief
 */
#include "register/register.h"
#include "framework/plugin_util.h"

namespace domi {
// TensorFlow FakeQuantWithMinMaxVarsPerChannel: auto mapping plus NHWC format setup on input/output.
static Status ParseParamsFakeQuantWithMinMaxVarsPerChannel(const google::protobuf::Message* op_src, ge::Operator& op)
{
    if (AutoMappingFn(op_src, op) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "auto mapping failed.");
        return FAILED;
    }
    auto tensor_in = op.GetInputDesc(0);
    tensor_in.SetOriginFormat(ge::FORMAT_NHWC);
    tensor_in.SetFormat(ge::FORMAT_NHWC);
    if (op.UpdateInputDesc(static_cast<uint32_t>(0), tensor_in) != ge::GRAPH_SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "update input format failed.");
        return FAILED;
    }
    auto tensor_out = op.GetOutputDesc(0);
    tensor_out.SetOriginFormat(ge::FORMAT_NHWC);
    tensor_out.SetFormat(ge::FORMAT_NHWC);
    if (op.UpdateOutputDesc(static_cast<uint32_t>(0), tensor_out) != ge::GRAPH_SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "update output format failed.");
        return FAILED;
    }
    return SUCCESS;
}

REGISTER_CUSTOM_OP("FakeQuantWithMinMaxVarsPerChannel")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("FakeQuantWithMinMaxVarsPerChannel")
    .ParseParamsFn(ParseParamsFakeQuantWithMinMaxVarsPerChannel)
    .ImplyType(ImplyType::TVM);
} // namespace domi

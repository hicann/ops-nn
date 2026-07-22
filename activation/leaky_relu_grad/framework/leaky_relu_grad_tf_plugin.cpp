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
 * \file leaky_relu_grad_tf_plugin.cpp
 * \brief
 */
#include "register/register.h"
#include "framework/plugin_util.h"

namespace domi {
// TensorFlow LeakyReluGrad uses attribute `alpha`, while the GE op expects `negative_slope`.
static Status ParseParamsLeakyReluGrad(const google::protobuf::Message* op_src, ge::Operator& op)
{
    if (AutoMappingFn(op_src, op) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "auto mapping failed.");
        return FAILED;
    }
    float alpha = 0.0f;
    if (op.GetAttr("alpha", alpha) != ge::GRAPH_SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get attr alpha failed.");
        return FAILED;
    }
    (void)op.SetAttr("negative_slope", alpha);
    return SUCCESS;
}

REGISTER_CUSTOM_OP("LeakyReluGrad")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("LeakyReluGrad")
    .ParseParamsFn(ParseParamsLeakyReluGrad)
    .ImplyType(ImplyType::TVM);
} // namespace domi

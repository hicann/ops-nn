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
 * \file max_pool3d_grad_tf_plugin.cpp
 * \brief
 */
#include "register/register.h"
#include "framework/plugin_util.h"

namespace domi {
// TensorFlow MaxPool3DGrad: auto mapping plus default pads attr to bypass GE required attr check.
static Status ParseParamsMaxPool3DGrad(const google::protobuf::Message* op_src, ge::Operator& op)
{
    if (AutoMappingFn(op_src, op) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "auto mapping failed.");
        return FAILED;
    }
    // Escape GE require attr [pads] check here
    std::vector<int32_t> pad_list = {0, 0, 0, 0, 0, 0};
    (void)op.SetAttr("pads", pad_list);
    return SUCCESS;
}

REGISTER_CUSTOM_OP("MaxPool3DGrad")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("MaxPool3DGrad")
    .ParseParamsFn(ParseParamsMaxPool3DGrad)
    .ImplyType(ImplyType::TVM);
} // namespace domi

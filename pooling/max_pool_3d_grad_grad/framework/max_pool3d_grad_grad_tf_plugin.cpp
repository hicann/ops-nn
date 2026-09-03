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
 * \file max_pool3d_grad_grad_tf_plugin.cpp
 * \brief
 */
#include "register/register.h"
#include "graph/operator.h"

#include "log/log.h"

namespace domi {
// Replace ge ParseParams fuction to process graph maxpool3dgradgrad node attrs
static Status ParseParamsMaxPool3DGradGrad(const ge::Operator& op_src, ge::Operator& op)
{
    // Convert original tf graph maxpool3dgradgrad attrs to GE graph attrs
    if (AutoMappingByOpFn(op_src, op) != SUCCESS) {
        return FAILED;
    }
    // Escape GE require attr [pads] check here
    std::vector<int32_t> padList = {0, 0, 0, 0, 0, 0};
    (void)op.SetAttr("pads", padList);
    return SUCCESS;
}

REGISTER_CUSTOM_OP("MaxPool3DGradGrad")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("MaxPool3DGradGrad")
    .ParseParamsByOperatorFn(ParseParamsMaxPool3DGradGrad)
    .ImplyType(ImplyType::TVM);
} // namespace domi

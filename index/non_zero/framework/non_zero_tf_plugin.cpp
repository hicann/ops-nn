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
 * \file non_zero_tf_plugin.cpp
 * \brief
 */
#include "register/register.h"
#include "framework/plugin_util.h"

namespace domi {
// TF NonZero: auto mapping plus copy output_type attr into dtype for the GE op.
static Status ParseParamsNonZero(const google::protobuf::Message* op_src, ge::Operator& op)
{
    if (AutoMappingFn(op_src, op) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "auto mapping failed.");
        return FAILED;
    }
    ge::DataType output_type;
    if (op.GetAttr("output_type", output_type) == ge::GRAPH_SUCCESS) {
        (void)op.SetAttr("dtype", output_type);
    }
    return SUCCESS;
}

REGISTER_CUSTOM_OP("NonZero")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("NonZero")
    .ParseParamsFn(ParseParamsNonZero)
    .ImplyType(ImplyType::TVM);
} // namespace domi

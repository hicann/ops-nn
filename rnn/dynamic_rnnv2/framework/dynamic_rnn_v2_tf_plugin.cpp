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
 * \file dynamic_rnn_v2_tf_plugin.cpp
 * \brief
 */
#include "register/register.h"

namespace domi {
// TF DynamicRNN (DynamicRnnV2): auto mapping plus is_misplaced attr for compatibility.
static Status ParseParamsDynamicRNNV2(const google::protobuf::Message* op_src, ge::Operator& op)
{
    if (AutoMappingFn(op_src, op) != SUCCESS) {
        return FAILED;
    }
    (void)op.SetAttr("is_misplaced", true);
    return SUCCESS;
}

REGISTER_CUSTOM_OP("DynamicRNN")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("DynamicRnnV2")
    .ParseParamsFn(ParseParamsDynamicRNNV2)
    .ImplyType(ImplyType::TVM);
} // namespace domi

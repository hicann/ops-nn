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
 * \file log_softmax_v2_tf_plugin.cpp
 * \brief
 */
#include "register/register.h"
#include "framework/plugin_util.h"

namespace domi {
// TensorFlow LogSoftmax uses attribute `axis`, while the GE op expects `axes`.
static Status ParseParamsLogSoftmaxV2(const google::protobuf::Message* op_src, ge::Operator& op)
{
    if (AutoMappingFn(op_src, op) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "auto mapping failed.");
        return FAILED;
    }
    std::vector<int64_t> axis = {-1};
    if (op.GetAttr("axis", axis) != ge::GRAPH_SUCCESS) {
        OP_LOGW(GetOpName(op).c_str(), "get attr axis failed, use default [-1].");
    }
    (void)op.SetAttr("axes", axis);
    return SUCCESS;
}

REGISTER_CUSTOM_OP("LogSoftmaxV2")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("LogSoftmax")
    .ParseParamsFn(ParseParamsLogSoftmaxV2)
    .ImplyType(ImplyType::TVM);
} // namespace domi

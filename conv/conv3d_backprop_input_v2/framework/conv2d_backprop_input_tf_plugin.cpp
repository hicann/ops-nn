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
 * \file conv2d_backprop_input_tf_plugin.cpp
 * \brief
 */
#include <map>
#include "graph/utils/type_utils.h"
#include "log/log.h"
#include "register/register.h"
#include "error_util.h"

namespace domi {
namespace {
const int32_t CV_NUM_1 = 1;
}
static Status ParseParamsConv2DBackpropInput(const ge::Operator& op_src, ge::Operator& op)
{
    ge::AscendString op_name;
    OP_LOGE_IF(op.GetName(op_name) != ge::GRAPH_SUCCESS, FAILED, "", "failed to get op_name");

    OP_LOGD(op_name.GetString(), "Enter ParseParamsConv2DBackpropInput.");

    if (AutoMappingByOpFn(op_src, op) != SUCCESS) {
        OP_LOGE(op_name.GetString(), "AutoMappingFn failed.");
        return FAILED;
    }

    ge::TensorDesc org_tensor_w = op.GetInputDesc(CV_NUM_1);
    org_tensor_w.SetOriginFormat(ge::FORMAT_HWCN);
    org_tensor_w.SetFormat(ge::FORMAT_HWCN);
    auto ret = op.UpdateInputDesc(CV_NUM_1, org_tensor_w);

    OP_LOGE_IF(ret != ge::GRAPH_SUCCESS, FAILED, op_name.GetString(), "failed to update filter format.");
    OP_LOGD(op_name.GetString(), "update filter format succeeded, now is [%s]",
            ge::TypeUtils::FormatToAscendString(op.GetInputDesc(CV_NUM_1).GetFormat()).GetString());

    // Escape GE require attr [pads] check here
    std::vector<int32_t> pad_list = {0, 0, 0, 0};
    op.SetAttr("pads", pad_list);

    return SUCCESS;
}

REGISTER_CUSTOM_OP("Conv2DBackpropInput")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("Conv2DBackpropInput")
    .ParseParamsByOperatorFn(ParseParamsConv2DBackpropInput)
    .ImplyType(ImplyType::TVM);
} // namespace domi

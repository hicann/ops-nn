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
 * \file conv3d_backprop_filter_tf_plugin.cpp
 * \brief
 */
#include <map>

#include "register/register.h"
#include "error_util.h"

#include "log/log.h"

namespace domi {
namespace {
const int32_t kInputIdx0 = 0;
const int32_t kInputIdx1 = 1;
const int32_t kOutputIdx0 = 0;
} // namespace

static Status ParseParamsConv3DBackpropFilter(const ge::Operator& op_src, ge::Operator& op)
{
    ge::AscendString op_name;
    OP_LOGE_IF(op.GetName(op_name) != ge::GRAPH_SUCCESS, FAILED, "", "failed to get op_name");

    OP_LOGD(op_name.GetString(), "Enter ParseParamsConv3DBackpropFilter.");
    if (AutoMappingByOpFn(op_src, op) != SUCCESS) {
        OP_LOGE(op_name.GetString(), "AutoMappingFn failed.");
        return FAILED;
    }

    ge::Format data_format = ge::FORMAT_NDHWC;
    std::string data_format_attr;
    if (op.GetAttr("data_format", data_format_attr) == ge::GRAPH_SUCCESS) {
        if (data_format_attr == "NCDHW") {
            data_format = ge::FORMAT_NCDHW;
        }
    }

    ge::TensorDesc org_tensor_x = op.GetInputDesc(kInputIdx0);
    org_tensor_x.SetOriginFormat(data_format);
    org_tensor_x.SetFormat(data_format);
    auto ret = op.UpdateInputDesc(kInputIdx0, org_tensor_x);
    OP_LOGE_IF(ret != ge::GRAPH_SUCCESS, FAILED, op_name.GetString(), "failed to update input_x format.");

    ge::TensorDesc org_tensor_y = op.GetInputDesc(kInputIdx1);
    org_tensor_y.SetOriginFormat(data_format);
    org_tensor_y.SetFormat(data_format);
    ret = op.UpdateInputDesc(kInputIdx1, org_tensor_y);
    OP_LOGE_IF(ret != ge::GRAPH_SUCCESS, FAILED, op_name.GetString(), "failed to update out_backprop format.");

    ge::TensorDesc org_tensor_w = op.GetOutputDesc(kOutputIdx0);
    org_tensor_w.SetOriginFormat(ge::FORMAT_DHWCN);
    org_tensor_w.SetFormat(ge::FORMAT_DHWCN);
    ret = op.UpdateOutputDesc(kInputIdx0, org_tensor_w);
    OP_LOGE_IF(ret != ge::GRAPH_SUCCESS, FAILED, op_name.GetString(), "failed to update output dw format.");
    std::vector<int32_t> pad_list = {0, 0, 0, 0, 0, 0};
    op.SetAttr("pads", pad_list);

    OP_LOGD(op_name.GetString(), "update output dw format succeeded.");

    OP_LOGD(op_name.GetString(), "Exit ParseParamsConv3DBackpropFilter.");
    return SUCCESS;
}

REGISTER_CUSTOM_OP("Conv3DBackpropFilter")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("Conv3DBackpropFilterV2")
    .ParseParamsByOperatorFn(ParseParamsConv3DBackpropFilter)
    .ImplyType(ImplyType::TVM);
} // namespace domi

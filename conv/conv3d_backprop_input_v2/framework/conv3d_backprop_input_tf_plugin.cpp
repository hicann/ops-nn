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
 * \file conv3d_backprop_input_tf_plugin.cpp
 * \brief
 */
#include <map>

#include "register/register.h"
#include "error_util.h"

#include "log/log.h"

namespace domi {
namespace {
const int32_t kIndex0 = 0;
const int32_t kIndex1 = 1;
} // namespace

static Status ParseParamsConv3DBackpropInput(const ge::Operator& op_src, ge::Operator& op)
{
    ge::AscendString op_name;
    OP_LOGE_IF(op.GetName(op_name) != ge::GRAPH_SUCCESS, FAILED, "", "failed to get op_name");

    OP_LOGD(op_name.GetString(), "Enter ParseParamsConv3DBackpropInput.");

    if (AutoMappingByOpFn(op_src, op) != SUCCESS) {
        OP_LOGE(op_name.GetString(), "AutoMappingFn failed.");
        return FAILED;
    }
    ge::TensorDesc org_tensor_w = op.GetInputDesc(kIndex1);
    org_tensor_w.SetOriginFormat(ge::FORMAT_DHWCN);
    org_tensor_w.SetFormat(ge::FORMAT_DHWCN);
    auto ret = op.UpdateInputDesc(kIndex1, org_tensor_w);
    OP_LOGE_IF(ret != ge::GRAPH_SUCCESS, FAILED, op_name.GetString(), "failed to update filter format.");
    OP_LOGD(op_name.GetString(), "Update filter format succeeded");

    ge::Format data_format = ge::FORMAT_NDHWC;
    std::string data_format_attr;
    if (op.GetAttr("data_format", data_format_attr) == ge::GRAPH_SUCCESS) {
        if (data_format_attr == "NCDHW") {
            data_format = ge::FORMAT_NCDHW;
        }
    }

    ge::TensorDesc org_tensor_y = op.GetInputDesc(kIndex0);
    org_tensor_y.SetOriginFormat(data_format);
    org_tensor_y.SetFormat(data_format);
    ret = op.UpdateInputDesc(kIndex0, org_tensor_y);
    OP_LOGE_IF(ret != ge::GRAPH_SUCCESS, FAILED, op_name.GetString(), "failed to update input out_backprop format.");
    OP_LOGD(op_name.GetString(), "update input out_backprop format succeeded.");

    ge::TensorDesc org_tensor_x = op.GetOutputDesc(kIndex0);
    org_tensor_x.SetOriginFormat(data_format);
    org_tensor_x.SetFormat(data_format);
    ret = op.UpdateOutputDesc(kIndex0, org_tensor_x);
    OP_LOGE_IF(ret != ge::GRAPH_SUCCESS, FAILED, op_name.GetString(), "failed to update output dx format.");

    std::vector<int32_t> pad_list = {0, 0, 0, 0, 0, 0};
    op.SetAttr("pads", pad_list);

    OP_LOGD(op_name.GetString(), "update output dx format succeeded.");

    OP_LOGD(op_name.GetString(), "Exit ParseParamsConv3DBackpropInput.");

    return SUCCESS;
}

REGISTER_CUSTOM_OP("Conv3DBackpropInput")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("Conv3DBackpropInputV2")
    .ParseParamsByOperatorFn(ParseParamsConv3DBackpropInput)
    .ImplyType(ImplyType::TVM);
} // namespace domi

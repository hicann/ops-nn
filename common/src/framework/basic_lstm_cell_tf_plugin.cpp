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
 * \file basic_lstm_cell_tf_plugin.cpp
 * \brief
 */
#include "register/register.h"
#include "framework/plugin_util.h"
#include "graph/operator.h"

#include "log/log.h"

namespace domi {
const uint32_t kPosition = 3;

static Status BasicLSTMCellParserParams(const std::vector<const google::protobuf::Message*> inside_nodes,
                                        ge::Operator& op)
{
    OP_LOGI(GetOpName(op).c_str(), "Enter BasicLSTMCell fusion parser.");

    ge::TensorDesc input_desc = op.GetInputDesc(kPosition);
    input_desc.SetOriginFormat(ge::FORMAT_HWCN);
    input_desc.SetFormat(ge::FORMAT_HWCN);

    if (op.UpdateInputDesc(kPosition, input_desc) != ge::GRAPH_SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "Update input desc fail, index:%u.", kPosition);
        return FAILED;
    }

    return SUCCESS;
}

static Status ParseParamsBasicLSTMCell(const ge::Operator& op_src, ge::Operator& op)
{
    if (AutoMappingByOpFn(op_src, op) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "tensorflow plugin parser failed. auto mapping failed.");
        return FAILED;
    }
    ge::TensorDesc orgTensorW = op.GetInputDesc(kPosition);
    orgTensorW.SetOriginFormat(ge::FORMAT_HWCN);
    orgTensorW.SetFormat(ge::FORMAT_HWCN);
    auto ret = op.UpdateInputDesc(kPosition, orgTensorW);
    if (ret != ge::GRAPH_SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "update filter format failed.");
        return FAILED;
    }
    return SUCCESS;
}

REGISTER_CUSTOM_OP("BasicLSTMCell")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("BasicLSTMCell")
    .ParseParamsByOperatorFn(ParseParamsBasicLSTMCell)
    .FusionParseParamsFn(BasicLSTMCellParserParams)
    .ImplyType(ImplyType::TVM);
} // namespace domi

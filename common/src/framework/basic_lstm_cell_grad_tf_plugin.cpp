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
 * \file basic_lstm_cell_grad_tf_plugin.cpp
 * \brief
 */
#include "register/register.h"
#include "framework/plugin_util.h"
#include "graph/operator.h"

#include "log/log.h"

namespace domi {
const int POS_0 = 0;
const int POS_1 = 1;

static Status ParseParamsBasicLSTMCellInputGrad(const ge::Operator& op_src, ge::Operator& op)
{
    if (AutoMappingByOpFn(op_src, op) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "tensorflow plugin parser failed. auto mapping failed.");
        return FAILED;
    }
    ge::TensorDesc orgTensorW = op.GetInputDesc(POS_1);
    orgTensorW.SetOriginFormat(ge::FORMAT_HWCN);
    orgTensorW.SetFormat(ge::FORMAT_HWCN);
    auto ret = op.UpdateInputDesc(POS_1, orgTensorW);
    if (ret != ge::GRAPH_SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "update filter format failed.");
        return FAILED;
    }
    return SUCCESS;
}

static Status ParseParamsBasicLSTMCellWeightGrad(const ge::Operator& op_src, ge::Operator& op)
{
    if (AutoMappingByOpFn(op_src, op) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "tensorflow plugin parser failed. auto mapping failed.");
        return FAILED;
    }
    ge::TensorDesc orgTensorW = op.GetOutputDesc(POS_0);
    orgTensorW.SetOriginFormat(ge::FORMAT_HWCN);
    orgTensorW.SetFormat(ge::FORMAT_HWCN);
    auto ret = op.UpdateOutputDesc(POS_0, orgTensorW);
    if (ret != ge::GRAPH_SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "update filter format failed.");
        return FAILED;
    }
    return SUCCESS;
}

static Status ParseParamsBasicLSTMCellCStateGrad(const ge::Operator& op_src, ge::Operator& op)
{
    if (AutoMappingByOpFn(op_src, op) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "tensorflow plugin parser failed. auto mapping failed.");
        return FAILED;
    }

    return SUCCESS;
}

REGISTER_CUSTOM_OP("BasicLSTMCellCStateGrad")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("BasicLSTMCellCStateGrad")
    .ParseParamsByOperatorFn(ParseParamsBasicLSTMCellCStateGrad)
    .ImplyType(ImplyType::TVM);

REGISTER_CUSTOM_OP("BasicLSTMCellWeightGrad")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("BasicLSTMCellWeightGrad")
    .ParseParamsByOperatorFn(ParseParamsBasicLSTMCellWeightGrad)
    .ImplyType(ImplyType::TVM);

REGISTER_CUSTOM_OP("BasicLSTMCellInputGrad")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("BasicLSTMCellInputGrad")
    .ParseParamsByOperatorFn(ParseParamsBasicLSTMCellInputGrad)
    .ImplyType(ImplyType::TVM);
} // namespace domi

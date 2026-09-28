/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the License).
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file dynamic_augru_grad_graph_infer.cpp
 * \brief DynamicAUGRUGrad InferDataType推导
 */

#include "register/op_impl_registry.h"
#include "log/log.h"

namespace ops {
using namespace ge;

static constexpr size_t IDX_X = 0;
static constexpr size_t FLOAT_INPUT_NUM = 14; // x~hidden_new，不含seq_length(INT32)/mask(UINT8)
static constexpr const char* K_FLOAT_INPUT_NAMES[FLOAT_INPUT_NUM] = {
    "x",  "weight_input", "weight_hidden", "weight_att", "y",     "init_h", "h",
    "dy", "dh",           "update",        "update_att", "reset", "new",    "hidden_new"};
static constexpr size_t FLOAT_OUTPUT_NUM = 7; // 全部7个输出dtype与x一致

static ge::graphStatus InferDataTypeDynamicAUGRUGrad(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferDataTypeDynamicAUGRUGrad");
    // 浮点输入dtype必须与x一致（proto契约）；混用会被GE precision_reduce自动cast
    // 产生静默精度劣化，须在推导阶段显式拒绝。seq_length为INT32、mask为UINT8不参与
    ge::DataType xDtype = context->GetInputDataType(IDX_X);
    for (size_t i = IDX_X + 1; i < FLOAT_INPUT_NUM; i++) {
        ge::DataType dtype = context->GetInputDataType(i);
        OP_CHECK_IF(
            dtype != xDtype,
            OP_LOGE(context,
                    "The dtype of %s should be same as x, but dtype %d of x and dtype %d of %s "
                    "were obtained.",
                    K_FLOAT_INPUT_NAMES[i], static_cast<int>(xDtype), static_cast<int>(dtype), K_FLOAT_INPUT_NAMES[i]),
            return GRAPH_FAILED);
    }
    // 全部浮点输出与x保持一致（seq_length为INT32不参与）
    for (size_t i = 0; i < FLOAT_OUTPUT_NUM; i++) {
        context->SetOutputDataType(i, context->GetInputDataType(IDX_X));
    }
    OP_LOGD(context->GetNodeName(), "End to do InferDataTypeDynamicAUGRUGrad");
    return GRAPH_SUCCESS;
}

IMPL_OP(DynamicAUGRUGrad).InferDataType(InferDataTypeDynamicAUGRUGrad);
} // namespace ops

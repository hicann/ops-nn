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
 * \file smooth_l1_loss_graph_infer.cpp
 * \brief 图模式 dtype 推导注册
 *
 * 输出 dtype 推导规则：loss 与 predict 同类型；predict 限 FP32/FP16/BF16 且与 label 一致。
 *
 * 注册方式：使用通用 IMPL_OP(SmoothL1Loss) 宏注册 InferDataType，
 * 与 op_host/smooth_l1_loss_infershape.cpp 中的 IMPL_OP_INFERSHAPE 注册并列存在、互不冲突。
 */
#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;

namespace ops {
static constexpr int64_t IDX_PREDICT = 0;
static constexpr int64_t IDX_LABEL = 1;
static constexpr int64_t IDX_LOSS = 0;

ge::graphStatus InferDataTypeSmoothL1Loss(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferDataTypeSmoothL1Loss");
    ge::DataType predictDtype = context->GetInputDataType(IDX_PREDICT);
    if (predictDtype != ge::DT_FLOAT && predictDtype != ge::DT_FLOAT16 && predictDtype != ge::DT_BF16) {
        OP_LOGE(context->GetNodeName(), "predict only supports float, float16 and bfloat16");
        return ge::GRAPH_FAILED;
    }
    if (context->GetInputDataType(IDX_LABEL) != predictDtype) {
        OP_LOGE(context->GetNodeName(), "predict and label dtype must be the same");
        return ge::GRAPH_FAILED;
    }
    context->SetOutputDataType(IDX_LOSS, predictDtype);
    OP_LOGD(context->GetNodeName(), "End to do InferDataTypeSmoothL1Loss");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(SmoothL1Loss).InferDataType(InferDataTypeSmoothL1Loss);
} // namespace ops

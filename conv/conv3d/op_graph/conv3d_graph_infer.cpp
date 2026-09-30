/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file conv3d_graph_infer.cpp
 * \brief Conv3D 算子 RT2.0 InferDataType 实现（自 canndev RT1.0 SetOutDtype 迁移）
 *
 * 输出 dtype 推导规则（对齐 RT1.0 nn_calculation_ops.cc SetOutDtype）：
 *   y.dtype = x.dtype，但当 x.dtype 为 int8 时 y.dtype 为 int32。
 */

#include "log/log.h"
#include "register/op_impl_registry.h"

#include "conv3d_proto.h"

namespace ops {
namespace {
constexpr size_t CONV3D_X_IDX = 0;
constexpr size_t CONV3D_Y_IDX = 0;
} // namespace

ge::graphStatus InferDataTypeConv3D(gert::InferDataTypeContext* context)
{
    if (context == nullptr) {
        OP_LOGE("Conv3D", "context is null.");
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context->GetNodeName(), "Begin InferDataTypeConv3D.");

    const ge::DataType xDtype = context->GetInputDataType(CONV3D_X_IDX);
    const ge::DataType yDtype = (xDtype == ge::DT_INT8) ? ge::DT_INT32 : xDtype;
    context->SetOutputDataType(CONV3D_Y_IDX, yDtype);

    OP_LOGD(context->GetNodeName(), "Set y dtype: %s success.", Ops::Base::ToString(yDtype).c_str());
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(Conv3D).InferDataType(InferDataTypeConv3D);
} // namespace ops

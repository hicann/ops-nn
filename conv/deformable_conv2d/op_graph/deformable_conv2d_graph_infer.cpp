/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file deformable_conv2d_graph_infer.cpp
 * \brief DeformableConv2D InferDataType.
 */

#include "log/log.h"
#include "register/op_impl_registry.h"

namespace ops {
namespace {
constexpr size_t X_IDX_DFM = 0;
constexpr size_t Y_IDX_DFM = 0;
} // namespace

ge::graphStatus InferDataTypeForDeformableConv2D(gert::InferDataTypeContext* context)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, context);
    OP_LOGD(context->GetNodeName(), "Begin InferDataTypeForDeformableConv2D.");

    const ge::DataType xDtype = context->GetInputDataType(X_IDX_DFM);
    // quantized int8 input maps to int32 output
    const ge::DataType yDtype = (xDtype == ge::DT_INT8) ? ge::DT_INT32 : xDtype;
    const ge::graphStatus ret = context->SetOutputDataType(Y_IDX_DFM, yDtype);

    OP_LOGD(context->GetNodeName(), "Set y dtype success.");
    return ret;
}

IMPL_OP(DeformableConv2D).InferDataType(InferDataTypeForDeformableConv2D);
} // namespace ops

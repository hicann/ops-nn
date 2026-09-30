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
 * \file conv2d_graph_infer.cpp
 * \brief Conv2D InferDataType (RT1.0 SetOutputDtypeConv2d, isQuantFlag==false).
 */

#include "exe_graph/runtime/infer_datatype_context.h"
#include "log/log.h"
#include "register/op_impl_registry.h"

namespace ops {
namespace {
constexpr size_t X_IDX_CONV2D = 0;
constexpr size_t W_IDX_CONV2D = 1;
constexpr size_t Y_IDX_CONV2D = 0;
} // namespace

ge::graphStatus InferDataTypeForConv2D(gert::InferDataTypeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const ge::DataType xDtype = context->GetInputDataType(X_IDX_CONV2D);
    const ge::DataType wDtype = context->GetInputDataType(W_IDX_CONV2D);
    ge::DataType yDtype = xDtype;
    switch (xDtype) {
        case ge::DT_INT8:
        case ge::DT_INT4:
        case ge::DT_INT16:
            yDtype = ge::DT_INT32;
            break;
        default:
            yDtype = xDtype;
            break;
    }
    if ((xDtype == ge::DT_FLOAT || xDtype == ge::DT_FLOAT16) && wDtype == ge::DT_INT8) {
        yDtype = ge::DT_INT32;
    }
    OP_LOGD(context->GetNodeName(), "Infer xDtype[%d] wDtype[%d] yDtype[%d].", static_cast<int32_t>(xDtype),
            static_cast<int32_t>(wDtype), static_cast<int32_t>(yDtype));
    return context->SetOutputDataType(Y_IDX_CONV2D, yDtype);
}

IMPL_OP(Conv2D).InferDataType(InferDataTypeForConv2D);
} // namespace ops

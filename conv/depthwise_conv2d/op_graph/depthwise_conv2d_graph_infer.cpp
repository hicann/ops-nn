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
 * \file depthwise_conv2d_graph_infer.cpp
 * \brief DepthwiseConv2D InferDataType (x is INT8/INT4/INT16 then y is INT32, otherwise same as x).
 */

#include "log/log.h"
#include "register/op_impl_registry.h"

#include "depthwise_conv2d_proto.h"

namespace ops {
namespace {
constexpr size_t X_IDX_DEPTHWISE = 0;
constexpr size_t Y_IDX_DEPTHWISE = 0;
} // namespace

ge::graphStatus InferDataTypeForDepthwiseConv2D(gert::InferDataTypeContext* context)
{
    if (context == nullptr) {
        OP_LOGE("DepthwiseConv2D", "context is null.");
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context->GetNodeName(), "Begin InferDataTypeForDepthwiseConv2D.");

    const ge::DataType xDtype = context->GetInputDataType(X_IDX_DEPTHWISE);
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
    const ge::graphStatus ret = context->SetOutputDataType(Y_IDX_DEPTHWISE, yDtype);

    OP_LOGD(context->GetNodeName(), "Set y dtype success.");
    return ret;
}

IMPL_OP(DepthwiseConv2D).InferDataType(InferDataTypeForDepthwiseConv2D);
} // namespace ops

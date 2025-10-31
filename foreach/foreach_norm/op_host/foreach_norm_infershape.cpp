/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file foreach_norm_infershape.cpp
 * \brief
 */

#include "log/log.h"
#include "register/op_impl_registry.h"
#include "runtime/infer_shape_context.h"
#include "runtime/storage_shape.h"
#include "../../foreach_utils/op_host/common_dtype.h"

using namespace ge;
namespace ops {
static ge::graphStatus InferShape4ForeachNorm(gert::InferShapeContext* context)
{
    uint32_t outputNum = context->GetComputeNodeOutputNum();
    uint32_t inputNum = context->GetComputeNodeInputNum();

    std::string errMsg = optiling::ConcatString(
        "num of dynamic input0 ", inputNum - 1, "not equal num of dynamic output0 ", outputNum);
    OP_CHECK_IF(
        (inputNum - 1) != outputNum, OP_LOGE(context->GetNodeName(), "%s", errMsg.c_str()),
        return ge::GRAPH_FAILED);

    for (uint32_t i = 0; i < inputNum - 1; i++) {
        auto xShape = context->GetInputShape(i);
        auto yShape = context->GetOutputShape(i);
        if ((xShape == nullptr) || (yShape == nullptr)) {
            return ge::GRAPH_FAILED;
        }
        yShape->SetDimNum(1);
        yShape->SetDim(0, 1);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4ForeachNorm(gert::InferDataTypeContext* context)
{
    uint32_t outputNum = context->GetComputeNodeOutputNum();
    uint32_t inputNum = context->GetComputeNodeInputNum();

    std::string errMsg = optiling::ConcatString(
        "num of dynamic input0 ", inputNum - 1, "not equal num of dynamic output0 ", outputNum);
    OP_CHECK_IF(
        (inputNum - 1) != outputNum, OP_LOGE(context->GetNodeName(), "%s", errMsg.c_str()),
        return ge::GRAPH_FAILED);

    for (uint32_t i = 0; i < inputNum - 1; i++) {
        auto xDtype = context->GetInputDataType(i);
        context->SetOutputDataType(i, xDtype);
    }
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(ForeachNorm).InferShape(InferShape4ForeachNorm).InferDataType(InferDataType4ForeachNorm);
} // namespace ops
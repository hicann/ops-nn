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
 * \file foreach_pow_scalar_and_tensor_infershape.cpp
 * \brief
 */

#include "log/log.h"
#include "../../foreach_utils/op_host/common_dtype.h"
#include "error_util.h"
#include "register/op_impl_registry.h"
#include "runtime/infer_shape_context.h"
#include "runtime/storage_shape.h"

using namespace ge;
namespace ops {
static ge::graphStatus InferShape4ForeachPowScalarAndTensor(gert::InferShapeContext* context)
{
    uint32_t outputNum = context->GetComputeNodeOutputNum();
    const auto inputInfo = context->GetIrInputInstanceInfo(1);
    if (inputInfo == nullptr) {
        return ge::GRAPH_FAILED;
    }

    std::string errMsg = optiling::ConcatString(
        "num of dynamic input1 ", inputInfo->GetInstanceNum(), "not equal num of dynamic output0 ", outputNum);
    OP_CHECK_IF(
        inputInfo->GetInstanceNum() != outputNum, OP_LOGE(context->GetNodeName(), "%s", errMsg.c_str()),
        return ge::GRAPH_FAILED);

    for (uint32_t i = 0; i < inputInfo->GetInstanceNum(); i++) {
        auto xShape = context->GetDynamicInputShape(1, i);
        auto yShape = context->GetOutputShape(i);
        if ((xShape == nullptr) || (yShape == nullptr)) {
            return ge::GRAPH_FAILED;
        }
        *yShape = *xShape;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4ForeachPowScalarAndTensor(gert::InferDataTypeContext* context)
{
    uint32_t outputNumber = context->GetComputeNodeOutputNum();
    const auto inputInfomation = context->GetIrInputInstanceInfo(1);
    if (inputInfomation == nullptr) {
        return ge::GRAPH_FAILED;
    }

    std::string errMsg = optiling::ConcatString(
        "num of dynamic input1 ", inputInfomation->GetInstanceNum(), "not equal num of dynamic output0 ", outputNumber);
    OP_CHECK_IF(
        inputInfomation->GetInstanceNum() != outputNumber, OP_LOGE(context->GetNodeName(), "%s", errMsg.c_str()),
        return ge::GRAPH_FAILED);

    for (uint32_t i = 0; i < inputInfomation->GetInstanceNum(); i++) {
        auto xDtype = context->GetDynamicInputDataType(1, i);
        context->SetOutputDataType(i, xDtype);
    }
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(ForeachPowScalarAndTensor)
    .InferShape(InferShape4ForeachPowScalarAndTensor)
    .InferDataType(InferDataType4ForeachPowScalarAndTensor);
} // namespace ops
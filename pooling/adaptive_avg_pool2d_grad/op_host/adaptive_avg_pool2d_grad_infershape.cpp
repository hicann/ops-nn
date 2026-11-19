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
 * \file adaptive_avg_pool2d_grad_infershape.cpp
 * \brief
 */

#include "register/op_impl_registry.h"
#include "log/log.h"
#include "util/shape_util.h"
#include "platform/platform_info.h"
#include "graph/utils/type_utils.h"

using namespace ge;
namespace {
constexpr size_t Y_GRAD_INDEX = 0;
constexpr size_t OUTPUT_INDEX = 0;
} // namespace
namespace ops {
static ge::graphStatus InferShape4AdaptiveAvgPool2dGrad(gert::InferShapeContext* context)
{
    // input 1: y_grad shape
    const gert::Shape* gradShape = context->GetInputShape(Y_GRAD_INDEX);
    OP_CHECK_IF(
        gradShape == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(Ops::Base::GetOpInfo(context).c_str(), "input_grad",
                                                 "required input, descriptor, attribute or attribute data is missing"),
        return ge::GRAPH_FAILED);

    // output shape
    gert::Shape* xGradShape = context->GetOutputShape(OUTPUT_INDEX);
    OP_CHECK_IF(xGradShape == nullptr,
                REPORT_PREDEFINED_ERR_MSG("EZ1014", (std::vector<const char*>{"opName", "reason"}),
                                          (std::vector<const char*>{Ops::Base::GetOpInfo(context).c_str(),
                                                                    "xGradShape is null; InferShape cannot continue"})),
                return ge::GRAPH_FAILED);

    if (Ops::Base::IsUnknownRank(*gradShape)) {
        Ops::Base::SetUnknownRank(*xGradShape);
        return ge::GRAPH_SUCCESS;
    }

    auto attr_ptr = context->GetAttrs();
    OP_CHECK_IF(
        attr_ptr == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(Ops::Base::GetOpInfo(context).c_str(), "attributes",
                                                 "required input, descriptor, attribute or attribute data is missing"),
        return ge::GRAPH_FAILED);
    auto output_size_ptr = attr_ptr->GetAttrPointer<gert::ContinuousVector>(0);
    size_t output_size_len = output_size_ptr->GetSize();
    auto output_size = static_cast<const int64_t*>(output_size_ptr->GetData());
    size_t inputDimNum = gradShape->GetDimNum();

    if (inputDimNum != output_size_len) {
        std::string dimMsg = std::to_string(inputDimNum) + " and " + std::to_string(output_size_len);
        std::string reasonMsg = "input and output's dim should be equal";
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(context->GetNodeName(), "input and output dim", dimMsg.c_str(),
                                                  reasonMsg.c_str());
        return GRAPH_FAILED;
    }

    for (int64_t i = 0; i < output_size_len; i++) {
        if (output_size[i] < 0) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "orig_input_shape",
                                                  std::to_string(output_size[i]),
                                                  "The value is invalid for orig_input_shape");
            return GRAPH_FAILED;
        }
    }

    // 维度数赋值
    xGradShape->SetDimNum(output_size_len);
    // 输出shape赋值+0维度校验
    for (size_t i = 0; i < output_size_len; ++i) {
        xGradShape->SetDim(i, output_size[i]);
    }
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(AdaptiveAvgPool2dGrad).InferShape(InferShape4AdaptiveAvgPool2dGrad);
} // namespace ops

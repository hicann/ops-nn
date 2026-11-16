/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/shape_util.h"
#include <string>
#include <vector>
#include "base/err_msg.h"

namespace ops {
static constexpr size_t MAX_DIM_NUM = 8;
static ge::graphStatus InferShape4HardSigmoid(gert::InferShapeContext* context)
{
    const gert::Shape* inputShape = context->GetInputShape(0);
    OP_CHECK_IF(
        inputShape == nullptr,
        OP_LOGE_WITH_INVALID_INPUT(
            ((context) == nullptr || (context)->GetNodeName() == nullptr ? "hard_sigmoid" : (context)->GetNodeName()),
            "input_x"),
        return ge::GRAPH_FAILED);

    gert::Shape* outputShape = context->GetOutputShape(0);
    if (outputShape == nullptr) {
        const std::string msgOpName = std::string(
            ((context) == nullptr || (context)->GetNodeName() == nullptr ? "hard_sigmoid" : (context)->GetNodeName()));
        OP_LOGE_WITHOUT_REPORT(msgOpName.c_str(), "%s",
                               "InferShape framework object outputShape is unavailable; check the operator "
                               "registration and context construction.");
        REPORT_PREDEFINED_ERR_MSG(
            "EZ1014", (std::vector<const char*>{"opName", "reason"}),
            (std::vector<const char*>{msgOpName.c_str(), "InferShape framework object outputShape is unavailable; "
                                                         "check the operator registration and context construction."}));
        return ge::GRAPH_FAILED;
    };

    if (Ops::Base::IsUnknownRank(*inputShape)) {
        Ops::Base::SetUnknownRank(*outputShape);
        return ge::GRAPH_SUCCESS;
    }
    OP_CHECK_IF(
        inputShape->GetDimNum() > MAX_DIM_NUM,
        OP_LOGE_WITHOUT_REPORT(context->GetNodeName(), "The dim num of input_x must be less than or equal to 8"),
        return ge::GRAPH_FAILED);
    *outputShape = *inputShape;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(HardSigmoid).InferShape(InferShape4HardSigmoid);
} // namespace ops

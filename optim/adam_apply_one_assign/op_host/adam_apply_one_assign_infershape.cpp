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
 * \file adam_apply_one_assign_infershape.cpp
 * \brief
 */
#include <vector>
#include "infershape_broadcast_util.h"
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "util/shape_util.h"

using namespace ge;
namespace ops {
static constexpr const char* INPUT_NAMES[] = {"input0", "input1", "input2", "input3", "input4",
                                              "mul0_x", "mul1_x", "mul2_x", "mul3_x", "add2_y"};

static ge::graphStatus ReportInferShapeFailure(const char* opName, const std::string& reason)
{
    const std::vector<const char*> keys = {"opName", "reason"};
    const std::vector<const char*> values = {opName, reason.c_str()};
    REPORT_PREDEFINED_ERR_MSG("EZ1014", keys, values);
    OP_LOGE_WITHOUT_REPORT(opName, "%s", reason.c_str());
    return ge::GRAPH_FAILED;
}

static ge::graphStatus InferShapeForAdam(gert::InferShapeContext* context)
{
    if (context == nullptr) {
        return ReportInferShapeFailure("AdamApplyOneAssign", "InferShapeContext is nullptr");
    }
    OP_LOGI(context->GetNodeName(), "ADAM_RTS2_ASSIGN_SHAPE_R1 begin");
    std::vector<const gert::Shape*> inputs(10);
    for (size_t i = 0; i < inputs.size(); ++i) {
        inputs[i] = context->GetInputShape(i);
        OP_CHECK_IF(inputs[i] == nullptr, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), INPUT_NAMES[i]),
                    return ge::GRAPH_FAILED);
    }
    // Validate the common broadcast and inplace states before touching outputs:
    // an output Shape can refer to metadata that the caller also uses as input.
    gert::Shape broadcastShape;
    OP_CHECK_IF(
        !Ops::Base::BroadcastShape(inputs, &broadcastShape),
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "inputs", Ops::Base::ToString(inputs).c_str(),
                                               "All inputs must be broadcast compatible"),
        return ge::GRAPH_FAILED);
    for (size_t state = 1; state <= 3; ++state) {
        // 未知rank（-2）或含未知维（-1）时无法确定不一致，跳过校验，避免动态场景误拦
        if (Ops::Base::IsUnknownRank(broadcastShape) || Ops::Base::IsUnknownRank(*inputs[state]) ||
            Ops::Base::IsUnknownShape(broadcastShape) || Ops::Base::IsUnknownShape(*inputs[state])) {
            continue;
        }
        OP_CHECK_IF(*inputs[state] != broadcastShape,
                    OP_LOGE_FOR_INVALID_SHAPE(context->GetNodeName(), INPUT_NAMES[state],
                                              Ops::Base::ToString(*inputs[state]).c_str(),
                                              Ops::Base::ToString(broadcastShape).c_str()),
                    return ge::GRAPH_FAILED);
    }
    gert::Shape* outputs[3];
    for (size_t i = 0; i < 3; ++i) {
        outputs[i] = context->GetOutputShape(i);
        if (outputs[i] == nullptr) {
            return ReportInferShapeFailure(context->GetNodeName(),
                                           "Output shape storage is nullptr for output " + std::to_string(i));
        }
    }
    for (size_t i = 0; i < 3; ++i) {
        *outputs[i] = broadcastShape;
        OP_LOGI(context->GetNodeName(), "ADAM_RTS2_ASSIGN_SHAPE_R1 output[%zu] shape: %s.", i,
                Ops::Base::ToString(*outputs[i]).c_str());
    }
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(AdamApplyOneAssign).InferShape(InferShapeForAdam);
} // namespace ops

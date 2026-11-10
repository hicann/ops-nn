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
 * \file scatter_min_infershape.cpp
 * \brief ScatterMin infershape: output var has the same shape/dtype as input var (in-place).
 */
#include <vector>
#include "register/op_impl_registry.h"
#include "log/log.h"

namespace ops {
static ge::graphStatus ReportInferShapeFailure(const char* opName, const char* reason)
{
    const std::vector<const char*> keys = {"opName", "reason"};
    const std::vector<const char*> values = {opName, reason};
    REPORT_PREDEFINED_ERR_MSG("EZ1014", keys, values);
    OP_LOGE_WITHOUT_REPORT(opName, "%s", reason);
    return ge::GRAPH_FAILED;
}

static constexpr size_t SC_IN_VAR_IDX = 0;
static constexpr size_t SC_OUT_VAR_IDX = 0;

static ge::graphStatus InferShape4ScatterMin(gert::InferShapeContext* context)
{
    OP_LOGI(context->GetNodeName(), "***runtime2.0 inferShape***");
    const gert::Shape* varShape = context->GetInputShape(SC_IN_VAR_IDX);
    OP_CHECK_IF(varShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "var"),
                return ge::GRAPH_FAILED);
    gert::Shape* outputShape = context->GetOutputShape(SC_OUT_VAR_IDX);
    if (outputShape == nullptr) {
        return ReportInferShapeFailure(context->GetNodeName(), "output var shape is null");
    }
    *outputShape = *varShape;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(ScatterMin).InferShape(InferShape4ScatterMin);
} // namespace ops

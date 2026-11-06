/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file gather_elements_infershape.cpp
 * \brief
 */
#include "log/log.h"
#include "register/op_impl_registry.h"
#include <vector>

using namespace ge;
namespace ops {
namespace {
void ReportGatherElementsInferShapeError(const char* opName, const char* detail)
{
    opName = opName != nullptr ? opName : "GatherElements";
    OP_LOGE_WITHOUT_REPORT(opName, "%s", detail);
    const std::vector<const char*> keys = {"opName", "reason"};
    const std::vector<const char*> values = {
        opName, "Operator shape inference execution failed. Preserve the execution logs and contact operator support."};
    REPORT_PREDEFINED_ERR_MSG("EZ1014", keys, values);
}
} // namespace
ge::graphStatus CopyShapeInput2OutputWithIdxForGatherElements(gert::InferShapeContext* context, int64_t input_idx,
                                                              int64_t output_idx)
{
    OP_CHECK_IF(context == nullptr, ReportGatherElementsInferShapeError("GatherElements", "InferShape context is null"),
                return ge::GRAPH_FAILED);
    const char* nodeName = context->GetNodeName();
    const char* opName = nodeName != nullptr ? nodeName : "GatherElements";
    auto in_shape = context->GetInputShape(input_idx);
    OP_CHECK_IF(in_shape == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName, "index"), return ge::GRAPH_FAILED);
    auto out_shape = context->GetOutputShape(output_idx);
    OP_CHECK_IF(out_shape == nullptr, ReportGatherElementsInferShapeError(opName, "Output shape storage is null"),
                return ge::GRAPH_FAILED);
    *out_shape = *in_shape;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShape4InIdx1AndOutIdx0(gert::InferShapeContext* context)
{
    constexpr size_t input_index = 1;
    constexpr size_t output_index = 0;
    return CopyShapeInput2OutputWithIdxForGatherElements(context, input_index, output_index);
}

IMPL_OP_INFERSHAPE(GatherElements).InferShape(InferShape4InIdx1AndOutIdx0);
} // namespace ops

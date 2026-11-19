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
 * \file ifmr_infershape.cpp
 * \brief
 */
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "error_util.h"

using namespace ge;

namespace ops {
static constexpr size_t IDX_SCALE = 0;
static constexpr size_t IDX_OFFSET = 1;

// InferShape框架输出shape容器缺失属于框架契约失败，统一上报EZ1014，不再生成EZ9999内部码
static void ReportIfmrInferShapeStageError(const char* opName, const std::string& reason)
{
    const std::vector<const char*> msgKey = {"opName", "reason"};
    const std::vector<const char*> msgValue = {opName != nullptr ? opName : "IFMR", reason.c_str()};
    REPORT_PREDEFINED_ERR_MSG("EZ1014", msgKey, msgValue);
}

static ge::graphStatus InferShapeIfmr(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShapeIfmr");
    auto outShape = context->GetOutputShape(IDX_SCALE);
    OP_CHECK_IF(outShape == nullptr,
                ReportIfmrInferShapeStageError(
                    context->GetNodeName(),
                    "the output shape container of scale (output index 0) is missing in the infer-shape context"),
                return ge::GRAPH_FAILED);
    outShape->SetDimNum(0);
    outShape->AppendDim(1);
    outShape = context->GetOutputShape(IDX_OFFSET);
    OP_CHECK_IF(outShape == nullptr,
                ReportIfmrInferShapeStageError(
                    context->GetNodeName(),
                    "the output shape container of offset (output index 1) is missing in the infer-shape context"),
                return ge::GRAPH_FAILED);
    outShape->SetDimNum(0);
    outShape->AppendDim(1);
    OP_LOGD(context->GetNodeName(), "End to do InferShapeIfmr");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeForIfmr(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferDataTypeForIfmr");
    context->SetOutputDataType(IDX_SCALE, ge::DT_FLOAT);
    context->SetOutputDataType(IDX_OFFSET, ge::DT_FLOAT);
    OP_LOGD(context->GetNodeName(), "End to do InferDataTypeForIfmr");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(IFMR).InferShape(InferShapeIfmr).InferDataType(InferDataTypeForIfmr);
} // namespace ops

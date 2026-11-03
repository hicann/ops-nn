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
 * \file normalize_bbox_graph_infer.cpp
 * \brief NormalizeBBox InferDataType: y.dtype == boxes.dtype.
 *        InferDataType is graph-only, hence op_graph; InferShape is shared by graph and
 *        single-op paths and stays in op_host.
 */

#include "log/log.h"
#include "register/op_impl_registry.h"

namespace ops {
namespace {
constexpr size_t INPUT_BOXES_IDX = 0;
constexpr size_t INPUT_SHAPE_HW_IDX = 1;
constexpr size_t OUTPUT_Y_IDX = 0;
} // namespace

ge::graphStatus NormalizeBBoxInferDataType(gert::InferDataTypeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context, "NormalizeBBoxInferDataType enter");
    const auto boxesDtype = context->GetInputDataType(INPUT_BOXES_IDX);
    OP_CHECK_IF(boxesDtype != ge::DT_FLOAT16 && boxesDtype != ge::DT_FLOAT,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context->GetNodeName(), "boxes", Ops::Base::ToString(boxesDtype),
                                                      "only float16 and float32 are supported"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context->GetInputDataType(INPUT_SHAPE_HW_IDX) != ge::DT_INT32,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                    context->GetNodeName(), "shape_hw",
                    Ops::Base::ToString(context->GetInputDataType(INPUT_SHAPE_HW_IDX)), "only int32 is supported"),
                return ge::GRAPH_FAILED);
    OP_LOGD(context, "NormalizeBBoxInferDataType end");
    return context->SetOutputDataType(OUTPUT_Y_IDX, boxesDtype);
}

IMPL_OP(NormalizeBBox).InferDataType(NormalizeBBoxInferDataType);
} // namespace ops

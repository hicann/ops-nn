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
 * \file chamfer_distance_infershape.cpp
 * \brief
 */
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "util/shape_util.h"

using namespace ge;

namespace {
constexpr uint64_t INPUT_XYZ1_IDX = 0;
constexpr uint64_t OUTPUT_DIST1_IDX = 0;
constexpr uint64_t OUTPUT_DIST2_IDX = 1;
constexpr uint64_t OUTPUT_IDX1_IDX = 2;
constexpr uint64_t OUTPUT_IDX2_IDX = 3;
constexpr size_t XYZ_DIM_NUM = 3;
constexpr size_t DIM_B = 1;
constexpr size_t DIM_N = 2;
} // namespace

namespace ops {
static graphStatus InferShape4ChamferDistance(gert::InferShapeContext* context)
{
    OP_LOGD(context, "Begin to do InferShape4ChamferDistance.");
    const gert::Shape* xyzShape = context->GetInputShape(INPUT_XYZ1_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, xyzShape);

    gert::Shape* dist1Shape = context->GetOutputShape(OUTPUT_DIST1_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, dist1Shape);
    gert::Shape* dist2Shape = context->GetOutputShape(OUTPUT_DIST2_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, dist2Shape);
    gert::Shape* idx1Shape = context->GetOutputShape(OUTPUT_IDX1_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, idx1Shape);
    gert::Shape* idx2Shape = context->GetOutputShape(OUTPUT_IDX2_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, idx2Shape);

    // 输入布局 (2, B, N): 首维是 x/y 两个坐标平面, 输出取其后两维 → (B, N)
    if (Ops::Base::IsUnknownRank(*xyzShape)) {
        Ops::Base::SetUnknownRank(*dist1Shape);
        Ops::Base::SetUnknownRank(*dist2Shape);
        Ops::Base::SetUnknownRank(*idx1Shape);
        Ops::Base::SetUnknownRank(*idx2Shape);
        return ge::GRAPH_SUCCESS;
    }
    OP_CHECK_IF(
        xyzShape->GetDimNum() != XYZ_DIM_NUM,
        OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "xyz1", std::to_string(xyzShape->GetDimNum()), "3"),
        return ge::GRAPH_FAILED);

    dist1Shape->SetDimNum(0);
    dist1Shape->AppendDim(xyzShape->GetDim(DIM_B));
    dist1Shape->AppendDim(xyzShape->GetDim(DIM_N));
    *dist2Shape = *dist1Shape;
    *idx1Shape = *dist1Shape;
    *idx2Shape = *dist1Shape;

    OP_LOGD(context, "InferShape4ChamferDistance End.");
    return ge::GRAPH_SUCCESS;
}

static graphStatus InferDataType4ChamferDistance(gert::InferDataTypeContext* context)
{
    OP_LOGD(context, "InferDataType4ChamferDistance Begin.");
    auto xyzDtype = context->GetInputDataType(INPUT_XYZ1_IDX);
    context->SetOutputDataType(OUTPUT_DIST1_IDX, xyzDtype);
    context->SetOutputDataType(OUTPUT_DIST2_IDX, xyzDtype);
    context->SetOutputDataType(OUTPUT_IDX1_IDX, ge::DT_INT32);
    context->SetOutputDataType(OUTPUT_IDX2_IDX, ge::DT_INT32);
    OP_LOGD(context, "InferDataType4ChamferDistance End.");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(ChamferDistance).InferShape(InferShape4ChamferDistance).InferDataType(InferDataType4ChamferDistance);
} // namespace ops

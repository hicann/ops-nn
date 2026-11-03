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
 * \file normalize_bbox_infershape.cpp
 * \brief y shares shape and dtype of boxes
 */
#include "log/log.h"
#include "runtime/infer_shape_context.h"
#include "register/op_impl_registry.h"
#include "error_util.h"
#include "util/shape_util.h"

using namespace ge;
namespace ops {
static constexpr size_t INPUT_BOXES_IDX = 0;
static constexpr size_t INPUT_SHAPE_HW_IDX = 1;
static constexpr size_t OUTPUT_Y_IDX = 0;
static constexpr size_t MAX_BOXES_RANK = 8;

static ge::graphStatus CheckNormalizeBBoxShapes(gert::InferShapeContext* context, const gert::Shape& boxesShape,
                                                const gert::Shape& shapeHwShape)
{
    const bool unknownBoxesRank = Ops::Base::IsUnknownRank(boxesShape);
    const bool unknownHwRank = Ops::Base::IsUnknownRank(shapeHwShape);
    if (!unknownHwRank) {
        OP_CHECK_IF(shapeHwShape.GetDimNum() != 2,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                        context->GetNodeName(), "shape_hw", std::to_string(shapeHwShape.GetDimNum()), "rank must be 2"),
                    return ge::GRAPH_FAILED);
        const auto width = shapeHwShape.GetDim(1);
        OP_CHECK_IF(width >= 0 && width != 3,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "shape_hw",
                                                          Ops::Base::ToString(shapeHwShape), "dimension 1 must be 3"),
                    return ge::GRAPH_FAILED);
    }
    if (!unknownBoxesRank) {
        const auto rank = boxesShape.GetDimNum();
        OP_CHECK_IF(rank < 2 || rank > MAX_BOXES_RANK,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "boxes", std::to_string(rank),
                                                             "rank must be in [2, 8]"),
                    return ge::GRAPH_FAILED);
        const auto* attrs = context->GetAttrs();
        const auto* reversed = attrs == nullptr ? nullptr : attrs->GetAttrPointer<bool>(0);
        const size_t coordinateAxis = reversed != nullptr && *reversed ? 1 : rank - 1;
        const auto coordinateSize = boxesShape.GetDim(coordinateAxis);
        OP_CHECK_IF(
            coordinateSize >= 0 && coordinateSize != 4,
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "boxes", Ops::Base::ToString(boxesShape),
                                                  "coordinate dimension must be 4"),
            return ge::GRAPH_FAILED);
    }
    if (!unknownBoxesRank && !unknownHwRank) {
        const auto boxesBatch = boxesShape.GetDim(0);
        const auto hwBatch = shapeHwShape.GetDim(0);
        OP_CHECK_IF(boxesBatch >= 0 && hwBatch >= 0 && boxesBatch != hwBatch,
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                        context->GetNodeName(), "boxes, shape_hw",
                        Ops::Base::ToString(boxesShape) + ", " + Ops::Base::ToString(shapeHwShape),
                        "batch dimensions must match"),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus NormalizeBBoxInferShape(gert::InferShapeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context, "Begin to do NormalizeBBoxInferShape");

    const gert::Shape* boxesShape = context->GetInputShape(INPUT_BOXES_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, boxesShape);
    const gert::Shape* shapeHwShape = context->GetInputShape(INPUT_SHAPE_HW_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, shapeHwShape);
    const auto shapeStatus = CheckNormalizeBBoxShapes(context, *boxesShape, *shapeHwShape);
    if (shapeStatus != ge::GRAPH_SUCCESS) {
        return shapeStatus;
    }
    gert::Shape* yShape = context->GetOutputShape(OUTPUT_Y_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);
    if (Ops::Base::IsUnknownRank(*boxesShape)) {
        Ops::Base::SetUnknownRank(*yShape);
    } else {
        *yShape = *boxesShape;
    }

    OP_LOGD(context, "End to do NormalizeBBoxInferShape");
    return ge::GRAPH_SUCCESS;
}

// InferDataType is graph-only and now lives in op_graph/normalize_bbox_graph_infer.cpp;
// only InferShape (shared by graph and single-op paths) is registered here.
IMPL_OP_INFERSHAPE(NormalizeBBox).InferShape(NormalizeBBoxInferShape);
} // namespace ops

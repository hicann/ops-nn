/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file chamfer_distance_grad.cc
 * \brief InferShape and InferDataType implementation for ChamferDistanceGrad.
 */

#include <cstdint>
#include "exe_graph/runtime/infer_shape_context.h"
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "util/shape_util.h"

using namespace ge;

namespace {
constexpr int64_t INDEX_XYZ1 = 0;
constexpr int64_t INDEX_XYZ2 = 1;
constexpr int64_t INDEX_IDX1 = 2;
constexpr int64_t INDEX_IDX2 = 3;
constexpr int64_t INDEX_GRAD_DIST1 = 4;
constexpr int64_t INDEX_GRAD_DIST2 = 5;

constexpr int64_t INDEX_GRAD_XYZ1 = 0;
constexpr int64_t INDEX_GRAD_XYZ2 = 1;

constexpr int64_t XYZ_RANK = 3;
constexpr int64_t POINTWISE_RANK = 2;
constexpr int64_t BATCH_AXIS = 0;
constexpr int64_t POINT_AXIS = 1;
constexpr int64_t COORD_AXIS = 2;
constexpr int64_t COORD_DIM = 2;

inline bool IsDimCompatible(const int64_t lhs, const int64_t rhs) { return lhs < 0 || rhs < 0 || lhs == rhs; }

inline bool IsPointwiseShapeCompatible(const gert::Shape& xyzShape, const gert::Shape& pointwiseShape)
{
    return IsDimCompatible(xyzShape.GetDim(BATCH_AXIS), pointwiseShape.GetDim(BATCH_AXIS)) &&
           IsDimCompatible(xyzShape.GetDim(POINT_AXIS), pointwiseShape.GetDim(POINT_AXIS));
}
} // namespace

namespace ops {

static ge::graphStatus InferDataType4ChamferDistanceGrad(gert::InferDataTypeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }

    OP_LOGD(context->GetNodeName(), "InferDataType4ChamferDistanceGrad start.");

    context->SetOutputDataType(INDEX_GRAD_XYZ1, context->GetInputDataType(INDEX_XYZ1));
    context->SetOutputDataType(INDEX_GRAD_XYZ2, context->GetInputDataType(INDEX_XYZ2));

    OP_LOGD(context->GetNodeName(), "InferDataType4ChamferDistanceGrad end.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShape4ChamferDistanceGrad(gert::InferShapeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }

    OP_LOGD(context->GetNodeName(), "Enter InferShapeChamferDistanceGrad");

    const gert::Shape* xyz1Shape = context->GetInputShape(INDEX_XYZ1);
    const gert::Shape* xyz2Shape = context->GetInputShape(INDEX_XYZ2);
    const gert::Shape* idx1Shape = context->GetInputShape(INDEX_IDX1);
    const gert::Shape* idx2Shape = context->GetInputShape(INDEX_IDX2);
    const gert::Shape* gradDist1Shape = context->GetInputShape(INDEX_GRAD_DIST1);
    const gert::Shape* gradDist2Shape = context->GetInputShape(INDEX_GRAD_DIST2);

    OP_CHECK_NULL_WITH_CONTEXT(context, xyz1Shape);
    OP_CHECK_NULL_WITH_CONTEXT(context, xyz2Shape);
    OP_CHECK_NULL_WITH_CONTEXT(context, idx1Shape);
    OP_CHECK_NULL_WITH_CONTEXT(context, idx2Shape);
    OP_CHECK_NULL_WITH_CONTEXT(context, gradDist1Shape);
    OP_CHECK_NULL_WITH_CONTEXT(context, gradDist2Shape);

    gert::Shape* gradXyz1Shape = context->GetOutputShape(INDEX_GRAD_XYZ1);
    gert::Shape* gradXyz2Shape = context->GetOutputShape(INDEX_GRAD_XYZ2);

    OP_CHECK_NULL_WITH_CONTEXT(context, gradXyz1Shape);
    OP_CHECK_NULL_WITH_CONTEXT(context, gradXyz2Shape);

    /*
     * A2/A3 supported shape:
     *
     *   xyz1:       [B, N, 2]
     *   xyz2:       [B, N, 2]
     *   idx1:       [B, N]
     *   idx2:       [B, N]
     *   grad_dist1: [B, N]
     *   grad_dist2: [B, N]
     *
     * Output:
     *
     *   grad_xyz1:  [B, N, 2]
     *   grad_xyz2:  [B, N, 2]
     */

    if (Ops::Base::IsUnknownRank(*xyz1Shape) || Ops::Base::IsUnknownRank(*xyz2Shape) ||
        Ops::Base::IsUnknownRank(*idx1Shape) || Ops::Base::IsUnknownRank(*idx2Shape) ||
        Ops::Base::IsUnknownRank(*gradDist1Shape) || Ops::Base::IsUnknownRank(*gradDist2Shape)) {
        Ops::Base::SetUnknownRank(*gradXyz1Shape);
        Ops::Base::SetUnknownRank(*gradXyz2Shape);

        OP_LOGD(context->GetNodeName(), "ChamferDistanceGrad InferShape handles unknown rank.");
        return ge::GRAPH_SUCCESS;
    }

    const int64_t xyz1Rank = static_cast<int64_t>(xyz1Shape->GetDimNum());
    const int64_t xyz2Rank = static_cast<int64_t>(xyz2Shape->GetDimNum());
    const int64_t idx1Rank = static_cast<int64_t>(idx1Shape->GetDimNum());
    const int64_t idx2Rank = static_cast<int64_t>(idx2Shape->GetDimNum());
    const int64_t gradDist1Rank = static_cast<int64_t>(gradDist1Shape->GetDimNum());
    const int64_t gradDist2Rank = static_cast<int64_t>(gradDist2Shape->GetDimNum());

    if (xyz1Rank != XYZ_RANK || xyz2Rank != XYZ_RANK) {
        OP_LOGE(context->GetNodeName(), "xyz1 and xyz2 must be rank-3 tensors, but got rank %ld and %ld.", xyz1Rank,
                xyz2Rank);
        return ge::GRAPH_FAILED;
    }

    if (idx1Rank != POINTWISE_RANK || idx2Rank != POINTWISE_RANK || gradDist1Rank != POINTWISE_RANK ||
        gradDist2Rank != POINTWISE_RANK) {
        OP_LOGE(context->GetNodeName(), "idx1, idx2, grad_dist1 and grad_dist2 must all be rank-2 tensors.");
        return ge::GRAPH_FAILED;
    }

    /*
     * The coordinate dimension is fixed to 2.
     *
     * An unknown coordinate dimension is accepted during dynamic
     * compilation, but a known value other than 2 is rejected.
     */
    if (!IsDimCompatible(xyz1Shape->GetDim(COORD_AXIS), COORD_DIM) ||
        !IsDimCompatible(xyz2Shape->GetDim(COORD_AXIS), COORD_DIM)) {
        OP_LOGE(context->GetNodeName(), "The last dimensions of xyz1 and xyz2 must both be 2, but got %ld and %ld.",
                xyz1Shape->GetDim(COORD_AXIS), xyz2Shape->GetDim(COORD_AXIS));
        return ge::GRAPH_FAILED;
    }

    /*
     * A2/A3 Tiling only has one batch_size and one num, so xyz1 and
     * xyz2 must have the same batch dimension and point dimension.
     */
    if (!IsDimCompatible(xyz1Shape->GetDim(BATCH_AXIS), xyz2Shape->GetDim(BATCH_AXIS))) {
        OP_LOGE(context->GetNodeName(), "xyz1 and xyz2 must have the same batch dimension, but got %ld and %ld.",
                xyz1Shape->GetDim(BATCH_AXIS), xyz2Shape->GetDim(BATCH_AXIS));
        return ge::GRAPH_FAILED;
    }

    if (!IsDimCompatible(xyz1Shape->GetDim(POINT_AXIS), xyz2Shape->GetDim(POINT_AXIS))) {
        OP_LOGE(context->GetNodeName(), "xyz1 and xyz2 must have the same point dimension, but got %ld and %ld.",
                xyz1Shape->GetDim(POINT_AXIS), xyz2Shape->GetDim(POINT_AXIS));
        return ge::GRAPH_FAILED;
    }

    if (!IsPointwiseShapeCompatible(*xyz1Shape, *idx1Shape)) {
        OP_LOGE(context->GetNodeName(), "idx1 shape must match the first two dimensions of xyz1.");
        return ge::GRAPH_FAILED;
    }

    if (!IsPointwiseShapeCompatible(*xyz2Shape, *idx2Shape)) {
        OP_LOGE(context->GetNodeName(), "idx2 shape must match the first two dimensions of xyz2.");
        return ge::GRAPH_FAILED;
    }

    if (!IsPointwiseShapeCompatible(*xyz1Shape, *gradDist1Shape)) {
        OP_LOGE(context->GetNodeName(), "grad_dist1 shape must match the first two dimensions of xyz1.");
        return ge::GRAPH_FAILED;
    }

    if (!IsPointwiseShapeCompatible(*xyz2Shape, *gradDist2Shape)) {
        OP_LOGE(context->GetNodeName(), "grad_dist2 shape must match the first two dimensions of xyz2.");
        return ge::GRAPH_FAILED;
    }

    /*
     * Preserve every known dynamic dimension from xyz1 and xyz2.
     *
     * For example:
     *
     *   xyz1 = [4, -1, 2] -> grad_xyz1 = [4, -1, 2]
     *
     * instead of changing the result to [-1, -1, -1].
     */
    *gradXyz1Shape = *xyz1Shape;
    *gradXyz2Shape = *xyz2Shape;

    OP_LOGD(context->GetNodeName(),
            "Runtime2.0 ChamferDistanceGrad InferShape success. "
            "grad_xyz1 shape is %s, grad_xyz2 shape is %s.",
            Ops::Base::ToString(*gradXyz1Shape).c_str(), Ops::Base::ToString(*gradXyz2Shape).c_str());

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(ChamferDistanceGrad)
    .InferShape(InferShape4ChamferDistanceGrad)
    .InferDataType(InferDataType4ChamferDistanceGrad);

} // namespace ops

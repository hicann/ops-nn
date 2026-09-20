/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// =============================================================================
// gn_training_update_package/op_host/gn_training_update_infershape.cpp
// =============================================================================
//
// ROLE: Shape inference for the GnTrainingUpdate operator (graph mode).
//   When the Graph Engine compiles a graph, it calls this infer-shape
//   function to derive the output shapes from the input shapes.
//
//   Semantics (design/InferShapeDtype.md, spec/spec.yaml outputs.shape_rule):
//     y.shape              = x.shape   (input 0) — normalization (+optional
//                                        affine) never changes the feature map
//     batch_mean.shape     = sum.shape (input 1)
//     batch_variance.shape = sum.shape (input 1)
//   No broadcast max-reduction is needed: outputs are plain copies of the
//   two reference inputs. Broadcasting only happens inside the kernel.
//
//   Static shapes are copied verbatim. Dynamic dims (-1) and unknown rank
//   (the V2 gert UNKNOWN_RANK marker, dim_num == 1 && dim0 == UNKNOWN_DIM_NUM,
//   i.e. {-2}) both propagate correctly through the same plain copy, so no
//   special-casing is required (same convention as ops-nn
//   bn3_d_training_reduce_grad_infershape.cpp).
//
// CONTENTS:
//   - InferShape4GnTrainingUpdate() — the shape inference function
//   - IMPL_OP_INFERSHAPE(...).InferShape(...) — registration
//
// REGISTRATION NAME: the graph-IR prototype is REG_OP(GNTrainingUpdate)
//   (frozen, verbatim from the official IR), while the OpDef / ACLNN op type
//   is GnTrainingUpdate (read-only). The impl registry is keyed by exact op
//   type string, so the same function is registered under BOTH names:
//   GnTrainingUpdate serves the ACLNN/two-stage path, GNTrainingUpdate serves
//   graph nodes whose type is the canonical REG_OP name.
//
// =============================================================================

#include "register/op_impl_registry.h"             // IMPL_OP_INFERSHAPE macro
#include "exe_graph/runtime/infer_shape_context.h" // InferShapeContext, gert::Shape
#include "op_common/log/log.h"
#include <string> // OP_CHECK_NULL_WITH_CONTEXT macro

using namespace ge;

namespace ops {

// ---------------------------------------------------------------------------
// Input-consistency validation (design/InferShapeDtype.md "broadcast 对齐规则"
// and "边界处理"). Static-shape violations are rejected with GRAPH_FAILED so
// illegal graphs fail at compile time instead of executing. Dynamic dims
// (any dim < 0, including the unknown-rank {-2} marker) are wildcards: a
// comparison involving an unknown dim is skipped, so legal dynamic graphs
// still infer correctly.
// ---------------------------------------------------------------------------
namespace {

constexpr int64_t kUnknownDimNum = -2; // gert unknown-rank marker

bool IsUnknownRankShape(const gert::Shape* shape)
{
    return shape->GetDimNum() == 1 && shape->GetDim(0) == kUnknownDimNum;
}

// "[d0,d1,...]" for diagnostics.
std::string ShapeStr(const gert::Shape* shape)
{
    std::string s = "[";
    for (size_t i = 0; i < shape->GetDimNum(); ++i) {
        s += std::to_string(shape->GetDim(i));
        if (i + 1 < shape->GetDimNum()) {
            s += ",";
        }
    }
    s += "]";
    return s;
}

// Per-dim equality with unknown-dim wildcard. An unknown-rank side matches
// anything: its dims cannot be validated at compile time.
bool DimsMatch(const gert::Shape* a, const gert::Shape* b)
{
    if (IsUnknownRankShape(a) || IsUnknownRankShape(b)) {
        return true;
    }
    if (a->GetDimNum() != b->GetDimNum()) {
        return false;
    }
    for (size_t i = 0; i < a->GetDimNum(); ++i) {
        const int64_t da = a->GetDim(i);
        const int64_t db = b->GetDim(i);
        if (da >= 0 && db >= 0 && da != db) {
            return false;
        }
    }
    return true;
}

// dim must be 1 unless unknown.
bool DimIsOne(const gert::Shape* shape, size_t axis)
{
    const int64_t d = shape->GetDim(axis);
    return d < 0 || d == 1;
}

// dim is "concrete and not 1".
bool DimIsConcreteNonOne(const gert::Shape* shape, size_t axis)
{
    const int64_t d = shape->GetDim(axis);
    return d >= 0 && d != 1;
}

// Statistics/affine 5D tensors carry the group axis at axis 1 (NCHW) or
// axis 3 (NHWC); every other non-batch axis must be 1. Both group axes
// concrete and non-1 means a mixed/ambiguous layout. Unknown-rank shapes
// cannot be validated and pass.
bool StatShapeCanonical(const gert::Shape* shape)
{
    if (IsUnknownRankShape(shape)) {
        return true;
    }
    if (!DimIsOne(shape, 2) || !DimIsOne(shape, 4)) {
        return false;
    }
    if (DimIsConcreteNonOne(shape, 1) && DimIsConcreteNonOne(shape, 3)) {
        return false;
    }
    return true;
}

// Affine tensors additionally require a leading broadcast dim of 1
// ([1,G,1,1,1] / [1,1,1,G,1]).
bool AffineShapeCanonical(const gert::Shape* shape) { return DimIsOne(shape, 0) && StatShapeCanonical(shape); }

} // namespace

// ---------------------------------------------------------------------------
// InferShape4GnTrainingUpdate(context) — shape inference function
//
// Output shape derivation (design/InferShapeDtype.md "InferShape" table):
//   output 0 (y)              <- input 0 (x),   4D [N,C,H,W] / [N,H,W,C]
//   output 1 (batch_mean)     <- input 1 (sum), 5D [N,G,1,1,1] / [N,1,1,G,1]
//   output 2 (batch_variance) <- input 1 (sum), same as batch_mean
//
// Covers static shapes, dynamic shapes (unknown dims -1) and unknown rank
// ({-2} marker) — all propagate through a plain shape copy.
//
// Parameters:
//   context — InferShapeContext providing access to input shapes and
//             allowing setting of output shapes
//
// Returns:
//   ge::graphStatus — ge::GRAPH_SUCCESS on success
// ---------------------------------------------------------------------------
static ge::graphStatus InferShape4GnTrainingUpdate(gert::InferShapeContext* context)
{
    // y (output 0) follows x (input 0): y.shape = x.shape
    const gert::Shape* x_shape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, x_shape);
    gert::Shape* y_shape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, y_shape);

    // batch_mean / batch_variance (outputs 1..2) follow sum (input 1)
    const gert::Shape* sum_shape = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, sum_shape);
    const gert::Shape* square_sum_shape = context->GetInputShape(2);
    OP_CHECK_NULL_WITH_CONTEXT(context, square_sum_shape);
    // Optional inputs are read by IR-prototype index via GetOptionalInputShape
    // (GetInputShape uses the packed instantiated-input index and is invalid
    // for OPTIONAL_INPUT ports); absent -> nullptr: scale(3), offset(4),
    // mean(5), variance(6).
    const gert::Shape* scale_shape = context->GetOptionalInputShape(3);
    const gert::Shape* offset_shape = context->GetOptionalInputShape(4);
    const gert::Shape* mean_shape = context->GetOptionalInputShape(5);
    const gert::Shape* variance_shape = context->GetOptionalInputShape(6);

    // ---- input-consistency validation ----
    // Unknown rank only defers the comparisons that actually involve the
    // unknown value: x-side checks run when x rank is known, sum-side checks
    // run when sum rank is known, cross checks when both are known. A
    // concretely illegal sum shape is rejected even when x is {-2}.
    const bool x_unknown_rank = IsUnknownRankShape(x_shape);
    const bool sum_unknown_rank = IsUnknownRankShape(sum_shape);
    if (!x_unknown_rank && x_shape->GetDimNum() != 4) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON("GnTrainingUpdate", "x", std::to_string(x_shape->GetDimNum()).c_str(),
                                                 "x must be 4D");
        return ge::GRAPH_FAILED;
    }
    if (!sum_unknown_rank) {
        if (sum_shape->GetDimNum() != 5 ||
            (!IsUnknownRankShape(square_sum_shape) && square_sum_shape->GetDimNum() != 5)) {
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON("GnTrainingUpdate", "sum/square_sum", "-",
                                                     "sum/square_sum must be 5D");
            return ge::GRAPH_FAILED;
        }
        if (!DimsMatch(square_sum_shape, sum_shape)) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON("GnTrainingUpdate", "square_sum", ShapeStr(square_sum_shape).c_str(),
                                                  ("must equal sum shape " + ShapeStr(sum_shape)).c_str());
            return ge::GRAPH_FAILED;
        }
        if (!StatShapeCanonical(sum_shape)) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON("GnTrainingUpdate", "sum", ShapeStr(sum_shape).c_str(),
                                                  "sum shape must be [N,G,1,1,1] or [N,1,1,G,1]");
            return ge::GRAPH_FAILED;
        }
        if (mean_shape != nullptr && !DimsMatch(mean_shape, sum_shape)) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON("GnTrainingUpdate", "mean", ShapeStr(mean_shape).c_str(),
                                                  ("must equal sum shape " + ShapeStr(sum_shape)).c_str());
            return ge::GRAPH_FAILED;
        }
        if (variance_shape != nullptr && !DimsMatch(variance_shape, sum_shape)) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON("GnTrainingUpdate", "variance", ShapeStr(variance_shape).c_str(),
                                                  ("must equal sum shape " + ShapeStr(sum_shape)).c_str());
            return ge::GRAPH_FAILED;
        }
        // Unknown-rank optional shapes cannot be validated and pass.
        if (scale_shape != nullptr && !IsUnknownRankShape(scale_shape) &&
            (scale_shape->GetDimNum() != 5 || !AffineShapeCanonical(scale_shape))) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON("GnTrainingUpdate", "scale", ShapeStr(scale_shape).c_str(),
                                                  "scale shape must be [1,G,1,1,1] or [1,1,1,G,1]");
            return ge::GRAPH_FAILED;
        }
        if (offset_shape != nullptr && !IsUnknownRankShape(offset_shape) &&
            (offset_shape->GetDimNum() != 5 || !AffineShapeCanonical(offset_shape))) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON("GnTrainingUpdate", "offset", ShapeStr(offset_shape).c_str(),
                                                  "offset shape must be [1,G,1,1,1] or [1,1,1,G,1]");
            return ge::GRAPH_FAILED;
        }
    }
    if (!x_unknown_rank && !sum_unknown_rank) {
        const int64_t x_n = x_shape->GetDim(0);
        const int64_t sum_n = sum_shape->GetDim(0);
        if (x_n >= 0 && sum_n >= 0 && x_n != sum_n) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                "GnTrainingUpdate", "sum", ShapeStr(sum_shape).c_str(),
                ("sum batch dim must equal x batch dim of " + ShapeStr(x_shape)).c_str());
            return ge::GRAPH_FAILED;
        }
    }

    *y_shape = *x_shape;
    for (size_t i = 1; i <= 2; ++i) {
        gert::Shape* out_shape = context->GetOutputShape(i);
        OP_CHECK_NULL_WITH_CONTEXT(context, out_shape);
        *out_shape = *sum_shape;
    }

    return ge::GRAPH_SUCCESS;
}

// Register the infer-shape function. Dual registration: GnTrainingUpdate
// (OpDef/ACLNN op type) and GNTrainingUpdate (canonical REG_OP graph type) —
// see the file header for the rationale.
IMPL_OP_INFERSHAPE(GnTrainingUpdate).InferShape(InferShape4GnTrainingUpdate);
IMPL_OP_INFERSHAPE(GNTrainingUpdate).InferShape(InferShape4GnTrainingUpdate);

} // namespace ops

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
// gn_training_update_package/op_graph/gn_training_update_graph_infer.cpp
// =============================================================================
//
// ROLE: Graph-level data type inference for the GnTrainingUpdate operator.
//   When the Graph Engine (GE) constructs a computational graph, it needs to
//   know the output data types of each node before execution. This file
//   registers a type inference function that tells GE the output dtypes.
//
//   Semantics (design/InferShapeDtype.md "InferDataType" table,
//   spec/spec.yaml outputs.dtype_rule / dtype_policy):
//     y.dtype              = x.dtype  (float16 or float32, follows input 0;
//                                      dtype_policy.promotion: same_as_first_input)
//     batch_mean.dtype     = float32  (copied from sum, whose dtype_set is [float32])
//     batch_variance.dtype = float32  (same as batch_mean)
//   No cross-input promotion exists; the statistics/affine inputs are always
//   float32, so copying sum's dtype yields exactly the declared combinations.
//
//   The INFER_SHAPE equivalent is in op_host/gn_training_update_infershape.cpp.
//
// CONTENTS:
//   - InferDataTypeForGnTrainingUpdate() — the type inference function
//   - IMPL_OP(...).InferDataType(...) — registration
//
// REGISTRATION NAME: the graph-IR prototype is REG_OP(GNTrainingUpdate)
//   (frozen, verbatim from the official IR), while the OpDef / ACLNN op type
//   is GnTrainingUpdate (read-only). The impl registry is keyed by exact op
//   type string, so the same function is registered under BOTH names:
//   GnTrainingUpdate serves the ACLNN/two-stage path, GNTrainingUpdate serves
//   graph nodes whose type is the canonical REG_OP name.
//
//   InferFormat / InferShapeRange callbacks are not registered: all tensors
//   are fixed FORMAT_ND (OpDef UnknownShapeFormat ND), so the framework
//   default inference is sufficient and the operator interface does not
//   require them.
//
// =============================================================================

#include "register/op_impl_registry.h" // IMPL_OP macro for operator registration
#include "graph/operator_reg.h"        // IMPLEMT_COMMON_INFERFUNC, COMMON_INFER_FUNC_REG
#include "graph/operator.h"            // ge::Operator, ge::TensorDesc
#include "op_common/log/log.h"         // OP_LOGE
#include <vector>

using namespace ge;

namespace ops {

// ---------------------------------------------------------------------------
// Shared input-validation helpers (design/InferShapeDtype.md "broadcast 对齐规则",
// "dtype 组合" and "边界处理"). Violations on statically-known facts are
// rejected (GRAPH_FAILED) so illegal graphs fail at compile time instead of
// executing. Dynamic dims (any dim < 0, including the unknown-rank {-2}
// marker) are wildcards: a comparison involving an unknown dim is skipped.
// ---------------------------------------------------------------------------
namespace {

// dtype contract: x in {float16, float32}; sum/square_sum and every present
// optional input (scale/offset/mean/variance) must be float32. An absent
// optional input reports DT_UNDEFINED and is skipped.
ge::graphStatus CheckInputDtypes(const std::vector<ge::DataType>& dtypes, const char* op_name)
{
    const ge::DataType x_dtype = dtypes[0];
    if (x_dtype != ge::DT_FLOAT16 && x_dtype != ge::DT_FLOAT) {
        OP_LOGE(op_name, "x dtype must be float16 or float32.");
        return ge::GRAPH_FAILED;
    }
    for (size_t i = 1; i < dtypes.size(); ++i) {
        const ge::DataType dt = dtypes[i];
        if (dt == ge::DT_UNDEFINED) {
            continue; // absent optional input
        }
        if (dt != ge::DT_FLOAT) {
            OP_LOGE(op_name, "input %zu dtype must be float32.", i);
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

constexpr int64_t kUnknownDimNum = -2; // unknown-rank marker {-2}

bool IsUnknownRankShape(const ge::Shape& s) { return s.GetDimNum() == 1 && s.GetDim(0) == kUnknownDimNum; }

// dim must be 1 unless unknown.
bool DimIsOne(int64_t d) { return d < 0 || d == 1; }

// Statistics/affine 5D tensors carry the group axis at axis 1 (NCHW) or
// axis 3 (NHWC); every other non-batch axis must be 1. Both group axes
// concrete and non-1 means a mixed/ambiguous layout. Unknown-rank shapes
// cannot be validated and pass.
bool StatShapeCanonical(const ge::Shape& shape)
{
    if (IsUnknownRankShape(shape)) {
        return true;
    }
    if (!DimIsOne(shape.GetDim(2)) || !DimIsOne(shape.GetDim(4))) {
        return false;
    }
    const int64_t d1 = shape.GetDim(1);
    const int64_t d3 = shape.GetDim(3);
    if (d1 >= 0 && d1 != 1 && d3 >= 0 && d3 != 1) {
        return false;
    }
    return true;
}

// Per-dim equality with unknown-dim wildcard. An unknown-rank side matches
// anything: its dims cannot be validated at compile time.
bool DimsMatch(const ge::Shape& a, const ge::Shape& b)
{
    if (IsUnknownRankShape(a) || IsUnknownRankShape(b)) {
        return true;
    }
    if (a.GetDimNum() != b.GetDimNum()) {
        return false;
    }
    for (size_t i = 0; i < a.GetDimNum(); ++i) {
        const int64_t da = a.GetDim(i);
        const int64_t db = b.GetDim(i);
        if (da >= 0 && db >= 0 && da != db) {
            return false;
        }
    }
    return true;
}

// Shape contract for the V1 route. Absent optional inputs (dtype
// DT_UNDEFINED) are skipped. Unknown rank only defers checks involving the
// unknown tensor; concrete illegal shapes of the other inputs are rejected.
ge::graphStatus CheckInputShapesV1(const ge::Shape& x_shape, const ge::Shape& sum_shape,
                                   const ge::Shape& square_sum_shape, const ge::Shape* scale_shape,
                                   const ge::Shape* offset_shape, const ge::Shape* mean_shape,
                                   const ge::Shape* variance_shape, const char* op_name)
{
    // Unknown rank only defers the comparisons that actually involve the
    // unknown value: x-side checks run when x rank is known, sum-side checks
    // run when sum rank is known, cross checks when both are known.
    const bool x_unknown_rank = IsUnknownRankShape(x_shape);
    const bool sum_unknown_rank = IsUnknownRankShape(sum_shape);
    if (!x_unknown_rank && x_shape.GetDimNum() != 4) {
        OP_LOGE(op_name, "x must be 4D.");
        return ge::GRAPH_FAILED;
    }
    if (!sum_unknown_rank) {
        if (sum_shape.GetDimNum() != 5 ||
            (!IsUnknownRankShape(square_sum_shape) && square_sum_shape.GetDimNum() != 5)) {
            OP_LOGE(op_name, "sum/square_sum must be 5D.");
            return ge::GRAPH_FAILED;
        }
        if (!DimsMatch(square_sum_shape, sum_shape)) {
            OP_LOGE(op_name, "square_sum shape must equal sum shape.");
            return ge::GRAPH_FAILED;
        }
        if (!StatShapeCanonical(sum_shape)) {
            OP_LOGE(op_name, "sum shape must be [N,G,1,1,1] or [N,1,1,G,1].");
            return ge::GRAPH_FAILED;
        }
        for (const ge::Shape* reserved : {mean_shape, variance_shape}) {
            if (reserved != nullptr && !DimsMatch(*reserved, sum_shape)) {
                OP_LOGE(op_name, "mean/variance shape must equal sum shape.");
                return ge::GRAPH_FAILED;
            }
        }
        for (const ge::Shape* affine : {scale_shape, offset_shape}) {
            if (affine != nullptr && !IsUnknownRankShape(*affine) &&
                (affine->GetDimNum() != 5 || !DimIsOne(affine->GetDim(0)) || !StatShapeCanonical(*affine))) {
                OP_LOGE(op_name, "scale/offset shape must be [1,G,1,1,1] or [1,1,1,G,1].");
                return ge::GRAPH_FAILED;
            }
        }
    }
    if (!x_unknown_rank && !sum_unknown_rank) {
        const int64_t x_n = x_shape.GetDim(0);
        const int64_t sum_n = sum_shape.GetDim(0);
        if (x_n >= 0 && sum_n >= 0 && x_n != sum_n) {
            OP_LOGE(op_name, "sum batch dim must equal x batch dim.");
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

} // namespace

// ---------------------------------------------------------------------------
// InferDataTypeForGnTrainingUpdate(context) — data type inference callback
//
// Output dtype derivation:
//   output 0 (y)              <- input 0 (x)   dtype (fp16 or fp32)
//   output 1 (batch_mean)     <- input 1 (sum) dtype (always fp32)
//   output 2 (batch_variance) <- input 1 (sum) dtype (always fp32)
//
// Parameters:
//   context — InferDataTypeContext that provides access to input dtypes
//             and allows setting output dtypes
//
// Returns:
//   ge::graphStatus — ge::GRAPH_SUCCESS on success
// ---------------------------------------------------------------------------
static ge::graphStatus InferDataTypeForGnTrainingUpdate(gert::InferDataTypeContext* context)
{
    // dtype contract validation (design/InferShapeDtype.md "dtype 组合").
    // Required inputs by packed index; OPTIONAL_INPUT ports by IR-prototype
    // index via GetOptionalInputDataType (absent -> DT_UNDEFINED, skipped).
    std::vector<ge::DataType> input_dtypes;
    for (size_t i = 0; i <= 2; ++i) { // x, sum, square_sum
        input_dtypes.push_back(context->GetInputDataType(i));
    }
    for (size_t ir = 3; ir <= 6; ++ir) { // scale, offset, mean, variance
        input_dtypes.push_back(context->GetOptionalInputDataType(ir));
    }
    if (CheckInputDtypes(input_dtypes, "GnTrainingUpdate") != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // y.dtype = x.dtype (same_as_first_input)
    const ge::DataType x_dtype = context->GetInputDataType(0);
    context->SetOutputDataType(0, x_dtype);

    // batch_mean.dtype = batch_variance.dtype = sum.dtype (float32)
    const ge::DataType sum_dtype = context->GetInputDataType(1);
    context->SetOutputDataType(1, sum_dtype);
    context->SetOutputDataType(2, sum_dtype);
    return ge::GRAPH_SUCCESS;
}

// Register the type inference function. Dual registration: GnTrainingUpdate
// (OpDef/ACLNN op type) and GNTrainingUpdate (canonical REG_OP graph type) —
// see the file header for the rationale.
IMPL_OP(GnTrainingUpdate).InferDataType(InferDataTypeForGnTrainingUpdate);
IMPL_OP(GNTrainingUpdate).InferDataType(InferDataTypeForGnTrainingUpdate);
} // namespace ops

// ---------------------------------------------------------------------------
// V1 (Runtime 1.0) infer-shape override for GNTrainingUpdate.
//
// CANN's built-in legacy op proto (libopgraph_legacy.so, reduce_ops.cc
// GNTrainingUpdateInferShape) registers a V1 common infer func for the same
// graph op type. GE's CallInferFunc prefers a V1 func on the OpDesc when one
// exists, and the built-in V1 func hard-fails unless the input format is
// NCHW/NHWC — this operator's interface is FORMAT_ND, so every GEIR case was
// rejected before the V2 IMPL_OP_INFERSHAPE callback could run.
//
// This V1 func implements the same derivation as the V2 callbacks (y <- x;
// batch_mean/batch_variance <- sum, shapes and dtypes) WITHOUT the
// NCHW/NHWC-only format restriction, so FORMAT_ND graphs infer correctly.
// The vendor opsproto so is loaded before the built-in package, so this
// registration shadows the built-in V1 func for GNTrainingUpdate.
// ---------------------------------------------------------------------------
IMPLEMT_COMMON_INFERFUNC(InferShapeV1ForGnTrainingUpdate)
{
    // ---- input validation (same contract as the V2 callbacks) ----
    // Name-based desc reads are safe for absent optional inputs: they report
    // a default desc with DT_UNDEFINED dtype and are skipped.
    const char* input_names[7] = {"x", "sum", "square_sum", "scale", "offset", "mean", "variance"};
    std::vector<ge::DataType> input_dtypes;
    std::vector<ge::Shape> input_shapes;
    for (const char* name : input_names) {
        auto desc = op.GetInputDescByName(name);
        input_dtypes.push_back(desc.GetDataType());
        input_shapes.push_back(desc.GetShape());
        // Format contract: x 支持 FORMAT_ND/NCHW/NHWC（与 OpDef/aclnn/tiling
        // 对齐），其余输入仅 FORMAT_ND。Absent optional inputs (DT_UNDEFINED)
        // are skipped.
        if (desc.GetDataType() != ge::DT_UNDEFINED) {
            bool isX = (name == input_names[0]);
            ge::Format fmt = desc.GetFormat();
            bool fmtOk = isX ? (fmt == ge::FORMAT_ND || fmt == ge::FORMAT_NCHW || fmt == ge::FORMAT_NHWC) :
                               (fmt == ge::FORMAT_ND);
            if (!fmtOk) {
                OP_LOGE("GnTrainingUpdate", "input %s format unsupported.", name);
                return GRAPH_FAILED;
            }
        }
    }
    if (ops::CheckInputDtypes(input_dtypes, "GnTrainingUpdate") != ge::GRAPH_SUCCESS) {
        return GRAPH_FAILED;
    }
    // Absent optional inputs report DT_UNDEFINED and are skipped in shape checks.
    const auto present = [&input_dtypes, &input_shapes](size_t i) -> const ge::Shape* {
        return input_dtypes[i] == ge::DT_UNDEFINED ? nullptr : &input_shapes[i];
    };
    if (ops::CheckInputShapesV1(input_shapes[0], input_shapes[1], input_shapes[2], present(3), present(4), present(5),
                                present(6), "GnTrainingUpdate") != ge::GRAPH_SUCCESS) {
        return GRAPH_FAILED;
    }

    // y <- x shape + dtype (normalization never changes the feature map)
    auto x_desc = op.GetInputDescByName("x");
    TensorDesc y_desc = op.GetOutputDescByName("y");
    y_desc.SetShape(x_desc.GetShape());
    y_desc.SetDataType(x_desc.GetDataType());
    if (op.UpdateOutputDesc("y", y_desc) != GRAPH_SUCCESS) {
        OP_LOGE("GnTrainingUpdate", "UpdateOutputDesc y failed.");
        return GRAPH_FAILED;
    }

    // batch_mean / batch_variance <- sum shape + dtype (float32)
    auto sum_desc = op.GetInputDescByName("sum");
    TensorDesc mean_desc = op.GetOutputDescByName("batch_mean");
    mean_desc.SetShape(sum_desc.GetShape());
    mean_desc.SetDataType(sum_desc.GetDataType());
    if (op.UpdateOutputDesc("batch_mean", mean_desc) != GRAPH_SUCCESS) {
        OP_LOGE("GnTrainingUpdate", "UpdateOutputDesc batch_mean failed.");
        return GRAPH_FAILED;
    }

    TensorDesc variance_desc = op.GetOutputDescByName("batch_variance");
    variance_desc.SetShape(sum_desc.GetShape());
    variance_desc.SetDataType(sum_desc.GetDataType());
    if (op.UpdateOutputDesc("batch_variance", variance_desc) != GRAPH_SUCCESS) {
        OP_LOGE("GnTrainingUpdate", "UpdateOutputDesc batch_variance failed.");
        return GRAPH_FAILED;
    }

    return GRAPH_SUCCESS;
}
COMMON_INFER_FUNC_REG(GNTrainingUpdate, InferShapeV1ForGnTrainingUpdate);

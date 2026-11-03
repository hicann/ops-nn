/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file l2_normalize_graph_infer.cpp
 * \brief Runtime 2.0 data type inference for L2Normalize.
 */

#include "register/op_impl_registry.h"
#include "op_common/log/log.h"
#include "l2_normalize_proto.h"

namespace {
bool DeclaredShapeCompatible(const ge::Shape& declared, const ge::Shape& input)
{
    if (declared.GetDimNum() == 0U || declared.GetDims() == ge::UNKNOWN_RANK || input.GetDims() == ge::UNKNOWN_RANK) {
        return true;
    }
    if (declared.GetDimNum() != input.GetDimNum()) {
        return false;
    }
    for (size_t index = 0U; index < input.GetDimNum(); ++index) {
        if (input.GetDim(index) >= 0 && declared.GetDim(index) >= 0 && input.GetDim(index) != declared.GetDim(index)) {
            return false;
        }
    }
    return true;
}
} // namespace

namespace ge {
// Validate graph declarations before inference replaces the output descriptors.
// Runtime 2.0 output buffers contain inferred values, not the original declarations.
IMPLEMT_VERIFIER(L2Normalize, VerifyL2Normalize)
{
    const auto x = op.GetInputDesc("x");
    const auto y = op.GetOutputDesc("y");
    if (x.GetDataType() != DT_UNDEFINED && x.GetDataType() != DT_FLOAT16 && x.GetDataType() != DT_FLOAT) {
        OP_LOGE("L2Normalize", "x must be FLOAT16 or FLOAT32.");
        return GRAPH_FAILED;
    }
    const auto shape = x.GetShape();
    if (shape.GetDims() != UNKNOWN_RANK && (shape.GetDimNum() == 0U || shape.GetDimNum() > 8U)) {
        OP_LOGE("L2Normalize", "x rank must be in [1, 8].");
        return GRAPH_FAILED;
    }
    if (!DeclaredShapeCompatible(y.GetShape(), shape)) {
        OP_LOGE("L2Normalize", "Declared y shape must match x shape.");
        return GRAPH_FAILED;
    }
    if (y.GetDataType() != DT_UNDEFINED && y.GetDataType() != x.GetDataType()) {
        OP_LOGE("L2Normalize", "Declared y dtype must match x dtype.");
        return GRAPH_FAILED;
    }
    if ((x.GetFormat() != FORMAT_ND && x.GetFormat() != FORMAT_RESERVED) ||
        (y.GetFormat() != FORMAT_ND && y.GetFormat() != FORMAT_RESERVED)) {
        OP_LOGE("L2Normalize", "x and y must use ND format.");
        return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
}
VERIFY_FUNC_REG(L2Normalize, VerifyL2Normalize);
} // namespace ge

namespace ops {
namespace {
constexpr size_t INPUT_X_IDX = 0;
constexpr size_t OUTPUT_Y_IDX = 0;
} // namespace

ge::graphStatus InferDataTypeForL2Normalize(gert::InferDataTypeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    return context->SetOutputDataType(OUTPUT_Y_IDX, context->GetInputDataType(INPUT_X_IDX));
}

IMPL_OP(L2Normalize).InferDataType(InferDataTypeForL2Normalize);
} // namespace ops

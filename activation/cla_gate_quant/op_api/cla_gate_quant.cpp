/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "activation/cla_gate_quant/op_api/cla_gate_quant.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(ClaGateQuant);

namespace {
constexpr int64_t ROW_BLOCK = 64;

op::Shape MakeRowDataShape(const op::Shape& xShape)
{
    op::Shape shape;
    shape.AppendDim(xShape.GetDim(0));
    shape.AppendDim(xShape.GetDim(1) * xShape.GetDim(2));
    return shape;
}

op::Shape MakeRowScaleShape(const op::Shape& xShape)
{
    op::Shape shape;
    shape.AppendDim(xShape.GetDim(0));
    shape.AppendDim((xShape.GetDim(1) * xShape.GetDim(2) + ROW_BLOCK - 1) / ROW_BLOCK);
    shape.AppendDim(2);
    return shape;
}

op::Shape MakeColScaleShape(const op::Shape& xShape)
{
    op::Shape shape;
    shape.AppendDim((xShape.GetDim(0) + ROW_BLOCK - 1) / ROW_BLOCK);
    shape.AppendDim(xShape.GetDim(1) * xShape.GetDim(2));
    shape.AppendDim(2);
    return shape;
}
} // namespace

std::tuple<aclTensor*, aclTensor*, aclTensor*, aclTensor*> ClaGateQuant(
    const aclTensor* globalAttn, const aclTensor* localAttn, const aclTensor* globalGateLogits,
    const aclTensor* localGateLogits, const char* roundMode, int64_t scaleAlg, int64_t dstType,
    const char* inputAttnLayout, bool dualAxisFlag, aclOpExecutor* executor)
{
    L0_DFX(ClaGateQuant, globalAttn, localAttn, globalGateLogits, localGateLogits, roundMode, scaleAlg, dstType,
           inputAttnLayout, dualAxisFlag);

    auto xShape = globalAttn->GetViewShape();
    auto rowDataShape = MakeRowDataShape(xShape);
    auto rowScaleShape = MakeRowScaleShape(xShape);
    op::Shape colDataShape;
    op::Shape colScaleShape;
    if (!dualAxisFlag) {
        colDataShape.AppendDim(0);
        colScaleShape.AppendDim(0);
    } else {
        colDataShape = rowDataShape;
        colScaleShape = MakeColScaleShape(xShape);
    }

    auto rowData = executor->AllocTensor(rowDataShape, op::DataType(dstType));
    auto rowScale = executor->AllocTensor(rowScaleShape, op::DataType::DT_FLOAT8_E8M0);
    auto colData = executor->AllocTensor(colDataShape, op::DataType(dstType));
    auto colScale = executor->AllocTensor(colScaleShape, op::DataType::DT_FLOAT8_E8M0);
    OP_CHECK_NULL(rowData, return {});
    OP_CHECK_NULL(rowScale, return {});
    OP_CHECK_NULL(colData, return {});
    OP_CHECK_NULL(colScale, return {});

    // OP_ATTR order must match op_def attr declaration:
    // dst_type(0), round_mode(1), scale_alg(2), input_attn_layout(3), dual_axis_flag(4)
    const char* normalizedRoundMode = (roundMode != nullptr) ? roundMode : "rint";
    const char* attnLayout = (inputAttnLayout != nullptr) ? inputAttnLayout : "TND";
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(ClaGateQuant,
                                           OP_INPUT(globalAttn, localAttn, globalGateLogits, localGateLogits),
                                           OP_OUTPUT(rowData, rowScale, colData, colScale),
                                           OP_ATTR(dstType, normalizedRoundMode, scaleAlg, attnLayout, dualAxisFlag));
    OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(ret != ACLNN_SUCCESS, return {},
                                         "ClaGateQuant ADD_TO_LAUNCHER_LIST_AICORE failed.");
    return std::make_tuple(rowData, rowScale, colData, colScale);
}

} // namespace l0op

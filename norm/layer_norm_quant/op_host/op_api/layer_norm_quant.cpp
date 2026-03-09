/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "layer_norm_quant.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(LayerNormQuant);

static inline void InferReduceShape(
    const op::Shape& xShape, op::Shape& reduceShape, int quantMode)
{
    bool isDyn = (quantMode == 1);  // 0 static; 1 dynamic
    if (isDyn) {
        size_t reduceShapeDim = xShape.GetDimNum() - 1;
        for (size_t i = 0; i < reduceShapeDim; i++) {
            reduceShape.AppendDim(xShape.GetDim(i));
        }
    } else {
        reduceShape.AppendDim(1);
    }
    return;
}

std::array<aclTensor*, LAYER_NORM_QUANT_OUT_NUM> LayerNormQuant(
    const aclTensor* x, const aclTensor* gamma, const aclTensor* beta, const aclTensor* scale,
    const aclTensor* zeroPointsOptional, int quantMode, double epsilon, aclOpExecutor* executor)
{
    OP_LOGI("LayerNormQuant L0 Start.");
    Shape reduceShape;
    Shape dummyShape({1});
    InferReduceShape(x->GetViewShape(), reduceShape, quantMode);

    L0_DFX(LayerNormQuant, x, gamma, beta, scale, zeroPointsOptional, quantMode, epsilon);

    OP_LOGI("LayerNormQuant L0_DFX.");

    auto res = executor->AllocTensor(x->GetViewShape(), DataType::DT_INT8, op::Format::FORMAT_ND);
    auto scaleOut = executor->AllocTensor(reduceShape, DataType::DT_FLOAT, op::Format::FORMAT_ND);

    OP_LOGI("LayerNormQuant alloc out.");

    OP_LOGI(
        "res=[%s], scaleOut=[%s].",
        op::ToString(res->GetViewShape()).GetString(), op::ToString(scaleOut->GetViewShape()).GetString());

    ADD_TO_LAUNCHER_LIST_AICORE(
        LayerNormQuant,
        OP_INPUT(x, gamma, beta, scale, zeroPointsOptional),
        OP_OUTPUT(res, scaleOut),
        OP_ATTR(static_cast<float>(epsilon), quantMode));

    OP_LOGI("LayerNormQuant Launch finish.");

    return {res, scaleOut};
}

} // namespace l0op

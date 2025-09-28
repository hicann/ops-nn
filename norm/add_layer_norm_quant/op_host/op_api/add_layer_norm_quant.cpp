/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "add_layer_norm_quant.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(AddLayerNormQuant);

static inline void InferReduceShape(
    const op::Shape& xShape, const op::Shape& gammaShape, op::Shape& reduceShape, const char* quantMode)
{
    bool isDyn = (strncmp(quantMode, "dynamic", strlen("dynamic")) == 0);
    if (isDyn) {
        size_t reduceShapeDim = xShape.GetDimNum() - gammaShape.GetDimNum();
        for (size_t i = 0; i < reduceShapeDim; i++) {
            reduceShape.AppendDim(xShape.GetDim(i));
        }
    } else {
        reduceShape.AppendDim(1);
    }
    return;
}

const std::array<aclTensor*, ADD_LAYER_NORM_QUANT_OUT_NUM> AddLayerNormQuant(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* gamma, const aclTensor* beta,
    const aclTensor* biasOptional, const aclTensor* scales1Optional, const aclTensor* scales2Optional,
    const aclTensor* zeroPoints1Optional, const aclTensor* zeroPoints2Optional, const char* quantMode, double epsilon,
    bool additionalOutput, bool divMode, aclOpExecutor* executor)
{
    OP_LOGI("AddLayerNormQuant L0 Start.");
    bool isDual = (nullptr != scales2Optional);
    Shape reduceShape;
    Shape dummyShape({1});
    InferReduceShape(x1->GetViewShape(), gamma->GetViewShape(), reduceShape, quantMode);

    L0_DFX(
        AddLayerNormQuant, x1, x2, gamma, beta, biasOptional, scales1Optional, scales2Optional, zeroPoints1Optional,
        zeroPoints2Optional, quantMode, epsilon, additionalOutput, divMode);

    OP_LOGI("AddLayerNormQuant L0_DFX.");

    auto y1Out = executor->AllocTensor(x1->GetViewShape(), DataType::DT_INT8, op::Format::FORMAT_ND);
    auto xOut = executor->AllocTensor(x1->GetViewShape(), x1->GetDataType(), op::Format::FORMAT_ND);
    auto outScales1Out = executor->AllocTensor(reduceShape, DataType::DT_FLOAT, op::Format::FORMAT_ND);

    auto y2Out = (isDual) ? executor->AllocTensor(x1->GetViewShape(), DataType::DT_INT8, op::Format::FORMAT_ND) :
                            executor->AllocTensor(dummyShape, DataType::DT_INT8, op::Format::FORMAT_ND);
    auto outScales2Out = (isDual) ? executor->AllocTensor(reduceShape, DataType::DT_FLOAT, op::Format::FORMAT_ND) :
                                    executor->AllocTensor(dummyShape, DataType::DT_FLOAT, op::Format::FORMAT_ND);

    OP_LOGI("AddLayerNormQuant alloc out.");

    OP_LOGI(
        "y1Out=[%s], y2Out=[%s], xOut=[%s], outScales1Out=[%s], outScales2Out=[%s].",
        op::ToString(y1Out->GetViewShape()).GetString(), op::ToString(y2Out->GetViewShape()).GetString(),
        op::ToString(xOut->GetViewShape()).GetString(), op::ToString(outScales1Out->GetViewShape()).GetString(),
        op::ToString(outScales2Out->GetViewShape()).GetString());

    ADD_TO_LAUNCHER_LIST_AICORE(
        AddLayerNormQuant,
        OP_INPUT(
            x1, x2, gamma, beta, biasOptional, scales1Optional, scales2Optional, zeroPoints1Optional,
            zeroPoints2Optional),
        OP_OUTPUT(y1Out, y2Out, xOut, outScales1Out, outScales2Out),
        OP_ATTR(quantMode, static_cast<float>(epsilon), additionalOutput, divMode));

    OP_LOGI("AddLayerNormQuant Launch finish.");

    return {y1Out, y2Out, xOut, outScales1Out, outScales2Out};
}

} // namespace l0op

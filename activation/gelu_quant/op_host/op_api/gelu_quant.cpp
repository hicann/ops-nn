/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gelu_quant.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/shape_utils.h"
#include "opdev/make_op_executor.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(GeluQuant);

static inline void InferReduceShape(const op::Shape& xShape, op::Shape& outputShape, const char* quantMode)
{
    bool isDyn = (strncmp(quantMode, "dynamic", strlen("dynamic")) == 0);
    if (isDyn) {
        size_t outputShapeDim = xShape.GetDimNum() - 1;
        for (int64_t i = 0; i < static_cast<int64_t>(outputShapeDim); i++) {
            outputShape.AppendDim(xShape.GetDim(i));
        }
    } else {
        outputShape.AppendDim(1);
    }
    return;
}

std::tuple<aclTensor*, aclTensor*> GeluQuant(
    const aclTensor* self, const aclTensor* inputScaleOptional, const aclTensor* inputOffsetOptional,
    const char* approximate, const char* quantMode, const char* roundMode, int64_t dstType, aclOpExecutor* executor)
{
    Shape outputShape;
    Shape dummyShape({1});
    InferReduceShape(self->GetViewShape(), outputShape, quantMode);
    L0_DFX(GeluQuant, self, inputScaleOptional, inputOffsetOptional, approximate, quantMode, roundMode, dstType);
    auto yOut = executor->AllocTensor(
        self->GetStorageShape(), self->GetViewShape(), op::DataType(dstType), self->GetStorageFormat(),
        self->GetOriginalFormat());
    bool isDyn = (strncmp(quantMode, "dynamic", strlen("dynamic")) == 0);
    auto output = (isDyn) ? executor->AllocTensor(outputShape, DataType::DT_FLOAT, op::Format::FORMAT_ND) :
                            executor->AllocTensor(dummyShape, DataType::DT_FLOAT, op::Format::FORMAT_ND);
    if (yOut == nullptr || output == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "alloc y or output tensor failed.");
        return std::tie(yOut, output);
    }

    OP_LOGD(
        "y shape=[%s], output shape=[%s].", op::ToString(yOut->GetViewShape()).GetString(),
        op::ToString(output->GetViewShape()).GetString());

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        GeluQuant, OP_INPUT(self, inputScaleOptional, inputOffsetOptional), OP_OUTPUT(yOut, output),
        OP_ATTR(approximate, quantMode, dstType, roundMode));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "GeluQuant launch kernel failed.");
        return std::tuple<aclTensor*, aclTensor*>(nullptr, nullptr);
    }
    return std::tuple<aclTensor*, aclTensor*>(yOut, output);
}

} // namespace l0op
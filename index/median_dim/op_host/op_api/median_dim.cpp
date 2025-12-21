/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "median_dim.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;
namespace l0op {

OP_TYPE_REGISTER(MedianDim);

static const std::tuple<aclTensor*, aclTensor*> MedianDimAiCore(const aclTensor *self, int64_t dim, bool keepDim,
                                                         aclTensor *valuesOut, aclTensor *indicesOut, 
                                                         aclOpExecutor *executor) {
    L0_DFX(MedianDimAiCore, self, dim, keepDim, valuesOut, indicesOut);
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(MedianDim, OP_INPUT(self), OP_OUTPUT(valuesOut, indicesOut), OP_ATTR(dim, keepDim));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "MedianDimAiCore ADD_TO_LAUNCHER_LIST_AICORE failed.");
        return std::tuple<aclTensor*, aclTensor*>(nullptr, nullptr);
    }
    return std::tuple<aclTensor*, aclTensor*>(valuesOut, indicesOut);
}

static bool MedianDimInferShape(const op::Shape &selfShape, int64_t dim,
                         bool keepDims, op::Shape &reduceShape) {
    int64_t reduceDim = 1;
    for (size_t i = 0; i < selfShape.GetDimNum(); i++) {
        if (keepDims) {
            if (static_cast<int64_t>(i) != dim) {
                reduceShape.AppendDim(selfShape.GetDim(i));
            } else {
                reduceShape.AppendDim(reduceDim);
            }
        } else {
            if (static_cast<int64_t>(i) != dim) {
                reduceShape.AppendDim(selfShape.GetDim(i));
            }
        }
    }
    return true;
}

const std::tuple<aclTensor*, aclTensor*> MedianDim(const aclTensor *self, int64_t dim, bool keepDim, aclOpExecutor *executor) {
    op::Shape reduceShape;
    MedianDimInferShape(self->GetViewShape(), dim, keepDim, reduceShape);
    auto values = executor->AllocTensor(reduceShape, self->GetDataType());
    auto indices = executor->AllocTensor(reduceShape, DataType::DT_INT32);
    return MedianDimAiCore(self, dim, keepDim, values, indices, executor);
}
}  // namespace l0op
 
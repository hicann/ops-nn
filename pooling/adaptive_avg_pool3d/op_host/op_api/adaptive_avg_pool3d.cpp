/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "adaptive_avg_pool3d.h"
#include "opdev/common_types.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
using namespace op;
namespace l0op {

OP_TYPE_REGISTER(AdaptiveAvgPool3d);

static constexpr size_t DIM_D = 0;
static constexpr size_t DIM_H = 1;
static constexpr size_t DIM_W = 2;

static const aclTensor* AdaptiveAvgPool3dAiCore(
    const aclTensor* self, const aclIntArray* outputSize, aclTensor* out, aclOpExecutor* executor)
{
    L0_DFX(AdaptiveAvgPool3dAiCore, self, outputSize, out);
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(AdaptiveAvgPool3d, OP_INPUT(self), OP_OUTPUT(out), OP_ATTR(outputSize));
    OP_CHECK(
        ret == ACLNN_SUCCESS,
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "AdaptiveAvgPool3dAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."),
        return nullptr);
    return out;
}

const aclTensor* AdaptiveAvgPool3d(const aclTensor* self, const aclIntArray* outputSize, aclOpExecutor* executor)
{
    L0_DFX(AdaptiveAvgPool3d, self, outputSize);
    op::Shape outShape = self->GetViewShape();
    outShape.SetDim(DIM_D + 1, (*outputSize)[DIM_D]);
    outShape.SetDim(DIM_H + 1, (*outputSize)[DIM_H]);
    outShape.SetDim(DIM_W + 1, (*outputSize)[DIM_W]);

    auto out = executor->AllocTensor(outShape, self->GetDataType(), self->GetStorageFormat());
    if (out == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "out is nullptr.");
        return nullptr;
    }
    return AdaptiveAvgPool3dAiCore(self, outputSize, out, executor);
}
} // namespace l0op
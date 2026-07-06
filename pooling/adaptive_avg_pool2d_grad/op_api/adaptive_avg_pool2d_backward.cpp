/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "adaptive_avg_pool2d_backward.h"
#include "opdev/common_types.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "op_api/aclnn_util.h"
using namespace op;

namespace l0op {

OP_TYPE_REGISTER(AdaptiveAvgPool2dGrad);
static constexpr size_t DIM_NCHW = 4;
static constexpr size_t DIM_CHW = 3;
static constexpr size_t DIM_H = 2;
static constexpr size_t DIM_W = 3;

static const aclTensor* AdaptiveAvgPool2dGradAiCore(const aclTensor* gradOutput, const aclIntArray* originInputShape,
                                                    aclTensor* out, aclOpExecutor* executor)
{
    L0_DFX(AdaptiveAvgPool2dGradAiCore, gradOutput, originInputShape, out);
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(AdaptiveAvgPool2dGrad, OP_INPUT(gradOutput), OP_OUTPUT(out),
                                           OP_ATTR(originInputShape));
    OP_CHECK(ret == ACLNN_SUCCESS,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "AdaptiveAvgPool2dGradAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."),
             return nullptr);
    return out;
}

const aclTensor* AdaptiveAvgPool2dGrad(const aclTensor* gradOutput, const aclTensor* self, aclOpExecutor* executor)
{
    L0_DFX(AdaptiveAvgPool2dGrad, gradOutput, self);
    if (!Ops::NN::AclnnUtil::IsRegbase()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "AdaptiveAvgPool2dGrad only support in Ascend950.");
        return nullptr;
    }
    op::Shape selfShape = self->GetViewShape();
    size_t dimNum = selfShape.GetDimNum();
    aclIntArray* inputShape = nullptr;
    if (dimNum == DIM_NCHW) {
        const int64_t n = selfShape.GetDim(0);
        const int64_t c = selfShape.GetDim(1);
        const int64_t h = selfShape.GetDim(DIM_H);
        const int64_t w = selfShape.GetDim(DIM_W);
        FVector<int64_t> originInput{n, c, h, w};
        inputShape = executor->AllocIntArray(originInput.data(), DIM_NCHW);
    } else {
        const int64_t c = selfShape.GetDim(0);
        const int64_t h = selfShape.GetDim(DIM_H - 1);
        const int64_t w = selfShape.GetDim(DIM_W - 1);
        FVector<int64_t> originInput{c, h, w};
        inputShape = executor->AllocIntArray(originInput.data(), DIM_CHW);
    }
    auto out = executor->AllocTensor(selfShape, self->GetDataType(), self->GetStorageFormat());
    if (out == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "out is nullptr.");
        return nullptr;
    }
    return AdaptiveAvgPool2dGradAiCore(gradOutput, inputShape, out, executor);
}
} // namespace l0op
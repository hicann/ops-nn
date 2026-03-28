/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "adaptive_avg_pool2d.h"
#include "opdev/common_types.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "op_api/aclnn_util.h"
using namespace op;
namespace l0op {

OP_TYPE_REGISTER(AdaptiveAvgPool2d);

static constexpr size_t SUB_H = -2;
static constexpr size_t SUB_W = -1;

static const aclTensor* AdaptiveAvgPool2dAiCore(
    const aclTensor* self, const aclIntArray* outputSize, aclTensor* out, aclOpExecutor* executor)
{
    L0_DFX(AdaptiveAvgPool2dAiCore, self, outputSize, out);
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(AdaptiveAvgPool2d, OP_INPUT(self), OP_OUTPUT(out), OP_ATTR(outputSize));
    OP_CHECK(
        ret == ACLNN_SUCCESS,
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "AdaptiveAvgPool2dAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."),
        return nullptr);
    return out;
}

const aclTensor* AdaptiveAvgPool2d(const aclTensor* self, const aclIntArray* outputSize, aclOpExecutor* executor)
{
    L0_DFX(AdaptiveAvgPool2d, self, outputSize);
    op::Shape outShape = self->GetViewShape();
    if (Ops::NN::AclnnUtil::IsRegbase()) {
        uint64_t size = outShape.GetDimNum();
        outShape.SetDim(size + SUB_H, (*outputSize)[0]);
        outShape.SetDim(size + SUB_W, (*outputSize)[1]); 
    }else {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "AdaptiveAvgPool2dAiCore only support ascendC950 failed.");
    }
    auto out = executor->AllocTensor(outShape, self->GetDataType(), self->GetStorageFormat());
    if (out == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "out is nullptr.");
        return nullptr;
    }
    return AdaptiveAvgPool2dAiCore(self, outputSize, out, executor);
}
} // namespace l0op
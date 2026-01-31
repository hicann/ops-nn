/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "adaptive_max_pool3d.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/common_types.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "op_api/aclnn_util.h"

using namespace op;
namespace l0op {
OP_TYPE_REGISTER(AdaptiveMaxPool3d);

static constexpr size_t NCDHW_DIM_NUM = 5;
static constexpr size_t DIM_D = 2;
static constexpr size_t DIM_H = 3;
static constexpr size_t DIM_W = 4;
static constexpr size_t DIM_ZERO = 0;
static constexpr size_t DIM_ONE = 1;
static constexpr size_t DIM_TWO = 2;
static const int32_t DTYPE_INT32 = 3;
static const int32_t DTYPE_INT64 = 9;
static const int32_t DHW_DIMS = 3;

static const std::tuple<aclTensor*, aclTensor*> AdaptiveMaxPool3dAiCore(
    const aclTensor* self, const aclIntArray* outputSize, aclTensor* outputOut, aclTensor* indicesOut,
    aclOpExecutor* executor)
{
    L0_DFX(AdaptiveMaxPool3dAiCore, self, outputSize, outputOut, indicesOut);

    if (Ops::NN::AclnnUtil::IsRegbase()) {
        int32_t indicesDtype = (indicesOut->GetDataType() == op::DataType::DT_INT64) ? DTYPE_INT64 : DTYPE_INT32;
        ADD_TO_LAUNCHER_LIST_AICORE(
        AdaptiveMaxPool3d, OP_INPUT(self), OP_OUTPUT(outputOut, indicesOut), OP_ATTR(outputSize, indicesDtype));
    } else {
        ADD_TO_LAUNCHER_LIST_AICORE(
        AdaptiveMaxPool3d, OP_INPUT(self), OP_OUTPUT(outputOut, indicesOut), OP_ATTR(outputSize));
    }

    return std::tuple<aclTensor*, aclTensor*>(outputOut, indicesOut);
}

const std::tuple<const aclTensor*, const aclTensor*> AdaptiveMaxPool3d(
    const aclTensor* self, const aclIntArray* outputSize, aclOpExecutor* executor, op::DataType indicesDtype)
{
    L0_DFX(AdaptiveMaxPool3d, self, outputSize);

    op::Shape outShape = self->GetViewShape();
    size_t dimNum = outShape.GetDimNum();
    if (dimNum != NCDHW_DIM_NUM) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Input dim num is not supported.");
        return std::tuple<aclTensor*, aclTensor*>(nullptr, nullptr);
    }

    outShape.SetDim(DIM_D, (*outputSize)[DIM_ZERO]);
    outShape.SetDim(DIM_H, (*outputSize)[DIM_ONE]);
    outShape.SetDim(DIM_W, (*outputSize)[DIM_TWO]);

    auto outputOut = executor->AllocTensor(outShape, self->GetDataType(), self->GetStorageFormat());
    auto indicesOut = executor->AllocTensor(outShape, indicesDtype, self->GetStorageFormat());
    if (outputOut == nullptr || indicesOut == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "outputOut or indicesOut is nullptr.");
        return std::tuple<aclTensor*, aclTensor*>(nullptr, nullptr);
    }
    return AdaptiveMaxPool3dAiCore(self, outputSize, outputOut, indicesOut, executor);
}
} // namespace l0op
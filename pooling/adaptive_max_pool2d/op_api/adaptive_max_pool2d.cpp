/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "adaptive_max_pool2d.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/common_types.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "op_api/aclnn_util.h"

using namespace op;
namespace l0op {
OP_TYPE_REGISTER(AdaptiveMaxPool2d);

static constexpr size_t DIM_H = 2;
static constexpr size_t DIM_W = 1;
static constexpr size_t NHWC_DIM_H = 3;
static constexpr size_t NHWC_DIM_W = 2;

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST_950 = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

static bool IsAscend950iCoreSupport(const aclTensor* self)
{
    if (!Ops::NN::AclnnUtil::IsRegbase()) {
        return false;
    }
    return CheckType(self->GetDataType(), AICORE_DTYPE_SUPPORT_LIST_950);
}

std::tuple<aclTensor*, aclTensor*> AdapativeMaxPool2dAiCpu(
    const aclTensor* self, const aclIntArray* outputSize, aclTensor* outputOut, aclTensor* indicesOut,
    aclOpExecutor* executor)
{
    L0_DFX(AdapativeMaxPool2dAiCpu, self, outputSize, outputOut, indicesOut);
    static internal::AicpuTaskSpace space("AdaptiveMaxPool2d");
    auto ret = ADD_TO_LAUNCHER_LIST_AICPU(
        AdaptiveMaxPool2d, OP_ATTR_NAMES({"output_size"}), OP_INPUT(self), OP_OUTPUT(outputOut, indicesOut),
        OP_ATTR(outputSize));
    if (ret != ACL_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "AdaptiveAvgPool2dAssistMatrixAiCpu ADD_TO_LAUNCHER_LIST_AICPU failed.");
        return std::tuple<aclTensor*, aclTensor*>(nullptr, nullptr);
    }
    return std::tuple<aclTensor*, aclTensor*>(outputOut, indicesOut);
}

std::tuple<aclTensor*, aclTensor*> AdapativeMaxPool2dAiCore(
    const aclTensor* self, const aclIntArray* outputSize, aclTensor* outputOut, aclTensor* indicesOut,
    aclOpExecutor* executor)
{
    L0_DFX(AdapativeMaxPool2dAiCore, self, outputSize, outputOut, indicesOut);
    ADD_TO_LAUNCHER_LIST_AICORE(
        AdaptiveMaxPool2d, OP_INPUT(self), OP_OUTPUT(outputOut, indicesOut), OP_ATTR(outputSize));
    return std::tuple<aclTensor*, aclTensor*>(outputOut, indicesOut);
}


std::tuple<aclTensor*, aclTensor*> AdaptiveMaxPool2d(
    const aclTensor* self, const aclIntArray* outputSize, aclOpExecutor* executor)
{
    L0_DFX(AdaptiveMaxPool2d, self, outputSize);

    op::Shape outShape = self->GetViewShape();
    size_t dimNum = outShape.GetDimNum();
    auto input_format = self->GetViewFormat();
    if (input_format == op::Format::FORMAT_NCHW) {
        outShape.SetDim(dimNum - DIM_H, (*outputSize)[0]);
        outShape.SetDim(dimNum - DIM_W, (*outputSize)[1]);
    } else {
        outShape.SetDim(dimNum - NHWC_DIM_H, (*outputSize)[0]);
        outShape.SetDim(dimNum - NHWC_DIM_W, (*outputSize)[1]);
    }

    auto outputOut = executor->AllocTensor(outShape, self->GetDataType(), self->GetStorageFormat());
    
    if (outputOut == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "outputOut is nullptr.");
        return std::tuple<aclTensor*, aclTensor*>(nullptr, nullptr);
    }
    if (IsAscend950iCoreSupport(self)) {
        auto indicesInt32Out = executor->AllocTensor(outShape, op::DataType::DT_INT32, self->GetStorageFormat());
        if (indicesInt32Out == nullptr) {
            OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "indicesInt32Out is nullptr.");
            return std::tuple<aclTensor*, aclTensor*>(nullptr, nullptr);
        }
        OP_LOGI("beigin adaptive maxpool2d.\n");
        return AdapativeMaxPool2dAiCore(self, outputSize, outputOut, indicesInt32Out, executor);
    }
    auto indicesInt64Out = executor->AllocTensor(outShape, op::DataType::DT_INT64, self->GetStorageFormat());
    if (indicesInt64Out == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "indicesInt64Out is nullptr.");
        return std::tuple<aclTensor*, aclTensor*>(nullptr, nullptr);
    }
    return AdapativeMaxPool2dAiCpu(self, outputSize, outputOut, indicesInt64Out, executor);
}
} // namespace l0op

/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file deformable_offsets_grad.cpp
 * \brief
 */
#include "deformable_offsets_grad.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(DeformableOffsetsGrad);

const std::tuple<aclTensor*, aclTensor*> DeformableOffsetsGrad(
    const aclTensor* grad_output, const aclTensor* input, const aclTensor* offsets, const aclIntArray* stride,
    const aclIntArray* pads, const aclIntArray* kernel_size, const aclIntArray* dilations, bool modulated,
    int64_t deformable_groups, aclOpExecutor* executor)
{
    L0_DFX(DeformableOffsetsGrad, grad_output, input, offsets);
    const char* data_format = "NHWC";
    auto grad_x = executor->AllocTensor(
        input->GetStorageShape(), input->GetOriginalShape(), input->GetDataType(), input->GetStorageFormat(),
        input->GetOriginalFormat());
    auto grad_offsets = executor->AllocTensor(
        offsets->GetStorageShape(), offsets->GetOriginalShape(), offsets->GetDataType(), offsets->GetStorageFormat(),
        offsets->GetOriginalFormat());
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        DeformableOffsetsGrad, OP_INPUT(grad_output, input, offsets), OP_OUTPUT(grad_x, grad_offsets),
        OP_ATTR(stride, pads, kernel_size, dilations, data_format, deformable_groups, modulated));
    if (ret != ACL_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "DeformableOffsetsGradAiCore ADD_TO_LAUNCHER_LIST_AICORE failed.");
        return std::tuple<aclTensor*, aclTensor*>(nullptr, nullptr);
    }
    return std::tuple<aclTensor*, aclTensor*>(grad_x, grad_offsets);
}
} // namespace l0op

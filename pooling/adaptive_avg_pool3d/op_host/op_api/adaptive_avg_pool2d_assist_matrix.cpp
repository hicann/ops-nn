/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "adaptive_avg_pool2d_assist_matrix.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/common_types.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;
namespace l0op {
OP_TYPE_REGISTER(AdaptiveAvgPool2dAssistMatrix);

bool AssistMatrixInferShape(
    const aclTensor* input, const aclIntArray* output_size, Shape& left_matrix_shape, Shape& right_matrix_shape,
    Shape& mul_matrix)
{
    auto shape = input->GetViewShape();
    size_t dim_num = shape.GetDimNum();
    auto h_dim_index = dim_num - 2;
    auto w_dim_index = dim_num - 1;

    int64_t input_h = shape.GetDim(h_dim_index);
    int64_t input_w = shape.GetDim(w_dim_index);

    int64_t output_h = (*output_size)[0];
    int64_t output_w = (*output_size)[1];

    int64_t max_value = 536870911;

    left_matrix_shape.AppendDim(output_h);
    left_matrix_shape.AppendDim(input_h);

    right_matrix_shape.AppendDim(input_w);
    right_matrix_shape.AppendDim(output_w);

    mul_matrix.AppendDim(output_h);
    mul_matrix.AppendDim(output_w);

    if (output_h * input_h > max_value || input_w * output_w > max_value || output_h * output_w > max_value) {
        return false;
    }
    return true;
}

static const std::array<aclTensor*, 3> AdaptiveAvgPool2dAssistMatrixAiCpu(
    const aclTensor* input, const aclIntArray* output_size, aclTensor* left_matrix, aclTensor* right_matrix,
    aclTensor* mul_matrix, aclOpExecutor* executor)
{
    L0_DFX(AdaptiveAvgPool2dAssistMatrixAiCpu, input, output_size);

    static internal::AicpuTaskSpace space("AdaptiveAvgPool2dAssistMatrix");
    auto ret = ADD_TO_LAUNCHER_LIST_AICPU(
        AdaptiveAvgPool2dAssistMatrix, OP_ATTR_NAMES({"output_size"}), OP_INPUT(input),
        OP_OUTPUT(left_matrix, right_matrix, mul_matrix), OP_ATTR(output_size));
    if (ret != ACL_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "AdaptiveAvgPool2dAssistMatrixAiCpu ADD_TO_LAUNCHER_LIST_AICPU failed.");
        return {nullptr, nullptr, nullptr};
    }
    return {left_matrix, right_matrix, mul_matrix};
}

const std::array<aclTensor*, 3> AdaptiveAvgPool2dAssistMatrix(
    const aclTensor* input, const aclTensor* origin_input, const aclIntArray* output_size, aclOpExecutor* executor)
{
    L0_DFX(AdaptiveAvgPool2dAssistMatrix, input, output_size);
    Shape left_matrix_shape;
    Shape right_matrix_shape;
    Shape mul_matrix_shape;
    if (!AssistMatrixInferShape(origin_input, output_size, left_matrix_shape, right_matrix_shape, mul_matrix_shape)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "AssistMatrixInferShape failed.");
        return {nullptr, nullptr, nullptr};
    }
    auto left_matrix = executor->AllocTensor(left_matrix_shape, DataType::DT_FLOAT, input->GetViewFormat());
    auto right_matrix = executor->AllocTensor(right_matrix_shape, DataType::DT_FLOAT, input->GetViewFormat());
    auto mul_matrix = executor->AllocTensor(mul_matrix_shape, DataType::DT_FLOAT, input->GetViewFormat());
    return AdaptiveAvgPool2dAssistMatrixAiCpu(input, output_size, left_matrix, right_matrix, mul_matrix, executor);
}

} // namespace l0op
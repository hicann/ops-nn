/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "non_zero_with_value_shape_aicpu.h"

#include "cpu_kernel_utils.h"
#include "cpu_tensor.h"
#include "cpu_tensor_shape.h"
#include "cpu_types.h"
#include "utils/kernel_util.h"

namespace {
const char* const kNonZeroWithValueShape = "NonZeroWithValueShape";
const uint32_t INPUTS_NUM = 3;
const uint32_t OUTPUTS_NUM = 2;

const uint32_t IDX_INPUT_TENSOR_COUNT = 2;
const uint32_t IDX_OUTPUT_VALUE = 0;
const uint32_t IDX_OUTPUT_INDEX = 1;
const uint32_t IDX_OUTPUT_INDEX_SHAPE = 2;
} // namespace
namespace aicpu {
uint32_t NonZeroWithValueShapeCpuKernel::Compute(CpuKernelContext& ctx)
{
    KERNEL_HANDLE_ERROR(NormalCheck(ctx, INPUTS_NUM, OUTPUTS_NUM), "Check input and output number failed.");
    Tensor* count = ctx.Input(IDX_INPUT_TENSOR_COUNT);
    KERNEL_CHECK_NULLPTR(count, KERNEL_STATUS_PARAM_INVALID, "[%s] get input_data[2] failed.", kNonZeroWithValueShape);
    KERNEL_CHECK_FALSE((count->NumElements() >= 1), KERNEL_STATUS_PARAM_INVALID,
                       "[%s] input count must contain at least one element.", kNonZeroWithValueShape);

    Tensor* out_value = ctx.Output(IDX_OUTPUT_VALUE);
    auto out_value_shape = out_value->GetTensorShape();
    int32_t count_num = static_cast<int32_t*>(count->GetData())[IDX_OUTPUT_VALUE];
    std::vector<int64_t> out_value_shape_values = {count_num};
    out_value_shape->SetDimSizes(out_value_shape_values);

    Tensor* out_index = ctx.Output(IDX_OUTPUT_INDEX);
    auto out_index_shape = out_index->GetTensorShape();
    std::vector<int64_t> out_index_shape_values = {IDX_OUTPUT_INDEX_SHAPE, count_num};
    out_index_shape->SetDimSizes(out_index_shape_values);

    return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(kNonZeroWithValueShape, NonZeroWithValueShapeCpuKernel);
} // namespace aicpu

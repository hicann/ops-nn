/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "non_zero_with_value_shape_v2_aicpu.h"

#include "cpu_kernel_utils.h"
#include "cpu_tensor.h"
#include "cpu_tensor_shape.h"
#include "cpu_types.h"
#include "utils/kernel_util.h"

namespace {
const char* const kNonZeroWithValueShapeV2 = "NonZeroWithValueShapeV2";
const uint32_t kOutputNum = 2;
const uint32_t kIndexFirstDim = 2;
} // namespace

namespace aicpu {
uint32_t NonZeroWithValueShapeV2CpuKernel::GetInputTypeAndCheck(const CpuKernelContext& ctx)
{
    DataType index_dtype = ctx.Input(kSecondOutputIndex)->GetDataType();
    DataType count_dtype = ctx.Input(kThirdInputIndex)->GetDataType();
    KERNEL_CHECK_FALSE((index_dtype == DT_INT32), KERNEL_STATUS_PARAM_INVALID,
                       "For '%s',  the dtype of '%s' must be '%s', but got [%s].", kNonZeroWithValueShapeV2, "index",
                       DTypeStr(DT_INT32).c_str(), DTypeStr(index_dtype).c_str());
    KERNEL_CHECK_FALSE((count_dtype == DT_INT32), KERNEL_STATUS_PARAM_INVALID,
                       "For '%s',  the dtype of '%s' must be '%s', but got [%s].", kNonZeroWithValueShapeV2, "count",
                       DTypeStr(DT_INT32).c_str(), DTypeStr(count_dtype).c_str());
    return KERNEL_STATUS_OK;
}

uint32_t NonZeroWithValueShapeV2CpuKernel::Compute(CpuKernelContext& ctx)
{
    KERNEL_HANDLE_ERROR(NormalCheck(ctx, INPUT_NUM3, kOutputNum), "[%s] Check input and output number failed.",
                        kNonZeroWithValueShapeV2);
    KERNEL_HANDLE_ERROR(GetInputTypeAndCheck(ctx), "[%s] Check input type failed.", kNonZeroWithValueShapeV2)
    Tensor* count = ctx.Input(kThirdInputIndex);
    KERNEL_CHECK_FALSE((count->NumElements() >= 1), KERNEL_STATUS_PARAM_INVALID,
                       "[%s] input count must contain at least one element.", kNonZeroWithValueShapeV2);
    int32_t count_num = static_cast<int32_t*>(count->GetData())[0];

    Tensor* out_value = ctx.Output(kFirstOutputIndex);
    auto out_value_shape = out_value->GetTensorShape();
    std::vector<int64_t> out_value_shape_values = {count_num};
    out_value_shape->SetDimSizes(out_value_shape_values);

    Tensor* out_index = ctx.Output(kSecondOutputIndex);
    auto out_index_shape = out_index->GetTensorShape();
    std::vector<int64_t> out_index_shape_values = {kIndexFirstDim, count_num};
    out_index_shape->SetDimSizes(out_index_shape_values);

    return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(kNonZeroWithValueShapeV2, NonZeroWithValueShapeV2CpuKernel);
} // namespace aicpu

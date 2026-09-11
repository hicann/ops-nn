/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "nan_median.h"

#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(NanMedian);

static const std::initializer_list<op::DataType> NAN_MEDIAN_DTYPES = {
    op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT,  op::DataType::DT_INT16, op::DataType::DT_INT8,
    op::DataType::DT_UINT8,   op::DataType::DT_INT32,  op::DataType::DT_INT64, op::DataType::DT_BF16,
    op::DataType::DT_UINT32,  op::DataType::DT_UINT16, op::DataType::DT_UINT64};

static constexpr int64_t NON_LAST_SMALL_AXIS_MIN = 2;
static constexpr int64_t NON_LAST_SMALL_AXIS_MAX = 2048;

static bool CheckNanMedianParams(const aclTensor* self, int64_t dim)
{
    OP_CHECK_NULL(self, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(self, NAN_MEDIAN_DTYPES, return false);
    int64_t rank = static_cast<int64_t>(self->GetViewShape().GetDimNum());
    if (rank <= 0 || dim < -rank || dim >= rank) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "NanMedian requires rank > 0 and a valid dim.");
        return false;
    }
    int64_t normDim = dim < 0 ? dim + rank : dim;
    int64_t axisLen = self->GetViewShape().GetDim(static_cast<size_t>(normDim));
    if (axisLen <= 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "NanMedian reduction axis must be non-empty.");
        return false;
    }
    if (normDim != rank - 1 && (axisLen < NON_LAST_SMALL_AXIS_MIN || axisLen > NON_LAST_SMALL_AXIS_MAX)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "NanMedian dim must be the last axis or a non-last axis with size in [2, 2048].");
        return false;
    }
    return true;
}

static bool AllocateNanMedianOutputs(aclOpExecutor* executor, const op::Shape& outputShape, op::DataType valueType,
                                     aclTensor*& values, aclTensor*& indices)
{
    values = executor->AllocTensor(outputShape, valueType, op::Format::FORMAT_ND);
    OP_CHECK_NULL(values, return false);
    indices = executor->AllocTensor(outputShape, op::DataType::DT_INT64, op::Format::FORMAT_ND);
    OP_CHECK_NULL(indices, return false);
    return true;
}

const std::tuple<aclTensor*, aclTensor*> NanMedian(const aclTensor* self, int64_t dim, aclOpExecutor* executor)
{
    L0_DFX(NanMedian, self, dim);
    if (!CheckNanMedianParams(self, dim)) {
        return {};
    }
    int64_t rank = static_cast<int64_t>(self->GetViewShape().GetDimNum());
    int64_t normDim = dim < 0 ? dim + rank : dim;
    auto outShape = self->GetViewShape();
    outShape.SetDim(static_cast<size_t>(normDim), 1);
    aclTensor* y = nullptr;
    aclTensor* indices = nullptr;
    if (!AllocateNanMedianOutputs(executor, outShape, self->GetDataType(), y, indices)) {
        return {};
    }
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(NanMedian, OP_INPUT(self), OP_OUTPUT(y, indices), OP_ATTR(dim));
    OP_CHECK(ret == ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "NanMedian ADD_TO_LAUNCHER_LIST_AICORE failed."),
             return {});
    return std::tie(y, indices);
}
} // namespace l0op

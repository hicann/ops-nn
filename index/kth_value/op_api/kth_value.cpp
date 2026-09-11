/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kth_value.h"

#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"

#include <initializer_list>

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(KthValue);

static const std::initializer_list<op::DataType> KTH_VALUE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT,  op::DataType::DT_INT16, op::DataType::DT_INT8,
    op::DataType::DT_UINT8,   op::DataType::DT_INT32,  op::DataType::DT_INT64, op::DataType::DT_BF16,
    op::DataType::DT_UINT32,  op::DataType::DT_UINT16, op::DataType::DT_UINT64};

static constexpr int64_t NON_LAST_SMALL_AXIS_MIN = 2;
static constexpr int64_t NON_LAST_SMALL_AXIS_MAX = 2048;

static bool IsLastAxisOrSupportedNonLastAxis(int64_t rank, int64_t normDim, int64_t axisLen)
{
    if (normDim == rank - 1) {
        return true;
    }
    return axisLen >= NON_LAST_SMALL_AXIS_MIN && axisLen <= NON_LAST_SMALL_AXIS_MAX;
}

static bool CheckParams(const aclTensor* self, int64_t k, int64_t dim)
{
    OP_CHECK_NULL(self, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(self, KTH_VALUE_DTYPE_SUPPORT_LIST, return false);
    auto rank = static_cast<int64_t>(self->GetViewShape().GetDimNum());
    if (rank <= 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "kth_value only supports tensor with rank > 0.");
        return false;
    }
    if (dim < -rank || dim >= rank) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "dim should be in range [%ld, %ld].", -rank, rank - 1);
        return false;
    }
    auto normDim = dim < 0 ? dim + rank : dim;
    auto axisLen = static_cast<int64_t>(self->GetViewShape().GetDim(static_cast<size_t>(normDim)));
    if (axisLen <= 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "kth_value axis length must be greater than 0.");
        return false;
    }
    if (!IsLastAxisOrSupportedNonLastAxis(rank, normDim, axisLen)) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "Dim must be the last axis or a supported non-last axis with axis size in [2, 2048]. Current dim is %ld.",
            dim);
        return false;
    }
    if (k < 1 || k > axisLen) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "k should be in range [1, %ld].", axisLen);
        return false;
    }
    return true;
}

static op::Shape MakeKthShape(const op::Shape& shape, int64_t dim)
{
    auto outShape = shape;
    outShape.SetDim(static_cast<size_t>(dim), 1);
    return outShape;
}

static std::tuple<aclTensor*, aclTensor*> KthValueAiCore(const aclTensor* self, int64_t k, int64_t dim,
                                                         aclOpExecutor* executor)
{
    auto rank = static_cast<int64_t>(self->GetViewShape().GetDimNum());
    auto normDim = dim < 0 ? dim + rank : dim;
    auto outShape = MakeKthShape(self->GetViewShape(), normDim);
    auto values = executor->AllocTensor(outShape, self->GetDataType(), op::Format::FORMAT_ND);
    auto indices = executor->AllocTensor(outShape, op::DataType::DT_INT64, op::Format::FORMAT_ND);
    OP_CHECK_NULL(values, return {});
    OP_CHECK_NULL(indices, return {});

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(KthValue, OP_INPUT(self), OP_OUTPUT(values, indices), OP_ATTR(k, dim));
    OP_CHECK(ret == ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "KthValue ADD_TO_LAUNCHER_LIST_AICORE failed."),
             return {});
    return std::tie(values, indices);
}

const std::tuple<aclTensor*, aclTensor*> KthValue(const aclTensor* self, int64_t k, int64_t dim,
                                                  aclOpExecutor* executor)
{
    L0_DFX(KthValue, self, k, dim);
    if (!CheckParams(self, k, dim)) {
        return {};
    }
    return KthValueAiCore(self, k, dim, executor);
}
} // namespace l0op

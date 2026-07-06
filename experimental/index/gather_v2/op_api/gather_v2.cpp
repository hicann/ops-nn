/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gather_v2_l0_experimental.h"

#include "opdev/data_type_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(GatherV2);

static const std::initializer_list<DataType> AICORE_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT,  DataType::DT_FLOAT16, DataType::DT_INT8,  DataType::DT_INT16,
    DataType::DT_INT32,  DataType::DT_INT64,   DataType::DT_UINT8, DataType::DT_UINT16,
    DataType::DT_UINT32, DataType::DT_UINT64,  DataType::DT_BOOL,  DataType::DT_DOUBLE};

static inline bool IsAiCoreSupport(const aclTensor* self)
{
    return self->GetViewShape().GetDimNum() != 0 && CheckType(self->GetDataType(), AICORE_DTYPE_SUPPORT_LIST);
}

static inline const aclTensor* MakeAxisTensor(int64_t axis, aclOpExecutor* executor)
{
    auto axisTensor = executor->ConvertToTensor(&axis, 1, DataType::DT_INT64);
    OP_CHECK(axisTensor != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "axisTensor is nullptr."), return nullptr);
    return axisTensor;
}

static inline const aclTensor* GatherV2AiCore(const aclTensor* self, int64_t axis, const aclTensor* indices,
                                              const aclTensor* out, aclOpExecutor* executor, int64_t batchDims,
                                              bool negativeIndexSupport)
{
    L0_DFX(GatherV2AiCore, self, axis, indices, batchDims, negativeIndexSupport);
    auto axisTensor = MakeAxisTensor(axis, executor);
    OP_CHECK(axisTensor != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "axisTensor is nullptr."), return nullptr);
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(GatherV2, OP_INPUT(self, indices, axisTensor), OP_OUTPUT(out),
                                           OP_ATTR(batchDims, negativeIndexSupport));
    OP_CHECK(ret == ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "GatherV2AiCore add to launcher failed."),
             return nullptr);
    return out;
}

const aclTensor* GatherV2(const aclTensor* self, int64_t axis, const aclTensor* indices, const op::Shape& outShape,
                          aclOpExecutor* executor, int64_t batchDims, bool negativeIndexSupport)
{
    auto out = executor->AllocTensor(outShape, self->GetDataType());
    OP_CHECK(out != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "out is nullptr."), return nullptr);
    return GatherV2(self, axis, indices, out, executor, batchDims, negativeIndexSupport);
}

const aclTensor* GatherV2(const aclTensor* self, int64_t axis, const aclTensor* indices, const aclTensor* out,
                          aclOpExecutor* executor, int64_t batchDims, bool negativeIndexSupport)
{
    L0_DFX(GatherV2, self, axis, indices, batchDims, negativeIndexSupport);
    OP_CHECK(out != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "out is nullptr."), return nullptr);
    OP_CHECK(IsAiCoreSupport(self),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "GatherV2 only supports non-scalar AICore inputs in this experimental implementation."),
             return nullptr);
    return GatherV2AiCore(self, axis, indices, out, executor, batchDims, negativeIndexSupport);
}
} // namespace l0op

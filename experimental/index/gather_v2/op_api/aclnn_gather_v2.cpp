/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_gather_v2_experimental.h"

#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "gather_v2_l0_experimental.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_def.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace {
static constexpr size_t MAX_DIM_LEN = 8;

static const std::initializer_list<DataType> SELF_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_INT8, DataType::DT_INT16, DataType::DT_INT32,
    DataType::DT_INT64, DataType::DT_UINT8, DataType::DT_UINT16, DataType::DT_UINT32, DataType::DT_UINT64,
    DataType::DT_BOOL, DataType::DT_DOUBLE};

static const std::initializer_list<DataType> INDEX_DTYPE_SUPPORT_LIST = {DataType::DT_INT32, DataType::DT_INT64};

static inline bool CheckNotNull(
    const aclTensor* self, const aclTensor* index, const aclTensor* out, const uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(index, return false);
    OP_CHECK_NULL(out, return false);
    OP_CHECK_NULL(workspaceSize, return false);
    OP_CHECK_NULL(executor, return false);
    return true;
}

static inline bool CheckDtypeValid(const aclTensor* self, const aclTensor* index, const aclTensor* out)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(self, SELF_DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(index, INDEX_DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_MATCH(out, self->GetDataType(), return false);
    return true;
}

static inline bool CheckShape(const aclTensor* self, const aclTensor* index, const aclTensor* out, int64_t dim)
{
    OP_CHECK_MAX_DIM(self, MAX_DIM_LEN, return false);
    OP_CHECK_MAX_DIM(index, MAX_DIM_LEN, return false);
    OP_CHECK_MAX_DIM(out, MAX_DIM_LEN, return false);

    auto selfDimNum = static_cast<int64_t>(self->GetViewShape().GetDimNum());
    if (selfDimNum == 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "self tensor must be non-scalar for the Ascend C experimental implementation.");
        return false;
    }
    if (dim < -selfDimNum || dim >= selfDimNum) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "dim %ld is out of range [%ld, %ld).", dim, -selfDimNum, selfDimNum);
        return false;
    }
    if (dim < 0) {
        dim += selfDimNum;
    }

    Shape expectedShape;
    for (int64_t i = 0; i < dim; ++i) {
        expectedShape.AppendDim(self->GetViewShape().GetDim(i));
    }
    if (index->GetViewShape().GetDimNum() == 0) {
        expectedShape.AppendDim(1);
    } else {
        for (size_t i = 0; i < index->GetViewShape().GetDimNum(); ++i) {
            expectedShape.AppendDim(index->GetViewShape().GetDim(i));
        }
    }
    for (size_t i = static_cast<size_t>(dim + 1); i < self->GetViewShape().GetDimNum(); ++i) {
        expectedShape.AppendDim(self->GetViewShape().GetDim(i));
    }
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(out, expectedShape, return false);
    return true;
}

static aclnnStatus CheckParams(const aclTensor* self, int64_t dim, const aclTensor* index, const aclTensor* out,
    const uint64_t* workspaceSize, aclOpExecutor** executor)
{
    CHECK_RET(CheckNotNull(self, index, out, workspaceSize, executor), ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckDtypeValid(self, index, out), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckShape(self, index, out, dim), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}
} // namespace

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnGatherV2GetWorkspaceSize(
    const aclTensor* self, int64_t dim, const aclTensor* index, aclTensor* out, uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnGatherV2, DFX_IN(self, dim, index), DFX_OUT(out));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto ret = CheckParams(self, dim, index, out, workspaceSize, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (self->IsEmpty() || index->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    auto contiguousSelf = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(contiguousSelf != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto contiguousIndex = l0op::Contiguous(index, uniqueExecutor.get());
    CHECK_RET(contiguousIndex != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto result = l0op::GatherV2(contiguousSelf, dim, contiguousIndex, out, uniqueExecutor.get());
    CHECK_RET(result != nullptr, ACLNN_ERR_INNER_NULLPTR);
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnGatherV2(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnGatherV2);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif

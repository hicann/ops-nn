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
 * \file aclnn_scatter_update.cpp
 * \brief
 */

#include "aclnn_scatter_update.h"
#include "level0/broadcast_to.h"
#include "aclnn_kernels/contiguous.h"
#include "scatter.h"
#include "level0/squeeze.h"
#include "level0/unsqueeze.h"

#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/make_op_executor.h"
#include "opdev/platform.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST_DATA = {
    op::DataType::DT_INT8, op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT, op::DataType::DT_BF16,
    op::DataType::DT_INT32};

static const std::initializer_list<op::DataType> ASCEND910_95_DTYPE_SUPPORT_LIST_DATA = {
    op::DataType::DT_INT8, op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT,
    op::DataType::DT_BF16, op::DataType::DT_UINT8,   op::DataType::DT_INT32};

static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST_INDICES = {
    op::DataType::DT_INT64, op::DataType::DT_INT32};

static const int64_t TWO_DIM_SIZE = 2;
static const int64_t ONE_DIM_SIZE = 1;
static const int64_t ZERO_DIM_SIZE = 0;
static const int64_t BATCH_DIM = 0;

static inline const std::initializer_list<op::DataType>& GetDtypeSupportListBySocVersion()
{
    auto socVersion = GetCurrentPlatformInfo().GetSocVersion();
    if (socVersion == SocVersion::ASCEND910_95) {
        return ASCEND910_95_DTYPE_SUPPORT_LIST_DATA;
    } else {
        return DTYPE_SUPPORT_LIST_DATA;
    }
}

static inline bool CheckSocVersionIsSupportBf16(void)
{
    return GetCurrentPlatformInfo().GetSocVersion() >= SocVersion::ASCEND910B &&
           GetCurrentPlatformInfo().GetSocVersion() <= SocVersion::ASCEND910E;
}

static inline bool CheckNotNull(aclTensor* data, const aclTensor* indices, const aclTensor* updates)
{
    OP_CHECK_NULL(data, return false);
    OP_CHECK_NULL(indices, return false);
    OP_CHECK_NULL(updates, return false);
    return true;
}

static bool CheckDtypeValid(aclTensor* data, const aclTensor* indices, const aclTensor* updates)
{
    const std::initializer_list<op::DataType> dtypeSupportList = GetDtypeSupportListBySocVersion();
    OP_CHECK_DTYPE_NOT_SUPPORT(data, dtypeSupportList, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(indices, DTYPE_SUPPORT_LIST_INDICES, return false);

    OP_CHECK_DTYPE_NOT_MATCH(updates, data->GetDataType(), return false);
    bool bf16flag = CheckSocVersionIsSupportBf16();
    auto socVersion = GetCurrentPlatformInfo().GetSocVersion();
    if (!bf16flag && data->GetDataType() == op::DataType::DT_BF16) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "data dtype %s is unsupported by the current SOC version [%s].",
            op::ToString(data->GetDataType()).GetString(), op::ToString(socVersion).GetString());
        return false;
    }
    return true;
}

static bool CheckShape(aclTensor* data, const aclTensor* indices, const aclTensor* updates)
{
    if (data->IsEmpty() || updates->IsEmpty() || indices->IsEmpty()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "input tensors can not be empty tensor");
        return false;
    }

    if (data->GetViewShape().GetDimNum() != updates->GetViewShape().GetDimNum()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "dimension of data and updates should equal");
        return false;
    }

    auto indicesDimSize = indices->GetViewShape().GetDimNum();
    if (indicesDimSize == ZERO_DIM_SIZE && updates->GetViewShape().GetDim(BATCH_DIM) != 1) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "when the dimension of indices are 0, batch axis of updates should be 1");
        return false;
    }

    if (indicesDimSize != ONE_DIM_SIZE && indicesDimSize != TWO_DIM_SIZE && indicesDimSize != ZERO_DIM_SIZE) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "dimension of indices should be 0, 1 or 2");
        return false;
    }
    return true;
}

static aclnnStatus CheckParams(aclTensor* data, const aclTensor* indices, const aclTensor* updates)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(data, indices, updates), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查参数的数据类型是否符合预期
    CHECK_RET(CheckDtypeValid(data, indices, updates), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查输入tensor的shape
    CHECK_RET(CheckShape(data, indices, updates), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnInplaceScatterUpdateGetWorkspaceSize(
    aclTensor* data, const aclTensor* indices, const aclTensor* updates, int64_t axis, uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnInplaceScatterUpdate, DFX_IN(data, indices, updates, axis), DFX_OUT(data));

    // 参数检查
    auto ret = CheckParams(data, indices, updates);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    // 创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 将输入data转换成连续的tensor
    auto dataContiguous = l0op::Contiguous(data, uniqueExecutor.get());
    CHECK_RET(dataContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    // 将输入indices转换成连续的tensor
    auto indicesContiguous = l0op::Contiguous(indices, uniqueExecutor.get());
    CHECK_RET(indicesContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    // 将输入updates转换成连续的tensor
    auto updatesContiguous = l0op::Contiguous(updates, uniqueExecutor.get());
    CHECK_RET(updatesContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 执行L0算子
    auto scatterUpdateRes =
        l0op::Scatter(dataContiguous, indicesContiguous, updatesContiguous, axis, uniqueExecutor.get());
    CHECK_RET(scatterUpdateRes != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 将计算结果拷贝到输出data上
    auto viewCopyResult = l0op::ViewCopy(scatterUpdateRes, data, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnInplaceScatterUpdate(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnInplaceScatterUpdate);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif

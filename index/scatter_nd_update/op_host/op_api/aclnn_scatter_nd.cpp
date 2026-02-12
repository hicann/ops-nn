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
 * \file aclnn_scatter_nd.cpp
 * \brief
 */

#include "aclnn_scatter_nd.h"
#include "level0/broadcast_to.h"
#include "aclnn_kernels/contiguous.h"
#include "scatter_nd_update.h"
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

static constexpr size_t MIN_INPUT_DIM_NUM = 1;
static constexpr size_t MAX_INPUT_DIM_MUM = 8;

static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST_DATA = {
    op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT, op::DataType::DT_BOOL, op::DataType::DT_BF16
};

static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST_INDICES = { op::DataType::DT_INT64, op::DataType::DT_INT32 };

static bool CheckNotNull(const aclTensor *data, const aclTensor *indices, const aclTensor *updates, const aclTensor *out)
{
    OP_CHECK_NULL(data, return false);
    OP_CHECK_NULL(indices, return false);
    OP_CHECK_NULL(updates, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static bool CheckDtypeValid(const aclTensor *data, const aclTensor *indices, const aclTensor *updates, const aclTensor *out)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(data, DTYPE_SUPPORT_LIST_DATA, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(indices, DTYPE_SUPPORT_LIST_INDICES, return false);
    OP_CHECK_DTYPE_NOT_MATCH(updates, data->GetDataType(), return false);
    OP_CHECK_DTYPE_NOT_MATCH(out, data->GetDataType(), return false);
    if (data->GetDataType() == op::DataType::DT_BF16 &&
        GetCurrentPlatformInfo().GetSocVersion() != SocVersion::ASCEND910B &&
        GetCurrentPlatformInfo().GetSocVersion() != SocVersion::ASCEND910_93 &&
        GetCurrentPlatformInfo().GetSocVersion() != SocVersion::ASCEND950) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "data dtype %s is unsupported by the current SOC version.",
                op::ToString(data->GetDataType()).GetString());
        return false;
    }
    return true;
}

static bool CheckShape(const aclTensor *data, const aclTensor *indices, const aclTensor *updates, const aclTensor *out)
{
    OP_CHECK_MIN_DIM(data, MIN_INPUT_DIM_NUM, return false);
    OP_CHECK_MIN_DIM(indices, MIN_INPUT_DIM_NUM, return false);
    OP_CHECK_MIN_DIM(updates, MIN_INPUT_DIM_NUM, return false);
    OP_CHECK_MAX_DIM(data, MAX_INPUT_DIM_MUM, return false);
    OP_CHECK_MAX_DIM(indices, MAX_INPUT_DIM_MUM, return false);
    OP_CHECK_MAX_DIM(updates, MAX_INPUT_DIM_MUM, return false);
    OP_CHECK_SHAPE_NOT_EQUAL(data, out, return false);

    auto indicesRank = indices->GetViewShape().GetDimNum();
    auto lastIndicesShapeDim = indices->GetViewShape().GetDim(indicesRank-1);
    auto dataRank = data->GetViewShape().GetDimNum();
    auto updatesRank = updates->GetViewShape().GetDimNum();
    // q + r - indices.shape[-1] - 1
    if (dataRank + indicesRank - lastIndicesShapeDim -1 != updatesRank) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "the rank of data,indices should match the shape of updates");
        return false;
    }

    // indices.shape[-1] <= len(data.shape)
    if (indices->GetViewShape().GetDim(indicesRank-1) > static_cast<int64_t>(data->GetViewShape().GetDimNum())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "the rank of data should bigger than indices");
        return false;
    }

    // updates.shape == indices.shape[:-1] + data.shape[indices.shape[-1] :]
    for (int i = 0;i < static_cast<int>(indicesRank) - 1;++i) {
        if (updates->GetViewShape().GetDim(i) != indices->GetViewShape().GetDim(i)) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "the shape of data, indices and updates are not matched");
            return false;
        }
    }
    for (int i = 0;i < static_cast<int>(dataRank) - lastIndicesShapeDim;++i) {
        if (data->GetViewShape().GetDim(i+lastIndicesShapeDim) != updates->GetViewShape().GetDim(i+indicesRank-1)) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "the shape of data, indices and updates are not matched");
            return false;
        }
    }

    return true;
}

static aclnnStatus CheckParams(const aclTensor *data, const aclTensor *indices, const aclTensor *updates, const aclTensor *out)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(data, indices, updates, out), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查参数的数据类型是否符合预期
    CHECK_RET(CheckDtypeValid(data, indices, updates, out), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查输入tensor的shape
    CHECK_RET(CheckShape(data, indices, updates, out), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnScatterNdGetWorkspaceSize(const aclTensor *data, const aclTensor *indices, const aclTensor *updates,
                                           aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnScatterNd, DFX_IN(data, indices, updates), DFX_OUT(out));
    // 参数检查
    auto ret = CheckParams(data, indices, updates, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    // 创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    if (data->IsEmpty() || indices->IsEmpty() || updates->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 拷贝data
    auto copyResult = l0op::ViewCopy(data, out, uniqueExecutor.get());
    CHECK_RET(copyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 将输入转换成连续的tensor
    auto dataContiguous = l0op::Contiguous(out, uniqueExecutor.get());
    CHECK_RET(dataContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    // 将输入indices转换成连续的tensor
    auto indicesContiguous = l0op::Contiguous(indices, uniqueExecutor.get());
    CHECK_RET(indicesContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    // 将输入updates转换成连续的tensor
    auto updatesContiguous = l0op::Contiguous(updates, uniqueExecutor.get());
    CHECK_RET(updatesContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 执行L0算子
    auto updateRet = l0op::ScatterNdUpdate(dataContiguous, indicesContiguous, updatesContiguous, false, uniqueExecutor.get());
    CHECK_RET(updateRet != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 将计算结果拷贝到输出输出上
    auto viewCopyResult = l0op::ViewCopy(updateRet, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnScatterNd(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnScatterNd);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
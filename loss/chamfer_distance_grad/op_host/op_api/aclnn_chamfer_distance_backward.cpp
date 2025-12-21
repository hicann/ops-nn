/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn_chamfer_distance_backward.h"
#include "chamfer_distance_backward.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/transdata.h"
#include "aclnn_kernels/reshape.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn/aclnn_base.h"
#include "level0/squeeze.h"
#include "level0/unsqueeze.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

static bool CheckNotNull(
    const aclTensor* xyz1, const aclTensor* xyz2, const aclTensor* idx1, const aclTensor* idx2,
    const aclTensor* gradDist1, const aclTensor* gradDist2, aclTensor* gradXyz1, aclTensor* gradXyz2)
{
    OP_CHECK_NULL(xyz1, return false);
    OP_CHECK_NULL(xyz2, return false);
    OP_CHECK_NULL(idx1, return false);
    OP_CHECK_NULL(idx2, return false);
    OP_CHECK_NULL(gradDist1, return false);
    OP_CHECK_NULL(gradDist2, return false);
    OP_CHECK_NULL(gradXyz1, return false);
    OP_CHECK_NULL(gradXyz2, return false);
    return true;
}

static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16};

static const std::initializer_list<op::DataType> IDX_DTYPE_SUPPORT_LIST = {op::DataType::DT_INT32};

static bool CheckDtypeValid(
    const aclTensor* xyz1, const aclTensor* xyz2, const aclTensor* idx1, const aclTensor* idx2,
    const aclTensor* gradDist1, const aclTensor* gradDist2, const aclTensor* gradXyz1, const aclTensor* gradXyz2)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(xyz1, DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(xyz2, DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(idx1, IDX_DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(idx2, IDX_DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(gradDist1, DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(gradDist2, DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(gradXyz1, DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(gradXyz2, DTYPE_SUPPORT_LIST, return false);

    return true;
}

static aclnnStatus CheckParams(
    const aclTensor* xyz1, const aclTensor* xyz2, const aclTensor* idx1, const aclTensor* idx2,
    const aclTensor* gradDist1, const aclTensor* gradDist2, aclTensor* gradXyz1, aclTensor* gradXyz2)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(xyz1, xyz2, idx1, idx2, gradDist1, gradDist2, gradXyz1, gradXyz2), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查数据类型是否正确
    CHECK_RET(
        CheckDtypeValid(xyz1, xyz2, idx1, idx2, gradDist1, gradDist2, gradXyz1, gradXyz2), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnChamferDistanceBackwardGetWorkspaceSize(
    const aclTensor* xyz1, const aclTensor* xyz2, const aclTensor* idx1, const aclTensor* idx2,
    const aclTensor* gradDist1, const aclTensor* gradDist2, aclTensor* gradXyz1, aclTensor* gradXyz2,
    uint64_t* workspaceSize, aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);

    L2_DFX_PHASE_1(
        aclnnChamferDistanceBackward, DFX_IN(xyz1, xyz2, idx1, idx2, gradDist1, gradDist2),
        DFX_OUT(gradXyz1, gradXyz2));
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(xyz1, xyz2, idx1, idx2, gradDist1, gradDist2, gradXyz1, gradXyz2);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (xyz1->IsEmpty() || xyz2->IsEmpty() || idx1->IsEmpty() || idx2->IsEmpty() || gradDist1->IsEmpty() ||
        gradDist2->IsEmpty()) {
        // 根据实际支持情况补充
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    auto xyz1Contiguous = l0op::Contiguous(xyz1, uniqueExecutor.get());
    CHECK_RET(xyz1Contiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto xyz2Contiguous = l0op::Contiguous(xyz2, uniqueExecutor.get());
    CHECK_RET(xyz2Contiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto idx1Contiguous = l0op::Contiguous(idx1, uniqueExecutor.get());
    CHECK_RET(idx1Contiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto idx2Contiguous = l0op::Contiguous(idx2, uniqueExecutor.get());
    CHECK_RET(idx2Contiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto gradDist1Contiguous = l0op::Contiguous(gradDist1, uniqueExecutor.get());
    CHECK_RET(gradDist1Contiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto gradDist2Contiguous = l0op::Contiguous(gradDist2, uniqueExecutor.get());
    CHECK_RET(gradDist2Contiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto curType = xyz1Contiguous->GetDataType();
    auto xyz1Cast = l0op::Cast(xyz1Contiguous, op::DataType::DT_FLOAT, uniqueExecutor.get());
    CHECK_RET(xyz1Cast != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto xyz2Cast = l0op::Cast(xyz2Contiguous, op::DataType::DT_FLOAT, uniqueExecutor.get());
    CHECK_RET(xyz2Cast != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto gradDist1Cast = l0op::Cast(gradDist1Contiguous, op::DataType::DT_FLOAT, uniqueExecutor.get());
    CHECK_RET(gradDist1Cast != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto gradDist2Cast = l0op::Cast(gradDist2Contiguous, op::DataType::DT_FLOAT, uniqueExecutor.get());
    CHECK_RET(gradDist2Cast != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto out = l0op::ChamferDistanceGrad(
        xyz1Cast, xyz2Cast, idx1Contiguous, idx2Contiguous, gradDist1Cast, gradDist2Cast, uniqueExecutor.get());
    auto gradXyz1Contiguous = std::get<0>(out);
    auto gradXyz2Contiguous = std::get<1>(out);
    auto gradXyz1Cast = l0op::Cast(gradXyz1Contiguous, curType, uniqueExecutor.get());
    CHECK_RET(gradXyz1Cast != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto gradXyz2Cast = l0op::Cast(gradXyz2Contiguous, curType, uniqueExecutor.get());
    CHECK_RET(gradXyz2Cast != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyResult = l0op::ViewCopy(gradXyz1Cast, gradXyz1, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyResult1 = l0op::ViewCopy(gradXyz2Cast, gradXyz2, uniqueExecutor.get());
    CHECK_RET(viewCopyResult1 != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnChamferDistanceBackward(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnChamferDistanceBackward);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
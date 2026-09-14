/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*!
 * \file aclnn_unique.cpp
 * \brief
 */
#include "aclnn_unique.h"
#include "unique_common.h"
#include "index/common/op_api/unique_with_counts_and_sorting.h"
#include "index/unique_consecutive/op_api/unique_consecutive.h"

#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/shape_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "op_api/aclnn_util.h"
#include "op_api/level2_base.h"
#include "op_api/op_api_def_nn.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

static bool CheckDtypeValid(const aclTensor* self, const aclTensor* inverseOut)
{
    CHECK_RET(UniqueCommon::CheckSelfDtypeValid(self), false);
    // 检查inverseOut数据类型
    OP_CHECK_DTYPE_NOT_MATCH(inverseOut, op::DataType::DT_INT64, return false);
    return true;
}

static bool CheckShapeValid(const aclTensor* self, bool returnInverse, const aclTensor* inverseOut)
{
    // self的数据维度不能超过8
    OP_CHECK_MAX_DIM(self, MAX_SUPPORT_DIMS_NUMS, return false);
    // self与inverseOut的shape必须保持一致
    if (returnInverse) {
        OP_CHECK_SHAPE_NOT_EQUAL(self, inverseOut, return false);
    }
    return true;
}

static aclnnStatus CheckParams(const aclTensor* self, bool returnInverse, aclTensor* valueOut, aclTensor* inverseOut)
{
    // 1. 检查输入输出是否为nullptr
    CHECK_RET(CheckNotNull3Tensor(self, valueOut, inverseOut), ACLNN_ERR_PARAM_NULLPTR);
    // 2. 检查数据类型
    CHECK_RET(CheckDtypeValid(self, inverseOut), ACLNN_ERR_PARAM_INVALID);
    // 3. 检查数据Shape
    CHECK_RET(CheckShapeValid(self, returnInverse, inverseOut), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

aclnnStatus ComputeUniqueViaAicore(const aclTensor* selfContiguous, bool returnInverse, aclTensor* valueOut,
                                   aclTensor* inverseOut, aclOpExecutor* executor)
{
    constexpr int64_t NONE_N = 1000;
    constexpr bool RET_INV_UC = false;
    constexpr bool RET_CNT_UC = false;

    auto indicesType = inverseOut->GetDataType();
    UniqueCommon::SortResult sortResult;
    auto ret = UniqueCommon::FlattenAndSort(selfContiguous, indicesType, executor, sortResult);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // uniqueCons for valueOut
    aclTensor* dummyInverseOut = nullptr;
    aclTensor* dummyCountsOut = nullptr;
    if (Ops::NN::AclnnUtil::IsRegbase()) {
        dummyInverseOut = executor->AllocTensor(inverseOut->GetStorageShape(), inverseOut->GetDataType(),
                                                Format::FORMAT_ND);
        dummyCountsOut = executor->AllocTensor(selfContiguous->GetStorageShape(), inverseOut->GetDataType(),
                                               Format::FORMAT_ND);
    } else {
        dummyInverseOut = executor->AllocTensor(inverseOut->GetStorageShape(), DataType::DT_INT32, Format::FORMAT_ND);
        dummyCountsOut = executor->AllocTensor(selfContiguous->GetStorageShape(), DataType::DT_INT32,
                                               Format::FORMAT_ND);
    }
    auto uniqueConsRet = l0op::UniqueConsecutive(sortResult.sortedValues, RET_INV_UC, RET_CNT_UC, NONE_N, valueOut,
                                                 dummyInverseOut, dummyCountsOut, executor);
    CHECK_RET(uniqueConsRet == ACLNN_SUCCESS, uniqueConsRet);

    // AdjDiff for inverseOut
    if (returnInverse) {
        ret = UniqueCommon::ComputeInverseIndices(selfContiguous, sortResult.sortedValues, sortResult.sortedIndices,
                                                  inverseOut, indicesType, executor);
        CHECK_RET(ret == ACLNN_SUCCESS, ret);
    }
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnUniqueGetWorkspaceSize(const aclTensor* self, bool sorted, bool returnInverse, aclTensor* valueOut,
                                        aclTensor* inverseOut, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);

    L2_DFX_PHASE_1(aclnnUnique, DFX_IN(self, sorted, returnInverse), DFX_OUT(valueOut, inverseOut));

    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(self, returnInverse, valueOut, inverseOut);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 空tensor在kernel中支持，对标竞品根据算子实际情况补充
    if (self->IsEmpty()) {
        // 根据实际支持情况补充
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 固定写法，将输入self转换成连续的tensor
    auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    if (returnInverse) {
        auto inverseViewShape = inverseOut->GetViewShape();
        inverseOut->SetStorageShape(inverseViewShape);
        inverseOut->SetOriginalShape(inverseViewShape);
    }

    if (UniqueCommon::SupportAicore4Unique(selfContiguous, "Unique")) {
        auto opRet = ComputeUniqueViaAicore(selfContiguous, returnInverse, valueOut, inverseOut, uniqueExecutor.get());
        CHECK_RET(opRet == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    } else {
        // 调用UniqueWithCountsAndSorting算子
        auto opRet = l0op::UniqueWithCountsAndSorting(selfContiguous, sorted, returnInverse, valueOut, inverseOut,
                                                      uniqueExecutor.get());
        CHECK_RET(opRet == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    }
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor); // 需要把 uniqueExecutor持有executor转移给executor
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnUnique(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnUnique);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif

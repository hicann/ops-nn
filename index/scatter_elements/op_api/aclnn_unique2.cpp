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
 * \file aclnn_unique2.cpp
 * \brief
 */
#include "aclnn_unique2.h"
#include "unique_common.h"
#include "level0/unique_with_counts_and_sorting.h"
#include "index/unique_consecutive/op_host/op_api/unique_consecutive.h"

#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/shape_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "op_api/op_api_def_nn.h"
#include "op_api/aclnn_util.h"
#include "op_api/level2_base.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

static bool CheckDtypeValid(const aclTensor* self, const aclTensor* inverseOut, const aclTensor* countsOut)
{
    CHECK_RET(UniqueCommon::CheckSelfDtypeValid(self), false);

    // 检查inverseOut数据类型
    OP_CHECK_DTYPE_NOT_MATCH(inverseOut, op::DataType::DT_INT64, return false);

    // 检查countsOut数据类型
    OP_CHECK_DTYPE_NOT_MATCH(countsOut, op::DataType::DT_INT64, return false);

    return true;
}

static bool CheckShapeValid(const aclTensor* self, bool returnInverse, bool returnCounts, const aclTensor* valueOut,
                            const aclTensor* inverseOut, const aclTensor* countsOut)
{
    // self的数据维度不能超过8
    OP_CHECK_MAX_DIM(self, MAX_SUPPORT_DIMS_NUMS, return false);

    // self与inverseOut的shape必须保持一致
    if (returnInverse || returnCounts) {
        OP_CHECK_SHAPE_NOT_EQUAL(self, inverseOut, return false);
    }

    // valueOut与countsOut的shape必须保持一致
    if (returnCounts) {
        OP_CHECK_SHAPE_NOT_EQUAL(valueOut, countsOut, return false);
    }
    return true;
}

static aclnnStatus CheckParams(const aclTensor* self, bool returnInverse, bool returnCounts, aclTensor* valueOut,
                               aclTensor* inverseOut, aclTensor* countsOut)
{
    // 1. 检查输入输出是否为nullptr
    CHECK_RET(CheckNotNull4Tensor(self, valueOut, inverseOut, countsOut), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查数据类型
    CHECK_RET(CheckDtypeValid(self, inverseOut, countsOut), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查数据Shape
    CHECK_RET(CheckShapeValid(self, returnInverse, returnCounts, valueOut, inverseOut, countsOut),
              ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

aclnnStatus ComputeUnique2ViaAicore(const aclTensor* selfContiguous, bool returnInverse, bool returnCounts,
                                    aclTensor* valueOut, aclTensor* inverseOut, aclTensor* countsOut,
                                    aclOpExecutor* executor)
{
    constexpr int64_t NONE_N = 1000;
    constexpr bool RET_INV_UC = false;

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
        dummyCountsOut = executor->AllocTensor(countsOut->GetStorageShape(), countsOut->GetDataType(),
                                               Format::FORMAT_ND);
    } else {
        dummyInverseOut = executor->AllocTensor(inverseOut->GetStorageShape(), DataType::DT_INT32, Format::FORMAT_ND);
        dummyCountsOut = executor->AllocTensor(countsOut->GetStorageShape(), DataType::DT_INT32, Format::FORMAT_ND);
    }
    auto uniqueConsRet = l0op::UniqueConsecutive(sortResult.sortedValues, RET_INV_UC, returnCounts, NONE_N, valueOut,
                                                 dummyInverseOut, dummyCountsOut, executor);
    CHECK_RET(uniqueConsRet == ACLNN_SUCCESS, uniqueConsRet);

    if (returnCounts) {
        auto countsOutInt64 = l0op::Cast(dummyCountsOut, DataType::DT_INT64, executor);
        auto viewCopyCnt = l0op::ViewCopy(countsOutInt64, countsOut, executor);
        CHECK_RET(viewCopyCnt != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    // AdjDiff for inverseOut
    if (returnInverse) {
        ret = UniqueCommon::ComputeInverseIndices(selfContiguous, sortResult.sortedValues, sortResult.sortedIndices,
                                                  inverseOut, indicesType, executor);
        CHECK_RET(ret == ACLNN_SUCCESS, ret);
    }
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnUnique2GetWorkspaceSize(const aclTensor* self, bool sorted, bool returnInverse, bool returnCounts,
                                         aclTensor* valueOut, aclTensor* inverseOut, aclTensor* countsOut,
                                         uint64_t* workspaceSize, aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);

    L2_DFX_PHASE_1(aclnnUnique2, DFX_IN(self, sorted, returnInverse, returnCounts),
                   DFX_OUT(valueOut, inverseOut, countsOut));

    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(self, returnInverse, returnCounts, valueOut, inverseOut, countsOut);
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

    if (returnInverse || returnCounts) {
        auto inverseViewShape = inverseOut->GetViewShape();
        inverseOut->SetStorageShape(inverseViewShape);
        inverseOut->SetOriginalShape(inverseViewShape);
    }

    if (UniqueCommon::SupportAicore4Unique(selfContiguous, "Unique2")) {
        auto opRet = ComputeUnique2ViaAicore(selfContiguous, returnInverse, returnCounts, valueOut, inverseOut,
                                             countsOut, uniqueExecutor.get());
        CHECK_RET(opRet == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    } else {
        // 调用UniqueWithCountsAndSorting算子
        auto opRet = l0op::UniqueWithCountsAndSorting(selfContiguous, sorted, returnInverse, returnCounts, valueOut,
                                                      inverseOut, countsOut, uniqueExecutor.get());
        CHECK_RET(opRet == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    }

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor); // 需要把 uniqueExecutor持有executor转移给executor
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnUnique2(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnUnique2);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif

/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*!
 * \file unique_common.cpp
 * \brief Shared utilities for aclnnUnique and aclnnUnique2
 */
#include "unique_common.h"

#include "level0/adjacent_difference.h"
#include "level0/cumsum.h"
#include "level0/sort.h"
#include "index/scatter_elements_v2/op_api/scatter_elements.h"

#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/reshape.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/data_type_utils.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "op_api/aclnn_util.h"
#include "op_api/level2_base.h"

using namespace op;

namespace UniqueCommon {
// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> ASCEND910_DTYPE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_BOOL,  op::DataType::DT_UINT8,  op::DataType::DT_INT8,  op::DataType::DT_UINT16,
    op::DataType::DT_INT16, op::DataType::DT_UINT32, op::DataType::DT_INT32, op::DataType::DT_UINT64,
    op::DataType::DT_INT64, op::DataType::DT_DOUBLE, op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16};

static const std::initializer_list<op::DataType> ASCEND910B_DTYPE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_BOOL,  op::DataType::DT_UINT8,  op::DataType::DT_INT8,  op::DataType::DT_UINT16,
    op::DataType::DT_INT16, op::DataType::DT_UINT32, op::DataType::DT_INT32, op::DataType::DT_UINT64,
    op::DataType::DT_INT64, op::DataType::DT_DOUBLE, op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16,
    op::DataType::DT_BF16};

static const std::initializer_list<op::DataType> XY_DTYPE_SUPPORT_LIST_ASCEND_REGBASE = {
    op::DataType::DT_INT64,  op::DataType::DT_INT32,   op::DataType::DT_INT16,  op::DataType::DT_INT8,
    op::DataType::DT_UINT64, op::DataType::DT_UINT32,  op::DataType::DT_UINT16, op::DataType::DT_UINT8,
    op::DataType::DT_BF16,   op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT};

int64_t GetTensorElementsNum(const aclTensor* tensor)
{
    int64_t num = 1;
    auto shape = tensor->GetViewShape();
    for (size_t i = 0; i < shape.GetDimNum(); i++) {
        num *= shape.GetDim(i);
    }
    return num;
}

const aclIntArray* GetFlattenShape(const aclTensor* self, aclOpExecutor* executor)
{
    int64_t valuePerm[1] = {GetTensorElementsNum(self)};
    return executor->AllocIntArray(valuePerm, 1);
}

bool CheckSelfDtypeValid(const aclTensor* self)
{
    if (Ops::NN::AclnnUtil::IsRegbase()) {
        OP_CHECK_DTYPE_NOT_SUPPORT(self, ASCEND910B_DTYPE_DTYPE_SUPPORT_LIST, return false);
    } else {
        auto supportList = GetDtypeSupportListV1(ASCEND910B_DTYPE_DTYPE_SUPPORT_LIST,
                                                 ASCEND910_DTYPE_DTYPE_SUPPORT_LIST);
        OP_CHECK_DTYPE_NOT_SUPPORT(self, supportList, return false);
    }
    return true;
}

bool SupportAicore4Unique(const aclTensor* self, const std::string& opName)
{
    OP_CHECK(Ops::NN::AclnnUtil::IsRegbase(), OP_LOGW("Aicore %s only support arch 3510.", opName.c_str()),
             return false);
    OP_CHECK(CheckType(self->GetDataType(), XY_DTYPE_SUPPORT_LIST_ASCEND_REGBASE),
             OP_LOGW("Unsupport input dtype for aicore UniqueConsecutive."), return false);
    return true;
}

aclnnStatus FlattenAndSort(const aclTensor* selfContiguous, op::DataType indicesType, aclOpExecutor* executor,
                           SortResult& result)
{
    // 将多维输入flatten成一维
    auto selfFlatten = l0op::Reshape(selfContiguous, GetFlattenShape(selfContiguous, executor), executor);
    OP_CHECK_NULL(selfFlatten, return ACLNN_ERR_INNER_NULLPTR);

    // sort
    auto sortRes = l0op::Sort(selfFlatten, 0, false, true, indicesType, executor);
    result.sortedValues = std::get<0>(sortRes);
    OP_CHECK_NULL(result.sortedValues, return ACLNN_ERR_INNER_NULLPTR);
    result.sortedIndices = std::get<1>(sortRes);
    OP_CHECK_NULL(result.sortedIndices, return ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

aclnnStatus ComputeInverseIndices(const aclTensor* selfContiguous, const aclTensor* sortedValues,
                                  const aclTensor* sortedIndices, aclTensor* inverseOut, op::DataType indicesType,
                                  aclOpExecutor* executor)
{
    const aclTensor* dimTensor = nullptr;
    int64_t firstDimOf1DTensor = 0;
    dimTensor = executor->ConvertToTensor(&firstDimOf1DTensor, 1, DataType::DT_INT64);
    auto adjDiff = l0op::AdjacentDifference(sortedValues, indicesType, executor);
    auto sumIdx = l0op::Cumsum(adjDiff, dimTensor, executor);
    auto newData = executor->AllocTensor(sumIdx->GetViewShape(), sumIdx->GetDataType(), sumIdx->GetViewFormat());
    CHECK_RET(newData != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto inverseIdx = l0op::ScatterElements(newData, sortedIndices, sumIdx, 0, "none", executor);

    // 将一维inverse indices reshape回原始多维shape
    auto inverseIdxReshape = l0op::Reshape(inverseIdx, selfContiguous->GetViewShape(), executor);
    OP_CHECK_NULL(inverseIdxReshape, return ACLNN_ERR_INNER_NULLPTR);

    const aclTensor* viewCopyInverseIdx = nullptr;
    if (Ops::NN::AclnnUtil::IsRegbase()) {
        viewCopyInverseIdx = l0op::ViewCopy(inverseIdxReshape, inverseOut, executor);
    } else {
        auto inverseIdxInt64 = l0op::Cast(inverseIdxReshape, DataType::DT_INT64, executor);
        viewCopyInverseIdx = l0op::ViewCopy(inverseIdxInt64, inverseOut, executor);
    }
    CHECK_RET(viewCopyInverseIdx != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}
} // namespace UniqueCommon

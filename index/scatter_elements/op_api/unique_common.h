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
 * \file unique_common.h
 * \brief Shared utilities for aclnnUnique and aclnnUnique2
 */
#ifndef OP_API_INC_UNIQUE_COMMON_H_
#define OP_API_INC_UNIQUE_COMMON_H_

#include "aclnn/aclnn_base.h"
#include "opdev/common_types.h"
#include <string>

namespace UniqueCommon {
int64_t GetTensorElementsNum(const aclTensor* tensor);
const aclIntArray* GetFlattenShape(const aclTensor* self, aclOpExecutor* executor);
bool CheckSelfDtypeValid(const aclTensor* self);
bool SupportAicore4Unique(const aclTensor* self, const std::string& opName);

struct SortResult {
    const aclTensor* sortedValues;
    const aclTensor* sortedIndices;
};
aclnnStatus FlattenAndSort(const aclTensor* selfContiguous, op::DataType indicesType, aclOpExecutor* executor,
                           SortResult& result);

aclnnStatus ComputeInverseIndices(const aclTensor* selfContiguous, const aclTensor* sortedValues,
                                  const aclTensor* sortedIndices, aclTensor* inverseOut, op::DataType indicesType,
                                  aclOpExecutor* executor);
} // namespace UniqueCommon

#endif // OP_API_INC_UNIQUE_COMMON_H_

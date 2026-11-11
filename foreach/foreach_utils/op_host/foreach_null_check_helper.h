/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file foreach_null_check_helper.h
 * \brief Common null-entry checks for foreach op_api TensorList parameters
 */

#ifndef FOREACH_NULL_CHECK_HELPER_H
#define FOREACH_NULL_CHECK_HELPER_H

#include "opdev/op_log.h"
#include "opdev/common_types.h"

namespace op {

/* 检查 TensorList 中每个 aclTensor 指针非空且 shape size 可表示,
   避免 CheckDtype/CheckShape/CheckFormat 解引用空指针及 numel×elementSize 溢出导致 SIGSEGV */
inline bool CheckTensorListNotNull(const aclTensorList* tensorList, const char* name)
{
    // 防御：shape size 超过可表示范围时拒绝，避免后续 numel×elementSize 溢出导致 SIGSEGV
    constexpr int64_t MAX_TENSOR_SHAPE_SIZE = INT64_MAX / 8; // 8 = 最大元素字节数(double)
    for (uint64_t i = 0; i < tensorList->Size(); i++) {
        if ((*tensorList)[i] == nullptr) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "%s[%lu] is null.", name, i);
            return false;
        }
        int64_t shapeSize = (*tensorList)[i]->GetViewShape().GetShapeSize();
        if (shapeSize < 0 || shapeSize > MAX_TENSOR_SHAPE_SIZE) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "%s[%lu] shape size %ld exceeds representable range.", name, i,
                    (long)shapeSize);
            return false;
        }
    }
    return true;
}

} // namespace op

#endif // FOREACH_NULL_CHECK_HELPER_H

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

/* 检查 TensorList 中每个 aclTensor 指针非空, 避免 CheckDtype/CheckShape/CheckFormat 解引用空指针 */
inline bool CheckTensorListNotNull(const aclTensorList* tensorList, const char* name)
{
    for (uint64_t i = 0; i < tensorList->Size(); i++) {
        if ((*tensorList)[i] == nullptr) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "%s[%lu] is null.", name, i);
            return false;
        }
    }
    return true;
}

} // namespace op

#endif // FOREACH_NULL_CHECK_HELPER_H

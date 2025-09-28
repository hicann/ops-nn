/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file scatter_add_with_sorted.cpp
 * \brief
 */
#include "scatter_add_with_sorted.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/platform.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(ScatterAddWithSorted);

const aclTensor* ScatterAddWithSorted(
    const aclTensor* self, const aclTensor* value, const aclTensor* sorted_index, const aclTensor* pos,
    const std::string& reduction, aclOpExecutor* executor)
{
    L0_DFX(ScatterAddWithSorted, self, value, sorted_index, pos);

    auto socVersion = GetCurrentPlatformInfo().GetSocVersion();
    if (socVersion != SocVersion::ASCEND910B && socVersion != SocVersion::ASCEND910_93) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "[ScatterAddWithSorted] only support ASCEND910B and ASCEND910_93");
    }

    auto selfOut = const_cast<aclTensor*>(self);
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        ScatterAddWithSorted, OP_INPUT(selfOut, value, sorted_index, pos), OP_OUTPUT(selfOut), OP_ATTR(reduction));
    OP_CHECK(
        ret == ACLNN_SUCCESS,
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "ScatterAddWithSortedAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."),
        return nullptr);
    return selfOut;
}
} // namespace l0op

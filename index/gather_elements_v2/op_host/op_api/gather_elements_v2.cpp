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
 * \file gather_elements_v2.cpp
 * \brief
 */
#include "gather_elements_v2.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/platform.h"

using namespace op;

static const std::initializer_list<op::DataType> ASCEND910B_AICORE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_INT32, op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

namespace l0op {
OP_TYPE_REGISTER(GatherElementsV2);

const aclTensor* GatherElementsV2(
    const aclTensor* self, const aclTensor* index, const int64_t dim, aclOpExecutor* executor)
{
    L0_DFX(GatherElementsV2, self, index, dim);
    CHECK_RET(self != nullptr, nullptr);
    CHECK_RET(index != nullptr, nullptr);

    auto socVersion = GetCurrentPlatformInfo().GetSocVersion();
    if (socVersion != SocVersion::ASCEND910B && socVersion != SocVersion::ASCEND910_93) {
        OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "[GatherElementsV2] only support ASCEND910B and ASCEND910_93");
        return nullptr;
    }

    if (!CheckType(self->GetDataType(), ASCEND910B_AICORE_DTYPE_SUPPORT_LIST)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "[GatherElementsV2] indices must be in int32, float32, float16, bfloat16");
        return nullptr;
    }

    if (!(index->GetDataType() == op::DataType::DT_INT32)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "[GatherElementsV2] index must be int32");
        return nullptr;
    }

    auto out = executor->AllocTensor(index->GetViewShape(), self->GetDataType());
    auto ret = ACL_SUCCESS;

    ret = ADD_TO_LAUNCHER_LIST_AICORE(GatherElementsV2, OP_INPUT(self, index), OP_OUTPUT(out), OP_ATTR(dim));
    OP_CHECK(
        ret == ACLNN_SUCCESS,
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "GatherElementsV2 AiCore ADD_TO_LAUNCHER_LIST_AICORE failed."),
        return nullptr);

    return out;
}
} // namespace l0op

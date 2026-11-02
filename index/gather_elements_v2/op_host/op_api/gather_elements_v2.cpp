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
 * \file gather_elements_v2.cpp
 * \brief
 */
#include "gather_elements_v2.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/platform.h"
#include "opdev/format_utils.h"
#include "base/err_msg.h"

using namespace op;

static const std::initializer_list<op::DataType> ASCEND910B_AICORE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_INT32, op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

namespace l0op {
OP_TYPE_REGISTER(GatherElementsV2);

const aclTensor* GatherElementsV2(const aclTensor* self, const aclTensor* index, const int64_t dim,
                                  aclOpExecutor* executor)
{
    L0_DFX(GatherElementsV2, self, index, dim);
    if (self == nullptr || index == nullptr) {
        const std::vector<const char*> keys = {"functionName", "paraName"};
        const std::vector<const char*> values = {"GatherElementsV2", self == nullptr ? "self" : "index"};
        REPORT_PREDEFINED_ERR_MSG("EZ1011", keys, values);
        return nullptr;
    }

    auto socVersion = GetCurrentPlatformInfo().GetSocVersion();
    if (socVersion != SocVersion::ASCEND910B && socVersion != SocVersion::ASCEND910_93) {
        const std::string reason = "kernel=GatherElementsV2, actual SoC=" +
                                   std::to_string(static_cast<int32_t>(socVersion)) +
                                   ", supported SoCs=[ASCEND910B, ASCEND910_93], self dtype=" +
                                   op::ToString(self->GetDataType()).GetString() +
                                   ", format=" + op::ToString(self->GetStorageFormat()).GetString() +
                                   "; no kernel selection is available on this SoC; use a supported SoC";
        OP_LOGE_WITHOUT_REPORT(ACLNN_ERR_RUNTIME_ERROR, "%s", reason.c_str());
        const std::vector<const char*> keys = {"opName", "reason"};
        const std::vector<const char*> values = {"GatherElementsV2", reason.c_str()};
        REPORT_PREDEFINED_ERR_MSG("EZ1009", keys, values);
        return nullptr;
    }

    if (!CheckType(self->GetDataType(), ASCEND910B_AICORE_DTYPE_SUPPORT_LIST)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "[GatherElementsV2] self must be in int32, float32, float16, bfloat16");
        return nullptr;
    }

    if (!(index->GetDataType() == op::DataType::DT_INT32)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "[GatherElementsV2] index must be int32");
        return nullptr;
    }

    auto out = executor->AllocTensor(index->GetViewShape(), self->GetDataType());
    if (out == nullptr) {
        OP_LOGE_WITHOUT_REPORT(ACLNN_ERR_INNER_NULLPTR, "GatherElementsV2 AllocTensor returned nullptr");
        return nullptr;
    }
    auto ret = ACL_SUCCESS;

    ret = ADD_TO_LAUNCHER_LIST_AICORE(GatherElementsV2, OP_INPUT(self, index), OP_OUTPUT(out), OP_ATTR(dim));
    OP_CHECK(ret == ACLNN_SUCCESS,
             OP_LOGE_WITHOUT_REPORT(ret, "GatherElementsV2 ADD_TO_LAUNCHER_LIST_AICORE failed, ret=%d", ret),
             return nullptr);

    return out;
}
} // namespace l0op

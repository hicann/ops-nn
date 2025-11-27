/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/

#include "tf_scatter_add.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/op_log.h"
#include "opdev/op_executor.h"
#include "opdev/make_op_executor.h"
#include "opdev/shape_utils.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(ScatterAdd);

// AiCore支持的ScatterAdd类型
static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_INT32};

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST_910_95 = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_INT32, op::DataType::DT_INT8,
    op::DataType::DT_UINT8};

inline static bool IsAiCoreSupport(const aclTensor* varRef)
{
    if (op::GetCurrentPlatformInfo().GetSocVersion() == op::SocVersion::ASCEND910_95) {
        return CheckType(varRef->GetDataType(), AICORE_DTYPE_SUPPORT_LIST_910_95);
    } else {
        return CheckType(varRef->GetDataType(), AICORE_DTYPE_SUPPORT_LIST);
    }
}
// AiCore的执行逻辑
const aclTensor* ScatterAddAiCore(
    const aclTensor* varRef, const aclTensor* indices, const aclTensor* updates, bool use_locking,
    aclOpExecutor* executor)
{
    L0_DFX(ScatterAddAiCore, varRef, indices, updates, use_locking);
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        ScatterAdd, OP_INPUT(varRef, indices, updates), OP_OUTPUT(varRef), OP_ATTR(use_locking));
    CHECK_RET(ret == ACLNN_SUCCESS, nullptr);
    return varRef;
}

// AiCPU的执行逻辑
const aclTensor* ScatterAddAiCPU(
    const aclTensor* varRef, const aclTensor* indices, const aclTensor* updates, bool use_locking,
    aclOpExecutor* executor)
{
    L0_DFX(ScatterAddAiCPU, varRef, indices, updates, use_locking);

    static internal::AicpuTaskSpace space("ScatterAdd", ge::DEPEND_IN_SHAPE, true);
    space.SetRef(0);
    auto ret = ADD_TO_LAUNCHER_LIST_AICPU(
        ScatterAdd, OP_ATTR_NAMES({"Tindices", "T", "use_locking"}), OP_INPUT(varRef, indices, updates),
        OP_OUTPUT(varRef), OP_ATTR(indices->GetDataType(), updates->GetDataType(), use_locking));
    CHECK_RET(ret == ACLNN_SUCCESS, nullptr);
    return varRef;
}

const aclTensor* ScatterAdd(
    const aclTensor* varRef, const aclTensor* indices, const aclTensor* updates, bool use_locking,
    aclOpExecutor* executor)
{
    if (IsAiCoreSupport(varRef)) {
        return ScatterAddAiCore(varRef, indices, updates, use_locking, executor);
    } else {
        return ScatterAddAiCPU(varRef, indices, updates, use_locking, executor);
    }
}

} // namespace l0op
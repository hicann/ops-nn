/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "opdev/op_executor.h"
#include "opdev/aicpu/aicpu_task.h"

aclnnStatus CreatAiCoreKernelLauncher([[maybe_unused]] const char *l0Name, uint32_t opType,
    aclOpExecutor *executor, op::OpArgContext *args)
{
    return ACLNN_SUCCESS;
}

namespace op::internal {
aclnnStatus CreatAicpuKernelLauncher(uint32_t opType, op::internal::AicpuTaskSpace &space,
    aclOpExecutor *executor, const op::FVector<std::string> &attrNames, op::OpArgContext *args)
{
    return ACLNN_SUCCESS;
}
}
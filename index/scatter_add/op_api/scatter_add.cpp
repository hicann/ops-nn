/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "scatter_add.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/op_log.h"
#include "opdev/op_executor.h"
#include "opdev/make_op_executor.h"
#include "opdev/shape_utils.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(ScatterAddWithAxis);

// AICPU算子kernel
static const aclTensor *ScatterAddAiCPU(const aclTensor *self, int64_t dim, const aclTensor *index,
                                        const aclTensor *src, aclOpExecutor *executor)
{
    // 使用框架宏ADD_TO_LAUNCHER_LIST，将AiCPU ScatterAddWithAxis
    // ScatterAddWithAxis, dim, src是算子的输入，out是算子的输出
    L0_DFX(ScatterAddAiCPU, self, dim, index, src);

    static internal::AicpuTaskSpace space("ScatterAddWithAxis");
    auto ret = ADD_TO_LAUNCHER_LIST_AICPU(ScatterAddWithAxis, OP_ATTR_NAMES({"axis"}), OP_INPUT(self, index, src),
                                          OP_OUTPUT(self), OP_ATTR(dim));
    CHECK_RET(ret == ACLNN_SUCCESS, nullptr);

    return self;
}

// 只支持 AICPU
const aclTensor *ScatterAddWithAxis(const aclTensor *self, int64_t dim, const aclTensor *index,
                                    const aclTensor *src, aclOpExecutor *executor)
{
    return ScatterAddAiCPU(self, dim, index, src, executor);
}
}
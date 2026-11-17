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
 * \file chunk_loss.cpp
 * \brief ChunkLoss Level0 算子接口实现
 *
 * 创建 GE 节点并设置属性后加入执行器。
 */
#include "chunk_loss.h"

#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"

namespace l0op {
OP_TYPE_REGISTER(ChunkLoss);

const aclTensor* ChunkLoss(const aclTensor* hidden, const aclTensor* weight, const aclTensor* labels, float alpha,
                           int64_t ignoreIndex, aclTensor* loss, aclTensor* maxOut, aclTensor* sumOut, aclTensor* dH,
                           aclOpExecutor* executor)
{
    L0_DFX(ChunkLoss, hidden, weight, labels, alpha, ignoreIndex, loss, maxOut, sumOut, dH);
    ADD_TO_LAUNCHER_LIST_AICORE(ChunkLoss, OP_INPUT(hidden, weight, labels), OP_OUTPUT(loss, maxOut, sumOut, dH),
                                OP_ATTR(alpha, ignoreIndex));
    return loss;
}
} // namespace l0op

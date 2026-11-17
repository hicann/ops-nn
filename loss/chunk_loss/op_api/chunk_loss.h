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
 * \file chunk_loss.h
 * \brief ChunkLoss Level0 算子接口声明
 *
 * 供 aclnn 层调用的 l0op::ChunkLoss。
 */
#ifndef OP_API_INC_LEVEL0_CHUNK_LOSS_H_
#define OP_API_INC_LEVEL0_CHUNK_LOSS_H_

#include "opdev/op_executor.h"

namespace l0op {
const aclTensor* ChunkLoss(const aclTensor* hidden, const aclTensor* weight, const aclTensor* labels, float alpha,
                           int64_t ignoreIndex, aclTensor* loss, aclTensor* maxOut, aclTensor* sumOut, aclTensor* dH,
                           aclOpExecutor* executor);
}

#endif

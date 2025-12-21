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
 * \file softmax_cross_entropy_with_logits.h
 * \brief
 */
#ifndef OP_API_INC_LEVEL0_SOFTMAX_CROSS_ENTROPY_WITH_H_
#define OP_API_INC_LEVEL0_SOFTMAX_CROSS_ENTROPY_WITH_H_

#include "opdev/op_executor.h"

namespace l0op {
std::tuple<aclTensor*, aclTensor*> SoftmaxCrossEntropyWithLogits(const aclTensor *features, const aclTensor *labels, aclOpExecutor *executor);
} // namespace l0op

#endif // OP_API_INC_LEVEL0_GELU_H_

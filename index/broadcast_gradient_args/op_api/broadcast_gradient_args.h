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
 * \file broadcast_gradient_args.h
 * \brief
 */
#ifndef OP_API_INC_LEVEL0_OP_BROADCAST_GRADIENT_ARGS_H_
#define OP_API_INC_LEVEL0_OP_BROADCAST_GRADIENT_ARGS_H_

#include "opdev/op_executor.h"

namespace l0op {
aclnnStatus BroadcastGradientArgs(const aclTensor* x1, const aclTensor* x2, aclTensor* y1, aclTensor* y2,
                                  aclOpExecutor* executor);
}

#endif // OP_API_INC_LEVEL0_OP_BROADCAST_GRADIENT_ARGS_H_

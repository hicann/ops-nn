/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef OP_API_INC_LEVEL0_OP_APPLY_ADAM_W_V2_H_
#define OP_API_INC_LEVEL0_OP_APPLY_ADAM_W_V2_H_

#include "opdev/op_executor.h"

namespace l0op {
void ApplyAdamWV2(
    aclTensor* varRef, aclTensor* mRef, aclTensor* vRef, aclTensor* maxGradNormOptionalRef, const aclTensor* grad,
    const aclTensor* step, float lr, float beta1, float beta2, float weightDecay, float eps, bool amsgrad,
    bool maximize, aclOpExecutor* executor);
}

#endif
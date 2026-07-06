/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */
/**
 * @file apply_adam_d.h
 * @brief ACLNN L0 API declaration for ApplyAdamD.
 */

#ifndef OP_API_INC_LEVEL0_APPLY_ADAM_D_H_
#define OP_API_INC_LEVEL0_APPLY_ADAM_D_H_

#include "opdev/op_executor.h"

namespace l0op {

struct ApplyAdamDOutputs {
    const aclTensor* varOut;
    const aclTensor* mOut;
    const aclTensor* vOut;
};

ApplyAdamDOutputs ApplyAdamD(const aclTensor* var, const aclTensor* m, const aclTensor* v, const aclTensor* beta1Power,
                             const aclTensor* beta2Power, const aclTensor* lr, const aclTensor* beta1,
                             const aclTensor* beta2, const aclTensor* epsilon, const aclTensor* grad, bool useLocking,
                             bool useNesterov, aclOpExecutor* executor);

} // namespace l0op

#endif // OP_API_INC_LEVEL0_APPLY_ADAM_D_H_

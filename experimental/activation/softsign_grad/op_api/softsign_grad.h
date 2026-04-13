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
 * @file softsign_grad.h
 * @brief ACLNN L0 API 接口声明 - SoftsignGrad
 *
 * L0 API 是 ACLNN 底层实现，由 L2 API 调用。
 * 职责：形状推导、Kernel 调度
 */

#ifndef OP_API_INC_LEVEL0_SOFTSIGN_GRAD_H_
#define OP_API_INC_LEVEL0_SOFTSIGN_GRAD_H_

#include "opdev/op_executor.h"

namespace l0op {

const aclTensor* SoftsignGrad(
    const aclTensor* gradients,
    const aclTensor* features,
    aclOpExecutor* executor);

} // namespace l0op

#endif // OP_API_INC_LEVEL0_SOFTSIGN_GRAD_H_

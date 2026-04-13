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
 * @file softsign.h
 * @brief ACLNN L0 API declaration for Softsign operator
 *
 * L0 API is called by L2 API.
 * Responsibilities: shape inference, kernel dispatch
 */

#ifndef OP_API_INC_LEVEL0_SOFTSIGN_H_
#define OP_API_INC_LEVEL0_SOFTSIGN_H_

#include "opdev/op_executor.h"

namespace l0op {

const aclTensor* Softsign(const aclTensor* self, aclOpExecutor* executor);

} // namespace l0op

#endif // OP_API_INC_LEVEL0_SOFTSIGN_H_

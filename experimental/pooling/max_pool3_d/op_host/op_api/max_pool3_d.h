/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_LEVEL0_MAX_POOL3_D_H_
#define OP_API_INC_LEVEL0_MAX_POOL3_D_H_

#include <string>

#include "opdev/op_executor.h"

namespace l0op {
const aclTensor* MaxPool3D(const aclTensor* x, const aclIntArray* ksize, const aclIntArray* strides,
                           const std::string& padding, const aclIntArray* pads, const aclIntArray* dilation,
                           int64_t ceilMode, const std::string& dataFormat, aclTensor* y, aclOpExecutor* executor);
} // namespace l0op

#endif // OP_API_INC_LEVEL0_MAX_POOL3_D_H_

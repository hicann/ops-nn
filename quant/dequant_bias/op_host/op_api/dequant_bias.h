/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_DEQUANT_BIAS_H_
#define OP_API_DEQUANT_BIAS_H_

#include "opdev/op_executor.h"

namespace l0op {

const aclTensor* DequantBias(const aclTensor* x, const aclTensor* weightScale, const aclTensor* activateScale,
                             const aclTensor* bias, int64_t outputDtype, aclOpExecutor* executor,
                             const aclTensor* out = nullptr);

} // namespace l0op

#endif // OP_API_DEQUANT_BIAS_H_

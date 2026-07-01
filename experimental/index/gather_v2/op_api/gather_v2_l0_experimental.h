/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_GATHER_V2_L0_EXPERIMENTAL_H
#define OP_API_GATHER_V2_L0_EXPERIMENTAL_H

#include "opdev/op_executor.h"
#include "opdev/shape_utils.h"

namespace l0op {
const aclTensor* GatherV2(
    const aclTensor* self, int64_t axis, const aclTensor* indices, const op::Shape& outShape, aclOpExecutor* executor,
    int64_t batchDims = 0, bool negativeIndexSupport = false);

const aclTensor* GatherV2(
    const aclTensor* self, int64_t axis, const aclTensor* indices, const aclTensor* out, aclOpExecutor* executor,
    int64_t batchDims = 0, bool negativeIndexSupport = false);
} // namespace l0op

#endif // OP_API_GATHER_V2_L0_EXPERIMENTAL_H

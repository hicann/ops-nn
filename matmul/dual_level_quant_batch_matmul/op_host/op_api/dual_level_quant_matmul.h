/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_COMMON_INC_LEVEL0_OP_DUAL_LEVEL_QUANT_MATMUL_H
#define OP_API_COMMON_INC_LEVEL0_OP_DUAL_LEVEL_QUANT_MATMUL_H

#include "opdev/op_executor.h"
#include "opdev/make_op_executor.h"

namespace l0op {
const aclTensor* DualLevelQuantBatchMatmul(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* x1Level0Scale, const aclTensor* x2Level0Scale,
    const aclTensor* x1Level1Scale, const aclTensor* x2Level1Scale, const aclTensor* optionalBias, int64_t dtype,
    bool transposeX1, bool transposeX2, int64_t level0GroupSize, int64_t level1GroupSize, aclOpExecutor* executor);
}

#endif // OP_API_COMMON_INC_LEVEL0_OP_DUAL_LEVEL_QUANT_MATMUL_H
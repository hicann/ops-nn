/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_HOST_OP_API_QUANT_BATCH_MATMUL_INPLACE_ADD_H
#define OP_HOST_OP_API_QUANT_BATCH_MATMUL_INPLACE_ADD_H

#include "opdev/op_executor.h"
#include "opdev/make_op_executor.h"

namespace l0op {
aclTensor* QuantBatchMatmulInplaceAdd(const aclTensor* x1, const aclTensor* x2, const aclTensor* x2Scale,
                                      aclTensor* yRef, const aclTensor* x1ScaleOptional, bool transposeX1, 
                                      bool transposeX2, int64_t groupSize, aclOpExecutor* executor);
}

#endif // OP_HOST_OP_API_QUANT_BATCH_MATMUL_INPLACE_ADD_H
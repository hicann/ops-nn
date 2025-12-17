/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/

/*!
 * \file quant_matmul_reduce_sum.h
 * \brief
 */
#ifndef PTA_NPU_OP_API_INC_LEVEL0_OP_QUANT_MATMUL_REDUCE_SUM_H
#define PTA_NPU_OP_API_INC_LEVEL0_OP_QUANT_MATMUL_REDUCE_SUM_H

#include "opdev/op_executor.h"

namespace l0op {
const aclTensor* QuantMatmulReduceSum(
    const aclTensor* x1, const aclTensor* x2, const aclIntArray* dims, const aclTensor* bias,
    const aclTensor* x1Scale, const aclTensor* x2Scale, const aclTensor* yScale,
    const aclTensor* x1Offset, const aclTensor* x2Offset, const aclTensor* yOffset, const aclTensor* x2Table,
    int64_t dtype, bool transposeX1, bool transposeX2, uint64_t groupSize, bool keepDims,
    aclOpExecutor* executor);
}

#endif // PTA_NPU_OP_API_INC_LEVEL0_OP_QUANT_MATMUL_REDUCE_SUM_H
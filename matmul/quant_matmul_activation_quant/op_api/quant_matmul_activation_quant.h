/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_HOST_OP_API_QUANT_MATMUL_ACTIVATION_QUANT_H
#define OP_HOST_OP_API_QUANT_MATMUL_ACTIVATION_QUANT_H

#include <array>
#include <cstddef>
#include <cstdint>

#include "aclnn/acl_meta.h"

namespace l0op {
constexpr std::size_t QUANT_MATMUL_ACTIVATION_QUANT_OUT_NUM = 2; // output y and yScale

const std::array<aclTensor*, QUANT_MATMUL_ACTIVATION_QUANT_OUT_NUM> QuantMatmulActivationQuant(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* bias, const aclTensor* x1Scale, const aclTensor* x2Scale,
    const aclTensor* x3, bool transposeX1, bool transposeX2, int64_t groupSize, const char* activationType,
    int64_t yDtype, const char* quantMode, const char* roundMode, int64_t scaleAlg, double dstTypeMax,
    aclOpExecutor* executor);
} // namespace l0op

#endif // OP_HOST_OP_API_QUANT_MATMUL_ACTIVATION_QUANT_H

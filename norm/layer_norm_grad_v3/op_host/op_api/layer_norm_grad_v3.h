/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_LEVEL0_LAYER_NORM_GRAD_V3_H_
#define OP_API_INC_LEVEL0_LAYER_NORM_GRAD_V3_H_

#include "opdev/op_executor.h"

using namespace op;

namespace l0op {
constexpr size_t GRAD_V3_OUT_NUM = 3;

const std::array<aclTensor*, GRAD_V3_OUT_NUM> LayerNormGradV3(
    const aclTensor* gradOut, const aclTensor* input, const aclTensor* rstd, const aclTensor* mean,
    const aclTensor* weight, const aclBoolArray* outputMask, const DataType gradWeightType, aclOpExecutor* executor);
} // namespace l0op

#endif // OP_API_INC_LEVEL0_LAYER_NORM_GRAD_V3_H_

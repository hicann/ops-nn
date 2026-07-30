/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_LEVEL0_SWIGLU_GROUP_GRAD_H_
#define OP_API_INC_LEVEL0_SWIGLU_GROUP_GRAD_H_

#include <array>

#include "opdev/op_executor.h"

namespace l0op {

constexpr int64_t SWIGLU_GROUP_GRAD_OUT_NUM = 2;

/**
 * @brief SwigluGroupGrad kernel-level interface.
 *
 * Allocates output tensors and registers kernel in the launch list.
 * Optional tensor inputs (weight, yOrigin, groupIndex) can be nullptr
 * when not provided by the caller.
 *
 * @param gradY               Required input: upstream gradient (T, H) or (B, S, H)
 * @param x                   Required input: forward input (T, 2H) or (B, S, 2H)
 * @param weightOptional      Optional input: MoE top-k weights (T, 1) or (B, S, 1), can be nullptr
 * @param yOriginOptional     Optional input: forward output y (T, H) or (B, S, H), can be nullptr
 * @param groupIndexOptional  Optional input: token count per group (G,), can be nullptr
 * @param clampLimit          Extracted from aclScalar; 0 means no clamp (c=+∞)
 * @param executor            Op executor for tensor allocation and kernel registration
 * @return std::array with [gradXOut, gradWeightOutOptional]. When weightOptional is nullptr,
 *         gradWeightOutOptional is a dummy one-element tensor that the kernel ignores.
 */
std::array<aclTensor*, SWIGLU_GROUP_GRAD_OUT_NUM> SwigluGroupGrad(const aclTensor* gradY, const aclTensor* x,
                                                                  const aclTensor* weightOptional,
                                                                  const aclTensor* yOriginOptional,
                                                                  const aclTensor* groupIndexOptional, float clampLimit,
                                                                  aclOpExecutor* executor);

} // namespace l0op
#endif // OP_API_INC_LEVEL0_SWIGLU_GROUP_GRAD_H_

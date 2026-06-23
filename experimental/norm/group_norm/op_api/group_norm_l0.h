/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef EXPERIMENTAL_NORM_GROUP_NORM_L0_H_
#define EXPERIMENTAL_NORM_GROUP_NORM_L0_H_

#include <tuple>
#include "opdev/op_executor.h"

namespace l0op {
const std::tuple<aclTensor*, aclTensor*, aclTensor*> GroupNorm(
    const aclTensor* x, const aclTensor* gamma, const aclTensor* beta, int64_t n, int64_t numGroups, float eps,
    aclOpExecutor* executor);

inline const std::tuple<aclTensor*, aclTensor*, aclTensor*> GroupNormExperimental(
    const aclTensor* x, const aclTensor* gamma, const aclTensor* beta, int64_t n, int64_t numGroups, float eps,
    aclOpExecutor* executor)
{
    return GroupNorm(x, gamma, beta, n, numGroups, eps, executor);
}
} // namespace l0op

#endif // EXPERIMENTAL_NORM_GROUP_NORM_L0_H_

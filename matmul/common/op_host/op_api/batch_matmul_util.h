/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include "aclnn/aclnn_base.h"
#include "matmul/common/op_host/op_api/matmul_util.h"
#include "opdev/common_types.h"
#include "opdev/platform.h"

namespace Ops {
namespace NN {
enum class NonContiguousMode : int32_t { CONTINUOUS = 0, B_NON_CONTINUOUS = 1, AB_NON_CONTINUOUS = 2 };

const aclTensor* ExecBmmOpWithBiasV2(const aclTensor* self, const aclTensor* mat2, const aclTensor* bias,
                                     const aclTensor* out, int8_t cubeMathType, aclOpExecutor* executor,
                                     bool isBaddbmm = false);

const aclTensor* ExecBatchMatmulOpWithBiasAndAttrsV2(
    const aclTensor* self, const aclTensor* mat2, const aclTensor* bias, const aclTensor* out, bool adjX1, bool adjX2,
    int8_t cubeMathType, aclOpExecutor* executor, NonContiguousMode nonContiguousMode = NonContiguousMode::CONTINUOUS,
    bool isBaddbmm = false);

const aclTensor* ExecBmmOpV2(const aclTensor* self, const aclTensor* mat2, const aclTensor* out, int8_t cubeMathType,
                             aclOpExecutor* executor, bool isBaddbmm = false);

const aclTensor* ExecBmmOpWithBias(const aclTensor* self, const aclTensor* mat2, const aclTensor* bias,
                                   const aclTensor* out, int8_t cubeMathType, aclOpExecutor* executor,
                                   bool isBaddbmm = false);

const aclTensor* ExecBatchMatmulOpWithBiasAndAttrs(const aclTensor* self, const aclTensor* mat2, const aclTensor* bias,
                                                   const aclTensor* out, bool adjX1, bool adjX2, int8_t cubeMathType,
                                                   aclOpExecutor* executor, bool isTransposeMat2Contiguous = false,
                                                   bool isBaddbmm = false);

const aclTensor* ExecBatchMatmulOp(const aclTensor* self, const aclTensor* mat2, const aclTensor* out, bool adjX1,
                                   bool adjX2, int8_t cubeMathType, aclOpExecutor* executor);

const aclTensor* ExecBmmOp(const aclTensor* self, const aclTensor* mat2, const aclTensor* out, int8_t cubeMathType,
                           aclOpExecutor* executor, bool isBaddbmm = false);

int64_t ProcessEqual1Cases(const aclTensor*& selfCast, const aclTensor*& mat2Cast, MmOpInfo& matmulOpInfo,
                           const aclTensor*& bias, bool& adjX1, bool& adjX2, const aclTensor*& selfReshape,
                           const aclTensor*& mat2Reshape, aclOpExecutor* executor, bool& ifKEqual1);

bool checkFusedmm(const aclTensor* bias, const aclTensor* self, const aclTensor* mat2, const aclScalar* alpha,
                  const aclScalar* beta, int8_t cubeMathType, bool& isNeedSwapInnerTwoDim);

const aclTensor* ExecFusedmmOp(const aclTensor* bias, const aclTensor* self, const aclTensor* mat2, int8_t cubeMathType,
                               const bool isNeedSwapInnerTwoDim, aclOpExecutor* executor);
} // namespace NN
} // namespace Ops

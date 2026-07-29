/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "fused_matmul_gelu.h"

#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(FusedMatmulGelu);

namespace {
op::Shape MakeMatmulOutputShape(const aclTensor* x, const aclTensor* weight)
{
    const auto xViewShape = x->GetViewShape();
    const auto outRank = xViewShape.GetDimNum();
    const auto outLastDim = weight->GetViewShape().GetDim(0);
    op::Shape matmulOutShape;

    for (size_t dimIdx = 0; dimIdx < outRank; ++dimIdx) {
        const auto dimValue = (dimIdx + 1U == outRank) ? outLastDim : xViewShape.GetDim(dimIdx);
        matmulOutShape.AppendDim(dimValue);
    }

    return matmulOutShape;
}
} // namespace

const aclTensor* FusedMatmulGelu(const aclTensor* x, const aclTensor* weight, const aclTensor* bias,
                                 int64_t approximate, aclOpExecutor* executor)
{
    auto yShape = MakeMatmulOutputShape(x, weight);

    auto y = executor->AllocTensor(yShape, yShape, x->GetDataType(), x->GetStorageFormat(), x->GetOriginalFormat());
    CHECK_RET(y != nullptr, nullptr);

    L0_DFX(FusedMatmulGelu, x, weight, bias, approximate, y);

    ADD_TO_LAUNCHER_LIST_AICORE(FusedMatmulGelu, OP_INPUT(x, weight, bias), OP_OUTPUT(y), OP_ATTR(approximate));

    return y;
}

} // namespace l0op

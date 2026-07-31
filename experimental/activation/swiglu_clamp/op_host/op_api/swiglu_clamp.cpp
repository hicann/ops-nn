/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file swiglu_clamp.cpp
 * \brief SwigluClamp l0op: x[...,2N] -> y[...,N], with scalar attr limit.
 *        Halved output shape mirrors GLU; scalar-attr threading mirrors LeakyRelu.
 */
#include "swiglu_clamp.h"

#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "aclnn_kernels/common/op_error_check.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(SwigluClamp);

// AICORE op kernel: limit threaded as a scalar attr.
static const aclTensor* SwigluClampAiCore(const aclTensor* x, const float limit, const aclTensor* y,
                                          aclOpExecutor* executor)
{
    L0_DFX(SwigluClampAiCore, x, limit, y);
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(SwigluClamp, OP_INPUT(x), OP_OUTPUT(y), OP_ATTR(limit));
    OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(ret != ACLNN_SUCCESS, return nullptr,
                                         "SwigluClamp ADD_TO_LAUNCHER_LIST_AICORE failed.");
    return y;
}

const aclTensor* SwigluClamp(const aclTensor* x, const float limit, aclOpExecutor* executor)
{
    // y shape = x shape with the last dim halved (gate/up split)
    op::Shape outShape = x->GetViewShape();
    size_t dimNum = outShape.GetDimNum();
    const int64_t SLICE_NUM = 2;
    outShape.SetDim(dimNum - 1, outShape.GetDim(dimNum - 1) / SLICE_NUM);

    auto y = executor->AllocTensor(outShape, x->GetDataType(), Format::FORMAT_ND);
    if (y == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "alloc out tensor failed.");
        return nullptr;
    }
    return SwigluClampAiCore(x, limit, y, executor);
}
} // namespace l0op

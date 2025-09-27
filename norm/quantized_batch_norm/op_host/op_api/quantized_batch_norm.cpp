/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file quantized_batch_norm.cpp
 * \brief
 */

#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "quantized_batch_norm.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(QuantizedBatchNorm);

const aclTensor* QuantizedBatchNorm(
    const l0op::QuantizedBatchNormParams& params, float epsilon, aclOpExecutor* executor)
{
    L0_DFX(
        QuantizedBatchNorm, params.x, params.mean, params.var, params.inputScale, params.inputZeroPoint,
        params.outputScale, params.outputZeroPoint, params.weight, params.bias, epsilon);

    auto y = executor->AllocTensor(
        params.x->GetStorageShape(), params.x->GetOriginalShape(), params.x->GetDataType(),
        params.x->GetStorageFormat(), params.x->GetOriginalFormat());

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        QuantizedBatchNorm,
        OP_INPUT(
            params.x, params.mean, params.var, params.inputScale, params.inputZeroPoint, params.outputScale,
            params.outputZeroPoint, params.weight, params.bias),
        OP_OUTPUT(y), OP_ATTR(epsilon));
    if (ret != ACL_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "QuantizedBatchNorm ADD_TO_LAUNCHER_LIST_AICORE failed.");
        return nullptr;
    }
    return y;
}
} // namespace l0op
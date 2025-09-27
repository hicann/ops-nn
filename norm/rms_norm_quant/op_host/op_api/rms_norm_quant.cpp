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
 * \file rms_norm_quant.cpp
 * \brief
 */
#include "rms_norm_quant.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/common_types.h"
#include "opdev/platform.h"
#include "aclnn_kernels/cast.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(RmsNormQuant);

const aclTensor* RmsNormQuant(
    const aclTensor* x, const aclTensor* gamma, const aclTensor* beta, const aclTensor* scale, const aclTensor* offset,
    double epsilon, aclOpExecutor* executor)
{
    L0_DFX(RmsNormQuant, x, gamma, beta, scale, offset, epsilon);

    auto y = executor->AllocTensor(x->GetViewShape(), DataType::DT_INT8, x->GetViewFormat());

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        RmsNormQuant, OP_INPUT(x, gamma, beta, scale, offset), OP_OUTPUT(y),
        OP_ATTR(static_cast<float>(epsilon), false, false));
    if (ret != ACL_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "RmsNormQuant ADD_TO_LAUNCHER_LIST_AICORE failed.");
        return nullptr;
    }
    return y;
}
} // namespace l0op
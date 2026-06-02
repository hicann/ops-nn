/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "rotate_quant.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/shape_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/format_utils.h"
#include "aclnn_kernels/common/op_error_check.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(RotateQuant);

std::tuple<aclTensor*, aclTensor*> RotateQuant(
    const aclTensor* x, const aclTensor* rot, int32_t dst_dtype, aclOpExecutor* executor)
{
    L0_DFX(RotateQuant, x, rot);

    Format format = Format::FORMAT_ND;

    auto yOut = executor->AllocTensor(op::DataType(dst_dtype), format, format);
    auto scaleOut = executor->AllocTensor(op::DataType::DT_FLOAT, format, format);

    auto ret = INFER_SHAPE(RotateQuant, OP_INPUT(x, rot), OP_OUTPUT(yOut, scaleOut), OP_ATTR(dst_dtype));
    OP_CHECK_INFERSHAPE(ret != ACLNN_SUCCESS, return std::tuple(nullptr, nullptr), "RotateQuant InferShape failed.");

    ret = ADD_TO_LAUNCHER_LIST_AICORE(RotateQuant, OP_INPUT(x, rot), OP_OUTPUT(yOut, scaleOut), OP_ATTR(dst_dtype));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "RotateQuant launch kernel failed.");
        return std::tuple<aclTensor*, aclTensor*>(nullptr, nullptr);
    }
    return std::tuple<aclTensor*, aclTensor*>(yOut, scaleOut);
}

} // namespace l0op

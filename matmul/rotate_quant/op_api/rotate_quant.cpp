/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file rotate_quant.cpp
 * \brief RotateQuant Level 0 Operator implementation
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

static std::tuple<aclTensor*, aclTensor*> RotateQuantAICore(const aclTensor* x, const aclTensor* rotation,
                                                            const aclTensor* alpha, int64_t axis, const char* roundMode,
                                                            int64_t scaleAlg, float dstTypeMax, bool trans,
                                                            int64_t dstType, aclTensor* yOut, aclTensor* scaleOut,
                                                            aclOpExecutor* executor)
{
    L0_DFX(RotateQuantAICore, x, rotation, alpha, axis, roundMode, scaleAlg, dstTypeMax, trans, dstType, yOut,
           scaleOut);
    auto retAicore = ADD_TO_LAUNCHER_LIST_AICORE(RotateQuant, OP_INPUT(x, rotation, alpha), OP_OUTPUT(yOut, scaleOut),
                                                 OP_ATTR(dstType, axis, roundMode, scaleAlg, dstTypeMax, trans));
    OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(retAicore != ACLNN_SUCCESS, return std::tuple(nullptr, nullptr),
                                         "RotateQuant add to aicore launch list failed.");
    return std::tie(yOut, scaleOut);
}

const std::tuple<aclTensor*, aclTensor*> RotateQuant(const aclTensor* x, const aclTensor* rotation,
                                                     const aclTensor* alpha, int64_t axis, const char* roundMode,
                                                     int64_t scaleAlg, float dstTypeMax, bool trans, int64_t dstType,
                                                     aclTensor* yOut, aclTensor* scaleOut, aclOpExecutor* executor)
{
    if (scaleOut->GetDataType() != DataType::DT_FLOAT8_E8M0) {
        yOut = executor->AllocTensor(x->GetViewShape(), op::DataType(dstType), op::Format::FORMAT_ND);
        int64_t M = x->GetViewShape().GetDim(0);
        scaleOut = executor->AllocTensor(op::Shape({M}), op::DataType::DT_FLOAT, op::Format::FORMAT_ND);
    }
    return RotateQuantAICore(x, rotation, alpha, axis, roundMode, scaleAlg, dstTypeMax, trans, dstType, yOut, scaleOut,
                             executor);
}

} // namespace l0op

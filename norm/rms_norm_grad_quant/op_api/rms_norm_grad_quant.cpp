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
 * \file rms_norm_grad_quant.cpp
 * \brief L0 operator implementation for RmsNormGradQuant
 */
#include "rms_norm_grad_quant.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(RmsNormGradQuant);

const std::array<aclTensor*, RMS_NORM_GRAD_QUANT_OUT_NUM> RmsNormGradQuant(
    const aclTensor* dy, const aclTensor* x, const aclTensor* rstd, const aclTensor* gamma,
    const aclTensor* scalesX, const aclTensor* offsetXOptional,
    const char* quantMode, bool divMode, int32_t dstType, aclTensor* dxOut,
    aclTensor* dgammaOut, aclOpExecutor* executor)
{
    L0_DFX(RmsNormGradQuant, dy, x, rstd, gamma, scalesX, offsetXOptional, quantMode, divMode, dstType);

    OP_LOGD(
        "dxOut=[%s], dgammaOut=[%s].", op::ToString(dxOut->GetViewShape()).GetString(),
        op::ToString(dgammaOut->GetViewShape()).GetString());

    std::string quantModeStr(quantMode != nullptr ? quantMode : "static");
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        RmsNormGradQuant,
        OP_INPUT(dy, x, rstd, gamma, scalesX, offsetXOptional),
        OP_OUTPUT(dxOut, dgammaOut),
        OP_ATTR(quantModeStr, divMode, dstType));
    if (ret != ACL_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "RmsNormGradQuant ADD_TO_LAUNCHER_LIST_AICORE failed.");
        return std::array<aclTensor*, RMS_NORM_GRAD_QUANT_OUT_NUM>{nullptr, nullptr};
    }
    return {dxOut, dgammaOut};
}
} // namespace l0op

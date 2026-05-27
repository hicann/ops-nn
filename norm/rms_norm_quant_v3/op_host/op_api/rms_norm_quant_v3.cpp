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
 * \file rms_norm_quant_v3.cpp
 * \brief
 */
#include "rms_norm_quant_v3.h"
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
OP_TYPE_REGISTER(RmsNormQuantV3);

const std::array<aclTensor*, RMS_NORM_QUANT_V3_OUT_NUM> RmsNormQuantV3(
    const aclTensor* x, const aclTensor* gamma, const aclTensor* scales1, const aclTensor* scales2Optional,
    const aclTensor* zeroPoints1Optional, const aclTensor* zeroPoints2Optional, const aclTensor* betaOptional,
    double epsilon, bool divMode, int32_t dstType, bool outputRstd, aclTensor* rstdOut, aclOpExecutor* executor)
{
    L0_DFX(
        RmsNormQuantV3, x, gamma, scales1, scales2Optional, zeroPoints1Optional, zeroPoints2Optional, betaOptional,
        epsilon, divMode, dstType, outputRstd);

    aclTensor* y1 = executor->AllocTensor(x->GetViewShape(), op::DataType(dstType), x->GetViewFormat());
    aclTensor* y2 = y1;
    if (scales2Optional != nullptr) {
        y2 = executor->AllocTensor(x->GetViewShape(), op::DataType(dstType), x->GetViewFormat());
    }

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        RmsNormQuantV3,
        OP_INPUT(x, gamma, scales1, scales2Optional, zeroPoints1Optional, zeroPoints2Optional, betaOptional),
        OP_OUTPUT(y1, y2, rstdOut),
        OP_ATTR(static_cast<float>(epsilon), divMode, dstType, outputRstd));
    if (ret != ACL_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "RmsNormQuantV3 ADD_TO_LAUNCHER_LIST_AICORE failed.");
        return std::array<aclTensor*, RMS_NORM_QUANT_V3_OUT_NUM>{nullptr, nullptr};
    }
    return {y1, rstdOut};
}
} // namespace l0op

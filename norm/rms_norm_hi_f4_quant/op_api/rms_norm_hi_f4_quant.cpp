/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "rms_norm_hi_f4_quant.h"
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
OP_TYPE_REGISTER(RmsNormHiF4Quant);

const std::array<aclTensor*, RMS_NORM_HI_F4_QUANT_OUT_NUM> RmsNormHiF4Quant(const aclTensor* x, const aclTensor* gamma,
                                                                            const aclTensor* beta, double epsilon,
                                                                            bool outputRstd, int64_t scaleAlg,
                                                                            aclTensor* yOut, aclTensor* scaleOut,
                                                                            aclTensor* rstdOut, aclOpExecutor* executor)
{
    L0_DFX(RmsNormHiF4Quant, x, gamma, beta, epsilon, outputRstd, scaleAlg);

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(RmsNormHiF4Quant, OP_INPUT(x, gamma, beta),
                                           OP_OUTPUT(yOut, scaleOut, rstdOut),
                                           OP_ATTR(static_cast<float>(epsilon), outputRstd, scaleAlg));
    if (ret != ACL_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "RmsNormHiF4Quant ADD_TO_LAUNCHER_LIST_AICORE failed.");
        return {nullptr, nullptr, nullptr};
    }
    return {yOut, scaleOut, rstdOut};
}

} // namespace l0op

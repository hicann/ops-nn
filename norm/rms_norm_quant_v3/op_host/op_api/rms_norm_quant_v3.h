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
 * \file rms_norm_quant_v3.h
 * \brief
 */
#ifndef PTA_NPU_OP_API_INC_LEVEL0_OP_RMS_NORM_QUANT_V3_H_
#define PTA_NPU_OP_API_INC_LEVEL0_OP_RMS_NORM_QUANT_V3_H_

#include "opdev/op_executor.h"

namespace l0op {
constexpr size_t RMS_NORM_QUANT_V3_OUT_NUM = 2;

const std::array<aclTensor*, RMS_NORM_QUANT_V3_OUT_NUM> RmsNormQuantV3(
    const aclTensor* x, const aclTensor* gamma, const aclTensor* scales1, const aclTensor* scales2Optional,
    const aclTensor* zeroPoints1Optional, const aclTensor* zeroPoints2Optional, const aclTensor* betaOptional,
    double epsilon, bool divMode, int32_t dstType, bool outputRstd, aclTensor* rstdOut, aclOpExecutor* executor);

} // namespace l0op

#endif // PTA_NPU_OP_API_INC_LEVEL0_OP_RMS_NORM_QUANT_V3_H_

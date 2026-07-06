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
 * \file gru.cpp
 * \brief
 */
#include "gru.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "aclnn_kernels/common/op_error_check.h"

using namespace op;
namespace l0op {
OP_TYPE_REGISTER(Gru);

static auto gruNullptrInner = std::tuple<aclTensor*, aclTensor*, aclTensor*, aclTensor*, aclTensor*, aclTensor*>(
    nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);

const std::tuple<const aclTensor*, const aclTensor*, const aclTensor*, const aclTensor*, const aclTensor*,
                 const aclTensor*>
Gru(const aclTensor* input, const aclTensor* weightInput, const aclTensor* weightHidden, const aclTensor* biasInput,
    const aclTensor* biasHidden, const aclTensor* seqLengthOptional, const aclTensor* initHOptional,
    const char* direction, bool train, aclTensor* yOut, aclTensor* outputHOut, aclTensor* rOut, aclTensor* zOut,
    aclTensor* nOut, aclTensor* nHOut, aclOpExecutor* executor)
{
    L0_DFX(Gru, input, weightInput, weightHidden, biasInput, biasHidden, seqLengthOptional, initHOptional, direction,
           train, yOut, outputHOut, rOut, zOut, nOut, nHOut);
    if (GetCurrentPlatformInfo().GetSocVersion() != SocVersion::ASCEND910B &&
        GetCurrentPlatformInfo().GetSocVersion() != SocVersion::ASCEND910_93) {
        return gruNullptrInner;
    }

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        Gru, OP_INPUT(input, weightInput, weightHidden, biasInput, biasHidden, seqLengthOptional, initHOptional),
        OP_OUTPUT(yOut, outputHOut, rOut, zOut, nOut, nHOut), OP_ATTR(direction, train));
    OP_CHECK(ret == ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Gru ADD_TO_LAUNCHER_LIST_AICORE failed."),
             return gruNullptrInner);

    return std::tuple<aclTensor*, aclTensor*, aclTensor*, aclTensor*, aclTensor*, aclTensor*>(yOut, outputHOut, rOut,
                                                                                              zOut, nOut, nHOut);
}
} // namespace l0op
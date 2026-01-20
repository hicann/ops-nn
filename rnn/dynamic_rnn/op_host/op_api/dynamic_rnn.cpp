/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "dynamic_rnn.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "aclnn_kernels/common/op_error_check.h"


using namespace op;
namespace l0op {
OP_TYPE_REGISTER(DynamicRNN);

auto nullptrInner = std::tuple<aclTensor*, aclTensor*, aclTensor*, aclTensor*, aclTensor*, aclTensor*, aclTensor*, aclTensor*>(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);

const std::tuple<const aclTensor*, const aclTensor*, const aclTensor*, const aclTensor*, const aclTensor*, const aclTensor*, const aclTensor*, const aclTensor*> DynamicRNN(
    const aclTensor* input, const aclTensor* weight, const aclTensor* bias, 
    const aclTensor *initHOptional,
    const aclTensor *initCOptional,
    const aclTensor *seqLengthOptional,
    const char *direction,
    bool train,
    aclTensor *yOutDirec,
    aclTensor *iOutDirec,
    aclTensor *jOutDirec,
    aclTensor *fOutDirec,
    aclTensor *oOutDirec,
    aclTensor *hOutDirec,
    aclTensor *cOutDirec,
    aclTensor *tanhCOutDirec,
    aclOpExecutor* executor) {
    L0_DFX(
        DynamicRNN,
        input, weight, bias, seqLengthOptional, initHOptional, initCOptional, direction, train,
        yOutDirec, hOutDirec, cOutDirec, iOutDirec, jOutDirec, fOutDirec, oOutDirec, tanhCOutDirec
    );
    if (GetCurrentPlatformInfo().GetSocVersion() != SocVersion::ASCEND910B &&
        GetCurrentPlatformInfo().GetSocVersion() != SocVersion::ASCEND910_93) {
        return nullptrInner;
    }

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(DynamicRNN, OP_INPUT(input, weight, bias, seqLengthOptional, initHOptional, initCOptional, nullptr, nullptr, nullptr, nullptr), 
                                OP_OUTPUT(yOutDirec, hOutDirec, cOutDirec, iOutDirec, jOutDirec, fOutDirec, oOutDirec, tanhCOutDirec),
                                OP_ATTR("LSTM", direction, 1, false, 1.0, -1.0, 0, true, "tanh", 0.0, "ifjo", train));
    OP_CHECK(
        ret == ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "DynamicRNN ADD_TO_LAUNCHER_LIST_AICORE failed."),
        return nullptrInner);

    return std::tuple<aclTensor*, aclTensor*, aclTensor*, aclTensor*, aclTensor*, aclTensor*, aclTensor*, aclTensor*>(yOutDirec, iOutDirec, jOutDirec, fOutDirec, oOutDirec, hOutDirec, cOutDirec, tanhCOutDirec);
}
}  // namespace l0op

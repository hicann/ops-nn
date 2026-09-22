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
 * \file cla_gate_backward.cpp
 * \brief
 */
#include "cla_gate_backward.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "aclnn_kernels/common/op_error_check.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(ClaGateBackward);

static inline ClaGateBackwardOut ClaGateBackwardAiCore(const aclTensor* gradMerged, const aclTensor* globalAttn,
                                                       const aclTensor* localAttn, const aclTensor* globalGateLogits,
                                                       const aclTensor* localGateLogits, const char* inputAttnLayout,
                                                       const aclTensor* gradGlobalAttnOut,
                                                       const aclTensor* gradLocalAttnOut,
                                                       const aclTensor* gradGlobalGateLogitsOut,
                                                       const aclTensor* gradLocalGateLogitsOut, aclOpExecutor* executor)
{
    L0_DFX(ClaGateBackwardAiCore, gradMerged, globalAttn, localAttn, globalGateLogits, localGateLogits, inputAttnLayout,
           gradGlobalAttnOut, gradLocalAttnOut, gradGlobalGateLogitsOut, gradLocalGateLogitsOut);

    const char* inputAttnLayoutStr = (inputAttnLayout != nullptr) ? inputAttnLayout : "TND";
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        ClaGateBackward, OP_INPUT(gradMerged, globalAttn, localAttn, globalGateLogits, localGateLogits),
        OP_OUTPUT(gradGlobalAttnOut, gradLocalAttnOut, gradGlobalGateLogitsOut, gradLocalGateLogitsOut),
        OP_ATTR(inputAttnLayoutStr));
    OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(ret != ACL_SUCCESS,
                                         return (ClaGateBackwardOut{nullptr, nullptr, nullptr, nullptr}),
                                         "ClaGateBackwardAiCore ADD_TO_LAUNCHER_LIST_AICORE failed.");
    return {gradGlobalAttnOut, gradLocalAttnOut, gradGlobalGateLogitsOut, gradLocalGateLogitsOut};
}

ClaGateBackwardOut ClaGateBackward(const aclTensor* gradMerged, const aclTensor* globalAttn, const aclTensor* localAttn,
                                   const aclTensor* globalGateLogits, const aclTensor* localGateLogits,
                                   const char* inputAttnLayout, aclOpExecutor* executor)
{
    // 输出与对应输入同 shape/dtype：分支梯度同 G，logits 梯度同 gate logits
    auto gradGlobalAttnOut = executor->AllocTensor(gradMerged->GetViewShape(), gradMerged->GetDataType());
    auto gradLocalAttnOut = executor->AllocTensor(gradMerged->GetViewShape(), gradMerged->GetDataType());
    auto gradGlobalGateLogitsOut = executor->AllocTensor(globalGateLogits->GetViewShape(),
                                                         globalGateLogits->GetDataType());
    auto gradLocalGateLogitsOut = executor->AllocTensor(localGateLogits->GetViewShape(),
                                                        localGateLogits->GetDataType());
    OP_CHECK_NULL(gradGlobalAttnOut, return (ClaGateBackwardOut{nullptr, nullptr, nullptr, nullptr}));
    OP_CHECK_NULL(gradLocalAttnOut, return (ClaGateBackwardOut{nullptr, nullptr, nullptr, nullptr}));
    OP_CHECK_NULL(gradGlobalGateLogitsOut, return (ClaGateBackwardOut{nullptr, nullptr, nullptr, nullptr}));
    OP_CHECK_NULL(gradLocalGateLogitsOut, return (ClaGateBackwardOut{nullptr, nullptr, nullptr, nullptr}));

    return ClaGateBackwardAiCore(gradMerged, globalAttn, localAttn, globalGateLogits, localGateLogits, inputAttnLayout,
                                 gradGlobalAttnOut, gradLocalAttnOut, gradGlobalGateLogitsOut, gradLocalGateLogitsOut,
                                 executor);
}
} // namespace l0op

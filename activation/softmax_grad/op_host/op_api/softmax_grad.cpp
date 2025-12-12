/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/
#include "opdev/op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/make_op_executor.h"
#include "softmax_grad.h"

using namespace op;
namespace l0op {
OP_TYPE_REGISTER(SoftmaxGrad);

aclTensor *SoftmaxGrad(const aclTensor *gradOutput, const aclTensor *output, int64_t dim,
                       aclOpExecutor *executor) {
    auto out = executor->AllocTensor(gradOutput->GetStorageShape(), gradOutput->GetDataType());

    // 使用框架宏ADD_TO_LAUNCHER_LIST，将SoftmaxGrad算子加入任务队列
    L0_DFX(SoftmaxGrad, gradOutput, output, dim);
    FVector<int64_t> newdim{dim};
    auto dims = executor->AllocIntArray(newdim.data(), 1);
    ADD_TO_LAUNCHER_LIST_AICORE(SoftmaxGrad, OP_INPUT(output, gradOutput), OP_OUTPUT(out), OP_ATTR(dims));
    return out;
}
}

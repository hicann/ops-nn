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
#include "opdev/aicpu/aicpu_task.h"
#include "softmax_v2_op.h"

using namespace op;
namespace l0op {

OP_TYPE_REGISTER(SoftmaxV2);

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT,
                                                                              op::DataType::DT_FLOAT16,
                                                                              op::DataType::DT_BF16};

static bool IsAiCoreSupport(const aclTensor *self) {
    return CheckType(self->GetDataType(), AICORE_DTYPE_SUPPORT_LIST);
}

// AICORE算子kernel
static const aclTensor *SoftmaxAiCore(const aclTensor *self, aclIntArray *dimList, aclTensor *out,
                                      aclOpExecutor *executor) {
    L0_DFX(SoftmaxAiCore);
    ADD_TO_LAUNCHER_LIST_AICORE(SoftmaxV2, OP_INPUT(self), OP_OUTPUT(out), OP_ATTR(dimList, false));
    return out;
}

// AICPU算子kernel
static const aclTensor *SoftmaxAiCpu(const aclTensor *self, aclIntArray *dimList, aclTensor *out,
                                     aclOpExecutor *executor) {
    L0_DFX(SoftmaxAiCpu);
    static internal::AicpuTaskSpace space("SoftmaxV2", ge::DEPEND_IN_SHAPE, false);
    auto ret = ADD_TO_LAUNCHER_LIST_AICPU(SoftmaxV2, OP_ATTR_NAMES({"axes"}), OP_INPUT(self),
                                          OP_OUTPUT(out), OP_ATTR(dimList));
    CHECK_RET(ret == ACLNN_SUCCESS, nullptr);
    return out;
}

const aclTensor *SoftmaxV2(const aclTensor *self, int64_t dim, aclOpExecutor *executor) {
    // 输出shape和self shape一致
    auto out = executor->AllocTensor(self->GetStorageShape(), self->GetDataType());
    FVector<int64_t> newdim{dim};
    auto dimList = executor->AllocIntArray(newdim.data(), 1);

    if (IsAiCoreSupport(self)) {
        return SoftmaxAiCore(self, dimList, out, executor);
    } else {
        return SoftmaxAiCpu(self, dimList, out, executor);
    }
}

}


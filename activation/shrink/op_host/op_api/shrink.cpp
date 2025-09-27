/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "shrink.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;
namespace l0op{
OP_TYPE_REGISTER(Shrink);

static const std::initializer_list<DataType> AICORE_DTYPE_SUPPORT_LIST = {DataType::DT_FLOAT, DataType::DT_FLOAT16};

//根据芯片类型、dtype判断算子是否支持走aicore
static inline bool IsAiCoreSupport(DataType inputDtype) {
    //只需要判断dtype
    return CheckType(inputDtype, AICORE_DTYPE_SUPPORT_LIST);
}

//AICORE算子kernel
static inline const aclTensor* ShrinkAiCore(const aclTensor* input, float lambd, float bias, aclTensor* output, aclOpExecutor* executor){
    L0_DFX(ShrinkAiCore, input, lambd, bias, output);

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(Shrink, OP_INPUT(input), OP_OUTPUT(output), OP_ATTR(lambd, bias));
    OP_CHECK(ret == ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "ShrinkAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."), return nullptr);
    return output;
}

const aclTensor* Shrink(const aclTensor* input, float lambd, float bias, aclOpExecutor* executor){
    auto output = executor->AllocTensor(input->GetViewShape(), input->GetDataType());

    if(IsAiCoreSupport(input->GetDataType())) {
        return ShrinkAiCore(input, lambd, bias, output, executor);
    }
    return nullptr;
}
} //namespace l0op
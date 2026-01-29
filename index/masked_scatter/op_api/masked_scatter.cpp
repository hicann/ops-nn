/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file masked_scatter.cpp
 * \brief
 */
#include "masked_scatter.h"
#include "opdev/data_type_utils.h"
#include "opdev/op_def.h"
#include "opdev/op_executor.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/platform.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(MaskedScatter);
static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_UINT8, op::DataType::DT_INT8,
    op::DataType::DT_INT32, op::DataType::DT_INT16,   op::DataType::DT_INT64, op::DataType::DT_BF16};
static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST_910D = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_DOUBLE, op::DataType::DT_UINT8,
    op::DataType::DT_INT8,  op::DataType::DT_INT16,   op::DataType::DT_INT32,  op::DataType::DT_INT64,
    op::DataType::DT_BOOL,  op::DataType::DT_BF16};

static bool CheckShapeLimit(const aclTensor* self, const aclTensor* mask)
{
    return self->GetViewShape() == mask->GetViewShape();
}

// 根据芯片类型、dtype判断算子是否支持走aicore
static bool IsAiCoreSupport(const aclTensor* self, const aclTensor* mask)
{
    // 只需要判断dtype
    auto supportList = GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND950 ?
                           AICORE_DTYPE_SUPPORT_LIST_910D :
                           AICORE_DTYPE_SUPPORT_LIST;
    bool result = CheckType(self->GetDataType(), supportList);
    result = result && (mask->GetDataType() == op::DataType::DT_BOOL);
    result = result && CheckShapeLimit(self, mask);
    return result;
}

// AICORE算子kernel
static const aclTensor* MaskedScatterAiCore(
    const aclTensor* self, const aclTensor* mask, const aclTensor* source, aclTensor* output, aclOpExecutor* executor)
{
    L0_DFX(MaskedScatterAiCore, self, mask, source, output);

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(MaskedScatter, OP_INPUT(self, mask, source), OP_OUTPUT(output));
    OP_CHECK(
        ret == ACLNN_SUCCESS,
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "MaskedScatterAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."), return nullptr);
    return output;
}

// AICPU算子kernel
static const aclTensor* MaskedScatterAiCpu(
    const aclTensor* self, const aclTensor* mask, const aclTensor* source, aclTensor* maskedScatterOut,
    aclOpExecutor* executor)
{
    L0_DFX(MaskedScatterAiCpu, self, mask, source, maskedScatterOut);

    static internal::AicpuTaskSpace space("MaskedScatter");
    auto ret = ADD_TO_LAUNCHER_LIST_AICPU(
        MaskedScatter, OP_ATTR_NAMES(), OP_INPUT(self, mask, source), OP_OUTPUT(maskedScatterOut));
    OP_CHECK(
        ret == ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "MaskedScatterAiCpu ADD_TO_LAUNCHER_LIST_AICPU failed."),
        return nullptr);
    return maskedScatterOut;
}

const aclTensor* MaskedScatter(
    const aclTensor* self, const aclTensor* mask, const aclTensor* source, aclOpExecutor* executor)
{
    L0_DFX(MaskedScatter, self, mask, source);
    auto maskedScatterOut = executor->AllocTensor(self->GetViewShape(), self->GetDataType());
    if (IsAiCoreSupport(self, mask)) {
        return MaskedScatterAiCore(self, mask, source, maskedScatterOut, executor);
    } else {
        return MaskedScatterAiCpu(self, mask, source, maskedScatterOut, executor);
    }
}

} // namespace l0op

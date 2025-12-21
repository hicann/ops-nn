/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "opdev/data_type_utils.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "ascend_anti_quant_v2.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(AscendAntiQuantV2);

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_INT4, op::DataType::DT_INT8};

// 根据芯片类型、dtype判断算子是否支持走aicore
static inline bool IsAiCoreSupport(const aclTensor* self)
{
    // AscendAntiQuantV2只需要判断dtype
    return CheckType(self->GetDataType(), AICORE_DTYPE_SUPPORT_LIST);
}

const aclTensor* AscendAntiQuantV2Aicore(
    const aclTensor* x, const aclTensor* scale, const aclTensor* offset, int64_t dstType, bool sqrtMode, aclTensor* out,
    aclOpExecutor* executor)
{
    L0_DFX(AscendAntiQuantV2Aicore, x, scale, offset, dstType, sqrtMode, out)
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        AscendAntiQuantV2, OP_INPUT(x, scale, offset), OP_OUTPUT(out), OP_ATTR(dstType, sqrtMode));
    OP_CHECK(
        ret == ACLNN_SUCCESS,
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "AscendAntiQuantV2AiCore ADD_TO_LAUNCHER_LIST_AICORE failed."),
        return nullptr);
    return out;
}

const aclTensor* AscendAntiQuantV2(
    const aclTensor* x, const aclTensor* scale, const aclTensor* offset, int64_t dstType, bool sqrtMode,
    aclOpExecutor* executor)
{
    op::Shape outShape = x->GetViewShape();
    auto out = executor->AllocTensor(outShape, op::DataType(dstType));

    if (!IsAiCoreSupport(x)) {
        OP_LOGE(ACL_ERROR_INVALID_PARAM, "dtype of input x is not supported");
        return nullptr;
    }

    return AscendAntiQuantV2Aicore(x, scale, offset, dstType, sqrtMode, out, executor);
}
} // namespace l0op

/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/data_type_utils.h"
#include "opdev/platform.h"
#include "quantize.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(Quantize);

namespace {
static std::map<op::DataType, std::string> quantizeDataTypeToStr = {
    {op::DataType::DT_UINT8, "torch.quint8"},
    {op::DataType::DT_INT8, "torch.qint8"},
    {op::DataType::DT_INT32, "torch.qint32"},
    {op::DataType::DT_HIFLOAT8, "torch.hifloat8"},
    {op::DataType::DT_FLOAT8_E4M3FN, "torch.float8_e4m3fn"},
    {op::DataType::DT_FLOAT8_E5M2, "torch.float8_e5m2"}};
}; // namespace

static const aclTensor* QuantizeAiCore(
    const aclTensor* x, const aclTensor* scales, const aclTensor* zeroPoints, op::DataType dtype, int32_t axis,
    aclTensor* out, aclOpExecutor* executor)
{
    L0_DFX(QuantizeAiCore, x, scales, zeroPoints, dtype, axis, out);
    std::string dtypeStr = quantizeDataTypeToStr[dtype];
    auto ret =
        ADD_TO_LAUNCHER_LIST_AICORE(Quantize, OP_INPUT(x, scales, zeroPoints), OP_OUTPUT(out), OP_ATTR(dtypeStr, axis));
    OP_CHECK(
        ret == ACL_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "QuantizeAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."),
        return nullptr);
    return out;
}

const aclTensor* Quantize(
    const aclTensor* x, const aclTensor* scales, const aclTensor* zeroPoints, op::DataType dtype, int32_t axis,
    aclOpExecutor* executor)
{
    auto out = executor->AllocTensor(x->GetViewShape(), dtype);
    if (out == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Alloc quantize out tensor failed");
        return nullptr;
    }
    return QuantizeAiCore(x, scales, zeroPoints, dtype, axis, out, executor);
}
} // namespace l0op

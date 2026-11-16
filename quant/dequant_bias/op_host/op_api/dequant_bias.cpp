/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "dequant_bias.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(DequantBias);

const aclTensor* DequantBias(const aclTensor* x, const aclTensor* weightScale, const aclTensor* activateScale,
                             const aclTensor* bias, int64_t outputDtype, aclOpExecutor* executor, const aclTensor* out)
{
    L0_DFX(DequantBias, x, weightScale, activateScale, bias, outputDtype);
    DataType outputDataType = (outputDtype == 1) ? DataType::DT_FLOAT16 : DataType::DT_BF16;
    const aclTensor* outTensor = out;
    if (outTensor == nullptr) {
        auto outputShape = x->GetViewShape();
        outTensor = executor->AllocTensor(outputShape, outputDataType, Format::FORMAT_ND);
        if (outTensor == nullptr) {
            return nullptr;
        }
    }
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(DequantBias, OP_INPUT(x, weightScale, activateScale, bias),
                                           OP_OUTPUT(outTensor), OP_ATTR(outputDtype));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER_STATIC_WORKSPACE_INVALID, "DequantBias launch failed.");
        return nullptr;
    }
    return outTensor;
}

} // namespace l0op

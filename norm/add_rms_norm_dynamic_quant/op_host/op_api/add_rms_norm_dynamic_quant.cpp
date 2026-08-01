/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "add_rms_norm_dynamic_quant.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(AddRmsNormDynamicQuant);
constexpr int IDX_0 = 0;
constexpr int IDX_1 = 1;
constexpr int OUTPUT_MASK_LEN = 2;

inline static const std::array<aclTensor*, ADD_RMS_NORM_DYNAMIC_QUANT_OUT_NUM> AddRmsNormDynamicQuantAiCore(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* gamma, const aclTensor* smoothScale1Optional,
    const aclTensor* smoothScale2Optional, const aclTensor* betaOptional, double epsilon,
    const aclBoolArray* outputMask, int dstType, const aclTensor* xOutRef, const aclTensor* scale1OutRef,
    const aclTensor* scale2OutRef, aclOpExecutor* executor)
{
    OP_LOGD("AddRmsNormDynamicQuant L0 Start.");
    L0_DFX(AddRmsNormDynamicQuantAiCore, x1, x2, gamma, smoothScale1Optional, smoothScale2Optional, epsilon, outputMask,
           dstType, xOutRef, scale1OutRef, scale2OutRef);
    OP_LOGD("AddRmsNormDynamicQuant L0_DFX.");

    bool out1Flag = true;
    bool out2Flag = false;
    if (outputMask != nullptr && outputMask->Size() == OUTPUT_MASK_LEN) {
        out1Flag = (*outputMask)[IDX_0];
        out2Flag = (*outputMask)[IDX_1];
    } else {
        out2Flag = (nullptr != smoothScale1Optional) && (nullptr != smoothScale2Optional);
    }
    Shape dummyShape({1});

    auto xOut = executor->AllocTensor(xOutRef->GetViewShape(), xOutRef->GetDataType(), op::Format::FORMAT_ND);

    aclTensor* y1Out = nullptr;
    aclTensor* scale1Out = nullptr;
    if (out1Flag) {
        y1Out = executor->AllocTensor(x1->GetViewShape(), op::DataType(dstType), op::Format::FORMAT_ND);
        scale1Out = executor->AllocTensor(scale1OutRef->GetViewShape(), scale1OutRef->GetDataType(),
                                          op::Format::FORMAT_ND);
    } else {
        y1Out = executor->AllocTensor(dummyShape, op::DataType(dstType), op::Format::FORMAT_ND);
        scale1Out = executor->AllocTensor(dummyShape, scale1OutRef->GetDataType(), op::Format::FORMAT_ND);
    }

    aclTensor* y2Out = nullptr;
    aclTensor* scale2Out = nullptr;
    if (out2Flag) {
        y2Out = executor->AllocTensor(x1->GetViewShape(), op::DataType(dstType), op::Format::FORMAT_ND);
        scale2Out = executor->AllocTensor(scale2OutRef->GetViewShape(), scale2OutRef->GetDataType(),
                                          op::Format::FORMAT_ND);
    } else {
        y2Out = executor->AllocTensor(dummyShape, op::DataType(dstType), op::Format::FORMAT_ND);
        scale2Out = executor->AllocTensor(dummyShape, scale2OutRef->GetDataType(), op::Format::FORMAT_ND);
    }

    OP_LOGD("y1Out=[%s], y2Out=[%s], xOut=[%s].", op::ToString(y1Out->GetViewShape()).GetString(),
            op::ToString(y2Out->GetViewShape()).GetString(), op::ToString(xOut->GetViewShape()).GetString());

    ADD_TO_LAUNCHER_LIST_AICORE(
        AddRmsNormDynamicQuant, OP_INPUT(x1, x2, gamma, smoothScale1Optional, smoothScale2Optional, betaOptional),
        OP_OUTPUT(y1Out, y2Out, xOut, scale1Out, scale2Out), OP_ATTR(static_cast<float>(epsilon), outputMask, dstType));
    OP_LOGI("AddRmsNormDynamicQuant Launch finish.");

    return {y1Out, y2Out, xOut, scale1Out, scale2Out};
}

const std::array<aclTensor*, ADD_RMS_NORM_DYNAMIC_QUANT_OUT_NUM> AddRmsNormDynamicQuant(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* gamma, const aclTensor* smoothScale1Optional,
    const aclTensor* smoothScale2Optional, const aclTensor* betaOptional, double epsilon,
    const aclBoolArray* outputMask, int dstType, const aclTensor* xOut, const aclTensor* scale1Out,
    const aclTensor* scale2Out, aclOpExecutor* executor)
{
    return AddRmsNormDynamicQuantAiCore(x1, x2, gamma, smoothScale1Optional, smoothScale2Optional, betaOptional,
                                        epsilon, outputMask, dstType, xOut, scale1Out, scale2Out, executor);
}

} // namespace l0op

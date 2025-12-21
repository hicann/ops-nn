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
 * \file ada_layer_norm_v2.cpp
 * \brief
 */
#include "ada_layer_norm_v2.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(AdaLayerNormV2);

const std::tuple<aclTensor*, aclTensor*, aclTensor*> AdaLayerNormV2(
    AdaLayerNormV2InputTensor inputTensor, float epsilon, aclOpExecutor* executor)
{
    L0_DFX(AdaLayerNormV2, inputTensor.x, inputTensor.scale, inputTensor.shift, inputTensor.weightOptional, inputTensor.biasOptional, epsilon);

    op::Shape xShape = inputTensor.x->GetViewShape();
    auto xDtype = inputTensor.x->GetDataType();
    auto xFormat = inputTensor.x->GetViewFormat();
    auto out = executor->AllocTensor(xShape, xDtype, xFormat);
    xShape[xShape.GetDimNum() - 1] = 1;
    auto mean = executor->AllocTensor(xShape, xDtype, xFormat);
    auto rstd = executor->AllocTensor(xShape, xDtype, xFormat);

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        AdaLayerNormV2, OP_INPUT(inputTensor.x, inputTensor.scale, inputTensor.shift, inputTensor.weightOptional, inputTensor.biasOptional), OP_OUTPUT(out, mean, rstd), OP_ATTR(epsilon));
    if (ret != ACL_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "AdaLayerNormV2AiCore ADD_TO_LAUNCHER_LIST_AICORE failed.");
        return std::tuple(nullptr, nullptr, nullptr);
    }
    return std::tie(out, mean, rstd);
}
} // namespace l0op
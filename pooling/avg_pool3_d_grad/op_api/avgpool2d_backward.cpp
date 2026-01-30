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
 * \file avgpool2d_backward.cpp
 * \brief
 */

#include "avgpool2d_backward.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/shape_utils.h"
#include "opdev/make_op_executor.h"

namespace l0op {

constexpr int64_t DIM_H = 2;
constexpr int64_t DIM_W = 3;
OP_TYPE_REGISTER(AvgPoolV2Grad);

const aclTensor* AvgPoolV2Grad(
    const aclTensor* self, const aclTensor* shapeOrigInput, const aclTensor* grad, const aclIntArray* ksize,
    const aclIntArray* strides, const std::string &paddingMode, const aclIntArray* pads, const std::string& dataFormat, 
    const bool globalPooling, bool ceilMode, bool exclusive, int divisorOverride, aclOpExecutor* executor)
{   

    L0_DFX(AvgPoolV2Grad, shapeOrigInput, grad, ksize, strides, paddingMode, pads, dataFormat, 
        globalPooling, ceilMode, exclusive, divisorOverride);

    auto selfShape = self->GetViewShape();

    int64_t outH = selfShape.GetDim(DIM_H);
    int64_t outW = selfShape.GetDim(DIM_W);
    
    auto outputShape = grad->GetViewShape();
    outputShape.SetDim(DIM_H, outH);
    outputShape.SetDim(DIM_W, outW);
    auto output = executor->AllocTensor(outputShape, grad->GetDataType(), grad->GetStorageFormat());
    CHECK_RET(output != nullptr, nullptr);

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(AvgPoolV2Grad, OP_INPUT(shapeOrigInput, grad), OP_OUTPUT(output),
    OP_ATTR(ksize, strides, paddingMode, pads, dataFormat, globalPooling, ceilMode, exclusive, divisorOverride));
    OP_CHECK(ret == ACLNN_SUCCESS,
    OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "AvgPoolV2Grad ADD_TO_LAUNCHER_LIST_AICORE failed."),
        return nullptr);
    return output;
}
} // namespace l0op
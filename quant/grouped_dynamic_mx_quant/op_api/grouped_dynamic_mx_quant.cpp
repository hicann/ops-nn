/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "grouped_dynamic_mx_quant.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/shape_utils.h"
#include "opdev/make_op_executor.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(GroupedDynamicMxQuant);
static constexpr size_t NUM_TWO = 2;
static op::Shape GetOutPutShape(const aclTensor* x, const aclTensor* groupIndex, int64_t blocksize) {
    op::Shape outputShape;
    auto axisDim0 = x->GetViewShape().GetDim(0);
    auto groupIndexDim = groupIndex->GetViewShape().GetDim(0);
    auto outputDim0 = (axisDim0 / (blocksize * NUM_TWO) + groupIndexDim);
    outputShape.AppendDim(outputDim0);
    outputShape.AppendDim(x->GetViewShape().GetDim(1));
    outputShape.AppendDim(NUM_TWO);
    return outputShape;
}

std::tuple<aclTensor*, aclTensor*> GroupedDynamicMxQuant(const aclTensor* x, const aclTensor* groupIndex, const char* roundMode,
                                                         int64_t dstType, int64_t blocksize,aclOpExecutor* executor) {
L0_DFX(GroupedDynamicMxQuant, x, groupIndex);
auto yOut = executor->AllocTensor(x->GetStorageShape(), x->GetViewShape(), op::DataType(dstType),
                                    x->GetStorageFormat(), x->GetOriginalFormat());

auto mxScaleShape = GetOutPutShape(x, groupIndex, blocksize);
auto mxScaleOut = executor->AllocTensor(mxScaleShape, op::DataType::DT_FLOAT8_E8M0);
if (yOut == nullptr || mxScaleOut == nullptr) {
    OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "alloc y or mxscale tensor failed.");
    return std::tie(yOut, mxScaleOut);
}

auto ret = ADD_TO_LAUNCHER_LIST_AICORE(GroupedDynamicMxQuant, OP_INPUT(x, groupIndex),
                                        OP_OUTPUT(yOut, mxScaleOut), OP_ATTR(roundMode, dstType, blocksize));
if (ret != ACLNN_SUCCESS) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "GroupedDynamicMxQuant launch kernel failed.");
    return std::tuple<aclTensor*, aclTensor*>(nullptr, nullptr);
}
return std::tuple<aclTensor*, aclTensor*>(yOut, mxScaleOut);
}

}  // namespace l0op
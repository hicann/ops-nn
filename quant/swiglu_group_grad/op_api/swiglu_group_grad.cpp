/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "swiglu_group_grad.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_def.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(SwigluGroupGrad);

std::array<aclTensor*, SWIGLU_GROUP_GRAD_OUT_NUM> SwigluGroupGrad(const aclTensor* gradY, const aclTensor* x,
                                                                  const aclTensor* weightOptional,
                                                                  const aclTensor* yOriginOptional,
                                                                  const aclTensor* groupIndexOptional, float clampLimit,
                                                                  aclOpExecutor* executor)
{
    L0_DFX(SwigluGroupGrad, gradY, x, weightOptional, yOriginOptional, groupIndexOptional, clampLimit);

    // Allocate gradXOut: shape = x.shape, dtype = gradY.dtype, format = ND
    auto gradXOut = executor->AllocTensor(x->GetViewShape(), gradY->GetDataType(), Format::FORMAT_ND);
    if (gradXOut == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "alloc gradXOut tensor failed.");
        return {nullptr, nullptr};
    }

    // Allocate gradWeightOutOptional when weight is present
    // When not present, allocate a dummy one-element tensor (the kernel will ignore it)
    aclTensor* gradWeightOutOptional = nullptr;
    if (weightOptional != nullptr) {
        gradWeightOutOptional = executor->AllocTensor(weightOptional->GetViewShape(), DataType::DT_FLOAT,
                                                      Format::FORMAT_ND);
        if (gradWeightOutOptional == nullptr) {
            OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "alloc gradWeightOutOptional tensor failed.");
            return {nullptr, nullptr};
        }
    } else {
        // Allocate a dummy one-element tensor for the kernel (unused)
        Shape dummyShape({1});
        gradWeightOutOptional = executor->AllocTensor(dummyShape, DataType::DT_FLOAT, Format::FORMAT_ND);
        if (gradWeightOutOptional == nullptr) {
            OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "alloc dummy gradWeightOutOptional tensor failed.");
            return {nullptr, nullptr};
        }
    }

    // Launch kernel with all inputs (including optional ones as nullptr when not present)
    // The kernel uses tiling data flags (isWeight, hasClamp, isGroupIndex) to determine
    // which optional inputs are valid, and template parameters to compile-time prune unused paths.
    auto retAicore = ADD_TO_LAUNCHER_LIST_AICORE(
        SwigluGroupGrad, OP_INPUT(gradY, x, weightOptional, yOriginOptional, groupIndexOptional),
        OP_OUTPUT(gradXOut, gradWeightOutOptional), OP_ATTR(clampLimit));
    OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(retAicore != ACLNN_SUCCESS, return {},
                                         "SwigluGroupGrad add to aicore launch list failed.");

    return {gradXOut, gradWeightOutOptional};
}

} // namespace l0op

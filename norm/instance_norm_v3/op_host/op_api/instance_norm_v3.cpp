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
#include "opdev/platform.h"
#include "instance_norm_v3.h"

using namespace op;

namespace l0op {

constexpr size_t DIM_2 = 2;
constexpr size_t DIM_3 = 3;

OP_TYPE_REGISTER(InstanceNormV3);

std::tuple<aclTensor*, aclTensor*, aclTensor*> InstanceNormV3(
    const aclTensor* x, const aclTensor* gamma, const aclTensor* beta, const char* dataFormat, double eps,
    aclOpExecutor* executor)
{
    OP_LOGD("Enter InstanceNorm Level 0");
    L0_DFX(InstanceNormV3, x, gamma, beta, dataFormat, eps);

    op::Shape reduceShape = x->GetStorageShape();
    reduceShape.SetDim(DIM_2, 1);
    reduceShape.SetDim(DIM_3, 1);

    auto y = executor->AllocTensor(x->GetViewShape(), x->GetDataType(), Format::FORMAT_ND);
    if (y == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "alloc out tensor y failed.");
        return std::make_tuple(nullptr, nullptr, nullptr);
    }
    auto mean = executor->AllocTensor(reduceShape, x->GetDataType(), Format::FORMAT_ND);
    if (mean == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "alloc out tensor mean failed.");
        return std::make_tuple(nullptr, nullptr, nullptr);
    }
    auto variance = executor->AllocTensor(reduceShape, x->GetDataType(), Format::FORMAT_ND);
    if (variance == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "alloc out tensor variance failed.");
        return std::make_tuple(nullptr, nullptr, nullptr);
    }

    ADD_TO_LAUNCHER_LIST_AICORE(
        InstanceNormV3, OP_INPUT(x, gamma, beta), OP_OUTPUT(y, mean, variance),
        OP_ATTR(dataFormat, static_cast<float>(eps)));

    OP_LOGD("Finish InstanceNorm level 0");
    return std::tuple<aclTensor*, aclTensor*, aclTensor*>(y, mean, variance);
}
} // namespace l0op

/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "max_pool3_d.h"

#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(MaxPool3D);

namespace {
void NormalizeNdc1hwc0Output(aclTensor* y)
{
    if (y == nullptr) {
        return;
    }
    if (y->GetViewShape().GetDimNum() == 6U && y->GetStorageFormat() != op::Format::FORMAT_NDC1HWC0) {
        y->SetOriginalFormat(op::Format::FORMAT_NDC1HWC0);
        y->SetViewFormat(op::Format::FORMAT_NDC1HWC0);
        y->SetStorageFormat(op::Format::FORMAT_NDC1HWC0);
    }
}

} // namespace

const aclTensor* MaxPool3D(const aclTensor* x, const aclIntArray* ksize, const aclIntArray* strides,
                           const std::string& padding, const aclIntArray* pads, const aclIntArray* dilation,
                           int64_t ceilMode, const std::string& dataFormat, aclTensor* y, aclOpExecutor* executor)
{
    L0_DFX(MaxPool3D, x, ksize, strides, padding, pads, dilation, ceilMode, dataFormat, y);
    NormalizeNdc1hwc0Output(y);
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(MaxPool3D, OP_INPUT(x), OP_OUTPUT(y),
                                           OP_ATTR(ksize, strides, padding, pads, dilation, ceilMode, dataFormat));
    OP_CHECK(ret == ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "MaxPool3D ADD_TO_LAUNCHER_LIST_AICORE failed."),
             return nullptr);
    return y;
}
} // namespace l0op

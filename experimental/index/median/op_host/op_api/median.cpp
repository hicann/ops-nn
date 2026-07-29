/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "median.h"
#include "opdev/data_type_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(Median);

std::tuple<const aclTensor*, const aclTensor*> Median(const aclTensor* self, int64_t dim, bool keepDim,
                                                      aclOpExecutor* executor)
{
    L0_DFX(Median, self, dim, keepDim);

    // 输出形状：保留 self 形状，去掉 dim 轴（kernel 已是末轴 reduce 语义，调用方保证 dim==末轴）
    auto selfShape = self->GetViewShape();
    int64_t dimNum = static_cast<int64_t>(selfShape.GetDimNum());
    if (!(dim == dimNum - 1 || dimNum == 0)) { // kernel 仅支持末轴 reduce；违反契约时尽早暴露
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Median kernel only supports last-axis reduce, dim=%ld, dimNum=%ld", dim,
                dimNum);
        return std::tuple<const aclTensor*, const aclTensor*>(nullptr, nullptr);
    }
    op::Shape outShape;
    for (int64_t i = 0; i < dimNum; ++i) {
        if (i == dim) {
            if (keepDim)
                outShape.AppendDim(1);
            continue;
        }
        outShape.AppendDim(selfShape.GetDim(i));
    }

    auto values = executor->AllocTensor(outShape, self->GetDataType(), Format::FORMAT_ND);
    auto indices = executor->AllocTensor(outShape, DataType::DT_INT32, Format::FORMAT_ND);
    if (values == nullptr || indices == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Median alloc out tensor failed.");
        return std::tuple<const aclTensor*, const aclTensor*>(nullptr, nullptr);
    }

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(Median, OP_INPUT(self), OP_OUTPUT(values, indices), OP_ATTR(dim, keepDim));
    if (ret != ACL_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Median ADD_TO_LAUNCHER_LIST_AICORE failed.");
        return std::tuple<const aclTensor*, const aclTensor*>(nullptr, nullptr);
    }

    return std::tie(values, indices);
}

} // namespace l0op

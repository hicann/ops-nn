/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. 
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "glu.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(Glu);

const aclTensor* Glu(
    const aclTensor* self, int64_t dim, aclOpExecutor* executor)
{
    L0_DFX(Glu, self, dim);

    op::Shape outShape = self->GetViewShape();
    size_t dimNum = outShape.GetDimNum();
    auto sliceDim = dim < 0 ? dimNum + dim : dim;
    const int64_t SLICE_NUM = 2;
    outShape.SetDim(sliceDim, outShape.GetDim(sliceDim) / SLICE_NUM);

    auto out = executor->AllocTensor(outShape, self->GetDataType(), Format::FORMAT_ND);
    if (out == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "alloc out tensor failed.");
        return nullptr;
    }

    ADD_TO_LAUNCHER_LIST_AICORE(
        Glu, OP_INPUT(self), OP_OUTPUT(out), OP_ATTR(dim));
    return out;
}

} // namespace l0op

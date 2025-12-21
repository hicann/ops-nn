/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "dilation.h"

#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;
namespace l0op {
const int kHDimNCHWORNC1HWC0Idx = 2;
const int kWDimNCHWORNC1HWC0Idx = 3;
const int kC0DimNC1HWC0Idx = 4;
const int kPADDINGUPIdx = 0;
const int kPADDINGDOWNIdx = 1;
const int kPADDINGLEFTIdx = 2;
const int kPADDINGRIGHTIdx = 3;
OP_TYPE_REGISTER(Dilation);
const aclTensor* Dilation(
    const aclTensor* x, const aclIntArray* dilations, const aclIntArray* pads, float paddingValue,
    aclOpExecutor* executor)
{
    L0_DFX(Dilation, x, dilations, pads, paddingValue);
    op::Shape y_view_shape;
    op::Shape y_storage_shape;
    auto x_storage_shape = x->GetStorageShape();
    auto x_view_shape = x->GetViewShape();
    int64_t y_shape_h = (x_view_shape.GetDim(kHDimNCHWORNC1HWC0Idx) - 1) * (*dilations)[kHDimNCHWORNC1HWC0Idx] + 1 +
                        (*pads)[kPADDINGUPIdx] + (*pads)[kPADDINGDOWNIdx];
    int64_t y_shape_w = (x_view_shape.GetDim(kWDimNCHWORNC1HWC0Idx) - 1) * (*dilations)[kWDimNCHWORNC1HWC0Idx] + 1 +
                        (*pads)[kPADDINGLEFTIdx] + (*pads)[kPADDINGRIGHTIdx];
    for (size_t i = 0; i < x_view_shape.GetDimNum(); i++) {
        y_storage_shape.AppendDim(
            (i == kHDimNCHWORNC1HWC0Idx) ? y_shape_h :
            (i == kWDimNCHWORNC1HWC0Idx) ? y_shape_w :
                                           x_storage_shape.GetDim(i));
        y_view_shape.AppendDim(
            (i == kHDimNCHWORNC1HWC0Idx) ? y_shape_h :
            (i == kWDimNCHWORNC1HWC0Idx) ? y_shape_w :
                                           x_view_shape.GetDim(i));
    }
    y_storage_shape.AppendDim(x_storage_shape.GetDim(kC0DimNC1HWC0Idx));

    auto y = executor->AllocTensor(
        y_storage_shape, y_view_shape, x->GetDataType(), x->GetStorageFormat(), x->GetOriginalFormat());
    ADD_TO_LAUNCHER_LIST_AICORE(Dilation, OP_INPUT(x), OP_OUTPUT(y), OP_ATTR(dilations, pads, paddingValue));
    return y;
}
} // namespace l0op
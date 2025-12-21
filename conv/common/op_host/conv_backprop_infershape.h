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
 * \file conv_backprop_infershape.h
 * \brief
 */

#ifndef OPS_CONV_BACKPROP_INFERSHAPE_H_
#define OPS_CONV_BACKPROP_INFERSHAPE_H_

#include <graph/utils/type_utils.h>
#include <exe_graph/runtime/infer_shape_context.h>
#include <log/log.h>
#include <util/shape_util.h>
#include <register/op_impl_registry.h>
#include "error_util.h"

namespace Ops {
namespace NN {
namespace Conv {
    ge::graphStatus InferShapeForConvBackprop(
        gert::InferShapeContext* context, size_t const_tensor_idx, const char* const_tensor_name, size_t dim_num);
    // Conv2DBackpropFilterV2 和 Conv2DBackpropFilterV3 共用
    ge::graphStatus InferShapeForConv2DBackpropFilter(gert::InferShapeContext* context);
    // Conv2DBackpropFilterV2 和 Conv2DBackpropFilterV3 共用
    ge::graphStatus InferDataTypeForConv2DBackpropFilter(gert::InferDataTypeContext* context);
    // Conv2DBackpropInputV2 和 Conv3DBackpropInputV2 共用
    ge::graphStatus InferDataTypeForConvBackpropInputV2(gert::InferDataTypeContext* context);
    // Conv2DTransposeV2 和 Conv3DTransposeV2 共用
    bool CheckOutputAllZero(const gert::Shape* shape);
    // Conv2DTransposeV2 和 Conv3DTransposeV2 共用
    ge::graphStatus InferDataTypeForConvTransposeV2(gert::InferDataTypeContext* context);
    ge::graphStatus InferShapeForConvBackpropExtend3D(
        gert::InferShapeContext* context, size_t const_tensor_idx, const char* const_tensor_name);
    ge::graphStatus SetOutputShapeDim(const gert::InferShapeContext* context, const gert::Tensor* const_tensor, gert::Shape* y_shape);
    int32_t GetConvBackpropIndex(ge::Format format);

}  // namespace Conv
}  // namespace NN
}  // namespace Ops

#endif // OPS_CONV_BACKPROP_INFERSHAPE_H_
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
 * \file matmul_compress_infershape.cpp
 * \brief
 */

#include <limits>
#include "log/log.h"
#include "runtime/infer_shape_context.h"
#include "register/op_impl_registry.h"
#include "error_util.h"

using namespace gert;
namespace {
static constexpr size_t DIMNUM3 = 3;

static ge::graphStatus InferShapeForMatmulCompress(InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Start to do InferShapeForMatmulCompress.");
    auto attrs = context->GetAttrs();
    const bool* trans_a = attrs->GetAttrPointer<bool>(0);
    const bool* trans_b = attrs->GetAttrPointer<bool>(1);
    auto shape_a = context->GetInputShape(0);
    auto shape_b = context->GetInputShape(1);
    auto shape_out = context->GetOutputShape(0);
    size_t dimNumA = shape_a->GetDimNum();
    shape_out->SetDimNum(dimNumA);
    if (dimNumA == DIMNUM3) {
        shape_out->SetDim(0, shape_a->GetDim(0));
        if (trans_a) {
            shape_out->SetDim(1, shape_a->GetDim(2));
        } else {
            shape_out->SetDim(1, shape_a->GetDim(1));
        }
        if (trans_b) {
            shape_out->SetDim(2, shape_b->GetDim(1));
        } else {
            shape_out->SetDim(2, shape_b->GetDim(2));
        }
    } else {
        if (trans_a) {
            shape_out->SetDim(0, shape_a->GetDim(1));
        } else {
            shape_out->SetDim(0, shape_a->GetDim(0));
        }
        if (trans_b) {
            shape_out->SetDim(1, shape_b->GetDim(0));
        } else {
            shape_out->SetDim(1, shape_b->GetDim(1));
        }
    }

    OP_LOGD(context->GetNodeName(), "End to do InferShapeForMatmulCompress.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeForMatmulCompress(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferDataTypeForMatmulCompress.");
    const ge::DataType a_data_type = context->GetInputDataType(0);
    ge::graphStatus ret = context->SetOutputDataType(0, a_data_type);
    OP_LOGD(context->GetNodeName(), "End to do InferDataTypeForMatmulCompress.");
    return ret;
}
} // namespace

namespace Ops::NN::MatMul {
IMPL_OP_INFERSHAPE(MatmulCompress)
    .InferShape(InferShapeForMatmulCompress)
    .InferDataType(InferDataTypeForMatmulCompress);
}

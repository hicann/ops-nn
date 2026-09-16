/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file update_tensor_desc_infershape.cpp
 * \brief UpdateTensorDesc 形状/类型推导：y.shape = tuple(attr shape)（覆写式强制，
 *   x 为占位输入不参与推导），y.dtype 恒 DT_INT64。
 *   早校验（fail-fast）：rank(attr shape) ∈ [1, kMaxRank]，每元素非负 int64，
 *   numel(attr shape) ≥ kDescSize。
 */

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "exe_graph/runtime/infer_datatype_context.h"
#include "op_common/log/log.h"
#include "../op_kernel/arch35/update_tensor_desc_tiling_data.h"

using namespace ge;

namespace ops {

static ge::graphStatus InferShape4UpdateTensorDesc(gert::InferShapeContext* context)
{
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const gert::TypedContinuousVector<int64_t>* shapeVec = attrs->GetListInt(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, shapeVec);

    const char* nodeName = context->GetNodeName();

    const int64_t rank = static_cast<int64_t>(shapeVec->GetSize());
    if (rank < 1 || rank > kMaxRank) {
        OP_LOGE(nodeName, "rank(attr shape)=%ld out of range [1, %ld]", rank, kMaxRank);
        return ge::GRAPH_FAILED;
    }
    const int64_t* dims = shapeVec->GetData();
    int64_t numel = 1;
    for (int64_t i = 0; i < rank; i++) {
        if (dims[i] < 0) {
            OP_LOGE(nodeName, "attr shape[%ld]=%ld is negative", i, dims[i]);
            return ge::GRAPH_FAILED;
        }
        numel *= dims[i];
    }
    if (numel < kDescSize) {
        OP_LOGE(nodeName, "numel(attr shape)=%ld below %ld", numel, kDescSize);
        return ge::GRAPH_FAILED;
    }

    gert::Shape* yShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);
    yShape->SetDimNum(0);
    for (int64_t i = 0; i < rank; i++) {
        yShape->AppendDim(dims[i]);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4UpdateTensorDesc(gert::InferDataTypeContext* context)
{
    return context->SetOutputDataType(0, ge::DT_INT64);
}

IMPL_OP_INFERSHAPE(UpdateTensorDesc)
    .InferShape(InferShape4UpdateTensorDesc)
    .InferDataType(InferDataType4UpdateTensorDesc);

} // namespace ops

/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "graph/types.h"
#include <vector>

using namespace ge;

namespace {

// GlobalLpPool: y = (sum(|x|^p along spatial dims))^(1/p), keepdims=True.
// Input: 4D (NCHW) or 5D (NCD0D1D2)
// Output: same rank, spatial dims set to 1
ge::graphStatus GlobalLpPoolInferShape(gert::InferShapeContext* context)
{
    const gert::Shape* xShape = context->GetInputShape(0);
    if (xShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    // Handle unknown rank: output shape also unknown
    if (xShape->GetDimNum() == static_cast<size_t>(UNKNOWN_DIM_NUM)) {
        gert::Shape* yShape = context->GetOutputShape(0);
        if (yShape != nullptr) {
            yShape->SetDimNum(static_cast<size_t>(UNKNOWN_DIM_NUM));
        }
        return ge::GRAPH_SUCCESS;
    }

    int64_t rank = static_cast<int64_t>(xShape->GetDimNum());
    if (rank != 4 && rank != 5) {
        return ge::GRAPH_FAILED;
    }

    gert::Shape* yShape = context->GetOutputShape(0);
    if (yShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    // keepdims=True: spatial dims [2, rank-1] → 1, N and C preserved
    yShape->SetDimNum(static_cast<size_t>(rank));
    for (int64_t i = 0; i < rank; ++i) {
        if (i < 2) {
            yShape->SetDim(static_cast<size_t>(i), xShape->GetDim(static_cast<size_t>(i)));
        } else {
            yShape->SetDim(static_cast<size_t>(i), 1);
        }
    }

    return ge::GRAPH_SUCCESS;
}

} // anonymous namespace

IMPL_OP_INFERSHAPE(GlobalLpPool).InferShape(GlobalLpPoolInferShape);

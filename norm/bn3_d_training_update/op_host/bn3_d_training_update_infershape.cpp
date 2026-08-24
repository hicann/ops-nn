/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file bn3_d_training_update_infershape.cpp
 * \brief
 */
#include "register/op_impl_registry.h" // IMPL_OP_INFERSHAPE, gert::InferShapeContext
#include "util/shape_util.h"           // Ops::Base::IsUnknownRank / SetUnknownRank

using namespace ge;

namespace ops {

static ge::graphStatus InferShapeForBN3DTrainingUpdate(gert::InferShapeContext* context)
{
    // y (output 0) follows x (input 0) shape. UNKNOWN_RANK(-2) propagates as
    // unknown rank (plain copy would materialize the -2 marker into a bogus
    // rank-1 shape).
    const gert::Shape* xShape = context->GetInputShape(0);
    gert::Shape* yShape = context->GetOutputShape(0);
    if (xShape != nullptr && yShape != nullptr) {
        if (Ops::Base::IsUnknownRank(*xShape)) {
            Ops::Base::SetUnknownRank(*yShape);
        } else {
            *yShape = *xShape;
        }
    }

    // Statistics outputs (1..4) follow sum (input 1) shape (-1/-2 same rule).
    const gert::Shape* sumShape = context->GetInputShape(1);
    for (size_t i = 1; i < 5; ++i) {
        gert::Shape* outShape = context->GetOutputShape(i);
        if (sumShape != nullptr && outShape != nullptr) {
            if (Ops::Base::IsUnknownRank(*sumShape)) {
                Ops::Base::SetUnknownRank(*outShape);
            } else {
                *outShape = *sumShape;
            }
        }
    }
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(BN3DTrainingUpdate).InferShape(InferShapeForBN3DTrainingUpdate);

} // namespace ops

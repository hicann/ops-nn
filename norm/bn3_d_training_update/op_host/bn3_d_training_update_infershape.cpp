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
 * \brief BN3DTrainingUpdate shape inference。
 */

#include "log/log.h"
#include "register/op_impl_registry.h" // IMPL_OP_INFERSHAPE, gert::InferShapeContext
#include "util/shape_util.h"           // Ops::Base::IsUnknownRank / SetUnknownRank

using namespace ge;

namespace ops {
namespace {
constexpr size_t kInputX = 0;
constexpr size_t kInputSum = 1;
constexpr size_t kOutputY = 0;
constexpr size_t kOutputStatFirst = 1;
constexpr size_t kOutputStatLast = 4;

inline void CopyShapeWithUnknownRank(const gert::Shape* src, gert::Shape* dst)
{
    if (Ops::Base::IsUnknownRank(*src)) {
        Ops::Base::SetUnknownRank(*dst);
    } else {
        *dst = *src;
    }
}
} // namespace

static ge::graphStatus InferShapeForBN3DTrainingUpdate(gert::InferShapeContext* context)
{
    const gert::Shape* xShape = context->GetInputShape(kInputX);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    gert::Shape* yShape = context->GetOutputShape(kOutputY);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);
    CopyShapeWithUnknownRank(xShape, yShape);

    const gert::Shape* sumShape = context->GetInputShape(kInputSum);
    OP_CHECK_NULL_WITH_CONTEXT(context, sumShape);
    for (size_t i = kOutputStatFirst; i <= kOutputStatLast; ++i) {
        gert::Shape* outShape = context->GetOutputShape(i);
        OP_CHECK_NULL_WITH_CONTEXT(context, outShape);
        CopyShapeWithUnknownRank(sumShape, outShape);
    }
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(BN3DTrainingUpdate).InferShape(InferShapeForBN3DTrainingUpdate);

} // namespace ops

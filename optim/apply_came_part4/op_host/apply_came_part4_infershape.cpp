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
 * \file apply_came_part4_infershape.cpp
 * \brief ApplyCamePart4 shape/dtype inference（对齐 canndev Infershape4ApplyCamePart4）
 *
 * Inputs:  param_in(0), m(1), r_in(2), c_in(3), ...
 * Outputs: param_out(0) <- shape(param_in), r_out(1) <- shape(r_in), c_out(2) <- shape(c_in)
 * Dtypes:  param_out <- dtype(param_in), r_out <- dtype(r_in), c_out <- dtype(c_in)
 */

#include <string>

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "op_common/op_host/util/shape_util.h"
#include "op_common/log/log.h"

using namespace ge;

namespace ops {

namespace {
constexpr size_t kIdxInParam = 0;
constexpr size_t kIdxInM = 1;
constexpr size_t kIdxInR = 2;
constexpr size_t kIdxInC = 3;
constexpr size_t kIdxOutParam = 0;
constexpr size_t kIdxOutR = 1;
constexpr size_t kIdxOutC = 2;
constexpr size_t kMaxDimNum = 2;
} // namespace

static ge::graphStatus Infershape4ApplyCamePart4(gert::InferShapeContext* context)
{
    const gert::Shape* paramShape = context->GetInputShape(kIdxInParam);
    const gert::Shape* mShape = context->GetInputShape(kIdxInM);
    const gert::Shape* rShape = context->GetInputShape(kIdxInR);
    const gert::Shape* cShape = context->GetInputShape(kIdxInC);
    gert::Shape* paramOutShape = context->GetOutputShape(kIdxOutParam);
    gert::Shape* rOutShape = context->GetOutputShape(kIdxOutR);
    gert::Shape* cOutShape = context->GetOutputShape(kIdxOutC);
    if (paramShape == nullptr || mShape == nullptr || rShape == nullptr || cShape == nullptr ||
        paramOutShape == nullptr || rOutShape == nullptr || cOutShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    // dynamic rank: preserve the unknown-rank contract on all outputs
    if (Ops::Base::IsUnknownRank(*paramShape) || Ops::Base::IsUnknownRank(*mShape) ||
        Ops::Base::IsUnknownRank(*rShape) || Ops::Base::IsUnknownRank(*cShape)) {
        Ops::Base::SetUnknownRank(*paramOutShape);
        Ops::Base::SetUnknownRank(*rOutShape);
        Ops::Base::SetUnknownRank(*cOutShape);
        return ge::GRAPH_SUCCESS;
    }

    if (paramShape->GetDimNum() > kMaxDimNum || mShape->GetDimNum() > kMaxDimNum) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            context->GetNodeName(), "param_in,m",
            std::to_string(paramShape->GetDimNum()) + "D," + std::to_string(mShape->GetDimNum()) + "D",
            "param_in and m must be no greater than 2D.");
        return ge::GRAPH_FAILED;
    }

    *paramOutShape = *paramShape;
    *rOutShape = *rShape;
    *cOutShape = *cShape;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4ApplyCamePart4(gert::InferDataTypeContext* context)
{
    // 输出 dtype 分别跟随 param/r/c 输入
    context->SetOutputDataType(kIdxOutParam, context->GetInputDataType(kIdxInParam));
    context->SetOutputDataType(kIdxOutR, context->GetInputDataType(kIdxInR));
    context->SetOutputDataType(kIdxOutC, context->GetInputDataType(kIdxInC));
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(ApplyCamePart4).InferShape(Infershape4ApplyCamePart4).InferDataType(InferDataType4ApplyCamePart4);

} // namespace ops

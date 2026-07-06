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
 * \file fake_quant_with_min_max_vars_per_channel_infershape.cpp
 * \brief FakeQuantWithMinMaxVarsPerChannel 形状/dtype 推导
 *
 * y.shape = x.shape
 * y.dtype = x.dtype (fp32)
 */

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "op_common/log/log.h"

using namespace ge;

namespace ops {

static ge::graphStatus InferShape4FakeQuantWithMinMaxVarsPerChannel(gert::InferShapeContext* context)
{
    const gert::Shape* xShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);

    gert::Shape* yShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);

    *yShape = *xShape;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4FakeQuantWithMinMaxVarsPerChannel(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(FakeQuantWithMinMaxVarsPerChannel)
    .InferShape(InferShape4FakeQuantWithMinMaxVarsPerChannel)
    .InferDataType(InferDataType4FakeQuantWithMinMaxVarsPerChannel);

} // namespace ops

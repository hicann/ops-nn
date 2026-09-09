/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 *
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

/*!
 * \file hard_swish_grad_infershape.cpp
 * \brief HardSwishGrad shape inference
 *
 * Output shape equals the first input (grad_output) shape.
 */

#include <string>

#include "util/shape_util.h"
#include "infershape_elewise_util.h"
#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "log/log.h"

using namespace ge;

namespace ops {

static ge::graphStatus InferShape4HardSwishGrad(gert::InferShapeContext* context)
{
    const ge::graphStatus ret = Ops::Base::InferShape4Elewise(context);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    const gert::Shape* outputShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputShape);
    const std::string outputShapeText = Ops::Base::ToString(*outputShape);
    OP_LOGI(context->GetNodeName(), "[InferShape] HardSwishGrad output0 shape=%s", outputShapeText.c_str());
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(HardSwishGrad).InferShape(InferShape4HardSwishGrad);

} // namespace ops

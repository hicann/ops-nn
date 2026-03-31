/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file inplace_index_fill_onfershape.cpp
 * \brief
 */

#include "log/log.h"
#include "op_host/infershape_elewise_util.h"
#include "register/op_impl_registry.h"

using namespace ge;
namespace ops {
static graphStatus InplaceIndexFillInferDtype(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InplaceIndexFillInferDtype rt2.0");

    auto gradDtype = context->GetInputDataType(0);

    context->SetOutputDataType(0, gradDtype);
    OP_LOGD(context->GetNodeName(), "Datatype of grad is:%s", Ops::Base::ToString(gradDtype).c_str());
    OP_LOGD(context->GetNodeName(), "End to do InplaceIndexFillInferDtype");

    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(InplaceIndexFill)
    .InferShape(Ops::Base::InferShape4Elewise)
    .InferDataType(InplaceIndexFillInferDtype);
} // namespace ops
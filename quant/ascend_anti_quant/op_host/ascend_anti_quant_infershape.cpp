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
 * \file ascend_anti_quant_infershape.cpp
 * \brief
 */
#include "infershape_elewise_util.h"
#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;
using namespace Ops::Base;

namespace ops {
constexpr size_t ATTR_INDEX_OF_DTYPE = 2;

static ge::graphStatus InferDataTypeForAscendAntiQuant(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferDataTypeForAscendAntiQuant");
    const auto& attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

    const int32_t* dstDtype = attrs->GetAttrPointer<int32_t>(ATTR_INDEX_OF_DTYPE);
    OP_CHECK_NULL_WITH_CONTEXT(context, dstDtype);
    ge::DataType outDtype = static_cast<ge::DataType>(*dstDtype);
    context->SetOutputDataType(0, outDtype);
    OP_LOGD(context->GetNodeName(), "End to do InferDataTypeForAscendAntiQuant");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(AscendAntiQuant).InferShape(InferShape4Elewise).InferDataType(InferDataTypeForAscendAntiQuant);
} // namespace ops

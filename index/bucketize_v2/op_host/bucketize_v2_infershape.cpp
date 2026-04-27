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
 * \file bucketize_v2_infershape.cpp
 * \brief
 */
#include "op_host/infershape_elewise_util.h"
#include "log/log.h"
#include "graph/utils/type_utils.h"
#include "register/op_impl_registry.h"

using namespace ge;
namespace ops {
static ge::graphStatus InferDataTypeForBucketizeV2(gert::InferDataTypeContext* context)
{
    if (context == nullptr) {
        return GRAPH_FAILED;
    }
    auto attrsPtr = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrsPtr);
    const bool* outInt32 = attrsPtr->GetAttrPointer<bool>(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outInt32);
    ge::DataType outDtype = *outInt32 ? ge::DT_INT32 : ge::DT_INT64;

    context->SetOutputDataType(0, outDtype);

    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(BucketizeV2)
    .InferShape(Ops::Base::InferShape4Elewise)
    .InferDataType(InferDataTypeForBucketizeV2);
}  // namespace ops
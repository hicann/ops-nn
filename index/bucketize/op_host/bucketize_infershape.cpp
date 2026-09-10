/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "op_host/infershape_elewise_util.h"
#include "log/log.h"
#include "register/op_impl_registry.h"

using namespace ge;

namespace ops {
static ge::graphStatus InferDataTypeForBucketize(gert::InferDataTypeContext* context)
{
    if (context == nullptr) {
        return GRAPH_FAILED;
    }
    const auto* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const ge::DataType* dtype = attrs->GetAttrPointer<ge::DataType>(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, dtype);
    if ((*dtype != ge::DT_INT32) && (*dtype != ge::DT_INT64)) {
        OP_LOGE(context->GetNodeName(), "Bucketize output dtype must be int32 or int64, got %d",
                static_cast<int>(*dtype));
        return GRAPH_FAILED;
    }
    context->SetOutputDataType(0, *dtype);
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(Bucketize).InferShape(Ops::Base::InferShape4Elewise).InferDataType(InferDataTypeForBucketize);
} // namespace ops

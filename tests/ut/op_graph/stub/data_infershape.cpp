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
#include <unistd.h>
#include <limits.h>
#include "platform/platform_info.h"
#include "base/registry/op_impl_space_registry_v2.h"

using namespace ge;
namespace ops {
static ge::graphStatus InferShape4Data(gert::InferShapeContext* context)
{
    const auto x_shape = context->GetInputShape(0);
    auto y_shape = context->GetOutputShape(0);
    *y_shape = *x_shape;
    return GRAPH_SUCCESS;
}
IMPL_OP(Data).InferShape(InferShape4Data).InferOutDataTypeSameWithFirstInput();
}//namespace ops
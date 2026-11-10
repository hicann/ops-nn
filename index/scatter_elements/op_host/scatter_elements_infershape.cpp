/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "log/log.h"
#include "register/op_impl_registry.h"
#include "infershape_elewise_util.h"

using namespace ge;

namespace ops {
static ge::graphStatus InferShape4ScatterElements(gert::InferShapeContext* context)
{
    OP_LOGI(context->GetNodeName(), "***runtime2.0 inferShape***");
    return Ops::Base::InferShape4Elewise(context);
}

IMPL_OP_INFERSHAPE(ScatterElements).InferShape(InferShape4ScatterElements);
} // namespace ops

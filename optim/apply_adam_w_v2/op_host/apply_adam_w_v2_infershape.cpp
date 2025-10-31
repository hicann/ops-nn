/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file apply_adam_w_v2_infershape.cpp
 * \brief
 */
#include "log/log.h"
#include "register/op_impl_registry.h"

using namespace ge;

namespace ops {
static ge::graphStatus InferShape4ApplyAdamWV2(gert::InferShapeContext* context)
{
    OP_LOGD(context, "Begin to do InferShape4ApplyAdamWV2");
    OP_LOGD(context, "End to do InferShape4AdvanceStep");
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType4ApplyAdamWV2(gert::InferDataTypeContext* context)
{
    OP_LOGD(context, "Begin to do InferDataType4ApplyAdamWV2");
    OP_LOGD(context, "End to do InferDataType4ApplyAdamWV2");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(ApplyAdamWV2).InferShape(InferShape4ApplyAdamWV2).InferDataType(InferDataType4ApplyAdamWV2);
} // namespace ops
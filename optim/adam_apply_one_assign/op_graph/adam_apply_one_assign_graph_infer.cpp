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
 * \file adam_apply_one_assign_graph_infer.cpp
 * \brief 
 */
#include "log/log.h"
#include "register/op_impl_registry.h"

using namespace ge;
namespace ops {

static ge::graphStatus InferDataType4AdamApplyOneAssign(gert::InferDataTypeContext* context)
{
    const ge::DataType dtype = context->GetInputDataType(0);
    context->SetOutputDataType(0, dtype);
    context->SetOutputDataType(1, dtype);
    context->SetOutputDataType(2, dtype);
    return GRAPH_SUCCESS;
}

IMPL_OP(AdamApplyOneAssign).InferDataType(InferDataType4AdamApplyOneAssign);

} // namespace ops

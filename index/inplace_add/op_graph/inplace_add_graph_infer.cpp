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
 * \file inplace_add_graph_infer.cpp
 * \brief inplace_add operator graph infer resource
 */

#include "register/op_impl_registry.h"

namespace ops {
using namespace ge;

static constexpr int64_t INPUT_X_INDEX = 0;
static constexpr int64_t OUTPUT_Y_INDEX = 0;

static ge::graphStatus InferDataTypeInplaceAdd(gert::InferDataTypeContext* context)
{
    ge::DataType xDtype = context->GetInputDataType(INPUT_X_INDEX);
    return context->SetOutputDataType(OUTPUT_Y_INDEX, xDtype);
}

IMPL_OP(InplaceAdd).InferDataType(InferDataTypeInplaceAdd);

} // namespace ops

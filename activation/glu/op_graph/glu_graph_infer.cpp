/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file glu_graph_infer.cpp
 * \brief GLU InferDataType implementation (graph mode)
 */

#include "log/log.h"
#include "register/op_impl_registry.h"

using namespace ge;

namespace {
constexpr size_t GLU_IN_X = 0;
constexpr size_t GLU_OUT_Y = 0;
} // namespace

namespace ops {
static ge::graphStatus InferDataTypeForGlu(gert::InferDataTypeContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("GLU", "InferDataType context is nullptr."), return ge::GRAPH_FAILED);
    OP_LOGD(context->GetNodeName(), "Begin to do InferDataTypeForGlu");
    const ge::DataType dtype = context->GetInputDataType(GLU_IN_X);
    ge::graphStatus ret = context->SetOutputDataType(GLU_OUT_Y, dtype);
    OP_LOGD(context->GetNodeName(), "End to do InferDataTypeForGlu");
    return ret;
}

IMPL_OP(GLU).InferDataType(InferDataTypeForGlu);
} // namespace ops

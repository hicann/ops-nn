/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file index_put_v2_graph_infer.cpp
 * \brief index_put_v2 operater graph infer resource
 */

#include "log/log.h"
#include "register/op_impl_registry.h"

using namespace ge;

namespace ops {
namespace {
constexpr size_t kInputIndex0 = 0U;
constexpr size_t kOutputIndex0 = 0U;
} // namespace

static ge::graphStatus InferDataTypeIndexPutV2(gert::InferDataTypeContext* context)
{
    OP_LOGI(context->GetNodeName(), "Begin to do InferDataTypeIndexPutV2");
    DataType xDataType = context->GetInputDataType(kInputIndex0);
    return context->SetOutputDataType(kOutputIndex0, xDataType);
}

IMPL_OP(IndexPutV2).InferDataType(InferDataTypeIndexPutV2);
} // namespace ops

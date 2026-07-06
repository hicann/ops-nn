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
 * \file acts_ulq_graph_infer.cpp
 * \brief acts_ulq operator graph infer resource
 */
#include "log/log.h"
#include "register/op_impl_registry.h"

using namespace ge;
namespace ops {

static constexpr uint32_t INPUT_DATA_IDX = 0;
static constexpr uint32_t OUTPUT_IDX = 0;
static constexpr uint32_t OUTPUT_CLAMP_MIN_MASK_IDX = 1;
static constexpr uint32_t OUTPUT_CLAMP_MAX_MASK_IDX = 2;
static constexpr uint32_t OUTPUT_X_CLAMPED_LOSS_IDX = 3;

static ge::graphStatus InferDataTypeForActsUlq(gert::InferDataTypeContext* context)
{
    const ge::DataType dataDtype = context->GetInputDataType(INPUT_DATA_IDX);
    context->SetOutputDataType(OUTPUT_IDX, dataDtype);
    context->SetOutputDataType(OUTPUT_CLAMP_MIN_MASK_IDX, dataDtype);
    context->SetOutputDataType(OUTPUT_CLAMP_MAX_MASK_IDX, dataDtype);
    context->SetOutputDataType(OUTPUT_X_CLAMPED_LOSS_IDX, dataDtype);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(ActsULQ).InferDataType(InferDataTypeForActsUlq);
} // namespace ops

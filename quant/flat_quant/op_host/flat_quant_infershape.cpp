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
 * \file flat_quant.cc
 * \brief
 */
#include "log/log.h"
#include "register/op_impl_registry.h"

using namespace ge;
namespace ops {

static constexpr size_t FLATQUANT_K_IDX = 0;
static constexpr size_t FLATQUANT_M_IDX = 1;
static constexpr size_t FLATQUANT_N_IDX = 2;

static ge::graphStatus InferShape4FlatQuant(gert::InferShapeContext *context)
{
    const gert::Shape *xShape = context->GetInputShape(0);
    gert::Shape *outShape = context->GetOutputShape(0);
    gert::Shape *qScaleShape = context->GetOutputShape(1);
    if ((xShape == nullptr) || (outShape == nullptr) || (qScaleShape == nullptr)) {
        return ge::GRAPH_FAILED;
    }
    outShape->SetDimNum(0);
    outShape->AppendDim(xShape->GetDim(FLATQUANT_K_IDX));
    outShape->AppendDim(xShape->GetDim(FLATQUANT_M_IDX));
    outShape->AppendDim(xShape->GetDim(FLATQUANT_N_IDX));

    qScaleShape->SetDimNum(0);
    qScaleShape->AppendDim(xShape->GetDim(FLATQUANT_K_IDX));
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4FlatQuant(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, ge::DT_INT4);
    context->SetOutputDataType(1, ge::DT_FLOAT);
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(FlatQuant).InferShape(InferShape4FlatQuant).InferDataType(InferDataType4FlatQuant);
}  // namespace ops

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
 * \file bn3_d_training_update_graph_infer.cpp
 * \brief
 */
#include "register/op_impl_registry.h" // IMPL_OP, gert::InferDataTypeContext

using namespace ge;

namespace ops {

static ge::graphStatus InferDataTypeForBN3DTrainingUpdate(gert::InferDataTypeContext* context)
{
    // y (output 0) follows x (input 0) dtype.
    context->SetOutputDataType(0, context->GetInputDataType(0));
    // Statistics outputs (1..4) follow sum (input 1) dtype.
    const ge::DataType sumDtype = context->GetInputDataType(1);
    for (size_t i = 1; i < 5; ++i) {
        context->SetOutputDataType(i, sumDtype);
    }
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(BN3DTrainingUpdate).InferDataType(InferDataTypeForBN3DTrainingUpdate);

} // namespace ops

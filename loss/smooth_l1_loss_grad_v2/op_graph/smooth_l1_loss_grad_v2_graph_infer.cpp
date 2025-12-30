/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file smooth_l1_loss_grad_v2_graph_infer.cpp
 * \brief
 */
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "infershape_broadcast_util.h"

using namespace ge;
namespace ops {
  
constexpr uint32_t INPUT_PREDICT_IDX = 0;
constexpr uint32_t OUTPUT_LOSS_GRAD_IDX = 0;

static ge::graphStatus InferDataTypeForSmoothL1LossGradV2(gert::InferDataTypeContext *context)
{
    const ge::DataType predictDtype = context->GetInputDataType(INPUT_PREDICT_IDX);
    context->SetOutputDataType(OUTPUT_LOSS_GRAD_IDX, predictDtype);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(SmoothL1LossGradV2).InferDataType(InferDataTypeForSmoothL1LossGradV2);
}  // namespace ops

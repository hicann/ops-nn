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
 * \file multi_scale_deformable_attention_grad_infershape.cpp
 * \brief
 */
#include "log/log.h"
#include "platform/platform_infos_def.h"
#include "platform/platform_info.h"
#include "register/op_impl_registry.h"
#include "register/op_def_registry.h"
#include "util/math_util.h"

using namespace ge;
namespace ops {
static ge::graphStatus InferShapeForMultiScaleDeformableAttentionGrad(gert::InferShapeContext *context)
{
    const gert::Shape *valueShape = context->GetInputShape(0);
    if (valueShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const gert::Shape *samplingLocationsShape = context->GetInputShape(3);
    if (samplingLocationsShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    gert::Shape *gradValueShape = context->GetOutputShape(0);
    gert::Shape *gradSampleLocShape = context->GetOutputShape(1);
    gert::Shape *gradAttnWeightShape = context->GetOutputShape(2);
    if ((gradValueShape == nullptr) || (gradSampleLocShape == nullptr) || (gradAttnWeightShape == nullptr)) {
        return ge::GRAPH_FAILED;
    }
    gradValueShape->SetDimNum(0);
    gradValueShape->AppendDim(valueShape->GetDim(0));
    gradValueShape->AppendDim(valueShape->GetDim(1));
    gradValueShape->AppendDim(valueShape->GetDim(2)); // dim 2
    gradValueShape->AppendDim(valueShape->GetDim(3)); // dim 3
    gradSampleLocShape->SetDimNum(0);
    gradSampleLocShape->AppendDim(samplingLocationsShape->GetDim(0));
    gradSampleLocShape->AppendDim(samplingLocationsShape->GetDim(1));
    gradSampleLocShape->AppendDim(samplingLocationsShape->GetDim(2)); // dim 2
    gradSampleLocShape->AppendDim(samplingLocationsShape->GetDim(3)); // dim 3
    gradSampleLocShape->AppendDim(samplingLocationsShape->GetDim(4)); // dim 4
    gradSampleLocShape->AppendDim(samplingLocationsShape->GetDim(5)); // dim 5
    gradAttnWeightShape->SetDimNum(0);
    gradAttnWeightShape->AppendDim(samplingLocationsShape->GetDim(0));
    gradAttnWeightShape->AppendDim(samplingLocationsShape->GetDim(1));
    gradAttnWeightShape->AppendDim(samplingLocationsShape->GetDim(2)); // dim 2
    gradAttnWeightShape->AppendDim(samplingLocationsShape->GetDim(3)); // dim 3
    gradAttnWeightShape->AppendDim(samplingLocationsShape->GetDim(5)); // dim5
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeMultiScaleDeformableAttentionGrad(gert::InferDataTypeContext* context)
{
    const ge::DataType valueDtype = context->GetInputDataType(0);
    context->SetOutputDataType(0, valueDtype);
    context->SetOutputDataType(1, valueDtype);
    context->SetOutputDataType(2, valueDtype); // 2 grad_attn_weight dtype
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MultiScaleDeformableAttentionGrad).InferShape(InferShapeForMultiScaleDeformableAttentionGrad).InferDataType(InferDataTypeMultiScaleDeformableAttentionGrad);
}  // namespace ops

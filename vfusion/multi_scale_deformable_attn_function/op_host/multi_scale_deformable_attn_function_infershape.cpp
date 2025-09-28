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
 * \file multi_scale_deformable_attn_function_infershape.cpp
 * \brief
 */
#include "log/log.h"
#include "platform/platform_infos_def.h"
#include "platform/platform_info.h"
#include "register/op_def_registry.h"
#include "util/math_util.h"
#include "register/op_impl_registry.h"
#include "util/shape_util.h"

using namespace ge;
namespace {
constexpr size_t INPUT_VALUE_INDEX = 0;
constexpr size_t INPUT_LOCAT_INDEX = 3;
constexpr size_t OUTPUT_Y_INDEX = 0;
constexpr size_t INPUT_VALUE_DIM_0 = 0;
constexpr size_t INPUT_VALUE_DIM_3 = 3;
constexpr size_t INPUT_LOCAT_DIM_5 = 5;
constexpr size_t INPUT_LOCAT_DIM_2 = 2;
constexpr size_t INPUT_LOCAT_DIM_1 = 1;
constexpr size_t OUTPUT_DIM_0 = 0;
constexpr size_t OUTPUT_DIM_1 = 1;
constexpr size_t OUTPUT_DIM_2 = 2;
}  // namespace

namespace ops {
static ge::graphStatus InferShapeForMultiScaleDeformableAttnFunction(gert::InferShapeContext *context)
{
    const gert::Shape *valueShape = context->GetInputShape(INPUT_VALUE_INDEX);
    if (valueShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const gert::Shape *samplingLocationsShape = context->GetInputShape(INPUT_LOCAT_INDEX);
    if (samplingLocationsShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    gert::Shape *yShape = context->GetOutputShape(OUTPUT_Y_INDEX);
    if (yShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    if (Ops::Base::IsUnknownRank(*valueShape)) {
        Ops::Base::SetUnknownRank(*yShape);
        return ge::GRAPH_SUCCESS;
    }
    bool isTranspose = samplingLocationsShape->GetDim(INPUT_LOCAT_DIM_1) < 32;
    uint64_t numHeads = INPUT_LOCAT_DIM_2;
    uint64_t numQueries = INPUT_LOCAT_DIM_1;
    if (isTranspose) {
        numHeads = INPUT_LOCAT_DIM_1;
        numQueries = INPUT_LOCAT_DIM_5;
    }
    const size_t outputRank = 3;
    yShape->SetDimNum(outputRank);

    yShape->SetDim(OUTPUT_DIM_0, valueShape->GetDim(INPUT_VALUE_DIM_0));
    yShape->SetDim(OUTPUT_DIM_1, samplingLocationsShape->GetDim(numQueries));
    if (samplingLocationsShape->GetDim(INPUT_LOCAT_DIM_1) == -1) {
        yShape->SetDim(OUTPUT_DIM_2, samplingLocationsShape->GetDim(INPUT_LOCAT_DIM_1));
    } else {
        yShape->SetDim(OUTPUT_DIM_2, samplingLocationsShape->GetDim(numHeads) * valueShape->GetDim(INPUT_VALUE_DIM_3)); // 1: dim 2; 3: dim 3
    }

    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeForMultiScaleDeformableAttnFunction(gert::InferDataTypeContext* context)
{
    const ge::DataType value_dtype = context->GetInputDataType(0);
    context->SetOutputDataType(0, value_dtype);
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MultiScaleDeformableAttnFunction).InferShape(InferShapeForMultiScaleDeformableAttnFunction).InferDataType(InferDataTypeForMultiScaleDeformableAttnFunction);
}  // namespace ops

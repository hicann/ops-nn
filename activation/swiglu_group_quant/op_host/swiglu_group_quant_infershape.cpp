/**
 * Copyright (c) 2026 Huawei Technologies
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file swiglu_group_quant_infershape.cpp
 * \brief InferShape implementation for SwiGLU Group Quant operator (quant_mode=3: HiF8 Dynamic Quant)
 */

#include "log/log.h"
#include "register/op_impl_registry.h"

using namespace ge;

namespace {
constexpr size_t INPUT_X = 0;
constexpr size_t INPUT_WEIGHT = 1;
constexpr size_t INPUT_GROUP_INDEX = 2;

constexpr size_t OUTPUT_Y = 0;
constexpr size_t OUTPUT_Y_SCALE = 1;
constexpr size_t OUTPUT_Y_ORIGIN = 2;

constexpr size_t ATTR_QUANT_MODE = 1;
constexpr size_t ATTR_OUTPUT_ORIGIN = 6;

constexpr int64_t QUANT_MODE_DYNAMIC = 3;
constexpr size_t SPLIT_NUM = 2;
}  // namespace

namespace ops {

static bool CheckQuantMode(gert::InferShapeContext* context)
{
    auto attrs = context->GetAttrs();
    if (attrs == nullptr) {
        OP_LOGE(context->GetNodeName(), "GetAttrs failed.");
        return false;
    }

    auto quant_mode_ptr = attrs->GetAttrPointer<int64_t>(ATTR_QUANT_MODE);
    if (quant_mode_ptr == nullptr) {
        OP_LOGE(context->GetNodeName(), "Get quant_mode attr failed.");
        return false;
    }

    return (*quant_mode_ptr == QUANT_MODE_DYNAMIC);
}

static bool CheckWeightShape(gert::InferShapeContext* context)
{
    const gert::Shape* x_shape = context->GetInputShape(INPUT_X);
    const gert::Shape* weight_shape = context->GetOptionalInputShape(INPUT_WEIGHT);
    
    if (weight_shape == nullptr) {
        return true;
    }

    if (weight_shape->GetDimNum() != x_shape->GetDimNum()) {
        OP_LOGE(context->GetNodeName(), 
                "weight dim num [%zu] must equal x dim num [%zu].",
                weight_shape->GetDimNum(), x_shape->GetDimNum());
        return false;
    }

    for (size_t i = 0; i < x_shape->GetDimNum() - 1; i++) {
        if (weight_shape->GetDim(i) != x_shape->GetDim(i)) {
            OP_LOGE(context->GetNodeName(),
                    "weight dim[%zu] [%ld] must equal x dim[%zu] [%ld].",
                    i, weight_shape->GetDim(i), i, x_shape->GetDim(i));
            return false;
        }
    }

    if (weight_shape->GetDim(weight_shape->GetDimNum() - 1) != 1) {
        OP_LOGE(context->GetNodeName(),
                "weight last dim [%ld] must be 1.",
                weight_shape->GetDim(weight_shape->GetDimNum() - 1));
        return false;
    }

    return true;
}

static bool InferYShape(gert::InferShapeContext* context)
{
    const gert::Shape* x_shape = context->GetInputShape(INPUT_X);
    gert::Shape* y_shape = context->GetOutputShape(OUTPUT_Y);
    
    if (x_shape == nullptr || y_shape == nullptr) {
        OP_LOGE(context->GetNodeName(), "Get x or y shape failed.");
        return false;
    }

    *y_shape = *x_shape;
    size_t last_dim_idx = x_shape->GetDimNum() - 1;

    if (x_shape->GetDim(last_dim_idx) != -1) {
        if (x_shape->GetDim(last_dim_idx) % SPLIT_NUM != 0) {
            OP_LOGE(context->GetNodeName(), 
                    "The last dim of x [%ld] is not divisible by 2.",
                    x_shape->GetDim(last_dim_idx));
            return false;
        }
        y_shape->SetDim(last_dim_idx, x_shape->GetDim(last_dim_idx) / SPLIT_NUM);
    }

    return true;
}

static bool InferYScaleShape(gert::InferShapeContext* context)
{
    gert::Shape* y_scale_shape = context->GetOutputShape(OUTPUT_Y_SCALE);
    if (y_scale_shape == nullptr) {
        OP_LOGE(context->GetNodeName(), "Get y_scale shape failed.");
        return false;
    }

    const gert::Shape* group_index_shape = context->GetOptionalInputShape(INPUT_GROUP_INDEX);
    y_scale_shape->SetDimNum(1);
    
    if (group_index_shape != nullptr) {
        y_scale_shape->SetDim(0, group_index_shape->GetDim(0));
    } else {
        y_scale_shape->SetDim(0, 1);
    }

    return true;
}

static bool InferYOriginShape(gert::InferShapeContext* context)
{
    auto attrs = context->GetAttrs();
    auto output_origin_ptr = attrs->GetAttrPointer<bool>(ATTR_OUTPUT_ORIGIN);
    
    if (output_origin_ptr == nullptr || *output_origin_ptr != true) {
        return true;
    }

    gert::Shape* y_origin_shape = context->GetOutputShape(OUTPUT_Y_ORIGIN);
    gert::Shape* y_shape = context->GetOutputShape(OUTPUT_Y);
    
    if (y_origin_shape == nullptr) {
        OP_LOGE(context->GetNodeName(), "Get y_origin shape failed.");
        return false;
    }

    *y_origin_shape = *y_shape;
    return true;
}

static ge::graphStatus InferShapeForSwigluGroupQuant(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShapeForSwigluGroupQuant");

    if (!CheckQuantMode(context)) {
        return GRAPH_SUCCESS;
    }

    if (!CheckWeightShape(context)) {
        return GRAPH_FAILED;
    }

    if (!InferYShape(context)) {
        return GRAPH_FAILED;
    }

    if (!InferYScaleShape(context)) {
        return GRAPH_FAILED;
    }

    if (!InferYOriginShape(context)) {
        return GRAPH_FAILED;
    }

    OP_LOGD(context->GetNodeName(), "End to do InferShapeForSwigluGroupQuant");
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeForSwigluGroupQuant(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferDataTypeForSwigluGroupQuant");

    auto attrs = context->GetAttrs();
    if (attrs == nullptr) {
        OP_LOGE(context->GetNodeName(), "GetAttrs failed.");
        return GRAPH_FAILED;
    }

    auto quant_mode_ptr = attrs->GetAttrPointer<int64_t>(ATTR_QUANT_MODE);
    if (quant_mode_ptr == nullptr) {
        OP_LOGE(context->GetNodeName(), "Get quant_mode attr failed.");
        return GRAPH_FAILED;
    }

    if (*quant_mode_ptr != QUANT_MODE_DYNAMIC) {
        return GRAPH_SUCCESS;
    }

    context->SetOutputDataType(OUTPUT_Y, ge::DT_HIFLOAT8);
    context->SetOutputDataType(OUTPUT_Y_SCALE, ge::DT_FLOAT);

    auto output_origin_ptr = attrs->GetAttrPointer<bool>(ATTR_OUTPUT_ORIGIN);
    if (output_origin_ptr != nullptr && *output_origin_ptr == true) {
        context->SetOutputDataType(OUTPUT_Y_ORIGIN, context->GetInputDataType(INPUT_X));
    }

    OP_LOGD(context->GetNodeName(), "End to do InferDataTypeForSwigluGroupQuant");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(SwigluGroupQuant)
    .InferShape(InferShapeForSwigluGroupQuant)
    .InferDataType(InferDataTypeForSwigluGroupQuant);

}  // namespace ops
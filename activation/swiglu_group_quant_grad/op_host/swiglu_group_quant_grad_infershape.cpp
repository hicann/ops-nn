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
 * \file swiglu_group_quant_grad_infershape.cpp
 * \brief SwiGLU Group Dynamic Quant Backward shape inference
 */

#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;

namespace {
constexpr size_t INPUT_GRAD_Y = 0;
constexpr size_t INPUT_X = 1;
constexpr size_t INPUT_WEIGHT = 2;
constexpr size_t INPUT_Y_ORIGIN = 3;
constexpr size_t INPUT_GROUP_INDEX = 4;
constexpr size_t OUTPUT_GRAD_X = 0;
constexpr size_t OUTPUT_GRAD_WEIGHT = 1;
constexpr size_t SPLIT_NUM = 2;
}  // namespace

namespace ops {

static ge::graphStatus CheckGradYAndXShape(const gert::Shape* gradYShape, const gert::Shape* xShape)
{
    int64_t gradYDimNum = gradYShape->GetDimNum();
    int64_t xDimNum = xShape->GetDimNum();
    
    if (gradYDimNum < 1 || xDimNum < 1) {
        D_OP_LOGE("SwigluGroupQuantGrad", "Input shape dimension must >= 1.");
        return GRAPH_FAILED;
    }
    
    if (gradYDimNum != xDimNum) {
        D_OP_LOGE("SwigluGroupQuantGrad", "gradY and x shape dimension must be same.");
        return GRAPH_FAILED;
    }
    
    for (int64_t i = 0; i < xDimNum; i++) {
        if (i < xDimNum - 1) {
            if (gradYShape->GetDim(i) != xShape->GetDim(i)) {
                D_OP_LOGE("SwigluGroupQuantGrad", "gradY and x shape must be same except last dim.");
                return GRAPH_FAILED;
            }
        } else {
            int64_t gradYDimLast = gradYShape->GetDim(i);
            int64_t xDimLast = xShape->GetDim(i);
            
            if (xDimLast % SPLIT_NUM != 0) {
                D_OP_LOGE("SwigluGroupQuantGrad", "Input x last dim must be divisible by 2.");
                return GRAPH_FAILED;
            }
            
            if (gradYDimLast != xDimLast / SPLIT_NUM) {
                D_OP_LOGE("SwigluGroupQuantGrad", "Input gradY last dim must be half of x last dim.");
                return GRAPH_FAILED;
            }
        }
    }
    
    return GRAPH_SUCCESS;
}

static ge::graphStatus CheckWeightShape(const gert::Shape* weightShape, const gert::Shape* gradYShape,
                                        int64_t gradYDimNum)
{
    int64_t weightDimNum = weightShape->GetDimNum();
    if (weightDimNum != gradYDimNum) {
        D_OP_LOGE("SwigluGroupQuantGrad", "weight and gradY shape dimension must be same.");
        return GRAPH_FAILED;
    }
    
    for (int64_t i = 0; i < gradYDimNum; i++) {
        if (i < gradYDimNum - 1) {
            if (weightShape->GetDim(i) != gradYShape->GetDim(i)) {
                D_OP_LOGE("SwigluGroupQuantGrad", "weight and gradY shape must be same except last dim.");
                return GRAPH_FAILED;
            }
        } else {
            int64_t weightDimLast = weightShape->GetDim(i);
            if (weightDimLast != 1) {
                D_OP_LOGE("SwigluGroupQuantGrad", "weight last dim must be 1.");
                return GRAPH_FAILED;
            }
        }
    }
    
    return GRAPH_SUCCESS;
}

static ge::graphStatus CheckYOriginShape(const gert::Shape* yOriginShape, const gert::Shape* gradYShape,
                                         int64_t gradYDimNum)
{
    int64_t yOriginDimNum = yOriginShape->GetDimNum();
    if (yOriginDimNum != gradYDimNum) {
        D_OP_LOGE("SwigluGroupQuantGrad", "yOrigin and gradY shape dimension must be same.");
        return GRAPH_FAILED;
    }
    
    for (int64_t i = 0; i < gradYDimNum; i++) {
        if (yOriginShape->GetDim(i) != gradYShape->GetDim(i)) {
            D_OP_LOGE("SwigluGroupQuantGrad", "yOrigin shape must be same as gradY.");
            return GRAPH_FAILED;
        }
    }
    
    return GRAPH_SUCCESS;
}

static ge::graphStatus CheckGroupIndexShape(const gert::Shape* groupIndexShape)
{
    int64_t groupIndexDimNum = groupIndexShape->GetDimNum();
    if (groupIndexDimNum != 1) {
        D_OP_LOGE("SwigluGroupQuantGrad", "groupIndex must be 1D tensor.");
        return GRAPH_FAILED;
    }
    
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferShapeForSwigluGroupQuantGrad(gert::InferShapeContext* context)
{
    OP_LOGD(context, "Enter SwigluGroupQuantGrad InferShape impl.");
    
    auto gradYShape = context->GetInputShape(INPUT_GRAD_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context, gradYShape);
    
    auto xShape = context->GetInputShape(INPUT_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    
    auto gradXShape = context->GetOutputShape(OUTPUT_GRAD_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, gradXShape);
    
    auto gradWeightShape = context->GetOutputShape(OUTPUT_GRAD_WEIGHT);
    OP_CHECK_NULL_WITH_CONTEXT(context, gradWeightShape);
    
    if (CheckGradYAndXShape(gradYShape, xShape) != GRAPH_SUCCESS) {
        return GRAPH_FAILED;
    }
    
    *gradXShape = *xShape;
    
    int64_t gradYDimNum = gradYShape->GetDimNum();
    
    auto weightShape = context->GetOptionalInputShape(INPUT_WEIGHT);
    if (weightShape != nullptr) {
        if (CheckWeightShape(weightShape, gradYShape, gradYDimNum) != GRAPH_SUCCESS) {
            return GRAPH_FAILED;
        }
        *gradWeightShape = *weightShape;
    }
    
    auto yOriginShape = context->GetOptionalInputShape(INPUT_Y_ORIGIN);
    if (yOriginShape != nullptr) {
        if (CheckYOriginShape(yOriginShape, gradYShape, gradYDimNum) != GRAPH_SUCCESS) {
            return GRAPH_FAILED;
        }
    }
    
    auto groupIndexShape = context->GetOptionalInputShape(INPUT_GROUP_INDEX);
    if (groupIndexShape != nullptr) {
        if (CheckGroupIndexShape(groupIndexShape) != GRAPH_SUCCESS) {
            return GRAPH_FAILED;
        }
    }
    
    OP_LOGD(context, "SwigluGroupQuantGrad InferShape impl end.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeForSwigluGroupQuantGrad(gert::InferDataTypeContext *context) {
    OP_LOGD(context, "Enter SwigluGroupQuantGrad inferDataType impl.");
    
    auto xDtype = context->GetInputDataType(INPUT_X);
    context->SetOutputDataType(OUTPUT_GRAD_X, xDtype);
    
    auto weightDesc = context->GetOptionalInputDesc(INPUT_WEIGHT);
    if (weightDesc != nullptr) {
        context->SetOutputDataType(OUTPUT_GRAD_WEIGHT, ge::DT_FLOAT);
    }
    
    OP_LOGD(context, "SwigluGroupQuantGrad inferDataType impl end.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(SwigluGroupQuantGrad)
    .InferShape(InferShapeForSwigluGroupQuantGrad)
    .InferDataType(InferDataTypeForSwigluGroupQuantGrad);

}  // namespace ops
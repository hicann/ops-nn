/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the License).
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file dynamic_augru_grad_infershape.cpp
 * \brief DynamicAUGRUGrad shape/DataType推导
 */

#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;

namespace ops {
static constexpr int64_t IDX_0 = 0;
static constexpr int64_t IDX_1 = 1;
static constexpr int64_t IDX_2 = 2;
static constexpr int64_t IDX_3 = 3;
static constexpr int64_t IDX_4 = 4;   // y（前向输出占位输入）
static constexpr int64_t IDX_5 = 5;   // init_h
static constexpr int64_t IDX_6 = 6;   // h
static constexpr int64_t IDX_7 = 7;   // dy
static constexpr int64_t IDX_8 = 8;   // dh
static constexpr int64_t IDX_9 = 9;   // update
static constexpr int64_t IDX_10 = 10; // update_att
static constexpr int64_t IDX_11 = 11; // reset
static constexpr int64_t IDX_12 = 12; // new
static constexpr int64_t IDX_13 = 13; // hidden_new
static constexpr int64_t IDX_14 = 14; // seq_length（可选）
static constexpr int64_t IDX_15 = 15; // mask（可选，占位）
static constexpr int64_t GATE_NUM = 3;
static constexpr size_t DIM_TBH = 3; // h/dy/update/update_att/reset/new/hidden_new/weight_att维度数
static constexpr size_t DIM_BH = 2;  // init_h/dh维度数

struct ShapeParams {
    int64_t t;
    int64_t b;
    int64_t i;
    int64_t h;
    int64_t threeH;
};

static ge::graphStatus CheckTbhShape(gert::InferShapeContext* context, int64_t idx, const char* name, int64_t t,
                                     int64_t b, int64_t h)
{
    const gert::Shape* shape = context->GetInputShape(idx);
    OP_CHECK_NULL_WITH_CONTEXT(context, shape);
    OP_CHECK_IF(shape->GetDimNum() != DIM_TBH,
                OP_LOGE(context, "The dim num of %s should be 3, but %zu was obtained.", name, shape->GetDimNum()),
                return GRAPH_FAILED);
    OP_CHECK_IF(shape->GetDim(IDX_0) != t || shape->GetDim(IDX_1) != b || shape->GetDim(IDX_2) != h,
                OP_LOGE(context, "The shape of %s should be [%lld, %lld, %lld], but [%lld, %lld, %lld] was obtained.",
                        name, t, b, h, shape->GetDim(IDX_0), shape->GetDim(IDX_1), shape->GetDim(IDX_2)),
                return GRAPH_FAILED);
    return GRAPH_SUCCESS;
}

static bool IsWeightHiddenShapeValid(const gert::Shape& shape, const ShapeParams& p)
{
    if (shape.GetDimNum() == DIM_BH) {
        return shape.GetDim(IDX_0) == p.h && shape.GetDim(IDX_1) == p.threeH;
    }
    return shape.GetDimNum() == DIM_TBH && shape.GetDim(IDX_0) == 1 && shape.GetDim(IDX_1) == p.h &&
           shape.GetDim(IDX_2) == p.threeH;
}

static ge::graphStatus GetAndCheckCoreShapes(gert::InferShapeContext* context, ShapeParams& p)
{
    const gert::Shape* xShape = context->GetInputShape(IDX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    const gert::Shape* wInputShape = context->GetInputShape(IDX_1);
    OP_CHECK_NULL_WITH_CONTEXT(context, wInputShape);
    const gert::Shape* wHiddenShape = context->GetInputShape(IDX_2);
    OP_CHECK_NULL_WITH_CONTEXT(context, wHiddenShape);
    const gert::Shape* hShape = context->GetInputShape(IDX_6);
    OP_CHECK_NULL_WITH_CONTEXT(context, hShape);
    OP_CHECK_IF(xShape->GetDimNum() != DIM_TBH,
                OP_LOGE(context, "The dim num of x should be 3, but %zu was obtained.", xShape->GetDimNum()),
                return GRAPH_FAILED);
    OP_CHECK_IF(wInputShape->GetDimNum() != DIM_BH, OP_LOGE(context, "The dim num of weight_input should be 2."),
                return GRAPH_FAILED);
    OP_CHECK_IF(hShape->GetDimNum() != DIM_TBH,
                OP_LOGE(context, "The dim num of h should be 3, but %zu was obtained.", hShape->GetDimNum()),
                return GRAPH_FAILED);
    p = {xShape->GetDim(IDX_0), xShape->GetDim(IDX_1), xShape->GetDim(IDX_2), hShape->GetDim(IDX_2), 0};
    p.threeH = GATE_NUM * p.h;
    OP_CHECK_IF(!IsWeightHiddenShapeValid(*wHiddenShape, p),
                OP_LOGE(context, "The shape of weight_hidden should be [%lld, %lld] or [1, %lld, %lld].", p.h, p.threeH,
                        p.h, p.threeH),
                return GRAPH_FAILED);
    OP_CHECK_IF(wInputShape->GetDim(IDX_0) != p.i || wInputShape->GetDim(IDX_1) != p.threeH,
                OP_LOGE(context, "The shape of weight_input should be [%lld, %lld].", p.i, p.threeH),
                return GRAPH_FAILED);
    return GRAPH_SUCCESS;
}

static ge::graphStatus CheckRelatedInputShapes(gert::InferShapeContext* context, const ShapeParams& p)
{
    const gert::Shape* initHShape = context->GetInputShape(IDX_5);
    OP_CHECK_NULL_WITH_CONTEXT(context, initHShape);
    const gert::Shape* dhShape = context->GetInputShape(IDX_8);
    OP_CHECK_NULL_WITH_CONTEXT(context, dhShape);
    OP_CHECK_IF(
        initHShape->GetDimNum() != DIM_BH || initHShape->GetDim(IDX_0) != p.b || initHShape->GetDim(IDX_1) != p.h,
        OP_LOGE(context, "The shape of init_h should be [%lld, %lld].", p.b, p.h), return GRAPH_FAILED);
    OP_CHECK_IF(dhShape->GetDimNum() != DIM_BH || dhShape->GetDim(IDX_0) != p.b || dhShape->GetDim(IDX_1) != p.h,
                OP_LOGE(context, "The shape of dh should be [%lld, %lld].", p.b, p.h), return GRAPH_FAILED);
    static const size_t kTbhIdx[] = {IDX_3, IDX_4, IDX_6, IDX_7, IDX_9, IDX_10, IDX_11, IDX_12, IDX_13};
    static const char* kTbhNames[] = {"weight_att", "y",     "h",   "dy",        "update",
                                      "update_att", "reset", "new", "hidden_new"};
    for (size_t i = 0; i < sizeof(kTbhIdx) / sizeof(kTbhIdx[0]); i++) {
        const char* name = kTbhNames[i];
        OP_CHECK_IF(CheckTbhShape(context, static_cast<int64_t>(kTbhIdx[i]), name, p.t, p.b, p.h) != GRAPH_SUCCESS,
                    OP_LOGE(context, "The shape of %s is invalid.", name), return GRAPH_FAILED);
    }
    const gert::Shape* seqLenShape = context->GetOptionalInputShape(IDX_14);
    if (seqLenShape != nullptr && !seqLenShape->IsScalar()) {
        OP_CHECK_IF(seqLenShape->GetDimNum() != 1 || seqLenShape->GetDim(IDX_0) != p.b,
                    OP_LOGE(context, "The shape of seq_length should be [%lld].", p.b), return GRAPH_FAILED);
    }
    return GRAPH_SUCCESS;
}

static ge::graphStatus SetOutputShapes(gert::InferShapeContext* context, const ShapeParams& p)
{
    gert::Shape* dwInputShape = context->GetOutputShape(IDX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context, dwInputShape);
    dwInputShape->SetDimNum(DIM_BH);
    dwInputShape->SetDim(IDX_0, p.i);
    dwInputShape->SetDim(IDX_1, p.threeH);
    gert::Shape* dwHiddenShape = context->GetOutputShape(IDX_1);
    OP_CHECK_NULL_WITH_CONTEXT(context, dwHiddenShape);
    dwHiddenShape->SetDimNum(DIM_BH);
    dwHiddenShape->SetDim(IDX_0, p.h);
    dwHiddenShape->SetDim(IDX_1, p.threeH);
    gert::Shape* dbInputShape = context->GetOutputShape(IDX_2);
    OP_CHECK_NULL_WITH_CONTEXT(context, dbInputShape);
    dbInputShape->SetDimNum(1);
    dbInputShape->SetDim(IDX_0, p.threeH);
    gert::Shape* dbHiddenShape = context->GetOutputShape(IDX_3);
    OP_CHECK_NULL_WITH_CONTEXT(context, dbHiddenShape);
    dbHiddenShape->SetDimNum(1);
    dbHiddenShape->SetDim(IDX_0, p.threeH);
    gert::Shape* dxShape = context->GetOutputShape(IDX_4);
    OP_CHECK_NULL_WITH_CONTEXT(context, dxShape);
    dxShape->SetDimNum(DIM_TBH);
    dxShape->SetDim(IDX_0, p.t);
    dxShape->SetDim(IDX_1, p.b);
    dxShape->SetDim(IDX_2, p.i);
    gert::Shape* dhPrevShape = context->GetOutputShape(IDX_5);
    OP_CHECK_NULL_WITH_CONTEXT(context, dhPrevShape);
    dhPrevShape->SetDimNum(DIM_BH);
    dhPrevShape->SetDim(IDX_0, p.b);
    dhPrevShape->SetDim(IDX_1, p.h);
    gert::Shape* dwAttShape = context->GetOutputShape(IDX_6);
    OP_CHECK_NULL_WITH_CONTEXT(context, dwAttShape);
    dwAttShape->SetDimNum(DIM_BH);
    dwAttShape->SetDim(IDX_0, p.t);
    dwAttShape->SetDim(IDX_1, p.b);
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferShapeDynamicAUGRUGrad(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShapeDynamicAUGRUGrad");
    ShapeParams params{};
    OP_CHECK_IF(GetAndCheckCoreShapes(context, params) != GRAPH_SUCCESS,
                OP_LOGE(context, "The core input shapes are invalid."), return GRAPH_FAILED);
    OP_CHECK_IF(CheckRelatedInputShapes(context, params) != GRAPH_SUCCESS,
                OP_LOGE(context, "The related input shapes are invalid."), return GRAPH_FAILED);
    OP_CHECK_IF(SetOutputShapes(context, params) != GRAPH_SUCCESS, OP_LOGE(context, "Failed to set output shapes."),
                return GRAPH_FAILED);
    OP_LOGD(context->GetNodeName(), "End to do InferShapeDynamicAUGRUGrad");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(DynamicAUGRUGrad).InferShape(InferShapeDynamicAUGRUGrad);
} // namespace ops

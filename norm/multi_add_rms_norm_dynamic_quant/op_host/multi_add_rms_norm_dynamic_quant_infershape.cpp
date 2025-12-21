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
 * \file multi_add_rms_norm_dynamic_quant_infershape.cpp
 * \brief
 */
#include <sstream>
#include <string>
#include <vector>
#include "register/op_def_registry.h"
#include "log/log.h"
#include "util/math_util.h"

using namespace ge;

namespace ops {

static constexpr int X1_IDX = 0;
static constexpr int X2_IDX = 1;
static constexpr int GAMMA_IDX = 2;
static constexpr int SMOOTH1_IDX = 3;
static constexpr int SMOOTH2_IDX = 4;

static constexpr int Y1_IDX = 0;
static constexpr int Y2_IDX = 1;
static constexpr int X_IDX = 2;
static constexpr int Y_IDX = 3;
static constexpr int OUT_SCALE1_IDX = 4;
static constexpr int OUT_SCALE2_IDX = 5;

static bool InferReduceShape(const gert::Shape* xShape, const gert::Shape* gammaShape, gert::Shape* reduceShape)
{
    size_t xDimNum = xShape->GetDimNum();
    size_t gammaDimNum = gammaShape->GetDimNum();
    reduceShape->SetDimNum(xDimNum - gammaDimNum);

    int64_t xDimValue = 0;

    OP_CHECK_IF((xDimNum - gammaDimNum < 0), OP_LOGD("InferReduceShape", "Get invalid x and gamma DimNum"), return false);
    for (size_t i = 0; i < xDimNum - gammaDimNum; i++) {
        xDimValue = xShape->GetDim(i);
        reduceShape->SetDim(i, xDimValue);
        OP_LOGI("InferShape4MultiAddRmsNormDynamicQuant", "reduceShape[%zu] = [%zu]", i, reduceShape->GetDim(i));
    }
    return true;
}

static bool CheckOptionalShapeExisting(const gert::Shape* smoothShape)
{
    OP_CHECK_IF(nullptr == smoothShape, OP_LOGD("CheckOptionalShapeExisting", "Get nullptr smoothShape"), return false);
    int64_t smoothShapeSize = smoothShape->GetShapeSize();
    OP_CHECK_IF((smoothShapeSize <= 0), OP_LOGD("CheckOptionalShapeExisting", "Get empty smoothShape"), return false);
    return true;
}

static ge::graphStatus InferShape4MultiAddRmsNormDynamicQuant(gert::InferShapeContext* context)
{
    OP_LOGI(context, "Begin to do InferShape4MultiAddRmsNormDynamicQuant");

    const auto inputInfo = context->GetIrInputInstanceInfo(0);
    if (inputInfo == nullptr) {
        return ge::GRAPH_FAILED;
    }
    int64_t x1Num = inputInfo->GetInstanceNum();

    // get input shapes
    const gert::Shape* x2Shape = context->GetInputShape(X2_IDX + x1Num - 1);
    OP_CHECK_NULL_WITH_CONTEXT(context, x2Shape);
    const gert::Shape* gammaShape = context->GetInputShape(GAMMA_IDX + x1Num - 1);
    OP_CHECK_NULL_WITH_CONTEXT(context, gammaShape);
    // get output shapes
    gert::Shape* y1Shape = context->GetOutputShape(Y1_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, y1Shape);
    gert::Shape* y2Shape = context->GetOutputShape(Y2_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, y2Shape);
    gert::Shape* xShape = context->GetOutputShape(X_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    gert::Shape* yShape = context->GetOutputShape(Y_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);
    gert::Shape* outScale1Shape = context->GetOutputShape(OUT_SCALE1_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, outScale1Shape);
    gert::Shape* outScale2Shape = context->GetOutputShape(OUT_SCALE2_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, outScale2Shape);

    *y1Shape = *x2Shape;
    *xShape = *x2Shape;
    *yShape = *x2Shape;

    const gert::Shape* smooth1Shape = context->GetOptionalInputShape(SMOOTH1_IDX);
    bool smooth1Exist = CheckOptionalShapeExisting(smooth1Shape);
    const gert::Shape* smooth2Shape = context->GetOptionalInputShape(SMOOTH2_IDX);
    bool smooth2Exist = CheckOptionalShapeExisting(smooth2Shape);

    OP_CHECK_IF(
        smooth1Exist && (*gammaShape != *smooth1Shape),
        OP_LOGE(context->GetNodeName(), "GammaShape is not same to smooth1Shape."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        smooth2Exist && (*gammaShape != *smooth2Shape),
        OP_LOGE(context->GetNodeName(), "GammaShape is not same to smooth2Shape."),
        return ge::GRAPH_FAILED);

    bool isOnlyExistSmooth2 = (!smooth1Exist) && smooth2Exist;
    OP_CHECK_IF(
        isOnlyExistSmooth2,
        OP_LOGE(
            context->GetNodeName(), "Dynamic MultiAddRmsNormDynamicQuant Not support only have scale2."),
        return ge::GRAPH_FAILED);
    InferReduceShape(x2Shape, gammaShape, outScale1Shape);
    if (smooth2Exist) {
        *y2Shape = *x2Shape;
        *outScale2Shape = *outScale1Shape;
    } else {
        *y2Shape = gert::Shape({1});
        *outScale2Shape = gert::Shape({1});
    }
    OP_LOGI(context, "End to do InferShape4MultiAddRmsNormDynamicQuant");
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType4MultiAddRmsNormDynamicQuant(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(Y1_IDX, DT_INT8);
    context->SetOutputDataType(Y2_IDX, DT_INT8);
    context->SetOutputDataType(X_IDX, context->GetInputDataType(X1_IDX));
    context->SetOutputDataType(Y_IDX, context->GetInputDataType(X1_IDX));
    context->SetOutputDataType(OUT_SCALE1_IDX, DT_FLOAT);
    context->SetOutputDataType(OUT_SCALE2_IDX, DT_FLOAT);
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MultiAddRmsNormDynamicQuant)
    .InferShape(InferShape4MultiAddRmsNormDynamicQuant)
    .InferDataType(InferDataType4MultiAddRmsNormDynamicQuant);
} // namespace ops
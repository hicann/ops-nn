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
 * \file rms_norm_grad_quant_infershape.cpp
 * \brief
 */

#include <algorithm>
#include <vector>
#include "log/log.h"
#include "util/shape_util.h"
#include "util/math_util.h"
#include "graph/utils/type_utils.h"
#include "register/op_impl_registry.h"
#include "op_common/log/log.h"

using namespace ge;

namespace ops {
static constexpr int RMS_NORM_GRAD_QUANT_IDX_DY = 0;
static constexpr int RMS_NORM_GRAD_QUANT_IDX_GAMMA = 3;
static constexpr int RMS_NORM_GRAD_QUANT_IDX_SCALE_X = 4;
static constexpr int RMS_NORM_GRAD_QUANT_ATTR_IDX_DST_TYPE = 2;

static constexpr int RMS_NORM_GRAD_QUANT_OUTPUT_IDX_DX = 0;
static constexpr int RMS_NORM_GRAD_QUANT_OUTPUT_IDX_DGAMMA = 1;

static const std::vector<ge::DataType> RMS_NORM_GRAD_QUANT_OUT_TYPE_LIST = {
    ge::DT_HIFLOAT8, ge::DT_INT8};
    
static ge::graphStatus InferShape4RmsNormGradQuant(gert::InferShapeContext* context)
{
    OP_LOGD(context, "Begin to do InferShape4RmsNormGradQuant.");
    const gert::Shape* dy_shape = context->GetInputShape(RMS_NORM_GRAD_QUANT_IDX_DY);
    const gert::Shape* gamma_shape = context->GetInputShape(RMS_NORM_GRAD_QUANT_IDX_GAMMA);
    OP_CHECK_NULL_WITH_CONTEXT(context, dy_shape);
    OP_CHECK_NULL_WITH_CONTEXT(context, gamma_shape);

    // get output shapes
    gert::Shape* dx_shape = context->GetOutputShape(RMS_NORM_GRAD_QUANT_OUTPUT_IDX_DX);
    gert::Shape* dgamma_shape = context->GetOutputShape(RMS_NORM_GRAD_QUANT_OUTPUT_IDX_DGAMMA);
    OP_CHECK_NULL_WITH_CONTEXT(context, dx_shape);
    OP_CHECK_NULL_WITH_CONTEXT(context, dgamma_shape);
    *dx_shape = *dy_shape;
    *dgamma_shape = *gamma_shape;

    OP_LOGD(context, "End to do InferShape4RmsNormGradQuant.");
    return ge::GRAPH_SUCCESS;
}

static graphStatus InferDataType4RmsNormGradQuant(gert::InferDataTypeContext* context)
{
    OP_LOGD(context, "Begin to do InferDataType4RmsNormGradQuant");

    // 1. 设置第一个输出dx的数据类型 (scales_x为必选输入，始终走量化场景)
    // 量化场景，根据dst_type这个attr的值进行设置，默认是INT8
    ge::DataType dxDtype = ge::DT_INT8;
    const auto* attrs = context->GetAttrs();
    if (attrs != nullptr) {
        const int64_t* dstTypePtr = attrs->GetAttrPointer<int64_t>(RMS_NORM_GRAD_QUANT_ATTR_IDX_DST_TYPE);
        if (dstTypePtr != nullptr) {
            dxDtype = static_cast<ge::DataType>(*dstTypePtr);
            OP_CHECK_IF(
                std::find(RMS_NORM_GRAD_QUANT_OUT_TYPE_LIST.begin(), RMS_NORM_GRAD_QUANT_OUT_TYPE_LIST.end(), dxDtype) == RMS_NORM_GRAD_QUANT_OUT_TYPE_LIST.end(),
                OP_LOGE(context,
                        "attr dst_type only support 2(int8), 34(hifloat8)"),
                return ge::GRAPH_FAILED);
        }
    }
    context->SetOutputDataType(RMS_NORM_GRAD_QUANT_OUTPUT_IDX_DX, dxDtype);

    // 2. 设置第二个输出dx的数据类型
    context->SetOutputDataType(RMS_NORM_GRAD_QUANT_OUTPUT_IDX_DGAMMA, DT_FLOAT);
    OP_LOGD(context, "End to do InferDataType4RmsNormGradQuant");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(RmsNormGradQuant).InferShape(InferShape4RmsNormGradQuant).InferDataType(InferDataType4RmsNormGradQuant);
} // namespace ops

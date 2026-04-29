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
 * \file adaptive_avg_pool3d_grad_infershape.cpp
 * \brief
 */

#include "register/op_impl_registry.h"
#include "log/log.h"
#include "util/shape_util.h"
#include "platform/platform_info.h"

using namespace ge;

namespace {
constexpr size_t Y_GRAD_INDEX = 0;
constexpr size_t X_INDEX = 1;
constexpr size_t X_GRAD_INDEX = 0;
constexpr size_t X_DIMS = 5;
constexpr size_t INDEX_DATAFORMAT = 0;
constexpr size_t NCDHW_DIM_NUM = 5;
constexpr size_t CDHW_DIM_NUM = 4;
} // namespace

namespace ops {
static ge::graphStatus InferShape4AdaptiveAvgPool3dGrad(gert::InferShapeContext* context)
{
    // input 1: y_grad shape
    const gert::Shape* gradShape = context->GetInputShape(Y_GRAD_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, gradShape);

    // input 2: x shape
    const gert::Shape* xShape = context->GetInputShape(X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);

    // output shape
    gert::Shape* xGradShape = context->GetOutputShape(X_GRAD_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, xGradShape);

    // attributes ptr
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

    // 获取数据格式属性
    const char* dataFormat = attrs->GetAttrPointer<char>(INDEX_DATAFORMAT);
    OP_CHECK_NULL_WITH_CONTEXT(context, dataFormat);
    std::string dataFormatStr = dataFormat;
    size_t xDimNum = xShape->GetDimNum();

    // 未知rank处理
    if (Ops::Base::IsUnknownRank(*xShape)) {
        OP_LOGI("InferShape4AdaptiveAvgPool3dGrad", "entering IsUnknownRank");
        Ops::Base::SetUnknownRank(*xGradShape);
        return ge::GRAPH_SUCCESS;
    }

    // 维度数赋值    
    xGradShape->SetDimNum(xDimNum);

    fe::PlatformInfo platform_info;
    fe::OptionalInfo optional_info;
    OP_CHECK_IF(
    (fe::PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platform_info, optional_info) !=
        ge::GRAPH_SUCCESS),
    OP_LOGE(context->GetNodeName(), "Cannot get platform info!"), return ge::GRAPH_FAILED);
    
    // 输出shape赋值+0维度校验
    for (size_t i = 0; i < xDimNum; ++i) {
        xGradShape->SetDim(i, xShape->GetDim(i));
    }

    return GRAPH_SUCCESS;
}

static graphStatus InferDtype4AdaptiveAvgPool3dGrad(gert::InferDataTypeContext* context)
{   
    auto gradDataType = context->GetInputDataType(Y_GRAD_INDEX);
    auto xDataType = context->GetInputDataType(X_INDEX);
    
    // 校验输入数据类型一致性
    OP_CHECK_IF(xDataType != gradDataType,
                OP_LOGE(context->GetNodeName(), "Data type mismatch: x(%d) != grad(%d)",
                        xDataType, gradDataType),
                return GRAPH_FAILED);
    
    // 校验数据类型合法性
    OP_CHECK_IF((xDataType != ge::DT_FLOAT) && (xDataType != ge::DT_FLOAT16) && (xDataType != ge::DT_BF16),
                OP_LOGE(context->GetNodeName(), "Data type invalid: x(%d), expect fp32/fp16/bf16.",
                        xDataType),
                return ge::GRAPH_FAILED);
    
    context->SetOutputDataType(X_GRAD_INDEX, xDataType);    
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(AdaptiveAvgPool3dGrad)
    .InferShape(InferShape4AdaptiveAvgPool3dGrad)
    .InferDataType(InferDtype4AdaptiveAvgPool3dGrad);
} // namespace ops

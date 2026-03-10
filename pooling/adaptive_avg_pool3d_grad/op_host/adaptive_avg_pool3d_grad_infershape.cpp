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

/**
 * @brief 提取的输入合法性校验辅助函数
 * @param context 推理上下文，用于日志输出和节点信息获取
 * @param gradShape y_grad输入的shape指针
 * @param xShape x输入的shape指针
 * @param dataFormatStr 数据格式字符串（NCDHW/NDHWC）
 * @return ge::GRAPH_SUCCESS 校验通过；ge::GRAPH_FAILED 校验失败（含详细日志）
 */
static ge::graphStatus CheckInputValidity(gert::InferShapeContext* context, 
                                         const gert::Shape* gradShape, 
                                         const gert::Shape* xShape, 
                                         const std::string& dataFormatStr) {
    // 1. 校验数据格式合法性
    if (!(dataFormatStr == "NCDHW" || dataFormatStr == "NDHWC")) {
        OP_LOGE(context->GetNodeName(), 
                "ATTR dataFormat is %s, expect [NDHWC] or [NCDHW].", 
                dataFormatStr.c_str());
        return ge::GRAPH_FAILED;
    }

    // 2. 校验输入维度数合法性
    size_t gradDimNum = gradShape->GetDimNum();
    size_t xDimNum = xShape->GetDimNum();
    bool dimNumValid = ((xDimNum == NCDHW_DIM_NUM) && (gradDimNum == NCDHW_DIM_NUM)) ||
                       ((xDimNum == CDHW_DIM_NUM) && (gradDimNum == CDHW_DIM_NUM));
    if (!dimNumValid) {
        OP_LOGE(context->GetNodeName(),
                "Input dim num should be %lu or %lu, actual: xDim=%lu, gradDim=%lu",
                NCDHW_DIM_NUM, CDHW_DIM_NUM, xDimNum, gradDimNum);
        return ge::GRAPH_FAILED;
    }

    // 3. 校验NC维度一致性
    uint32_t cPosIdx = (dataFormatStr == "NDHWC") ? xDimNum - 1 : xDimNum - 4;
    uint64_t xNDim = (xDimNum == CDHW_DIM_NUM) ? 1 : xShape->GetDim(0);
    uint64_t gradNDim = (gradDimNum == CDHW_DIM_NUM) ? 1 : gradShape->GetDim(0);    
    uint64_t xCDim = xShape->GetDim(cPosIdx);
    uint64_t gradCDim = gradShape->GetDim(cPosIdx);
    OP_LOGI("InferShape4AdaptiveAvgPool3dGrad", 
            "NC: grad(%lu,%lu), x(%lu,%lu)", gradNDim, gradCDim, xNDim, xCDim);

    if ((xNDim != gradNDim) || (xCDim != gradCDim)) {
        OP_LOGE(context->GetNodeName(), 
                "Input N/C dim mismatch: grad(N=%lu,C=%lu), x(N=%lu,C=%lu)",
                gradNDim, gradCDim, xNDim, xCDim);
        return ge::GRAPH_FAILED;
    }

    // 所有校验通过
    return ge::GRAPH_SUCCESS;
}

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

    // 未知rank处理
    if (Ops::Base::IsUnknownRank(*xShape)) {
        OP_LOGI("InferShape4AdaptiveAvgPool3dGrad", "entering IsUnknownRank");
        Ops::Base::SetUnknownRank(*xGradShape);
        return ge::GRAPH_SUCCESS;
    }

    // 维度数赋值
    size_t xDimNum = xShape->GetDimNum();
    xGradShape->SetDimNum(xDimNum);

    fe::PlatformInfo platform_info;
    fe::OptionalInfo optional_info;
    OP_CHECK_IF(
    (fe::PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platform_info, optional_info) !=
        ge::GRAPH_SUCCESS),
    OP_LOGE(context->GetNodeName(), "Cannot get platform info!"), return ge::GRAPH_FAILED);
    
    if (platform_info.str_info.short_soc_version == "Ascend950") {
        ge::graphStatus checkStatus = CheckInputValidity(context, gradShape, xShape, dataFormatStr);
        if (checkStatus != ge::GRAPH_SUCCESS) {
            OP_LOGE(context->GetNodeName(), "Input validity check failed!");
            return GRAPH_FAILED;
        }
    }

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

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
 * \file batch_norm_grad_v3_infershape.cpp
 * \brief
 */
#include "op_common/log/log.h"
#include "error_util.h"
#include "register/op_impl_registry.h"

using namespace ge;
namespace ops {
// InferShape阶段框架契约类失败(输出shape容器为空)统一上报EZ1014, 不再生成EZ9999内部码
static void ReportInferShapeStageError(const char* opName, const std::string& reason)
{
    const std::vector<const char*> msgKey = {"opName", "reason"};
    const std::vector<const char*> msgValue = {opName != nullptr ? opName : "BatchNormGradV3", reason.c_str()};
    REPORT_PREDEFINED_ERR_MSG("EZ1014", msgKey, msgValue);
}

static constexpr int64_t DY_INPUT_IDX = 0;
static constexpr int64_t X_INPUT_IDX = 1;
static constexpr int64_t WEIGHT_INPUT_IDX = 2;
static constexpr int64_t DX_OUTPUT_IDX = 0;
static constexpr int64_t DWEIGHT_OUTPUT_IDX = 1;
static constexpr int64_t DBIAS_OUTPUT_IDX = 2;
static constexpr int DIM_0 = 0;
static constexpr int DIM_1 = 1;
static constexpr int DIM_3 = 3;
static constexpr int DIM_4 = 4;
static constexpr int DIM_LEN_ONE = 1;

static ge::graphStatus BatchNormGradV3InferShape(gert::InferShapeContext* context)
{
    if (context == nullptr) {
        return GRAPH_FAILED;
    }

    const gert::Shape* dyShape = context->GetInputShape(DY_INPUT_IDX);
    OP_CHECK_IF(
        dyShape == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "dy",
                                                 "the input shape of required input dy is missing in the infer-shape "
                                                 "context; check that the graph node provides all required inputs"),
        return ge::GRAPH_FAILED);
    const gert::Shape* weightShape = context->GetInputShape(WEIGHT_INPUT_IDX);
    OP_CHECK_IF(weightShape == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                    context->GetNodeName(), "weight",
                    "the input shape of required input weight is missing in the infer-shape context; check that the "
                    "graph node provides all required inputs"),
                return ge::GRAPH_FAILED);
    gert::Shape* dxShape = context->GetOutputShape(DX_OUTPUT_IDX);
    OP_CHECK_IF(dxShape == nullptr,
                ReportInferShapeStageError(context->GetNodeName(),
                                           "the output shape container of dx (output index 0) is missing in the "
                                           "infer-shape context; check the InferShapeContext construction"),
                return ge::GRAPH_FAILED);
    gert::Shape* dweightShape = context->GetOutputShape(DWEIGHT_OUTPUT_IDX);
    OP_CHECK_IF(dweightShape == nullptr,
                ReportInferShapeStageError(context->GetNodeName(),
                                           "the output shape container of dweight (output index 1) is missing in the "
                                           "infer-shape context; check the InferShapeContext construction"),
                return ge::GRAPH_FAILED);
    gert::Shape* dbiasShape = context->GetOutputShape(DBIAS_OUTPUT_IDX);
    OP_CHECK_IF(dbiasShape == nullptr,
                ReportInferShapeStageError(context->GetNodeName(),
                                           "the output shape container of dbias (output index 2) is missing in the "
                                           "infer-shape context; check the InferShapeContext construction"),
                return ge::GRAPH_FAILED);

    *dxShape = *dyShape;
    *dweightShape = *weightShape;
    *dbiasShape = *weightShape;

    // 推理模式仅计算 dx；dy 明确为空时，将无数值语义的辅助输出也推为空，交由 GE 处理。
    // 训练模式的参数梯度包含归约：归约轴为 0 不代表 [C] 输出为空，须保留 weight 的形状。
    const auto* attrs = context->GetAttrs();
    const bool* isTraining = attrs == nullptr ? nullptr : attrs->GetBool(0);
    if (isTraining != nullptr && !*isTraining && dyShape->GetShapeSize() == 0) {
        *dweightShape = {0};
        *dbiasShape = {0};
    }

    return GRAPH_SUCCESS;
}

static ge::graphStatus BatchNormGradV3InferDataType(gert::InferDataTypeContext* context)
{
    if (context == nullptr) {
        return GRAPH_FAILED;
    }

    const ge::DataType dyDtype = context->GetInputDataType(DY_INPUT_IDX);
    const ge::DataType weightDtype = context->GetInputDataType(WEIGHT_INPUT_IDX);

    context->SetOutputDataType(DX_OUTPUT_IDX, dyDtype);
    context->SetOutputDataType(DWEIGHT_OUTPUT_IDX, weightDtype);
    context->SetOutputDataType(DBIAS_OUTPUT_IDX, weightDtype);
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(BatchNormGradV3).InferShape(BatchNormGradV3InferShape).InferDataType(BatchNormGradV3InferDataType);
} // namespace ops

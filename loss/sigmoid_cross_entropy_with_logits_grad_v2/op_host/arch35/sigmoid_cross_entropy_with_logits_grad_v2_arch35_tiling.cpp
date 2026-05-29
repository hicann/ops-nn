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
 * \file sigmoid_cross_entropy_with_logits_grad_v2_arch35_tiling.cpp
 * \brief
 */
#include "sigmoid_cross_entropy_with_logits_grad_v2_arch35_tiling.h"

using namespace AscendC;
using namespace ge;
using namespace SigmoidCrossEntropyWithLogitsGradV2Op;
using namespace SigmoidCrossEntropyWithLogitsGradV2Struct;

#define SIGMOID_CROSS_ENTROPY_WITH_LOGITS_GRAD_V2_DO_BROADCAST_TILING(DTYPE)                             \
    if (hasWeight_) {                                                                                    \
        if (hasPosWeight_) {                                                                             \
            SIGMOID_CROSS_ENTROPY_WITH_LOGITS_GRAD_V2_DO_BROADCAST_TILING_DAG(DTYPE, DagWeightPosWight); \
        } else {                                                                                         \
            SIGMOID_CROSS_ENTROPY_WITH_LOGITS_GRAD_V2_DO_BROADCAST_TILING_DAG(DTYPE, DagWeight);         \
        }                                                                                                \
    } else {                                                                                             \
        if (hasPosWeight_) {                                                                             \
            SIGMOID_CROSS_ENTROPY_WITH_LOGITS_GRAD_V2_DO_BROADCAST_TILING_DAG(DTYPE, DagPosWeight);      \
        } else {                                                                                         \
            SIGMOID_CROSS_ENTROPY_WITH_LOGITS_GRAD_V2_DO_BROADCAST_TILING_DAG(DTYPE, Dag);               \
        }                                                                                                \
    }

#define SIGMOID_CROSS_ENTROPY_WITH_LOGITS_GRAD_V2_DO_BROADCAST_TILING_DAG(DTYPE, DAG)           \
    if (isMean_) {                                                                              \
        SIGMOID_CROSS_ENTROPY_WITH_LOGITS_GRAD_V2_DO_BROADCAST_TILING_WITH_MEAN(DTYPE, DAG);    \
    } else {                                                                                    \
        SIGMOID_CROSS_ENTROPY_WITH_LOGITS_GRAD_V2_DO_BROADCAST_TILING_WITHOUT_MEAN(DTYPE, DAG); \
    }

#define SIGMOID_CROSS_ENTROPY_WITH_LOGITS_GRAD_V2_DO_BROADCAST_TILING_WITH_MEAN(DTYPE, DAG)                         \
    BroadcastBaseTiling<SigmoidCrossEntropyWithLogitsGradV2##DAG<DTYPE, ATTR_MEAN>::OpDag> brcBaseTiling(context_); \
    SIGMOID_CROSS_ENTROPY_WITH_LOGITS_GRAD_V2_CAL_BROADCAST_TILING

#define SIGMOID_CROSS_ENTROPY_WITH_LOGITS_GRAD_V2_DO_BROADCAST_TILING_WITHOUT_MEAN(DTYPE, DAG)                       \
    BroadcastBaseTiling<SigmoidCrossEntropyWithLogitsGradV2##DAG<DTYPE, ATTR_OTHER>::OpDag> brcBaseTiling(context_); \
    SIGMOID_CROSS_ENTROPY_WITH_LOGITS_GRAD_V2_CAL_BROADCAST_TILING

#define SIGMOID_CROSS_ENTROPY_WITH_LOGITS_GRAD_V2_CAL_BROADCAST_TILING                               \
    ret = brcBaseTiling.DoTiling();                                                                  \
    tilingKey_ = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode(), hasWeight_, hasPosWeight_, isMean_); \
    brcBaseTiling.SetScalar<float>(scale_)

namespace optiling {
static constexpr size_t INPUT_PREDICT = 0;
static constexpr size_t INPUT_TARGET = 1;
static constexpr size_t INPUT_DOUT = 2;
static constexpr size_t INPUT_WEIGHT = 3;
static constexpr size_t INPUT_POS_WEIGHT = 4;
static constexpr size_t ATTR_REDUCTION_INDEX = 0;
static constexpr size_t OUTPUT_GRADIENT = 0;

static constexpr uint32_t ATTR_MEAN = 1;
static constexpr uint32_t ATTR_OTHER = 0;

ge::graphStatus SigmoidCrossEntropyWithLogitsGradV2Tiling::GetDtype()
{
    auto predictedDesc = context_->GetInputDesc(INPUT_PREDICT);
    OP_CHECK_NULL_WITH_CONTEXT(context_, predictedDesc);
    dtype_ = predictedDesc->GetDataType();
    auto targetDesc = context_->GetInputDesc(INPUT_TARGET);
    OP_CHECK_NULL_WITH_CONTEXT(context_, targetDesc);
    ge::DataType targetDtype = targetDesc->GetDataType();
    OP_CHECK_IF(
        dtype_ != targetDtype,
        OP_LOGE(
            context_->GetNodeName(), "the dtype of target %s should be same with predicted %s.",
            ge::TypeUtils::DataTypeToAscendString(targetDtype).GetString(),
            ge::TypeUtils::DataTypeToAscendString(dtype_).GetString()),
        return ge::GRAPH_FAILED);

    auto doutDesc = context_->GetInputDesc(INPUT_DOUT);
    OP_CHECK_NULL_WITH_CONTEXT(context_, doutDesc);
    ge::DataType doutDtype = doutDesc->GetDataType();
    OP_CHECK_IF(
        dtype_ != doutDtype,
        OP_LOGE(
            context_->GetNodeName(), "the dtype of dout %s should be same with predicted %s.",
            ge::TypeUtils::DataTypeToAscendString(doutDtype).GetString(),
            ge::TypeUtils::DataTypeToAscendString(dtype_).GetString()),
        return ge::GRAPH_FAILED);

    auto weightDesc = context_->GetOptionalInputDesc(INPUT_WEIGHT);
    if (weightDesc != nullptr) {
        hasWeight_ = true;
        ge::DataType weightDtype = weightDesc->GetDataType();
        OP_CHECK_IF(
            dtype_ != weightDtype,
            OP_LOGE(
                context_->GetNodeName(), "the dtype of weight %s should be same with predicted %s.",
                ge::TypeUtils::DataTypeToAscendString(weightDtype).GetString(),
                ge::TypeUtils::DataTypeToAscendString(dtype_).GetString()),
            return ge::GRAPH_FAILED);
    }
    auto posWeightDesc = context_->GetOptionalInputDesc(INPUT_POS_WEIGHT);
    if (posWeightDesc != nullptr) {
        hasPosWeight_ = true;
        ge::DataType posWeightDtype = posWeightDesc->GetDataType();
        OP_CHECK_IF(
            dtype_ != posWeightDtype,
            OP_LOGE(
                context_->GetNodeName(), "the dtype of weight %s should be same with predicted %s.",
                ge::TypeUtils::DataTypeToAscendString(posWeightDtype).GetString(),
                ge::TypeUtils::DataTypeToAscendString(dtype_).GetString()),
            return ge::GRAPH_FAILED);
    }

    auto gradientDesc = context_->GetOutputDesc(OUTPUT_GRADIENT);
    OP_CHECK_NULL_WITH_CONTEXT(context_, gradientDesc);
    ge::DataType gradientDtype = gradientDesc->GetDataType();
    OP_CHECK_IF(
        dtype_ != gradientDtype,
        OP_LOGE(
            context_->GetNodeName(), "the dtype of dout %s should be same with predicted %s.",
            ge::TypeUtils::DataTypeToAscendString(gradientDtype).GetString(),
            ge::TypeUtils::DataTypeToAscendString(dtype_).GetString()),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SigmoidCrossEntropyWithLogitsGradV2Tiling::GetScale()
{
    auto predictShapePtr = context_->GetInputShape(INPUT_PREDICT);
    OP_CHECK_NULL_WITH_CONTEXT(context_, predictShapePtr);
    auto predictShape = predictShapePtr->GetStorageShape();
    int64_t predictNumel = 1;
    for (size_t i = 0; i < predictShape.GetDimNum(); ++i) {
        predictNumel *= predictShape.GetDim(i);
    }
    scale_ = 1.0f / std::max(1.0f, static_cast<float>(predictNumel));
    return ge::GRAPH_SUCCESS;
}

void SigmoidCrossEntropyWithLogitsGradV2Tiling::DumpTilingInfo()
{
    std::ostringstream info;
    info << "tilingId: " << tilingKey_;
    info << "hasWeight: " << hasWeight_;
    info << "hasPosWeight: " << hasPosWeight_;
    info << "isMean: " << isMean_;
    info << "scale: " << scale_;
    OP_LOGI(context_->GetNodeName(), "%s", info.str().c_str());
}

ge::graphStatus SigmoidCrossEntropyWithLogitsGradV2Tiling::GetShapeAttrsInfo()
{
    // 校验并获取输入参数
    if (GetDtype() == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }

    auto* attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    const char* reductionStr = attrs->GetAttrPointer<char>(ATTR_REDUCTION_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, reductionStr);
    if (strcmp(reductionStr, "mean") == 0) {
        isMean_ = true;
        return GetScale();
    } else if ((strcmp(reductionStr, "none") != 0) && (strcmp(reductionStr, "sum") != 0)) {
        OP_LOGE(context_->GetNodeName(), "reduction should be one of ['none', 'mean', 'sum'].");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SigmoidCrossEntropyWithLogitsGradV2Tiling::GetPlatformInfo()
{
    return ge::GRAPH_SUCCESS;
}

bool SigmoidCrossEntropyWithLogitsGradV2Tiling::IsCapable()
{
    return true;
}

ge::graphStatus SigmoidCrossEntropyWithLogitsGradV2Tiling::DoOpTiling()
{
    ge::graphStatus ret = ge::GRAPH_SUCCESS;
    if (dtype_ == ge::DT_FLOAT) {
        SIGMOID_CROSS_ENTROPY_WITH_LOGITS_GRAD_V2_DO_BROADCAST_TILING(float);
    } else if (dtype_ == ge::DT_FLOAT16) {
        SIGMOID_CROSS_ENTROPY_WITH_LOGITS_GRAD_V2_DO_BROADCAST_TILING(half);
    } else if (dtype_ == ge::DT_BF16) {
        SIGMOID_CROSS_ENTROPY_WITH_LOGITS_GRAD_V2_DO_BROADCAST_TILING(Ops::Base::bfloat16_t);
    }
    DumpTilingInfo();

    return ret;
}

ge::graphStatus SigmoidCrossEntropyWithLogitsGradV2Tiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SigmoidCrossEntropyWithLogitsGradV2Tiling::GetWorkspaceSize()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SigmoidCrossEntropyWithLogitsGradV2Tiling::PostTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t SigmoidCrossEntropyWithLogitsGradV2Tiling::GetTilingKey() const
{
    return tilingKey_;
}

REGISTER_OPS_TILING_TEMPLATE(SigmoidCrossEntropyWithLogitsGradV2, SigmoidCrossEntropyWithLogitsGradV2Tiling, 0);
} // namespace optiling

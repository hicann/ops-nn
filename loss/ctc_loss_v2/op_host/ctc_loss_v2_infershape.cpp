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
 * \file ctc_loss_v2_infershape.cpp
 * \brief
 */
#include "log/log.h"
#include "error_util.h"
#include "util/shape_util.h"
#include "register/op_impl_registry.h"
#include "common/op_util.h"

namespace {
constexpr int64_t LOG_ALPHA_LEN = 3;
constexpr int64_t STEP_INFO_IDX = 1;
constexpr int64_t LABEL_INFO_IDX = 2;
constexpr int64_t LABEL_COE = 2;
constexpr int64_t NUM_ALIGN = 8;
constexpr int64_t NEG_LOG_LIKELIHOOD_IDX = 0;
constexpr int64_t LOG_ALPHA_IDX = 1;
constexpr int64_t INPUT_TARGETS_IDX = 1;
constexpr int64_t INPUT_LENGTHS_IDX = 2;
constexpr int64_t TARGET_LENGTHS_IDX = 3;
constexpr int64_t EXPECTED_LOG_PROBS_DIM_NUM = 3;
constexpr int64_t EXPECTED_TARGETS_DIM_NUM_1D = 1;
constexpr int64_t EXPECTED_TARGETS_DIM_NUM_2D = 2;
} // namespace
using namespace ge;
namespace ops {
template <typename T>
static ge::graphStatus GetMaxValueAndSumFromTensor(
    const T* value, const int64_t batchSize, int64_t& maxLength, int64_t& targetLengthsSum)
{
    for (int64_t i = 0; i < batchSize; i++) {
        T currentValue = value[i];
        maxLength = maxLength > currentValue ? maxLength : currentValue;
        targetLengthsSum = targetLengthsSum + currentValue;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetMaxTargetLengths(
    const gert::Tensor* targetLengthsTensor, const int64_t batchSize, int64_t& maxLength, int64_t& targetLengthsSum)
{
    auto status = ge::GRAPH_SUCCESS;
    if (targetLengthsTensor->GetDataType() == ge::DT_INT32) {
        const int32_t* value = targetLengthsTensor->GetData<int32_t>();
        status = GetMaxValueAndSumFromTensor<int32_t>(value, batchSize, maxLength, targetLengthsSum);
    } else {
        const int64_t* value = targetLengthsTensor->GetData<int64_t>();
        status = GetMaxValueAndSumFromTensor<int64_t>(value, batchSize, maxLength, targetLengthsSum);
    }
    return status;
}

static ge::graphStatus CheckTensorAndBlankValid(
    const gert::InferShapeContext* context, const gert::Shape* logProbsShape)
{
    int64_t logProbsDimNum = logProbsShape->GetDimNum();
    if (logProbsDimNum != EXPECTED_LOG_PROBS_DIM_NUM) {
        OP_LOGE(context->GetNodeName(), "the dim num of log_probs should be 3, but got %ld.", logProbsDimNum);
        return ge::GRAPH_FAILED;
    }
    int64_t const symbolSet = logProbsShape->GetDim(2); // symbol set size info
    OP_CHECK_IF(
        symbolSet == 0, OP_LOGE(context->GetNodeName(), "The value of symbolSet cannot be 0."), return GRAPH_FAILED);
    auto const targetsShape = context->GetInputShape(INPUT_TARGETS_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, targetsShape);
    int64_t targetsDimNum = targetsShape->GetDimNum();
    if ((targetsDimNum != EXPECTED_TARGETS_DIM_NUM_1D) && (targetsDimNum != EXPECTED_TARGETS_DIM_NUM_2D)) {
        OP_LOGE(context->GetNodeName(), "the dim num of targets should be 1 or 2, but got %ld.", targetsDimNum);
        return ge::GRAPH_FAILED;
    }

    auto const attrs = context->GetAttrs();
    const int64_t* blank = attrs->GetAttrPointer<int64_t>(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, blank);
    if (*blank < 0) {
        OP_LOGE(context->GetNodeName(), "blank must be greater and equal than 0.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckTarget(const gert::InferShapeContext* context, int64_t targetLengthsSum)
{
    auto const targetsShape = context->GetInputShape(INPUT_TARGETS_IDX);
    if ((targetsShape->GetDimNum() == 1) && (targetsShape->GetShapeSize() != targetLengthsSum) &&
        (targetsShape->GetShapeSize() >= 0)) {
        OP_LOGE(
            context->GetNodeName(),
            "targets is 1D, the size of targets should be euqal to the sum of target_lengths, but got targets %s, "
            "the sum of target_lengths %ld.",
            Shape2String(*targetsShape).c_str(), targetLengthsSum);
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CTCLossV2InferShapeFunc(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "CTCLossV2InferShape run.");
    auto const logProbsShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, logProbsShape);

    auto negLossShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, negLossShape);
    auto logAlphaShape = context->GetOutputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, logAlphaShape);

    if (Ops::Base::IsUnknownRank(*logProbsShape)) {
        OP_LOGD(context->GetNodeName(), "log_probs shape is UnknownRank, set neg_loss and log_alpha shape to -2");
        Ops::Base::SetUnknownRank(*negLossShape);
        Ops::Base::SetUnknownRank(*logAlphaShape);
        return ge::GRAPH_SUCCESS;
    }

    if (CheckTensorAndBlankValid(context, logProbsShape) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    int64_t const timeStep = logProbsShape->GetDim(0);  // time step size info
    int64_t const batchSize = logProbsShape->GetDim(1); // batch size size info

    negLossShape->SetDimNum(1);
    negLossShape->SetDim(0, batchSize);
    OP_LOGD(context->GetNodeName(), "negLossShape: %s.", Shape2String(*negLossShape).c_str());

    logAlphaShape->SetDimNum(LOG_ALPHA_LEN);
    auto targetLengthsTensor = context->GetInputTensor(TARGET_LENGTHS_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, targetLengthsTensor);

    if (Ops::Nn::IsConstTensor(targetLengthsTensor)) {
        logAlphaShape->SetDim(0, batchSize);
        logAlphaShape->SetDim(STEP_INFO_IDX, timeStep);
        int64_t labelLen = -1;
        int64_t targetLengthsSum = 0;
        if (GetMaxTargetLengths(targetLengthsTensor, batchSize, labelLen, targetLengthsSum) != ge::GRAPH_SUCCESS) {
            OP_LOGE(context->GetNodeName(), "GetMaxTargetLengths failed.");
            return ge::GRAPH_FAILED;
        }

        if (CheckTarget(context, targetLengthsSum) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }

        int64_t alphaTailSizeAlign = (labelLen * LABEL_COE + 1 + NUM_ALIGN - 1) / NUM_ALIGN * NUM_ALIGN;
        alphaTailSizeAlign = labelLen < 0 ? -1 : alphaTailSizeAlign;
        logAlphaShape->SetDim(LABEL_INFO_IDX, alphaTailSizeAlign);
        OP_LOGD(context->GetNodeName(), "logAlphaShape: %s.", Shape2String(*logAlphaShape).c_str());
        return ge::GRAPH_SUCCESS;
    } else {
        OP_LOGD(context->GetNodeName(), "target_lengths is not ConstTensor, set log_alpha shape to -1");
        logAlphaShape->SetDim(0, -1);
        logAlphaShape->SetDim(STEP_INFO_IDX, -1);
        logAlphaShape->SetDim(LABEL_INFO_IDX, -1);
        return ge::GRAPH_SUCCESS;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CTCLossV2InferDataTypeFunc(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "CTCLossV2InferDataTypeFunc begin");
    auto inputDtype = context->GetInputDataType(0);
    OP_LOGD(context->GetNodeName(), "log_probs dtype: %s", Ops::Base::ToString(inputDtype).c_str());
    context->SetOutputDataType(NEG_LOG_LIKELIHOOD_IDX, inputDtype);
    context->SetOutputDataType(LOG_ALPHA_IDX, inputDtype);
    OP_LOGD(
        context->GetNodeName(), "neg_log_likelihood dtype: %s",
        Ops::Base::ToString(context->GetOutputDataType(NEG_LOG_LIKELIHOOD_IDX)).c_str());
    OP_LOGD(
        context->GetNodeName(), "log_alpha dtype: %s",
        Ops::Base::ToString(context->GetOutputDataType(LOG_ALPHA_IDX)).c_str());
    OP_LOGD(context->GetNodeName(), "CTCLossV2InferDataTypeFunc end");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(CTCLossV2)
    .InferShape(CTCLossV2InferShapeFunc)
    .InferDataType(CTCLossV2InferDataTypeFunc)
    .InputsDataDependency({INPUT_LENGTHS_IDX, TARGET_LENGTHS_IDX});
} // namespace ops

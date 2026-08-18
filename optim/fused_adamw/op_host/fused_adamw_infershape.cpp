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
 * \file fused_adamw_infershape.cpp
 * \brief
 */

#include "log/log.h"
#include "register/op_impl_registry.h"

using namespace ge;
static constexpr size_t INPUT_PARAMS_INDEX = 0;
static constexpr size_t INPUT_GRADS_INDEX = 1;
static constexpr size_t INPUT_EXP_AVGS_INDEX = 2;
static constexpr size_t INPUT_EXP_AVG_SQS_INDEX = 3;
static constexpr size_t INPUT_MAX_EXP_AVG_SQS_INDEX = 4;
static constexpr size_t OUTPUT_PARAMS_INDEX = 0;
static constexpr size_t OUTPUT_EXP_AVGS_INDEX = 1;
static constexpr size_t OUTPUT_EXP_AVG_SQS_INDEX = 2;
static constexpr size_t OUTPUT_MAX_EXP_AVG_SQS_INDEX = 3;

namespace ops {
static ge::graphStatus InferShapeForFusedAdamW(gert::InferShapeContext* context)
{
    OP_LOGD(context, "Begin to do InferShapeForFusedAdamW.");

    auto computeNodeInfo = context->GetComputeNodeInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, computeNodeInfo);

    auto paramsInstanceInfo = computeNodeInfo->GetInputInstanceInfo(INPUT_PARAMS_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, paramsInstanceInfo);
    auto inputNum = paramsInstanceInfo->GetInstanceNum();
    if (inputNum == 0) {
        OP_LOGE(context, "input num must be greater than 0");
        return ge::GRAPH_FAILED;
    }

    auto paramsOutInstanceInfo = context->GetIrOutputInstanceInfo(OUTPUT_PARAMS_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, paramsOutInstanceInfo);
    auto expAvgsOutInstanceInfo = context->GetIrOutputInstanceInfo(OUTPUT_EXP_AVGS_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, expAvgsOutInstanceInfo);
    auto expAvgSqsOutInstanceInfo = context->GetIrOutputInstanceInfo(OUTPUT_EXP_AVG_SQS_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, expAvgSqsOutInstanceInfo);
    auto maxExpAvgSqsOutInstanceInfo = context->GetIrOutputInstanceInfo(OUTPUT_MAX_EXP_AVG_SQS_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, maxExpAvgSqsOutInstanceInfo);

    for (uint32_t i = 0; i < inputNum; i++) {
        const gert::Shape* paramsShape = context->GetDynamicInputShape(INPUT_PARAMS_INDEX, i);
        OP_CHECK_NULL_WITH_CONTEXT(context, paramsShape);
        const gert::Shape* gradsShape = context->GetDynamicInputShape(INPUT_GRADS_INDEX, i);
        OP_CHECK_NULL_WITH_CONTEXT(context, gradsShape);
        const gert::Shape* expAvgsShape = context->GetDynamicInputShape(INPUT_EXP_AVGS_INDEX, i);
        OP_CHECK_NULL_WITH_CONTEXT(context, expAvgsShape);
        const gert::Shape* expAvgSqsShape = context->GetDynamicInputShape(INPUT_EXP_AVG_SQS_INDEX, i);
        OP_CHECK_NULL_WITH_CONTEXT(context, expAvgSqsShape);
        const gert::Shape* maxExpAvgSqsShape = context->GetDynamicInputShape(INPUT_MAX_EXP_AVG_SQS_INDEX, i);
        OP_CHECK_NULL_WITH_CONTEXT(context, maxExpAvgSqsShape);

        if (*paramsShape != *gradsShape || *paramsShape != *expAvgsShape || *paramsShape != *expAvgSqsShape ||
            *paramsShape != *maxExpAvgSqsShape) {
            OP_LOGE(context, "params, grads, exp_avgs, exp_avg_sqs and max_exp_avg_sqs should have the same shape");
            return ge::GRAPH_FAILED;
        }

        gert::Shape* paramsRefShape = context->GetOutputShape(paramsOutInstanceInfo->GetInstanceStart() + i);
        OP_CHECK_NULL_WITH_CONTEXT(context, paramsRefShape);
        gert::Shape* expAvgsRefShape = context->GetOutputShape(expAvgsOutInstanceInfo->GetInstanceStart() + i);
        OP_CHECK_NULL_WITH_CONTEXT(context, expAvgsRefShape);
        gert::Shape* expAvgSqsRefShape = context->GetOutputShape(expAvgSqsOutInstanceInfo->GetInstanceStart() + i);
        OP_CHECK_NULL_WITH_CONTEXT(context, expAvgSqsRefShape);
        gert::Shape* maxExpAvgSqsRefShape = context->GetOutputShape(maxExpAvgSqsOutInstanceInfo->GetInstanceStart() +
                                                                    i);
        OP_CHECK_NULL_WITH_CONTEXT(context, maxExpAvgSqsRefShape);

        *paramsRefShape = *paramsShape;
        *expAvgsRefShape = *expAvgsShape;
        *expAvgSqsRefShape = *expAvgSqsShape;
        *maxExpAvgSqsRefShape = *maxExpAvgSqsShape;
    }

    OP_LOGD(context, "End to do InferShapeForFusedAdamW.");
    return ge::GRAPH_SUCCESS;
}

static graphStatus InferDataTypeForFusedAdamW(gert::InferDataTypeContext* context)
{
    auto computeNodeInfo = context->GetComputeNodeInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, computeNodeInfo);

    auto paramsInstanceInfo = computeNodeInfo->GetInputInstanceInfo(INPUT_PARAMS_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, paramsInstanceInfo);
    auto inputNum = paramsInstanceInfo->GetInstanceNum();

    auto paramsOutInstanceInfo = context->GetIrOutputInstanceInfo(OUTPUT_PARAMS_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, paramsOutInstanceInfo);
    auto expAvgsOutInstanceInfo = context->GetIrOutputInstanceInfo(OUTPUT_EXP_AVGS_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, expAvgsOutInstanceInfo);
    auto expAvgSqsOutInstanceInfo = context->GetIrOutputInstanceInfo(OUTPUT_EXP_AVG_SQS_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, expAvgSqsOutInstanceInfo);
    auto maxExpAvgSqsOutInstanceInfo = context->GetIrOutputInstanceInfo(OUTPUT_MAX_EXP_AVG_SQS_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, maxExpAvgSqsOutInstanceInfo);

    for (uint32_t i = 0; i < inputNum; i++) {
        context->SetOutputDataType(paramsOutInstanceInfo->GetInstanceStart() + i,
                                   context->GetDynamicInputDataType(INPUT_PARAMS_INDEX, i));
        context->SetOutputDataType(expAvgsOutInstanceInfo->GetInstanceStart() + i,
                                   context->GetDynamicInputDataType(INPUT_EXP_AVGS_INDEX, i));
        context->SetOutputDataType(expAvgSqsOutInstanceInfo->GetInstanceStart() + i,
                                   context->GetDynamicInputDataType(INPUT_EXP_AVG_SQS_INDEX, i));
        context->SetOutputDataType(maxExpAvgSqsOutInstanceInfo->GetInstanceStart() + i,
                                   context->GetDynamicInputDataType(INPUT_MAX_EXP_AVG_SQS_INDEX, i));
    }
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(FusedAdamw).InferShape(InferShapeForFusedAdamW).InferDataType(InferDataTypeForFusedAdamW);
} // namespace ops

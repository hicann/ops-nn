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
 * \file fused_sgd_infershape.cpp
 * \brief
 */
#include "log/log.h"
#include "register/op_impl_registry.h"

using namespace ge;
static constexpr size_t INPUT_PARAMS_INDEX = 0;
static constexpr size_t INPUT_GRADS_INDEX = 1;
static constexpr size_t INPUT_MOMENTUM_BUFFER_INDEX = 2;
static constexpr size_t OUTPUT_PARAMS_INDEX = 0;
static constexpr size_t OUTPUT_GRADS_INDEX = 1;
static constexpr size_t OUTPUT_MOMENTUM_BUFFER_INDEX = 2;

namespace ops {
static ge::graphStatus InferShapeForFusedSgd(gert::InferShapeContext* context)
{
    OP_LOGD(context, "Begin to do InferShapeForFusedSgd.");

    auto computeNodeInfo = context->GetComputeNodeInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, computeNodeInfo);

    auto paramsInstanceInfo = computeNodeInfo->GetInputInstanceInfo(INPUT_PARAMS_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, paramsInstanceInfo);
    auto inputNum = paramsInstanceInfo->GetInstanceNum();
    if (inputNum == 0) {
        OP_LOGE(context, "input num must be greater than 0");
        return ge::GRAPH_FAILED;
    }

    // 判断tensorlist是否为空
    auto momentumBufferListInput = context->GetDynamicInputShape(INPUT_MOMENTUM_BUFFER_INDEX, 0);
    if (momentumBufferListInput != nullptr) {
        uint32_t momentumBufferListDims = momentumBufferListInput->GetDimNum();
        bool flag = true;
        if (momentumBufferListDims < 1) {
            flag = false;
        }
        if (flag) {
            for(uint32_t i = 0; i < momentumBufferListDims; i++) {
                int64_t dimValue = momentumBufferListInput->GetDim(i);
                if (dimValue == 0) {
                    flag = false;
                    break;
                }
            }
        }
        if (!flag) {
            momentumBufferListInput = nullptr;
        }
    }

    auto paramsOutInstanceInfo = context->GetIrOutputInstanceInfo(OUTPUT_PARAMS_INDEX);
    auto gradsOutInstanceInfo = context->GetIrOutputInstanceInfo(OUTPUT_GRADS_INDEX);
    auto momentumOutInstanceInfo = context->GetIrOutputInstanceInfo(OUTPUT_MOMENTUM_BUFFER_INDEX);

    for (uint32_t i = 0; i < inputNum; i++) {
        const gert::Shape* paramsShape = context->GetDynamicInputShape(INPUT_PARAMS_INDEX, i);
        OP_CHECK_NULL_WITH_CONTEXT(context, paramsShape);
        const gert::Shape* gradsShape = context->GetDynamicInputShape(INPUT_GRADS_INDEX, i);
        OP_CHECK_NULL_WITH_CONTEXT(context, gradsShape);
        const gert::Shape* momentumShape = nullptr;
        if (momentumBufferListInput != nullptr) {
            momentumShape = context->GetDynamicInputShape(INPUT_MOMENTUM_BUFFER_INDEX, i);
            OP_CHECK_NULL_WITH_CONTEXT(context, momentumShape);
        }

        if (*paramsShape != *gradsShape || (momentumShape != nullptr && *paramsShape != *momentumShape)) {
            OP_LOGE(context, "params, grads and momentum_buffer_list should have the same shape");
            return ge::GRAPH_FAILED;
        }

        gert::Shape* paramsRefShape = context->GetOutputShape(paramsOutInstanceInfo->GetInstanceStart() + i);
        OP_CHECK_NULL_WITH_CONTEXT(context, paramsRefShape);
        gert::Shape* gradsRefShape = context->GetOutputShape(gradsOutInstanceInfo->GetInstanceStart() + i);
        OP_CHECK_NULL_WITH_CONTEXT(context, gradsRefShape);
        gert::Shape* momentumRefShape = nullptr;
        if (momentumBufferListInput != nullptr) {
            momentumRefShape = context->GetOutputShape(momentumOutInstanceInfo->GetInstanceStart() + i);
            OP_CHECK_NULL_WITH_CONTEXT(context, momentumRefShape);
        }

        *paramsRefShape = *paramsShape;
        *gradsRefShape = *gradsShape;
        if (momentumBufferListInput != nullptr) {
            *momentumRefShape = *momentumShape;
        }
    }

    OP_LOGD(context, "End to do InferShapeForFusedSgd.");
    return ge::GRAPH_SUCCESS;
}

static graphStatus InferDataTypeForFusedSgd(gert::InferDataTypeContext* context)
{
    auto computeNodeInfo = context->GetComputeNodeInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, computeNodeInfo);

    auto paramsInstanceInfo = computeNodeInfo->GetInputInstanceInfo(INPUT_PARAMS_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, paramsInstanceInfo);
    auto inputNum = paramsInstanceInfo->GetInstanceNum();

    auto paramsOutInstanceInfo = context->GetIrOutputInstanceInfo(OUTPUT_PARAMS_INDEX);
    auto gradsOutInstanceInfo = context->GetIrOutputInstanceInfo(OUTPUT_GRADS_INDEX);
    auto momentumOutInstanceInfo = context->GetIrOutputInstanceInfo(OUTPUT_MOMENTUM_BUFFER_INDEX);

    for (uint32_t i = 0; i < inputNum; i++) {
        context->SetOutputDataType(paramsOutInstanceInfo->GetInstanceStart() + i,
            context->GetDynamicInputDataType(INPUT_PARAMS_INDEX, i));
        context->SetOutputDataType(gradsOutInstanceInfo->GetInstanceStart() + i,
            context->GetDynamicInputDataType(INPUT_GRADS_INDEX, i));
        context->SetOutputDataType(momentumOutInstanceInfo->GetInstanceStart() + i,
            context->GetDynamicInputDataType(INPUT_MOMENTUM_BUFFER_INDEX, i));
    }
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(FusedSgd)
    .InferShape(InferShapeForFusedSgd)
    .InferDataType(InferDataTypeForFusedSgd);
} // namespace ops
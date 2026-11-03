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
 * \file sgd_infershape.cpp
 * \brief SGD InferShape（InferDataType 见 op_graph/sgd_graph_infer.cpp）
 *
 * 行为对齐 ascend910b（canndev nn_training_ops.cc:1906-1973）：
 *   - parameters_out 的 shape / dtype 等于 parameters；
 *   - rank 落在 1 ~ 8 之外即 GRAPH_FAILED（故 rank-0 标量被拒）；
 *   - UNKNOWN_RANK(-2) 透传 —— GE 下 dims = {-2}（DimNum == 1），不触发 rank 拒绝；
 *   - nesterov == true 时 dampening 必须为 0；weight_decay >= 0。
 *
 * InferShape与op_host/arch35/sgd_tiling.cpp分别校验属性，保证图推导和
 * 独立Tiling入口均拒绝非法属性。
 */
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "op_host/infershape_elewise_util.h"

using namespace ge;

namespace ops {
namespace {
constexpr size_t PARAMETERS_INDEX = 0;
constexpr size_t MAX_DIM_NUM = 8;
constexpr size_t MIN_DIM_NUM = 1;
constexpr size_t ATTR_IDX_DAMPENING = 0;
constexpr size_t ATTR_IDX_WEIGHT_DECAY = 1;
constexpr size_t ATTR_IDX_NESTEROV = 2;
constexpr int64_t UNKNOWN_RANK_DIM = -2;

// UNKNOWN_RANK 在 GE 下表现为 dims == {-2}，须透传而非按 rank 拒绝。
bool IsUnknownRank(const gert::Shape* shape) { return shape->GetDimNum() == 1 && shape->GetDim(0) == UNKNOWN_RANK_DIM; }

ge::graphStatus CheckInputShapes(gert::InferShapeContext* context, const gert::Shape* paramShape)
{
    const char* inputNames[] = {"parameters", "gradient", "learning_rate", "accum", "momentum", "stat"};
    for (size_t input = 0; input < 6; ++input) {
        const auto* shape = context->GetInputShape(input);
        OP_CHECK_NULL_WITH_CONTEXT(context, shape);
        if (IsUnknownRank(shape)) {
            continue;
        }
        const bool isScalarInput = input == 2 || input == 4;
        const size_t rank = shape->GetDimNum();
        if (!isScalarInput) {
            OP_CHECK_IF(rank < MIN_DIM_NUM || rank > MAX_DIM_NUM,
                        OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), inputNames[input],
                                                     std::to_string(rank).c_str(), "1 ~ 8 dims"),
                        return ge::GRAPH_FAILED);
        }
        for (size_t axis = 0; axis < rank; ++axis) {
            const int64_t dim = shape->GetDim(axis);
            OP_CHECK_IF(dim == 0 && !isScalarInput,
                        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(context->GetNodeName(), inputNames[input], "0",
                                                                  "empty tensors are not supported"),
                        return ge::GRAPH_FAILED);
            OP_CHECK_IF(isScalarInput && (dim == 0 || dim > 1),
                        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), inputNames[input],
                                                              Ops::Base::ToString(*shape).c_str(),
                                                              "must contain exactly one element"),
                        return ge::GRAPH_FAILED);
        }
        // Unknown dimensions and unknown rank are resolved at runtime. Only
        // reject shape relationships that are already known to be invalid.
        if (isScalarInput || IsUnknownRank(paramShape)) {
            continue;
        }
        OP_CHECK_IF(rank != paramShape->GetDimNum(),
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                        context->GetNodeName(), (std::string("parameters, ") + inputNames[input]).c_str(),
                        Ops::Base::ToString(std::vector<const gert::Shape*>{paramShape, shape}).c_str(),
                        "must have the same shape as parameters"),
                    return ge::GRAPH_FAILED);
        for (size_t axis = 0; axis < rank; ++axis) {
            const int64_t dim = shape->GetDim(axis);
            const int64_t paramDim = paramShape->GetDim(axis);
            OP_CHECK_IF(dim >= 0 && paramDim >= 0 && dim != paramDim,
                        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                            context->GetNodeName(), (std::string("parameters, ") + inputNames[input]).c_str(),
                            Ops::Base::ToString(std::vector<const gert::Shape*>{paramShape, shape}).c_str(),
                            "must have the same shape as parameters"),
                        return ge::GRAPH_FAILED);
        }
    }
    return ge::GRAPH_SUCCESS;
}
} // namespace

static ge::graphStatus InferShapeForSgd(gert::InferShapeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context, "InferShapeForSgd begin.");

    const auto* attrs = context->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OP_LOGE(context, "Get attrs failed."), return ge::GRAPH_FAILED);
    const float* dampening = attrs->GetAttrPointer<float>(ATTR_IDX_DAMPENING);
    const float* weightDecay = attrs->GetAttrPointer<float>(ATTR_IDX_WEIGHT_DECAY);
    const bool* nesterov = attrs->GetAttrPointer<bool>(ATTR_IDX_NESTEROV);
    const float dampeningVal = (dampening != nullptr) ? *dampening : 0.0f;
    const float weightDecayVal = (weightDecay != nullptr) ? *weightDecay : 0.0f;
    const bool nesterovVal = (nesterov != nullptr) ? *nesterov : false;

    OP_CHECK_IF(
        nesterovVal && dampeningVal != 0.0f,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "dampening", std::to_string(dampeningVal).c_str(),
                                              "attr dampening must be 0 when attr nesterov is true"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(!(weightDecayVal >= 0.0f),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "weight_decay",
                                                      std::to_string(weightDecayVal).c_str(),
                                                      "attr weight_decay must be more than or equal to 0"),
                return ge::GRAPH_FAILED);

    const gert::Shape* paramShape = context->GetInputShape(PARAMETERS_INDEX);
    OP_CHECK_IF(paramShape == nullptr, OP_LOGE(context, "Get input shape of parameters returns nullptr."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckInputShapes(context, paramShape) != ge::GRAPH_SUCCESS,
                OP_LOGD(context, "Input shape validation failed."), return ge::GRAPH_FAILED);

    ge::graphStatus ret = Ops::Base::InferShape4Elewise(context);
    OP_CHECK_IF(ret == ge::GRAPH_FAILED, OP_LOGE(context, "InferShapeForSgd failed."), return ge::GRAPH_FAILED);
    OP_LOGD(context, "InferShapeForSgd end.");
    return ret;
}

// InferDataType is registered separately in op_graph/sgd_graph_infer.cpp.
IMPL_OP_INFERSHAPE(SGD).InferShape(InferShapeForSgd);
} // namespace ops

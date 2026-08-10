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
 * 注：ascend910b 把这两条属性校验放在 InferShape 里而非 Verify（其 SGDVerify 只调
 * CheckSgdDimension），属职责错位（见 01 §6.5）。ascend950 侧在 InferShape 与 Tiling
 * 两个独立入口都做，保证任一入口进来都拦得住。
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
} // namespace

static ge::graphStatus InferShapeForSgd(gert::InferShapeContext* context)
{
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
    OP_CHECK_IF(weightDecayVal < 0.0f,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "weight_decay",
                                                      std::to_string(weightDecayVal).c_str(),
                                                      "attr weight_decay must be more than or equal to 0"),
                return ge::GRAPH_FAILED);

    // 这里守的是「框架没把 shape 递过来」（返回 nullptr），属于内部失败，不是 parameters 取值非法，
    // 所以用裸 OP_LOGE 而非 OP_LOGE_FOR_INVALID_*：后者会走 REPORT_PREDEFINED_ERR_MSG 报 EZ0009/EZ0010，
    // 对用户提示「你的 shape 不合法」，而此刻并不知道用户给了什么。真正的 shape 取值校验在下面的 dimNum 分支。
    const gert::Shape* paramShape = context->GetInputShape(PARAMETERS_INDEX);
    OP_CHECK_IF(paramShape == nullptr, OP_LOGE(context, "Get input shape of parameters returns nullptr."),
                return ge::GRAPH_FAILED);
    if (!IsUnknownRank(paramShape)) {
        const size_t dimNum = paramShape->GetDimNum();
        OP_CHECK_IF(dimNum < MIN_DIM_NUM || dimNum > MAX_DIM_NUM,
                    OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "parameters", std::to_string(dimNum).c_str(),
                                                 "1 ~ 8 dims"),
                    return ge::GRAPH_FAILED);
    }

    ge::graphStatus ret = Ops::Base::InferShape4Elewise(context);
    OP_CHECK_IF(ret == ge::GRAPH_FAILED, OP_LOGE(context, "InferShapeForSgd failed."), return ge::GRAPH_FAILED);
    OP_LOGD(context, "InferShapeForSgd end.");
    return ret;
}

// 双挂仍然必须：漏 InferDataType 会导致 GE 侧输出 dtype 推导缺失/推错，是拷模板时的高频漏项
// （step3 GATE 明列）。但 InferDataType 只在图场景使用，按交付件划分落在
// op_graph/sgd_graph_infer.cpp；此处只挂图与单算子共用的 InferShape。
IMPL_OP_INFERSHAPE(SGD).InferShape(InferShapeForSgd);
} // namespace ops

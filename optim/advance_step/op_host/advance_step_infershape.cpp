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
 * \file advance_step_infershape.cc
 * \brief
 */
#include "util/shape_util.h"
#include "log/log.h"
#include "register/op_impl_registry.h"
#include <string>
#include <vector>
#include "base/err_msg.h"

using namespace ge;

namespace ops {

static constexpr int64_t IDX_0 = 0;
static constexpr int64_t IDX_1 = 1;
static constexpr int64_t IDX_2 = 2;
static constexpr int64_t IDX_3 = 3;
static constexpr int64_t IDX_4 = 4;
static constexpr int64_t IDX_5 = 5;
static constexpr int64_t IDX_6 = 6;
static constexpr int64_t IDX_7 = 7;

static constexpr size_t SIZE_2 = 2;

// Host 形状门辅助：与 tiling CheckOptionalShapeExisting 同口径的"存在性"判定——
// 任一维度为负（未知 -1 / 未知 rank -2）时无法证明非空，跳过检查（不比 tiling 更严格）；
// 全部维度具体时 numel>0 才视为"已传入"，空张量（任意 rank）= 未传入，tiling 侧
// 跳过全部可选校验并回落 V1，门必须同样放行
static bool IsConcreteNonEmptyOptional(const gert::Shape* shape)
{
    if (shape == nullptr) {
        return false;
    }
    int64_t numel = 1;
    for (size_t i = 0; i < shape->GetDimNum(); i++) {
        const int64_t dim = shape->GetDim(i);
        if (dim < 0) {
            return false;
        }
        numel *= dim;
    }
    return numel > 0;
}

// Host 形状门辅助：描述符含未知 rank 标记(-2)时跳过该输入的全部检查
static bool HasUnknownRankMark(const gert::Shape* shape)
{
    for (size_t i = 0; i < shape->GetDimNum(); i++) {
        if (shape->GetDim(i) == ge::UNKNOWN_DIM_NUM) {
            return true;
        }
    }
    return false;
}

// Host 形状门（GE 图模式早期校验）：rank、空 Tensor 与状态张量跨输入一致性。
// 仅覆盖路径无关的可判定子集；动态 shape(-1) 跳过取值检查，未知 rank(-2) 跳过
// 全部检查，V1/V2 路径相关的完整形状约束仍由 tiling 阶段校验。
static ge::graphStatus CheckAdvanceStepInputShape(gert::InferShapeContext* context)
{
    // 必选 1-D 状态张量：input_tokens/input_positions/seq_lens/slot_mapping
    for (int64_t idx : {IDX_0, IDX_2, IDX_3, IDX_4}) {
        auto shape = context->GetInputShape(idx);
        OP_CHECK_NULL_WITH_CONTEXT(context, shape);
        if (HasUnknownRankMark(shape)) {
            continue;
        }
        if (shape->GetDimNum() != 1) {
            OP_LOGE(context, "input %ld must be 1-D, got rank %zu.", idx, shape->GetDimNum());
            return ge::GRAPH_FAILED;
        }
        if (shape->GetDim(0) == 0) {
            OP_LOGE(context, "input %ld does not support empty tensor.", idx);
            return ge::GRAPH_FAILED;
        }
    }
    // 必选 2-D 张量：sampled_token_ids/block_tables
    for (int64_t idx : {IDX_1, IDX_5}) {
        auto shape = context->GetInputShape(idx);
        OP_CHECK_NULL_WITH_CONTEXT(context, shape);
        if (HasUnknownRankMark(shape)) {
            continue;
        }
        if (shape->GetDimNum() != SIZE_2) {
            OP_LOGE(context, "input %ld must be 2-D, got rank %zu.", idx, shape->GetDimNum());
            return ge::GRAPH_FAILED;
        }
        if (shape->GetDim(0) == 0 || shape->GetDim(1) == 0) {
            OP_LOGE(context, "input %ld does not support empty tensor.", idx);
            return ge::GRAPH_FAILED;
        }
    }
    // 可选输入：使用 GetOptionalInputShape 按实例化状态访问（未连接返回 nullptr，
    // 直接 GetInputShape 会越过已连接实例布局造成越界）。存在性判定与 tiling
    // CheckOptionalShapeExisting 同口径：空张量（numel==0，任意 rank）视为未传入，
    // 跳过检查回落 V1；仅在已传入且 rank 已知时校验 rank（不比 tiling 更严格）
    auto specShape = context->GetOptionalInputShape(IDX_6);
    if (IsConcreteNonEmptyOptional(specShape) && specShape->GetDimNum() != SIZE_2) {
        OP_LOGE(context, "input %ld must be 2-D, got rank %zu.", IDX_6, specShape->GetDimNum());
        return ge::GRAPH_FAILED;
    }
    auto accShape = context->GetOptionalInputShape(IDX_7);
    if (IsConcreteNonEmptyOptional(accShape) && accShape->GetDimNum() != 1) {
        OP_LOGE(context, "input %ld must be 1-D, got rank %zu.", IDX_7, accShape->GetDimNum());
        return ge::GRAPH_FAILED;
    }
    // 跨输入一致性：4 个状态张量的第一维取具体值时应一致（V1=numSeqs，V2=numSeqs*(1+specNum)）
    int64_t refDim0 = -1;
    for (int64_t idx : {IDX_0, IDX_2, IDX_3, IDX_4}) {
        auto shape = context->GetInputShape(idx);
        if (shape->GetDimNum() == 1 && shape->GetDim(0) >= 0) {
            if (refDim0 < 0) {
                refDim0 = shape->GetDim(0);
            } else if (shape->GetDim(0) != refDim0) {
                OP_LOGE(context, "1-D state tensors must share the same length, input 0 is %ld but input %ld is %ld.",
                        refDim0, idx, shape->GetDim(0));
                return ge::GRAPH_FAILED;
            }
        }
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShape4AdvanceStep(gert::InferShapeContext* context)
{
    OP_LOGD(context, "Begin to do InferShape4AdvanceStep");

    if (CheckAdvanceStepInputShape(context) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // get input shapes
    auto x1Shape = context->GetInputShape(IDX_0);
    OP_CHECK_IF(
        x1Shape == nullptr,
        OP_LOGE_WITH_INVALID_INPUT(
            ((context) == nullptr || (context)->GetNodeName() == nullptr ? "advance_step" : (context)->GetNodeName()),
            "input_tokens"),
        return ge::GRAPH_FAILED);

    // get output shapes
    auto y1Shape = context->GetOutputShape(IDX_0);
    auto y2Shape = context->GetOutputShape(IDX_1);
    auto y3Shape = context->GetOutputShape(IDX_2);
    auto y4Shape = context->GetOutputShape(IDX_3);
    if (y1Shape == nullptr) {
        const std::string msgOpName = std::string(
            ((context) == nullptr || (context)->GetNodeName() == nullptr ? "advance_step" : (context)->GetNodeName()));
        OP_LOGE_WITHOUT_REPORT(msgOpName.c_str(), "%s",
                               "InferShape framework object y1Shape is unavailable; check the operator registration "
                               "and context construction.");
        REPORT_PREDEFINED_ERR_MSG(
            "EZ1014", (std::vector<const char*>{"opName", "reason"}),
            (std::vector<const char*>{msgOpName.c_str(), "InferShape framework object y1Shape is unavailable; check "
                                                         "the operator registration and context construction."}));
        return ge::GRAPH_FAILED;
    };
    if (y2Shape == nullptr) {
        const std::string msgOpName = std::string(
            ((context) == nullptr || (context)->GetNodeName() == nullptr ? "advance_step" : (context)->GetNodeName()));
        OP_LOGE_WITHOUT_REPORT(msgOpName.c_str(), "%s",
                               "InferShape framework object y2Shape is unavailable; check the operator registration "
                               "and context construction.");
        REPORT_PREDEFINED_ERR_MSG(
            "EZ1014", (std::vector<const char*>{"opName", "reason"}),
            (std::vector<const char*>{msgOpName.c_str(), "InferShape framework object y2Shape is unavailable; check "
                                                         "the operator registration and context construction."}));
        return ge::GRAPH_FAILED;
    };
    if (y3Shape == nullptr) {
        const std::string msgOpName = std::string(
            ((context) == nullptr || (context)->GetNodeName() == nullptr ? "advance_step" : (context)->GetNodeName()));
        OP_LOGE_WITHOUT_REPORT(msgOpName.c_str(), "%s",
                               "InferShape framework object y3Shape is unavailable; check the operator registration "
                               "and context construction.");
        REPORT_PREDEFINED_ERR_MSG(
            "EZ1014", (std::vector<const char*>{"opName", "reason"}),
            (std::vector<const char*>{msgOpName.c_str(), "InferShape framework object y3Shape is unavailable; check "
                                                         "the operator registration and context construction."}));
        return ge::GRAPH_FAILED;
    };
    if (y4Shape == nullptr) {
        const std::string msgOpName = std::string(
            ((context) == nullptr || (context)->GetNodeName() == nullptr ? "advance_step" : (context)->GetNodeName()));
        OP_LOGE_WITHOUT_REPORT(msgOpName.c_str(), "%s",
                               "InferShape framework object y4Shape is unavailable; check the operator registration "
                               "and context construction.");
        REPORT_PREDEFINED_ERR_MSG(
            "EZ1014", (std::vector<const char*>{"opName", "reason"}),
            (std::vector<const char*>{msgOpName.c_str(), "InferShape framework object y4Shape is unavailable; check "
                                                         "the operator registration and context construction."}));
        return ge::GRAPH_FAILED;
    };

    *y1Shape = *x1Shape;
    *y2Shape = *x1Shape;
    *y3Shape = *x1Shape;
    *y4Shape = *x1Shape;

    OP_LOGD(context, "End to do InferShape4AdvanceStep");
    OP_LOGI(context->GetNodeName(), "[InferShape] output0 shape=%s", Ops::Base::ToString(*y1Shape).c_str());
    return GRAPH_SUCCESS;
}

// InferDataType4AdvanceStep 已按转测布局规则拆分至 op_graph/advance_step_graph_infer.cpp
IMPL_OP_INFERSHAPE(AdvanceStep).InferShape(InferShape4AdvanceStep);
} // namespace ops

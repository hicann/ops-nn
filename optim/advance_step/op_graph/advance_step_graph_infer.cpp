/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file advance_step_graph_infer.cpp
 * \brief AdvanceStep 算子 RT2.0 InferDataType 实现（按转测布局规则自 op_host 拆分）
 *
 * 输出 dtype 推导规则：4 个 inplace 状态张量输出与输入同型，均取 input_tokens 的 dtype。
 * Host dtype 门：必选输入强制 DT_INT64；可选输入 dtype 交由 tiling 按存在性判定后校验
 * （门不比 tiling 更严格）。
 */

#include "log/log.h"
#include "register/op_impl_registry.h"

#include "advance_step_proto.h"

namespace ops {
namespace {
constexpr size_t ADVANCE_STEP_INPUT_IDX = 0;
constexpr size_t ADVANCE_STEP_REQUIRED_INPUT_NUM = 6;
constexpr size_t ADVANCE_STEP_OUTPUT_NUM = 4;
} // namespace

ge::graphStatus InferDataType4AdvanceStep(gert::InferDataTypeContext* context)
{
    if (context == nullptr) {
        OP_LOGE("AdvanceStep", "context is null.");
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context->GetNodeName(), "Begin InferDataType4AdvanceStep.");

    // Host dtype 门（GE 图模式早期校验）：算子仅支持 INT64，必选输入必须为 DT_INT64。
    // 可选输入的 dtype 不在门内校验：tiling 按"存在性"（numel>0）判定后才校验已传入
    // 的可选输入，空张量（任意 dtype）视为未传入回落 V1；InferDataType 上下文无法获知
    // shape 存在性，为不比 tiling 更严格，可选输入的 dtype 校验完全交由 tiling 阶段
    for (size_t i = 0; i < ADVANCE_STEP_REQUIRED_INPUT_NUM; i++) {
        if (context->GetInputDataType(i) != ge::DT_INT64) {
            OP_LOGE(context->GetNodeName(), "input %zu dtype must be INT64.", i);
            return ge::GRAPH_FAILED;
        }
    }

    const ge::DataType inputDtype = context->GetInputDataType(ADVANCE_STEP_INPUT_IDX);
    for (size_t i = 0; i < ADVANCE_STEP_OUTPUT_NUM; i++) {
        context->SetOutputDataType(i, inputDtype);
    }

    OP_LOGD(context->GetNodeName(), "Set 4 output dtype: %s success.", Ops::Base::ToString(inputDtype).c_str());
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(AdvanceStep).InferDataType(InferDataType4AdvanceStep);
} // namespace ops

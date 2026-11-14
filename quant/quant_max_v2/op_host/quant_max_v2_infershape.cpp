/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// QuantMaxV2 形状推导：y 与 x 同形，amax 为 [numGroupsLocal]；输出类型推导在 op_graph 注册。

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "exe_graph/runtime/infer_datatype_context.h"
#include "graph/ge_error_codes.h"
#include "op_common/log/log.h"

using namespace ge;

namespace ops {

constexpr size_t INPUT_X_INDEX = 0;
constexpr size_t INPUT_SCALE_INDEX = 1;
constexpr int64_t MAX_GROUP_NUM = 384;
constexpr size_t INPUT_GROUP_LIST_INDEX = 2;
constexpr size_t OUTPUT_Y_INDEX = 0;
constexpr size_t OUTPUT_AMAX_INDEX = 1;
constexpr size_t OUTPUT_Y_SCALE_INDEX = 2;
constexpr size_t ATTR_ROUND_MODE_INDEX = 0;
constexpr size_t ATTR_DST_TYPE_INDEX = 1;
constexpr int64_t DST_TYPE_HIFLOAT8 = 34;
constexpr int64_t DST_TYPE_FLOAT8_E5M2 = 35;
constexpr int64_t DST_TYPE_FLOAT8_E4M3FN = 36;

static const std::initializer_list<ge::DataType> SUPPORTED_X_DTYPE_LIST = {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16};

static ge::DataType GetOutputYDtype(int64_t dst_type)
{
    switch (dst_type) {
        case DST_TYPE_HIFLOAT8:
            return ge::DT_HIFLOAT8;
        case DST_TYPE_FLOAT8_E5M2:
            return ge::DT_FLOAT8_E5M2;
        case DST_TYPE_FLOAT8_E4M3FN:
            return ge::DT_FLOAT8_E4M3FN;
        default:
            return ge::DT_UNDEFINED;
    }
}

// 形状推导：y 与 x 同形；amax = [numGroups]（groupList 决定，单张量为 [1]）；y_scale 仅动态输出 [numGroups]。
static ge::graphStatus InferShape4QuantMaxV2(gert::InferShapeContext* context)
{
    const gert::Shape* inputXShape = context->GetInputShape(INPUT_X_INDEX);
    if (inputXShape == nullptr) {
        OP_LOGE("QuantMaxV2", "InferShape got nullptr x shape.");
        return ge::GRAPH_FAILED;
    }

    const gert::Shape* groupListShape = context->GetOptionalInputShape(INPUT_GROUP_LIST_INDEX);
    // group_list 决定 amax/y_scale 的形状，维数必须为 1；第 0 维为动态值 -1 或位于 [1, 384]，与 tiling 的约束一致。
    if (groupListShape != nullptr) {
        if (groupListShape->GetDimNum() != 1) {
            OP_LOGE("QuantMaxV2", "group_list must be rank one, got rank %zu.", groupListShape->GetDimNum());
            return ge::GRAPH_FAILED;
        }
        const int64_t groups = groupListShape->GetDim(0);
        if (groups != -1 && (groups < 1 || groups > MAX_GROUP_NUM)) {
            OP_LOGE("QuantMaxV2", "group_list dim0 must be in [1, 384] or dynamic, got %ld.", groups);
            return ge::GRAPH_FAILED;
        }
    }

    // 校验 y 和 amax 的输出形状。
    gert::Shape* outputYShape = context->GetOutputShape(OUTPUT_Y_INDEX);
    if (outputYShape == nullptr) {
        OP_LOGE("QuantMaxV2", "InferShape got nullptr y output shape.");
        return ge::GRAPH_FAILED;
    }
    *outputYShape = *inputXShape;

    // amax 必选，形状为 [numGroups]；不分组时为 [1]。
    gert::Shape* outputAmaxShape = context->GetOutputShape(OUTPUT_AMAX_INDEX);
    auto numGroupsLocal = groupListShape == nullptr ? 1 : groupListShape->GetDim(0);
    if (outputAmaxShape == nullptr) {
        OP_LOGE("QuantMaxV2", "InferShape got nullptr amax output shape.");
        return ge::GRAPH_FAILED;
    }
    outputAmaxShape->SetDimNum(1);
    outputAmaxShape->SetDim(0, numGroupsLocal);

    gert::Shape* outputYScaleShape = context->GetOutputShape(OUTPUT_Y_SCALE_INDEX);
    // y_scale 可选；静态模式的形状指针可以为空，动态模式才生成该输出。
    if (outputYScaleShape != nullptr) {
        outputYScaleShape->SetDimNum(1);
        outputYScaleShape->SetDim(0, context->GetOptionalInputShape(INPUT_SCALE_INDEX) == nullptr ?
                                         (groupListShape == nullptr ? 1 : groupListShape->GetDim(0)) :
                                         0);
    }

    return ge::GRAPH_SUCCESS;
}

// 注册形状推导；类型推导在 op_graph 中注册。
IMPL_OP_INFERSHAPE(QuantMaxV2).InferShape(InferShape4QuantMaxV2);

} // namespace ops

/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// QuantMaxV2 图侧类型推导；通过 IMPL_OP 注册，op_host 仅注册形状推导。

#include "log/log.h"
#include "register/op_impl_registry.h"

using namespace ge;

namespace ops {

constexpr size_t INPUT_X_INDEX = 0;
constexpr size_t OUTPUT_Y_INDEX = 0;
constexpr size_t OUTPUT_AMAX_INDEX = 1;
constexpr size_t OUTPUT_Y_SCALE_INDEX = 2;
constexpr size_t ATTR_DST_TYPE_INDEX = 1;
constexpr int64_t MAX_GROUP_NUM = 384;
constexpr int64_t DST_TYPE_HIFLOAT8 = 34;
constexpr int64_t DST_TYPE_FLOAT8_E5M2 = 35;
constexpr int64_t DST_TYPE_FLOAT8_E4M3FN = 36;

static const std::initializer_list<ge::DataType> SUPPORTED_X_DTYPE_LIST = {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16};

static ge::DataType GetOutputYDtype(int64_t dstType)
{
    switch (dstType) {
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

// 校验输入输出类型范围，以及 y 的类型与 dst_type 是否匹配。
static ge::graphStatus InferDataType4QuantMaxV2(gert::InferDataTypeContext* context)
{
    const ge::DataType inputXDtype = context->GetInputDataType(INPUT_X_INDEX);
    if (inputXDtype == ge::DT_UNDEFINED) {
        OP_LOGE("QuantMaxV2", "InferDataType got undefined x dtype.");
        return ge::GRAPH_FAILED;
    }

    bool isValidXDtype = false;
    for (const auto& dtype : SUPPORTED_X_DTYPE_LIST) {
        if (inputXDtype == dtype) {
            isValidXDtype = true;
            break;
        }
    }
    if (!isValidXDtype) {
        OP_LOGE("QuantMaxV2", "x dtype %d is not supported.", static_cast<int32_t>(inputXDtype));
        return ge::GRAPH_FAILED;
    }

    // 可选输入使用 IR 索引访问，避免 scale 缺省后将 group_list 错认成 scale。
    const auto scaleDtype = context->GetOptionalInputDataType(1);
    const auto groupListDtype = context->GetOptionalInputDataType(2);
    if (scaleDtype != ge::DT_UNDEFINED && scaleDtype != ge::DT_FLOAT) {
        OP_LOGE("QuantMaxV2", "scale dtype must be FLOAT, got %d.", static_cast<int32_t>(scaleDtype));
        return ge::GRAPH_FAILED;
    }
    if (groupListDtype != ge::DT_UNDEFINED && groupListDtype != ge::DT_INT64) {
        OP_LOGE("QuantMaxV2", "group_list dtype must be INT64, got %d.", static_cast<int32_t>(groupListDtype));
        return ge::GRAPH_FAILED;
    }

    const auto* attrs = context->GetAttrs();
    if (attrs == nullptr) {
        OP_LOGE("QuantMaxV2", "InferDataType got nullptr attrs.");
        return ge::GRAPH_FAILED;
    }

    const int64_t* dstTypePtr = attrs->GetAttrPointer<int64_t>(ATTR_DST_TYPE_INDEX);
    if (dstTypePtr == nullptr) {
        OP_LOGE("QuantMaxV2", "required attr dst_type is missing.");
        return ge::GRAPH_FAILED;
    }
    int64_t dstType = *dstTypePtr;

    if (dstType != DST_TYPE_HIFLOAT8 && dstType != DST_TYPE_FLOAT8_E5M2 && dstType != DST_TYPE_FLOAT8_E4M3FN) {
        OP_LOGE("QuantMaxV2", "dst_type %ld is invalid, must be %ld/%ld/%ld.", dstType, DST_TYPE_HIFLOAT8,
                DST_TYPE_FLOAT8_E5M2, DST_TYPE_FLOAT8_E4M3FN);
        return ge::GRAPH_FAILED;
    }

    ge::DataType outputYDtype = GetOutputYDtype(dstType);
    if (outputYDtype == ge::DT_UNDEFINED) {
        OP_LOGE("QuantMaxV2", "dst_type %ld has no matching output y dtype.", dstType);
        return ge::GRAPH_FAILED;
    }
    // 输出 dtype 规则：y 由 dst_type 决定；amax 与 x 同 dtype；y_scale 恒为 FLOAT
    // （静态模式下该输出为占位，类型推导保持一致以便图复用）。
    context->SetOutputDataType(OUTPUT_Y_INDEX, outputYDtype);
    context->SetOutputDataType(OUTPUT_AMAX_INDEX, inputXDtype);
    context->SetOutputDataType(OUTPUT_Y_SCALE_INDEX, ge::DT_FLOAT);

    return ge::GRAPH_SUCCESS;
}

IMPL_OP(QuantMaxV2).InferDataType(InferDataType4QuantMaxV2);
} // namespace ops

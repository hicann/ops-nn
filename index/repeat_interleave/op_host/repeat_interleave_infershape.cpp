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
 * \file repeat_interleave_infershape.cpp
 * \brief
 */
#include "log/log.h"
#include "util/shape_util.h"
#include "register/op_impl_registry.h"

using namespace gert;
using namespace ge;
namespace {
constexpr size_t ATTR_AXIS_IDX = 0;
constexpr size_t INPUT_X_ID = 0;
constexpr size_t INPUT_REPEATS_ID = 1;
constexpr size_t OUTPUT_Y_ID = 0;
} // namespace
namespace ops {
inline bool IsConstTensor(const gert::Tensor* input_tensor)
{
    if (input_tensor != nullptr) {
        if (input_tensor->GetAddr() == nullptr) {
            // empty tensor
            return input_tensor->GetShapeSize() == 0;
        }
        return true;
    }
    return false;
}

template <typename T>
static int64_t GetOutputAxisLen(
    const gert::Shape* xShape, const gert::Shape* repeatsShape, const gert::Tensor* repeatsInput,
    const int64_t axisAttr)
{
    int64_t repeatsSum = 0;
    if (repeatsShape->GetShapeSize() == 1) {
        repeatsSum = xShape->GetDim(axisAttr) * repeatsInput->GetData<T>()[0];
    } else {
        for (int64_t i = 0; i < repeatsShape->GetShapeSize(); i++) {
            repeatsSum += repeatsInput->GetData<T>()[i];
        }
    }
    return repeatsSum;
}

static graphStatus InferShape4RepeatInterleave(gert::InferShapeContext* context)
{
    OP_LOGI(context->GetNodeName(), "Enter RepeatInterleaveInferShape");

    const gert::Shape* xShape = context->GetInputShape(INPUT_X_ID);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);

    const gert::Shape* repeatsShape = context->GetInputShape(INPUT_REPEATS_ID);
    OP_CHECK_NULL_WITH_CONTEXT(context, repeatsShape);

    gert::Shape* yShape = context->GetOutputShape(OUTPUT_Y_ID);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);

    const auto repeatsInput = context->GetInputTensor(INPUT_REPEATS_ID);
    OP_CHECK_NULL_WITH_CONTEXT(context, repeatsInput);

    const RuntimeAttrs* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

    const int64_t* axis = attrs->GetAttrPointer<int64_t>(ATTR_AXIS_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, axis);

    if (Ops::Base::IsUnknownRank(*xShape)) {
        OP_LOGD(context->GetNodeName(), "input shape is UnknownRank, set output shape to -2");
        Ops::Base::SetUnknownRank(*yShape);
    }

    int64_t xDimNum = xShape->GetDimNum();
    int64_t axisAttr = *axis;
    if (axisAttr < -xDimNum || axisAttr >= xDimNum) {
        OP_LOGE(context->GetNodeName(), "InferShape4RepeatInterleave FAILED, axisAttr is %ld, not support", axisAttr);
        return ge::GRAPH_FAILED;
    }
    if (axisAttr < 0) {
        axisAttr = *axis + xDimNum;
    }

    *yShape = *xShape;

    if (IsConstTensor(repeatsInput) && xShape->GetDim(axisAttr) != -1) {
        ge::DataType repeatsDtype = repeatsInput->GetDataType();
        OP_LOGD(
            context->GetNodeName(), "InferShape4RepeatInterleave repeatsDtype is %s.",
            Ops::Base::ToString(repeatsDtype).c_str());
        switch (repeatsDtype) {
            case DT_INT32:
                OP_CHECK_IF(
                    repeatsInput->GetData<int32_t>() == nullptr,
                    OP_LOGE(context->GetNodeName(), "repeatsInput->GetData() is nullptr"), return ge::GRAPH_FAILED);
                yShape->SetDim(axisAttr, GetOutputAxisLen<int32_t>(xShape, repeatsShape, repeatsInput, axisAttr));
                break;
            case DT_INT64:
                OP_CHECK_IF(
                    repeatsInput->GetData<int64_t>() == nullptr,
                    OP_LOGE(context->GetNodeName(), "repeatsInput->GetData() is nullptr"), return ge::GRAPH_FAILED);
                yShape->SetDim(axisAttr, GetOutputAxisLen<int64_t>(xShape, repeatsShape, repeatsInput, axisAttr));
                break;
            default:
                OP_LOGE_WITH_INVALID_INPUT_DTYPE(
                    context->GetNodeName(), "repeats", Ops::Base::ToString(repeatsDtype).c_str(), "[int32_t, int64_t]");
                return ge::GRAPH_FAILED;
        }
    } else {
        OP_LOGD(
            context->GetNodeName(),
            "If not satisfy that repeats is ConstTensor and input[axisAttr] != -1, then set output shape to -2");
        Ops::Base::SetUnknownRank(*yShape);
    }

    return ge::GRAPH_SUCCESS;
}

static graphStatus InferDataType4RepeatInterleave(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(OUTPUT_Y_ID, context->GetInputDataType(INPUT_X_ID));
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(RepeatInterleave)
    .InferShape(InferShape4RepeatInterleave)
    .InputsDataDependency({INPUT_REPEATS_ID})
    .InferDataType(InferDataType4RepeatInterleave);
} // namespace ops

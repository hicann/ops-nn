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
 * \file repeat_interleave_grad.cc
 * \brief
 */

#include "register/op_impl_registry.h"
#include "log/log.h"
#include "util/shape_util.h"

using namespace ge;

namespace ops {

static constexpr size_t INPUT_GRAD_IDX = 0;
static constexpr size_t INPUT_REPEATS_IDX = 1;
static constexpr size_t OUTPUT_GRAD_IDX = 0;
static constexpr size_t ATTR_AXIS_IDX = 0;

inline bool IsConstTensor(const gert::Tensor* input_tensor) {
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
static int64_t GetOutputZeroAxisLen(
    const gert::Shape* inputGradShape, const gert::Shape* repeatsShape, const gert::Tensor* repeatsInput,
    int64_t& axisAttr)
{
    int64_t repeatsAxisDim = 0;
    if (repeatsShape->GetShapeSize() == 1) {
        repeatsAxisDim = inputGradShape->GetDim(axisAttr) / repeatsInput->GetData<T>()[0];
    } else {
        repeatsAxisDim = repeatsShape->GetShapeSize();
    }
    return repeatsAxisDim;
}

static graphStatus InferShape4RepeatInterleaveGrad(gert::InferShapeContext* context)
{
    OP_LOGD(context, "Begin to do InferShape4RepeatInterleaveGrad");
    const gert::Shape* inputGradShape = context->GetInputShape(INPUT_GRAD_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputGradShape);
    const gert::Shape* repeatsShape = context->GetInputShape(INPUT_REPEATS_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, repeatsShape);
    const auto repeatsInput = context->GetInputTensor(INPUT_REPEATS_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, repeatsInput);

    gert::Shape* outputGradShape = context->GetOutputShape(OUTPUT_GRAD_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputGradShape);

    if (Ops::Base::IsUnknownRank(*inputGradShape)) {
        OP_LOGD(context, "input shape is UnknownRank, set output shape to -2");
        Ops::Base::SetUnknownRank(*outputGradShape);
        return ge::GRAPH_SUCCESS;
    }
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t* axis = attrs->GetAttrPointer<int64_t>(ATTR_AXIS_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, axis);

    int64_t axisAttr = *axis;
    int64_t inputDimNum = inputGradShape->GetDimNum();
    if (axisAttr < -inputDimNum || axisAttr >= inputDimNum) {
        OP_LOGE(context, "InferShape4RepeatInterleaveGrad FAILED, axisAttr is %ld, not support", axisAttr);
        return ge::GRAPH_FAILED;
    }
    if (axisAttr < 0) {
        axisAttr = axisAttr + inputDimNum;
    }

    *outputGradShape = *inputGradShape;
    if (IsConstTensor(repeatsInput) && inputGradShape->GetDim(axisAttr) != -1) {
        ge::DataType repeatsDtype = repeatsInput->GetDataType();
        OP_LOGD(context, "repeatsDtype is %s.", Ops::Base::ToString(repeatsDtype).c_str());
        switch (repeatsDtype) {
            case DT_INT32:
                OP_CHECK_IF(
                    repeatsInput->GetData<int32_t>() == nullptr, OP_LOGE(context, "repeatsInput->GetData() is nullptr"),
                    return ge::GRAPH_FAILED);
                outputGradShape->SetDim(
                    axisAttr, GetOutputZeroAxisLen<int32_t>(inputGradShape, repeatsShape, repeatsInput, axisAttr));
                break;
            case DT_INT64:
                OP_CHECK_IF(
                    repeatsInput->GetData<int64_t>() == nullptr, OP_LOGE(context, "repeatsInput->GetData() is nullptr"),
                    return ge::GRAPH_FAILED);
                outputGradShape->SetDim(
                    axisAttr, GetOutputZeroAxisLen<int64_t>(inputGradShape, repeatsShape, repeatsInput, axisAttr));
                break;
            default:
                OP_LOGE(context, "InferShape4RepeatInterleaveGrad FAILED, repeats must be int32 or int64");
                return ge::GRAPH_FAILED;
        }
    } else {
        OP_LOGD(
            context, "If not satisfy that repeats is ConstTensor and input[dim] != -1, then set output shape to -2");
        Ops::Base::SetUnknownRank(*outputGradShape);
        return ge::GRAPH_SUCCESS;
    }
    OP_LOGD(context, "End to do InferShape4RepeatInterleaveGrad");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4RepeatInterleaveGrad(gert::InferDataTypeContext* context)
{
    OP_LOGD(context, "Begin to do InferDataType4RepeatInterleaveGrad");
    OP_LOGD(context, "input y_grad dtype: %s", Ops::Base::ToString(context->GetInputDataType(INPUT_GRAD_IDX)).c_str());
    context->SetOutputDataType(OUTPUT_GRAD_IDX, context->GetInputDataType(INPUT_GRAD_IDX));
    OP_LOGD(context, "output x_grad dtype: %s", Ops::Base::ToString(context->GetOutputDataType(OUTPUT_GRAD_IDX)).c_str());
    OP_LOGD(context, "End to do InferDataType4RepeatInterleaveGrad");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(RepeatInterleaveGrad)
    .InferShape(InferShape4RepeatInterleaveGrad)
    .InputsDataDependency({INPUT_REPEATS_IDX})
    .InferDataType(InferDataType4RepeatInterleaveGrad);
} // namespace ops

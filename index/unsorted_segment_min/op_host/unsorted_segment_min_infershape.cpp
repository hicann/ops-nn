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
 * \file unsorted_segment_infershape.cpp
 * \brief
 */

#include "register/op_impl_registry.h"
#include "log/log.h"
#include "util/shape_util.h"
#include "util/const_util.h"

using namespace ge;
namespace ops {
static constexpr size_t INPUT_X_IDX = 0;
static constexpr size_t INPUT_SEGMENTS_ID_IDX = 1;
static constexpr size_t INPUT_NUM_SEGMENTS_IDX = 2;
constexpr int64_t UNKNOWN_DIM_VALUE_ = -1LL;

static ge::graphStatus UnsortedSegmentInferShapeImpl(const int64_t &first_dim, const gert::Shape *x_shape,
                                                     const gert::Shape *segment_ids_shape, gert::Shape *output_shape) 
{
  const size_t x_rank = x_shape->GetDimNum();
  const size_t segment_ids_rank = segment_ids_shape->GetDimNum();
  size_t output_rank = x_rank - segment_ids_rank + 1;
  output_shape->SetDimNum(output_rank);
  output_shape->SetDim(0, first_dim);
  for (size_t i = 1; i < output_rank; i++) {
    output_shape->SetDim(i, x_shape->GetDim(i + segment_ids_rank - 1));
  }

  return ge::GRAPH_SUCCESS;
}

static graphStatus InferShape4UnsortedSegment(gert::InferShapeContext* context) 
{
  OP_LOGD(context->GetNodeName(), "Begin to do unsortedSegmentInferShape");
  const gert::Shape *x_shape = context->GetInputShape(INPUT_X_IDX);
  OP_CHECK_NULL_WITH_CONTEXT(context, x_shape);
  const gert::Shape *segment_ids_shape = context->GetInputShape(INPUT_SEGMENTS_ID_IDX);
  OP_CHECK_NULL_WITH_CONTEXT(context, segment_ids_shape);
  const gert::Shape *num_segments_tensor = context->GetInputShape(INPUT_NUM_SEGMENTS_IDX);
  OP_CHECK_NULL_WITH_CONTEXT(context, num_segments_tensor);
  gert::Shape *output_shape = context->GetOutputShape(0);
  OP_CHECK_NULL_WITH_CONTEXT(context, output_shape);

  OP_CHECK_IF(num_segments_tensor->GetShapeSize() > 1,
           OP_LOGE(context->GetNodeName(), "The size of num_segments must be 1!"),
           return ge::GRAPH_FAILED);

  // dynamic shape is -2
  if (Ops::Base::IsUnknownRank(*x_shape)) {
    OP_LOGD(context->GetNodeName(), "x_shape is UnknownRank, set output_shape to -2");
    *output_shape = *x_shape;
    return ge::GRAPH_SUCCESS;
  }

  if (Ops::Base::IsUnknownRank(*segment_ids_shape)) {
    OP_LOGD(context->GetNodeName(), "segment_ids_shape is UnknownRank, set output_shape to -2");
    *output_shape = *segment_ids_shape;
    return ge::GRAPH_SUCCESS;
  }

  // dynamic shape is -1 and not dynamic shape
  int64_t num_segments = 0;
  if (!Ops::Base::GetConstInt(context, INPUT_NUM_SEGMENTS_IDX, num_segments)) {
    OP_LOGD(context->GetNodeName(), "get num_segments_tensor unsuccessful, set num_segments to -1");
    num_segments = UNKNOWN_DIM_VALUE_;
  }

  return UnsortedSegmentInferShapeImpl(num_segments, x_shape, segment_ids_shape, output_shape);
}

IMPL_OP_INFERSHAPE(UnsortedSegmentMin)
    .InferShape(InferShape4UnsortedSegment)
    .InputsDataDependency({INPUT_NUM_SEGMENTS_IDX});

}
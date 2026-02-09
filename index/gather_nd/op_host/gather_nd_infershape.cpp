/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file gather_nd_infershape.cpp
 * \brief
 */
#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;
namespace ops {
const size_t INPUT_IDX_X = 0;
const size_t INPUT_IDX_INDICES = 1;

static bool CheckGatherNdParamsSize(const gert::InferShapeContext* context, const int64_t last_dim, const int64_t shape_size) {
  if (last_dim > shape_size) {
    OP_LOGE(context->GetNodeName(), "The last dim(%ld) of indices must be <= params.rank(%ld).", last_dim, shape_size);
    return false;
  }
  return true;
}

static ge::graphStatus InferShape4GatherNd(gert::InferShapeContext* context) {
  OP_LOGD(context->GetNodeName(), "infershape is begin");
  auto x_shape = context->GetInputShape(INPUT_IDX_X);
  OP_CHECK_NULL_WITH_CONTEXT(context, x_shape);
  int64_t x_real_dim_cnt = x_shape->GetDimNum();
  auto indies_shape = context->GetInputShape(INPUT_IDX_INDICES);
  OP_CHECK_NULL_WITH_CONTEXT(context, indies_shape);
  int64_t rank_indices = indies_shape->GetDimNum();
  auto out_shape = context->GetOutputShape(INPUT_IDX_X);
  OP_CHECK_NULL_WITH_CONTEXT(context, out_shape);
  out_shape->SetDimNum(0);
  if (rank_indices < 1) {
    OP_LOGE(context->GetNodeName(), "rank_indices is less than 1， rank_indices is %ld", rank_indices);
    return GRAPH_FAILED;
  }
  int64_t indices_last_element = indies_shape->GetDim(rank_indices - 1);
  OP_CHECK_IF(!CheckGatherNdParamsSize(context, indices_last_element, x_real_dim_cnt),
           OP_LOGE(context->GetNodeName(), "check params is failed"),
           return ge::GRAPH_FAILED);
  for (int64_t i = 0; i < rank_indices - 1; ++i) {
    out_shape->AppendDim(indies_shape->GetDim(i));
  }
  for (int64_t i = indices_last_element; i < x_real_dim_cnt; ++i) {
    out_shape->AppendDim(x_shape->GetDim(i));
  }
  OP_LOGD(context->GetNodeName(), "out_shape is %s", Ops::Base::ToString(*out_shape).c_str());
  return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(GatherNd).InferShape(InferShape4GatherNd);
}  // namespace ops

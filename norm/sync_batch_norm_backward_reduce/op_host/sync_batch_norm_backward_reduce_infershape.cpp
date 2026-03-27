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
 * \file sync_batch_norm_backward_reduce_infershape.cpp
 * \brief
 */
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "op_api/runtime2_util.h"

using namespace ge;
namespace ops {
ge::graphStatus InferShape4InIdxAndOutVector(gert::InferShapeContext* context, int64_t input_idx,
                                             const std::vector<int64_t>& output_idxs) {
  auto in_shape = context->GetInputShape(input_idx);
  OP_CHECK_NULL_WITH_CONTEXT(context, in_shape);
  for (int64_t idx : output_idxs) {
    auto out_shape = context->GetOutputShape(idx);
    OP_CHECK_NULL_WITH_CONTEXT(context, out_shape);
    *out_shape = *in_shape;
  }
  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShape4SyncBatchNormBackwardReduce(gert::InferShapeContext* context) {
  static const std::vector<int64_t> out_idxs{0, 1};
  return InferShape4InIdxAndOutVector(context, 0, out_idxs);
}

IMPL_OP_INFERSHAPE(SyncBatchNormBackwardReduce).InferShape(InferShape4SyncBatchNormBackwardReduce);
}  // namespace ops
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
 * \file apply_adagrad_d_infershape.cpp
 * \brief
 */
#include "log/log.h"
#include "register/op_impl_registry.h"

using namespace ge;
using namespace gert;
using namespace std;
inline ge::graphStatus CopyShapeInputToOutputWithIdx(gert::InferShapeContext* context, int64_t input_idx, int64_t output_idx) {
    auto in_shape = context->GetInputShape(input_idx);
    OP_CHECK_NULL_WITH_CONTEXT(context, in_shape);
    auto out_shape = context->GetOutputShape(output_idx);
    OP_CHECK_NULL_WITH_CONTEXT(context, out_shape);
    *out_shape = *in_shape;
    return ge::GRAPH_SUCCESS;
}
namespace ops {
static ge::graphStatus InferShape4ApplyAdagradD(gert::InferShapeContext* context) {
  constexpr size_t input_var_idx = 0;
  constexpr size_t input_accum_idx = 1;
  constexpr size_t output_var_idx = 0;
  constexpr size_t output_accum_idx = 1;
  OP_CHECK_IF(CopyShapeInputToOutputWithIdx(context, input_var_idx, output_var_idx) != GRAPH_SUCCESS,
           OP_LOGE(context->GetNodeName(),
                                               "elewise one %zu error!", input_var_idx),
           return GRAPH_FAILED);

  return CopyShapeInputToOutputWithIdx(context, input_accum_idx, output_accum_idx);
}

IMPL_OP_INFERSHAPE(ApplyAdagradD).InferShape(InferShape4ApplyAdagradD);

}  // namespace ops

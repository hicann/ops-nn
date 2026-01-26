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
 * \file apply_adam_d_inershape.cpp
 * \brief
 */
#include "log/log.h"
#include "register/op_impl_registry.h"
using namespace ge;
using namespace gert;
using namespace std;

namespace ops {
template <typename T>
static std::string ConcatString(const T& arg) {
  std::ostringstream oss;
  oss << arg;
  return oss.str();
}

template <typename T, typename... Ts>
static std::string ConcatString(const T& arg, const Ts& ... arg_left) {
  std::ostringstream oss;
  oss << arg;
  oss << ConcatString(arg_left...);
  return oss.str();
}  

ge::graphStatus CopyShapeInput2OutputWithIdxInApply(gert::InferShapeContext* context, int64_t input_idx, int64_t output_idx) {
  auto in_shape = context->GetInputShape(input_idx);
  OP_CHECK_NULL_WITH_CONTEXT(context, in_shape);
  auto out_shape = context->GetOutputShape(output_idx);
  OP_CHECK_NULL_WITH_CONTEXT(context, out_shape);
  *out_shape = *in_shape;
  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShape4ApplyAdamD(gert::InferShapeContext* context) {
  constexpr size_t input_var_idx = 0;
  constexpr size_t input_m_idx = 1;
  constexpr size_t input_v_idx = 2;
  constexpr size_t output_var_idx = 0;
  constexpr size_t output_m_idx = 1;
  constexpr size_t output_v_idx = 2;
  OP_CHECK_IF(CopyShapeInput2OutputWithIdxInApply(context, input_var_idx, output_var_idx) != GRAPH_SUCCESS,
           OP_LOGE(context->GetNodeName(), "%s", ConcatString("elewise one ", input_var_idx, " error!").c_str()),
           return GRAPH_FAILED);

  OP_CHECK_IF(
      CopyShapeInput2OutputWithIdxInApply(context, input_m_idx, output_m_idx) != GRAPH_SUCCESS,
      OP_LOGE(context->GetNodeName(), "%s", ConcatString("elewise one ", input_m_idx, " error!").c_str()),
      return GRAPH_FAILED);

  return CopyShapeInput2OutputWithIdxInApply(context, input_v_idx, output_v_idx);
}

IMPL_OP_INFERSHAPE(ApplyAdamD).InferShape(InferShape4ApplyAdamD);
}  // namespace ops

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
 * \file gemm_v2_infer.cpp
 * \brief
 */
#include "common/op_host/matmul_common_infershape.h"

using namespace gert;
namespace {
#define CHECK(cond, log_func, return_expr) \
  do {                                   \
      if (cond) {                        \
          log_func;                      \
          return_expr;                   \
      }                                  \
  } while (0)

const size_t kMatmulV2MinShapeSize = 2;

static ge::graphStatus InferShapeForGemmV2(InferShapeContext *context)
{
  auto op_name = context->GetNodeName();
  auto shape_a = context->GetInputShape(0);
  auto shape_b = context->GetInputShape(1);
  auto shape_c = context->GetInputShape(4);
  auto shape_out = context->GetOutputShape(0);
  auto attrs = context->GetAttrs();
  CHECK(shape_a == nullptr || shape_b == nullptr || shape_c == nullptr || shape_out == nullptr || attrs == nullptr,
        CUBE_INNER_ERR_REPORT(op_name, "shape or attrs is null"), return ge::GRAPH_FAILED);

  const bool *trans_a = attrs->GetAttrPointer<bool>(0);
  const bool *trans_b = attrs->GetAttrPointer<bool>(1);
  CHECK(trans_a == nullptr || trans_b == nullptr, CUBE_INNER_ERR_REPORT(op_name, "attribute is null"),
        return ge::GRAPH_FAILED);

  OP_LOGD(context->GetNodeName(), "a_shape: %s, b_shape: %s, c_shape:%s, transpose_a: %d, transpose_b: %d",
          Ops::Base::ToString(*shape_a).c_str(), Ops::Base::ToString(*shape_b).c_str(), Ops::Base::ToString(*shape_c).c_str(),
          *trans_a, *trans_b);

  OP_LOGD(op_name, "check the input shape length.");
  CHECK((shape_a->GetDimNum() != kMatmulV2MinShapeSize || shape_b->GetDimNum() != kMatmulV2MinShapeSize ||
      shape_c->GetDimNum() != kMatmulV2MinShapeSize),
      CUBE_INNER_ERR_REPORT(op_name, "input dim num[%zu] [%zu] [%zu]is not 2!",
                            shape_a->GetDimNum(), shape_b->GetDimNum(), shape_c->GetDimNum()), return ge::GRAPH_FAILED);

  size_t idx_m = static_cast<size_t>(*trans_a);
  size_t idx_k_a = idx_m == 0UL ? 1UL : 0UL;
  size_t idx_k_b = static_cast<size_t>(*trans_b);
  size_t idx_n = idx_k_b == 0UL ? 1UL : 0UL;

  CHECK(shape_a->GetDim(idx_k_a) != shape_b->GetDim(idx_k_b),
      CUBE_INNER_ERR_REPORT(op_name, "The k-axis of a(%ld) and b(%ld) tensors must be the same",
                            shape_a->GetDim(idx_k_a), shape_b->GetDim(idx_k_b)),
      return ge::GRAPH_FAILED);

  CHECK(shape_a->GetDim(idx_m) != shape_c->GetDim(0) || shape_b->GetDim(idx_n) != shape_c->GetDim(1),
      CUBE_INNER_ERR_REPORT(op_name, "The m(%ld), n(%ld) tensors must be the same c(%ld, %ld)",
                            shape_a->GetDim(idx_m), shape_b->GetDim(idx_n), shape_c->GetDim(0), shape_c->GetDim(1)),
      return ge::GRAPH_FAILED);

  shape_out->SetDimNum(kMatmulV2MinShapeSize);
  shape_out->SetDim(0, shape_a->GetDim(idx_m));
  shape_out->SetDim(1, shape_b->GetDim(idx_n));

  OP_LOGI(op_name, "end infershape.");
  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeForGemmV2(gert::InferDataTypeContext *context)
{
  auto op_name = context->GetNodeName();
  const ge::DataType a_data_type = context->GetInputDataType(0);
  const ge::DataType b_data_type = context->GetInputDataType(1);
  const ge::DataType c_data_type = context->GetInputDataType(4);
  CHECK(((a_data_type != ge::DT_FLOAT16 && a_data_type != ge::DT_BF16) ||
        (b_data_type != ge::DT_FLOAT16 && b_data_type != ge::DT_BF16) ||
        c_data_type != ge::DT_FLOAT),
        CUBE_INNER_ERR_REPORT(op_name, "input dtype not support"),
        return ge::GRAPH_FAILED);
  ge::graphStatus ret = context->SetOutputDataType(0, ge::DT_FLOAT);
  return ret;
}
} // namespace

namespace Ops::NN::MatMul {
IMPL_OP_INFERSHAPE(GemmV2)
  .InferShape(InferShapeForGemmV2)
  .InferDataType(InferDataTypeForGemmV2);
}
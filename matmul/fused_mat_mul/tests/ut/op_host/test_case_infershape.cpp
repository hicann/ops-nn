/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "test_case_infershape.h"

#include "infershape_test_util.h"

using namespace std;
using namespace ge;
using namespace op;

FusedMatMul CreateFusedMatmulOp(OP_TUPLE x1, OP_TUPLE x2, OP_TUPLE bias, OP_TUPLE x3,
                                                  const bool &transpose_x1, const bool &transpose_x2,
                                                  const bool &enable_hf32,
                                                  const std::string &fused_op_type) {
  auto tensor_x1 = create_desc_shape_range(get<0>(x1), get<1>(x1), FORMAT_ND, get<0>(x1), FORMAT_ND, get<2>(x1));
  auto tensor_x2 = create_desc_shape_range(get<0>(x2), get<1>(x2), FORMAT_ND, get<0>(x2), FORMAT_ND, get<2>(x2));
  TensorDesc tensor_bias;
  auto shape_bias = get<0>(bias);
  if (!shape_bias.empty()) {
    tensor_bias = create_desc_shape_range(get<0>(bias), get<1>(bias), FORMAT_ND, get<0>(bias), FORMAT_ND, get<2>(bias));
  }
  auto shape_x3 = get<0>(x3);
  TensorDesc tensor_x3;
  if (!shape_x3.empty()) {
    tensor_x3 = create_desc_shape_range(get<0>(x3), get<1>(x3), FORMAT_ND, get<0>(x3), FORMAT_ND, get<2>(x3));
  }

  FusedMatMul op;
  op.UpdateInputDesc("x1", tensor_x1);
  op.UpdateInputDesc("x2", tensor_x2);
  if (!shape_bias.empty()) {
    op.UpdateInputDesc("bias", tensor_bias);
  }
  if (!shape_x3.empty()) {
    op.UpdateInputDesc("x3", tensor_x3);
  }

  op.SetAttr("transpose_x1", transpose_x1);
  op.SetAttr("transpose_x2", transpose_x2);
  op.SetAttr("enable_hf32", enable_hf32);
  op.SetAttr("fused_op_type", fused_op_type);
  return op;
}
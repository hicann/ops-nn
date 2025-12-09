/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/

#include <gtest/gtest.h> // NOLINT
#include <iostream>
#include "infershape_test_util.h"
#include "ut_op_common.h"
#include "log/log.h"
#include "../../../op_graph/masked_softmax_with_rel_pos_bias_proto.h"

class MaskedSoftmaxWithRelPosBias : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "MaskedSoftmaxWithRelPosBias Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "MaskedSoftmaxWithRelPosBias Proto Test TearDown" << std::endl;
  }
};

TEST_F(MaskedSoftmaxWithRelPosBias, MaskedSoftmaxWithRelPosBias_infershape_case_0) {
  ge::op::MaskedSoftmaxWithRelPosBias op;
  op.UpdateInputDesc("x", create_desc({1, 2, 2, 2048, 64}, ge::DT_FLOAT));
  op.UpdateInputDesc("relative_pos_bias", create_desc({2, 2048, 64}, ge::DT_FLOAT));
  op.UpdateInputDesc("atten_mask", create_desc({2, 2048, 64}, ge::DT_FLOAT));
  op.SetAttr("scale_value", float(1.0));
  op.SetAttr("inner_precision_mode", int(0));
  std::vector<int64_t> y_shape = {1, 2, 2, 2048, 64};
  Runtime2TestParam param{
      {"scale_value", "inner_precision_mode"}}; // attr
  EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);
  auto y_desc = op.GetOutputDescByName("y");
  EXPECT_EQ(y_desc.GetShape().GetDims(), y_shape);
}
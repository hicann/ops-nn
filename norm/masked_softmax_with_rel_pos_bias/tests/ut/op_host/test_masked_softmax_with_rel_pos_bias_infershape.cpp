/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "experiment_ops.h"
#include "common/utils/ut_op_common.h"
#include "array_ops.h"
#include "util/util.h"
#include "matrix_calculation_ops.h"


class masked_softmax_with_rel_pos_bias : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "MaskedSoftmaxWithRelPosBias Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "MaskedSoftmaxWithRelPosBias Proto Test TearDown" << std::endl;
  }
};

TEST_F(masked_softmax_with_rel_pos_bias, masked_softmax_with_rel_pos_bias_infershape_test_2_BSH_FLOAT) {
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
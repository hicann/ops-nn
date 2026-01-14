/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <iostream>
#include "infershape_test_util.h"
#include "ut_op_common.h"
#include "../../../op_graph/softplus_proto.h"

class SoftplusInfershapeTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "SoftplusInfershapeTest SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "SoftplusInfershapeTest TearDown" << std::endl;
  }
};

TEST_F(SoftplusInfershapeTest, softplus_infershape_diff_test){
  ge::op::Softplus op;
  op.UpdateInputDesc("x", create_desc({4, 3, 4}, ge::DT_FLOAT16));
  
  EXPECT_EQ(InferShapeTest(op), ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("y");
  std::vector<int64_t> expected_output_shape = {4, 3, 4};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(SoftplusInfershapeTest, softplus_infershape_same_test){
  ge::op::Softplus op;
  op.UpdateInputDesc("x", create_desc({1, 3, 4}, ge::DT_FLOAT));

  EXPECT_EQ(InferShapeTest(op), ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("y");
  std::vector<int64_t> expected_output_shape = {1, 3, 4};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(SoftplusInfershapeTest,softplus_infershape_diff_test_1){
    ge::op::Softplus op;
    std::vector<std::pair<int64_t,int64_t>> shape_range = {{2,100}};
    auto tensor_desc = create_desc_shape_range({-1},
                                                ge::DT_FLOAT16,ge::FORMAT_ND,
                                                {64},
                                                ge::FORMAT_ND,shape_range);
    op.UpdateInputDesc("x",tensor_desc);

    EXPECT_EQ(InferShapeTest(op),ge::GRAPH_SUCCESS);
    auto output_y1_desc = op.GetOutputDesc("y");
    std::vector<int64_t> expected_output_shape = {-1};
    EXPECT_EQ(output_y1_desc.GetShape().GetDims(), expected_output_shape);
}

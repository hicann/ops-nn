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
 * \file test_group_norm_v2_infershape.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include <iostream>
#include "infershape_test_util.h"
#include "ut_op_common.h"
#include "log/log.h"
#include "../../../op_graph/group_norm_v2_proto.h"

class GroupNormV2 : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "GroupNormV2 Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "GroupNormV2 Proto Test TearDown" << std::endl;
  }
};

TEST_F(GroupNormV2, group_norm_v2_infershape_test_1){
  ge::op::GroupNormV2 op;
  op.UpdateInputDesc("x", create_desc({8, 16, 15, 15}, ge::DT_FLOAT16));
  op.SetAttr("num_groups", 8);
  std::vector<int64_t> expected_output_shape = {8, 16, 15, 15};
  std::vector<int64_t> expected_mean_shape = {64};
  std::vector<int64_t> expected_rstd_shape = {64};

  // run rt 2.0
  Runtime2TestParam param{{"num_groups"}};
  EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);
  auto output0_desc = op.GetOutputDesc(0);
  EXPECT_EQ(output0_desc.GetShape().GetDims(), expected_output_shape);
  auto output1_desc = op.GetOutputDesc(1);
  EXPECT_EQ(output1_desc.GetShape().GetDims(), expected_mean_shape);
  auto output2_desc = op.GetOutputDesc(2);
  EXPECT_EQ(output2_desc.GetShape().GetDims(), expected_rstd_shape);
}

TEST_F(GroupNormV2, group_norm_v2_infershape_test_2){
  ge::op::GroupNormV2 op;
  op.UpdateInputDesc("x", create_desc({-1, -1, -1, -1}, ge::DT_FLOAT16));
  op.SetAttr("num_groups", 8);
  std::vector<int64_t> expected_output_shape = {-1, -1, -1, -1};
  std::vector<int64_t> expected_mean_shape = {-1};
  std::vector<int64_t> expected_rstd_shape = {-1};

  // run rt 2.0
  Runtime2TestParam param{{"num_groups"}};
  EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);
  auto output0_desc = op.GetOutputDesc(0);
  EXPECT_EQ(output0_desc.GetShape().GetDims(), expected_output_shape);
  auto output1_desc = op.GetOutputDesc(1);
  EXPECT_EQ(output1_desc.GetShape().GetDims(), expected_mean_shape);
  auto output2_desc = op.GetOutputDesc(2);
  EXPECT_EQ(output2_desc.GetShape().GetDims(), expected_rstd_shape);
}

TEST_F(GroupNormV2, group_norm_v2_inferdtype_david_test1) {
  ge::op::GroupNormV2 op;
  op.UpdateInputDesc("x", create_desc({1, 1152, 64, 64}, ge::DT_FLOAT16));
  op.UpdateInputDesc("gamma", create_desc({1152}, ge::DT_FLOAT));
  op.UpdateInputDesc("beta", create_desc({1152}, ge::DT_FLOAT));

  auto ret = InferDataTypeTest(op);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_dtype = op.GetOutputDesc("y").GetDataType();
  auto mean_dtype = op.GetOutputDesc("mean").GetDataType();
  auto rstd_dtype = op.GetOutputDesc("rstd").GetDataType();

  EXPECT_EQ(output_dtype, ge::DT_FLOAT16);
  EXPECT_EQ(mean_dtype, ge::DT_FLOAT16);
  EXPECT_EQ(rstd_dtype, ge::DT_FLOAT16);
}
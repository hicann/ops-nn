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
 * \file test_lp_loss_infershape.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include <iostream>
#include "log/log.h"
#include "ut_op_common.h"
#include "infershape_test_util.h"
#include "platform/platform_info.h"

#include "../../../op_graph/lp_loss_proto.h"

class LpLossTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "lp_loss SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "lp_loss TearDown" << std::endl;
  }
};

TEST_F(LpLossTest, lp_loss_test_case01_rt2_sum) {
  ge::op::LpLoss lp_loss_op;
  ge::TensorDesc tensorDesc;
  ge::Shape shape({10, 200});
  tensorDesc.SetDataType(ge::DT_FLOAT16);
  tensorDesc.SetShape(shape);
  tensorDesc.SetOriginShape(shape);

  lp_loss_op.UpdateInputDesc("predict", tensorDesc);
  lp_loss_op.UpdateInputDesc("label", tensorDesc);
  lp_loss_op.SetAttr("p", 1);
  lp_loss_op.SetAttr("reduction", "sum");
  std::vector<int64_t> expected_output_shape = {};

  auto ret_Verify = lp_loss_op.VerifyAllAttr(true);
  EXPECT_EQ(ret_Verify, ge::GRAPH_SUCCESS);

  Runtime2TestParam param{{"p", "reduction"}};
  EXPECT_EQ(InferShapeTest(lp_loss_op, param), ge::GRAPH_SUCCESS);
  auto output0_desc = lp_loss_op.GetOutputDesc(0);
  EXPECT_EQ(output0_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(LpLossTest, lp_loss_test_case02_rt2_none) {
  ge::op::LpLoss lp_loss_op;
  ge::TensorDesc tensorDesc;
  ge::Shape shape({10, 200});
  tensorDesc.SetDataType(ge::DT_FLOAT16);
  tensorDesc.SetShape(shape);
  tensorDesc.SetOriginShape(shape);

  lp_loss_op.UpdateInputDesc("predict", tensorDesc);
  lp_loss_op.UpdateInputDesc("label", tensorDesc);
  lp_loss_op.SetAttr("p", 1);
  lp_loss_op.SetAttr("reduction", "none");
  std::vector<int64_t> expected_output_shape = {10, 200};

  auto ret_Verify = lp_loss_op.VerifyAllAttr(true);
  EXPECT_EQ(ret_Verify, ge::GRAPH_SUCCESS);

  Runtime2TestParam param{{"p", "reduction"}};
  EXPECT_EQ(InferShapeTest(lp_loss_op, param), ge::GRAPH_SUCCESS);
  auto output0_desc = lp_loss_op.GetOutputDesc(0);
  EXPECT_EQ(output0_desc.GetShape().GetDims(), expected_output_shape);
}
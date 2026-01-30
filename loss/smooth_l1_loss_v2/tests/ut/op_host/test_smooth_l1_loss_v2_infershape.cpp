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
 * \file test_smooth_l1_loss_v2_infershape.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include <iostream>
#include "log/log.h"
#include "ut_op_common.h"
#include "infershape_test_util.h"
#include "platform/platform_info.h"

#include "../../../op_graph/smooth_l1_loss_v2_proto.h"

class SmoothL1LossV2InfershapeTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "smooth_l1_loss_v2 SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "smooth_l1_loss_v2 TearDown" << std::endl;
  }
};

TEST_F(SmoothL1LossV2InfershapeTest, smooth_l1_loss_v2_infershape_test_1) {
  ge::op::SmoothL1LossV2 smooth_l1_loss_v2_op;
  ge::TensorDesc tensorDesc;
  ge::Shape shape({10, 200});
  tensorDesc.SetDataType(ge::DT_FLOAT16);
  tensorDesc.SetShape(shape);
  tensorDesc.SetOriginShape(shape);

  smooth_l1_loss_v2_op.UpdateInputDesc("predict", tensorDesc);
  smooth_l1_loss_v2_op.UpdateInputDesc("label", tensorDesc);
  smooth_l1_loss_v2_op.SetAttr("p", 1);
  smooth_l1_loss_v2_op.SetAttr("reduction", "sum");
  std::vector<int64_t> expected_output_shape = {};

  auto ret_Verify = smooth_l1_loss_v2_op.VerifyAllAttr(true);
  EXPECT_EQ(ret_Verify, ge::GRAPH_SUCCESS);

  Runtime2TestParam param{{"p", "reduction"}};
  EXPECT_EQ(InferShapeTest(smooth_l1_loss_v2_op, param), ge::GRAPH_SUCCESS);
  auto output0_desc = smooth_l1_loss_v2_op.GetOutputDesc(0);
  EXPECT_EQ(output0_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(SmoothL1LossV2InfershapeTest, smooth_l1_loss_v2_infershape_test_2) {
  ge::op::SmoothL1LossV2 smooth_l1_loss_v2_op;
  ge::TensorDesc tensorDesc;
  ge::Shape shape({10, 200});
  tensorDesc.SetDataType(ge::DT_FLOAT16);
  tensorDesc.SetShape(shape);
  tensorDesc.SetOriginShape(shape);

  smooth_l1_loss_v2_op.UpdateInputDesc("predict", tensorDesc);
  smooth_l1_loss_v2_op.UpdateInputDesc("label", tensorDesc);
  smooth_l1_loss_v2_op.SetAttr("p", 1);
  smooth_l1_loss_v2_op.SetAttr("reduction", "none");
  std::vector<int64_t> expected_output_shape = {10, 200};

  auto ret_Verify = smooth_l1_loss_v2_op.VerifyAllAttr(true);
  EXPECT_EQ(ret_Verify, ge::GRAPH_SUCCESS);

  Runtime2TestParam param{{"p", "reduction"}};
  EXPECT_EQ(InferShapeTest(smooth_l1_loss_v2_op, param), ge::GRAPH_SUCCESS);
  auto output0_desc = smooth_l1_loss_v2_op.GetOutputDesc(0);
  EXPECT_EQ(output0_desc.GetShape().GetDims(), expected_output_shape);
}
/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>  // NOLINT
#include <iostream>
#include "infershape_test_util.h"  // NOLINT
#include "ut_op_common.h"
#include "log/log.h"
#include "../../../op_graph/dequant_bias_proto.h"

class DequantBias : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "DequantBias SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "DequantBias TearDown" << std::endl;
  }
};

TEST_F(DequantBias, DequantBias_infershape_case_0) {
  ge::op::DequantBias op;
  op.UpdateInputDesc("x", create_desc({4,256}, ge::DT_FLOAT16));

  EXPECT_EQ(InferShapeTest(op), ge::GRAPH_SUCCESS);
  EXPECT_EQ(InferDataTypeTest(op), ge::GRAPH_SUCCESS);
}
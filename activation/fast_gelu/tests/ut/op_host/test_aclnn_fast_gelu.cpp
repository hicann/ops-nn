/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include "../../../op_host/op_api/aclnn_fast_gelu.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"

class l2_fast_gelu_test : public testing::Test {
protected:
  static void SetUpTestCase() {
    std::cout << "l2_fast_gelu_test SetUp" << std::endl;
  }

  static void TearDownTestCase() { std::cout << "l2_fast_gelu_test TearDown" << std::endl; }
};

// 正常路径，float32
TEST_F(l2_fast_gelu_test, l2_fast_gelu_test_005) {
  auto selfDesc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND);
  auto outDesc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnFastGelu, INPUT(selfDesc), OUTPUT(outDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);

  //ut.TestPrecision();
}

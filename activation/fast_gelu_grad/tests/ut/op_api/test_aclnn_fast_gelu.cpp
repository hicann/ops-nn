/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <vector>
#include <array>
#include "gtest/gtest.h"

#include "../../../op_api/aclnn_fast_gelu_backward.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"

using namespace std;


class l2_fast_gelu_backward_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    cout << "fast_gelu_backward_test SetUp" << endl;
  }

  static void TearDownTestCase() {
    cout << "fast_gelu_backward_test TearDown" << endl;
  }
};

TEST_F(l2_fast_gelu_backward_test, test_fast_gelu_backward_float32) {
  auto self = TensorDesc({1,16}, ACL_FLOAT, ACL_FORMAT_ND);
  auto input_tensor = TensorDesc({1,16}, ACL_FLOAT, ACL_FORMAT_ND);

  auto output_tensor = TensorDesc({1,16}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnFastGeluBackward, INPUT(self, input_tensor),
                      OUTPUT(output_tensor));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}
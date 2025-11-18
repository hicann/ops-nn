/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <array>
#include <vector>
#include <math.h>
#include "gtest/gtest.h"

#include "../../../op_host/op_api/aclnn_median.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace std;

class l2_nanmedian_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    cout << "nanmedian_test SetUp" << endl;
  }

  static void TearDownTestCase() {
    cout << "nanmedian_test TearDown" << endl;
  }
};

TEST_F(l2_nanmedian_test, case_001_self_out_shape_not_euqal) {
  auto selfTensor = TensorDesc({3, 3}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
  auto outTensor = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);

  auto ut = OP_API_UT(aclnnNanMedian, INPUT(selfTensor), OUTPUT(outTensor));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

/**
* Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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

#include <vector>
#include <array>
#include "gtest/gtest.h"
#include "../../../op_host/op_api/aclnn_index_fill.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"

using namespace std;

class l2_inplace_index_fill_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    cout << "inpace_index_fill_test SetUp" << endl;
  }

  static void TearDownTestCase() { cout << "inpace_index_fill_test TearDown" << endl; }
};

// 正常场景 self:int32 fillVal:int32
TEST_F(l2_inplace_index_fill_test, Ascend910B2_inplace_index_fill_case_00) {
  int64_t dim = 1;
  auto self = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND)
                         .ValueRange(-10, 10)
                         .Value(vector<int32_t>{3, 4, 9, 6, 7, 11});
  auto index = TensorDesc({3}, ACL_INT32, ACL_FORMAT_ND)
                         .ValueRange(0, 3)
                         .Value(vector<int32_t>{0, 1, 2});
  auto fillVal = ScalarDesc(static_cast<int32_t>(1));

  auto ut = OP_API_UT(aclnnInplaceIndexFill, INPUT(self, dim, index, fillVal), OUTPUT());

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
}
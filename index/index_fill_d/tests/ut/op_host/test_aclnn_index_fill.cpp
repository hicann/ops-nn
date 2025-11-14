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

class l2_index_fill_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    cout << "index_fill_test SetUp" << endl;
  }

  static void TearDownTestCase() { cout << "index_fill_test TearDown" << endl; }
};

TEST_F(l2_index_fill_test, Ascend910B2_index_fill_case_00) {
  int64_t dim = 1;
  auto self = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND)
                         .ValueRange(-10, 10)
                         .Value(vector<int32_t>{3, 4, 9, 6, 7, 11});
  auto index = TensorDesc({3}, ACL_INT32, ACL_FORMAT_ND)
                         .ValueRange(0, 3)
                         .Value(vector<int32_t>{0, 1, 2});
  auto fillVal = ScalarDesc(static_cast<int32_t>(1));
  auto out = TensorDesc(self);
  auto ut = OP_API_UT(aclnnIndexFill, INPUT(self, dim, index, fillVal), OUTPUT(out));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
}

TEST_F(l2_index_fill_test, Ascend910B2_index_fill_case_01) {
  int64_t dim = 1;
  auto self = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND)
                         .ValueRange(-10, 10)
                         .Value(vector<float>{3, 4, 9, 6, 7, 11});
  auto index = TensorDesc({3}, ACL_INT32, ACL_FORMAT_ND)
                         .ValueRange(0, 3)
                         .Value(vector<int32_t>{0, 1, 2});
  auto fillVal = ScalarDesc(static_cast<int32_t>(1));
  auto out = TensorDesc(self);
  auto ut = OP_API_UT(aclnnIndexFill, INPUT(self, dim, index, fillVal), OUTPUT(out));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
}

TEST_F(l2_index_fill_test, Ascend910B2_index_fill_case_02) {
  int64_t dim = 100;
  auto self = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND)
                         .ValueRange(-10, 10)
                         .Value(vector<int32_t>{3, 4, 9, 6, 7, 11});
  auto index = TensorDesc({3}, ACL_INT32, ACL_FORMAT_ND)
                         .ValueRange(0, 3)
                         .Value(vector<int32_t>{0, 1, 2});
  auto fillVal = ScalarDesc(static_cast<int32_t>(1));
  auto out = TensorDesc(self);
  auto ut = OP_API_UT(aclnnIndexFill, INPUT(self, dim, index, fillVal), OUTPUT(out));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
}

TEST_F(l2_index_fill_test, Ascend910B2_index_fill_case_03) {
  int64_t dim = 1;
  auto self = TensorDesc({2, 3}, ACL_INT64, ACL_FORMAT_ND)
                         .ValueRange(-10, 10)
                         .Value(vector<int32_t>{3, 4, 9, 6, 7, 11});
  auto index = TensorDesc({3}, ACL_INT32, ACL_FORMAT_ND)
                         .ValueRange(0, 3)
                         .Value(vector<int32_t>{0, 1, 2});
  auto fillVal = ScalarDesc(static_cast<int64_t>(1));
  auto out = TensorDesc(self);
  auto ut = OP_API_UT(aclnnIndexFill, INPUT(self, dim, index, fillVal), OUTPUT(out));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
}

TEST_F(l2_index_fill_test, Ascend910B2_index_fill_case_04) {
  int64_t dim = 1;
  auto self = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND)
                         .ValueRange(-10, 10)
                         .Value(vector<int32_t>{3, 4, 9, 6, 7, 11});
  auto index = TensorDesc({}, ACL_INT32, ACL_FORMAT_ND);
  auto fillVal = ScalarDesc(static_cast<int32_t>(1));
  auto out = TensorDesc(self);
  auto ut = OP_API_UT(aclnnIndexFill, INPUT(self, dim, index, fillVal), OUTPUT(out));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
}

TEST_F(l2_index_fill_test, Ascend910B2_index_fill_case_05) {
  int64_t dim = 1;
  auto self = TensorDesc({}, ACL_INT32, ACL_FORMAT_ND);
  auto index = TensorDesc({3}, ACL_INT32, ACL_FORMAT_ND)
                         .ValueRange(0, 3)
                         .Value(vector<int32_t>{0, 1, 2});
  auto fillVal = ScalarDesc(static_cast<int32_t>(1));
  auto out = TensorDesc(self);
  auto ut = OP_API_UT(aclnnIndexFill, INPUT(self, dim, index, fillVal), OUTPUT(out));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
}

TEST_F(l2_index_fill_test, Ascend910B2_index_fill_case_06) {
  int64_t dim = -100;
  auto self = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND)
                         .ValueRange(-10, 10)
                         .Value(vector<int32_t>{3, 4, 9, 6, 7, 11});
  auto index = TensorDesc({3}, ACL_INT32, ACL_FORMAT_ND)
                         .ValueRange(0, 3)
                         .Value(vector<int32_t>{0, 1, 2});
  auto fillVal = ScalarDesc(static_cast<int32_t>(1));
  auto out = TensorDesc(self);
  auto ut = OP_API_UT(aclnnIndexFill, INPUT(self, dim, index, fillVal), OUTPUT(out));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
}

TEST_F(l2_index_fill_test, index_fill_case_07) {
  int64_t dim = 1;
  auto self = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND)
                         .ValueRange(-10, 10)
                         .Value(vector<int32_t>{3, 4, 9, 6, 7, 11});
  auto index = TensorDesc({3}, ACL_INT32, ACL_FORMAT_ND)
                         .ValueRange(0, 3)
                         .Value(vector<int32_t>{0, 1, 2});
  auto fillVal = ScalarDesc(static_cast<int32_t>(1));
  auto out = TensorDesc(self);
  auto ut = OP_API_UT(aclnnIndexFill, INPUT(self, dim, index, fillVal), OUTPUT(out));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
}
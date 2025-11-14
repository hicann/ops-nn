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
#include <vector>
#include <array>
#include "gtest/gtest.h"

#include "level2/aclnn_median.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/inner/types.h"


using namespace std;

class l2_nanmedian_dim_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "l2_nanmedian_dim_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "l2_nanmedian_dim_test TearDown" << endl;
    }
};

TEST_F(l2_nanmedian_dim_test, aclnnNanMedianDim_test_zero_dim_tensor) {
  auto self_tensor_desc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);
  const int64_t dim = 0;
  const bool keepdim = false;
  auto indices_tensor_desc = TensorDesc({}, ACL_INT64, ACL_FORMAT_ND);
  auto out_tensor_desc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnNanMedianDim, INPUT(self_tensor_desc, dim, keepdim), OUTPUT(out_tensor_desc, indices_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_nanmedian_dim_test, aclnnNanMedianDim_3_4_float_nd_0_nd_checkdim) {
  const int64_t dim = 5;
  const bool keepdim = false;

  auto self_tensor_desc = TensorDesc({3,4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto indices_tensor_desc = TensorDesc({4}, ACL_INT64, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
  auto out_tensor_desc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnNanMedianDim, INPUT(self_tensor_desc, dim, keepdim), OUTPUT(out_tensor_desc, indices_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

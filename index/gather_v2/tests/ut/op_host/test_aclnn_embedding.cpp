/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <vector>
#include <array>
#include "gtest/gtest.h"

#include "../../../op_host/op_api/aclnn_embedding.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"


using namespace std;
using namespace op;

class l2_embedding_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    cout << "embedding_test SetUp" << endl;
  }

  static void TearDownTestCase() { cout << "embedding_test TearDown" << endl; }
};

// normal
TEST_F(l2_embedding_test, case_1) {
  auto tensorWeightDesc = TensorDesc({10, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto tensorIndicesDesc = TensorDesc({4}, ACL_INT32, ACL_FORMAT_ND)
      .ValueRange(0, 5)
      .Value(vector<int>{1,2,3,4});
  auto tensorOutDesc = TensorDesc({4, 3}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnEmbedding,INPUT(tensorWeightDesc, tensorIndicesDesc), OUTPUT(tensorOutDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

}

// indice nullptr
TEST_F(l2_embedding_test, case_2) {
  auto tensorWeightDesc = TensorDesc({10, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto tensorIndicesDesc = nullptr;
  auto tensorOutDesc = TensorDesc({4, 3}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnEmbedding,INPUT(tensorWeightDesc, tensorIndicesDesc), OUTPUT(tensorOutDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// weight nullptr
TEST_F(l2_embedding_test, case_3) {
  auto tensorWeightDesc = nullptr;
  auto tensorIndicesDesc = TensorDesc({4}, ACL_INT32, ACL_FORMAT_ND)
      .ValueRange(0, 5)
      .Value(vector<int>{1,2,3,4});
  auto tensorOutDesc = TensorDesc({4, 3}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnEmbedding,INPUT(tensorWeightDesc, tensorIndicesDesc), OUTPUT(tensorOutDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// out nullptr
TEST_F(l2_embedding_test, case_4) {
  auto tensorWeightDesc = TensorDesc({10, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto tensorIndicesDesc = TensorDesc({4}, ACL_INT32, ACL_FORMAT_ND)
      .ValueRange(0, 5)
      .Value(vector<int>{1,2,3,4});
  auto tensorOutDesc = nullptr;

  auto ut = OP_API_UT(aclnnEmbedding,INPUT(tensorWeightDesc, tensorIndicesDesc), OUTPUT(tensorOutDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// indices support int64, weight support float16
TEST_F(l2_embedding_test, case_5) {
  auto tensorWeightDesc = TensorDesc({10, 3}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto tensorIndicesDesc = TensorDesc({4}, ACL_INT64, ACL_FORMAT_ND)
      .ValueRange(0, 5)
      .Value(vector<int>{1,2,3,4});
  auto tensorOutDesc = TensorDesc({4, 3}, ACL_FLOAT16, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnEmbedding,INPUT(tensorWeightDesc, tensorIndicesDesc), OUTPUT(tensorOutDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

}

// indices support int32, weight support float64
TEST_F(l2_embedding_test, case_6) {
  auto tensorWeightDesc = TensorDesc({10, 3}, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto tensorIndicesDesc = TensorDesc({4}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 5).Value(vector<int>{1,2,3,4});
  auto tensorOutDesc = TensorDesc({4, 3}, ACL_DOUBLE, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnEmbedding,INPUT(tensorWeightDesc, tensorIndicesDesc), OUTPUT(tensorOutDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// weight not support bfloat16
TEST_F(l2_embedding_test, case_7) {
  auto tensorWeightDesc = TensorDesc({10, 3}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto tensorIndicesDesc = TensorDesc({4}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 5).Value(vector<int>{1,2,3,4});
  auto tensorOutDesc = TensorDesc({4, 3}, ACL_BF16, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnEmbedding,INPUT(tensorWeightDesc, tensorIndicesDesc), OUTPUT(tensorOutDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, 0);
}

// indices support int32, weight support int8
TEST_F(l2_embedding_test, case_8) {
  auto tensorWeightDesc = TensorDesc({10, 3}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto tensorIndicesDesc = TensorDesc({4}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 5).Value(vector<int>{1,2,3,4});
  auto tensorOutDesc = TensorDesc({4, 3}, ACL_INT8, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnEmbedding,INPUT(tensorWeightDesc, tensorIndicesDesc), OUTPUT(tensorOutDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

}

// indices support int32, weight support uint8
TEST_F(l2_embedding_test, case_9) {
  auto tensorWeightDesc = TensorDesc({10, 3}, ACL_UINT8, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto tensorIndicesDesc = TensorDesc({4}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 5).Value(vector<int>{1,2,3,4});
  auto tensorOutDesc = TensorDesc({4, 3}, ACL_UINT8, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnEmbedding,INPUT(tensorWeightDesc, tensorIndicesDesc), OUTPUT(tensorOutDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

}

// indices support int32, weight support int32
TEST_F(l2_embedding_test, case_10) {
  auto tensorWeightDesc = TensorDesc({10, 3}, ACL_UINT8, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto tensorIndicesDesc = TensorDesc({4}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 5).Value(vector<int>{1,2,3,4});
  auto tensorOutDesc = TensorDesc({4, 3}, ACL_UINT8, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnEmbedding,INPUT(tensorWeightDesc, tensorIndicesDesc), OUTPUT(tensorOutDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

}

// indices support int32, weight support int64
TEST_F(l2_embedding_test, case_11) {
  auto tensorWeightDesc = TensorDesc({10, 3}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto tensorIndicesDesc = TensorDesc({4}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 5).Value(vector<int>{1,2,3,4});
  auto tensorOutDesc = TensorDesc({4, 3}, ACL_INT64, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnEmbedding,INPUT(tensorWeightDesc, tensorIndicesDesc), OUTPUT(tensorOutDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

}

// indices support int32, weight support bool
TEST_F(l2_embedding_test, case_12) {
  auto tensorWeightDesc = TensorDesc({10, 3}, ACL_BOOL, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto tensorIndicesDesc = TensorDesc({4}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 5).Value(vector<int>{1,2,3,4});
  auto tensorOutDesc = TensorDesc({4, 3}, ACL_BOOL, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnEmbedding,INPUT(tensorWeightDesc, tensorIndicesDesc), OUTPUT(tensorOutDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

}

// indices support int32, weight support complex64
TEST_F(l2_embedding_test, case_13) {
  auto tensorWeightDesc = TensorDesc({10, 3}, ACL_COMPLEX64, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto tensorIndicesDesc = TensorDesc({4}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 5).Value(vector<int>{1,2,3,4});
  auto tensorOutDesc = TensorDesc({4, 3}, ACL_COMPLEX64, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnEmbedding,INPUT(tensorWeightDesc, tensorIndicesDesc), OUTPUT(tensorOutDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// indices support int32, weight support complex128
TEST_F(l2_embedding_test, case_14) {
  auto tensorWeightDesc = TensorDesc({10, 3}, ACL_COMPLEX128, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto tensorIndicesDesc = TensorDesc({4}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 5).Value(vector<int>{1,2,3,4});
  auto tensorOutDesc = TensorDesc({4, 3}, ACL_COMPLEX128, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnEmbedding,INPUT(tensorWeightDesc, tensorIndicesDesc), OUTPUT(tensorOutDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// indices not support float16
TEST_F(l2_embedding_test, case_15) {
  auto tensorWeightDesc = TensorDesc({10, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto tensorIndicesDesc = TensorDesc({4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 5).Value(vector<int>{1,2,3,4});
  auto tensorOutDesc = TensorDesc({4, 3}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnEmbedding,INPUT(tensorWeightDesc, tensorIndicesDesc), OUTPUT(tensorOutDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// indices not support float
TEST_F(l2_embedding_test, case_16) {
  auto tensorWeightDesc = TensorDesc({10, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto tensorIndicesDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 5).Value(vector<int>{1,2,3,4});
  auto tensorOutDesc = TensorDesc({4, 3}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnEmbedding,INPUT(tensorWeightDesc, tensorIndicesDesc), OUTPUT(tensorOutDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// indices not support float64
TEST_F(l2_embedding_test, case_17) {
  auto tensorWeightDesc = TensorDesc({10, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto tensorIndicesDesc = TensorDesc({4}, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(0, 5).Value(vector<int>{1,2,3,4});
  auto tensorOutDesc = TensorDesc({4, 3}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnEmbedding,INPUT(tensorWeightDesc, tensorIndicesDesc), OUTPUT(tensorOutDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// indices not support uint8
TEST_F(l2_embedding_test, case_19) {
  auto tensorWeightDesc = TensorDesc({10, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto tensorIndicesDesc = TensorDesc({4}, ACL_UINT8, ACL_FORMAT_ND).ValueRange(0, 5).Value(vector<int>{1,2,3,4});
  auto tensorOutDesc = TensorDesc({4, 3}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnEmbedding,INPUT(tensorWeightDesc, tensorIndicesDesc), OUTPUT(tensorOutDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// indices not support int8
TEST_F(l2_embedding_test, case_20) {
  auto tensorWeightDesc = TensorDesc({10, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto tensorIndicesDesc = TensorDesc({4}, ACL_INT8, ACL_FORMAT_ND).ValueRange(0, 5).Value(vector<int>{1,2,3,4});
  auto tensorOutDesc = TensorDesc({4, 3}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnEmbedding,INPUT(tensorWeightDesc, tensorIndicesDesc), OUTPUT(tensorOutDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// indices not support int16
TEST_F(l2_embedding_test, case_21) {
  auto tensorWeightDesc = TensorDesc({10, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto tensorIndicesDesc = TensorDesc({4}, ACL_INT16, ACL_FORMAT_ND).ValueRange(0, 5).Value(vector<int>{1,2,3,4});
  auto tensorOutDesc = TensorDesc({4, 3}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnEmbedding,INPUT(tensorWeightDesc, tensorIndicesDesc), OUTPUT(tensorOutDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// indices not support bool
TEST_F(l2_embedding_test, case_22) {
  auto tensorWeightDesc = TensorDesc({10, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto tensorIndicesDesc = TensorDesc({4}, ACL_BOOL, ACL_FORMAT_ND).ValueRange(0, 5).Value(vector<int>{1,2,3,4});
  auto tensorOutDesc = TensorDesc({4, 3}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnEmbedding,INPUT(tensorWeightDesc, tensorIndicesDesc), OUTPUT(tensorOutDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// indices not support complex64
TEST_F(l2_embedding_test, case_23) {
  auto tensorWeightDesc = TensorDesc({10, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto tensorIndicesDesc = TensorDesc({4}, ACL_COMPLEX64, ACL_FORMAT_ND).ValueRange(0, 5).Value(vector<int>{1,2,3,4});
  auto tensorOutDesc = TensorDesc({4, 3}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnEmbedding,INPUT(tensorWeightDesc, tensorIndicesDesc), OUTPUT(tensorOutDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// indices not support complex128
TEST_F(l2_embedding_test, case_24) {
  auto tensorWeightDesc = TensorDesc({10, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto tensorIndicesDesc = TensorDesc({4}, ACL_COMPLEX128, ACL_FORMAT_ND).ValueRange(0, 5).Value(vector<int>{1,2,3,4});
  auto tensorOutDesc = TensorDesc({4, 3}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnEmbedding,INPUT(tensorWeightDesc, tensorIndicesDesc), OUTPUT(tensorOutDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// support empty tensor
TEST_F(l2_embedding_test, case_25) {
  auto tensorWeightDesc = TensorDesc({10, 0}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto tensorIndicesDesc = TensorDesc({4}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 5).Value(vector<int>{1,2,3,4});
  auto tensorOutDesc = TensorDesc({4, 0}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnEmbedding,INPUT(tensorWeightDesc, tensorIndicesDesc), OUTPUT(tensorOutDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// max dimension less than 9
TEST_F(l2_embedding_test, case_26) {
  auto tensorWeightDesc = TensorDesc({10, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto tensorIndicesDesc = TensorDesc({4,4,4,4,4,4,4,4,4}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 5).Value(vector<int>{1,2,3,4});
  auto tensorOutDesc = TensorDesc({4, 3}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnEmbedding,INPUT(tensorWeightDesc, tensorIndicesDesc), OUTPUT(tensorOutDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// support not contiguous
TEST_F(l2_embedding_test, case_27) {
  auto tensorWeightDesc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 2}, 0, {4, 2}).Value(vector<float>{1,2,3,4,5,6,7,8});
  auto tensorIndicesDesc = TensorDesc({3}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 5).Value(vector<int>{1,1,1});
  auto tensorOutDesc = TensorDesc({3, 4}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnEmbedding,INPUT(tensorWeightDesc, tensorIndicesDesc), OUTPUT(tensorOutDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

}

// support nchw
TEST_F(l2_embedding_test, case_28) {
  auto tensorWeightDesc = TensorDesc({10, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 10);
  auto tensorIndicesDesc = TensorDesc({2, 2, 2, 2}, ACL_INT32, ACL_FORMAT_NCHW).ValueRange(0, 9);
  auto tensorOutDesc = TensorDesc({2, 2, 2, 2, 3}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnEmbedding,INPUT(tensorWeightDesc, tensorIndicesDesc), OUTPUT(tensorOutDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

}

// support nhwc
TEST_F(l2_embedding_test, case_29) {
  auto tensorWeightDesc = TensorDesc({10, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 10);
  auto tensorIndicesDesc = TensorDesc({2, 2, 2, 2}, ACL_INT32, ACL_FORMAT_NHWC).ValueRange(0, 9);
  auto tensorOutDesc = TensorDesc({2, 2, 2, 2, 3}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnEmbedding,INPUT(tensorWeightDesc, tensorIndicesDesc), OUTPUT(tensorOutDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

}

// support hwcn
TEST_F(l2_embedding_test, case_30) {
  auto tensorWeightDesc = TensorDesc({10, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 10);
  auto tensorIndicesDesc = TensorDesc({2, 2, 2, 2}, ACL_INT32, ACL_FORMAT_HWCN).ValueRange(0, 9);
  auto tensorOutDesc = TensorDesc({2, 2, 2, 2, 3}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnEmbedding,INPUT(tensorWeightDesc, tensorIndicesDesc), OUTPUT(tensorOutDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

}

// support ndhwc
TEST_F(l2_embedding_test, case_31) {
  auto tensorWeightDesc = TensorDesc({10, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 10);
  auto tensorIndicesDesc = TensorDesc({2, 2, 2, 2, 2}, ACL_INT32, ACL_FORMAT_NDHWC).ValueRange(0, 9);
  auto tensorOutDesc = TensorDesc({2, 2, 2, 2, 2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnEmbedding,INPUT(tensorWeightDesc, tensorIndicesDesc), OUTPUT(tensorOutDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

}

// support ncdhw
TEST_F(l2_embedding_test, case_32) {
  auto tensorWeightDesc = TensorDesc({10, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 10);
  auto tensorIndicesDesc = TensorDesc({2, 2, 2, 2, 2}, ACL_INT32, ACL_FORMAT_NCDHW).ValueRange(0, 9);
  auto tensorOutDesc = TensorDesc({2, 2, 2, 2, 2, 3}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnEmbedding,INPUT(tensorWeightDesc, tensorIndicesDesc), OUTPUT(tensorOutDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

}

// support impl_mode high_performance
TEST_F(l2_embedding_test, ascend910B2_high_performance_case_33) {
  auto tensorWeightDesc = TensorDesc({2268, 32}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto tensorIndicesDesc = TensorDesc({8192, 80}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 200);
  auto tensorOutDesc = TensorDesc({8192, 80, 32}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnEmbedding,INPUT(tensorWeightDesc, tensorIndicesDesc), OUTPUT(tensorOutDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

}

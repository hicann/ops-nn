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

#include "../../../op_api/aclnn_embedding_dense_backward.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"


using namespace std;
using namespace op;

class l2_embedding_dense_backward_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    cout << "embedding_dense_backward_test SetUp" << endl;
  }

  static void TearDownTestCase() { cout << "embedding_dense_backward_test TearDown" << endl; }
};

// 正常场景
TEST_F(l2_embedding_dense_backward_test, case_1) {
  int64_t numWeights = 10;
  int64_t paddingIdx = 0;
  bool scaleGradByFreq = false;
  auto tensorGradDesc = TensorDesc({4, 3}, ACL_FLOAT, ACL_FORMAT_ND)
                            .ValueRange(-1, 1)
                            .Value(vector<float>{0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2});
  auto tensorIndicesDesc = TensorDesc({4}, ACL_INT32, ACL_FORMAT_ND)
                           .ValueRange(0, 5)
                           .Value(vector<int>{1,2,3,4});
  auto tensorOutDesc = TensorDesc({10, 3}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnEmbeddingDenseBackward,
                      INPUT(tensorGradDesc, tensorIndicesDesc, numWeights, paddingIdx, scaleGradByFreq),
                      OUTPUT(tensorOutDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  ut.TestPrecision();
}

// grad为空指针
TEST_F(l2_embedding_dense_backward_test, case_2) {
  int64_t numWeights = 10;
  int64_t paddingIdx = 0;
  bool scaleGradByFreq = false;
  auto tensorGradDesc = nullptr;
  auto tensorIndicesDesc = TensorDesc({4}, ACL_INT32, ACL_FORMAT_ND)
                               .ValueRange(0, 5);
  auto tensorOutDesc = TensorDesc({10, 3}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnEmbeddingDenseBackward,
                      INPUT(tensorGradDesc, tensorIndicesDesc, numWeights, paddingIdx, scaleGradByFreq),
                      OUTPUT(tensorOutDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// indices为空指针
TEST_F(l2_embedding_dense_backward_test, case_3) {
  int64_t numWeights = 10;
  int64_t paddingIdx = 0;
  bool scaleGradByFreq = false;
  auto tensorGradDesc = TensorDesc({4, 3}, ACL_FLOAT, ACL_FORMAT_ND)
                            .ValueRange(-1, 1);
  auto tensorIndicesDesc = nullptr;
  auto tensorOutDesc = TensorDesc({10, 3}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnEmbeddingDenseBackward,
                      INPUT(tensorGradDesc, tensorIndicesDesc, numWeights, paddingIdx, scaleGradByFreq),
                      OUTPUT(tensorOutDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// out为空指针
TEST_F(l2_embedding_dense_backward_test, case_4) {
  int64_t numWeights = 10;
  int64_t paddingIdx = 0;
  bool scaleGradByFreq = false;
  auto tensorGradDesc = TensorDesc({4, 3}, ACL_FLOAT, ACL_FORMAT_ND)
                            .ValueRange(-1, 1);
  auto tensorIndicesDesc = TensorDesc({4}, ACL_INT32, ACL_FORMAT_ND)
                               .ValueRange(0, 5);
  auto tensorOutDesc = nullptr;

  auto ut = OP_API_UT(aclnnEmbeddingDenseBackward,
                      INPUT(tensorGradDesc, tensorIndicesDesc, numWeights, paddingIdx, scaleGradByFreq),
                      OUTPUT(tensorOutDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// grad数据类型不支持(FLOAT16)
TEST_F(l2_embedding_dense_backward_test, case_5) {
  int64_t numWeights = 10;
  int64_t paddingIdx = 0;
  bool scaleGradByFreq = false;
  auto tensorGradDesc = TensorDesc({4, 3}, ACL_FLOAT16, ACL_FORMAT_ND)
                            .ValueRange(-1, 1);
  auto tensorIndicesDesc = TensorDesc({4}, ACL_INT32, ACL_FORMAT_ND)
                               .ValueRange(0, 5);
  auto tensorOutDesc = TensorDesc({10, 3}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnEmbeddingDenseBackward,
                      INPUT(tensorGradDesc, tensorIndicesDesc, numWeights, paddingIdx, scaleGradByFreq),
                      OUTPUT(tensorOutDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// grad数据类型不支持(BFLOAT16)
TEST_F(l2_embedding_dense_backward_test, case_6) {
  int64_t numWeights = 10;
  int64_t paddingIdx = 0;
  bool scaleGradByFreq = false;
  auto tensorGradDesc = TensorDesc({4, 3}, ACL_BF16, ACL_FORMAT_ND)
                            .ValueRange(-1, 1);
  auto tensorIndicesDesc = TensorDesc({4}, ACL_INT32, ACL_FORMAT_ND)
                               .ValueRange(0, 5);
  auto tensorOutDesc = TensorDesc({10, 3}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnEmbeddingDenseBackward,
                      INPUT(tensorGradDesc, tensorIndicesDesc, numWeights, paddingIdx, scaleGradByFreq),
                      OUTPUT(tensorOutDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B) {
    EXPECT_EQ(aclRet, ACL_SUCCESS);
  } else {
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
  }
}

// grad数据类型不支持(DOUBLE)
TEST_F(l2_embedding_dense_backward_test, case_7) {
  int64_t numWeights = 10;
  int64_t paddingIdx = 0;
  bool scaleGradByFreq = false;
  auto tensorGradDesc = TensorDesc({4, 3}, ACL_DOUBLE, ACL_FORMAT_ND)
                            .ValueRange(-1, 1);
  auto tensorIndicesDesc = TensorDesc({4}, ACL_INT32, ACL_FORMAT_ND)
                               .ValueRange(0, 5);
  auto tensorOutDesc = TensorDesc({10, 3}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnEmbeddingDenseBackward,
                      INPUT(tensorGradDesc, tensorIndicesDesc, numWeights, paddingIdx, scaleGradByFreq),
                      OUTPUT(tensorOutDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// grad数据类型不支持(INT32)
TEST_F(l2_embedding_dense_backward_test, case_8) {
  int64_t numWeights = 10;
  int64_t paddingIdx = 0;
  bool scaleGradByFreq = false;
  auto tensorGradDesc = TensorDesc({4, 3}, ACL_INT32, ACL_FORMAT_ND)
                            .ValueRange(-1, 1);
  auto tensorIndicesDesc = TensorDesc({4}, ACL_INT32, ACL_FORMAT_ND)
                               .ValueRange(0, 5);
  auto tensorOutDesc = TensorDesc({10, 3}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnEmbeddingDenseBackward,
                      INPUT(tensorGradDesc, tensorIndicesDesc, numWeights, paddingIdx, scaleGradByFreq),
                      OUTPUT(tensorOutDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// grad数据类型不支持(INT64)
TEST_F(l2_embedding_dense_backward_test, case_9) {
  int64_t numWeights = 10;
  int64_t paddingIdx = 0;
  bool scaleGradByFreq = false;
  auto tensorGradDesc = TensorDesc({4, 3}, ACL_INT64, ACL_FORMAT_ND)
                            .ValueRange(-1, 1);
  auto tensorIndicesDesc = TensorDesc({4}, ACL_INT32, ACL_FORMAT_ND)
                               .ValueRange(0, 5);
  auto tensorOutDesc = TensorDesc({10, 3}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnEmbeddingDenseBackward,
                      INPUT(tensorGradDesc, tensorIndicesDesc, numWeights, paddingIdx, scaleGradByFreq),
                      OUTPUT(tensorOutDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// grad数据类型不支持(INT16)
TEST_F(l2_embedding_dense_backward_test, case_10) {
  int64_t numWeights = 10;
  int64_t paddingIdx = 0;
  bool scaleGradByFreq = false;
  auto tensorGradDesc = TensorDesc({4, 3}, ACL_INT16, ACL_FORMAT_ND)
                            .ValueRange(-1, 1);
  auto tensorIndicesDesc = TensorDesc({4}, ACL_INT32, ACL_FORMAT_ND)
                               .ValueRange(0, 5);
  auto tensorOutDesc = TensorDesc({10, 3}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnEmbeddingDenseBackward,
                      INPUT(tensorGradDesc, tensorIndicesDesc, numWeights, paddingIdx, scaleGradByFreq),
                      OUTPUT(tensorOutDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// grad数据类型不支持(INT8)
TEST_F(l2_embedding_dense_backward_test, case_11) {
  int64_t numWeights = 10;
  int64_t paddingIdx = 0;
  bool scaleGradByFreq = false;
  auto tensorGradDesc = TensorDesc({4, 3}, ACL_INT8, ACL_FORMAT_ND)
                            .ValueRange(-1, 1);
  auto tensorIndicesDesc = TensorDesc({4}, ACL_INT32, ACL_FORMAT_ND)
                               .ValueRange(0, 5);
  auto tensorOutDesc = TensorDesc({10, 3}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnEmbeddingDenseBackward,
                      INPUT(tensorGradDesc, tensorIndicesDesc, numWeights, paddingIdx, scaleGradByFreq),
                      OUTPUT(tensorOutDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// grad数据类型不支持(UINT8)
TEST_F(l2_embedding_dense_backward_test, case_12) {
  int64_t numWeights = 10;
  int64_t paddingIdx = 0;
  bool scaleGradByFreq = false;
  auto tensorGradDesc = TensorDesc({4, 3}, ACL_UINT8, ACL_FORMAT_ND)
                            .ValueRange(-1, 1);
  auto tensorIndicesDesc = TensorDesc({4}, ACL_INT32, ACL_FORMAT_ND)
                               .ValueRange(0, 5);
  auto tensorOutDesc = TensorDesc({10, 3}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnEmbeddingDenseBackward,
                      INPUT(tensorGradDesc, tensorIndicesDesc, numWeights, paddingIdx, scaleGradByFreq),
                      OUTPUT(tensorOutDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// grad数据类型不支持(BOOL)
TEST_F(l2_embedding_dense_backward_test, case_13) {
  int64_t numWeights = 10;
  int64_t paddingIdx = 0;
  bool scaleGradByFreq = false;
  auto tensorGradDesc = TensorDesc({4, 3}, ACL_BOOL, ACL_FORMAT_ND)
                            .ValueRange(-1, 1);
  auto tensorIndicesDesc = TensorDesc({4}, ACL_INT32, ACL_FORMAT_ND)
                               .ValueRange(0, 5);
  auto tensorOutDesc = TensorDesc({10, 3}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnEmbeddingDenseBackward,
                      INPUT(tensorGradDesc, tensorIndicesDesc, numWeights, paddingIdx, scaleGradByFreq),
                      OUTPUT(tensorOutDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// grad数据类型不支持(COMPLEX64)
TEST_F(l2_embedding_dense_backward_test, case_14) {
  int64_t numWeights = 10;
  int64_t paddingIdx = 0;
  bool scaleGradByFreq = false;
  auto tensorGradDesc = TensorDesc({4, 3}, ACL_COMPLEX64, ACL_FORMAT_ND)
                            .ValueRange(-1, 1);
  auto tensorIndicesDesc = TensorDesc({4}, ACL_INT32, ACL_FORMAT_ND)
                               .ValueRange(0, 5);
  auto tensorOutDesc = TensorDesc({10, 3}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnEmbeddingDenseBackward,
                      INPUT(tensorGradDesc, tensorIndicesDesc, numWeights, paddingIdx, scaleGradByFreq),
                      OUTPUT(tensorOutDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// grad数据类型不支持(COMPLEX128)
TEST_F(l2_embedding_dense_backward_test, case_15) {
  int64_t numWeights = 10;
  int64_t paddingIdx = 0;
  bool scaleGradByFreq = false;
  auto tensorGradDesc = TensorDesc({4, 3}, ACL_COMPLEX128, ACL_FORMAT_ND)
                            .ValueRange(-1, 1);
  auto tensorIndicesDesc = TensorDesc({4}, ACL_INT32, ACL_FORMAT_ND)
                               .ValueRange(0, 5);
  auto tensorOutDesc = TensorDesc({10, 3}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnEmbeddingDenseBackward,
                      INPUT(tensorGradDesc, tensorIndicesDesc, numWeights, paddingIdx, scaleGradByFreq),
                      OUTPUT(tensorOutDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// indices数据类型支持(FLOAT)
TEST_F(l2_embedding_dense_backward_test, case_16) {
  int64_t numWeights = 10;
  int64_t paddingIdx = 0;
  bool scaleGradByFreq = false;
  auto tensorGradDesc = TensorDesc({4, 3}, ACL_FLOAT, ACL_FORMAT_ND)
                            .ValueRange(-1, 1);
  auto tensorIndicesDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND)
                               .ValueRange(0, 5);
  auto tensorOutDesc = TensorDesc({10, 3}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnEmbeddingDenseBackward,
                      INPUT(tensorGradDesc, tensorIndicesDesc, numWeights, paddingIdx, scaleGradByFreq),
                      OUTPUT(tensorOutDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// indices数据类型不支持(FLOAT16)
TEST_F(l2_embedding_dense_backward_test, case_17) {
  int64_t numWeights = 10;
  int64_t paddingIdx = 0;
  bool scaleGradByFreq = false;
  auto tensorGradDesc = TensorDesc({4, 3}, ACL_FLOAT, ACL_FORMAT_ND)
                            .ValueRange(-1, 1);
  auto tensorIndicesDesc = TensorDesc({4}, ACL_FLOAT16, ACL_FORMAT_ND)
                               .ValueRange(0, 5);
  auto tensorOutDesc = TensorDesc({10, 3}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnEmbeddingDenseBackward,
                      INPUT(tensorGradDesc, tensorIndicesDesc, numWeights, paddingIdx, scaleGradByFreq),
                      OUTPUT(tensorOutDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// indices数据类型不支持(BFLOAT16)
TEST_F(l2_embedding_dense_backward_test, case_18) {
  int64_t numWeights = 10;
  int64_t paddingIdx = 0;
  bool scaleGradByFreq = false;
  auto tensorGradDesc = TensorDesc({4, 3}, ACL_FLOAT, ACL_FORMAT_ND)
                            .ValueRange(-1, 1);
  auto tensorIndicesDesc = TensorDesc({4}, ACL_BF16, ACL_FORMAT_ND)
                               .ValueRange(0, 5);
  auto tensorOutDesc = TensorDesc({10, 3}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnEmbeddingDenseBackward,
                      INPUT(tensorGradDesc, tensorIndicesDesc, numWeights, paddingIdx, scaleGradByFreq),
                      OUTPUT(tensorOutDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// indices数据类型支持(DOUBLE)
TEST_F(l2_embedding_dense_backward_test, case_19) {
  int64_t numWeights = 10;
  int64_t paddingIdx = 0;
  bool scaleGradByFreq = false;
  auto tensorGradDesc = TensorDesc({4, 3}, ACL_FLOAT, ACL_FORMAT_ND)
                            .ValueRange(-1, 1);
  auto tensorIndicesDesc = TensorDesc({4}, ACL_DOUBLE, ACL_FORMAT_ND)
                               .ValueRange(0, 5);
  auto tensorOutDesc = TensorDesc({10, 3}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnEmbeddingDenseBackward,
                      INPUT(tensorGradDesc, tensorIndicesDesc, numWeights, paddingIdx, scaleGradByFreq),
                      OUTPUT(tensorOutDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// indices数据类型支持(INT64)
TEST_F(l2_embedding_dense_backward_test, case_20) {
  int64_t numWeights = 10;
  int64_t paddingIdx = 0;
  bool scaleGradByFreq = false;
  auto tensorGradDesc = TensorDesc({4, 3}, ACL_FLOAT, ACL_FORMAT_ND)
                            .ValueRange(-1, 1);
  auto tensorIndicesDesc = TensorDesc({4}, ACL_INT64, ACL_FORMAT_ND)
                               .ValueRange(0, 5);
  auto tensorOutDesc = TensorDesc({10, 3}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnEmbeddingDenseBackward,
                      INPUT(tensorGradDesc, tensorIndicesDesc, numWeights, paddingIdx, scaleGradByFreq),
                      OUTPUT(tensorOutDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// indices数据类型支持(INT16)
TEST_F(l2_embedding_dense_backward_test, case_21) {
  int64_t numWeights = 10;
  int64_t paddingIdx = 0;
  bool scaleGradByFreq = false;
  auto tensorGradDesc = TensorDesc({4, 3}, ACL_FLOAT, ACL_FORMAT_ND)
                            .ValueRange(-1, 1);
  auto tensorIndicesDesc = TensorDesc({4}, ACL_INT16, ACL_FORMAT_ND)
                               .ValueRange(0, 5);
  auto tensorOutDesc = TensorDesc({10, 3}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnEmbeddingDenseBackward,
                      INPUT(tensorGradDesc, tensorIndicesDesc, numWeights, paddingIdx, scaleGradByFreq),
                      OUTPUT(tensorOutDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// indices数据类型支持(INT8)
TEST_F(l2_embedding_dense_backward_test, case_22) {
  int64_t numWeights = 10;
  int64_t paddingIdx = 0;
  bool scaleGradByFreq = false;
  auto tensorGradDesc = TensorDesc({4, 3}, ACL_FLOAT, ACL_FORMAT_ND)
                            .ValueRange(-1, 1);
  auto tensorIndicesDesc = TensorDesc({4}, ACL_INT8, ACL_FORMAT_ND)
                               .ValueRange(0, 5);
  auto tensorOutDesc = TensorDesc({10, 3}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnEmbeddingDenseBackward,
                      INPUT(tensorGradDesc, tensorIndicesDesc, numWeights, paddingIdx, scaleGradByFreq),
                      OUTPUT(tensorOutDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// indices数据类型支持(UINT8)
TEST_F(l2_embedding_dense_backward_test, case_23) {
  int64_t numWeights = 10;
  int64_t paddingIdx = 0;
  bool scaleGradByFreq = false;
  auto tensorGradDesc = TensorDesc({4, 3}, ACL_FLOAT, ACL_FORMAT_ND)
                            .ValueRange(-1, 1);
  auto tensorIndicesDesc = TensorDesc({4}, ACL_UINT8, ACL_FORMAT_ND)
                               .ValueRange(0, 5);
  auto tensorOutDesc = TensorDesc({10, 3}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnEmbeddingDenseBackward,
                      INPUT(tensorGradDesc, tensorIndicesDesc, numWeights, paddingIdx, scaleGradByFreq),
                      OUTPUT(tensorOutDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// indices数据类型支持(BOOL)
TEST_F(l2_embedding_dense_backward_test, case_24) {
  int64_t numWeights = 10;
  int64_t paddingIdx = 0;
  bool scaleGradByFreq = false;
  auto tensorGradDesc = TensorDesc({4, 3}, ACL_FLOAT, ACL_FORMAT_ND)
                            .ValueRange(-1, 1);
  auto tensorIndicesDesc = TensorDesc({4}, ACL_BOOL, ACL_FORMAT_ND)
                               .ValueRange(0, 5);
  auto tensorOutDesc = TensorDesc({10, 3}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnEmbeddingDenseBackward,
                      INPUT(tensorGradDesc, tensorIndicesDesc, numWeights, paddingIdx, scaleGradByFreq),
                      OUTPUT(tensorOutDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// indices数据类型不支持(COMPLEX64)
TEST_F(l2_embedding_dense_backward_test, case_25) {
  int64_t numWeights = 10;
  int64_t paddingIdx = 0;
  bool scaleGradByFreq = false;
  auto tensorGradDesc = TensorDesc({4, 3}, ACL_FLOAT, ACL_FORMAT_ND)
                            .ValueRange(-1, 1);
  auto tensorIndicesDesc = TensorDesc({4}, ACL_COMPLEX64, ACL_FORMAT_ND)
                               .ValueRange(0, 5);
  auto tensorOutDesc = TensorDesc({10, 3}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnEmbeddingDenseBackward,
                      INPUT(tensorGradDesc, tensorIndicesDesc, numWeights, paddingIdx, scaleGradByFreq),
                      OUTPUT(tensorOutDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// indices数据类型不支持(COMPLEX128)
TEST_F(l2_embedding_dense_backward_test, case_26) {
  int64_t numWeights = 10;
  int64_t paddingIdx = 0;
  bool scaleGradByFreq = false;
  auto tensorGradDesc = TensorDesc({4, 3}, ACL_FLOAT, ACL_FORMAT_ND)
                            .ValueRange(-1, 1);
  auto tensorIndicesDesc = TensorDesc({4}, ACL_COMPLEX128, ACL_FORMAT_ND)
                               .ValueRange(0, 5);
  auto tensorOutDesc = TensorDesc({10, 3}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnEmbeddingDenseBackward,
                      INPUT(tensorGradDesc, tensorIndicesDesc, numWeights, paddingIdx, scaleGradByFreq),
                      OUTPUT(tensorOutDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// out数据类型不支持
TEST_F(l2_embedding_dense_backward_test, case_27) {
  int64_t numWeights = 10;
  int64_t paddingIdx = 0;
  bool scaleGradByFreq = false;
  auto tensorGradDesc = TensorDesc({4, 3}, ACL_FLOAT, ACL_FORMAT_ND)
                            .ValueRange(-1, 1);
  auto tensorIndicesDesc = TensorDesc({4}, ACL_INT32, ACL_FORMAT_ND)
                               .ValueRange(0, 5);
  auto tensorOutDesc = TensorDesc({10, 3}, ACL_DOUBLE, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnEmbeddingDenseBackward,
                      INPUT(tensorGradDesc, tensorIndicesDesc, numWeights, paddingIdx, scaleGradByFreq),
                      OUTPUT(tensorOutDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// grad和indices维度不匹配
TEST_F(l2_embedding_dense_backward_test, case_31) {
  int64_t numWeights = 10;
  int64_t paddingIdx = 0;
  bool scaleGradByFreq = false;
  auto tensorGradDesc = TensorDesc({4, 3}, ACL_FLOAT, ACL_FORMAT_ND)
                            .ValueRange(-1, 1);
  auto tensorIndicesDesc = TensorDesc({4, 3}, ACL_INT32, ACL_FORMAT_ND)
                               .ValueRange(0, 5);
  auto tensorOutDesc = TensorDesc({10, 3}, ACL_FLOAT, ACL_FORMAT_UNDEFINED);

  auto ut = OP_API_UT(aclnnEmbeddingDenseBackward,
                      INPUT(tensorGradDesc, tensorIndicesDesc, numWeights, paddingIdx, scaleGradByFreq),
                      OUTPUT(tensorOutDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// grad和indices shape不满足对应关系
TEST_F(l2_embedding_dense_backward_test, case_32) {
  int64_t numWeights = 10;
  int64_t paddingIdx = 0;
  bool scaleGradByFreq = false;
  auto tensorGradDesc = TensorDesc({4, 3}, ACL_FLOAT, ACL_FORMAT_ND)
                            .ValueRange(-1, 1);
  auto tensorIndicesDesc = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND)
                               .ValueRange(0, 5);
  auto tensorOutDesc = TensorDesc({10, 3}, ACL_FLOAT, ACL_FORMAT_UNDEFINED);

  auto ut = OP_API_UT(aclnnEmbeddingDenseBackward,
                      INPUT(tensorGradDesc, tensorIndicesDesc, numWeights, paddingIdx, scaleGradByFreq),
                      OUTPUT(tensorOutDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// nchw ncdhw
TEST_F(l2_embedding_dense_backward_test, case_33) {
  int64_t numWeights = 10;
  int64_t paddingIdx = 0;
  bool scaleGradByFreq = false;
  auto tensorGradDesc = TensorDesc({4, 3, 3, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW)
                            .ValueRange(0, 2);
  auto tensorIndicesDesc = TensorDesc({4, 3, 3, 3}, ACL_INT32, ACL_FORMAT_NCHW)
                               .ValueRange(0, 9);
  auto tensorOutDesc = TensorDesc({10, 3}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnEmbeddingDenseBackward,
                      INPUT(tensorGradDesc, tensorIndicesDesc, numWeights, paddingIdx, scaleGradByFreq),
                      OUTPUT(tensorOutDesc));

  uint64_t workspaceSize1 = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize1);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate

}

// nhwc ndhwc
TEST_F(l2_embedding_dense_backward_test, case_34) {
  int64_t numWeights = 10;
  int64_t paddingIdx = 0;
  bool scaleGradByFreq = false;
  auto tensorGradDesc = TensorDesc({4, 3, 3, 3, 3}, ACL_FLOAT, ACL_FORMAT_NDHWC)
                            .ValueRange(0, 2);
  auto tensorIndicesDesc = TensorDesc({4,3,3,3}, ACL_INT32, ACL_FORMAT_NHWC)
                               .ValueRange(0, 9);
  auto tensorOutDesc = TensorDesc({10, 3}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnEmbeddingDenseBackward,
                      INPUT(tensorGradDesc, tensorIndicesDesc, numWeights, paddingIdx, scaleGradByFreq),
                      OUTPUT(tensorOutDesc));

  uint64_t workspaceSize1 = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize1);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate

}

// hwcn ndhwc
TEST_F(l2_embedding_dense_backward_test, case_35) {
  int64_t numWeights = 10;
  int64_t paddingIdx = 0;
  bool scaleGradByFreq = false;
  auto tensorGradDesc = TensorDesc({4, 3, 3, 3, 3}, ACL_FLOAT, ACL_FORMAT_NDHWC)
                            .ValueRange(0, 2);
  auto tensorIndicesDesc = TensorDesc({4,3,3,3}, ACL_INT32, ACL_FORMAT_HWCN)
                               .ValueRange(0, 9);
  auto tensorOutDesc = TensorDesc({10, 3}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnEmbeddingDenseBackward,
                      INPUT(tensorGradDesc, tensorIndicesDesc, numWeights, paddingIdx, scaleGradByFreq),
                      OUTPUT(tensorOutDesc));

  uint64_t workspaceSize1 = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize1);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate

}

// 空tensor
TEST_F(l2_embedding_dense_backward_test, case_36) {
  int64_t numWeights = 10;
  int64_t paddingIdx = 0;
  bool scaleGradByFreq = false;
  auto tensorGradDesc = TensorDesc({4, 3, 3, 3, 0}, ACL_FLOAT, ACL_FORMAT_NDHWC)
                            .ValueRange(0, 2);
  auto tensorIndicesDesc = TensorDesc({4,3,3,3}, ACL_INT32, ACL_FORMAT_HWCN)
                               .ValueRange(0, 9);
  auto tensorOutDesc = TensorDesc({10, 3}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnEmbeddingDenseBackward,
                      INPUT(tensorGradDesc, tensorIndicesDesc, numWeights, paddingIdx, scaleGradByFreq),
                      OUTPUT(tensorOutDesc));

  uint64_t workspaceSize1 = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize1);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_embedding_dense_backward_test, case_37) {
  int64_t numWeights = 10;
  int64_t paddingIdx = 0;
  bool scaleGradByFreq = false;
  auto tensorGradDesc = TensorDesc({3, 3, 3, 3, 3, 3, 3, 3, 3}, ACL_FLOAT, ACL_FORMAT_ND);
  auto tensorIndicesDesc = TensorDesc({3, 3, 3, 3, 3, 3, 3, 3}, ACL_INT32, ACL_FORMAT_ND);
  auto tensorOutDesc = TensorDesc({10, 3}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnEmbeddingDenseBackward,
                      INPUT(tensorGradDesc, tensorIndicesDesc, numWeights, paddingIdx, scaleGradByFreq),
                      OUTPUT(tensorOutDesc));

  uint64_t workspaceSize1 = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize1);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_embedding_dense_backward_test, case_38) {
  int64_t numWeights = 10;
  int64_t paddingIdx = 0;
  bool scaleGradByFreq = false;
  auto tensorGradDesc = TensorDesc({4, 3}, ACL_FLOAT, ACL_FORMAT_ND, {1, 4}, 0, {3, 4}).ValueRange(-1, 1);
  auto tensorIndicesDesc = TensorDesc({4}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 5);
  auto tensorOutDesc = TensorDesc({10, 3}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnEmbeddingDenseBackward,
                      INPUT(tensorGradDesc, tensorIndicesDesc, numWeights, paddingIdx, scaleGradByFreq),
                      OUTPUT(tensorOutDesc));

  uint64_t workspaceSize1 = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize1);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  ut.TestPrecision();
}

// embedding_dense_grad_v2
TEST_F(l2_embedding_dense_backward_test, ascend910B2_case_39) {
  int64_t numWeights = 10;
  int64_t paddingIdx = 0;
  bool scaleGradByFreq = false;
  auto tensorGradDesc = TensorDesc({4, 3, 3, 3, 3}, ACL_FLOAT, ACL_FORMAT_NDHWC)
                            .ValueRange(0, 2);
  auto tensorIndicesDesc = TensorDesc({4,3,3,3}, ACL_INT32, ACL_FORMAT_HWCN)
                               .ValueRange(0, 9);
  auto tensorOutDesc = TensorDesc({10, 3}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnEmbeddingDenseBackward,
                      INPUT(tensorGradDesc, tensorIndicesDesc, numWeights, paddingIdx, scaleGradByFreq),
                      OUTPUT(tensorOutDesc));

  uint64_t workspaceSize1 = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize1);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

}

TEST_F(l2_embedding_dense_backward_test, ascend910B2_case_40) {
  int64_t numWeights = 10;
  int64_t paddingIdx = -1;
  bool scaleGradByFreq = false;
  auto tensorGradDesc = TensorDesc({1, 3}, ACL_FLOAT, ACL_FORMAT_NDHWC)
                            .ValueRange(0, 2);
  auto tensorIndicesDesc = TensorDesc({1}, ACL_INT32, ACL_FORMAT_HWCN)
                               .ValueRange(0, 9);
  auto tensorOutDesc = TensorDesc({10, 3}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnEmbeddingDenseBackward,
                      INPUT(tensorGradDesc, tensorIndicesDesc, numWeights, paddingIdx, scaleGradByFreq),
                      OUTPUT(tensorOutDesc));

  uint64_t workspaceSize1 = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize1);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_embedding_dense_backward_test, case_41) {
  int64_t numWeights = 10;
  int64_t paddingIdx = 0;
  bool scaleGradByFreq = false;
  auto tensorGradDesc = TensorDesc({4, 3}, ACL_FLOAT, ACL_FORMAT_NC1HWC0)
                            .ValueRange(-1, 1)
                            .Value(vector<float>{0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2});
  auto tensorIndicesDesc = TensorDesc({4}, ACL_INT32, ACL_FORMAT_ND)
                           .ValueRange(0, 5)
                           .Value(vector<int>{1,2,3,4});
  auto tensorOutDesc = TensorDesc({10, 3}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnEmbeddingDenseBackward,
                      INPUT(tensorGradDesc, tensorIndicesDesc, numWeights, paddingIdx, scaleGradByFreq),
                      OUTPUT(tensorOutDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 不支持indices私有格式
TEST_F(l2_embedding_dense_backward_test, case_42) {
  int64_t numWeights = 10;
  int64_t paddingIdx = 0;
  bool scaleGradByFreq = false;
  auto tensorGradDesc = TensorDesc({4, 3}, ACL_FLOAT, ACL_FORMAT_ND)
                            .ValueRange(-1, 1)
                            .Value(vector<float>{0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2});
  auto tensorIndicesDesc = TensorDesc({4}, ACL_INT32, ACL_FORMAT_NC1HWC0)
                           .ValueRange(0, 5)
                           .Value(vector<int>{1,2,3,4});
  auto tensorOutDesc = TensorDesc({10, 3}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnEmbeddingDenseBackward,
                      INPUT(tensorGradDesc, tensorIndicesDesc, numWeights, paddingIdx, scaleGradByFreq),
                      OUTPUT(tensorOutDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 不支持out私有格式
TEST_F(l2_embedding_dense_backward_test, case_43) {
  int64_t numWeights = 10;
  int64_t paddingIdx = 0;
  bool scaleGradByFreq = false;
  auto tensorGradDesc = TensorDesc({4, 3}, ACL_FLOAT, ACL_FORMAT_ND)
                            .ValueRange(-1, 1)
                            .Value(vector<float>{0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2});
  auto tensorIndicesDesc = TensorDesc({4}, ACL_INT32, ACL_FORMAT_ND)
                           .ValueRange(0, 5)
                           .Value(vector<int>{1,2,3,4});
  auto tensorOutDesc = TensorDesc({10, 3}, ACL_FLOAT, ACL_FORMAT_NC1HWC0);

  auto ut = OP_API_UT(aclnnEmbeddingDenseBackward,
                      INPUT(tensorGradDesc, tensorIndicesDesc, numWeights, paddingIdx, scaleGradByFreq),
                      OUTPUT(tensorOutDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// grad为空tensor
TEST_F(l2_embedding_dense_backward_test, case_44) {
  int64_t numWeights = 155136;
  int64_t paddingIdx = -1;
  bool scaleGradByFreq = false;
  auto tensorGradDesc = TensorDesc({0, 4096}, ACL_FLOAT, ACL_FORMAT_NDHWC)
                            .ValueRange(0, 2);
  auto tensorIndicesDesc = TensorDesc({0}, ACL_INT32, ACL_FORMAT_HWCN)
                               .ValueRange(0, 9);
  auto tensorOutDesc = TensorDesc({155136, 4096}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnEmbeddingDenseBackward,
                      INPUT(tensorGradDesc, tensorIndicesDesc, numWeights, paddingIdx, scaleGradByFreq),
                      OUTPUT(tensorOutDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_embedding_dense_backward_test, ascend950PR_9589_case) {
  int64_t numWeights = 20;
  int64_t paddingIdx = -1;
  bool scaleGradByFreq = false;
  auto tensorGradDesc = TensorDesc({10, 5}, ACL_FLOAT, ACL_FORMAT_ND)
                            .ValueRange(1, 3);
  auto tensorIndicesDesc = TensorDesc({10}, ACL_INT32, ACL_FORMAT_ND)
                               .ValueRange(0, 5);
  auto tensorOutDesc = TensorDesc({20, 5}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnEmbeddingDenseBackward,
                      INPUT(tensorGradDesc, tensorIndicesDesc, numWeights, paddingIdx, scaleGradByFreq),
                      OUTPUT(tensorOutDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
  ut.TestPrecision();
}
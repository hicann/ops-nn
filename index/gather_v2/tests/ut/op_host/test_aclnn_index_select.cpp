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

#include "../../../op_host/op_api/aclnn_index_select.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"


using namespace std;

class l2IndexSelectTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    cout << "index_select_test SetUp" << endl;
  }

  static void TearDownTestCase() { cout << "index_select_test TearDown" << endl; }
};

TEST_F(l2IndexSelectTest, l2_index_select_case_001) {
  // self张量的shape
  vector<int64_t> inputDims = {2, 3, 4, 1};
  vector<int64_t> outputDims = inputDims;

  // 索引张量
  vector<int64_t> index = {0, 1};

  // 根据dim, index来推导输出张量的最终shape
  int64_t dim = 2;
  outputDims[dim] = index.size();

  auto selfDesc = TensorDesc(inputDims, ACL_FLOAT, ACL_FORMAT_ND)
                         .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                                              17, 18, 19, 20, 21, 22, 23, 24});
  int64_t indexSize = index.size();
  auto indexDesc = TensorDesc({indexSize}, ACL_INT64, ACL_FORMAT_ND).Value(index);
  auto outputDesc = TensorDesc(outputDims, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnIndexSelect,  // host api第二段接口名称
                      INPUT(selfDesc, dim, indexDesc),   // host api输入
                      OUTPUT(outputDesc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  ut.TestPrecision();
}

// test_002 index_dtype = int64  self_dtype= INT32
TEST_F(l2IndexSelectTest, l2_index_select_case_002) {
  vector<int64_t> inputDims = {1, 2, 2, 2};
  vector<int64_t> outputDims = inputDims;
  vector<int64_t> index = {0};
  int64_t dim = 2;
  outputDims[dim] = index.size();

  auto selfDesc = TensorDesc(inputDims, ACL_INT32, ACL_FORMAT_HWCN)
                         .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8,});
  int64_t indexSize = index.size();
  auto indexDesc = TensorDesc({indexSize}, ACL_INT64, ACL_FORMAT_ND).Value(index);
  auto outputDesc = TensorDesc(outputDims, ACL_INT32, ACL_FORMAT_NDHWC);

  auto ut = OP_API_UT(aclnnIndexSelect,  // host api第二段接口名称
                      INPUT(selfDesc, dim, indexDesc),   // host api输入
                      OUTPUT(outputDesc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  ut.TestPrecision();
}

// test_003 index_dtype = int32, self_dtype = int64
TEST_F(l2IndexSelectTest, l2_index_select_case_003) {
  vector<int64_t> inputDims = {1, 2, 2, 2};
  vector<int64_t> outputDims = inputDims;
  vector<int64_t> index = {0};
  int64_t dim = 2;
  outputDims[dim] = index.size();

  auto selfDesc = TensorDesc(inputDims, ACL_INT64, ACL_FORMAT_NCDHW)
                         .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8,});
  int64_t indexSize = index.size();
  auto indexDesc = TensorDesc({indexSize}, ACL_INT32, ACL_FORMAT_ND).Value(index);
  auto outputDesc = TensorDesc(outputDims, ACL_INT64, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnIndexSelect,  // host api第二段接口名称
                      INPUT(selfDesc, dim, indexDesc),   // host api输入
                      OUTPUT(outputDesc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  ut.TestPrecision();
}

// test_004 index_dtype = int32, self_dtype = float16
TEST_F(l2IndexSelectTest, l2_index_select_case_004) {
  vector<int64_t> inputDims = {1, 2, 2, 2};
  vector<int64_t> outputDims = inputDims;
  vector<int64_t> index = {0};
  int64_t dim = 2;
  outputDims[dim] = index.size();

  auto selfDesc = TensorDesc(inputDims, ACL_FLOAT16, ACL_FORMAT_ND)
                         .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8,});
  int64_t indexSize = index.size();
  auto indexDesc = TensorDesc({indexSize}, ACL_INT32, ACL_FORMAT_ND).Value(index);
  auto outputDesc = TensorDesc(outputDims, ACL_FLOAT16, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnIndexSelect,  // host api第二段接口名称
                      INPUT(selfDesc, dim, indexDesc),   // host api输入
                      OUTPUT(outputDesc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  ut.TestPrecision();
}

// test_005 index_dtype = int32, self_dtype = int16
TEST_F(l2IndexSelectTest, l2_index_select_case_005) {
  vector<int64_t> inputDims = {1, 2, 2, 2};
  vector<int64_t> outputDims = inputDims;
  vector<int64_t> index = {0};
  int64_t dim = 2;
  outputDims[dim] = index.size();

  auto selfDesc = TensorDesc(inputDims, ACL_INT16, ACL_FORMAT_ND)
                         .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8,});
  int64_t indexSize = index.size();
  auto indexDesc = TensorDesc({indexSize}, ACL_INT32, ACL_FORMAT_ND).Value(index);
  auto outputDesc = TensorDesc(outputDims, ACL_INT16, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnIndexSelect,  // host api第二段接口名称
                      INPUT(selfDesc, dim, indexDesc),   // host api输入
                      OUTPUT(outputDesc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  ut.TestPrecision();
}

// test_006 index_dtype = int32, self_dtype = int8
TEST_F(l2IndexSelectTest, l2_index_select_case_006) {
  vector<int64_t> inputDims = {1, 2, 2, 2};
  vector<int64_t> outputDims = inputDims;
  vector<int64_t> index = {0};
  int64_t dim = 2;
  outputDims[dim] = index.size();

  auto selfDesc = TensorDesc(inputDims, ACL_INT8, ACL_FORMAT_ND)
                         .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8,});
  int64_t indexSize = index.size();
  auto indexDesc = TensorDesc({indexSize}, ACL_INT32, ACL_FORMAT_ND).Value(index);
  auto outputDesc = TensorDesc(outputDims, ACL_INT8, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnIndexSelect,  // host api第二段接口名称
                      INPUT(selfDesc, dim, indexDesc),   // host api输入
                      OUTPUT(outputDesc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  ut.TestPrecision();
}

// test_007 index_dtype = int64, self_dtype = uint8
TEST_F(l2IndexSelectTest, l2_index_select_case_007) {
  vector<int64_t> inputDims = {1, 2, 2, 2};
  vector<int64_t> outputDims = inputDims;
  vector<int64_t> index = {0};
  int64_t dim = 2;
  outputDims[dim] = index.size();

  auto selfDesc = TensorDesc(inputDims, ACL_UINT8, ACL_FORMAT_ND)
                         .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8,});
  int64_t indexSize = index.size();
  auto indexDesc = TensorDesc({indexSize}, ACL_INT64, ACL_FORMAT_ND).Value(index);
  auto outputDesc = TensorDesc(outputDims, ACL_UINT8, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnIndexSelect,  // host api第二段接口名称
                      INPUT(selfDesc, dim, indexDesc),   // host api输入
                      OUTPUT(outputDesc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  ut.TestPrecision();
}

// test_008 index_dtype = int32, self_dtype = bool
TEST_F(l2IndexSelectTest, l2_index_select_case_008) {
  vector<int64_t> inputDims = {1, 2, 2, 2};
  vector<int64_t> outputDims = inputDims;
  vector<int64_t> index = {0};
  int64_t dim = 2;
  outputDims[dim] = index.size();

  auto selfDesc = TensorDesc(inputDims, ACL_BOOL, ACL_FORMAT_NCHW)
                         .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8,});
  int64_t indexSize = index.size();
  auto indexDesc = TensorDesc({indexSize}, ACL_INT32, ACL_FORMAT_ND).Value(index);
  auto outputDesc = TensorDesc(outputDims, ACL_BOOL, ACL_FORMAT_NHWC);

  auto ut = OP_API_UT(aclnnIndexSelect,  // host api第二段接口名称
                      INPUT(selfDesc, dim, indexDesc),   // host api输入
                      OUTPUT(outputDesc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  ut.TestPrecision();
}

// test_009 index_dtype = int32, self_dtype = uint64
TEST_F(l2IndexSelectTest, l2_index_select_case_009) {
  vector<int64_t> inputDims = {1, 2, 2, 2};
  vector<int64_t> outputDims = inputDims;
  vector<int64_t> index = {0};
  int64_t dim = 2;
  outputDims[dim] = index.size();

  auto selfDesc = TensorDesc(inputDims, ACL_UINT64, ACL_FORMAT_ND)
                         .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8,});
  int64_t indexSize = index.size();
  auto indexDesc = TensorDesc({indexSize}, ACL_INT32, ACL_FORMAT_ND).Value(index);
  auto outputDesc = TensorDesc(outputDims, ACL_UINT64, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnIndexSelect,  // host api第二段接口名称
                      INPUT(selfDesc, dim, indexDesc),   // host api输入
                      OUTPUT(outputDesc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  ut.TestPrecision();
}

// test_010 index_dtype = int32, self_dtype = uint32
TEST_F(l2IndexSelectTest, l2_index_select_case_010) {
  vector<int64_t> inputDims = {1, 2, 2, 2};
  vector<int64_t> outputDims = inputDims;
  vector<int64_t> index = {0};
  int64_t dim = 2;
  outputDims[dim] = index.size();

  auto selfDesc = TensorDesc(inputDims, ACL_UINT32, ACL_FORMAT_ND)
                         .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8,});
  int64_t indexSize = index.size();
  auto indexDesc = TensorDesc({indexSize}, ACL_INT32, ACL_FORMAT_ND).Value(index);
  auto outputDesc = TensorDesc(outputDims, ACL_UINT32, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnIndexSelect,  // host api第二段接口名称
                      INPUT(selfDesc, dim, indexDesc),   // host api输入
                      OUTPUT(outputDesc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  ut.TestPrecision();
}

// test_011 index_dtype = int32, self_dtype = Uint16
TEST_F(l2IndexSelectTest, l2_index_select_case_011) {
  vector<int64_t> inputDims = {1, 2, 2, 2};
  vector<int64_t> outputDims = inputDims;
  vector<int64_t> index = {0};
  int64_t dim = 2;
  outputDims[dim] = index.size();

  auto selfDesc = TensorDesc(inputDims, ACL_UINT16, ACL_FORMAT_ND)
                         .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8,});
  int64_t indexSize = index.size();
  auto indexDesc = TensorDesc({indexSize}, ACL_INT32, ACL_FORMAT_ND).Value(index);
  auto outputDesc = TensorDesc(outputDims, ACL_UINT16, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnIndexSelect,  // host api第二段接口名称
                      INPUT(selfDesc, dim, indexDesc),   // host api输入
                      OUTPUT(outputDesc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  ut.TestPrecision();
}

// test_012 index_dtype = int32, self_dtype = int16, dim = -2
TEST_F(l2IndexSelectTest, l2_index_select_case_012) {
  vector<int64_t> inputDims = {3, 3};
  vector<int64_t> outputDims = inputDims;
  vector<int64_t> index = {0,0,0};
  int64_t dim = 0;
  outputDims[dim] = index.size();

  auto selfDesc = TensorDesc(inputDims, ACL_INT16, ACL_FORMAT_ND)
                         .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9});
  int64_t indexSize = index.size();
  auto indexDesc = TensorDesc({indexSize}, ACL_INT32, ACL_FORMAT_ND).Value(index);
  auto outputDesc = TensorDesc(outputDims, ACL_INT16, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnIndexSelect,  // host api第二段接口名称
                      INPUT(selfDesc, dim, indexDesc),   // host api输入
                      OUTPUT(outputDesc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  ut.TestPrecision();
}

// test_013 index_dtype = int32, self_dtype = int16, dim = 1
TEST_F(l2IndexSelectTest, l2_index_select_case_013) {
  vector<int64_t> inputDims = {3, 3};
  vector<int64_t> outputDims = inputDims;
  vector<int64_t> index = {0,0,0};
  int64_t dim = 1;
  outputDims[dim] = index.size();

  auto selfDesc = TensorDesc(inputDims, ACL_INT16, ACL_FORMAT_ND)
                         .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9});
  int64_t indexSize = index.size();
  auto indexDesc = TensorDesc({indexSize}, ACL_INT32, ACL_FORMAT_ND).Value(index);
  auto outputDesc = TensorDesc(outputDims, ACL_INT16, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnIndexSelect,  // host api第二段接口名称
                      INPUT(selfDesc, dim, indexDesc),   // host api输入
                      OUTPUT(outputDesc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  ut.TestPrecision();
}

// test_012 index_dtype = int32, self_dtype = FLOAT64
TEST_F(l2IndexSelectTest, l2_index_select_test_FLOAT64) {
  vector<int64_t> inputDims = {3, 3};
  vector<int64_t> outputDims = inputDims;
  vector<int64_t> index = {0,0,0};
  int64_t dim = 0;
  outputDims[dim] = index.size();

  auto selfDesc = TensorDesc(inputDims, ACL_DOUBLE, ACL_FORMAT_ND)
                         .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9});
  int64_t indexSize = index.size();
  auto indexDesc = TensorDesc({indexSize}, ACL_INT64, ACL_FORMAT_ND).Value(index);
  auto outputDesc = TensorDesc(outputDims, ACL_DOUBLE, ACL_FORMAT_NCHW);

  auto ut = OP_API_UT(aclnnIndexSelect,  // host api第二段接口名称
                      INPUT(selfDesc, dim, indexDesc),   // host api输入
                      OUTPUT(outputDesc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  ut.TestPrecision();
}

// test_dtype: index_dtype = int32, self_dtype = COMPLEX64
TEST_F(l2IndexSelectTest, l2_index_select_test_COMPLEX64) {
  vector<int64_t> inputDims = {3, 3};
  vector<int64_t> outputDims = inputDims;
  vector<int64_t> index = {0,0,0};
  int64_t dim = 0;
  outputDims[dim] = index.size();

  auto selfDesc = TensorDesc(inputDims, ACL_COMPLEX64, ACL_FORMAT_ND)
                         .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9});
  int64_t indexSize = index.size();
  auto indexDesc = TensorDesc({indexSize}, ACL_INT64, ACL_FORMAT_ND).Value(index);
  auto outputDesc = TensorDesc(outputDims, ACL_COMPLEX64, ACL_FORMAT_NCHW);

  auto ut = OP_API_UT(aclnnIndexSelect,  // host api第二段接口名称
                      INPUT(selfDesc, dim, indexDesc),   // host api输入
                      OUTPUT(outputDesc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  ut.TestPrecision();
}

// test_dtype: index_dtype = int32, self_dtype = COMPLEX128
TEST_F(l2IndexSelectTest, l2_index_select_test_COMPLEX128) {
  vector<int64_t> inputDims = {3, 3};
  vector<int64_t> outputDims = inputDims;
  vector<int64_t> index = {0};
  int64_t dim = 0;
  outputDims[dim] = index.size();

  auto selfDesc = TensorDesc(inputDims, ACL_COMPLEX128, ACL_FORMAT_ND)
                         .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9});
  int64_t indexSize = index.size();
  auto indexDesc = TensorDesc({indexSize}, ACL_INT64, ACL_FORMAT_ND).Value(index);
  auto outputDesc = TensorDesc(outputDims, ACL_COMPLEX128, ACL_FORMAT_NCHW);

  auto ut = OP_API_UT(aclnnIndexSelect,  // host api第二段接口名称
                      INPUT(selfDesc, dim, indexDesc),   // host api输入
                      OUTPUT(outputDesc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  ut.TestPrecision();
}

// test_dim_invalid
TEST_F(l2IndexSelectTest, l2_index_select_test_dim_invalid) {
  vector<int64_t> inputDims = {3, 3};
  vector<int64_t> outputDims = inputDims;
  vector<int64_t> index = {0,0,0};
  int64_t dim = 4;
  int64_t outputSize = outputDims.size();
  if (dim < outputSize){
    outputDims[dim] = index.size();
  }


  auto selfDesc = TensorDesc(inputDims, ACL_DOUBLE, ACL_FORMAT_ND)
                         .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9});
  int64_t indexSize = index.size();
  auto indexDesc = TensorDesc({indexSize}, ACL_INT32, ACL_FORMAT_ND).Value(index);
  auto outputDesc = TensorDesc(outputDims, ACL_DOUBLE, ACL_FORMAT_NCHW);

  auto ut = OP_API_UT(aclnnIndexSelect,  // host api第二段接口名称
                      INPUT(selfDesc, dim, indexDesc),   // host api输入
                      OUTPUT(outputDesc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);

}

TEST_F(l2IndexSelectTest, l2_index_select_test_nullptr) {
  int64_t dim = -2;
  vector<int64_t> outputDims = {};
  auto outputDesc = TensorDesc(outputDims, ACL_INT16, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnIndexSelect,
                      INPUT(nullptr, dim, nullptr),
                      OUTPUT(outputDesc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2IndexSelectTest, l2_index_select_abnormal_dtype_UNDEFINED) {
  auto selfDesc = TensorDesc({2, 2, 2}, ACL_DT_UNDEFINED, ACL_FORMAT_ND);
  auto indexDesc = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int>{0});
  auto outDesc = TensorDesc({2, 1, 2}, ACL_DT_UNDEFINED, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnIndexSelect, INPUT(selfDesc, 0, indexDesc), OUTPUT(outDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2IndexSelectTest, l2_index_select_abnormal_self_out_dtype_not_same) {
  auto selfDesc = TensorDesc({2, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
  auto indexDesc = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int>{0});
  auto outDesc = TensorDesc({2, 1, 2}, ACL_DT_UNDEFINED, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnIndexSelect, INPUT(selfDesc, 1, indexDesc), OUTPUT(outDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2IndexSelectTest, l2_index_select_abnormal_format_NC1HWC0) {
  auto selfDesc = TensorDesc({2, 2, 2}, ACL_FLOAT, ACL_FORMAT_NC1HWC0);
  auto indexDesc = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int>{0});
  auto outDesc = TensorDesc({2, 1, 2}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnIndexSelect, INPUT(selfDesc, 1, indexDesc), OUTPUT(outDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2IndexSelectTest, l2_index_select_abnormal_format_index_dimNum_gt_1) {
  auto selfDesc = TensorDesc({2, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
  auto indexDesc = TensorDesc({1, 2}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int>{0, 1});
  auto outDesc = TensorDesc({2, 1, 2}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnIndexSelect, INPUT(selfDesc, 1, indexDesc), OUTPUT(outDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2IndexSelectTest, l2_index_select_abnormal_format_index_dimNum_gt_8) {
  auto selfDesc = TensorDesc({1, 2, 3, 4, 5, 6, 7, 8, 9}, ACL_FLOAT, ACL_FORMAT_ND);
  auto indexDesc = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int>{0});
  auto outDesc = TensorDesc({1, 1, 3, 4, 5, 6, 7, 8, 9}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnIndexSelect, INPUT(selfDesc, 1, indexDesc), OUTPUT(outDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 数据范围[-1, 1]
TEST_F(l2IndexSelectTest, l2_index_select_data_value_between_negative_1_and_postive_1) {
  auto selfDesc = TensorDesc({2, 4, 6, 8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto indexDesc = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int>{0});
  auto outDesc = TensorDesc({1, 4, 6, 8}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnIndexSelect, INPUT(selfDesc, 0, indexDesc), OUTPUT(outDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);

  ut.TestPrecision();
}

// 非连续
TEST_F(l2IndexSelectTest, l2_index_select_self_not_contiguous) {
  auto selfDesc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 2}, 0, {4, 2})
                      .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8});
  auto indexDesc = TensorDesc({3}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int>{1, 0, 2});
  auto outDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnIndexSelect, INPUT(selfDesc, 1, indexDesc), OUTPUT(outDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);

  ut.TestPrecision();
}

// index 0维
TEST_F(l2IndexSelectTest, l2_index_select_index_0_dim) {
  auto selfDesc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND);
  auto indexDesc = TensorDesc({}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int>{1});
  auto outDesc = TensorDesc({2, 1}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnIndexSelect, INPUT(selfDesc, 1, indexDesc), OUTPUT(outDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);

  ut.TestPrecision();
}

// self 0维
TEST_F(l2IndexSelectTest, l2_index_select_self_0_dim) {
  auto selfDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<int>{1});
  auto indexDesc = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int>{0, 0});
  auto outDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnIndexSelect, INPUT(selfDesc, 0, indexDesc), OUTPUT(outDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);

  ut.TestPrecision();
}

// self BFloat16
TEST_F(l2IndexSelectTest, l2_index_select_self_bf16) {
  auto selfDesc = TensorDesc({4}, ACL_BF16, ACL_FORMAT_ND).Value(vector<float>{1, 2.89, -3.01, 4});
  auto indexDesc = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int>{0, 2});
  auto outDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnIndexSelect, INPUT(selfDesc, 0, indexDesc), OUTPUT(outDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// dim negative
TEST_F(l2IndexSelectTest, l2_index_select_dim_negative) {
  auto selfDesc = TensorDesc({4, 3}, ACL_FLOAT, ACL_FORMAT_ND);
  auto indexDesc = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int>{0, 2});
  auto outDesc = TensorDesc({4, 2}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnIndexSelect, INPUT(selfDesc, -1, indexDesc), OUTPUT(outDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);

  ut.TestPrecision();
}

TEST_F(l2IndexSelectTest, ascend910B2_l2_index_select_case_001) {
  // self张量的shape
  vector<int64_t> inputDims = {2, 3, 4, 1};
  vector<int64_t> outputDims = inputDims;

  // 索引张量
  vector<int64_t> index = {0, 1};

  // 根据dim, index来推导输出张量的最终shape
  int64_t dim = 2;
  outputDims[dim] = index.size();

  auto selfDesc = TensorDesc(inputDims, ACL_FLOAT, ACL_FORMAT_ND)
                         .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                                              17, 18, 19, 20, 21, 22, 23, 24});
  int64_t indexSize = index.size();
  auto indexDesc = TensorDesc({indexSize}, ACL_INT64, ACL_FORMAT_ND).Value(index);
  auto outputDesc = TensorDesc(outputDims, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnIndexSelect,  // host api第二段接口名称
                      INPUT(selfDesc, dim, indexDesc),   // host api输入
                      OUTPUT(outputDesc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  ut.TestPrecision();
}

TEST_F(l2IndexSelectTest, Ascend910_9589_l2_index_select_case_001) {
  // self张量的shape
  vector<int64_t> inputDims = {2, 3, 4, 1};
  vector<int64_t> outputDims = inputDims;

  // 索引张量
  vector<int64_t> index = {0, 1};

  // 根据dim, index来推导输出张量的最终shape
  int64_t dim = 2;
  outputDims[dim] = index.size();

  auto selfDesc = TensorDesc(inputDims, ACL_FLOAT, ACL_FORMAT_ND)
                         .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                                              17, 18, 19, 20, 21, 22, 23, 24});
  int64_t indexSize = index.size();
  auto indexDesc = TensorDesc({indexSize}, ACL_INT64, ACL_FORMAT_ND).Value(index);
  auto outputDesc = TensorDesc(outputDims, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnIndexSelect,  // host api第二段接口名称
                      INPUT(selfDesc, dim, indexDesc),   // host api输入
                      OUTPUT(outputDesc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  ut.TestPrecision();
}
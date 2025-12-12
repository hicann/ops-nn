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

#include "../../../op_host/op_api/aclnn_flip.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"


using namespace std;

class l2_flip_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    cout << "l2_flip_test SetUp" << endl;
  }

  static void TearDownTestCase() { cout << "l2_flip_test TearDown" << endl; }
};

// 正常场景：数据类型为float
TEST_F(l2_flip_test, normal_1) {
  auto tensor_1_desc = TensorDesc({3, 5}, ACL_FLOAT, ACL_FORMAT_ND)
                         .ValueRange(-2, 2)
                         .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
  auto out_tensor_desc = TensorDesc({3, 5}, ACL_FLOAT, ACL_FORMAT_ND);
  auto dim = IntArrayDesc(vector<int64_t>{0, 1});

  auto ut = OP_API_UT(aclnnFlip, INPUT(tensor_1_desc, dim), OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);

  // SAMPLE: precision simulate
  ut.TestPrecision();
}

// 正常场景：数据类型为int32
TEST_F(l2_flip_test, normal_2) {
  auto tensor_1_desc = TensorDesc({3, 5}, ACL_INT32, ACL_FORMAT_ND)
                         .ValueRange(-2, 2)
                         .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
  auto out_tensor_desc = TensorDesc({3, 5}, ACL_INT32, ACL_FORMAT_ND);
  auto dim = IntArrayDesc(vector<int64_t>{0, 1});

  auto ut = OP_API_UT(aclnnFlip, INPUT(tensor_1_desc, dim), OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);

  // SAMPLE: precision simulate
  ut.TestPrecision();
}

// 正常场景：空tensor
TEST_F(l2_flip_test, normal_3) {
    auto tensor_1_desc = TensorDesc({3, 0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc({3, 0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto dim = IntArrayDesc(vector<int64_t>{0, 1});

    auto ut = OP_API_UT(aclnnFlip, INPUT(tensor_1_desc, dim), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

// 正常场景：ACL_INT16
TEST_F(l2_flip_test, normal_4) {
    auto tensor_1_desc = TensorDesc({3, 5}, ACL_INT16, ACL_FORMAT_ND)
            .ValueRange(-2, 2)
            .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
    auto out_tensor_desc = TensorDesc({3, 5}, ACL_INT16, ACL_FORMAT_ND);
    auto dim = IntArrayDesc(vector<int64_t>{0, 1});

    auto ut = OP_API_UT(aclnnFlip, INPUT(tensor_1_desc, dim), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

// 正常场景：ACL_INT64
TEST_F(l2_flip_test, normal_5) {
    auto tensor_1_desc = TensorDesc({3, 5}, ACL_INT64, ACL_FORMAT_ND)
            .ValueRange(-2, 2)
            .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
    auto out_tensor_desc = TensorDesc({3, 5}, ACL_INT64, ACL_FORMAT_ND);
    auto dim = IntArrayDesc(vector<int64_t>{0, 1});

    auto ut = OP_API_UT(aclnnFlip, INPUT(tensor_1_desc, dim), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

// 正常场景：ACL_FLOAT16
TEST_F(l2_flip_test, normal_6) {
    auto tensor_1_desc = TensorDesc({3, 5}, ACL_FLOAT16, ACL_FORMAT_ND)
            .ValueRange(-2, 2)
            .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
    auto out_tensor_desc = TensorDesc({3, 5}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto dim = IntArrayDesc(vector<int64_t>{0, 1});

    auto ut = OP_API_UT(aclnnFlip, INPUT(tensor_1_desc, dim), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

// 正常场景：format为NCHW
TEST_F(l2_flip_test, normal_11) {
    auto tensor_1_desc = TensorDesc({3, 2, 1, 2}, ACL_FLOAT, ACL_FORMAT_NCHW)
                             .ValueRange(-2, 2)
                             .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    auto out_tensor_desc = TensorDesc({3, 2, 1, 2}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto dim = IntArrayDesc(vector<int64_t>{0, 1});

    auto ut = OP_API_UT(aclnnFlip, INPUT(tensor_1_desc, dim), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

// 正常场景：format为NHWC
TEST_F(l2_flip_test, normal_12) {
    auto tensor_1_desc = TensorDesc({3, 2, 1, 2}, ACL_FLOAT, ACL_FORMAT_NHWC)
                             .ValueRange(-2, 2)
                             .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    auto out_tensor_desc = TensorDesc({3, 2, 1, 2}, ACL_FLOAT, ACL_FORMAT_NHWC);
    auto dim = IntArrayDesc(vector<int64_t>{0, 1});

    auto ut = OP_API_UT(aclnnFlip, INPUT(tensor_1_desc, dim), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

// 正常场景：format为HWCN
TEST_F(l2_flip_test, normal_13) {
    auto tensor_1_desc = TensorDesc({3, 2, 1, 2}, ACL_FLOAT, ACL_FORMAT_HWCN)
                             .ValueRange(-2, 2)
                             .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    auto out_tensor_desc = TensorDesc({3, 2, 1, 2}, ACL_FLOAT, ACL_FORMAT_HWCN);
    auto dim = IntArrayDesc(vector<int64_t>{0, 1});

    auto ut = OP_API_UT(aclnnFlip, INPUT(tensor_1_desc, dim), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}


// 正常场景：format为NDHWC
TEST_F(l2_flip_test, normal_14) {
    auto tensor_1_desc = TensorDesc({3, 2, 1, 2, 1}, ACL_FLOAT, ACL_FORMAT_NDHWC)
                             .ValueRange(-2, 2)
                             .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    auto out_tensor_desc = TensorDesc({3, 2, 1, 2, 1}, ACL_FLOAT, ACL_FORMAT_NDHWC);
    auto dim = IntArrayDesc(vector<int64_t>{0, 1});

    auto ut = OP_API_UT(aclnnFlip, INPUT(tensor_1_desc, dim), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

// 正常场景：format为NCDHW
TEST_F(l2_flip_test, normal_15) {
    auto tensor_1_desc = TensorDesc({3, 2, 1, 2, 1}, ACL_FLOAT, ACL_FORMAT_NCDHW)
                             .ValueRange(-2, 2)
                             .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    auto out_tensor_desc = TensorDesc({3, 2, 1, 2, 1}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto dim = IntArrayDesc(vector<int64_t>{0, 1});

    auto ut = OP_API_UT(aclnnFlip, INPUT(tensor_1_desc, dim), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

// 正常场景：dims包含所有轴
TEST_F(l2_flip_test, normal_16) {
    auto tensor_1_desc = TensorDesc({3, 2, 1, 2, 1}, ACL_FLOAT, ACL_FORMAT_NCDHW)
                             .ValueRange(-2, 2)
                             .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    auto out_tensor_desc = TensorDesc({3, 2, 1, 2, 1}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto dim = IntArrayDesc(vector<int64_t>{0, 1, 2, 3, 4});

    auto ut = OP_API_UT(aclnnFlip, INPUT(tensor_1_desc, dim), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

// 正常场景：dims包含最小负数的轴
TEST_F(l2_flip_test, normal_17) {
    auto tensor_1_desc = TensorDesc({3, 2, 1, 2, 1}, ACL_FLOAT, ACL_FORMAT_NCDHW)
                             .ValueRange(-2, 2)
                             .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    auto out_tensor_desc = TensorDesc({3, 2, 1, 2, 1}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto dim = IntArrayDesc(vector<int64_t>{-5, 1, 2, 3, 4});

    auto ut = OP_API_UT(aclnnFlip, INPUT(tensor_1_desc, dim), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

// 异常场景：self数据类型与out的数据类型不一致
TEST_F(l2_flip_test, abnormal_2) {
  auto tensor_1_desc = TensorDesc({3, 5}, ACL_INT8, ACL_FORMAT_ND)
                         .ValueRange(-2, 2)
                         .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
  auto out_tensor_desc = TensorDesc({3, 5}, ACL_FLOAT, ACL_FORMAT_ND);
  auto dim = IntArrayDesc(vector<int64_t>{0, 1});

  auto ut = OP_API_UT(aclnnFlip, INPUT(tensor_1_desc, dim), OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);

  // SAMPLE: precision simulate
  // ut.TestPrecision();
}


// 异常场景：self与out的shape不一致
TEST_F(l2_flip_test, abnormal_3) {
  auto tensor_1_desc = TensorDesc({3, 3}, ACL_FLOAT, ACL_FORMAT_ND)
                         .ValueRange(-2, 2)
                         .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
  auto out_tensor_desc = TensorDesc({3, 5}, ACL_FLOAT, ACL_FORMAT_ND);
  auto dim = IntArrayDesc(vector<int64_t>{0, 1});

  auto ut = OP_API_UT(aclnnFlip, INPUT(tensor_1_desc, dim), OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);

  // SAMPLE: precision simulate
  // ut.TestPrecision();
}


// 异常场景：self为空
TEST_F(l2_flip_test, abnormal_4) {
  // auto tensor_1_desc = TensorDesc({3, 3}, ACL_FLOAT, ACL_FORMAT_ND)
  //                        .ValueRange(-2, 2)
  //                        .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
  auto out_tensor_desc = TensorDesc({3, 5}, ACL_FLOAT, ACL_FORMAT_ND);
  auto dim = IntArrayDesc(vector<int64_t>{0, 1});

  auto ut = OP_API_UT(aclnnFlip, INPUT((aclTensor*)nullptr, dim), OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

  // SAMPLE: precision simulate
  // ut.TestPrecision();
}

// 异常场景：dim为空
TEST_F(l2_flip_test, abnormal_dim_null) {
  auto tensor_1_desc = TensorDesc({3, 3}, ACL_FLOAT, ACL_FORMAT_ND)
                         .ValueRange(-2, 2)
                         .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
  auto out_tensor_desc = TensorDesc({3, 5}, ACL_FLOAT, ACL_FORMAT_ND);
  // auto dim = IntArrayDesc(vector<int64_t>{0, 1});

  auto ut = OP_API_UT(aclnnFlip, INPUT(tensor_1_desc, (aclIntArray*)nullptr), OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

  // SAMPLE: precision simulate
  // ut.TestPrecision();
}

// 异常场景：out为空
TEST_F(l2_flip_test, abnormal_out_null) {
  auto tensor_1_desc = TensorDesc({3, 3}, ACL_FLOAT, ACL_FORMAT_ND)
                         .ValueRange(-2, 2)
                         .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
  // auto out_tensor_desc = TensorDesc({3, 5}, ACL_FLOAT, ACL_FORMAT_ND);
  auto dim = IntArrayDesc(vector<int64_t>{0, 1});

  auto ut = OP_API_UT(aclnnFlip, INPUT(tensor_1_desc, dim), OUTPUT((aclTensor*)nullptr));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

  // SAMPLE: precision simulate
  // ut.TestPrecision();
}

// 异常场景：dims中轴有重复
TEST_F(l2_flip_test, abnormal_5) {
  auto tensor_1_desc = TensorDesc({3, 2, 1, 2, 1}, ACL_FLOAT, ACL_FORMAT_NCDHW)
                           .ValueRange(-2, 2)
                           .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
  auto out_tensor_desc = TensorDesc({3, 2, 1, 2, 1}, ACL_FLOAT, ACL_FORMAT_NCDHW);
  auto dim = IntArrayDesc(vector<int64_t>{0, 1, 2, 3, 4, -1});

  auto ut = OP_API_UT(aclnnFlip, INPUT(tensor_1_desc, dim), OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);

  // SAMPLE: precision simulate
  //  ut.TestPrecision();
}

// 异常场景：dims中轴有重复
TEST_F(l2_flip_test, abnormal_6) {
  auto tensor_1_desc = TensorDesc({3, 2, 1, 2, 1}, ACL_FLOAT, ACL_FORMAT_NCDHW)
                           .ValueRange(-2, 2)
                           .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
  auto out_tensor_desc = TensorDesc({3, 2, 1, 2, 1}, ACL_FLOAT, ACL_FORMAT_NCDHW);
  auto dim = IntArrayDesc(vector<int64_t>{0, 1, 2, 3, 4, -5});

  auto ut = OP_API_UT(aclnnFlip, INPUT(tensor_1_desc, dim), OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);

  // SAMPLE: precision simulate
  //  ut.TestPrecision();
}

// 异常场景：dims中轴有重复
TEST_F(l2_flip_test, abnormal_7) {
  auto tensor_1_desc = TensorDesc({3, 2, 1, 2, 1}, ACL_FLOAT, ACL_FORMAT_NCDHW)
                           .ValueRange(-2, 2)
                           .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
  auto out_tensor_desc = TensorDesc({3, 2, 1, 2, 1}, ACL_FLOAT, ACL_FORMAT_NCDHW);
  auto dim = IntArrayDesc(vector<int64_t>{0, 1, 2, 3, 4, -2});

  auto ut = OP_API_UT(aclnnFlip, INPUT(tensor_1_desc, dim), OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);

  // SAMPLE: precision simulate
  //  ut.TestPrecision();
}

// 异常场景：dims中value不在有效范围
TEST_F(l2_flip_test, abnormal_8) {
  auto tensor_1_desc = TensorDesc({3, 2, 1, 2, 1}, ACL_FLOAT, ACL_FORMAT_NCDHW)
                           .ValueRange(-2, 2)
                           .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
  auto out_tensor_desc = TensorDesc({3, 2, 1, 2, 1}, ACL_FLOAT, ACL_FORMAT_NCDHW);
  auto dim = IntArrayDesc(vector<int64_t>{0, 1, 2, 3, 8});

  auto ut = OP_API_UT(aclnnFlip, INPUT(tensor_1_desc, dim), OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);

  // SAMPLE: precision simulate
  //  ut.TestPrecision();
}


// 异常场景：dims中value不在有效范围
TEST_F(l2_flip_test, abnormal_9) {
  auto tensor_1_desc = TensorDesc({3, 2, 1, 2, 1}, ACL_FLOAT, ACL_FORMAT_NCDHW)
                           .ValueRange(-2, 2)
                           .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
  auto out_tensor_desc = TensorDesc({3, 2, 1, 2, 1}, ACL_FLOAT, ACL_FORMAT_NCDHW);
  auto dim = IntArrayDesc(vector<int64_t>{0, 1, 2, 3, -6});

  auto ut = OP_API_UT(aclnnFlip, INPUT(tensor_1_desc, dim), OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);

  // SAMPLE: precision simulate
  //  ut.TestPrecision();
}

// 异常场景：维度大于8
TEST_F(l2_flip_test, abnormal_10) {
  auto tensor_1_desc = TensorDesc({3, 2, 1, 2, 1, 3, 2, 1, 2, 1}, ACL_FLOAT, ACL_FORMAT_NCDHW);
  auto out_tensor_desc = TensorDesc({3, 2, 1, 2, 1, 3, 2, 1, 2, 1}, ACL_FLOAT, ACL_FORMAT_NCDHW);
  auto dim = IntArrayDesc(vector<int64_t>{0, 1, 2, 3});

  auto ut = OP_API_UT(aclnnFlip, INPUT(tensor_1_desc, dim), OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);

  // SAMPLE: precision simulate
  //  ut.TestPrecision();
}












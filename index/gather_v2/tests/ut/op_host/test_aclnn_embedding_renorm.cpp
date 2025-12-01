/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <array>
#include <vector>
#include "gtest/gtest.h"
#include "../../../op_host/op_api/aclnn_embedding_renorm.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"


using namespace std;

class l2EmbeddingRenormTest : public testing::Test {
protected:
 static void SetUpTestCase() { cout << "embedding_renorm_test SetUp" << endl; }

 static void TearDownTestCase() { cout << "embedding_renorm_test TearDown" << endl; }
};
// 测试入参self为空指针
TEST_F(l2EmbeddingRenormTest, testcase_002_nullptr) {
 auto selfDesc = TensorDesc({3,3}, ACL_FLOAT, ACL_FORMAT_ND);
 auto indicesDesc = TensorDesc({2}, ACL_INT64, ACL_FORMAT_ND).Value(vector<long>{1, 0});
 double normType = 1;
 double maxNorm = 1;
 auto ut = OP_API_UT(aclnnEmbeddingRenorm, INPUT((aclTensor*)nullptr, indicesDesc, maxNorm, normType),
                     OUTPUT());

 // test GetWorkspaceSize
 uint64_t workspaceSize = 7;
 aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
 EXPECT_EQ(aclRet, ACLNN_ERR_INNER_NULLPTR);
 EXPECT_EQ(workspaceSize, 7UL);
}

// 测试入参indices为空指针
TEST_F(l2EmbeddingRenormTest, testcase_003_nullptr) {
 auto selfDesc = TensorDesc({3,3}, ACL_FLOAT, ACL_FORMAT_ND);
 auto indicesDesc = TensorDesc({2}, ACL_INT64, ACL_FORMAT_ND).Value(vector<long>{1, 0});
 double normType = 1;
 double maxNorm = 1;
 auto ut = OP_API_UT(aclnnEmbeddingRenorm, INPUT(selfDesc, (aclTensor*)nullptr, maxNorm, normType),
                     OUTPUT());

 // test GetWorkspaceSize
 uint64_t workspaceSize = 7;
 aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
 EXPECT_EQ(aclRet, ACLNN_ERR_INNER_NULLPTR);
 EXPECT_EQ(workspaceSize, 7UL);
}

// CheckFormat 测试入参indices的dtype不为int64
TEST_F(l2EmbeddingRenormTest, testcase_005_invalid_dtype) {
 auto selfDesc = TensorDesc({3,3}, ACL_FLOAT16, ACL_FORMAT_ND);
 auto indicesDesc = TensorDesc({2}, ACL_INT8, ACL_FORMAT_ND);
 double normType = 1;
 double maxNorm = 1;
 auto ut = OP_API_UT(aclnnEmbeddingRenorm, INPUT(selfDesc, indicesDesc, maxNorm, normType),
                     OUTPUT());

 // test GetWorkspaceSize
 uint64_t workspaceSize = 7;
 aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
 EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
 EXPECT_EQ(workspaceSize, 7UL);
}
// CheckDim 测试self数据是否为2维
TEST_F(l2EmbeddingRenormTest, testcase_009_invalid_dimension) {
 auto selfDesc = TensorDesc({3,3,3}, ACL_FLOAT16, ACL_FORMAT_ND);
 auto indicesDesc = TensorDesc({2,2}, ACL_INT64, ACL_FORMAT_ND).Value(vector<long>{1, 0});
 double normType = 1;
 double maxNorm = 1;
 auto ut = OP_API_UT(aclnnEmbeddingRenorm, INPUT(selfDesc, indicesDesc, maxNorm, normType),
                     OUTPUT());

 // test GetWorkspaceSize
 uint64_t workspaceSize = 7;
 aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
 EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
 EXPECT_EQ(workspaceSize, 7);
}

// 测试self为空
TEST_F(l2EmbeddingRenormTest, testcase_010_self_empty) {
 auto selfDesc = TensorDesc({0,0}, ACL_FLOAT16, ACL_FORMAT_ND);
 auto indicesDesc = TensorDesc({2,2}, ACL_INT64, ACL_FORMAT_ND).Value(vector<long>{1, 0, 1, 1});
 double normType = 1;
 double maxNorm = 1;
 auto ut = OP_API_UT(aclnnEmbeddingRenorm, INPUT(selfDesc, indicesDesc, maxNorm, normType),
                     OUTPUT());

 // test GetWorkspaceSize
 uint64_t workspaceSize = 7;
 aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
 EXPECT_EQ(aclRet, ACL_SUCCESS);
 EXPECT_EQ(workspaceSize, 0);
}

// 测试indices为空
TEST_F(l2EmbeddingRenormTest, testcase_011_indices_empty) {
 auto selfDesc = TensorDesc({3,3}, ACL_FLOAT16, ACL_FORMAT_ND);
 auto indicesDesc = TensorDesc({0}, ACL_INT64, ACL_FORMAT_ND);
 double normType = 1;
 double maxNorm = 1;
 auto ut = OP_API_UT(aclnnEmbeddingRenorm, INPUT(selfDesc, indicesDesc, maxNorm, normType),
                     OUTPUT());

 // test GetWorkspaceSize
 uint64_t workspaceSize = 7;
 aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
 EXPECT_EQ(aclRet, ACL_SUCCESS);
 EXPECT_EQ(workspaceSize, 0);
}

// 测试self和indices为空
TEST_F(l2EmbeddingRenormTest, testcase_012_self_indices_empty) {
 auto selfDesc = TensorDesc({0, 0}, ACL_FLOAT16, ACL_FORMAT_ND);
 auto indicesDesc = TensorDesc({0}, ACL_INT64, ACL_FORMAT_ND).Value(vector<long>{1, 0});
 double normType = 1;
 double maxNorm = 1;
 auto ut = OP_API_UT(aclnnEmbeddingRenorm, INPUT(selfDesc, indicesDesc, maxNorm, normType),
                     OUTPUT());

 // test GetWorkspaceSize
 uint64_t workspaceSize = 7;
 aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
 EXPECT_EQ(aclRet, ACL_SUCCESS);
 EXPECT_EQ(workspaceSize, 0);
}

// CheckFormat 测试入参self的dtype不为float或者float16 - DOUBLE
TEST_F(l2EmbeddingRenormTest, testcase_013_invalid_dtype) {
 auto selfDesc = TensorDesc({3,3}, ACL_DOUBLE, ACL_FORMAT_ND);
 auto indicesDesc = TensorDesc({2}, ACL_INT64, ACL_FORMAT_ND).Value(vector<long>{1, 0});
 double normType = 1;
 double maxNorm = 1;
 auto ut = OP_API_UT(aclnnEmbeddingRenorm, INPUT(selfDesc, indicesDesc, maxNorm, normType),
                     OUTPUT());

 // test GetWorkspaceSize
 uint64_t workspaceSize = 7;
 aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
 EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
 EXPECT_EQ(workspaceSize, 7UL);
}

// CheckFormat 测试入参self的dtype不为float或者float16 - INT32
TEST_F(l2EmbeddingRenormTest, testcase_014_invalid_dtype) {
 auto selfDesc = TensorDesc({3,3}, ACL_INT32, ACL_FORMAT_ND);
 auto indicesDesc = TensorDesc({2}, ACL_INT64, ACL_FORMAT_ND).Value(vector<long>{1, 0});
 double normType = 1;
 double maxNorm = 1;
 auto ut = OP_API_UT(aclnnEmbeddingRenorm, INPUT(selfDesc, indicesDesc, maxNorm, normType),
                     OUTPUT());

 // test GetWorkspaceSize
 uint64_t workspaceSize = 7;
 aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
 EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
 EXPECT_EQ(workspaceSize, 7UL);
}

// CheckFormat 测试入参self的dtype不为float或者float16 - INT64
TEST_F(l2EmbeddingRenormTest, testcase_015_invalid_dtype) {
 auto selfDesc = TensorDesc({3,3}, ACL_INT64, ACL_FORMAT_ND);
 auto indicesDesc = TensorDesc({2}, ACL_INT64, ACL_FORMAT_ND).Value(vector<long>{1, 0});
 double normType = 1;
 double maxNorm = 1;
 auto ut = OP_API_UT(aclnnEmbeddingRenorm, INPUT(selfDesc, indicesDesc, maxNorm, normType),
                     OUTPUT());

 // test GetWorkspaceSize
 uint64_t workspaceSize = 7;
 aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
 EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
 EXPECT_EQ(workspaceSize, 7UL);
}

// CheckFormat 测试入参self的dtype不为float或者float16 - INT16
TEST_F(l2EmbeddingRenormTest, testcase_016_invalid_dtype) {
 auto selfDesc = TensorDesc({3,3}, ACL_INT16, ACL_FORMAT_ND);
 auto indicesDesc = TensorDesc({2}, ACL_INT64, ACL_FORMAT_ND).Value(vector<long>{1, 0});
 double normType = 1;
 double maxNorm = 1;
 auto ut = OP_API_UT(aclnnEmbeddingRenorm, INPUT(selfDesc, indicesDesc, maxNorm, normType),
                     OUTPUT());

 // test GetWorkspaceSize
 uint64_t workspaceSize = 7;
 aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
 EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
 EXPECT_EQ(workspaceSize, 7UL);
}

// CheckFormat 测试入参self的dtype不为float或者float16 - INT8
TEST_F(l2EmbeddingRenormTest, testcase_017_invalid_dtype) {
 auto selfDesc = TensorDesc({3,3}, ACL_INT8, ACL_FORMAT_ND);
 auto indicesDesc = TensorDesc({2}, ACL_INT64, ACL_FORMAT_ND).Value(vector<long>{1, 0});
 double normType = 1;
 double maxNorm = 1;
 auto ut = OP_API_UT(aclnnEmbeddingRenorm, INPUT(selfDesc, indicesDesc, maxNorm, normType),
                     OUTPUT());

 // test GetWorkspaceSize
 uint64_t workspaceSize = 7;
 aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
 EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
 EXPECT_EQ(workspaceSize, 7UL);
}

// CheckFormat 测试入参self的dtype不为float或者float16 - UINT8
TEST_F(l2EmbeddingRenormTest, testcase_018_invalid_dtype) {
 auto selfDesc = TensorDesc({3,3}, ACL_UINT8, ACL_FORMAT_ND);
 auto indicesDesc = TensorDesc({2}, ACL_INT64, ACL_FORMAT_ND).Value(vector<long>{1, 0});
 double normType = 1;
 double maxNorm = 1;
 auto ut = OP_API_UT(aclnnEmbeddingRenorm, INPUT(selfDesc, indicesDesc, maxNorm, normType),
                     OUTPUT());

 // test GetWorkspaceSize
 uint64_t workspaceSize = 7;
 aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
 EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
 EXPECT_EQ(workspaceSize, 7UL);
}

// CheckFormat 测试入参self的dtype不为float或者float16 - BOOL
TEST_F(l2EmbeddingRenormTest, testcase_019_invalid_dtype) {
 auto selfDesc = TensorDesc({3,3}, ACL_BOOL, ACL_FORMAT_ND);
 auto indicesDesc = TensorDesc({2}, ACL_INT64, ACL_FORMAT_ND).Value(vector<long>{1, 0});
 double normType = 1;
 double maxNorm = 1;
 auto ut = OP_API_UT(aclnnEmbeddingRenorm, INPUT(selfDesc, indicesDesc, maxNorm, normType),
                     OUTPUT());

 // test GetWorkspaceSize
 uint64_t workspaceSize = 7;
 aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
 EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
 EXPECT_EQ(workspaceSize, 7UL);
}

// CheckFormat 测试入参self的dtype不为float或者float16 - COMPLEX64
TEST_F(l2EmbeddingRenormTest, testcase_020_invalid_dtype) {
 auto selfDesc = TensorDesc({3,3}, ACL_COMPLEX64, ACL_FORMAT_ND);
 auto indicesDesc = TensorDesc({2}, ACL_INT64, ACL_FORMAT_ND).Value(vector<long>{1, 0});
 double normType = 1;
 double maxNorm = 1;
 auto ut = OP_API_UT(aclnnEmbeddingRenorm, INPUT(selfDesc, indicesDesc, maxNorm, normType),
                     OUTPUT());

 // test GetWorkspaceSize
 uint64_t workspaceSize = 7;
 aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
 EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
 EXPECT_EQ(workspaceSize, 7UL);
}

// CheckFormat 测试入参self的dtype不为float或者float16 - COMPLEX128
TEST_F(l2EmbeddingRenormTest, testcase_021_invalid_dtype) {
 auto selfDesc = TensorDesc({3,3}, ACL_COMPLEX128, ACL_FORMAT_ND);
 auto indicesDesc = TensorDesc({2}, ACL_INT64, ACL_FORMAT_ND).Value(vector<long>{1, 0});
 double normType = 1;
 double maxNorm = 1;
 auto ut = OP_API_UT(aclnnEmbeddingRenorm, INPUT(selfDesc, indicesDesc, maxNorm, normType),
                     OUTPUT());

 // test GetWorkspaceSize
 uint64_t workspaceSize = 7;
 aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
 EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
 EXPECT_EQ(workspaceSize, 7UL);
}

// CheckFormat 测试入参indices的dtype不为int64 - FLOAT
TEST_F(l2EmbeddingRenormTest, testcase_022_invalid_dtype) {
 auto selfDesc = TensorDesc({3,3}, ACL_FLOAT16, ACL_FORMAT_ND);
 auto indicesDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
 double normType = 1;
 double maxNorm = 1;
 auto ut = OP_API_UT(aclnnEmbeddingRenorm, INPUT(selfDesc, indicesDesc, maxNorm, normType),
                     OUTPUT());

 // test GetWorkspaceSize
 uint64_t workspaceSize = 7;
 aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
 EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
 EXPECT_EQ(workspaceSize, 7UL);
}

// CheckFormat 测试入参indices的dtype不为int64 - FLOAT16
TEST_F(l2EmbeddingRenormTest, testcase_023_invalid_dtype) {
 auto selfDesc = TensorDesc({3,3}, ACL_FLOAT16, ACL_FORMAT_ND);
 auto indicesDesc = TensorDesc({2}, ACL_FLOAT16, ACL_FORMAT_ND);
 double normType = 1;
 double maxNorm = 1;
 auto ut = OP_API_UT(aclnnEmbeddingRenorm, INPUT(selfDesc, indicesDesc, maxNorm, normType),
                     OUTPUT());

 // test GetWorkspaceSize
 uint64_t workspaceSize = 7;
 aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
 EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
 EXPECT_EQ(workspaceSize, 7UL);
}

// CheckFormat 测试入参indices的dtype不为int64 - DOUBLE
TEST_F(l2EmbeddingRenormTest, testcase_025_invalid_dtype) {
 auto selfDesc = TensorDesc({3,3}, ACL_FLOAT16, ACL_FORMAT_ND);
 auto indicesDesc = TensorDesc({2}, ACL_DOUBLE, ACL_FORMAT_ND);
 double normType = 1;
 double maxNorm = 1;
 auto ut = OP_API_UT(aclnnEmbeddingRenorm, INPUT(selfDesc, indicesDesc, maxNorm, normType),
                     OUTPUT());

 // test GetWorkspaceSize
 uint64_t workspaceSize = 7;
 aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
 EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
 EXPECT_EQ(workspaceSize, 7UL);
}

// CheckFormat 测试入参indices的dtype不为int64 - INT16
TEST_F(l2EmbeddingRenormTest, testcase_027_invalid_dtype) {
 auto selfDesc = TensorDesc({3,3}, ACL_FLOAT16, ACL_FORMAT_ND);
 auto indicesDesc = TensorDesc({2}, ACL_INT16, ACL_FORMAT_ND);
 double normType = 1;
 double maxNorm = 1;
 auto ut = OP_API_UT(aclnnEmbeddingRenorm, INPUT(selfDesc, indicesDesc, maxNorm, normType),
                     OUTPUT());

 // test GetWorkspaceSize
 uint64_t workspaceSize = 7;
 aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
 EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
 EXPECT_EQ(workspaceSize, 7UL);
}

// CheckFormat 测试入参indices的dtype不为int64 - UINT8
TEST_F(l2EmbeddingRenormTest, testcase_028_invalid_dtype) {
 auto selfDesc = TensorDesc({3,3}, ACL_FLOAT16, ACL_FORMAT_ND);
 auto indicesDesc = TensorDesc({2}, ACL_UINT8, ACL_FORMAT_ND);
 double normType = 1;
 double maxNorm = 1;
 auto ut = OP_API_UT(aclnnEmbeddingRenorm, INPUT(selfDesc, indicesDesc, maxNorm, normType),
                     OUTPUT());

 // test GetWorkspaceSize
 uint64_t workspaceSize = 7;
 aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
 EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
 EXPECT_EQ(workspaceSize, 7UL);
}

// CheckFormat 测试入参indices的dtype不为int64 - BOOL
TEST_F(l2EmbeddingRenormTest, testcase_029_invalid_dtype) {
 auto selfDesc = TensorDesc({3,3}, ACL_FLOAT16, ACL_FORMAT_ND);
 auto indicesDesc = TensorDesc({2}, ACL_BOOL, ACL_FORMAT_ND);
 double normType = 1;
 double maxNorm = 1;
 auto ut = OP_API_UT(aclnnEmbeddingRenorm, INPUT(selfDesc, indicesDesc, maxNorm, normType),
                     OUTPUT());

 // test GetWorkspaceSize
 uint64_t workspaceSize = 7;
 aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
 EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
 EXPECT_EQ(workspaceSize, 7UL);
}

// CheckFormat 测试入参indices的dtype不为int64 - COMPLEX64
TEST_F(l2EmbeddingRenormTest, testcase_030_invalid_dtype) {
 auto selfDesc = TensorDesc({3,3}, ACL_FLOAT16, ACL_FORMAT_ND);
 auto indicesDesc = TensorDesc({2}, ACL_COMPLEX64, ACL_FORMAT_ND);
 double normType = 1;
 double maxNorm = 1;
 auto ut = OP_API_UT(aclnnEmbeddingRenorm, INPUT(selfDesc, indicesDesc, maxNorm, normType),
                     OUTPUT());

 // test GetWorkspaceSize
 uint64_t workspaceSize = 7;
 aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
 EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
 EXPECT_EQ(workspaceSize, 7UL);
}

// CheckFormat 测试入参indices的dtype不为int64 - COMPLEX128
TEST_F(l2EmbeddingRenormTest, testcase_031_invalid_dtype) {
 auto selfDesc = TensorDesc({3,3}, ACL_FLOAT16, ACL_FORMAT_ND);
 auto indicesDesc = TensorDesc({2}, ACL_COMPLEX128, ACL_FORMAT_ND);
 double normType = 1;
 double maxNorm = 1;
 auto ut = OP_API_UT(aclnnEmbeddingRenorm, INPUT(selfDesc, indicesDesc, maxNorm, normType),
                     OUTPUT());

 // test GetWorkspaceSize
 uint64_t workspaceSize = 7;
 aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
 EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
 EXPECT_EQ(workspaceSize, 7UL);
}

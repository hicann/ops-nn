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
#include "level2/aclnn_embedding_renorm.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"


using namespace std;

class l2EmbeddingRenormTest : public testing::Test {
protected:
 static void SetUpTestCase() { cout << "embedding_renorm_test SetUp" << endl; }

 static void TearDownTestCase() { cout << "embedding_renorm_test TearDown" << endl; }
};

// 正常输入输出，需要golden
TEST_F(l2EmbeddingRenormTest, testcase_001_normal) {
 auto selfDesc = TensorDesc({3,3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10,10).Value(vector<float>{1,1,1,2,3,4,5,6,7});
 auto indicesDesc = TensorDesc({3}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-3,2).Value(vector<int64_t>{0,2,1});
 double normType = 1.;
 double maxNorm = 1.;
 auto ut = OP_API_UT(aclnnEmbeddingRenorm, INPUT(selfDesc, indicesDesc, maxNorm, normType),OUTPUT());

 // test GetWorkspaceSize
 uint64_t workspaceSize = 0;
 aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
 EXPECT_EQ(aclRet, ACL_SUCCESS);
 // precision simulate
 ut.TestPrecision();
}

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

TEST_F(l2EmbeddingRenormTest, ascend910B2_promote_type) {
 auto selfDesc = TensorDesc({3,3}, ACL_BF16, ACL_FORMAT_ND);
 auto indicesDesc = TensorDesc({2}, ACL_INT64, ACL_FORMAT_ND).Value(vector<long>{1, 0});
 double normType = 1;
 double maxNorm = 1;
 auto ut = OP_API_UT(aclnnEmbeddingRenorm, INPUT(selfDesc, indicesDesc, maxNorm, normType),
                     OUTPUT());

 // test GetWorkspaceSize
 uint64_t workspaceSize = 7;
 aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
 EXPECT_EQ(aclRet, ACL_SUCCESS);
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

// CheckPrompteType 测试入参是否需要做数据类型转换
TEST_F(l2EmbeddingRenormTest, testcase_006_promote_type) {
 auto selfDesc = TensorDesc({3,3}, ACL_FLOAT16, ACL_FORMAT_ND);
 auto indicesDesc = TensorDesc({2}, ACL_INT64, ACL_FORMAT_ND).Value(vector<long>{1, 0});
 double normType = 1;
 double maxNorm = 1;
 auto ut = OP_API_UT(aclnnEmbeddingRenorm, INPUT(selfDesc, indicesDesc, maxNorm, normType),
                     OUTPUT());

 // test GetWorkspaceSize
 uint64_t workspaceSize = 7;
 aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
 EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// CheckFormat 测试self数据格式是否支持
TEST_F(l2EmbeddingRenormTest, testcase_007_invalid_format) {
 auto selfDesc = TensorDesc({3,3}, ACL_FLOAT16, ACL_FORMAT_NHWC);
 auto indicesDesc = TensorDesc({2}, ACL_INT64, ACL_FORMAT_ND).Value(vector<long>{1, 0});
 double normType = 1;
 double maxNorm = 1;
 auto ut = OP_API_UT(aclnnEmbeddingRenorm, INPUT(selfDesc, indicesDesc, maxNorm, normType),
                     OUTPUT());

 // test GetWorkspaceSize
 uint64_t workspaceSize = 7;
 aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
 EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// CheckFormat 测试indices数据格式是否支持
TEST_F(l2EmbeddingRenormTest, testcase_008_invalid_format) {
 auto selfDesc = TensorDesc({3,3}, ACL_FLOAT16, ACL_FORMAT_ND);
 auto indicesDesc = TensorDesc({2}, ACL_INT64, ACL_FORMAT_NHWC).Value(vector<long>{1, 0});
 double normType = 1;
 double maxNorm = 1;
 auto ut = OP_API_UT(aclnnEmbeddingRenorm, INPUT(selfDesc, indicesDesc, maxNorm, normType),
                     OUTPUT());

 // test GetWorkspaceSize
 uint64_t workspaceSize = 7;
 aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
 EXPECT_EQ(aclRet, ACL_SUCCESS);
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

// CheckFormat 测试入参indices的dtype不为int64 - INT32
TEST_F(l2EmbeddingRenormTest, testcase_026_invalid_dtype) {
 auto selfDesc = TensorDesc({3,3}, ACL_FLOAT16, ACL_FORMAT_ND);
 auto indicesDesc = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND);
 double normType = 1;
 double maxNorm = 1;
 auto ut = OP_API_UT(aclnnEmbeddingRenorm, INPUT(selfDesc, indicesDesc, maxNorm, normType),
                     OUTPUT());

 // test GetWorkspaceSize
 uint64_t workspaceSize = 7;
 aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
 EXPECT_EQ(aclRet, ACLNN_SUCCESS);
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

//CheckMaxNormNotNegative 测试maxNorm取非法值是否拦截
TEST_F(l2EmbeddingRenormTest, testcase_032_invalid_maxNorm) {
 auto selfDesc = TensorDesc({3,3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10,10).Value(vector<float>{1,1,1,2,3,4,5,6,7});
 auto indicesDesc = TensorDesc({3}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-3,2).Value(vector<int64_t>{0,2,1});
 double normType = 1.;
 double maxNorm = -1.;
 auto ut = OP_API_UT(aclnnEmbeddingRenorm, INPUT(selfDesc, indicesDesc, maxNorm, normType),OUTPUT());

 // test GetWorkspaceSize
 uint64_t workspaceSize = 7;
 aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
 EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
 EXPECT_EQ(workspaceSize, 7UL);
}

// CheckFormat 测试self\indices数据格式是否支持
TEST_F(l2EmbeddingRenormTest, testcase_032_invalid_format) {
 auto selfDesc = TensorDesc({3,3}, ACL_FLOAT16, ACL_FORMAT_NHWC);
 auto indicesDesc = TensorDesc({2}, ACL_INT64, ACL_FORMAT_NCHW).Value(vector<long>{1, 0});
 double normType = 1;
 double maxNorm = 1;
 auto ut = OP_API_UT(aclnnEmbeddingRenorm, INPUT(selfDesc, indicesDesc, maxNorm, normType),
                     OUTPUT());

 // test GetWorkspaceSize
 uint64_t workspaceSize = 7;
 aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
 EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// CheckFormat 测试self\indices数据格式是否支持
TEST_F(l2EmbeddingRenormTest, testcase_033_invalid_format) {
 auto selfDesc = TensorDesc({3,3}, ACL_FLOAT16, ACL_FORMAT_HWCN);
 auto indicesDesc = TensorDesc({2}, ACL_INT64, ACL_FORMAT_NCHW).Value(vector<long>{1, 0});
 double normType = 1;
 double maxNorm = 1;
 auto ut = OP_API_UT(aclnnEmbeddingRenorm, INPUT(selfDesc, indicesDesc, maxNorm, normType),
                     OUTPUT());

 // test GetWorkspaceSize
 uint64_t workspaceSize = 7;
 aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
 EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// CheckFormat 测试self\indices数据格式是否支持
TEST_F(l2EmbeddingRenormTest, testcase_034_invalid_format) {
 auto selfDesc = TensorDesc({3,3}, ACL_FLOAT16, ACL_FORMAT_NDHWC);
 auto indicesDesc = TensorDesc({2}, ACL_INT64, ACL_FORMAT_NCDHW).Value(vector<long>{1, 0});
 double normType = 1;
 double maxNorm = 1;
 auto ut = OP_API_UT(aclnnEmbeddingRenorm, INPUT(selfDesc, indicesDesc, maxNorm, normType),
                     OUTPUT());

 // test GetWorkspaceSize
 uint64_t workspaceSize = 7;
 aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
 EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常输入输出，需要golden
TEST_F(l2EmbeddingRenormTest, testcase_035_normal) {
 auto selfDesc = TensorDesc({3,3}, ACL_FLOAT16, ACL_FORMAT_ND)
                     .Value(vector<float>{2,3,4,5,6,7,8,9,9}).Precision(0.01,0.01);
 auto indicesDesc = TensorDesc({3}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-3,2).Value(vector<int64_t>{0,2,1});
 double normType = 2.;
 double maxNorm = 2.;
 auto ut = OP_API_UT(aclnnEmbeddingRenorm, INPUT(selfDesc, indicesDesc, maxNorm, normType),OUTPUT());

 // test GetWorkspaceSize
 uint64_t workspaceSize = 0;
 aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
 EXPECT_EQ(aclRet, ACL_SUCCESS);
 // precision simulate
 ut.TestPrecision();
}

TEST_F(l2EmbeddingRenormTest, testcase_036_normal) {
 auto selfDesc = TensorDesc({25,25}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10,10);
 auto indicesDesc = TensorDesc({25}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0,24);
 double normType = 2.;
 double maxNorm = 2.;
 auto ut = OP_API_UT(aclnnEmbeddingRenorm, INPUT(selfDesc, indicesDesc, maxNorm, normType),OUTPUT());

 // test GetWorkspaceSize
 uint64_t workspaceSize = 0;
 aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
 EXPECT_EQ(aclRet, ACL_SUCCESS);
 // precision simulate
 // ut.TestPrecision();  // comment bcz of timeout in model tests (300921 ms)
}

TEST_F(l2EmbeddingRenormTest, testcase_037_normal) {
 auto selfDesc = TensorDesc({25,25}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1,1);
 auto indicesDesc = TensorDesc({11000}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0,24);
 double normType = 2.;
 double maxNorm = 2.;
 auto ut = OP_API_UT(aclnnEmbeddingRenorm, INPUT(selfDesc, indicesDesc, maxNorm, normType),OUTPUT());

 // test GetWorkspaceSize
 uint64_t workspaceSize = 0;
 aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
 EXPECT_EQ(aclRet, ACL_SUCCESS);
 // precision simulate
//  ut.TestPrecision();
}

// CheckFormat 测试indices数据格式是否支持
TEST_F(l2EmbeddingRenormTest, testcase_038_invalid_format) {
 auto selfDesc = TensorDesc({3,3}, ACL_FLOAT16, ACL_FORMAT_ND);
 auto indicesDesc = TensorDesc({2}, ACL_INT64, ACL_FORMAT_NHWC).Value(vector<long>{-1, 0});
 double normType = 1;
 double maxNorm = 1;
 auto ut = OP_API_UT(aclnnEmbeddingRenorm, INPUT(selfDesc, indicesDesc, maxNorm, normType),
                     OUTPUT());

 // test GetWorkspaceSize
 uint64_t workspaceSize = 7;
 aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
 EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// CheckFormat 测试indices数据格式是否支持
TEST_F(l2EmbeddingRenormTest, testcase_039_invalid_format) {
 auto selfDesc = TensorDesc({3,3}, ACL_FLOAT16, ACL_FORMAT_ND);
 auto indicesDesc = TensorDesc({2,2,3}, ACL_INT64, ACL_FORMAT_NHWC).ValueRange(0,2);
 double normType = 1;
 double maxNorm = 1;
 auto ut = OP_API_UT(aclnnEmbeddingRenorm, INPUT(selfDesc, indicesDesc, maxNorm, normType),
                     OUTPUT());

 // test GetWorkspaceSize
 uint64_t workspaceSize = 7;
 aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
 EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// 测试indices为0维
TEST_F(l2EmbeddingRenormTest, testcase_040_indices_0d) {
 auto selfDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
 auto indicesDesc = TensorDesc({}, ACL_INT64, ACL_FORMAT_ND);
 double normType = 1;
 double maxNorm = 1;
 auto ut = OP_API_UT(aclnnEmbeddingRenorm, INPUT(selfDesc, indicesDesc, maxNorm, normType),
                     OUTPUT());

 // test GetWorkspaceSize
 uint64_t workspaceSize = 0;
 aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
 EXPECT_EQ(aclRet, ACL_SUCCESS);
}

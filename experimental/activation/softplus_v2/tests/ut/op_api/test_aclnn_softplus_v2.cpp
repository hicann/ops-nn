/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <vector>

#include "gtest/gtest.h"

#include "../../../op_api/aclnn_softplus_v2.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace op;
using namespace std;

class l2_softplus_v2_test : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "softplus_v2_test SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "softplus_v2_test TearDown" << std::endl; }
};

// 正常场景_FLOAT_ND
TEST_F(l2_softplus_v2_test, case_001_float)
{
    auto selfDesc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-8, 8);
    auto betaDesc = ScalarDesc(1.0f, ACL_FLOAT);
    auto thresholdDesc = ScalarDesc(20.0f, ACL_FLOAT);
    auto outDesc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnSoftplusV2, INPUT(selfDesc, betaDesc, thresholdDesc), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_SUCCESS);
}

// 正常场景_FLOAT16_ND
TEST_F(l2_softplus_v2_test, case_002_float16)
{
    auto selfDesc = TensorDesc({2, 3, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-8, 8);
    auto betaDesc = ScalarDesc(1.0f, ACL_FLOAT);
    auto thresholdDesc = ScalarDesc(20.0f, ACL_FLOAT);
    auto outDesc = TensorDesc({2, 3, 4}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnSoftplusV2, INPUT(selfDesc, betaDesc, thresholdDesc), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_SUCCESS);
}

// 正常场景_BFLOAT16_ND
TEST_F(l2_softplus_v2_test, case_003_bfloat16)
{
    auto selfDesc = TensorDesc({2, 3, 4}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-8, 8);
    auto betaDesc = ScalarDesc(1.0f, ACL_FLOAT);
    auto thresholdDesc = ScalarDesc(20.0f, ACL_FLOAT);
    auto outDesc = TensorDesc({2, 3, 4}, ACL_BF16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnSoftplusV2, INPUT(selfDesc, betaDesc, thresholdDesc), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_SUCCESS);
}

// 空tensor场景
TEST_F(l2_softplus_v2_test, case_004_empty_tensor)
{
    auto selfDesc = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto betaDesc = ScalarDesc(1.0f, ACL_FLOAT);
    auto thresholdDesc = ScalarDesc(20.0f, ACL_FLOAT);
    auto outDesc = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnSoftplusV2, INPUT(selfDesc, betaDesc, thresholdDesc), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_SUCCESS);
}

// 非连续tensor场景
TEST_F(l2_softplus_v2_test, case_005_not_contiguous)
{
    auto selfDesc = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 5}, 0, {4, 5}).ValueRange(-8, 8);
    auto betaDesc = ScalarDesc(1.0f, ACL_FLOAT);
    auto thresholdDesc = ScalarDesc(20.0f, ACL_FLOAT);
    auto outDesc = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 5}, 0, {4, 5});

    auto ut = OP_API_UT(aclnnSoftplusV2, INPUT(selfDesc, betaDesc, thresholdDesc), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_SUCCESS);
}

// beta/threshold 为double类型，可转换为self的数据类型
TEST_F(l2_softplus_v2_test, case_006_scalar_double)
{
    auto selfDesc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-8, 8);
    auto betaDesc = ScalarDesc(static_cast<double>(1.0));
    auto thresholdDesc = ScalarDesc(static_cast<double>(20.0));
    auto outDesc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnSoftplusV2, INPUT(selfDesc, betaDesc, thresholdDesc), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_SUCCESS);
}

// self为空指针
TEST_F(l2_softplus_v2_test, case_007_nullptr_self)
{
    auto outDesc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto betaDesc = ScalarDesc(1.0f, ACL_FLOAT);
    auto thresholdDesc = ScalarDesc(20.0f, ACL_FLOAT);

    auto ut = OP_API_UT(aclnnSoftplusV2, INPUT(static_cast<aclTensor*>(nullptr), betaDesc, thresholdDesc),
                        OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_NULLPTR);
}

// beta为空指针
TEST_F(l2_softplus_v2_test, case_008_nullptr_beta)
{
    auto selfDesc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto thresholdDesc = ScalarDesc(20.0f, ACL_FLOAT);
    auto outDesc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnSoftplusV2, INPUT(selfDesc, static_cast<aclScalar*>(nullptr), thresholdDesc),
                        OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_NULLPTR);
}

// threshold为空指针
TEST_F(l2_softplus_v2_test, case_009_nullptr_threshold)
{
    auto selfDesc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto betaDesc = ScalarDesc(1.0f, ACL_FLOAT);
    auto outDesc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnSoftplusV2, INPUT(selfDesc, betaDesc, static_cast<aclScalar*>(nullptr)), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_NULLPTR);
}

// out为空指针
TEST_F(l2_softplus_v2_test, case_010_nullptr_out)
{
    auto selfDesc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto betaDesc = ScalarDesc(1.0f, ACL_FLOAT);
    auto thresholdDesc = ScalarDesc(20.0f, ACL_FLOAT);

    auto ut = OP_API_UT(aclnnSoftplusV2, INPUT(selfDesc, betaDesc, thresholdDesc),
                        OUTPUT(static_cast<aclTensor*>(nullptr)));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_NULLPTR);
}

// 非法输入dtype
TEST_F(l2_softplus_v2_test, case_011_invalid_input_dtype)
{
    auto selfDesc = TensorDesc({2, 3, 4}, ACL_INT32, ACL_FORMAT_ND);
    auto betaDesc = ScalarDesc(1.0f, ACL_FLOAT);
    auto thresholdDesc = ScalarDesc(20.0f, ACL_FLOAT);
    auto outDesc = TensorDesc({2, 3, 4}, ACL_INT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnSoftplusV2, INPUT(selfDesc, betaDesc, thresholdDesc), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
}

// 输入输出shape不一致
TEST_F(l2_softplus_v2_test, case_012_shape_mismatch)
{
    auto selfDesc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto betaDesc = ScalarDesc(1.0f, ACL_FLOAT);
    auto thresholdDesc = ScalarDesc(20.0f, ACL_FLOAT);
    auto outDesc = TensorDesc({2, 3, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnSoftplusV2, INPUT(selfDesc, betaDesc, thresholdDesc), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
}

// 超过最大维度
TEST_F(l2_softplus_v2_test, case_013_max_dim)
{
    auto selfDesc = TensorDesc({1, 2, 3, 4, 5, 6, 7, 8, 9}, ACL_FLOAT, ACL_FORMAT_ND);
    auto betaDesc = ScalarDesc(1.0f, ACL_FLOAT);
    auto thresholdDesc = ScalarDesc(20.0f, ACL_FLOAT);
    auto outDesc = TensorDesc({1, 2, 3, 4, 5, 6, 7, 8, 9}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnSoftplusV2, INPUT(selfDesc, betaDesc, thresholdDesc), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
}

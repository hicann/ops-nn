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
#include <float.h>
#include "gtest/gtest.h"

#include "../../../op_host/op_api/aclnn_addbmm.h"
#include "op_api/op_api_def.h"
#include "opdev/platform.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"

using namespace op;
using namespace std;

class l2_addbmm_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "addbmm_test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "addbmm_test TearDown" << std::endl;
    }
};

TEST_F(l2_addbmm_test, case_fp16_not_align)
{
    auto self = TensorDesc({3, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto batch1 = TensorDesc({1, 3, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto batch2 = TensorDesc({1, 4, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out = TensorDesc({3, 5}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto beta = ScalarDesc(1.0f);
    auto alpha = ScalarDesc(1.0f);
    int8_t cubeMathType = ALLOW_FP32_DOWN_PRECISION;

    auto ut = OP_API_UT(aclnnAddbmm, INPUT(self, batch1, batch2, beta, alpha), OUTPUT(out), cubeMathType);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 直接调用addbmm算子 1xN bias broadcast batch==1
TEST_F(l2_addbmm_test, case_fp16_4mm4_1mm4bias)
{
    auto self = TensorDesc({1, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat1 = TensorDesc({1, 4, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat2 = TensorDesc({1, 4, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out = TensorDesc({4, 4}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto beta = ScalarDesc(1.0f);
    auto alpha = ScalarDesc(1.0f);
    int8_t cubeMathType = ALLOW_FP32_DOWN_PRECISION;

    auto ut = OP_API_UT(aclnnAddbmm, INPUT(self, mat1, mat2, beta, alpha), OUTPUT(out), cubeMathType);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 直接调用addbmm算子 1xN bias broadcast batch==1
TEST_F(l2_addbmm_test, ascend910B2_case_bf16_4mm4_1mm4bias)
{
    auto self = TensorDesc({1, 4}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat1 = TensorDesc({1, 4, 4}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat2 = TensorDesc({1, 4, 4}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out = TensorDesc({4, 4}, ACL_BF16, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto beta = ScalarDesc(1.0f);
    auto alpha = ScalarDesc(1.0f);
    int8_t cubeMathType = ALLOW_FP32_DOWN_PRECISION;

    auto ut = OP_API_UT(aclnnAddbmm, INPUT(self, mat1, mat2, beta, alpha), OUTPUT(out), cubeMathType);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 直接调用addbmm算子 N维 bias broadcast batch==1
TEST_F(l2_addbmm_test, ascend910B2_case_bf16_4mm4_4bias)
{
    auto self = TensorDesc({4}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat1 = TensorDesc({1, 4, 4}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat2 = TensorDesc({1, 4, 4}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out = TensorDesc({4, 4}, ACL_BF16, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto beta = ScalarDesc(1.0f);
    auto alpha = ScalarDesc(1.0f);
    int8_t cubeMathType = ALLOW_FP32_DOWN_PRECISION;

    auto ut = OP_API_UT(aclnnAddbmm, INPUT(self, mat1, mat2, beta, alpha), OUTPUT(out), cubeMathType);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// bias cast to float
TEST_F(l2_addbmm_test, ascend910B2_case_bias_cast_to_float)
{
    auto self = TensorDesc({4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat1 = TensorDesc({1, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat2 = TensorDesc({1, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto beta = ScalarDesc(1.0f);
    auto alpha = ScalarDesc(1.0f);
    int8_t cubeMathType = ALLOW_FP32_DOWN_PRECISION;

    auto ut = OP_API_UT(aclnnAddbmm, INPUT(self, mat1, mat2, beta, alpha), OUTPUT(out), cubeMathType);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 直接调用addbmm算子 N维 bias broadcast batch==1
TEST_F(l2_addbmm_test, case_fp16_4mm4_4bias)
{
    auto self = TensorDesc({4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat1 = TensorDesc({1, 4, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat2 = TensorDesc({1, 4, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out = TensorDesc({4, 4}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto beta = ScalarDesc(1.0f);
    auto alpha = ScalarDesc(1.0f);
    int8_t cubeMathType = ALLOW_FP32_DOWN_PRECISION;

    auto ut = OP_API_UT(aclnnAddbmm, INPUT(self, mat1, mat2, beta, alpha), OUTPUT(out), cubeMathType);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_addbmm_test, case_fp16_align)
{
    auto self = TensorDesc({16, 64}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto batch1 = TensorDesc({16, 16, 32}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto batch2 = TensorDesc({16, 32, 64}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out = TensorDesc({16, 64}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto beta = ScalarDesc(1.0f);
    auto alpha = ScalarDesc(1.0f);
    int8_t cubeMathType = ALLOW_FP32_DOWN_PRECISION;

    auto ut = OP_API_UT(aclnnAddbmm, INPUT(self, batch1, batch2, beta, alpha), OUTPUT(out), cubeMathType);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_addbmm_test, case_fp32_not_align)
{
    auto self = TensorDesc({3, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto batch1 = TensorDesc({1, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto batch2 = TensorDesc({1, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out = TensorDesc({3, 5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto beta = ScalarDesc(1.0f);
    auto alpha = ScalarDesc(1.0f);
    int8_t cubeMathType = ALLOW_FP32_DOWN_PRECISION;

    auto ut = OP_API_UT(aclnnAddbmm, INPUT(self, batch1, batch2, beta, alpha), OUTPUT(out), cubeMathType);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_addbmm_test, case_fp32_align)
{
    auto self = TensorDesc({16, 64}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto batch1 = TensorDesc({16, 16, 32}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto batch2 = TensorDesc({16, 32, 64}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out = TensorDesc({16, 64}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto beta = ScalarDesc(1.0f);
    auto alpha = ScalarDesc(1.0f);
    int8_t cubeMathType = ALLOW_FP32_DOWN_PRECISION;

    auto ut = OP_API_UT(aclnnAddbmm, INPUT(self, batch1, batch2, beta, alpha), OUTPUT(out), cubeMathType);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_addbmm_test, case_NCL)
{
    auto self = TensorDesc({3, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto batch1 = TensorDesc({1, 3, 4}, ACL_FLOAT, ACL_FORMAT_NCL).ValueRange(0, 2);
    auto batch2 = TensorDesc({1, 4, 5}, ACL_FLOAT, ACL_FORMAT_NCL).ValueRange(0, 2);
    auto out = TensorDesc({3, 5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto beta = ScalarDesc(1.0f);
    auto alpha = ScalarDesc(1.0f);
    int8_t cubeMathType = ALLOW_FP32_DOWN_PRECISION;

    auto ut = OP_API_UT(aclnnAddbmm, INPUT(self, batch1, batch2, beta, alpha), OUTPUT(out), cubeMathType);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_addbmm_test, case_not_support_dtype)
{
    auto self = TensorDesc({3, 5}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 2);
    auto batch1 = TensorDesc({2, 3, 4}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 2);
    auto batch2 = TensorDesc({2, 4, 5}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out = TensorDesc({3, 5}, ACL_INT32, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto beta = ScalarDesc(1.0f);
    auto alpha = ScalarDesc(1.0f);
    int8_t cubeMathType = ALLOW_FP32_DOWN_PRECISION;

    auto ut = OP_API_UT(aclnnAddbmm, INPUT(self, batch1, batch2, beta, alpha), OUTPUT(out), cubeMathType);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_addbmm_test, case_not_3d)
{
    auto self = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto batch1 = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto batch2 = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto beta = ScalarDesc(2.5f);
    auto alpha = ScalarDesc(5.2f);
    int8_t cubeMathType = ALLOW_FP32_DOWN_PRECISION;

    auto ut = OP_API_UT(aclnnAddbmm, INPUT(self, batch1, batch2, beta, alpha), OUTPUT(out), cubeMathType);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_addbmm_test, case_only_one_empty_tensor)
{
    auto self = TensorDesc({3, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto batch1 = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto batch2 = TensorDesc({2, 4, 0}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out = TensorDesc({3, 5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto beta = ScalarDesc(2.5f);
    auto alpha = ScalarDesc(5.2f);
    int8_t cubeMathType = ALLOW_FP32_DOWN_PRECISION;

    auto ut = OP_API_UT(aclnnAddbmm, INPUT(self, batch1, batch2, beta, alpha), OUTPUT(out), cubeMathType);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_addbmm_test, addbmm_950_case_two_empty_tensor)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto self = TensorDesc({3, 0}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto batch1 = TensorDesc({2, 3, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto batch2 = TensorDesc({2, 5, 0}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out = TensorDesc({3, 0}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto beta = ScalarDesc(1.0f);
    auto alpha = ScalarDesc(1.0f);
    int8_t cubeMathType = KEEP_DTYPE;

    auto ut = OP_API_UT(aclnnAddbmm, INPUT(self, batch1, batch2, beta, alpha), OUTPUT(out), cubeMathType);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_addbmm_test, case_bmm_not_match1)
{
    auto self = TensorDesc({3, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto batch1 = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto batch2 = TensorDesc({5, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out = TensorDesc({3, 5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto beta = ScalarDesc(2.5f);
    auto alpha = ScalarDesc(5.2f);
    int8_t cubeMathType = ALLOW_FP32_DOWN_PRECISION;

    auto ut = OP_API_UT(aclnnAddbmm, INPUT(self, batch1, batch2, beta, alpha), OUTPUT(out), cubeMathType);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_addbmm_test, case_bmm_match1)
{
    auto self = TensorDesc({3, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto batch1 = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto batch2 = TensorDesc({1, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out = TensorDesc({3, 5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto beta = ScalarDesc(2.5f);
    auto alpha = ScalarDesc(5.2f);
    int8_t cubeMathType = ALLOW_FP32_DOWN_PRECISION;

    auto ut = OP_API_UT(aclnnAddbmm, INPUT(self, batch1, batch2, beta, alpha), OUTPUT(out), cubeMathType);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_addbmm_test, case_bmm_not_match2)
{
    auto self = TensorDesc({3, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto batch1 = TensorDesc({2, 4, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto batch2 = TensorDesc({2, 5, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out = TensorDesc({3, 5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto beta = ScalarDesc(2.5f);
    auto alpha = ScalarDesc(5.2f);
    int8_t cubeMathType = ALLOW_FP32_DOWN_PRECISION;

    auto ut = OP_API_UT(aclnnAddbmm, INPUT(self, batch1, batch2, beta, alpha), OUTPUT(out), cubeMathType);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_addbmm_test, case_self_broadcast1)
{
    auto self = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto batch1 = TensorDesc({1, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto batch2 = TensorDesc({1, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out = TensorDesc({3, 5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto beta = ScalarDesc(1.0f);
    auto alpha = ScalarDesc(1.0f);
    int8_t cubeMathType = ALLOW_FP32_DOWN_PRECISION;

    auto ut = OP_API_UT(aclnnAddbmm, INPUT(self, batch1, batch2, beta, alpha), OUTPUT(out), cubeMathType);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_addbmm_test, case_self_broadcast2)
{
    auto self = TensorDesc({1, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto batch1 = TensorDesc({1, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto batch2 = TensorDesc({1, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out = TensorDesc({3, 5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto beta = ScalarDesc(1.0f);
    auto alpha = ScalarDesc(1.0f);
    int8_t cubeMathType = ALLOW_FP32_DOWN_PRECISION;

    auto ut = OP_API_UT(aclnnAddbmm, INPUT(self, batch1, batch2, beta, alpha), OUTPUT(out), cubeMathType);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_addbmm_test, case_self_broadcast3)
{
    auto self = TensorDesc({3, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto batch1 = TensorDesc({1, 1, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto batch2 = TensorDesc({1, 4, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out = TensorDesc({3, 5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto beta = ScalarDesc(1.0f);
    auto alpha = ScalarDesc(1.0f);
    int8_t cubeMathType = ALLOW_FP32_DOWN_PRECISION;

    auto ut = OP_API_UT(aclnnAddbmm, INPUT(self, batch1, batch2, beta, alpha), OUTPUT(out), cubeMathType);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_addbmm_test, case_self_empty_not_broadcast)
{
    auto self = TensorDesc({3, 0}, ACL_FLOAT, ACL_FORMAT_NCL).ValueRange(0, 2);
    auto batch1 = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_NCL).ValueRange(0, 2);
    auto batch2 = TensorDesc({2, 4, 0}, ACL_FLOAT, ACL_FORMAT_NCL).ValueRange(0, 2);
    auto out = TensorDesc({3, 0}, ACL_FLOAT, ACL_FORMAT_NCL).Precision(0.001, 0.001);
    auto beta = ScalarDesc(1.0f);
    auto alpha = ScalarDesc(1.0f);
    int8_t cubeMathType = ALLOW_FP32_DOWN_PRECISION;

    auto ut = OP_API_UT(aclnnAddbmm, INPUT(self, batch1, batch2, beta, alpha), OUTPUT(out), cubeMathType);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_addbmm_test, case_self_empty_broadcast)
{
    auto self = TensorDesc({1, 0}, ACL_FLOAT, ACL_FORMAT_NCL).ValueRange(0, 2);
    auto batch1 = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_NCL).ValueRange(0, 2);
    auto batch2 = TensorDesc({2, 4, 0}, ACL_FLOAT, ACL_FORMAT_NCL).ValueRange(0, 2);
    auto out = TensorDesc({3, 0}, ACL_FLOAT, ACL_FORMAT_NCL).Precision(0.001, 0.001);
    auto beta = ScalarDesc(1.0f);
    auto alpha = ScalarDesc(1.0f);
    int8_t cubeMathType = ALLOW_FP32_DOWN_PRECISION;

    auto ut = OP_API_UT(aclnnAddbmm, INPUT(self, batch1, batch2, beta, alpha), OUTPUT(out), cubeMathType);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_addbmm_test, case_self_empty_cannot_broadcast)
{
    auto self = TensorDesc({2, 0}, ACL_FLOAT, ACL_FORMAT_NCL).ValueRange(0, 2);
    auto batch1 = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_NCL).ValueRange(0, 2);
    auto batch2 = TensorDesc({2, 4, 0}, ACL_FLOAT, ACL_FORMAT_NCL).ValueRange(0, 2);
    auto out = TensorDesc({3, 0}, ACL_FLOAT, ACL_FORMAT_NCL).Precision(0.001, 0.001);
    auto beta = ScalarDesc(1.0f);
    auto alpha = ScalarDesc(1.0f);
    int8_t cubeMathType = ALLOW_FP32_DOWN_PRECISION;

    auto ut = OP_API_UT(aclnnAddbmm, INPUT(self, batch1, batch2, beta, alpha), OUTPUT(out), cubeMathType);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_addbmm_test, case_self_nullptr)
{
    auto self = TensorDesc({3, 5}, ACL_FLOAT, ACL_FORMAT_NCL).ValueRange(0, 2);
    auto batch1 = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_NCL).ValueRange(0, 2);
    auto batch2 = TensorDesc({2, 4, 5}, ACL_FLOAT, ACL_FORMAT_NCL).ValueRange(0, 2);
    auto out = TensorDesc({3, 5}, ACL_FLOAT, ACL_FORMAT_NCL).Precision(0.001, 0.001);
    auto beta = ScalarDesc(1.0f);
    auto alpha = ScalarDesc(1.0f);
    int8_t cubeMathType = ALLOW_FP32_DOWN_PRECISION;

    auto ut = OP_API_UT(aclnnAddbmm, INPUT(nullptr, batch1, batch2, beta, alpha), OUTPUT(out), cubeMathType);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_addbmm_test, case_batch1_nullptr)
{
    auto self = TensorDesc({3, 5}, ACL_FLOAT, ACL_FORMAT_NCL).ValueRange(0, 2);
    auto batch1 = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_NCL).ValueRange(0, 2);
    auto batch2 = TensorDesc({2, 4, 5}, ACL_FLOAT, ACL_FORMAT_NCL).ValueRange(0, 2);
    auto out = TensorDesc({3, 5}, ACL_FLOAT, ACL_FORMAT_NCL).Precision(0.001, 0.001);
    auto beta = ScalarDesc(1.0f);
    auto alpha = ScalarDesc(1.0f);
    int8_t cubeMathType = ALLOW_FP32_DOWN_PRECISION;

    auto ut = OP_API_UT(aclnnAddbmm, INPUT(self, nullptr, batch2, beta, alpha), OUTPUT(out), cubeMathType);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_addbmm_test, case_batch2_nullptr)
{
    auto self = TensorDesc({3, 5}, ACL_FLOAT, ACL_FORMAT_NCL).ValueRange(0, 2);
    auto batch1 = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_NCL).ValueRange(0, 2);
    auto batch2 = TensorDesc({2, 4, 5}, ACL_FLOAT, ACL_FORMAT_NCL).ValueRange(0, 2);
    auto out = TensorDesc({3, 5}, ACL_FLOAT, ACL_FORMAT_NCL).Precision(0.001, 0.001);
    auto beta = ScalarDesc(1.0f);
    auto alpha = ScalarDesc(1.0f);
    int8_t cubeMathType = ALLOW_FP32_DOWN_PRECISION;

    auto ut = OP_API_UT(aclnnAddbmm, INPUT(self, batch1, nullptr, beta, alpha), OUTPUT(out), cubeMathType);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_addbmm_test, case_beta_nullptr)
{
    auto self = TensorDesc({3, 5}, ACL_FLOAT, ACL_FORMAT_NCL).ValueRange(0, 2);
    auto batch1 = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_NCL).ValueRange(0, 2);
    auto batch2 = TensorDesc({2, 4, 5}, ACL_FLOAT, ACL_FORMAT_NCL).ValueRange(0, 2);
    auto out = TensorDesc({3, 5}, ACL_FLOAT, ACL_FORMAT_NCL).Precision(0.001, 0.001);
    auto beta = ScalarDesc(1.0f);
    auto alpha = ScalarDesc(1.0f);
    int8_t cubeMathType = ALLOW_FP32_DOWN_PRECISION;

    auto ut = OP_API_UT(aclnnAddbmm, INPUT(self, batch1, batch2, nullptr, alpha), OUTPUT(out), cubeMathType);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_addbmm_test, case_alpha_nullptr)
{
    auto self = TensorDesc({3, 5}, ACL_FLOAT, ACL_FORMAT_NCL).ValueRange(0, 2);
    auto batch1 = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_NCL).ValueRange(0, 2);
    auto batch2 = TensorDesc({2, 4, 5}, ACL_FLOAT, ACL_FORMAT_NCL).ValueRange(0, 2);
    auto out = TensorDesc({3, 5}, ACL_FLOAT, ACL_FORMAT_NCL).Precision(0.001, 0.001);
    auto beta = ScalarDesc(1.0f);
    auto alpha = ScalarDesc(1.0f);
    int8_t cubeMathType = ALLOW_FP32_DOWN_PRECISION;

    auto ut = OP_API_UT(aclnnAddbmm, INPUT(self, batch1, batch2, beta, nullptr), OUTPUT(out), cubeMathType);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_addbmm_test, case_out_nullptr)
{
    auto self = TensorDesc({3, 5}, ACL_FLOAT, ACL_FORMAT_NCL).ValueRange(0, 2);
    auto batch1 = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_NCL).ValueRange(0, 2);
    auto batch2 = TensorDesc({2, 4, 5}, ACL_FLOAT, ACL_FORMAT_NCL).ValueRange(0, 2);
    auto out = TensorDesc({3, 5}, ACL_FLOAT, ACL_FORMAT_NCL).Precision(0.001, 0.001);
    auto beta = ScalarDesc(1.0f);
    auto alpha = ScalarDesc(1.0f);
    int8_t cubeMathType = ALLOW_FP32_DOWN_PRECISION;

    auto ut = OP_API_UT(aclnnAddbmm, INPUT(self, batch1, batch2, beta, alpha), OUTPUT(nullptr), cubeMathType);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// self非连续，带storage shape
TEST_F(l2_addbmm_test, case_not_contiguous)
{
    auto self = TensorDesc({3, 5}, ACL_FLOAT, ACL_FORMAT_ND, {1, 3}, 0, {5, 3}).ValueRange(0, 2);
    auto batch1 = TensorDesc({1, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto batch2 = TensorDesc({1, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out = TensorDesc({3, 5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.005, 0.005);
    auto beta = ScalarDesc(1.0f);
    auto alpha = ScalarDesc(1.0f);
    int8_t cubeMathType = ALLOW_FP32_DOWN_PRECISION;

    auto ut = OP_API_UT(aclnnAddbmm, INPUT(self, batch1, batch2, beta, alpha), OUTPUT(out), cubeMathType);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// batch1 / batch2非连续，带storage shape
TEST_F(l2_addbmm_test, case_not_contiguous2)
{
    auto self = TensorDesc({3, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto batch1 = TensorDesc({1, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND, {12, 1, 3}, 0, {1, 4, 3}).ValueRange(0, 2);
    auto batch2 = TensorDesc({1, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND, {20, 1, 4}, 0, {1, 5, 4}).ValueRange(0, 2);
    auto out = TensorDesc({3, 5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.005, 0.005);
    auto beta = ScalarDesc(1.0f);
    auto alpha = ScalarDesc(1.0f);
    int8_t cubeMathType = ALLOW_FP32_DOWN_PRECISION;

    auto ut = OP_API_UT(aclnnAddbmm, INPUT(self, batch1, batch2, beta, alpha), OUTPUT(out), cubeMathType);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 非连续，不带storage shape
TEST_F(l2_addbmm_test, case_not_contiguous3)
{
    auto self = TensorDesc({3, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto batch1 = TensorDesc({1, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND, {12, 1, 3}, 0, {12}).ValueRange(0, 2);
    auto batch2 = TensorDesc({1, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND, {20, 1, 4}, 0, {20}).ValueRange(0, 2);
    auto out = TensorDesc({3, 5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.005, 0.005);
    auto beta = ScalarDesc(1.0f);
    auto alpha = ScalarDesc(1.0f);
    int8_t cubeMathType = ALLOW_FP32_DOWN_PRECISION;

    auto ut = OP_API_UT(aclnnAddbmm, INPUT(self, batch1, batch2, beta, alpha), OUTPUT(out), cubeMathType);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 检查原地操作
TEST_F(l2_addbmm_test, case_inplace)
{
    auto self = TensorDesc({3, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto batch1 = TensorDesc({1, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND, {12, 1, 3}, 0, {12}).ValueRange(0, 2);
    auto batch2 = TensorDesc({1, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND, {20, 1, 4}, 0, {20}).ValueRange(0, 2);
    auto beta = ScalarDesc(1.0f);
    auto alpha = ScalarDesc(1.0f);
    int8_t cubeMathType = ALLOW_FP32_DOWN_PRECISION;

    auto ut = OP_API_UT(aclnnInplaceAddbmm, INPUT(self, batch1, batch2, beta, alpha), OUTPUT(), cubeMathType);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// beta is zero and alpha is not zero
TEST_F(l2_addbmm_test, case_beta_0_alpha_not0)
{
    auto self = TensorDesc({3, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto batch1 = TensorDesc({1, 3, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto batch2 = TensorDesc({1, 4, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out = TensorDesc({3, 5}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto beta = ScalarDesc(0.0f);
    auto alpha = ScalarDesc(1.0f);
    int8_t cubeMathType = ALLOW_FP32_DOWN_PRECISION;

    auto ut = OP_API_UT(aclnnAddbmm, INPUT(self, batch1, batch2, beta, alpha), OUTPUT(out), cubeMathType);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// beta is not zero and alpha is zero
TEST_F(l2_addbmm_test, case_beta_not0_alpha_0)
{
    auto self = TensorDesc({3, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto batch1 = TensorDesc({1, 3, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto batch2 = TensorDesc({1, 4, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out = TensorDesc({3, 5}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto beta = ScalarDesc(1.0f);
    auto alpha = ScalarDesc(0.0f);
    int8_t cubeMathType = ALLOW_FP32_DOWN_PRECISION;

    auto ut = OP_API_UT(aclnnAddbmm, INPUT(self, batch1, batch2, beta, alpha), OUTPUT(out), cubeMathType);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// beta is zero and alpha is zero
TEST_F(l2_addbmm_test, case_beta_0_alpha_0)
{
    auto self = TensorDesc({3, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto batch1 = TensorDesc({1, 3, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto batch2 = TensorDesc({1, 4, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out = TensorDesc({3, 5}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto beta = ScalarDesc(0.0f);
    auto alpha = ScalarDesc(0.0f);
    int8_t cubeMathType = ALLOW_FP32_DOWN_PRECISION;

    auto ut = OP_API_UT(aclnnAddbmm, INPUT(self, batch1, batch2, beta, alpha), OUTPUT(out), cubeMathType);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_addbmm_test, case_alpha_is_not_one)
{
    auto self = TensorDesc({3, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto batch1 = TensorDesc({1, 3, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto batch2 = TensorDesc({1, 4, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out = TensorDesc({3, 5}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto beta = ScalarDesc(1.0f);
    auto alpha = ScalarDesc(2.0f);
    int8_t cubeMathType = ALLOW_FP32_DOWN_PRECISION;

    auto ut = OP_API_UT(aclnnAddbmm, INPUT(self, batch1, batch2, beta, alpha), OUTPUT(out), cubeMathType);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_addbmm_test, case_different_input_dtype_1)
{
    auto self = TensorDesc({3, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto batch1 = TensorDesc({1, 3, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto batch2 = TensorDesc({1, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out = TensorDesc({3, 5}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto beta = ScalarDesc(1.0f);
    auto alpha = ScalarDesc(2.0f);
    int8_t cubeMathType = ALLOW_FP32_DOWN_PRECISION;

    auto ut = OP_API_UT(aclnnAddbmm, INPUT(self, batch1, batch2, beta, alpha), OUTPUT(out), cubeMathType);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_addbmm_test, case_different_input_dtype_2)
{
    auto self = TensorDesc({3, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto batch1 = TensorDesc({1, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto batch2 = TensorDesc({1, 4, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out = TensorDesc({3, 5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto beta = ScalarDesc(1.0f);
    auto alpha = ScalarDesc(2.0f);
    int8_t cubeMathType = ALLOW_FP32_DOWN_PRECISION;

    auto ut = OP_API_UT(aclnnAddbmm, INPUT(self, batch1, batch2, beta, alpha), OUTPUT(out), cubeMathType);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_addbmm_test, case_different_input_dtype_3)
{
    auto self = TensorDesc({3, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto batch1 = TensorDesc({1, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto batch2 = TensorDesc({1, 4, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out = TensorDesc({3, 5}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto beta = ScalarDesc(1.0f);
    auto alpha = ScalarDesc(2.0f);
    int8_t cubeMathType = ALLOW_FP32_DOWN_PRECISION;

    auto ut = OP_API_UT(aclnnAddbmm, INPUT(self, batch1, batch2, beta, alpha), OUTPUT(out), cubeMathType);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_addbmm_test, case_different_input_dtype_4)
{
    auto self = TensorDesc({3, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto batch1 = TensorDesc({1, 3, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto batch2 = TensorDesc({1, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out = TensorDesc({3, 5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto beta = ScalarDesc(1.0f);
    auto alpha = ScalarDesc(2.0f);
    int8_t cubeMathType = ALLOW_FP32_DOWN_PRECISION;

    auto ut = OP_API_UT(aclnnAddbmm, INPUT(self, batch1, batch2, beta, alpha), OUTPUT(out), cubeMathType);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 接口整改异常用例 - 950
TEST_F(l2_addbmm_test, addbmm_950_FP32_FP32_FP16FP32_KEEP_DTYPE)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto self = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat1 = TensorDesc({1, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat2 = TensorDesc({1, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto beta = ScalarDesc(1.0f);
    auto alpha = ScalarDesc(1.0f);
    int8_t cubeMathType = FP16FP32_KEEP_DTYPE;

    auto ut = OP_API_UT(aclnnAddbmm, INPUT(self, mat1, mat2, beta, alpha), OUTPUT(out), cubeMathType);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_addbmm_test, addbmm_950_FP32_FP16_FP16FP32_KEEP_DTYPE)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto self = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat1 = TensorDesc({1, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat2 = TensorDesc({1, 4, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto beta = ScalarDesc(1.0f);
    auto alpha = ScalarDesc(1.0f);
    int8_t cubeMathType = FP16FP32_KEEP_DTYPE;

    auto ut = OP_API_UT(aclnnAddbmm, INPUT(self, mat1, mat2, beta, alpha), OUTPUT(out), cubeMathType);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_addbmm_test, addbmm_950_FP32_BF16_FP16FP32_KEEP_DTYPE)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto self = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat1 = TensorDesc({1, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat2 = TensorDesc({1, 4, 4}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto beta = ScalarDesc(1.0f);
    auto alpha = ScalarDesc(1.0f);
    int8_t cubeMathType = FP16FP32_KEEP_DTYPE;

    auto ut = OP_API_UT(aclnnAddbmm, INPUT(self, mat1, mat2, beta, alpha), OUTPUT(out), cubeMathType);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_addbmm_test, addbmm_950_FP16_FP16_FP16FP32_KEEP_DTYPE)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto self = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat1 = TensorDesc({1, 4, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat2 = TensorDesc({1, 4, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto beta = ScalarDesc(1.0f);
    auto alpha = ScalarDesc(1.0f);
    int8_t cubeMathType = FP16FP32_KEEP_DTYPE;

    auto ut = OP_API_UT(aclnnAddbmm, INPUT(self, mat1, mat2, beta, alpha), OUTPUT(out), cubeMathType);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_addbmm_test, addbmm_950_FP16_BF16_FP16FP32_KEEP_DTYPE)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto self = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat1 = TensorDesc({1, 4, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat2 = TensorDesc({1, 4, 4}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto beta = ScalarDesc(1.0f);
    auto alpha = ScalarDesc(1.0f);
    int8_t cubeMathType = FP16FP32_KEEP_DTYPE;

    auto ut = OP_API_UT(aclnnAddbmm, INPUT(self, mat1, mat2, beta, alpha), OUTPUT(out), cubeMathType);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_addbmm_test, addbmm_950_BF16_BF16_FP16FP32_KEEP_DTYPE)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto self = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat1 = TensorDesc({1, 4, 4}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat2 = TensorDesc({1, 4, 4}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto beta = ScalarDesc(1.0f);
    auto alpha = ScalarDesc(1.0f);
    int8_t cubeMathType = FP16FP32_KEEP_DTYPE;

    auto ut = OP_API_UT(aclnnAddbmm, INPUT(self, mat1, mat2, beta, alpha), OUTPUT(out), cubeMathType);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 接口整改异常用例 - 310
TEST_F(l2_addbmm_test, addbmm_310_FP32_FP32_KEEP_DTYPE)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND310);
    auto self = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat1 = TensorDesc({1, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat2 = TensorDesc({1, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto beta = ScalarDesc(1.0f);
    auto alpha = ScalarDesc(1.0f);
    int8_t cubeMathType = KEEP_DTYPE;

    auto ut = OP_API_UT(aclnnAddbmm, INPUT(self, mat1, mat2, beta, alpha), OUTPUT(out), cubeMathType);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_addbmm_test, addbmm_310_FP32_FP16_KEEP_DTYPE)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND310);
    auto self = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat1 = TensorDesc({1, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat2 = TensorDesc({1, 4, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto beta = ScalarDesc(1.0f);
    auto alpha = ScalarDesc(1.0f);
    int8_t cubeMathType = KEEP_DTYPE;

    auto ut = OP_API_UT(aclnnAddbmm, INPUT(self, mat1, mat2, beta, alpha), OUTPUT(out), cubeMathType);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_addbmm_test, addbmm_310_FP32_FP32_USE_HF32)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND310);
    auto self = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat1 = TensorDesc({1, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat2 = TensorDesc({1, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto beta = ScalarDesc(1.0f);
    auto alpha = ScalarDesc(1.0f);
    int8_t cubeMathType = USE_HF32;

    auto ut = OP_API_UT(aclnnAddbmm, INPUT(self, mat1, mat2, beta, alpha), OUTPUT(out), cubeMathType);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_addbmm_test, addbmm_310_FP32_FP16_USE_HF32)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND310);
    auto self = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat1 = TensorDesc({1, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat2 = TensorDesc({1, 4, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto beta = ScalarDesc(1.0f);
    auto alpha = ScalarDesc(1.0f);
    int8_t cubeMathType = USE_HF32;

    auto ut = OP_API_UT(aclnnAddbmm, INPUT(self, mat1, mat2, beta, alpha), OUTPUT(out), cubeMathType);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_addbmm_test, addbmm_310_FP32_FP32_FP16FP32_KEEP_DTYPE)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND310);
    auto self = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat1 = TensorDesc({1, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat2 = TensorDesc({1, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto beta = ScalarDesc(1.0f);
    auto alpha = ScalarDesc(1.0f);
    int8_t cubeMathType = FP16FP32_KEEP_DTYPE;

    auto ut = OP_API_UT(aclnnAddbmm, INPUT(self, mat1, mat2, beta, alpha), OUTPUT(out), cubeMathType);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_addbmm_test, addbmm_310_FP32_FP16_FP16FP32_KEEP_DTYPE)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND310);
    auto self = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat1 = TensorDesc({1, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat2 = TensorDesc({1, 4, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto beta = ScalarDesc(1.0f);
    auto alpha = ScalarDesc(1.0f);
    int8_t cubeMathType = FP16FP32_KEEP_DTYPE;

    auto ut = OP_API_UT(aclnnAddbmm, INPUT(self, mat1, mat2, beta, alpha), OUTPUT(out), cubeMathType);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 原地操作
// 接口整改异常用例 - 950
TEST_F(l2_addbmm_test, addbmm_inplace_950_FP32_FP32_FP16FP32_KEEP_DTYPE)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto self = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat1 = TensorDesc({1, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat2 = TensorDesc({1, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto beta = ScalarDesc(1.0f);
    auto alpha = ScalarDesc(1.0f);
    int8_t cubeMathType = FP16FP32_KEEP_DTYPE;

    auto ut = OP_API_UT(aclnnInplaceAddbmm, INPUT(self, mat1, mat2, beta, alpha), OUTPUT(), cubeMathType);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_addbmm_test, addbmm_inplace_950_FP32_FP16_FP16FP32_KEEP_DTYPE)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto self = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat1 = TensorDesc({1, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat2 = TensorDesc({1, 4, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto beta = ScalarDesc(1.0f);
    auto alpha = ScalarDesc(1.0f);
    int8_t cubeMathType = FP16FP32_KEEP_DTYPE;

    auto ut = OP_API_UT(aclnnInplaceAddbmm, INPUT(self, mat1, mat2, beta, alpha), OUTPUT(), cubeMathType);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_addbmm_test, addbmm_inplace_950_FP32_BF16_FP16FP32_KEEP_DTYPE)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto self = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat1 = TensorDesc({1, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat2 = TensorDesc({1, 4, 4}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto beta = ScalarDesc(1.0f);
    auto alpha = ScalarDesc(1.0f);
    int8_t cubeMathType = FP16FP32_KEEP_DTYPE;

    auto ut = OP_API_UT(aclnnInplaceAddbmm, INPUT(self, mat1, mat2, beta, alpha), OUTPUT(), cubeMathType);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_addbmm_test, addbmm_inplace_950_FP16_FP16_FP16FP32_KEEP_DTYPE)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto self = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat1 = TensorDesc({1, 4, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat2 = TensorDesc({1, 4, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto beta = ScalarDesc(1.0f);
    auto alpha = ScalarDesc(1.0f);
    int8_t cubeMathType = FP16FP32_KEEP_DTYPE;

    auto ut = OP_API_UT(aclnnInplaceAddbmm, INPUT(self, mat1, mat2, beta, alpha), OUTPUT(), cubeMathType);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_addbmm_test, addbmm_inplace_950_FP16_BF16_FP16FP32_KEEP_DTYPE)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto self = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat1 = TensorDesc({1, 4, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat2 = TensorDesc({1, 4, 4}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto beta = ScalarDesc(1.0f);
    auto alpha = ScalarDesc(1.0f);
    int8_t cubeMathType = FP16FP32_KEEP_DTYPE;

    auto ut = OP_API_UT(aclnnInplaceAddbmm, INPUT(self, mat1, mat2, beta, alpha), OUTPUT(), cubeMathType);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_addbmm_test, addbmm_inplace_950_BF16_BF16_FP16FP32_KEEP_DTYPE)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto self = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat1 = TensorDesc({1, 4, 4}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat2 = TensorDesc({1, 4, 4}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto beta = ScalarDesc(1.0f);
    auto alpha = ScalarDesc(1.0f);
    int8_t cubeMathType = FP16FP32_KEEP_DTYPE;

    auto ut = OP_API_UT(aclnnInplaceAddbmm, INPUT(self, mat1, mat2, beta, alpha), OUTPUT(), cubeMathType);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 原地操作
// 接口整改异常用例 - 310
TEST_F(l2_addbmm_test, addbmm_inplace_310_FP32_FP32_KEEP_DTYPE)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND310);
    auto self = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat1 = TensorDesc({1, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat2 = TensorDesc({1, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto beta = ScalarDesc(1.0f);
    auto alpha = ScalarDesc(1.0f);
    int8_t cubeMathType = KEEP_DTYPE;

    auto ut = OP_API_UT(aclnnInplaceAddbmm, INPUT(self, mat1, mat2, beta, alpha), OUTPUT(), cubeMathType);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_addbmm_test, addbmm_inplace_310_FP32_FP16_KEEP_DTYPE)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND310);
    auto self = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat1 = TensorDesc({1, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat2 = TensorDesc({1, 4, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto beta = ScalarDesc(1.0f);
    auto alpha = ScalarDesc(1.0f);
    int8_t cubeMathType = KEEP_DTYPE;

    auto ut = OP_API_UT(aclnnInplaceAddbmm, INPUT(self, mat1, mat2, beta, alpha), OUTPUT(), cubeMathType);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_addbmm_test, addbmm_inplace_310_FP32_FP32_USE_HF32)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND310);
    auto self = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat1 = TensorDesc({1, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat2 = TensorDesc({1, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto beta = ScalarDesc(1.0f);
    auto alpha = ScalarDesc(1.0f);
    int8_t cubeMathType = USE_HF32;

    auto ut = OP_API_UT(aclnnInplaceAddbmm, INPUT(self, mat1, mat2, beta, alpha), OUTPUT(), cubeMathType);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_addbmm_test, addbmm_inplace_310_FP32_FP16_USE_HF32)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND310);
    auto self = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat1 = TensorDesc({1, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat2 = TensorDesc({1, 4, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto beta = ScalarDesc(1.0f);
    auto alpha = ScalarDesc(1.0f);
    int8_t cubeMathType = USE_HF32;

    auto ut = OP_API_UT(aclnnInplaceAddbmm, INPUT(self, mat1, mat2, beta, alpha), OUTPUT(), cubeMathType);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_addbmm_test, addbmm_inplace_310_FP32_FP32_FP16FP32_KEEP_DTYPE)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND310);
    auto self = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat1 = TensorDesc({1, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat2 = TensorDesc({1, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto beta = ScalarDesc(1.0f);
    auto alpha = ScalarDesc(1.0f);
    int8_t cubeMathType = FP16FP32_KEEP_DTYPE;

    auto ut = OP_API_UT(aclnnInplaceAddbmm, INPUT(self, mat1, mat2, beta, alpha), OUTPUT(), cubeMathType);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_addbmm_test, addbmm_inplace_310_FP32_FP16_FP16FP32_KEEP_DTYPE)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND310);
    auto self = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat1 = TensorDesc({1, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat2 = TensorDesc({1, 4, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto beta = ScalarDesc(1.0f);
    auto alpha = ScalarDesc(1.0f);
    int8_t cubeMathType = FP16FP32_KEEP_DTYPE;

    auto ut = OP_API_UT(aclnnInplaceAddbmm, INPUT(self, mat1, mat2, beta, alpha), OUTPUT(), cubeMathType);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}
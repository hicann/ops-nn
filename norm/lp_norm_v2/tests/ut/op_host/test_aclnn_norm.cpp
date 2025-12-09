/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/
#include <array>
#include <vector>
#include "gtest/gtest.h"
#include "../../../op_host/op_api/aclnn_norm.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace std;

class l2_norm_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "l2_norm SetUp" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "l2_norm TearDown" << endl;
    }
};

// p0_float_ND 场景
TEST_F(l2_norm_test, case_3_3_4_p0_float_ND_001)
{
    auto selfDesc = TensorDesc({3, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto resultDesc = TensorDesc({3, 1, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto pDesc = ScalarDesc(0.0f);
    vector<int64_t> inpDims = {1};
    auto dims = IntArrayDesc(inpDims);
    bool keepdim = true;

    auto ut = OP_API_UT(aclnnNorm, INPUT(selfDesc, pDesc, dims, keepdim), OUTPUT(resultDesc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// p1_float_ND 场景
TEST_F(l2_norm_test, case_3_3_4_p1_float_ND_002)
{
    auto selfDesc = TensorDesc({3, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto resultDesc = TensorDesc({3, 1, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto pDesc = ScalarDesc(1.0f);
    vector<int64_t> inpDims = {1};
    auto dims = IntArrayDesc(inpDims);
    bool keepdim = true;

    auto ut = OP_API_UT(aclnnNorm, INPUT(selfDesc, pDesc, dims, keepdim), OUTPUT(resultDesc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// p2_float_ND 场景
TEST_F(l2_norm_test, case_3_3_4_p2_float_ND_003)
{
    auto selfDesc = TensorDesc({3, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto resultDesc = TensorDesc({3, 1, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto pDesc = ScalarDesc(2.0f);
    vector<int64_t> inpDims = {1};
    auto dims = IntArrayDesc(inpDims);
    bool keepdim = true;

    auto ut = OP_API_UT(aclnnNorm, INPUT(selfDesc, pDesc, dims, keepdim), OUTPUT(resultDesc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// p2_float_2dim_ND 场景
TEST_F(l2_norm_test, case_3_3_4_p2_2dim_float_ND_004)
{
    auto selfDesc = TensorDesc({3, 3, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto resultDesc = TensorDesc({3, 1, 1, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto pDesc = ScalarDesc(2.0f);
    vector<int64_t> inpDims = {1, 2};
    auto dims = IntArrayDesc(inpDims);
    bool keepdim = true;

    auto ut = OP_API_UT(aclnnNorm, INPUT(selfDesc, pDesc, dims, keepdim), OUTPUT(resultDesc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// p2_float_3dim_ND 场景
TEST_F(l2_norm_test, case_3_3_4_p2_3dim_float_ND_005)
{
    auto selfDesc = TensorDesc({3, 3, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto resultDesc = TensorDesc({3, 1, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto pDesc = ScalarDesc(2.0f);
    vector<int64_t> inpDims = {1, 2, 3};
    auto dims = IntArrayDesc(inpDims);
    bool keepdim = true;

    auto ut = OP_API_UT(aclnnNorm, INPUT(selfDesc, pDesc, dims, keepdim), OUTPUT(resultDesc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// p2_float_6dim_ND_keepdim-false 场景
TEST_F(l2_norm_test, case_3_3_4_p2_3dim_float_ND_006)
{
    auto selfDesc = TensorDesc({3, 3, 3, 4, 2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto resultDesc = TensorDesc({3, 3, 4, 2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto pDesc = ScalarDesc(2.0f);
    vector<int64_t> inpDims = {1};
    auto dims = IntArrayDesc(inpDims);
    bool keepdim = false;

    auto ut = OP_API_UT(aclnnNorm, INPUT(selfDesc, pDesc, dims, keepdim), OUTPUT(resultDesc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// p2_float_NCHW 场景
TEST_F(l2_norm_test, case_3_3_4_p2_3dim_float_NCHW_007)
{
    auto selfDesc = TensorDesc({3, 3, 4}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto resultDesc = TensorDesc({3, 1, 4}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto pDesc = ScalarDesc(2.0f);
    vector<int64_t> inpDims = {1};
    auto dims = IntArrayDesc(inpDims);
    bool keepdim = true;

    auto ut = OP_API_UT(aclnnNorm, INPUT(selfDesc, pDesc, dims, keepdim), OUTPUT(resultDesc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// p2_float_NHWC 场景
TEST_F(l2_norm_test, case_3_3_4_p2_3dim_float_NHWC_008)
{
    auto selfDesc = TensorDesc({3, 3, 4}, ACL_FLOAT, ACL_FORMAT_NHWC);
    auto resultDesc = TensorDesc({3, 1, 4}, ACL_FLOAT, ACL_FORMAT_NHWC);
    auto pDesc = ScalarDesc(2.0f);
    vector<int64_t> inpDims = {1};
    auto dims = IntArrayDesc(inpDims);
    bool keepdim = true;

    auto ut = OP_API_UT(aclnnNorm, INPUT(selfDesc, pDesc, dims, keepdim), OUTPUT(resultDesc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

}

// p2_float_HWCN 场景
TEST_F(l2_norm_test, case_3_3_4_p2_3dim_float_HWCN_009)
{
    auto selfDesc = TensorDesc({3, 3, 4}, ACL_FLOAT, ACL_FORMAT_HWCN);
    auto resultDesc = TensorDesc({3, 1, 4}, ACL_FLOAT, ACL_FORMAT_HWCN);
    auto pDesc = ScalarDesc(2.0f);
    vector<int64_t> inpDims = {1};
    auto dims = IntArrayDesc(inpDims);
    bool keepdim = true;

    auto ut = OP_API_UT(aclnnNorm, INPUT(selfDesc, pDesc, dims, keepdim), OUTPUT(resultDesc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// p2_float_NDHWC 场景
TEST_F(l2_norm_test, case_3_3_4_p2_3dim_float_NDHWC_010)
{
    auto selfDesc = TensorDesc({3, 3, 4}, ACL_FLOAT, ACL_FORMAT_NDHWC);
    auto resultDesc = TensorDesc({3, 1, 4}, ACL_FLOAT, ACL_FORMAT_NDHWC);
    auto pDesc = ScalarDesc(2.0f);
    vector<int64_t> inpDims = {1};
    auto dims = IntArrayDesc(inpDims);
    bool keepdim = true;

    auto ut = OP_API_UT(aclnnNorm, INPUT(selfDesc, pDesc, dims, keepdim), OUTPUT(resultDesc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// p2_float_NCDHW 场景
TEST_F(l2_norm_test, case_3_3_4_p2_3dim_float_NCDHW_011)
{
    auto selfDesc = TensorDesc({3, 3, 4}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto resultDesc = TensorDesc({3, 1, 4}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto pDesc = ScalarDesc(2.0f);
    vector<int64_t> inpDims = {1};
    auto dims = IntArrayDesc(inpDims);
    bool keepdim = true;

    auto ut = OP_API_UT(aclnnNorm, INPUT(selfDesc, pDesc, dims, keepdim), OUTPUT(resultDesc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// p2_float16_NCDHW 场景
TEST_F(l2_norm_test, case_3_3_4_p2_3dim_float16_NCDHW_013)
{
    auto selfDesc = TensorDesc({3, 3, 4}, ACL_FLOAT16, ACL_FORMAT_NCDHW);
    auto resultDesc = TensorDesc({3, 1, 4}, ACL_FLOAT16, ACL_FORMAT_NCDHW);
    auto pDesc = ScalarDesc(2.0f);
    vector<int64_t> inpDims = {1};
    auto dims = IntArrayDesc(inpDims);
    bool keepdim = true;

    auto ut = OP_API_UT(aclnnNorm, INPUT(selfDesc, pDesc, dims, keepdim), OUTPUT(resultDesc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 0维tensor输入的场景
TEST_F(l2_norm_test, case_3_3_0_4_p2_single_dim_float_ND_014)
{
    auto selfDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);
    auto resultDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);
    auto pDesc = ScalarDesc(2.0f);
    vector<int64_t> inpDims = {0};
    auto dims = IntArrayDesc(inpDims);
    bool keepdim = true;

    auto ut = OP_API_UT(aclnnNorm, INPUT(selfDesc, pDesc, dims, keepdim), OUTPUT(resultDesc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 多维空tensor输入的场景
TEST_F(l2_norm_test, case_3_3_0_4_p2_multi_dim_float_ND_015)
{
    auto selfDesc = TensorDesc({3, 3, 0, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto resultDesc = TensorDesc({3, 1, 0, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto pDesc = ScalarDesc(2.0f);
    vector<int64_t> inpDims = {1};
    auto dims = IntArrayDesc(inpDims);
    bool keepdim = true;

    auto ut = OP_API_UT(aclnnNorm, INPUT(selfDesc, pDesc, dims, keepdim), OUTPUT(resultDesc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// self为空指针的场景
TEST_F(l2_norm_test, case_3_3_4_p2_3dim_float_nullptr_self_ND_016)
{
    auto selfDesc = TensorDesc({3, 3, 2, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto resultDesc = TensorDesc({3, 1, 2, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto pDesc = ScalarDesc(2.0f);
    vector<int64_t> inpDims = {1};
    auto dims = IntArrayDesc(inpDims);
    bool keepdim = true;

    auto ut = OP_API_UT(aclnnNorm, INPUT((aclTensor*)nullptr, pDesc, dims, keepdim), OUTPUT(resultDesc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// p为空指针的场景
TEST_F(l2_norm_test, case_3_3_4_p2_3dim_float_nullptr_p_ND_017)
{
    auto selfDesc = TensorDesc({3, 3, 2, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto resultDesc = TensorDesc({3, 1, 2, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto pDesc = ScalarDesc(2.0f);
    vector<int64_t> inpDims = {1};
    auto dims = IntArrayDesc(inpDims);
    bool keepdim = true;

    auto ut = OP_API_UT(aclnnNorm, INPUT(selfDesc, (aclScalar*)nullptr, dims, keepdim), OUTPUT(resultDesc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// dim为空指针的场景
TEST_F(l2_norm_test, case_3_3_4_p2_3dim_float_nullptr_dim_ND_018)
{
    auto selfDesc = TensorDesc({3, 3, 2, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto resultDesc = TensorDesc({3, 1, 2, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto pDesc = ScalarDesc(2.0f);
    vector<int64_t> inpDims = {1};
    auto dims = IntArrayDesc(inpDims);
    bool keepdim = true;

    auto ut = OP_API_UT(aclnnNorm, INPUT(selfDesc, pDesc, (aclIntArray*)nullptr, keepdim), OUTPUT(resultDesc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// out为空指针的场景
TEST_F(l2_norm_test, case_3_3_4_p2_3dim_float_nullptr_out_ND_019)
{
    auto selfDesc = TensorDesc({3, 3, 2, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto resultDesc = TensorDesc({3, 1, 2, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto pDesc = ScalarDesc(2.0f);
    vector<int64_t> inpDims = {1};
    auto dims = IntArrayDesc(inpDims);
    bool keepdim = true;

    auto ut = OP_API_UT(aclnnNorm, INPUT(selfDesc, pDesc, dims, keepdim), OUTPUT((aclTensor*)nullptr));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// self取值范围是(-1, 1)的场景
TEST_F(l2_norm_test, case_3_3_4_p2_3dim_float_nullptr_ND_020)
{
    auto selfDesc = TensorDesc({3, 3, 2, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto resultDesc = TensorDesc({3, 1, 2, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto pDesc = ScalarDesc(2.0f);
    vector<int64_t> inpDims = {1};
    auto dims = IntArrayDesc(inpDims);
    bool keepdim = true;

    auto ut = OP_API_UT(aclnnNorm, INPUT(selfDesc, pDesc, dims, keepdim), OUTPUT(resultDesc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// p2_float_8dim的场景
TEST_F(l2_norm_test, case_3_3_4_p2_8dim_float_ND_021)
{
    auto selfDesc = TensorDesc({3, 3, 2, 4, 2, 3, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto resultDesc = TensorDesc({3, 1, 2, 4, 2, 3, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto pDesc = ScalarDesc(2.0f);
    vector<int64_t> inpDims = {1};
    auto dims = IntArrayDesc(inpDims);
    bool keepdim = true;

    auto ut = OP_API_UT(aclnnNorm, INPUT(selfDesc, pDesc, dims, keepdim), OUTPUT(resultDesc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// p2_float_9dim的异常场景
TEST_F(l2_norm_test, case_3_3_4_p2_9dim_float_ND_022)
{
    auto selfDesc = TensorDesc({3, 3, 3, 4, 2, 3, 3, 4, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto resultDesc = TensorDesc({3, 1, 3, 4, 2, 3, 3, 4, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto pDesc = ScalarDesc(2.0f);
    vector<int64_t> inpDims = {1};
    auto dims = IntArrayDesc(inpDims);
    bool keepdim = true;

    auto ut = OP_API_UT(aclnnNorm, INPUT(selfDesc, pDesc, dims, keepdim), OUTPUT(resultDesc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// self和out的dtype不一致的场景
TEST_F(l2_norm_test, case_3_3_4_p2_diff_type_float_ND_023)
{
    auto selfDesc = TensorDesc({3, 3, 2, 4, 2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto resultDesc = TensorDesc({3, 1, 2, 4, 2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto pDesc = ScalarDesc(2.0f);
    vector<int64_t> inpDims = {1};
    auto dims = IntArrayDesc(inpDims);
    bool keepdim = true;

    auto ut = OP_API_UT(aclnnNorm, INPUT(selfDesc, pDesc, dims, keepdim), OUTPUT(resultDesc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// p2_double_ND的场景
TEST_F(l2_norm_test, case_3_3_2_4_p2_4dim_double_ND_24)
{
    auto selfDesc = TensorDesc({3, 3, 2, 4}, ACL_DOUBLE, ACL_FORMAT_ND);
    auto resultDesc = TensorDesc({3, 1, 2, 4}, ACL_DOUBLE, ACL_FORMAT_ND);
    auto pDesc = ScalarDesc(2.0f);
    vector<int64_t> inpDims = {1};
    auto dims = IntArrayDesc(inpDims);
    bool keepdim = true;

    auto ut = OP_API_UT(aclnnNorm, INPUT(selfDesc, pDesc, dims, keepdim), OUTPUT(resultDesc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// p2_complex64_ND的场景
TEST_F(l2_norm_test, case_3_3_2_4_p2_4dim_complex64_ND_025)
{
    auto selfDesc = TensorDesc({3, 3, 2, 4}, ACL_COMPLEX64, ACL_FORMAT_ND);
    auto resultDesc = TensorDesc({3, 1, 2, 4}, ACL_COMPLEX64, ACL_FORMAT_ND);
    auto pDesc = ScalarDesc(2.0f);
    vector<int64_t> inpDims = {1};
    auto dims = IntArrayDesc(inpDims);
    bool keepdim = true;

    auto ut = OP_API_UT(aclnnNorm, INPUT(selfDesc, pDesc, dims, keepdim), OUTPUT(resultDesc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// p2_complex128_ND的场景
TEST_F(l2_norm_test, case_3_3_2_4_p2_4dim_complex128_ND_026)
{
    auto selfDesc = TensorDesc({3, 3, 2, 4}, ACL_COMPLEX128, ACL_FORMAT_ND);
    auto resultDesc = TensorDesc({3, 1, 2, 4}, ACL_COMPLEX128, ACL_FORMAT_ND);
    auto pDesc = ScalarDesc(2.0f);
    vector<int64_t> inpDims = {1};
    auto dims = IntArrayDesc(inpDims);
    bool keepdim = true;

    auto ut = OP_API_UT(aclnnNorm, INPUT(selfDesc, pDesc, dims, keepdim), OUTPUT(resultDesc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// p2_bfloat16_ND 场景(1980-异常场景)
// TEST_F(l2_norm_test, case_3_3_4_p2_3dim_bfloat16_910_ND_027)
// {
//     auto selfDesc = TensorDesc({3, 3, 4}, ACL_BF16, ACL_FORMAT_ND);
//     auto resultDesc = TensorDesc({3, 1, 4}, ACL_BF16, ACL_FORMAT_ND);
//     auto pDesc = ScalarDesc(2.0f);
//     vector<int64_t> inpDims = {1};
//     auto dims = IntArrayDesc(inpDims);
//     bool keepdim = true;

//     auto ut = OP_API_UT(aclnnNorm, INPUT(selfDesc, pDesc, dims, keepdim), OUTPUT(resultDesc));
//     uint64_t workspace_size = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
//     EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
// }

// p2_bfloat16_ND 场景（1971-正常场景）
TEST_F(l2_norm_test, ascend910B2_case_3_3_4_p2_3dim_bfloat16_ND_028)
{
    auto selfDesc = TensorDesc({3, 3, 4}, ACL_BF16, ACL_FORMAT_ND);
    auto resultDesc = TensorDesc({3, 1, 4}, ACL_BF16, ACL_FORMAT_ND);
    auto pDesc = ScalarDesc(2.0f);
    vector<int64_t> inpDims = {1};
    auto dims = IntArrayDesc(inpDims);
    bool keepdim = true;

    auto ut = OP_API_UT(aclnnNorm, INPUT(selfDesc, pDesc, dims, keepdim), OUTPUT(resultDesc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_norm_test, case_diff_type_float)
{
    auto selfDesc = TensorDesc({3, 3, 2, 4}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto resultDesc = TensorDesc({3, 1, 2, 4}, ACL_DOUBLE, ACL_FORMAT_ND);
    auto pDesc = ScalarDesc(2.0f);
    vector<int64_t> inpDims = {1};
    auto dims = IntArrayDesc(inpDims);
    bool keepdim = true;

    auto ut = OP_API_UT(aclnnNorm, INPUT(selfDesc, pDesc, dims, keepdim), OUTPUT(resultDesc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_norm_test, case_diff_type_double)
{
    auto selfDesc = TensorDesc({3, 3, 2, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto resultDesc = TensorDesc({3, 1, 2, 4}, ACL_DOUBLE, ACL_FORMAT_ND);
    auto pDesc = ScalarDesc(2.0f);
    vector<int64_t> inpDims = {1};
    auto dims = IntArrayDesc(inpDims);
    bool keepdim = true;

    auto ut = OP_API_UT(aclnnNorm, INPUT(selfDesc, pDesc, dims, keepdim), OUTPUT(resultDesc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_norm_test, case_diff_type_fp16)
{
    auto selfDesc = TensorDesc({3, 3, 2, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto resultDesc = TensorDesc({3, 1, 2, 4}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto pDesc = ScalarDesc(2.0f);
    vector<int64_t> inpDims = {1};
    auto dims = IntArrayDesc(inpDims);
    bool keepdim = true;

    auto ut = OP_API_UT(aclnnNorm, INPUT(selfDesc, pDesc, dims, keepdim), OUTPUT(resultDesc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}


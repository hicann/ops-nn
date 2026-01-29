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

#include "../../../op_host/op_api/aclnn_addmm.h"
#include "op_api/op_api_def.h"
#include "opdev/platform.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace std;
using namespace op;

namespace {
class l2_addmmWeightNz_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << "l2_addmm_weight_nz_test SetUp" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "l2_addmm_weight_nz_test TearDown" << endl;
    }
};

// // 正常流程16*16 self dims == 1 dtype == float
// TEST_F(l2_addmmWeightNz_test, ascend910B2_case_self_fp32_16mm16)
// {
//     auto self = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
//     auto mat1 = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
//     auto mat2 = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_FRACTAL_NZ, {}, 0, {1, 1, 16, 16});
//     auto out = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);
//     auto beta = ScalarDesc(1.0f);
//     auto alpha = ScalarDesc(1.0f);
//     int8_t cubeMathType = ALLOW_FP32_DOWN_PRECISION;

//     auto ut = OP_API_UT(aclnnAddmmWeightNz, INPUT(self, mat1, mat2, beta, alpha), OUTPUT(out), cubeMathType);

    // uint64_t workspace_size = 0;
    // aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    // EXPECT_EQ(aclRet, ACL_SUCCESS);
// }

// // 正常流程16*16 self dims == 1 dtype == fp16
// TEST_F(l2_addmmWeightNz_test, ascend910B2_case_self_fp16_16mm16)
// {
//     auto self = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
//     auto mat1 = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
//     auto mat2 = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_FRACTAL_NZ, {}, 0, {1, 1, 16, 16});
//     auto out = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);
//     auto beta = ScalarDesc(1.0f);
//     auto alpha = ScalarDesc(1.0f);
//     int8_t cubeMathType = ALLOW_FP32_DOWN_PRECISION;

//     auto ut = OP_API_UT(aclnnAddmmWeightNz, INPUT(self, mat1, mat2, beta, alpha), OUTPUT(out), cubeMathType);

    // uint64_t workspace_size = 0;
    // aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    // EXPECT_EQ(aclRet, ACL_SUCCESS);
// }

// // 正常流程16*16 self dims == 2  dtype == fp16
// TEST_F(l2_addmmWeightNz_test, ascend910B2_case_fp16_16mm16_self_dims2)
// {
//     auto self = TensorDesc({1, 16}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
//     auto mat1 = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
//     auto mat2 = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_FRACTAL_NZ, {}, 0, {1, 1, 16, 16});
//     auto out = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);
//     auto beta = ScalarDesc(1.0f);
//     auto alpha = ScalarDesc(1.0f);
//     int8_t cubeMathType = ALLOW_FP32_DOWN_PRECISION;

//     auto ut = OP_API_UT(aclnnAddmmWeightNz, INPUT(self, mat1, mat2, beta, alpha), OUTPUT(out), cubeMathType);

    // uint64_t workspace_size = 0;
    // aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    // EXPECT_EQ(aclRet, ACL_SUCCESS);
// }

// // 正常流程16*16 self dtype == fp16  mat dtype == bf16
// TEST_F(l2_addmmWeightNz_test, ascend910B2_case_bf16_16mm16_self_dims2)
// {
//     auto self = TensorDesc({1, 16}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
//     auto mat1 = TensorDesc({16, 16}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 2);
//     auto mat2 = TensorDesc({16, 16}, ACL_BF16, ACL_FORMAT_FRACTAL_NZ, {}, 0, {1, 1, 16, 16});
//     auto out = TensorDesc({16, 16}, ACL_BF16, ACL_FORMAT_ND).Precision(0.001, 0.001);
//     auto beta = ScalarDesc(1.0f);
//     auto alpha = ScalarDesc(1.0f);
//     int8_t cubeMathType = ALLOW_FP32_DOWN_PRECISION;

//     auto ut = OP_API_UT(aclnnAddmmWeightNz, INPUT(self, mat1, mat2, beta, alpha), OUTPUT(out), cubeMathType);

    // uint64_t workspace_size = 0;
    // aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    // EXPECT_EQ(aclRet, ACL_SUCCESS);
// }

// // 正常流程16*16
// TEST_F(l2_addmmWeightNz_test, ascend910B2_case_bf16_16mm16)
// {
//     auto self = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
//     auto mat1 = TensorDesc({16, 16}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 2);
//     auto mat2 = TensorDesc({16, 16}, ACL_BF16, ACL_FORMAT_FRACTAL_NZ, {}, 0, {1, 1, 16, 16});
//     auto out = TensorDesc({16, 16}, ACL_BF16, ACL_FORMAT_ND).Precision(0.001, 0.001);
//     auto beta = ScalarDesc(1.0f);
//     auto alpha = ScalarDesc(1.0f);
//     int8_t cubeMathType = ALLOW_FP32_DOWN_PRECISION;

//     auto ut = OP_API_UT(aclnnAddmmWeightNz, INPUT(self, mat1, mat2, beta, alpha), OUTPUT(out), cubeMathType);

    // uint64_t workspace_size = 0;
    // aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    // EXPECT_EQ(aclRet, ACL_SUCCESS);
// }

// 异常，计算weightNzshape有误
TEST_F(l2_addmmWeightNz_test, ascend910B2_case_error_bf16_shapecal_err)
{
    auto self = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat1 = TensorDesc({4, 4}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat2 = TensorDesc({4, 4}, ACL_BF16, ACL_FORMAT_FRACTAL_NZ, {}, 0, {1, 1, 4, 4});
    auto out = TensorDesc({4, 4}, ACL_BF16, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto beta = ScalarDesc(1.0f);
    auto alpha = ScalarDesc(1.0f);
    int8_t cubeMathType = ALLOW_FP32_DOWN_PRECISION;

    auto ut = OP_API_UT(aclnnAddmmWeightNz, INPUT(self, mat1, mat2, beta, alpha), OUTPUT(out), cubeMathType);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_NE(aclRet, ACL_SUCCESS);
}

// 异常，K轴不一致
TEST_F(l2_addmmWeightNz_test, ascend910B2_case_error_bf16_16mm16_Kdiff)
{
    auto self = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat1 = TensorDesc({16, 16}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat2 = TensorDesc({18, 16}, ACL_BF16, ACL_FORMAT_FRACTAL_NZ, {}, 0, {1, 1, 16, 16});
    auto out = TensorDesc({16, 16}, ACL_BF16, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto beta = ScalarDesc(1.0f);
    auto alpha = ScalarDesc(1.0f);
    int8_t cubeMathType = ALLOW_FP32_DOWN_PRECISION;

    auto ut = OP_API_UT(aclnnAddmmWeightNz, INPUT(self, mat1, mat2, beta, alpha), OUTPUT(out), cubeMathType);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_NE(aclRet, ACL_SUCCESS);
}

// 异常：mat1, mat2为float
//TEST_F(l2_addmmWeightNz_test, ascend910B2_case_error_mat1_mat2_not_fp16bf16_1)
//{
//    auto self = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
//    auto mat1 = TensorDesc({16, 16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
//    auto mat2 = TensorDesc({16, 16}, ACL_FLOAT, ACL_FORMAT_FRACTAL_NZ, {}, 0, {1, 1, 16, 16});
//    auto out = TensorDesc({16, 16}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
//    auto beta = ScalarDesc(1.0f);
//    auto alpha = ScalarDesc(1.0f);
//    int8_t cubeMathType = ALLOW_FP32_DOWN_PRECISION;
//
//    auto ut = OP_API_UT(aclnnAddmmWeightNz, INPUT(self, mat1, mat2, beta, alpha), OUTPUT(out), cubeMathType);
//
//    uint64_t workspace_size = 0;
//    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
//    EXPECT_NE(aclRet, ACL_SUCCESS);
//}

//TEST_F(l2_addmmWeightNz_test, ascend910B2_case_error_mat1_mat2_not_fp16bf16_2)
//{
//    auto self = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
//    auto mat1 = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
//    auto mat2 = TensorDesc({16, 16}, ACL_FLOAT, ACL_FORMAT_FRACTAL_NZ, {}, 0, {1, 1, 16, 16});
//    auto out = TensorDesc({16, 16}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
//    auto beta = ScalarDesc(1.0f);
//    auto alpha = ScalarDesc(1.0f);
//    int8_t cubeMathType = ALLOW_FP32_DOWN_PRECISION;
//
//    auto ut = OP_API_UT(aclnnAddmmWeightNz, INPUT(self, mat1, mat2, beta, alpha), OUTPUT(out), cubeMathType);
//
//    uint64_t workspace_size = 0;
//    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
//    EXPECT_NE(aclRet, ACL_SUCCESS);
//}

// 正常：self为(1, n)且为float类型
TEST_F(l2_addmmWeightNz_test, ascend910B2_case_fp16_16mm16_self_float_dims2)
{
    auto self = TensorDesc({1, 16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat1 = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat2 = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_FRACTAL_NZ, {}, 0, {1, 1, 16, 16});
    auto out = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto beta = ScalarDesc(1.0f);
    auto alpha = ScalarDesc(1.0f);
    int8_t cubeMathType = ALLOW_FP32_DOWN_PRECISION;

    auto ut = OP_API_UT(aclnnAddmmWeightNz, INPUT(self, mat1, mat2, beta, alpha), OUTPUT(out), cubeMathType);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_GE(aclRet, ACL_SUCCESS);
}

// // 正常：self为(m,n)
// TEST_F(l2_addmmWeightNz_test, ascend910B2_case_self_dim_ne_1)
// {
//     auto self = TensorDesc({16, 16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
//     auto mat1 = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
//     auto mat2 = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_FRACTAL_NZ, {}, 0, {1, 1, 16, 16});
//     auto out = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);
//     auto beta = ScalarDesc(1.0f);
//     auto alpha = ScalarDesc(1.0f);
//     int8_t cubeMathType = ALLOW_FP32_DOWN_PRECISION;

//     auto ut = OP_API_UT(aclnnAddmmWeightNz, INPUT(self, mat1, mat2, beta, alpha), OUTPUT(out), cubeMathType);

    // uint64_t workspace_size = 0;
    // aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    // EXPECT_EQ(aclRet, ACL_SUCCESS);
// }

// 异常：mat2为NZ，但是shape为2维
TEST_F(l2_addmmWeightNz_test, ascend910B2_case_error_mat2_2dim)
{
    auto self = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat1 = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat2 = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_FRACTAL_NZ, {}, 0, {16, 16});
    auto out = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto beta = ScalarDesc(1.0f);
    auto alpha = ScalarDesc(1.0f);
    int8_t cubeMathType = ALLOW_FP32_DOWN_PRECISION;

    auto ut = OP_API_UT(aclnnAddmmWeightNz, INPUT(self, mat1, mat2, beta, alpha), OUTPUT(out), cubeMathType);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_NE(aclRet, ACL_SUCCESS);
}

// 异常：mat2为NZ，但是shape为1维
TEST_F(l2_addmmWeightNz_test, ascend910B2_case_error_mat2_1dim)
{
    auto self = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat1 = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat2 = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_FRACTAL_NZ, {}, 0, {16});
    auto out = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto beta = ScalarDesc(1.0f);
    auto alpha = ScalarDesc(1.0f);
    int8_t cubeMathType = ALLOW_FP32_DOWN_PRECISION;

    auto ut = OP_API_UT(aclnnAddmmWeightNz, INPUT(self, mat1, mat2, beta, alpha), OUTPUT(out), cubeMathType);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_NE(aclRet, ACL_SUCCESS);
}

// 异常：mat1 mat2 dtype不一致
//TEST_F(l2_addmmWeightNz_test, ascend910B2_case_error_mat1bf16_mat2fp16)
//{
//    auto self = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
//    auto mat1 = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
//    auto mat2 = TensorDesc({16, 16}, ACL_BF16, ACL_FORMAT_FRACTAL_NZ, {}, 0, {1, 1, 16, 16});
//    auto out = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);
//    auto beta = ScalarDesc(1.0f);
//    auto alpha = ScalarDesc(1.0f);
//    int8_t cubeMathType = ALLOW_FP32_DOWN_PRECISION;
//
//    auto ut = OP_API_UT(aclnnAddmmWeightNz, INPUT(self, mat1, mat2, beta, alpha), OUTPUT(out), cubeMathType);
//
//    uint64_t workspace_size = 0;
//    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
//    EXPECT_NE(aclRet, ACL_SUCCESS);
//}

// 异常：mat1 mat2 dtype不一致
//TEST_F(l2_addmmWeightNz_test, ascend910B2_case_error_mat2bf16_mat1fp16)
//{
//    auto self = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
//    auto mat1 = TensorDesc({16, 16}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 2);
//    auto mat2 = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_FRACTAL_NZ, {}, 0, {1, 1, 16, 16});
//    auto out = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);
//    auto beta = ScalarDesc(1.0f);
//    auto alpha = ScalarDesc(1.0f);
//    int8_t cubeMathType = ALLOW_FP32_DOWN_PRECISION;
//
//    auto ut = OP_API_UT(aclnnAddmmWeightNz, INPUT(self, mat1, mat2, beta, alpha), OUTPUT(out), cubeMathType);
//
//    uint64_t workspace_size = 0;
//    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
//    EXPECT_NE(aclRet, ACL_SUCCESS);
//}

// // 正常：beta alpha 不是缺省值
// TEST_F(l2_addmmWeightNz_test, ascend910B2_case_beta_ne_1)
// {
//     auto self = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
//     auto mat1 = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
//     auto mat2 = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_FRACTAL_NZ, {}, 0, {1, 1, 16, 16});
//     auto out = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);
//     auto beta = ScalarDesc(0.0f);
//     auto alpha = ScalarDesc(1.0f);
//     int8_t cubeMathType = ALLOW_FP32_DOWN_PRECISION;

//     auto ut = OP_API_UT(aclnnAddmmWeightNz, INPUT(self, mat1, mat2, beta, alpha), OUTPUT(out), cubeMathType);

    // uint64_t workspace_size = 0;
    // aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    // EXPECT_EQ(aclRet, ACL_SUCCESS);
// }

// TEST_F(l2_addmmWeightNz_test, ascend910B2_case_alpha_ne_1)
// {
//     auto self = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
//     auto mat1 = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
//     auto mat2 = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_FRACTAL_NZ, {}, 0, {1, 1, 16, 16});
//     auto out = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);
//     auto beta = ScalarDesc(1.0f);
//     auto alpha = ScalarDesc(0.0f);
//     int8_t cubeMathType = ALLOW_FP32_DOWN_PRECISION;

//     auto ut = OP_API_UT(aclnnAddmmWeightNz, INPUT(self, mat1, mat2, beta, alpha), OUTPUT(out), cubeMathType);

    // uint64_t workspace_size = 0;
    // aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    // EXPECT_EQ(aclRet, ACL_SUCCESS);
// }

// 异常：self与mat1为NZ格式
TEST_F(l2_addmmWeightNz_test, ascend910B2_case_error_self_mat1_dtype_nz)
{
    auto self = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_FRACTAL_NZ).ValueRange(0, 2);
    auto mat1 = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_FRACTAL_NZ).ValueRange(0, 2);
    auto mat2 = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_FRACTAL_NZ, {}, 0, {1, 1, 16, 16});
    auto out = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto beta = ScalarDesc(1.0f);
    auto alpha = ScalarDesc(1.0f);
    int8_t cubeMathType = ALLOW_FP32_DOWN_PRECISION;

    auto ut = OP_API_UT(aclnnAddmmWeightNz, INPUT(self, mat1, mat2, beta, alpha), OUTPUT(out), cubeMathType);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_NE(aclRet, ACL_SUCCESS);
}

// 接口整改异常用例 - 950
TEST_F(l2_addmmWeightNz_test, addmm_NZ_950_FP32_FP32_FP16FP32_KEEP_DTYPE)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto self = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat1 = TensorDesc({16, 16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat2 = TensorDesc({16, 16}, ACL_FLOAT, ACL_FORMAT_FRACTAL_NZ, {}, 0, {1, 1, 16, 16});
    auto out = TensorDesc({16, 16}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto beta = ScalarDesc(1.0f);
    auto alpha = ScalarDesc(1.0f);
    int8_t cubeMathType = FP16FP32_KEEP_DTYPE;

    auto ut = OP_API_UT(aclnnAddmmWeightNz, INPUT(self, mat1, mat2, beta, alpha), OUTPUT(out), cubeMathType);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_addmmWeightNz_test, addmm_NZ_950_FP32_FP16_FP16FP32_KEEP_DTYPE)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto self = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat1 = TensorDesc({16, 16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat2 = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_FRACTAL_NZ, {}, 0, {1, 1, 16, 16});
    auto out = TensorDesc({16, 16}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto beta = ScalarDesc(1.0f);
    auto alpha = ScalarDesc(1.0f);
    int8_t cubeMathType = FP16FP32_KEEP_DTYPE;

    auto ut = OP_API_UT(aclnnAddmmWeightNz, INPUT(self, mat1, mat2, beta, alpha), OUTPUT(out), cubeMathType);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_addmmWeightNz_test, addmm_NZ_950_FP32_BF16_FP16FP32_KEEP_DTYPE)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto self = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat1 = TensorDesc({16, 16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat2 = TensorDesc({16, 16}, ACL_BF16, ACL_FORMAT_FRACTAL_NZ, {}, 0, {1, 1, 16, 16});
    auto out = TensorDesc({16, 16}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto beta = ScalarDesc(1.0f);
    auto alpha = ScalarDesc(1.0f);
    int8_t cubeMathType = FP16FP32_KEEP_DTYPE;

    auto ut = OP_API_UT(aclnnAddmmWeightNz, INPUT(self, mat1, mat2, beta, alpha), OUTPUT(out), cubeMathType);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_addmmWeightNz_test, addmm_NZ_950_FP16_FP16_FP16FP32_KEEP_DTYPE)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto self = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat1 = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat2 = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_FRACTAL_NZ, {}, 0, {1, 1, 16, 16});
    auto out = TensorDesc({16, 16}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto beta = ScalarDesc(1.0f);
    auto alpha = ScalarDesc(1.0f);
    int8_t cubeMathType = FP16FP32_KEEP_DTYPE;

    auto ut = OP_API_UT(aclnnAddmmWeightNz, INPUT(self, mat1, mat2, beta, alpha), OUTPUT(out), cubeMathType);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_addmmWeightNz_test, addmm_NZ_950_FP16_BF16_FP16FP32_KEEP_DTYPE)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto self = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat1 = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat2 = TensorDesc({16, 16}, ACL_BF16, ACL_FORMAT_FRACTAL_NZ, {}, 0, {1, 1, 16, 16});
    auto out = TensorDesc({16, 16}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto beta = ScalarDesc(1.0f);
    auto alpha = ScalarDesc(1.0f);
    int8_t cubeMathType = FP16FP32_KEEP_DTYPE;

    auto ut = OP_API_UT(aclnnAddmmWeightNz, INPUT(self, mat1, mat2, beta, alpha), OUTPUT(out), cubeMathType);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_addmmWeightNz_test, addmm_NZ_950_BF16_BF16_FP16FP32_KEEP_DTYPE)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto self = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat1 = TensorDesc({16, 16}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat2 = TensorDesc({16, 16}, ACL_BF16, ACL_FORMAT_FRACTAL_NZ, {}, 0, {1, 1, 16, 16});
    auto out = TensorDesc({16, 16}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto beta = ScalarDesc(1.0f);
    auto alpha = ScalarDesc(1.0f);
    int8_t cubeMathType = FP16FP32_KEEP_DTYPE;

    auto ut = OP_API_UT(aclnnAddmmWeightNz, INPUT(self, mat1, mat2, beta, alpha), OUTPUT(out), cubeMathType);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 接口整改异常用例 - 310
TEST_F(l2_addmmWeightNz_test, addmm_NZ_310_FP32_FP32_KEEP_DTYPE)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND310);
    auto self = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat1 = TensorDesc({16, 16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat2 = TensorDesc({16, 16}, ACL_FLOAT, ACL_FORMAT_FRACTAL_NZ, {}, 0, {1, 1, 16, 16});
    auto out = TensorDesc({16, 16}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto beta = ScalarDesc(1.0f);
    auto alpha = ScalarDesc(1.0f);
    int8_t cubeMathType = KEEP_DTYPE;

    auto ut = OP_API_UT(aclnnAddmmWeightNz, INPUT(self, mat1, mat2, beta, alpha), OUTPUT(out), cubeMathType);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_addmmWeightNz_test, addmm_NZ_310_FP32_FP16_KEEP_DTYPE)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND310);
    auto self = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat1 = TensorDesc({16, 16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat2 = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_FRACTAL_NZ, {}, 0, {1, 1, 16, 16});
    auto out = TensorDesc({16, 16}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto beta = ScalarDesc(1.0f);
    auto alpha = ScalarDesc(1.0f);
    int8_t cubeMathType = KEEP_DTYPE;

    auto ut = OP_API_UT(aclnnAddmmWeightNz, INPUT(self, mat1, mat2, beta, alpha), OUTPUT(out), cubeMathType);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_addmmWeightNz_test, addmm_NZ_310_FP32_FP32_USE_HF32)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND310);
    auto self = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat1 = TensorDesc({16, 16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat2 = TensorDesc({16, 16}, ACL_FLOAT, ACL_FORMAT_FRACTAL_NZ, {}, 0, {1, 1, 16, 16});
    auto out = TensorDesc({16, 16}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto beta = ScalarDesc(1.0f);
    auto alpha = ScalarDesc(1.0f);
    int8_t cubeMathType = USE_HF32;

    auto ut = OP_API_UT(aclnnAddmmWeightNz, INPUT(self, mat1, mat2, beta, alpha), OUTPUT(out), cubeMathType);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_addmmWeightNz_test, addmm_NZ_310_FP32_FP16_USE_HF32)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND310);
    auto self = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat1 = TensorDesc({16, 16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat2 = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_FRACTAL_NZ, {}, 0, {1, 1, 16, 16});
    auto out = TensorDesc({16, 16}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto beta = ScalarDesc(1.0f);
    auto alpha = ScalarDesc(1.0f);
    int8_t cubeMathType = USE_HF32;

    auto ut = OP_API_UT(aclnnAddmmWeightNz, INPUT(self, mat1, mat2, beta, alpha), OUTPUT(out), cubeMathType);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_addmmWeightNz_test, addmm_NZ_310_FP32_FP32_FP16FP32_KEEP_DTYPE)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND310);
    auto self = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat1 = TensorDesc({16, 16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat2 = TensorDesc({16, 16}, ACL_FLOAT, ACL_FORMAT_FRACTAL_NZ, {}, 0, {1, 1, 16, 16});
    auto out = TensorDesc({16, 16}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto beta = ScalarDesc(1.0f);
    auto alpha = ScalarDesc(1.0f);
    int8_t cubeMathType = FP16FP32_KEEP_DTYPE;

    auto ut = OP_API_UT(aclnnAddmmWeightNz, INPUT(self, mat1, mat2, beta, alpha), OUTPUT(out), cubeMathType);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_addmmWeightNz_test, addmm_NZ_310_FP32_FP16_FP16FP32_KEEP_DTYPE)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND310);
    auto self = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat1 = TensorDesc({16, 16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat2 = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_FRACTAL_NZ, {}, 0, {1, 1, 16, 16});
    auto out = TensorDesc({16, 16}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto beta = ScalarDesc(1.0f);
    auto alpha = ScalarDesc(1.0f);
    int8_t cubeMathType = FP16FP32_KEEP_DTYPE;

    auto ut = OP_API_UT(aclnnAddmmWeightNz, INPUT(self, mat1, mat2, beta, alpha), OUTPUT(out), cubeMathType);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}
}
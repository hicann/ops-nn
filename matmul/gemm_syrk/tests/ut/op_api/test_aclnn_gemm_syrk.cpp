/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <iostream>
#include "gtest/gtest.h"

#include "../../../op_api/aclnn_gemm_syrk.h"
#include "op_api/op_api_def_nn.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"

using namespace std;
using namespace op;

class l2_gemm_syrk_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "gemm_syrk_test SetUp" << endl; }

    static void TearDownTestCase() { cout << "gemm_syrk_test TearDown" << endl; }
};

// 2D fp16, 显式 alpha/beta
TEST_F(l2_gemm_syrk_test, gemm_syrk_2d_fp16)
{
    // UT stub 平台默认为 910B(DAV_2201)，GemmSyrk 仅支持 3510 系列，需显式切换
    op::SocVersionManager socVersionManager(op::SocVersion::ASCEND950);
    auto a_desc = TensorDesc({256, 512}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 1);
    auto c_desc = TensorDesc({256, 256}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 1);
    auto alpha_desc = ScalarDesc(3.0f);
    auto beta_desc = ScalarDesc(2.0f);

    auto ut = OP_API_UT(aclnnGemmSyrk, INPUT(a_desc, c_desc, alpha_desc, beta_desc), OUTPUT(), false, "full");

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// 3D bf16, batch=2, 默认 alpha/beta (nullptr)
TEST_F(l2_gemm_syrk_test, gemm_syrk_3d_bf16_default_scale)
{
    // UT stub 平台默认为 910B(DAV_2201)，GemmSyrk 仅支持 3510 系列，需显式切换
    op::SocVersionManager socVersionManager(op::SocVersion::ASCEND950);
    auto a_desc = TensorDesc({2, 128, 64}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 1);
    auto c_desc = TensorDesc({2, 128, 128}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 1);

    auto ut = OP_API_UT(aclnnGemmSyrk, INPUT(a_desc, c_desc, nullptr, nullptr), OUTPUT(), false, "full");

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// transpose_x=true: a 为转置 (k, m) 存储
TEST_F(l2_gemm_syrk_test, gemm_syrk_transpose_x)
{
    // UT stub 平台默认为 910B(DAV_2201)，GemmSyrk 仅支持 3510 系列，需显式切换
    op::SocVersionManager socVersionManager(op::SocVersion::ASCEND950);
    auto a_desc = TensorDesc({512, 256}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 1);
    auto c_desc = TensorDesc({256, 256}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 1);
    auto alpha_desc = ScalarDesc(1.0f);
    auto beta_desc = ScalarDesc(1.5f);

    auto ut = OP_API_UT(aclnnGemmSyrk, INPUT(a_desc, c_desc, alpha_desc, beta_desc), OUTPUT(), true, "full");

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// 6D 维度上限
TEST_F(l2_gemm_syrk_test, gemm_syrk_6d)
{
    // UT stub 平台默认为 910B(DAV_2201)，GemmSyrk 仅支持 3510 系列，需显式切换
    op::SocVersionManager socVersionManager(op::SocVersion::ASCEND950);
    auto a_desc = TensorDesc({2, 2, 2, 1, 64, 64}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 1);
    auto c_desc = TensorDesc({2, 2, 2, 1, 64, 64}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 1);

    auto ut = OP_API_UT(aclnnGemmSyrk, INPUT(a_desc, c_desc, nullptr, nullptr), OUTPUT(), false, "full");

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// 拒绝: 1D (低于最小维度)
TEST_F(l2_gemm_syrk_test, gemm_syrk_reject_1d)
{
    auto a_desc = TensorDesc({64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto c_desc = TensorDesc({64}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnGemmSyrk, INPUT(a_desc, c_desc, nullptr, nullptr), OUTPUT(), false, "full");

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 拒绝: 7D (超过最大维度)
TEST_F(l2_gemm_syrk_test, gemm_syrk_reject_7d)
{
    auto a_desc = TensorDesc({2, 2, 2, 2, 2, 64, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto c_desc = TensorDesc({2, 2, 2, 2, 2, 64, 64}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnGemmSyrk, INPUT(a_desc, c_desc, nullptr, nullptr), OUTPUT(), false, "full");

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 拒绝: a 与 c 维数不一致
TEST_F(l2_gemm_syrk_test, gemm_syrk_reject_dim_mismatch)
{
    auto a_desc = TensorDesc({2, 128, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto c_desc = TensorDesc({128, 128}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnGemmSyrk, INPUT(a_desc, c_desc, nullptr, nullptr), OUTPUT(), false, "full");

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 拒绝: c 非方阵 (m != n)
TEST_F(l2_gemm_syrk_test, gemm_syrk_reject_non_square_c)
{
    auto a_desc = TensorDesc({32, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto c_desc = TensorDesc({32, 16}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnGemmSyrk, INPUT(a_desc, c_desc, nullptr, nullptr), OUTPUT(), false, "full");

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 拒绝: a 的 m 轴与 c 不匹配
TEST_F(l2_gemm_syrk_test, gemm_syrk_reject_m_mismatch)
{
    auto a_desc = TensorDesc({64, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto c_desc = TensorDesc({32, 32}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnGemmSyrk, INPUT(a_desc, c_desc, nullptr, nullptr), OUTPUT(), false, "full");

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 拒绝: transpose_x 下 m 轴 (a 最后一维) 与 c 不匹配
TEST_F(l2_gemm_syrk_test, gemm_syrk_reject_m_mismatch_transpose)
{
    auto a_desc = TensorDesc({64, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto c_desc = TensorDesc({32, 32}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnGemmSyrk, INPUT(a_desc, c_desc, nullptr, nullptr), OUTPUT(), true, "full");

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 拒绝: batch 轴不匹配 (原地更新不支持广播)
TEST_F(l2_gemm_syrk_test, gemm_syrk_reject_batch_mismatch)
{
    auto a_desc = TensorDesc({2, 128, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto c_desc = TensorDesc({3, 128, 128}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnGemmSyrk, INPUT(a_desc, c_desc, nullptr, nullptr), OUTPUT(), false, "full");

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 拒绝: 不支持的数据类型 (fp32)
TEST_F(l2_gemm_syrk_test, gemm_syrk_reject_fp32)
{
    auto a_desc = TensorDesc({64, 32}, ACL_FLOAT, ACL_FORMAT_ND);
    auto c_desc = TensorDesc({64, 64}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnGemmSyrk, INPUT(a_desc, c_desc, nullptr, nullptr), OUTPUT(), false, "full");

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 拒绝: a 与 c 数据类型不一致
TEST_F(l2_gemm_syrk_test, gemm_syrk_reject_dtype_mismatch)
{
    auto a_desc = TensorDesc({64, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto c_desc = TensorDesc({64, 64}, ACL_BF16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnGemmSyrk, INPUT(a_desc, c_desc, nullptr, nullptr), OUTPUT(), false, "full");

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 拒绝: fill_mode 仅实现 "full", "up"/"low" 返回参数错误
TEST_F(l2_gemm_syrk_test, gemm_syrk_reject_fill_mode_up)
{
    auto a_desc = TensorDesc({64, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto c_desc = TensorDesc({64, 64}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnGemmSyrk, INPUT(a_desc, c_desc, nullptr, nullptr), OUTPUT(), false, "up");

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 拒绝: a 为空指针
TEST_F(l2_gemm_syrk_test, gemm_syrk_reject_null_a)
{
    auto c_desc = TensorDesc({64, 64}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnGemmSyrk, INPUT(nullptr, c_desc, nullptr, nullptr), OUTPUT(), false, "full");

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

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
#include "../../../op_host/op_api/aclnn_chamfer_distance_backward.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"

using namespace op;
using namespace std;

class l2_chamfer_distance_backward_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "chamfer_distance_backward_test SetUp" << endl; }

    static void TearDownTestCase() { cout << "chamfer_distance_backward_test TearDown" << endl; }
};

TEST_F(l2_chamfer_distance_backward_test, Ascend910B_case_float_normal)
{
    if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B) {
        const int64_t batch = 2;
        const int64_t num = 1000;
        auto xyz1 = TensorDesc({batch, num, 2}, ACL_FLOAT, ACL_FORMAT_ND);
        auto xyz2 = TensorDesc({batch, num, 2}, ACL_FLOAT, ACL_FORMAT_ND);
        auto idx1 = TensorDesc({batch, num}, ACL_INT32, ACL_FORMAT_ND);
        auto idx2 = TensorDesc({batch, num}, ACL_INT32, ACL_FORMAT_ND);
        auto grad_dist1 = TensorDesc({batch, num}, ACL_FLOAT, ACL_FORMAT_ND);
        auto grad_dist2 = TensorDesc({batch, num}, ACL_FLOAT, ACL_FORMAT_ND);
        auto grad_xyz1 = TensorDesc({batch, num, 2}, ACL_FLOAT, ACL_FORMAT_ND);
        auto grad_xyz2 = TensorDesc({batch, num, 2}, ACL_FLOAT, ACL_FORMAT_ND);

        auto ut = OP_API_UT(aclnnChamferDistanceBackward, INPUT(xyz1, xyz2, idx1, idx2, grad_dist1, grad_dist2),
                            OUTPUT(grad_xyz1, grad_xyz2));

        uint64_t workspace_size = 0;
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
        EXPECT_EQ(aclRet, ACL_SUCCESS);
    }
}

TEST_F(l2_chamfer_distance_backward_test, Ascend910B_case_half_normal)
{
    if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B) {
        const int64_t batch = 32;
        const int64_t num = 10000;
        auto xyz1 = TensorDesc({batch, num, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
        auto xyz2 = TensorDesc({batch, num, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
        auto idx1 = TensorDesc({batch, num}, ACL_INT32, ACL_FORMAT_ND);
        auto idx2 = TensorDesc({batch, num}, ACL_INT32, ACL_FORMAT_ND);
        auto grad_dist1 = TensorDesc({batch, num}, ACL_FLOAT16, ACL_FORMAT_ND);
        auto grad_dist2 = TensorDesc({batch, num}, ACL_FLOAT16, ACL_FORMAT_ND);
        auto grad_xyz1 = TensorDesc({batch, num, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
        auto grad_xyz2 = TensorDesc({batch, num, 2}, ACL_FLOAT16, ACL_FORMAT_ND);

        auto ut = OP_API_UT(aclnnChamferDistanceBackward, INPUT(xyz1, xyz2, idx1, idx2, grad_dist1, grad_dist2),
                            OUTPUT(grad_xyz1, grad_xyz2));

        uint64_t workspace_size = 0;
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
        EXPECT_EQ(aclRet, ACL_SUCCESS);
    }
}

TEST_F(l2_chamfer_distance_backward_test, Ascend910B_case_nullptr_normal)
{
    if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B) {
        const int64_t batch = 32;
        const int64_t num = 10000;
        auto xyz1 = TensorDesc({batch, num, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
        auto xyz2 = TensorDesc({batch, num, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
        auto idx1 = TensorDesc({batch, num}, ACL_INT32, ACL_FORMAT_ND);
        auto idx2 = TensorDesc({batch, num}, ACL_INT32, ACL_FORMAT_ND);
        auto grad_dist1 = TensorDesc({batch, num}, ACL_FLOAT16, ACL_FORMAT_ND);
        auto grad_dist2 = TensorDesc({batch, num}, ACL_FLOAT16, ACL_FORMAT_ND);
        auto grad_xyz1 = TensorDesc({batch, num, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
        auto grad_xyz2 = TensorDesc({batch, num, 2}, ACL_FLOAT16, ACL_FORMAT_ND);

        auto ut = OP_API_UT(aclnnChamferDistanceBackward,
                            INPUT((aclTensor*)nullptr, xyz2, idx1, idx2, grad_dist1, grad_dist2),
                            OUTPUT(grad_xyz1, grad_xyz2));

        uint64_t workspace_size = 0;
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
        EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
    }
}

TEST_F(l2_chamfer_distance_backward_test, Ascend910B_case_invalid_normal)
{
    const int64_t batch = 32;
    const int64_t num = 10000;
    auto xyz1 = TensorDesc({batch, num, 2}, ACL_INT32, ACL_FORMAT_ND);
    auto xyz2 = TensorDesc({batch, num, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto idx1 = TensorDesc({batch, num}, ACL_INT32, ACL_FORMAT_ND);
    auto idx2 = TensorDesc({batch, num}, ACL_INT32, ACL_FORMAT_ND);
    auto grad_dist1 = TensorDesc({batch, num}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto grad_dist2 = TensorDesc({batch, num}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto grad_xyz1 = TensorDesc({batch, num, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto grad_xyz2 = TensorDesc({batch, num, 2}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnChamferDistanceBackward, INPUT(xyz1, xyz2, idx1, idx2, grad_dist1, grad_dist2),
                        OUTPUT(grad_xyz1, grad_xyz2));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// issue#7042: 混合dtype(xyz1=fp32、xyz2=fp16等)应在第一段入口返回161002，而非内部错误码561103
TEST_F(l2_chamfer_distance_backward_test, Ascend910B_case_mix_dtype_invalid)
{
    const int64_t batch = 2;
    const int64_t num = 1000;
    auto xyz1 = TensorDesc({batch, num, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto xyz2 = TensorDesc({batch, num, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto idx1 = TensorDesc({batch, num}, ACL_INT32, ACL_FORMAT_ND);
    auto idx2 = TensorDesc({batch, num}, ACL_INT32, ACL_FORMAT_ND);
    auto grad_dist1 = TensorDesc({batch, num}, ACL_FLOAT, ACL_FORMAT_ND);
    auto grad_dist2 = TensorDesc({batch, num}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto grad_xyz1 = TensorDesc({batch, num, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto grad_xyz2 = TensorDesc({batch, num, 2}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnChamferDistanceBackward, INPUT(xyz1, xyz2, idx1, idx2, grad_dist1, grad_dist2),
                        OUTPUT(grad_xyz1, grad_xyz2));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// issue#7042: 输出dtype与输入不一致时应返回161002
TEST_F(l2_chamfer_distance_backward_test, Ascend910B_case_output_dtype_mismatch_invalid)
{
    const int64_t batch = 2;
    const int64_t num = 1000;
    auto xyz1 = TensorDesc({batch, num, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto xyz2 = TensorDesc({batch, num, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto idx1 = TensorDesc({batch, num}, ACL_INT32, ACL_FORMAT_ND);
    auto idx2 = TensorDesc({batch, num}, ACL_INT32, ACL_FORMAT_ND);
    auto grad_dist1 = TensorDesc({batch, num}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto grad_dist2 = TensorDesc({batch, num}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto grad_xyz1 = TensorDesc({batch, num, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto grad_xyz2 = TensorDesc({batch, num, 2}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnChamferDistanceBackward, INPUT(xyz1, xyz2, idx1, idx2, grad_dist1, grad_dist2),
                        OUTPUT(grad_xyz1, grad_xyz2));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// issue#7041: xyz1与xyz2点数不一致(N=32、M=16)时应返回161002，不应放行到执行段
TEST_F(l2_chamfer_distance_backward_test, Ascend910B_case_point_dim_mismatch_invalid)
{
    const int64_t num1 = 32;
    const int64_t num2 = 16;
    auto xyz1 = TensorDesc({1, num1, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto xyz2 = TensorDesc({1, num2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto idx1 = TensorDesc({1, num1}, ACL_INT32, ACL_FORMAT_ND);
    auto idx2 = TensorDesc({1, num2}, ACL_INT32, ACL_FORMAT_ND);
    auto grad_dist1 = TensorDesc({1, num1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto grad_dist2 = TensorDesc({1, num2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto grad_xyz1 = TensorDesc({1, num1, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto grad_xyz2 = TensorDesc({1, num2, 2}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnChamferDistanceBackward, INPUT(xyz1, xyz2, idx1, idx2, grad_dist1, grad_dist2),
                        OUTPUT(grad_xyz1, grad_xyz2));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// issue#7041: xyz1与xyz2的batch维不一致时应返回161002
TEST_F(l2_chamfer_distance_backward_test, Ascend910B_case_batch_dim_mismatch_invalid)
{
    auto xyz1 = TensorDesc({2, 16, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto xyz2 = TensorDesc({1, 16, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto idx1 = TensorDesc({2, 16}, ACL_INT32, ACL_FORMAT_ND);
    auto idx2 = TensorDesc({1, 16}, ACL_INT32, ACL_FORMAT_ND);
    auto grad_dist1 = TensorDesc({2, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    auto grad_dist2 = TensorDesc({1, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    auto grad_xyz1 = TensorDesc({2, 16, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto grad_xyz2 = TensorDesc({1, 16, 2}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnChamferDistanceBackward, INPUT(xyz1, xyz2, idx1, idx2, grad_dist1, grad_dist2),
                        OUTPUT(grad_xyz1, grad_xyz2));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// issue#7041: idx的shape与xyz前两维不一致时应返回161002
TEST_F(l2_chamfer_distance_backward_test, Ascend910B_case_idx_shape_mismatch_invalid)
{
    auto xyz1 = TensorDesc({1, 16, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto xyz2 = TensorDesc({1, 16, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto idx1 = TensorDesc({1, 8}, ACL_INT32, ACL_FORMAT_ND);
    auto idx2 = TensorDesc({1, 16}, ACL_INT32, ACL_FORMAT_ND);
    auto grad_dist1 = TensorDesc({1, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    auto grad_dist2 = TensorDesc({1, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    auto grad_xyz1 = TensorDesc({1, 16, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto grad_xyz2 = TensorDesc({1, 16, 2}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnChamferDistanceBackward, INPUT(xyz1, xyz2, idx1, idx2, grad_dist1, grad_dist2),
                        OUTPUT(grad_xyz1, grad_xyz2));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// issue#7041: xyz最后一维非2时应返回161002
TEST_F(l2_chamfer_distance_backward_test, Ascend910B_case_coord_dim_invalid)
{
    auto xyz1 = TensorDesc({1, 16, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto xyz2 = TensorDesc({1, 16, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto idx1 = TensorDesc({1, 16}, ACL_INT32, ACL_FORMAT_ND);
    auto idx2 = TensorDesc({1, 16}, ACL_INT32, ACL_FORMAT_ND);
    auto grad_dist1 = TensorDesc({1, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    auto grad_dist2 = TensorDesc({1, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    auto grad_xyz1 = TensorDesc({1, 16, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto grad_xyz2 = TensorDesc({1, 16, 2}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnChamferDistanceBackward, INPUT(xyz1, xyz2, idx1, idx2, grad_dist1, grad_dist2),
                        OUTPUT(grad_xyz1, grad_xyz2));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// issue#7041: xyz的秩非3时应返回161002
TEST_F(l2_chamfer_distance_backward_test, Ascend910B_case_rank_invalid)
{
    auto xyz1 = TensorDesc({16, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto xyz2 = TensorDesc({1, 16, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto idx1 = TensorDesc({1, 16}, ACL_INT32, ACL_FORMAT_ND);
    auto idx2 = TensorDesc({1, 16}, ACL_INT32, ACL_FORMAT_ND);
    auto grad_dist1 = TensorDesc({1, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    auto grad_dist2 = TensorDesc({1, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    auto grad_xyz1 = TensorDesc({16, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto grad_xyz2 = TensorDesc({1, 16, 2}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnChamferDistanceBackward, INPUT(xyz1, xyz2, idx1, idx2, grad_dist1, grad_dist2),
                        OUTPUT(grad_xyz1, grad_xyz2));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// issue#7041: 输出shape与对应输入不一致时应返回161002
TEST_F(l2_chamfer_distance_backward_test, Ascend910B_case_output_shape_mismatch_invalid)
{
    auto xyz1 = TensorDesc({1, 16, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto xyz2 = TensorDesc({1, 16, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto idx1 = TensorDesc({1, 16}, ACL_INT32, ACL_FORMAT_ND);
    auto idx2 = TensorDesc({1, 16}, ACL_INT32, ACL_FORMAT_ND);
    auto grad_dist1 = TensorDesc({1, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    auto grad_dist2 = TensorDesc({1, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    auto grad_xyz1 = TensorDesc({1, 8, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto grad_xyz2 = TensorDesc({1, 16, 2}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnChamferDistanceBackward, INPUT(xyz1, xyz2, idx1, idx2, grad_dist1, grad_dist2),
                        OUTPUT(grad_xyz1, grad_xyz2));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// issue#8186: xyz1/xyz2非空、idx/gradDist为空时，两个非空输出gradXyz1/gradXyz2需补写全0，
// 接口应返回SUCCESS，不能静默跳过输出写入
TEST_F(l2_chamfer_distance_backward_test, Ascend910B_case_empty_idx_fill_grad_outputs)
{
    const int64_t batch = 2;
    const int64_t num = 4;
    auto xyz1 = TensorDesc({batch, num, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 10);
    auto xyz2 = TensorDesc({batch, num, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 10);
    auto idx1 = TensorDesc({batch, 0}, ACL_INT32, ACL_FORMAT_ND);
    auto idx2 = TensorDesc({batch, 0}, ACL_INT32, ACL_FORMAT_ND);
    auto grad_dist1 = TensorDesc({batch, 0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto grad_dist2 = TensorDesc({batch, 0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto grad_xyz1 = TensorDesc({batch, num, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto grad_xyz2 = TensorDesc({batch, num, 2}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnChamferDistanceBackward, INPUT(xyz1, xyz2, idx1, idx2, grad_dist1, grad_dist2),
                        OUTPUT(grad_xyz1, grad_xyz2));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    // 冒烟断言：空tensor早退路径不报错且下发输出清零。
    // Fill类算子workspace可能为0，不能以workspaceSize>0作为行为判据，stub环境无输出值校验能力。
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// issue#8186: 全空输入（xyz/idx/gradDist均为空tensor）时接口应返回SUCCESS，输出为空无需填充
TEST_F(l2_chamfer_distance_backward_test, Ascend910B_case_all_empty_input_success)
{
    const int64_t batch = 2;
    auto xyz1 = TensorDesc({batch, 0, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto xyz2 = TensorDesc({batch, 0, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto idx1 = TensorDesc({batch, 0}, ACL_INT32, ACL_FORMAT_ND);
    auto idx2 = TensorDesc({batch, 0}, ACL_INT32, ACL_FORMAT_ND);
    auto grad_dist1 = TensorDesc({batch, 0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto grad_dist2 = TensorDesc({batch, 0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto grad_xyz1 = TensorDesc({batch, 0, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto grad_xyz2 = TensorDesc({batch, 0, 2}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnChamferDistanceBackward, INPUT(xyz1, xyz2, idx1, idx2, grad_dist1, grad_dist2),
                        OUTPUT(grad_xyz1, grad_xyz2));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    EXPECT_EQ(workspace_size, 0U);
}

// issue#8186: 空idx场景的fp16变体，非空输出需按fp16补写全0
TEST_F(l2_chamfer_distance_backward_test, Ascend910B_case_empty_idx_fill_grad_outputs_half)
{
    const int64_t batch = 2;
    const int64_t num = 4;
    auto xyz1 = TensorDesc({batch, num, 2}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(1, 10);
    auto xyz2 = TensorDesc({batch, num, 2}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(1, 10);
    auto idx1 = TensorDesc({batch, 0}, ACL_INT32, ACL_FORMAT_ND);
    auto idx2 = TensorDesc({batch, 0}, ACL_INT32, ACL_FORMAT_ND);
    auto grad_dist1 = TensorDesc({batch, 0}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto grad_dist2 = TensorDesc({batch, 0}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto grad_xyz1 = TensorDesc({batch, num, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto grad_xyz2 = TensorDesc({batch, num, 2}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnChamferDistanceBackward, INPUT(xyz1, xyz2, idx1, idx2, grad_dist1, grad_dist2),
                        OUTPUT(grad_xyz1, grad_xyz2));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

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

#include "opdev/op_log.h"
#include "../../../op_api/aclnn_adaptive_max_pool3d_backward.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace op;
using namespace std;

class l2_adaptive_max_pool3d_backward_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "l2_adaptive_max_pool3d_backward_test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "l2_adaptive_max_pool3d_backward_test TearDown" << std::endl;
    }
};

// 正常场景：数据类型为float
TEST_F(l2_adaptive_max_pool3d_backward_test, ascend910B2_normal_float)
{
    vector<int64_t> self_dims = {2, 3, 2, 2, 2};
    vector<int64_t> out_dims = {2, 3, 1, 1, 1};
    vector<int64_t> indices_dims = {2, 3, 1, 1, 1};

    auto tensor_gradOutput = TensorDesc(out_dims, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto tensor_self = TensorDesc(self_dims, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto indices_tensor_desc = TensorDesc(indices_dims, ACL_INT32, ACL_FORMAT_NCDHW);
    auto gradInput_tensor_desc = TensorDesc(self_dims, ACL_FLOAT, ACL_FORMAT_NCDHW);

    auto ut = OP_API_UT(
        aclnnAdaptiveMaxPool3dBackward, INPUT(tensor_gradOutput, tensor_self, indices_tensor_desc),
        OUTPUT(gradInput_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常场景：数据类型是float16
TEST_F(l2_adaptive_max_pool3d_backward_test, ascend910B2_normal_float16)
{
    vector<int64_t> self_dims = {2, 3, 2, 2, 2};
    vector<int64_t> out_dims = {2, 3, 1, 1, 1};
    vector<int64_t> indices_dims = {2, 3, 1, 1, 1};

    auto tensor_gradOutput = TensorDesc(out_dims, ACL_FLOAT16, ACL_FORMAT_NCDHW);
    auto tensor_self = TensorDesc(self_dims, ACL_FLOAT16, ACL_FORMAT_NCDHW);
    auto indices_tensor_desc = TensorDesc(indices_dims, ACL_INT32, ACL_FORMAT_NCDHW);
    auto gradInput_tensor_desc = TensorDesc(self_dims, ACL_FLOAT16, ACL_FORMAT_NCDHW);

    auto ut = OP_API_UT(
        aclnnAdaptiveMaxPool3dBackward, INPUT(tensor_gradOutput, tensor_self, indices_tensor_desc),
        OUTPUT(gradInput_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常场景：数据类型是bf16
TEST_F(l2_adaptive_max_pool3d_backward_test, ascend910B2_normal_bfloat16)
{
    vector<int64_t> self_dims = {2, 3, 2, 2, 2};
    vector<int64_t> out_dims = {2, 3, 1, 1, 1};
    vector<int64_t> indices_dims = {2, 3, 1, 1, 1};

    auto tensor_gradOutput = TensorDesc(out_dims, ACL_BF16, ACL_FORMAT_NCDHW);
    auto tensor_self = TensorDesc(self_dims, ACL_BF16, ACL_FORMAT_NCDHW);
    auto indices_tensor_desc = TensorDesc(indices_dims, ACL_INT32, ACL_FORMAT_NCDHW);
    auto gradInput_tensor_desc = TensorDesc(self_dims, ACL_BF16, ACL_FORMAT_NCDHW);

    auto ut = OP_API_UT(
        aclnnAdaptiveMaxPool3dBackward, INPUT(tensor_gradOutput, tensor_self, indices_tensor_desc),
        OUTPUT(gradInput_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 异常场景：非法数据类型
TEST_F(l2_adaptive_max_pool3d_backward_test, ascend910B2_5d_invalid_dtype)
{
    vector<int64_t> self_dims = {2, 3, 2, 2, 2};
    vector<int64_t> out_dims = {2, 3, 1, 1, 1};
    vector<int64_t> indices_dims = {2, 3, 1, 1, 1};

    auto tensor_gradOutput = TensorDesc(out_dims, ACL_INT32, ACL_FORMAT_NCDHW);
    auto tensor_self = TensorDesc(self_dims, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto indices_tensor_desc = TensorDesc(indices_dims, ACL_INT32, ACL_FORMAT_NCDHW);
    auto gradInput_tensor_desc = TensorDesc(self_dims, ACL_FLOAT, ACL_FORMAT_NCDHW);

    auto ut = OP_API_UT(
        aclnnAdaptiveMaxPool3dBackward, INPUT(tensor_gradOutput, tensor_self, indices_tensor_desc),
        OUTPUT(gradInput_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 异常场景：非法数据格式
TEST_F(l2_adaptive_max_pool3d_backward_test, ascend910B2_5d_invalid_dformat)
{
    vector<int64_t> self_dims = {2, 3, 2, 2, 2};
    vector<int64_t> out_dims = {2, 3, 1, 1, 1};
    vector<int64_t> indices_dims = {2, 3, 1, 1, 1};

    auto tensor_gradOutput = TensorDesc(out_dims, ACL_INT32, ACL_FORMAT_NCL);
    auto tensor_self = TensorDesc(self_dims, ACL_FLOAT, ACL_FORMAT_NCL);
    auto indices_tensor_desc = TensorDesc(indices_dims, ACL_INT32, ACL_FORMAT_NCL);
    auto gradInput_tensor_desc = TensorDesc(self_dims, ACL_FLOAT, ACL_FORMAT_NCL);

    auto ut = OP_API_UT(
        aclnnAdaptiveMaxPool3dBackward, INPUT(tensor_gradOutput, tensor_self, indices_tensor_desc),
        OUTPUT(gradInput_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 异常场景：数据格式不一致
TEST_F(l2_adaptive_max_pool3d_backward_test, ascend910B2_5d_dformat_not_same)
{
    vector<int64_t> self_dims = {2, 3, 2, 2, 2};
    vector<int64_t> out_dims = {2, 3, 1, 1, 1};
    vector<int64_t> indices_dims = {2, 3, 1, 1, 1};

    auto tensor_gradOutput = TensorDesc(out_dims, ACL_INT32, ACL_FORMAT_NCDHW);
    auto tensor_self = TensorDesc(self_dims, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto indices_tensor_desc = TensorDesc(indices_dims, ACL_INT32, ACL_FORMAT_NCDHW);
    auto gradInput_tensor_desc = TensorDesc(self_dims, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(
        aclnnAdaptiveMaxPool3dBackward, INPUT(tensor_gradOutput, tensor_self, indices_tensor_desc),
        OUTPUT(gradInput_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 正常场景：self维度为4d
TEST_F(l2_adaptive_max_pool3d_backward_test, ascend910B2_normal_4d)
{
    vector<int64_t> self_dims = {2, 2, 2, 2};
    vector<int64_t> out_dims = {2, 1, 1, 1};
    vector<int64_t> indices_dims = {2, 1, 1, 1};

    auto tensor_gradOutput = TensorDesc(out_dims, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto tensor_self = TensorDesc(self_dims, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto indices_tensor_desc = TensorDesc(indices_dims, ACL_INT32, ACL_FORMAT_NCHW);
    auto gradInput_tensor_desc = TensorDesc(self_dims, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(
        aclnnAdaptiveMaxPool3dBackward, INPUT(tensor_gradOutput, tensor_self, indices_tensor_desc),
        OUTPUT(gradInput_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    if (GetCurrentPlatformInfo().GetSocVersion() != SocVersion::ASCEND910B) {
        EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    } else {
        EXPECT_EQ(aclRet, ACL_SUCCESS);
    }
}

// 异常场景：输入输出shape不是4d或5d
TEST_F(l2_adaptive_max_pool3d_backward_test, ascend910B2_normal_invalid_dim)
{
    vector<int64_t> self_dims = {2, 2, 2};
    vector<int64_t> out_dims = {2, 1, 1};
    vector<int64_t> indices_dims = {2, 1, 1};

    auto tensor_gradOutput = TensorDesc(out_dims, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto tensor_self = TensorDesc(self_dims, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto indices_tensor_desc = TensorDesc(indices_dims, ACL_INT32, ACL_FORMAT_NCHW);
    auto gradInput_tensor_desc = TensorDesc(self_dims, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(
        aclnnAdaptiveMaxPool3dBackward, INPUT(tensor_gradOutput, tensor_self, indices_tensor_desc),
        OUTPUT(gradInput_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 异常场景：输入输出存在空指针
TEST_F(l2_adaptive_max_pool3d_backward_test, ascend910B2_5d_nullptr)
{
    vector<int64_t> self_dims = {2, 3, 1, 1, 1};
    vector<int64_t> indices_dims = {2, 3, 1, 1, 1};

    auto tensor_gradOutput = nullptr;
    auto tensor_self = TensorDesc(self_dims, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto indices_tensor_desc = TensorDesc(indices_dims, ACL_INT32, ACL_FORMAT_NCDHW);
    auto gradInput_tensor_desc = TensorDesc(self_dims, ACL_FLOAT, ACL_FORMAT_NCDHW);

    auto ut = OP_API_UT(
        aclnnAdaptiveMaxPool3dBackward, INPUT(tensor_gradOutput, tensor_self, indices_tensor_desc),
        OUTPUT(gradInput_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 异常场景：gradOutput与indices的shape不一致
TEST_F(l2_adaptive_max_pool3d_backward_test, ascend910B2_5d_shape_not_same_1)
{
    vector<int64_t> self_dims = {2, 3, 2, 2, 2};
    vector<int64_t> out_dims = {2, 3, 1, 1, 1};
    vector<int64_t> indices_dims = {2, 3, 2, 1, 1};

    auto tensor_gradOutput = TensorDesc(out_dims, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto tensor_self = TensorDesc(self_dims, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto indices_tensor_desc = TensorDesc(indices_dims, ACL_INT32, ACL_FORMAT_NCDHW);
    auto gradInput_tensor_desc = TensorDesc(self_dims, ACL_FLOAT, ACL_FORMAT_NCDHW);

    auto ut = OP_API_UT(
        aclnnAdaptiveMaxPool3dBackward, INPUT(tensor_gradOutput, tensor_self, indices_tensor_desc),
        OUTPUT(gradInput_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 异常场景：gradInput与self的shape不一致
TEST_F(l2_adaptive_max_pool3d_backward_test, ascend910B2_5d_shape_not_same_2)
{
    vector<int64_t> self_dims = {2, 3, 2, 2, 2};
    vector<int64_t> out_dims = {2, 3, 1, 1, 1};
    vector<int64_t> indices_dims = {2, 3, 1, 1, 1};
    vector<int64_t> gradIn_dims = {2, 3, 2, 2, 1};

    auto tensor_gradOutput = TensorDesc(out_dims, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto tensor_self = TensorDesc(self_dims, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto indices_tensor_desc = TensorDesc(indices_dims, ACL_INT32, ACL_FORMAT_NCDHW);
    auto gradInput_tensor_desc = TensorDesc(gradIn_dims, ACL_FLOAT, ACL_FORMAT_NCDHW);

    auto ut = OP_API_UT(
        aclnnAdaptiveMaxPool3dBackward, INPUT(tensor_gradOutput, tensor_self, indices_tensor_desc),
        OUTPUT(gradInput_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 异常场景：self的dhw超出indices的表达范围
TEST_F(l2_adaptive_max_pool3d_backward_test, ascend910B2_5d_shape_over_range)
{
    vector<int64_t> self_dims = {2, 3, 10000, 10000, 1000};
    vector<int64_t> out_dims = {2, 3, 1, 1, 1};
    vector<int64_t> indices_dims = {2, 3, 1, 1, 1};

    auto tensor_gradOutput = TensorDesc(out_dims, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto tensor_self = TensorDesc(self_dims, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto indices_tensor_desc = TensorDesc(indices_dims, ACL_INT32, ACL_FORMAT_NCDHW);
    auto gradInput_tensor_desc = TensorDesc(self_dims, ACL_FLOAT, ACL_FORMAT_NCDHW);

    auto ut = OP_API_UT(
        aclnnAdaptiveMaxPool3dBackward, INPUT(tensor_gradOutput, tensor_self, indices_tensor_desc),
        OUTPUT(gradInput_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}
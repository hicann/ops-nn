/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <vector>
#include <array>
#include <iostream>
#include <limits>
#include "gtest/gtest.h"
#include "../../../../op_api/aclnn_swiglu_group_grad.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"

using namespace op;
using namespace std;

class l2_swiglu_group_grad_test : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "swiglu_group_grad_test SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "swiglu_group_grad_test TearDown" << std::endl; }
};

TEST_F(l2_swiglu_group_grad_test, l2_normal_FLOAT_ND_no_options)
{
    auto dyDesc = TensorDesc({4, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    auto xDesc = TensorDesc({4, 32}, ACL_FLOAT, ACL_FORMAT_ND);
    auto dxOutDesc = TensorDesc({4, 32}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnSwigluGroupGrad, INPUT(dyDesc, xDesc, nullptr, nullptr, nullptr, 0.0f),
                        OUTPUT(dxOutDesc, nullptr));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_swiglu_group_grad_test, l2_normal_FLOAT16_ND_no_options)
{
    auto dyDesc = TensorDesc({4, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto xDesc = TensorDesc({4, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto dxOutDesc = TensorDesc({4, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnSwigluGroupGrad, INPUT(dyDesc, xDesc, nullptr, nullptr, nullptr, 0.0f),
                        OUTPUT(dxOutDesc, nullptr));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_swiglu_group_grad_test, l2_normal_BF16_ND_no_options)
{
    auto dyDesc = TensorDesc({4, 16}, ACL_BF16, ACL_FORMAT_ND);
    auto xDesc = TensorDesc({4, 32}, ACL_BF16, ACL_FORMAT_ND);
    auto dxOutDesc = TensorDesc({4, 32}, ACL_BF16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnSwigluGroupGrad, INPUT(dyDesc, xDesc, nullptr, nullptr, nullptr, 0.0f),
                        OUTPUT(dxOutDesc, nullptr));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_swiglu_group_grad_test, l2_normal_FLOAT_ND_with_clamp)
{
    auto dyDesc = TensorDesc({4, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    auto xDesc = TensorDesc({4, 32}, ACL_FLOAT, ACL_FORMAT_ND);
    auto dxOutDesc = TensorDesc({4, 32}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnSwigluGroupGrad, INPUT(dyDesc, xDesc, nullptr, nullptr, nullptr, 3.0f),
                        OUTPUT(dxOutDesc, nullptr));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_swiglu_group_grad_test, l2_normal_FLOAT_ND_with_topk_weight)
{
    auto dyDesc = TensorDesc({4, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    auto xDesc = TensorDesc({4, 32}, ACL_FLOAT, ACL_FORMAT_ND);
    auto weightDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto yOriginDesc = TensorDesc({4, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    auto dxOutDesc = TensorDesc({4, 32}, ACL_FLOAT, ACL_FORMAT_ND);
    auto dWeightDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnSwigluGroupGrad, INPUT(dyDesc, xDesc, weightDesc, yOriginDesc, nullptr, 0.0f),
                        OUTPUT(dxOutDesc, dWeightDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_swiglu_group_grad_test, l2_normal_FLOAT_ND_with_all_options)
{
    auto dyDesc = TensorDesc({4, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    auto xDesc = TensorDesc({4, 32}, ACL_FLOAT, ACL_FORMAT_ND);
    auto weightDesc = TensorDesc({4, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto yOriginDesc = TensorDesc({4, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    auto groupIndexDesc = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);
    auto dxOutDesc = TensorDesc({4, 32}, ACL_FLOAT, ACL_FORMAT_ND);
    auto dWeightDesc = TensorDesc({4, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnSwigluGroupGrad, INPUT(dyDesc, xDesc, weightDesc, yOriginDesc, groupIndexDesc, 3.0f),
                        OUTPUT(dxOutDesc, dWeightDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_swiglu_group_grad_test, l2_normal_FLOAT_ND_3d_no_options)
{
    auto dyDesc = TensorDesc({2, 4, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    auto xDesc = TensorDesc({2, 4, 32}, ACL_FLOAT, ACL_FORMAT_ND);
    auto dxOutDesc = TensorDesc({2, 4, 32}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnSwigluGroupGrad, INPUT(dyDesc, xDesc, nullptr, nullptr, nullptr, 0.0f),
                        OUTPUT(dxOutDesc, nullptr));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_swiglu_group_grad_test, l2_normal_FLOAT_ND_3d_with_all_options)
{
    auto dyDesc = TensorDesc({2, 4, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    auto xDesc = TensorDesc({2, 4, 32}, ACL_FLOAT, ACL_FORMAT_ND);
    auto weightDesc = TensorDesc({2, 4, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto yOriginDesc = TensorDesc({2, 4, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    auto groupIndexDesc = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);
    auto dxOutDesc = TensorDesc({2, 4, 32}, ACL_FLOAT, ACL_FORMAT_ND);
    auto dWeightDesc = TensorDesc({2, 4, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnSwigluGroupGrad, INPUT(dyDesc, xDesc, weightDesc, yOriginDesc, groupIndexDesc, 3.0f),
                        OUTPUT(dxOutDesc, dWeightDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_swiglu_group_grad_test, l2_normal_FLOAT16_ND_large_shape)
{
    auto dyDesc = TensorDesc({64, 1024}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto xDesc = TensorDesc({64, 2048}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto dxOutDesc = TensorDesc({64, 2048}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnSwigluGroupGrad, INPUT(dyDesc, xDesc, nullptr, nullptr, nullptr, 0.0f),
                        OUTPUT(dxOutDesc, nullptr));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_swiglu_group_grad_test, l2_normal_FLOAT_ND_empty_tensor)
{
    auto dyDesc = TensorDesc({0, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    auto xDesc = TensorDesc({0, 32}, ACL_FLOAT, ACL_FORMAT_ND);
    auto dxOutDesc = TensorDesc({0, 32}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnSwigluGroupGrad, INPUT(dyDesc, xDesc, nullptr, nullptr, nullptr, 0.0f),
                        OUTPUT(dxOutDesc, nullptr));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_swiglu_group_grad_test, l2_abnormal_dy_nullptr)
{
    auto xDesc = TensorDesc({4, 32}, ACL_FLOAT, ACL_FORMAT_ND);
    auto dxOutDesc = TensorDesc({4, 32}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnSwigluGroupGrad, INPUT(nullptr, xDesc, nullptr, nullptr, nullptr, 0.0f),
                        OUTPUT(dxOutDesc, nullptr));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_swiglu_group_grad_test, l2_abnormal_x_nullptr)
{
    auto dyDesc = TensorDesc({4, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    auto dxOutDesc = TensorDesc({4, 32}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnSwigluGroupGrad, INPUT(dyDesc, nullptr, nullptr, nullptr, nullptr, 0.0f),
                        OUTPUT(dxOutDesc, nullptr));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_swiglu_group_grad_test, l2_abnormal_dxOut_nullptr)
{
    auto dyDesc = TensorDesc({4, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    auto xDesc = TensorDesc({4, 32}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnSwigluGroupGrad, INPUT(dyDesc, xDesc, nullptr, nullptr, nullptr, 0.0f),
                        OUTPUT(nullptr, nullptr));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_swiglu_group_grad_test, l2_abnormal_topk_present_dtopk_nullptr)
{
    auto dyDesc = TensorDesc({4, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    auto xDesc = TensorDesc({4, 32}, ACL_FLOAT, ACL_FORMAT_ND);
    auto weightDesc = TensorDesc({4, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto yOriginDesc = TensorDesc({4, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    auto dxOutDesc = TensorDesc({4, 32}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnSwigluGroupGrad, INPUT(dyDesc, xDesc, weightDesc, yOriginDesc, nullptr, 0.0f),
                        OUTPUT(dxOutDesc, nullptr));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_swiglu_group_grad_test, l2_abnormal_dtype_int8)
{
    auto dyDesc = TensorDesc({4, 16}, ACL_INT8, ACL_FORMAT_ND);
    auto xDesc = TensorDesc({4, 32}, ACL_INT8, ACL_FORMAT_ND);
    auto dxOutDesc = TensorDesc({4, 32}, ACL_INT8, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnSwigluGroupGrad, INPUT(dyDesc, xDesc, nullptr, nullptr, nullptr, 0.0f),
                        OUTPUT(dxOutDesc, nullptr));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_swiglu_group_grad_test, l2_abnormal_dtype_double)
{
    auto dyDesc = TensorDesc({4, 16}, ACL_DOUBLE, ACL_FORMAT_ND);
    auto xDesc = TensorDesc({4, 32}, ACL_DOUBLE, ACL_FORMAT_ND);
    auto dxOutDesc = TensorDesc({4, 32}, ACL_DOUBLE, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnSwigluGroupGrad, INPUT(dyDesc, xDesc, nullptr, nullptr, nullptr, 0.0f),
                        OUTPUT(dxOutDesc, nullptr));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_swiglu_group_grad_test, l2_abnormal_dtype_mismatch_dy_x)
{
    auto dyDesc = TensorDesc({4, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    auto xDesc = TensorDesc({4, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto dxOutDesc = TensorDesc({4, 32}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnSwigluGroupGrad, INPUT(dyDesc, xDesc, nullptr, nullptr, nullptr, 0.0f),
                        OUTPUT(dxOutDesc, nullptr));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_swiglu_group_grad_test, l2_normal_h_not_aligned)
{
    auto dyDesc = TensorDesc({4, 17}, ACL_FLOAT, ACL_FORMAT_ND);
    auto xDesc = TensorDesc({4, 34}, ACL_FLOAT, ACL_FORMAT_ND);
    auto dxOutDesc = TensorDesc({4, 34}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnSwigluGroupGrad, INPUT(dyDesc, xDesc, nullptr, nullptr, nullptr, 0.0f),
                        OUTPUT(dxOutDesc, nullptr));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_swiglu_group_grad_test, l2_normal_small_h)
{
    auto dyDesc = TensorDesc({4, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto xDesc = TensorDesc({4, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    auto dxOutDesc = TensorDesc({4, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnSwigluGroupGrad, INPUT(dyDesc, xDesc, nullptr, nullptr, nullptr, 0.0f),
                        OUTPUT(dxOutDesc, nullptr));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_swiglu_group_grad_test, l2_abnormal_x_dim1_not_2h)
{
    auto dyDesc = TensorDesc({4, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    auto xDesc = TensorDesc({4, 33}, ACL_FLOAT, ACL_FORMAT_ND);
    auto dxOutDesc = TensorDesc({4, 33}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnSwigluGroupGrad, INPUT(dyDesc, xDesc, nullptr, nullptr, nullptr, 0.0f),
                        OUTPUT(dxOutDesc, nullptr));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_swiglu_group_grad_test, l2_abnormal_dy_1d)
{
    auto dyDesc = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND);
    auto xDesc = TensorDesc({32}, ACL_FLOAT, ACL_FORMAT_ND);
    auto dxOutDesc = TensorDesc({32}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnSwigluGroupGrad, INPUT(dyDesc, xDesc, nullptr, nullptr, nullptr, 0.0f),
                        OUTPUT(dxOutDesc, nullptr));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_swiglu_group_grad_test, l2_abnormal_zero_hidden_size)
{
    auto dyDesc = TensorDesc({4, 0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto xDesc = TensorDesc({4, 0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto dxOutDesc = TensorDesc({4, 0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnSwigluGroupGrad, INPUT(dyDesc, xDesc, nullptr, nullptr, nullptr, 0.0f),
                        OUTPUT(dxOutDesc, nullptr));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_swiglu_group_grad_test, l2_abnormal_topk_dtype_int32)
{
    auto dyDesc = TensorDesc({4, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    auto xDesc = TensorDesc({4, 32}, ACL_FLOAT, ACL_FORMAT_ND);
    auto weightDesc = TensorDesc({4, 1}, ACL_INT32, ACL_FORMAT_ND);
    auto yOriginDesc = TensorDesc({4, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    auto dxOutDesc = TensorDesc({4, 32}, ACL_FLOAT, ACL_FORMAT_ND);
    auto dWeightDesc = TensorDesc({4, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnSwigluGroupGrad, INPUT(dyDesc, xDesc, weightDesc, yOriginDesc, nullptr, 0.0f),
                        OUTPUT(dxOutDesc, dWeightDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_swiglu_group_grad_test, l2_abnormal_avail_token_dtype_float)
{
    auto dyDesc = TensorDesc({4, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    auto xDesc = TensorDesc({4, 32}, ACL_FLOAT, ACL_FORMAT_ND);
    auto availTokenDesc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto dxOutDesc = TensorDesc({4, 32}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnSwigluGroupGrad, INPUT(dyDesc, xDesc, nullptr, nullptr, availTokenDesc, 0.0f),
                        OUTPUT(dxOutDesc, nullptr));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_swiglu_group_grad_test, l2_abnormal_empty_group_index)
{
    auto dyDesc = TensorDesc({4, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    auto xDesc = TensorDesc({4, 32}, ACL_FLOAT, ACL_FORMAT_ND);
    auto groupIndexDesc = TensorDesc({0}, ACL_INT64, ACL_FORMAT_ND);
    auto dxOutDesc = TensorDesc({4, 32}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnSwigluGroupGrad, INPUT(dyDesc, xDesc, nullptr, nullptr, groupIndexDesc, 0.0f),
                        OUTPUT(dxOutDesc, nullptr));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_swiglu_group_grad_test, l2_abnormal_weight_without_y_origin)
{
    auto dyDesc = TensorDesc({4, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    auto xDesc = TensorDesc({4, 32}, ACL_FLOAT, ACL_FORMAT_ND);
    auto weightDesc = TensorDesc({4, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto dxOutDesc = TensorDesc({4, 32}, ACL_FLOAT, ACL_FORMAT_ND);
    auto dWeightDesc = TensorDesc({4, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnSwigluGroupGrad, INPUT(dyDesc, xDesc, weightDesc, nullptr, nullptr, 0.0f),
                        OUTPUT(dxOutDesc, dWeightDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_swiglu_group_grad_test, l2_abnormal_y_origin_without_weight)
{
    auto dyDesc = TensorDesc({4, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    auto xDesc = TensorDesc({4, 32}, ACL_FLOAT, ACL_FORMAT_ND);
    auto yOriginDesc = TensorDesc({4, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    auto dxOutDesc = TensorDesc({4, 32}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnSwigluGroupGrad, INPUT(dyDesc, xDesc, nullptr, yOriginDesc, nullptr, 0.0f),
                        OUTPUT(dxOutDesc, nullptr));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_swiglu_group_grad_test, l2_abnormal_negative_clamp_limit)
{
    auto dyDesc = TensorDesc({4, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    auto xDesc = TensorDesc({4, 32}, ACL_FLOAT, ACL_FORMAT_ND);
    auto dxOutDesc = TensorDesc({4, 32}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnSwigluGroupGrad, INPUT(dyDesc, xDesc, nullptr, nullptr, nullptr, -1.0f),
                        OUTPUT(dxOutDesc, nullptr));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_swiglu_group_grad_test, l2_abnormal_nan_clamp_limit)
{
    auto dyDesc = TensorDesc({4, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    auto xDesc = TensorDesc({4, 32}, ACL_FLOAT, ACL_FORMAT_ND);
    auto dxOutDesc = TensorDesc({4, 32}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnSwigluGroupGrad,
                        INPUT(dyDesc, xDesc, nullptr, nullptr, nullptr, std::numeric_limits<float>::quiet_NaN()),
                        OUTPUT(dxOutDesc, nullptr));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

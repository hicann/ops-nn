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
#include "gtest/gtest.h"

#include "../../../op_host/op_api/aclnn_ascend_anti_quant.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace std;

class l2_ascend_anti_quant_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << "l2_ascend_anti_quant_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "l2_ascend_anti_quant_test TearDown" << endl;
    }
};

TEST_F(l2_ascend_anti_quant_test, ascend910B2_normal_1)
{
    auto tensor_1_desc = TensorDesc({3, 5}, ACL_INT8, ACL_FORMAT_ND)
                             .ValueRange(-2, 2)
                             .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
    auto tensor_scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_offset = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc({3, 5}, ACL_FLOAT16, ACL_FORMAT_ND);
    int dstType = ACL_FLOAT16;
    bool sqrtMode = false;

    auto ut = OP_API_UT(
        aclnnAscendAntiQuant, INPUT(tensor_1_desc, tensor_scale, tensor_offset, dstType, sqrtMode),
        OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_ascend_anti_quant_test, ascend910B2_normal_3)
{
    auto tensor_1_desc = TensorDesc({3, 5}, ACL_INT8, ACL_FORMAT_ND)
                             .ValueRange(-2, 2)
                             .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
    auto tensor_scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc({3, 5}, ACL_FLOAT16, ACL_FORMAT_ND);
    int dstType = ACL_FLOAT16;
    bool sqrtMode = false;

    auto ut = OP_API_UT(
        aclnnAscendAntiQuant, INPUT(tensor_1_desc, tensor_scale, (aclTensor*)nullptr, dstType, sqrtMode),
        OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_ascend_anti_quant_test, ascend910B2_normal_2)
{
    auto tensor_1_desc = TensorDesc({3, 5}, ACL_INT8, ACL_FORMAT_ND)
                             .ValueRange(-2, 2)
                             .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
    auto tensor_scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_offset = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc({3, 5}, ACL_BF16, ACL_FORMAT_ND);
    int dstType = ACL_BF16;
    bool sqrtMode = false;

    auto ut = OP_API_UT(
        aclnnAscendAntiQuant, INPUT(tensor_1_desc, tensor_scale, tensor_offset, dstType, sqrtMode),
        OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_ascend_anti_quant_test, ascend910B2_normal_4)
{
    auto tensor_1_desc = TensorDesc({3, 5}, ACL_INT8, ACL_FORMAT_ND)
                             .ValueRange(-2, 2)
                             .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
    auto tensor_scale = TensorDesc({1}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_offset = TensorDesc({1}, ACL_BF16, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc({3, 5}, ACL_BF16, ACL_FORMAT_ND);
    int dstType = ACL_BF16;
    bool sqrtMode = false;

    auto ut = OP_API_UT(
        aclnnAscendAntiQuant, INPUT(tensor_1_desc, tensor_scale, tensor_offset, dstType, sqrtMode),
        OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_ascend_anti_quant_test, ascend910B2_normal_5)
{
    auto tensor_1_desc = TensorDesc({3, 5}, ACL_INT8, ACL_FORMAT_ND)
                             .ValueRange(-2, 2)
                             .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
    auto tensor_scale = TensorDesc({1}, ACL_BF16, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc({3, 5}, ACL_BF16, ACL_FORMAT_ND);
    int dstType = ACL_BF16;
    bool sqrtMode = false;

    auto ut = OP_API_UT(
        aclnnAscendAntiQuant, INPUT(tensor_1_desc, tensor_scale, (aclTensor*)nullptr, dstType, sqrtMode),
        OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_ascend_anti_quant_test, ascend910B2_int4)
{
    auto tensor_1_desc = TensorDesc({3, 4}, ACL_INT4, ACL_FORMAT_ND)
                             .ValueRange(-2, 2)
                             .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    auto tensor_scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_offset = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc({3, 4}, ACL_FLOAT16, ACL_FORMAT_ND);
    int dstType = ACL_FLOAT16;
    bool sqrtMode = false;

    auto ut = OP_API_UT(
        aclnnAscendAntiQuant, INPUT(tensor_1_desc, tensor_scale, tensor_offset, dstType, sqrtMode),
        OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_ascend_anti_quant_test, ascend910B2_int32)
{
    auto tensor_1_desc = TensorDesc({3, 5}, ACL_INT32, ACL_FORMAT_ND)
                             .ValueRange(-2, 2)
                             .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
    auto tensor_scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_offset = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc({3, 40}, ACL_FLOAT16, ACL_FORMAT_ND);
    int dstType = ACL_FLOAT16;
    bool sqrtMode = false;

    auto ut = OP_API_UT(
        aclnnAscendAntiQuant, INPUT(tensor_1_desc, tensor_scale, tensor_offset, dstType, sqrtMode),
        OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_ascend_anti_quant_test, ascend910B2_int32_scalar)
{
    auto tensor_1_desc = TensorDesc({}, ACL_INT32, ACL_FORMAT_ND)
                             .ValueRange(-2, 2)
                             .Value(vector<float>{
                                 1,
                             });
    auto tensor_scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_offset = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc(
        {
            8,
        },
        ACL_FLOAT16, ACL_FORMAT_ND);
    int dstType = ACL_FLOAT16;
    bool sqrtMode = false;

    auto ut = OP_API_UT(
        aclnnAscendAntiQuant, INPUT(tensor_1_desc, tensor_scale, tensor_offset, dstType, sqrtMode),
        OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_ascend_anti_quant_test, ascend910B2_input_out_dtype_dif_1)
{
    auto tensor_1_desc = TensorDesc({3, 5}, ACL_INT8, ACL_FORMAT_ND)
                             .ValueRange(-2, 2)
                             .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
    auto tensor_scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_offset = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc({3, 5}, ACL_BF16, ACL_FORMAT_ND);
    int dstType = ACL_FLOAT16;
    bool sqrtMode = false;

    auto ut = OP_API_UT(
        aclnnAscendAntiQuant, INPUT(tensor_1_desc, tensor_scale, tensor_offset, dstType, sqrtMode),
        OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_ascend_anti_quant_test, ascend910B2_input_out_dtype_dif_2)
{
    auto tensor_1_desc = TensorDesc({3, 5}, ACL_INT8, ACL_FORMAT_ND)
                             .ValueRange(-2, 2)
                             .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
    auto tensor_scale = TensorDesc({1}, ACL_BF16, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc({3, 5}, ACL_BF16, ACL_FORMAT_ND);
    int dstType = ACL_FLOAT16;
    bool sqrtMode = false;

    auto ut = OP_API_UT(
        aclnnAscendAntiQuant, INPUT(tensor_1_desc, tensor_scale, (aclTensor*)nullptr, dstType, sqrtMode),
        OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_ascend_anti_quant_test, ascend910B2_offset_dim_not_1_01)
{
    auto tensor_1_desc = TensorDesc({3, 5}, ACL_INT8, ACL_FORMAT_ND)
                             .ValueRange(-2, 2)
                             .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
    auto tensor_scale = TensorDesc({1, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_offset = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc({3, 5}, ACL_FLOAT16, ACL_FORMAT_ND);
    int dstType = ACL_FLOAT16;
    bool sqrtMode = false;

    auto ut = OP_API_UT(
        aclnnAscendAntiQuant, INPUT(tensor_1_desc, tensor_scale, tensor_offset, dstType, sqrtMode),
        OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_ascend_anti_quant_test, ascend910B2_offset_dim_not_1_02)
{
    auto tensor_1_desc = TensorDesc({3, 5}, ACL_INT8, ACL_FORMAT_ND)
                             .ValueRange(-2, 2)
                             .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
    auto tensor_scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_offset = TensorDesc({1, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc({3, 5}, ACL_FLOAT16, ACL_FORMAT_ND);
    int dstType = ACL_FLOAT16;
    bool sqrtMode = false;

    auto ut = OP_API_UT(
        aclnnAscendAntiQuant, INPUT(tensor_1_desc, tensor_scale, tensor_offset, dstType, sqrtMode),
        OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_ascend_anti_quant_test, ascend910B2_scale_dim_not_1_Nonoffset)
{
    auto tensor_1_desc = TensorDesc({3, 5}, ACL_INT8, ACL_FORMAT_ND)
                             .ValueRange(-2, 2)
                             .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
    auto tensor_scale = TensorDesc({1, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc({3, 5}, ACL_BF16, ACL_FORMAT_ND);
    int dstType = ACL_FLOAT16;
    bool sqrtMode = false;

    auto ut = OP_API_UT(
        aclnnAscendAntiQuant, INPUT(tensor_1_desc, tensor_scale, (aclTensor*)nullptr, dstType, sqrtMode),
        OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_ascend_anti_quant_test, ascend910B2_scale_cannot_broadcast)
{
    auto tensor_1_desc = TensorDesc({3, 5}, ACL_INT8, ACL_FORMAT_ND)
                             .ValueRange(-2, 2)
                             .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
    auto tensor_scale = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_offset = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc({3, 5}, ACL_FLOAT16, ACL_FORMAT_ND);
    int dstType = ACL_FLOAT16;
    bool sqrtMode = false;

    auto ut = OP_API_UT(
        aclnnAscendAntiQuant, INPUT(tensor_1_desc, tensor_scale, tensor_offset, dstType, sqrtMode),
        OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_ascend_anti_quant_test, ascend910B2_empty_tensor)
{
    auto tensor_1_desc = TensorDesc({3, 0}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto tensor_scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_offset = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc({3, 0}, ACL_FLOAT16, ACL_FORMAT_ND);
    int dstType = ACL_FLOAT16;
    bool sqrtMode = false;

    auto ut = OP_API_UT(
        aclnnAscendAntiQuant, INPUT(tensor_1_desc, tensor_scale, tensor_offset, dstType, sqrtMode),
        OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_ascend_anti_quant_test, ascend910A_empty_tensor0)
{
    auto tensor_1_desc = TensorDesc({3, 0}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto tensor_scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_offset = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc({3, 0}, ACL_FLOAT16, ACL_FORMAT_ND);
    int dstType = ACL_FLOAT16;
    bool sqrtMode = false;

    auto ut = OP_API_UT(
        aclnnAscendAntiQuant, INPUT(tensor_1_desc, tensor_scale, tensor_offset, dstType, sqrtMode),
        OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    // EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_ascend_anti_quant_test, ascend910B2_empty_tensor1)
{
    auto tensor_1_desc = TensorDesc({3, 0}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto tensor_scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_offset = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc({3, 0}, ACL_FLOAT16, ACL_FORMAT_ND);
    int dstType = ACL_FLOAT16;
    bool sqrtMode = false;

    auto ut = OP_API_UT(
        aclnnAscendAntiQuant, INPUT(tensor_1_desc, tensor_scale, tensor_offset, dstType, sqrtMode),
        OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_ascend_anti_quant_test, ascend310P_empty_tensor2)
{
    auto tensor_1_desc = TensorDesc({3, 0}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto tensor_scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_offset = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc({3, 0}, ACL_FLOAT16, ACL_FORMAT_ND);
    int dstType = ACL_FLOAT16;
    bool sqrtMode = false;

    auto ut = OP_API_UT(
        aclnnAscendAntiQuant, INPUT(tensor_1_desc, tensor_scale, tensor_offset, dstType, sqrtMode),
        OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    // EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_ascend_anti_quant_test, ascend310P_empty_tensor3)
{
    auto tensor_1_desc = TensorDesc({3, 0}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto tensor_scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_offset = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc({3, 0}, ACL_FLOAT16, ACL_FORMAT_ND);
    int dstType = ACL_BF16;
    bool sqrtMode = false;

    auto ut = OP_API_UT(
        aclnnAscendAntiQuant, INPUT(tensor_1_desc, tensor_scale, tensor_offset, dstType, sqrtMode),
        OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    // EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_ascend_anti_quant_test, ascend310P_check_input_bf16)
{
    auto tensor_1_desc = TensorDesc({3, 5}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto tensor_scale = TensorDesc({1}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_offset = TensorDesc({1}, ACL_BF16, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc({3, 5}, ACL_FLOAT16, ACL_FORMAT_ND);
    int dstType = ACL_FLOAT16;
    bool sqrtMode = false;

    auto ut = OP_API_UT(
        aclnnAscendAntiQuant, INPUT(tensor_1_desc, tensor_scale, tensor_offset, dstType, sqrtMode),
        OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    // EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_ascend_anti_quant_test, ascend310P_check_output_bf16)
{
    auto tensor_1_desc = TensorDesc({3, 5}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto tensor_scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_offset = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc({3, 5}, ACL_BF16, ACL_FORMAT_ND);
    int dstType = ACL_BF16;
    bool sqrtMode = false;

    auto ut = OP_API_UT(
        aclnnAscendAntiQuant, INPUT(tensor_1_desc, tensor_scale, tensor_offset, dstType, sqrtMode),
        OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    // EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_ascend_anti_quant_test, ascend910B2_scale_cannot_broadcast_1)
{
    auto tensor_1_desc = TensorDesc({3, 1}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto tensor_scale = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_offset = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc({3, 1}, ACL_FLOAT16, ACL_FORMAT_ND);
    int dstType = ACL_FLOAT16;
    bool sqrtMode = false;

    auto ut = OP_API_UT(
        aclnnAscendAntiQuant, INPUT(tensor_1_desc, tensor_scale, tensor_offset, dstType, sqrtMode),
        OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_ascend_anti_quant_test, ascend950PR_normal_1)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto tensor_1_desc = TensorDesc({3, 5}, ACL_HIFLOAT8, ACL_FORMAT_ND)
                             .ValueRange(-2, 2)
                             .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
    auto tensor_scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_offset = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc({3, 5}, ACL_FLOAT16, ACL_FORMAT_ND);
    int dstType = ACL_FLOAT16;
    bool sqrtMode = false;

    auto ut = OP_API_UT(
        aclnnAscendAntiQuant, INPUT(tensor_1_desc, tensor_scale, tensor_offset, dstType, sqrtMode),
        OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_ascend_anti_quant_test, ascend950PR_normal_2)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto tensor_1_desc = TensorDesc({3, 5}, ACL_HIFLOAT8, ACL_FORMAT_ND)
                             .ValueRange(-2, 2)
                             .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
    auto tensor_scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc({3, 5}, ACL_BF16, ACL_FORMAT_ND);
    int dstType = ACL_BF16;
    bool sqrtMode = false;

    auto ut = OP_API_UT(
        aclnnAscendAntiQuant, INPUT(tensor_1_desc, tensor_scale, (aclTensor*)nullptr, dstType, sqrtMode),
        OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_ascend_anti_quant_test, ascend950PR_normal_3)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto tensor_1_desc = TensorDesc({3, 5}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND)
                             .ValueRange(-2, 2)
                             .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
    auto tensor_scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_offset = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc({3, 5}, ACL_FLOAT16, ACL_FORMAT_ND);
    int dstType = ACL_FLOAT16;
    bool sqrtMode = false;

    auto ut = OP_API_UT(
        aclnnAscendAntiQuant, INPUT(tensor_1_desc, tensor_scale, tensor_offset, dstType, sqrtMode),
        OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_ascend_anti_quant_test, ascend950PR_normal_4)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto tensor_1_desc = TensorDesc({3, 5}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND)
                             .ValueRange(-2, 2)
                             .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
    auto tensor_scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc({3, 5}, ACL_BF16, ACL_FORMAT_ND);
    int dstType = ACL_BF16;
    bool sqrtMode = false;

    auto ut = OP_API_UT(
        aclnnAscendAntiQuant, INPUT(tensor_1_desc, tensor_scale, (aclTensor*)nullptr, dstType, sqrtMode),
        OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_ascend_anti_quant_test, ascend950PR_normal_5)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto tensor_1_desc = TensorDesc({3, 5}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND)
                             .ValueRange(-2, 2)
                             .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
    auto tensor_scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_offset = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc({3, 5}, ACL_FLOAT16, ACL_FORMAT_ND);
    int dstType = ACL_FLOAT16;
    bool sqrtMode = false;

    auto ut = OP_API_UT(
        aclnnAscendAntiQuant, INPUT(tensor_1_desc, tensor_scale, tensor_offset, dstType, sqrtMode),
        OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_ascend_anti_quant_test, ascend950PR_normal_6)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto tensor_1_desc = TensorDesc({3, 5}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND)
                             .ValueRange(-2, 2)
                             .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
    auto tensor_scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc({3, 5}, ACL_BF16, ACL_FORMAT_ND);
    int dstType = ACL_BF16;
    bool sqrtMode = false;

    auto ut = OP_API_UT(
        aclnnAscendAntiQuant, INPUT(tensor_1_desc, tensor_scale, (aclTensor*)nullptr, dstType, sqrtMode),
        OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_ascend_anti_quant_test, ascend950PR_normal_7)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto tensor_1_desc = TensorDesc({3, 5}, ACL_INT8, ACL_FORMAT_ND)
                             .ValueRange(-2, 2)
                             .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
    auto tensor_scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_offset = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc({3, 5}, ACL_FLOAT16, ACL_FORMAT_ND);
    int dstType = ACL_FLOAT16;
    bool sqrtMode = false;

    auto ut = OP_API_UT(
        aclnnAscendAntiQuant, INPUT(tensor_1_desc, tensor_scale, tensor_offset, dstType, sqrtMode),
        OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_ascend_anti_quant_test, ascend950PR_normal_8)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto tensor_1_desc = TensorDesc({3, 5}, ACL_INT8, ACL_FORMAT_ND)
                             .ValueRange(-2, 2)
                             .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
    auto tensor_scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc({3, 5}, ACL_FLOAT16, ACL_FORMAT_ND);
    int dstType = ACL_FLOAT16;
    bool sqrtMode = false;

    auto ut = OP_API_UT(
        aclnnAscendAntiQuant, INPUT(tensor_1_desc, tensor_scale, (aclTensor*)nullptr, dstType, sqrtMode),
        OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_ascend_anti_quant_test, ascend950PR_normal_9)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto tensor_1_desc = TensorDesc({3, 5}, ACL_INT8, ACL_FORMAT_ND)
                             .ValueRange(-2, 2)
                             .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
    auto tensor_scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_offset = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc({3, 5}, ACL_BF16, ACL_FORMAT_ND);
    int dstType = ACL_BF16;
    bool sqrtMode = false;

    auto ut = OP_API_UT(
        aclnnAscendAntiQuant, INPUT(tensor_1_desc, tensor_scale, tensor_offset, dstType, sqrtMode),
        OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_ascend_anti_quant_test, ascend950PR_normal_10)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto tensor_1_desc = TensorDesc({3, 5}, ACL_INT8, ACL_FORMAT_ND)
                             .ValueRange(-2, 2)
                             .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
    auto tensor_scale = TensorDesc({1}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_offset = TensorDesc({1}, ACL_BF16, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc({3, 5}, ACL_BF16, ACL_FORMAT_ND);
    int dstType = ACL_BF16;
    bool sqrtMode = true;

    auto ut = OP_API_UT(
        aclnnAscendAntiQuant, INPUT(tensor_1_desc, tensor_scale, tensor_offset, dstType, sqrtMode),
        OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_ascend_anti_quant_test, ascend950PR_normal_11)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto tensor_1_desc = TensorDesc({3, 5}, ACL_INT8, ACL_FORMAT_ND)
                             .ValueRange(-2, 2)
                             .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
    auto tensor_scale = TensorDesc({1}, ACL_BF16, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc({3, 5}, ACL_BF16, ACL_FORMAT_ND);
    int dstType = ACL_BF16;
    bool sqrtMode = true;

    auto ut = OP_API_UT(
        aclnnAscendAntiQuant, INPUT(tensor_1_desc, tensor_scale, (aclTensor*)nullptr, dstType, sqrtMode),
        OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_ascend_anti_quant_test, ascend950PR_int4)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto tensor_1_desc = TensorDesc({3, 4}, ACL_INT4, ACL_FORMAT_ND)
                             .ValueRange(-2, 2)
                             .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    auto tensor_scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_offset = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc({3, 4}, ACL_FLOAT16, ACL_FORMAT_ND);
    int dstType = ACL_FLOAT16;
    bool sqrtMode = false;

    auto ut = OP_API_UT(
        aclnnAscendAntiQuant, INPUT(tensor_1_desc, tensor_scale, tensor_offset, dstType, sqrtMode),
        OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_ascend_anti_quant_test, ascend950PR_int32)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto tensor_1_desc = TensorDesc({3, 5}, ACL_INT32, ACL_FORMAT_ND)
                             .ValueRange(-2, 2)
                             .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
    auto tensor_scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_offset = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc({3, 40}, ACL_FLOAT16, ACL_FORMAT_ND);
    int dstType = ACL_FLOAT16;
    bool sqrtMode = false;

    auto ut = OP_API_UT(
        aclnnAscendAntiQuant, INPUT(tensor_1_desc, tensor_scale, tensor_offset, dstType, sqrtMode),
        OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_ascend_anti_quant_test, ascend950PR_scale_dim_not_1_01)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto tensor_1_desc = TensorDesc({3, 5}, ACL_HIFLOAT8, ACL_FORMAT_ND)
                             .ValueRange(-2, 2)
                             .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
    auto tensor_scale = TensorDesc({1, 5}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_offset = TensorDesc({1, 5}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc({3, 5}, ACL_FLOAT16, ACL_FORMAT_ND);
    int dstType = ACL_FLOAT16;
    bool sqrtMode = false;

    auto ut = OP_API_UT(
        aclnnAscendAntiQuant, INPUT(tensor_1_desc, tensor_scale, tensor_offset, dstType, sqrtMode),
        OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_ascend_anti_quant_test, ascend950PR_per_head_1)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto tensor_1_desc = TensorDesc({3, 5}, ACL_HIFLOAT8, ACL_FORMAT_ND)
                             .ValueRange(-2, 2)
                             .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
    auto tensor_scale = TensorDesc({3, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_offset = TensorDesc({3, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc({3, 5}, ACL_BF16, ACL_FORMAT_ND);
    int dstType = ACL_BF16;
    bool sqrtMode = false;

    auto ut = OP_API_UT(
        aclnnAscendAntiQuant, INPUT(tensor_1_desc, tensor_scale, tensor_offset, dstType, sqrtMode),
        OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_ascend_anti_quant_test, ascend950PR_per_head_2)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto tensor_1_desc = TensorDesc({3, 4}, ACL_INT4, ACL_FORMAT_ND)
                             .ValueRange(-2, 2);
    auto tensor_scale = TensorDesc({3}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_offset = TensorDesc({3}, ACL_BF16, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc({3, 4}, ACL_BF16, ACL_FORMAT_ND);
    int dstType = ACL_BF16;
    bool sqrtMode = true;

    auto ut = OP_API_UT(
        aclnnAscendAntiQuant, INPUT(tensor_1_desc, tensor_scale, tensor_offset, dstType, sqrtMode),
        OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_ascend_anti_quant_test, ascend950PR_int32_scalar)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto tensor_1_desc = TensorDesc({}, ACL_INT32, ACL_FORMAT_ND)
                             .ValueRange(-2, 2)
                             .Value(vector<float>{
                                 1,
                             });
    auto tensor_scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_offset = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc(
        {
            8,
        },
        ACL_FLOAT16, ACL_FORMAT_ND);
    int dstType = ACL_FLOAT16;
    bool sqrtMode = false;

    auto ut = OP_API_UT(
        aclnnAscendAntiQuant, INPUT(tensor_1_desc, tensor_scale, tensor_offset, dstType, sqrtMode),
        OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_ascend_anti_quant_test, ascend950PR_hifloat8_scalar)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto tensor_1_desc = TensorDesc({}, ACL_HIFLOAT8, ACL_FORMAT_ND)
                             .ValueRange(-2, 2)
                             .Value(vector<float>{
                                 1,
                             });
    auto tensor_scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_offset = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc(
        {
            8,
        },
        ACL_FLOAT16, ACL_FORMAT_ND);
    int dstType = ACL_FLOAT16;
    bool sqrtMode = false;

    auto ut = OP_API_UT(
        aclnnAscendAntiQuant, INPUT(tensor_1_desc, tensor_scale, tensor_offset, dstType, sqrtMode),
        OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_ascend_anti_quant_test, ascend950PR_input_out_dtype_dif_1)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto tensor_1_desc = TensorDesc({3, 5}, ACL_INT8, ACL_FORMAT_ND)
                             .ValueRange(-2, 2)
                             .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
    auto tensor_scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_offset = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc({3, 5}, ACL_BF16, ACL_FORMAT_ND);
    int dstType = ACL_FLOAT16;
    bool sqrtMode = false;

    auto ut = OP_API_UT(
        aclnnAscendAntiQuant, INPUT(tensor_1_desc, tensor_scale, tensor_offset, dstType, sqrtMode),
        OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_ascend_anti_quant_test, ascend950PR_input_out_dtype_dif_2)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto tensor_1_desc = TensorDesc({3, 5}, ACL_INT8, ACL_FORMAT_ND)
                             .ValueRange(-2, 2)
                             .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
    auto tensor_scale = TensorDesc({1}, ACL_BF16, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc({3, 5}, ACL_BF16, ACL_FORMAT_ND);
    int dstType = ACL_FLOAT16;
    bool sqrtMode = false;

    auto ut = OP_API_UT(
        aclnnAscendAntiQuant, INPUT(tensor_1_desc, tensor_scale, (aclTensor*)nullptr, dstType, sqrtMode),
        OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_ascend_anti_quant_test, ascend950PR_offset_dim_not_1)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto tensor_1_desc = TensorDesc({3, 5}, ACL_INT8, ACL_FORMAT_ND)
                             .ValueRange(-2, 2)
                             .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
    auto tensor_scale = TensorDesc({1, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_offset = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc({3, 5}, ACL_FLOAT16, ACL_FORMAT_ND);
    int dstType = ACL_FLOAT16;
    bool sqrtMode = false;

    auto ut = OP_API_UT(
        aclnnAscendAntiQuant, INPUT(tensor_1_desc, tensor_scale, tensor_offset, dstType, sqrtMode),
        OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_ascend_anti_quant_test, ascend950PR_scale_dim_not_1_Nonoffset)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto tensor_1_desc = TensorDesc({3, 5}, ACL_INT8, ACL_FORMAT_ND)
                             .ValueRange(-2, 2)
                             .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
    auto tensor_scale = TensorDesc({1, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc({3, 5}, ACL_BF16, ACL_FORMAT_ND);
    int dstType = ACL_FLOAT16;
    bool sqrtMode = false;

    auto ut = OP_API_UT(
        aclnnAscendAntiQuant, INPUT(tensor_1_desc, tensor_scale, (aclTensor*)nullptr, dstType, sqrtMode),
        OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_ascend_anti_quant_test, ascend950PR_empty_tensor)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto tensor_1_desc = TensorDesc({3, 0}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto tensor_scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_offset = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc({3, 0}, ACL_FLOAT16, ACL_FORMAT_ND);
    int dstType = ACL_FLOAT16;
    bool sqrtMode = false;

    auto ut = OP_API_UT(
        aclnnAscendAntiQuant, INPUT(tensor_1_desc, tensor_scale, tensor_offset, dstType, sqrtMode),
        OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}
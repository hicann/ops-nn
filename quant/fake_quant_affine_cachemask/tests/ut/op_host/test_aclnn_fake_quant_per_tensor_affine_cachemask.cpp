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

#include "../../../op_host/op_api/aclnn_fake_quant_per_tensor_affine_cachemask.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"

using namespace std;

class fake_quant_per_tensor_affine_cachemask_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << "fake_quant_per_tensor_affine_cachemask_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "fake_quant_per_tensor_affine_cachemask_test TearDown" << endl;
    }
};

TEST_F(fake_quant_per_tensor_affine_cachemask_test, case_norm_float32)
{
    auto selfDesc = TensorDesc({1, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto scaleDesc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto zeroPointDesc = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND);

    auto outDesc = TensorDesc({1, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto maskDesc = TensorDesc({1, 2}, ACL_BOOL, ACL_FORMAT_ND);

    float fake_quant_enbled = 0;
    int64_t quant_min = -10;
    int64_t quant_max = 10;

    auto ut = OP_API_UT(
        aclnnFakeQuantPerTensorAffineCachemask,
        INPUT(selfDesc, scaleDesc, zeroPointDesc, fake_quant_enbled, quant_min, quant_max), OUTPUT(outDesc, maskDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(fake_quant_per_tensor_affine_cachemask_test, case_norm_fake_quant_enabled)
{
    auto selfDesc = TensorDesc({1, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto scaleDesc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto zeroPointDesc = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND);

    auto outDesc = TensorDesc({1, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto maskDesc = TensorDesc({1, 2}, ACL_BOOL, ACL_FORMAT_ND);

    float fake_quant_enbled = 2.0;
    int64_t quant_min = -10;
    int64_t quant_max = 10;

    auto ut = OP_API_UT(
        aclnnFakeQuantPerTensorAffineCachemask,
        INPUT(selfDesc, scaleDesc, zeroPointDesc, fake_quant_enbled, quant_min, quant_max), OUTPUT(outDesc, maskDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    // EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(fake_quant_per_tensor_affine_cachemask_test, case_norm_float16)
{
    auto selfDesc = TensorDesc({1, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto scaleDesc = TensorDesc({1}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto zeroPointDesc = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND);

    auto outDesc = TensorDesc({1, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto maskDesc = TensorDesc({1, 2}, ACL_BOOL, ACL_FORMAT_ND);

    float fake_quant_enbled = 0;
    int64_t quant_min = -10;
    int64_t quant_max = 10;

    auto ut = OP_API_UT(
        aclnnFakeQuantPerTensorAffineCachemask,
        INPUT(selfDesc, scaleDesc, zeroPointDesc, fake_quant_enbled, quant_min, quant_max), OUTPUT(outDesc, maskDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(fake_quant_per_tensor_affine_cachemask_test, case_norm_int32)
{
    auto selfDesc = TensorDesc({1, 2}, ACL_INT32, ACL_FORMAT_ND);
    auto scaleDesc = TensorDesc({1}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto zeroPointDesc = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND);

    auto outDesc = TensorDesc({1, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto maskDesc = TensorDesc({1, 2}, ACL_BOOL, ACL_FORMAT_ND);

    float fake_quant_enbled = 0;
    int64_t quant_min = -10;
    int64_t quant_max = 10;

    auto ut = OP_API_UT(
        aclnnFakeQuantPerTensorAffineCachemask,
        INPUT(selfDesc, scaleDesc, zeroPointDesc, fake_quant_enbled, quant_min, quant_max), OUTPUT(outDesc, maskDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_NE(aclRet, ACLNN_SUCCESS);
}

TEST_F(fake_quant_per_tensor_affine_cachemask_test, case_norm_int8)
{
    auto selfDesc = TensorDesc({1, 2}, ACL_INT8, ACL_FORMAT_ND);
    auto scaleDesc = TensorDesc({1}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto zeroPointDesc = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND);

    auto outDesc = TensorDesc({1, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto maskDesc = TensorDesc({1, 2}, ACL_BOOL, ACL_FORMAT_ND);

    float fake_quant_enbled = 0;
    int64_t quant_min = -10;
    int64_t quant_max = 10;

    auto ut = OP_API_UT(
        aclnnFakeQuantPerTensorAffineCachemask,
        INPUT(selfDesc, scaleDesc, zeroPointDesc, fake_quant_enbled, quant_min, quant_max), OUTPUT(outDesc, maskDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_NE(aclRet, ACLNN_SUCCESS);
}

TEST_F(fake_quant_per_tensor_affine_cachemask_test, case_norm_uint8)
{
    auto selfDesc = TensorDesc({1, 2}, ACL_UINT8, ACL_FORMAT_ND);
    auto scaleDesc = TensorDesc({1}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto zeroPointDesc = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND);

    auto outDesc = TensorDesc({1, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto maskDesc = TensorDesc({1, 2}, ACL_BOOL, ACL_FORMAT_ND);

    float fake_quant_enbled = 0;
    int64_t quant_min = -10;
    int64_t quant_max = 10;

    auto ut = OP_API_UT(
        aclnnFakeQuantPerTensorAffineCachemask,
        INPUT(selfDesc, scaleDesc, zeroPointDesc, fake_quant_enbled, quant_min, quant_max), OUTPUT(outDesc, maskDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_NE(aclRet, ACLNN_SUCCESS);
}

TEST_F(fake_quant_per_tensor_affine_cachemask_test, case_norm_int16)
{
    auto selfDesc = TensorDesc({1, 2}, ACL_INT16, ACL_FORMAT_ND);
    auto scaleDesc = TensorDesc({1}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto zeroPointDesc = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND);

    auto outDesc = TensorDesc({1, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto maskDesc = TensorDesc({1, 2}, ACL_BOOL, ACL_FORMAT_ND);

    float fake_quant_enbled = 0;
    int64_t quant_min = -10;
    int64_t quant_max = 10;

    auto ut = OP_API_UT(
        aclnnFakeQuantPerTensorAffineCachemask,
        INPUT(selfDesc, scaleDesc, zeroPointDesc, fake_quant_enbled, quant_min, quant_max), OUTPUT(outDesc, maskDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_NE(aclRet, ACLNN_SUCCESS);
}

TEST_F(fake_quant_per_tensor_affine_cachemask_test, case_dtype_invalid_out)
{
    auto selfDesc = TensorDesc({1, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto scaleDesc = TensorDesc({1}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto zeroPointDesc = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND);

    auto outDesc = TensorDesc({1, 2}, ACL_INT32, ACL_FORMAT_ND);
    auto maskDesc = TensorDesc({1, 2}, ACL_BOOL, ACL_FORMAT_ND);

    float fake_quant_enbled = 0;
    int64_t quant_min = -10;
    int64_t quant_max = 10;

    auto ut = OP_API_UT(
        aclnnFakeQuantPerTensorAffineCachemask,
        INPUT(selfDesc, scaleDesc, zeroPointDesc, fake_quant_enbled, quant_min, quant_max), OUTPUT(outDesc, maskDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    // EXPECT_NE(aclRet, ACLNN_SUCCESS);
}

TEST_F(fake_quant_per_tensor_affine_cachemask_test, case_shape_invalid_out)
{
    auto selfDesc = TensorDesc({1, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto scaleDesc = TensorDesc({1}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto zeroPointDesc = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND);

    auto outDesc = TensorDesc({2, 2}, ACL_INT32, ACL_FORMAT_ND);
    auto maskDesc = TensorDesc({1, 2}, ACL_BOOL, ACL_FORMAT_ND);

    float fake_quant_enbled = 0;
    int64_t quant_min = -10;
    int64_t quant_max = 10;

    auto ut = OP_API_UT(
        aclnnFakeQuantPerTensorAffineCachemask,
        INPUT(selfDesc, scaleDesc, zeroPointDesc, fake_quant_enbled, quant_min, quant_max), OUTPUT(outDesc, maskDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_NE(aclRet, ACLNN_SUCCESS);
}

TEST_F(fake_quant_per_tensor_affine_cachemask_test, case_quant_val)
{
    auto selfDesc = TensorDesc({1, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto scaleDesc = TensorDesc({1}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto zeroPointDesc = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND);

    auto outDesc = TensorDesc({2, 2}, ACL_INT32, ACL_FORMAT_ND);
    auto maskDesc = TensorDesc({1, 2}, ACL_BOOL, ACL_FORMAT_ND);

    float fake_quant_enbled = 0;
    int64_t quant_min = 10;
    int64_t quant_max = -10;

    auto ut = OP_API_UT(
        aclnnFakeQuantPerTensorAffineCachemask,
        INPUT(selfDesc, scaleDesc, zeroPointDesc, fake_quant_enbled, quant_min, quant_max), OUTPUT(outDesc, maskDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_NE(aclRet, ACLNN_SUCCESS);
}

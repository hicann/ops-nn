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

#include "../../../op_host/op_api/aclnn_ascend_quant.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"

using namespace std;

class l2_ascend_quant_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "l2_ascend_quant_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "l2_ascend_quant_test TearDown" << endl;
    }
};

TEST_F(l2_ascend_quant_test, ascend910B2_normal_1)
{
    auto tensor_1_desc = TensorDesc({3, 5}, ACL_FLOAT, ACL_FORMAT_ND)
                             .ValueRange(-2, 2)
                             .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
    auto tensor_scale = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_offset = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc({3, 5}, ACL_INT8, ACL_FORMAT_ND);
    int32_t dstType = ACL_INT8;
    bool sqrtMode = false;
    const char* roundMode = "round";

    auto ut = OP_API_UT(
        aclnnAscendQuant, INPUT(tensor_1_desc, tensor_scale, tensor_offset, sqrtMode, roundMode, dstType),
        OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);

    // ut.TestPrecision();
}

TEST_F(l2_ascend_quant_test, ascend910B2_normal_2)
{
    auto tensor_1_desc = TensorDesc({3, 5}, ACL_BF16, ACL_FORMAT_ND)
                             .ValueRange(-2, 2)
                             .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
    auto tensor_scale = TensorDesc({5}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_offset = TensorDesc({5}, ACL_BF16, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc({3, 5}, ACL_INT8, ACL_FORMAT_ND);
    int32_t dstType = ACL_INT8;
    bool sqrtMode = false;
    const char* roundMode = "round";

    auto ut = OP_API_UT(
        aclnnAscendQuant, INPUT(tensor_1_desc, tensor_scale, tensor_offset, sqrtMode, roundMode, dstType),
        OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);

    // ut.TestPrecision();
}

TEST_F(l2_ascend_quant_test, ascend910B2_normal_3)
{
    auto tensor_1_desc = TensorDesc({3, 5}, ACL_FLOAT16, ACL_FORMAT_ND)
                             .ValueRange(-2, 2)
                             .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
    auto tensor_scale = TensorDesc({1, 5}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_offset = TensorDesc({1, 5}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc({3, 5}, ACL_INT8, ACL_FORMAT_ND);
    int32_t dstType = ACL_INT8;
    const bool sqrtMode = false;
    const char* roundMode = "round";

    auto ut = OP_API_UT(
        aclnnAscendQuant, INPUT(tensor_1_desc, tensor_scale, tensor_offset, sqrtMode, roundMode, dstType),
        OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);

    // ut.TestPrecision();
}

TEST_F(l2_ascend_quant_test, ascend910B2_normal_4)
{
    auto tensor_1_desc = TensorDesc({3, 5}, ACL_FLOAT16, ACL_FORMAT_ND)
                             .ValueRange(-2, 2)
                             .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
    auto tensor_scale = TensorDesc({1, 5}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc({3, 5}, ACL_INT8, ACL_FORMAT_ND);
    int32_t dstType = ACL_INT8;
    const bool sqrtMode = false;
    const char* roundMode = "round";

    auto ut = OP_API_UT(
        aclnnAscendQuant, INPUT(tensor_1_desc, tensor_scale, nullptr, sqrtMode, roundMode, dstType),
        OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);

    // ut.TestPrecision();
}

TEST_F(l2_ascend_quant_test, ascend310P_normal_1)
{
    auto tensor_1_desc = TensorDesc({3, 5}, ACL_FLOAT, ACL_FORMAT_ND)
                             .ValueRange(-2, 2)
                             .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
    auto tensor_scale = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_offset = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc({3, 5}, ACL_INT8, ACL_FORMAT_ND);
    int32_t dstType = ACL_INT8;
    bool sqrtMode = false;
    const char* roundMode = "round";

    auto ut = OP_API_UT(
        aclnnAscendQuant, INPUT(tensor_1_desc, tensor_scale, tensor_offset, sqrtMode, roundMode, dstType),
        OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);

    // ut.TestPrecision();
}

TEST_F(l2_ascend_quant_test, ascend910B2_input_dtype_dif_1)
{
    auto tensor_1_desc = TensorDesc({3, 5}, ACL_FLOAT16, ACL_FORMAT_ND)
                             .ValueRange(-2, 2)
                             .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
    auto tensor_scale = TensorDesc({1}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_offset = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc({3, 5}, ACL_INT8, ACL_FORMAT_ND);
    int32_t dstType = ACL_INT8;
    bool sqrtMode = false;
    const char* roundMode = "round";

    auto ut = OP_API_UT(
        aclnnAscendQuant, INPUT(tensor_1_desc, tensor_scale, tensor_offset, sqrtMode, roundMode, dstType),
        OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_ascend_quant_test, ascend910B2_input_dtype_dif_2)
{
    auto tensor_1_desc = TensorDesc({3, 5}, ACL_FLOAT16, ACL_FORMAT_ND)
                             .ValueRange(-2, 2)
                             .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
    auto tensor_scale = TensorDesc({2, 5}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_offset = TensorDesc({5}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc({3, 5}, ACL_INT8, ACL_FORMAT_ND);
    int32_t dstType = ACL_INT8;
    bool sqrtMode = false;
    const char* roundMode = "round";

    auto ut = OP_API_UT(
        aclnnAscendQuant, INPUT(tensor_1_desc, tensor_scale, tensor_offset, sqrtMode, roundMode, dstType),
        OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_ascend_quant_test, ascend910B2_output_int4)
{
    auto tensor_1_desc = TensorDesc({2, 8}, ACL_FLOAT16, ACL_FORMAT_ND)
                             .ValueRange(-2, 2)
                             .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    auto tensor_scale = TensorDesc({1, 8}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_offset = TensorDesc({1, 8}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc({2, 8}, ACL_INT4, ACL_FORMAT_ND);
    int32_t dstType = ACL_INT4;
    bool sqrtMode = false;
    const char* roundMode = "round";

    auto ut = OP_API_UT(
        aclnnAscendQuant, INPUT(tensor_1_desc, tensor_scale, tensor_offset, sqrtMode, roundMode, dstType),
        OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_ascend_quant_test, ascend910B2_output_int32)
{
    auto tensor_1_desc = TensorDesc({2, 8}, ACL_FLOAT16, ACL_FORMAT_ND)
                             .ValueRange(-2, 2)
                             .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    auto tensor_scale = TensorDesc({1, 8}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_offset = TensorDesc({1, 8}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc({2, 1}, ACL_INT32, ACL_FORMAT_ND);
    int32_t dstType = ACL_INT32;
    bool sqrtMode = false;
    const char* roundMode = "round";

    auto ut = OP_API_UT(
        aclnnAscendQuant, INPUT(tensor_1_desc, tensor_scale, tensor_offset, sqrtMode, roundMode, dstType),
        OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_ascend_quant_test, ascend910B2_output_int32_error_shape)
{
    auto tensor_1_desc = TensorDesc({2, 8}, ACL_FLOAT16, ACL_FORMAT_ND)
                             .ValueRange(-2, 2)
                             .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    auto tensor_scale = TensorDesc({1, 8}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_offset = TensorDesc({1, 8}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc({2, 2}, ACL_INT32, ACL_FORMAT_ND);
    int32_t dstType = ACL_INT32;
    bool sqrtMode = false;
    const char* roundMode = "round";

    auto ut = OP_API_UT(
        aclnnAscendQuant, INPUT(tensor_1_desc, tensor_scale, tensor_offset, sqrtMode, roundMode, dstType),
        OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_ascend_quant_test, ascend910B2_output_int4_scalar_error)
{
    auto tensor_1_desc = TensorDesc({}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2).Value(vector<float>{1});
    auto tensor_scale = TensorDesc({1}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_offset = TensorDesc({1}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc({}, ACL_INT4, ACL_FORMAT_ND);
    int32_t dstType = ACL_INT4;
    bool sqrtMode = false;
    const char* roundMode = "round";

    auto ut = OP_API_UT(
        aclnnAscendQuant, INPUT(tensor_1_desc, tensor_scale, tensor_offset, sqrtMode, roundMode, dstType),
        OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_ascend_quant_test, ascend910B2_output_int32_scalar_error)
{
    auto tensor_1_desc = TensorDesc({}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2).Value(vector<float>{1});
    auto tensor_scale = TensorDesc({1}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_offset = TensorDesc({1}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc({}, ACL_INT32, ACL_FORMAT_ND);
    int32_t dstType = ACL_INT32;
    bool sqrtMode = false;
    const char* roundMode = "round";

    auto ut = OP_API_UT(
        aclnnAscendQuant, INPUT(tensor_1_desc, tensor_scale, tensor_offset, sqrtMode, roundMode, dstType),
        OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_ascend_quant_test, ascend910B_rountMode_error)
{
    auto tensor_1_desc = TensorDesc({3, 5}, ACL_FLOAT, ACL_FORMAT_ND)
                             .ValueRange(-2, 2)
                             .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
    auto tensor_scale = TensorDesc({5}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_offset = TensorDesc({5}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc({3, 5}, ACL_INT8, ACL_FORMAT_ND);
    int32_t dstType = ACL_INT8;
    bool sqrtMode = false;
    const char* roundMode = "r";

    auto ut = OP_API_UT(
        aclnnAscendQuant, INPUT(tensor_1_desc, tensor_scale, tensor_offset, sqrtMode, roundMode, dstType),
        OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_ascend_quant_test, ascend910B_rountMode_is_null_error)
{
    auto tensor_1_desc = TensorDesc({3, 5}, ACL_FLOAT, ACL_FORMAT_ND)
                             .ValueRange(-2, 2)
                             .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
    auto tensor_scale = TensorDesc({5}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_offset = TensorDesc({5}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc({3, 5}, ACL_INT8, ACL_FORMAT_ND);
    int32_t dstType = ACL_INT8;
    bool sqrtMode = false;
    const char* roundMode = nullptr;

    auto ut = OP_API_UT(
        aclnnAscendQuant, INPUT(tensor_1_desc, tensor_scale, tensor_offset, sqrtMode, roundMode, dstType),
        OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}
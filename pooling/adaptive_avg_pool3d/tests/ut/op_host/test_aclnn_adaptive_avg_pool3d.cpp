/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "gtest/gtest.h"
#include <array>
#include <vector>

#include "../../../op_host/op_api/aclnn_adaptive_avg_pool3d.h"

#include "op_api_ut_common/array_desc.h"
#include "op_api_ut_common/inner/types.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace std;

class l2_adaptive_avg_pool3d_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << "l2_adaptive_avg_pool3d_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "l2_adaptive_avg_pool3d_test TearDown" << endl;
    }
};

TEST_F(l2_adaptive_avg_pool3d_test, adaptive_avg_pool3d_outputsize_one_one_one)
{
    vector<aclDataType> dtypes{ACL_FLOAT, ACL_FLOAT16};
    vector<int64_t> output_size = {1, 1, 1};
    for (auto dtype : dtypes) {
        auto input_tensor = TensorDesc({2, 1, 2, 2, 2}, dtype, ACL_FORMAT_ND)
                                .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
        auto int_array_desc = IntArrayDesc(output_size);
        auto output_tensor = TensorDesc({2, 1, 1, 1, 1}, dtype, ACL_FORMAT_ND);
        auto ut = OP_API_UT(aclnnAdaptiveAvgPool3d, INPUT(input_tensor, int_array_desc), OUTPUT(output_tensor));
        uint64_t workspace_size = 0;
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
        EXPECT_EQ(aclRet, ACLNN_SUCCESS);
    }
}

TEST_F(l2_adaptive_avg_pool3d_test, adaptive_avg_pool3d_outputsize_common)
{
    vector<aclDataType> dtypes{ACL_FLOAT};
    vector<int64_t> output_size = {2, 2, 2};
    for (auto dtype : dtypes) {
        auto input_tensor =
            TensorDesc({2, 1, 2, 3, 3}, dtype, ACL_FORMAT_ND).Value(vector<float>{1,  2,  3,  4,  5,  6,  7,  8,  9,
                                                                                  10, 11, 12, 13, 14, 15, 16, 17, 18,
                                                                                  19, 20, 21, 22, 23, 24, 25, 26, 27,
                                                                                  28, 29, 30, 31, 32, 33, 34, 35, 36});
        auto int_array_desc = IntArrayDesc(output_size);
        auto output_tensor = TensorDesc({2, 1, 2, 2, 2}, dtype, ACL_FORMAT_ND);
        auto ut = OP_API_UT(aclnnAdaptiveAvgPool3d, INPUT(input_tensor, int_array_desc), OUTPUT(output_tensor));
        uint64_t workspace_size = 0;
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    }
}

TEST_F(l2_adaptive_avg_pool3d_test, adaptive_avg_pool3d_outputsize_input_discontinuous)
{
    vector<aclDataType> dtypes{ACL_FLOAT};
    vector<int64_t> output_size = {2, 2, 2};
    for (auto dtype : dtypes) {
        auto input_tensor =
            TensorDesc({2, 1, 2, 4, 3}, dtype, ACL_FORMAT_ND, {64, 32, 8, 1, 2}, 0, {2, 2, 4, 8, 8}).ValueRange(40, 50);
        auto int_array_desc = IntArrayDesc(output_size);
        auto output_tensor = TensorDesc({2, 1, 2, 2, 2}, dtype, ACL_FORMAT_ND).Precision(0.1, 0.1);
        auto ut = OP_API_UT(aclnnAdaptiveAvgPool3d, INPUT(input_tensor, int_array_desc), OUTPUT(output_tensor));
        uint64_t workspace_size = 0;
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    }
}

TEST_F(l2_adaptive_avg_pool3d_test, adaptive_avg_pool3d_outputsize_common_DHW_not_same)
{
    vector<aclDataType> dtypes{ACL_FLOAT};
    vector<int64_t> output_size = {3, 2, 1};
    for (auto dtype : dtypes) {
        auto input_tensor =
            TensorDesc({2, 2, 4, 3, 1}, dtype, ACL_FORMAT_ND)
                .Value(vector<float>{0.5,  1.5,  2.5,  3.5,  4.5,  5.5,  6.5,  7.5,  8.5,  9.5,  10.5, 11.5,
                                     12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5, 20.5, 21.5, 22.5, 23.5,
                                     24.5, 25.5, 26.5, 27.5, 28.5, 29.5, 30.5, 31.5, 32.5, 33.5, 34.5, 35.5,
                                     36.5, 37.5, 38.5, 39.5, 40.5, 41.5, 42.5, 43.5, 44.5, 45.5, 46.5, 47.5});
        auto int_array_desc = IntArrayDesc(output_size);
        auto output_tensor = TensorDesc({2, 2, 3, 2, 1}, dtype, ACL_FORMAT_ND);
        auto ut = OP_API_UT(aclnnAdaptiveAvgPool3d, INPUT(input_tensor, int_array_desc), OUTPUT(output_tensor));
        uint64_t workspace_size = 0;
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    }
}

TEST_F(l2_adaptive_avg_pool3d_test, adaptive_avg_pool3d_cdhw_common)
{
    vector<aclDataType> dtypes{ACL_FLOAT};
    vector<int64_t> output_size = {2, 3, 2};
    for (auto dtype : dtypes) {
        auto input_tensor = TensorDesc({2, 2, 4, 3}, dtype, ACL_FORMAT_ND).ValueRange(40, 50);
        auto int_array_desc = IntArrayDesc(output_size);
        auto output_tensor = TensorDesc({2, 2, 3, 2}, dtype, ACL_FORMAT_ND).Precision(0.03, 0.03);
        auto ut = OP_API_UT(aclnnAdaptiveAvgPool3d, INPUT(input_tensor, int_array_desc), OUTPUT(output_tensor));
        uint64_t workspace_size = 0;
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    }
}

TEST_F(l2_adaptive_avg_pool3d_test, adaptive_avg_pool3d_cdhw_common_fp16)
{
    vector<aclDataType> dtypes{ACL_FLOAT16};
    vector<int64_t> output_size = {2, 3, 2};
    for (auto dtype : dtypes) {
        auto input_tensor = TensorDesc({2, 2, 4, 3}, dtype, ACL_FORMAT_ND).ValueRange(40, 50);
        auto int_array_desc = IntArrayDesc(output_size);
        auto output_tensor = TensorDesc({2, 2, 3, 2}, dtype, ACL_FORMAT_ND).Precision(0.03, 0.03);
        auto ut = OP_API_UT(aclnnAdaptiveAvgPool3d, INPUT(input_tensor, int_array_desc), OUTPUT(output_tensor));
        uint64_t workspace_size = 0;
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    }
}

TEST_F(l2_adaptive_avg_pool3d_test, adaptive_avg_pool3d_cdhw_bf16)
{
    vector<aclDataType> dtypes{ACL_BF16};
    vector<int64_t> output_size = {2, 3, 2};
    for (auto dtype : dtypes) {
        auto input_tensor = TensorDesc({2, 2, 4, 3}, dtype, ACL_FORMAT_ND).ValueRange(40, 50);
        auto int_array_desc = IntArrayDesc(output_size);
        auto output_tensor = TensorDesc({2, 2, 3, 2}, dtype, ACL_FORMAT_ND).Precision(0.03, 0.03);
        auto ut = OP_API_UT(aclnnAdaptiveAvgPool3d, INPUT(input_tensor, int_array_desc), OUTPUT(output_tensor));
        uint64_t workspace_size = 0;
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
        EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    }
}

TEST_F(l2_adaptive_avg_pool3d_test, ascend910B2_adaptive_avg_pool3d_cdhw_common_bf16)
{
    vector<aclDataType> dtypes{ACL_BF16};
    vector<int64_t> output_size = {2, 3, 2};
    for (auto dtype : dtypes) {
        auto input_tensor = TensorDesc({2, 2, 4, 3}, dtype, ACL_FORMAT_ND).ValueRange(40, 50);
        auto int_array_desc = IntArrayDesc(output_size);
        auto output_tensor = TensorDesc({2, 2, 3, 2}, dtype, ACL_FORMAT_ND).Precision(0.03, 0.03);
        auto ut = OP_API_UT(aclnnAdaptiveAvgPool3d, INPUT(input_tensor, int_array_desc), OUTPUT(output_tensor));
        uint64_t workspace_size = 0;
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
        EXPECT_EQ(aclRet, ACL_SUCCESS);
    }
}

TEST_F(l2_adaptive_avg_pool3d_test, adaptive_avg_pool3d_cdhw_one_one)
{
    vector<aclDataType> dtypes{ACL_FLOAT};
    vector<int64_t> output_size = {1, 1, 1};
    for (auto dtype : dtypes) {
        auto input_tensor = TensorDesc({2, 2, 4, 3}, dtype, ACL_FORMAT_ND).ValueRange(40, 50);
        auto int_array_desc = IntArrayDesc(output_size);
        auto output_tensor = TensorDesc({2, 1, 1, 1}, dtype, ACL_FORMAT_ND).Precision(0.03, 0.03);
        auto ut = OP_API_UT(aclnnAdaptiveAvgPool3d, INPUT(input_tensor, int_array_desc), OUTPUT(output_tensor));
        uint64_t workspace_size = 0;
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
        EXPECT_EQ(aclRet, ACL_SUCCESS);
    }
}

TEST_F(l2_adaptive_avg_pool3d_test, adaptive_avg_pool3d_dimnum_invalid)
{
    vector<aclDataType> dtypes{ACL_FLOAT};
    vector<int64_t> output_size = {3, 2, 2};
    for (auto dtype : dtypes) {
        auto input_tensor =
            TensorDesc({2, 2, 2, 2, 3}, dtype, ACL_FORMAT_ND)
                .Value(vector<float>{0.5,  1.5,  2.5,  3.5,  4.5,  5.5,  6.5,  7.5,  8.5,  9.5,  10.5, 11.5,
                                     12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5, 20.5, 21.5, 22.5, 23.5,
                                     24.5, 25.5, 26.5, 27.5, 28.5, 29.5, 30.5, 31.5, 32.5, 33.5, 34.5, 35.5,
                                     36.5, 37.5, 38.5, 39.5, 40.5, 41.5, 42.5, 43.5, 44.5, 45.5, 46.5, 47.5});
        auto int_array_desc = IntArrayDesc(output_size);
        auto output_tensor = TensorDesc({2, 2, 3, 2}, dtype, ACL_FORMAT_ND);
        auto ut = OP_API_UT(aclnnAdaptiveAvgPool3d, INPUT(input_tensor, int_array_desc), OUTPUT(output_tensor));
        uint64_t workspace_size = 0;
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
        EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    }
}

TEST_F(l2_adaptive_avg_pool3d_test, adaptive_avg_pool3d_dtype_invalid)
{
    vector<aclDataType> dtypes{ACL_INT64, ACL_UINT64, ACL_INT32};
    vector<int64_t> output_size = {2, 3, 2};
    for (auto dtype : dtypes) {
        auto input_tensor = TensorDesc({2, 2, 2, 4, 3}, dtype, ACL_FORMAT_ND).ValueRange(40, 50);
        auto int_array_desc = IntArrayDesc(output_size);
        auto output_tensor = TensorDesc({2, 2, 2, 3, 2}, dtype, ACL_FORMAT_ND);
        auto ut = OP_API_UT(aclnnAdaptiveAvgPool3d, INPUT(input_tensor, int_array_desc), OUTPUT(output_tensor));
        uint64_t workspace_size = 0;
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
        EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    }
}

TEST_F(l2_adaptive_avg_pool3d_test, adaptive_avg_pool3d_format_invalid)
{
    vector<aclDataType> dtypes{ACL_FLOAT};
    vector<int64_t> output_size = {2, 3, 2};
    for (auto dtype : dtypes) {
        auto input_tensor = TensorDesc({2, 2, 2, 4, 3}, dtype, ACL_FORMAT_NCDHW).ValueRange(40, 50);
        auto int_array_desc = IntArrayDesc(output_size);
        auto output_tensor = TensorDesc({2, 2, 2, 3, 2}, dtype, ACL_FORMAT_ND);
        auto ut = OP_API_UT(aclnnAdaptiveAvgPool3d, INPUT(input_tensor, int_array_desc), OUTPUT(output_tensor));
        uint64_t workspace_size = 0;
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
        EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    }
}

TEST_F(l2_adaptive_avg_pool3d_test, adaptive_avg_pool3d_outputsize_invalid)
{
    vector<aclDataType> dtypes{ACL_FLOAT};
    vector<int64_t> output_size = {2, 2, 2};
    for (auto dtype : dtypes) {
        auto input_tensor = TensorDesc({2, 2, 2, 4, 3}, dtype, ACL_FORMAT_ND).ValueRange(40, 50);
        auto int_array_desc = IntArrayDesc(output_size);
        auto output_tensor = TensorDesc({2, 2, 2, 3, 2}, dtype, ACL_FORMAT_ND);
        auto ut = OP_API_UT(aclnnAdaptiveAvgPool3d, INPUT(input_tensor, int_array_desc), OUTPUT(output_tensor));
        uint64_t workspace_size = 0;
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
        EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    }
}

TEST_F(l2_adaptive_avg_pool3d_test, adaptive_avg_pool3d_outputsize_dim_invalid)
{
    vector<aclDataType> dtypes{ACL_FLOAT};
    vector<int64_t> output_size = {2, 3};
    for (auto dtype : dtypes) {
        auto input_tensor = TensorDesc({2, 2, 2, 4, 3}, dtype, ACL_FORMAT_ND).ValueRange(40, 50);
        auto int_array_desc = IntArrayDesc(output_size);
        auto output_tensor = TensorDesc({2, 2, 2, 3, 2}, dtype, ACL_FORMAT_ND);
        auto ut = OP_API_UT(aclnnAdaptiveAvgPool3d, INPUT(input_tensor, int_array_desc), OUTPUT(output_tensor));
        uint64_t workspace_size = 0;
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
        EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    }
}

TEST_F(l2_adaptive_avg_pool3d_test, adaptive_avg_pool3d_out_nc_dim_invalid)
{
    vector<aclDataType> dtypes{ACL_FLOAT};
    vector<int64_t> output_size = {2, 3, 2};
    for (auto dtype : dtypes) {
        auto input_tensor = TensorDesc({2, 2, 2, 4, 3}, dtype, ACL_FORMAT_ND).ValueRange(40, 50);
        auto int_array_desc = IntArrayDesc(output_size);
        auto output_tensor = TensorDesc({2, 1, 2, 3, 2}, dtype, ACL_FORMAT_ND);
        auto ut = OP_API_UT(aclnnAdaptiveAvgPool3d, INPUT(input_tensor, int_array_desc), OUTPUT(output_tensor));
        uint64_t workspace_size = 0;
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
        EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    }
}
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

#include "opdev/op_log.h"
#include "../../../op_host/op_api/aclnn_adaptive_max_pool3d.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace op;
using namespace std;

class l2_adaptive_max_pool3d_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "l2_adaptive_max_pool3d_test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "l2_adaptive_max_pool3d_test TearDown" << std::endl;
    }
};

TEST_F(l2_adaptive_max_pool3d_test, ascend910B1_case_aicore_maxpool3d)
{
    vector<int64_t> self_dims = {2, 1, 2, 2, 2};
    vector<int64_t> output_size_dims = {1, 1, 1};
    vector<int64_t> out_dims = {2, 1, 1, 1, 1};
    vector<int64_t> indices_dims = {2, 1, 1, 1, 1};

    auto self_desc = TensorDesc(self_dims, ACL_FLOAT, ACL_FORMAT_NCDHW)
                         .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    auto output_size_desc = IntArrayDesc(output_size_dims);
    auto out_tensor_desc = TensorDesc(out_dims, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto indices_tensor_desc = TensorDesc(indices_dims, ACL_INT32, ACL_FORMAT_NCDHW);

    auto ut = OP_API_UT(
        aclnnAdaptiveMaxPool3d, INPUT(self_desc, output_size_desc), OUTPUT(out_tensor_desc, indices_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    if (GetCurrentPlatformInfo().GetSocVersion() != SocVersion::ASCEND910B) {
        EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    } else {
        EXPECT_EQ(aclRet, ACL_SUCCESS);
    }
}

TEST_F(l2_adaptive_max_pool3d_test, ascend910B1_case_aicore_maxpool3d_4d)
{
    vector<int64_t> self_dims = {2, 1, 2, 2};
    vector<int64_t> output_size_dims = {1, 1, 1};
    vector<int64_t> out_dims = {2, 1, 1, 1};
    vector<int64_t> indices_dims = {2, 1, 1, 1};

    auto self_desc = TensorDesc(self_dims, ACL_FLOAT, ACL_FORMAT_NCHW).Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8});
    auto output_size_desc = IntArrayDesc(output_size_dims);
    auto out_tensor_desc = TensorDesc(out_dims, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto indices_tensor_desc = TensorDesc(indices_dims, ACL_INT32, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(
        aclnnAdaptiveMaxPool3d, INPUT(self_desc, output_size_desc), OUTPUT(out_tensor_desc, indices_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    if (GetCurrentPlatformInfo().GetSocVersion() != SocVersion::ASCEND910B) {
        EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    } else {
        EXPECT_EQ(aclRet, ACL_SUCCESS);
    }
}

TEST_F(l2_adaptive_max_pool3d_test, ascend910B1_case_aicore_adaptive_maxpool3d)
{
    vector<int64_t> self_dims = {2, 1, 3, 3, 3};
    vector<int64_t> output_size_dims = {2, 2, 2};
    vector<int64_t> out_dims = {2, 1, 2, 2, 2};
    vector<int64_t> indices_dims = {2, 1, 2, 2, 2};

    auto self_desc = TensorDesc(self_dims, ACL_FLOAT, ACL_FORMAT_NCDHW).ValueRange(10, 20);
    auto output_size_desc = IntArrayDesc(output_size_dims);
    auto out_tensor_desc = TensorDesc(out_dims, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto indices_tensor_desc = TensorDesc(indices_dims, ACL_INT32, ACL_FORMAT_NCDHW);

    auto ut = OP_API_UT(
        aclnnAdaptiveMaxPool3d, INPUT(self_desc, output_size_desc), OUTPUT(out_tensor_desc, indices_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    if (GetCurrentPlatformInfo().GetSocVersion() != SocVersion::ASCEND910B) {
        EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    } else {
        EXPECT_EQ(aclRet, ACL_SUCCESS);
    }
}

TEST_F(l2_adaptive_max_pool3d_test, ascend910B1_case_aicore_adaptive_maxpool3d_4d)
{
    vector<int64_t> self_dims = {2, 1, 3, 3};
    vector<int64_t> output_size_dims = {1, 2, 2};
    vector<int64_t> out_dims = {2, 1, 2, 2};
    vector<int64_t> indices_dims = {2, 1, 2, 2};

    auto self_desc = TensorDesc(self_dims, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(10, 20);
    auto output_size_desc = IntArrayDesc(output_size_dims);
    auto out_tensor_desc = TensorDesc(out_dims, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto indices_tensor_desc = TensorDesc(indices_dims, ACL_INT32, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(
        aclnnAdaptiveMaxPool3d, INPUT(self_desc, output_size_desc), OUTPUT(out_tensor_desc, indices_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    if (GetCurrentPlatformInfo().GetSocVersion() != SocVersion::ASCEND910B) {
        EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    } else {
        EXPECT_EQ(aclRet, ACL_SUCCESS);
    }
}

TEST_F(l2_adaptive_max_pool3d_test, ascend910B1_case_aicore_input_nullptr)
{
    vector<int64_t> output_size_dims = {1, 1, 1};
    vector<int64_t> out_dims = {2, 1, 1, 1, 1};
    vector<int64_t> indices_dims = {2, 1, 1, 1, 1};

    auto self_desc = nullptr;
    auto output_size_desc = IntArrayDesc(output_size_dims);
    auto out_tensor_desc = TensorDesc(out_dims, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto indices_tensor_desc = TensorDesc(indices_dims, ACL_INT32, ACL_FORMAT_NCDHW);

    auto ut = OP_API_UT(
        aclnnAdaptiveMaxPool3d, INPUT(self_desc, output_size_desc), OUTPUT(out_tensor_desc, indices_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_adaptive_max_pool3d_test, ascend910B1_case_aicore_outsize_nullptr)
{
    vector<int64_t> self_dims = {2, 1, 2, 2, 2};
    vector<int64_t> out_dims = {2, 1, 1, 1, 1};
    vector<int64_t> indices_dims = {2, 1, 1, 1, 1};

    auto self_desc = TensorDesc(self_dims, ACL_FLOAT, ACL_FORMAT_NCDHW)
                         .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    auto output_size_desc = nullptr;
    auto out_tensor_desc = TensorDesc(out_dims, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto indices_tensor_desc = TensorDesc(indices_dims, ACL_INT32, ACL_FORMAT_NCDHW);

    auto ut = OP_API_UT(
        aclnnAdaptiveMaxPool3d, INPUT(self_desc, output_size_desc), OUTPUT(out_tensor_desc, indices_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_adaptive_max_pool3d_test, ascend910B1_case_aicore_out_nullptr)
{
    vector<int64_t> self_dims = {2, 1, 2, 2, 2};
    vector<int64_t> output_size_dims = {1, 1, 1};
    vector<int64_t> indices_dims = {2, 1, 1, 1, 1};

    auto self_desc = TensorDesc(self_dims, ACL_FLOAT, ACL_FORMAT_NCDHW)
                         .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    auto output_size_desc = IntArrayDesc(output_size_dims);
    auto out_tensor_desc = nullptr;
    auto indices_tensor_desc = TensorDesc(indices_dims, ACL_INT32, ACL_FORMAT_NCDHW);

    auto ut = OP_API_UT(
        aclnnAdaptiveMaxPool3d, INPUT(self_desc, output_size_desc), OUTPUT(out_tensor_desc, indices_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_adaptive_max_pool3d_test, ascend910B1_case_aicore_input_dims)
{
    vector<int64_t> self_dims = {2, 2, 2};
    vector<int64_t> output_size_dims = {1, 1, 1};
    vector<int64_t> out_dims = {2, 1, 1, 1, 1};
    vector<int64_t> indices_dims = {2, 1, 1, 1, 1};

    auto self_desc = TensorDesc(self_dims, ACL_FLOAT, ACL_FORMAT_NCDHW)
                         .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    auto output_size_desc = IntArrayDesc(output_size_dims);
    auto out_tensor_desc = TensorDesc(out_dims, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto indices_tensor_desc = TensorDesc(indices_dims, ACL_INT32, ACL_FORMAT_NCDHW);

    auto ut = OP_API_UT(
        aclnnAdaptiveMaxPool3d, INPUT(self_desc, output_size_desc), OUTPUT(out_tensor_desc, indices_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_adaptive_max_pool3d_test, ascend910B1_case_aicore_input_format)
{
    vector<int64_t> self_dims = {2, 1, 2, 2, 2};
    vector<int64_t> output_size_dims = {1, 1, 1};
    vector<int64_t> out_dims = {2, 1, 1, 1, 1};
    vector<int64_t> indices_dims = {2, 1, 1, 1, 1};

    auto self_desc = TensorDesc(self_dims, ACL_FLOAT, ACL_FORMAT_FRACTAL_NZ)
                         .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    auto output_size_desc = IntArrayDesc(output_size_dims);
    auto out_tensor_desc = TensorDesc(out_dims, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto indices_tensor_desc = TensorDesc(indices_dims, ACL_INT32, ACL_FORMAT_NCDHW);

    auto ut = OP_API_UT(
        aclnnAdaptiveMaxPool3d, INPUT(self_desc, output_size_desc), OUTPUT(out_tensor_desc, indices_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_adaptive_max_pool3d_test, ascend910B1_case_aicore_out_format)
{
    vector<int64_t> self_dims = {2, 1, 2, 2, 2};
    vector<int64_t> output_size_dims = {1, 1, 1};
    vector<int64_t> out_dims = {2, 1, 1, 1, 1};
    vector<int64_t> indices_dims = {2, 1, 1, 1, 1};

    auto self_desc = TensorDesc(self_dims, ACL_FLOAT, ACL_FORMAT_NCDHW)
                         .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    auto output_size_desc = IntArrayDesc(output_size_dims);
    auto out_tensor_desc = TensorDesc(out_dims, ACL_FLOAT, ACL_FORMAT_FRACTAL_NZ);
    auto indices_tensor_desc = TensorDesc(indices_dims, ACL_INT32, ACL_FORMAT_NCDHW);

    auto ut = OP_API_UT(
        aclnnAdaptiveMaxPool3d, INPUT(self_desc, output_size_desc), OUTPUT(out_tensor_desc, indices_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_adaptive_max_pool3d_test, ascend910B1_case_aicore_input_data_type)
{
    vector<int64_t> self_dims = {2, 1, 2, 2, 2};
    vector<int64_t> output_size_dims = {1, 1, 1};
    vector<int64_t> out_dims = {2, 1, 1, 1, 1};
    vector<int64_t> indices_dims = {2, 1, 1, 1, 1};

    auto self_desc = TensorDesc(self_dims, ACL_INT32, ACL_FORMAT_NCDHW)
                         .Value(vector<int32_t>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    auto output_size_desc = IntArrayDesc(output_size_dims);
    auto out_tensor_desc = TensorDesc(out_dims, ACL_INT32, ACL_FORMAT_NCDHW);
    auto indices_tensor_desc = TensorDesc(indices_dims, ACL_INT32, ACL_FORMAT_NCDHW);

    auto ut = OP_API_UT(
        aclnnAdaptiveMaxPool3d, INPUT(self_desc, output_size_desc), OUTPUT(out_tensor_desc, indices_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_adaptive_max_pool3d_test, ascend910B1_case_aicore_self_output_mismatch)
{
    vector<int64_t> self_dims = {2, 1, 2, 2, 2};
    vector<int64_t> output_size_dims = {1, 1, 1};
    vector<int64_t> out_dims = {2, 1, 1, 1, 1};
    vector<int64_t> indices_dims = {2, 1, 1, 1, 1};

    auto self_desc = TensorDesc(self_dims, ACL_FLOAT16, ACL_FORMAT_NCDHW)
                         .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    auto output_size_desc = IntArrayDesc(output_size_dims);
    auto out_tensor_desc = TensorDesc(out_dims, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto indices_tensor_desc = TensorDesc(indices_dims, ACL_INT32, ACL_FORMAT_NCDHW);

    auto ut = OP_API_UT(
        aclnnAdaptiveMaxPool3d, INPUT(self_desc, output_size_desc), OUTPUT(out_tensor_desc, indices_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_adaptive_max_pool3d_test, ascend910B1_case_aicore_input_size)
{
    vector<int64_t> self_dims = {2, 0, 2, 2, 2};
    vector<int64_t> output_size_dims = {1, 1, 1};
    vector<int64_t> out_dims = {2, 1, 1, 1, 1};
    vector<int64_t> indices_dims = {2, 1, 1, 1, 1};

    auto self_desc = TensorDesc(self_dims, ACL_FLOAT, ACL_FORMAT_NCDHW)
                         .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    auto output_size_desc = IntArrayDesc(output_size_dims);
    auto out_tensor_desc = TensorDesc(out_dims, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto indices_tensor_desc = TensorDesc(indices_dims, ACL_INT32, ACL_FORMAT_NCDHW);

    auto ut = OP_API_UT(
        aclnnAdaptiveMaxPool3d, INPUT(self_desc, output_size_desc), OUTPUT(out_tensor_desc, indices_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_adaptive_max_pool3d_test, ascend910B1_case_aicore_out_indices_mismatch)
{
    vector<int64_t> self_dims = {2, 1, 2, 2, 2};
    vector<int64_t> output_size_dims = {1, 1, 1};
    vector<int64_t> out_dims = {2, 1, 1, 1, 1};
    vector<int64_t> indices_dims = {1, 1, 1, 1, 1};

    auto self_desc = TensorDesc(self_dims, ACL_FLOAT, ACL_FORMAT_NCDHW)
                         .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    auto output_size_desc = IntArrayDesc(output_size_dims);
    auto out_tensor_desc = TensorDesc(out_dims, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto indices_tensor_desc = TensorDesc(indices_dims, ACL_INT32, ACL_FORMAT_NCDHW);

    auto ut = OP_API_UT(
        aclnnAdaptiveMaxPool3d, INPUT(self_desc, output_size_desc), OUTPUT(out_tensor_desc, indices_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_adaptive_max_pool3d_test, ascend910B1_case_aicore_output_size)
{
    vector<int64_t> self_dims = {2, 1, 2, 2, 2};
    vector<int64_t> output_size_dims = {1, 1};
    vector<int64_t> out_dims = {2, 1, 1, 1, 1};
    vector<int64_t> indices_dims = {2, 1, 1, 1, 1};

    auto self_desc = TensorDesc(self_dims, ACL_FLOAT, ACL_FORMAT_NCDHW)
                         .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    auto output_size_desc = IntArrayDesc(output_size_dims);
    auto out_tensor_desc = TensorDesc(out_dims, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto indices_tensor_desc = TensorDesc(indices_dims, ACL_INT32, ACL_FORMAT_NCDHW);

    auto ut = OP_API_UT(
        aclnnAdaptiveMaxPool3d, INPUT(self_desc, output_size_desc), OUTPUT(out_tensor_desc, indices_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_adaptive_max_pool3d_test, ascend910B1_case_aicore_dhw_size)
{
    vector<int64_t> self_dims = {2, 1, 4294967295, 2, 1};
    vector<int64_t> output_size_dims = {1, 1, 1};
    vector<int64_t> out_dims = {2, 1, 4294967295, 2, 1};
    vector<int64_t> indices_dims = {2, 1, 4294967295, 2, 1};

    auto self_desc = TensorDesc(self_dims, ACL_FLOAT, ACL_FORMAT_NCDHW)
                         .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    auto output_size_desc = IntArrayDesc(output_size_dims);
    auto out_tensor_desc = TensorDesc(out_dims, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto indices_tensor_desc = TensorDesc(indices_dims, ACL_INT32, ACL_FORMAT_NCDHW);

    auto ut = OP_API_UT(
        aclnnAdaptiveMaxPool3d, INPUT(self_desc, output_size_desc), OUTPUT(out_tensor_desc, indices_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

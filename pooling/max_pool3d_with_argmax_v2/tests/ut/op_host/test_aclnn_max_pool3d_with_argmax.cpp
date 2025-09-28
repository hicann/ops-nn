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
#include "../../../op_host/op_api/aclnn_max_pool3d_with_argmax.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace op;
using namespace std;

class l2_max_pool3d_with_argmax_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "l2_max_pool3d_with_argmax_test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "l2_max_pool3d_with_argmax_test TearDown" << std::endl;
    }
};

TEST_F(l2_max_pool3d_with_argmax_test, ascend910B1_case_aicore_float)
{
    vector<int64_t> self_dims = {1, 8, 6, 6, 6};
    vector<int64_t> kernel_dims = {2, 2, 2};
    vector<int64_t> stride_dims = {2, 2, 2};
    vector<int64_t> padding_dims = {0, 0, 0};
    vector<int64_t> dilation_dims = {1, 1, 1};
    vector<int64_t> out_dims = {1, 8, 3, 3, 3};
    vector<int64_t> indices_dims = {1, 8, 3, 3, 3};

    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    bool ceilMode = false;

    auto self_desc = TensorDesc(self_dims, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc(out_dims, ACL_FLOAT, ACL_FORMAT_ND);
    auto indices_tensor_desc = TensorDesc(indices_dims, ACL_INT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnMaxPool3dWithArgmax, INPUT(self_desc, kernel_desc, stride_desc, padding_desc, dilation_desc, ceilMode),
        OUTPUT(out_tensor_desc, indices_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    if (GetCurrentPlatformInfo().GetSocVersion() != SocVersion::ASCEND910B) {
        EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    } else {
        EXPECT_EQ(aclRet, ACL_SUCCESS);
        // ut.TestPrecision();
    }
}

TEST_F(l2_max_pool3d_with_argmax_test, ascend910B1_case_aicore_padding)
{
    vector<int64_t> self_dims = {1, 8, 6, 6, 6};
    vector<int64_t> kernel_dims = {2, 2, 2};
    vector<int64_t> stride_dims = {2, 2, 2};
    vector<int64_t> padding_dims = {1, 1, 1};
    vector<int64_t> dilation_dims = {1, 1, 1};
    vector<int64_t> out_dims = {1, 8, 4, 4, 4};
    vector<int64_t> indices_dims = {1, 8, 4, 4, 4};

    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    bool ceilMode = false;

    auto self_desc = TensorDesc(self_dims, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc(out_dims, ACL_FLOAT, ACL_FORMAT_ND);
    auto indices_tensor_desc = TensorDesc(indices_dims, ACL_INT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnMaxPool3dWithArgmax, INPUT(self_desc, kernel_desc, stride_desc, padding_desc, dilation_desc, ceilMode),
        OUTPUT(out_tensor_desc, indices_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    if (GetCurrentPlatformInfo().GetSocVersion() != SocVersion::ASCEND910B) {
        EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    } else {
        EXPECT_EQ(aclRet, ACL_SUCCESS);
        // ut.TestPrecision();
    }
}

TEST_F(l2_max_pool3d_with_argmax_test, ascend910B1_case_aicore_input_4d)
{
    vector<int64_t> self_dims = {8, 6, 6, 6};
    vector<int64_t> kernel_dims = {2, 2, 2};
    vector<int64_t> stride_dims = {2, 2, 2};
    vector<int64_t> padding_dims = {0, 0, 0};
    vector<int64_t> dilation_dims = {1, 1, 1};
    vector<int64_t> out_dims = {8, 3, 3, 3};
    vector<int64_t> indices_dims = {8, 3, 3, 3};

    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    bool ceilMode = false;

    auto self_desc = TensorDesc(self_dims, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc(out_dims, ACL_FLOAT, ACL_FORMAT_ND);
    auto indices_tensor_desc = TensorDesc(indices_dims, ACL_INT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnMaxPool3dWithArgmax, INPUT(self_desc, kernel_desc, stride_desc, padding_desc, dilation_desc, ceilMode),
        OUTPUT(out_tensor_desc, indices_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    if (GetCurrentPlatformInfo().GetSocVersion() != SocVersion::ASCEND910B) {
        EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    } else {
        EXPECT_EQ(aclRet, ACL_SUCCESS);
        // ut.TestPrecision();
    }
}

TEST_F(l2_max_pool3d_with_argmax_test, ascend910B1_case_aicore_kernel_size_non_square)
{
    vector<int64_t> self_dims = {1, 8, 6, 6, 6};
    vector<int64_t> kernel_dims = {3, 4, 5};
    vector<int64_t> stride_dims = {3, 4, 5};
    vector<int64_t> padding_dims = {0, 0, 0};
    vector<int64_t> dilation_dims = {1, 1, 1};
    vector<int64_t> out_dims = {1, 8, 2, 1, 1};
    vector<int64_t> indices_dims = {1, 8, 2, 1, 1};

    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    bool ceilMode = false;

    auto self_desc = TensorDesc(self_dims, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc(out_dims, ACL_FLOAT, ACL_FORMAT_ND);
    auto indices_tensor_desc = TensorDesc(indices_dims, ACL_INT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnMaxPool3dWithArgmax, INPUT(self_desc, kernel_desc, stride_desc, padding_desc, dilation_desc, ceilMode),
        OUTPUT(out_tensor_desc, indices_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    if (GetCurrentPlatformInfo().GetSocVersion() != SocVersion::ASCEND910B) {
        EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    } else {
        EXPECT_EQ(aclRet, ACL_SUCCESS);
        // ut.TestPrecision();
    }
}

TEST_F(l2_max_pool3d_with_argmax_test, ascend910B1_case_aicore_ceil_mode)
{
    vector<int64_t> self_dims = {1, 8, 8, 8, 8};
    vector<int64_t> kernel_dims = {3, 3, 3};
    vector<int64_t> stride_dims = {3, 3, 3};
    vector<int64_t> padding_dims = {0, 0, 0};
    vector<int64_t> dilation_dims = {1, 1, 1};
    vector<int64_t> out_dims = {1, 8, 3, 3, 3};
    vector<int64_t> indices_dims = {1, 8, 3, 3, 3};

    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    bool ceilMode = true;

    auto self_desc = TensorDesc(self_dims, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc(out_dims, ACL_FLOAT, ACL_FORMAT_ND);
    auto indices_tensor_desc = TensorDesc(indices_dims, ACL_INT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnMaxPool3dWithArgmax, INPUT(self_desc, kernel_desc, stride_desc, padding_desc, dilation_desc, ceilMode),
        OUTPUT(out_tensor_desc, indices_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    if (GetCurrentPlatformInfo().GetSocVersion() != SocVersion::ASCEND910B) {
        EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    } else {
        EXPECT_EQ(aclRet, ACL_SUCCESS);
        // ut.TestPrecision();
    }
}

// ?tensor,n???0,chw????0
TEST_F(l2_max_pool3d_with_argmax_test, ascend910B1_case_aicore_batches)
{
    vector<int64_t> self_dims = {48, 8, 6, 6, 6};
    vector<int64_t> kernel_dims = {2, 2, 2};
    vector<int64_t> stride_dims = {2, 2, 2};
    vector<int64_t> padding_dims = {0, 0, 0};
    vector<int64_t> dilation_dims = {1, 1, 1};
    vector<int64_t> out_dims = {48, 8, 3, 3, 3};
    vector<int64_t> indices_dims = {48, 8, 3, 3, 3};

    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    bool ceilMode = false;

    auto self_desc = TensorDesc(self_dims, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc(out_dims, ACL_FLOAT, ACL_FORMAT_ND);
    auto indices_tensor_desc = TensorDesc(indices_dims, ACL_INT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnMaxPool3dWithArgmax, INPUT(self_desc, kernel_desc, stride_desc, padding_desc, dilation_desc, ceilMode),
        OUTPUT(out_tensor_desc, indices_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    if (GetCurrentPlatformInfo().GetSocVersion() != SocVersion::ASCEND910B) {
        EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    } else {
        EXPECT_EQ(aclRet, ACL_SUCCESS);
        // ut.TestPrecision();
    }
}

TEST_F(l2_max_pool3d_with_argmax_test, ascend910B1_case_aicore_int)
{
    vector<int64_t> self_dims = {1, 8, 6, 6, 6};
    vector<int64_t> kernel_dims = {2, 2, 2};
    vector<int64_t> stride_dims = {2, 2, 2};
    vector<int64_t> padding_dims = {0, 0, 0};
    vector<int64_t> dilation_dims = {1, 1, 1};
    vector<int64_t> out_dims = {1, 8, 3, 3, 3};
    vector<int64_t> indices_dims = {1, 8, 3, 3, 3};

    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    bool ceilMode = false;

    auto self_desc = TensorDesc(self_dims, ACL_INT32, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc(out_dims, ACL_INT32, ACL_FORMAT_ND);
    auto indices_tensor_desc = TensorDesc(indices_dims, ACL_INT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnMaxPool3dWithArgmax, INPUT(self_desc, kernel_desc, stride_desc, padding_desc, dilation_desc, ceilMode),
        OUTPUT(out_tensor_desc, indices_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_max_pool3d_with_argmax_test, ascend910B1_case_aicore_self_nullptr)
{
    vector<int64_t> self_dims = {1, 8, 6, 6, 6};
    vector<int64_t> kernel_dims = {2, 2, 2};
    vector<int64_t> stride_dims = {2, 2, 2};
    vector<int64_t> padding_dims = {0, 0, 0};
    vector<int64_t> dilation_dims = {1, 1, 1};
    vector<int64_t> out_dims = {1, 8, 3, 3, 3};
    vector<int64_t> indices_dims = {1, 8, 3, 3, 3};

    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    bool ceilMode = false;

    auto self_desc = nullptr;
    auto out_tensor_desc = TensorDesc(out_dims, ACL_FLOAT, ACL_FORMAT_ND);
    auto indices_tensor_desc = TensorDesc(indices_dims, ACL_INT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnMaxPool3dWithArgmax, INPUT(self_desc, kernel_desc, stride_desc, padding_desc, dilation_desc, ceilMode),
        OUTPUT(out_tensor_desc, indices_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_max_pool3d_with_argmax_test, ascend910B1_case_aicore_kernel_nullptr)
{
    vector<int64_t> self_dims = {1, 8, 6, 6, 6};
    vector<int64_t> kernel_dims = {2, 2, 2};
    vector<int64_t> stride_dims = {2, 2, 2};
    vector<int64_t> padding_dims = {0, 0, 0};
    vector<int64_t> dilation_dims = {1, 1, 1};
    vector<int64_t> out_dims = {1, 8, 3, 3, 3};
    vector<int64_t> indices_dims = {1, 8, 3, 3, 3};

    auto kernel_desc = nullptr;
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    bool ceilMode = false;

    auto self_desc = TensorDesc(self_dims, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc(out_dims, ACL_FLOAT, ACL_FORMAT_ND);
    auto indices_tensor_desc = TensorDesc(indices_dims, ACL_INT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnMaxPool3dWithArgmax, INPUT(self_desc, kernel_desc, stride_desc, padding_desc, dilation_desc, ceilMode),
        OUTPUT(out_tensor_desc, indices_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_max_pool3d_with_argmax_test, ascend910B1_case_aicore_indices_float)
{
    vector<int64_t> self_dims = {1, 8, 6, 6, 6};
    vector<int64_t> kernel_dims = {2, 2, 2};
    vector<int64_t> stride_dims = {2, 2, 2};
    vector<int64_t> padding_dims = {0, 0, 0};
    vector<int64_t> dilation_dims = {1, 1, 1};
    vector<int64_t> out_dims = {1, 8, 3, 3, 3};
    vector<int64_t> indices_dims = {1, 8, 3, 3, 3};

    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    bool ceilMode = false;

    auto self_desc = TensorDesc(self_dims, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc(out_dims, ACL_FLOAT, ACL_FORMAT_ND);
    auto indices_tensor_desc = TensorDesc(indices_dims, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnMaxPool3dWithArgmax, INPUT(self_desc, kernel_desc, stride_desc, padding_desc, dilation_desc, ceilMode),
        OUTPUT(out_tensor_desc, indices_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_max_pool3d_with_argmax_test, ascend910B1_case_aicore_out_int)
{
    vector<int64_t> self_dims = {1, 8, 6, 6, 6};
    vector<int64_t> kernel_dims = {2, 2, 2};
    vector<int64_t> stride_dims = {2, 2, 2};
    vector<int64_t> padding_dims = {0, 0, 0};
    vector<int64_t> dilation_dims = {1, 1, 1};
    vector<int64_t> out_dims = {1, 8, 3, 3, 3};
    vector<int64_t> indices_dims = {1, 8, 3, 3, 3};

    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    bool ceilMode = false;

    auto self_desc = TensorDesc(self_dims, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc(out_dims, ACL_INT32, ACL_FORMAT_ND);
    auto indices_tensor_desc = TensorDesc(indices_dims, ACL_INT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnMaxPool3dWithArgmax, INPUT(self_desc, kernel_desc, stride_desc, padding_desc, dilation_desc, ceilMode),
        OUTPUT(out_tensor_desc, indices_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_max_pool3d_with_argmax_test, ascend910B1_case_aicore_self_out_diff)
{
    vector<int64_t> self_dims = {1, 8, 6, 6, 6};
    vector<int64_t> kernel_dims = {2, 2, 2};
    vector<int64_t> stride_dims = {2, 2, 2};
    vector<int64_t> padding_dims = {0, 0, 0};
    vector<int64_t> dilation_dims = {1, 1, 1};
    vector<int64_t> out_dims = {1, 8, 3, 3, 3};
    vector<int64_t> indices_dims = {1, 8, 3, 3, 3};

    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    bool ceilMode = false;

    auto self_desc = TensorDesc(self_dims, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc(out_dims, ACL_FLOAT, ACL_FORMAT_NCL);
    auto indices_tensor_desc = TensorDesc(indices_dims, ACL_INT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnMaxPool3dWithArgmax, INPUT(self_desc, kernel_desc, stride_desc, padding_desc, dilation_desc, ceilMode),
        OUTPUT(out_tensor_desc, indices_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_max_pool3d_with_argmax_test, ascend910B1_case_aicore_3d)
{
    vector<int64_t> self_dims = {6, 6, 6};
    vector<int64_t> kernel_dims = {2, 2, 2};
    vector<int64_t> stride_dims = {2, 2, 2};
    vector<int64_t> padding_dims = {0, 0, 0};
    vector<int64_t> dilation_dims = {1, 1, 1};
    vector<int64_t> out_dims = {3, 3, 3};
    vector<int64_t> indices_dims = {3, 3, 3};

    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    bool ceilMode = false;

    auto self_desc = TensorDesc(self_dims, ACL_FLOAT, ACL_FORMAT_NCL);
    auto out_tensor_desc = TensorDesc(out_dims, ACL_FLOAT, ACL_FORMAT_NCL);
    auto indices_tensor_desc = TensorDesc(indices_dims, ACL_INT32, ACL_FORMAT_NCL);

    auto ut = OP_API_UT(
        aclnnMaxPool3dWithArgmax, INPUT(self_desc, kernel_desc, stride_desc, padding_desc, dilation_desc, ceilMode),
        OUTPUT(out_tensor_desc, indices_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_max_pool3d_with_argmax_test, ascend910B1_case_aicore_kernel_value_0)
{
    vector<int64_t> self_dims = {1, 8, 6, 6, 6};
    vector<int64_t> kernel_dims = {0, 2, 2};
    vector<int64_t> stride_dims = {2, 2, 2};
    vector<int64_t> padding_dims = {0, 0, 0};
    vector<int64_t> dilation_dims = {1, 1, 1};
    vector<int64_t> out_dims = {1, 8, 3, 3, 3};
    vector<int64_t> indices_dims = {1, 8, 3, 3, 3};

    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    bool ceilMode = false;

    auto self_desc = TensorDesc(self_dims, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc(out_dims, ACL_FLOAT, ACL_FORMAT_ND);
    auto indices_tensor_desc = TensorDesc(indices_dims, ACL_INT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnMaxPool3dWithArgmax, INPUT(self_desc, kernel_desc, stride_desc, padding_desc, dilation_desc, ceilMode),
        OUTPUT(out_tensor_desc, indices_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_max_pool3d_with_argmax_test, ascend910B1_case_aicore_stride_value_0)
{
    vector<int64_t> self_dims = {1, 8, 6, 6, 6};
    vector<int64_t> kernel_dims = {2, 2, 2};
    vector<int64_t> stride_dims = {0, 2, 2};
    vector<int64_t> padding_dims = {0, 0, 0};
    vector<int64_t> dilation_dims = {1, 1, 1};
    vector<int64_t> out_dims = {1, 8, 3, 3, 3};
    vector<int64_t> indices_dims = {1, 8, 3, 3, 3};

    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    bool ceilMode = false;

    auto self_desc = TensorDesc(self_dims, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc(out_dims, ACL_FLOAT, ACL_FORMAT_ND);
    auto indices_tensor_desc = TensorDesc(indices_dims, ACL_INT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnMaxPool3dWithArgmax, INPUT(self_desc, kernel_desc, stride_desc, padding_desc, dilation_desc, ceilMode),
        OUTPUT(out_tensor_desc, indices_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_max_pool3d_with_argmax_test, ascend910B1_case_aicore_dilation_value_0)
{
    vector<int64_t> self_dims = {1, 8, 6, 6, 6};
    vector<int64_t> kernel_dims = {2, 2, 2};
    vector<int64_t> stride_dims = {2, 2, 2};
    vector<int64_t> padding_dims = {0, 0, 0};
    vector<int64_t> dilation_dims = {0, 1, 1};
    vector<int64_t> out_dims = {1, 8, 3, 3, 3};
    vector<int64_t> indices_dims = {1, 8, 3, 3, 3};

    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    bool ceilMode = false;

    auto self_desc = TensorDesc(self_dims, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc(out_dims, ACL_FLOAT, ACL_FORMAT_ND);
    auto indices_tensor_desc = TensorDesc(indices_dims, ACL_INT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnMaxPool3dWithArgmax, INPUT(self_desc, kernel_desc, stride_desc, padding_desc, dilation_desc, ceilMode),
        OUTPUT(out_tensor_desc, indices_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_max_pool3d_with_argmax_test, ascend910B1_case_aicore_incorrect_padding)
{
    vector<int64_t> self_dims = {1, 8, 6, 6, 6};
    vector<int64_t> kernel_dims = {2, 2, 2};
    vector<int64_t> stride_dims = {2, 2, 2};
    vector<int64_t> padding_dims = {2, 2, 2};
    vector<int64_t> dilation_dims = {1, 1, 1};
    vector<int64_t> out_dims = {1, 8, 5, 5, 5};
    vector<int64_t> indices_dims = {1, 8, 5, 5, 5};

    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    bool ceilMode = false;

    auto self_desc = TensorDesc(self_dims, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc(out_dims, ACL_FLOAT, ACL_FORMAT_ND);
    auto indices_tensor_desc = TensorDesc(indices_dims, ACL_INT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnMaxPool3dWithArgmax, INPUT(self_desc, kernel_desc, stride_desc, padding_desc, dilation_desc, ceilMode),
        OUTPUT(out_tensor_desc, indices_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_max_pool3d_with_argmax_test, ascend910B1_case_aicore_incorrect_out_shape)
{
    vector<int64_t> self_dims = {1, 8, 6, 6, 6};
    vector<int64_t> kernel_dims = {2, 2, 2};
    vector<int64_t> stride_dims = {2, 2, 2};
    vector<int64_t> padding_dims = {0, 0, 0};
    vector<int64_t> dilation_dims = {1, 1, 1};
    vector<int64_t> out_dims = {1, 8, 4, 4, 4};
    vector<int64_t> indices_dims = {1, 8, 4, 4, 4};

    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    bool ceilMode = false;

    auto self_desc = TensorDesc(self_dims, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc(out_dims, ACL_FLOAT, ACL_FORMAT_ND);
    auto indices_tensor_desc = TensorDesc(indices_dims, ACL_INT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnMaxPool3dWithArgmax, INPUT(self_desc, kernel_desc, stride_desc, padding_desc, dilation_desc, ceilMode),
        OUTPUT(out_tensor_desc, indices_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_max_pool3d_with_argmax_test, ascend910B1_case_aicore_incorrect_indices_format)
{
    vector<int64_t> self_dims = {1, 8, 6, 6, 6};
    vector<int64_t> kernel_dims = {2, 2, 2};
    vector<int64_t> stride_dims = {2, 2, 2};
    vector<int64_t> padding_dims = {0, 0, 0};
    vector<int64_t> dilation_dims = {1, 1, 1};
    vector<int64_t> out_dims = {1, 8, 4, 4, 4};
    vector<int64_t> indices_dims = {1, 8, 4, 4, 4};

    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    bool ceilMode = false;

    auto self_desc = TensorDesc(self_dims, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc(out_dims, ACL_FLOAT, ACL_FORMAT_ND);
    auto indices_tensor_desc = TensorDesc(indices_dims, ACL_INT32, ACL_FORMAT_NCL);

    auto ut = OP_API_UT(
        aclnnMaxPool3dWithArgmax, INPUT(self_desc, kernel_desc, stride_desc, padding_desc, dilation_desc, ceilMode),
        OUTPUT(out_tensor_desc, indices_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}
/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_Gather.cpp
 * \brief
 */

#include <vector>
#include <array>
#include "gtest/gtest.h"

#include "../../../op_host/op_api/aclnn_gather.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace op;
using namespace std;

class l2Gather : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "l2Gather SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "l2Gather TearDown" << std::endl;
    }

public:
    aclTensor* CreateAclTensor(
        vector<int64_t> view_shape, vector<int64_t> stride, int64_t offset, vector<int64_t> storage_shape,
        aclDataType dataType = ACL_FLOAT)
    {
        return aclCreateTensor(
            view_shape.data(), view_shape.size(), dataType, stride.data(), offset, ACL_FORMAT_ND, storage_shape.data(),
            storage_shape.size(), nullptr);
    }
};

TEST_F(l2Gather, case_norm_float32)
{
    auto tensor_desc = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    int64_t dim = 1;
    auto out_tensor_desc = TensorDesc({4, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto index_desc = TensorDesc({4, 5}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 4);

    auto ut = OP_API_UT(aclnnGather, INPUT(tensor_desc, dim, index_desc), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2Gather, case_norm_float32_aicpu)
{
    auto tensor_desc = TensorDesc({1, 4, 8, 4, 4, 9, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    int64_t dim = 0;
    auto out_tensor_desc = TensorDesc({1, 1, 1, 1, 1, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto index_desc = TensorDesc({1, 1, 1, 1, 1, 1, 1}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 1);

    auto ut = OP_API_UT(aclnnGather, INPUT(tensor_desc, dim, index_desc), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);

    
}

TEST_F(l2Gather, case_norm_float16)
{
    auto tensor_desc = TensorDesc({4, 4}, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(-1, 1);
    int64_t dim = -1;
    auto out_tensor_desc = TensorDesc({4, 5}, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(-1, 1);
    auto index_desc = TensorDesc({4, 5}, ACL_INT64, ACL_FORMAT_NCHW).ValueRange(0, 4);

    auto ut = OP_API_UT(aclnnGather, INPUT(tensor_desc, dim, index_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2Gather, case_norm_int32_dim0)
{
    auto tensor_desc = TensorDesc({4, 4}, ACL_INT32, ACL_FORMAT_NCHW).ValueRange(-1, 1);
    int64_t dim = 0;
    auto out_tensor_desc = TensorDesc({4, 4}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto index_desc = TensorDesc({4, 4}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 4);

    auto ut = OP_API_UT(aclnnGather, INPUT(tensor_desc, dim, index_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}


TEST_F(l2Gather, case_norm_int16)
{
    auto tensor_desc = TensorDesc({4, 4}, ACL_INT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    int64_t dim = 1;
    auto out_tensor_desc = TensorDesc({4, 4}, ACL_INT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto index_desc = TensorDesc({4, 4}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 4);

    auto ut = OP_API_UT(aclnnGather, INPUT(tensor_desc, dim, index_desc), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2Gather, case_norm_uint16)
{
    auto tensor_desc = TensorDesc({4, 4}, ACL_UINT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    int64_t dim = 1;
    auto out_tensor_desc = TensorDesc({4, 4}, ACL_UINT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto index_desc = TensorDesc({4, 4}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 4);

    auto ut = OP_API_UT(aclnnGather, INPUT(tensor_desc, dim, index_desc), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);

    // 
}

TEST_F(l2Gather, case_norm_uint32)
{
    auto tensor_desc = TensorDesc({4, 4}, ACL_UINT32, ACL_FORMAT_ND).ValueRange(-1, 1);
    int64_t dim = 1;
    auto out_tensor_desc = TensorDesc({4, 4}, ACL_UINT32, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto index_desc = TensorDesc({4, 4}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 4);

    auto ut = OP_API_UT(aclnnGather, INPUT(tensor_desc, dim, index_desc), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);

    // 
}

TEST_F(l2Gather, case_norm_uint64)
{
    auto tensor_desc = TensorDesc({4, 4}, ACL_UINT64, ACL_FORMAT_ND).ValueRange(-1, 1);
    int64_t dim = 1;
    auto out_tensor_desc = TensorDesc({4, 4}, ACL_UINT64, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto index_desc = TensorDesc({4, 4}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 4);

    auto ut = OP_API_UT(aclnnGather, INPUT(tensor_desc, dim, index_desc), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);

    // 
}

TEST_F(l2Gather, case_norm_int64)
{
    auto tensor_desc = TensorDesc({4, 4}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-1, 1);
    int64_t dim = 1;
    auto out_tensor_desc = TensorDesc({4, 4}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto index_desc = TensorDesc({4, 4}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 4);

    auto ut = OP_API_UT(aclnnGather, INPUT(tensor_desc, dim, index_desc), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);

    
}

TEST_F(l2Gather, case_nullptr_self)
{
    auto tensor_desc = TensorDesc({4, 4}, ACL_UINT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    int64_t dim = 1;
    auto out_tensor_desc = TensorDesc({4, 4}, ACL_UINT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto index_desc = TensorDesc({4, 4}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 4);

    auto ut = OP_API_UT(aclnnGather, INPUT((aclTensor*)nullptr, dim, index_desc), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2Gather, case_nullptr_index)
{
    auto tensor_desc = TensorDesc({4, 4}, ACL_UINT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    int64_t dim = 1;
    auto out_tensor_desc = TensorDesc({4, 4}, ACL_UINT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto index_desc = nullptr;

    auto ut = OP_API_UT(aclnnGather, INPUT(tensor_desc, dim, index_desc), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2Gather, case_nullptr_out)
{
    auto tensor_desc = TensorDesc({4, 4}, ACL_UINT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    int64_t dim = 1;
    auto out_tensor_desc = nullptr;
    auto index_desc = TensorDesc({4, 4}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 4);

    auto ut = OP_API_UT(aclnnGather, INPUT(tensor_desc, dim, index_desc), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2Gather, case_dtype_invalid)
{
    auto tensor_desc = TensorDesc({4, 4}, ACL_COMPLEX64, ACL_FORMAT_ND).ValueRange(-1, 1);
    int64_t dim = 1;
    auto out_tensor_desc = TensorDesc({4, 4}, ACL_COMPLEX64, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto index_desc = TensorDesc({4, 4}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 4);

    auto ut = OP_API_UT(aclnnGather, INPUT(tensor_desc, dim, index_desc), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2Gather, case_dims_invalid)
{
    auto tensor_desc = TensorDesc({1, 1, 1, 1, 1, 1, 1, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t dim = 1;
    auto out_tensor_desc = TensorDesc({1, 1, 1, 1, 1, 1, 1, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto index_desc = TensorDesc({1, 1, 1, 1, 1, 1, 1, 1, 1}, ACL_INT64, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnGather, INPUT(tensor_desc, dim, index_desc), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2Gather, case_dim_diff)
{
    auto tensor_desc = TensorDesc({1, 1, 1, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t dim = 1;
    auto out_tensor_desc = TensorDesc({1, 1, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto index_desc = TensorDesc({1, 1, 1, 1}, ACL_INT64, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnGather, INPUT(tensor_desc, dim, index_desc), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2Gather, case_dim_invalid)
{
    auto tensor_desc = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    int64_t dim = 5;
    auto out_tensor_desc = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto index_desc = TensorDesc({4, 4}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 4);

    auto ut = OP_API_UT(aclnnGather, INPUT(tensor_desc, dim, index_desc), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2Gather, case_out_shape_invalid)
{
    auto tensor_desc = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    int64_t dim = 1;
    auto out_tensor_desc = TensorDesc({4, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto index_desc = TensorDesc({4, 4}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 4);

    auto ut = OP_API_UT(aclnnGather, INPUT(tensor_desc, dim, index_desc), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2Gather, case_out_dtype_invalid)
{
    auto tensor_desc = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    int64_t dim = 1;
    auto out_tensor_desc = TensorDesc({4, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto index_desc = TensorDesc({4, 4}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 4);

    auto ut = OP_API_UT(aclnnGather, INPUT(tensor_desc, dim, index_desc), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2Gather, case_format_internal)
{
    auto tensor_desc = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_NC1HWC0).ValueRange(-1, 1);
    int64_t dim = 1;
    auto out_tensor_desc = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_NC1HWC0).ValueRange(-1, 1);
    auto index_desc = TensorDesc({4, 4}, ACL_INT64, ACL_FORMAT_NC1HWC0).ValueRange(0, 4);

    auto ut = OP_API_UT(aclnnGather, INPUT(tensor_desc, dim, index_desc), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2Gather, case_empty_tensor)
{
    auto tensor_desc = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    int64_t dim = 1;
    auto out_tensor_desc = TensorDesc({4, 0}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto index_desc = TensorDesc({4, 0}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 4);

    auto ut = OP_API_UT(aclnnGather, INPUT(tensor_desc, dim, index_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);

    
}

TEST_F(l2Gather, case_format_abnormal)
{
    auto tensor_desc = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_UNDEFINED).ValueRange(-1, 1);
    int64_t dim = 1;
    auto out_tensor_desc = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_UNDEFINED).ValueRange(-1, 1);
    auto index_desc = TensorDesc({4, 4}, ACL_INT64, ACL_FORMAT_UNDEFINED).ValueRange(0, 4);
    auto ut = OP_API_UT(aclnnGather, INPUT(tensor_desc, dim, index_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2Gather, case_dtype_abnormal)
{
    auto tensor_desc = TensorDesc({4, 4}, ACL_DT_UNDEFINED, ACL_FORMAT_ND).ValueRange(-1, 1);
    int64_t dim = 1;
    auto out_tensor_desc = TensorDesc({4, 4}, ACL_DT_UNDEFINED, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto index_desc = TensorDesc({4, 4}, ACL_DT_UNDEFINED, ACL_FORMAT_ND).ValueRange(0, 4);

    auto ut = OP_API_UT(aclnnGather, INPUT(tensor_desc, dim, index_desc), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2Gather, case_strided)
{
    auto tensor_desc =
        TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 2}, 0, {4, 2}).Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8});
    int64_t dim = 0;
    auto out_tensor_desc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 2}, 0, {4, 2});
    auto index_desc = TensorDesc({2, 4}, ACL_INT64, ACL_FORMAT_ND, {1, 2}, 0, {4, 2}).ValueRange(0, 2);

    auto ut = OP_API_UT(aclnnGather, INPUT(tensor_desc, dim, index_desc), OUTPUT(out_tensor_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2Gather, case_ascend910B2_expand)
{
    auto xTensor = CreateAclTensor({4096, 6144}, {6144, 1}, 0, {4096, 6144});
    auto indexTensor = CreateAclTensor(
        {2076, 6144}, {1, 0}, 0,
        {
            2076,
        },
        ACL_INT64);
    auto outTensor = CreateAclTensor({2076, 6144}, {6144, 1}, 0, {2076, 6144});
    int64_t dim = 0;
    uint64_t workspaceSize = 0U;
    aclOpExecutor* exe = nullptr;
    auto aclRet = aclnnGatherGetWorkspaceSize(xTensor, dim, indexTensor, outTensor, &workspaceSize, &exe);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    EXPECT_NE(exe, nullptr);
}

TEST_F(l2Gather, case_ascend910B2_last_dim)
{
    auto xTensor = CreateAclTensor({4096, 6144}, {6144, 1}, 0, {4096, 6144});
    auto indexTensor = CreateAclTensor(
        {2076, 6144}, {1, 0}, 0,
        {
            2076,
        },
        ACL_INT64);
    auto outTensor = CreateAclTensor({2076, 6144}, {6144, 1}, 0, {2076, 6144});
    int64_t dim = -1;
    uint64_t workspaceSize = 0U;
    aclOpExecutor* exe = nullptr;
    auto aclRet = aclnnGatherGetWorkspaceSize(xTensor, dim, indexTensor, outTensor, &workspaceSize, &exe);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    EXPECT_NE(exe, nullptr);
}

TEST_F(l2Gather, ascend910_95_case_norm_int32)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND910_95);
    auto tensor_desc = TensorDesc({4, 4}, ACL_INT32, ACL_FORMAT_NCHW).ValueRange(-1, 1);
    int64_t dim = 1;
    auto out_tensor_desc = TensorDesc({4, 4}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto index_desc = TensorDesc({4, 4}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 4);

    auto ut = OP_API_UT(aclnnGather, INPUT(tensor_desc, dim, index_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);

    
}

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
#include "../../../op_host/op_api/aclnn_threshold.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"

class l2_threshold_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "l2_threshold_test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "l2_threshold_test TearDown" << std::endl;
    }
};

// self为空指针
TEST_F(l2_threshold_test, l2_test_null_self)
{
    auto selfDesc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto threshold = ScalarDesc(1);
    auto value = ScalarDesc(10);

    auto ut = OP_API_UT(aclnnThreshold, INPUT((aclTensor*)nullptr, threshold, value), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

// out为空指针
TEST_F(l2_threshold_test, l2_test_null_out)
{
    auto selfDesc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto threshold = ScalarDesc(1);
    auto value = ScalarDesc(10);

    auto ut = OP_API_UT(aclnnThreshold, INPUT(selfDesc, threshold, value), OUTPUT((aclTensor*)nullptr));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

// self为空tensor
TEST_F(l2_threshold_test, l2_test_0tensor_self)
{
    auto selfDesc = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto threshold = ScalarDesc(1);
    auto value = ScalarDesc(10);

    auto ut = OP_API_UT(aclnnThreshold, INPUT(selfDesc, threshold, value), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// self的int64数据类型在支持范围内
TEST_F(l2_threshold_test, l2_test_self_int64)
{
    auto selfDesc = TensorDesc({2, 3}, ACL_INT64, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2, 3}, ACL_INT64, ACL_FORMAT_ND);
    auto threshold = ScalarDesc(4);
    auto value = ScalarDesc(5);

    auto ut = OP_API_UT(aclnnThreshold, INPUT(selfDesc, threshold, value), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// self的double数据类型不在支持范围内
TEST_F(l2_threshold_test, l2_test_self_double)
{
    auto selfDesc = TensorDesc({2, 3}, ACL_DOUBLE, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2, 3}, ACL_DOUBLE, ACL_FORMAT_ND);
    auto threshold = ScalarDesc(4);
    auto value = ScalarDesc(5);

    auto ut = OP_API_UT(aclnnThreshold, INPUT(selfDesc, threshold, value), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// self的bfloat16数据类型不在支持范围内
// TEST_F(l2_threshold_test, l2_test_self_bf16)
// {
//     auto selfDesc = TensorDesc({2, 3}, ACL_BF16, ACL_FORMAT_ND);
//     auto outDesc = TensorDesc({2, 3}, ACL_BF16, ACL_FORMAT_ND);
//     auto threshold = ScalarDesc(4.0f);
//     auto value = ScalarDesc(5);

//     auto ut = OP_API_UT(aclnnThreshold, INPUT(selfDesc, threshold, value), OUTPUT(outDesc));

//     uint64_t workspaceSize = 0;
//     aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
//     EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
// }

// self的bool数据类型不在支持范围内
TEST_F(l2_threshold_test, l2_test_self_bool)
{
    auto selfDesc = TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND);
    auto threshold = ScalarDesc(4.0f);
    auto value = ScalarDesc(5);

    auto ut = OP_API_UT(aclnnThreshold, INPUT(selfDesc, threshold, value), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// self的complex64数据类型不在支持范围内
TEST_F(l2_threshold_test, l2_test_self_complex64)
{
    auto selfDesc = TensorDesc({2, 3}, ACL_COMPLEX64, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2, 3}, ACL_COMPLEX64, ACL_FORMAT_ND);
    auto threshold = ScalarDesc(4.0f);
    auto value = ScalarDesc(5);

    auto ut = OP_API_UT(aclnnThreshold, INPUT(selfDesc, threshold, value), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// self的complex128数据类型不在支持范围内
TEST_F(l2_threshold_test, l2_test_self_complex128)
{
    auto selfDesc = TensorDesc({2, 3}, ACL_COMPLEX128, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2, 3}, ACL_COMPLEX128, ACL_FORMAT_ND);
    auto threshold = ScalarDesc(4.0f);
    auto value = ScalarDesc(5);

    auto ut = OP_API_UT(aclnnThreshold, INPUT(selfDesc, threshold, value), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// self的int16数据类型不在支持范围内
TEST_F(l2_threshold_test, l2_test_self_int16)
{
    auto selfDesc = TensorDesc({2, 3}, ACL_INT16, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2, 3}, ACL_INT16, ACL_FORMAT_ND);
    auto threshold = ScalarDesc(4.0f);
    auto value = ScalarDesc(5);

    auto ut = OP_API_UT(aclnnThreshold, INPUT(selfDesc, threshold, value), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// self的float32数据类型在支持范围内
TEST_F(l2_threshold_test, l2_test_self_fp32)
{
    auto selfDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto threshold = ScalarDesc(4.0f);
    auto value = ScalarDesc(10.1f);

    auto ut = OP_API_UT(aclnnThreshold, INPUT(selfDesc, threshold, value), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// self的float16数据类型在支持范围内
TEST_F(l2_threshold_test, l2_test_self_fp16)
{
    auto selfDesc = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto threshold = ScalarDesc(4.0f);
    auto value = ScalarDesc(10.1f);

    auto ut = OP_API_UT(aclnnThreshold, INPUT(selfDesc, threshold, value), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// self的int8数据类型在支持范围内
TEST_F(l2_threshold_test, l2_test_self_int8)
{
    auto selfDesc = TensorDesc({2, 3}, ACL_INT8, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2, 3}, ACL_INT8, ACL_FORMAT_ND);
    auto threshold = ScalarDesc(4);
    auto value = ScalarDesc(10);

    auto ut = OP_API_UT(aclnnThreshold, INPUT(selfDesc, threshold, value), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// self的uint8数据类型在支持范围内
TEST_F(l2_threshold_test, l2_test_self_uint8)
{
    auto selfDesc = TensorDesc({2, 3}, ACL_UINT8, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2, 3}, ACL_UINT8, ACL_FORMAT_ND);
    auto threshold = ScalarDesc(4);
    auto value = ScalarDesc(10);

    auto ut = OP_API_UT(aclnnThreshold, INPUT(selfDesc, threshold, value), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// self的int32数据类型在支持范围内
TEST_F(l2_threshold_test, l2_test_self_int32)
{
    auto selfDesc = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND);
    auto threshold = ScalarDesc(4);
    auto value = ScalarDesc(10);

    auto ut = OP_API_UT(aclnnThreshold, INPUT(selfDesc, threshold, value), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// self为8维
TEST_F(l2_threshold_test, l2_test_self_shape8)
{
    auto selfDesc = TensorDesc({1, 2, 3, 4, 5, 6, 7, 8}, ACL_INT32, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({1, 2, 3, 4, 5, 6, 7, 8}, ACL_INT32, ACL_FORMAT_ND);
    auto threshold = ScalarDesc(4);
    auto value = ScalarDesc(10);

    auto ut = OP_API_UT(aclnnThreshold, INPUT(selfDesc, threshold, value), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// self为9维,不支持
TEST_F(l2_threshold_test, l2_test_self_shape9)
{
    auto selfDesc = TensorDesc({1, 2, 3, 4, 5, 6, 7, 8, 9}, ACL_INT32, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({1, 2, 3, 4, 5, 6, 7, 8, 9}, ACL_INT32, ACL_FORMAT_ND);
    auto threshold = ScalarDesc(4);
    auto value = ScalarDesc(10);

    auto ut = OP_API_UT(aclnnThreshold, INPUT(selfDesc, threshold, value), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// threshold为空指针,不支持
TEST_F(l2_threshold_test, l2_test_threshold_nullptr)
{
    auto selfDesc = TensorDesc({1, 2, 3, 4, 5, 6}, ACL_INT32, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({1, 2, 3, 4, 5, 6}, ACL_INT32, ACL_FORMAT_ND);
    auto threshold = ScalarDesc(4);
    auto value = ScalarDesc(10);

    auto ut = OP_API_UT(aclnnThreshold, INPUT(selfDesc, (aclScalar*)nullptr, value), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

// threshold为空指针,不支持
TEST_F(l2_threshold_test, l2_test_self_dtype_cant_cast_out_dtype)
{
    auto selfDesc = TensorDesc({1, 2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({1, 2, 3, 4}, ACL_INT32, ACL_FORMAT_ND);
    auto threshold = ScalarDesc(4);
    auto value = ScalarDesc(10);

    auto ut = OP_API_UT(aclnnThreshold, INPUT(selfDesc, threshold, value), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// threshold_测试：self的float32数据类型在支持范围内
TEST_F(l2_threshold_test, l2_test_selfRef_fp32)
{
    auto selfRefDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto threshold = ScalarDesc(4.0f);
    auto value = ScalarDesc(10.1f);

    auto ut = OP_API_UT(aclnnInplaceThreshold, INPUT(selfRefDesc, threshold, value), OUTPUT());

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// threshold_测试：self的float16数据类型在支持范围内
TEST_F(l2_threshold_test, l2_test_selfRef_fp16)
{
    auto selfRefDesc = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto threshold = ScalarDesc(4.0f);
    auto value = ScalarDesc(10.1f);

    auto ut = OP_API_UT(aclnnInplaceThreshold, INPUT(selfRefDesc, threshold, value), OUTPUT());

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// threshold_测试：self的int8数据类型在支持范围内
TEST_F(l2_threshold_test, l2_test_selfRef_int8)
{
    auto selfRefDesc = TensorDesc({2, 3}, ACL_INT8, ACL_FORMAT_ND);
    auto threshold = ScalarDesc(4);
    auto value = ScalarDesc(10);

    auto ut = OP_API_UT(aclnnInplaceThreshold, INPUT(selfRefDesc, threshold, value), OUTPUT());

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// threshold_测试：self的uint8数据类型在支持范围内
TEST_F(l2_threshold_test, l2_test_selfRef_uint8)
{
    auto selfRefDesc = TensorDesc({2, 3}, ACL_UINT8, ACL_FORMAT_ND);
    auto threshold = ScalarDesc(4);
    auto value = ScalarDesc(10);

    auto ut = OP_API_UT(aclnnInplaceThreshold, INPUT(selfRefDesc, threshold, value), OUTPUT());

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// threshold_测试：self的int32数据类型在支持范围内
TEST_F(l2_threshold_test, l2_test_selfRef_int32)
{
    auto selfRefDesc = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND);
    auto threshold = ScalarDesc(4);
    auto value = ScalarDesc(10);

    auto ut = OP_API_UT(aclnnInplaceThreshold, INPUT(selfRefDesc, threshold, value), OUTPUT());

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// threshold_测试：self的int64数据类型在支持范围内
TEST_F(l2_threshold_test, l2_test_selfRef_int64)
{
    auto selfRefDesc = TensorDesc({2, 3}, ACL_INT64, ACL_FORMAT_ND);
    auto threshold = ScalarDesc(4);
    auto value = ScalarDesc(5);

    auto ut = OP_API_UT(aclnnInplaceThreshold, INPUT(selfRefDesc, threshold, value), OUTPUT());

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

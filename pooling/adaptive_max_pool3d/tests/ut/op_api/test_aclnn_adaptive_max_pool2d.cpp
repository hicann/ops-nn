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

#include "opdev/op_log.h"
#include "../../../op_api/aclnn_adaptive_max_pool2d.h"
#include "op_api_ut_common/array_desc.h"
#include "op_api_ut_common/inner/types.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace op;
using namespace std;

class l2_adaptive_max_pool2d_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "l2_adaptive_max_pool2d_test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "l2_adaptive_max_pool2d_test TearDown" << std::endl;
    }
};

TEST_F(l2_adaptive_max_pool2d_test, nullptr_self)
{
    auto selfDesc = TensorDesc({1, 1, 5, 5}, ACL_FLOAT, ACL_FORMAT_NCHW);
    vector<int64_t> outputSize = {2, 2};
    auto outputSizeArray = IntArrayDesc(outputSize);

    auto valDesc = TensorDesc({1, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto indDesc = TensorDesc({1, 1, 2, 2}, ACL_INT64, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnAdaptiveMaxPool2d, INPUT(nullptr, outputSizeArray), OUTPUT(valDesc, indDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_adaptive_max_pool2d_test, nullptr_output_size)
{
    auto selfDesc = TensorDesc({1, 1, 5, 5}, ACL_FLOAT, ACL_FORMAT_NCHW);
    vector<int64_t> outputSize = {2, 2};
    auto outputSizeArray = IntArrayDesc(outputSize);

    auto valDesc = TensorDesc({1, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto indDesc = TensorDesc({1, 1, 2, 2}, ACL_INT64, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnAdaptiveMaxPool2d, INPUT(selfDesc, nullptr), OUTPUT(valDesc, indDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_adaptive_max_pool2d_test, nullptr_outputOut)
{
    auto selfDesc = TensorDesc({1, 1, 5, 5}, ACL_FLOAT, ACL_FORMAT_NCHW);
    vector<int64_t> outputSize = {2, 2};
    auto outputSizeArray = IntArrayDesc(outputSize);

    auto valDesc = TensorDesc({1, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto indDesc = TensorDesc({1, 1, 2, 2}, ACL_INT64, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnAdaptiveMaxPool2d, INPUT(selfDesc, outputSizeArray), OUTPUT(nullptr, indDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_adaptive_max_pool2d_test, nullptr_indicesOut)
{
    auto selfDesc = TensorDesc({1, 1, 5, 5}, ACL_FLOAT, ACL_FORMAT_NCHW);
    vector<int64_t> outputSize = {2, 2};
    auto outputSizeArray = IntArrayDesc(outputSize);

    auto valDesc = TensorDesc({1, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto indDesc = TensorDesc({1, 1, 2, 2}, ACL_INT64, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnAdaptiveMaxPool2d, INPUT(selfDesc, outputSizeArray), OUTPUT(valDesc, nullptr));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_adaptive_max_pool2d_test, self_float16)
{
    auto selfDesc = TensorDesc({1, 1, 5, 5}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    vector<int64_t> outputSize = {2, 2};
    auto outputSizeArray = IntArrayDesc(outputSize);

    auto valDesc = TensorDesc({1, 1, 2, 2}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto indDesc = TensorDesc({1, 1, 2, 2}, ACL_INT64, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnAdaptiveMaxPool2d, INPUT(selfDesc, outputSizeArray), OUTPUT(valDesc, indDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
}

TEST_F(l2_adaptive_max_pool2d_test, ascend910B2_self_float16)
{
    auto selfDesc = TensorDesc({1, 1, 5, 5}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    vector<int64_t> outputSize = {2, 2};
    auto outputSizeArray = IntArrayDesc(outputSize);

    auto valDesc = TensorDesc({1, 1, 2, 2}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto indDesc = TensorDesc({1, 1, 2, 2}, ACL_INT64, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnAdaptiveMaxPool2d, INPUT(selfDesc, outputSizeArray), OUTPUT(valDesc, indDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
}

TEST_F(l2_adaptive_max_pool2d_test, ascend910B2_self_0_tensor)
{
    auto selfDesc = TensorDesc({0, 1, 5, 5}, ACL_FLOAT, ACL_FORMAT_NCHW);
    vector<int64_t> outputSize = {2, 2};
    auto outputSizeArray = IntArrayDesc(outputSize);

    auto valDesc = TensorDesc({1, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto indDesc = TensorDesc({1, 1, 2, 2}, ACL_INT64, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnAdaptiveMaxPool2d, INPUT(selfDesc, outputSizeArray), OUTPUT(valDesc, indDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
}

TEST_F(l2_adaptive_max_pool2d_test, self_0_tensor)
{
    auto selfDesc = TensorDesc({0, 1, 5, 5}, ACL_FLOAT, ACL_FORMAT_NCHW);
    vector<int64_t> outputSize = {2, 2};
    auto outputSizeArray = IntArrayDesc(outputSize);

    auto valDesc = TensorDesc({1, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto indDesc = TensorDesc({1, 1, 2, 2}, ACL_INT64, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnAdaptiveMaxPool2d, INPUT(selfDesc, outputSizeArray), OUTPUT(valDesc, indDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
}

TEST_F(l2_adaptive_max_pool2d_test, self_float32)
{
    auto selfDesc = TensorDesc({1, 1, 5, 5}, ACL_FLOAT, ACL_FORMAT_NCHW);
    vector<int64_t> outputSize = {2, 2};
    auto outputSizeArray = IntArrayDesc(outputSize);

    auto valDesc = TensorDesc({1, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto indDesc = TensorDesc({1, 1, 2, 2}, ACL_INT64, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnAdaptiveMaxPool2d, INPUT(selfDesc, outputSizeArray), OUTPUT(valDesc, indDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
    ut.TestPrecision();
}

TEST_F(l2_adaptive_max_pool2d_test, ascend910B2_self_float32)
{
    auto selfDesc = TensorDesc({1, 1, 5, 5}, ACL_FLOAT, ACL_FORMAT_NCHW);
    vector<int64_t> outputSize = {2, 2};
    auto outputSizeArray = IntArrayDesc(outputSize);

    auto valDesc = TensorDesc({1, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto indDesc = TensorDesc({1, 1, 2, 2}, ACL_INT64, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnAdaptiveMaxPool2d, INPUT(selfDesc, outputSizeArray), OUTPUT(valDesc, indDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
    ut.TestPrecision();
}

TEST_F(l2_adaptive_max_pool2d_test, ascend910B2_self_double)
{
    auto selfDesc = TensorDesc({1, 1, 5, 5}, ACL_DOUBLE, ACL_FORMAT_NCHW);
    vector<int64_t> outputSize = {2, 2};
    auto outputSizeArray = IntArrayDesc(outputSize);

    auto valDesc = TensorDesc({1, 1, 2, 2}, ACL_DOUBLE, ACL_FORMAT_NCHW);
    auto indDesc = TensorDesc({1, 1, 2, 2}, ACL_INT64, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnAdaptiveMaxPool2d, INPUT(selfDesc, outputSizeArray), OUTPUT(valDesc, indDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
    ut.TestPrecision();
}

TEST_F(l2_adaptive_max_pool2d_test, self_double)
{
    auto selfDesc = TensorDesc({1, 1, 5, 5}, ACL_DOUBLE, ACL_FORMAT_NCHW);
    vector<int64_t> outputSize = {2, 2};
    auto outputSizeArray = IntArrayDesc(outputSize);

    auto valDesc = TensorDesc({1, 1, 2, 2}, ACL_DOUBLE, ACL_FORMAT_NCHW);
    auto indDesc = TensorDesc({1, 1, 2, 2}, ACL_INT64, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnAdaptiveMaxPool2d, INPUT(selfDesc, outputSizeArray), OUTPUT(valDesc, indDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
    ut.TestPrecision();
}

TEST_F(l2_adaptive_max_pool2d_test, self_invalid_int32)
{
    auto selfDesc = TensorDesc({1, 1, 5, 5}, ACL_INT32, ACL_FORMAT_NCHW);
    vector<int64_t> outputSize = {2, 2};
    auto outputSizeArray = IntArrayDesc(outputSize);

    auto valDesc = TensorDesc({1, 1, 2, 2}, ACL_INT32, ACL_FORMAT_NCHW);
    auto indDesc = TensorDesc({1, 1, 2, 2}, ACL_INT64, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnAdaptiveMaxPool2d, INPUT(selfDesc, outputSizeArray), OUTPUT(valDesc, indDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_adaptive_max_pool2d_test, self_invalid_int8)
{
    auto selfDesc = TensorDesc({1, 1, 5, 5}, ACL_INT8, ACL_FORMAT_NCHW);
    vector<int64_t> outputSize = {2, 2};
    auto outputSizeArray = IntArrayDesc(outputSize);

    auto valDesc = TensorDesc({1, 1, 2, 2}, ACL_INT8, ACL_FORMAT_NCHW);
    auto indDesc = TensorDesc({1, 1, 2, 2}, ACL_INT64, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnAdaptiveMaxPool2d, INPUT(selfDesc, outputSizeArray), OUTPUT(valDesc, indDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_adaptive_max_pool2d_test, self_invalid_int16)
{
    auto selfDesc = TensorDesc({1, 1, 5, 5}, ACL_INT16, ACL_FORMAT_NCHW);
    vector<int64_t> outputSize = {2, 2};
    auto outputSizeArray = IntArrayDesc(outputSize);

    auto valDesc = TensorDesc({1, 1, 2, 2}, ACL_INT16, ACL_FORMAT_NCHW);
    auto indDesc = TensorDesc({1, 1, 2, 2}, ACL_INT64, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnAdaptiveMaxPool2d, INPUT(selfDesc, outputSizeArray), OUTPUT(valDesc, indDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_adaptive_max_pool2d_test, self_invalid_uint8)
{
    auto selfDesc = TensorDesc({1, 1, 5, 5}, ACL_UINT8, ACL_FORMAT_NCHW);
    vector<int64_t> outputSize = {2, 2};
    auto outputSizeArray = IntArrayDesc(outputSize);

    auto valDesc = TensorDesc({1, 1, 2, 2}, ACL_UINT8, ACL_FORMAT_NCHW);
    auto indDesc = TensorDesc({1, 1, 2, 2}, ACL_INT64, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnAdaptiveMaxPool2d, INPUT(selfDesc, outputSizeArray), OUTPUT(valDesc, indDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_adaptive_max_pool2d_test, self_invalid_int64)
{
    auto selfDesc = TensorDesc({1, 1, 5, 5}, ACL_INT64, ACL_FORMAT_NCHW);
    vector<int64_t> outputSize = {2, 2};
    auto outputSizeArray = IntArrayDesc(outputSize);

    auto valDesc = TensorDesc({1, 1, 2, 2}, ACL_INT64, ACL_FORMAT_NCHW);
    auto indDesc = TensorDesc({1, 1, 2, 2}, ACL_INT64, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnAdaptiveMaxPool2d, INPUT(selfDesc, outputSizeArray), OUTPUT(valDesc, indDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_adaptive_max_pool2d_test, self_float16_out_float32)
{
    auto selfDesc = TensorDesc({1, 1, 5, 5}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    vector<int64_t> outputSize = {2, 2};
    auto outputSizeArray = IntArrayDesc(outputSize);

    auto valDesc = TensorDesc({1, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto indDesc = TensorDesc({1, 1, 2, 2}, ACL_INT64, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnAdaptiveMaxPool2d, INPUT(selfDesc, outputSizeArray), OUTPUT(valDesc, indDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_adaptive_max_pool2d_test, self_shape2)
{
    auto selfDesc = TensorDesc({1, 5}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    vector<int64_t> outputSize = {2, 2};
    auto outputSizeArray = IntArrayDesc(outputSize);

    auto valDesc = TensorDesc({1, 1, 2, 2}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto indDesc = TensorDesc({1, 1, 2, 2}, ACL_INT64, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnAdaptiveMaxPool2d, INPUT(selfDesc, outputSizeArray), OUTPUT(valDesc, indDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_adaptive_max_pool2d_test, self_shape5)
{
    auto selfDesc = TensorDesc({1, 2, 3, 4, 5}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    vector<int64_t> outputSize = {2, 2};
    auto outputSizeArray = IntArrayDesc(outputSize);

    auto valDesc = TensorDesc({1, 1, 2, 2}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto indDesc = TensorDesc({1, 1, 2, 2}, ACL_INT64, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnAdaptiveMaxPool2d, INPUT(selfDesc, outputSizeArray), OUTPUT(valDesc, indDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_adaptive_max_pool2d_test, self_size_is_neg)
{
    auto selfDesc = TensorDesc({1, -1, 5, 5}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    vector<int64_t> outputSize = {2, 2};
    auto outputSizeArray = IntArrayDesc(outputSize);

    auto valDesc = TensorDesc({1, 1, 2, 2}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto indDesc = TensorDesc({1, 1, 2, 2}, ACL_INT64, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnAdaptiveMaxPool2d, INPUT(selfDesc, outputSizeArray), OUTPUT(valDesc, indDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_adaptive_max_pool2d_test, output_size3)
{
    auto selfDesc = TensorDesc({1, 1, 5, 5}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    vector<int64_t> outputSize = {2, 2, 3};
    auto outputSizeArray = IntArrayDesc(outputSize);

    auto valDesc = TensorDesc({1, 1, 2, 2}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto indDesc = TensorDesc({1, 1, 2, 2}, ACL_INT64, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnAdaptiveMaxPool2d, INPUT(selfDesc, outputSizeArray), OUTPUT(valDesc, indDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_adaptive_max_pool2d_test, output_size_is_neg)
{
    auto selfDesc = TensorDesc({1, 1, 5, 5}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    vector<int64_t> outputSize = {-2, 2};
    auto outputSizeArray = IntArrayDesc(outputSize);

    auto valDesc = TensorDesc({1, 1, 2, 2}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto indDesc = TensorDesc({1, 1, 2, 2}, ACL_INT64, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnAdaptiveMaxPool2d, INPUT(selfDesc, outputSizeArray), OUTPUT(valDesc, indDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_adaptive_max_pool2d_test, output_shape5_indices_shape4)
{
    auto selfDesc = TensorDesc({1, 1, 5, 5}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    vector<int64_t> outputSize = {2, 2};
    auto outputSizeArray = IntArrayDesc(outputSize);

    auto valDesc = TensorDesc({1, 1, 2, 2, 3}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto indDesc = TensorDesc({1, 1, 2, 2}, ACL_INT64, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnAdaptiveMaxPool2d, INPUT(selfDesc, outputSizeArray), OUTPUT(valDesc, indDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// TEST_F(l2_adaptive_max_pool2d_test, self_float32_invalid_format)
// {
//     auto selfDesc = TensorDesc({1, 1, 5, 5}, ACL_FLOAT, ACL_FORMAT_ND);
//     vector<int64_t> outputSize = {2, 2};
//     auto outputSizeArray = IntArrayDesc(outputSize);

//     auto valDesc = TensorDesc({1, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
//     auto indDesc = TensorDesc({1, 1, 2, 2}, ACL_INT64, ACL_FORMAT_ND);

//     auto ut = OP_API_UT(aclnnAdaptiveMaxPool2d, INPUT(selfDesc, outputSizeArray), OUTPUT(valDesc, indDesc));

//     uint64_t workspaceSize = 0;
//     aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
//     EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
// }

TEST_F(l2_adaptive_max_pool2d_test, self_float32_nhwc)
{
    auto selfDesc = TensorDesc({1, 5, 5, 1}, ACL_FLOAT, ACL_FORMAT_NHWC);
    vector<int64_t> outputSize = {2, 2};
    auto outputSizeArray = IntArrayDesc(outputSize);

    auto valDesc = TensorDesc({1, 2, 2, 1}, ACL_FLOAT, ACL_FORMAT_NHWC);
    auto indDesc = TensorDesc({1, 2, 2, 1}, ACL_INT64, ACL_FORMAT_NHWC);

    auto ut = OP_API_UT(aclnnAdaptiveMaxPool2d, INPUT(selfDesc, outputSizeArray), OUTPUT(valDesc, indDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
}

TEST_F(l2_adaptive_max_pool2d_test, ascend910B2_self_float32_nhwc)
{
    auto selfDesc = TensorDesc({1, 5, 5, 1}, ACL_FLOAT, ACL_FORMAT_NHWC);
    vector<int64_t> outputSize = {2, 2};
    auto outputSizeArray = IntArrayDesc(outputSize);

    auto valDesc = TensorDesc({1, 2, 2, 1}, ACL_FLOAT, ACL_FORMAT_NHWC);
    auto indDesc = TensorDesc({1, 2, 2, 1}, ACL_INT64, ACL_FORMAT_NHWC);

    auto ut = OP_API_UT(aclnnAdaptiveMaxPool2d, INPUT(selfDesc, outputSizeArray), OUTPUT(valDesc, indDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
}

TEST_F(l2_adaptive_max_pool2d_test, ascend910B2_self_float16_ncl_maxpool3dwithargmaxv2)
{
    auto selfDesc = TensorDesc({1, 64, 16, 8}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    vector<int64_t> outputSize = {1, 8};
    auto outputSizeArray = IntArrayDesc(outputSize);

    auto valDesc = TensorDesc({1, 64, 1, 8}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto indDesc = TensorDesc({1, 64, 1, 8}, ACL_INT64, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnAdaptiveMaxPool2d, INPUT(selfDesc, outputSizeArray), OUTPUT(valDesc, indDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
}
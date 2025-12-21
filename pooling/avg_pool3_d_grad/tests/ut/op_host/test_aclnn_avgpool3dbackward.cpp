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

#include "opdev/platform.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"

#include "../../../op_host/op_api/aclnn_avgpool3d_backward.h"

using namespace op;
using namespace std;

class l2_avgpool3dbackward_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "avgpool3dbackward_test Setup" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "avgpool3dbackward_test TearDown" << std::endl;
    }
};

// 正常场景
TEST_F(l2_avgpool3dbackward_test, Ascend910B2_case_1)
{
    int divisor_override = 0;
    bool count_include_pad = false;
    bool ceil_mode = true;

    vector<int64_t> vector_grad_output = {1, 16, 1, 1, 1};
    vector<int64_t> vector_kernel_size = {4, 4, 4};
    vector<int64_t> vector_stride = {1, 1, 1};
    vector<int64_t> vector_padding = {0, 0, 0};
    vector<int64_t> vector_grad_input = {1, 16, 4, 4, 4};
    vector<int64_t> vector_output = {1, 16, 4, 4, 4};

    auto grad_output_desc = TensorDesc(vector_grad_output, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto kernel_size_desc = IntArrayDesc(vector_kernel_size);
    auto stride_desc = IntArrayDesc(vector_stride);
    auto padding_desc = IntArrayDesc(vector_padding);
    auto grad_input_desc = TensorDesc(vector_grad_input, ACL_FLOAT, ACL_FORMAT_ND);
    auto output = TensorDesc(vector_output, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnAvgPool3dBackward, // host api第二段接口名称
        INPUT(
            grad_output_desc, grad_input_desc, kernel_size_desc, stride_desc, padding_desc, ceil_mode,
            count_include_pad, divisor_override), // host api输入
        OUTPUT(output));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size); // check op graph
    //  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// gradoutput 为空指针
TEST_F(l2_avgpool3dbackward_test, Ascend910B2_case_2)
{
    int divisor_override = 0;
    bool count_include_pad = false;
    bool ceil_mode = true;

    vector<int64_t> vector_grad_output = {1, 16, 1, 1, 1};
    vector<int64_t> vector_kernel_size = {4, 4, 4};
    vector<int64_t> vector_stride = {1, 1, 1};
    vector<int64_t> vector_padding = {0, 0, 0};
    vector<int64_t> vector_grad_input = {1, 16, 4, 4, 4};
    vector<int64_t> vector_output = {1, 16, 4, 4, 4};

    const aclTensor* grad_output_desc = nullptr;
    auto kernel_size_desc = IntArrayDesc(vector_kernel_size);
    auto stride_desc = IntArrayDesc(vector_stride);
    auto padding_desc = IntArrayDesc(vector_padding);
    auto grad_input_desc = TensorDesc(vector_grad_input, ACL_FLOAT, ACL_FORMAT_ND);
    auto output = TensorDesc(vector_output, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnAvgPool3dBackward, // host api第二段接口名称
        INPUT(
            grad_output_desc, grad_input_desc, kernel_size_desc, stride_desc, padding_desc, ceil_mode,
            count_include_pad, divisor_override), // host api输入
        OUTPUT(output));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size); // check op graph
    //  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// ksize 为空指针
TEST_F(l2_avgpool3dbackward_test, Ascend910B2_case_3)
{
    int divisor_override = 0;
    bool count_include_pad = false;
    bool ceil_mode = true;

    vector<int64_t> vector_grad_output = {1, 16, 1, 1, 1};
    vector<int64_t> vector_kernel_size = {4, 4, 4};
    vector<int64_t> vector_stride = {1, 1, 1};
    vector<int64_t> vector_padding = {0, 0, 0};
    vector<int64_t> vector_grad_input = {1, 16, 4, 4, 4};
    vector<int64_t> vector_output = {1, 16, 4, 4, 4};

    auto grad_output_desc = TensorDesc(vector_grad_output, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    const aclIntArray* kernel_size_desc = nullptr;
    auto stride_desc = IntArrayDesc(vector_stride);
    auto padding_desc = IntArrayDesc(vector_padding);
    auto grad_input_desc = TensorDesc(vector_grad_input, ACL_FLOAT, ACL_FORMAT_ND);
    auto output = TensorDesc(vector_output, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnAvgPool3dBackward, // host api第二段接口名称
        INPUT(
            grad_output_desc, grad_input_desc, kernel_size_desc, stride_desc, padding_desc, ceil_mode,
            count_include_pad, divisor_override), // host api输入
        OUTPUT(output));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size); // check op graph
    //  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// output为空指针
TEST_F(l2_avgpool3dbackward_test, Ascend910B2_case_4)
{
    int divisor_override = 0;
    bool count_include_pad = false;
    bool ceil_mode = true;

    vector<int64_t> vector_grad_output = {1, 16, 1, 1, 1};
    vector<int64_t> vector_kernel_size = {4, 4, 4};
    vector<int64_t> vector_stride = {1, 1, 1};
    vector<int64_t> vector_padding = {0, 0, 0};
    vector<int64_t> vector_grad_input = {1, 16, 4, 4, 4};
    vector<int64_t> vector_output = {1, 16, 4, 4, 4};

    auto grad_output_desc = TensorDesc(vector_grad_output, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto kernel_size_desc = IntArrayDesc(vector_kernel_size);
    auto stride_desc = IntArrayDesc(vector_stride);
    auto padding_desc = IntArrayDesc(vector_padding);
    auto grad_input_desc = TensorDesc(vector_grad_input, ACL_FLOAT, ACL_FORMAT_ND);
    aclTensor* output = nullptr;

    auto ut = OP_API_UT(
        aclnnAvgPool3dBackward, // host api第二段接口名称
        INPUT(
            grad_output_desc, grad_input_desc, kernel_size_desc, stride_desc, padding_desc, ceil_mode,
            count_include_pad, divisor_override), // host api输入
        OUTPUT(output));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size); // check op graph
    //  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// gradoutput数据类型不合法
TEST_F(l2_avgpool3dbackward_test, Ascend910B2_case_5)
{
    int divisor_override = 0;
    bool count_include_pad = false;
    bool ceil_mode = true;

    vector<int64_t> vector_grad_output = {1, 16, 1, 1, 1};
    vector<int64_t> vector_kernel_size = {4, 4, 4};
    vector<int64_t> vector_stride = {1, 1, 1};
    vector<int64_t> vector_padding = {0, 0, 0};
    vector<int64_t> vector_grad_input = {1, 16, 4, 4, 4};
    vector<int64_t> vector_output = {1, 16, 4, 4, 4};

    auto grad_output_desc = TensorDesc(vector_grad_output, ACL_INT32, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto kernel_size_desc = IntArrayDesc(vector_kernel_size);
    auto stride_desc = IntArrayDesc(vector_stride);
    auto padding_desc = IntArrayDesc(vector_padding);
    auto grad_input_desc = TensorDesc(vector_grad_input, ACL_FLOAT, ACL_FORMAT_ND);
    auto output = TensorDesc(vector_output, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnAvgPool3dBackward, // host api第二段接口名称
        INPUT(
            grad_output_desc, grad_input_desc, kernel_size_desc, stride_desc, padding_desc, ceil_mode,
            count_include_pad, divisor_override), // host api输入
        OUTPUT(output));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size); // check op graph
    //  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// gradoutput shape不合法
TEST_F(l2_avgpool3dbackward_test, Ascend910B2_case_6)
{
    int divisor_override = 0;
    bool count_include_pad = false;
    bool ceil_mode = true;

    vector<int64_t> vector_grad_output = {1, 16, 1};
    vector<int64_t> vector_kernel_size = {4, 4, 4};
    vector<int64_t> vector_stride = {1, 1, 1};
    vector<int64_t> vector_padding = {0, 0, 0};
    vector<int64_t> vector_grad_input = {1, 16, 4, 4, 4};
    vector<int64_t> vector_output = {1, 16, 4, 4, 4};

    auto grad_output_desc = TensorDesc(vector_grad_output, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto kernel_size_desc = IntArrayDesc(vector_kernel_size);
    auto stride_desc = IntArrayDesc(vector_stride);
    auto padding_desc = IntArrayDesc(vector_padding);
    auto grad_input_desc = TensorDesc(vector_grad_input, ACL_FLOAT, ACL_FORMAT_ND);
    auto output = TensorDesc(vector_output, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnAvgPool3dBackward, // host api第二段接口名称
        INPUT(
            grad_output_desc, grad_input_desc, kernel_size_desc, stride_desc, padding_desc, ceil_mode,
            count_include_pad, divisor_override), // host api输入
        OUTPUT(output));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size); // check op graph
    //  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// ksize shape不合法
TEST_F(l2_avgpool3dbackward_test, Ascend910B2_case_7)
{
    int divisor_override = 0;
    bool count_include_pad = false;
    bool ceil_mode = true;

    vector<int64_t> vector_grad_output = {1, 16, 1, 1, 1};
    vector<int64_t> vector_kernel_size = {4, 4};
    vector<int64_t> vector_stride = {1, 1, 1};
    vector<int64_t> vector_padding = {0, 0, 0};
    vector<int64_t> vector_grad_input = {1, 16, 4, 4, 4};
    vector<int64_t> vector_output = {1, 16, 4, 4, 4};

    auto grad_output_desc = TensorDesc(vector_grad_output, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto kernel_size_desc = IntArrayDesc(vector_kernel_size);
    auto stride_desc = IntArrayDesc(vector_stride);
    auto padding_desc = IntArrayDesc(vector_padding);
    auto grad_input_desc = TensorDesc(vector_grad_input, ACL_FLOAT, ACL_FORMAT_ND);
    auto output = TensorDesc(vector_output, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnAvgPool3dBackward, // host api第二段接口名称
        INPUT(
            grad_output_desc, grad_input_desc, kernel_size_desc, stride_desc, padding_desc, ceil_mode,
            count_include_pad, divisor_override), // host api输入
        OUTPUT(output));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size); // check op graph
    //  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// ksize 数据不正确
TEST_F(l2_avgpool3dbackward_test, Ascend910B2_case_8)
{
    int divisor_override = 0;
    bool count_include_pad = false;
    bool ceil_mode = true;

    vector<int64_t> vector_grad_output = {1, 16, 1, 1, 1};
    vector<int64_t> vector_kernel_size = {4, 4, -4};
    vector<int64_t> vector_stride = {1, 1, 1};
    vector<int64_t> vector_padding = {0, 0, 0};
    vector<int64_t> vector_grad_input = {1, 16, 4, 4, 4};
    vector<int64_t> vector_output = {1, 16, 4, 4, 4};

    auto grad_output_desc = TensorDesc(vector_grad_output, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto kernel_size_desc = IntArrayDesc(vector_kernel_size);
    auto stride_desc = IntArrayDesc(vector_stride);
    auto padding_desc = IntArrayDesc(vector_padding);
    auto grad_input_desc = TensorDesc(vector_grad_input, ACL_FLOAT, ACL_FORMAT_ND);
    auto output = TensorDesc(vector_output, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnAvgPool3dBackward, // host api第二段接口名称
        INPUT(
            grad_output_desc, grad_input_desc, kernel_size_desc, stride_desc, padding_desc, ceil_mode,
            count_include_pad, divisor_override), // host api输入
        OUTPUT(output));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size); // check op graph
    //  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// output维度与gradoutput维度不一致
TEST_F(l2_avgpool3dbackward_test, Ascend910B2_case_9)
{
    int divisor_override = 0;
    bool count_include_pad = false;
    bool ceil_mode = true;

    vector<int64_t> vector_grad_output = {1, 16, 1, 1, 1};
    vector<int64_t> vector_kernel_size = {4, 4, 4};
    vector<int64_t> vector_stride = {1, 1, 1};
    vector<int64_t> vector_padding = {0, 0, 0};
    vector<int64_t> vector_grad_input = {1, 16, 4, 4, 4};
    vector<int64_t> vector_output = {1, 16, 4, 4};

    auto grad_output_desc = TensorDesc(vector_grad_output, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto kernel_size_desc = IntArrayDesc(vector_kernel_size);
    auto stride_desc = IntArrayDesc(vector_stride);
    auto padding_desc = IntArrayDesc(vector_padding);
    auto grad_input_desc = TensorDesc(vector_grad_input, ACL_FLOAT, ACL_FORMAT_ND);
    auto output = TensorDesc(vector_output, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnAvgPool3dBackward, // host api第二段接口名称
        INPUT(
            grad_output_desc, grad_input_desc, kernel_size_desc, stride_desc, padding_desc, ceil_mode,
            count_include_pad, divisor_override), // host api输入
        OUTPUT(output));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size); // check op graph
    //  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// input shape 与 grad shape维度不一致
TEST_F(l2_avgpool3dbackward_test, Ascend910B2_case_10)
{
    int divisor_override = 0;
    bool count_include_pad = false;
    bool ceil_mode = true;

    vector<int64_t> vector_grad_output = {1, 16, 1, 1, 1};
    vector<int64_t> vector_kernel_size = {4, 4, 4};
    vector<int64_t> vector_stride = {1, 1, 1};
    vector<int64_t> vector_padding = {0, 0, 0};
    vector<int64_t> vector_grad_input = {1, 16, 4, 4};
    vector<int64_t> vector_output = {1, 16, 4, 4, 4};

    auto grad_output_desc = TensorDesc(vector_grad_output, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto kernel_size_desc = IntArrayDesc(vector_kernel_size);
    auto stride_desc = IntArrayDesc(vector_stride);
    auto padding_desc = IntArrayDesc(vector_padding);
    auto grad_input_desc = TensorDesc(vector_grad_input, ACL_FLOAT, ACL_FORMAT_ND);
    auto output = TensorDesc(vector_output, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnAvgPool3dBackward, // host api第二段接口名称
        INPUT(
            grad_output_desc, grad_input_desc, kernel_size_desc, stride_desc, padding_desc, ceil_mode,
            count_include_pad, divisor_override), // host api输入
        OUTPUT(output));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size); // check op graph
    //  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// input shape 与 grad shape维度值不一致
TEST_F(l2_avgpool3dbackward_test, Ascend910B2_case_11)
{
    int divisor_override = 0;
    bool count_include_pad = false;
    bool ceil_mode = true;

    vector<int64_t> vector_grad_output = {1, 16, 1, 1, 1};
    vector<int64_t> vector_kernel_size = {4, 4, 4};
    vector<int64_t> vector_stride = {1, 1, 1};
    vector<int64_t> vector_padding = {0, 0, 0};
    vector<int64_t> vector_grad_input = {1, 16, 4, 4, 5};
    vector<int64_t> vector_output = {1, 16, 4, 4, 4};

    auto grad_output_desc = TensorDesc(vector_grad_output, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto kernel_size_desc = IntArrayDesc(vector_kernel_size);
    auto stride_desc = IntArrayDesc(vector_stride);
    auto padding_desc = IntArrayDesc(vector_padding);
    auto grad_input_desc = TensorDesc(vector_grad_input, ACL_FLOAT, ACL_FORMAT_ND);
    auto output = TensorDesc(vector_output, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnAvgPool3dBackward, // host api第二段接口名称
        INPUT(
            grad_output_desc, grad_input_desc, kernel_size_desc, stride_desc, padding_desc, ceil_mode,
            count_include_pad, divisor_override), // host api输入
        OUTPUT(output));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size); // check op graph
    //  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// kernel为-1
TEST_F(l2_avgpool3dbackward_test, Ascend910B2_case_12)
{
    int divisor_override = 0;
    bool count_include_pad = false;
    bool ceil_mode = true;

    vector<int64_t> vector_grad_output = {1, 16, 1, 1, 1};
    vector<int64_t> vector_kernel_size = {-1, -1, -1};
    vector<int64_t> vector_stride = {0, 0, 0};
    vector<int64_t> vector_padding = {0, 0, 0};
    vector<int64_t> vector_grad_input = {1, 16, 4, 4, 4};
    vector<int64_t> vector_output = {1, 16, 4, 4, 4};

    auto grad_output_desc = TensorDesc(vector_grad_output, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto kernel_size_desc = IntArrayDesc(vector_kernel_size);
    auto stride_desc = IntArrayDesc(vector_stride);
    auto padding_desc = IntArrayDesc(vector_padding);
    auto grad_input_desc = TensorDesc(vector_grad_input, ACL_FLOAT, ACL_FORMAT_ND);
    auto output = TensorDesc(vector_output, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnAvgPool3dBackward, // host api第二段接口名称
        INPUT(
            grad_output_desc, grad_input_desc, kernel_size_desc, stride_desc, padding_desc, ceil_mode,
            count_include_pad, divisor_override), // host api输入
        OUTPUT(output));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size); // check op graph
    //  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// stride为-1
TEST_F(l2_avgpool3dbackward_test, Ascend910B2_case_13)
{
    int divisor_override = 0;
    bool count_include_pad = false;
    bool ceil_mode = true;

    vector<int64_t> vector_grad_output = {1, 16, 1, 1, 1};
    vector<int64_t> vector_kernel_size = {4, 4, 4};
    vector<int64_t> vector_stride = {-1, -1, -1};
    vector<int64_t> vector_padding = {0, 0, 0};
    vector<int64_t> vector_grad_input = {1, 16, 4, 4, 4};
    vector<int64_t> vector_output = {1, 16, 4, 4, 4};

    auto grad_output_desc = TensorDesc(vector_grad_output, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto kernel_size_desc = IntArrayDesc(vector_kernel_size);
    auto stride_desc = IntArrayDesc(vector_stride);
    auto padding_desc = IntArrayDesc(vector_padding);
    auto grad_input_desc = TensorDesc(vector_grad_input, ACL_FLOAT, ACL_FORMAT_ND);
    auto output = TensorDesc(vector_output, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnAvgPool3dBackward, // host api第二段接口名称
        INPUT(
            grad_output_desc, grad_input_desc, kernel_size_desc, stride_desc, padding_desc, ceil_mode,
            count_include_pad, divisor_override), // host api输入
        OUTPUT(output));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size); // check op graph
    //  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// padding为-1
TEST_F(l2_avgpool3dbackward_test, Ascend910B2_case_14)
{
    int divisor_override = 0;
    bool count_include_pad = false;
    bool ceil_mode = true;

    vector<int64_t> vector_grad_output = {1, 16, 1, 1, 1};
    vector<int64_t> vector_kernel_size = {4, 4, 4};
    vector<int64_t> vector_stride = {1, 1, 1};
    vector<int64_t> vector_padding = {-1, -1, -1};
    vector<int64_t> vector_grad_input = {1, 16, 4, 4, 4};
    vector<int64_t> vector_output = {1, 16, 4, 4, 4};

    auto grad_output_desc = TensorDesc(vector_grad_output, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto kernel_size_desc = IntArrayDesc(vector_kernel_size);
    auto stride_desc = IntArrayDesc(vector_stride);
    auto padding_desc = IntArrayDesc(vector_padding);
    auto grad_input_desc = TensorDesc(vector_grad_input, ACL_FLOAT, ACL_FORMAT_ND);
    auto output = TensorDesc(vector_output, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnAvgPool3dBackward, // host api第二段接口名称
        INPUT(
            grad_output_desc, grad_input_desc, kernel_size_desc, stride_desc, padding_desc, ceil_mode,
            count_include_pad, divisor_override), // host api输入
        OUTPUT(output));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size); // check op graph
    //  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

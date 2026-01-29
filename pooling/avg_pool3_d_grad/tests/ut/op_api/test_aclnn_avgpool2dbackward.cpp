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
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

#include "../../../op_api/aclnn_avgpool2d_backward.h"

using namespace op;
using namespace std;

class l2_avgpool2dbackward_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "avgpool2dbackward_test Setup" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "avgpool2dbackward_test TearDown" << std::endl;
    }
};

const int8_t cubeMathType = 1;
/*
// 输入gradOutput的shape取值不合法拦截
TEST_F(l2_avgpool2dbackward_test, ascend910B2_test_avgpool2dbackward_gradOutput_illegal_shape) {
 int64_t divisor_override = 0;
 bool count_include_pad = false;
 bool ceil_mode = true;

 vector<int64_t> vector_grad_output = {-1, 1, 8, 8};
 vector<int64_t> vector_self = {1, 1, 16, 16};
 vector<int64_t> vector_kernel_size = {4, 4};
 vector<int64_t> vector_stride = {2, 2};
 vector<int64_t> vector_padding = {2, 2};
 vector<int64_t> vector_grad_input = {1, 1, 16, 16};

 auto grad_output_desc = TensorDesc(vector_grad_output, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(-1, 1);
 auto self_desc = TensorDesc(vector_self, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(-1, 1);
 auto kernel_size_desc = IntArrayDesc(vector_kernel_size);
 auto stride_desc = IntArrayDesc(vector_stride);
 auto padding_desc = IntArrayDesc(vector_padding);
 auto grad_input_desc = TensorDesc(vector_grad_input, ACL_FLOAT, ACL_FORMAT_NCHW);

 auto ut = OP_API_UT(aclnnAvgPool2dBackward, // host api第二段接口名称
                     INPUT(grad_output_desc, self_desc, kernel_size_desc, stride_desc, padding_desc, ceil_mode,
                           count_include_pad, divisor_override, cubeMathType), // host api输入
                     OUTPUT(grad_input_desc));
 // SAMPLE: only test GetWorkspaceSize
 uint64_t workspace_size = 0;
 aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size); // check op graph
 EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 输入self的shape取值不合法拦截
TEST_F(l2_avgpool2dbackward_test, ascend910B2_test_avgpool2dbackward_self_illegal_shape) {
 int64_t divisor_override = 0;
 bool count_include_pad = false;
 bool ceil_mode = true;

 vector<int64_t> vector_grad_output = {1, 1, 8, 8};
 vector<int64_t> vector_self = {-1, 1, 16, 16};
 vector<int64_t> vector_kernel_size = {4, 4};
 vector<int64_t> vector_stride = {2, 2};
 vector<int64_t> vector_padding = {2, 2};
 vector<int64_t> vector_grad_input = {1, 1, 16, 16};

 auto grad_output_desc = TensorDesc(vector_grad_output, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(-1, 1);
 auto self_desc = TensorDesc(vector_self, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(-1, 1);
 auto kernel_size_desc = IntArrayDesc(vector_kernel_size);
 auto stride_desc = IntArrayDesc(vector_stride);
 auto padding_desc = IntArrayDesc(vector_padding);
 auto grad_input_desc = TensorDesc(vector_grad_input, ACL_FLOAT, ACL_FORMAT_NCHW);

 auto ut = OP_API_UT(aclnnAvgPool2dBackward, // host api第二段接口名称
                     INPUT(grad_output_desc, self_desc, kernel_size_desc, stride_desc, padding_desc, ceil_mode,
                           count_include_pad, divisor_override, cubeMathType), // host api输入
                     OUTPUT(grad_input_desc));
 // SAMPLE: only test GetWorkspaceSize
 uint64_t workspace_size = 0;
 aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size); // check op graph
 EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}
*/

// 输入kernelSize的shape取值不合法拦截
TEST_F(l2_avgpool2dbackward_test, ascend910B2_test_avgpool2dbackward_kernelSize_illegal_shape)
{
    int64_t divisor_override = 0;
    bool count_include_pad = false;
    bool ceil_mode = true;

    vector<int64_t> vector_grad_output = {1, 1, 8, 8};
    vector<int64_t> vector_self = {1, 1, 16, 16};
    vector<int64_t> vector_kernel_size = {-4, 4};
    vector<int64_t> vector_stride = {2, 2};
    vector<int64_t> vector_padding = {2, 2};
    vector<int64_t> vector_grad_input = {1, 1, 16, 16};

    auto grad_output_desc = TensorDesc(vector_grad_output, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(-1, 1);
    auto self_desc = TensorDesc(vector_self, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(-1, 1);
    auto kernel_size_desc = IntArrayDesc(vector_kernel_size);
    auto stride_desc = IntArrayDesc(vector_stride);
    auto padding_desc = IntArrayDesc(vector_padding);
    auto grad_input_desc = TensorDesc(vector_grad_input, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(
        aclnnAvgPool2dBackward, // host api第二段接口名称
        INPUT(
            grad_output_desc, self_desc, kernel_size_desc, stride_desc, padding_desc, ceil_mode, count_include_pad,
            divisor_override, cubeMathType), // host api输入
        OUTPUT(grad_input_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size); // check op graph
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 输入stride的shape取值不合法拦截
TEST_F(l2_avgpool2dbackward_test, ascend910B2_test_avgpool2dbackward_stride_illegal_shape)
{
    int64_t divisor_override = 0;
    bool count_include_pad = false;
    bool ceil_mode = true;

    vector<int64_t> vector_grad_output = {1, 1, 8, 8};
    vector<int64_t> vector_self = {1, 1, 16, 16};
    vector<int64_t> vector_kernel_size = {4, 4};
    vector<int64_t> vector_stride = {-2, 2};
    vector<int64_t> vector_padding = {2, 2};
    vector<int64_t> vector_grad_input = {1, 1, 16, 16};

    auto grad_output_desc = TensorDesc(vector_grad_output, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(-1, 1);
    auto self_desc = TensorDesc(vector_self, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(-1, 1);
    auto kernel_size_desc = IntArrayDesc(vector_kernel_size);
    auto stride_desc = IntArrayDesc(vector_stride);
    auto padding_desc = IntArrayDesc(vector_padding);
    auto grad_input_desc = TensorDesc(vector_grad_input, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(
        aclnnAvgPool2dBackward, // host api第二段接口名称
        INPUT(
            grad_output_desc, self_desc, kernel_size_desc, stride_desc, padding_desc, ceil_mode, count_include_pad,
            divisor_override, cubeMathType), // host api输入
        OUTPUT(grad_input_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size); // check op graph
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}
/*
// 输出grad_input的shape取值不合法拦截
TEST_F(l2_avgpool2dbackward_test, ascend910B2_test_avgpool2dbackward_gradInput_illegal_shape) {
 int64_t divisor_override = 0;
 bool count_include_pad = false;
 bool ceil_mode = true;

 vector<int64_t> vector_grad_output = {1, 1, 8, 8};
 vector<int64_t> vector_self = {1, 1, 16, 16};
 vector<int64_t> vector_kernel_size = {4, 4};
 vector<int64_t> vector_stride = {2, 2};
 vector<int64_t> vector_padding = {2, 2};
 vector<int64_t> vector_grad_input = {-1, 1, 16, 16};

 auto grad_output_desc = TensorDesc(vector_grad_output, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(-1, 1);
 auto self_desc = TensorDesc(vector_self, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(-1, 1);
 auto kernel_size_desc = IntArrayDesc(vector_kernel_size);
 auto stride_desc = IntArrayDesc(vector_stride);
 auto padding_desc = IntArrayDesc(vector_padding);
 auto grad_input_desc = TensorDesc(vector_grad_input, ACL_FLOAT, ACL_FORMAT_NCHW);

 auto ut = OP_API_UT(aclnnAvgPool2dBackward, // host api第二段接口名称
                     INPUT(grad_output_desc, self_desc, kernel_size_desc, stride_desc, padding_desc, ceil_mode,
                           count_include_pad, divisor_override, cubeMathType), // host api输入
                     OUTPUT(grad_input_desc));
 // SAMPLE: only test GetWorkspaceSize
 uint64_t workspace_size = 0;
 aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size); // check op graph
 EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}
*/


// 输出grad_input的shape取值不合法拦截
TEST_F(l2_avgpool2dbackward_test, ascend910B2_test_avgpool2dbackward_kernel_size_empty)
{
    vector<int64_t> vector_grad_output = {8, 192, 1, 1};
    vector<int64_t> vector_self = {8, 192, 64, 64};
    vector<int64_t> vector_kernel_size = {};
    vector<int64_t> vector_stride = {0, 0};
    vector<int64_t> vector_padding = {0, 0};
    vector<int64_t> vector_grad_input = {8, 192, 64, 64};
    bool count_include_pad = false;
    bool ceil_mode = false;
    int64_t divisor_override = 0;

    auto grad_output_desc = TensorDesc(vector_grad_output, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(-1.0, 1.0);
    auto self_desc = TensorDesc(vector_self, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(-1.0, 1.0);

    auto kernel_size_desc = IntArrayDesc(vector_kernel_size);
    auto stride_desc = IntArrayDesc(vector_stride);
    auto padding_desc = IntArrayDesc(vector_padding);
    auto grad_input_desc = TensorDesc(vector_grad_input, ACL_FLOAT16, ACL_FORMAT_NCHW).Precision(0.001, 0.001);
    auto ut = OP_API_UT(
        aclnnAvgPool2dBackward, // host api第二段接口名称
        INPUT(
            grad_output_desc, self_desc, kernel_size_desc, stride_desc, padding_desc, ceil_mode, count_include_pad,
            divisor_override, cubeMathType), // host api输入
        OUTPUT(grad_input_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// BF16，输入是四维，不支持的芯片插Mul, cubeMathType = 0 支持的走3d
TEST_F(l2_avgpool2dbackward_test, ascend310P_test_avgpool2dbackwardbackward_global_avg_pool_bf16)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND310P);
    vector<int64_t> vector_grad_output = {5, 4, 6, 6};
    vector<int64_t> vector_self = {5, 4, 13, 13};
    vector<int64_t> vector_kernel_size = {3, 3};
    vector<int64_t> vector_stride = {2, 2};
    vector<int64_t> vector_padding = {1, 1};
    vector<int64_t> vector_grad_input = {5, 4, 13, 13};
    bool count_include_pad = false;
    bool ceil_mode = false;
    int64_t divisor_override = 300;
    int8_t cubeMathTypeZero = 0;

    auto grad_output_desc = TensorDesc(vector_grad_output, ACL_BF16, ACL_FORMAT_NCHW).ValueRange(-1.0, 1.0);
    auto self_desc = TensorDesc(vector_self, ACL_BF16, ACL_FORMAT_NCHW).ValueRange(-1.0, 1.0);

    auto kernel_size_desc = IntArrayDesc(vector_kernel_size);
    auto stride_desc = IntArrayDesc(vector_stride);
    auto padding_desc = IntArrayDesc(vector_padding);
    auto grad_input_desc = TensorDesc(vector_grad_input, ACL_BF16, ACL_FORMAT_NCHW).Precision(0.001, 0.001);
    auto ut = OP_API_UT(
        aclnnAvgPool2dBackward, // host api第二段接口名称
        INPUT(
            grad_output_desc, self_desc, kernel_size_desc, stride_desc, padding_desc, ceil_mode, count_include_pad,
            divisor_override, cubeMathTypeZero), // host api输入
        OUTPUT(grad_input_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}
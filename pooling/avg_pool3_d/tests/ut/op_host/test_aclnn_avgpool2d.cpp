/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "../../../op_host/op_api/aclnn_avgpool2d.h"

#include <vector>
#include <array>
#include "gtest/gtest.h"

#include "opdev/op_log.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api/op_api_def.h"

using namespace op;
using namespace std;

class l2_avgpool2d_test : public testing::Test {
protected:
 static void SetUpTestCase() { std::cout << "avgpool2d_test Setup" << std::endl; }
 static void TearDownTestCase() { std::cout << "avgpool2d_test TearDown" << std::endl; }
};

const int8_t cubeMathType = ALLOW_FP32_DOWN_PRECISION;

// cubeMathType取值非法拦截
TEST_F(l2_avgpool2d_test, test_avgpool2d_illegal_cubeMathType) {
    int64_t divisor_override = 0;
    bool count_include_pad = false;
    bool ceil_mode = true;
    const int8_t cubeMathType_0 = 4;

    vector<int64_t> self_dims = {1, 1, 16, 16};
    vector<int64_t> kernel_dims = {4, 4};
    vector<int64_t> stride_dims = {2, 2};
    vector<int64_t> padding_dims = {-1, -1}; // 这里有-1
    vector<int64_t> out_dims = {1, 1, 8, 8};

    auto self_desc = TensorDesc(self_dims, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(-1, 1);
    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto out_desc = TensorDesc(out_dims, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnAvgPool2d,  // host api第二段接口名称
                        INPUT(self_desc, kernel_desc, stride_desc, padding_desc,
                              ceil_mode, count_include_pad, divisor_override, cubeMathType_0),  // host api输入
                        OUTPUT(out_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size); // check op graph
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 参数kernel的shape取值不合法拦截
TEST_F(l2_avgpool2d_test, test_avgpool2d_kernel_illegal_shape) {
    int64_t divisor_override = 0;
    bool count_include_pad = true;
    bool ceil_mode = false;

    vector<int64_t> self_dims = {1, 1, 16, 16};
    vector<int64_t> kernel_dims = {-4, 4}; // 这里有个-4
    vector<int64_t> stride_dims = {2, 2};
    vector<int64_t> padding_dims = {1, 1};
    vector<int64_t> out_dims = {1, 1, 8, 8};

    auto self_desc = TensorDesc(self_dims, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto out_desc = TensorDesc(out_dims, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnAvgPool2d,  // host api第二段接口名称
                        INPUT(self_desc, kernel_desc, stride_desc, padding_desc,
                              ceil_mode, count_include_pad, divisor_override , cubeMathType),  // host api输入
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size); // check op graph
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 参数kernel的shape取值不合法拦截
TEST_F(l2_avgpool2d_test, test_avgpool2d_kernel_illegal_shape_1) {
    int64_t divisor_override = 0;
    bool count_include_pad = true;
    bool ceil_mode = false;

    vector<int64_t> self_dims = {1, 1, 16, 16};
    vector<int64_t> kernel_dims = {}; // 这里kernel为空
    vector<int64_t> stride_dims = {2, 2};
    vector<int64_t> padding_dims = {1, 1};
    vector<int64_t> out_dims = {1, 1, 8, 8};

    auto self_desc = TensorDesc(self_dims, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto out_desc = TensorDesc(out_dims, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnAvgPool2d,  // host api第二段接口名称
                        INPUT(self_desc, kernel_desc, stride_desc, padding_desc,
                              ceil_mode, count_include_pad, divisor_override , cubeMathType),  // host api输入
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size); // check op graph
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 参数kernel的shape取值不合法拦截
TEST_F(l2_avgpool2d_test, test_avgpool2d_kernel_illegal_shape_2) {
    int64_t divisor_override = 0;
    bool count_include_pad = true;
    bool ceil_mode = true;

    vector<int64_t> self_dims = {1, 1, 16, 16};
    vector<int64_t> kernel_dims = {1, 1};
    vector<int64_t> stride_dims = {2, 2};
    vector<int64_t> padding_dims = {1, 1};
    vector<int64_t> out_dims = {1, 1, 10, 10}; // outshape 不正确，正确的是{1,1,9,9}

    auto self_desc = TensorDesc(self_dims, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto out_desc = TensorDesc(out_dims, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnAvgPool2d,  // host api第二段接口名称
                        INPUT(self_desc, kernel_desc, stride_desc, padding_desc,
                              ceil_mode, count_include_pad, divisor_override , cubeMathType),  // host api输入
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size); // check op graph
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}
// 参数stride的shape取值不合法拦截
TEST_F(l2_avgpool2d_test, test_avgpool2d_stride_illegal_shape) {
    int64_t divisor_override = 0;
    bool count_include_pad = true;
    bool ceil_mode = false;

    vector<int64_t> self_dims = {1, 1, 16, 16};
    vector<int64_t> kernel_dims = {4, 4};
    vector<int64_t> stride_dims = {2, -2}; // 这里有个-2
    vector<int64_t> padding_dims = {1, 1};
    vector<int64_t> out_dims = {1, 1, 8, 8};

    auto self_desc = TensorDesc(self_dims, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto out_desc = TensorDesc(out_dims, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnAvgPool2d,  // host api第二段接口名称
                        INPUT(self_desc, kernel_desc, stride_desc, padding_desc,
                              ceil_mode, count_include_pad, divisor_override, cubeMathType),  // host api输入
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size); // check op graph
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// FP16，走Vector分支
TEST_F(l2_avgpool2d_test, test_avgpool2d_globalpool_fp16) {
    int64_t divisor_override = 0;
    bool count_include_pad = true;
    bool ceil_mode = false;

    vector<int64_t> self_dims = {1, 16, 12, 12}; // NCHW
    vector<int64_t> kernel_dims = {12, 12};
    vector<int64_t> stride_dims = {1, 1};
    vector<int64_t> padding_dims = {0, 0};
    vector<int64_t> out_dims = {1, 16, 1, 1};

    auto self_desc = TensorDesc(self_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(-1.0, 1.0);
    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto out_desc = TensorDesc(out_dims, ACL_FLOAT16, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnAvgPool2d, // host api第二段接口名称
                        INPUT(self_desc, kernel_desc, stride_desc, padding_desc,
                              ceil_mode, count_include_pad, divisor_override, cubeMathType), // host api输入
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size); // check op graph
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// FP16，走Cube分支，countIncludePad=True，不插入AvgPoolUpdate算子
TEST_F(l2_avgpool2d_test, test_avgpool2d_avgpool_include_fp16) {
    int64_t divisor_override = 0;
    bool count_include_pad = true;
    bool ceil_mode = false;

    vector<int64_t> self_dims = {1, 16, 12, 20}; // NCHW
    vector<int64_t> kernel_dims = {12, 12};
    vector<int64_t> stride_dims = {1, 1};
    vector<int64_t> padding_dims = {0, 0};
    vector<int64_t> out_dims = {1, 16, 1, 9};

    auto self_desc = TensorDesc(self_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(-1.0, 1.0);
    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto out_desc = TensorDesc(out_dims, ACL_FLOAT16, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnAvgPool2d, // host api第二段接口名称
                        INPUT(self_desc, kernel_desc, stride_desc, padding_desc,
                              ceil_mode, count_include_pad, divisor_override, cubeMathType), // host api输入
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size); // check op graph
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// FP16，走3d分支，countIncludePad=True
TEST_F(l2_avgpool2d_test, test_avgpool2d_avgpool_include_fp16_to3d) {
    int64_t divisor_override = 0;
    bool count_include_pad = true;
    bool ceil_mode = false;

    vector<int64_t> self_dims = {1, 16, 12, 40}; // NCHW
    vector<int64_t> kernel_dims = {12, 12};
    vector<int64_t> stride_dims = {1, 1};
    vector<int64_t> padding_dims = {0, 0};
    vector<int64_t> out_dims = {1, 16, 1, 29};

    auto self_desc = TensorDesc(self_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(-1.0, 1.0);
    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto out_desc = TensorDesc(out_dims, ACL_FLOAT16, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnAvgPool2d, // host api第二段接口名称
                        INPUT(self_desc, kernel_desc, stride_desc, padding_desc,
                              ceil_mode, count_include_pad, divisor_override, cubeMathType), // host api输入
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size); // check op graph
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// FP16，走3d分支,countIncludePad=False
TEST_F(l2_avgpool2d_test, test_avgpool2d_avgpool_exclude_fp16_to3d) {
    int64_t divisor_override = 0;
    bool count_include_pad = false;
    bool ceil_mode = false;

    vector<int64_t> self_dims = {1, 16, 12, 40}; // NCHW
    vector<int64_t> kernel_dims = {12, 12};
    vector<int64_t> stride_dims = {1, 1};
    vector<int64_t> padding_dims = {0, 0};
    vector<int64_t> out_dims = {1, 16, 1, 29};

    auto self_desc = TensorDesc(self_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(-1.0, 1.0);
    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto out_desc = TensorDesc(out_dims, ACL_FLOAT16, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnAvgPool2d, // host api第二段接口名称
                        INPUT(self_desc, kernel_desc, stride_desc, padding_desc,
                              ceil_mode, count_include_pad, divisor_override, cubeMathType), // host api输入
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size); // check op graph
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// FP16，走Cube分支，countIncludePad=False，插入AvgPoolUpdate算子
TEST_F(l2_avgpool2d_test, test_avgpool2d_avgpool_exclude_fp16) {
    int64_t divisor_override = 0;
    bool count_include_pad = false;
    bool ceil_mode = false;

    vector<int64_t> self_dims = {1, 16, 12, 20}; // NCHW
    vector<int64_t> kernel_dims = {12, 12};
    vector<int64_t> stride_dims = {1, 1};
    vector<int64_t> padding_dims = {0, 0};
    vector<int64_t> out_dims = {1, 16, 1, 9};

    auto self_desc = TensorDesc(self_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(-1.0, 1.0);
    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto out_desc = TensorDesc(out_dims, ACL_FLOAT16, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnAvgPool2d, // host api第二段接口名称
                        INPUT(self_desc, kernel_desc, stride_desc, padding_desc,
                              ceil_mode, count_include_pad, divisor_override, cubeMathType), // host api输入
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size); // check op graph
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// FP16，走AICPU分支
TEST_F(l2_avgpool2d_test, test_avgpool2d_no_global_no_average_fp16) {
    int64_t divisor_override = 0;
    bool count_include_pad = true;
    bool ceil_mode = false;

    vector<int64_t> self_dims = {1, 16, 100, 4}; // NCHW
    vector<int64_t> kernel_dims = {34, 34};
    vector<int64_t> stride_dims = {1, 1};
    vector<int64_t> padding_dims = {15, 15};
    vector<int64_t> out_dims = {1, 16, 97, 1};

    auto self_desc = TensorDesc(self_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(-1.0, 1.0);
    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto out_desc = TensorDesc(out_dims, ACL_FLOAT16, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnAvgPool2d,  // host api第二段接口名称
                        INPUT(self_desc, kernel_desc, stride_desc, padding_desc,
                              ceil_mode, count_include_pad, divisor_override, cubeMathType),  // host api输入
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size); // check op graph
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// FP32，走3d分支，countIncludePad=True
TEST_F(l2_avgpool2d_test, test_avgpool2d_avgpool_include_fp32_to3d) {
    int64_t divisor_override = 0;
    bool count_include_pad = true;
    bool ceil_mode = false;

    vector<int64_t> self_dims = {1, 16, 12, 40}; // NCHW
    vector<int64_t> kernel_dims = {12, 12};
    vector<int64_t> stride_dims = {1, 1};
    vector<int64_t> padding_dims = {0, 0};
    vector<int64_t> out_dims = {1, 16, 1, 29};

    auto self_desc = TensorDesc(self_dims, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(-1.0, 1.0);
    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto out_desc = TensorDesc(out_dims, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnAvgPool2d,  // host api第二段接口名称
                        INPUT(self_desc, kernel_desc, stride_desc, padding_desc,
                              ceil_mode, count_include_pad, divisor_override, cubeMathType), // host api输入
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size); // check op graph
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// FP32，走3d分支，countIncludePad=False
TEST_F(l2_avgpool2d_test, test_avgpool2d_avgpool_exclude_fp32_to3d) {
    int64_t divisor_override = 0;
    bool count_include_pad = false;
    bool ceil_mode = false;

    vector<int64_t> self_dims = {1, 16, 12, 40}; // NCHW
    vector<int64_t> kernel_dims = {12, 12};
    vector<int64_t> stride_dims = {1, 1};
    vector<int64_t> padding_dims = {0, 0};
    vector<int64_t> out_dims = {1, 16, 1, 29};

    auto self_desc = TensorDesc(self_dims, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(-1.0, 1.0);
    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto out_desc = TensorDesc(out_dims, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnAvgPool2d,  // host api第二段接口名称
                        INPUT(self_desc, kernel_desc, stride_desc, padding_desc,
                              ceil_mode, count_include_pad, divisor_override, cubeMathType),  // host api输入
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size); // check op graph
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// FP32，走AICPU分支
TEST_F(l2_avgpool2d_test, test_avgpool2d_no_global_no_average_fp32) {
    int64_t divisor_override = 0;
    bool count_include_pad = true;
    bool ceil_mode = true;

    vector<int64_t> self_dims = {1, 16, 100, 4}; // NCHW
    vector<int64_t> kernel_dims = {34, 34};
    vector<int64_t> stride_dims = {1, 1};
    vector<int64_t> padding_dims = {15, 15};
    vector<int64_t> out_dims = {1, 16, 97, 1};

    auto self_desc = TensorDesc(self_dims, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(-1.0, 1.0);
    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto out_desc = TensorDesc(out_dims, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnAvgPool2d,  // host api第二段接口名称
                        INPUT(self_desc, kernel_desc, stride_desc, padding_desc,
                              ceil_mode, count_include_pad, divisor_override, cubeMathType),  // host api输入
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size); // check op graph
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// self空指针拦截
TEST_F(l2_avgpool2d_test, test_avgpool2d_nullptr_self) {
    int64_t divisor_override = 0;
    bool count_include_pad = true;
    bool ceil_mode = false;

    vector<int64_t> self_dims = {1, 1, 16, 16};
    vector<int64_t> kernel_dims = {4, 4};
    vector<int64_t> stride_dims = {2, 2};
    vector<int64_t> padding_dims = {1, 1};
    vector<int64_t> out_dims = {1, 1, 8, 8};

    auto self_desc = TensorDesc(self_dims, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto out_desc = TensorDesc(out_dims, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnAvgPool2d,  // host api第二段接口名称
                        INPUT((aclTensor*)nullptr, kernel_desc, stride_desc, padding_desc,
                              ceil_mode, count_include_pad, divisor_override, cubeMathType),  // host api输入
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size); // check op graph
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// kernel空指针拦截
TEST_F(l2_avgpool2d_test, test_avgpool2d_nullptr_kernel) {
    int64_t divisor_override = 0;
    bool count_include_pad = true;
    bool ceil_mode = false;

    vector<int64_t> self_dims = {1, 1, 16, 16};
    vector<int64_t> kernel_dims = {4, 4};
    vector<int64_t> stride_dims = {2, 2};
    vector<int64_t> padding_dims = {1, 1};
    vector<int64_t> out_dims = {1, 1, 8, 8};

    auto self_desc = TensorDesc(self_dims, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto out_desc = TensorDesc(out_dims, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnAvgPool2d,  // host api第二段接口名称
                        INPUT(self_desc, (aclIntArray*)nullptr, stride_desc, padding_desc,
                              ceil_mode, count_include_pad, divisor_override, cubeMathType),  // host api输入
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size); // check op graph
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// stride空指针拦截
TEST_F(l2_avgpool2d_test, test_avgpool2d_nullptr_stride) {
    int64_t divisor_override = 0;
    bool count_include_pad = true;
    bool ceil_mode = false;

    vector<int64_t> self_dims = {1, 1, 16, 16};
    vector<int64_t> kernel_dims = {4, 4};
    vector<int64_t> stride_dims = {2, 2};
    vector<int64_t> padding_dims = {1, 1};
    vector<int64_t> out_dims = {1, 1, 8, 8};

    auto self_desc = TensorDesc(self_dims, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto out_desc = TensorDesc(out_dims, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnAvgPool2d,  // host api第二段接口名称
                        INPUT(self_desc, kernel_desc, (aclIntArray*)nullptr, padding_desc,
                              ceil_mode, count_include_pad, divisor_override, cubeMathType),  // host api输入
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size); // check op graph
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// padding空指针拦截
TEST_F(l2_avgpool2d_test, test_avgpool2d_nullptr_padding) {
    int64_t divisor_override = 0;
    bool count_include_pad = true;
    bool ceil_mode = false;

    vector<int64_t> self_dims = {1, 1, 16, 16};
    vector<int64_t> kernel_dims = {4, 4};
    vector<int64_t> stride_dims = {2, 2};
    vector<int64_t> padding_dims = {1, 1};
    vector<int64_t> out_dims = {1, 1, 8, 8};

    auto self_desc = TensorDesc(self_dims, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto out_desc = TensorDesc(out_dims, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnAvgPool2d,  // host api第二段接口名称
                        INPUT(self_desc, kernel_desc, stride_desc, (aclIntArray*)nullptr,
                              ceil_mode, count_include_pad, divisor_override, cubeMathType),  // host api输入
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size); // check op graph
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// out空指针拦截
TEST_F(l2_avgpool2d_test, test_avgpool2d_nullptr_out) {
    int64_t divisor_override = 0;
    bool count_include_pad = true;
    bool ceil_mode = false;

    vector<int64_t> self_dims = {1, 1, 16, 16};
    vector<int64_t> kernel_dims = {4, 4};
    vector<int64_t> stride_dims = {2, 2};
    vector<int64_t> padding_dims = {1, 1};
    vector<int64_t> out_dims = {1, 1, 8, 8};

    auto self_desc = TensorDesc(self_dims, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto out_desc = TensorDesc(out_dims, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnAvgPool2d,  // host api第二段接口名称
                        INPUT(self_desc, kernel_desc, stride_desc, padding_desc,
                              ceil_mode, count_include_pad, divisor_override, cubeMathType),  // host api输入
                        OUTPUT((aclTensor*)nullptr));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size); // check op graph
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// 输出out是空Tensor
TEST_F(l2_avgpool2d_test, test_avgpool2d_out_empty_tensor) {
    int64_t divisor_override = 0;
    bool count_include_pad = true;
    bool ceil_mode = false;

    vector<int64_t> self_dims = {0, 1, 16, 16};
    vector<int64_t> kernel_dims = {4, 4};
    vector<int64_t> stride_dims = {2, 2};
    vector<int64_t> padding_dims = {1, 1};
    vector<int64_t> out_dims = {0, 1, 8, 8}; // 这里有一维是0，是空tensor

    auto self_desc = TensorDesc(self_dims, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto out_desc = TensorDesc(out_dims, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnAvgPool2d,  // host api第二段接口名称
                        INPUT(self_desc, kernel_desc, stride_desc, padding_desc,
                              ceil_mode, count_include_pad, divisor_override, cubeMathType),  // host api输入
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size); // check op graph
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// 输入输出Format不是NCHW拦截
TEST_F(l2_avgpool2d_test, test_avgpool2d_format_not_nchw) {
    int64_t divisor_override = 0;
    bool count_include_pad = true;
    bool ceil_mode = false;

    vector<int64_t> self_dims = {1, 1, 16, 16};
    vector<int64_t> kernel_dims = {4, 4};
    vector<int64_t> stride_dims = {2, 2};
    vector<int64_t> padding_dims = {1, 1};
    vector<int64_t> out_dims = {1, 1, 8, 8};

    auto self_desc = TensorDesc(self_dims, ACL_FLOAT, ACL_FORMAT_NHWC);
    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto out_desc = TensorDesc(out_dims, ACL_FLOAT, ACL_FORMAT_NHWC);

    auto ut = OP_API_UT(aclnnAvgPool2d,  // host api第二段接口名称
                        INPUT(self_desc, kernel_desc, stride_desc, padding_desc,
                              ceil_mode, count_include_pad, divisor_override, cubeMathType),  // host api输入
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size); // check op graph
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 异常Dtype拦截
TEST_F(l2_avgpool2d_test, test_avgpool2d_dtype_abnormal) {
    int64_t divisor_override = 0;
    bool count_include_pad = true;
    bool ceil_mode = false;

    vector<int64_t> self_dims = {1, 1, 16, 16};
    vector<int64_t> kernel_dims = {4, 4};
    vector<int64_t> stride_dims = {2, 2};
    vector<int64_t> padding_dims = {1, 1};
    vector<int64_t> out_dims = {1, 1, 8, 8};

    auto self_desc = TensorDesc(self_dims, ACL_DT_UNDEFINED, ACL_FORMAT_NCHW);
    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto out_desc = TensorDesc(out_dims, ACL_DT_UNDEFINED, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnAvgPool2d,  // host api第二段接口名称
                        INPUT(self_desc, kernel_desc, stride_desc, padding_desc,
                              ceil_mode, count_include_pad, divisor_override, cubeMathType),  // host api输入
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size); // check op graph
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 构造一个不支持的shape，看能否拦截
TEST_F(l2_avgpool2d_test, test_avgpool2d_format_abnormal) {
    int64_t divisor_override = 0;
    bool count_include_pad = true;
    bool ceil_mode = false;

    vector<int64_t> self_dims = {1, 16, 16}; // 不满足NCHW
    vector<int64_t> kernel_dims = {4, 4};
    vector<int64_t> stride_dims = {2, 2};
    vector<int64_t> padding_dims = {1, 1};
    vector<int64_t> out_dims = {1, 1, 8, 8};

    auto self_desc = TensorDesc(self_dims, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto out_desc = TensorDesc(out_dims, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnAvgPool2d,  // host api第二段接口名称
                        INPUT(self_desc, kernel_desc, stride_desc, padding_desc,
                              ceil_mode, count_include_pad, divisor_override, cubeMathType),  // host api输入
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size); // check op graph
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_avgpool2d_test, test_avgpool2d_input_output_different_dtype) {
    int64_t divisor_override = 0;
    bool count_include_pad = true;
    bool ceil_mode = false;

    vector<int64_t> self_dims = {1, 1, 16, 16};
    vector<int64_t> kernel_dims = {4, 4};
    vector<int64_t> stride_dims = {2, 2};
    vector<int64_t> padding_dims = {1, 1};
    vector<int64_t> out_dims = {1, 1, 8, 8};

    auto self_desc = TensorDesc(self_dims, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto out_desc = TensorDesc(out_dims, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnAvgPool2d,  // host api第二段接口名称
                        INPUT(self_desc, kernel_desc, stride_desc, padding_desc,
                              ceil_mode, count_include_pad, divisor_override, cubeMathType),  // host api输入
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size); // check op graph
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 输入输出Format不匹配拦截
TEST_F(l2_avgpool2d_test, test_avgpool2d_input_output_different_format) {
    int64_t divisor_override = 0;
    bool count_include_pad = true;
    bool ceil_mode = false;

    vector<int64_t> self_dims = {1, 1, 16, 16};
    vector<int64_t> kernel_dims = {4, 4};
    vector<int64_t> stride_dims = {2, 2};
    vector<int64_t> padding_dims = {1, 1};
    vector<int64_t> out_dims = {1, 1, 8, 8};

    auto self_desc = TensorDesc(self_dims, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto out_desc = TensorDesc(out_dims, ACL_FLOAT, ACL_FORMAT_NHWC);

    auto ut = OP_API_UT(aclnnAvgPool2d,  // host api第二段接口名称
                        INPUT(self_desc, kernel_desc, stride_desc, padding_desc,
                              ceil_mode, count_include_pad, divisor_override, cubeMathType),  // host api输入
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size); // check op graph
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 参数ceil_mode取true,走3d
TEST_F(l2_avgpool2d_test, test_avgpool2d_ceilmode_true_to3d) {
    int64_t divisor_override = 0;
    bool count_include_pad = true;
    bool ceil_mode = true;

    vector<int64_t> self_dims = {1, 16, 12, 40};
    vector<int64_t> kernel_dims = {12, 12};
    vector<int64_t> stride_dims = {1, 1};
    vector<int64_t> padding_dims = {0, 0};
    vector<int64_t> out_dims = {1, 16, 1, 29};

    auto self_desc = TensorDesc(self_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(-1.0, 1.0);
    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto out_desc = TensorDesc(out_dims, ACL_FLOAT16, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnAvgPool2d,  // host api第二段接口名称
                        INPUT(self_desc, kernel_desc, stride_desc, padding_desc,
                              ceil_mode, count_include_pad, divisor_override, cubeMathType),  // host api输入
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size); // check op graph
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// 参数ceil_mode取true,走2d
TEST_F(l2_avgpool2d_test, test_avgpool2d_ceilmode_true) {
    int64_t divisor_override = 0;
    bool count_include_pad = true;
    bool ceil_mode = true;

    vector<int64_t> self_dims = {1, 16, 12, 20};
    vector<int64_t> kernel_dims = {12, 12};
    vector<int64_t> stride_dims = {1, 1};
    vector<int64_t> padding_dims = {0, 0};
    vector<int64_t> out_dims = {1, 16, 1, 9};

    auto self_desc = TensorDesc(self_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(-1.0, 1.0);
    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto out_desc = TensorDesc(out_dims, ACL_FLOAT16, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnAvgPool2d,  // host api第二段接口名称
                        INPUT(self_desc, kernel_desc, stride_desc, padding_desc,
                              ceil_mode, count_include_pad, divisor_override, cubeMathType),  // host api输入
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size); // check op graph
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// 参数ceil_mode取false,走2d
TEST_F(l2_avgpool2d_test, test_avgpool2d_ceilmode_false) {
    int64_t divisor_override = -4;
    bool count_include_pad = true;
    bool ceil_mode = false;

    vector<int64_t> self_dims = {1, 16, 12, 20};
    vector<int64_t> kernel_dims = {12, 12};
    vector<int64_t> stride_dims = {1, 1};
    vector<int64_t> padding_dims = {0, 0};
    vector<int64_t> out_dims = {1, 16, 1, 9};

    auto self_desc = TensorDesc(self_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(-1.0, 1.0);
    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto out_desc = TensorDesc(out_dims, ACL_FLOAT16, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnAvgPool2d,  // host api第二段接口名称
                        INPUT(self_desc, kernel_desc, stride_desc, padding_desc,
                              ceil_mode, count_include_pad, divisor_override, cubeMathType),  // host api输入
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size); // check op graph
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// 参数ceil_mode取false,走3d
TEST_F(l2_avgpool2d_test, test_avgpool2d_ceilmode_false_to3d) {
    int64_t divisor_override = -4;
    bool count_include_pad = true;
    bool ceil_mode = false;

    vector<int64_t> self_dims = {1, 16, 12, 40};
    vector<int64_t> kernel_dims = {12, 12};
    vector<int64_t> stride_dims = {1, 1};
    vector<int64_t> padding_dims = {0, 0};
    vector<int64_t> out_dims = {1, 16, 1, 29};

    auto self_desc = TensorDesc(self_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(-1.0, 1.0);
    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto out_desc = TensorDesc(out_dims, ACL_FLOAT16, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnAvgPool2d,  // host api第二段接口名称
                        INPUT(self_desc, kernel_desc, stride_desc, padding_desc,
                              ceil_mode, count_include_pad, divisor_override, cubeMathType),  // host api输入
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size); // check op graph
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_avgpool2d_test, ascend910B2_test_avgpool2d_cubeMathType_keepdtype_fp32) {
    int64_t divisor_override = 0;
    bool count_include_pad = true;
    bool ceil_mode = true;
    const int8_t cubeMathType_0 = KEEP_DTYPE;

    vector<int64_t> self_dims = {1, 16, 12, 40}; // NCHW
    vector<int64_t> kernel_dims = {12, 12};
    vector<int64_t> stride_dims = {1, 1};
    vector<int64_t> padding_dims = {0, 0};
    vector<int64_t> out_dims = {1, 16, 1, 29};

    auto self_desc = TensorDesc(self_dims, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(-1.0, 1.0);
    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto out_desc = TensorDesc(out_dims, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnAvgPool2d,  // host api第二段接口名称
                        INPUT(self_desc, kernel_desc, stride_desc, padding_desc,
                              ceil_mode, count_include_pad, divisor_override, cubeMathType_0),  // host api输入
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size); // check op graph
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_avgpool2d_test, Ascend910B2_test_avgpool2d_cubeMathType_keepdtype_bf16) {
    int64_t divisor_override = 0;
    bool count_include_pad = true;
    bool ceil_mode = true;
    const int8_t cubeMathType_0 = KEEP_DTYPE;

    vector<int64_t> self_dims = {1, 16, 12, 40}; // NCHW
    vector<int64_t> kernel_dims = {12, 12};
    vector<int64_t> stride_dims = {1, 1};
    vector<int64_t> padding_dims = {0, 0};
    vector<int64_t> out_dims = {1, 16, 1, 29};

    auto self_desc = TensorDesc(self_dims, ACL_BF16, ACL_FORMAT_NCHW).ValueRange(-1.0, 1.0);
    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto out_desc = TensorDesc(out_dims, ACL_BF16, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnAvgPool2d,  // host api第二段接口名称
                        INPUT(self_desc, kernel_desc, stride_desc, padding_desc,
                              ceil_mode, count_include_pad, divisor_override, cubeMathType_0),  // host api输入
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size); // check op graph
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_avgpool2d_test, test_avgpool2d_cubeMathType_usefp16_fp32) {
    int64_t divisor_override = 0;
    bool count_include_pad = true;
    bool ceil_mode = true;
    const int8_t cubeMathType_0 = USE_FP16;

    vector<int64_t> self_dims = {1, 16, 12, 40}; // NCHW
    vector<int64_t> kernel_dims = {12, 12};
    vector<int64_t> stride_dims = {1, 1};
    vector<int64_t> padding_dims = {0, 0};
    vector<int64_t> out_dims = {1, 16, 1, 29};

    auto self_desc = TensorDesc(self_dims, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(-1.0, 1.0);
    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto out_desc = TensorDesc(out_dims, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnAvgPool2d,  // host api第二段接口名称
                        INPUT(self_desc, kernel_desc, stride_desc, padding_desc,
                              ceil_mode, count_include_pad, divisor_override, cubeMathType_0),  // host api输入
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size); // check op graph
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_avgpool2d_test, ascend910B2_test_avgpool2d_cubeMathType_usefp16_fp32) {
    int64_t divisor_override = 0;
    bool count_include_pad = true;
    bool ceil_mode = true;
    const int8_t cubeMathType_0 = USE_FP16;

    vector<int64_t> self_dims = {1, 16, 12, 40}; // NCHW
    vector<int64_t> kernel_dims = {12, 12};
    vector<int64_t> stride_dims = {1, 1};
    vector<int64_t> padding_dims = {0, 0};
    vector<int64_t> out_dims = {1, 16, 1, 29};

    auto self_desc = TensorDesc(self_dims, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(-1.0, 1.0);
    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto out_desc = TensorDesc(out_dims, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnAvgPool2d,  // host api第二段接口名称
                        INPUT(self_desc, kernel_desc, stride_desc, padding_desc,
                              ceil_mode, count_include_pad, divisor_override, cubeMathType_0),  // host api输入
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size); // check op graph
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_avgpool2d_test, ascend910B2_test_avgpool2d_cubeMathType_usehf32_fp32) {
    int64_t divisor_override = 0;
    bool count_include_pad = true;
    bool ceil_mode = true;
    const int8_t cubeMathType_0 = USE_HF32;

    vector<int64_t> self_dims = {1, 16, 12, 40}; // NCHW
    vector<int64_t> kernel_dims = {12, 12};
    vector<int64_t> stride_dims = {1, 1};
    vector<int64_t> padding_dims = {0, 0};
    vector<int64_t> out_dims = {1, 16, 1, 29};

    auto self_desc = TensorDesc(self_dims, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(-1.0, 1.0);
    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto out_desc = TensorDesc(out_dims, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnAvgPool2d,  // host api第二段接口名称
                        INPUT(self_desc, kernel_desc, stride_desc, padding_desc,
                              ceil_mode, count_include_pad, divisor_override, cubeMathType_0),  // host api输入
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size); // check op graph
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_avgpool2d_test, test_avgpool2d_cubeMathType_allowfp32downprecision_fp32) {
    int64_t divisor_override = 0;
    bool count_include_pad = true;
    bool ceil_mode = true;
    const int8_t cubeMathType_0 = ALLOW_FP32_DOWN_PRECISION;

    vector<int64_t> self_dims = {1, 16, 12, 40}; // NCHW
    vector<int64_t> kernel_dims = {12, 12};
    vector<int64_t> stride_dims = {1, 1};
    vector<int64_t> padding_dims = {0, 0};
    vector<int64_t> out_dims = {1, 16, 1, 29};

    auto self_desc = TensorDesc(self_dims, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(-1.0, 1.0);
    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto out_desc = TensorDesc(out_dims, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnAvgPool2d,  // host api第二段接口名称
                        INPUT(self_desc, kernel_desc, stride_desc, padding_desc,
                              ceil_mode, count_include_pad, divisor_override, cubeMathType_0),  // host api输入
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size); // check op graph
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_avgpool2d_test, ascend910B2_test_avgpool2d_cubeMathType_allowfp32downprecision_fp32) {
    int64_t divisor_override = 0;
    bool count_include_pad = true;
    bool ceil_mode = true;
    const int8_t cubeMathType_0 = ALLOW_FP32_DOWN_PRECISION;

    vector<int64_t> self_dims = {1, 16, 12, 40}; // NCHW
    vector<int64_t> kernel_dims = {12, 12};
    vector<int64_t> stride_dims = {1, 1};
    vector<int64_t> padding_dims = {0, 0};
    vector<int64_t> out_dims = {1, 16, 1, 29};

    auto self_desc = TensorDesc(self_dims, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(-1.0, 1.0);
    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto out_desc = TensorDesc(out_dims, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnAvgPool2d,  // host api第二段接口名称
                        INPUT(self_desc, kernel_desc, stride_desc, padding_desc,
                              ceil_mode, count_include_pad, divisor_override, cubeMathType_0),  // host api输入
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size); // check op graph
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_avgpool2d_test, test_avgpool2d_divisorOverride_aicpu_fp16) {
    int64_t divisor_override = -256;
    bool count_include_pad = true;
    bool ceil_mode = false;
    const int8_t cubeMathType_0 = KEEP_DTYPE;

    vector<int64_t> self_dims = {1, 16, 12, 40}; // NCHW
    vector<int64_t> kernel_dims = {12, 12};
    vector<int64_t> stride_dims = {1, 1};
    vector<int64_t> padding_dims = {0, 0};
    vector<int64_t> out_dims = {1, 16, 1, 29};

    auto self_desc = TensorDesc(self_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(-1.0, 1.0);
    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto out_desc = TensorDesc(out_dims, ACL_FLOAT16, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnAvgPool2d,  // host api第二段接口名称
                        INPUT(self_desc, kernel_desc, stride_desc, padding_desc,
                              ceil_mode, count_include_pad, divisor_override, cubeMathType_0),  // host api输入
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size); // check op graph
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_avgpool2d_test, test_avgpool2d_divisorOverride_cube_fp32) {
    int64_t divisor_override = -4;
    bool count_include_pad = true;
    bool ceil_mode = false;

    vector<int64_t> self_dims = {1, 16, 12, 40}; // NCHW
    vector<int64_t> kernel_dims = {12, 12};
    vector<int64_t> stride_dims = {1, 1};
    vector<int64_t> padding_dims = {0, 0};
    vector<int64_t> out_dims = {1, 16, 1, 29};

    auto self_desc = TensorDesc(self_dims, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(-1.0, 1.0);
    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto out_desc = TensorDesc(out_dims, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnAvgPool2d,  // host api第二段接口名称
                        INPUT(self_desc, kernel_desc, stride_desc, padding_desc,
                              ceil_mode, count_include_pad, divisor_override, cubeMathType),  // host api输入
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size); // check op graph
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_avgpool2d_test, ascend910B2_test_avgpool2d_divisorOverride_cube_fp32) {
    int64_t divisor_override = -4;
    bool count_include_pad = true;
    bool ceil_mode = false;

    vector<int64_t> self_dims = {1, 16, 12, 20}; // NCHW
    vector<int64_t> kernel_dims = {12, 12};
    vector<int64_t> stride_dims = {1, 1};
    vector<int64_t> padding_dims = {0, 0};
    vector<int64_t> out_dims = {1, 16, 1, 9};

    auto self_desc = TensorDesc(self_dims, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(-1.0, 1.0);
    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto out_desc = TensorDesc(out_dims, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnAvgPool2d,  // host api第二段接口名称
                        INPUT(self_desc, kernel_desc, stride_desc, padding_desc,
                              ceil_mode, count_include_pad, divisor_override, cubeMathType),  // host api输入
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size); // check op graph
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_avgpool2d_test, test_avgpool2d_illegal_pad) {
    int64_t divisor_override = -4;
    bool count_include_pad = true;
    bool ceil_mode = false;

    vector<int64_t> self_dims = {1, 16, 12, 40};
    vector<int64_t> kernel_dims = {12, 12};
    vector<int64_t> stride_dims = {1, 1};
    vector<int64_t> padding_dims = {0, 0, 0};

    vector<int64_t> out_dims = {1, 16, 1, 29};

    auto self_desc = TensorDesc(self_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(-1.0, 1.0);
    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto out_desc = TensorDesc(out_dims, ACL_FLOAT16, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnAvgPool2d,  // host api第二段接口名称
                        INPUT(self_desc, kernel_desc, stride_desc, padding_desc,
                              ceil_mode, count_include_pad, divisor_override, cubeMathType),  // host api输入
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size); // check op graph
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// FP16, illegal stride ut
TEST_F(l2_avgpool2d_test, test_avgpool2d_illegal_stride) {
    int64_t divisor_override = 0;
    bool count_include_pad = true;
    bool ceil_mode = false;

    vector<int64_t> self_dims = {1, 16, 12, 40}; // NCHW
    vector<int64_t> kernel_dims = {12, 12};
    vector<int64_t> stride_dims = {1, 1, 1};
    vector<int64_t> padding_dims = {0, 0};
    vector<int64_t> out_dims = {1, 16, 1, 29};

    auto self_desc = TensorDesc(self_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(-1.0, 1.0);
    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto out_desc = TensorDesc(out_dims, ACL_FLOAT16, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnAvgPool2d, // host api第二段接口名称
                        INPUT(self_desc, kernel_desc, stride_desc, padding_desc,
                              ceil_mode, count_include_pad, divisor_override, cubeMathType), // host api输入
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size); // check op graph
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// FP16, illegal dim 3 and format NCHW
TEST_F(l2_avgpool2d_test, test_avgpool2d_illegal_dim_and_format) {
    int64_t divisor_override = 0;
    bool count_include_pad = true;
    bool ceil_mode = false;

    vector<int64_t> self_dims = {1, 16, 12}; // NCHW
    vector<int64_t> kernel_dims = {12, 12};
    vector<int64_t> stride_dims = {1, 1};
    vector<int64_t> padding_dims = {0, 0};
    vector<int64_t> out_dims = {1, 16, 1, 29};

    auto self_desc = TensorDesc(self_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(-1.0, 1.0);
    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto out_desc = TensorDesc(out_dims, ACL_FLOAT16, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnAvgPool2d, // host api第二段接口名称
                        INPUT(self_desc, kernel_desc, stride_desc, padding_desc,
                              ceil_mode, count_include_pad, divisor_override, cubeMathType), // host api输入
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size); // check op graph
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_avgpool2d_test, test_avgpool2d_illegal_pad_value1) {
    int64_t divisor_override = -4;
    bool count_include_pad = true;
    bool ceil_mode = false;

    vector<int64_t> self_dims = {1, 16, 12, 40};
    vector<int64_t> kernel_dims = {12, 12};
    vector<int64_t> stride_dims = {1, 1};
    vector<int64_t> padding_dims = {0, 8};

    vector<int64_t> out_dims = {1, 16, 1, 29};

    auto self_desc = TensorDesc(self_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(-1.0, 1.0);
    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto out_desc = TensorDesc(out_dims, ACL_FLOAT16, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnAvgPool2d,  // host api第二段接口名称
                        INPUT(self_desc, kernel_desc, stride_desc, padding_desc,
                              ceil_mode, count_include_pad, divisor_override, cubeMathType),  // host api输入
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size); // check op graph
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_avgpool2d_test, test_avgpool2d_illegal_pad_value2) {
    int64_t divisor_override = -4;
    bool count_include_pad = true;
    bool ceil_mode = false;

    vector<int64_t> self_dims = {1, 16, 12, 40};
    vector<int64_t> kernel_dims = {12, 12};
    vector<int64_t> stride_dims = {1, 1};
    vector<int64_t> padding_dims = {8, 0};

    vector<int64_t> out_dims = {1, 16, 1, 29};

    auto self_desc = TensorDesc(self_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(-1.0, 1.0);
    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto out_desc = TensorDesc(out_dims, ACL_FLOAT16, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnAvgPool2d,  // host api第二段接口名称
                        INPUT(self_desc, kernel_desc, stride_desc, padding_desc,
                              ceil_mode, count_include_pad, divisor_override, cubeMathType),  // host api输入
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size); // check op graph
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}
/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "../../../op_host/op_api/aclnn_avgpool3d.h"

#include <vector>
#include <array>
#include "gtest/gtest.h"

#include "opdev/op_log.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api/op_api_def.h"

#include "opdev/platform.h"

using namespace op;
using namespace std;

class l2_avgpool3d_test : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "avgpool3d_test Setup" << std::endl; }
    static void TearDownTestCase() { std::cout << "avgpool3d_test TearDown" << std::endl; }
};

// 输入self的shape取值不合法拦截
TEST_F(l2_avgpool3d_test, test_avgpool3d_self_illegal_shape) {
    int64_t divisor_override = 0;
    bool count_include_pad = false;
    bool ceil_mode = true;

    vector<int64_t> self_dims = {-1, 1, 16, 16, 16}; // 这里有个-1
    vector<int64_t> kernel_dims = {4, 4, 4};
    vector<int64_t> stride_dims = {2, 2, 2};
    vector<int64_t> padding_dims = {1, 1, 1};
    vector<int64_t> out_dims = {1, 1, 8, 8, 8};

    auto self_desc = TensorDesc(self_dims, ACL_FLOAT, ACL_FORMAT_NCDHW).ValueRange(-1, 1);
    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto out_desc = TensorDesc(out_dims, ACL_FLOAT, ACL_FORMAT_NCDHW);

    auto ut = OP_API_UT(aclnnAvgPool3d,  // host api第二段接口名称
                        INPUT(self_desc, kernel_desc, stride_desc, padding_desc,
                              ceil_mode, count_include_pad, divisor_override),  // host api输入
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size); // check op graph
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 参数kernel的shape取值不合法拦截
TEST_F(l2_avgpool3d_test, test_avgpool3d_kernel_illegal_shape) {
    int64_t divisor_override = 0;
    bool count_include_pad = true;
    bool ceil_mode = false;

    vector<int64_t> self_dims = {1, 1, 16, 16, 16};
    vector<int64_t> kernel_dims = {-4, 4, 4}; // 这里有个-4
    vector<int64_t> stride_dims = {2, 2, 2};
    vector<int64_t> padding_dims = {1, 1, 1};
    vector<int64_t> out_dims = {1, 1, 8, 8, 8};

    auto self_desc = TensorDesc(self_dims, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto out_desc = TensorDesc(out_dims, ACL_FLOAT, ACL_FORMAT_NCDHW);

    auto ut = OP_API_UT(aclnnAvgPool3d,  // host api第二段接口名称
                        INPUT(self_desc, kernel_desc, stride_desc, padding_desc,
                              ceil_mode, count_include_pad, divisor_override),  // host api输入
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size); // check op graph
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 参数kernel的shape取值不合法拦截
TEST_F(l2_avgpool3d_test, test_avgpool3d_kernel_illegal_shape_1) {
    int64_t divisor_override = 0;
    bool count_include_pad = true;
    bool ceil_mode = false;

    vector<int64_t> self_dims = {1, 1, 16, 16, 16};
    vector<int64_t> kernel_dims = {}; // 这里kernel为空
    vector<int64_t> stride_dims = {2, 2, 2};
    vector<int64_t> padding_dims = {1, 1, 1};
    vector<int64_t> out_dims = {1, 1, 8, 8, 8};

    auto self_desc = TensorDesc(self_dims, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto out_desc = TensorDesc(out_dims, ACL_FLOAT, ACL_FORMAT_NCDHW);

    auto ut = OP_API_UT(aclnnAvgPool3d,  // host api第二段接口名称
                        INPUT(self_desc, kernel_desc, stride_desc, padding_desc,
                              ceil_mode, count_include_pad, divisor_override),  // host api输入
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size); // check op graph
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 参数kernel的shape取值不合法拦截
TEST_F(l2_avgpool3d_test, test_avgpool3d_kernel_illegal_shape_2) {
    int64_t divisor_override = 0;
    bool count_include_pad = true;
    bool ceil_mode = true;

    vector<int64_t> self_dims = {1, 1, 16, 16, 16};
    vector<int64_t> kernel_dims = {1, 1, 1};
    vector<int64_t> stride_dims = {2, 2, 2};
    vector<int64_t> padding_dims = {1, 1, 1};
    vector<int64_t> out_dims = {1, 1, 10, 10, 10}; // outshape 不正确,正确的是{1,1,9,9}

    auto self_desc = TensorDesc(self_dims, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto out_desc = TensorDesc(out_dims, ACL_FLOAT, ACL_FORMAT_NCDHW);

    auto ut = OP_API_UT(aclnnAvgPool3d,  // host api第二段接口名称
                        INPUT(self_desc, kernel_desc, stride_desc, padding_desc,
                              ceil_mode, count_include_pad, divisor_override),  // host api输入
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size); // check op graph
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 参数stride的shape取值不合法拦截
TEST_F(l2_avgpool3d_test, test_avgpool3d_stride_illegal_shape) {
    int64_t divisor_override = 0;
    bool count_include_pad = true;
    bool ceil_mode = false;

    vector<int64_t> self_dims = {1, 1, 16, 16, 16};
    vector<int64_t> kernel_dims = {4, 4, 4};
    vector<int64_t> stride_dims = {2, -2, 2}; // 这里有个-2
    vector<int64_t> padding_dims = {1, 1, 1};
    vector<int64_t> out_dims = {1, 1, 8, 8, 8};

    auto self_desc = TensorDesc(self_dims, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto out_desc = TensorDesc(out_dims, ACL_FLOAT, ACL_FORMAT_NCDHW);

    auto ut = OP_API_UT(aclnnAvgPool3d,  // host api第二段接口名称
                        INPUT(self_desc, kernel_desc, stride_desc, padding_desc,
                              ceil_mode, count_include_pad, divisor_override),  // host api输入
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size); // check op graph
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_avgpool3d_test, test_avgpool3d_fp16) {
    int64_t divisor_override = 0;
    bool count_include_pad = true;
    bool ceil_mode = false;

    vector<int64_t> self_dims = {1, 16, 12, 12, 12}; // NCDHW
    vector<int64_t> kernel_dims = {12, 12, 12};
    vector<int64_t> stride_dims = {1, 1, 1};
    vector<int64_t> padding_dims = {0, 0, 0};
    vector<int64_t> out_dims = {1, 16, 1, 1, 1};

    auto self_desc = TensorDesc(self_dims, ACL_FLOAT16, ACL_FORMAT_NCDHW).ValueRange(-1.0, 1.0);
    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto out_desc = TensorDesc(out_dims, ACL_FLOAT16, ACL_FORMAT_NCDHW);

    auto ut = OP_API_UT(aclnnAvgPool3d, // host api第二段接口名称
                        INPUT(self_desc, kernel_desc, stride_desc, padding_desc,
                              ceil_mode, count_include_pad, divisor_override), // host api输入
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size); // check op graph
    auto curSoc = GetCurrentPlatformInfo().GetSocVersion();
    if (curSoc == SocVersion::ASCEND910B || curSoc == SocVersion::ASCEND910_93) {
        EXPECT_EQ(aclRet, ACL_SUCCESS);
    } else {
        EXPECT_NE(aclRet, ACL_SUCCESS);
    }
}

// FP16,countIncludePad=True
TEST_F(l2_avgpool3d_test, test_avgpool3d_include_pad_fp16) {
    int64_t divisor_override = 0;
    bool count_include_pad = true;
    bool ceil_mode = false;

    vector<int64_t> self_dims = {1, 16, 12, 40, 40}; // NCDHW
    vector<int64_t> kernel_dims = {12, 12, 12};
    vector<int64_t> stride_dims = {1, 1, 1};
    vector<int64_t> padding_dims = {0, 0, 0};
    vector<int64_t> out_dims = {1, 16, 1, 29, 29};

    auto self_desc = TensorDesc(self_dims, ACL_FLOAT16, ACL_FORMAT_NCDHW).ValueRange(-1.0, 1.0);
    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto out_desc = TensorDesc(out_dims, ACL_FLOAT16, ACL_FORMAT_NCDHW);

    auto ut = OP_API_UT(aclnnAvgPool3d, // host api第二段接口名称
                        INPUT(self_desc, kernel_desc, stride_desc, padding_desc,
                              ceil_mode, count_include_pad, divisor_override), // host api输入
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size); // check op graph
    auto curSoc = GetCurrentPlatformInfo().GetSocVersion();
    if (curSoc == SocVersion::ASCEND910B || curSoc == SocVersion::ASCEND910_93) {
        EXPECT_EQ(aclRet, ACL_SUCCESS);
    } else {
        EXPECT_NE(aclRet, ACL_SUCCESS);
    }
}

// FP16,countIncludePad=False
TEST_F(l2_avgpool3d_test, test_avgpool3d_exclude_pad_fp16) {
    int64_t divisor_override = 0;
    bool count_include_pad = false;
    bool ceil_mode = false;

    vector<int64_t> self_dims = {1, 16, 12, 40, 40}; // NCDHW
    vector<int64_t> kernel_dims = {12, 12, 12};
    vector<int64_t> stride_dims = {1, 1, 1};
    vector<int64_t> padding_dims = {0, 0, 0};
    vector<int64_t> out_dims = {1, 16, 1, 29, 29};

    auto self_desc = TensorDesc(self_dims, ACL_FLOAT16, ACL_FORMAT_NCDHW).ValueRange(-1.0, 1.0);
    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto out_desc = TensorDesc(out_dims, ACL_FLOAT16, ACL_FORMAT_NCDHW);

    auto ut = OP_API_UT(aclnnAvgPool3d, // host api第二段接口名称
                        INPUT(self_desc, kernel_desc, stride_desc, padding_desc,
                              ceil_mode, count_include_pad, divisor_override), // host api输入
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size); // check op graph
    auto curSoc = GetCurrentPlatformInfo().GetSocVersion();
    if (curSoc == SocVersion::ASCEND910B || curSoc == SocVersion::ASCEND910_93) {
        EXPECT_EQ(aclRet, ACL_SUCCESS);
    } else {
        EXPECT_NE(aclRet, ACL_SUCCESS);
    }
}

// FP16,输入是四维
TEST_F(l2_avgpool3d_test, test_avgpool3d_fp16_input_4d) {
    int64_t divisor_override = 0;
    bool count_include_pad = true;
    bool ceil_mode = false;

    vector<int64_t> self_dims = {16, 100, 34, 34}; // NCHW
    vector<int64_t> kernel_dims = {34, 34, 34};
    vector<int64_t> stride_dims = {1, 1, 1};
    vector<int64_t> padding_dims = {15, 1, 1};
    vector<int64_t> out_dims = {16, 97, 3, 3};

    auto self_desc = TensorDesc(self_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(-1.0, 1.0);
    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto out_desc = TensorDesc(out_dims, ACL_FLOAT16, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnAvgPool3d,  // host api第二段接口名称
                        INPUT(self_desc, kernel_desc, stride_desc, padding_desc,
                              ceil_mode, count_include_pad, divisor_override),  // host api输入
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size); // check op graph
    auto curSoc = GetCurrentPlatformInfo().GetSocVersion();
    if (curSoc == SocVersion::ASCEND910B || curSoc == SocVersion::ASCEND910_93) {
        EXPECT_EQ(aclRet, ACL_SUCCESS);
    } else {
        EXPECT_NE(aclRet, ACL_SUCCESS);
    }
}

// FP32
TEST_F(l2_avgpool3d_test, test_avgpool3d_fp32) {
    int64_t divisor_override = 0;
    bool count_include_pad = true;
    bool ceil_mode = false;

    vector<int64_t> self_dims = {1, 16, 12, 12, 12}; // NCDHW
    vector<int64_t> kernel_dims = {12, 12, 12};
    vector<int64_t> stride_dims = {1, 1, 1};
    vector<int64_t> padding_dims = {0, 0, 0};
    vector<int64_t> out_dims = {1, 16, 1, 1, 1};

    auto self_desc = TensorDesc(self_dims, ACL_FLOAT, ACL_FORMAT_NCDHW).ValueRange(-1.0, 1.0);
    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto out_desc = TensorDesc(out_dims, ACL_FLOAT, ACL_FORMAT_NCDHW);

    auto ut = OP_API_UT(aclnnAvgPool3d,  // host api第二段接口名称
                        INPUT(self_desc, kernel_desc, stride_desc, padding_desc,
                              ceil_mode, count_include_pad, divisor_override),  // host api输入
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size); // check op graph
    auto curSoc = GetCurrentPlatformInfo().GetSocVersion();
    if (curSoc == SocVersion::ASCEND910B || curSoc == SocVersion::ASCEND910_93) {
        EXPECT_EQ(aclRet, ACL_SUCCESS);
    } else {
        EXPECT_NE(aclRet, ACL_SUCCESS);
    }
}

// FP32, 输入是四维
TEST_F(l2_avgpool3d_test, test_avgpool3d_fp32_input_4d) {
    int64_t divisor_override = 0;
    bool count_include_pad = true;
    bool ceil_mode = false;

    vector<int64_t> self_dims = {16, 100, 34, 34}; // NCHW
    vector<int64_t> kernel_dims = {34, 34, 34};
    vector<int64_t> stride_dims = {1, 1, 1};
    vector<int64_t> padding_dims = {15, 1, 1};
    vector<int64_t> out_dims = {16, 97, 3, 3};

    auto self_desc = TensorDesc(self_dims, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(-1.0, 1.0);
    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto out_desc = TensorDesc(out_dims, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnAvgPool3d,  // host api第二段接口名称
                        INPUT(self_desc, kernel_desc, stride_desc, padding_desc,
                              ceil_mode, count_include_pad, divisor_override),  // host api输入
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size); // check op graph
    auto curSoc = GetCurrentPlatformInfo().GetSocVersion();
    if (curSoc == SocVersion::ASCEND910B || curSoc == SocVersion::ASCEND910_93) {
        EXPECT_EQ(aclRet, ACL_SUCCESS);
    } else {
        EXPECT_NE(aclRet, ACL_SUCCESS);
    }
}

// self空指针拦截
TEST_F(l2_avgpool3d_test, test_avgpool3d_nullptr_self) {
    int64_t divisor_override = 0;
    bool count_include_pad = true;
    bool ceil_mode = false;

    vector<int64_t> self_dims = {1, 1, 16, 16, 16};
    vector<int64_t> kernel_dims = {4, 4, 4};
    vector<int64_t> stride_dims = {2, 2, 2};
    vector<int64_t> padding_dims = {1, 1, 1};
    vector<int64_t> out_dims = {1, 1, 8, 8, 8};

    auto self_desc = TensorDesc(self_dims, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto out_desc = TensorDesc(out_dims, ACL_FLOAT, ACL_FORMAT_NCDHW);

    auto ut = OP_API_UT(aclnnAvgPool3d,  // host api第二段接口名称
                        INPUT((aclTensor*)nullptr, kernel_desc, stride_desc, padding_desc,
                              ceil_mode, count_include_pad, divisor_override),  // host api输入
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size); // check op graph
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// kernel空指针拦截
TEST_F(l2_avgpool3d_test, test_avgpool3d_nullptr_kernel) {
    int64_t divisor_override = 0;
    bool count_include_pad = true;
    bool ceil_mode = false;

    vector<int64_t> self_dims = {1, 1, 16, 16, 16};
    vector<int64_t> kernel_dims = {4, 4, 4};
    vector<int64_t> stride_dims = {2, 2, 2};
    vector<int64_t> padding_dims = {1, 1, 1};
    vector<int64_t> out_dims = {1, 1, 8, 8, 8};

    auto self_desc = TensorDesc(self_dims, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto out_desc = TensorDesc(out_dims, ACL_FLOAT, ACL_FORMAT_NCDHW);

    auto ut = OP_API_UT(aclnnAvgPool3d,  // host api第二段接口名称
                        INPUT(self_desc, (aclIntArray*)nullptr, stride_desc, padding_desc,
                              ceil_mode, count_include_pad, divisor_override),  // host api输入
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size); // check op graph
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// stride空指针拦截
TEST_F(l2_avgpool3d_test, test_avgpool3d_nullptr_stride) {
    int64_t divisor_override = 0;
    bool count_include_pad = true;
    bool ceil_mode = false;

    vector<int64_t> self_dims = {1, 1, 16, 16, 16};
    vector<int64_t> kernel_dims = {4, 4, 4};
    vector<int64_t> stride_dims = {2, 2, 2};
    vector<int64_t> padding_dims = {1, 1, 1};
    vector<int64_t> out_dims = {1, 1, 8, 8, 8};

    auto self_desc = TensorDesc(self_dims, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto out_desc = TensorDesc(out_dims, ACL_FLOAT, ACL_FORMAT_NCDHW);

    auto ut = OP_API_UT(aclnnAvgPool3d,  // host api第二段接口名称
                        INPUT(self_desc, kernel_desc, (aclIntArray*)nullptr, padding_desc,
                              ceil_mode, count_include_pad, divisor_override),  // host api输入
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size); // check op graph
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// padding空指针拦截
TEST_F(l2_avgpool3d_test, test_avgpool3d_nullptr_padding) {
    int64_t divisor_override = 0;
    bool count_include_pad = true;
    bool ceil_mode = false;

    vector<int64_t> self_dims = {1, 1, 16, 16, 16};
    vector<int64_t> kernel_dims = {4, 4, 4};
    vector<int64_t> stride_dims = {2, 2, 2};
    vector<int64_t> padding_dims = {1, 1, 1};
    vector<int64_t> out_dims = {1, 1, 8, 8, 8};

    auto self_desc = TensorDesc(self_dims, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto out_desc = TensorDesc(out_dims, ACL_FLOAT, ACL_FORMAT_NCDHW);

    auto ut = OP_API_UT(aclnnAvgPool3d,  // host api第二段接口名称
                        INPUT(self_desc, kernel_desc, stride_desc, (aclIntArray*)nullptr,
                              ceil_mode, count_include_pad, divisor_override),  // host api输入
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size); // check op graph
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// out空指针拦截
TEST_F(l2_avgpool3d_test, test_avgpool3d_nullptr_out) {
    int64_t divisor_override = 0;
    bool count_include_pad = true;
    bool ceil_mode = false;

    vector<int64_t> self_dims = {1, 1, 16, 16, 16};
    vector<int64_t> kernel_dims = {4, 4, 4};
    vector<int64_t> stride_dims = {2, 2, 2};
    vector<int64_t> padding_dims = {1, 1, 1};
    vector<int64_t> out_dims = {1, 1, 8, 8, 8};

    auto self_desc = TensorDesc(self_dims, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto out_desc = TensorDesc(out_dims, ACL_FLOAT, ACL_FORMAT_NCDHW);

    auto ut = OP_API_UT(aclnnAvgPool3d,  // host api第二段接口名称
                        INPUT(self_desc, kernel_desc, stride_desc, padding_desc,
                              ceil_mode, count_include_pad, divisor_override),  // host api输入
                        OUTPUT((aclTensor*)nullptr));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size); // check op graph
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// 输出out的shape非法拦截
TEST_F(l2_avgpool3d_test, test_avgpool3d_illegal_output_shape) {
    int64_t divisor_override = 0;
    bool count_include_pad = true;
    bool ceil_mode = false;

    vector<int64_t> self_dims = {1, 1, 16, 16, 16};
    vector<int64_t> kernel_dims = {4, 4, 4};
    vector<int64_t> stride_dims = {2, 2, 2};
    vector<int64_t> padding_dims = {1, 1, 1};
    vector<int64_t> out_dims = {0, 1, 8, 8, 8}; // 这里有一维是0,是空tensor

    auto self_desc = TensorDesc(self_dims, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto out_desc = TensorDesc(out_dims, ACL_FLOAT, ACL_FORMAT_NCDHW);

    auto ut = OP_API_UT(aclnnAvgPool3d,  // host api第二段接口名称
                        INPUT(self_desc, kernel_desc, stride_desc, padding_desc,
                              ceil_mode, count_include_pad, divisor_override),  // host api输入
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size); // check op graph
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 异常Dtype拦截
TEST_F(l2_avgpool3d_test, test_avgpool3d_dtype_abnormal) {
    int64_t divisor_override = 0;
    bool count_include_pad = true;
    bool ceil_mode = false;

    vector<int64_t> self_dims = {1, 1, 16, 16, 16};
    vector<int64_t> kernel_dims = {4, 4, 4};
    vector<int64_t> stride_dims = {2, 2, 2};
    vector<int64_t> padding_dims = {1, 1, 1};
    vector<int64_t> out_dims = {1, 1, 8, 8, 8};

    auto self_desc = TensorDesc(self_dims, ACL_DT_UNDEFINED, ACL_FORMAT_NCDHW);
    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto out_desc = TensorDesc(out_dims, ACL_DT_UNDEFINED, ACL_FORMAT_NCDHW);

    auto ut = OP_API_UT(aclnnAvgPool3d,  // host api第二段接口名称
                        INPUT(self_desc, kernel_desc, stride_desc, padding_desc,
                              ceil_mode, count_include_pad, divisor_override),  // host api输入
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size); // check op graph
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 构造一个不支持的shape,看能否拦截
TEST_F(l2_avgpool3d_test, test_avgpool3d_format_abnormal) {
    int64_t divisor_override = 0;
    bool count_include_pad = true;
    bool ceil_mode = false;

    vector<int64_t> self_dims = {1, 16, 16}; // 不满足NCDHW
    vector<int64_t> kernel_dims = {4, 4, 4};
    vector<int64_t> stride_dims = {2, 2, 2};
    vector<int64_t> padding_dims = {1, 1, 1};
    vector<int64_t> out_dims = {1, 1, 8, 8, 8};

    auto self_desc = TensorDesc(self_dims, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto out_desc = TensorDesc(out_dims, ACL_FLOAT, ACL_FORMAT_NCDHW);

    auto ut = OP_API_UT(aclnnAvgPool3d,  // host api第二段接口名称
                        INPUT(self_desc, kernel_desc, stride_desc, padding_desc,
                              ceil_mode, count_include_pad, divisor_override),  // host api输入
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size); // check op graph
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_avgpool3d_test, test_avgpool3d_input_output_same_dtype) {
    int64_t divisor_override = 0;
    bool count_include_pad = true;
    bool ceil_mode = false;

    vector<int64_t> self_dims = {1, 1, 16, 16, 16};
    vector<int64_t> kernel_dims = {4, 4, 4};
    vector<int64_t> stride_dims = {2, 2, 2};
    vector<int64_t> padding_dims = {1, 1, 1};
    vector<int64_t> out_dims = {1, 1, 8, 8, 8};

    auto self_desc = TensorDesc(self_dims, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto out_desc = TensorDesc(out_dims, ACL_FLOAT, ACL_FORMAT_NCDHW);

    auto ut = OP_API_UT(aclnnAvgPool3d,  // host api第二段接口名称
                        INPUT(self_desc, kernel_desc, stride_desc, padding_desc,
                              ceil_mode, count_include_pad, divisor_override),  // host api输入
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size); // check op graph
    auto curSoc = GetCurrentPlatformInfo().GetSocVersion();
    if (curSoc == SocVersion::ASCEND910B || curSoc == SocVersion::ASCEND910_93) {
        EXPECT_EQ(aclRet, ACL_SUCCESS);
    } else {
        EXPECT_NE(aclRet, ACL_SUCCESS);
    }
}

TEST_F(l2_avgpool3d_test, test_avgpool3d_input_output_different_dtype) {
    int64_t divisor_override = 0;
    bool count_include_pad = true;
    bool ceil_mode = false;

    vector<int64_t> self_dims = {1, 1, 16, 16, 16};
    vector<int64_t> kernel_dims = {4, 4, 4};
    vector<int64_t> stride_dims = {2, 2, 2};
    vector<int64_t> padding_dims = {1, 1, 1};
    vector<int64_t> out_dims = {1, 1, 8, 8, 8};

    auto self_desc = TensorDesc(self_dims, ACL_FLOAT16, ACL_FORMAT_NCDHW);
    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto out_desc = TensorDesc(out_dims, ACL_FLOAT, ACL_FORMAT_NCDHW);

    auto ut = OP_API_UT(aclnnAvgPool3d,  // host api第二段接口名称
                        INPUT(self_desc, kernel_desc, stride_desc, padding_desc,
                              ceil_mode, count_include_pad, divisor_override),  // host api输入
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size); // check op graph
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_avgpool3d_test, test_avgpool3d_input_output_same_format) {
    int64_t divisor_override = 0;
    bool count_include_pad = true;
    bool ceil_mode = false;

    vector<int64_t> self_dims = {1, 1, 16, 16, 16};
    vector<int64_t> kernel_dims = {4, 4, 4};
    vector<int64_t> stride_dims = {2, 2, 2};
    vector<int64_t> padding_dims = {1, 1, 1};
    vector<int64_t> out_dims = {1, 1, 8, 8, 8};

    auto self_desc = TensorDesc(self_dims, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto out_desc = TensorDesc(out_dims, ACL_FLOAT, ACL_FORMAT_NCDHW);

    auto ut = OP_API_UT(aclnnAvgPool3d,  // host api第二段接口名称
                        INPUT(self_desc, kernel_desc, stride_desc, padding_desc,
                              ceil_mode, count_include_pad, divisor_override),  // host api输入
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size); // check op graph
    auto curSoc = GetCurrentPlatformInfo().GetSocVersion();
    if (curSoc == SocVersion::ASCEND910B || curSoc == SocVersion::ASCEND910_93) {
        EXPECT_EQ(aclRet, ACL_SUCCESS);
    } else {
        EXPECT_NE(aclRet, ACL_SUCCESS);
    }
}

// 输入输出Format不匹配拦截
TEST_F(l2_avgpool3d_test, test_avgpool3d_input_output_different_format) {
    int64_t divisor_override = 0;
    bool count_include_pad = true;
    bool ceil_mode = false;

    vector<int64_t> self_dims = {1, 1, 16, 16, 16};
    vector<int64_t> kernel_dims = {4, 4, 4};
    vector<int64_t> stride_dims = {2, 2, 2};
    vector<int64_t> padding_dims = {1, 1, 1};
    vector<int64_t> out_dims = {1, 1, 8, 8, 8};

    auto self_desc = TensorDesc(self_dims, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto out_desc = TensorDesc(out_dims, ACL_FLOAT, ACL_FORMAT_NHWC);

    auto ut = OP_API_UT(aclnnAvgPool3d,  // host api第二段接口名称
                        INPUT(self_desc, kernel_desc, stride_desc, padding_desc,
                              ceil_mode, count_include_pad, divisor_override),  // host api输入
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size); // check op graph
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 参数ceil_mode取true
TEST_F(l2_avgpool3d_test, test_avgpool3d_ceilmode_true) {
    int64_t divisor_override = 0;
    bool count_include_pad = true;
    bool ceil_mode = true;

    vector<int64_t> self_dims = {1, 16, 12, 40, 40};
    vector<int64_t> kernel_dims = {12, 12, 12};
    vector<int64_t> stride_dims = {1, 1, 1};
    vector<int64_t> padding_dims = {0, 0, 0};
    vector<int64_t> out_dims = {1, 16, 1, 29, 29};

    auto self_desc = TensorDesc(self_dims, ACL_FLOAT16, ACL_FORMAT_NCDHW).ValueRange(-1.0, 1.0);
    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto out_desc = TensorDesc(out_dims, ACL_FLOAT16, ACL_FORMAT_NCDHW);

    auto ut = OP_API_UT(aclnnAvgPool3d,  // host api第二段接口名称
                        INPUT(self_desc, kernel_desc, stride_desc, padding_desc,
                              ceil_mode, count_include_pad, divisor_override),  // host api输入
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size); // check op graph
    auto curSoc = GetCurrentPlatformInfo().GetSocVersion();
    if (curSoc == SocVersion::ASCEND910B || curSoc == SocVersion::ASCEND910_93) {
        EXPECT_EQ(aclRet, ACL_SUCCESS);
    } else {
        EXPECT_NE(aclRet, ACL_SUCCESS);
    }
}

// 参数ceil_mode取false
TEST_F(l2_avgpool3d_test, test_avgpool3d_ceilmode_false) {
    int64_t divisor_override = -4;
    bool count_include_pad = true;
    bool ceil_mode = false;

    vector<int64_t> self_dims = {1, 16, 12, 40, 40};
    vector<int64_t> kernel_dims = {12, 12, 12};
    vector<int64_t> stride_dims = {1, 1, 1};
    vector<int64_t> padding_dims = {0, 0, 0};
    vector<int64_t> out_dims = {1, 16, 1, 29, 29};

    auto self_desc = TensorDesc(self_dims, ACL_FLOAT16, ACL_FORMAT_NCDHW).ValueRange(-1.0, 1.0);
    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto out_desc = TensorDesc(out_dims, ACL_FLOAT16, ACL_FORMAT_NCDHW);

    auto ut = OP_API_UT(aclnnAvgPool3d,  // host api第二段接口名称
                        INPUT(self_desc, kernel_desc, stride_desc, padding_desc,
                              ceil_mode, count_include_pad, divisor_override),  // host api输入
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size); // check op graph
    auto curSoc = GetCurrentPlatformInfo().GetSocVersion();
    if (curSoc == SocVersion::ASCEND910B || curSoc == SocVersion::ASCEND910_93) {
        EXPECT_EQ(aclRet, ACL_SUCCESS);
    } else {
        EXPECT_NE(aclRet, ACL_SUCCESS);
    }
}

TEST_F(l2_avgpool3d_test, test_avgpool3d_divisorOverride_fp32) {
    int64_t divisor_override = 4;
    bool count_include_pad = true;
    bool ceil_mode = true;

    vector<int64_t> self_dims = {1, 16, 12, 12, 12}; // NCDHW
    vector<int64_t> kernel_dims = {12, 12, 12};
    vector<int64_t> stride_dims = {1, 1, 1};
    vector<int64_t> padding_dims = {0, 0, 0};
    vector<int64_t> out_dims = {1, 16, 1, 1, 1};

    auto self_desc = TensorDesc(self_dims, ACL_FLOAT, ACL_FORMAT_NCDHW).ValueRange(-1.0, 1.0);
    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto out_desc = TensorDesc(out_dims, ACL_FLOAT, ACL_FORMAT_NCDHW);

    auto ut = OP_API_UT(aclnnAvgPool3d,  // host api第二段接口名称
                        INPUT(self_desc, kernel_desc, stride_desc, padding_desc,
                              ceil_mode, count_include_pad, divisor_override),  // host api输入
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size); // check op graph
    auto curSoc = GetCurrentPlatformInfo().GetSocVersion();
    if (curSoc == SocVersion::ASCEND910B || curSoc == SocVersion::ASCEND910_93) {
        EXPECT_EQ(aclRet, ACL_SUCCESS);
    } else {
        EXPECT_NE(aclRet, ACL_SUCCESS);
    }
}

TEST_F(l2_avgpool3d_test, test_avgpool3d_illegal_pad) {
    int64_t divisor_override = -4;
    bool count_include_pad = true;
    bool ceil_mode = false;

    vector<int64_t> self_dims = {1, 16, 12, 40, 40};
    vector<int64_t> kernel_dims = {12, 12, 12};
    vector<int64_t> stride_dims = {1, 1, 1};
    vector<int64_t> padding_dims = {0, 0, 0, 0};

    vector<int64_t> out_dims = {1, 16, 1, 29, 29};

    auto self_desc = TensorDesc(self_dims, ACL_FLOAT16, ACL_FORMAT_NCDHW).ValueRange(-1.0, 1.0);
    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto out_desc = TensorDesc(out_dims, ACL_FLOAT16, ACL_FORMAT_NCDHW);

    auto ut = OP_API_UT(aclnnAvgPool3d,  // host api第二段接口名称
                        INPUT(self_desc, kernel_desc, stride_desc, padding_desc,
                              ceil_mode, count_include_pad, divisor_override),  // host api输入
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size); // check op graph
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_avgpool3d_test, test_avgpool3d_illegal_pad_value) {
    int64_t divisor_override = -4;
    bool count_include_pad = true;
    bool ceil_mode = false;

    vector<int64_t> self_dims = {1, 16, 12, 40, 40};
    vector<int64_t> kernel_dims = {12, 12, 12};
    vector<int64_t> stride_dims = {1, 1, 1};
    vector<int64_t> padding_dims = {0, 0, 8};

    vector<int64_t> out_dims = {1, 16, 1, 29, 29};

    auto self_desc = TensorDesc(self_dims, ACL_FLOAT16, ACL_FORMAT_NCDHW).ValueRange(-1.0, 1.0);
    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto out_desc = TensorDesc(out_dims, ACL_FLOAT16, ACL_FORMAT_NCDHW);

    auto ut = OP_API_UT(aclnnAvgPool3d,  // host api第二段接口名称
                        INPUT(self_desc, kernel_desc, stride_desc, padding_desc,
                              ceil_mode, count_include_pad, divisor_override),  // host api输入
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size); // check op graph
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// FP16, illegal stride ut
TEST_F(l2_avgpool3d_test, test_avgpool3d_illegal_stride) {
    int64_t divisor_override = 0;
    bool count_include_pad = true;
    bool ceil_mode = false;

    vector<int64_t> self_dims = {1, 16, 12, 40, 40}; // NCDHW
    vector<int64_t> kernel_dims = {12, 12, 12};
    vector<int64_t> stride_dims = {1, 1, 1, 1};
    vector<int64_t> padding_dims = {0};
    vector<int64_t> out_dims = {1, 16, 1, 29, 29};

    auto self_desc = TensorDesc(self_dims, ACL_FLOAT16, ACL_FORMAT_NCDHW).ValueRange(-1.0, 1.0);
    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto out_desc = TensorDesc(out_dims, ACL_FLOAT16, ACL_FORMAT_NCDHW);

    auto ut = OP_API_UT(aclnnAvgPool3d, // host api第二段接口名称
                        INPUT(self_desc, kernel_desc, stride_desc, padding_desc,
                              ceil_mode, count_include_pad, divisor_override), // host api输入
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size); // check op graph
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// FP16, illegal dim 4 and format NCDHW
TEST_F(l2_avgpool3d_test, test_avgpool3d_illegal_dim_and_format) {
    int64_t divisor_override = 0;
    bool count_include_pad = true;
    bool ceil_mode = false;

    vector<int64_t> self_dims = {1, 16, 12, 12}; // NCDHW
    vector<int64_t> kernel_dims = {12, 12, 12};
    vector<int64_t> stride_dims = {1, 1, 1};
    vector<int64_t> padding_dims = {0, 0, 0};
    vector<int64_t> out_dims = {1, 16, 1, 29, 29};

    auto self_desc = TensorDesc(self_dims, ACL_FLOAT16, ACL_FORMAT_NCDHW).ValueRange(-1.0, 1.0);
    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto out_desc = TensorDesc(out_dims, ACL_FLOAT16, ACL_FORMAT_NCDHW);

    auto ut = OP_API_UT(aclnnAvgPool3d, // host api第二段接口名称
                        INPUT(self_desc, kernel_desc, stride_desc, padding_desc,
                              ceil_mode, count_include_pad, divisor_override), // host api输入
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size); // check op graph
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_avgpool3d_test, test_avgpool3d_empty_input_tensor) {
    int64_t divisor_override = 4;
    bool count_include_pad = true;
    bool ceil_mode = true;

    vector<int64_t> self_dims = {1, 0, 12, 12, 12}; // 0: empty tensor
    vector<int64_t> kernel_dims = {12, 12, 12};
    vector<int64_t> stride_dims = {1, 1, 1};
    vector<int64_t> padding_dims = {0, 0, 0};
    vector<int64_t> out_dims = {1, 0, 1, 1, 1};

    auto self_desc = TensorDesc(self_dims, ACL_FLOAT, ACL_FORMAT_NCDHW).ValueRange(-1.0, 1.0);
    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto out_desc = TensorDesc(out_dims, ACL_FLOAT, ACL_FORMAT_NCDHW);

    auto ut = OP_API_UT(aclnnAvgPool3d,  // host api第二段接口名称
                        INPUT(self_desc, kernel_desc, stride_desc, padding_desc,
                              ceil_mode, count_include_pad, divisor_override),  // host api输入
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size); // check op graph
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_avgpool3d_test, test_avgpool3d_kernel_over_input_shape) {
    int64_t divisor_override = 4;
    bool count_include_pad = true;
    bool ceil_mode = true;

    vector<int64_t> self_dims = {1, 16, 12, 12, 12};
    vector<int64_t> kernel_dims = {14, 12, 12}; // kernel > inputD
    vector<int64_t> stride_dims = {1, 1, 1};
    vector<int64_t> padding_dims = {0, 0, 0};
    vector<int64_t> out_dims = {1, 16, 1, 1, 1};

    auto self_desc = TensorDesc(self_dims, ACL_FLOAT, ACL_FORMAT_NCDHW).ValueRange(-1.0, 1.0);
    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto out_desc = TensorDesc(out_dims, ACL_FLOAT, ACL_FORMAT_NCDHW);

    auto ut = OP_API_UT(aclnnAvgPool3d,  // host api第二段接口名称
                        INPUT(self_desc, kernel_desc, stride_desc, padding_desc,
                              ceil_mode, count_include_pad, divisor_override),  // host api输入
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size); // check op graph
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_avgpool3d_test, ascend910_9589_test_avgpool3d_fp16) {
    int64_t divisor_override = 0;
    bool count_include_pad = true;
    bool ceil_mode = false;

    vector<int64_t> self_dims = {1, 16, 12, 12, 12}; // NCDHW
    vector<int64_t> kernel_dims = {12, 12, 12};
    vector<int64_t> stride_dims = {1, 1, 1};
    vector<int64_t> padding_dims = {0, 0, 0};
    vector<int64_t> out_dims = {1, 16, 1, 1, 1};

    auto self_desc = TensorDesc(self_dims, ACL_FLOAT16, ACL_FORMAT_NCDHW).ValueRange(-1.0, 1.0);
    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto out_desc = TensorDesc(out_dims, ACL_FLOAT16, ACL_FORMAT_NCDHW);

    auto ut = OP_API_UT(aclnnAvgPool3d, // host api第二段接口名称
                        INPUT(self_desc, kernel_desc, stride_desc, padding_desc,
                              ceil_mode, count_include_pad, divisor_override), // host api输入
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size); // check op graph
    auto curSoc = GetCurrentPlatformInfo().GetSocVersion();
    if (curSoc == SocVersion::ASCEND910_95) {
        EXPECT_EQ(aclRet, ACL_SUCCESS);
    } else {
        EXPECT_NE(aclRet, ACL_SUCCESS);
    }
}

TEST_F(l2_avgpool3d_test, test_avgpool3d_ncdhw_fp16) {
    int64_t divisor_override = 1;
    bool count_include_pad = true;
    bool ceil_mode = false;

    vector<int64_t> self_dims = {1, 50, 12, 40, 40}; // NCDHW
    vector<int64_t> kernel_dims = {2, 2, 2};
    vector<int64_t> stride_dims = {1, 1, 1};
    vector<int64_t> padding_dims = {0, 0, 0};
    vector<int64_t> out_dims = {1, 50, 11, 39, 39};

    auto self_desc = TensorDesc(self_dims, ACL_FLOAT16, ACL_FORMAT_NCDHW).ValueRange(-1.0, 1.0);
    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto out_desc = TensorDesc(out_dims, ACL_FLOAT16, ACL_FORMAT_NCDHW);

    auto ut = OP_API_UT(aclnnAvgPool3d, // host api第二段接口名称
                        INPUT(self_desc, kernel_desc, stride_desc, padding_desc,
                              ceil_mode, count_include_pad, divisor_override), // host api输入
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size); // check op graph
    auto curSoc = GetCurrentPlatformInfo().GetSocVersion();
    if (curSoc == SocVersion::ASCEND910B || curSoc == SocVersion::ASCEND910_93) {
        EXPECT_EQ(aclRet, ACL_SUCCESS);
    } else {
        EXPECT_NE(aclRet, ACL_SUCCESS);
    }
}

TEST_F(l2_avgpool3d_test, test_avgpool3d_ncdhw_fp32) {
    int64_t divisor_override = 1;
    bool count_include_pad = true;
    bool ceil_mode = false;

    vector<int64_t> self_dims = {1, 50, 12, 40, 40}; // NCDHW
    vector<int64_t> kernel_dims = {2, 2, 2};
    vector<int64_t> stride_dims = {1, 1, 1};
    vector<int64_t> padding_dims = {0, 0, 0};
    vector<int64_t> out_dims = {1, 50, 11, 39, 39};

    auto self_desc = TensorDesc(self_dims, ACL_FLOAT, ACL_FORMAT_NCDHW).ValueRange(-1.0, 1.0);
    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto out_desc = TensorDesc(out_dims, ACL_FLOAT, ACL_FORMAT_NCDHW);

    auto ut = OP_API_UT(aclnnAvgPool3d, // host api第二段接口名称
                        INPUT(self_desc, kernel_desc, stride_desc, padding_desc,
                              ceil_mode, count_include_pad, divisor_override), // host api输入
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size); // check op graph
    auto curSoc = GetCurrentPlatformInfo().GetSocVersion();
    if (curSoc == SocVersion::ASCEND910B || curSoc == SocVersion::ASCEND910_93) {
        EXPECT_EQ(aclRet, ACL_SUCCESS);
    } else {
        EXPECT_NE(aclRet, ACL_SUCCESS);
    }
}

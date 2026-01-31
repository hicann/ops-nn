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
#include "../../../op_api/aclnn_max_pool.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace op;
using namespace std;

class l2_max_pool_test : public testing::Test {
    protected:
    static void SetUpTestCase() {
        std::cout << "l2_max_pool_test SetUp" << std::endl;
    }

    static void TearDownTestCase() {std::cout << "l2_max_pool_test TearDown" << std::endl;}
};

// 正常场景：数据类型为float
TEST_F(l2_max_pool_test, ascend910B2_normal_float_testcase01) {
    vector<int64_t> self_dims = {10, 1, 1, 244};
    vector<int64_t> kernel_dims = {1, 2};
    vector<int64_t> stride_dims = {1, 2};
    const int64_t autoPad = 0;
    vector<int64_t> padding_dims = {0, 0, 0, 0};
    vector<int64_t> dilation_dims = {1, 1};
    int64_t ceilMode = 1;
    vector<int64_t> out_dims = {10, 1, 1, 122};


    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);

    auto tensor_self_desc = TensorDesc(self_dims, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto out_tensor_desc = TensorDesc(out_dims, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnMaxPool,
                        INPUT(tensor_self_desc, kernel_desc, stride_desc, autoPad, padding_desc, dilation_desc,
                              ceilMode),
                        OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    ut.TestPrecision();
}

TEST_F(l2_max_pool_test, ascend910B2_normal_float_testcase02) {
    vector<int64_t> self_dims = {10, 1, 1, 122};
    vector<int64_t> kernel_dims = {1, 2};
    vector<int64_t> stride_dims = {1, 2};
    const int64_t autoPad = 0;
    vector<int64_t> padding_dims = {0, 0, 0, 0};
    vector<int64_t> dilation_dims = {1, 1};
    int64_t ceilMode = 1;
    vector<int64_t> out_dims = {10, 1, 1, 61};


    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);

    auto tensor_self_desc = TensorDesc(self_dims, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto out_tensor_desc = TensorDesc(out_dims, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnMaxPool,
                        INPUT(tensor_self_desc, kernel_desc, stride_desc, autoPad, padding_desc, dilation_desc,
                              ceilMode),
                        OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    ut.TestPrecision();
}

TEST_F(l2_max_pool_test, ascend910B2_normal_float_testcase03) {
    vector<int64_t> self_dims = {10, 64, 40, 244};
    vector<int64_t> kernel_dims = {1, 2};
    vector<int64_t> stride_dims = {1, 2};
    const int64_t autoPad = 0;
    vector<int64_t> padding_dims = {0, 0, 0, 0};
    vector<int64_t> dilation_dims = {1, 1};
    int64_t ceilMode = 1;
    vector<int64_t> out_dims = {10, 64, 40, 122};


    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);

    auto tensor_self_desc = TensorDesc(self_dims, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto out_tensor_desc = TensorDesc(out_dims, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnMaxPool,
                        INPUT(tensor_self_desc, kernel_desc, stride_desc, autoPad, padding_desc, dilation_desc,
                              ceilMode),
                        OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    ut.TestPrecision();
}

TEST_F(l2_max_pool_test, ascend910B2_normal_float_testcase04) {
    vector<int64_t> self_dims = {10, 80, 40, 122};
    vector<int64_t> kernel_dims = {2, 2};
    vector<int64_t> stride_dims = {2, 2};
    const int64_t autoPad = 0;
    vector<int64_t> padding_dims = {0, 0, 0, 0};
    vector<int64_t> dilation_dims = {1, 1};
    int64_t ceilMode = 1;
    vector<int64_t> out_dims = {10, 80, 20, 61};


    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);

    auto tensor_self_desc = TensorDesc(self_dims, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto out_tensor_desc = TensorDesc(out_dims, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnMaxPool,
                        INPUT(tensor_self_desc, kernel_desc, stride_desc, autoPad, padding_desc, dilation_desc,
                              ceilMode),
                        OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    ut.TestPrecision();
}

TEST_F(l2_max_pool_test, ascend910B2_normal_float_testcase05) {
    vector<int64_t> self_dims = {10, 104, 20, 61};
    vector<int64_t> kernel_dims = {2, 1};
    vector<int64_t> stride_dims = {2, 1};
    const int64_t autoPad = 0;
    vector<int64_t> padding_dims = {0, 0, 0, 0};
    vector<int64_t> dilation_dims = {1, 1};
    int64_t ceilMode = 1;
    vector<int64_t> out_dims = {10, 104, 10, 61};


    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);

    auto tensor_self_desc = TensorDesc(self_dims, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto out_tensor_desc = TensorDesc(out_dims, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnMaxPool,
                        INPUT(tensor_self_desc, kernel_desc, stride_desc, autoPad, padding_desc, dilation_desc,
                              ceilMode),
                        OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    ut.TestPrecision();
}

// 正常场景：数据类型为float16
TEST_F(l2_max_pool_test, ascend910_normal_float16_testcase06) {
    vector<int64_t> self_dims = {10, 1, 1, 244};
    vector<int64_t> kernel_dims = {1, 2};
    vector<int64_t> stride_dims = {1, 2};
    const int64_t autoPad= 0;
    vector<int64_t> padding_dims = {0, 0, 0, 0};
    vector<int64_t> dilation_dims = {1, 1};
    int64_t ceilMode = 1;
    vector<int64_t> out_dims = {10, 1, 1, 122};


    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);

    auto tensor_self_desc = TensorDesc(self_dims, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto out_tensor_desc = TensorDesc(out_dims, ACL_FLOAT16, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnMaxPool,
                        INPUT(tensor_self_desc, kernel_desc, stride_desc, autoPad, padding_desc, dilation_desc,
                              ceilMode),
                        OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    ut.TestPrecision();
}

TEST_F(l2_max_pool_test, ascend910_normal_float16_testcase07) {
    vector<int64_t> self_dims = {10, 1, 1, 122};
    vector<int64_t> kernel_dims = {1, 2};
    vector<int64_t> stride_dims = {1, 2};
    const int64_t autoPad = 0;
    vector<int64_t> padding_dims = {0, 0, 0, 0};
    vector<int64_t> dilation_dims = {1, 1};
    int64_t ceilMode = 1;
    vector<int64_t> out_dims = {10, 1, 1, 61};


    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);

    auto tensor_self_desc = TensorDesc(self_dims, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto out_tensor_desc = TensorDesc(out_dims, ACL_FLOAT16, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnMaxPool,
                        INPUT(tensor_self_desc, kernel_desc, stride_desc, autoPad, padding_desc, dilation_desc,
                              ceilMode),
                        OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    ut.TestPrecision();
}

TEST_F(l2_max_pool_test, ascend910_normal_float16_testcase08) {
    vector<int64_t> self_dims = {10, 64, 40, 244};
    vector<int64_t> kernel_dims = {1, 2};
    vector<int64_t> stride_dims = {1, 2};
    const int64_t autoPad = 0;
    vector<int64_t> padding_dims = {0, 0, 0, 0};
    vector<int64_t> dilation_dims = {1, 1};
    int64_t ceilMode = 1;
    vector<int64_t> out_dims = {10, 64, 40, 122};


    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);

    auto tensor_self_desc = TensorDesc(self_dims, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto out_tensor_desc = TensorDesc(out_dims, ACL_FLOAT16, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnMaxPool,
                        INPUT(tensor_self_desc, kernel_desc, stride_desc, autoPad, padding_desc, dilation_desc,
                              ceilMode),
                        OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    ut.TestPrecision();
}

TEST_F(l2_max_pool_test, ascend910_normal_float16_testcase09) {
    vector<int64_t> self_dims = {10, 80, 40, 122};
    vector<int64_t> kernel_dims = {2, 2};
    vector<int64_t> stride_dims = {2, 2};
    const int64_t autoPad = 0;
    vector<int64_t> padding_dims = {0, 0, 0, 0};
    vector<int64_t> dilation_dims = {1, 1};
    int64_t ceilMode = 1;
    vector<int64_t> out_dims = {10, 80, 20, 61};


    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);

    auto tensor_self_desc = TensorDesc(self_dims, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto out_tensor_desc = TensorDesc(out_dims, ACL_FLOAT16, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnMaxPool,
                        INPUT(tensor_self_desc, kernel_desc, stride_desc, autoPad, padding_desc, dilation_desc,
                              ceilMode),
                        OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    ut.TestPrecision();
}

TEST_F(l2_max_pool_test, ascend910_normal_float16_testcase10) {
    vector<int64_t> self_dims = {10, 104, 20, 61};
    vector<int64_t> kernel_dims = {2, 1};
    vector<int64_t> stride_dims = {2, 1};
    const int64_t autoPad = 0;
    vector<int64_t> padding_dims = {0, 0, 0, 0};
    vector<int64_t> dilation_dims = {1, 1};
    int64_t ceilMode = 1;
    vector<int64_t> out_dims = {10, 104, 10, 61};

    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);

    auto tensor_self_desc = TensorDesc(self_dims, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto out_tensor_desc = TensorDesc(out_dims, ACL_FLOAT16, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnMaxPool,
                        INPUT(tensor_self_desc, kernel_desc, stride_desc, autoPad, padding_desc, dilation_desc,
                              ceilMode),
                        OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    ut.TestPrecision();
}

TEST_F(l2_max_pool_test, ascend910B2_normal_float_precision_testcase11) {
    vector<int64_t> self_dims = {10, 104, 20, 61};
    vector<int64_t> kernel_dims = {2, 1};
    vector<int64_t> stride_dims = {2, 1};
    const int64_t autoPad = 0;
    vector<int64_t> padding_dims = {0, 0, 0, 0};
    vector<int64_t> dilation_dims = {1, 1};
    int64_t ceilMode = 1;
    vector<int64_t> out_dims = {10, 104, 10, 61};


    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);

    auto tensor_self_desc = TensorDesc(self_dims, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(-100, 100);
    auto out_tensor_desc = TensorDesc(out_dims, ACL_FLOAT, ACL_FORMAT_NCHW).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnMaxPool,
                        INPUT(tensor_self_desc, kernel_desc, stride_desc, autoPad, padding_desc, dilation_desc,
                              ceilMode),
                        OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    ut.TestPrecision();
}

TEST_F(l2_max_pool_test, ascend910_normal_float16_precision_testcase12) {
    vector<int64_t> self_dims = {10, 104, 20, 61};
    vector<int64_t> kernel_dims = {2, 1};
    vector<int64_t> stride_dims = {2, 1};
    const int64_t autoPad = 0;
    vector<int64_t> padding_dims = {0, 0, 0, 0};
    vector<int64_t> dilation_dims = {1, 1};
    int64_t ceilMode = 1;
    vector<int64_t> out_dims = {10, 104, 10, 61};


    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);

    auto tensor_self_desc = TensorDesc(self_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(-100, 100);
    auto out_tensor_desc = TensorDesc(out_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).Precision(0.001, 0.001);

    auto ut = OP_API_UT(aclnnMaxPool,
                        INPUT(tensor_self_desc, kernel_desc, stride_desc, autoPad, padding_desc, dilation_desc,
                              ceilMode),
                        OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    ut.TestPrecision();
}

// 测试3D tensor 输入场景
// 正常场景：format是CHW,dtype是float;
TEST_F(l2_max_pool_test, ascend910B2_normal_float_self_3D_testcase13) {
    vector<int64_t> self_dims = {3, 2, 2};
    vector<int64_t> kernel_dims = {2};
    vector<int64_t> stride_dims = {2};
    const int64_t autoPad = 0;
    vector<int64_t> padding_dims = {0};
    vector<int64_t> dilation_dims = {1};
    int64_t ceilMode = 1;
    vector<int64_t> out_dims = {3, 1, 1};


    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);

    auto tensor_self_desc = TensorDesc(self_dims, ACL_FLOAT, ACL_FORMAT_NCL);
    auto out_tensor_desc = TensorDesc(out_dims, ACL_FLOAT, ACL_FORMAT_NCL);

    auto ut = OP_API_UT(aclnnMaxPool,
                        INPUT(tensor_self_desc, kernel_desc, stride_desc, autoPad, padding_desc, dilation_desc,
                              ceilMode),
                        OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    ut.TestPrecision();
}

// 正常场景：format是CHW,dtype是float16;
TEST_F(l2_max_pool_test, ascend910_normal_float16_self_3D_testcase14) {
    vector<int64_t> self_dims = {3, 2, 2};
    vector<int64_t> kernel_dims = {2};
    vector<int64_t> stride_dims = {2};
    const int64_t autoPad = 0;
    vector<int64_t> padding_dims = {0};
    vector<int64_t> dilation_dims = {1};
    int64_t ceilMode = 1;
    vector<int64_t> out_dims = {3, 1, 1};


    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);

    auto tensor_self_desc = TensorDesc(self_dims, ACL_FLOAT16, ACL_FORMAT_NCL);
    auto out_tensor_desc = TensorDesc(out_dims, ACL_FLOAT16, ACL_FORMAT_NCL);

    auto ut = OP_API_UT(aclnnMaxPool,
                        INPUT(tensor_self_desc, kernel_desc, stride_desc, autoPad, padding_desc, dilation_desc,
                              ceilMode),
                        OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    ut.TestPrecision();
}

// 测试 pads非零值，dilation二元组的任意值场景
TEST_F(l2_max_pool_test, ascend910_normal_pads_4D_testcase15) {
    vector<int64_t> self_dims = {10, 16, 50, 32};
    vector<int64_t> kernel_dims = {5, 3};
    vector<int64_t> stride_dims = {2, 1};
    const int64_t autoPad = 0;
    vector<int64_t> padding_dims = {2, 1};
    vector<int64_t> dilation_dims = {1, 1};
    int64_t ceilMode = 1;
    vector<int64_t> out_dims = {10, 16, 26, 32};


    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);

    auto tensor_self_desc = TensorDesc(self_dims, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto out_tensor_desc = TensorDesc(out_dims, ACL_FLOAT16, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnMaxPool,
                        INPUT(tensor_self_desc, kernel_desc, stride_desc, autoPad, padding_desc, dilation_desc,
                              ceilMode),
                        OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    ut.TestPrecision();
}

TEST_F(l2_max_pool_test, ascend910_normal_pads_3D_testcase16) {
    vector<int64_t> self_dims = {16, 50, 32};
    vector<int64_t> kernel_dims = {5, 3};
    vector<int64_t> stride_dims = {2, 1};
    const int64_t autoPad = 0;
    vector<int64_t> padding_dims = {2, 1};
    vector<int64_t> dilation_dims = {1, 1};
    int64_t ceilMode = 1;
    vector<int64_t> out_dims = {16, 26, 32};


    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);

    auto tensor_self_desc = TensorDesc(self_dims, ACL_FLOAT16, ACL_FORMAT_NCL);
    auto out_tensor_desc = TensorDesc(out_dims, ACL_FLOAT16, ACL_FORMAT_NCL);

    auto ut = OP_API_UT(aclnnMaxPool,
                        INPUT(tensor_self_desc, kernel_desc, stride_desc, autoPad, padding_desc, dilation_desc,
                              ceilMode),
                        OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    ut.TestPrecision();
}

// 异常场景
// 异常场景：数据类型是int32
TEST_F(l2_max_pool_test, ascend910_abnormal_datatype_int32_testcase17) {
    vector<int64_t> self_dims = {10, 80, 40, 122};
    vector<int64_t> kernel_dims = {2, 2};
    vector<int64_t> stride_dims = {2, 2};
    const int64_t autoPad = 0;
    vector<int64_t> padding_dims = {0, 0, 0, 0};
    vector<int64_t> dilation_dims = {1, 1};
    int64_t ceilMode = 1;
    vector<int64_t> out_dims = {10, 80, 20, 61};


    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);

    auto tensor_self_desc = TensorDesc(self_dims, ACL_INT32, ACL_FORMAT_NCHW);
    auto out_tensor_desc = TensorDesc(out_dims, ACL_INT32, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnMaxPool,
                        INPUT(tensor_self_desc, kernel_desc, stride_desc, autoPad, padding_desc, dilation_desc,
                              ceilMode),
                        OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    // EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 异常场景：数据类型是double
TEST_F(l2_max_pool_test, ascend910_abnormal_datatype_double_testcase18) {
    vector<int64_t> self_dims = {10, 80, 40, 122};
    vector<int64_t> kernel_dims = {2, 2};
    vector<int64_t> stride_dims = {2, 2};
    const int64_t autoPad = 0;
    vector<int64_t> padding_dims = {0, 0, 0, 0};
    vector<int64_t> dilation_dims = {1, 1};
    int64_t ceilMode = 1;
    vector<int64_t> out_dims = {10, 80, 20, 61};


    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);

    auto tensor_self_desc = TensorDesc(self_dims, ACL_DOUBLE, ACL_FORMAT_NCHW);
    auto out_tensor_desc = TensorDesc(out_dims, ACL_DOUBLE, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnMaxPool,
                        INPUT(tensor_self_desc, kernel_desc, stride_desc, autoPad, padding_desc, dilation_desc,
                              ceilMode),
                        OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_max_pool_test, ascend910B2_abnormal_datatype_self_out_diff_testcase19) {
    vector<int64_t> self_dims = {10, 80, 40, 122};
    vector<int64_t> kernel_dims = {2, 2};
    vector<int64_t> stride_dims = {2, 2};
    const int64_t autoPad = 0;
    vector<int64_t> padding_dims = {0, 0, 0, 0};
    vector<int64_t> dilation_dims = {1, 1};
    int64_t ceilMode = 1;
    vector<int64_t> out_dims = {10, 80, 20, 61};


    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);

    auto tensor_self_desc = TensorDesc(self_dims, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto out_tensor_desc = TensorDesc(out_dims, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnMaxPool,
                        INPUT(tensor_self_desc, kernel_desc, stride_desc, autoPad, padding_desc, dilation_desc,
                              ceilMode),
                        OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_max_pool_test, ascend910_abnormal_self_nullptr_testcase20) {
    vector<int64_t> self_dims = {10, 80, 40, 122};
    vector<int64_t> kernel_dims = {2, 2};
    vector<int64_t> stride_dims = {2, 2};
    const int64_t autoPad = 0;
    vector<int64_t> padding_dims = {0, 0, 0, 0};
    vector<int64_t> dilation_dims = {1, 1};
    int64_t ceilMode = 1;
    vector<int64_t> out_dims = {10, 80, 20, 61};


    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);

    auto tensor_self_desc = nullptr;
    auto out_tensor_desc = TensorDesc(out_dims, ACL_FLOAT16, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnMaxPool,
                        INPUT(tensor_self_desc, kernel_desc, stride_desc, autoPad, padding_desc, dilation_desc,
                              ceilMode),
                        OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_max_pool_test, ascend910_abnormal_kernel_nullptr_testcase21) {
    vector<int64_t> self_dims = {10, 80, 40, 122};
    vector<int64_t> kernel_dims = {2, 2};
    vector<int64_t> stride_dims = {2, 2};
    const int64_t autoPad = 0;
    vector<int64_t> padding_dims = {0, 0, 0, 0};
    vector<int64_t> dilation_dims = {1, 1};
    int64_t ceilMode = 1;
    vector<int64_t> out_dims = {10, 80, 20, 61};


    auto kernel_desc = nullptr;
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);

    auto tensor_self_desc = TensorDesc(self_dims, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto out_tensor_desc = TensorDesc(out_dims, ACL_FLOAT16, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnMaxPool,
                        INPUT(tensor_self_desc, kernel_desc, stride_desc, autoPad, padding_desc, dilation_desc,
                              ceilMode),
                        OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_max_pool_test, ascend910_abnormal_autoPad_testcase22) {
    vector<int64_t> self_dims = {10, 80, 40, 122};
    vector<int64_t> kernel_dims = {2, 2};
    vector<int64_t> stride_dims = {2, 2};
    const int64_t autoPad = 1;
    vector<int64_t> padding_dims = {0, 0, 0, 0};
    vector<int64_t> dilation_dims = {1, 1};
    int64_t ceilMode = 1;
    vector<int64_t> out_dims = {10, 80, 20, 61};


    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);

    auto tensor_self_desc = TensorDesc(self_dims, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto out_tensor_desc = TensorDesc(out_dims, ACL_FLOAT16, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnMaxPool,
                        INPUT(tensor_self_desc, kernel_desc, stride_desc, autoPad, padding_desc, dilation_desc,
                              ceilMode),
                        OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_max_pool_test, ascend910_abnormal_dilation_testcase23) {
    vector<int64_t> self_dims = {10, 80, 40, 122};
    vector<int64_t> kernel_dims = {2, 2};
    vector<int64_t> stride_dims = {2, 2};
    const int64_t autoPad = 0;
    vector<int64_t> padding_dims = {0, 0, 0, 0};
    vector<int64_t> dilation_dims = {1, 2};
    int64_t ceilMode = 1;
    vector<int64_t> out_dims = {10, 80, 20, 61};


    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);

    auto tensor_self_desc = TensorDesc(self_dims, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto out_tensor_desc = TensorDesc(out_dims, ACL_FLOAT16, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnMaxPool,
                        INPUT(tensor_self_desc, kernel_desc, stride_desc, autoPad, padding_desc, dilation_desc,
                              ceilMode),
                        OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 异常shape场景；
// 异常场景：self是2维
TEST_F(l2_max_pool_test, ascend910_abnormal_self_2D_testcase24) {
    vector<int64_t> self_dims = {40, 244};
    vector<int64_t> kernel_dims = {1, 2};
    vector<int64_t> stride_dims = {1, 2};
    const int64_t autoPad = 0;
    vector<int64_t> padding_dims = {0, 0, 0, 0};
    vector<int64_t> dilation_dims = {1, 1};
    int64_t ceilMode = 1;
    vector<int64_t> out_dims = {40, 122};


    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);

    auto tensor_self_desc = TensorDesc(self_dims, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto out_tensor_desc = TensorDesc(out_dims, ACL_FLOAT16, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnMaxPool,
                        INPUT(tensor_self_desc, kernel_desc, stride_desc, autoPad, padding_desc, dilation_desc,
                              ceilMode),
                        OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_max_pool_test, ascend910_abnormal_self_5D_testcase25) {
    vector<int64_t> self_dims = {40, 40, 64, 40, 244};
    vector<int64_t> kernel_dims = {1, 2};
    vector<int64_t> stride_dims = {1, 2};
    const int64_t autoPad = 0;
    vector<int64_t> padding_dims = {0, 0, 0, 0};
    vector<int64_t> dilation_dims = {1, 1};
    int64_t ceilMode = 1;
    vector<int64_t> out_dims = {40, 40, 64, 40, 122};


    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);

    auto tensor_self_desc = TensorDesc(self_dims, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto out_tensor_desc = TensorDesc(out_dims, ACL_FLOAT16, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnMaxPool,
                        INPUT(tensor_self_desc, kernel_desc, stride_desc, autoPad, padding_desc, dilation_desc,
                              ceilMode),
                        OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 异常场景：kernelShape为空，即元素个数是0
TEST_F(l2_max_pool_test, ascend910_abnormal_kernelShape_0_testcase26) {
    vector<int64_t> self_dims = {10, 104, 20, 61};
    vector<int64_t> kernel_dims = {};
    vector<int64_t> stride_dims = {2, 1};
    const int64_t autoPad = 0;
    vector<int64_t> padding_dims = {0, 0, 0, 0};
    vector<int64_t> dilation_dims = {1, 1};
    int64_t ceilMode = 1;
    vector<int64_t> out_dims = {10, 104, 10, 61};

    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);

    auto tensor_self_desc = TensorDesc(self_dims, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto out_tensor_desc = TensorDesc(out_dims, ACL_FLOAT16, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnMaxPool,
                        INPUT(tensor_self_desc, kernel_desc, stride_desc, autoPad, padding_desc, dilation_desc,
                              ceilMode),
                        OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 异常场景：kernel的值是0
TEST_F(l2_max_pool_test, ascend910_abnormal_kernelValue_0_testcase27) {
    vector<int64_t> self_dims = {10, 104, 20, 61};
    vector<int64_t> kernel_dims = {0, 1};
    vector<int64_t> stride_dims = {2, 1};
    const int64_t autoPad = 0;
    vector<int64_t> padding_dims = {0, 0, 0, 0};
    vector<int64_t> dilation_dims = {1, 1};
    int64_t ceilMode = 1;
    vector<int64_t> out_dims = {10, 104, 10, 61};

    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);

    auto tensor_self_desc = TensorDesc(self_dims, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto out_tensor_desc = TensorDesc(out_dims, ACL_FLOAT16, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnMaxPool,
                        INPUT(tensor_self_desc, kernel_desc, stride_desc, autoPad, padding_desc, dilation_desc,
                              ceilMode),
                        OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 异常场景：strides的值是0
TEST_F(l2_max_pool_test, ascend910_abnormal_stridesValue_0_testcase28) {
    vector<int64_t> self_dims = {10, 104, 20, 61};
    vector<int64_t> kernel_dims = {2, 1};
    vector<int64_t> stride_dims = {0, 0};
    const int64_t autoPad = 0;
    vector<int64_t> padding_dims = {0, 0, 0, 0};
    vector<int64_t> dilation_dims = {1, 1};
    int64_t ceilMode = 1;
    vector<int64_t> out_dims = {10, 104, 10, 61};

    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);

    auto tensor_self_desc = TensorDesc(self_dims, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto out_tensor_desc = TensorDesc(out_dims, ACL_FLOAT16, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnMaxPool,
                        INPUT(tensor_self_desc, kernel_desc, stride_desc, autoPad, padding_desc, dilation_desc,
                              ceilMode),
                        OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 异常场景：padding为空
TEST_F(l2_max_pool_test, ascend910_abnormal_paddingValue_nullptr_testcase29) {
    vector<int64_t> self_dims = {10, 104, 20, 61};
    vector<int64_t> kernel_dims = {2, 1};
    vector<int64_t> stride_dims = {2, 1};
    const int64_t autoPad = 0;
    vector<int64_t> padding_dims = {0, 0, 0, 0};
    vector<int64_t> dilation_dims = {1, 1};
    int64_t ceilMode = 1;
    vector<int64_t> out_dims = {10, 104, 10, 61};

    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = nullptr;
    auto dilation_desc = IntArrayDesc(dilation_dims);

    auto tensor_self_desc = TensorDesc(self_dims, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto out_tensor_desc = TensorDesc(out_dims, ACL_FLOAT16, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnMaxPool,
                        INPUT(tensor_self_desc, kernel_desc, stride_desc, autoPad, padding_desc, dilation_desc,
                              ceilMode),
                        OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// 异常场景：pads大于kernelShape/2
TEST_F(l2_max_pool_test, ascend910_abnormal_padsValue_testcase30) {
    vector<int64_t> self_dims = {10, 104, 20, 61};
    vector<int64_t> kernel_dims = {2, 1};
    vector<int64_t> stride_dims = {2, 1};
    const int64_t autoPad = 0;
    vector<int64_t> padding_dims = {2, 2, 2, 2};
    vector<int64_t> dilation_dims = {1, 1};
    int64_t ceilMode = 1;
    vector<int64_t> out_dims = {10, 104, 10, 61};

    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);

    auto tensor_self_desc = TensorDesc(self_dims, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto out_tensor_desc = TensorDesc(out_dims, ACL_FLOAT16, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnMaxPool,
                        INPUT(tensor_self_desc, kernel_desc, stride_desc, autoPad, padding_desc, dilation_desc,
                              ceilMode),
                        OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 异常场景：dilations是的值不为1
TEST_F(l2_max_pool_test, ascend910_abnormal_dilationsValue_testcase31) {
    vector<int64_t> self_dims = {10, 104, 20, 61};
    vector<int64_t> kernel_dims = {2, 1};
    vector<int64_t> stride_dims = {2, 1};
    const int64_t autoPad = 0;
    vector<int64_t> padding_dims = {0, 0, 0, 0};
    vector<int64_t> dilation_dims = {2, 2, 2, 2};
    int64_t ceilMode = 1;
    vector<int64_t> out_dims = {10, 104, 10, 61};

    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);

    auto tensor_self_desc = TensorDesc(self_dims, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto out_tensor_desc = TensorDesc(out_dims, ACL_FLOAT16, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnMaxPool,
                        INPUT(tensor_self_desc, kernel_desc, stride_desc, autoPad, padding_desc, dilation_desc,
                              ceilMode),
                        OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 异常场景：out的shape异常，输入shape经计算输出与指定的out shape不匹配
TEST_F(l2_max_pool_test, ascend910_abnormal_out_shape_testcase32) {
    vector<int64_t> self_dims = {10, 104, 20, 61};
    vector<int64_t> kernel_dims = {2, 1};
    vector<int64_t> stride_dims = {2, 1};
    const int64_t autoPad = 0;
    vector<int64_t> padding_dims = {0, 0, 0, 0};
    vector<int64_t> dilation_dims = {1, 1};
    int64_t ceilMode = 1;
    vector<int64_t> out_dims = {10, 104, 20, 61};

    auto kernel_desc = IntArrayDesc(kernel_dims);
    auto stride_desc = IntArrayDesc(stride_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);

    auto tensor_self_desc = TensorDesc(self_dims, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto out_tensor_desc = TensorDesc(out_dims, ACL_FLOAT16, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnMaxPool,
                        INPUT(tensor_self_desc, kernel_desc, stride_desc, autoPad, padding_desc, dilation_desc,
                              ceilMode),
                        OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

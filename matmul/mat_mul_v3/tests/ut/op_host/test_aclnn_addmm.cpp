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
#include <float.h>
#include "gtest/gtest.h"

#include "../../../op_host/op_api/aclnn_addmm.h"
#include "op_api/op_api_def.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"

using namespace std;
using namespace op;

class l2_addmm_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << "addmm_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "addmm_test TearDown" << endl;
    }

    void test_run(
        vector<int64_t> selfDims, aclDataType selfDtype, aclFormat selfFormat, vector<int64_t> selfRange,
        vector<int64_t> mat1Dims, aclDataType mat1Dtype, aclFormat mat1Format, vector<int64_t> mat1Range,
        vector<int64_t> mat2Dims, aclDataType mat2Dtype, aclFormat mat2Format, vector<int64_t> mat2Range,
        ScalarDesc beta, ScalarDesc alpha, vector<int64_t> outDims, aclDataType outDtype, aclFormat outFormat,
        int8_t cubeMathType)
    {
        auto self = TensorDesc(selfDims, selfDtype, selfFormat).ValueRange(selfRange[0], selfRange[1]);
        auto mat1 = TensorDesc(mat1Dims, mat1Dtype, mat1Format).ValueRange(mat1Range[0], mat1Range[1]);
        auto mat2 = TensorDesc(mat2Dims, mat2Dtype, mat2Format).ValueRange(mat2Range[0], mat2Range[1]);
        auto out = TensorDesc(outDims, outDtype, outFormat).Precision(0.00001, 0.00001);

        auto ut = OP_API_UT(aclnnAddmm, INPUT(self, mat1, mat2, beta, alpha), OUTPUT(out), cubeMathType);
        uint64_t workspaceSize = 0;
        aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
        EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
        // ut.TestPrecision();
    }

    void test_run_invalid(
        vector<int64_t> selfDims, aclDataType selfDtype, aclFormat selfFormat, vector<int64_t> selfRange,
        vector<int64_t> mat1Dims, aclDataType mat1Dtype, aclFormat mat1Format, vector<int64_t> mat1Range,
        vector<int64_t> mat2Dims, aclDataType mat2Dtype, aclFormat mat2Format, vector<int64_t> mat2Range,
        ScalarDesc beta, ScalarDesc alpha, vector<int64_t> outDims, aclDataType outDtype, aclFormat outFormat,
        int8_t cubeMathType)
    {
        auto self = TensorDesc(selfDims, selfDtype, selfFormat).ValueRange(selfRange[0], selfRange[1]);
        auto mat1 = TensorDesc(mat1Dims, mat1Dtype, mat1Format).ValueRange(mat1Range[0], mat1Range[1]);
        auto mat2 = TensorDesc(mat2Dims, mat2Dtype, mat2Format).ValueRange(mat2Range[0], mat2Range[1]);
        auto out = TensorDesc(outDims, outDtype, outFormat).Precision(0.00001, 0.00001);

        auto ut = OP_API_UT(aclnnAddmm, INPUT(self, mat1, mat2, beta, alpha), OUTPUT(out), cubeMathType);
        uint64_t workspaceSize = 0;
        aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
        EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
    }
};

// 正常流程16*16
TEST_F(l2_addmm_test, case_fp_16mm16)
{
    auto self = TensorDesc({16, 16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat1 = TensorDesc({16, 16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat2 = TensorDesc({16, 16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out = TensorDesc({16, 16}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto beta = ScalarDesc(1.0f);
    auto alpha = ScalarDesc(1.0f);
    int8_t cubeMathType = ALLOW_FP32_DOWN_PRECISION;

    auto ut = OP_API_UT(aclnnAddmm, INPUT(self, mat1, mat2, beta, alpha), OUTPUT(out), cubeMathType);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常流程4*4
TEST_F(l2_addmm_test, case_fp_4mm4)
{
    auto self = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat1 = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat2 = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto beta = ScalarDesc(1.0f);
    auto alpha = ScalarDesc(1.0f);
    int8_t cubeMathType = ALLOW_FP32_DOWN_PRECISION;

    auto ut = OP_API_UT(aclnnAddmm, INPUT(self, mat1, mat2, beta, alpha), OUTPUT(out), cubeMathType);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 直接调用addmm算子 1xN bias broadcast
TEST_F(l2_addmm_test, case_fp16_4mm4_1mm4bias)
{
    auto self = TensorDesc({1, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat1 = TensorDesc({4, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat2 = TensorDesc({4, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out = TensorDesc({4, 4}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto beta = ScalarDesc(1.0f);
    auto alpha = ScalarDesc(1.0f);
    int8_t cubeMathType = ALLOW_FP32_DOWN_PRECISION;

    auto ut = OP_API_UT(aclnnAddmm, INPUT(self, mat1, mat2, beta, alpha), OUTPUT(out), cubeMathType);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 直接调用addmm算子 1xN bias broadcast
TEST_F(l2_addmm_test, ascend910B2_case_bf16_4mm4_1mm4bias)
{
    auto self = TensorDesc({1, 4}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat1 = TensorDesc({4, 4}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat2 = TensorDesc({4, 4}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out = TensorDesc({4, 4}, ACL_BF16, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto beta = ScalarDesc(1.0f);
    auto alpha = ScalarDesc(1.0f);
    int8_t cubeMathType = ALLOW_FP32_DOWN_PRECISION;

    auto ut = OP_API_UT(aclnnAddmm, INPUT(self, mat1, mat2, beta, alpha), OUTPUT(out), cubeMathType);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 直接调用addmm算子 910A fp32进出
TEST_F(l2_addmm_test, ascend910_case_fp32_mm_add)
{
    auto self = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat1 = TensorDesc({4, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat2 = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out = TensorDesc({4, 3}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto beta = ScalarDesc(1.0f);
    auto alpha = ScalarDesc(1.0f);
    int8_t cubeMathType = USE_FP16;

    auto ut = OP_API_UT(aclnnAddmm, INPUT(self, mat1, mat2, beta, alpha), OUTPUT(out), cubeMathType);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 直接调用addmm算子 N维 bias broadcast
TEST_F(l2_addmm_test, case_fp16_4mm4_4bias)
{
    auto self = TensorDesc({4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat1 = TensorDesc({4, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat2 = TensorDesc({4, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out = TensorDesc({4, 4}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto beta = ScalarDesc(1.0f);
    auto alpha = ScalarDesc(1.0f);
    int8_t cubeMathType = ALLOW_FP32_DOWN_PRECISION;

    auto ut = OP_API_UT(aclnnAddmm, INPUT(self, mat1, mat2, beta, alpha), OUTPUT(out), cubeMathType);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 直接调用addmm算子 N维 bias broadcast
TEST_F(l2_addmm_test, ascend910B2_case_bf16_4mm4_4bias)
{
    auto self = TensorDesc({4}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat1 = TensorDesc({4, 4}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat2 = TensorDesc({4, 4}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out = TensorDesc({4, 4}, ACL_BF16, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto beta = ScalarDesc(1.0f);
    auto alpha = ScalarDesc(1.0f);
    int8_t cubeMathType = ALLOW_FP32_DOWN_PRECISION;

    auto ut = OP_API_UT(aclnnAddmm, INPUT(self, mat1, mat2, beta, alpha), OUTPUT(out), cubeMathType);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常流程4*4 alpha!=1
TEST_F(l2_addmm_test, case_fp_4mm4_1)
{
    auto self = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat1 = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat2 = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto beta = ScalarDesc(2.5f);
    auto alpha = ScalarDesc(5.2f);
    int8_t cubeMathType = ALLOW_FP32_DOWN_PRECISION;

    auto ut = OP_API_UT(aclnnAddmm, INPUT(self, mat1, mat2, beta, alpha), OUTPUT(out), cubeMathType);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常流程4*4
TEST_F(l2_addmm_test, case_fp16_4mm4)
{
    auto self = TensorDesc({4, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat1 = TensorDesc({4, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat2 = TensorDesc({4, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out = TensorDesc({4, 4}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.005, 0.005);
    auto beta = ScalarDesc(1.0f);
    auto alpha = ScalarDesc(1.0f);
    int8_t cubeMathType = ALLOW_FP32_DOWN_PRECISION;

    auto ut = OP_API_UT(aclnnAddmm, INPUT(self, mat1, mat2, beta, alpha), OUTPUT(out), cubeMathType);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常流程16*16
TEST_F(l2_addmm_test, case_fp16_16mm16)
{
    auto self = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat1 = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat2 = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.005, 0.005);
    auto beta = ScalarDesc(1.0f);
    auto alpha = ScalarDesc(1.0f);
    int8_t cubeMathType = ALLOW_FP32_DOWN_PRECISION;

    auto ut = OP_API_UT(aclnnAddmm, INPUT(self, mat1, mat2, beta, alpha), OUTPUT(out), cubeMathType);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// beta == 0 场景
TEST_F(l2_addmm_test, case_beta_0)
{
    auto self = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat1 = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat2 = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.005, 0.005);
    auto beta = ScalarDesc(0.0f);
    auto alpha = ScalarDesc(1.0f);
    int8_t cubeMathType = ALLOW_FP32_DOWN_PRECISION;

    auto ut = OP_API_UT(aclnnAddmm, INPUT(self, mat1, mat2, beta, alpha), OUTPUT(out), cubeMathType);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// self空指针
TEST_F(l2_addmm_test, case_empty_tensor_input_1)
{
    auto mat1 = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat2 = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto beta = ScalarDesc(1.0f);
    auto alpha = ScalarDesc(1.0f);
    int8_t cubeMathType = ALLOW_FP32_DOWN_PRECISION;

    auto ut = OP_API_UT(aclnnAddmm, INPUT(nullptr, mat1, mat2, beta, alpha), OUTPUT(out), cubeMathType);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// mat1空指针
TEST_F(l2_addmm_test, case_empty_tensor_input_2)
{
    auto self = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat2 = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto beta = ScalarDesc(1.0f);
    auto alpha = ScalarDesc(1.0f);
    int8_t cubeMathType = ALLOW_FP32_DOWN_PRECISION;

    auto ut = OP_API_UT(aclnnAddmm, INPUT(self, nullptr, mat2, beta, alpha), OUTPUT(out), cubeMathType);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// mat2空指针
TEST_F(l2_addmm_test, case_empty_tensor_input_3)
{
    auto self = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat1 = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto beta = ScalarDesc(1.0f);
    auto alpha = ScalarDesc(1.0f);
    int8_t cubeMathType = ALLOW_FP32_DOWN_PRECISION;

    auto ut = OP_API_UT(aclnnAddmm, INPUT(self, mat1, nullptr, beta, alpha), OUTPUT(out), cubeMathType);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// beta空指针
TEST_F(l2_addmm_test, case_empty_tensor_input_4)
{
    auto self = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat1 = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat2 = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto alpha = ScalarDesc(1.0f);
    int8_t cubeMathType = ALLOW_FP32_DOWN_PRECISION;

    auto ut = OP_API_UT(aclnnAddmm, INPUT(self, mat1, mat2, nullptr, alpha), OUTPUT(out), cubeMathType);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// alpha空指针
TEST_F(l2_addmm_test, case_empty_tensor_input_5)
{
    auto self = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat1 = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat2 = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto beta = ScalarDesc(1.0f);
    int8_t cubeMathType = ALLOW_FP32_DOWN_PRECISION;

    auto ut = OP_API_UT(aclnnAddmm, INPUT(self, mat1, mat2, beta, nullptr), OUTPUT(out), cubeMathType);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// out空指针
TEST_F(l2_addmm_test, case_empty_tensor_out)
{
    auto self = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat1 = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat2 = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto beta = ScalarDesc(1.0f);
    auto alpha = ScalarDesc(1.0f);
    int8_t cubeMathType = ALLOW_FP32_DOWN_PRECISION;

    auto ut = OP_API_UT(aclnnAddmm, INPUT(self, mat1, mat2, beta, alpha), OUTPUT(nullptr), cubeMathType);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// 数据类型不支持 dtype = INT32
TEST_F(l2_addmm_test, case_dtype_int32)
{
    auto self = TensorDesc({16, 16}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat1 = TensorDesc({16, 16}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat2 = TensorDesc({16, 16}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out = TensorDesc({16, 16}, ACL_INT32, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto beta = ScalarDesc(1.0f);
    auto alpha = ScalarDesc(1.0f);
    int8_t cubeMathType = ALLOW_FP32_DOWN_PRECISION;

    auto ut = OP_API_UT(aclnnAddmm, INPUT(self, mat1, mat2, beta, alpha), OUTPUT(out), cubeMathType);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 数据格式支持 NCHW
TEST_F(l2_addmm_test, case_format_nchw)
{
    auto self = TensorDesc({16, 16}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(0, 2);
    auto mat1 = TensorDesc({16, 16}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(0, 2);
    auto mat2 = TensorDesc({16, 16}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(0, 2);
    auto out = TensorDesc({16, 16}, ACL_FLOAT, ACL_FORMAT_NCHW).Precision(0.001, 0.001);
    auto beta = ScalarDesc(1.0f);
    auto alpha = ScalarDesc(1.0f);
    int8_t cubeMathType = ALLOW_FP32_DOWN_PRECISION;

    auto ut = OP_API_UT(aclnnAddmm, INPUT(self, mat1, mat2, beta, alpha), OUTPUT(out), cubeMathType);

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// mat1不是2维
TEST_F(l2_addmm_test, case_mat1_dim_not_2)
{
    auto self = TensorDesc({16, 16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat1 = TensorDesc({16, 16, 16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat2 = TensorDesc({16, 16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out = TensorDesc({16, 16}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto beta = ScalarDesc(1.0f);
    auto alpha = ScalarDesc(1.0f);
    int8_t cubeMathType = ALLOW_FP32_DOWN_PRECISION;

    auto ut = OP_API_UT(aclnnAddmm, INPUT(self, mat1, mat2, beta, alpha), OUTPUT(out), cubeMathType);

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// mat1, mat2不满足相乘条件
TEST_F(l2_addmm_test, case_cannot_matmul)
{
    auto self = TensorDesc({16, 16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat1 = TensorDesc({16, 17}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat2 = TensorDesc({16, 16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out = TensorDesc({16, 16}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto beta = ScalarDesc(1.0f);
    auto alpha = ScalarDesc(1.0f);
    int8_t cubeMathType = ALLOW_FP32_DOWN_PRECISION;

    auto ut = OP_API_UT(aclnnAddmm, INPUT(self, mat1, mat2, beta, alpha), OUTPUT(out), cubeMathType);

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// bias need cast to float32
TEST_F(l2_addmm_test, case_input_float_bias_float16_matmul)
{
    auto self = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat1 = TensorDesc({16, 16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat2 = TensorDesc({16, 16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out = TensorDesc({16, 16}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto beta = ScalarDesc(1.0f);
    auto alpha = ScalarDesc(1.0f);
    int8_t cubeMathType = ALLOW_FP32_DOWN_PRECISION;

    auto ut = OP_API_UT(aclnnAddmm, INPUT(self, mat1, mat2, beta, alpha), OUTPUT(out), cubeMathType);

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// self 和 mat1@mat2无法broadcast
TEST_F(l2_addmm_test, case_cannot_axpy)
{
    auto self = TensorDesc({17, 17}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat1 = TensorDesc({16, 16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat2 = TensorDesc({16, 16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out = TensorDesc({16, 16}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto beta = ScalarDesc(1.0f);
    auto alpha = ScalarDesc(1.0f);
    int8_t cubeMathType = ALLOW_FP32_DOWN_PRECISION;

    auto ut = OP_API_UT(aclnnAddmm, INPUT(self, mat1, mat2, beta, alpha), OUTPUT(out), cubeMathType);

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 非连续
TEST_F(l2_addmm_test, case_not_contiguous)
{
    auto self = TensorDesc({16, 16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat1 = TensorDesc({16, 32}, ACL_FLOAT, ACL_FORMAT_ND, {1, 16}, 0, {32, 16}).ValueRange(0, 2);
    auto mat2 = TensorDesc({32, 16}, ACL_FLOAT, ACL_FORMAT_ND, {1, 32}, 0, {16, 32}).ValueRange(0, 2);
    auto out = TensorDesc({16, 16}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.005, 0.005);
    auto beta = ScalarDesc(1.0f);
    auto alpha = ScalarDesc(1.0f);
    int8_t cubeMathType = ALLOW_FP32_DOWN_PRECISION;

    auto ut = OP_API_UT(aclnnAddmm, INPUT(self, mat1, mat2, beta, alpha), OUTPUT(out), cubeMathType);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 空tensor1
TEST_F(l2_addmm_test, case_empty_tensor_1)
{
    auto self = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat1 = TensorDesc({4, 0}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat2 = TensorDesc({0, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto beta = ScalarDesc(1.0f);
    auto alpha = ScalarDesc(1.0f);
    int8_t cubeMathType = ALLOW_FP32_DOWN_PRECISION;

    auto ut = OP_API_UT(aclnnAddmm, INPUT(self, mat1, mat2, beta, alpha), OUTPUT(out), cubeMathType);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// inplace FP16 4*4
TEST_F(l2_addmm_test, case_inplace_fp16_4mm4)
{
    auto self = TensorDesc({4, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat1 = TensorDesc({4, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat2 = TensorDesc({4, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out = TensorDesc({4, 4}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.005, 0.005);
    auto beta = ScalarDesc(1.0f);
    auto alpha = ScalarDesc(1.0f);
    int8_t cubeMathType = ALLOW_FP32_DOWN_PRECISION;

    auto ut = OP_API_UT(aclnnInplaceAddmm, INPUT(self, mat1, mat2, beta, alpha), OUTPUT(), cubeMathType);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// inplace FP16 515*512
TEST_F(l2_addmm_test, ascend910_95_inplace_fp16_512mm512_01)
{
    auto self = TensorDesc({512, 512}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat1 = TensorDesc({512, 512}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat2 = TensorDesc({512, 512}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out = TensorDesc({512, 512}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.005, 0.005);
    auto beta = ScalarDesc(1.0f);
    auto alpha = ScalarDesc(1.0f);
    int8_t cubeMathType = ALLOW_FP32_DOWN_PRECISION;

    auto ut = OP_API_UT(aclnnInplaceAddmm, INPUT(self, mat1, mat2, beta, alpha), OUTPUT(), cubeMathType);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    // EXPECT_EQ(aclRet, ACL_SUCCESS);  // no install 910_95 kernel
}

// inplace FP16 515*512
TEST_F(l2_addmm_test, ascend910_95_inplace_fp16_512mm512_02)
{
    auto self = TensorDesc({512, 512}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat1 = TensorDesc({512, 512}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat2 = TensorDesc({512, 512}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out = TensorDesc({512, 512}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.005, 0.005);
    auto beta = ScalarDesc(1.0f);
    auto alpha = ScalarDesc(1.0f);
    int8_t cubeMathType = KEEP_DTYPE;

    auto ut = OP_API_UT(aclnnInplaceAddmm, INPUT(self, mat1, mat2, beta, alpha), OUTPUT(), cubeMathType);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    // EXPECT_EQ(aclRet, ACL_SUCCESS);  // no install 910_95 kernel
}

// inplace FP16 515*512 c和y类型FP32  a和b类型FP16  ALLOW_FP32_DOWN_PRECISION
TEST_F(l2_addmm_test, ascend910_95_inplace_fp16_512mm512_03)
{
    auto self = TensorDesc({512, 512}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat1 = TensorDesc({512, 512}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat2 = TensorDesc({512, 512}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out = TensorDesc({512, 512}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.005, 0.005);
    auto beta = ScalarDesc(1.0f);
    auto alpha = ScalarDesc(1.0f);
    int8_t cubeMathType = ALLOW_FP32_DOWN_PRECISION;

    auto ut = OP_API_UT(aclnnInplaceAddmm, INPUT(self, mat1, mat2, beta, alpha), OUTPUT(), cubeMathType);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    // EXPECT_EQ(aclRet, ACL_SUCCESS);  // no install 910_95 kernel
}

// inplace BF16
TEST_F(l2_addmm_test, ascend910_95_inplace_bf16_512mm512_01)
{
    auto self = TensorDesc({512, 512}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat1 = TensorDesc({512, 512}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat2 = TensorDesc({512, 512}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out = TensorDesc({512, 512}, ACL_BF16, ACL_FORMAT_ND).Precision(0.005, 0.005);
    auto beta = ScalarDesc(1.0f);
    auto alpha = ScalarDesc(1.0f);
    int8_t cubeMathType = KEEP_DTYPE;

    auto ut = OP_API_UT(aclnnInplaceAddmm, INPUT(self, mat1, mat2, beta, alpha), OUTPUT(), cubeMathType);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    // EXPECT_EQ(aclRet, ACL_SUCCESS);  // no install 910_95 kernel
}

// out == fp32 && BF16 && ALLOW_FP32_DOWN_PRECISION => invalid
TEST_F(l2_addmm_test, ascend910_95_inplace_bf16_512mm512_02)
{
    auto self = TensorDesc({512, 512}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat1 = TensorDesc({512, 512}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat2 = TensorDesc({512, 512}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out = TensorDesc({512, 512}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.005, 0.005);
    auto beta = ScalarDesc(1.0f);
    auto alpha = ScalarDesc(1.0f);
    int8_t cubeMathType = ALLOW_FP32_DOWN_PRECISION;

    auto ut = OP_API_UT(aclnnInplaceAddmm, INPUT(self, mat1, mat2, beta, alpha), OUTPUT(), cubeMathType);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    // EXPECT_EQ(aclRet, ACL_SUCCESS);  // no install 910_95 kernel
}

// out == fp32 && BF16 && KEEP_DTYPE => success
TEST_F(l2_addmm_test, ascend910_95_inplace_bf16_512mm512_03)
{
    auto self = TensorDesc({512, 512}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat1 = TensorDesc({512, 512}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat2 = TensorDesc({512, 512}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out = TensorDesc({512, 512}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.005, 0.005);
    auto beta = ScalarDesc(1.0f);
    auto alpha = ScalarDesc(1.0f);
    int8_t cubeMathType = KEEP_DTYPE;

    auto ut = OP_API_UT(aclnnInplaceAddmm, INPUT(self, mat1, mat2, beta, alpha), OUTPUT(), cubeMathType);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    // EXPECT_EQ(aclRet, ACL_SUCCESS);  // no install 910_95 kernel
}

// inplace fp32
TEST_F(l2_addmm_test, ascend910_95_inplace_fp32_512mm512_allow_fp32_down_precision)
{
    auto self = TensorDesc({512, 512}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat1 = TensorDesc({512, 512}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat2 = TensorDesc({512, 512}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out = TensorDesc({512, 512}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.005, 0.005);
    auto beta = ScalarDesc(1.0f);
    auto alpha = ScalarDesc(1.0f);
    int8_t cubeMathType = ALLOW_FP32_DOWN_PRECISION;

    auto ut = OP_API_UT(aclnnInplaceAddmm, INPUT(self, mat1, mat2, beta, alpha), OUTPUT(), cubeMathType);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    // EXPECT_EQ(aclRet, ACL_SUCCESS);  // no install 910_95 kernel
}

// inplace fp32
TEST_F(l2_addmm_test, ascend910_95_inplace_fp32_512mm512_keep_dtype)
{
    auto self = TensorDesc({512, 512}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat1 = TensorDesc({512, 512}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat2 = TensorDesc({512, 512}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out = TensorDesc({512, 512}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.005, 0.005);
    auto beta = ScalarDesc(1.0f);
    auto alpha = ScalarDesc(1.0f);
    int8_t cubeMathType = KEEP_DTYPE;

    auto ut = OP_API_UT(aclnnInplaceAddmm, INPUT(self, mat1, mat2, beta, alpha), OUTPUT(), cubeMathType);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    // EXPECT_EQ(aclRet, ACL_SUCCESS);  // no install 910_95 kernel
}

// 空tensor2
TEST_F(l2_addmm_test, case_empty_tensor_2)
{
    auto self = TensorDesc({3, 0}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat1 = TensorDesc({3, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat2 = TensorDesc({1, 0}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out = TensorDesc({3, 0}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto beta = ScalarDesc(1.0f);
    auto alpha = ScalarDesc(1.0f);
    int8_t cubeMathType = ALLOW_FP32_DOWN_PRECISION;

    auto ut = OP_API_UT(aclnnAddmm, INPUT(self, mat1, mat2, beta, alpha), OUTPUT(out), cubeMathType);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 校验output shape要与预期一致
TEST_F(l2_addmm_test, case_output_shape)
{
    auto beta = ScalarDesc(2.0f);
    auto alpha = ScalarDesc(2.0f);
    int8_t cubeMathType = ALLOW_FP32_DOWN_PRECISION;
    // self: 2 x 3, mat1: 2 x 5, mat2: 5 x 3, out: 2 x 3
    test_run(
        {2, 3}, ACL_FLOAT, ACL_FORMAT_ND, {-5, 5}, {2, 5}, ACL_FLOAT, ACL_FORMAT_ND, {-5, 5}, {5, 3}, ACL_FLOAT,
        ACL_FORMAT_ND, {-5, 5}, beta, alpha, {2, 3}, ACL_FLOAT, ACL_FORMAT_ND, cubeMathType);

    // self: 2 x 3, mat1: 2 x 5, mat2: 5 x 3, out: 2 x 4
    test_run_invalid(
        {2, 3}, ACL_FLOAT, ACL_FORMAT_ND, {-5, 5}, {2, 5}, ACL_FLOAT, ACL_FORMAT_ND, {-5, 5}, {5, 3}, ACL_FLOAT,
        ACL_FORMAT_ND, {-5, 5}, beta, alpha, {2, 4}, ACL_FLOAT, ACL_FORMAT_ND, cubeMathType);
}

// 空tensor混合场景
TEST_F(l2_addmm_test, case_empty_tensor_3)
{
    auto beta = ScalarDesc(2.0f);
    auto alpha = ScalarDesc(2.0f);
    int8_t cubeMathType = ALLOW_FP32_DOWN_PRECISION;
    // self: 2 x 2, mat1: 2 x 0, mat2: 0 x 2, out: 2 x 2
    test_run(
        {2, 2}, ACL_FLOAT, ACL_FORMAT_ND, {-5, 5}, {2, 0}, ACL_FLOAT, ACL_FORMAT_ND, {-5, 5}, {0, 2}, ACL_FLOAT,
        ACL_FORMAT_ND, {-5, 5}, beta, alpha, {2, 2}, ACL_FLOAT, ACL_FORMAT_ND, cubeMathType);

    // self: 1 / 1 x 1, mat1: 2 x 0, mat2: 0 x 2, out: 2 x 2
    test_run(
        {1}, ACL_FLOAT, ACL_FORMAT_ND, {-5, 5}, {2, 0}, ACL_FLOAT, ACL_FORMAT_ND, {-5, 5}, {0, 2}, ACL_FLOAT,
        ACL_FORMAT_ND, {-5, 5}, beta, alpha, {2, 2}, ACL_FLOAT, ACL_FORMAT_ND, cubeMathType);
    test_run(
        {1, 1}, ACL_FLOAT, ACL_FORMAT_ND, {-5, 5}, {2, 0}, ACL_FLOAT, ACL_FORMAT_ND, {-5, 5}, {0, 2}, ACL_FLOAT,
        ACL_FORMAT_ND, {-5, 5}, beta, alpha, {2, 2}, ACL_FLOAT, ACL_FORMAT_ND, cubeMathType);

    // self: 0, mat1: 2 x 0, mat2: 0 x 2, out: 2 x 2  拦截报错
    test_run_invalid(
        {0}, ACL_FLOAT, ACL_FORMAT_ND, {-5, 5}, {2, 0}, ACL_FLOAT, ACL_FORMAT_ND, {-5, 5}, {0, 2}, ACL_FLOAT,
        ACL_FORMAT_ND, {-5, 5}, beta, alpha, {2, 2}, ACL_FLOAT, ACL_FORMAT_ND, cubeMathType);

    // self: 2 x 0, mat1: 2 x 0, mat2: 0 x 0, out: 0 x 0
    test_run(
        {2, 0}, ACL_FLOAT, ACL_FORMAT_ND, {-5, 5}, {2, 0}, ACL_FLOAT, ACL_FORMAT_ND, {-5, 5}, {0, 0}, ACL_FLOAT,
        ACL_FORMAT_ND, {-5, 5}, beta, alpha, {0, 0}, ACL_FLOAT, ACL_FORMAT_ND, cubeMathType);
    // self: 0 / 1 / 1 x 1, mat1: 2 x 0, mat2: 0 x 0, out: 0 x 0
    test_run(
        {0}, ACL_FLOAT, ACL_FORMAT_ND, {-5, 5}, {2, 0}, ACL_FLOAT, ACL_FORMAT_ND, {-5, 5}, {0, 0}, ACL_FLOAT,
        ACL_FORMAT_ND, {-5, 5}, beta, alpha, {0, 0}, ACL_FLOAT, ACL_FORMAT_ND, cubeMathType);
    test_run(
        {1}, ACL_FLOAT, ACL_FORMAT_ND, {-5, 5}, {2, 0}, ACL_FLOAT, ACL_FORMAT_ND, {-5, 5}, {0, 0}, ACL_FLOAT,
        ACL_FORMAT_ND, {-5, 5}, beta, alpha, {0, 0}, ACL_FLOAT, ACL_FORMAT_ND, cubeMathType);
    test_run(
        {1, 1}, ACL_FLOAT, ACL_FORMAT_ND, {-5, 5}, {2, 0}, ACL_FLOAT, ACL_FORMAT_ND, {-5, 5}, {0, 0}, ACL_FLOAT,
        ACL_FORMAT_ND, {-5, 5}, beta, alpha, {0, 0}, ACL_FLOAT, ACL_FORMAT_ND, cubeMathType);
}
/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <array>
#include <vector>
#include "gtest/gtest.h"

#include "../../../op_host/op_api/aclnn_quant_matmul.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace std;

class l2_quant_matmul_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "l2_quant_matmul_test SetUp" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "l2_quant_matmul_test TearDown" << endl;
    }
};

TEST_F(l2_quant_matmul_test, ascend910B2_test_normal_input)
{
    // 使用**Desc描述host api输入输出
    auto x1_desc = TensorDesc({16, 32}, ACL_INT8, ACL_FORMAT_ND);
    auto x2_desc = TensorDesc({32, 16}, ACL_INT8, ACL_FORMAT_ND);
    auto bias_desc = TensorDesc({16, 16}, ACL_INT32, ACL_FORMAT_ND);
    float deqScale = 1.0;
    auto out_desc = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnQuantMatmul, INPUT(x1_desc, x2_desc, bias_desc, deqScale), OUTPUT(out_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    // EXPECT_EQ(aclRet, ACLNN_SUCCESS);

    // SAMPLE: precision simulate
    // ut.TestPrecision();
}

TEST_F(l2_quant_matmul_test, ascend910B2_test_empty)
{
    // 使用**Desc描述host api输入输出
    auto x1_desc = TensorDesc({16, 0}, ACL_INT8, ACL_FORMAT_ND);
    auto x2_desc = TensorDesc({0, 16}, ACL_INT8, ACL_FORMAT_ND);
    auto bias_desc = TensorDesc({16}, ACL_INT32, ACL_FORMAT_ND);
    float deqScale = 1.0;
    auto out_desc = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnQuantMatmul, INPUT(x1_desc, x2_desc, bias_desc, deqScale), OUTPUT(out_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    // EXPECT_EQ(aclRet, ACLNN_ERR_INNER_NULLPTR);

    // SAMPLE: precision simulate
    // ut.TestPrecision();
}

TEST_F(l2_quant_matmul_test, ascend910B2_test_normal_input_perchannel)
{
    // 使用**Desc描述host api输入输出，并转换为aclTensor指针
    auto x1_desc = TensorDesc({5, 2}, ACL_INT8, ACL_FORMAT_ND);
    auto x2_desc = TensorDesc({2, 3}, ACL_INT8, ACL_FORMAT_ND);
    auto bias_desc = TensorDesc({3}, ACL_INT32, ACL_FORMAT_ND);
    auto deqScale_desc = TensorDesc({16}, ACL_UINT64, ACL_FORMAT_ND);
    bool adjX1 = false;
    bool adjX2 = false;
    auto out_desc = TensorDesc({5, 3}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnQuantMatmulV2, INPUT(x1_desc, x2_desc, bias_desc, deqScale_desc, adjX1, adjX2), OUTPUT(out_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    // EXPECT_EQ(aclRet, ACLNN_SUCCESS);

    // SAMPLE: precision simulate
    // ut.TestPrecision();
}

TEST_F(l2_quant_matmul_test, ascend910B2_test_abnormal_input_x1)
{
    // 使用**Desc描述host api输入输出，并转换为aclTensor指针
    auto x1_desc = TensorDesc({1, 1, 5, 2}, ACL_INT8, ACL_FORMAT_ND);
    auto x2_desc = TensorDesc({2, 3}, ACL_INT8, ACL_FORMAT_ND);
    auto bias_desc = TensorDesc({3}, ACL_INT32, ACL_FORMAT_ND);
    auto deqScale_desc = TensorDesc({16}, ACL_UINT64, ACL_FORMAT_ND);
    bool adjX1 = false;
    bool adjX2 = false;
    auto out_desc = TensorDesc({5, 3}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnQuantMatmulV2, INPUT(x1_desc, x2_desc, bias_desc, deqScale_desc, adjX1, adjX2), OUTPUT(out_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);

    // SAMPLE: precision simulate
    // ut.TestPrecision();
}

TEST_F(l2_quant_matmul_test, ascend910B2_test_abnormal_input_x2)
{
    // 使用**Desc描述host api输入输出，并转换为aclTensor指针
    auto x1_desc = TensorDesc({5, 2}, ACL_INT8, ACL_FORMAT_ND);
    auto x2_desc = TensorDesc({3}, ACL_INT8, ACL_FORMAT_ND);
    auto bias_desc = TensorDesc({3}, ACL_INT32, ACL_FORMAT_ND);
    auto deqScale_desc = TensorDesc({16}, ACL_UINT64, ACL_FORMAT_ND);
    bool adjX1 = false;
    bool adjX2 = false;
    auto out_desc = TensorDesc({5, 3}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnQuantMatmulV2, INPUT(x1_desc, x2_desc, bias_desc, deqScale_desc, adjX1, adjX2), OUTPUT(out_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);

    // SAMPLE: precision simulate
    // ut.TestPrecision();
}

TEST_F(l2_quant_matmul_test, ascend910B2_test_abnormal_input_bias)
{
    // 使用**Desc描述host api输入输出，并转换为aclTensor指针
    auto x1_desc = TensorDesc({5, 2}, ACL_INT8, ACL_FORMAT_ND);
    auto x2_desc = TensorDesc({2, 3}, ACL_INT8, ACL_FORMAT_ND);
    auto bias_desc = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND);
    auto deqScale_desc = TensorDesc({16}, ACL_UINT64, ACL_FORMAT_ND);
    bool adjX1 = false;
    bool adjX2 = false;
    auto out_desc = TensorDesc({5, 3}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnQuantMatmulV2, INPUT(x1_desc, x2_desc, bias_desc, deqScale_desc, adjX1, adjX2), OUTPUT(out_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);

    // SAMPLE: precision simulate
    // ut.TestPrecision();
}

TEST_F(l2_quant_matmul_test, ascend910B2_test_abnormal_input_deqScale)
{
    // 使用**Desc描述host api输入输出，并转换为aclTensor指针
    auto x1_desc = TensorDesc({5, 2}, ACL_INT8, ACL_FORMAT_ND);
    auto x2_desc = TensorDesc({2, 3}, ACL_INT8, ACL_FORMAT_ND);
    auto bias_desc = TensorDesc({3}, ACL_INT32, ACL_FORMAT_ND);
    auto deqScale_desc = TensorDesc({15}, ACL_UINT64, ACL_FORMAT_ND);
    bool adjX1 = false;
    bool adjX2 = false;
    auto out_desc = TensorDesc({5, 3}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnQuantMatmulV2, INPUT(x1_desc, x2_desc, bias_desc, deqScale_desc, adjX1, adjX2), OUTPUT(out_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);

    // SAMPLE: precision simulate
    // ut.TestPrecision();
}

TEST_F(l2_quant_matmul_test, ascend910B2_test_abnormal_input_format)
{
    // 使用**Desc描述host api输入输出
    auto x1_desc = TensorDesc({16, 32}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ);
    auto x2_desc = TensorDesc({32, 16}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ);
    auto bias_desc = TensorDesc({16}, ACL_INT32, ACL_FORMAT_FRACTAL_NZ);
    float deqScale = 1.0;
    auto out_desc = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_FRACTAL_NZ);

    auto ut = OP_API_UT(aclnnQuantMatmul, INPUT(x1_desc, x2_desc, bias_desc, deqScale), OUTPUT(out_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    // EXPECT_EQ(aclRet, ACLNN_SUCCESS);

    // SAMPLE: precision simulate
    // ut.TestPrecision();
}

TEST_F(l2_quant_matmul_test, ascend910B2_test_abnormal_input_dim)
{
    // 使用**Desc描述host api输入输出
    auto x1_desc = TensorDesc({16}, ACL_INT8, ACL_FORMAT_ND);
    auto x2_desc = TensorDesc({32, 16}, ACL_INT8, ACL_FORMAT_ND);
    auto bias_desc = TensorDesc({16}, ACL_INT32, ACL_FORMAT_ND);
    float deqScale = 1.0;
    auto out_desc = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnQuantMatmul, INPUT(x1_desc, x2_desc, bias_desc, deqScale), OUTPUT(out_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    // EXPECT_EQ(aclRet, ACLNN_SUCCESS);

    // SAMPLE: precision simulate
    // ut.TestPrecision();
}

TEST_F(l2_quant_matmul_test, ascend910B2_test_abnormal_input_sameDim)
{
    // 使用**Desc描述host api输入输出
    auto x1_desc = TensorDesc({16, 32}, ACL_INT8, ACL_FORMAT_ND);
    auto x2_desc = TensorDesc({1, 32, 16}, ACL_INT8, ACL_FORMAT_ND);
    auto bias_desc = TensorDesc({16}, ACL_INT32, ACL_FORMAT_ND);
    float deqScale = 1.0;
    auto out_desc = TensorDesc({1, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnQuantMatmul, INPUT(x1_desc, x2_desc, bias_desc, deqScale), OUTPUT(out_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    // EXPECT_EQ(aclRet, ACLNN_SUCCESS);

    // SAMPLE: precision simulate
    // ut.TestPrecision();
}

TEST_F(l2_quant_matmul_test, ascend910B2_test_abnormal_input_sameBatch)
{
    // 使用**Desc描述host api输入输出
    auto x1_desc = TensorDesc({2, 16, 32}, ACL_INT8, ACL_FORMAT_ND);
    auto x2_desc = TensorDesc({1, 32, 16}, ACL_INT8, ACL_FORMAT_ND);
    auto bias_desc = TensorDesc({16}, ACL_INT32, ACL_FORMAT_ND);
    float deqScale = 1.0;
    auto out_desc = TensorDesc({1, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnQuantMatmul, INPUT(x1_desc, x2_desc, bias_desc, deqScale), OUTPUT(out_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    // EXPECT_EQ(aclRet, ACLNN_SUCCESS);

    // SAMPLE: precision simulate
    // ut.TestPrecision();
}

TEST_F(l2_quant_matmul_test, ascend910B2_test_abnormal_input_sameK)
{
    // 使用**Desc描述host api输入输出
    auto x1_desc = TensorDesc({16, 32}, ACL_INT8, ACL_FORMAT_ND);
    auto x2_desc = TensorDesc({31, 16}, ACL_INT8, ACL_FORMAT_ND);
    auto bias_desc = TensorDesc({16}, ACL_INT32, ACL_FORMAT_ND);
    float deqScale = 1.0;
    auto out_desc = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnQuantMatmul, INPUT(x1_desc, x2_desc, bias_desc, deqScale), OUTPUT(out_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    // EXPECT_EQ(aclRet, ACLNN_SUCCESS);

    // SAMPLE: precision simulate
    // ut.TestPrecision();
}

TEST_F(l2_quant_matmul_test, ascend910B2_test_abnormal_input_biasN)
{
    // 使用**Desc描述host api输入输出
    auto x1_desc = TensorDesc({16, 32}, ACL_INT8, ACL_FORMAT_ND);
    auto x2_desc = TensorDesc({31, 16}, ACL_INT8, ACL_FORMAT_ND);
    auto bias_desc = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND);
    float deqScale = 1.0;
    auto out_desc = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnQuantMatmul, INPUT(x1_desc, x2_desc, bias_desc, deqScale), OUTPUT(out_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    // EXPECT_EQ(aclRet, ACLNN_SUCCESS);

    // SAMPLE: precision simulate
    // ut.TestPrecision();
}

// checkout out shape
TEST_F(l2_quant_matmul_test, ascend910B2_test_abnormal_inferedOutDimNum_outDimNum)
{
    // inferedOutDimNum != outDimNum
    TensorDesc x1_desc = TensorDesc({1, 512}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_desc = TensorDesc({512, 128}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc bias_desc = TensorDesc({128}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-1, 1);
    float deqScale = 1.0;
    TensorDesc out_desc = TensorDesc({1, 1, 1, 128}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto ut = OP_API_UT(aclnnQuantMatmul, INPUT(x1_desc, x2_desc, bias_desc, deqScale), OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_quant_matmul_test, ascend910B2_test_abnormal_outMDim_x1MDim)
{
    // outMDim != x1MDim
    TensorDesc x1_desc = TensorDesc({1, 512}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_desc = TensorDesc({512, 128}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc bias_desc = TensorDesc({128}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-1, 1);
    float deqScale = 1.0;
    TensorDesc out_desc = TensorDesc({32, 128}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto ut = OP_API_UT(aclnnQuantMatmul, INPUT(x1_desc, x2_desc, bias_desc, deqScale), OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_quant_matmul_test, ascend910B2_test_abnormal_outNDim_x2NDim)
{
    // outNDim != x2NDim
    TensorDesc x1_desc = TensorDesc({1, 512}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_desc = TensorDesc({512, 128}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc bias_desc = TensorDesc({128}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-1, 1);
    float deqScale = 1.0;
    TensorDesc out_desc = TensorDesc({1, 32}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto ut = OP_API_UT(aclnnQuantMatmul, INPUT(x1_desc, x2_desc, bias_desc, deqScale), OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_quant_matmul_test, ascend910B2_test_abnormal_outBatchDim)
{
    // infer out batch != out batch
    TensorDesc x1_desc = TensorDesc({1, 512}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_desc = TensorDesc({10, 512, 128}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc bias_desc = TensorDesc({128}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-1, 1);
    float deqScale = 1.0;
    TensorDesc out_desc = TensorDesc({1, 1, 128}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto ut = OP_API_UT(aclnnQuantMatmul, INPUT(x1_desc, x2_desc, bias_desc, deqScale), OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

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

#include "../../../op_host/op_api/aclnn_batchmatmul_quant.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace std;

class l2_batchmatmul_quant_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "l2_batchmatmul_quant_test SetUp" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "l2_batchmatmul_quant_test TearDown" << endl;
    }
};
// bmm quantParamN
TEST_F(l2_batchmatmul_quant_test, ascend910B2_test_bmm_quantParamN)
{
    auto x1_desc = TensorDesc({1, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto x2_desc = TensorDesc({1, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto quantParam_desc = TensorDesc({16}, ACL_UINT64, ACL_FORMAT_NHWC);
    auto bias_desc = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
    bool adjX1 = false;
    bool adjX2 = false;
    auto out_desc = TensorDesc({1, 16, 16}, ACL_INT8, ACL_FORMAT_ND);
    auto ut = OP_API_UT(
        aclnnBatchMatmulQuant, INPUT(x1_desc, x2_desc, quantParam_desc, bias_desc, adjX1, adjX2), OUTPUT(out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}
// bmm tansposex1
TEST_F(l2_batchmatmul_quant_test, ascend910B2_test_bmm_transposex1)
{
    auto x1_desc = TensorDesc({1, 16, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto x2_desc = TensorDesc({1, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto quantParam_desc = TensorDesc({16}, ACL_UINT64, ACL_FORMAT_NHWC);
    auto bias_desc = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
    bool adjX1 = true;
    bool adjX2 = false;
    auto out_desc = TensorDesc({1, 128, 16}, ACL_INT8, ACL_FORMAT_ND);
    auto ut = OP_API_UT(
        aclnnBatchMatmulQuant, INPUT(x1_desc, x2_desc, quantParam_desc, bias_desc, adjX1, adjX2), OUTPUT(out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}
// bmm transposex2
TEST_F(l2_batchmatmul_quant_test, ascend910B2_test_bmm_transposex2)
{
    auto x1_desc = TensorDesc({1, 128, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto x2_desc = TensorDesc({1, 8, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto quantParam_desc = TensorDesc({8}, ACL_UINT64, ACL_FORMAT_NHWC);
    auto bias_desc = TensorDesc({8}, ACL_FLOAT16, ACL_FORMAT_ND);
    bool adjX1 = false;
    bool adjX2 = true;
    auto out_desc = TensorDesc({1, 128, 8}, ACL_INT8, ACL_FORMAT_ND);
    auto ut = OP_API_UT(
        aclnnBatchMatmulQuant, INPUT(x1_desc, x2_desc, quantParam_desc, bias_desc, adjX1, adjX2), OUTPUT(out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}
// quant quantParamN
TEST_F(l2_batchmatmul_quant_test, ascend910B2_test_mm_quantParamN)
{
    auto x1_desc = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto x2_desc = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto quantParam_desc = TensorDesc({16}, ACL_UINT64, ACL_FORMAT_NHWC);
    auto bias_desc = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
    bool adjX1 = false;
    bool adjX2 = false;
    auto out_desc = TensorDesc({16, 16}, ACL_INT8, ACL_FORMAT_ND);
    auto ut = OP_API_UT(
        aclnnBatchMatmulQuant, INPUT(x1_desc, x2_desc, quantParam_desc, bias_desc, adjX1, adjX2), OUTPUT(out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}
// quant transposex1
TEST_F(l2_batchmatmul_quant_test, ascend910B2_test_mm_transposex1)
{
    auto x1_desc = TensorDesc({16, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto x2_desc = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto quantParam_desc = TensorDesc({16}, ACL_UINT64, ACL_FORMAT_NHWC);
    auto bias_desc = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
    bool adjX1 = true;
    bool adjX2 = false;
    auto out_desc = TensorDesc({128, 16}, ACL_INT8, ACL_FORMAT_ND);
    auto ut = OP_API_UT(
        aclnnBatchMatmulQuant, INPUT(x1_desc, x2_desc, quantParam_desc, bias_desc, adjX1, adjX2), OUTPUT(out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}
// quant transposex2
TEST_F(l2_batchmatmul_quant_test, ascend910B2_test_mm_transposex2)
{
    auto x1_desc = TensorDesc({128, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto x2_desc = TensorDesc({8, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto quantParam_desc = TensorDesc({8}, ACL_UINT64, ACL_FORMAT_NHWC);
    auto bias_desc = TensorDesc({8}, ACL_FLOAT16, ACL_FORMAT_ND);
    bool adjX1 = false;
    bool adjX2 = true;
    auto out_desc = TensorDesc({128, 8}, ACL_INT8, ACL_FORMAT_ND);
    auto ut = OP_API_UT(
        aclnnBatchMatmulQuant, INPUT(x1_desc, x2_desc, quantParam_desc, bias_desc, adjX1, adjX2), OUTPUT(out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}
// quant wrongbias
TEST_F(l2_batchmatmul_quant_test, ascend910B2_test_mm_wrongbias)
{
    auto x1_desc = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto x2_desc = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto quantParam_desc = TensorDesc({16}, ACL_UINT64, ACL_FORMAT_NHWC);
    auto bias_desc = TensorDesc({1, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    bool adjX1 = false;
    bool adjX2 = false;
    auto out_desc = TensorDesc({16, 16}, ACL_INT8, ACL_FORMAT_ND);
    auto ut = OP_API_UT(
        aclnnBatchMatmulQuant, INPUT(x1_desc, x2_desc, quantParam_desc, bias_desc, adjX1, adjX2), OUTPUT(out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}
// bmm wrongbias
TEST_F(l2_batchmatmul_quant_test, ascend910B2_test_bmm_wrongbias)
{
    auto x1_desc = TensorDesc({1, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto x2_desc = TensorDesc({1, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto quantParam_desc = TensorDesc({16}, ACL_UINT64, ACL_FORMAT_NHWC);
    auto bias_desc = TensorDesc({1, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    bool adjX1 = false;
    bool adjX2 = false;
    auto out_desc = TensorDesc({1, 16, 16}, ACL_INT8, ACL_FORMAT_ND);
    auto ut = OP_API_UT(
        aclnnBatchMatmulQuant, INPUT(x1_desc, x2_desc, quantParam_desc, bias_desc, adjX1, adjX2), OUTPUT(out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}
// bmm quantParam strange dim
TEST_F(l2_batchmatmul_quant_test, ascend910B2_test_bmm_wrongquantParamdim)
{
    auto x1_desc = TensorDesc({1, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto x2_desc = TensorDesc({1, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto quantParam_desc = TensorDesc({16, 1}, ACL_UINT64, ACL_FORMAT_NHWC);
    auto bias_desc = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
    bool adjX1 = false;
    bool adjX2 = false;
    auto out_desc = TensorDesc({1, 16, 16}, ACL_INT8, ACL_FORMAT_ND);
    auto ut = OP_API_UT(
        aclnnBatchMatmulQuant, INPUT(x1_desc, x2_desc, quantParam_desc, bias_desc, adjX1, adjX2), OUTPUT(out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}
// quant quantParam strange dim
TEST_F(l2_batchmatmul_quant_test, ascend910B2_test_mm_wrongquantParamdim)
{
    auto x1_desc = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto x2_desc = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto quantParam_desc = TensorDesc({16, 1}, ACL_UINT64, ACL_FORMAT_NHWC);
    auto bias_desc = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
    bool adjX1 = false;
    bool adjX2 = false;
    auto out_desc = TensorDesc({16, 16}, ACL_INT8, ACL_FORMAT_ND);
    auto ut = OP_API_UT(
        aclnnBatchMatmulQuant, INPUT(x1_desc, x2_desc, quantParam_desc, bias_desc, adjX1, adjX2), OUTPUT(out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}
// bmm quantParam strange input
TEST_F(l2_batchmatmul_quant_test, ascend910B2_test_bmm_wrongquantParaminput)
{
    auto x1_desc = TensorDesc({1, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto x2_desc = TensorDesc({1, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto quantParam_desc = TensorDesc({15}, ACL_UINT64, ACL_FORMAT_NHWC);
    auto bias_desc = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
    bool adjX1 = false;
    bool adjX2 = false;
    auto out_desc = TensorDesc({1, 16, 16}, ACL_INT8, ACL_FORMAT_ND);
    auto ut = OP_API_UT(
        aclnnBatchMatmulQuant, INPUT(x1_desc, x2_desc, quantParam_desc, bias_desc, adjX1, adjX2), OUTPUT(out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}
// quant quantParam strange input
TEST_F(l2_batchmatmul_quant_test, ascend910B2_test_mm_wrongquantParaminput)
{
    auto x1_desc = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto x2_desc = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto quantParam_desc = TensorDesc({15}, ACL_UINT64, ACL_FORMAT_NHWC);
    auto bias_desc = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
    bool adjX1 = false;
    bool adjX2 = false;
    auto out_desc = TensorDesc({16, 16}, ACL_INT8, ACL_FORMAT_ND);
    auto ut = OP_API_UT(
        aclnnBatchMatmulQuant, INPUT(x1_desc, x2_desc, quantParam_desc, bias_desc, adjX1, adjX2), OUTPUT(out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}
// bmm quantParam1
TEST_F(l2_batchmatmul_quant_test, ascend910B2_test_bmm_quantParam1)
{
    auto x1_desc = TensorDesc({1, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto x2_desc = TensorDesc({1, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto quantParam_desc = TensorDesc({1}, ACL_UINT64, ACL_FORMAT_NHWC);
    auto bias_desc = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
    bool adjX1 = false;
    bool adjX2 = false;
    auto out_desc = TensorDesc({1, 16, 16}, ACL_INT8, ACL_FORMAT_ND);
    auto ut = OP_API_UT(
        aclnnBatchMatmulQuant, INPUT(x1_desc, x2_desc, quantParam_desc, bias_desc, adjX1, adjX2), OUTPUT(out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}
// quant quantParam1
TEST_F(l2_batchmatmul_quant_test, ascend910B2_test_mm_quantParam1)
{
    auto x1_desc = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto x2_desc = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto quantParam_desc = TensorDesc({1}, ACL_UINT64, ACL_FORMAT_NHWC);
    auto bias_desc = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
    bool adjX1 = false;
    bool adjX2 = false;
    auto out_desc = TensorDesc({16, 16}, ACL_INT8, ACL_FORMAT_ND);
    auto ut = OP_API_UT(
        aclnnBatchMatmulQuant, INPUT(x1_desc, x2_desc, quantParam_desc, bias_desc, adjX1, adjX2), OUTPUT(out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}
// bmm x1 strange dim
TEST_F(l2_batchmatmul_quant_test, ascend910B2_test_bmm_x1wrongdim)
{
    auto x1_desc = TensorDesc({1, 16, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto x2_desc = TensorDesc({1, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto quantParam_desc = TensorDesc({16}, ACL_UINT64, ACL_FORMAT_NHWC);
    auto bias_desc = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
    bool adjX1 = false;
    bool adjX2 = false;
    auto out_desc = TensorDesc({1, 16, 16}, ACL_INT8, ACL_FORMAT_ND);
    auto ut = OP_API_UT(
        aclnnBatchMatmulQuant, INPUT(x1_desc, x2_desc, quantParam_desc, bias_desc, adjX1, adjX2), OUTPUT(out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}
// quant x2 strange dim
TEST_F(l2_batchmatmul_quant_test, ascend910B2_test_mm_x2wrongdim)
{
    auto x1_desc = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto x2_desc = TensorDesc({16, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto quantParam_desc = TensorDesc({16}, ACL_UINT64, ACL_FORMAT_NHWC);
    auto bias_desc = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
    bool adjX1 = false;
    bool adjX2 = false;
    auto out_desc = TensorDesc({16, 16}, ACL_INT8, ACL_FORMAT_ND);
    auto ut = OP_API_UT(
        aclnnBatchMatmulQuant, INPUT(x1_desc, x2_desc, quantParam_desc, bias_desc, adjX1, adjX2), OUTPUT(out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}
// quant bias strange input
TEST_F(l2_batchmatmul_quant_test, ascend910B2_test_mm_wrongbiasinput)
{
    auto x1_desc = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto x2_desc = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto quantParam_desc = TensorDesc({16}, ACL_UINT64, ACL_FORMAT_NHWC);
    auto bias_desc = TensorDesc({1}, ACL_FLOAT16, ACL_FORMAT_ND);
    bool adjX1 = false;
    bool adjX2 = false;
    auto out_desc = TensorDesc({16, 16}, ACL_INT8, ACL_FORMAT_ND);
    auto ut = OP_API_UT(
        aclnnBatchMatmulQuant, INPUT(x1_desc, x2_desc, quantParam_desc, bias_desc, adjX1, adjX2), OUTPUT(out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}
// bmm bias strange input
TEST_F(l2_batchmatmul_quant_test, ascend910B2_test_bmm_wrongbiasinput)
{
    auto x1_desc = TensorDesc({1, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto x2_desc = TensorDesc({1, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto quantParam_desc = TensorDesc({16}, ACL_UINT64, ACL_FORMAT_NHWC);
    auto bias_desc = TensorDesc({1}, ACL_FLOAT16, ACL_FORMAT_ND);
    bool adjX1 = false;
    bool adjX2 = false;
    auto out_desc = TensorDesc({1, 16, 16}, ACL_INT8, ACL_FORMAT_ND);
    auto ut = OP_API_UT(
        aclnnBatchMatmulQuant, INPUT(x1_desc, x2_desc, quantParam_desc, bias_desc, adjX1, adjX2), OUTPUT(out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// quant empty tensor1
TEST_F(l2_batchmatmul_quant_test, ascend910B2_test_mm_emptyTensor1)
{
    auto x1_desc = TensorDesc({0, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto x2_desc = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto quantParam_desc = TensorDesc({16}, ACL_UINT64, ACL_FORMAT_NHWC);
    auto bias_desc = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
    bool adjX1 = false;
    bool adjX2 = false;
    auto out_desc = TensorDesc({0, 16}, ACL_INT8, ACL_FORMAT_ND);
    auto ut = OP_API_UT(
        aclnnBatchMatmulQuant, INPUT(x1_desc, x2_desc, quantParam_desc, bias_desc, adjX1, adjX2), OUTPUT(out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// quant empty tensor2
TEST_F(l2_batchmatmul_quant_test, ascend910B2_test_mm_emptyTensor2)
{
    auto x1_desc = TensorDesc({16, 0}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto x2_desc = TensorDesc({0, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto quantParam_desc = TensorDesc({16}, ACL_UINT64, ACL_FORMAT_NHWC);
    auto bias_desc = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
    bool adjX1 = false;
    bool adjX2 = false;
    auto out_desc = TensorDesc({16, 16}, ACL_INT8, ACL_FORMAT_ND);
    auto ut = OP_API_UT(
        aclnnBatchMatmulQuant, INPUT(x1_desc, x2_desc, quantParam_desc, bias_desc, adjX1, adjX2), OUTPUT(out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}
// different k-axis
TEST_F(l2_batchmatmul_quant_test, ascend910B2_test_mm_diffKaxis)
{
    auto x1_desc = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto x2_desc = TensorDesc({15, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto quantParam_desc = TensorDesc({16}, ACL_UINT64, ACL_FORMAT_NHWC);
    auto bias_desc = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
    bool adjX1 = false;
    bool adjX2 = false;
    auto out_desc = TensorDesc({16, 16}, ACL_INT8, ACL_FORMAT_ND);
    auto ut = OP_API_UT(
        aclnnBatchMatmulQuant, INPUT(x1_desc, x2_desc, quantParam_desc, bias_desc, adjX1, adjX2), OUTPUT(out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_batchmatmul_quant_test, ascend910B2_invalid_min_dim)
{
    auto x1_desc = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto x2_desc = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto quantParam_desc = TensorDesc({16}, ACL_UINT64, ACL_FORMAT_NHWC);
    auto bias_desc = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
    bool adjX1 = false;
    bool adjX2 = false;
    auto out_desc = TensorDesc({16}, ACL_INT8, ACL_FORMAT_ND);
    auto ut = OP_API_UT(
        aclnnBatchMatmulQuant, INPUT(x1_desc, x2_desc, quantParam_desc, bias_desc, adjX1, adjX2), OUTPUT(out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_batchmatmul_quant_test, ascend910B2_invalid_max_dim)
{
    auto x1_desc = TensorDesc({1, 2, 3, 4, 5, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto x2_desc = TensorDesc({1, 2, 3, 4, 5, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto quantParam_desc = TensorDesc({16}, ACL_UINT64, ACL_FORMAT_NHWC);
    auto bias_desc = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
    bool adjX1 = false;
    bool adjX2 = false;
    auto out_desc = TensorDesc({1, 2, 3, 4, 5, 16, 16}, ACL_INT8, ACL_FORMAT_ND);
    auto ut = OP_API_UT(
        aclnnBatchMatmulQuant, INPUT(x1_desc, x2_desc, quantParam_desc, bias_desc, adjX1, adjX2), OUTPUT(out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_batchmatmul_quant_test, ascend910B2_valid_max_dim)
{
    auto x1_desc = TensorDesc({1, 2, 3, 4, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto x2_desc = TensorDesc({1, 2, 3, 4, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto quantParam_desc = TensorDesc({16}, ACL_UINT64, ACL_FORMAT_NHWC);
    auto bias_desc = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
    bool adjX1 = false;
    bool adjX2 = false;
    auto out_desc = TensorDesc({1, 2, 3, 4, 16, 16}, ACL_INT8, ACL_FORMAT_ND);
    auto ut = OP_API_UT(
        aclnnBatchMatmulQuant, INPUT(x1_desc, x2_desc, quantParam_desc, bias_desc, adjX1, adjX2), OUTPUT(out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

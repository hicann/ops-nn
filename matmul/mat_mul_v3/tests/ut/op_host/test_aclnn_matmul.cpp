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

#include "../../../op_host/op_api/aclnn_matmul.h"
#include "op_api/op_api_def.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace std;
using namespace op;

class l2_matmul_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << "l2_matmul_test SetUp" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "l2_matmul_test TearDown" << endl;
    }
    static void MatMulCommonTest(
        TensorDesc a_desc, TensorDesc b_desc, TensorDesc out_desc, aclnnStatus expect_status,
        int8_t cubeMathType = ALLOW_FP32_DOWN_PRECISION)
    {
        auto ut = OP_API_UT(aclnnMatmul, INPUT(a_desc, b_desc), OUTPUT(out_desc), cubeMathType);

        // SAMPLE: only test GetWorkspaceSize
        uint64_t workspace_size = 0;
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
        EXPECT_EQ(aclRet, expect_status);
        // SAMPLE: precision simulate
        if (expect_status == ACL_SUCCESS) {
            // ut.TestPrecision();  // soc version  2. 二段接口
        }
    }
};

TEST_F(l2_matmul_test, test_middle_shape_fp16)
{
    // 使用**Desc描述host api输入输出
    TensorDesc a_desc = TensorDesc({10368, 10368}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc b_desc = TensorDesc({10368, 10368}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({10368, 10368}, ACL_FLOAT16, ACL_FORMAT_ND);
    MatMulCommonTest(a_desc, b_desc, out_desc, ACL_SUCCESS);
}

TEST_F(l2_matmul_test, test_aligned_fp16_false_false)
{
    // 使用**Desc描述host api输入输出
    TensorDesc a_desc = TensorDesc({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc b_desc = TensorDesc({32, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    MatMulCommonTest(a_desc, b_desc, out_desc, ACL_SUCCESS);
}

TEST_F(l2_matmul_test, test_aligned_fp16_true_false)
{
    // 使用**Desc描述host api输入输出
    TensorDesc a_t_desc = TensorDesc({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND, {1, 16}, 0, {32, 16});
    TensorDesc b_desc = TensorDesc({32, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    cout << "-----------Testing trans_x1 = ture trans_x2 = false ----------" << endl;
    MatMulCommonTest(a_t_desc, b_desc, out_desc, ACL_SUCCESS);
}

TEST_F(l2_matmul_test, test_aligned_fp16_false_true)
{
    // 使用**Desc描述host api输入输出
    TensorDesc a_desc = TensorDesc({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc b_t_desc = TensorDesc({32, 16}, ACL_FLOAT16, ACL_FORMAT_ND, {1, 32}, 0, {16, 32});
    TensorDesc out_desc = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    cout << "-----------Testing trans_x1 = false trans_x2 = true ----------" << endl;
    MatMulCommonTest(a_desc, b_t_desc, out_desc, ACL_SUCCESS);
}

TEST_F(l2_matmul_test, test_aligned_fp16_true_true)
{
    // 使用**Desc描述host api输入输出
    TensorDesc a_t_desc = TensorDesc({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND, {1, 16}, 0, {32, 16});
    TensorDesc b_t_desc = TensorDesc({32, 16}, ACL_FLOAT16, ACL_FORMAT_ND, {1, 32}, 0, {16, 32});
    TensorDesc out_desc = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    cout << "-----------Testing trans_x1 = true trans_x2 = ture ----------" << endl;
    MatMulCommonTest(a_t_desc, b_t_desc, out_desc, ACL_SUCCESS);
}

TEST_F(l2_matmul_test, test_aligned_fp32_false_false)
{
    // 使用**Desc描述host api输入输出
    TensorDesc a_desc = TensorDesc({16, 32}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2).ValueRange(0, 2);
    TensorDesc b_desc = TensorDesc({32, 16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2).ValueRange(0, 2);
    TensorDesc out_desc = TensorDesc({16, 16}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.005, 0.005);
    cout << "-----------Testing trans_x1 = false trans_x2 = false ----------" << endl;
    MatMulCommonTest(a_desc, b_desc, out_desc, ACL_SUCCESS);
}

TEST_F(l2_matmul_test, test_aligned_fp32_true_false)
{
    // 使用**Desc描述host api输入输出
    TensorDesc a_t_desc = TensorDesc({16, 32}, ACL_FLOAT, ACL_FORMAT_ND, {1, 16}, 0, {32, 16}).ValueRange(0, 2);
    TensorDesc b_desc = TensorDesc({32, 16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2).ValueRange(0, 2);
    TensorDesc out_desc = TensorDesc({16, 16}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.005, 0.005);
    cout << "-----------Testing trans_x1 = true trans_x2 = false ----------" << endl;
    MatMulCommonTest(a_t_desc, b_desc, out_desc, ACL_SUCCESS);
}
TEST_F(l2_matmul_test, test_aligned_fp32_false_true)
{
    // 使用**Desc描述host api输入输出
    TensorDesc a_desc = TensorDesc({16, 32}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2).ValueRange(0, 2);
    TensorDesc b_t_desc = TensorDesc({32, 16}, ACL_FLOAT, ACL_FORMAT_ND, {1, 32}, 0, {16, 32}).ValueRange(0, 2);
    TensorDesc out_desc = TensorDesc({16, 16}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.05, 0.05);
    cout << "-----------Testing trans_x1 = false trans_x2 = true ----------" << endl;
    MatMulCommonTest(a_desc, b_t_desc, out_desc, ACL_SUCCESS);
}
TEST_F(l2_matmul_test, test_aligned_fp32_true_true)
{
    // 使用**Desc描述host api输入输出
    TensorDesc a_desc = TensorDesc({16, 32}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2).ValueRange(0, 2);
    TensorDesc a_t_desc = TensorDesc({16, 32}, ACL_FLOAT, ACL_FORMAT_ND, {1, 16}, 0, {32, 16}).ValueRange(0, 2);
    TensorDesc b_desc = TensorDesc({32, 16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2).ValueRange(0, 2);
    TensorDesc b_t_desc = TensorDesc({32, 16}, ACL_FLOAT, ACL_FORMAT_ND, {1, 32}, 0, {16, 32}).ValueRange(0, 2);
    TensorDesc out_desc = TensorDesc({16, 16}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.005, 0.005);
    cout << "-----------Testing trans_x1 = true trans_x2 = true ----------" << endl;
    MatMulCommonTest(a_t_desc, b_t_desc, out_desc, ACL_SUCCESS);
}

TEST_F(l2_matmul_test, test_no_aligned_fp16_false_false)
{
    // 使用**Desc描述host api输入输出
    TensorDesc a_desc = TensorDesc({8, 16}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    TensorDesc b_desc = TensorDesc({16, 8}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    TensorDesc out_desc = TensorDesc({8, 8}, ACL_FLOAT16, ACL_FORMAT_ND);
    cout << "-----------Testing trans_x1 = false trans_x2 = false ----------" << endl;
    MatMulCommonTest(a_desc, b_desc, out_desc, ACL_SUCCESS);
}

TEST_F(l2_matmul_test, test_no_aligned_fp16_true_false)
{
    // 使用**Desc描述host api输入输出
    TensorDesc a_t_desc = TensorDesc({8, 16}, ACL_FLOAT16, ACL_FORMAT_ND, {1, 8}, 0, {16, 8}).ValueRange(0, 2);
    TensorDesc b_desc = TensorDesc({16, 8}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    TensorDesc out_desc = TensorDesc({8, 8}, ACL_FLOAT16, ACL_FORMAT_ND);
    cout << "-----------Testing trans_x1 = true trans_x2 = false ----------" << endl;
    MatMulCommonTest(a_t_desc, b_desc, out_desc, ACL_SUCCESS);
}

TEST_F(l2_matmul_test, test_no_aligned_fp16_false_true)
{
    // 使用**Desc描述host api输入输出
    TensorDesc a_desc = TensorDesc({8, 16}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    TensorDesc b_t_desc = TensorDesc({16, 8}, ACL_FLOAT16, ACL_FORMAT_ND, {1, 16}, 0, {8, 16}).ValueRange(0, 2);
    TensorDesc out_desc = TensorDesc({8, 8}, ACL_FLOAT16, ACL_FORMAT_ND);
    cout << "-----------Testing trans_x1 = false trans_x2 = true ----------" << endl;
    MatMulCommonTest(a_desc, b_t_desc, out_desc, ACL_SUCCESS);
}

TEST_F(l2_matmul_test, test_no_aligned_fp16_true_true)
{
    // 使用**Desc描述host api输入输出
    TensorDesc a_t_desc = TensorDesc({8, 16}, ACL_FLOAT16, ACL_FORMAT_ND, {1, 8}, 0, {16, 8}).ValueRange(0, 2);
    TensorDesc b_t_desc = TensorDesc({16, 8}, ACL_FLOAT16, ACL_FORMAT_ND, {1, 16}, 0, {8, 16}).ValueRange(0, 2);
    TensorDesc out_desc = TensorDesc({8, 8}, ACL_FLOAT16, ACL_FORMAT_ND);
    cout << "-----------Testing trans_x1 = true trans_x2 = true ----------" << endl;
    MatMulCommonTest(a_t_desc, b_t_desc, out_desc, ACL_SUCCESS);
}

TEST_F(l2_matmul_test, test_split_k_ND)
{
    // 使用**Desc描述host api输入输出
    TensorDesc a_desc = TensorDesc({16, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc b_desc = TensorDesc({128, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    MatMulCommonTest(a_desc, b_desc, out_desc, ACL_SUCCESS);
}

TEST_F(l2_matmul_test, test_transpsoe_fail)
{
    // 使用**Desc描述host api输入输出
    TensorDesc a_desc = TensorDesc({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc b_desc = TensorDesc({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    MatMulCommonTest(a_desc, b_desc, out_desc, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_matmul_test, test_invalid_dtype)
{
    // 使用**Desc描述host api输入输出
    TensorDesc a_desc = TensorDesc({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc b_desc = TensorDesc({16, 32}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    MatMulCommonTest(a_desc, b_desc, out_desc, ACLNN_ERR_PARAM_INVALID);

    TensorDesc a2_desc = TensorDesc({16, 32}, ACL_BOOL, ACL_FORMAT_ND);
    TensorDesc b2_desc = TensorDesc({16, 32}, ACL_BOOL, ACL_FORMAT_ND);
    TensorDesc out2_desc = TensorDesc({16, 16}, ACL_BOOL, ACL_FORMAT_ND);
    MatMulCommonTest(a2_desc, b2_desc, out2_desc, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_matmul_test, test_null_ptr_input)
{
    // 使用**Desc描述host api输入输出
    TensorDesc b_desc = TensorDesc({16, 32}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    int8_t cubeMathType = ALLOW_FP32_DOWN_PRECISION;
    auto ut = OP_API_UT(
        aclnnMatmul,                        // host api第二段接口名称
        INPUT((aclTensor*)nullptr, b_desc), // host api输入
        OUTPUT(out_desc),
        cubeMathType); // host api输出
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size); // todo: check op graph
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_matmul_test, test_empty_tensor)
{
    // 使用**Desc描述host api输入输出
    TensorDesc a_desc = TensorDesc({0, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc b_desc = TensorDesc({32, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({0, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    MatMulCommonTest(a_desc, b_desc, out_desc, ACL_SUCCESS);

    TensorDesc a2_desc = TensorDesc({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc b2_desc = TensorDesc({32, 0}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc out2_desc = TensorDesc({16, 0}, ACL_FLOAT16, ACL_FORMAT_ND);
    MatMulCommonTest(a2_desc, b2_desc, out2_desc, ACL_SUCCESS);

    TensorDesc a3_desc = TensorDesc({16, 0}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc b3_desc = TensorDesc({0, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc out3_desc = TensorDesc({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    MatMulCommonTest(a3_desc, b3_desc, out3_desc, ACL_SUCCESS);
}

TEST_F(l2_matmul_test, test_bmm_empty_fp32_d5_d5)
{
    // 使用**Desc描述host api输入输出
    TensorDesc a_t_desc = TensorDesc({1, 2, 0, 6, 6}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc b_desc = TensorDesc({0, 1, 1, 6, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({0, 2, 0, 6, 1}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.005, 0.005);
    MatMulCommonTest(a_t_desc, b_desc, out_desc, ACL_SUCCESS);
    TensorDesc a_t_desc2 = TensorDesc({1, 2, 2, 6, 6}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc b_desc2 = TensorDesc({2, 1, 1, 6, 0}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc out_desc2 = TensorDesc({2, 2, 2, 6, 0}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.005, 0.005);
    MatMulCommonTest(a_t_desc2, b_desc2, out_desc2, ACL_SUCCESS);
    TensorDesc a_t_desc3 = TensorDesc({1, 2, 0, 6, 0}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc b_desc3 = TensorDesc({0, 1, 1, 0, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc out_desc3 = TensorDesc({0, 2, 0, 6, 1}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.005, 0.005);
    MatMulCommonTest(a_t_desc3, b_desc3, out_desc3, ACL_SUCCESS);
}

TEST_F(l2_matmul_test, test_invalid_shape_d1_d1)
{
    // 使用**Desc描述host api输入输出
    TensorDesc a_t_desc = TensorDesc({1}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc b_desc = TensorDesc({10}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({10}, ACL_FLOAT16, ACL_FORMAT_ND);
    MatMulCommonTest(a_t_desc, b_desc, out_desc, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_matmul_test, test_invalid_dtype_i8_fp32_fp32)
{
    // 使用**Desc描述host api输入输出
    TensorDesc a_t_desc = TensorDesc({10}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc b_desc = TensorDesc({10}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({1}, ACL_INT8, ACL_FORMAT_ND);
    MatMulCommonTest(a_t_desc, b_desc, out_desc, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_matmul_test, test_invalid_dtype_i8_fp16_fp32)
{
    // 使用**Desc描述host api输入输出
    TensorDesc a_t_desc = TensorDesc({10}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc b_desc = TensorDesc({10}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    MatMulCommonTest(a_t_desc, b_desc, out_desc, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_matmul_test, test_protome_dtype_fp32_fp16_fp32)
{
    // 使用**Desc描述host api输入输出
    TensorDesc a_t_desc = TensorDesc({10}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc b_desc = TensorDesc({10}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.005, 0.005);
    MatMulCommonTest(a_t_desc, b_desc, out_desc, ACL_SUCCESS);
}

TEST_F(l2_matmul_test, test_dot_fp16_d1_d1)
{
    // 使用**Desc描述host api输入输出
    TensorDesc a_t_desc = TensorDesc({10}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc b_desc = TensorDesc({10}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({1}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.005, 0.005);
    MatMulCommonTest(a_t_desc, b_desc, out_desc, ACL_SUCCESS);
}

//TEST_F(l2_matmul_test, test_mv_fp16_d2_d1)
//{
//    // 使用**Desc描述host api输入输出
//    TensorDesc a_t_desc = TensorDesc({10, 10}, ACL_FLOAT16, ACL_FORMAT_ND);
//    TensorDesc b_desc = TensorDesc({10}, ACL_FLOAT16, ACL_FORMAT_ND);
//    TensorDesc out_desc = TensorDesc({10}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.005, 0.005);
//    MatMulCommonTest(a_t_desc, b_desc, out_desc, ACL_SUCCESS);
//}

//TEST_F(l2_matmul_test, test_mv_fp32_d1_d2)
//{
//    // 使用**Desc描述host api输入输出
//    TensorDesc a_t_desc = TensorDesc({10}, ACL_FLOAT, ACL_FORMAT_ND);
//    TensorDesc b_desc = TensorDesc({10, 10}, ACL_FLOAT, ACL_FORMAT_ND);
//    TensorDesc out_desc = TensorDesc({10}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.005, 0.005);
//    MatMulCommonTest(a_t_desc, b_desc, out_desc, ACL_SUCCESS);
//}

TEST_F(l2_matmul_test, test_bmm_fp16_d2_d3)
{
    // 使用**Desc描述host api输入输出
    TensorDesc a_t_desc = TensorDesc({10, 10}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc b_desc = TensorDesc({2, 10, 1}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({2, 10, 1}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.005, 0.005);
    MatMulCommonTest(a_t_desc, b_desc, out_desc, ACL_SUCCESS);
}

// TEST_F(l2_matmul_test, test_bmm_fp32_d3_d2)
// {
//     // 使用**Desc描述host api输入输出
//     TensorDesc a_t_desc = TensorDesc({3, 10, 10}, ACL_FLOAT, ACL_FORMAT_ND);
//     TensorDesc b_desc = TensorDesc({10, 1}, ACL_FLOAT, ACL_FORMAT_ND);
//     TensorDesc out_desc = TensorDesc({3, 10, 1}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.005, 0.005);
//     MatMulCommonTest(a_t_desc, b_desc, out_desc, ACL_SUCCESS);
// }

TEST_F(l2_matmul_test, test_bmm_fp32_d3_d3)
{
    // 使用**Desc描述host api输入输出
    TensorDesc a_t_desc = TensorDesc({5, 6, 6}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc b_desc = TensorDesc({1, 6, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({5, 6, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    MatMulCommonTest(a_t_desc, b_desc, out_desc, ACL_SUCCESS);
}

TEST_F(l2_matmul_test, test_bmm_fp16_mn_1_d3_d3)
{
    // 使用**Desc描述host api输入输出
    TensorDesc a_desc = TensorDesc({5, 1, 6}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc b_desc = TensorDesc({5, 6, 1}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({5, 1, 1}, ACL_FLOAT16, ACL_FORMAT_ND);
    MatMulCommonTest(a_desc, b_desc, out_desc, ACL_SUCCESS);
}

TEST_F(l2_matmul_test, test_bmm_fp16_k_1_d3_d3)
{
    // 使用**Desc描述host api输入输出
    TensorDesc a_desc = TensorDesc({5, 6, 1}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc b_desc = TensorDesc({5, 1, 6}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({5, 6, 6}, ACL_FLOAT16, ACL_FORMAT_ND);
    MatMulCommonTest(a_desc, b_desc, out_desc, ACL_SUCCESS);
}

TEST_F(l2_matmul_test, test_bmm_fp32_d4_d4)
{
    // 使用**Desc描述host api输入输出
    TensorDesc a_t_desc = TensorDesc({1, 5, 6, 6}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc b_desc = TensorDesc({4, 1, 6, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({4, 5, 6, 1}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.005, 0.005);
    MatMulCommonTest(a_t_desc, b_desc, out_desc, ACL_SUCCESS);
}

TEST_F(l2_matmul_test, test_bmm_fp16_d3_d4)
{
    // 使用**Desc描述host api输入输出
    TensorDesc a_t_desc = TensorDesc({2, 6, 6}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc b_desc = TensorDesc({4, 1, 6, 1}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({4, 2, 6, 1}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.005, 0.005);
    MatMulCommonTest(a_t_desc, b_desc, out_desc, ACL_SUCCESS);
}

TEST_F(l2_matmul_test, test_bmm_fp32_d4_d3)
{
    // 使用**Desc描述host api输入输出
    TensorDesc a_t_desc = TensorDesc({2, 5, 6, 6}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc b_desc = TensorDesc({1, 6, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({2, 5, 6, 1}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.005, 0.005);
    MatMulCommonTest(a_t_desc, b_desc, out_desc, ACL_SUCCESS);
}

TEST_F(l2_matmul_test, test_bmm_fp32_fp32_d1_d4)
{
    // 使用**Desc描述host api输入输出
    TensorDesc a_t_desc = TensorDesc({6}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc b_desc = TensorDesc({4, 1, 6, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({4, 1, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.005, 0.005);
    MatMulCommonTest(a_t_desc, b_desc, out_desc, ACL_SUCCESS);
}

// TEST_F(l2_matmul_test, test_bmm_fp32_d5_d1)
// {
//     // 使用**Desc描述host api输入输出
//     TensorDesc a_t_desc = TensorDesc({4, 10, 1, 5, 6}, ACL_FLOAT, ACL_FORMAT_ND);
//     TensorDesc b_desc = TensorDesc({6}, ACL_FLOAT, ACL_FORMAT_ND);
//     TensorDesc out_desc = TensorDesc({4, 10, 1, 5, 1}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.005, 0.005);
//     MatMulCommonTest(a_t_desc, b_desc, out_desc, ACL_SUCCESS);
// }
//
// TEST_F(l2_matmul_test, test_bmm_fp32_d6_d2)
// {
//     // 使用**Desc描述host api输入输出
//     TensorDesc a_t_desc = TensorDesc({1, 4, 10, 1, 5, 6}, ACL_FLOAT, ACL_FORMAT_ND);
//     TensorDesc b_desc = TensorDesc({6, 1}, ACL_FLOAT, ACL_FORMAT_ND);
//     TensorDesc out_desc = TensorDesc({1, 4, 10, 1, 5, 1}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.005, 0.005);
//     MatMulCommonTest(a_t_desc, b_desc, out_desc, ACL_SUCCESS);
// }

TEST_F(l2_matmul_test, test_bmm_fp32_d2_d6)
{
    // 使用**Desc描述host api输入输出
    TensorDesc a_t_desc = TensorDesc({5, 6}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc b_desc = TensorDesc({1, 4, 10, 1, 6, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({1, 4, 10, 1, 5, 1}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.005, 0.005);
    MatMulCommonTest(a_t_desc, b_desc, out_desc, ACL_SUCCESS);
}

TEST_F(l2_matmul_test, test_max_dimentions_fp32_d8_d9)
{
    // 使用**Desc描述host api输入输出
    TensorDesc a_t_desc = TensorDesc({1, 1, 1, 1, 1, 1, 5, 6}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc b_desc = TensorDesc({1, 1, 1, 1, 1, 1, 1, 6, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({1, 1, 1, 1, 1, 1, 1, 5, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    MatMulCommonTest(a_t_desc, b_desc, out_desc, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_matmul_test, test_broadcast_fp32_d3_d3)
{
    // 使用**Desc描述host api输入输出
    TensorDesc a_t_desc = TensorDesc({1, 5, 6}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc b_desc = TensorDesc({2, 6, 5}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({2, 5, 5}, ACL_FLOAT, ACL_FORMAT_ND);
    MatMulCommonTest(a_t_desc, b_desc, out_desc, ACL_SUCCESS);
}

TEST_F(l2_matmul_test, test_broadcast_fp32_d4_d4)
{
    // 使用**Desc描述host api输入输出
    TensorDesc a_t_desc = TensorDesc({1, 3, 5, 6}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc b_desc = TensorDesc({3, 1, 6, 5}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({3, 3, 5, 5}, ACL_FLOAT, ACL_FORMAT_ND);
    MatMulCommonTest(a_t_desc, b_desc, out_desc, ACL_SUCCESS);
}

TEST_F(l2_matmul_test, test_invalid_broadcast_fp32_d4_d4)
{
    // 使用**Desc描述host api输入输出
    TensorDesc a_t_desc = TensorDesc({2, 3, 5, 6}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc b_desc = TensorDesc({3, 2, 6, 5}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({3, 2, 5, 5}, ACL_FLOAT, ACL_FORMAT_ND);
    MatMulCommonTest(a_t_desc, b_desc, out_desc, ACLNN_ERR_PARAM_INVALID);
}

// TEST_F(l2_matmul_test, test_invalid_broadcast_fp32_d3_d2)
// {
//     // 使用**Desc描述host api输入输出
//     TensorDesc a_t_desc = TensorDesc({3, 5, 6}, ACL_FLOAT, ACL_FORMAT_ND);
//     TensorDesc b_desc = TensorDesc({6, 5}, ACL_FLOAT, ACL_FORMAT_ND);
//     TensorDesc out_desc = TensorDesc({3, 5, 5}, ACL_FLOAT, ACL_FORMAT_ND);
//     MatMulCommonTest(a_t_desc, b_desc, out_desc, ACL_SUCCESS);
// }

// TEST_F(l2_matmul_test, test_keeptype_ng)
// {
//     // 使用**Desc描述host api输入输出
//     TensorDesc a_t_desc = TensorDesc({5, 6}, ACL_FLOAT, ACL_FORMAT_ND);
//     TensorDesc b_desc = TensorDesc({6, 5}, ACL_FLOAT, ACL_FORMAT_ND);
//     TensorDesc out_desc = TensorDesc({5, 5}, ACL_FLOAT, ACL_FORMAT_ND);
//     MatMulCommonTest(a_t_desc, b_desc, out_desc, ACLNN_ERR_PARAM_INVALID, KEEP_DTYPE);
// }

TEST_F(l2_matmul_test, test_keeptype_ok)
{
    // 使用**Desc描述host api输入输出
    TensorDesc a_t_desc = TensorDesc({5, 6}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc b_desc = TensorDesc({6, 5}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({5, 5}, ACL_FLOAT16, ACL_FORMAT_ND);
    MatMulCommonTest(a_t_desc, b_desc, out_desc, ACL_SUCCESS, KEEP_DTYPE);
}

TEST_F(l2_matmul_test, test_custom_fault)
{
    // 使用**Desc描述host api输入输出
    TensorDesc a_t_desc = TensorDesc({1, 8, 52}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc b_desc = TensorDesc({52, 3, 52}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({1, 8, 5}, ACL_FLOAT16, ACL_FORMAT_ND);
    MatMulCommonTest(a_t_desc, b_desc, out_desc, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_matmul_test, cubeMathType_0_fp16_fp16)
{
    auto tensor_1_desc = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto tensor_2_desc = TensorDesc({3, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out_tensor_desc = TensorDesc({2, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2).Precision(0.005, 0.005);
    int8_t cube_math_type = 0;
    auto ut = OP_API_UT(aclnnMatmul, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc), cube_math_type);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_matmul_test, cubeMathType_1_fp16_fp16)
{
    auto tensor_1_desc = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto tensor_2_desc = TensorDesc({3, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out_tensor_desc = TensorDesc({2, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2).Precision(0.005, 0.005);
    int8_t cube_math_type = 1;
    auto ut = OP_API_UT(aclnnMatmul, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc), cube_math_type);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_matmul_test, cubeMathType_2_fp16_fp16)
{
    auto tensor_1_desc = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto tensor_2_desc = TensorDesc({3, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out_tensor_desc = TensorDesc({2, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2).Precision(0.005, 0.005);
    int8_t cube_math_type = 2;
    auto ut = OP_API_UT(aclnnMatmul, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc), cube_math_type);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_matmul_test, cubeMathType_3_fp16_fp16)
{
    auto tensor_1_desc = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto tensor_2_desc = TensorDesc({3, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out_tensor_desc = TensorDesc({2, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2).Precision(0.005, 0.005);
    int8_t cube_math_type = 3;
    auto ut = OP_API_UT(aclnnMatmul, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc), cube_math_type);

    uint64_t workspace_size = 3;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// TEST_F(l2_matmul_test, cubeMathType_0_fp32_fp16)
// {
//     auto tensor_1_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
//     auto tensor_2_desc = TensorDesc({3, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
//     auto out_tensor_desc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2).Precision(0.005, 0.005);
//     int8_t cube_math_type = 0;
//     auto ut = OP_API_UT(aclnnMatmul, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc), cube_math_type);
//
//     uint64_t workspace_size = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
//     EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
// }

TEST_F(l2_matmul_test, cubeMathType_1_fp32_fp16)
{
    auto tensor_1_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto tensor_2_desc = TensorDesc({3, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out_tensor_desc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2).Precision(0.005, 0.005);
    int8_t cube_math_type = 1;
    auto ut = OP_API_UT(aclnnMatmul, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc), cube_math_type);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_matmul_test, cubeMathType_2_fp32_fp16)
{
    auto tensor_1_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto tensor_2_desc = TensorDesc({3, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out_tensor_desc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2).Precision(0.005, 0.005);
    int8_t cube_math_type = 2;
    auto ut = OP_API_UT(aclnnMatmul, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc), cube_math_type);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// TEST_F(l2_matmul_test, cubeMathType_3_fp32_fp16)
// {
//     auto tensor_1_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
//     auto tensor_2_desc = TensorDesc({3, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
//     auto out_tensor_desc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2).Precision(0.005, 0.005);
//     int8_t cube_math_type = 3;
//     auto ut = OP_API_UT(aclnnMatmul, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc), cube_math_type);
//
//     uint64_t workspace_size = 3;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
//     EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
// }

// TEST_F(l2_matmul_test, cubeMathType_0_fp16_fp32)
// {
//     auto tensor_1_desc = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
//     auto tensor_2_desc = TensorDesc({3, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
//     auto out_tensor_desc = TensorDesc({2, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2).Precision(0.005, 0.005);
//     int8_t cube_math_type = 0;
//     auto ut = OP_API_UT(aclnnMatmul, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc), cube_math_type);
//
//     uint64_t workspace_size = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
//     EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
// }

TEST_F(l2_matmul_test, cubeMathType_1_fp16_fp32)
{
    auto tensor_1_desc = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto tensor_2_desc = TensorDesc({3, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out_tensor_desc = TensorDesc({2, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2).Precision(0.005, 0.005);
    int8_t cube_math_type = 1;
    auto ut = OP_API_UT(aclnnMatmul, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc), cube_math_type);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_matmul_test, cubeMathType_2_fp16_fp32)
{
    auto tensor_1_desc = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto tensor_2_desc = TensorDesc({3, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out_tensor_desc = TensorDesc({2, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2).Precision(0.005, 0.005);
    int8_t cube_math_type = 2;
    auto ut = OP_API_UT(aclnnMatmul, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc), cube_math_type);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// TEST_F(l2_matmul_test, cubeMathType_3_fp16_fp32)
// {
//     auto tensor_1_desc = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
//     auto tensor_2_desc = TensorDesc({3, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
//     auto out_tensor_desc = TensorDesc({2, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2).Precision(0.005, 0.005);
//     int8_t cube_math_type = 3;
//     auto ut = OP_API_UT(aclnnMatmul, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc), cube_math_type);
//
//     uint64_t workspace_size = 3;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
//     EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
// }

// TEST_F(l2_matmul_test, cubeMathType_0_fp32_fp32)
// {
//     auto tensor_1_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
//     auto tensor_2_desc = TensorDesc({3, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
//     auto out_tensor_desc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2).Precision(0.005, 0.005);
//     int8_t cube_math_type = 0;
//     auto ut = OP_API_UT(aclnnMatmul, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc), cube_math_type);
//
//     uint64_t workspace_size = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
//     EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
// }

TEST_F(l2_matmul_test, cubeMathType_1_fp32_fp32)
{
    auto tensor_1_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto tensor_2_desc = TensorDesc({3, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out_tensor_desc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2).Precision(0.005, 0.005);
    int8_t cube_math_type = 1;
    auto ut = OP_API_UT(aclnnMatmul, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc), cube_math_type);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_matmul_test, cubeMathType_2_fp32_fp32)
{
    auto tensor_1_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto tensor_2_desc = TensorDesc({3, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out_tensor_desc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2).Precision(0.005, 0.005);
    int8_t cube_math_type = 2;
    auto ut = OP_API_UT(aclnnMatmul, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc), cube_math_type);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// TEST_F(l2_matmul_test, cubeMathType_3_fp32_fp32)
// {
//     auto tensor_1_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
//     auto tensor_2_desc = TensorDesc({3, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
//     auto out_tensor_desc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2).Precision(0.005, 0.005);
//     int8_t cube_math_type = 3;
//     auto ut = OP_API_UT(aclnnMatmul, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc), cube_math_type);
//
//     uint64_t workspace_size = 3;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
//     EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
// }
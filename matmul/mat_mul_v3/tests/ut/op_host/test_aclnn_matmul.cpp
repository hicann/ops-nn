/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
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
#include "op_api/op_api_def_nn.h"
#include "opdev/platform.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_common/op_host/util/op_const_def.h"

using namespace std;
using namespace op;

class l2_matmul_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "l2_matmul_test SetUp" << endl; }
    static void TearDownTestCase() { cout << "l2_matmul_test TearDown" << endl; }
    static void MatMulCommonTest(TensorDesc a_desc, TensorDesc b_desc, TensorDesc out_desc, aclnnStatus expect_status,
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
    auto ut = OP_API_UT(aclnnMatmul,                        // host api第二段接口名称
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

// TEST_F(l2_matmul_test, test_mv_fp16_d2_d1)
TEST_F(l2_matmul_test, test_mv_fp16_d2_d1)
{
    // 使用**Desc描述host api输入输出
    TensorDesc a_t_desc = TensorDesc({10, 10}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc b_desc = TensorDesc({10}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({10}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.005, 0.005);
    MatMulCommonTest(a_t_desc, b_desc, out_desc, ACL_SUCCESS);
}

// TEST_F(l2_matmul_test, test_mv_fp32_d1_d2)
TEST_F(l2_matmul_test, test_mv_fp32_d1_d2)
{
    // 使用**Desc描述host api输入输出
    TensorDesc a_t_desc = TensorDesc({10}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc b_desc = TensorDesc({10, 10}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({10}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.005, 0.005);
    MatMulCommonTest(a_t_desc, b_desc, out_desc, ACL_SUCCESS);
}

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

TEST_F(l2_matmul_test, test_bmm_fp32_d8_d8)
{
    TensorDesc a_t_desc = TensorDesc({1, 1, 2, 1, 2, 1, 4, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc b_desc = TensorDesc({1, 1, 1, 2, 1, 1, 8, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({1, 1, 2, 2, 2, 1, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.005, 0.005);
    MatMulCommonTest(a_t_desc, b_desc, out_desc, ACL_SUCCESS);
}

TEST_F(l2_matmul_test, test_bmm_fp32_d8_d2)
{
    TensorDesc a_t_desc = TensorDesc({1, 2, 1, 1, 2, 1, 4, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc b_desc = TensorDesc({8, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({1, 2, 1, 1, 2, 1, 4, 3}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.005, 0.005);
    MatMulCommonTest(a_t_desc, b_desc, out_desc, ACL_SUCCESS);
}

TEST_F(l2_matmul_test, test_bmm_fp32_d2_d8)
{
    TensorDesc a_t_desc = TensorDesc({4, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc b_desc = TensorDesc({1, 2, 1, 1, 2, 1, 8, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({1, 2, 1, 1, 2, 1, 4, 3}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.005, 0.005);
    MatMulCommonTest(a_t_desc, b_desc, out_desc, ACL_SUCCESS);
}

TEST_F(l2_matmul_test, test_bmm_fp32_d7_d7)
{
    TensorDesc a_t_desc = TensorDesc({1, 2, 1, 1, 2, 4, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc b_desc = TensorDesc({2, 1, 1, 2, 1, 8, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({2, 2, 1, 2, 2, 4, 3}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.005, 0.005);
    MatMulCommonTest(a_t_desc, b_desc, out_desc, ACL_SUCCESS);
}

TEST_F(l2_matmul_test, test_bmm_fp16_d8_d6_broadcast)
{
    TensorDesc a_t_desc = TensorDesc({2, 1, 1, 1, 1, 1, 4, 8}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc b_desc = TensorDesc({1, 2, 1, 1, 8, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({2, 1, 1, 2, 1, 1, 4, 3}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.005, 0.005);
    MatMulCommonTest(a_t_desc, b_desc, out_desc, ACL_SUCCESS);
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

TEST_F(l2_matmul_test, cubeMathType_4_fp32_fp32)
{
    auto tensor_1_desc = TensorDesc({256, 2048}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto tensor_2_desc = TensorDesc({2048, 1024}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out_tensor_desc = TensorDesc({256, 1024}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2).Precision(0.005, 0.005);
    int8_t cube_math_type = 4;
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

TEST_F(l2_matmul_test, dav3510_test_mm_slice_invalid)
{
    op::NpuArchManager archManager(Ops::Base::DAV_3510);
    auto tensor_1_desc = TensorDesc({5, 2, 7}, ACL_FLOAT, ACL_FORMAT_ND, {42, 7, 1}, 21, {210}).ValueRange(-2, 2);
    auto tensor_2_desc = TensorDesc({7, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out_tensor_desc = TensorDesc({15, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2).Precision(0.005, 0.005);
    int8_t cube_math_type = 0;
    auto ut = OP_API_UT(aclnnMatmul, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc), cube_math_type);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_matmul_test, dav3510_test_mm_slice_valid)
{
    op::NpuArchManager archManager(Ops::Base::DAV_3510);
    op::SetCubeCoreNum(32U);
    auto tensor_1_desc = TensorDesc({5, 3, 7}, ACL_FLOAT, ACL_FORMAT_ND, {42, 7, 1}, 21, {210}).ValueRange(-2, 2);
    auto tensor_2_desc = TensorDesc({7, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out_tensor_desc = TensorDesc({15, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2).Precision(0.005, 0.005);
    int8_t cube_math_type = 0;
    auto ut = OP_API_UT(aclnnMatmul, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc), cube_math_type);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_matmul_test, dav3510_test_mm_slice_k_large)
{
    op::NpuArchManager archManager(Ops::Base::DAV_3510);
    op::SetCubeCoreNum(32U);
    // 2D K-slice: view [128, 64], storage [128, 128], K=64, fp16 -> K*dtype=128 bytes >= 32
    auto tensor_1_desc = TensorDesc({128, 64}, ACL_FLOAT16, ACL_FORMAT_ND, {128, 1}, 0, {16384}).ValueRange(-2, 2);
    auto tensor_2_desc = TensorDesc({64, 32}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out_tensor_desc = TensorDesc({128, 32}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2).Precision(0.005, 0.005);
    int8_t cube_math_type = 0;
    auto ut = OP_API_UT(aclnnMatmul, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc), cube_math_type);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_matmul_test, dav3510_test_mm_2d_slice_k_small)
{
    op::NpuArchManager archManager(Ops::Base::DAV_3510);
    op::SetCubeCoreNum(32U);
    // 2D K-slice: view [128, 2], storage [128, 18], K=2, fp16 -> K*dtype=4 bytes < 32, should be blocked
    auto tensor_1_desc = TensorDesc({128, 2}, ACL_FLOAT16, ACL_FORMAT_ND, {18, 1}, 0, {2304}).ValueRange(-2, 2);
    auto tensor_2_desc = TensorDesc({2, 32}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out_tensor_desc = TensorDesc({128, 32}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2).Precision(0.005, 0.005);
    int8_t cube_math_type = 0;
    auto ut = OP_API_UT(aclnnMatmul, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc), cube_math_type);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_matmul_test, dav3510_test_mm_3d_slice_k_large)
{
    op::NpuArchManager archManager(Ops::Base::DAV_3510);
    op::SetCubeCoreNum(32U);
    // 3D K-slice: view [5, 3, 32], storage [5, 3, 64], K=32, fp16 -> K*dtype=64 bytes >= 32
    auto tensor_1_desc = TensorDesc({5, 3, 32}, ACL_FLOAT16, ACL_FORMAT_ND, {192, 64, 1}, 0, {960}).ValueRange(-2, 2);
    auto tensor_2_desc = TensorDesc({32, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out_tensor_desc = TensorDesc({15, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2).Precision(0.005, 0.005);
    int8_t cube_math_type = 0;
    auto ut = OP_API_UT(aclnnMatmul, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc), cube_math_type);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_matmul_test, dav3510_test_mm_3d_slice_k_small)
{
    op::NpuArchManager archManager(Ops::Base::DAV_3510);
    op::SetCubeCoreNum(32U);
    // 3D K-slice: view [5, 3, 2], storage [5, 3, 18], K=2, fp16 -> K*dtype=4 bytes < 32, should be blocked
    auto tensor_1_desc = TensorDesc({5, 3, 2}, ACL_FLOAT16, ACL_FORMAT_ND, {54, 18, 1}, 0, {270}).ValueRange(-2, 2);
    auto tensor_2_desc = TensorDesc({2, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out_tensor_desc = TensorDesc({15, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2).Precision(0.005, 0.005);
    int8_t cube_math_type = 0;
    auto ut = OP_API_UT(aclnnMatmul, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc), cube_math_type);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_matmul_test, dav3510_test_bmm_transpose_scene1)
{
    op::NpuArchManager archManager(Ops::Base::DAV_3510);
    op::SetCubeCoreNum(32U);
    auto tensor_1_desc = TensorDesc({512, 150, 150}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 2);
    // 原始shape排布k b n 转换后 b k n
    auto tensor_2_desc = TensorDesc({512, 150, 32}, ACL_BF16, ACL_FORMAT_ND, {32, 16384, 1}, 0, {2457600})
                             .ValueRange(0, 2);
    auto out_tensor_desc = TensorDesc({512, 150, 32}, ACL_BF16, ACL_FORMAT_ND)
                               .ValueRange(0, 2)
                               .Precision(0.0001, 0.0001);

    int8_t math_type = 1;
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = 0;
    auto ut_false_false = OP_API_UT(aclnnMatmul, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc),
                                    math_type);
    aclRet = ut_false_false.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_matmul_test, dav3510_test_bmm_transpose_scene2)
{
    op::NpuArchManager archManager(Ops::Base::DAV_3510);
    op::SetCubeCoreNum(32U);
    auto tensor_1_desc = TensorDesc({512, 150, 150}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 2);
    // 原始shape排布n b k [32 512 150] 转换后 b k n [512 150 32]
    auto tensor_2_desc = TensorDesc({512, 150, 32}, ACL_BF16, ACL_FORMAT_ND, {150, 1, 76800}, 0, {2457600})
                             .ValueRange(0, 2);
    auto out_tensor_desc = TensorDesc({512, 150, 32}, ACL_BF16, ACL_FORMAT_ND)
                               .ValueRange(0, 2)
                               .Precision(0.0001, 0.0001);

    int8_t math_type = 1;
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = 0;
    auto ut_false_false = OP_API_UT(aclnnMatmul, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc),
                                    math_type);
    aclRet = ut_false_false.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_matmul_test, dav3510_fp16_fp16_fp32_output)
{
    op::NpuArchManager archManager(Ops::Base::DAV_3510);
    TensorDesc a_desc = TensorDesc({128, 4096}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    TensorDesc b_desc = TensorDesc({4096, 256}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    TensorDesc out_desc = TensorDesc({128, 256}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2).Precision(0.005, 0.005);
    MatMulCommonTest(a_desc, b_desc, out_desc, ACL_SUCCESS, KEEP_DTYPE);
}

TEST_F(l2_matmul_test, dav3510_bf16_bf16_fp32_output)
{
    op::NpuArchManager archManager(Ops::Base::DAV_3510);
    TensorDesc a_desc = TensorDesc({128, 4096}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-2, 2);
    TensorDesc b_desc = TensorDesc({4096, 256}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-2, 2);
    TensorDesc out_desc = TensorDesc({128, 256}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2).Precision(0.005, 0.005);
    MatMulCommonTest(a_desc, b_desc, out_desc, ACL_SUCCESS, KEEP_DTYPE);
}
// }

// 接口整改异常用例 - dav_3510
TEST_F(l2_matmul_test, matmul_dav3510_FP32_FP32_FP16FP32_KEEP_DTYPE)
{
    op::NpuArchManager archManager(Ops::Base::DAV_3510);
    auto tensor_1_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto tensor_2_desc = TensorDesc({3, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out_tensor_desc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2).Precision(0.005, 0.005);
    int8_t cube_math_type = -1;
    auto ut = OP_API_UT(aclnnMatmul, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc), cube_math_type);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_matmul_test, matmul_dav3510_FP32_FP16_FP16FP32_KEEP_DTYPE)
{
    op::NpuArchManager archManager(Ops::Base::DAV_3510);
    auto tensor_1_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto tensor_2_desc = TensorDesc({3, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out_tensor_desc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2).Precision(0.005, 0.005);
    int8_t cube_math_type = -1;
    auto ut = OP_API_UT(aclnnMatmul, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc), cube_math_type);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_matmul_test, matmul_dav3510_FP32_BF16_FP16FP32_KEEP_DTYPE)
{
    op::NpuArchManager archManager(Ops::Base::DAV_3510);
    auto tensor_1_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto tensor_2_desc = TensorDesc({3, 4}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out_tensor_desc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2).Precision(0.005, 0.005);
    int8_t cube_math_type = -1;
    auto ut = OP_API_UT(aclnnMatmul, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc), cube_math_type);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_matmul_test, matmul_dav3510_FP16_FP16_FP16FP32_KEEP_DTYPE)
{
    op::NpuArchManager archManager(Ops::Base::DAV_3510);
    auto tensor_1_desc = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto tensor_2_desc = TensorDesc({3, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out_tensor_desc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2).Precision(0.005, 0.005);
    int8_t cube_math_type = -1;
    auto ut = OP_API_UT(aclnnMatmul, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc), cube_math_type);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_matmul_test, matmul_dav3510_FP16_BF16_FP16FP32_KEEP_DTYPE)
{
    op::NpuArchManager archManager(Ops::Base::DAV_3510);
    auto tensor_1_desc = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto tensor_2_desc = TensorDesc({3, 4}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out_tensor_desc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2).Precision(0.005, 0.005);
    int8_t cube_math_type = -1;
    auto ut = OP_API_UT(aclnnMatmul, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc), cube_math_type);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_matmul_test, matmul_dav3510_BF16_BF16_FP16FP32_KEEP_DTYPE)
{
    op::NpuArchManager archManager(Ops::Base::DAV_3510);
    auto tensor_1_desc = TensorDesc({2, 3}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto tensor_2_desc = TensorDesc({3, 4}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out_tensor_desc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2).Precision(0.005, 0.005);
    int8_t cube_math_type = -1;
    auto ut = OP_API_UT(aclnnMatmul, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc), cube_math_type);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 接口整改异常用例 - 310
TEST_F(l2_matmul_test, matmul_310_FP32_FP32_KEEP_DTYPE)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND310);
    auto tensor_1_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto tensor_2_desc = TensorDesc({3, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out_tensor_desc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2).Precision(0.005, 0.005);
    int8_t cube_math_type = 0;
    auto ut = OP_API_UT(aclnnMatmul, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc), cube_math_type);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_matmul_test, matmul_310_FP32_FP16_KEEP_DTYPE)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND310);
    auto tensor_1_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto tensor_2_desc = TensorDesc({3, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out_tensor_desc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2).Precision(0.005, 0.005);
    int8_t cube_math_type = 0;
    auto ut = OP_API_UT(aclnnMatmul, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc), cube_math_type);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_matmul_test, matmul_310_FP32_FP32_USE_HF32)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND310);
    auto tensor_1_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto tensor_2_desc = TensorDesc({3, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out_tensor_desc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2).Precision(0.005, 0.005);
    int8_t cube_math_type = 3;
    auto ut = OP_API_UT(aclnnMatmul, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc), cube_math_type);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_matmul_test, matmul_310_FP32_FP16_USE_HF32)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND310);
    auto tensor_1_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto tensor_2_desc = TensorDesc({3, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out_tensor_desc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2).Precision(0.005, 0.005);
    int8_t cube_math_type = 3;
    auto ut = OP_API_UT(aclnnMatmul, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc), cube_math_type);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_matmul_test, matmul_310_FP32_FP32_FP16FP32_KEEP_DTYPE)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND310);
    auto tensor_1_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto tensor_2_desc = TensorDesc({3, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out_tensor_desc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2).Precision(0.005, 0.005);
    int8_t cube_math_type = -1;
    auto ut = OP_API_UT(aclnnMatmul, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc), cube_math_type);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_matmul_test, test_self_nz_format_invalid)
{
    TensorDesc a_desc = TensorDesc({16, 32}, ACL_FLOAT16, ACL_FORMAT_FRACTAL_NZ);
    TensorDesc b_desc = TensorDesc({32, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    MatMulCommonTest(a_desc, b_desc, out_desc, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_matmul_test, test_out_nz_format_invalid)
{
    TensorDesc a_desc = TensorDesc({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc b_desc = TensorDesc({32, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_FRACTAL_NZ);
    MatMulCommonTest(a_desc, b_desc, out_desc, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_matmul_test, test_bmm_fp32_d5_d1)
{
    TensorDesc a_t_desc = TensorDesc({4, 10, 1, 5, 6}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc b_desc = TensorDesc({6}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({4, 10, 1, 5, 1}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.005, 0.005);
    MatMulCommonTest(a_t_desc, b_desc, out_desc, ACL_SUCCESS);
}

TEST_F(l2_matmul_test, test_weightnz_unsupported_arch)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND910);
    TensorDesc a_desc = TensorDesc({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc b_desc = TensorDesc({32, 16}, ACL_FLOAT16, ACL_FORMAT_FRACTAL_NZ, {}, 0, {2, 1, 16, 16});
    TensorDesc out_desc = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    int8_t cube_math_type = ALLOW_FP32_DOWN_PRECISION;
    auto ut = OP_API_UT(aclnnMatmulWeightNz, INPUT(a_desc, b_desc), OUTPUT(out_desc), cube_math_type);
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_matmul_test, test_weightnz_n_axis_1)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND910B);
    TensorDesc a_desc = TensorDesc({16, 1}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc b_desc = TensorDesc({1, 16}, ACL_FLOAT16, ACL_FORMAT_FRACTAL_NZ, {}, 0, {1, 1, 16, 16});
    TensorDesc out_desc = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    int8_t cube_math_type = ALLOW_FP32_DOWN_PRECISION;
    auto ut = OP_API_UT(aclnnMatmulWeightNz, INPUT(a_desc, b_desc), OUTPUT(out_desc), cube_math_type);
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_matmul_test, test_weightnz_k_mismatch)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND910B);
    TensorDesc a_desc = TensorDesc({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc b_desc = TensorDesc({48, 16}, ACL_FLOAT16, ACL_FORMAT_FRACTAL_NZ, {}, 0, {1, 3, 16, 16});
    TensorDesc out_desc = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    int8_t cube_math_type = ALLOW_FP32_DOWN_PRECISION;
    auto ut = OP_API_UT(aclnnMatmulWeightNz, INPUT(a_desc, b_desc), OUTPUT(out_desc), cube_math_type);
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_matmul_test, test_weightnz_empty_self)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND910B);
    TensorDesc a_desc = TensorDesc({0, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc b_desc = TensorDesc({32, 16}, ACL_FLOAT16, ACL_FORMAT_FRACTAL_NZ, {}, 0, {2, 1, 16, 16});
    TensorDesc out_desc = TensorDesc({0, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    int8_t cube_math_type = ALLOW_FP32_DOWN_PRECISION;
    auto ut = OP_API_UT(aclnnMatmulWeightNz, INPUT(a_desc, b_desc), OUTPUT(out_desc), cube_math_type);
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_matmul_test, test_weightnz_310_fp32_nd_keepdtype)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND310);
    TensorDesc a_desc = TensorDesc({16, 32}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc b_desc = TensorDesc({32, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({16, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    int8_t cube_math_type = KEEP_DTYPE;
    auto ut = OP_API_UT(aclnnMatmulWeightNz, INPUT(a_desc, b_desc), OUTPUT(out_desc), cube_math_type);
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_matmul_test, test_weightnz_fp32_in_fp16_out_keepdtype)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND910B);
    TensorDesc a_desc = TensorDesc({16, 32}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc b_desc = TensorDesc({32, 16}, ACL_FLOAT, ACL_FORMAT_FRACTAL_NZ, {}, 0, {2, 1, 16, 16});
    TensorDesc out_desc = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    int8_t cube_math_type = KEEP_DTYPE;
    auto ut = OP_API_UT(aclnnMatmulWeightNz, INPUT(a_desc, b_desc), OUTPUT(out_desc), cube_math_type);
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// ScSplitK + AL1FullLoad opapi 准入：910B + fp16/bf16 ND + 形状窗内走通
TEST_F(l2_matmul_test, ascend910B_sc_splitk_al1_fullload_fp16_hit)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND910B);
    TensorDesc a_desc = TensorDesc({128, 9216}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc b_desc = TensorDesc({9216, 16384}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({128, 16384}, ACL_FLOAT16, ACL_FORMAT_ND);
    MatMulCommonTest(a_desc, b_desc, out_desc, ACL_SUCCESS);
}

TEST_F(l2_matmul_test, ascend910B_sc_splitk_al1_fullload_bf16_b_trans_hit)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND910B);
    TensorDesc a_desc = TensorDesc({64, 12288}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc b_desc = TensorDesc({12288, 20480}, ACL_BF16, ACL_FORMAT_ND, {1, 12288}, 0, {20480, 12288});
    TensorDesc out_desc = TensorDesc({64, 20480}, ACL_BF16, ACL_FORMAT_ND);
    MatMulCommonTest(a_desc, b_desc, out_desc, ACL_SUCCESS);
}

#include <cstdint>
#include <dlfcn.h>

#include "gtest/gtest.h"
#include "opdev/platform.h"
#include "opdev/common_types.h"
#include "opdev/format_utils.h"
#include "opdev/op_executor.h"

#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/reshape.h"
#include "aclnn_kernels/transdata.h"
#include "aclnn_kernels/transpose.h"
#include "level0/fill.h"
#include "level0/mul.h"
#include "level0/padv3.h"
#include "op_api/op_api_def_nn.h"
#include "opdev/data_type_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"

// Keep ErrorManager declarations in global scope when including cube_util.cpp below.
#include "base/err_msg.h"

namespace matmul_util_whitebox {

namespace l0op {
__attribute__((visibility("hidden"))) const aclTensor* Cast(const aclTensor* self, op::DataType dstDtype,
                                                            aclOpExecutor* executor);
__attribute__((visibility("hidden"))) const aclTensor* Contiguous(const aclTensor* x, aclOpExecutor* executor);
__attribute__((visibility("hidden"))) const aclTensor* Fill(const aclTensor* dims, const aclTensor* value,
                                                            const aclIntArray* outShape, aclOpExecutor* executor);
__attribute__((visibility("hidden"))) const aclTensor* Mul(const aclTensor* self, const aclTensor* other,
                                                           aclOpExecutor* executor);
__attribute__((visibility("hidden"))) const aclTensor* ReFormat(const aclTensor* x, const op::Format& format,
                                                                aclOpExecutor* executor = nullptr);
__attribute__((visibility("hidden"))) const aclTensor* Reshape(const aclTensor* x, const op::Shape& shape,
                                                               aclOpExecutor* executor);
__attribute__((visibility("hidden"))) const aclTensor* Reshape(const aclTensor* x, const aclIntArray* shape,
                                                               aclOpExecutor* executor);
__attribute__((visibility("hidden"))) const aclTensor* TransData(const aclTensor* x, op::Format dstPrimaryFormat,
                                                                 int64_t groups, aclOpExecutor* executor);
__attribute__((visibility("hidden"))) const aclTensor* Transpose(const aclTensor* x, const aclIntArray* perm,
                                                                 aclOpExecutor* executor);
__attribute__((visibility("hidden"))) const aclTensor* ViewCopy(const aclTensor* x, const aclTensor* y,
                                                                aclOpExecutor* executor);
__attribute__((visibility("hidden"))) bool MmCheckHitV3Shape(const aclTensor* x1, const aclTensor* x2,
                                                             const aclTensor* bias, const bool transposeX1,
                                                             const bool transposeX2, op::Format mat2_format,
                                                             bool supportSplitK);
} // namespace l0op

#include "../../../../common/op_host/op_api/cube_util.cpp"
#include "../../../../common/op_host/op_api/matmul.cpp"
#include "../../../../common/op_host/op_api/matmul_util.cpp"

bool gCastToNullForTest = false;

namespace l0op {
const aclTensor* Cast(const aclTensor* self, op::DataType, aclOpExecutor*)
{
    return gCastToNullForTest ? nullptr : self;
}
const aclTensor* Contiguous(const aclTensor* x, aclOpExecutor*) { return x; }
const aclTensor* Fill(const aclTensor*, const aclTensor*, const aclIntArray*, aclOpExecutor*) { return nullptr; }
const aclTensor* Mul(const aclTensor*, const aclTensor*, aclOpExecutor*) { return nullptr; }
const aclTensor* ReFormat(const aclTensor* x, const op::Format& format, aclOpExecutor*)
{
    return (x != nullptr && x->GetStorageFormat() == format) ? x : nullptr;
}
const aclTensor* Reshape(const aclTensor* x, const op::Shape& shape, aclOpExecutor* executor)
{
    op::Shape out = shape;
    int64_t known = 1;
    int64_t inferIdx = -1;
    for (size_t i = 0; i < out.GetDimNum(); i++) {
        if (out.GetDim(i) == -1) {
            inferIdx = static_cast<int64_t>(i);
        } else {
            known *= out.GetDim(i);
        }
    }
    if (inferIdx >= 0) {
        out.SetDim(inferIdx, x->GetViewShape().GetShapeSize() / known);
    }
    return executor->CreateView(x, out, 0);
}
const aclTensor* Reshape(const aclTensor* x, const aclIntArray* shape, aclOpExecutor* executor)
{
    using ReshapeFn = const aclTensor* (*)(const aclTensor*, const aclIntArray*, aclOpExecutor*);
    static ReshapeFn realReshape = reinterpret_cast<ReshapeFn>(
        dlsym(RTLD_DEFAULT, "_ZN4l0op7ReshapeEPK9aclTensorPK11aclIntArrayP13aclOpExecutor"));
    if (realReshape == nullptr) {
        return nullptr;
    }
    return realReshape(x, shape, executor);
}
const aclTensor* TransData(const aclTensor*, op::Format, int64_t, aclOpExecutor*) { return nullptr; }
const aclTensor* Transpose(const aclTensor* x, const aclIntArray*, aclOpExecutor*) { return x; }
const aclTensor* ViewCopy(const aclTensor*, const aclTensor*, aclOpExecutor*) { return nullptr; }
bool MmCheckHitV3Shape(const aclTensor*, const aclTensor*, const aclTensor*, bool, bool, op::Format, bool)
{
    return false;
}
} // namespace l0op

} // namespace matmul_util_whitebox

namespace matmul_util_whitebox {
bool CallCheckAscendCScenario2(const aclTensor* x1, const aclTensor* x2, const Ops::NN::MmOpInfo& mmOpInfo,
                               bool transposeX1, bool transposeX2)
{
    return CheckAscendCScenario2(x1, x2, mmOpInfo, transposeX1, transposeX2);
}
} // namespace matmul_util_whitebox

namespace matmul_util_whitebox {
op::DataType CallPromoteOutputAndBiasDtype(op::DataType outputDtype, op::DataType biasDtype)
{
    Ops::NN::DefaultMatMulRule rule(::NpuArch::DAV_1001);
    return rule.PromoteOutputAndBiasDtype(outputDtype, biasDtype);
}
} // namespace matmul_util_whitebox

namespace matmul_util_whitebox {
bool CallCheckDtypeValid(const aclTensor* self, const aclTensor* mat2, const aclTensor* bias, const aclTensor* out,
                         int8_t cubeMathType)
{
    return CheckDtypeValid(self, mat2, bias, out, cubeMathType);
}
} // namespace matmul_util_whitebox

namespace matmul_util_whitebox {
bool CallCheckStreamKSKTiling(Ops::NN::MmOpInfo& mmOpInfo) { return CheckStreamKSKTiling(mmOpInfo); }
} // namespace matmul_util_whitebox

namespace matmul_util_whitebox {
bool CallCheckStreamKDPSKTiling(Ops::NN::MmOpInfo& mmOpInfo) { return CheckStreamKDPSKTiling(mmOpInfo); }
} // namespace matmul_util_whitebox

namespace matmul_util_whitebox {
const aclTensor* CallHandleEmptyTensor(const aclTensor* self, const aclTensor* mat2, aclOpExecutor* executor,
                                       int8_t cubeMathType)
{
    return HandleEmptyTensor(self, mat2, executor, cubeMathType);
}
} // namespace matmul_util_whitebox

namespace matmul_util_whitebox {
const aclTensor* CallSetTensorToNDFormat(const aclTensor* input) { return SetTensorToNDFormat(input); }
} // namespace matmul_util_whitebox

using namespace op;

namespace {

constexpr int8_t kCubeMathKeepDtype = 0;

aclTensor* MakeTensor(aclOpExecutor* executor, const std::initializer_list<int64_t>& dims, op::DataType dtype,
                      op::Format format)
{
    return executor->AllocTensor(op::Shape(dims), dtype, format);
}

const aclTensor* MakeViewTensor(aclOpExecutor* executor, op::DataType dtype, const op::Shape& storageShape,
                                const op::Shape& viewShape, const op::Strides& strides)
{
    auto* storage = executor->AllocTensor(storageShape, dtype, op::Format::FORMAT_ND);
    if (storage == nullptr) {
        return nullptr;
    }
    return executor->CreateView(storage, viewShape, storageShape, strides, 0);
}

bool CheckAscendCScenario2Case(
    const std::initializer_list<int64_t>& x1Dims, const std::initializer_list<int64_t>& x2Dims,
    op::DataType x1Dtype = op::DataType::DT_FLOAT16, op::DataType x2Dtype = op::DataType::DT_FLOAT16,
    op::Format selfFormat = op::Format::FORMAT_ND, op::Format mat2Format = op::Format::FORMAT_FRACTAL_NZ,
    op::Format outFormat = op::Format::FORMAT_ND, bool transposeX1 = false, bool transposeX2 = false)
{
    auto uniqueExecutor = CREATE_EXECUTOR();
    if (uniqueExecutor.get() == nullptr) {
        ADD_FAILURE() << "CREATE_EXECUTOR failed";
        return false;
    }
    auto* executor = uniqueExecutor.get();
    const aclTensor* x1 = MakeTensor(executor, x1Dims, x1Dtype, op::Format::FORMAT_ND);
    const aclTensor* x2 = MakeTensor(executor, x2Dims, x2Dtype, op::Format::FORMAT_ND);
    if (x1 == nullptr || x2 == nullptr) {
        ADD_FAILURE() << "AllocTensor failed";
        return false;
    }
    matmul_util_whitebox::Ops::NN::MmOpInfo mmOpInfo{};
    mmOpInfo.support_info.self_format = selfFormat;
    mmOpInfo.support_info.mat2_format = mat2Format;
    mmOpInfo.support_info.output_format = outFormat;
    return matmul_util_whitebox::CallCheckAscendCScenario2(x1, x2, mmOpInfo, transposeX1, transposeX2);
}

matmul_util_whitebox::Ops::NN::MmOpInfo MakeMmOpInfo(op::DataType selfDtype, op::DataType mat2Dtype,
                                                     op::DataType outputDtype)
{
    matmul_util_whitebox::Ops::NN::MmOpInfo info{};
    info.ori_info.self_dtype = selfDtype;
    info.ori_info.mat2_dtype = mat2Dtype;
    info.ori_info.output_dtype = outputDtype;
    info.support_info = info.ori_info;
    return info;
}

bool CheckDtypeValidCase(op::DataType selfDtype, op::DataType mat2Dtype, bool hasBias = false,
                         op::DataType biasDtype = op::DataType::DT_FLOAT16, bool hasOut = false,
                         op::DataType outDtype = op::DataType::DT_FLOAT16, int8_t cubeMathType = kCubeMathKeepDtype)
{
    auto uniqueExecutor = CREATE_EXECUTOR();
    if (uniqueExecutor.get() == nullptr) {
        ADD_FAILURE() << "CREATE_EXECUTOR failed";
        return false;
    }
    auto* executor = uniqueExecutor.get();
    const aclTensor* self = MakeTensor(executor, {4, 8}, selfDtype, op::Format::FORMAT_ND);
    const aclTensor* mat2 = MakeTensor(executor, {8, 4}, mat2Dtype, op::Format::FORMAT_ND);
    const aclTensor* bias = hasBias ? MakeTensor(executor, {4}, biasDtype, op::Format::FORMAT_ND) : nullptr;
    const aclTensor* out = hasOut ? MakeTensor(executor, {4, 4}, outDtype, op::Format::FORMAT_ND) : nullptr;
    if (self == nullptr || mat2 == nullptr || (hasBias && bias == nullptr) || (hasOut && out == nullptr)) {
        ADD_FAILURE() << "AllocTensor failed";
        return false;
    }
    return matmul_util_whitebox::CallCheckDtypeValid(self, mat2, bias, out, cubeMathType);
}

matmul_util_whitebox::Ops::NN::MmOpInfo MakeStreamKMmOpInfo(int64_t mDim, int64_t nDim, int64_t kDim,
                                                            int64_t dtypeASize, bool enableHf32, int64_t aiCoreCnt)
{
    matmul_util_whitebox::Ops::NN::MmOpInfo info{};
    info.shapeInfo.mDim = mDim;
    info.shapeInfo.nDim = nDim;
    info.shapeInfo.kDim = kDim;
    info.shapeInfo.dtypeASize = dtypeASize;
    info.enableHf32 = enableHf32;
    info.aiCoreCnt = aiCoreCnt;
    return info;
}

matmul_util_whitebox::Ops::NN::MmOpInfo MakeDPSKMmOpInfo(int64_t mDim, int64_t nDim, int64_t kDim, int64_t dtypeASize,
                                                         bool enableHf32, int64_t aiCoreCnt)
{
    matmul_util_whitebox::Ops::NN::MmOpInfo info{};
    info.shapeInfo.mDim = mDim;
    info.shapeInfo.nDim = nDim;
    info.shapeInfo.kDim = kDim;
    info.shapeInfo.dtypeASize = dtypeASize;
    info.enableHf32 = enableHf32;
    info.aiCoreCnt = aiCoreCnt;
    return info;
}

struct CastNullGuard {
    CastNullGuard() { matmul_util_whitebox::gCastToNullForTest = true; }
    ~CastNullGuard() { matmul_util_whitebox::gCastToNullForTest = false; }
};

matmul_util_whitebox::Ops::NN::MmOpInfo MakeNonContigMmOpInfo(int64_t mDim, int64_t nDim, int64_t kDim,
                                                              int64_t dtypeASize, bool enableHf32, int64_t aiCoreCnt,
                                                              op::DataType mat2Dtype)
{
    matmul_util_whitebox::Ops::NN::MmOpInfo info{};
    info.shapeInfo.mDim = mDim;
    info.shapeInfo.nDim = nDim;
    info.shapeInfo.kDim = kDim;
    info.shapeInfo.dtypeASize = dtypeASize;
    info.enableHf32 = enableHf32;
    info.aiCoreCnt = aiCoreCnt;
    info.ori_info.mat2_dtype = mat2Dtype;
    return info;
}

bool CheckGemmV3WithAlphaBetaCase(op::DataType biasDtype, op::DataType selfDtype, op::DataType mat2Dtype,
                                  int8_t cubeMathType = op::USE_FP32_ADD)
{
    auto uniqueExecutor = CREATE_EXECUTOR();
    if (uniqueExecutor.get() == nullptr) {
        ADD_FAILURE() << "CREATE_EXECUTOR failed";
        return false;
    }
    auto* executor = uniqueExecutor.get();
    const aclTensor* bias = MakeTensor(executor, {4}, biasDtype, op::Format::FORMAT_ND);
    const aclTensor* self = MakeTensor(executor, {4, 8}, selfDtype, op::Format::FORMAT_ND);
    const aclTensor* mat2 = MakeTensor(executor, {8, 4}, mat2Dtype, op::Format::FORMAT_ND);
    if (bias == nullptr || self == nullptr || mat2 == nullptr) {
        ADD_FAILURE() << "AllocTensor failed";
        return false;
    }
    return matmul_util_whitebox::Ops::NN::CheckGemmV3WithAlphaBeta(bias, self, mat2, cubeMathType);
}

bool IsSliceCase(const std::initializer_list<int64_t>& storageDims, const std::initializer_list<int64_t>& viewDims,
                 const op::Strides& strides, op::DataType selfDtype = op::DataType::DT_FLOAT16,
                 op::DataType mat2Dtype = op::DataType::DT_FLOAT16, op::Format mat2Format = op::Format::FORMAT_ND,
                 int8_t cubeMathType = kCubeMathKeepDtype, int64_t viewOffset = 0,
                 op::Format selfFormat = op::Format::FORMAT_ND)
{
    auto uniqueExecutor = CREATE_EXECUTOR();
    if (uniqueExecutor.get() == nullptr) {
        ADD_FAILURE() << "CREATE_EXECUTOR failed";
        return false;
    }
    auto* executor = uniqueExecutor.get();
    auto* storage = executor->AllocTensor(op::Shape(storageDims), selfDtype, selfFormat);
    if (storage == nullptr) {
        ADD_FAILURE() << "AllocTensor failed";
        return false;
    }
    const aclTensor* self = executor->CreateView(storage, op::Shape(viewDims), op::Shape(storageDims), strides,
                                                 viewOffset);
    const aclTensor* mat2 = MakeTensor(executor, {8, 8}, mat2Dtype, mat2Format);
    if (self == nullptr || mat2 == nullptr) {
        ADD_FAILURE() << "CreateView/AllocTensor failed";
        return false;
    }
    return matmul_util_whitebox::Ops::NN::IsSliceNonContiguous(self, mat2, cubeMathType);
}

bool IsCapableContiguousCase(const std::initializer_list<int64_t>& selfDims,
                             const std::initializer_list<int64_t>& mat2Dims,
                             op::DataType selfDtype = op::DataType::DT_FLOAT,
                             op::DataType mat2Dtype = op::DataType::DT_FLOAT,
                             op::Format selfFormat = op::Format::FORMAT_ND,
                             op::Format mat2Format = op::Format::FORMAT_ND, bool hasBias = false,
                             int8_t cubeMathType = op::USE_FP32_ADD)
{
    auto uniqueExecutor = CREATE_EXECUTOR();
    if (uniqueExecutor.get() == nullptr) {
        ADD_FAILURE() << "CREATE_EXECUTOR failed";
        return false;
    }
    auto* executor = uniqueExecutor.get();
    const aclTensor* self = MakeTensor(executor, selfDims, selfDtype, selfFormat);
    const aclTensor* mat2 = MakeTensor(executor, mat2Dims, mat2Dtype, mat2Format);
    const aclTensor* bias = hasBias ? MakeTensor(executor, {4}, op::DataType::DT_FLOAT, op::Format::FORMAT_ND) :
                                      nullptr;
    if (self == nullptr || mat2 == nullptr || (hasBias && bias == nullptr)) {
        ADD_FAILURE() << "AllocTensor failed";
        return false;
    }
    return matmul_util_whitebox::Ops::NN::IsCapableForMultiMul(self, mat2, bias, cubeMathType);
}

bool IsCapableViewCase(const std::initializer_list<int64_t>& selfStorageDims,
                       const std::initializer_list<int64_t>& selfViewDims, const op::Strides& selfStrides,
                       const std::initializer_list<int64_t>& mat2StorageDims,
                       const std::initializer_list<int64_t>& mat2ViewDims, const op::Strides& mat2Strides)
{
    auto uniqueExecutor = CREATE_EXECUTOR();
    if (uniqueExecutor.get() == nullptr) {
        ADD_FAILURE() << "CREATE_EXECUTOR failed";
        return false;
    }
    auto* executor = uniqueExecutor.get();
    auto* selfStorage = executor->AllocTensor(op::Shape(selfStorageDims), op::DataType::DT_FLOAT,
                                              op::Format::FORMAT_ND);
    auto* mat2Storage = executor->AllocTensor(op::Shape(mat2StorageDims), op::DataType::DT_FLOAT,
                                              op::Format::FORMAT_ND);
    if (selfStorage == nullptr || mat2Storage == nullptr) {
        ADD_FAILURE() << "AllocTensor failed";
        return false;
    }
    const aclTensor* self = executor->CreateView(selfStorage, op::Shape(selfViewDims), op::Shape(selfStorageDims),
                                                 selfStrides, 0);
    const aclTensor* mat2 = executor->CreateView(mat2Storage, op::Shape(mat2ViewDims), op::Shape(mat2StorageDims),
                                                 mat2Strides, 0);
    if (self == nullptr || mat2 == nullptr) {
        ADD_FAILURE() << "CreateView failed";
        return false;
    }
    return matmul_util_whitebox::Ops::NN::IsCapableForMultiMul(self, mat2, nullptr, op::USE_FP32_ADD);
}

} // namespace

class MatmulUtilWhiteboxTest : public testing::Test {};

TEST_F(MatmulUtilWhiteboxTest, MultiMulTranspose_NoTransposeNeeded)
{
    auto uniqueExecutor = CREATE_EXECUTOR();
    ASSERT_NE(uniqueExecutor.get(), nullptr);
    auto* executor = uniqueExecutor.get();
    const aclTensor* self = MakeTensor(executor, {4, 8}, op::DataType::DT_FLOAT, op::Format::FORMAT_ND);
    const aclTensor* mat2 = MakeViewTensor(executor, op::DataType::DT_FLOAT, {4, 8}, {8, 4}, op::Strides{1, 8});
    ASSERT_NE(self, nullptr);
    ASSERT_NE(mat2, nullptr);
    bool transposeX2 = false;

    EXPECT_TRUE(matmul_util_whitebox::Ops::NN::MultiMulTranspose(self, mat2, transposeX2, executor));
    EXPECT_FALSE(transposeX2);
}

TEST_F(MatmulUtilWhiteboxTest, MultiMulTranspose_ATransposeOnly)
{
    auto uniqueExecutor = CREATE_EXECUTOR();
    ASSERT_NE(uniqueExecutor.get(), nullptr);
    auto* executor = uniqueExecutor.get();
    const aclTensor* self = MakeViewTensor(executor, op::DataType::DT_FLOAT, {4, 8}, {8, 4}, op::Strides{1, 8});
    const aclTensor* mat2 = MakeViewTensor(executor, op::DataType::DT_FLOAT, {4, 8}, {8, 4}, op::Strides{1, 8});
    ASSERT_NE(self, nullptr);
    ASSERT_NE(mat2, nullptr);
    bool transposeX2 = false;

    EXPECT_TRUE(matmul_util_whitebox::Ops::NN::MultiMulTranspose(self, mat2, transposeX2, executor));
    EXPECT_FALSE(transposeX2);
}

TEST_F(MatmulUtilWhiteboxTest, MultiMulTranspose_BNotTransposed)
{
    auto uniqueExecutor = CREATE_EXECUTOR();
    ASSERT_NE(uniqueExecutor.get(), nullptr);
    auto* executor = uniqueExecutor.get();
    const aclTensor* self = MakeTensor(executor, {4, 8}, op::DataType::DT_FLOAT, op::Format::FORMAT_ND);
    const aclTensor* mat2 = MakeTensor(executor, {8, 4}, op::DataType::DT_FLOAT, op::Format::FORMAT_ND);
    ASSERT_NE(self, nullptr);
    ASSERT_NE(mat2, nullptr);
    bool transposeX2 = false;

    EXPECT_TRUE(matmul_util_whitebox::Ops::NN::MultiMulTranspose(self, mat2, transposeX2, executor));
    EXPECT_TRUE(transposeX2);
}

TEST_F(MatmulUtilWhiteboxTest, CheckAscendCScenario2_ScSplitKAl1FullLoad_Hit)
{
    NpuArchManager archManager(NpuArch::DAV_2201);
    EXPECT_TRUE(CheckAscendCScenario2Case({128, 9216}, {9216, 9472}, op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT16,
                                          op::Format::FORMAT_ND, op::Format::FORMAT_ND, op::Format::FORMAT_ND));
}

TEST_F(MatmulUtilWhiteboxTest, CheckAscendCScenario2_ArchNotDav2002)
{
    NpuArchManager archManager(NpuArch::DAV_2201);
    EXPECT_FALSE(CheckAscendCScenario2Case({4096, 4096}, {4096, 4096}, op::DataType::DT_FLOAT16,
                                           op::DataType::DT_FLOAT16, op::Format::FORMAT_ND, op::Format::FORMAT_ND,
                                           op::Format::FORMAT_ND));
}

TEST_F(MatmulUtilWhiteboxTest, CheckAscendCScenario2_DtypeX1NotFp16)
{
    NpuArchManager archManager(NpuArch::DAV_2002);
    EXPECT_FALSE(
        CheckAscendCScenario2Case({4096, 4096}, {4096, 4096}, op::DataType::DT_BF16, op::DataType::DT_FLOAT16));
}

TEST_F(MatmulUtilWhiteboxTest, CheckAscendCScenario2_DtypeX2NotFp16)
{
    NpuArchManager archManager(NpuArch::DAV_2002);
    EXPECT_FALSE(
        CheckAscendCScenario2Case({4096, 4096}, {4096, 4096}, op::DataType::DT_FLOAT16, op::DataType::DT_BF16));
}

TEST_F(MatmulUtilWhiteboxTest, CheckAscendCScenario2_FormatSelfNotNd)
{
    NpuArchManager archManager(NpuArch::DAV_2002);
    EXPECT_FALSE(CheckAscendCScenario2Case({4096, 4096}, {4096, 4096}, op::DataType::DT_FLOAT16,
                                           op::DataType::DT_FLOAT16, op::Format::FORMAT_FRACTAL_NZ,
                                           op::Format::FORMAT_FRACTAL_NZ, op::Format::FORMAT_ND));
}

TEST_F(MatmulUtilWhiteboxTest, CheckAscendCScenario2_FormatMat2NotNz)
{
    NpuArchManager archManager(NpuArch::DAV_2002);
    EXPECT_FALSE(CheckAscendCScenario2Case({4096, 4096}, {4096, 4096}, op::DataType::DT_FLOAT16,
                                           op::DataType::DT_FLOAT16, op::Format::FORMAT_ND, op::Format::FORMAT_ND,
                                           op::Format::FORMAT_ND));
}

TEST_F(MatmulUtilWhiteboxTest, CheckAscendCScenario2_FormatOutNotNd)
{
    NpuArchManager archManager(NpuArch::DAV_2002);
    EXPECT_FALSE(CheckAscendCScenario2Case({4096, 4096}, {4096, 4096}, op::DataType::DT_FLOAT16,
                                           op::DataType::DT_FLOAT16, op::Format::FORMAT_ND,
                                           op::Format::FORMAT_FRACTAL_NZ, op::Format::FORMAT_FRACTAL_NZ));
}

TEST_F(MatmulUtilWhiteboxTest, CheckAscendCScenario2_BoundsM)
{
    NpuArchManager archManager(NpuArch::DAV_2002);
    EXPECT_FALSE(CheckAscendCScenario2Case({5505, 4096}, {4096, 4096}));
    EXPECT_FALSE(CheckAscendCScenario2Case({2047, 4096}, {4096, 4096}));
}

TEST_F(MatmulUtilWhiteboxTest, CheckAscendCScenario2_BoundsK)
{
    NpuArchManager archManager(NpuArch::DAV_2002);
    EXPECT_FALSE(CheckAscendCScenario2Case({4096, 5505}, {4096, 4096}));
    EXPECT_FALSE(CheckAscendCScenario2Case({4096, 2047}, {4096, 4096}));
}

TEST_F(MatmulUtilWhiteboxTest, CheckAscendCScenario2_BoundsN)
{
    NpuArchManager archManager(NpuArch::DAV_2002);
    EXPECT_FALSE(CheckAscendCScenario2Case({4096, 4096}, {4096, 5505}));
    EXPECT_FALSE(CheckAscendCScenario2Case({4096, 4096}, {4096, 2047}));
}

TEST_F(MatmulUtilWhiteboxTest, CheckAscendCScenario2_InRange_True)
{
    NpuArchManager archManager(NpuArch::DAV_2002);
    EXPECT_TRUE(CheckAscendCScenario2Case({4096, 4096}, {4096, 4096}));
}

TEST_F(MatmulUtilWhiteboxTest, CheckAscendCScenario2_TransposeTrue_True)
{
    NpuArchManager archManager(NpuArch::DAV_2002);
    EXPECT_TRUE(CheckAscendCScenario2Case({3000, 4000}, {4000, 3000}, op::DataType::DT_FLOAT16,
                                          op::DataType::DT_FLOAT16, op::Format::FORMAT_ND,
                                          op::Format::FORMAT_FRACTAL_NZ, op::Format::FORMAT_ND, true, true));
}

TEST_F(MatmulUtilWhiteboxTest, PromoteOutputAndBiasDtype_BiasFp32)
{
    EXPECT_EQ(matmul_util_whitebox::CallPromoteOutputAndBiasDtype(op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT),
              op::DataType::DT_FLOAT);
}

TEST_F(MatmulUtilWhiteboxTest, PromoteOutputAndBiasDtype_OutputFp32)
{
    EXPECT_EQ(matmul_util_whitebox::CallPromoteOutputAndBiasDtype(op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16),
              op::DataType::DT_FLOAT);
}

TEST_F(MatmulUtilWhiteboxTest, PromoteOutputAndBiasDtype_BothFp16)
{
    EXPECT_EQ(matmul_util_whitebox::CallPromoteOutputAndBiasDtype(op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT16),
              op::DataType::DT_FLOAT16);
}

TEST_F(MatmulUtilWhiteboxTest, SetMmSupportDType_ArchNotSupportFp32)
{
    NpuArchManager archManager(NpuArch::DAV_1001);
    auto info = MakeMmOpInfo(op::DataType::DT_FLOAT, op::DataType::DT_FLOAT, op::DataType::DT_FLOAT);
    EXPECT_EQ(matmul_util_whitebox::Ops::NN::SetMmSupportDType(info, kCubeMathKeepDtype), ACLNN_SUCCESS);
    EXPECT_EQ(info.support_info.self_dtype, op::DataType::DT_FLOAT16);
    EXPECT_EQ(info.support_info.mat2_dtype, op::DataType::DT_FLOAT16);
}

TEST_F(MatmulUtilWhiteboxTest, SetMmSupportDType_UseFp16)
{
    NpuArchManager archManager(NpuArch::DAV_2201);
    {
        auto info = MakeMmOpInfo(op::DataType::DT_FLOAT, op::DataType::DT_FLOAT, op::DataType::DT_FLOAT);
        matmul_util_whitebox::Ops::NN::SetMmSupportDType(info, op::USE_FP16);
        EXPECT_EQ(info.support_info.self_dtype, op::DataType::DT_FLOAT16);
        EXPECT_EQ(info.support_info.mat2_dtype, op::DataType::DT_FLOAT16);
        EXPECT_EQ(info.support_info.output_dtype, op::DataType::DT_FLOAT16);
    }
    {
        auto info = MakeMmOpInfo(op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT16);
        matmul_util_whitebox::Ops::NN::SetMmSupportDType(info, op::USE_FP16);
        EXPECT_EQ(info.support_info.self_dtype, op::DataType::DT_FLOAT16);
        EXPECT_EQ(info.support_info.mat2_dtype, op::DataType::DT_FLOAT16);
        EXPECT_EQ(info.support_info.output_dtype, op::DataType::DT_FLOAT16);
    }
    {
        auto info = MakeMmOpInfo(op::DataType::DT_BF16, op::DataType::DT_BF16, op::DataType::DT_BF16);
        matmul_util_whitebox::Ops::NN::SetMmSupportDType(info, op::USE_FP16);
        EXPECT_EQ(info.support_info.self_dtype, op::DataType::DT_BF16);
        EXPECT_EQ(info.support_info.mat2_dtype, op::DataType::DT_BF16);
        EXPECT_EQ(info.support_info.output_dtype, op::DataType::DT_BF16);
    }
}

TEST_F(MatmulUtilWhiteboxTest, SetMmSupportDType_DtypeMismatchToFp32)
{
    NpuArchManager archManager(NpuArch::DAV_2201);
    {
        auto info = MakeMmOpInfo(op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT16);
        matmul_util_whitebox::Ops::NN::SetMmSupportDType(info, kCubeMathKeepDtype);
        EXPECT_EQ(info.support_info.self_dtype, op::DataType::DT_FLOAT);
        EXPECT_EQ(info.support_info.mat2_dtype, op::DataType::DT_FLOAT);
        EXPECT_EQ(info.support_info.output_dtype, op::DataType::DT_FLOAT);
    }
    {
        auto info = MakeMmOpInfo(op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16);
        matmul_util_whitebox::Ops::NN::SetMmSupportDType(info, kCubeMathKeepDtype);
        EXPECT_EQ(info.support_info.self_dtype, op::DataType::DT_FLOAT);
        EXPECT_EQ(info.support_info.mat2_dtype, op::DataType::DT_FLOAT);
        EXPECT_EQ(info.support_info.output_dtype, op::DataType::DT_FLOAT);
    }
    {
        auto info = MakeMmOpInfo(op::DataType::DT_FLOAT16, op::DataType::DT_BF16, op::DataType::DT_FLOAT16);
        matmul_util_whitebox::Ops::NN::SetMmSupportDType(info, kCubeMathKeepDtype);
        EXPECT_EQ(info.support_info.self_dtype, op::DataType::DT_FLOAT);
        EXPECT_EQ(info.support_info.mat2_dtype, op::DataType::DT_FLOAT);
        EXPECT_EQ(info.support_info.output_dtype, op::DataType::DT_FLOAT);
    }
    {
        auto info = MakeMmOpInfo(op::DataType::DT_BF16, op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT16);
        matmul_util_whitebox::Ops::NN::SetMmSupportDType(info, kCubeMathKeepDtype);
        EXPECT_EQ(info.support_info.self_dtype, op::DataType::DT_FLOAT);
        EXPECT_EQ(info.support_info.mat2_dtype, op::DataType::DT_FLOAT);
        EXPECT_EQ(info.support_info.output_dtype, op::DataType::DT_FLOAT);
    }
}

TEST_F(MatmulUtilWhiteboxTest, SetMmSupportDType_UseFp32Add)
{
    NpuArchManager archManager(NpuArch::DAV_2201);
    {
        auto info = MakeMmOpInfo(op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT16);
        matmul_util_whitebox::Ops::NN::SetMmSupportDType(info, op::USE_FP32_ADD);
        EXPECT_EQ(info.support_info.self_dtype, op::DataType::DT_FLOAT16);
        EXPECT_EQ(info.support_info.mat2_dtype, op::DataType::DT_FLOAT16);
        EXPECT_EQ(info.support_info.output_dtype, op::DataType::DT_FLOAT);
    }
    {
        auto info = MakeMmOpInfo(op::DataType::DT_BF16, op::DataType::DT_BF16, op::DataType::DT_BF16);
        matmul_util_whitebox::Ops::NN::SetMmSupportDType(info, op::USE_FP32_ADD);
        EXPECT_EQ(info.support_info.self_dtype, op::DataType::DT_BF16);
        EXPECT_EQ(info.support_info.mat2_dtype, op::DataType::DT_BF16);
        EXPECT_EQ(info.support_info.output_dtype, op::DataType::DT_FLOAT);
    }
}

TEST_F(MatmulUtilWhiteboxTest, SetMmSupportDType_LowPrecisionFp32Out)
{
    NpuArchManager archManager(NpuArch::DAV_2201);
    {
        auto info = MakeMmOpInfo(op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT);
        matmul_util_whitebox::Ops::NN::SetMmSupportDType(info, kCubeMathKeepDtype);
        EXPECT_EQ(info.support_info.self_dtype, op::DataType::DT_FLOAT16);
        EXPECT_EQ(info.support_info.mat2_dtype, op::DataType::DT_FLOAT16);
        EXPECT_EQ(info.support_info.output_dtype, op::DataType::DT_FLOAT);
    }
    {
        auto info = MakeMmOpInfo(op::DataType::DT_BF16, op::DataType::DT_BF16, op::DataType::DT_FLOAT);
        matmul_util_whitebox::Ops::NN::SetMmSupportDType(info, kCubeMathKeepDtype);
        EXPECT_EQ(info.support_info.self_dtype, op::DataType::DT_BF16);
        EXPECT_EQ(info.support_info.mat2_dtype, op::DataType::DT_BF16);
        EXPECT_EQ(info.support_info.output_dtype, op::DataType::DT_FLOAT);
    }
}

TEST_F(MatmulUtilWhiteboxTest, SetMmSupportDType_NoChange)
{
    NpuArchManager archManager(NpuArch::DAV_2201);
    {
        auto info = MakeMmOpInfo(op::DataType::DT_FLOAT, op::DataType::DT_FLOAT, op::DataType::DT_FLOAT);
        matmul_util_whitebox::Ops::NN::SetMmSupportDType(info, kCubeMathKeepDtype);
        EXPECT_EQ(info.support_info.self_dtype, op::DataType::DT_FLOAT);
        EXPECT_EQ(info.support_info.mat2_dtype, op::DataType::DT_FLOAT);
        EXPECT_EQ(info.support_info.output_dtype, op::DataType::DT_FLOAT);
    }
    {
        auto info = MakeMmOpInfo(op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT16);
        matmul_util_whitebox::Ops::NN::SetMmSupportDType(info, kCubeMathKeepDtype);
        EXPECT_EQ(info.support_info.self_dtype, op::DataType::DT_FLOAT16);
        EXPECT_EQ(info.support_info.mat2_dtype, op::DataType::DT_FLOAT16);
        EXPECT_EQ(info.support_info.output_dtype, op::DataType::DT_FLOAT16);
    }
    {
        auto info = MakeMmOpInfo(op::DataType::DT_BF16, op::DataType::DT_BF16, op::DataType::DT_BF16);
        matmul_util_whitebox::Ops::NN::SetMmSupportDType(info, kCubeMathKeepDtype);
        EXPECT_EQ(info.support_info.self_dtype, op::DataType::DT_BF16);
        EXPECT_EQ(info.support_info.mat2_dtype, op::DataType::DT_BF16);
        EXPECT_EQ(info.support_info.output_dtype, op::DataType::DT_BF16);
    }
}

TEST_F(MatmulUtilWhiteboxTest, SetMmSupportDType_Arch3510UseFp16)
{
    NpuArchManager archManager(NpuArch::DAV_3510);
    auto info = MakeMmOpInfo(op::DataType::DT_FLOAT, op::DataType::DT_FLOAT, op::DataType::DT_FLOAT);
    matmul_util_whitebox::Ops::NN::SetMmSupportDType(info, op::USE_FP16);
    EXPECT_EQ(info.support_info.self_dtype, op::DataType::DT_FLOAT16);
    EXPECT_EQ(info.support_info.mat2_dtype, op::DataType::DT_FLOAT16);
    EXPECT_EQ(info.support_info.output_dtype, op::DataType::DT_FLOAT16);
}

TEST_F(MatmulUtilWhiteboxTest, GetOffSet_BmmAndMm)
{
    EXPECT_EQ(matmul_util_whitebox::Ops::NN::GetOffSet(3), 1);
    EXPECT_EQ(matmul_util_whitebox::Ops::NN::GetOffSet(2), 0);
    EXPECT_EQ(matmul_util_whitebox::Ops::NN::GetOffSet(4), 0);
}

TEST_F(MatmulUtilWhiteboxTest, CheckDtypeValid_Bf16ArchSupported)
{
    NpuArchManager archManager(NpuArch::DAV_2201);
    EXPECT_TRUE(CheckDtypeValidCase(op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT16));
    EXPECT_TRUE(CheckDtypeValidCase(op::DataType::DT_BF16, op::DataType::DT_BF16, true, op::DataType::DT_BF16));
    EXPECT_TRUE(CheckDtypeValidCase(op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16));
}

TEST_F(MatmulUtilWhiteboxTest, CheckDtypeValid_Bf16ArchUnsupportedSelf)
{
    NpuArchManager archManager(NpuArch::DAV_2201);
    EXPECT_FALSE(CheckDtypeValidCase(op::DataType::DT_INT8, op::DataType::DT_FLOAT16));
}

TEST_F(MatmulUtilWhiteboxTest, CheckDtypeValid_Bf16ArchUnsupportedMat2)
{
    NpuArchManager archManager(NpuArch::DAV_2201);
    EXPECT_FALSE(CheckDtypeValidCase(op::DataType::DT_FLOAT16, op::DataType::DT_INT8));
}

TEST_F(MatmulUtilWhiteboxTest, CheckDtypeValid_Bf16ArchUnsupportedBias)
{
    NpuArchManager archManager(NpuArch::DAV_2201);
    EXPECT_FALSE(CheckDtypeValidCase(op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT16, true, op::DataType::DT_INT8));
}

TEST_F(MatmulUtilWhiteboxTest, CheckDtypeValid_NoBf16ArchSupported)
{
    NpuArchManager archManager(NpuArch::DAV_1001);
    EXPECT_TRUE(CheckDtypeValidCase(op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT16));
    EXPECT_TRUE(CheckDtypeValidCase(op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16));
    EXPECT_TRUE(
        CheckDtypeValidCase(op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT16, true, op::DataType::DT_FLOAT16));
}

TEST_F(MatmulUtilWhiteboxTest, CheckDtypeValid_NoBf16ArchRejectSelfBf16)
{
    NpuArchManager archManager(NpuArch::DAV_1001);
    EXPECT_FALSE(CheckDtypeValidCase(op::DataType::DT_BF16, op::DataType::DT_FLOAT16));
}

TEST_F(MatmulUtilWhiteboxTest, CheckDtypeValid_NoBf16ArchRejectMat2Bf16)
{
    NpuArchManager archManager(NpuArch::DAV_1001);
    EXPECT_FALSE(CheckDtypeValidCase(op::DataType::DT_FLOAT16, op::DataType::DT_BF16));
}

TEST_F(MatmulUtilWhiteboxTest, CheckDtypeValid_NoBf16ArchRejectBiasBf16)
{
    NpuArchManager archManager(NpuArch::DAV_1001);
    EXPECT_FALSE(CheckDtypeValidCase(op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT16, true, op::DataType::DT_BF16));
}

TEST_F(MatmulUtilWhiteboxTest, CheckDtypeValid_KeepDtypeFp16OutFp32Self)
{
    NpuArchManager archManager(NpuArch::DAV_2201);
    EXPECT_FALSE(CheckDtypeValidCase(op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, false, op::DataType::DT_FLOAT16,
                                     true, op::DataType::DT_FLOAT16, kCubeMathKeepDtype));
}

TEST_F(MatmulUtilWhiteboxTest, CheckDtypeValid_OutNull)
{
    NpuArchManager archManager(NpuArch::DAV_2201);
    EXPECT_TRUE(CheckDtypeValidCase(op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16));
}

TEST_F(MatmulUtilWhiteboxTest, CheckDtypeValid_NotKeepDtype)
{
    NpuArchManager archManager(NpuArch::DAV_2201);
    EXPECT_TRUE(CheckDtypeValidCase(op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, false, op::DataType::DT_FLOAT16,
                                    true, op::DataType::DT_FLOAT16, op::ALLOW_FP32_DOWN_PRECISION));
}

TEST_F(MatmulUtilWhiteboxTest, CheckDtypeValid_SelfNotFp32)
{
    NpuArchManager archManager(NpuArch::DAV_2201);
    EXPECT_TRUE(CheckDtypeValidCase(op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT16, false, op::DataType::DT_FLOAT16,
                                    true, op::DataType::DT_FLOAT16, kCubeMathKeepDtype));
}

TEST_F(MatmulUtilWhiteboxTest, CheckDtypeValid_OutNotFp16)
{
    NpuArchManager archManager(NpuArch::DAV_2201);
    EXPECT_TRUE(CheckDtypeValidCase(op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, false, op::DataType::DT_FLOAT16,
                                    true, op::DataType::DT_FLOAT, kCubeMathKeepDtype));
}

TEST_F(MatmulUtilWhiteboxTest, CheckStreamKSKTiling_Fp32LargeK_Rejected)
{
    auto info = MakeStreamKMmOpInfo(1024, 1024, 2000001, 4, false, 16);
    EXPECT_FALSE(matmul_util_whitebox::CallCheckStreamKSKTiling(info));
}

TEST_F(MatmulUtilWhiteboxTest, CheckStreamKSKTiling_Fp32LargeKHf32_True)
{
    auto info = MakeStreamKMmOpInfo(256, 256, 2000001, 4, true, 16);
    EXPECT_TRUE(matmul_util_whitebox::CallCheckStreamKSKTiling(info));
}

TEST_F(MatmulUtilWhiteboxTest, CheckStreamKSKTiling_Fp16LargeK_True)
{
    auto info = MakeStreamKMmOpInfo(256, 256, 2000001, 2, false, 16);
    EXPECT_TRUE(matmul_util_whitebox::CallCheckStreamKSKTiling(info));
}

TEST_F(MatmulUtilWhiteboxTest, CheckStreamKSKTiling_KBelowThreshold_Rejected)
{
    auto info = MakeStreamKMmOpInfo(256, 256, 256, 2, false, 32);
    EXPECT_FALSE(matmul_util_whitebox::CallCheckStreamKSKTiling(info));
}

TEST_F(MatmulUtilWhiteboxTest, CheckStreamKSKTiling_Fp32Align32_Rejected)
{
    auto info = MakeStreamKMmOpInfo(1024, 1024, 8192, 4, false, 16);
    EXPECT_FALSE(matmul_util_whitebox::CallCheckStreamKSKTiling(info));
}

TEST_F(MatmulUtilWhiteboxTest, CheckStreamKSKTiling_Fp32Align32_True)
{
    auto info = MakeStreamKMmOpInfo(64, 64, 8192, 4, false, 16);
    EXPECT_TRUE(matmul_util_whitebox::CallCheckStreamKSKTiling(info));
}

TEST_F(MatmulUtilWhiteboxTest, CheckStreamKSKTiling_Fp16Align256_Rejected)
{
    auto info = MakeStreamKMmOpInfo(1024, 1024, 8192, 2, false, 1);
    EXPECT_FALSE(matmul_util_whitebox::CallCheckStreamKSKTiling(info));
}

TEST_F(MatmulUtilWhiteboxTest, CheckStreamKSKTiling_Fp16Align256_True)
{
    auto info = MakeStreamKMmOpInfo(256, 256, 8192, 2, false, 16);
    EXPECT_TRUE(matmul_util_whitebox::CallCheckStreamKSKTiling(info));
}

TEST_F(MatmulUtilWhiteboxTest, CheckStreamKSKTiling_ZeroAiCore_Rejected)
{
    auto info = MakeStreamKMmOpInfo(256, 256, 8192, 2, false, 0);
    EXPECT_FALSE(matmul_util_whitebox::CallCheckStreamKSKTiling(info));
}

TEST_F(MatmulUtilWhiteboxTest, CheckStreamKDPSKTiling_MNotAligned)
{
    auto info = MakeDPSKMmOpInfo(257, 256, 8192, 2, false, 8);
    EXPECT_FALSE(matmul_util_whitebox::CallCheckStreamKDPSKTiling(info));
}

TEST_F(MatmulUtilWhiteboxTest, CheckStreamKDPSKTiling_NNotAligned)
{
    auto info = MakeDPSKMmOpInfo(256, 257, 8192, 2, false, 8);
    EXPECT_FALSE(matmul_util_whitebox::CallCheckStreamKDPSKTiling(info));
}

TEST_F(MatmulUtilWhiteboxTest, CheckStreamKDPSKTiling_KBelowThreshold)
{
    auto info = MakeDPSKMmOpInfo(256, 256, 4095, 2, false, 8);
    EXPECT_FALSE(matmul_util_whitebox::CallCheckStreamKDPSKTiling(info));
}

TEST_F(MatmulUtilWhiteboxTest, CheckStreamKDPSKTiling_Fp32NoHf32)
{
    auto info = MakeDPSKMmOpInfo(256, 256, 8192, 4, false, 8);
    EXPECT_FALSE(matmul_util_whitebox::CallCheckStreamKDPSKTiling(info));
}

TEST_F(MatmulUtilWhiteboxTest, CheckStreamKDPSKTiling_Fp32Hf32_True)
{
    auto info = MakeDPSKMmOpInfo(768, 256, 8192, 4, true, 2);
    EXPECT_TRUE(matmul_util_whitebox::CallCheckStreamKDPSKTiling(info));
}

TEST_F(MatmulUtilWhiteboxTest, CheckStreamKDPSKTiling_TotalLessThanAiCore)
{
    auto info = MakeDPSKMmOpInfo(256, 256, 8192, 2, false, 16);
    EXPECT_FALSE(matmul_util_whitebox::CallCheckStreamKDPSKTiling(info));
}

TEST_F(MatmulUtilWhiteboxTest, CheckStreamKDPSKTiling_TotalModAiCoreZero)
{
    auto info = MakeDPSKMmOpInfo(512, 1024, 8192, 2, false, 8);
    EXPECT_FALSE(matmul_util_whitebox::CallCheckStreamKDPSKTiling(info));
}

TEST_F(MatmulUtilWhiteboxTest, CheckStreamKDPSKTiling_TotalModAiCoreOverHalf)
{
    auto info = MakeDPSKMmOpInfo(3328, 256, 8192, 2, false, 8);
    EXPECT_FALSE(matmul_util_whitebox::CallCheckStreamKDPSKTiling(info));
}

TEST_F(MatmulUtilWhiteboxTest, CheckStreamKDPSKTiling_AllConditionsPass_True)
{
    auto info = MakeDPSKMmOpInfo(3072, 256, 8192, 2, false, 8);
    EXPECT_TRUE(matmul_util_whitebox::CallCheckStreamKDPSKTiling(info));
}

TEST_F(MatmulUtilWhiteboxTest, CheckStreamKDPSKTiling_BoundaryValues)
{
    {
        auto info = MakeDPSKMmOpInfo(256, 256, 8192, 2, false, 0);
        EXPECT_FALSE(matmul_util_whitebox::CallCheckStreamKDPSKTiling(info));
    }
    {
        auto info = MakeDPSKMmOpInfo(256, 256, 8192, 0, false, 8);
        EXPECT_FALSE(matmul_util_whitebox::CallCheckStreamKDPSKTiling(info));
    }
}

TEST_F(MatmulUtilWhiteboxTest, HandleEmptyTensor_ProcessEmptyTensorNull)
{
    NpuArchManager archManager(NpuArch::DAV_2201);
    auto uniqueExecutor = CREATE_EXECUTOR();
    ASSERT_NE(uniqueExecutor.get(), nullptr);
    auto* executor = uniqueExecutor.get();
    const aclTensor* self = MakeTensor(executor, {4, 0}, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND);
    const aclTensor* mat2 = MakeTensor(executor, {0, 4}, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND);
    ASSERT_NE(self, nullptr);
    ASSERT_NE(mat2, nullptr);

    EXPECT_EQ(matmul_util_whitebox::CallHandleEmptyTensor(self, mat2, executor, kCubeMathKeepDtype), nullptr);
}

TEST_F(MatmulUtilWhiteboxTest, HandleEmptyTensor_CastDav2201KeepDtype)
{
    NpuArchManager archManager(NpuArch::DAV_2201);
    auto uniqueExecutor = CREATE_EXECUTOR();
    ASSERT_NE(uniqueExecutor.get(), nullptr);
    auto* executor = uniqueExecutor.get();
    const aclTensor* self = MakeTensor(executor, {0, 4}, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND);
    const aclTensor* mat2 = MakeTensor(executor, {4, 5}, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND);
    ASSERT_NE(self, nullptr);
    ASSERT_NE(mat2, nullptr);

    auto* out = matmul_util_whitebox::CallHandleEmptyTensor(self, mat2, executor, op::FP16FP32_KEEP_DTYPE);
    ASSERT_NE(out, nullptr);
    EXPECT_EQ(out->GetViewShape().GetDimNum(), 2);
    EXPECT_EQ(out->GetViewShape().GetDim(0), 0);
    EXPECT_EQ(out->GetViewShape().GetDim(1), 5);
}

TEST_F(MatmulUtilWhiteboxTest, HandleEmptyTensor_NotDav2201)
{
    NpuArchManager archManager(NpuArch::DAV_1001);
    auto uniqueExecutor = CREATE_EXECUTOR();
    ASSERT_NE(uniqueExecutor.get(), nullptr);
    auto* executor = uniqueExecutor.get();
    const aclTensor* self = MakeTensor(executor, {0, 4}, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND);
    const aclTensor* mat2 = MakeTensor(executor, {4, 5}, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND);
    ASSERT_NE(self, nullptr);
    ASSERT_NE(mat2, nullptr);

    auto* out = matmul_util_whitebox::CallHandleEmptyTensor(self, mat2, executor, op::FP16FP32_KEEP_DTYPE);
    ASSERT_NE(out, nullptr);
    EXPECT_EQ(out->GetViewShape().GetDim(0), 0);
    EXPECT_EQ(out->GetViewShape().GetDim(1), 5);
}

TEST_F(MatmulUtilWhiteboxTest, HandleEmptyTensor_NotKeepDtype)
{
    NpuArchManager archManager(NpuArch::DAV_2201);
    auto uniqueExecutor = CREATE_EXECUTOR();
    ASSERT_NE(uniqueExecutor.get(), nullptr);
    auto* executor = uniqueExecutor.get();
    const aclTensor* self = MakeTensor(executor, {0, 4}, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND);
    const aclTensor* mat2 = MakeTensor(executor, {4, 5}, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND);
    ASSERT_NE(self, nullptr);
    ASSERT_NE(mat2, nullptr);

    auto* out = matmul_util_whitebox::CallHandleEmptyTensor(self, mat2, executor, kCubeMathKeepDtype);
    ASSERT_NE(out, nullptr);
    EXPECT_EQ(out->GetViewShape().GetDim(0), 0);
    EXPECT_EQ(out->GetViewShape().GetDim(1), 5);
}

TEST_F(MatmulUtilWhiteboxTest, HandleEmptyTensor_CastNull)
{
    NpuArchManager archManager(NpuArch::DAV_2201);
    auto uniqueExecutor = CREATE_EXECUTOR();
    ASSERT_NE(uniqueExecutor.get(), nullptr);
    auto* executor = uniqueExecutor.get();
    const aclTensor* self = MakeTensor(executor, {0, 4}, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND);
    const aclTensor* mat2 = MakeTensor(executor, {4, 5}, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND);
    ASSERT_NE(self, nullptr);
    ASSERT_NE(mat2, nullptr);

    CastNullGuard guard;
    EXPECT_EQ(matmul_util_whitebox::CallHandleEmptyTensor(self, mat2, executor, op::FP16FP32_KEEP_DTYPE), nullptr);
}

TEST_F(MatmulUtilWhiteboxTest, SetTensorToNDFormat_NchwToNd)
{
    auto uniqueExecutor = CREATE_EXECUTOR();
    ASSERT_NE(uniqueExecutor.get(), nullptr);
    auto* executor = uniqueExecutor.get();
    const aclTensor* input = MakeTensor(executor, {2, 3}, op::DataType::DT_FLOAT16, op::Format::FORMAT_NCHW);
    ASSERT_NE(input, nullptr);

    const aclTensor* out = matmul_util_whitebox::CallSetTensorToNDFormat(input);
    EXPECT_EQ(out, input);
    EXPECT_EQ(out->GetViewFormat(), op::Format::FORMAT_ND);
    EXPECT_EQ(out->GetOriginalFormat(), op::Format::FORMAT_ND);
    EXPECT_EQ(out->GetStorageFormat(), op::Format::FORMAT_ND);
}

TEST_F(MatmulUtilWhiteboxTest, SetTensorToNDFormat_AlreadyNd)
{
    auto uniqueExecutor = CREATE_EXECUTOR();
    ASSERT_NE(uniqueExecutor.get(), nullptr);
    auto* executor = uniqueExecutor.get();
    const aclTensor* input = MakeTensor(executor, {2, 3}, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND);
    ASSERT_NE(input, nullptr);

    const aclTensor* out = matmul_util_whitebox::CallSetTensorToNDFormat(input);
    EXPECT_EQ(out, input);
    EXPECT_EQ(out->GetStorageFormat(), op::Format::FORMAT_ND);
}

TEST_F(MatmulUtilWhiteboxTest, SetTensorToNDFormat_AlreadyNz)
{
    auto uniqueExecutor = CREATE_EXECUTOR();
    ASSERT_NE(uniqueExecutor.get(), nullptr);
    auto* executor = uniqueExecutor.get();
    const aclTensor* input = MakeTensor(executor, {2, 3}, op::DataType::DT_FLOAT16, op::Format::FORMAT_FRACTAL_NZ);
    ASSERT_NE(input, nullptr);

    const aclTensor* out = matmul_util_whitebox::CallSetTensorToNDFormat(input);
    EXPECT_EQ(out, input);
    EXPECT_EQ(out->GetStorageFormat(), op::Format::FORMAT_FRACTAL_NZ);
}

TEST_F(MatmulUtilWhiteboxTest, CheckNonContiguousShapeSupport_SkTiling)
{
    auto info = MakeNonContigMmOpInfo(256, 256, 8192, 2, false, 16, op::DataType::DT_FLOAT16);
    EXPECT_FALSE(matmul_util_whitebox::Ops::NN::CheckNonContiguousShapeSupport(info));
}

TEST_F(MatmulUtilWhiteboxTest, CheckNonContiguousShapeSupport_DpskTiling)
{
    auto info = MakeNonContigMmOpInfo(2560, 256, 8192, 2, false, 8, op::DataType::DT_FLOAT16);
    EXPECT_FALSE(matmul_util_whitebox::Ops::NN::CheckNonContiguousShapeSupport(info));
}

TEST_F(MatmulUtilWhiteboxTest, CheckNonContiguousShapeSupport_Fp32LargeK)
{
    auto info = MakeNonContigMmOpInfo(256, 256, 8192, 4, false, 16, op::DataType::DT_FLOAT);
    EXPECT_FALSE(matmul_util_whitebox::Ops::NN::CheckNonContiguousShapeSupport(info));
}

TEST_F(MatmulUtilWhiteboxTest, CheckNonContiguousShapeSupport_Fp32Hf32)
{
    auto info = MakeNonContigMmOpInfo(1536, 256, 8192, 4, true, 8, op::DataType::DT_FLOAT);
    EXPECT_TRUE(matmul_util_whitebox::Ops::NN::CheckNonContiguousShapeSupport(info));
}

TEST_F(MatmulUtilWhiteboxTest, CheckNonContiguousShapeSupport_Fp32SmallK)
{
    auto info = MakeNonContigMmOpInfo(256, 256, 1024, 4, false, 8, op::DataType::DT_FLOAT);
    EXPECT_TRUE(matmul_util_whitebox::Ops::NN::CheckNonContiguousShapeSupport(info));
}

TEST_F(MatmulUtilWhiteboxTest, CheckNonContiguousShapeSupport_NonFp32)
{
    auto info = MakeNonContigMmOpInfo(1536, 256, 8192, 2, false, 8, op::DataType::DT_FLOAT16);
    EXPECT_TRUE(matmul_util_whitebox::Ops::NN::CheckNonContiguousShapeSupport(info));
}

TEST_F(MatmulUtilWhiteboxTest, CheckGemmV3WithAlphaBeta_NotUseFp32Add)
{
    NpuArchManager archManager(NpuArch::DAV_2201);
    EXPECT_FALSE(CheckGemmV3WithAlphaBetaCase(op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT16,
                                              op::DataType::DT_FLOAT16, kCubeMathKeepDtype));
}

TEST_F(MatmulUtilWhiteboxTest, CheckGemmV3WithAlphaBeta_NotDav2201)
{
    NpuArchManager archManager(NpuArch::DAV_1001);
    EXPECT_FALSE(CheckGemmV3WithAlphaBetaCase(op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT16,
                                              op::DataType::DT_FLOAT16, op::USE_FP32_ADD));
}

TEST_F(MatmulUtilWhiteboxTest, CheckGemmV3WithAlphaBeta_AllFp16)
{
    NpuArchManager archManager(NpuArch::DAV_2201);
    EXPECT_TRUE(CheckGemmV3WithAlphaBetaCase(op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT16,
                                             op::DataType::DT_FLOAT16, op::USE_FP32_ADD));
}

TEST_F(MatmulUtilWhiteboxTest, CheckGemmV3WithAlphaBeta_AllBf16)
{
    NpuArchManager archManager(NpuArch::DAV_2201);
    EXPECT_TRUE(CheckGemmV3WithAlphaBetaCase(op::DataType::DT_BF16, op::DataType::DT_BF16, op::DataType::DT_BF16,
                                             op::USE_FP32_ADD));
}

TEST_F(MatmulUtilWhiteboxTest, CheckGemmV3WithAlphaBeta_MixedFp32)
{
    NpuArchManager archManager(NpuArch::DAV_2201);
    EXPECT_FALSE(
        CheckGemmV3WithAlphaBetaCase(op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT16));
    EXPECT_FALSE(
        CheckGemmV3WithAlphaBetaCase(op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16));
    EXPECT_FALSE(
        CheckGemmV3WithAlphaBetaCase(op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT));
}

TEST_F(MatmulUtilWhiteboxTest, CheckGemmV3WithAlphaBeta_MixedBf16)
{
    NpuArchManager archManager(NpuArch::DAV_2201);
    EXPECT_FALSE(CheckGemmV3WithAlphaBetaCase(op::DataType::DT_BF16, op::DataType::DT_BF16, op::DataType::DT_FLOAT16));
    EXPECT_FALSE(CheckGemmV3WithAlphaBetaCase(op::DataType::DT_BF16, op::DataType::DT_FLOAT16, op::DataType::DT_BF16));
    EXPECT_FALSE(CheckGemmV3WithAlphaBetaCase(op::DataType::DT_FLOAT16, op::DataType::DT_BF16, op::DataType::DT_BF16));
}

TEST_F(MatmulUtilWhiteboxTest, IsSliceNonContiguous_EarlyRejects)
{
    {
        NpuArchManager archManager(NpuArch::DAV_2201);
        EXPECT_FALSE(IsSliceCase({4, 64}, {4, 32}, op::Strides{64, 1}));
    }
    {
        NpuArchManager archManager(NpuArch::DAV_3510);
        EXPECT_FALSE(IsSliceCase({4, 32}, {4, 32}, op::Strides{32, 1}));
        EXPECT_FALSE(IsSliceCase({4, 64}, {4, 32}, op::Strides{64, 1}, op::DataType::DT_FLOAT16,
                                 op::DataType::DT_FLOAT16, op::Format::FORMAT_FRACTAL_NZ));
        EXPECT_FALSE(IsSliceCase({4, 64}, {4, 32}, op::Strides{64, 1}, op::DataType::DT_FLOAT16,
                                 op::DataType::DT_FLOAT16, op::Format::FORMAT_ND, kCubeMathKeepDtype, 0,
                                 op::Format::FORMAT_FRACTAL_NZ));
        EXPECT_FALSE(IsSliceCase({4, 64}, {4, 32}, op::Strides{64, 1}, op::DataType::DT_FLOAT16,
                                 op::DataType::DT_FLOAT16, op::Format::FORMAT_ND, op::USE_FP32_ADD));
        EXPECT_FALSE(
            IsSliceCase({4, 64}, {4, 32}, op::Strides{64, 1}, op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT));
        EXPECT_FALSE(IsSliceCase({2, 2, 8, 8}, {2, 2, 4, 8}, op::Strides{128, 64, 8, 1}));
    }
}

TEST_F(MatmulUtilWhiteboxTest, IsSliceNonContiguous_StrideChecks)
{
    NpuArchManager archManager(NpuArch::DAV_3510);
    EXPECT_FALSE(IsSliceCase({4, 64}, {4, 32}, op::Strides{64, 0}));
    EXPECT_FALSE(IsSliceCase({4, 64}, {4, 1}, op::Strides{64, 1}));
    EXPECT_FALSE(IsSliceCase({32, 64}, {4, 32}, op::Strides{1, 64}));
    EXPECT_FALSE(IsSliceCase({32, 64}, {4, 32}, op::Strides{6, 4}));
    EXPECT_FALSE(IsSliceCase({64, 64}, {4, 16}, op::Strides{8, 1}));
    EXPECT_FALSE(IsSliceCase({4, 64}, {4, 32}, op::Strides{64, 2}));
}

TEST_F(MatmulUtilWhiteboxTest, IsSliceNonContiguous_ValidCases)
{
    NpuArchManager archManager(NpuArch::DAV_3510);
    EXPECT_TRUE(IsSliceCase({4, 64}, {4, 32}, op::Strides{64, 1}));
    EXPECT_TRUE(IsSliceCase({2, 4, 64}, {2, 4, 32}, op::Strides{256, 64, 1}));
    EXPECT_TRUE(IsSliceCase({2, 32, 32}, {2, 16, 32}, op::Strides{1024, 32, 1}));
}

TEST_F(MatmulUtilWhiteboxTest, IsSliceNonContiguous_ValidateRejects)
{
    NpuArchManager archManager(NpuArch::DAV_3510);
    EXPECT_FALSE(IsSliceCase({2, 32, 32}, {2, 6, 32}, op::Strides{1024, 32, 1}));
    EXPECT_FALSE(IsSliceCase({4, 64}, {4, 16}, op::Strides{64, 1}));
    EXPECT_FALSE(IsSliceCase({4, 57}, {4, 32}, op::Strides{64, 1}));
    EXPECT_FALSE(IsSliceCase({4, 4, 64}, {2, 4, 32}, op::Strides{256, 64, 1}));
    EXPECT_FALSE(IsSliceCase({3, 64}, {4, 32}, op::Strides{64, 1}));
}

TEST_F(MatmulUtilWhiteboxTest, IsCapableForMultiMul_Rejects)
{
    {
        NpuArchManager archManager(NpuArch::DAV_3510);
        EXPECT_FALSE(IsCapableContiguousCase({4, 8}, {8, 4}, op::DataType::DT_FLOAT, op::DataType::DT_FLOAT,
                                             op::Format::FORMAT_ND, op::Format::FORMAT_ND, false, kCubeMathKeepDtype));
        EXPECT_FALSE(IsCapableContiguousCase({4, 8}, {8, 4}, op::DataType::DT_FLOAT, op::DataType::DT_FLOAT,
                                             op::Format::FORMAT_ND, op::Format::FORMAT_ND, true));
        EXPECT_FALSE(IsCapableContiguousCase({0, 8}, {8, 4}));
        EXPECT_FALSE(IsCapableContiguousCase({4, 8}, {8, 0}));
        EXPECT_FALSE(IsCapableContiguousCase({2, 4, 8}, {8, 4}));
        EXPECT_FALSE(IsCapableContiguousCase({4, 8}, {2, 8, 4}));
        EXPECT_FALSE(IsCapableContiguousCase({4, 8}, {8, 4}, op::DataType::DT_FLOAT, op::DataType::DT_FLOAT,
                                             op::Format::FORMAT_FRACTAL_NZ, op::Format::FORMAT_ND));
        EXPECT_FALSE(IsCapableContiguousCase({4, 8}, {8, 4}, op::DataType::DT_FLOAT, op::DataType::DT_FLOAT,
                                             op::Format::FORMAT_ND, op::Format::FORMAT_FRACTAL_NZ));
        EXPECT_FALSE(IsCapableContiguousCase({4, 8}, {8, 4}, op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT));
        EXPECT_FALSE(IsCapableContiguousCase({4, 8}, {8, 4}, op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16));
        EXPECT_FALSE(IsCapableContiguousCase({1, 8}, {8, 4}));
        EXPECT_FALSE(IsCapableContiguousCase({4, 8}, {8, 1}));
        EXPECT_FALSE(IsCapableContiguousCase({4, 1}, {1, 4}));
    }
    {
        NpuArchManager archManager(NpuArch::DAV_2201);
        EXPECT_FALSE(IsCapableContiguousCase({4, 8}, {8, 4}));
    }
}

TEST_F(MatmulUtilWhiteboxTest, IsCapableForMultiMul_True)
{
    NpuArchManager archManager(NpuArch::DAV_3510);
    EXPECT_TRUE(IsCapableContiguousCase({4, 8}, {8, 4}));
}

TEST_F(MatmulUtilWhiteboxTest, IsCapableForMultiMul_Transposed_True)
{
    NpuArchManager archManager(NpuArch::DAV_3510);
    EXPECT_TRUE(IsCapableViewCase({4, 8}, {8, 4}, op::Strides{1, 8}, {8, 4}, {4, 8}, op::Strides{1, 4}));
}

TEST_F(MatmulUtilWhiteboxTest, L0_MatMulNdFp162Fp32)
{
    auto uniqueExecutor = CREATE_EXECUTOR();
    ASSERT_NE(uniqueExecutor.get(), nullptr);
    auto* executor = uniqueExecutor.get();
    const aclTensor* x1 = MakeTensor(executor, {64, 128}, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND);
    const aclTensor* x2 = MakeTensor(executor, {128, 64}, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND);
    ASSERT_NE(x1, nullptr);
    ASSERT_NE(x2, nullptr);
    EXPECT_NE(matmul_util_whitebox::l0op::MatMulNdFp162Fp32(x1, x2, nullptr, nullptr, false, false, false, 0, executor),
              nullptr);
}

TEST_F(MatmulUtilWhiteboxTest, L0_MatMulNdFp162Fp32Adj)
{
    auto uniqueExecutor = CREATE_EXECUTOR();
    ASSERT_NE(uniqueExecutor.get(), nullptr);
    auto* executor = uniqueExecutor.get();
    const aclTensor* x1 = MakeTensor(executor, {128, 64}, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND);
    const aclTensor* x2 = MakeTensor(executor, {64, 128}, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND);
    ASSERT_NE(x1, nullptr);
    ASSERT_NE(x2, nullptr);
    EXPECT_NE(matmul_util_whitebox::l0op::MatMulNdFp162Fp32(x1, x2, nullptr, nullptr, true, true, false, 0, executor),
              nullptr);
}

TEST_F(MatmulUtilWhiteboxTest, L0_MatMulNzNzNd)
{
    auto uniqueExecutor = CREATE_EXECUTOR();
    ASSERT_NE(uniqueExecutor.get(), nullptr);
    auto* executor = uniqueExecutor.get();
    const aclTensor* x1 = MakeTensor(executor, {64, 128}, op::DataType::DT_FLOAT16, op::Format::FORMAT_FRACTAL_NZ);
    const aclTensor* x2 = MakeTensor(executor, {128, 64}, op::DataType::DT_FLOAT16, op::Format::FORMAT_FRACTAL_NZ);
    ASSERT_NE(x1, nullptr);
    ASSERT_NE(x2, nullptr);
    EXPECT_NE(matmul_util_whitebox::l0op::MatMulNzNzNd(x1, x2, nullptr, nullptr, false, false, false, 0, executor),
              nullptr);
}

TEST_F(MatmulUtilWhiteboxTest, L0_MatMulNzNzNdAdj)
{
    auto uniqueExecutor = CREATE_EXECUTOR();
    ASSERT_NE(uniqueExecutor.get(), nullptr);
    auto* executor = uniqueExecutor.get();
    const aclTensor* x1 = MakeTensor(executor, {128, 64}, op::DataType::DT_FLOAT16, op::Format::FORMAT_FRACTAL_NZ);
    const aclTensor* x2 = MakeTensor(executor, {64, 128}, op::DataType::DT_FLOAT16, op::Format::FORMAT_FRACTAL_NZ);
    ASSERT_NE(x1, nullptr);
    ASSERT_NE(x2, nullptr);
    EXPECT_NE(matmul_util_whitebox::l0op::MatMulNzNzNd(x1, x2, nullptr, nullptr, true, true, false, 0, executor),
              nullptr);
}

TEST_F(MatmulUtilWhiteboxTest, L0_MatMulV3NzNzNd)
{
    auto uniqueExecutor = CREATE_EXECUTOR();
    ASSERT_NE(uniqueExecutor.get(), nullptr);
    auto* executor = uniqueExecutor.get();
    const aclTensor* x1 = MakeTensor(executor, {64, 128}, op::DataType::DT_FLOAT16, op::Format::FORMAT_FRACTAL_NZ);
    const aclTensor* x2 = MakeTensor(executor, {128, 64}, op::DataType::DT_FLOAT16, op::Format::FORMAT_FRACTAL_NZ);
    ASSERT_NE(x1, nullptr);
    ASSERT_NE(x2, nullptr);
    EXPECT_NE(matmul_util_whitebox::l0op::MatMulV3NzNzNd(x1, x2, nullptr, false, false, false, 0, executor), nullptr);
}

TEST_F(MatmulUtilWhiteboxTest, L0_MatMulV3NzNzNdAdj)
{
    auto uniqueExecutor = CREATE_EXECUTOR();
    ASSERT_NE(uniqueExecutor.get(), nullptr);
    auto* executor = uniqueExecutor.get();
    const aclTensor* x1 = MakeTensor(executor, {128, 64}, op::DataType::DT_FLOAT16, op::Format::FORMAT_FRACTAL_NZ);
    const aclTensor* x2 = MakeTensor(executor, {64, 128}, op::DataType::DT_FLOAT16, op::Format::FORMAT_FRACTAL_NZ);
    ASSERT_NE(x1, nullptr);
    ASSERT_NE(x2, nullptr);
    EXPECT_NE(matmul_util_whitebox::l0op::MatMulV3NzNzNd(x1, x2, nullptr, true, true, false, 0, executor), nullptr);
}

TEST_F(MatmulUtilWhiteboxTest, L0_MatMulV3NzNzNdFp162Fp32)
{
    auto uniqueExecutor = CREATE_EXECUTOR();
    ASSERT_NE(uniqueExecutor.get(), nullptr);
    auto* executor = uniqueExecutor.get();
    const aclTensor* x1 = MakeTensor(executor, {64, 128}, op::DataType::DT_FLOAT16, op::Format::FORMAT_FRACTAL_NZ);
    const aclTensor* x2 = MakeTensor(executor, {128, 64}, op::DataType::DT_FLOAT16, op::Format::FORMAT_FRACTAL_NZ);
    ASSERT_NE(x1, nullptr);
    ASSERT_NE(x2, nullptr);
    EXPECT_NE(matmul_util_whitebox::l0op::MatMulV3NzNzNdFp162Fp32(x1, x2, nullptr, false, false, false, 0, executor),
              nullptr);
}

TEST_F(MatmulUtilWhiteboxTest, L0_MatMulV3NzNzNdFp162Fp32Adj)
{
    auto uniqueExecutor = CREATE_EXECUTOR();
    ASSERT_NE(uniqueExecutor.get(), nullptr);
    auto* executor = uniqueExecutor.get();
    const aclTensor* x1 = MakeTensor(executor, {128, 64}, op::DataType::DT_FLOAT16, op::Format::FORMAT_FRACTAL_NZ);
    const aclTensor* x2 = MakeTensor(executor, {64, 128}, op::DataType::DT_FLOAT16, op::Format::FORMAT_FRACTAL_NZ);
    ASSERT_NE(x1, nullptr);
    ASSERT_NE(x2, nullptr);
    EXPECT_NE(matmul_util_whitebox::l0op::MatMulV3NzNzNdFp162Fp32(x1, x2, nullptr, true, true, false, 0, executor),
              nullptr);
}

TEST_F(l2_matmul_test, ascend960dt_matmul_fp16_basic)
{
    NpuArchManager archManager(NpuArch::DAV_9201);
    TensorDesc a_desc = TensorDesc({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc b_desc = TensorDesc({32, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    MatMulCommonTest(a_desc, b_desc, out_desc, ACL_SUCCESS);
}

TEST_F(l2_matmul_test, ascend960dt_matmul_fp16_trans)
{
    NpuArchManager archManager(NpuArch::DAV_9201);
    TensorDesc a_t_desc = TensorDesc({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND, {1, 16}, 0, {32, 16});
    TensorDesc b_desc = TensorDesc({32, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    MatMulCommonTest(a_t_desc, b_desc, out_desc, ACL_SUCCESS);
}

TEST_F(l2_matmul_test, ascend960dt_matmul_fp32_splitk)
{
    NpuArchManager archManager(NpuArch::DAV_9201);
    TensorDesc a_desc = TensorDesc({16, 128}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc b_desc = TensorDesc({128, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({16, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    MatMulCommonTest(a_desc, b_desc, out_desc, ACL_SUCCESS);
}

TEST_F(l2_matmul_test, ascend960dt_matmul_bf16)
{
    NpuArchManager archManager(NpuArch::DAV_9201);
    TensorDesc a_desc = TensorDesc({16, 32}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc b_desc = TensorDesc({32, 16}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({16, 16}, ACL_BF16, ACL_FORMAT_ND);
    MatMulCommonTest(a_desc, b_desc, out_desc, ACL_SUCCESS);
}

TEST_F(l2_matmul_test, ascend960dt_matmul_empty_tensor)
{
    NpuArchManager archManager(NpuArch::DAV_9201);
    TensorDesc a_desc = TensorDesc({0, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc b_desc = TensorDesc({32, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({0, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    MatMulCommonTest(a_desc, b_desc, out_desc, ACL_SUCCESS);
}

TEST_F(l2_matmul_test, ascend960dt_matmul_invalid_dtype)
{
    NpuArchManager archManager(NpuArch::DAV_9201);
    TensorDesc a_desc = TensorDesc({16, 32}, ACL_BOOL, ACL_FORMAT_ND);
    TensorDesc b_desc = TensorDesc({32, 16}, ACL_BOOL, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({16, 16}, ACL_BOOL, ACL_FORMAT_ND);
    MatMulCommonTest(a_desc, b_desc, out_desc, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_matmul_test, ascend960pr_matmul_fp16_basic)
{
    NpuArchManager archManager(NpuArch::DAV_9202);
    TensorDesc a_desc = TensorDesc({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc b_desc = TensorDesc({32, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    MatMulCommonTest(a_desc, b_desc, out_desc, ACL_SUCCESS);
}

TEST_F(l2_matmul_test, ascend960pr_matmul_fp16_trans)
{
    NpuArchManager archManager(NpuArch::DAV_9202);
    TensorDesc a_t_desc = TensorDesc({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND, {1, 16}, 0, {32, 16});
    TensorDesc b_desc = TensorDesc({32, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    MatMulCommonTest(a_t_desc, b_desc, out_desc, ACL_SUCCESS);
}

TEST_F(l2_matmul_test, ascend960pr_matmul_fp32_splitk)
{
    NpuArchManager archManager(NpuArch::DAV_9202);
    TensorDesc a_desc = TensorDesc({16, 128}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc b_desc = TensorDesc({128, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({16, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    MatMulCommonTest(a_desc, b_desc, out_desc, ACL_SUCCESS);
}

TEST_F(l2_matmul_test, ascend960pr_matmul_bf16)
{
    NpuArchManager archManager(NpuArch::DAV_9202);
    TensorDesc a_desc = TensorDesc({16, 32}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc b_desc = TensorDesc({32, 16}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({16, 16}, ACL_BF16, ACL_FORMAT_ND);
    MatMulCommonTest(a_desc, b_desc, out_desc, ACL_SUCCESS);
}

TEST_F(l2_matmul_test, ascend960pr_matmul_empty_tensor)
{
    NpuArchManager archManager(NpuArch::DAV_9202);
    TensorDesc a_desc = TensorDesc({0, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc b_desc = TensorDesc({32, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({0, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    MatMulCommonTest(a_desc, b_desc, out_desc, ACL_SUCCESS);
}

TEST_F(l2_matmul_test, ascend960pr_matmul_invalid_dtype)
{
    NpuArchManager archManager(NpuArch::DAV_9202);
    TensorDesc a_desc = TensorDesc({16, 32}, ACL_BOOL, ACL_FORMAT_ND);
    TensorDesc b_desc = TensorDesc({32, 16}, ACL_BOOL, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({16, 16}, ACL_BOOL, ACL_FORMAT_ND);
    MatMulCommonTest(a_desc, b_desc, out_desc, ACLNN_ERR_PARAM_INVALID);
}

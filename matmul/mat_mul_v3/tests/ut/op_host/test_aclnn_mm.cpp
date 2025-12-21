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

#include "../../../op_host/op_api/aclnn_mm.h"
#include "op_api/op_api_def.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace std;
using namespace op;

class l2_mm_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << "l2_mm_test SetUp" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "l2_mm_test TearDown" << endl;
    }
    static void MmCommonTest(TensorDesc a_desc, TensorDesc b_desc, TensorDesc out_desc, aclnnStatus expect_status)
    {
        int8_t cubeMathType = ALLOW_FP32_DOWN_PRECISION;
        auto ut = OP_API_UT(
            aclnnMm,               // host api第二段接口名称
            INPUT(a_desc, b_desc), // host api输入
            OUTPUT(out_desc),
            cubeMathType); // host api输出

        // SAMPLE: only test GetWorkspaceSize
        uint64_t workspace_size = 0;
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size); // check op graph
        EXPECT_EQ(aclRet, expect_status);
        // SAMPLE: precision simulate
        // if (expect_status == ACL_SUCCESS){
        // ut.TestPrecision();  // soc version  2. 二段接口
        // }
    }
};

TEST_F(l2_mm_test, test_aligned_fp16_false_false)
{
    // 使用**Desc描述host api输入输出
    TensorDesc a_desc = TensorDesc({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc b_desc = TensorDesc({32, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.005, 0.005);
    MmCommonTest(a_desc, b_desc, out_desc, ACL_SUCCESS);
}

TEST_F(l2_mm_test, ascend910B2_test_aligned_fp16_false_false)
{
    // 使用**Desc描述host api输入输出
    TensorDesc a_desc = TensorDesc({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc b_desc = TensorDesc({32, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.005, 0.005);
    MmCommonTest(a_desc, b_desc, out_desc, ACL_SUCCESS);
}

TEST_F(l2_mm_test, test_aligned_fp16_true_false)
{
    // 使用**Desc描述host api输入输出
    TensorDesc a_t_desc = TensorDesc({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND, {1, 16}, 0, {32, 16});
    TensorDesc b_desc = TensorDesc({32, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.005, 0.005);
    cout << "-----------Testing trans_x1 = ture trans_x2 = false ----------" << endl;
    MmCommonTest(a_t_desc, b_desc, out_desc, ACL_SUCCESS);
}

TEST_F(l2_mm_test, test_aligned_fp16_false_true)
{
    // 使用**Desc描述host api输入输出
    TensorDesc a_desc = TensorDesc({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc b_t_desc = TensorDesc({32, 16}, ACL_FLOAT16, ACL_FORMAT_ND, {1, 32}, 0, {16, 32});
    TensorDesc out_desc = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.005, 0.005);
    cout << "-----------Testing trans_x1 = false trans_x2 = true ----------" << endl;
    MmCommonTest(a_desc, b_t_desc, out_desc, ACL_SUCCESS);
}

TEST_F(l2_mm_test, test_aligned_fp16_true_true)
{
    // 使用**Desc描述host api输入输出
    TensorDesc a_t_desc = TensorDesc({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND, {1, 16}, 0, {32, 16});
    TensorDesc b_t_desc = TensorDesc({32, 16}, ACL_FLOAT16, ACL_FORMAT_ND, {1, 32}, 0, {16, 32});
    TensorDesc out_desc = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.005, 0.005);
    cout << "-----------Testing trans_x1 = true trans_x2 = ture ----------" << endl;
    MmCommonTest(a_t_desc, b_t_desc, out_desc, ACL_SUCCESS);
}

TEST_F(l2_mm_test, test_aligned_fp32_false_false)
{
    // 使用**Desc描述host api输入输出
    TensorDesc a_desc = TensorDesc({16, 32}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2).ValueRange(0, 2);
    TensorDesc b_desc = TensorDesc({32, 16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2).ValueRange(0, 2);
    TensorDesc out_desc = TensorDesc({16, 16}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.005, 0.005);
    cout << "-----------Testing trans_x1 = false trans_x2 = false ----------" << endl;
    MmCommonTest(a_desc, b_desc, out_desc, ACL_SUCCESS);
}

TEST_F(l2_mm_test, test_aligned_fp32_true_false)
{
    // 使用**Desc描述host api输入输出
    TensorDesc a_t_desc = TensorDesc({16, 32}, ACL_FLOAT, ACL_FORMAT_ND, {1, 16}, 0, {32, 16}).ValueRange(0, 2);
    TensorDesc b_desc = TensorDesc({32, 16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2).ValueRange(0, 2);
    TensorDesc out_desc = TensorDesc({16, 16}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.005, 0.005);
    cout << "-----------Testing trans_x1 = true trans_x2 = false ----------" << endl;
    MmCommonTest(a_t_desc, b_desc, out_desc, ACL_SUCCESS);
}
TEST_F(l2_mm_test, test_aligned_fp32_false_true)
{
    // 使用**Desc描述host api输入输出
    TensorDesc a_desc = TensorDesc({16, 32}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2).ValueRange(0, 2);
    TensorDesc b_t_desc = TensorDesc({32, 16}, ACL_FLOAT, ACL_FORMAT_ND, {1, 32}, 0, {16, 32}).ValueRange(0, 2);
    TensorDesc out_desc = TensorDesc({16, 16}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.05, 0.05);
    cout << "-----------Testing trans_x1 = false trans_x2 = true ----------" << endl;
    MmCommonTest(a_desc, b_t_desc, out_desc, ACL_SUCCESS);
}
TEST_F(l2_mm_test, test_aligned_fp32_true_true)
{
    // 使用**Desc描述host api输入输出
    TensorDesc a_desc = TensorDesc({16, 32}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2).ValueRange(0, 2);
    TensorDesc a_t_desc = TensorDesc({16, 32}, ACL_FLOAT, ACL_FORMAT_ND, {1, 16}, 0, {32, 16}).ValueRange(0, 2);
    TensorDesc b_desc = TensorDesc({32, 16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2).ValueRange(0, 2);
    TensorDesc b_t_desc = TensorDesc({32, 16}, ACL_FLOAT, ACL_FORMAT_ND, {1, 32}, 0, {16, 32}).ValueRange(0, 2);
    TensorDesc out_desc = TensorDesc({16, 16}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.005, 0.005);
    cout << "-----------Testing trans_x1 = true trans_x2 = true ----------" << endl;
    MmCommonTest(a_t_desc, b_t_desc, out_desc, ACL_SUCCESS);
}

TEST_F(l2_mm_test, test_no_aligned_fp16_false_false)
{
    // 使用**Desc描述host api输入输出
    TensorDesc a_desc = TensorDesc({8, 16}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    TensorDesc b_desc = TensorDesc({16, 8}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    TensorDesc out_desc = TensorDesc({8, 8}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.005, 0.005);
    cout << "-----------Testing trans_x1 = false trans_x2 = false ----------" << endl;
    MmCommonTest(a_desc, b_desc, out_desc, ACL_SUCCESS);
}

TEST_F(l2_mm_test, ascend910B2_test_no_aligned_fp16_false_false)
{
    // 使用**Desc描述host api输入输出
    TensorDesc a_desc = TensorDesc({8, 16}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    TensorDesc b_desc = TensorDesc({16, 8}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    TensorDesc out_desc = TensorDesc({8, 8}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.005, 0.005);
    cout << "-----------Testing trans_x1 = false trans_x2 = false ----------" << endl;
    MmCommonTest(a_desc, b_desc, out_desc, ACL_SUCCESS);
}

TEST_F(l2_mm_test, test_no_aligned_fp16_true_false)
{
    // 使用**Desc描述host api输入输出
    TensorDesc a_t_desc = TensorDesc({8, 16}, ACL_FLOAT16, ACL_FORMAT_ND, {1, 8}, 0, {16, 8}).ValueRange(0, 2);
    TensorDesc b_desc = TensorDesc({16, 8}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    TensorDesc out_desc = TensorDesc({8, 8}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.005, 0.005);
    cout << "-----------Testing trans_x1 = true trans_x2 = false ----------" << endl;
    MmCommonTest(a_t_desc, b_desc, out_desc, ACL_SUCCESS);
}

TEST_F(l2_mm_test, test_no_aligned_fp16_false_true)
{
    // 使用**Desc描述host api输入输出
    TensorDesc a_desc = TensorDesc({8, 16}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    TensorDesc b_t_desc = TensorDesc({16, 8}, ACL_FLOAT16, ACL_FORMAT_ND, {1, 16}, 0, {8, 16}).ValueRange(0, 2);
    TensorDesc out_desc = TensorDesc({8, 8}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.005, 0.005);
    cout << "-----------Testing trans_x1 = false trans_x2 = true ----------" << endl;
    MmCommonTest(a_desc, b_t_desc, out_desc, ACL_SUCCESS);
}

TEST_F(l2_mm_test, test_no_aligned_fp16_true_true)
{
    // 使用**Desc描述host api输入输出
    TensorDesc a_t_desc = TensorDesc({8, 16}, ACL_FLOAT16, ACL_FORMAT_ND, {1, 8}, 0, {16, 8}).ValueRange(0, 2);
    TensorDesc b_t_desc = TensorDesc({16, 8}, ACL_FLOAT16, ACL_FORMAT_ND, {1, 16}, 0, {8, 16}).ValueRange(0, 2);
    TensorDesc out_desc = TensorDesc({8, 8}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.005, 0.005);
    cout << "-----------Testing trans_x1 = true trans_x2 = true ----------" << endl;
    MmCommonTest(a_t_desc, b_t_desc, out_desc, ACL_SUCCESS);
}

TEST_F(l2_mm_test, ascend910B2_mmv3_test_fp16_false_false)
{
    // 使用**Desc描述host api输入输出
    TensorDesc a_desc = TensorDesc({20480, 6656}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    TensorDesc b_desc = TensorDesc({6656, 8192}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    TensorDesc out_desc = TensorDesc({20480, 8192}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.005, 0.005);
    cout << "-----------Testing MMV3 trans_x1 = false trans_x2 = false ----------" << endl;
    MmCommonTest(a_desc, b_desc, out_desc, ACL_SUCCESS);
}

// TEST_F(l2_mm_test, ascend910B2_mmv3_ndnz_fp32_fp32_fp32)
// {
//     // 使用**Desc描述host api输入输出
//     TensorDesc a_desc = TensorDesc({1152, 640}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
//     TensorDesc b_desc = TensorDesc({640, 1152}, ACL_FLOAT, ACL_FORMAT_FRACTAL_NZ).ValueRange(-1, 1);
//     TensorDesc out_desc = TensorDesc({1152, 1152}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.005, 0.005);
//     cout << "-----------Testing MMV3_B2 fp32 NDNZ ----------" << endl;
//     MmCommonTest(a_desc, b_desc, out_desc, ACLNN_ERR_INNER_NULLPTR);
// }

TEST_F(l2_mm_test, ascend910B2_mmv3_ndnz_bf16_bf16_bf16)
{
    // 使用**Desc描述host api输入输出
    TensorDesc a_desc = TensorDesc({1152, 640}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc b_desc = TensorDesc({640, 1152}, ACL_BF16, ACL_FORMAT_FRACTAL_NZ).ValueRange(-1, 1);
    TensorDesc out_desc = TensorDesc({1152, 1152}, ACL_BF16, ACL_FORMAT_ND).Precision(0.005, 0.005);
    cout << "-----------Testing MMV3_B2 bf16 NDNZ ----------" << endl;
    MmCommonTest(a_desc, b_desc, out_desc, ACL_SUCCESS);
}

TEST_F(l2_mm_test, ascend910B2_mmv3_ndnz_fp16_fp16_fp16)
{
    // 使用**Desc描述host api输入输出
    TensorDesc a_desc = TensorDesc({1152, 640}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc b_desc = TensorDesc({640, 1152}, ACL_FLOAT16, ACL_FORMAT_FRACTAL_NZ).ValueRange(-1, 1);
    TensorDesc out_desc = TensorDesc({1152, 1152}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.005, 0.005);
    cout << "-----------Testing MMV3_B2 fp16 NDNZ ----------" << endl;
    MmCommonTest(a_desc, b_desc, out_desc, ACL_SUCCESS);
}

TEST_F(l2_mm_test, ascend910B4_mmv3_test_fp16_false_false)
{
    // 使用**Desc描述host api输入输出
    TensorDesc a_desc = TensorDesc({2560, 6656}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    TensorDesc b_desc = TensorDesc({6656, 8192}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    TensorDesc out_desc = TensorDesc({2560, 8192}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.005, 0.005);
    cout << "-----------Testing MMV3_B4 trans_x1 = false trans_x2 = false ----------" << endl;
    MmCommonTest(a_desc, b_desc, out_desc, ACL_SUCCESS);
}

TEST_F(l2_mm_test, ascend910B4_mmv3ndnz_test_fp16_false_false)
{
    // 使用**Desc描述host api输入输出
    TensorDesc a_desc = TensorDesc({1, 8192}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    TensorDesc b_desc =
        TensorDesc({8192, 5120}, ACL_FLOAT16, ACL_FORMAT_FRACTAL_NZ, {5120, 1}, 0, {320, 80, 16, 16}).ValueRange(0, 2);
    TensorDesc out_desc = TensorDesc({1, 5120}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.005, 0.005);
    cout << "-----------Testing MMV3_B4_NDNZ trans_x1 = false trans_x2 = false ----------" << endl;
    int8_t cubeMathType = ALLOW_FP32_DOWN_PRECISION;
    auto ut = OP_API_UT(
        aclnnMm,               // host api第二段接口名称
        INPUT(a_desc, b_desc), // host api输入
        OUTPUT(out_desc),
        cubeMathType); // host api输出
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
}

TEST_F(l2_mm_test, ascend910B2_mmv3_splitK_fp16_false_false)
{
    // 使用**Desc描述host api输入输出
    TensorDesc a_desc = TensorDesc({1024, 27392}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    TensorDesc b_desc = TensorDesc({27392, 2048}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    TensorDesc out_desc = TensorDesc({1024, 2048}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.005, 0.005);
    cout << "-----------Testing MMV3_splitK  trans_x1 = false trans_x2 = false ----------" << endl;
    MmCommonTest(a_desc, b_desc, out_desc, ACL_SUCCESS);
}

TEST_F(l2_mm_test, test_split_k_ND)
{
    // 使用**Desc描述host api输入输出
    TensorDesc a_desc = TensorDesc({16, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc b_desc = TensorDesc({128, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.005, 0.005);
    MmCommonTest(a_desc, b_desc, out_desc, ACL_SUCCESS);
}

TEST_F(l2_mm_test, test_split_k_NZ)
{
    // 使用**Desc描述host api输入输出
    TensorDesc a_desc = TensorDesc({7, 75765}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc b_desc = TensorDesc({75765, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({7, 3}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.005, 0.005);
    MmCommonTest(a_desc, b_desc, out_desc, ACL_SUCCESS);
}

TEST_F(l2_mm_test, test_split_k_NZ_2)
{
    // 使用**Desc描述host api输入输出
    TensorDesc a_desc = TensorDesc({128, 75765}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc b_desc = TensorDesc({75765, 129}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({128, 129}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    MmCommonTest(a_desc, b_desc, out_desc, ACL_SUCCESS);
}

TEST_F(l2_mm_test, ascend910B2_test_split_k_ND)
{
    // 使用**Desc描述host api输入输出
    TensorDesc a_desc = TensorDesc({16, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc b_desc = TensorDesc({128, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.005, 0.005);
    MmCommonTest(a_desc, b_desc, out_desc, ACL_SUCCESS);
}

TEST_F(l2_mm_test, test_transpsoe_fail)
{
    // 使用**Desc描述host api输入输出
    TensorDesc a_desc = TensorDesc({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc b_desc = TensorDesc({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    MmCommonTest(a_desc, b_desc, out_desc, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_mm_test, test_invalid_dtype)
{
    // 使用**Desc描述host api输入输出
    TensorDesc a_desc = TensorDesc({16, 32}, ACL_BOOL, ACL_FORMAT_ND);
    TensorDesc b_desc = TensorDesc({16, 32}, ACL_BOOL, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({16, 16}, ACL_BOOL, ACL_FORMAT_ND);
    MmCommonTest(a_desc, b_desc, out_desc, ACLNN_ERR_PARAM_INVALID);

    TensorDesc a2_desc = TensorDesc({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc b2_desc = TensorDesc({16, 32}, ACL_BOOL, ACL_FORMAT_ND);
    TensorDesc out2_desc = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    MmCommonTest(a2_desc, b2_desc, out2_desc, ACLNN_ERR_PARAM_INVALID);

    TensorDesc a3_desc = TensorDesc({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc b3_desc = TensorDesc({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc out3_desc = TensorDesc({16, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    MmCommonTest(a3_desc, b3_desc, out3_desc, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_mm_test, test_invalid_shape)
{
    // 使用**Desc描述host api输入输出
    TensorDesc a_desc = TensorDesc({10, 16, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc b_desc = TensorDesc({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    MmCommonTest(a_desc, b_desc, out_desc, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_mm_test, test_invalid_k_dim_shape)
{
    // 使用**Desc描述host api输入输出
    TensorDesc a_desc = TensorDesc({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc b_desc = TensorDesc({46, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({16, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    MmCommonTest(a_desc, b_desc, out_desc, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_mm_test, test_promote_dtype_aligned_mast2_float)
{
    // 使用**Desc描述host api输入输出
    TensorDesc a_desc = TensorDesc({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc b_desc = TensorDesc({32, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.005, 0.005);
    MmCommonTest(a_desc, b_desc, out_desc, ACL_SUCCESS);
}

TEST_F(l2_mm_test, test_promote_dtype_aligned_self_float)
{
    // 使用**Desc描述host api输入输出
    TensorDesc a_desc = TensorDesc({16, 32}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc b_desc = TensorDesc({32, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({16, 16}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.005, 0.005);
    MmCommonTest(a_desc, b_desc, out_desc, ACL_SUCCESS);
}

TEST_F(l2_mm_test, test_promote_dtype_no_aligned)
{
    // 使用**Desc描述host api输入输出
    TensorDesc a_desc = TensorDesc({8, 30}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc b_desc = TensorDesc({30, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({8, 8}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.005, 0.005);
    MmCommonTest(a_desc, b_desc, out_desc, ACL_SUCCESS);
}

TEST_F(l2_mm_test, ascend910B2_test_promote_dtype_no_aligned)
{
    // 使用**Desc描述host api输入输出
    TensorDesc a_desc = TensorDesc({8, 30}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc b_desc = TensorDesc({30, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({8, 8}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.005, 0.005);
    MmCommonTest(a_desc, b_desc, out_desc, ACL_SUCCESS);
}

TEST_F(l2_mm_test, test_promote_dtype_split_k)
{
    // 使用**Desc描述host api输入输出
    TensorDesc a_desc = TensorDesc({16, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc b_desc = TensorDesc({256, 32}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.005, 0.005);
    MmCommonTest(a_desc, b_desc, out_desc, ACL_SUCCESS);
}

TEST_F(l2_mm_test, test_null_ptr_input)
{
    // 使用**Desc描述host api输入输出
    TensorDesc b_desc = TensorDesc({16, 32}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    int8_t cubeMathType = ALLOW_FP32_DOWN_PRECISION;
    auto ut = OP_API_UT(
        aclnnMm,                            // host api第二段接口名称
        INPUT((aclTensor*)nullptr, b_desc), // host api输入
        OUTPUT(out_desc),
        cubeMathType); // host api输出
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size); // todo: check op graph
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_mm_test, test_empty_tensor)
{
    // 使用**Desc描述host api输入输出
    TensorDesc a_desc = TensorDesc({0, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc b_desc = TensorDesc({32, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({0, 16}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.005, 0.005);
    MmCommonTest(a_desc, b_desc, out_desc, ACL_SUCCESS);

    TensorDesc a2_desc = TensorDesc({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc b2_desc = TensorDesc({32, 0}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc out2_desc = TensorDesc({16, 0}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.005, 0.005);
    MmCommonTest(a2_desc, b2_desc, out2_desc, ACL_SUCCESS);

    TensorDesc a3_desc = TensorDesc({16, 0}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc b3_desc = TensorDesc({0, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc out3_desc = TensorDesc({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.005, 0.005);
    MmCommonTest(a3_desc, b3_desc, out3_desc, ACL_SUCCESS);
}

TEST_F(l2_mm_test, test_aligned_fp32_false_true_1D_storage_shape)
{
    // 使用**Desc描述host api输入输出
    TensorDesc a_desc = TensorDesc({256, 192}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    TensorDesc b_t_desc = TensorDesc(
                              {192, 576}, ACL_FLOAT, ACL_FORMAT_ND, {1, 192}, 0,
                              {
                                  110592,
                              })
                              .ValueRange(0, 2);
    TensorDesc out_desc = TensorDesc({256, 576}, ACL_FLOAT, ACL_FORMAT_ND);

    int8_t cube_math_type = ALLOW_FP32_DOWN_PRECISION;
    auto ut = OP_API_UT(aclnnMm, INPUT(a_desc, b_t_desc), OUTPUT(out_desc), cube_math_type);

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_mm_test, ascend910B2_test_insert_pad)
{
    // 使用**Desc描述host api输入输出
    TensorDesc a_desc = TensorDesc({16384, 1708}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc b_desc = TensorDesc({1708, 5120}, ACL_FLOAT16, ACL_FORMAT_ND, {1, 1708}, 0, {5120, 1708});
    TensorDesc out_desc = TensorDesc({16384, 5120}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.005, 0.005);
    MmCommonTest(a_desc, b_desc, out_desc, ACL_SUCCESS);
}

TEST_F(l2_mm_test, ascend910B2_test_insert_self_transpose)
{
    // 使用**Desc描述host api输入输出
    TensorDesc a_desc = TensorDesc({3416, 16384}, ACL_FLOAT16, ACL_FORMAT_ND, {1, 3416}, 0, {16384, 3416});
    TensorDesc b_desc = TensorDesc({16384, 5120}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({3416, 5120}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.005, 0.005);
    MmCommonTest(a_desc, b_desc, out_desc, ACL_SUCCESS);
}

TEST_F(l2_mm_test, ascend910B2_test_insert_mat2_transpose)
{
    // 使用**Desc描述host api输入输出
    TensorDesc a_desc = TensorDesc({5120, 16384}, ACL_FLOAT16, ACL_FORMAT_ND, {1, 5120}, 0, {16384, 5120});
    TensorDesc b_desc = TensorDesc({16384, 1708}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({5120, 1708}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.005, 0.005);
    MmCommonTest(a_desc, b_desc, out_desc, ACL_SUCCESS);
}

TEST_F(l2_mm_test, cubeMathType_0_fp16_fp16)
{
    auto tensor_1_desc = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto tensor_2_desc = TensorDesc({3, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out_tensor_desc = TensorDesc({2, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2).Precision(0.005, 0.005);
    int8_t cube_math_type = 0;
    auto ut = OP_API_UT(aclnnMm, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc), cube_math_type);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_mm_test, cubeMathType_1_fp16_fp16)
{
    auto tensor_1_desc = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto tensor_2_desc = TensorDesc({3, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out_tensor_desc = TensorDesc({2, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2).Precision(0.005, 0.005);
    int8_t cube_math_type = 1;
    auto ut = OP_API_UT(aclnnMm, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc), cube_math_type);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_mm_test, cubeMathType_2_fp16_fp16)
{
    auto tensor_1_desc = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto tensor_2_desc = TensorDesc({3, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out_tensor_desc = TensorDesc({2, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2).Precision(0.005, 0.005);
    int8_t cube_math_type = 2;
    auto ut = OP_API_UT(aclnnMm, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc), cube_math_type);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_mm_test, cubeMathType_3_fp16_fp16)
{
    auto tensor_1_desc = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto tensor_2_desc = TensorDesc({3, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out_tensor_desc = TensorDesc({2, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2).Precision(0.005, 0.005);
    int8_t cube_math_type = 3;
    auto ut = OP_API_UT(aclnnMm, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc), cube_math_type);

    uint64_t workspace_size = 3;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// TEST_F(l2_mm_test, cubeMathType_0_fp32_fp16)
// {
//     auto tensor_1_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
//     auto tensor_2_desc = TensorDesc({3, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
//     auto out_tensor_desc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2).Precision(0.005, 0.005);
//     int8_t cube_math_type = 0;
//     auto ut = OP_API_UT(aclnnMm, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc), cube_math_type);
//
//     uint64_t workspace_size = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
//     EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
// }

TEST_F(l2_mm_test, cubeMathType_1_fp32_fp16)
{
    auto tensor_1_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto tensor_2_desc = TensorDesc({3, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out_tensor_desc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2).Precision(0.005, 0.005);
    int8_t cube_math_type = 1;
    auto ut = OP_API_UT(aclnnMm, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc), cube_math_type);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_mm_test, cubeMathType_2_fp32_fp16)
{
    auto tensor_1_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto tensor_2_desc = TensorDesc({3, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out_tensor_desc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2).Precision(0.005, 0.005);
    int8_t cube_math_type = 2;
    auto ut = OP_API_UT(aclnnMm, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc), cube_math_type);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// TEST_F(l2_mm_test, cubeMathType_3_fp32_fp16)
// {
//     auto tensor_1_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
//     auto tensor_2_desc = TensorDesc({3, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
//     auto out_tensor_desc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2).Precision(0.005, 0.005);
//     int8_t cube_math_type = 3;
//     auto ut = OP_API_UT(aclnnMm, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc), cube_math_type);
//
//     uint64_t workspace_size = 3;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
//     EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
// }

// TEST_F(l2_mm_test, cubeMathType_0_fp16_fp32)
// {
//     auto tensor_1_desc = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
//     auto tensor_2_desc = TensorDesc({3, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
//     auto out_tensor_desc = TensorDesc({2, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2).Precision(0.005, 0.005);
//     int8_t cube_math_type = 0;
//     auto ut = OP_API_UT(aclnnMm, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc), cube_math_type);
//
//     uint64_t workspace_size = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
//     EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
// }

TEST_F(l2_mm_test, cubeMathType_1_fp16_fp32)
{
    auto tensor_1_desc = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto tensor_2_desc = TensorDesc({3, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out_tensor_desc = TensorDesc({2, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2).Precision(0.005, 0.005);
    int8_t cube_math_type = 1;
    auto ut = OP_API_UT(aclnnMm, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc), cube_math_type);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_mm_test, cubeMathType_2_fp16_fp32)
{
    auto tensor_1_desc = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto tensor_2_desc = TensorDesc({3, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out_tensor_desc = TensorDesc({2, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2).Precision(0.005, 0.005);
    int8_t cube_math_type = 2;
    auto ut = OP_API_UT(aclnnMm, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc), cube_math_type);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// TEST_F(l2_mm_test, cubeMathType_3_fp16_fp32)
// {
//     auto tensor_1_desc = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
//     auto tensor_2_desc = TensorDesc({3, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
//     auto out_tensor_desc = TensorDesc({2, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2).Precision(0.005, 0.005);
//     int8_t cube_math_type = 3;
//     auto ut = OP_API_UT(aclnnMm, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc), cube_math_type);
//
//     uint64_t workspace_size = 3;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
//     EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
// }

// TEST_F(l2_mm_test, cubeMathType_0_fp32_fp32)
// {
//     auto tensor_1_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
//     auto tensor_2_desc = TensorDesc({3, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
//     auto out_tensor_desc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2).Precision(0.005, 0.005);
//     int8_t cube_math_type = 0;
//     auto ut = OP_API_UT(aclnnMm, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc), cube_math_type);
//
//     uint64_t workspace_size = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
//     EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
// }

TEST_F(l2_mm_test, cubeMathType_1_fp32_fp32)
{
    auto tensor_1_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto tensor_2_desc = TensorDesc({3, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out_tensor_desc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2).Precision(0.005, 0.005);
    int8_t cube_math_type = 1;
    auto ut = OP_API_UT(aclnnMm, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc), cube_math_type);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_mm_test, cubeMathType_2_fp32_fp32)
{
    auto tensor_1_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto tensor_2_desc = TensorDesc({3, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out_tensor_desc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2).Precision(0.005, 0.005);
    int8_t cube_math_type = 2;
    auto ut = OP_API_UT(aclnnMm, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc), cube_math_type);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// TEST_F(l2_mm_test, cubeMathType_3_fp32_fp32)
// {
//     auto tensor_1_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
//     auto tensor_2_desc = TensorDesc({3, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
//     auto out_tensor_desc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2).Precision(0.005, 0.005);
//     int8_t cube_math_type = 3;
//     auto ut = OP_API_UT(aclnnMm, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc), cube_math_type);
//
//     uint64_t workspace_size = 3;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
//     EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
// }

TEST_F(l2_mm_test, cubeMathType_4_fp32_fp32)
{
    auto tensor_1_desc = TensorDesc({256, 2048}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto tensor_2_desc = TensorDesc({2048, 1024}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out_tensor_desc = TensorDesc({256, 1024}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2).Precision(0.005, 0.005);
    int8_t cube_math_type = 4;
    auto ut = OP_API_UT(aclnnMm, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc), cube_math_type);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_mm_test, test_matmul_v3)
{
    TensorDesc a_desc = TensorDesc({10832, 4224}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc b_desc = TensorDesc({4224, 1408}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({10832, 1408}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.005, 0.005);
    MmCommonTest(a_desc, b_desc, out_desc, ACL_SUCCESS);
}

TEST_F(l2_mm_test, test_matmul_v3_float)
{
    TensorDesc a_desc = TensorDesc({10832, 4224}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc b_desc = TensorDesc({4224, 1408}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({10832, 1408}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    MmCommonTest(a_desc, b_desc, out_desc, ACL_SUCCESS);
}

TEST_F(l2_mm_test, test_matmul_v3_float_1)
{
    TensorDesc a_desc = TensorDesc({108, 4224}, ACL_FLOAT, ACL_FORMAT_ND, {1, 108}, 0, {4224, 108});
    TensorDesc b_desc = TensorDesc({4224, 24}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({108, 24}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    MmCommonTest(a_desc, b_desc, out_desc, ACL_SUCCESS);
}

TEST_F(l2_mm_test, test_mat_mul_v3_multik)
{
    TensorDesc a_desc = TensorDesc({1024, 30720}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc b_desc = TensorDesc({30720, 1024}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({1024, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);
    MmCommonTest(a_desc, b_desc, out_desc, ACL_SUCCESS);
}

TEST_F(l2_mm_test, test_mat_mul_v3_multik_1)
{
    TensorDesc a_desc = TensorDesc({1024, 3072}, ACL_FLOAT, ACL_FORMAT_ND, {1, 1024}, 0, {3072, 1024});
    TensorDesc b_desc = TensorDesc({3072, 1024}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({1024, 1024}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    MmCommonTest(a_desc, b_desc, out_desc, ACL_SUCCESS);
}

TEST_F(l2_mm_test, test_mat_mul_v3_multik_2)
{
    TensorDesc a_desc = TensorDesc({1024, 3072}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc b_desc = TensorDesc({3072, 1024}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({1024, 1024}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    MmCommonTest(a_desc, b_desc, out_desc, ACL_SUCCESS);
}

TEST_F(l2_mm_test, test_matmul_v3_n_1)
{
    TensorDesc a_desc = TensorDesc({32, 322222}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc b_desc = TensorDesc({322222, 1}, ACL_FLOAT, ACL_FORMAT_ND, {1, 1}, 0, {1, 322222});
    TensorDesc out_desc = TensorDesc({32, 1}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    MmCommonTest(a_desc, b_desc, out_desc, ACL_SUCCESS);
}

TEST_F(l2_mm_test, test_matmul_v3_n_1_2)
{
    TensorDesc a_desc = TensorDesc({33, 322223}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc b_desc = TensorDesc({322223, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({33, 1}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    MmCommonTest(a_desc, b_desc, out_desc, ACL_SUCCESS);
}

TEST_F(l2_mm_test, test_matmul_k_1)
{
    TensorDesc a_desc = TensorDesc({4224, 1}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc b_desc = TensorDesc({1, 1408}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({4224, 1408}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);
    MmCommonTest(a_desc, b_desc, out_desc, ACL_SUCCESS);
}

TEST_F(l2_mm_test, test_matmul_k_1_oom)
{
    TensorDesc a_desc = TensorDesc({1073741824, 1}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc b_desc = TensorDesc({1, 1}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({1073741824, 1}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);
    MmCommonTest(a_desc, b_desc, out_desc, ACL_SUCCESS);
}

TEST_F(l2_mm_test, test_matmul_m_1_n_1)
{
    TensorDesc a_desc = TensorDesc({1, 4224}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc b_desc = TensorDesc({4224, 1}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({1, 1}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);
    MmCommonTest(a_desc, b_desc, out_desc, ACL_SUCCESS);
}

TEST_F(l2_mm_test, cubeMathType_2_fp16_fp16_fp32)
{
    auto tensor_1_desc = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto tensor_2_desc = TensorDesc({3, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out_tensor_desc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2).Precision(0.005, 0.005);
    int8_t cube_math_type = 2;
    auto ut = OP_API_UT(aclnnMm, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc), cube_math_type);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_mm_test, test_nd2nz_nvchwconv_1)
{
    // 使用**Desc描述host api输入输出
    TensorDesc a_desc = TensorDesc({1500, 88}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 5);
    TensorDesc b_desc = TensorDesc({88, 32}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 5);
    TensorDesc out_desc = TensorDesc({1500, 32}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    MmCommonTest(a_desc, b_desc, out_desc, ACL_SUCCESS);
}

TEST_F(l2_mm_test, test_nd2nz_nvchwconv_2)
{
    // 使用**Desc描述host api输入输出
    TensorDesc a_desc = TensorDesc({8200, 88}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 5);
    TensorDesc b_desc = TensorDesc({88, 32}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 5);
    TensorDesc out_desc = TensorDesc({8200, 32}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    MmCommonTest(a_desc, b_desc, out_desc, ACL_SUCCESS);
}

TEST_F(l2_mm_test, ascend910B2_test_multi_split_k)
{
    // 使用**Desc描述host api输入输出
    TensorDesc a_desc = TensorDesc({49, 191568}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 5);
    TensorDesc b_desc = TensorDesc({191568, 142}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 5);
    TensorDesc out_desc = TensorDesc({49, 142}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    MmCommonTest(a_desc, b_desc, out_desc, ACL_SUCCESS);
}

TEST_F(l2_mm_test, test_multi_smallmn_split_k_1)
{
    // 使用**Desc描述host api输入输出
    TensorDesc a_t_desc = TensorDesc({157, 1440}, ACL_FLOAT, ACL_FORMAT_ND, {1, 157}, 0, {1440, 157});
    TensorDesc b_desc = TensorDesc({1440, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({157, 16}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.005, 0.005);
    cout << "-----------Testing trans_x1 = true trans_x2 = false ----------" << endl;
    MmCommonTest(a_t_desc, b_desc, out_desc, ACL_SUCCESS);
}

TEST_F(l2_mm_test, test_multi_smallmn_split_k_2)
{
    // 使用**Desc描述host api输入输出
    TensorDesc a_t_desc = TensorDesc({212, 2380}, ACL_FLOAT, ACL_FORMAT_ND, {1, 212}, 0, {2380, 212});
    TensorDesc b_desc = TensorDesc({2380, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({212, 16}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.005, 0.005);
    cout << "-----------Testing trans_x1 = true trans_x2 = false ----------" << endl;
    MmCommonTest(a_t_desc, b_desc, out_desc, ACL_SUCCESS);
}

TEST_F(l2_mm_test, test_multi_smallmn_split_k_3)
{
    // 使用**Desc描述host api输入输出
    TensorDesc a_t_desc = TensorDesc({129, 4000}, ACL_FLOAT16, ACL_FORMAT_ND, {1, 129}, 0, {4000, 129});
    TensorDesc b_desc = TensorDesc({4000, 136}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({129, 136}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.005, 0.005);
    cout << "-----------Testing trans_x1 = true trans_x2 = false ----------" << endl;
    MmCommonTest(a_t_desc, b_desc, out_desc, ACL_SUCCESS);
}

TEST_F(l2_mm_test, test_m_from_1_to_256)
{
    // 使用**Desc描述host api输入输出
    TensorDesc a_t_desc = TensorDesc({128, 7168}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc b_desc = TensorDesc({7168, 256}, ACL_FLOAT, ACL_FORMAT_ND, {1, 7168}, 0, {256, 7168});
    TensorDesc out_desc = TensorDesc({128, 256}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.005, 0.005);
    cout << "-----------Testing trans_x1 = false trans_x2 = true ----------" << endl;
    MmCommonTest(a_t_desc, b_desc, out_desc, ACL_SUCCESS);
}

TEST_F(l2_mm_test, test_fix_nd2nz_1)
{
    // 使用**Desc描述host api输入输出
    TensorDesc a_desc = TensorDesc({16024, 124}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 5);
    TensorDesc b_desc = TensorDesc({124, 215}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 5);
    TensorDesc out_desc = TensorDesc({16024, 215}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    MmCommonTest(a_desc, b_desc, out_desc, ACL_SUCCESS);
}

TEST_F(l2_mm_test, test_fix_nd2nz_2)
{
    // 使用**Desc描述host api输入输出
    TensorDesc a_desc = TensorDesc({12366, 223}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 5);
    TensorDesc b_desc = TensorDesc({223, 9}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 5);
    TensorDesc out_desc = TensorDesc({12366, 9}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);
    MmCommonTest(a_desc, b_desc, out_desc, ACL_SUCCESS);
}

TEST_F(l2_mm_test, ascend910B2_test_empty_tensor_cubeMathType_0_fp16_fp32_fp32)
{
    auto tensor_1_desc = TensorDesc({2, 0}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto tensor_2_desc = TensorDesc({0, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out_tensor_desc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2).Precision(0.005, 0.005);
    int8_t cube_math_type = 0;
    auto ut = OP_API_UT(aclnnMm, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc), cube_math_type);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_mm_test, ascend910B2_test_cubeMathType_2_bf16_bf16_bf16)
{
    auto tensor_1_desc = TensorDesc({2, 3}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto tensor_2_desc = TensorDesc({3, 4}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out_tensor_desc = TensorDesc({2, 4}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-2, 2).Precision(0.005, 0.005);
    int8_t cube_math_type = 2;
    auto ut = OP_API_UT(aclnnMm, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc), cube_math_type);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_mm_test, ascend910B2_test_small128_and_align16_inner_axis)
{
    // 使用**Desc描述host api输入输出
    TensorDesc a_desc = TensorDesc({1500, 32}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 5);
    TensorDesc b_desc = TensorDesc({32, 32}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 5);
    TensorDesc out_desc = TensorDesc({1500, 32}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    MmCommonTest(a_desc, b_desc, out_desc, ACL_SUCCESS);
}

TEST_F(l2_mm_test, test_multi_smallmn_split_k_4)
{
    // 使用**Desc描述host api输入输出
    TensorDesc a_t_desc = TensorDesc({1, 7168}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc b_desc = TensorDesc({7168, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({1, 16}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.005, 0.005);
    MmCommonTest(a_t_desc, b_desc, out_desc, ACL_SUCCESS);
}

TEST_F(l2_mm_test, ascend910B2_test_kEqual1_bf16_bf16_bf16)
{
    auto tensor_1_desc = TensorDesc({2, 1}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto tensor_2_desc = TensorDesc({1, 4}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out_tensor_desc = TensorDesc({2, 4}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-2, 2).Precision(0.005, 0.005);
    int8_t cube_math_type = 2;
    auto ut = OP_API_UT(aclnnMm, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc), cube_math_type);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_mm_test, ascend910B2_test_kEqual1_transpose_bf16_bf16_bf16)
{
    auto tensor_1_desc = TensorDesc({2, 1}, ACL_BF16, ACL_FORMAT_ND, {1, 2}, 0, {1, 2}).ValueRange(-2, 2);
    auto tensor_2_desc = TensorDesc({1, 4}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out_tensor_desc = TensorDesc({2, 4}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-2, 2).Precision(0.005, 0.005);
    int8_t cube_math_type = 2;
    auto ut = OP_API_UT(aclnnMm, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc), cube_math_type);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_mm_test, ascend910B2_test_kEqual1_over4000_bf16_bf16_bf16)
{
    auto tensor_1_desc = TensorDesc({2, 1}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto tensor_2_desc = TensorDesc({1, 4500}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out_tensor_desc = TensorDesc({2, 4500}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-2, 2).Precision(0.005, 0.005);
    int8_t cube_math_type = 2;
    auto ut = OP_API_UT(aclnnMm, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc), cube_math_type);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_mm_test, ascend910B2_test_netshape_generalization)
{
    TensorDesc a_desc = TensorDesc({32768, 7168}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc b_desc = TensorDesc({7168, 576}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({32768, 576}, ACL_FLOAT16, ACL_FORMAT_ND);
    MmCommonTest(a_desc, b_desc, out_desc, ACL_SUCCESS);

    TensorDesc a2_desc = TensorDesc({32768, 7168}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc b2_desc = TensorDesc({7168, 256}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc out2_desc = TensorDesc({32768, 256}, ACL_FLOAT, ACL_FORMAT_ND);
    MmCommonTest(a2_desc, b2_desc, out2_desc, ACL_SUCCESS);
}

TEST_F(l2_mm_test, ascend910B2_test_splitk_bf16_bf16_bf16)
{
    auto tensor_1_desc = TensorDesc({2, 27408}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto tensor_2_desc = TensorDesc({27408, 2}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out_tensor_desc = TensorDesc({2, 2}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-2, 2).Precision(0.005, 0.005);
    int8_t cube_math_type = 2;
    auto ut = OP_API_UT(aclnnMm, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc), cube_math_type);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_mm_test, ascend910B2_test_splitk_fp16_fp16_fp16)
{
    auto tensor_1_desc = TensorDesc({2, 27408}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto tensor_2_desc = TensorDesc({27408, 2}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out_tensor_desc = TensorDesc({2, 2}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2).Precision(0.005, 0.005);
    int8_t cube_math_type = 2;
    auto ut = OP_API_UT(aclnnMm, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc), cube_math_type);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_mm_test, ascend910B2_test_splitk_bf16_bf16_bf16_0)
{
    auto tensor_1_desc = TensorDesc({62080, 1536}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto tensor_2_desc = TensorDesc({1536, 384}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out_tensor_desc = TensorDesc({62080, 384}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-2, 2).Precision(0.005, 0.005);
    int8_t cube_math_type = 2;
    auto ut = OP_API_UT(aclnnMm, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc), cube_math_type);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_mm_test, ascend910B2_test_splitk_bf16_bf16_bf16_1)
{
    auto tensor_1_desc = TensorDesc({384, 1536}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto tensor_2_desc = TensorDesc({1536, 62080}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out_tensor_desc = TensorDesc({384, 62080}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-2, 2).Precision(0.005, 0.005);
    int8_t cube_math_type = 2;
    auto ut = OP_API_UT(aclnnMm, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc), cube_math_type);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_mm_test, ascend910B2_test_splitk_fp16_fp16_fp16_0)
{
    auto tensor_1_desc = TensorDesc({62080, 1536}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto tensor_2_desc = TensorDesc({1536, 384}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out_tensor_desc = TensorDesc({62080, 384}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2).Precision(0.005, 0.005);
    int8_t cube_math_type = 2;
    auto ut = OP_API_UT(aclnnMm, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc), cube_math_type);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_mm_test, ascend910B2_test_splitk_fp16_fp16_fp16_1)
{
    auto tensor_1_desc = TensorDesc({384, 1536}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto tensor_2_desc = TensorDesc({1536, 62080}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out_tensor_desc = TensorDesc({384, 62080}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2).Precision(0.005, 0.005);
    int8_t cube_math_type = 2;
    auto ut = OP_API_UT(aclnnMm, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc), cube_math_type);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}
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

#include "aclnn_matmul_compress_dequant.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace std;

class l2_matmul_compress_dequant_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "l2_matmul_compress_dequant_test SetUp" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "l2_matmul_compress_dequant_test TearDown" << endl;
    }
    static void MatMulCompressDequantCommonTest(
        TensorDesc a_desc, TensorDesc b_desc, TensorDesc index_desc, TensorDesc bias_desc, TensorDesc deqScale_desc,
        IntArrayDesc compress_info_desc, TensorDesc out_desc, aclnnStatus expect_status)
    {
        auto ut = OP_API_UT(
            aclnnMatmulCompressDequant,
            INPUT(a_desc, b_desc, index_desc, bias_desc, deqScale_desc, nullptr, 1, compress_info_desc),
            OUTPUT(out_desc));

        // SAMPLE: only test GetWorkspaceSize
        uint64_t workspace_size = 0;
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
        EXPECT_EQ(aclRet, expect_status);
    }
};

TEST_F(l2_matmul_compress_dequant_test, ascend310P_test_normal_input)
{
    // 使用**Desc描述host api输入输出
    TensorDesc a_desc = TensorDesc({16, 32}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc b_desc = TensorDesc({64}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc index_desc = TensorDesc({8}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc bias_desc = TensorDesc({16, 16}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc deqScale_desc = TensorDesc({16, 32}, ACL_UINT64, ACL_FORMAT_NC1HWC0);
    vector<int64_t> compress_info = {8, 8, 32, 16, 1};
    IntArrayDesc compress_info_desc = IntArrayDesc(compress_info);
    TensorDesc out_desc = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    MatMulCompressDequantCommonTest(
        a_desc, b_desc, index_desc, bias_desc, deqScale_desc, compress_info_desc, out_desc, ACLNN_SUCCESS);
}

TEST_F(l2_matmul_compress_dequant_test, ascend310P_test_empty)
{
    // 使用**Desc描述host api输入输出
    TensorDesc a_desc = TensorDesc({16, 0}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc b_desc = TensorDesc({0}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc index_desc = TensorDesc({8}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc bias_desc = TensorDesc({16, 16}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc deqScale_desc = TensorDesc({16, 32}, ACL_UINT64, ACL_FORMAT_NC1HWC0);
    vector<int64_t> compress_info = {8, 8, 0, 16, 1};
    IntArrayDesc compress_info_desc = IntArrayDesc(compress_info);
    TensorDesc out_desc = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    MatMulCompressDequantCommonTest(
        a_desc, b_desc, index_desc, bias_desc, deqScale_desc, compress_info_desc, out_desc, ACLNN_SUCCESS);
}

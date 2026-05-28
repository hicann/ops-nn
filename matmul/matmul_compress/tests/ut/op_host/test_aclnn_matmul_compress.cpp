/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <vector>
#include <array>
#include <float.h>
#include "gtest/gtest.h"

#include "../../../op_api/aclnn_matmul_compress.h"
#include "op_api/op_api_def.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"

using namespace std;
using namespace op;


class l2_matmul_compress_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "matmul_compress_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "matmul_compress_test TearDown" << endl;
    }

    static void MatMulCompressTest(
        TensorDesc x_desc, TensorDesc weight_desc, TensorDesc bias_desc, TensorDesc compressIndex_desc, 
        TensorDesc out_desc, aclnnStatus expect_status)
    {
        auto ut = OP_API_UT(aclnnMatmulCompress, INPUT(x_desc, weight_desc, bias_desc, compressIndex_desc), OUTPUT(out_desc));
        uint64_t workspaceSize = 0;
        aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
        EXPECT_EQ(getWorkspaceResult, expect_status);
    }
};

TEST_F(l2_matmul_compress_test, test_fp16)
{
    // 使用**Desc描述host api输入输出
    TensorDesc x_desc = TensorDesc({1024, 1024}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc weight_desc = TensorDesc({1024, 1024}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc bias_desc = TensorDesc({1024}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc compressIndex_desc = TensorDesc({1024}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({1024, 1024}, ACL_FLOAT16, ACL_FORMAT_ND);
    MatMulCompressTest(x_desc, weight_desc, bias_desc, compressIndex_desc, out_desc, ACL_SUCCESS);
}
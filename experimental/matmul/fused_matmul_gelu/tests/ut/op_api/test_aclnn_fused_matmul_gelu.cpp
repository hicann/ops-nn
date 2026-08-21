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
#include "gtest/gtest.h"

#include "../../../op_api/aclnn_fused_matmul_gelu.h"
#include "op_api/op_api_def_nn.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace std;
using namespace op;

class l2_fused_matmul_gelu_op_api_test : public testing::Test {
protected:
    static void FusedMatmulGeluCommonTest(const vector<int64_t>& xShape, const vector<int64_t>& weightShape,
                                          const vector<int64_t>& biasShape, const vector<int64_t>& yShape,
                                          aclDataType dtype, bool biasIsNull, int64_t approximate,
                                          aclnnStatus expectStatus)
    {
        auto xDesc = TensorDesc(xShape, dtype, ACL_FORMAT_ND);
        auto weightDesc = TensorDesc(weightShape, dtype, ACL_FORMAT_ND);
        auto yDesc = TensorDesc(yShape, dtype, ACL_FORMAT_ND);

        uint64_t workspaceSize = 0;
        aclnnStatus ret;

        if (biasIsNull) {
            auto biasDesc = nullptr;
            auto ut = OP_API_UT(aclnnFusedMatmulGelu, INPUT(xDesc, weightDesc, biasDesc, approximate), OUTPUT(yDesc));
            ret = ut.TestGetWorkspaceSize(&workspaceSize);
        } else {
            auto biasDesc = TensorDesc(biasShape, dtype, ACL_FORMAT_ND);
            auto ut = OP_API_UT(aclnnFusedMatmulGelu, INPUT(xDesc, weightDesc, biasDesc, approximate), OUTPUT(yDesc));
            ret = ut.TestGetWorkspaceSize(&workspaceSize);
        }

        EXPECT_EQ(ret, expectStatus);
    }
};

TEST_F(l2_fused_matmul_gelu_op_api_test, fp16_with_bias_success)
{
    FusedMatmulGeluCommonTest({2, 4}, // x: [M, K]
                              {3, 4}, // weight: [N, K]
                              {3},    // bias: [N]
                              {2, 3}, // y: [M, N]
                              ACL_FLOAT16, false, 1, ACL_SUCCESS);
}

TEST_F(l2_fused_matmul_gelu_op_api_test, fp16_without_bias_success)
{
    FusedMatmulGeluCommonTest({2, 4}, {3, 4}, {}, {2, 3}, ACL_FLOAT16, true, 1, ACL_SUCCESS);
}

TEST_F(l2_fused_matmul_gelu_op_api_test, bf16_with_bias_success)
{
    FusedMatmulGeluCommonTest({2, 4}, {3, 4}, {3}, {2, 3}, ACL_BF16, false, 1, ACL_SUCCESS);
}

TEST_F(l2_fused_matmul_gelu_op_api_test, invalid_approximate)
{
    FusedMatmulGeluCommonTest({2, 4}, {3, 4}, {3}, {2, 3}, ACL_FLOAT16, false, 2, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_fused_matmul_gelu_op_api_test, invalid_weight_k_mismatch)
{
    FusedMatmulGeluCommonTest({2, 4}, {3, 5}, {3}, {2, 3}, ACL_FLOAT16, false, 1, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_fused_matmul_gelu_op_api_test, invalid_bias_shape)
{
    FusedMatmulGeluCommonTest({2, 4}, {3, 4}, {4}, {2, 3}, ACL_FLOAT16, false, 1, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_fused_matmul_gelu_op_api_test, invalid_output_shape)
{
    FusedMatmulGeluCommonTest({2, 4}, {3, 4}, {3}, {2, 4}, ACL_FLOAT16, false, 1, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_fused_matmul_gelu_op_api_test, invalid_dtype_mismatch)
{
    auto xDesc = TensorDesc({2, 4}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto weightDesc = TensorDesc({3, 4}, ACL_BF16, ACL_FORMAT_ND);
    auto biasDesc = TensorDesc({3}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto yDesc = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnFusedMatmulGelu, INPUT(xDesc, weightDesc, biasDesc, 1), OUTPUT(yDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus ret = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(ret, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_fused_matmul_gelu_op_api_test, input_nullptr_invalid)
{
    auto xDesc = nullptr;
    auto weightDesc = TensorDesc({3, 4}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto biasDesc = TensorDesc({3}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto yDesc = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnFusedMatmulGelu, INPUT(xDesc, weightDesc, biasDesc, 1), OUTPUT(yDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus ret = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(ret, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_fused_matmul_gelu_op_api_test, weight_nullptr_invalid)
{
    auto xDesc = TensorDesc({2, 4}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto weightDesc = nullptr;
    auto biasDesc = TensorDesc({3}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto yDesc = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnFusedMatmulGelu, INPUT(xDesc, weightDesc, biasDesc, 1), OUTPUT(yDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus ret = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(ret, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_fused_matmul_gelu_op_api_test, output_nullptr_invalid)
{
    auto xDesc = TensorDesc({2, 4}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto weightDesc = TensorDesc({3, 4}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto biasDesc = TensorDesc({3}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto yDesc = nullptr;

    auto ut = OP_API_UT(aclnnFusedMatmulGelu, INPUT(xDesc, weightDesc, biasDesc, 1), OUTPUT(yDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus ret = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(ret, ACLNN_ERR_PARAM_NULLPTR);
}

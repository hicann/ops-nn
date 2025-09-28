/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_aclnn_apply_top_k_top_p.cpp
 * \brief
 */

#include "gtest/gtest.h"
#include "../../../op_host/op_api/aclnn_apply_top_k_top_p.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"

namespace TestAclnnApplyTopKTopP {
class l2_apply_top_k_top_p_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "l2_apply_top_k_top_p_test SetUp" << std::endl;
    }

    static void TearDownTestCase() { std::cout << "l2_apply_top_k_top_p_test TearDown" << std::endl; }

    void test_run(
        vector<int64_t> logitsDims, aclDataType logitsDtype, aclFormat logitsFormat, vector<int64_t> logitsRange,
        vector<int64_t> pDims, aclDataType pDtype, aclFormat pFormat, vector<int64_t> pRange,
        vector<int64_t> kDims, aclDataType kDtype, aclFormat kFormat, vector<int64_t> kRange,
        vector<int64_t> outDims, aclDataType outDtype, aclFormat outFormat, aclnnStatus expectedStatus)
    {
        auto logits = TensorDesc(logitsDims, logitsDtype, logitsFormat).ValueRange(logitsRange[0], logitsRange[1]);
        auto p = TensorDesc(pDims, pDtype, pFormat).ValueRange(pRange[0], pRange[1]);
        auto k = TensorDesc(kDims, kDtype, kFormat).ValueRange(kRange[0], kRange[1]);
        auto out = TensorDesc(outDims, outDtype, outFormat);

        auto ut = OP_API_UT(aclnnApplyTopKTopP, INPUT(logits_desc, p_desc, k_desc), OUTPUT(output_desc));
        uint64_t workspaceSize = 0;
        aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
        EXPECT_EQ(getWorkspaceResult, expectedStatus);
    }
};

// 空tensor
TEST_F(l2_apply_top_k_top_p_test, l2_apply_top_k_top_p_test_empty_success)
{
    test_run({4, 0}, ACL_FLOAT, ACL_FORMAT_ND, {-1, 1},
             {4}, ACL_FLOAT, ACL_FORMAT_ND, {0, 1},
             {4}, ACL_INT32, ACL_FORMAT_ND, {1, 10},
             {4, 0}, ACL_FLOAT, ACL_FORMAT_ND, ACLNN_SUCCESS);
}

// p空tensor
TEST_F(l2_apply_top_k_top_p_test, l2_apply_top_k_top_p_test_p_empty_success)
{
    test_run({4, 1024}, ACL_FLOAT, ACL_FORMAT_ND, {-1, 1},
             {0}, ACL_FLOAT, ACL_FORMAT_ND, {0, 1},
             {4}, ACL_INT32, ACL_FORMAT_ND, {1, 10},
             {4, 1024}, ACL_FLOAT, ACL_FORMAT_ND, ACLNN_SUCCESS);
}

// k空tensor
TEST_F(l2_apply_top_k_top_p_test, l2_apply_top_k_top_p_test_k_empty_success)
{
    test_run({4, 1024}, ACL_FLOAT, ACL_FORMAT_ND, {-1, 1},
             {4}, ACL_FLOAT, ACL_FORMAT_ND, {0, 1},
             {0}, ACL_INT32, ACL_FORMAT_ND, {1, 10},
             {4, 1024}, ACL_FLOAT, ACL_FORMAT_ND, ACLNN_SUCCESS);
}

// 正常路径，float32
TEST_F(l2_apply_top_k_top_p_test, l2_apply_top_k_top_p_test_fp32_success)
{
    test_run({4, 1024}, ACL_FLOAT, ACL_FORMAT_ND, {-1, 1},
             {4}, ACL_FLOAT, ACL_FORMAT_ND, {0, 1},
             {4}, ACL_INT32, ACL_FORMAT_ND, {1, 10},
             {4, 1024}, ACL_FLOAT, ACL_FORMAT_ND, ACLNN_SUCCESS);
}

// 正常路径，float16
TEST_F(l2_apply_top_k_top_p_test, l2_apply_top_k_top_p_test_fp16_success)
{
    test_run({4, 1024}, ACL_FLOAT16, ACL_FORMAT_ND, {-1, 1},
             {4}, ACL_FLOAT16, ACL_FORMAT_ND, {0, 1},
             {4}, ACL_INT32, ACL_FORMAT_ND, {1, 10},
             {4, 1024}, ACL_FLOAT16, ACL_FORMAT_ND, ACLNN_SUCCESS);
}

// 正常路径，bfloat16
TEST_F(l2_apply_top_k_top_p_test, ascend910B2_l2_apply_top_k_top_p_test_bf16_success)
{
    test_run({4, 1024}, ACL_BF16, ACL_FORMAT_ND, {-1, 1},
             {4}, ACL_BF16, ACL_FORMAT_ND, {0, 1},
             {4}, ACL_INT32, ACL_FORMAT_ND, {1, 10},
             {4, 1024}, ACL_BF16, ACL_FORMAT_ND, ACLNN_SUCCESS);
}

// 正常路径，float16
TEST_F(l2_apply_top_k_top_p_test, l2_apply_top_k_top_p_test_fp16_last_dim_1_success)
{
    test_run({4, 1}, ACL_FLOAT16, ACL_FORMAT_ND, {-1, 1},
             {4}, ACL_FLOAT16, ACL_FORMAT_ND, {0, 1},
             {4}, ACL_INT32, ACL_FORMAT_ND, {1, 1},
             {4, 1}, ACL_FLOAT16, ACL_FORMAT_ND, ACLNN_SUCCESS);
}

// CheckDtypeEqual
TEST_F(l2_apply_top_k_top_p_test, l2_apply_top_k_top_p_test_logits_dtype_invalid)
{
    test_run({4, 1024}, ACL_INT32, ACL_FORMAT_ND, {-1, 1},
             {4}, ACL_FLOAT, ACL_FORMAT_ND, {0, 1},
             {4}, ACL_INT32, ACL_FORMAT_ND, {1, 10},
             {4, 1024}, ACL_FLOAT, ACL_FORMAT_ND, ACLNN_ERR_PARAM_INVALID);
}

// CheckDtypeEqual
TEST_F(l2_apply_top_k_top_p_test, l2_apply_top_k_top_p_test_p_dtype_invalid)
{
    test_run({4, 1024}, ACL_FLOAT, ACL_FORMAT_ND, {-1, 1},
             {4}, ACL_INT32, ACL_FORMAT_ND, {0, 1},
             {4}, ACL_INT32, ACL_FORMAT_ND, {1, 10},
             {4, 1024}, ACL_FLOAT, ACL_FORMAT_ND, ACLNN_ERR_PARAM_INVALID);
}

// CheckDtypeEqual
TEST_F(l2_apply_top_k_top_p_test, l2_apply_top_k_top_p_test_k_dtype_invalid)
{
    test_run({4, 1024}, ACL_FLOAT, ACL_FORMAT_ND, {-1, 1},
             {4}, ACL_FLOAT, ACL_FORMAT_ND, {0, 1},
             {4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 10},
             {4, 1024}, ACL_FLOAT, ACL_FORMAT_ND, ACLNN_ERR_PARAM_INVALID);
}

// CheckDtypeEqual
TEST_F(l2_apply_top_k_top_p_test, l2_apply_top_k_top_p_test_out_dtype_invalid)
{
    test_run({4, 1024}, ACL_FLOAT, ACL_FORMAT_ND, {-1, 1},
             {4}, ACL_FLOAT, ACL_FORMAT_ND, {0, 1},
             {4}, ACL_INT32, ACL_FORMAT_ND, {1, 10},
             {4, 1024}, ACL_INT32, ACL_FORMAT_ND, ACLNN_ERR_PARAM_INVALID);
}

// CheckDtypeEqual
TEST_F(l2_apply_top_k_top_p_test, l2_apply_top_k_top_p_test_p_logits_dtype_not_equal)
{
    test_run({4, 1024}, ACL_FLOAT, ACL_FORMAT_ND, {-1, 1},
             {4}, ACL_FLOAT16, ACL_FORMAT_ND, {0, 1},
             {4}, ACL_INT32, ACL_FORMAT_ND, {1, 10},
             {4, 1024}, ACL_FLOAT, ACL_FORMAT_ND, ACLNN_ERR_PARAM_INVALID);
}

// CheckDtypeEqual
TEST_F(l2_apply_top_k_top_p_test, l2_apply_top_k_top_p_test_out_logits_dtype_not_equal)
{
    test_run({4, 1024}, ACL_FLOAT, ACL_FORMAT_ND, {-1, 1},
             {4}, ACL_FLOAT, ACL_FORMAT_ND, {0, 1},
             {4}, ACL_INT32, ACL_FORMAT_ND, {1, 10},
             {4, 1024}, ACL_FLOAT16, ACL_FORMAT_ND, ACLNN_ERR_PARAM_INVALID);
}

// CheckNotNull
TEST_F(l2_apply_top_k_top_p_test, l2_apply_top_k_top_p_test_logits_nullptr)
{
    auto logits_desc = nullptr;
    auto p_desc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.0, 1.0);
    auto k_desc = TensorDesc({4}, ACL_INT32, ACL_FORMAT_ND).ValueRange(1, 10);
    auto output_desc = TensorDesc({4, 1024}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnApplyTopKTopP, INPUT(logits_desc, p_desc, k_desc), OUTPUT(output_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

// CheckNotNull
TEST_F(l2_apply_top_k_top_p_test, l2_apply_top_k_top_p_test_p_nullptr)
{
    auto logits_desc = TensorDesc({4, 1024}, ACL_FLOAT, ACL_FORMAT_ND);
    auto p_desc = nullptr;
    auto k_desc = TensorDesc({4}, ACL_INT32, ACL_FORMAT_ND);
    auto output_desc = TensorDesc({4, 1024}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnApplyTopKTopP, INPUT(logits_desc, p_desc, k_desc), OUTPUT(output_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

// CheckNotNull
TEST_F(l2_apply_top_k_top_p_test, l2_apply_top_k_top_p_test_k_nullptr)
{
    auto logits_desc = TensorDesc({4, 1024}, ACL_FLOAT, ACL_FORMAT_ND);
    auto p_desc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto k_desc = nullptr;
    auto output_desc = TensorDesc({4, 1024}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnApplyTopKTopP, INPUT(logits_desc, p_desc, k_desc), OUTPUT(output_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

// CheckNotNull
TEST_F(l2_apply_top_k_top_p_test, l2_apply_top_k_top_p_test_output_nullptr)
{
    auto logits_desc = TensorDesc({4, 1024}, ACL_FLOAT, ACL_FORMAT_ND);
    auto p_desc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto k_desc = TensorDesc({4}, ACL_INT32, ACL_FORMAT_ND);
    auto output_desc = nullptr;

    auto ut = OP_API_UT(aclnnApplyTopKTopP, INPUT(logits_desc, p_desc, k_desc), OUTPUT(output_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

// CheckShapeValid
TEST_F(l2_apply_top_k_top_p_test, l2_apply_top_k_top_p_test_logits_shape_invalid)
{
    test_run({4, 1024, 1}, ACL_FLOAT, ACL_FORMAT_ND, {-1, 1},
             {4}, ACL_FLOAT, ACL_FORMAT_ND, {0, 1},
             {4}, ACL_INT32, ACL_FORMAT_ND, {1, 10},
             {4, 1024}, ACL_FLOAT, ACL_FORMAT_ND, ACLNN_ERR_PARAM_INVALID);
}

// CheckShapeValid
TEST_F(l2_apply_top_k_top_p_test, l2_apply_top_k_top_p_test_p_shape_invalid)
{
    test_run({4, 1024}, ACL_FLOAT, ACL_FORMAT_ND, {-1, 1},
             {4, 1}, ACL_FLOAT, ACL_FORMAT_ND, {0, 1},
             {4}, ACL_INT32, ACL_FORMAT_ND, {1, 10},
             {4, 1024}, ACL_FLOAT, ACL_FORMAT_ND, ACLNN_ERR_PARAM_INVALID);
}

// CheckShapeValid
TEST_F(l2_apply_top_k_top_p_test, l2_apply_top_k_top_p_test_k_shape_invalid)
{
    test_run({4, 1024}, ACL_FLOAT, ACL_FORMAT_ND, {-1, 1},
             {4}, ACL_FLOAT, ACL_FORMAT_ND, {0, 1},
             {4, 1}, ACL_INT32, ACL_FORMAT_ND, {1, 10},
             {4, 1024}, ACL_FLOAT, ACL_FORMAT_ND, ACLNN_ERR_PARAM_INVALID);
}

// CheckShapeValid
TEST_F(l2_apply_top_k_top_p_test, l2_apply_top_k_top_p_test_out_shape_invalid)
{
    test_run({4, 1024}, ACL_FLOAT, ACL_FORMAT_ND, {-1, 1},
             {4}, ACL_FLOAT, ACL_FORMAT_ND, {0, 1},
             {4}, ACL_INT32, ACL_FORMAT_ND, {1, 10},
             {4, 1024, 1}, ACL_FLOAT, ACL_FORMAT_ND, ACLNN_ERR_PARAM_INVALID);
}

// CheckShapeValid
TEST_F(l2_apply_top_k_top_p_test, l2_apply_top_k_top_p_test_out_logits_shape_invalid)
{
    test_run({4, 1024}, ACL_FLOAT, ACL_FORMAT_ND, {-1, 1},
             {4}, ACL_FLOAT, ACL_FORMAT_ND, {0, 1},
             {4}, ACL_INT32, ACL_FORMAT_ND, {1, 10},
             {1, 1024}, ACL_FLOAT, ACL_FORMAT_ND, ACLNN_ERR_PARAM_INVALID);
}

// CheckShapeValid
TEST_F(l2_apply_top_k_top_p_test, l2_apply_top_k_top_p_test_p_logits_shape_invalid)
{
    test_run({4, 1024}, ACL_FLOAT, ACL_FORMAT_ND, {-1, 1},
             {2}, ACL_FLOAT, ACL_FORMAT_ND, {0, 1},
             {4}, ACL_INT32, ACL_FORMAT_ND, {1, 10},
             {4, 1024}, ACL_FLOAT, ACL_FORMAT_ND, ACLNN_ERR_PARAM_INVALID);
}

// CheckShapeValid
TEST_F(l2_apply_top_k_top_p_test, l2_apply_top_k_top_p_test_k_logits_shape_invalid)
{
    test_run({4, 1024}, ACL_FLOAT, ACL_FORMAT_ND, {-1, 1},
             {4}, ACL_FLOAT, ACL_FORMAT_ND, {0, 1},
             {2}, ACL_INT32, ACL_FORMAT_ND, {1, 10},
             {4, 1024}, ACL_FLOAT, ACL_FORMAT_ND, ACLNN_ERR_PARAM_INVALID);
}

// CheckFormatValid
TEST_F(l2_apply_top_k_top_p_test, l2_apply_top_k_top_p_test_logits_format_invalid)
{
    test_run({4, 1024}, ACL_FLOAT, ACL_FORMAT_UNDEFINED, {-1, 1},
             {4}, ACL_FLOAT, ACL_FORMAT_ND, {0, 1},
             {4}, ACL_INT32, ACL_FORMAT_ND, {1, 10},
             {4, 1024}, ACL_FLOAT, ACL_FORMAT_ND, ACLNN_ERR_PARAM_INVALID);
}

// CheckFormatValid
TEST_F(l2_apply_top_k_top_p_test, l2_apply_top_k_top_p_test_p_format_invalid)
{
    test_run({4, 1024}, ACL_FLOAT, ACL_FORMAT_ND, {-1, 1},
             {4}, ACL_FLOAT, ACL_FORMAT_UNDEFINED, {0, 1},
             {4}, ACL_INT32, ACL_FORMAT_ND, {1, 10},
             {4, 1024}, ACL_FLOAT, ACL_FORMAT_ND, ACLNN_ERR_PARAM_INVALID);
}

// CheckFormatValid
TEST_F(l2_apply_top_k_top_p_test, l2_apply_top_k_top_p_test_k_format_invalid)
{
    test_run({4, 1024}, ACL_FLOAT, ACL_FORMAT_ND, {-1, 1},
             {4}, ACL_FLOAT, ACL_FORMAT_ND, {0, 1},
             {4}, ACL_INT32, ACL_FORMAT_UNDEFINED, {1, 10},
             {4, 1024}, ACL_FLOAT, ACL_FORMAT_ND, ACLNN_ERR_PARAM_INVALID);
}

// CheckFormatValid
TEST_F(l2_apply_top_k_top_p_test, l2_apply_top_k_top_p_test_out_format_invalid)
{
    test_run({4, 1024}, ACL_FLOAT, ACL_FORMAT_ND, {-1, 1},
             {4}, ACL_FLOAT, ACL_FORMAT_ND, {0, 1},
             {4}, ACL_INT32, ACL_FORMAT_ND, {1, 10},
             {4, 1024}, ACL_FLOAT, ACL_FORMAT_UNDEFINED, ACLNN_ERR_PARAM_INVALID);
}
} // namespace TestAclnnApplyTopKTopP

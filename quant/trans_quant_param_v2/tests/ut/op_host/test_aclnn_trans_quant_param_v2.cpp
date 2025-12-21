/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <float.h>

#include <array>
#include <vector>

#include "gtest/gtest.h"
#include <gmock/gmock.h>
#include "../../../op_host/op_api/aclnn_trans_quant_param_v2.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace op;
using namespace std;

class l2_trans_quant_param_v2_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << "l2_trans_quant_param_v2_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "l2_trans_quant_param_v2_test TearDown" << endl;
    }
};

TEST_F(l2_trans_quant_param_v2_test, test_trans_quant_param_first_api)
{
    TensorDesc scale_desc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc offset_desc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({3}, ACL_UINT64, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnTransQuantParamV2, INPUT(scale_desc, offset_desc), OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
}

TEST_F(l2_trans_quant_param_v2_test, test_trans_quant_param_third_api)
{
    TensorDesc scale_desc = TensorDesc({3, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc offset_desc = TensorDesc({3, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({3}, ACL_UINT64, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnTransQuantParamV2, INPUT(scale_desc, offset_desc), OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
}

TEST_F(l2_trans_quant_param_v2_test, test_trans_quant_param_fourth_api)
{
    TensorDesc scale_desc = TensorDesc({1, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc offset_desc = TensorDesc({1, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({3}, ACL_UINT64, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnTransQuantParamV2, INPUT(scale_desc, offset_desc), OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
}

TEST_F(l2_trans_quant_param_v2_test, test_trans_quant_param_fifth_api)
{
    TensorDesc scale_desc = TensorDesc({1, 3}, ACL_UINT64, ACL_FORMAT_ND);
    TensorDesc offset_desc = TensorDesc({1, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({3}, ACL_UINT64, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnTransQuantParamV2, INPUT(scale_desc, offset_desc), OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
}

TEST_F(l2_trans_quant_param_v2_test, test_trans_quant_param_sixth_api)
{
    TensorDesc scale_desc = TensorDesc({1, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc offset_desc = TensorDesc({1, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({3}, ACL_UINT64, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnTransQuantParamV2, INPUT(scale_desc, offset_desc), OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
}

TEST_F(l2_trans_quant_param_v2_test, test_offset_nullptr_invaild_api)
{
    TensorDesc scale_desc = TensorDesc({1, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({3}, ACL_UINT64, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnTransQuantParamV2, INPUT(scale_desc, (aclTensor*)nullptr), OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
}

TEST_F(l2_trans_quant_param_v2_test, test_format_invalid_api)
{
    TensorDesc scale_desc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_FRACTAL_NZ);
    TensorDesc offset_desc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({3}, ACL_UINT64, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnTransQuantParamV2, INPUT(scale_desc, offset_desc), OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
}

TEST_F(l2_trans_quant_param_v2_test, ascend910_95_test_trans_quant_param_first_dim_less_0)
{
    TensorDesc scale_desc = TensorDesc({-1, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({-1, 3}, ACL_UINT64, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnTransQuantParamV2, INPUT(scale_desc, (aclTensor*)nullptr), OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

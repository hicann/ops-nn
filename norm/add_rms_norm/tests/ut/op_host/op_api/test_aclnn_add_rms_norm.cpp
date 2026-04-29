/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <vector>
#include <array>
#include "gtest/gtest.h"
#include "../../../../op_host/op_api/aclnn_add_rms_norm.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"

using namespace std;

class l2_add_rms_norm_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << "l2_add_rms_norm_test SetUp" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "l2_add_rms_norm_test TearDown" << endl;
    }

public:
    void CommonTest(
        const vector<int64_t>& xShape, const vector<int64_t>& weightShape, const vector<int64_t>& rstdShape, aclDataType dtype, aclnnStatus expectRet)
    {
        auto x1 = TensorDesc(xShape, dtype, ACL_FORMAT_ND);
        auto x2 = TensorDesc(xShape, dtype, ACL_FORMAT_ND);
        auto weight = TensorDesc(weightShape, dtype, ACL_FORMAT_ND);
        auto yOut = TensorDesc(xShape, dtype, ACL_FORMAT_ND);
        auto rstdOut = TensorDesc(rstdShape, ACL_FLOAT, ACL_FORMAT_ND);
        auto xOut = TensorDesc(xShape, dtype, ACL_FORMAT_ND);
        uint64_t workspace_size = 0;
        auto ut = OP_API_UT(aclnnAddRmsNorm, INPUT(x1, x2, weight, 0.00001), OUTPUT(yOut, rstdOut, xOut));
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
        EXPECT_EQ(aclRet, expectRet);
    }
};

TEST_F(l2_add_rms_norm_test, ascend910B2_success)
{
    // data type cases
    CommonTest({4, 16, 128}, {128}, {4, 16, 1}, ACL_FLOAT, ACLNN_SUCCESS);
    CommonTest({4, 4, 128}, {128}, {4, 4, 1}, ACL_FLOAT16, ACLNN_SUCCESS);
    CommonTest({4, 16, 12288}, {12288}, {4, 16, 1}, ACL_FLOAT16, ACLNN_SUCCESS);
    CommonTest({4, 0, 128}, {128}, {4, 0, 1}, ACL_FLOAT, ACLNN_SUCCESS);
}

TEST_F(l2_add_rms_norm_test, ascend910B2_param_invalid)
{
    // invalid dtype
    CommonTest({4, 4, 128}, {128}, {4, 4, 1}, ACL_DOUBLE, ACLNN_ERR_PARAM_INVALID);
}
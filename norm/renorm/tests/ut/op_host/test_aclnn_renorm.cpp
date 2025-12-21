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

#include "../../../op_host/op_api/aclnn_renorm.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"

using namespace std;

class l2_renorm_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << "renorm_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "renorm_test TearDown" << endl;
    }
};

TEST_F(l2_renorm_test, case_dtype_fp16)
{
    auto self = TensorDesc({1, 3, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 5);
    auto out = TensorDesc({1, 3, 5}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto p = ScalarDesc(1.0f);
    int64_t dim = 0;
    auto maxNorm = ScalarDesc(5.0f);

    auto ut = OP_API_UT(aclnnRenorm, INPUT(self, p, dim, maxNorm), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // ut.TestPrecision();
}

TEST_F(l2_renorm_test, case_dtype_fp32)
{
    auto self = TensorDesc({1, 3, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 5);
    auto out = TensorDesc({1, 3, 5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto p = ScalarDesc(1.0f);
    int64_t dim = 0;
    auto maxNorm = ScalarDesc(5.0f);

    auto ut = OP_API_UT(aclnnRenorm, INPUT(self, p, dim, maxNorm), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // ut.TestPrecision();
}

TEST_F(l2_renorm_test, case_dtype_float16_float)
{
    auto self = TensorDesc({1, 3, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 5);
    auto out = TensorDesc({1, 3, 5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto p = ScalarDesc(1.0f);
    int64_t dim = 0;
    auto maxNorm = ScalarDesc(5.0f);

    auto ut = OP_API_UT(aclnnRenorm, INPUT(self, p, dim, maxNorm), OUTPUT(out));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_renorm_test, case_dtype_float_float16)
{
    auto self = TensorDesc({1, 3, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 5);
    auto out = TensorDesc({1, 3, 5}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto p = ScalarDesc(1.0f);
    int64_t dim = 0;
    auto maxNorm = ScalarDesc(5.0f);

    auto ut = OP_API_UT(aclnnRenorm, INPUT(self, p, dim, maxNorm), OUTPUT(out));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_renorm_test, case_dtype_dim_1)
{
    auto self = TensorDesc({10}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 5);
    auto out = TensorDesc({10}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto p = ScalarDesc(0.0f);
    int64_t dim = 0;
    auto maxNorm = ScalarDesc(5.0f);

    auto ut = OP_API_UT(aclnnRenorm, INPUT(self, p, dim, maxNorm), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_renorm_test, case_dtype_dim2)
{
    auto self = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 5);
    auto out = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto p = ScalarDesc(1.0f);
    int64_t dim = 0;
    auto maxNorm = ScalarDesc(5.0f);

    auto ut = OP_API_UT(aclnnRenorm, INPUT(self, p, dim, maxNorm), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // ut.TestPrecision();
}

TEST_F(l2_renorm_test, case_dtype_dim8)
{
    auto self = TensorDesc({2, 3, 4, 1, 2, 3, 4, 1}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 5);
    auto out = TensorDesc({2, 3, 4, 1, 2, 3, 4, 1}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto p = ScalarDesc(1.0f);
    int64_t dim = 0;
    auto maxNorm = ScalarDesc(5.0f);

    auto ut = OP_API_UT(aclnnRenorm, INPUT(self, p, dim, maxNorm), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // ut.TestPrecision();
}

TEST_F(l2_renorm_test, case_dtype_dim9)
{
    auto self = TensorDesc({2, 3, 4, 1, 2, 3, 4, 1, 2}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 5);
    auto out = TensorDesc({2, 3, 4, 1, 2, 3, 4, 1, 2}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto p = ScalarDesc(1.0f);
    int64_t dim = 0;
    auto maxNorm = ScalarDesc(5.0f);

    auto ut = OP_API_UT(aclnnRenorm, INPUT(self, p, dim, maxNorm), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_renorm_test, case_dtype_not_contiguous)
{
    auto self = TensorDesc({1, 3, 4}, ACL_FLOAT16, ACL_FORMAT_ND, {12, 1, 3}, 0, {1, 4, 3}).ValueRange(0, 5);
    auto out = TensorDesc({1, 3, 4}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto p = ScalarDesc(1.0f);
    int64_t dim = 0;
    auto maxNorm = ScalarDesc(5.0f);

    auto ut = OP_API_UT(aclnnRenorm, INPUT(self, p, dim, maxNorm), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // ut.TestPrecision();
}

TEST_F(l2_renorm_test, case_self_nullptr)
{
    auto self = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 5);
    auto out = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto p = ScalarDesc(1.0f);
    int64_t dim = 0;
    auto maxNorm = ScalarDesc(5.0f);

    auto ut = OP_API_UT(aclnnRenorm, INPUT(nullptr, p, dim, maxNorm), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_renorm_test, case_p_nullptr)
{
    auto self = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 5);
    auto out = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto p = ScalarDesc(1.0f);
    int64_t dim = 0;
    auto maxNorm = ScalarDesc(5.0f);

    auto ut = OP_API_UT(aclnnRenorm, INPUT(self, nullptr, dim, maxNorm), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_renorm_test, case_maxNorm_nullptr)
{
    auto self = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 5);
    auto out = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto p = ScalarDesc(1.0f);
    int64_t dim = 0;
    auto maxNorm = ScalarDesc(5.0f);

    auto ut = OP_API_UT(aclnnRenorm, INPUT(self, p, dim, nullptr), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_renorm_test, case_output_nullptr)
{
    auto self = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 5);
    auto out = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto p = ScalarDesc(1.0f);
    int64_t dim = 0;
    auto maxNorm = ScalarDesc(5.0f);

    auto ut = OP_API_UT(aclnnRenorm, INPUT(self, p, dim, maxNorm), OUTPUT(nullptr));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_renorm_test, case_dtype_int32)
{
    auto self = TensorDesc({1, 3, 5}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 5);
    auto out = TensorDesc({1, 3, 5}, ACL_INT32, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto p = ScalarDesc(1.0f);
    int64_t dim = 0;
    auto maxNorm = ScalarDesc(5.0f);

    auto ut = OP_API_UT(aclnnRenorm, INPUT(self, p, dim, maxNorm), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_renorm_test, case_dtype_double)
{
    auto self = TensorDesc({1, 3, 5}, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(0, 5);
    auto out = TensorDesc({1, 3, 5}, ACL_DOUBLE, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto p = ScalarDesc(1.0f);
    int64_t dim = 0;
    auto maxNorm = ScalarDesc(5.0f);

    auto ut = OP_API_UT(aclnnRenorm, INPUT(self, p, dim, maxNorm), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_renorm_test, case_p_support)
{
    auto self = TensorDesc({1, 3, 5, 6}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 5);
    auto out = TensorDesc({1, 3, 5, 6}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto p1 = ScalarDesc(0.0f);
    auto p2 = ScalarDesc(1.0f);
    auto p3 = ScalarDesc(2.0f);
    auto p4 = ScalarDesc(3.0f);
    int64_t dim = 0;
    auto maxNorm = ScalarDesc(5.0f);

    auto ut1 = OP_API_UT(aclnnRenorm, INPUT(self, p1, dim, maxNorm), OUTPUT(out));

    uint64_t workspace_size1 = 0;
    aclnnStatus aclRet1 = ut1.TestGetWorkspaceSize(&workspace_size1);
    EXPECT_EQ(aclRet1, ACL_SUCCESS);

    auto ut2 = OP_API_UT(aclnnRenorm, INPUT(self, p2, dim, maxNorm), OUTPUT(out));

    uint64_t workspace_size2 = 0;
    aclnnStatus aclRet2 = ut2.TestGetWorkspaceSize(&workspace_size2);
    EXPECT_EQ(aclRet2, ACL_SUCCESS);

    // ut2.TestPrecision();

    auto ut3 = OP_API_UT(aclnnRenorm, INPUT(self, p3, dim, maxNorm), OUTPUT(out));

    uint64_t workspace_size3 = 0;
    aclnnStatus aclRet3 = ut3.TestGetWorkspaceSize(&workspace_size3);
    EXPECT_EQ(aclRet3, ACL_SUCCESS);

    // ut3.TestPrecision();

    auto ut4 = OP_API_UT(aclnnRenorm, INPUT(self, p4, dim, maxNorm), OUTPUT(out));

    uint64_t workspace_size4 = 0;
    aclnnStatus aclRet4 = ut4.TestGetWorkspaceSize(&workspace_size4);
    EXPECT_EQ(aclRet4, ACL_SUCCESS);

    // ut4.TestPrecision();
}

TEST_F(l2_renorm_test, case_dim_support)
{
    auto self = TensorDesc({1, 3, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 5);
    auto out = TensorDesc({1, 3, 5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto p = ScalarDesc(1.0f);
    int64_t dim1 = 0;
    int64_t dim2 = 2;
    int64_t dim3 = -3;
    auto maxNorm = ScalarDesc(5.0f);

    auto ut1 = OP_API_UT(aclnnRenorm, INPUT(self, p, dim1, maxNorm), OUTPUT(out));

    uint64_t workspace_size1 = 0;
    aclnnStatus aclRet1 = ut1.TestGetWorkspaceSize(&workspace_size1);
    EXPECT_EQ(aclRet1, ACL_SUCCESS);

    // ut1.TestPrecision();

    auto ut2 = OP_API_UT(aclnnRenorm, INPUT(self, p, dim2, maxNorm), OUTPUT(out));

    uint64_t workspace_size2 = 0;
    aclnnStatus aclRet2 = ut2.TestGetWorkspaceSize(&workspace_size2);
    EXPECT_EQ(aclRet2, ACL_SUCCESS);

    // ut2.TestPrecision();

    auto ut3 = OP_API_UT(aclnnRenorm, INPUT(self, p, dim3, maxNorm), OUTPUT(out));

    uint64_t workspace_size3 = 0;
    aclnnStatus aclRet3 = ut3.TestGetWorkspaceSize(&workspace_size3);
    EXPECT_EQ(aclRet3, ACL_SUCCESS);

    // ut3.TestPrecision();
}

TEST_F(l2_renorm_test, case_dim_unsupport)
{
    auto self = TensorDesc({1, 3, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 5);
    auto out = TensorDesc({1, 3, 5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto p = ScalarDesc(1.0f);
    int64_t dim1 = 3;
    int64_t dim2 = -4;
    auto maxNorm = ScalarDesc(5.0f);

    auto ut1 = OP_API_UT(aclnnRenorm, INPUT(self, p, dim1, maxNorm), OUTPUT(out));

    uint64_t workspace_size1 = 0;
    aclnnStatus aclRet1 = ut1.TestGetWorkspaceSize(&workspace_size1);
    EXPECT_EQ(aclRet1, ACLNN_ERR_PARAM_INVALID);

    auto ut2 = OP_API_UT(aclnnRenorm, INPUT(self, p, dim2, maxNorm), OUTPUT(out));

    uint64_t workspace_size2 = 0;
    aclnnStatus aclRet2 = ut2.TestGetWorkspaceSize(&workspace_size2);
    EXPECT_EQ(aclRet2, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_renorm_test, case_maxNorm_support)
{
    auto self = TensorDesc({1, 3, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 5);
    auto out = TensorDesc({1, 3, 5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto p = ScalarDesc(1.0f);
    int64_t dim = 0;
    auto maxNorm1 = ScalarDesc(1.0f);
    auto maxNorm2 = ScalarDesc(10.0f);
    auto maxNorm3 = ScalarDesc(100.0f);

    auto ut1 = OP_API_UT(aclnnRenorm, INPUT(self, p, dim, maxNorm1), OUTPUT(out));

    uint64_t workspace_size1 = 0;
    aclnnStatus aclRet1 = ut1.TestGetWorkspaceSize(&workspace_size1);
    EXPECT_EQ(aclRet1, ACL_SUCCESS);

    // ut1.TestPrecision();

    auto ut2 = OP_API_UT(aclnnRenorm, INPUT(self, p, dim, maxNorm2), OUTPUT(out));

    uint64_t workspace_size2 = 0;
    aclnnStatus aclRet2 = ut2.TestGetWorkspaceSize(&workspace_size2);
    EXPECT_EQ(aclRet2, ACL_SUCCESS);

    // ut2.TestPrecision();

    auto ut3 = OP_API_UT(aclnnRenorm, INPUT(self, p, dim, maxNorm3), OUTPUT(out));

    uint64_t workspace_size3 = 0;
    aclnnStatus aclRet3 = ut3.TestGetWorkspaceSize(&workspace_size3);
    EXPECT_EQ(aclRet3, ACL_SUCCESS);

    // ut3.TestPrecision();
}

TEST_F(l2_renorm_test, case_maxNorm_unsupport)
{
    auto self = TensorDesc({1, 3, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 5);
    auto out = TensorDesc({1, 3, 5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto p = ScalarDesc(1.0f);
    int64_t dim = 0;
    auto maxNorm1 = ScalarDesc(-1.0f);
    auto maxNorm2 = ScalarDesc(-2.0f);
    auto maxNorm3 = ScalarDesc(-10.0f);

    auto ut1 = OP_API_UT(aclnnRenorm, INPUT(self, p, dim, maxNorm1), OUTPUT(out));

    uint64_t workspace_size1 = 0;
    aclnnStatus aclRet1 = ut1.TestGetWorkspaceSize(&workspace_size1);
    EXPECT_EQ(aclRet1, ACLNN_ERR_PARAM_INVALID);

    auto ut2 = OP_API_UT(aclnnRenorm, INPUT(self, p, dim, maxNorm2), OUTPUT(out));

    uint64_t workspace_size2 = 0;
    aclnnStatus aclRet2 = ut2.TestGetWorkspaceSize(&workspace_size2);
    EXPECT_EQ(aclRet2, ACLNN_ERR_PARAM_INVALID);

    auto ut3 = OP_API_UT(aclnnRenorm, INPUT(self, p, dim, maxNorm3), OUTPUT(out));

    uint64_t workspace_size3 = 0;
    aclnnStatus aclRet3 = ut3.TestGetWorkspaceSize(&workspace_size3);
    EXPECT_EQ(aclRet3, ACLNN_ERR_PARAM_INVALID);
}

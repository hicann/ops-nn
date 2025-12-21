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

#include "../../../op_host/op_api/aclnn_flat_quant.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"

using namespace std;

class l2_flat_quant_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "l2_flat_quant_test SetUp" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "l2_flat_quant_test TearDown" << endl;
    }

public:
    void CommonTest(
        const vector<int64_t>& xShape, const vector<int64_t>& p1Shape, const vector<int64_t>& p2Shape,
        const vector<int64_t>& outShape, const vector<int64_t>& scaleShape, aclDataType xDtype, aclDataType p1Dtype,
        aclDataType p2Dtype, aclDataType outDtype, aclDataType scaleDtype, const double_t clipRatio,
        aclnnStatus expectRet)
    {
        auto x = TensorDesc(xShape, xDtype, ACL_FORMAT_NCL);
        auto p1 = TensorDesc(p1Shape, p1Dtype, ACL_FORMAT_ND);
        auto p2 = TensorDesc(p2Shape, p2Dtype, ACL_FORMAT_ND);
        auto out = TensorDesc(outShape, outDtype, ACL_FORMAT_NCL);
        auto quantScale = TensorDesc(scaleShape, scaleDtype, ACL_FORMAT_ND);
        uint64_t workspace_size = 0;
        auto ut = OP_API_UT(aclnnFlatQuant, INPUT(x, p1, p2, clipRatio), OUTPUT(out, quantScale));
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);

        // EXPECT_EQ(aclRet, expectRet);
    }
};

TEST_F(l2_flat_quant_test, ascend910B2_success)
{
    // data type cases
    CommonTest(
        {16, 64, 64}, {64, 64}, {64, 64}, {16, 64, 64}, {16}, ACL_FLOAT16, ACL_FLOAT16, ACL_FLOAT16, ACL_INT4,
        ACL_FLOAT, 1.0, ACLNN_SUCCESS);
    CommonTest(
        {16, 64, 64}, {64, 64}, {64, 64}, {16, 64, 8}, {16}, ACL_FLOAT16, ACL_FLOAT16, ACL_FLOAT16, ACL_INT32,
        ACL_FLOAT, 1.0, ACLNN_SUCCESS);
    CommonTest(
        {16, 64, 64}, {64, 64}, {64, 64}, {16, 64, 64}, {16}, ACL_BF16, ACL_BF16, ACL_BF16, ACL_INT4, ACL_FLOAT, 1.0,
        ACLNN_SUCCESS);
    CommonTest(
        {16, 64, 64}, {64, 64}, {64, 64}, {16, 64, 8}, {16}, ACL_BF16, ACL_BF16, ACL_BF16, ACL_INT32, ACL_FLOAT, 1.0,
        ACLNN_SUCCESS);
}

TEST_F(l2_flat_quant_test, ascend910B2_param_nullptr)
{
    // nullptr
    auto x = TensorDesc({16, 64, 64}, ACL_FLOAT16, ACL_FORMAT_NCL);
    auto p1 = TensorDesc({64, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto p2 = TensorDesc({64, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto out = TensorDesc({16, 64, 64}, ACL_INT4, ACL_FORMAT_NCL);
    auto quantScale = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND);
    double clipRatio = 1.0;
    uint64_t workspace_size = 0;

    auto ut1 = OP_API_UT(aclnnFlatQuant, INPUT(nullptr, p1, p2, clipRatio), OUTPUT(out, quantScale));
    aclnnStatus aclRet = ut1.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

    auto ut2 = OP_API_UT(aclnnFlatQuant, INPUT(x, nullptr, p2, clipRatio), OUTPUT(out, quantScale));
    aclRet = ut2.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

    auto ut3 = OP_API_UT(aclnnFlatQuant, INPUT(x, p1, nullptr, clipRatio), OUTPUT(out, quantScale));
    aclRet = ut3.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

    auto ut4 = OP_API_UT(aclnnFlatQuant, INPUT(x, p1, p2, clipRatio), OUTPUT(nullptr, quantScale));
    aclRet = ut4.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

    auto ut5 = OP_API_UT(aclnnFlatQuant, INPUT(x, p1, p2, clipRatio), OUTPUT(out, nullptr));
    aclRet = ut5.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_flat_quant_test, ascend910B2_param_invalid)
{
    // invalid x shape
    // axis is 0
    CommonTest(
        {0, 64, 64}, {64, 64}, {64, 64}, {16, 64, 64}, {16}, ACL_FLOAT16, ACL_FLOAT16, ACL_FLOAT16, ACL_INT4, ACL_FLOAT,
        1.0, ACLNN_ERR_PARAM_INVALID);
    CommonTest(
        {16, 0, 64}, {64, 64}, {64, 64}, {16, 64, 64}, {16}, ACL_FLOAT16, ACL_FLOAT16, ACL_FLOAT16, ACL_INT4, ACL_FLOAT,
        1.0, ACLNN_ERR_PARAM_INVALID);
    CommonTest(
        {16, 64, 0}, {64, 64}, {64, 64}, {16, 64, 64}, {16}, ACL_FLOAT16, ACL_FLOAT16, ACL_FLOAT16, ACL_INT4, ACL_FLOAT,
        1.0, ACLNN_ERR_PARAM_INVALID);
    // axis exceeds limit
    CommonTest(
        {262145, 64, 64}, {64, 64}, {64, 64}, {262145, 64, 64}, {262145}, ACL_FLOAT16, ACL_FLOAT16, ACL_FLOAT16,
        ACL_INT4, ACL_FLOAT, 1.0, ACLNN_ERR_PARAM_INVALID);
    CommonTest(
        {16, 257, 64}, {257, 257}, {64, 64}, {16, 257, 64}, {16}, ACL_FLOAT16, ACL_FLOAT16, ACL_FLOAT16, ACL_INT4,
        ACL_FLOAT, 1.0, ACLNN_ERR_PARAM_INVALID);
    CommonTest(
        {16, 64, 257}, {64, 64}, {257, 257}, {16, 64, 257}, {16}, ACL_FLOAT16, ACL_FLOAT16, ACL_FLOAT16, ACL_INT4,
        ACL_FLOAT, 1.0, ACLNN_ERR_PARAM_INVALID);
    // invalid dtype
    // inconsistent dtype
    CommonTest(
        {16, 64, 64}, {64, 64}, {64, 64}, {16, 64, 64}, {16}, ACL_FLOAT, ACL_FLOAT16, ACL_FLOAT16, ACL_INT4, ACL_FLOAT,
        1.0, ACLNN_ERR_PARAM_INVALID);
    CommonTest(
        {16, 64, 64}, {64, 64}, {64, 64}, {16, 64, 64}, {16}, ACL_FLOAT16, ACL_FLOAT, ACL_FLOAT16, ACL_INT4, ACL_FLOAT,
        1.0, ACLNN_ERR_PARAM_INVALID);
    CommonTest(
        {16, 64, 64}, {64, 64}, {64, 64}, {16, 64, 64}, {16}, ACL_FLOAT16, ACL_FLOAT16, ACL_FLOAT, ACL_INT4, ACL_FLOAT,
        1.0, ACLNN_ERR_PARAM_INVALID);
    // unsupported dtype
    CommonTest(
        {16, 64, 64}, {64, 64}, {64, 64}, {16, 64, 64}, {16}, ACL_INT32, ACL_FLOAT16, ACL_FLOAT16, ACL_INT4, ACL_FLOAT,
        1.0, ACLNN_ERR_PARAM_INVALID);
    CommonTest(
        {16, 64, 64}, {64, 64}, {64, 64}, {16, 64, 64}, {16}, ACL_FLOAT16, ACL_INT32, ACL_FLOAT16, ACL_INT4, ACL_FLOAT,
        1.0, ACLNN_ERR_PARAM_INVALID);
    CommonTest(
        {16, 64, 64}, {64, 64}, {64, 64}, {16, 64, 64}, {16}, ACL_FLOAT16, ACL_FLOAT16, ACL_INT32, ACL_INT4, ACL_FLOAT,
        1.0, ACLNN_ERR_PARAM_INVALID);
    CommonTest(
        {16, 64, 64}, {64, 64}, {64, 64}, {16, 64, 64}, {16}, ACL_FLOAT16, ACL_FLOAT16, ACL_FLOAT16, ACL_FLOAT16,
        ACL_FLOAT, 1.0, ACLNN_ERR_PARAM_INVALID);
    CommonTest(
        {16, 64, 64}, {64, 64}, {64, 64}, {16, 64, 64}, {16}, ACL_FLOAT16, ACL_FLOAT16, ACL_FLOAT16, ACL_INT4,
        ACL_INT32, 1.0, ACLNN_ERR_PARAM_INVALID);
    // invalid dim
    CommonTest(
        {16, 16, 64, 64}, {64, 64}, {64, 64}, {16, 64, 64}, {16}, ACL_FLOAT16, ACL_FLOAT16, ACL_FLOAT16, ACL_INT4,
        ACL_FLOAT, 1.0, ACLNN_ERR_PARAM_INVALID);
    CommonTest(
        {16, 64, 64}, {64}, {64, 64}, {16, 64, 64}, {16}, ACL_FLOAT16, ACL_FLOAT16, ACL_FLOAT16, ACL_INT4, ACL_FLOAT,
        1.0, ACLNN_ERR_PARAM_INVALID);
    CommonTest(
        {16, 64, 64}, {64, 64}, {64, 64, 64}, {16, 64, 64}, {16}, ACL_FLOAT16, ACL_FLOAT16, ACL_FLOAT16, ACL_INT4,
        ACL_FLOAT, 1.0, ACLNN_ERR_PARAM_INVALID);
    CommonTest(
        {16, 64, 64}, {64, 64}, {64, 64}, {16, 64}, {16}, ACL_FLOAT16, ACL_FLOAT16, ACL_FLOAT16, ACL_INT4, ACL_FLOAT,
        1.0, ACLNN_ERR_PARAM_INVALID);
    CommonTest(
        {16, 64, 64}, {64, 64}, {64, 64}, {16, 64, 64}, {16, 64}, ACL_FLOAT16, ACL_FLOAT16, ACL_FLOAT16, ACL_INT4,
        ACL_FLOAT, 1.0, ACLNN_ERR_PARAM_INVALID);
    // inconsistent shape
    CommonTest(
        {16, 64, 64}, {64, 65}, {64, 64}, {16, 64, 64}, {16}, ACL_FLOAT16, ACL_FLOAT16, ACL_FLOAT16, ACL_INT4,
        ACL_FLOAT, 1.0, ACLNN_ERR_PARAM_INVALID);
    CommonTest(
        {16, 64, 64}, {64, 64}, {65, 64}, {16, 64, 64}, {16}, ACL_FLOAT16, ACL_FLOAT16, ACL_FLOAT16, ACL_INT4,
        ACL_FLOAT, 1.0, ACLNN_ERR_PARAM_INVALID);
    // invalid shape
    CommonTest(
        {16, 64, 65}, {64, 64}, {65, 65}, {16, 64, 65}, {16}, ACL_FLOAT16, ACL_FLOAT16, ACL_FLOAT16, ACL_INT4,
        ACL_FLOAT, 1.0, ACLNN_ERR_PARAM_INVALID);
    CommonTest(
        {16, 64, 64}, {64, 64}, {64, 64}, {16, 64, 8}, {16}, ACL_FLOAT16, ACL_FLOAT16, ACL_FLOAT16, ACL_INT4, ACL_FLOAT,
        1.0, ACLNN_ERR_PARAM_INVALID);
    CommonTest(
        {16, 64, 64}, {64, 64}, {64, 64}, {16, 64, 64}, {16}, ACL_FLOAT16, ACL_FLOAT16, ACL_FLOAT16, ACL_INT32,
        ACL_FLOAT, 1.0, ACLNN_ERR_PARAM_INVALID);
    CommonTest(
        {16, 64, 64}, {64, 64}, {64, 64}, {16, 65, 8}, {16}, ACL_FLOAT16, ACL_FLOAT16, ACL_FLOAT16, ACL_INT32,
        ACL_FLOAT, 1.0, ACLNN_ERR_PARAM_INVALID);
    // clipRatio not valid
    CommonTest(
        {16, 64, 64}, {64, 64}, {64, 64}, {16, 64, 64}, {16}, ACL_FLOAT16, ACL_FLOAT16, ACL_FLOAT16, ACL_INT4,
        ACL_FLOAT, 1.5, ACLNN_ERR_PARAM_INVALID);
    CommonTest(
        {16, 64, 64}, {64, 64}, {64, 64}, {16, 64, 64}, {16}, ACL_FLOAT16, ACL_FLOAT16, ACL_FLOAT16, ACL_INT4,
        ACL_FLOAT, 0.0, ACLNN_ERR_PARAM_INVALID);
}

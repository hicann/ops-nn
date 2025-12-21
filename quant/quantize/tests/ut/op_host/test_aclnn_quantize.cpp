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
#include <vector>
#include <array>
#include "gtest/gtest.h"

#include "../../../op_host/op_api/aclnn_quantize.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include <cstdlib>
#include <ctime>

using namespace std;

class l2_quantize_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << "quantize_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "quantize_test TearDown" << endl;
    }
};

// CheckNotNull
TEST_F(l2_quantize_test, quantize_testcase_001_exception_null_x)
{
    auto scalesDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto zeroPointsDesc = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND).ValueRange(1, 5);
    aclDataType dType = ACL_INT8;
    int32_t axis = 1;
    const vector<int64_t>& outShape = {3, 2};
    auto outTensorDesc = TensorDesc(outShape, dType, ACL_FORMAT_ND).ValidCount(6);
    auto ut = OP_API_UT(
        aclnnQuantize, INPUT((aclTensor*)nullptr, scalesDesc, zeroPointsDesc, dType, axis), OUTPUT(outTensorDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// CheckNotNull
TEST_F(l2_quantize_test, quantize_testcase_002_exception_null_scales)
{
    auto xDesc = TensorDesc({3, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto zeroPointsDesc = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND).ValueRange(1, 5);
    aclDataType dType = ACL_INT8;
    int32_t axis = 1;
    const vector<int64_t>& outShape = {3, 2};
    auto outTensorDesc = TensorDesc(outShape, dType, ACL_FORMAT_ND).ValidCount(6);
    auto ut =
        OP_API_UT(aclnnQuantize, INPUT(xDesc, (aclTensor*)nullptr, zeroPointsDesc, dType, axis), OUTPUT(outTensorDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// CheckNotNull
TEST_F(l2_quantize_test, quantize_testcase_003_exception_null_out)
{
    auto xDesc = TensorDesc({3, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto scalesDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto zeroPointsDesc = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND).ValueRange(1, 5);
    aclDataType dType = ACL_INT8;
    int32_t axis = 1;
    const vector<int64_t>& outShape = {3, 2};
    auto ut = OP_API_UT(
        aclnnQuantize, INPUT((aclTensor*)nullptr, scalesDesc, zeroPointsDesc, dType, axis),
        OUTPUT((aclTensor*)nullptr));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// CheckDimValid
TEST_F(l2_quantize_test, quantize_testcase_004_exception_dim_not_valid)
{
    auto xDesc = TensorDesc({3, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto scalesDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto zeroPointsDesc = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND).ValueRange(1, 5);
    aclDataType dType = ACL_INT8;
    int32_t axis = 1;
    const vector<int64_t>& outShape = {3, 2};
    auto outTensorDesc = TensorDesc(outShape, dType, ACL_FORMAT_ND).ValidCount(6);
    auto ut = OP_API_UT(aclnnQuantize, INPUT(xDesc, scalesDesc, zeroPointsDesc, dType, axis), OUTPUT(outTensorDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// Check x dtype not in supportList
TEST_F(l2_quantize_test, quantize_testcase_005_exception_x_dtype_not_supported)
{
    auto xDesc = TensorDesc({3, 2}, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto scalesDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto zeroPointsDesc = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND).ValueRange(1, 5);
    aclDataType dType = ACL_INT8;
    int32_t axis = 1;
    const vector<int64_t>& outShape = {3, 2};
    auto outTensorDesc = TensorDesc(outShape, dType, ACL_FORMAT_ND).ValidCount(6);
    auto ut = OP_API_UT(aclnnQuantize, INPUT(xDesc, scalesDesc, zeroPointsDesc, dType, axis), OUTPUT(outTensorDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// Check scales dtype not in supportList
TEST_F(l2_quantize_test, quantize_testcase_006_exception_scales_dtype_not_supported)
{
    auto xDesc = TensorDesc({3, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto scalesDesc = TensorDesc({2}, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto zeroPointsDesc = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND).ValueRange(1, 5);
    aclDataType dType = ACL_INT8;
    int32_t axis = 1;
    const vector<int64_t>& outShape = {3, 2};
    auto outTensorDesc = TensorDesc(outShape, dType, ACL_FORMAT_ND).ValidCount(6);
    auto ut = OP_API_UT(aclnnQuantize, INPUT(xDesc, scalesDesc, zeroPointsDesc, dType, axis), OUTPUT(outTensorDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// Check zeroPoints dtype not in supportList
TEST_F(l2_quantize_test, quantize_testcase_007_exception_zeroPoints_dtype_not_supported)
{
    auto xDesc = TensorDesc({3, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto scalesDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto zeroPointsDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 5);
    aclDataType dType = ACL_INT8;
    int32_t axis = 1;
    const vector<int64_t>& outShape = {3, 2};
    auto outTensorDesc = TensorDesc(outShape, dType, ACL_FORMAT_ND).ValidCount(6);
    auto ut = OP_API_UT(aclnnQuantize, INPUT(xDesc, scalesDesc, zeroPointsDesc, dType, axis), OUTPUT(outTensorDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    // EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// Check scales' Size not equal x[axis]'s Size
TEST_F(l2_quantize_test, quantize_testcase_008_exception_scales_size_not_valid)
{
    auto xDesc = TensorDesc({3, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto scalesDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto zeroPointsDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 5);
    aclDataType dType = ACL_INT8;
    int32_t axis = 1;
    const vector<int64_t>& outShape = {3, 2};
    auto outTensorDesc = TensorDesc(outShape, dType, ACL_FORMAT_ND).ValidCount(6);
    auto ut = OP_API_UT(aclnnQuantize, INPUT(xDesc, scalesDesc, zeroPointsDesc, dType, axis), OUTPUT(outTensorDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// Check BF16 Dtype
TEST_F(l2_quantize_test, quantize_testcase_009_exception_bf16_dtype_not_valid)
{
    auto xDesc = TensorDesc({3, 2}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto scalesDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto zeroPointsDesc = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND).ValueRange(1, 5);
    aclDataType dType = ACL_INT8;
    int32_t axis = 1;
    const vector<int64_t>& outShape = {3, 2};
    auto outTensorDesc = TensorDesc(outShape, dType, ACL_FORMAT_ND).ValidCount(6);
    auto ut = OP_API_UT(aclnnQuantize, INPUT(xDesc, scalesDesc, zeroPointsDesc, dType, axis), OUTPUT(outTensorDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// float32 no zeroPoints
TEST_F(l2_quantize_test, quantize_testcase_010_normal_fp32_no_zeroPoints)
{
    const vector<int64_t>& xShape = {3, 2};
    const vector<int64_t>& scalesShape = {2};
    aclDataType dType = ACL_INT8;
    int32_t axis = 1;
    const vector<int64_t>& outShape = {3, 2};

    auto xTensorDesc =
        TensorDesc(xShape, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0.5f, 1.2f, 1.3f, 1.4f, 2.9f, 3.2f});
    auto scalesTensorDesc = TensorDesc(scalesShape, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0.5f, 0.8f});
    auto outTensorDesc = TensorDesc(outShape, dType, ACL_FORMAT_ND).ValidCount(6);
    auto ut = OP_API_UT(
        aclnnQuantize, INPUT(xTensorDesc, scalesTensorDesc, (aclTensor*)nullptr, dType, axis), OUTPUT(outTensorDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// float16
TEST_F(l2_quantize_test, quantize_testcase_011_normal_fp16)
{
    auto xDesc = TensorDesc({3, 2}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto scalesDesc = TensorDesc({2}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto zeroPointsDesc = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND).ValueRange(1, 5);
    aclDataType dType = ACL_INT8;
    int32_t axis = 1;
    const vector<int64_t>& outShape = {3, 2};
    auto outTensorDesc = TensorDesc(outShape, dType, ACL_FORMAT_ND).ValidCount(6);
    auto ut = OP_API_UT(aclnnQuantize, INPUT(xDesc, scalesDesc, zeroPointsDesc, dType, axis), OUTPUT(outTensorDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// float32
TEST_F(l2_quantize_test, quantize_testcase_012_normal_fp32)
{
    auto xDesc = TensorDesc({3, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto scalesDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto zeroPointsDesc = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND).ValueRange(1, 5);
    aclDataType dType = ACL_INT8;
    int32_t axis = 1;
    const vector<int64_t>& outShape = {3, 2};
    auto outTensorDesc = TensorDesc(outShape, dType, ACL_FORMAT_ND).ValidCount(6);
    auto ut = OP_API_UT(aclnnQuantize, INPUT(xDesc, scalesDesc, zeroPointsDesc, dType, axis), OUTPUT(outTensorDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// float16
TEST_F(l2_quantize_test, quantize_testcase_013_normal_fp16_pertensor)
{
    auto xDesc = TensorDesc({3, 2}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto scalesDesc = TensorDesc({1}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto zeroPointsDesc = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND).ValueRange(1, 5);
    aclDataType dType = ACL_INT8;
    int32_t axis = 1;
    const vector<int64_t>& outShape = {3, 2};
    auto outTensorDesc = TensorDesc(outShape, dType, ACL_FORMAT_ND).ValidCount(6);
    auto ut = OP_API_UT(aclnnQuantize, INPUT(xDesc, scalesDesc, zeroPointsDesc, dType, axis), OUTPUT(outTensorDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_quantize_test, quantize_testcase_014_exception_x_private_format_invaild)
{
    auto xDesc = TensorDesc({3, 2}, ACL_FLOAT, ACL_FORMAT_NC1HWC0).ValueRange(-1, 1);
    auto scalesDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto zeroPointsDesc = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND).ValueRange(1, 5);
    aclDataType dType = ACL_INT8;
    int32_t axis = 1;
    const vector<int64_t>& outShape = {3, 2};
    auto outTensorDesc = TensorDesc(outShape, dType, ACL_FORMAT_ND).ValidCount(6);
    auto ut = OP_API_UT(aclnnQuantize, INPUT(xDesc, scalesDesc, zeroPointsDesc, dType, axis), OUTPUT(outTensorDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_quantize_test, quantize_testcase_015_exception_scales_private_format_invaild)
{
    auto xDesc = TensorDesc({3, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto scalesDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_NC1HWC0).ValueRange(-1, 1);
    auto zeroPointsDesc = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND).ValueRange(1, 5);
    aclDataType dType = ACL_INT8;
    int32_t axis = 1;
    const vector<int64_t>& outShape = {3, 2};
    auto outTensorDesc = TensorDesc(outShape, dType, ACL_FORMAT_ND).ValidCount(6);
    auto ut = OP_API_UT(aclnnQuantize, INPUT(xDesc, scalesDesc, zeroPointsDesc, dType, axis), OUTPUT(outTensorDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_quantize_test, quantize_testcase_016_exception_zeroPoints_private_format_invaild)
{
    auto xDesc = TensorDesc({3, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto scalesDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto zeroPointsDesc = TensorDesc({2}, ACL_INT32, ACL_FORMAT_NC1HWC0).ValueRange(1, 5);
    aclDataType dType = ACL_INT8;
    int32_t axis = 1;
    const vector<int64_t>& outShape = {3, 2};
    auto outTensorDesc = TensorDesc(outShape, dType, ACL_FORMAT_ND).ValidCount(6);
    auto ut = OP_API_UT(aclnnQuantize, INPUT(xDesc, scalesDesc, zeroPointsDesc, dType, axis), OUTPUT(outTensorDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_quantize_test, quantize_testcase_017_exception_out_private_format_invaild)
{
    auto xDesc = TensorDesc({3, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto scalesDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto zeroPointsDesc = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND).ValueRange(1, 5);
    aclDataType dType = ACL_INT8;
    int32_t axis = 1;
    const vector<int64_t>& outShape = {3, 2};
    auto outTensorDesc = TensorDesc(outShape, dType, ACL_FORMAT_NC1HWC0).ValidCount(6);
    auto ut = OP_API_UT(aclnnQuantize, INPUT(xDesc, scalesDesc, zeroPointsDesc, dType, axis), OUTPUT(outTensorDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

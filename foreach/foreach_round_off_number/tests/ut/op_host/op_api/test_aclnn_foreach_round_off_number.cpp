/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <memory>
#include <vector>
#include "gtest/gtest.h"
#include "../../../../op_host/op_api/aclnn_foreach_round_off_number.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace std;

class l2_foreach_round_off_number_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "foreach_round_off_number_test SetUp" << endl; }

    static void TearDownTestCase() { cout << "foreach_round_off_number_test TearDown" << endl; }
};

// Test names below carry no platform gating: the UT framework derives only dump
// file prefixes from them, and cases run on the default platform unless a case
// body sets a SocVersionManager scope explicitly (see unsupported_soc_*).
// support non-contiguous input and output
TEST_F(l2_foreach_round_off_number_test, foreach_round_off_number_test_non_contiguous)
{
    vector<int64_t> selfDims = {2, 4};
    auto roundMode = TensorDesc({1}, ACL_INT8, ACL_FORMAT_ND).ValueRange(0, 1);
    vector<int64_t> outDims = {2, 4};
    auto x = TensorDesc(selfDims, ACL_FLOAT, ACL_FORMAT_ND, {1, 2}, 0, {4, 2}).ValueRange(-1, 1);
    auto out = TensorDesc(outDims, ACL_FLOAT, ACL_FORMAT_ND, {1, 2}, 0, {4, 2}).Precision(0.001, 0.001);
    auto xList = TensorListDesc({x});
    auto outList = TensorListDesc({out});

    auto ut = OP_API_UT(aclnnForeachRoundOffNumber, INPUT(xList, roundMode), OUTPUT(outList));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
}

// INT16 is on both the 910BC and the 950 support lists
TEST_F(l2_foreach_round_off_number_test, foreach_round_off_number_test_int16)
{
    vector<int64_t> selfDims = {2, 4};
    auto roundMode = TensorDesc({1}, ACL_INT8, ACL_FORMAT_ND).ValueRange(0, 1);
    auto x = TensorDesc(selfDims, ACL_INT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto out = TensorDesc(selfDims, ACL_INT16, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto xList = TensorListDesc({x});
    auto outList = TensorListDesc({out});

    auto ut = OP_API_UT(aclnnForeachRoundOffNumber, INPUT(xList, roundMode), OUTPUT(outList));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
}

// roundMode carries scalar semantics: its element count must be 1
TEST_F(l2_foreach_round_off_number_test, foreach_round_off_number_test_roundmode_numel_not_1)
{
    vector<int64_t> selfDims = {2, 4};
    auto roundMode = TensorDesc({2}, ACL_INT8, ACL_FORMAT_ND).ValueRange(0, 1);
    auto x = TensorDesc(selfDims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto out = TensorDesc(selfDims, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto xList = TensorListDesc({x});
    auto outList = TensorListDesc({out});

    auto ut = OP_API_UT(aclnnForeachRoundOffNumber, INPUT(xList, roundMode), OUTPUT(outList));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// x and out must hold the same number of tensors
TEST_F(l2_foreach_round_off_number_test, foreach_round_off_number_test_x_out_size_mismatch)
{
    auto x = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto out = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto roundMode = TensorDesc({1}, ACL_INT8, ACL_FORMAT_ND).ValueRange(0, 1);
    auto xList = TensorListDesc({x, x});
    auto outList = TensorListDesc({out});

    auto ut = OP_API_UT(aclnnForeachRoundOffNumber, INPUT(xList, roundMode), OUTPUT(outList));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// tensor list capacity is capped by the arch35 TilingData fixed array
TEST_F(l2_foreach_round_off_number_test, foreach_round_off_number_test_tensor_num_exceed)
{
    auto x = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto out = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto roundMode = TensorDesc({1}, ACL_INT8, ACL_FORMAT_ND).ValueRange(0, 1);
    auto xList = TensorListDesc(257, x);
    auto outList = TensorListDesc(257, out);

    auto ut = OP_API_UT(aclnnForeachRoundOffNumber, INPUT(xList, roundMode), OUTPUT(outList));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// x dtype outside the support list (fp16/fp32/bf16/int16) is rejected
TEST_F(l2_foreach_round_off_number_test, foreach_round_off_number_test_unsupported_dtype)
{
    vector<int64_t> selfDims = {2, 2};
    auto roundMode = TensorDesc({1}, ACL_INT8, ACL_FORMAT_ND).ValueRange(0, 1);
    auto x = TensorDesc(selfDims, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto out = TensorDesc(selfDims, ACL_DOUBLE, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto xList = TensorListDesc({x});
    auto outList = TensorListDesc({out});

    auto ut = OP_API_UT(aclnnForeachRoundOffNumber, INPUT(xList, roundMode), OUTPUT(outList));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// out and x must share one dtype: a mismatched out member is rejected
TEST_F(l2_foreach_round_off_number_test, foreach_round_off_number_test_out_dtype_different)
{
    vector<int64_t> selfDims = {2, 2};
    auto roundMode = TensorDesc({1}, ACL_INT8, ACL_FORMAT_ND).ValueRange(0, 1);
    auto x = TensorDesc(selfDims, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto out = TensorDesc(selfDims, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto xList = TensorListDesc({x});
    auto outList = TensorListDesc({out});

    auto ut = OP_API_UT(aclnnForeachRoundOffNumber, INPUT(xList, roundMode), OUTPUT(outList));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// out and x shapes must be equal memberwise
TEST_F(l2_foreach_round_off_number_test, foreach_round_off_number_test_shape_different)
{
    auto roundMode = TensorDesc({1}, ACL_INT8, ACL_FORMAT_ND).ValueRange(0, 1);
    auto x = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto out = TensorDesc({2, 6}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto xList = TensorListDesc({x});
    auto outList = TensorListDesc({out});

    auto ut = OP_API_UT(aclnnForeachRoundOffNumber, INPUT(xList, roundMode), OUTPUT(outList));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// empty tensor list is a legal no-op: early SUCCESS with zero workspace
TEST_F(l2_foreach_round_off_number_test, foreach_round_off_number_test_empty_tensor_list)
{
    std::unique_ptr<aclTensorList, decltype(&aclDestroyTensorList)> x(aclCreateTensorList(nullptr, 0),
                                                                      &aclDestroyTensorList);
    ASSERT_NE(x.get(), nullptr);
    std::unique_ptr<aclTensorList, decltype(&aclDestroyTensorList)> out(aclCreateTensorList(nullptr, 0),
                                                                        &aclDestroyTensorList);
    ASSERT_NE(out.get(), nullptr);
    auto roundMode = TensorDesc({1}, ACL_INT8, ACL_FORMAT_ND).ValueRange(0, 1).ToAclType();
    uint64_t workspaceSize = 100;
    aclOpExecutor* executor = nullptr;
    auto status = aclnnForeachRoundOffNumberGetWorkspaceSize(x.get(), roundMode.get(), out.get(), &workspaceSize,
                                                             &executor);
    std::unique_ptr<aclOpExecutor> executorGuard(executor);
    EXPECT_EQ(status, ACLNN_SUCCESS);
    EXPECT_EQ(workspaceSize, 0);
}

// soc outside {950 regbase, 910BC}: dtype support list is empty
TEST_F(l2_foreach_round_off_number_test, unsupported_soc_foreach_round_off_number_test)
{
    op::SocVersionManager socVersion(op::SocVersion::ASCEND910);
    vector<int64_t> selfDims = {2, 4};
    auto roundMode = TensorDesc({1}, ACL_INT8, ACL_FORMAT_ND).ValueRange(0, 1);
    auto x = TensorDesc(selfDims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto out = TensorDesc(selfDims, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto xList = TensorListDesc({x});
    auto outList = TensorListDesc({out});

    auto ut = OP_API_UT(aclnnForeachRoundOffNumber, INPUT(xList, roundMode), OUTPUT(outList));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// null x / roundMode / out are rejected by CheckNotNull before any dereference
TEST_F(l2_foreach_round_off_number_test, foreach_round_off_number_test_nullptr)
{
    auto roundMode = TensorDesc({1}, ACL_INT8, ACL_FORMAT_ND).ValueRange(0, 1).ToAclType();
    auto x = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto out = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto xList = TensorListDesc({x});
    auto outList = TensorListDesc({out});
    uint64_t workspaceSize = 100;
    aclOpExecutor* executor = nullptr;

    auto status = aclnnForeachRoundOffNumberGetWorkspaceSize(nullptr, roundMode.get(), outList.ToAclTypeRawPtr(),
                                                             &workspaceSize, &executor);
    EXPECT_EQ(status, ACLNN_ERR_PARAM_NULLPTR);
    status = aclnnForeachRoundOffNumberGetWorkspaceSize(xList.ToAclTypeRawPtr(), nullptr, outList.ToAclTypeRawPtr(),
                                                        &workspaceSize, &executor);
    EXPECT_EQ(status, ACLNN_ERR_PARAM_NULLPTR);
    status = aclnnForeachRoundOffNumberGetWorkspaceSize(xList.ToAclTypeRawPtr(), roundMode.get(), nullptr,
                                                        &workspaceSize, &executor);
    EXPECT_EQ(status, ACLNN_ERR_PARAM_NULLPTR);
}

// x members must share one dtype: a mixed list cannot be dispatched by a single template
TEST_F(l2_foreach_round_off_number_test, foreach_round_off_number_test_dtype_mixed_in_list)
{
    auto roundMode = TensorDesc({1}, ACL_INT8, ACL_FORMAT_ND).ValueRange(0, 1);
    auto xFp16 = TensorDesc({2, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto xFp32 = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto out = TensorDesc({2, 4}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto xList = TensorListDesc({xFp16, xFp32});
    auto outList = TensorListDesc({out, out});

    auto ut = OP_API_UT(aclnnForeachRoundOffNumber, INPUT(xList, roundMode), OUTPUT(outList));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

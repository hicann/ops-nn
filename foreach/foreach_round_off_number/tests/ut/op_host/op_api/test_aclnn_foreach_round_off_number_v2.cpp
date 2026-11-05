/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <memory>
#include <vector>
#include "gtest/gtest.h"

#include "../../../../op_host/op_api/aclnn_foreach_round_off_number_v2.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include <iostream>
#include "opdev/platform.h"

using namespace std;

class l2_foreach_round_off_number_v2_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "l2 foreach_round_off_number_test SetUp" << endl; }

    static void TearDownTestCase() { cout << "l2 foreach_round_off_number_test TearDown" << endl; }
};

// dtype DOUBLE
TEST_F(l2_foreach_round_off_number_v2_test, foreach_round_off_number_v2_test_double)
{
    vector<vector<int64_t>> selfDims = {{2, 2}};
    int8_t val = 1;
    auto scalar_desc = ScalarDesc(val);
    vector<vector<int64_t>> outDims = {{2, 2}};
    auto x = TensorDesc(selfDims[0], ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto out = TensorDesc(outDims[0], ACL_DOUBLE, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto xList = TensorListDesc({x});
    auto outList = TensorListDesc({out});

    auto ut = OP_API_UT(aclnnForeachRoundOffNumberV2, INPUT(xList, scalar_desc), OUTPUT(outList));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_foreach_round_off_number_v2_test, foreach_round_off_number_v2_test_fp32)
{
    vector<vector<int64_t>> selfDims = {{2, 2}};
    int8_t val = 1;
    auto scalar_desc = ScalarDesc(val);
    vector<vector<int64_t>> outDims = {{2, 2}};
    auto x = TensorDesc(selfDims[0], ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto out = TensorDesc(outDims[0], ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto xList = TensorListDesc({x});
    auto outList = TensorListDesc({out});

    auto ut = OP_API_UT(aclnnForeachRoundOffNumberV2, INPUT(xList, scalar_desc), OUTPUT(outList));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
}

// out and self different dtype
TEST_F(l2_foreach_round_off_number_v2_test, foreach_round_off_number_v2_test_dtype_different)
{
    vector<vector<int64_t>> selfDims = {{2, 2}};
    int8_t val = 1;
    auto scalar_desc = ScalarDesc(val);
    vector<vector<int64_t>> outDims = {{2, 2}};
    auto x = TensorDesc(selfDims[0], ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto out = TensorDesc(outDims[0], ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto xList = TensorListDesc({x});
    auto outList = TensorListDesc({out});

    auto ut = OP_API_UT(aclnnForeachRoundOffNumberV2, INPUT(xList, scalar_desc), OUTPUT(outList));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// out and self different shape
TEST_F(l2_foreach_round_off_number_v2_test, foreach_round_off_number_v2_test_shape_different)
{
    vector<vector<int64_t>> selfDims = {{2, 2}};
    int8_t val = 1;
    auto scalar_desc = ScalarDesc(val);
    vector<vector<int64_t>> outDims = {{2, 6}};
    auto x = TensorDesc(selfDims[0], ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto out = TensorDesc(outDims[0], ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto xList = TensorListDesc({x});
    auto outList = TensorListDesc({out});

    auto ut = OP_API_UT(aclnnForeachRoundOffNumberV2, INPUT(xList, scalar_desc), OUTPUT(outList));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// support non-contiguous input and output
TEST_F(l2_foreach_round_off_number_v2_test, foreach_round_off_number_v2_test_non_contiguous)
{
    vector<int64_t> selfDims = {2, 4};
    int8_t val = 1;
    auto scalar_desc = ScalarDesc(val);
    vector<int64_t> outDims = {2, 4};
    auto x = TensorDesc(selfDims, ACL_FLOAT, ACL_FORMAT_ND, {1, 2}, 0, {4, 2}).ValueRange(-1, 1);
    auto out = TensorDesc(outDims, ACL_FLOAT, ACL_FORMAT_ND, {1, 2}, 0, {4, 2}).Precision(0.001, 0.001);
    auto xList = TensorListDesc({x});
    auto outList = TensorListDesc({out});

    auto ut = OP_API_UT(aclnnForeachRoundOffNumberV2, INPUT(xList, scalar_desc), OUTPUT(outList));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
}

// INT16 is on both the 910BC and the 950 support lists
TEST_F(l2_foreach_round_off_number_v2_test, foreach_round_off_number_v2_test_int16)
{
    vector<int64_t> selfDims = {2, 4};
    int8_t val = 1;
    auto scalar_desc = ScalarDesc(val);
    auto x = TensorDesc(selfDims, ACL_INT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto out = TensorDesc(selfDims, ACL_INT16, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto xList = TensorListDesc({x});
    auto outList = TensorListDesc({out});

    auto ut = OP_API_UT(aclnnForeachRoundOffNumberV2, INPUT(xList, scalar_desc), OUTPUT(outList));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
}

// roundMode scalar is narrowed into an INT8 tensor: values outside the INT8 range are rejected
TEST_F(l2_foreach_round_off_number_v2_test, foreach_round_off_number_v2_test_scalar_out_of_int8_range)
{
    vector<int64_t> selfDims = {2, 4};
    int64_t val = 128;
    auto scalar_desc = ScalarDesc(val);
    auto x = TensorDesc(selfDims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto out = TensorDesc(selfDims, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto xList = TensorListDesc({x});
    auto outList = TensorListDesc({out});

    auto ut = OP_API_UT(aclnnForeachRoundOffNumberV2, INPUT(xList, scalar_desc), OUTPUT(outList));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// x and out must hold the same number of tensors
TEST_F(l2_foreach_round_off_number_v2_test, foreach_round_off_number_v2_test_x_out_size_mismatch)
{
    auto x = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto out = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    int8_t val = 1;
    auto scalar_desc = ScalarDesc(val);
    auto xList = TensorListDesc({x, x});
    auto outList = TensorListDesc({out});

    auto ut = OP_API_UT(aclnnForeachRoundOffNumberV2, INPUT(xList, scalar_desc), OUTPUT(outList));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// tensor list capacity is capped by the arch35 TilingData fixed array
TEST_F(l2_foreach_round_off_number_v2_test, foreach_round_off_number_v2_test_tensor_num_exceed)
{
    auto x = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto out = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    int8_t val = 1;
    auto scalar_desc = ScalarDesc(val);
    auto xList = TensorListDesc(257, x);
    auto outList = TensorListDesc(257, out);

    auto ut = OP_API_UT(aclnnForeachRoundOffNumberV2, INPUT(xList, scalar_desc), OUTPUT(outList));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// empty tensor list is a legal no-op: early SUCCESS with zero workspace
TEST_F(l2_foreach_round_off_number_v2_test, foreach_round_off_number_v2_test_empty_tensor_list)
{
    std::unique_ptr<aclTensorList, decltype(&aclDestroyTensorList)> x(aclCreateTensorList(nullptr, 0),
                                                                      &aclDestroyTensorList);
    ASSERT_NE(x.get(), nullptr);
    std::unique_ptr<aclTensorList, decltype(&aclDestroyTensorList)> out(aclCreateTensorList(nullptr, 0),
                                                                        &aclDestroyTensorList);
    ASSERT_NE(out.get(), nullptr);
    auto scalar = ScalarDesc(static_cast<int8_t>(1)).ToAclType();
    uint64_t workspaceSize = 100;
    aclOpExecutor* executor = nullptr;
    auto status = aclnnForeachRoundOffNumberV2GetWorkspaceSize(x.get(), scalar.get(), out.get(), &workspaceSize,
                                                               &executor);
    std::unique_ptr<aclOpExecutor> executorGuard(executor);
    EXPECT_EQ(status, ACLNN_SUCCESS);
    EXPECT_EQ(workspaceSize, 0);
}

// null x / roundMode / out are rejected by CheckNotNull before any dereference
TEST_F(l2_foreach_round_off_number_v2_test, foreach_round_off_number_v2_test_nullptr)
{
    auto scalar = ScalarDesc(static_cast<int8_t>(1)).ToAclType();
    auto x = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto out = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto xList = TensorListDesc({x});
    auto outList = TensorListDesc({out});
    uint64_t workspaceSize = 100;
    aclOpExecutor* executor = nullptr;

    auto status = aclnnForeachRoundOffNumberV2GetWorkspaceSize(nullptr, scalar.get(), outList.ToAclTypeRawPtr(),
                                                               &workspaceSize, &executor);
    EXPECT_EQ(status, ACLNN_ERR_PARAM_NULLPTR);
    status = aclnnForeachRoundOffNumberV2GetWorkspaceSize(xList.ToAclTypeRawPtr(), nullptr, outList.ToAclTypeRawPtr(),
                                                          &workspaceSize, &executor);
    EXPECT_EQ(status, ACLNN_ERR_PARAM_NULLPTR);
    status = aclnnForeachRoundOffNumberV2GetWorkspaceSize(xList.ToAclTypeRawPtr(), scalar.get(), nullptr,
                                                          &workspaceSize, &executor);
    EXPECT_EQ(status, ACLNN_ERR_PARAM_NULLPTR);
}

// soc outside {950 regbase, 910BC}: dtype support list is empty
TEST_F(l2_foreach_round_off_number_v2_test, unsupported_soc_foreach_round_off_number_v2_test)
{
    op::SocVersionManager socVersion(op::SocVersion::ASCEND910);
    vector<int64_t> selfDims = {2, 4};
    int8_t val = 1;
    auto scalar_desc = ScalarDesc(val);
    auto x = TensorDesc(selfDims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto out = TensorDesc(selfDims, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto xList = TensorListDesc({x});
    auto outList = TensorListDesc({out});

    auto ut = OP_API_UT(aclnnForeachRoundOffNumberV2, INPUT(xList, scalar_desc), OUTPUT(outList));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

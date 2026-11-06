/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <vector>
#include <array>
#include <float.h>
#include "gtest/gtest.h"
#include "../../../../op_host/op_api/aclnn_foreach_log1p.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace std;

class l2_foreach_log1p_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "foreach_log1p_test SetUp" << endl; }

    static void TearDownTestCase() { cout << "foreach_log1p_test TearDown" << endl; }
};

// support non-contiguous input and output
TEST_F(l2_foreach_log1p_test, ascend910B2_foreach_log1p_test_non_contiguous)
{
    vector<int64_t> selfDims = {2, 4};
    vector<int64_t> outDims = {2, 4};
    auto x = TensorDesc(selfDims, ACL_FLOAT, ACL_FORMAT_ND, {1, 2}, 0, {4, 2}).ValueRange(0.5, 2.0);
    auto out = TensorDesc(outDims, ACL_FLOAT, ACL_FORMAT_ND, {1, 2}, 0, {4, 2}).Precision(0.001, 0.001);
    auto xList = TensorListDesc({x});
    auto outList = TensorListDesc({out});

    auto ut = OP_API_UT(aclnnForeachLog1p, INPUT(xList), OUTPUT(outList));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
}

// An empty tensor list carries no data, workspace stays empty and the call must succeed.
TEST_F(l2_foreach_log1p_test, foreach_log1p_test_empty_tensor_list)
{
    aclTensorList* xList = aclCreateTensorList(nullptr, 0);
    aclTensorList* outList = aclCreateTensorList(nullptr, 0);
    ASSERT_NE(xList, nullptr);
    ASSERT_NE(outList, nullptr);

    uint64_t workspaceSize = 1;
    aclOpExecutor* executor = nullptr;
    aclnnStatus ret = aclnnForeachLog1pGetWorkspaceSize(xList, outList, &workspaceSize, &executor);
    EXPECT_EQ(ret, ACL_SUCCESS);
    EXPECT_EQ(workspaceSize, 0U);
    delete executor;
    aclDestroyTensorList(xList);
    aclDestroyTensorList(outList);
}

// self and out must hold the same number of tensors.
TEST_F(l2_foreach_log1p_test, foreach_log1p_test_tensor_count_mismatch)
{
    auto x = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.5, 2.0);
    auto out = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto xList = TensorListDesc({x});
    auto outList = TensorListDesc({out, out});

    auto ut = OP_API_UT(aclnnForeachLog1p, INPUT(xList), OUTPUT(outList));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// The tensor list count is capped at the tiling-side static capacity (256).
TEST_F(l2_foreach_log1p_test, foreach_log1p_test_tensor_count_exceeds_limit)
{
    auto x = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.5, 2.0);
    auto out = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto xList = TensorListDesc(257, x);
    auto outList = TensorListDesc(257, out);

    auto ut = OP_API_UT(aclnnForeachLog1p, INPUT(xList), OUTPUT(outList));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_INNER_TILING_ERROR);
}

// Platforms outside the supported soc list have no dtype support list and must be rejected.
TEST_F(l2_foreach_log1p_test, foreach_log1p_test_unsupported_soc)
{
    op::SocVersionManager socVersion(op::SocVersion::RESERVED_VERSION);
    auto x = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.5, 2.0);
    auto out = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto xList = TensorListDesc({x});
    auto outList = TensorListDesc({out});

    auto ut = OP_API_UT(aclnnForeachLog1p, INPUT(xList), OUTPUT(outList));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// The second-phase run entry must fail cleanly on a null executor instead of crashing.
TEST_F(l2_foreach_log1p_test, foreach_log1p_test_run_with_null_executor)
{
    aclnnStatus runResult = aclnnForeachLog1p(nullptr, 0, nullptr, nullptr);
    EXPECT_NE(runResult, ACL_SUCCESS);
}

// Positive: 9-dim input is a legal input domain and passes parameter validation.
TEST_F(l2_foreach_log1p_test, foreach_log1p_test_dim9_supported)
{
    vector<int64_t> selfDims = {1, 1, 1, 1, 1, 1, 1, 1, 2};
    vector<int64_t> outDims = {1, 1, 1, 1, 1, 1, 1, 1, 2};
    auto x = TensorDesc(selfDims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.5, 2.0);
    auto out = TensorDesc(outDims, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto xList = TensorListDesc({x});
    auto outList = TensorListDesc({out});

    auto ut = OP_API_UT(aclnnForeachLog1p, INPUT(xList), OUTPUT(outList));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
}

// Negative: bool dtype is outside the supported dtype set and must be rejected.
TEST_F(l2_foreach_log1p_test, foreach_log1p_test_bool_dtype_rejected)
{
    auto x = TensorDesc({2, 4}, ACL_BOOL, ACL_FORMAT_ND).ValueRange(0, 1);
    auto out = TensorDesc({2, 4}, ACL_BOOL, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto xList = TensorListDesc({x});
    auto outList = TensorListDesc({out});

    auto ut = OP_API_UT(aclnnForeachLog1p, INPUT(xList), OUTPUT(outList));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// Negative: complex128 dtype is outside the supported dtype set and must be rejected.
TEST_F(l2_foreach_log1p_test, foreach_log1p_test_complex128_dtype_rejected)
{
    auto x = TensorDesc({2, 4}, ACL_COMPLEX128, ACL_FORMAT_ND).ValueRange(0.5, 2.0);
    auto out = TensorDesc({2, 4}, ACL_COMPLEX128, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto xList = TensorListDesc({x});
    auto outList = TensorListDesc({out});

    auto ut = OP_API_UT(aclnnForeachLog1p, INPUT(xList), OUTPUT(outList));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// Negative: the out dtype must match the x dtype.
TEST_F(l2_foreach_log1p_test, foreach_log1p_test_out_dtype_mismatch_rejected)
{
    auto x = TensorDesc({2, 4}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0.5, 2.0);
    auto out = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto xList = TensorListDesc({x});
    auto outList = TensorListDesc({out});

    auto ut = OP_API_UT(aclnnForeachLog1p, INPUT(xList), OUTPUT(outList));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// Negative: a null x tensor list pointer must be rejected by the first-phase null check.
TEST_F(l2_foreach_log1p_test, foreach_log1p_test_null_self_rejected)
{
    auto out = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto outList = TensorListDesc({out});

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus ret = aclnnForeachLog1pGetWorkspaceSize(nullptr, outList.ToAclTypeRawPtr(), &workspaceSize, &executor);
    EXPECT_EQ(ret, ACLNN_ERR_PARAM_NULLPTR);
}

// Negative: a null out tensor list pointer must be rejected by the first-phase null check.
TEST_F(l2_foreach_log1p_test, foreach_log1p_test_null_out_rejected)
{
    auto x = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.5, 2.0);
    auto xList = TensorListDesc({x});

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus ret = aclnnForeachLog1pGetWorkspaceSize(xList.ToAclTypeRawPtr(), nullptr, &workspaceSize, &executor);
    EXPECT_EQ(ret, ACLNN_ERR_PARAM_NULLPTR);
}

// Negative: out shape must equal the corresponding x shape.
TEST_F(l2_foreach_log1p_test, foreach_log1p_test_shape_mismatch_rejected)
{
    auto x = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.5, 2.0);
    auto out = TensorDesc({3, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto xList = TensorListDesc({x});
    auto outList = TensorListDesc({out});

    auto ut = OP_API_UT(aclnnForeachLog1p, INPUT(xList), OUTPUT(outList));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// Negative: dtypes inside the x tensor list must be consistent.
TEST_F(l2_foreach_log1p_test, foreach_log1p_test_in_list_dtype_mismatch_rejected)
{
    auto xFp16 = TensorDesc({2, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0.5, 2.0);
    auto xFp32 = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.5, 2.0);
    auto out = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto xList = TensorListDesc({xFp16, xFp32});
    auto outList = TensorListDesc({out, out});

    auto ut = OP_API_UT(aclnnForeachLog1p, INPUT(xList), OUTPUT(outList));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

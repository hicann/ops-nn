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
#include <float.h>
#include "gtest/gtest.h"
#include "../../../op_host/op_api/aclnn_foreach_tan.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace std;

namespace {

// Direct-call helper for cases the OP_API_UT harness cannot express (empty list, null entries).
static aclnnStatus GetWorkspaceStatus(const aclTensorList* x, aclTensorList* out, uint64_t* workspaceSize)
{
    aclOpExecutor* executor = nullptr;
    auto status = aclnnForeachTanGetWorkspaceSize(x, out, workspaceSize, &executor);
    std::unique_ptr<aclOpExecutor> executorGuard(executor);
    return status;
}

static TensorDesc MakeTanTensor(aclDataType dtype) { return TensorDesc({2, 4}, dtype, ACL_FORMAT_ND); }

} // namespace

class l2_foreach_tan_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "foreach_tan_test SetUp" << endl; }

    static void TearDownTestCase() { cout << "foreach_tan_test TearDown" << endl; }
};

// support non-contiguous input and output
TEST_F(l2_foreach_tan_test, ascend910B2_foreach_tan_test_non_contiguous)
{
    vector<int64_t> selfDims = {2, 4};
    vector<int64_t> outDims = {2, 4};
    auto x = TensorDesc(selfDims, ACL_FLOAT, ACL_FORMAT_ND, {1, 2}, 0, {4, 2}).ValueRange(-1, 1);
    auto out = TensorDesc(outDims, ACL_FLOAT, ACL_FORMAT_ND, {1, 2}, 0, {4, 2}).Precision(0.001, 0.001);
    auto xList = TensorListDesc({x});
    auto outList = TensorListDesc({out});

    auto ut = OP_API_UT(aclnnForeachTan, INPUT(xList), OUTPUT(outList));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
}

// all dtypes advertised for ASCEND910B/910_93/950 are accepted
TEST_F(l2_foreach_tan_test, ascend910b2_supported_dtypes)
{
    for (auto dtype : {ACL_FLOAT, ACL_FLOAT16, ACL_BF16}) {
        SCOPED_TRACE(static_cast<int>(dtype));
        auto xList = TensorListDesc({MakeTanTensor(dtype)});
        auto outList = TensorListDesc({MakeTanTensor(dtype)});
        auto ut = OP_API_UT(aclnnForeachTan, INPUT(xList), OUTPUT(outList));
        uint64_t workspaceSize = 0;
        EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACL_SUCCESS);
    }
}

// soc without a registered support list is rejected
TEST_F(l2_foreach_tan_test, unsupported_soc_rejected)
{
    op::SocVersionManager versionManager(op::SocVersion::RESERVED_VERSION);
    auto xList = TensorListDesc({MakeTanTensor(ACL_FLOAT)});
    auto outList = TensorListDesc({MakeTanTensor(ACL_FLOAT)});
    auto ut = OP_API_UT(aclnnForeachTan, INPUT(xList), OUTPUT(outList));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
}

// input and output lists must have the same number of tensors
TEST_F(l2_foreach_tan_test, list_size_mismatch_rejected)
{
    auto xList = TensorListDesc({MakeTanTensor(ACL_FLOAT), MakeTanTensor(ACL_FLOAT)});
    auto outList = TensorListDesc({MakeTanTensor(ACL_FLOAT)});
    auto ut = OP_API_UT(aclnnForeachTan, INPUT(xList), OUTPUT(outList));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
}

// lists longer than 256 tensors are rejected
TEST_F(l2_foreach_tan_test, excessive_list_length_rejected)
{
    auto xList = TensorListDesc(257, MakeTanTensor(ACL_FLOAT));
    auto outList = TensorListDesc(257, MakeTanTensor(ACL_FLOAT));
    auto ut = OP_API_UT(aclnnForeachTan, INPUT(xList), OUTPUT(outList));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
}

// exactly 256 tensors is the accepted upper bound
TEST_F(l2_foreach_tan_test, maximum_list_length_accepted)
{
    auto xList = TensorListDesc(256, MakeTanTensor(ACL_FLOAT));
    auto outList = TensorListDesc(256, MakeTanTensor(ACL_FLOAT));
    auto ut = OP_API_UT(aclnnForeachTan, INPUT(xList), OUTPUT(outList));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACL_SUCCESS);
}

// empty tensor lists need no kernel launch and report zero workspace
TEST_F(l2_foreach_tan_test, empty_list_accepted_with_zero_workspace)
{
    std::unique_ptr<aclTensorList, decltype(&aclDestroyTensorList)> x(aclCreateTensorList(nullptr, 0),
                                                                      &aclDestroyTensorList);
    std::unique_ptr<aclTensorList, decltype(&aclDestroyTensorList)> out(aclCreateTensorList(nullptr, 0),
                                                                        &aclDestroyTensorList);
    ASSERT_NE(x.get(), nullptr);
    ASSERT_NE(out.get(), nullptr);
    uint64_t workspaceSize = 1;
    EXPECT_EQ(GetWorkspaceStatus(x.get(), out.get(), &workspaceSize), ACL_SUCCESS);
    EXPECT_EQ(workspaceSize, 0U);
}

// null list pointers are rejected
TEST_F(l2_foreach_tan_test, null_inputs_rejected)
{
    auto xList = TensorListDesc({MakeTanTensor(ACL_FLOAT)});
    auto outList = TensorListDesc({MakeTanTensor(ACL_FLOAT)});
    uint64_t workspaceSize = 0;
    EXPECT_EQ(GetWorkspaceStatus(nullptr, outList.ToAclTypeRawPtr(), &workspaceSize), ACLNN_ERR_PARAM_NULLPTR);
    EXPECT_EQ(GetWorkspaceStatus(xList.ToAclTypeRawPtr(), nullptr, &workspaceSize), ACLNN_ERR_PARAM_NULLPTR);
}

// a null tensor inside the list must be caught before dtype/shape checks dereference it
TEST_F(l2_foreach_tan_test, null_element_after_valid_tensor_rejected)
{
    const aclTensor* items[] = {MakeTanTensor(ACL_FLOAT).ToAclTypeRawPtr(), nullptr};
    // Use the ACL deleter directly because the UT descriptor deleter dereferences each entry.
    std::unique_ptr<aclTensorList, decltype(&aclDestroyTensorList)> x(aclCreateTensorList(items, 2),
                                                                      &aclDestroyTensorList);
    ASSERT_NE(x.get(), nullptr);
    auto outList = TensorListDesc({MakeTanTensor(ACL_FLOAT), MakeTanTensor(ACL_FLOAT)});
    uint64_t workspaceSize = 0;
    EXPECT_EQ(GetWorkspaceStatus(x.get(), outList.ToAclTypeRawPtr(), &workspaceSize), ACLNN_ERR_PARAM_INVALID);
}

// dtype outside the support list is rejected
TEST_F(l2_foreach_tan_test, unsupported_input_dtype_rejected)
{
    auto xList = TensorListDesc({MakeTanTensor(ACL_INT32)});
    auto outList = TensorListDesc({MakeTanTensor(ACL_INT32)});
    auto ut = OP_API_UT(aclnnForeachTan, INPUT(xList), OUTPUT(outList));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
}

// tensors inside the input list must share one dtype
TEST_F(l2_foreach_tan_test, mixed_input_dtypes_rejected)
{
    auto xList = TensorListDesc({MakeTanTensor(ACL_FLOAT), MakeTanTensor(ACL_FLOAT16)});
    auto outList = TensorListDesc({MakeTanTensor(ACL_FLOAT), MakeTanTensor(ACL_FLOAT16)});
    auto ut = OP_API_UT(aclnnForeachTan, INPUT(xList), OUTPUT(outList));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
}

// output list dtype must match the input list dtype
TEST_F(l2_foreach_tan_test, output_dtype_mismatch_rejected)
{
    auto xList = TensorListDesc({MakeTanTensor(ACL_FLOAT)});
    auto outList = TensorListDesc({MakeTanTensor(ACL_FLOAT16)});
    auto ut = OP_API_UT(aclnnForeachTan, INPUT(xList), OUTPUT(outList));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
}

// input and output shapes must match element-wise
TEST_F(l2_foreach_tan_test, shape_mismatch_rejected)
{
    auto xList = TensorListDesc({TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND)});
    auto outList = TensorListDesc({TensorDesc({3, 2}, ACL_FLOAT, ACL_FORMAT_ND)});
    auto ut = OP_API_UT(aclnnForeachTan, INPUT(xList), OUTPUT(outList));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
}

// Positive: 9-dim input is a legal input domain and passes parameter validation.
TEST_F(l2_foreach_tan_test, foreach_tan_test_dim9_supported)
{
    vector<int64_t> selfDims = {1, 1, 1, 1, 1, 1, 1, 1, 2};
    vector<int64_t> outDims = {1, 1, 1, 1, 1, 1, 1, 1, 2};
    auto x = TensorDesc(selfDims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto out = TensorDesc(outDims, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto xList = TensorListDesc({x});
    auto outList = TensorListDesc({out});

    auto ut = OP_API_UT(aclnnForeachTan, INPUT(xList), OUTPUT(outList));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
}

// phase-2 entry must fail fast on a null executor instead of crashing
TEST_F(l2_foreach_tan_test, phase2_null_executor_rejected)
{
    aclnnStatus ret = aclnnForeachTan(nullptr, 0, nullptr, nullptr);
    EXPECT_NE(ret, ACL_SUCCESS);
}

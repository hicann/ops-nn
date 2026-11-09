/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <array>
#include <vector>
#include "gtest/gtest.h"

#include "../../../op_host/op_api/aclnn_index_add.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"
#include "acl/acl_rt.h"
#include <dlfcn.h>

using namespace op;
using namespace std;

class l2_index_add_test : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "indexadd_test SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "indexadd_test TearDown" << std::endl; }
};

TEST_F(l2_index_add_test, case_NULLPTR)
{
    auto ut = OP_API_UT(aclnnIndexAdd,
                        INPUT((aclTensor*)nullptr, 0, (aclTensor*)nullptr, (aclTensor*)nullptr, (aclScalar*)nullptr),
                        OUTPUT((aclTensor*)nullptr));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_index_add_test, case_NULLPTR01)
{
    auto ut = OP_API_UT(aclnnIndexAdd,
                        INPUT((aclTensor*)nullptr, 0, (aclTensor*)nullptr, (aclTensor*)nullptr, (aclScalar*)nullptr),
                        OUTPUT((aclTensor*)nullptr));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_index_add_test, case_NULLPTR02)
{
    auto ut = OP_API_UT(aclnnIndexAdd,
                        INPUT((aclTensor*)nullptr, 0, (aclTensor*)nullptr, (aclTensor*)nullptr, (aclScalar*)nullptr),
                        OUTPUT((aclTensor*)nullptr));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_index_add_test, case_NULLPTR03)
{
    auto ut = OP_API_UT(aclnnIndexAdd,
                        INPUT((aclTensor*)nullptr, 0, (aclTensor*)nullptr, (aclTensor*)nullptr, (aclScalar*)nullptr),
                        OUTPUT((aclTensor*)nullptr));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_index_add_test, case_NULLPTR04_workspace)
{
    auto tensor_desc = TensorDesc({5, 3}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(-10, 10);
    int32_t dims = 0;
    auto index_desc = TensorDesc({3}, ACL_INT32, ACL_FORMAT_NCHW).Value(vector<int32_t>{0, 2, 4});
    auto source_desc = TensorDesc({3, 3}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(-10, 10);
    auto scalar_desc = ScalarDesc(1.0f);

    auto ut = OP_API_UT(aclnnIndexAdd, INPUT(tensor_desc, dims, index_desc, source_desc, scalar_desc),
                        OUTPUT(tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(nullptr);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_index_add_test, case_BF16_910_not_supported)
{
    auto tensor_desc = TensorDesc({5, 3}, ACL_INT32, ACL_FORMAT_NCHW).ValueRange(-10, 10);
    int32_t dims = 1;
    auto index_desc = TensorDesc({3}, ACL_INT32, ACL_FORMAT_NCHW).Value(vector<int32_t>{0, 2, 1});
    auto source_desc = TensorDesc({5, 3}, ACL_INT32, ACL_FORMAT_NCHW).ValueRange(-10, 10);
    auto scalar_desc = ScalarDesc(1.0f);
    auto out_desc = TensorDesc({5, 3}, ACL_BF16, ACL_FORMAT_NCHW).ValueRange(-10, 10);
    auto ut = OP_API_UT(aclnnIndexAdd, INPUT(tensor_desc, dims, index_desc, source_desc, scalar_desc),
                        OUTPUT(out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_index_add_test, case_self_dtype_INT64)
{
    auto tensor_desc = TensorDesc({5, 3}, ACL_INT64, ACL_FORMAT_NCHW).ValueRange(-10, 10);
    int32_t dims = 0;
    auto index_desc = TensorDesc({3}, ACL_INT32, ACL_FORMAT_NCHW).Value(vector<int32_t>{0, 2, 4});
    auto source_desc = TensorDesc({3, 3}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(-10, 10);
    auto scalar_desc = ScalarDesc(1.0f);

    auto ut = OP_API_UT(aclnnIndexAdd, INPUT(tensor_desc, dims, index_desc, source_desc, scalar_desc),
                        OUTPUT(tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_index_add_test, case_source_dtype_INT64)
{
    auto tensor_desc = TensorDesc({5, 3}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(-10, 10);
    int32_t dims = 0;
    auto index_desc = TensorDesc({3}, ACL_INT32, ACL_FORMAT_NCHW).Value(vector<int32_t>{0, 2, 4});
    auto source_desc = TensorDesc({3, 3}, ACL_INT64, ACL_FORMAT_NCHW).ValueRange(-10, 10);
    auto scalar_desc = ScalarDesc(1.0f);

    auto ut = OP_API_UT(aclnnIndexAdd, INPUT(tensor_desc, dims, index_desc, source_desc, scalar_desc),
                        OUTPUT(tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_index_add_test, case_source_error_dim)
{
    auto tensor_desc = TensorDesc({5, 3}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(-10, 10);
    int32_t dims = 0;
    auto index_desc = TensorDesc({3}, ACL_INT32, ACL_FORMAT_NCHW).Value(vector<int32_t>{0, 2, 4});
    auto source_desc = TensorDesc({3, 3, 5}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(-10, 10);
    auto scalar_desc = ScalarDesc(1.0f);

    auto ut = OP_API_UT(aclnnIndexAdd, INPUT(tensor_desc, dims, index_desc, source_desc, scalar_desc),
                        OUTPUT(tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_index_add_test, case_index_dtype_FLOAT)
{
    auto tensor_desc = TensorDesc({5, 3}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(-10, 10);
    int32_t dims = 0;
    auto index_desc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_NCHW).Value(vector<int32_t>{0, 2, 4});
    auto source_desc = TensorDesc({3, 3}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(-10, 10);
    auto scalar_desc = ScalarDesc(1.0f);

    auto ut = OP_API_UT(aclnnIndexAdd, INPUT(tensor_desc, dims, index_desc, source_desc, scalar_desc),
                        OUTPUT(tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_index_add_test, case_index_dtype_FLOAT_source_dtype_INT32)
{
    auto tensor_desc = TensorDesc({5, 3}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(-10, 10);
    int32_t dims = 0;
    auto index_desc = TensorDesc({3}, ACL_INT32, ACL_FORMAT_NCHW).Value(vector<int32_t>{0, 2, 4});
    auto source_desc = TensorDesc({3, 3}, ACL_INT32, ACL_FORMAT_NCHW).ValueRange(-10, 10);
    auto scalar_desc = ScalarDesc(1.0f);

    auto ut = OP_API_UT(aclnnIndexAdd, INPUT(tensor_desc, dims, index_desc, source_desc, scalar_desc),
                        OUTPUT(tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_index_add_test, case_index_dtype_FLOAT_out_dtype_INT32)
{
    auto tensor_desc = TensorDesc({5, 3}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(-10, 10);
    int32_t dims = 0;
    auto index_desc = TensorDesc({3}, ACL_INT32, ACL_FORMAT_NCHW).Value(vector<int32_t>{0, 2, 4});
    auto source_desc = TensorDesc({3, 3}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(-10, 10);
    auto scalar_desc = ScalarDesc(1.0f);
    auto out_desc = TensorDesc({5, 3}, ACL_INT32, ACL_FORMAT_NCHW).ValueRange(-10, 10);

    auto ut = OP_API_UT(aclnnIndexAdd, INPUT(tensor_desc, dims, index_desc, source_desc, scalar_desc),
                        OUTPUT(out_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// A 0-dim self is rejected by the first-stage validation with 161002 ("DimNum of self should be in
// range [1, 8]") instead of escaping the dim check and being rejected as 561103 by the
// InplaceIndexAddAiCore tiling.
TEST_F(l2_index_add_test, case_self_zero_dim)
{
    auto tensor_desc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_NCHW);
    int64_t dims = 0;
    auto index_desc = TensorDesc({3}, ACL_INT32, ACL_FORMAT_NCHW).Value(vector<int32_t>{0, 2, 4});
    auto source_desc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(-10, 10);
    auto scalar_desc = ScalarDesc(1.0f);

    auto ut = OP_API_UT(aclnnIndexAdd, INPUT(tensor_desc, dims, index_desc, source_desc, scalar_desc),
                        OUTPUT(tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// Out-of-range dim is rejected with 161002 (dim outside [-self.dims(), self.dims())).
TEST_F(l2_index_add_test, case_dim_out_of_range)
{
    auto tensor_desc = TensorDesc({5, 3}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(-10, 10);
    int64_t dims = 10;
    auto index_desc = TensorDesc({3}, ACL_INT32, ACL_FORMAT_NCHW).Value(vector<int32_t>{0, 2, 4});
    auto source_desc = TensorDesc({3, 3}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(-10, 10);
    auto scalar_desc = ScalarDesc(1.0f);

    auto ut = OP_API_UT(aclnnIndexAdd, INPUT(tensor_desc, dims, index_desc, source_desc, scalar_desc),
                        OUTPUT(tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// Guard for definedStride[0] > INT32_MAX on the deterministic index_put path: overflowing strides
// fall back to the AICPU IndexPut path instead of ConvertToTensor(DT_INT32) narrowing. The UT builds
// descriptors only (GetWorkspaceSize does not allocate real memory); NpuArchManager(DAV_3510) enables
// IsRegbase and aclrtSetSysParamOpt enables the deterministic switch (the symbol is resolved at runtime
// via dlsym — the UT link line has no libacl_rt, mirroring the runtime resolution of
// aclrtGetSysParamOpt in the source under test). The deterministic switch is restored to 0 afterwards
// (process-wide global, to avoid cross-case pollution).
TEST_F(l2_index_add_test, case_index_put_stride_overflow_fallback)
{
    // stride[0] = dim1 = 2^31 > INT32_MAX (2^31-1): hits the new guard condition
    auto tensor_desc = TensorDesc({2, 2147483648L}, ACL_FLOAT, ACL_FORMAT_NCHW);
    int64_t dims = 0;
    auto index_desc = TensorDesc({3}, ACL_INT32, ACL_FORMAT_NCHW).Value(vector<int32_t>{0, 1, 1});
    auto source_desc = TensorDesc({3, 2147483648L}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto scalar_desc = ScalarDesc(1.0f);

    using SetSysParamOptFn = aclError (*)(aclSysParamOpt, int64_t);
    auto setSysParamOpt = reinterpret_cast<SetSysParamOptFn>(dlsym(RTLD_DEFAULT, "aclrtSetSysParamOpt"));
    ASSERT_NE(setSysParamOpt, nullptr);

    NpuArchManager platform(NpuArch::DAV_3510);
    aclError setRet = setSysParamOpt(ACL_OPT_DETERMINISTIC, 1);

    auto ut = OP_API_UT(aclnnIndexAdd, INPUT(tensor_desc, dims, index_desc, source_desc, scalar_desc),
                        OUTPUT(tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    // Restore the process-wide deterministic switch before asserting, so neither pass nor fail leaves pollution behind
    (void)setSysParamOpt(ACL_OPT_DETERMINISTIC, 0);
    ASSERT_EQ(setRet, ACL_SUCCESS);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

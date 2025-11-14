/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <vector>
#include <array>
#include "gtest/gtest.h"

#include "level2/aclnn_unique.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"

using namespace std;

class l2_unique_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "unique_test SetUp" << endl; }

    static void TearDownTestCase() { cout << "unique_test TearDown" << endl; }

    vector<int64_t> GetViewDims(int num_of_dim)
    {
        vector<int64_t> view_dims= {1, 2, 2, 2};      // NCHW + NHWC + HWCN
        if (num_of_dim == 5)                          // NDHWC + NCDHW
            view_dims = {1, 4, 1, 1, 2};
        else if (num_of_dim == 0)                     // 空tensor
            view_dims = {0};
        else if (num_of_dim == 1)                     // 1维
            view_dims = {10};
        else if (num_of_dim == 8)                     // 8维
            view_dims = {2, 1, 2, 1, 2, 4, 5, 7};
        else if (num_of_dim == 10)                    // > 8维
            view_dims = {2, 1, 4, 4, 3, 1, 2, 4, 5, 7};
        else if (num_of_dim != 4)                     // ND
            view_dims = {1, 4, 1, 1, 2, 1};

        return view_dims;
    }

    // 测试合法数据类型
    void test_run(aclDataType test_dtype, aclFormat test_format, int num_of_dim,
                  bool sorted=false, bool return_inverse=true)
    {
        auto view_dims = GetViewDims(num_of_dim);
        const vector<int64_t>& view_dims_final= const_cast <vector<int64_t>&>(view_dims);

        auto self = TensorDesc(view_dims_final, test_dtype, test_format).ValueRange(-10, 10);
        auto output = TensorDesc(view_dims_final, test_dtype, test_format);
        auto indices = TensorDesc(view_dims_final, ACL_INT64, test_format);

        auto ut = OP_API_UT(aclnnUnique, INPUT(self, sorted, return_inverse), OUTPUT(output, indices));

        uint64_t workspaceSize = 0;
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
        EXPECT_EQ(aclRet, ACL_SUCCESS);
    }

    // 测试不合法数据类型
    void test_run_invalid(aclDataType test_dtype, aclFormat test_format, int num_of_dim,
                          bool sorted=false, bool return_inverse=false)
    {
        auto view_dims = GetViewDims(num_of_dim);
        const vector<int64_t>& view_dims_final= const_cast <vector<int64_t>&>(view_dims);

        auto self = TensorDesc(view_dims_final, test_dtype, test_format).ValueRange(-10, 10);
        auto output = TensorDesc(view_dims_final, test_dtype, test_format);
        auto indices = TensorDesc(view_dims_final, ACL_INT64, test_format);

        auto ut = OP_API_UT(aclnnUnique, INPUT(self, sorted, return_inverse), OUTPUT(output, indices));

        uint64_t workspaceSize = 0;
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
        EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    }

    // 测试数据类型是否一致
    void test_run_dtype_not_consistent(aclDataType self_dtype, aclDataType output_dtype,
                                       aclDataType indices_dtype, bool invalid,
                                       aclFormat test_format, int num_of_dim, bool sorted=false,
                                       bool return_inverse=false)
    {
        auto view_dims = GetViewDims(num_of_dim);
        const vector<int64_t>& view_dims_final = const_cast <vector<int64_t>&>(view_dims);

        auto self = TensorDesc(view_dims_final, self_dtype, test_format).ValueRange(-10, 10);
        auto output = TensorDesc(view_dims_final, output_dtype, test_format);
        auto indices = TensorDesc(view_dims_final, indices_dtype, test_format);

        auto ut = OP_API_UT(aclnnUnique, INPUT(self, sorted, return_inverse), OUTPUT(output, indices));

        uint64_t workspaceSize = 0;
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
        if (invalid) {
            EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
        } else {
            EXPECT_EQ(aclRet, ACL_SUCCESS);
        }
    }

    // 测试数据shape是否一致
    void test_run_shape_not_consistent(vector<int64_t> self_shape, vector<int64_t> output_shape,
                                       vector<int64_t> indices_shape,
                                       aclDataType test_dtype, bool sorted=false,
                                       bool return_inverse=true)
    {
        const vector<int64_t>& view_dims_self = const_cast <vector<int64_t>&>(self_shape);
        const vector<int64_t>& view_dims_output = const_cast <vector<int64_t>&>(output_shape);
        const vector<int64_t>& view_dims_indices = const_cast <vector<int64_t>&>(indices_shape);

        auto self = TensorDesc(view_dims_self, test_dtype, ACL_FORMAT_ND).ValueRange(-10, 10);
        auto output = TensorDesc(view_dims_output, test_dtype, ACL_FORMAT_ND);
        auto indices = TensorDesc(view_dims_indices, ACL_INT64, ACL_FORMAT_ND);

        auto ut = OP_API_UT(aclnnUnique, INPUT(self, sorted, return_inverse), OUTPUT(output, indices));

        uint64_t workspace_size = 0;
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
        EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    }

};

///////////////////////////////////////
/////      测试数据类型合法        /////
///////////////////////////////////////

// 测试不合法数据类型：COMPLEX64
TEST_F(l2_unique_test, case_014_COMPLEX64)
{
    test_run_invalid(ACL_COMPLEX64, ACL_FORMAT_ND, 6);
}

// 测试不合法数据类型：COMPLEX64
TEST_F(l2_unique_test, case_015_COMPLEX128)
{
    test_run_invalid(ACL_COMPLEX128, ACL_FORMAT_ND, 6);
}

// 测试不合法数据类型：UNIDEFINED
TEST_F(l2_unique_test, case_016_UNDEFINED)
{
    test_run_invalid(ACL_DT_UNDEFINED, ACL_FORMAT_ND, 6);
}

// 测试self和output输出不一致情况
TEST_F(l2_unique_test, case_017_DTYPE_UNCONSISTENT)
{
    test_run_dtype_not_consistent(ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, true,
                                  ACL_FORMAT_ND, 6, false, true);  // indices 非INT64
}

///////////////////////////////////////
/////          测试空指针          /////
///////////////////////////////////////

// 测试空指针
TEST_F(l2_unique_test, case_018_NULLPTR)
{
    const int num_of_dim = 4;
    bool sorted = false;
    bool return_inverse = true;

    auto view_dims = GetViewDims(num_of_dim);
    const vector<int64_t>& view_dims_final= const_cast <vector<int64_t>&>(view_dims);

    auto self = TensorDesc(view_dims_final, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto output = TensorDesc(view_dims_final, ACL_FLOAT16, ACL_FORMAT_ND);
    auto indices = TensorDesc(view_dims_final, ACL_INT64, ACL_FORMAT_ND);

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet;

    auto ut1 = OP_API_UT(aclnnUnique, INPUT(nullptr, sorted, return_inverse),
                                       OUTPUT(output, indices));
    aclRet = ut1.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

    auto ut2 = OP_API_UT(aclnnUnique, INPUT(self, sorted, return_inverse),
                                       OUTPUT(nullptr, indices));
    aclRet = ut2.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

    auto ut3 = OP_API_UT(aclnnUnique, INPUT(self, sorted, return_inverse),
                                       OUTPUT(output, nullptr));
    aclRet = ut3.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

///////////////////////////////////////
/////         支持空tensor         /////
///////////////////////////////////////

// 测试空tensor
TEST_F(l2_unique_test, case_019_EMPTY)
{
    test_run(ACL_INT8, ACL_FORMAT_ND, 0);
}

///////////////////////////////////////
/////        shape必须一致         /////
///////////////////////////////////////

// shape必须一致
TEST_F(l2_unique_test, case_020_SHAPE_UNCONSISTENT)
{
    test_run_shape_not_consistent({3, 4, 2}, {3, 4, 2}, {3, 2}, ACL_FLOAT16);  // self 与 indices shape不一致
}

// dim长度必须小于8
TEST_F(l2_unique_test, case_021_DIM_OVERSIZE)
{
    test_run_invalid(ACL_FLOAT, ACL_FORMAT_ND, 10);
}


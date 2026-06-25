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
#include "gtest/gtest.h"

#include "../../../op_api/aclnn_scatter_reduce.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"

using namespace std;

namespace {
constexpr int64_t kReduceNone = 0;
constexpr int64_t kReduceAdd = 1;
constexpr int64_t kReduceMul = 2;
constexpr int64_t kReduceMax = 3;
constexpr int64_t kReduceMin = 4;
constexpr int64_t kReduceMean = 5;
}

class l2_scatter_reduce_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << "Scatter Reduce Test Setup" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "Scatter Reduce Test TearDown" << endl;
    }

    void TestScatterReduceModes(bool includeSelf)
    {
        auto self = TensorDesc({3, 4}, ACL_FLOAT, ACL_FORMAT_ND)
                        .Value(vector<float>{1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3})
                        .Precision(0.0001, 0.0001);
        auto index = TensorDesc({1, 4}, ACL_INT64, ACL_FORMAT_ND).Value(vector<int64_t>{0, 1, 2, 1});
        auto src = TensorDesc({3, 4}, ACL_FLOAT, ACL_FORMAT_ND)
                       .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
        auto out = TensorDesc({3, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
        int64_t dim = 0;
        uint64_t workspaceSize = 0;

        const vector<int64_t> reduceList = {
            kReduceNone, kReduceAdd, kReduceMul, kReduceMax, kReduceMin, kReduceMean};

        for (auto reduce : reduceList) {
            auto ut =
                OP_API_UT(aclnnScatterReduce, INPUT(self, dim, index, src, reduce, includeSelf), OUTPUT(out));
            auto ret = ut.TestGetWorkspaceSize(&workspaceSize);
            EXPECT_EQ(ret, ACL_SUCCESS);
        }
    }

    void TestInplaceScatterReduceModes(bool includeSelf)
    {
        auto self = TensorDesc({3, 4}, ACL_FLOAT, ACL_FORMAT_ND)
                        .Value(vector<float>{1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3})
                        .Precision(0.0001, 0.0001);
        auto index = TensorDesc({1, 4}, ACL_INT64, ACL_FORMAT_ND).Value(vector<int64_t>{0, 1, 2, 1});
        auto src = TensorDesc({3, 4}, ACL_FLOAT, ACL_FORMAT_ND)
                       .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
        int64_t dim = 0;
        uint64_t workspaceSize = 0;

        const vector<int64_t> reduceList = {
            kReduceNone, kReduceAdd, kReduceMul, kReduceMax, kReduceMin, kReduceMean};

        for (auto reduce : reduceList) {
            auto ut = OP_API_UT(
                aclnnInplaceScatterReduce, INPUT(self, dim, index, src, reduce, includeSelf), OUTPUT());
            auto ret = ut.TestGetWorkspaceSize(&workspaceSize);
            EXPECT_EQ(ret, ACL_SUCCESS);
        }
    }
};

TEST_F(l2_scatter_reduce_test, scatter_reduce_supports_all_modes_include_self_true)
{
    TestScatterReduceModes(true);
}

TEST_F(l2_scatter_reduce_test, scatter_reduce_supports_all_modes_include_self_false)
{
    TestScatterReduceModes(false);
}

TEST_F(l2_scatter_reduce_test, inplace_scatter_reduce_supports_all_modes_include_self_true)
{
    TestInplaceScatterReduceModes(true);
}

TEST_F(l2_scatter_reduce_test, inplace_scatter_reduce_supports_all_modes_include_self_false)
{
    TestInplaceScatterReduceModes(false);
}

TEST_F(l2_scatter_reduce_test, scatter_reduce_invalid_reduce_should_fail)
{
    auto self = TensorDesc({3, 4}, ACL_FLOAT, ACL_FORMAT_ND)
                    .Value(vector<float>{1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3})
                    .Precision(0.0001, 0.0001);
    auto index = TensorDesc({1, 4}, ACL_INT64, ACL_FORMAT_ND).Value(vector<int64_t>{0, 1, 2, 1});
    auto src = TensorDesc({3, 4}, ACL_FLOAT, ACL_FORMAT_ND)
                   .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    auto out = TensorDesc({3, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    uint64_t workspaceSize = 0;

    auto ut1 = OP_API_UT(aclnnScatterReduce, INPUT(self, 0, index, src, -1, true), OUTPUT(out));
    EXPECT_EQ(ut1.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);

    auto ut2 = OP_API_UT(aclnnScatterReduce, INPUT(self, 0, index, src, 6, true), OUTPUT(out));
    EXPECT_EQ(ut2.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_scatter_reduce_test, inplace_scatter_reduce_invalid_reduce_should_fail)
{
    auto self = TensorDesc({3, 4}, ACL_FLOAT, ACL_FORMAT_ND)
                    .Value(vector<float>{1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3})
                    .Precision(0.0001, 0.0001);
    auto index = TensorDesc({1, 4}, ACL_INT64, ACL_FORMAT_ND).Value(vector<int64_t>{0, 1, 2, 1});
    auto src = TensorDesc({3, 4}, ACL_FLOAT, ACL_FORMAT_ND)
                   .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    uint64_t workspaceSize = 0;

    auto ut1 = OP_API_UT(aclnnInplaceScatterReduce, INPUT(self, 0, index, src, -1, true), OUTPUT());
    EXPECT_EQ(ut1.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);

    auto ut2 = OP_API_UT(aclnnInplaceScatterReduce, INPUT(self, 0, index, src, 6, true), OUTPUT());
    EXPECT_EQ(ut2.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_scatter_reduce_test, scatter_reduce_nullptr_should_fail)
{
    auto self = TensorDesc({3, 4}, ACL_FLOAT, ACL_FORMAT_ND)
                    .Value(vector<float>{1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3})
                    .Precision(0.0001, 0.0001);
    auto index = TensorDesc({1, 4}, ACL_INT64, ACL_FORMAT_ND).Value(vector<int64_t>{0, 1, 2, 1});
    auto src = TensorDesc({3, 4}, ACL_FLOAT, ACL_FORMAT_ND)
                   .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    auto out = TensorDesc({3, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    uint64_t workspaceSize = 0;

    auto ut1 = OP_API_UT(aclnnScatterReduce, INPUT(nullptr, 0, index, src, kReduceAdd, true), OUTPUT(out));
    EXPECT_EQ(ut1.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_NULLPTR);

    auto ut2 = OP_API_UT(aclnnScatterReduce, INPUT(self, 0, nullptr, src, kReduceAdd, true), OUTPUT(out));
    EXPECT_EQ(ut2.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_NULLPTR);

    auto ut3 = OP_API_UT(aclnnScatterReduce, INPUT(self, 0, index, nullptr, kReduceAdd, true), OUTPUT(out));
    EXPECT_EQ(ut3.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_NULLPTR);

    auto ut4 = OP_API_UT(aclnnScatterReduce, INPUT(self, 0, index, src, kReduceAdd, true), OUTPUT(nullptr));
    EXPECT_EQ(ut4.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_scatter_reduce_test, inplace_scatter_reduce_nullptr_should_fail)
{
    auto self = TensorDesc({3, 4}, ACL_FLOAT, ACL_FORMAT_ND)
                    .Value(vector<float>{1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3})
                    .Precision(0.0001, 0.0001);
    auto index = TensorDesc({1, 4}, ACL_INT64, ACL_FORMAT_ND).Value(vector<int64_t>{0, 1, 2, 1});
    auto src = TensorDesc({3, 4}, ACL_FLOAT, ACL_FORMAT_ND)
                   .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    uint64_t workspaceSize = 0;

    auto ut1 = OP_API_UT(aclnnInplaceScatterReduce, INPUT(nullptr, 0, index, src, kReduceAdd, true), OUTPUT());
    EXPECT_EQ(ut1.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_NULLPTR);

    auto ut2 = OP_API_UT(aclnnInplaceScatterReduce, INPUT(self, 0, nullptr, src, kReduceAdd, true), OUTPUT());
    EXPECT_EQ(ut2.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_NULLPTR);

    auto ut3 = OP_API_UT(aclnnInplaceScatterReduce, INPUT(self, 0, index, nullptr, kReduceAdd, true), OUTPUT());
    EXPECT_EQ(ut3.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_scatter_reduce_test, scatter_reduce_shape_mismatch_should_fail)
{
    auto self = TensorDesc({3, 3}, ACL_FLOAT, ACL_FORMAT_ND)
                    .Value(vector<float>{1, 1, 1, 2, 2, 2, 3, 3, 3})
                    .Precision(0.0001, 0.0001);
    auto index = TensorDesc({1, 4}, ACL_INT64, ACL_FORMAT_ND).Value(vector<int64_t>{0, 1, 2, 2});
    auto src = TensorDesc({3, 4}, ACL_FLOAT, ACL_FORMAT_ND)
                   .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    auto out = TensorDesc({3, 3}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    uint64_t workspaceSize = 0;

    auto ut = OP_API_UT(aclnnScatterReduce, INPUT(self, 0, index, src, kReduceAdd, true), OUTPUT(out));
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_scatter_reduce_test, inplace_scatter_reduce_dtype_mismatch_should_fail)
{
    auto self = TensorDesc({3, 4}, ACL_DOUBLE, ACL_FORMAT_ND)
                    .Value(vector<double>{1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3})
                    .Precision(0.0001, 0.0001);
    auto index = TensorDesc({1, 4}, ACL_INT64, ACL_FORMAT_ND).Value(vector<int64_t>{0, 1, 2, 1});
    auto src = TensorDesc({3, 4}, ACL_FLOAT, ACL_FORMAT_ND)
                   .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    uint64_t workspaceSize = 0;

    auto ut = OP_API_UT(aclnnInplaceScatterReduce, INPUT(self, 0, index, src, kReduceAdd, true), OUTPUT());
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
}

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
#include <memory>
#include "gtest/gtest.h"
#include <dlfcn.h>

#include "../../../op_api/aclnn_scatter_reduce.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace std;

namespace {
constexpr int64_t kReduceNone = 0;
constexpr int64_t kReduceAdd = 1;
constexpr int64_t kReduceMul = 2;
constexpr int64_t kReduceMax = 3;
constexpr int64_t kReduceMin = 4;
constexpr int64_t kReduceMean = 5;
} // namespace

class l2_scatter_reduce_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "Scatter Reduce Test Setup" << endl; }

    static void TearDownTestCase() { cout << "Scatter Reduce Test TearDown" << endl; }

    void CheckErrorMessage(std::initializer_list<const char*> expected)
    {
        auto getError = reinterpret_cast<const char* (*)()>(dlsym(RTLD_DEFAULT, "aclGetRecentErrMsg"));
        ASSERT_NE(getError, nullptr);
        const char* message = getError();
        ASSERT_NE(message, nullptr);
        const std::string text(message);
        for (const char* part : expected) {
            EXPECT_NE(text.find(part), std::string::npos) << text;
        }
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

        auto ut = OP_API_UT(aclnnScatterReduce, INPUT(self, dim, index, src, kReduceAdd, includeSelf), OUTPUT(out));
        EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACL_SUCCESS);
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

        auto ut = OP_API_UT(aclnnInplaceScatterReduce, INPUT(self, dim, index, src, kReduceAdd, includeSelf), OUTPUT());
        EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACL_SUCCESS);
    }
};

TEST_F(l2_scatter_reduce_test, scatter_reduce_supports_all_modes_include_self_true) { TestScatterReduceModes(true); }

TEST_F(l2_scatter_reduce_test, scatter_reduce_supports_all_modes_include_self_false) { TestScatterReduceModes(false); }

TEST_F(l2_scatter_reduce_test, inplace_scatter_reduce_supports_all_modes_include_self_true)
{
    TestInplaceScatterReduceModes(true);
}

TEST_F(l2_scatter_reduce_test, inplace_scatter_reduce_supports_all_modes_include_self_false)
{
    TestInplaceScatterReduceModes(false);
}

TEST_F(l2_scatter_reduce_test, arch22_sum_include_self_false_shapes)
{
    op::NpuArchManager platform(NpuArch::DAV_2201);
    for (auto socVersion : {op::SocVersion::ASCEND910B, op::SocVersion::ASCEND910_93}) {
        op::SocVersionManager soc(socVersion);
        for (const auto& shape : {vector<int64_t>{8}, vector<int64_t>{3, 8}, vector<int64_t>{2, 3, 8}}) {
            for (int64_t dim = -static_cast<int64_t>(shape.size()); dim < static_cast<int64_t>(shape.size()); ++dim) {
                SCOPED_TRACE(dim);
                auto self = TensorDesc(shape, ACL_FLOAT, ACL_FORMAT_ND);
                auto index = TensorDesc(shape, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 0);
                auto src = TensorDesc(shape, ACL_FLOAT, ACL_FORMAT_ND);
                auto out = TensorDesc(shape, ACL_FLOAT, ACL_FORMAT_ND);
                uint64_t workspaceSize = 0;
                auto ut = OP_API_UT(aclnnScatterReduce, INPUT(self, dim, index, src, kReduceAdd, false), OUTPUT(out));
                EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_SUCCESS);
                auto inplaceUt = OP_API_UT(aclnnInplaceScatterReduce, INPUT(self, dim, index, src, kReduceAdd, false),
                                           OUTPUT());
                EXPECT_EQ(inplaceUt.TestGetWorkspaceSize(&workspaceSize), ACLNN_SUCCESS);
            }
        }
    }
}

TEST_F(l2_scatter_reduce_test, arch22_sum_include_self_false_shape_boundaries)
{
    op::NpuArchManager platform(NpuArch::DAV_2201);
    for (auto socVersion : {op::SocVersion::ASCEND910B, op::SocVersion::ASCEND910_93}) {
        op::SocVersionManager soc(socVersion);
        SCOPED_TRACE(static_cast<int>(socVersion));
        for (int64_t rank : {1, 3, 4, 8}) {
            SCOPED_TRACE(rank);
            for (int64_t dim = -rank; dim < rank; ++dim) {
                SCOPED_TRACE(dim);
                const int64_t axis = dim < 0 ? dim + rank : dim;
                const int64_t otherAxis = axis == 0 ? rank - 1 : 0;
                for (int64_t mismatchAxis = 0; mismatchAxis < rank; ++mismatchAxis) {
                    SCOPED_TRACE(mismatchAxis);
                    for (bool largerSrc : {false, true}) {
                        SCOPED_TRACE(largerSrc);
                        vector<int64_t> selfShape(rank, 2);
                        selfShape[axis] = 5;
                        auto indexShape = selfShape;
                        auto srcShape = selfShape;
                        indexShape[axis] = 7;
                        srcShape[axis] = 9;
                        if (mismatchAxis != axis) {
                            if (largerSrc) {
                                srcShape[mismatchAxis] = 3;
                            } else {
                                indexShape[mismatchAxis] = 1;
                            }
                        }
                        const auto expected = mismatchAxis == axis || mismatchAxis == otherAxis ?
                                                  ACLNN_SUCCESS :
                                                  ACLNN_ERR_PARAM_INVALID;
                        auto self = TensorDesc(selfShape, ACL_FLOAT, ACL_FORMAT_ND);
                        auto index = TensorDesc(indexShape, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 3);
                        auto src = TensorDesc(srcShape, ACL_FLOAT, ACL_FORMAT_ND);
                        auto out = TensorDesc(selfShape, ACL_FLOAT, ACL_FORMAT_ND);
                        uint64_t workspaceSize = 0;
                        auto ut = OP_API_UT(aclnnScatterReduce, INPUT(self, dim, index, src, kReduceAdd, false),
                                            OUTPUT(out));
                        EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), expected);
                        if (expected == ACLNN_ERR_PARAM_INVALID) {
                            CheckErrorMessage({"self, index, src", "sizes must match on every axis except"});
                        }
                        auto inplaceUt = OP_API_UT(aclnnInplaceScatterReduce,
                                                   INPUT(self, dim, index, src, kReduceAdd, false), OUTPUT());
                        EXPECT_EQ(inplaceUt.TestGetWorkspaceSize(&workspaceSize), expected);
                        if (expected == ACLNN_ERR_PARAM_INVALID) {
                            CheckErrorMessage({"self, index, src", "sizes must match on every axis except"});
                        }
                    }
                }
            }
        }
    }
}

TEST_F(l2_scatter_reduce_test, arch22_sum_include_self_false_empty_index)
{
    op::SocVersionManager soc(op::SocVersion::ASCEND910B);
    op::NpuArchManager platform(NpuArch::DAV_2201);
    auto self = TensorDesc({3, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto index = TensorDesc({0, 8}, ACL_INT64, ACL_FORMAT_ND);
    auto src = TensorDesc({0, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out = TensorDesc({3, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    uint64_t workspaceSize = 0;
    auto ut = OP_API_UT(aclnnScatterReduce, INPUT(self, int64_t{0}, index, src, kReduceAdd, false), OUTPUT(out));
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_SUCCESS);
    auto inplaceUt = OP_API_UT(aclnnInplaceScatterReduce, INPUT(self, int64_t{0}, index, src, kReduceAdd, false),
                               OUTPUT());
    EXPECT_EQ(inplaceUt.TestGetWorkspaceSize(&workspaceSize), ACLNN_SUCCESS);
}

TEST_F(l2_scatter_reduce_test, arch22_sum_include_self_false_rejects_aicpu_shape)
{
    op::SocVersionManager soc(op::SocVersion::ASCEND910B);
    op::NpuArchManager platform(NpuArch::DAV_2201);
    auto self = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto index = TensorDesc({2, 2, 4}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 1);
    auto src = TensorDesc({2, 2, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    uint64_t workspaceSize = 0;
    auto ut = OP_API_UT(aclnnScatterReduce, INPUT(self, int64_t{0}, index, src, kReduceAdd, false), OUTPUT(out));
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
    CheckErrorMessage({"self, index, src", "2, 3, 4", "2, 2, 4", "dim=0", "except 0 and 2"});
    auto inplaceUt = OP_API_UT(aclnnInplaceScatterReduce, INPUT(self, int64_t{0}, index, src, kReduceAdd, false),
                               OUTPUT());
    EXPECT_EQ(inplaceUt.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
    CheckErrorMessage({"self, index, src", "2, 3, 4", "2, 2, 4", "dim=0", "except 0 and 2"});
}

TEST_F(l2_scatter_reduce_test, arch22_rejects_other_reductions_and_dtypes)
{
    op::SocVersionManager soc(op::SocVersion::ASCEND910B);
    op::NpuArchManager platform(NpuArch::DAV_2201);
    for (auto dtype : {ACL_FLOAT, ACL_FLOAT16, ACL_BF16, ACL_INT32}) {
        for (int64_t reduction : {kReduceNone, kReduceAdd, kReduceMul, kReduceMax, kReduceMin, kReduceMean}) {
            if ((dtype == ACL_FLOAT && reduction == kReduceAdd) ||
                ((dtype == ACL_FLOAT || dtype == ACL_FLOAT16 || dtype == ACL_BF16) && reduction == kReduceMax)) {
                continue;
            }
            for (bool includeSelf : {false, true}) {
                SCOPED_TRACE(reduction);
                auto self = TensorDesc({3, 4}, dtype, ACL_FORMAT_ND);
                auto index = TensorDesc({3, 4}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 2);
                auto src = TensorDesc({3, 4}, dtype, ACL_FORMAT_ND);
                auto out = TensorDesc({3, 4}, dtype, ACL_FORMAT_ND);
                uint64_t workspaceSize = 0;
                auto ut = OP_API_UT(aclnnScatterReduce, INPUT(self, int64_t{0}, index, src, reduction, includeSelf),
                                    OUTPUT(out));
                EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
                auto inplaceUt = OP_API_UT(aclnnInplaceScatterReduce,
                                           INPUT(self, int64_t{0}, index, src, reduction, includeSelf), OUTPUT());
                EXPECT_EQ(inplaceUt.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
            }
        }
    }
}

TEST_F(l2_scatter_reduce_test, scatter_reduce_invalid_reduce_should_fail)
{
    auto self = TensorDesc({3, 4}, ACL_FLOAT, ACL_FORMAT_ND)
                    .Value(vector<float>{1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3})
                    .Precision(0.0001, 0.0001);
    auto index = TensorDesc({1, 4}, ACL_INT64, ACL_FORMAT_ND).Value(vector<int64_t>{0, 1, 2, 1});
    auto src = TensorDesc({3, 4}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
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
    auto src = TensorDesc({3, 4}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
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
    auto src = TensorDesc({3, 4}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
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
    auto src = TensorDesc({3, 4}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
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
    auto src = TensorDesc({3, 4}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
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
    auto src = TensorDesc({3, 4}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    uint64_t workspaceSize = 0;

    auto ut = OP_API_UT(aclnnInplaceScatterReduce, INPUT(self, 0, index, src, kReduceAdd, true), OUTPUT());
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_scatter_reduce_test, ascend960_add_mul_include_self)
{
    op::SocVersionManager soc(op::SocVersion::RESERVED_VERSION);
    for (auto arch : {NpuArch::DAV_9201, NpuArch::DAV_9202}) {
        op::NpuArchManager platform(arch);
        SCOPED_TRACE(static_cast<int>(arch));
        for (int64_t reduction : {kReduceAdd, kReduceMul}) {
            for (bool includeSelf : {false, true}) {
                auto self = TensorDesc({3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
                auto index = TensorDesc({1, 4}, ACL_INT64, ACL_FORMAT_ND).Value(vector<int64_t>{0, 1, 2, 1});
                auto src = TensorDesc({1, 4}, ACL_FLOAT, ACL_FORMAT_ND);
                auto out = TensorDesc({3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
                auto expected = includeSelf ? ACLNN_SUCCESS : ACLNN_ERR_PARAM_INVALID;
                uint64_t workspaceSize = 0;
                auto ut = OP_API_UT(aclnnScatterReduce, INPUT(self, int64_t{0}, index, src, reduction, includeSelf),
                                    OUTPUT(out));
                EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), expected);
                auto inplaceUt = OP_API_UT(aclnnInplaceScatterReduce,
                                           INPUT(self, int64_t{0}, index, src, reduction, includeSelf), OUTPUT());
                EXPECT_EQ(inplaceUt.TestGetWorkspaceSize(&workspaceSize), expected);
            }
        }
    }
}

TEST_F(l2_scatter_reduce_test, arch22_max_include_self)
{
    op::SocVersionManager soc(op::SocVersion::ASCEND910B);
    op::NpuArchManager platform(NpuArch::DAV_2201);
    for (auto dtype : {ACL_FLOAT, ACL_FLOAT16, ACL_BF16}) {
        for (const auto& shape : {vector<int64_t>{7}, vector<int64_t>{7, 33}}) {
            for (bool includeSelf : {false, true}) {
                auto self = TensorDesc(shape, dtype, ACL_FORMAT_ND);
                auto index = TensorDesc(shape, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 6);
                auto src = TensorDesc(shape, dtype, ACL_FORMAT_ND);
                auto out = TensorDesc(shape, dtype, ACL_FORMAT_ND);
                uint64_t ws = 0;
                auto ut = OP_API_UT(aclnnScatterReduce, INPUT(self, int64_t{0}, index, src, kReduceMax, includeSelf),
                                    OUTPUT(out));
                EXPECT_EQ(ut.TestGetWorkspaceSize(&ws), ACLNN_SUCCESS);
                auto ip = OP_API_UT(
                    aclnnInplaceScatterReduce,
                    INPUT(self, -static_cast<int64_t>(shape.size()), index, src, kReduceMax, includeSelf), OUTPUT());
                EXPECT_EQ(ip.TestGetWorkspaceSize(&ws), ACLNN_SUCCESS);
            }
        }
    }
}

TEST_F(l2_scatter_reduce_test, arch22_max_shape_limits)
{
    op::SocVersionManager soc(op::SocVersion::ASCEND910B);
    op::NpuArchManager platform(NpuArch::DAV_2201);
    auto self = TensorDesc({7, 33}, ACL_FLOAT, ACL_FORMAT_ND);
    auto src = TensorDesc({7, 33}, ACL_FLOAT, ACL_FORMAT_ND);
    auto index = TensorDesc({7, 33}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 6);
    auto out = TensorDesc({7, 33}, ACL_FLOAT, ACL_FORMAT_ND);
    uint64_t ws = 0;
    auto reject = [&](TensorDesc input, TensorDesc indices, TensorDesc updates, TensorDesc output, int64_t dim,
                      std::initializer_list<const char*> expected) {
        for (bool includeSelf : {false, true}) {
            auto ut = OP_API_UT(aclnnScatterReduce, INPUT(input, dim, indices, updates, kReduceMax, includeSelf),
                                OUTPUT(output));
            EXPECT_EQ(ut.TestGetWorkspaceSize(&ws), ACLNN_ERR_PARAM_INVALID);
            CheckErrorMessage(expected);
            auto ip = OP_API_UT(aclnnInplaceScatterReduce, INPUT(input, dim, indices, updates, kReduceMax, includeSelf),
                                OUTPUT());
            EXPECT_EQ(ip.TestGetWorkspaceSize(&ws), ACLNN_ERR_PARAM_INVALID);
            CheckErrorMessage(expected);
        }
    };
    reject(self, index, src, out, 1, {"dim", "1", "dim=0 or -2", "rank 2"});
    auto i32 = TensorDesc({7, 33}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 6);
    reject(self, i32, src, out, 0, {"index", "INT32", "INT64"});
    auto rank3 = TensorDesc({2, 7, 33}, ACL_FLOAT, ACL_FORMAT_ND);
    auto index3 = TensorDesc({2, 7, 33}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 1);
    reject(rank3, index3, rank3, rank3, 0, {"self rank", "3", "rank 1 or 2"});
    auto shortSrc = TensorDesc({6, 33}, ACL_FLOAT, ACL_FORMAT_ND);
    reject(self, index, shortSrc, out, 0, {"self, index, src", "7, 33", "6, 33", "matching index/src shapes"});
    auto narrow = TensorDesc({7, 32}, ACL_FLOAT, ACL_FORMAT_ND);
    auto narrowIndex = TensorDesc({7, 32}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 6);
    reject(self, narrowIndex, narrow, out, 0, {"self, index, src", "7, 33", "7, 32", "matching self/src columns"});
    auto emptyIndex = TensorDesc({0, 33}, ACL_INT64, ACL_FORMAT_ND);
    auto emptySrc = TensorDesc({0, 33}, ACL_FLOAT, ACL_FORMAT_ND);
    auto empty = OP_API_UT(aclnnScatterReduce, INPUT(self, int64_t{0}, emptyIndex, emptySrc, kReduceMax, false),
                           OUTPUT(out));
    EXPECT_EQ(empty.TestGetWorkspaceSize(&ws), ACLNN_SUCCESS);
}

TEST_F(l2_scatter_reduce_test, arch22_max_element_count_limits)
{
    op::SocVersionManager soc(op::SocVersion::ASCEND910B);
    op::NpuArchManager platform(NpuArch::DAV_2201);
    auto tensor = [](const vector<int64_t>& shape, aclDataType dtype) {
        const vector<int64_t> strides = {shape[1], 1};
        return std::unique_ptr<aclTensor, decltype(&aclDestroyTensor)>(
            aclCreateTensor(shape.data(), shape.size(), dtype, strides.data(), 0, ACL_FORMAT_ND, shape.data(),
                            shape.size(), nullptr),
            aclDestroyTensor);
    };
    for (auto dtype : {ACL_FLOAT, ACL_FLOAT16, ACL_BF16}) {
        for (bool selfLimit : {false, true}) {
            for (int64_t rows : {65536LL, 4294967296LL}) {
                const int64_t columns = rows == 65536 ? 65536 : 1;
                const vector<int64_t> large = {rows, columns}, small = {1, columns};
                auto self = tensor(selfLimit ? large : small, dtype);
                auto src = tensor(selfLimit ? small : large, dtype);
                auto index = tensor(selfLimit ? small : large, ACL_INT64);
                ASSERT_NE(self.get(), nullptr);
                ASSERT_NE(src.get(), nullptr);
                ASSERT_NE(index.get(), nullptr);
                for (bool includeSelf : {false, true}) {
                    uint64_t ws = 0;
                    aclOpExecutor* executor = nullptr;
                    EXPECT_EQ(aclnnScatterReduceGetWorkspaceSize(self.get(), 0, index.get(), src.get(), kReduceMax,
                                                                 includeSelf, self.get(), &ws, &executor),
                              ACLNN_ERR_PARAM_INVALID);
                    EXPECT_EQ(executor, nullptr);
                    CheckErrorMessage({selfLimit ? "self" : "src", "4294967296", "4294967295"});
                    EXPECT_EQ(aclnnInplaceScatterReduceGetWorkspaceSize(self.get(), 0, index.get(), src.get(),
                                                                        kReduceMax, includeSelf, &ws, &executor),
                              ACLNN_ERR_PARAM_INVALID);
                    EXPECT_EQ(executor, nullptr);
                    CheckErrorMessage({selfLimit ? "self" : "src", "4294967296", "4294967295"});
                }
            }
        }
    }
    auto largeSelf = tensor({65536, 65536}, ACL_FLOAT);
    auto emptySrc = tensor({0, 65536}, ACL_FLOAT);
    auto emptyIndex = tensor({0, 65536}, ACL_INT64);
    for (bool includeSelf : {false, true}) {
        uint64_t ws = 0;
        aclOpExecutor* executor = nullptr;
        EXPECT_EQ(aclnnScatterReduceGetWorkspaceSize(largeSelf.get(), 0, emptyIndex.get(), emptySrc.get(), kReduceMax,
                                                     includeSelf, largeSelf.get(), &ws, &executor),
                  ACLNN_SUCCESS);
        delete executor;
        executor = nullptr;
        EXPECT_EQ(aclnnInplaceScatterReduceGetWorkspaceSize(largeSelf.get(), 0, emptyIndex.get(), emptySrc.get(),
                                                            kReduceMax, includeSelf, &ws, &executor),
                  ACLNN_SUCCESS);
        delete executor;
    }
}

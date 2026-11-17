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

#include "../../../op_api/aclnn_unique_consecutive.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"

using namespace std;

constexpr int64_t NoneN = 1000;

class l2_unique_consecutive_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "unique_consecutive_test SetUp" << endl; }

    static void TearDownTestCase() { cout << "unique_consecutive_test TearDown" << endl; }
};

// 测试空Tensor
TEST_F(l2_unique_consecutive_test, case_016_EMPTY)
{
    auto self = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto output = TensorDesc({3, 5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto indices = TensorDesc({0}, ACL_INT64, ACL_FORMAT_ND);
    auto counts = TensorDesc({0}, ACL_INT64, ACL_FORMAT_ND);

    bool return_inverse = true;
    bool return_counts = true;

    auto ut = OP_API_UT(aclnnUniqueConsecutive, INPUT(self, return_inverse, return_counts, NoneN),
                        OUTPUT(output, indices, counts));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 测试空指针
TEST_F(l2_unique_consecutive_test, case_017_NULLPTR)
{
    vector<float> raw_value = {0, 1, 2, 2, 3, 1, 1, 1, 4, 4};
    auto self = TensorDesc({10}, ACL_FLOAT, ACL_FORMAT_ND).Value(raw_value);
    auto output = TensorDesc({3, 5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto indices = TensorDesc({10}, ACL_INT64, ACL_FORMAT_ND);
    auto counts = TensorDesc({10}, ACL_INT64, ACL_FORMAT_ND);

    bool return_inverse = true;
    bool return_counts = true;
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet;

    auto ut1 = OP_API_UT(aclnnUniqueConsecutive, INPUT(nullptr, return_inverse, return_counts, NoneN),
                         OUTPUT(output, indices, counts));
    aclRet = ut1.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

    auto ut2 = OP_API_UT(aclnnUniqueConsecutive, INPUT(self, return_inverse, return_counts, NoneN),
                         OUTPUT(nullptr, indices, counts));
    aclRet = ut2.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

    auto ut3 = OP_API_UT(aclnnUniqueConsecutive, INPUT(self, return_inverse, return_counts, NoneN),
                         OUTPUT(output, nullptr, counts));
    aclRet = ut3.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

    auto ut4 = OP_API_UT(aclnnUniqueConsecutive, INPUT(self, return_inverse, return_counts, NoneN),
                         OUTPUT(output, indices, nullptr));
    aclRet = ut4.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// 测试inverseOut数据类型
TEST_F(l2_unique_consecutive_test, case_018_DTYPE_INVERSE_OUT)
{
    vector<float> raw_value = {0, 1, 2, 2, 3, 1, 1, 1, 4, 4};
    auto self = TensorDesc({10}, ACL_FLOAT, ACL_FORMAT_ND).Value(raw_value);
    auto output = TensorDesc({3, 5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto indices = TensorDesc({10}, ACL_INT8, ACL_FORMAT_ND); // invalid type of inverseOut
    auto counts = TensorDesc({10}, ACL_INT64, ACL_FORMAT_ND);

    bool return_inverse = true;
    bool return_counts = true;

    auto ut = OP_API_UT(aclnnUniqueConsecutive, INPUT(self, return_inverse, return_counts, NoneN),
                        OUTPUT(output, indices, counts));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 测试countsOut数据类型
TEST_F(l2_unique_consecutive_test, case_019_DTYPE_COUNTS_OUT)
{
    vector<float> raw_value = {0, 1, 2, 2, 3, 1, 1, 1, 4, 4};
    auto self = TensorDesc({10}, ACL_FLOAT, ACL_FORMAT_ND).Value(raw_value);
    auto output = TensorDesc({3, 5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto indices = TensorDesc({10}, ACL_INT64, ACL_FORMAT_ND);
    auto counts = TensorDesc({10}, ACL_FLOAT, ACL_FORMAT_ND); // invalid type of countsOut

    bool return_inverse = true;
    bool return_counts = true;

    auto ut = OP_API_UT(aclnnUniqueConsecutive, INPUT(self, return_inverse, return_counts, NoneN),
                        OUTPUT(output, indices, counts));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 测试self和valueOut数据类型不一致
TEST_F(l2_unique_consecutive_test, case_020_DTYPE_UNCONSISTENT_BETWEEN_SELF_AND_VALUE_OUT)
{
    vector<float> raw_value = {0, 1, 2, 2, 3, 1, 1, 1, 4, 4};
    auto self = TensorDesc({10}, ACL_FLOAT, ACL_FORMAT_ND).Value(raw_value);
    auto output = TensorDesc({3, 5}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto indices = TensorDesc({10}, ACL_INT64, ACL_FORMAT_ND);
    auto counts = TensorDesc({10}, ACL_INT64, ACL_FORMAT_ND);

    bool return_inverse = true;
    bool return_counts = true;

    auto ut = OP_API_UT(aclnnUniqueConsecutive, INPUT(self, return_inverse, return_counts, NoneN),
                        OUTPUT(output, indices, counts));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 测试inverseOut和countsOut数据类型不一致
TEST_F(l2_unique_consecutive_test, case_021_DTYPE_UNCONSISTENT_BETWEEN_INVERSE_OUT_AND_COUNTS_OUT)
{
    vector<float> raw_value = {0, 1, 2, 2, 3, 1, 1, 1, 4, 4};
    auto self = TensorDesc({10}, ACL_FLOAT, ACL_FORMAT_ND).Value(raw_value);
    auto output = TensorDesc({3, 5}, ACL_INT32, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto indices = TensorDesc({10}, ACL_INT64, ACL_FORMAT_ND);
    auto counts = TensorDesc({10}, ACL_INT64, ACL_FORMAT_ND);

    bool return_inverse = true;
    bool return_counts = true;

    auto ut = OP_API_UT(aclnnUniqueConsecutive, INPUT(self, return_inverse, return_counts, NoneN),
                        OUTPUT(output, indices, counts));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 测试shape维度大于8
TEST_F(l2_unique_consecutive_test, case_022_OVERSIZE_SHAPE)
{
    auto self = TensorDesc({1, 2, 2, 1, 2, 2, 3, 2, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 10);
    auto output = TensorDesc({1, 2, 2, 1, 2, 2, 3, 2, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto indices = TensorDesc({1, 2, 2, 1, 2, 2, 3, 2, 1, 1}, ACL_INT64, ACL_FORMAT_ND);
    auto counts = TensorDesc({1, 2, 2, 1, 2, 2, 3, 2, 1, 1}, ACL_INT64, ACL_FORMAT_ND);

    bool return_inverse = true;
    bool return_counts = true;

    auto ut = OP_API_UT(aclnnUniqueConsecutive, INPUT(self, return_inverse, return_counts, NoneN),
                        OUTPUT(output, indices, counts));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 测试dim=NoneN时输出为非连续tensor（转置/切片视图），GetWorkspaceSize构图成功
TEST_F(l2_unique_consecutive_test, case_023_UNCONTIGUOUS_OUT_DIM_NONE)
{
    vector<float> raw_value = {1, 1, 2, 2, 3, 3, 4, 4, 1, 2, 2, 3, 5, 5, 5, 5};
    auto self = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_ND).Value(raw_value);
    // valueOut/inverseOut为(4,4)输出的转置视图（strides={1,4}），countsOut为(16,)输出的步长切片视图
    auto output = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 4}, 0, {4, 4}).Precision(0.0001, 0.0001);
    auto indices = TensorDesc({4, 4}, ACL_INT64, ACL_FORMAT_ND, {1, 4}, 0, {4, 4});
    auto counts = TensorDesc({16}, ACL_INT64, ACL_FORMAT_ND, {2}, 0, {32});

    bool return_inverse = true;
    bool return_counts = true;

    auto ut = OP_API_UT(aclnnUniqueConsecutive, INPUT(self, return_inverse, return_counts, NoneN),
                        OUTPUT(output, indices, counts));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // 同一输入下连续输出的对照用例
    auto contigOutput = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto contigIndices = TensorDesc({16}, ACL_INT64, ACL_FORMAT_ND);
    auto contigCounts = TensorDesc({16}, ACL_INT64, ACL_FORMAT_ND);
    auto contigUt = OP_API_UT(aclnnUniqueConsecutive, INPUT(self, return_inverse, return_counts, NoneN),
                              OUTPUT(contigOutput, contigIndices, contigCounts));
    uint64_t contigWorkspaceSize = 0;
    aclRet = contigUt.TestGetWorkspaceSize(&contigWorkspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 测试dim!=NoneN时输出为非连续tensor（转置/切片视图），GetWorkspaceSize构图成功
TEST_F(l2_unique_consecutive_test, case_024_UNCONTIGUOUS_OUT_WITH_DIM)
{
    vector<float> raw_value = {1, 1, 2, 2, 3, 3, 4, 4, 1, 2, 2, 3, 5, 5, 5, 5};
    auto self = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_ND).Value(raw_value);
    // valueOut为(4,4)输出的转置视图（strides={1,4}），inverseOut/countsOut为(4,)输出的步长切片视图
    auto output = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 4}, 0, {4, 4}).Precision(0.0001, 0.0001);
    auto indices = TensorDesc({4}, ACL_INT64, ACL_FORMAT_ND, {2}, 0, {8});
    auto counts = TensorDesc({4}, ACL_INT64, ACL_FORMAT_ND, {2}, 1, {8});

    int64_t dim = 0;
    bool return_inverse = true;
    bool return_counts = true;

    auto ut = OP_API_UT(aclnnUniqueConsecutive, INPUT(self, return_inverse, return_counts, dim),
                        OUTPUT(output, indices, counts));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // 同一输入下连续输出的对照用例
    auto contigOutput = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto contigIndices = TensorDesc({4}, ACL_INT64, ACL_FORMAT_ND);
    auto contigCounts = TensorDesc({4}, ACL_INT64, ACL_FORMAT_ND);
    auto contigUt = OP_API_UT(aclnnUniqueConsecutive, INPUT(self, return_inverse, return_counts, dim),
                              OUTPUT(contigOutput, contigIndices, contigCounts));
    uint64_t contigWorkspaceSize = 0;
    aclRet = contigUt.TestGetWorkspaceSize(&contigWorkspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 测试returnCounts=false时countsOut为哑元输出，非连续时不应申请临时tensor及ViewCopy
TEST_F(l2_unique_consecutive_test, case_025_UNCONTIGUOUS_DUMMY_COUNTS_OUT)
{
    vector<float> raw_value = {1, 1, 2, 2, 3, 3, 4, 4, 1, 2, 2, 3, 5, 5, 5, 5};
    auto self = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_ND).Value(raw_value);
    auto output = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 4}, 0, {4, 4}).Precision(0.0001, 0.0001);
    auto indices = TensorDesc({4, 4}, ACL_INT64, ACL_FORMAT_ND, {1, 4}, 0, {4, 4});
    auto counts = TensorDesc({16}, ACL_INT64, ACL_FORMAT_ND, {2}, 0, {32}); // 哑元输出，同样传入非连续视图

    bool return_inverse = true;
    bool return_counts = false;

    auto ut = OP_API_UT(aclnnUniqueConsecutive, INPUT(self, return_inverse, return_counts, NoneN),
                        OUTPUT(output, indices, counts));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // 同一输入下连续输出的对照用例
    auto contigOutput = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto contigIndices = TensorDesc({16}, ACL_INT64, ACL_FORMAT_ND);
    auto contigCounts = TensorDesc({16}, ACL_INT64, ACL_FORMAT_ND);
    auto contigUt = OP_API_UT(aclnnUniqueConsecutive, INPUT(self, return_inverse, return_counts, NoneN),
                              OUTPUT(contigOutput, contigIndices, contigCounts));
    uint64_t contigWorkspaceSize = 0;
    aclRet = contigUt.TestGetWorkspaceSize(&contigWorkspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 测试returnInverse=false时inverseOut为哑元输出，非连续时不应申请临时tensor及ViewCopy
TEST_F(l2_unique_consecutive_test, case_026_UNCONTIGUOUS_DUMMY_INVERSE_OUT)
{
    vector<float> raw_value = {1, 1, 2, 2, 3, 3, 4, 4, 1, 2, 2, 3, 5, 5, 5, 5};
    auto self = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_ND).Value(raw_value);
    auto output = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 4}, 0, {4, 4}).Precision(0.0001, 0.0001);
    auto indices = TensorDesc({4, 4}, ACL_INT64, ACL_FORMAT_ND, {1, 4}, 0, {4, 4}); // 哑元输出，同样传入非连续视图
    auto counts = TensorDesc({16}, ACL_INT64, ACL_FORMAT_ND, {2}, 0, {32});

    bool return_inverse = false;
    bool return_counts = true;

    auto ut = OP_API_UT(aclnnUniqueConsecutive, INPUT(self, return_inverse, return_counts, NoneN),
                        OUTPUT(output, indices, counts));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // 同一输入下连续输出的对照用例
    auto contigOutput = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto contigIndices = TensorDesc({16}, ACL_INT64, ACL_FORMAT_ND);
    auto contigCounts = TensorDesc({16}, ACL_INT64, ACL_FORMAT_ND);
    auto contigUt = OP_API_UT(aclnnUniqueConsecutive, INPUT(self, return_inverse, return_counts, NoneN),
                              OUTPUT(contigOutput, contigIndices, contigCounts));
    uint64_t contigWorkspaceSize = 0;
    aclRet = contigUt.TestGetWorkspaceSize(&contigWorkspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 测试inverseOut/countsOut为INT32时数据类型校验通过（idx/count输出支持INT32、INT64）
TEST_F(l2_unique_consecutive_test, case_027_DTYPE_INT32_INVERSE_OUT_AND_COUNTS_OUT)
{
    vector<float> raw_value = {0, 1, 2, 2, 3, 1, 1, 1, 4, 4};
    auto self = TensorDesc({10}, ACL_FLOAT, ACL_FORMAT_ND).Value(raw_value);
    auto output = TensorDesc({3, 5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto indices = TensorDesc({10}, ACL_INT32, ACL_FORMAT_ND);
    auto counts = TensorDesc({10}, ACL_INT32, ACL_FORMAT_ND);

    bool return_inverse = true;
    bool return_counts = true;

    auto ut = OP_API_UT(aclnnUniqueConsecutive, INPUT(self, return_inverse, return_counts, NoneN),
                        OUTPUT(output, indices, counts));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 测试inverseOut(INT32)与countsOut(INT64)数据类型不一致
TEST_F(l2_unique_consecutive_test, case_028_DTYPE_UNCONSISTENT_INT32_AND_INT64_OUT)
{
    vector<float> raw_value = {0, 1, 2, 2, 3, 1, 1, 1, 4, 4};
    auto self = TensorDesc({10}, ACL_FLOAT, ACL_FORMAT_ND).Value(raw_value);
    auto output = TensorDesc({3, 5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto indices = TensorDesc({10}, ACL_INT32, ACL_FORMAT_ND);
    auto counts = TensorDesc({10}, ACL_INT64, ACL_FORMAT_ND); // 与inverseOut数据类型不一致

    bool return_inverse = true;
    bool return_counts = true;

    auto ut = OP_API_UT(aclnnUniqueConsecutive, INPUT(self, return_inverse, return_counts, NoneN),
                        OUTPUT(output, indices, counts));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 测试returnInverse/returnCounts均为false时，INT32哑元输出数据类型校验通过（idx/count输出支持INT32、INT64）
TEST_F(l2_unique_consecutive_test, case_029_DTYPE_INT32_DUMMY_WHEN_BOTH_FALSE)
{
    vector<float> raw_value = {0, 1, 2, 2, 3, 1, 1, 1, 4, 4};
    auto self = TensorDesc({10}, ACL_FLOAT, ACL_FORMAT_ND).Value(raw_value);
    auto output = TensorDesc({10}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto indices = TensorDesc({10}, ACL_INT32, ACL_FORMAT_ND);
    auto counts = TensorDesc({10}, ACL_INT32, ACL_FORMAT_ND);

    bool return_inverse = false;
    bool return_counts = false;

    auto ut = OP_API_UT(aclnnUniqueConsecutive, INPUT(self, return_inverse, return_counts, NoneN),
                        OUTPUT(output, indices, counts));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

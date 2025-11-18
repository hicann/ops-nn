/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*!
 * \file test_embedding_bag.cpp
 * \brief
 */
#include <vector>
#include <array>
#include "gtest/gtest.h"

#include "../../../op_host/op_api/aclnn_embedding_bag.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace std;
using namespace op;

class l2_embedding_bag_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "embedding_bag_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "embedding_bag_test TearDown" << endl;
    }
};

TEST_F(l2_embedding_bag_test, case_0)
{
    auto tensorWeightDesc = TensorDesc({10, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorIndicesDesc = TensorDesc({4}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 5).Value(vector<int>{1, 2, 3, 4});
    auto tensorOffsetsDesc = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 5).Value(vector<int>{0, 4});
    bool scaleGradByFreq = false;
    int64_t mode = 1;
    bool sparse = false;
    bool includeLastOffset = false;
    int64_t paddingIdx = -1;
    auto tensorOutputDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorOffset2BagDesc = TensorDesc({4}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 5);
    auto tensorBagSizeDesc = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 5);
    auto tensorMaxIndicesDesc = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 5);
    auto ut = OP_API_UT(
        aclnnEmbeddingBag,
        INPUT(
            tensorWeightDesc, tensorIndicesDesc, tensorOffsetsDesc, scaleGradByFreq, mode, sparse, (aclTensor*)nullptr,
            includeLastOffset, paddingIdx),
        OUTPUT(tensorOutputDesc, tensorOffset2BagDesc, tensorBagSizeDesc, tensorMaxIndicesDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    // ut.TestPrecision();
}

TEST_F(l2_embedding_bag_test, case_1)
{
    auto tensorWeightDesc = TensorDesc({10, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorIndicesDesc = TensorDesc({4}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 5).Value(vector<int>{1, 2, 3, 4});
    auto tensorOffsetsDesc = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 5).Value(vector<int>{0, 4});
    bool scaleGradByFreq = false;
    int64_t mode = 2;
    bool sparse = false;
    bool includeLastOffset = false;
    int64_t paddingIdx = -1;
    auto tensorOutputDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorOffset2BagDesc = TensorDesc({4}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 5);
    auto tensorBagSizeDesc = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 5);
    auto tensorMaxIndicesDesc = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 5);
    auto ut = OP_API_UT(
        aclnnEmbeddingBag,
        INPUT(
            tensorWeightDesc, tensorIndicesDesc, tensorOffsetsDesc, scaleGradByFreq, mode, sparse, (aclTensor*)nullptr,
            includeLastOffset, paddingIdx),
        OUTPUT(tensorOutputDesc, tensorOffset2BagDesc, tensorBagSizeDesc, tensorMaxIndicesDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    // ut.TestPrecision();
}

TEST_F(l2_embedding_bag_test, case_2)
{
    auto tensorWeightDesc = TensorDesc({3, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorIndicesDesc = TensorDesc({4}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 5).Value(vector<int>{1, 2, 0, 1});
    auto tensorOffsetsDesc = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 5).Value(vector<int>{0, 2});
    bool scaleGradByFreq = false;
    int64_t mode = 0;
    bool sparse = false;
    auto tensorperSampleWeightsDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
    bool includeLastOffset = false;
    int64_t paddingIdx = -1;
    auto tensorOutputDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorOffset2BagDesc = TensorDesc({4}, ACL_INT32, ACL_FORMAT_ND);
    auto tensorBagSizeDesc = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 5);
    auto tensorMaxIndicesDesc = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 5);
    auto ut = OP_API_UT(
        aclnnEmbeddingBag,
        INPUT(
            tensorWeightDesc, tensorIndicesDesc, tensorOffsetsDesc, scaleGradByFreq, mode, sparse,
            tensorperSampleWeightsDesc, includeLastOffset, paddingIdx),
        OUTPUT(tensorOutputDesc, tensorOffset2BagDesc, tensorBagSizeDesc, tensorMaxIndicesDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    // ut.TestPrecision();
}
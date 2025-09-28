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
#include <float.h>
#include "gtest/gtest.h"
#include "opdev/op_log.h"

#include "level2/aclnn_binary_cross_entropy.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"


class l2_binary_cross_entropy_test : public testing::Test {
 protected:
  static void SetUpTestCase() { std::cout << "binary_cross_entropy_test SetUp" << std::endl; }

  static void TearDownTestCase() { std::cout << "binary_cross_entropy_test TearDown" << std::endl; }
};

//空指针1
TEST_F(l2_binary_cross_entropy_test, case_nullptr_1) {
  auto target = TensorDesc({5, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto weight = TensorDesc({5, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto out = TensorDesc({5, 5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
  int64_t reduction = Reduction::None;

  auto ut = OP_API_UT(aclnnBinaryCrossEntropy,
                      INPUT(nullptr, target, weight, reduction),
                      OUTPUT(out));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_INNER_NULLPTR);
}

//空指针2
TEST_F(l2_binary_cross_entropy_test, case_nullptr_2) {
  auto self = TensorDesc({5, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto weight = TensorDesc({5, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto out = TensorDesc({5, 5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
  int64_t reduction = Reduction::None;

  auto ut = OP_API_UT(aclnnBinaryCrossEntropy,
                      INPUT(self, nullptr, weight, reduction),
                      OUTPUT(out));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_INNER_NULLPTR);
}

//空指针3
TEST_F(l2_binary_cross_entropy_test, case_nullptr_3) {
  auto self = TensorDesc({5, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto target = TensorDesc({5, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto weight = TensorDesc({5, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  int64_t reduction = Reduction::None;

  auto ut = OP_API_UT(aclnnBinaryCrossEntropy,
                      INPUT(self, target, weight, reduction),
                      OUTPUT(nullptr));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_INNER_NULLPTR);
}

//异常数据类型1
TEST_F(l2_binary_cross_entropy_test, case_double_1) {
  auto self = TensorDesc({5, 5}, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(0, 1);
  auto target = TensorDesc({5, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto weight = TensorDesc({5, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto out = TensorDesc({5, 5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
  int64_t reduction = Reduction::None;

  auto ut = OP_API_UT(aclnnBinaryCrossEntropy,
                      INPUT(self, target, weight, reduction),
                      OUTPUT(out));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

//异常数据类型2
TEST_F(l2_binary_cross_entropy_test, case_double_2) {
  auto self = TensorDesc({5, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto target = TensorDesc({5, 5}, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(0, 1);
  auto weight = TensorDesc({5, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto out = TensorDesc({5, 5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
  int64_t reduction = Reduction::None;

  auto ut = OP_API_UT(aclnnBinaryCrossEntropy,
                      INPUT(self, target, weight, reduction),
                      OUTPUT(out));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

//异常数据类型3
TEST_F(l2_binary_cross_entropy_test, case_double_3) {
  auto self = TensorDesc({5, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto target = TensorDesc({5, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto weight = TensorDesc({5, 5}, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(0, 1);
  auto out = TensorDesc({5, 5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
  int64_t reduction = Reduction::None;

  auto ut = OP_API_UT(aclnnBinaryCrossEntropy,
                      INPUT(self, target, weight, reduction),
                      OUTPUT(out));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

//异常数据类型4
TEST_F(l2_binary_cross_entropy_test, case_double_4) {
  auto self = TensorDesc({5, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto target = TensorDesc({5, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto weight = TensorDesc({5, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto out = TensorDesc({5, 5}, ACL_DOUBLE, ACL_FORMAT_ND).Precision(0.001, 0.001);
  int64_t reduction = Reduction::None;

  auto ut = OP_API_UT(aclnnBinaryCrossEntropy,
                      INPUT(self, target, weight, reduction),
                      OUTPUT(out));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

//异常输入数据1
TEST_F(l2_binary_cross_entropy_test, case_reduction_3) {
  auto self = TensorDesc({5, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto target = TensorDesc({5, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto weight = TensorDesc({5, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto out = TensorDesc({5, 5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
  int64_t reduction = 3;

  auto ut = OP_API_UT(aclnnBinaryCrossEntropy,
                      INPUT(self, target, weight, reduction),
                      OUTPUT(out));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

//异常输入数据2
TEST_F(l2_binary_cross_entropy_test, case_reduction_1) {
  auto self = TensorDesc({5, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto target = TensorDesc({5, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto weight = TensorDesc({5, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto out = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
  int64_t reduction = 1;

  auto ut = OP_API_UT(aclnnBinaryCrossEntropy,
                      INPUT(self, target, weight, reduction),
                      OUTPUT(out));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

//输入shape不一致
TEST_F(l2_binary_cross_entropy_test, case_shape_abnormal) {
  auto self = TensorDesc({5, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto target = TensorDesc({5, 7}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto weight = TensorDesc({5, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto out = TensorDesc({5, 5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
  int64_t reduction = Reduction::None;

  auto ut = OP_API_UT(aclnnBinaryCrossEntropy,
                      INPUT(self, target, weight, reduction),
                      OUTPUT(out));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

//数据覆盖 float Reduction::None
TEST_F(l2_binary_cross_entropy_test, case_float_reduction_none) {
  auto self = TensorDesc({5, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto target = TensorDesc({5, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto weight = TensorDesc({5, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto out = TensorDesc({5, 5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
  int64_t reduction = Reduction::None;

  auto ut = OP_API_UT(aclnnBinaryCrossEntropy,
                      INPUT(self, target, weight, reduction),
                      OUTPUT(out));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  //ut.TestPrecision();
}

//数据覆盖 float Reduction::None   weight为nullptr也支持
TEST_F(l2_binary_cross_entropy_test, case_float_reduction_none_weight_nullptr) {
  auto self = TensorDesc({5, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto target = TensorDesc({5, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto weight = nullptr;
  auto out = TensorDesc({5, 5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
  int64_t reduction = Reduction::None;

  auto ut = OP_API_UT(aclnnBinaryCrossEntropy, INPUT(self, target, weight, reduction), OUTPUT(out));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  //ut.TestPrecision();
}


//数据覆盖 float Reduction::Sum
TEST_F(l2_binary_cross_entropy_test, case_float_reduction_sum) {
  auto self = TensorDesc({5, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto target = TensorDesc({5, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto weight = TensorDesc({5, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto out = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
  int64_t reduction = Reduction::Sum;

  auto ut = OP_API_UT(aclnnBinaryCrossEntropy,
                      INPUT(self, target, weight, reduction),
                      OUTPUT(out));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  //ut.TestPrecision();
}

//数据覆盖 float Reduction::Sum
TEST_F(l2_binary_cross_entropy_test, case_float_reduction_sum_weight_nullptr) {
  auto self = TensorDesc({5, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto target = TensorDesc({5, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto weight = nullptr;
  auto out = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
  int64_t reduction = Reduction::Sum;

  auto ut = OP_API_UT(aclnnBinaryCrossEntropy,
                      INPUT(self, target, weight, reduction),
                      OUTPUT(out));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  //ut.TestPrecision();
}

//数据覆盖 float Reduction::Mean
TEST_F(l2_binary_cross_entropy_test, case_float_reduction_mean) {
  auto self = TensorDesc({5, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto target = TensorDesc({5, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto weight = TensorDesc({5, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto out = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
  int64_t reduction = Reduction::Mean;

  auto ut = OP_API_UT(aclnnBinaryCrossEntropy,
                      INPUT(self, target, weight, reduction),
                      OUTPUT(out));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  //ut.TestPrecision();
}

//数据覆盖 float Reduction::Mean
TEST_F(l2_binary_cross_entropy_test, case_float_reduction_mean_weight_nullptr) {
  auto self = TensorDesc({5, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto target = TensorDesc({5, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto weight = nullptr;
  auto out = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
  int64_t reduction = Reduction::Mean;

  auto ut = OP_API_UT(aclnnBinaryCrossEntropy,
                      INPUT(self, target, weight, reduction),
                      OUTPUT(out));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  //ut.TestPrecision();
}

//数据覆盖 float16 Reduction::Mean
TEST_F(l2_binary_cross_entropy_test, case_float16_reduction_none) {
  auto self = TensorDesc({5, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 1);
  auto target = TensorDesc({5, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 1);
  auto weight = TensorDesc({5, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 1);
  auto out = TensorDesc({5, 5}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);
  int64_t reduction = Reduction::None;

  auto ut = OP_API_UT(aclnnBinaryCrossEntropy,
                      INPUT(self, target, weight, reduction),
                      OUTPUT(out));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  //ut.TestPrecision();
}

//数据覆盖 float16 Reduction::Sum
TEST_F(l2_binary_cross_entropy_test, case_float16_reduction_sum) {
  auto self = TensorDesc({5, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 1);
  auto target = TensorDesc({5, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 1);
  auto weight = TensorDesc({5, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 1);
  auto out = TensorDesc({1}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);
  int reduction = Reduction::Sum;

  auto ut = OP_API_UT(aclnnBinaryCrossEntropy,
                      INPUT(self, target, weight, reduction),
                      OUTPUT(out));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  //ut.TestPrecision();
}

//数据覆盖 float16 Reduction::Mean
TEST_F(l2_binary_cross_entropy_test, case_float16_reduction_mean) {
  auto self = TensorDesc({5, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 1);
  auto target = TensorDesc({5, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 1);
  auto weight = TensorDesc({5, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 1);
  auto out = TensorDesc({1}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);
  int reduction = Reduction::Mean;

  auto ut = OP_API_UT(aclnnBinaryCrossEntropy,
                      INPUT(self, target, weight, reduction),
                      OUTPUT(out));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  //ut.TestPrecision();
}

//空tensor float Reduction::None
TEST_F(l2_binary_cross_entropy_test, case_fp_empty_tensor_none) {
  auto self = TensorDesc({0, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto target = TensorDesc({0, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto weight = TensorDesc({0, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto out = TensorDesc({0, 5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
  int reduction = Reduction::None;

  auto ut = OP_API_UT(aclnnBinaryCrossEntropy,
                      INPUT(self, target, weight, reduction),
                      OUTPUT(out));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  //ut.TestPrecision();
}

//空tensor float16 Reduction::None
TEST_F(l2_binary_cross_entropy_test, case_fp16_empty_tensor_none) {
  auto self = TensorDesc({0, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 1);
  auto target = TensorDesc({0, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 1);
  auto weight = TensorDesc({0, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 1);
  auto out = TensorDesc({0, 5}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);
  int reduction = Reduction::None;

  auto ut = OP_API_UT(aclnnBinaryCrossEntropy,
                      INPUT(self, target, weight, reduction),
                      OUTPUT(out));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  //ut.TestPrecision();
}

//非连续
TEST_F(l2_binary_cross_entropy_test, case_not_contiguous) {
  auto self = TensorDesc({5, 10}, ACL_FLOAT16, ACL_FORMAT_ND, {1, 5}, 0, {10, 5}).ValueRange(0, 1);
  auto target = TensorDesc({5, 10}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 1);
  auto weight = TensorDesc({5, 10}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 1);
  auto out = TensorDesc({5, 10}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);
  int reduction = Reduction::None;

  auto ut = OP_API_UT(aclnnBinaryCrossEntropy,
                      INPUT(self, target, weight, reduction),
                      OUTPUT(out));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  //ut.TestPrecision();
}

//输入输出dtype不一致
TEST_F(l2_binary_cross_entropy_test, case_dtype_inconsistent) {
  auto self = TensorDesc({5, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 1);
  auto target = TensorDesc({5, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 1);
  auto weight = TensorDesc({5, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 1);
  auto out = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
  int reduction = Reduction::Mean;

  auto ut = OP_API_UT(aclnnBinaryCrossEntropy,
                      INPUT(self, target, weight, reduction),
                      OUTPUT(out));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

//输入输出format不一致
TEST_F(l2_binary_cross_entropy_test, case_format_inconsistent) {
  auto self = TensorDesc({5, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 1);
  auto target = TensorDesc({5, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 1);
  auto weight = TensorDesc({5, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 1);
  auto out = TensorDesc({1}, ACL_FLOAT16, ACL_FORMAT_NCHW).Precision(0.001, 0.001);
  int reduction = Reduction::Mean;

  auto ut = OP_API_UT(aclnnBinaryCrossEntropy,
                      INPUT(self, target, weight, reduction),
                      OUTPUT(out));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  //ut.TestPrecision();
}

// format测试
TEST_F(l2_binary_cross_entropy_test, case_format)
{
    vector<aclFormat> ValidList = {
    ACL_FORMAT_UNDEFINED,
    ACL_FORMAT_NCHW,
    ACL_FORMAT_NHWC,
    ACL_FORMAT_ND,
    ACL_FORMAT_NC1HWC0,
    ACL_FORMAT_FRACTAL_Z,
    ACL_FORMAT_NC1HWC0_C04,
    ACL_FORMAT_HWCN,
    ACL_FORMAT_NDHWC,
    ACL_FORMAT_FRACTAL_NZ,
    ACL_FORMAT_NCDHW,
    ACL_FORMAT_NDC1HWC0,
    ACL_FRACTAL_Z_3D};

    int length = ValidList.size();
    for (int i = 0; i < length; i++) {
        auto self = TensorDesc({5, 5}, ACL_FLOAT16, ValidList[i]).ValueRange(0, 1);
        auto target = TensorDesc({5, 5}, ACL_FLOAT16, ValidList[i]).ValueRange(0, 1);
        auto weight = TensorDesc({5, 5}, ACL_FLOAT16, ValidList[i]).ValueRange(0, 1);
        auto out = TensorDesc({5, 5}, ACL_FLOAT16, ValidList[i]).Precision(0.001, 0.001);
        int reduction = Reduction::None;

        auto ut = OP_API_UT(aclnnBinaryCrossEntropy,
                            INPUT(self, target, weight, reduction),
                            OUTPUT(out));

        uint64_t workspace_size = 0;
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
        EXPECT_EQ(aclRet, ACL_SUCCESS);

        //ut.TestPrecision();
    }

    for (int i = 0; i < length; i++) {
        auto self = TensorDesc({5, 5}, ACL_FLOAT16, ValidList[i]).ValueRange(0, 1);
        auto target = TensorDesc({5, 5}, ACL_FLOAT16, ValidList[i]).ValueRange(0, 1);
        auto weight = TensorDesc({5, 5}, ACL_FLOAT16, ValidList[i]).ValueRange(0, 1);
        auto out = TensorDesc({1}, ACL_FLOAT16, ValidList[i]).Precision(0.001, 0.001);
        int reduction = Reduction::Mean;

        auto ut = OP_API_UT(aclnnBinaryCrossEntropy,
                            INPUT(self, target, weight, reduction),
                            OUTPUT(out));

        uint64_t workspace_size = 0;
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
        EXPECT_EQ(aclRet, ACL_SUCCESS);

        //ut.TestPrecision();
    }

    for (int i = 0; i < length; i++) {
        auto self = TensorDesc({5, 5}, ACL_FLOAT16, ValidList[i]).ValueRange(0, 1);
        auto target = TensorDesc({5, 5}, ACL_FLOAT16, ValidList[i]).ValueRange(0, 1);
        auto weight = TensorDesc({5, 5}, ACL_FLOAT16, ValidList[i]).ValueRange(0, 1);
        auto out = TensorDesc({1}, ACL_FLOAT16, ValidList[i]).Precision(0.001, 0.001);
        int reduction = Reduction::Sum;

        auto ut = OP_API_UT(aclnnBinaryCrossEntropy,
                            INPUT(self, target, weight, reduction),
                            OUTPUT(out));

        uint64_t workspace_size = 0;
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
        EXPECT_EQ(aclRet, ACL_SUCCESS);

        //ut.TestPrecision();
    }
}
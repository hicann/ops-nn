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
#include <unistd.h>

#include "gtest/gtest.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"
#include "../../../op_api/aclnn_smooth_l1_loss_backward.h"


using namespace op;
using namespace std;

enum Reduction {
    NONE,
    MEAN,
    SUM,
    END
};

class l2_smooth_l1_loss_backward_test : public testing::Test {
protected:
    static void SetUpTestCase() 
    {
        std::cout << "smooth_l1_loss_backward_test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "smooth_l1_loss_backward_test TearDown" << std::endl;
    }
};

TEST_F(l2_smooth_l1_loss_backward_test, case_normal)
{
    auto grad_output_desc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND)
                            .Value(vector<float>{0.1});
    auto self_desc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND)
                     .Value(vector<float>{0.32});
    auto target_desc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND)
                       .Value(vector<float>{1});
    int64_t reduction = NONE;
    float beta = 1;
    auto grad_input_desc = TensorDesc(grad_output_desc).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnSmoothL1LossBackward,
                        INPUT(grad_output_desc, self_desc, target_desc, reduction, beta),
                        OUTPUT(grad_input_desc));
    // SAMPLE: only test GetworkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// check NULLPTR
TEST_F(l2_smooth_l1_loss_backward_test, case_nullptr)
{
    auto grad_output_desc = TensorDesc({1, 0, 4, 1}, ACL_FLOAT, ACL_FORMAT_ND)
                            .ValueRange(-2, 2);
    auto self_desc = TensorDesc({1, 0, 4, 1}, ACL_FLOAT, ACL_FORMAT_ND)
                     .ValueRange(-2, 2);
    auto target_desc = TensorDesc({1, 0, 4, 1}, ACL_FLOAT, ACL_FORMAT_ND)
                       .ValueRange(-2, 2);
    int64_t reduction = NONE;
    float beta = 1;
    auto grad_input_desc = TensorDesc(grad_output_desc).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnSmoothL1LossBackward,
                        INPUT(nullptr, self_desc, target_desc, reduction, beta),
                        OUTPUT(grad_input_desc));
    // SAMPLE: only test GetworkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

    auto ut2 = OP_API_UT(aclnnSmoothL1LossBackward,
                         INPUT(grad_output_desc, nullptr, target_desc, reduction, beta),
                         OUTPUT(grad_input_desc));
    // SAMPLE: only test GetworkspaceSize
    workspaceSize = 0;
    aclRet = ut2.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

    auto ut3 = OP_API_UT(aclnnSmoothL1LossBackward,
                         INPUT(grad_output_desc, self_desc, nullptr, reduction, beta),
                         OUTPUT(grad_input_desc));
    // SAMPLE: only test GetworkspaceSize
    workspaceSize = 0;
    aclRet = ut3.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

    auto ut4 = OP_API_UT(aclnnSmoothL1LossBackward,
                         INPUT(grad_output_desc, self_desc, target_desc, reduction, beta),
                         OUTPUT(nullptr));
    // SAMPLE: only test GetworkspaceSize
    workspaceSize = 0;
    aclRet = ut4.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// empty tensor
TEST_F(l2_smooth_l1_loss_backward_test, case_empty_tensor)
{
    auto grad_output_desc = TensorDesc({1, 0, 4, 1}, ACL_FLOAT, ACL_FORMAT_ND)
                            .ValueRange(-2, 2);
    auto self_desc = TensorDesc({1, 0, 4, 1}, ACL_FLOAT, ACL_FORMAT_ND)
                     .ValueRange(-2, 2);
    auto target_desc = TensorDesc({1, 0, 4, 1}, ACL_FLOAT, ACL_FORMAT_ND)
                       .ValueRange(-2, 2);
    int64_t reduction = NONE;
    float beta = 1;
    auto grad_input_desc = TensorDesc(grad_output_desc).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnSmoothL1LossBackward,
                        INPUT(grad_output_desc, self_desc, target_desc, reduction, beta),
                        OUTPUT(grad_input_desc));
    // SAMPLE: only test GetworkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);

    // // SAMPLE: precision simulate
    // ut.TestPrecision();
}

// Checkdtype FLOAT
TEST_F(l2_smooth_l1_loss_backward_test, case_dtype_float)
{
    auto grad_output_desc = TensorDesc({1, 4, 4, 1}, ACL_FLOAT, ACL_FORMAT_ND)
                            .ValueRange(-2, 2);
    auto self_desc = TensorDesc({1, 4, 4, 1}, ACL_FLOAT, ACL_FORMAT_ND)
                     .ValueRange(-2, 2);
    auto target_desc = TensorDesc({1, 4, 4, 1}, ACL_FLOAT, ACL_FORMAT_ND)
                       .ValueRange(-2, 2);
    int64_t reduction = NONE;
    float beta = 1;
    auto grad_input_desc = TensorDesc(grad_output_desc).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnSmoothL1LossBackward,
                        INPUT(grad_output_desc, self_desc, target_desc, reduction, beta),
                        OUTPUT(grad_input_desc));
    // SAMPLE: only test GetworkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);

    // // SAMPLE: precision simulate
    // ut.TestPrecision();
}

// Checkdtype FLOAT16
TEST_F(l2_smooth_l1_loss_backward_test, case_dtype_float16)
{
    auto grad_output_desc = TensorDesc({1, 4, 4, 1}, ACL_FLOAT16, ACL_FORMAT_ND)
                            .ValueRange(-2, 2);
    auto self_desc = TensorDesc({1, 4, 4, 1}, ACL_FLOAT16, ACL_FORMAT_ND)
                     .ValueRange(-2, 2);
    auto target_desc = TensorDesc({1, 4, 4, 1}, ACL_FLOAT16, ACL_FORMAT_ND)
                       .ValueRange(-2, 2);
    int64_t reduction = NONE;
    float beta = 1;
    auto grad_input_desc = TensorDesc(grad_output_desc).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnSmoothL1LossBackward,
                        INPUT(grad_output_desc, self_desc, target_desc, reduction, beta),
                        OUTPUT(grad_input_desc));
    // SAMPLE: only test GetworkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// Checkdtype promote Dtype
TEST_F(l2_smooth_l1_loss_backward_test, case_dtype_promote)
{
    auto grad_output_desc = TensorDesc({1, 4, 4, 1}, ACL_FLOAT16, ACL_FORMAT_ND)
                            .ValueRange(-2, 2);
    auto self_desc = TensorDesc({1, 4, 4, 1}, ACL_FLOAT, ACL_FORMAT_ND)
                     .ValueRange(-2, 2);
    auto target_desc = TensorDesc({1, 4, 4, 1}, ACL_FLOAT16, ACL_FORMAT_ND)
                       .ValueRange(-2, 2);
    int64_t reduction = NONE;
    float beta = 1;
    auto grad_input_desc = TensorDesc(grad_output_desc).Precision(0.001, 0.001);

    auto ut = OP_API_UT(aclnnSmoothL1LossBackward,
                        INPUT(grad_output_desc, self_desc, target_desc, reduction, beta),
                        OUTPUT(grad_input_desc));
    // SAMPLE: only test GetworkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);

    // // SAMPLE: precision simulate
    // ut.TestPrecision();
}

// checkDtype failed
TEST_F(l2_smooth_l1_loss_backward_test, case_dtype_invalid)
{
    vector<aclDataType> DtypeList = {
        ACL_FLOAT,
        ACL_FLOAT,
        ACL_FLOAT,
        ACL_DOUBLE};
    for (int i = 0; i < 4; i++) {
        auto grad_output_desc = TensorDesc({1, 4, 4, 1}, DtypeList[i % 4], ACL_FORMAT_ND)
                                .ValueRange(-2, 2);
        auto self_desc = TensorDesc({1, 4, 4, 1}, DtypeList[(i + 1) % 4], ACL_FORMAT_ND)
                         .ValueRange(-2, 2);
        auto target_desc = TensorDesc({1, 4, 4, 1}, DtypeList[(i + 2) % 4], ACL_FORMAT_ND)
                          .ValueRange(-2, 2);
        int64_t reduction = NONE;
        float beta = 1;
        auto grad_input_desc = TensorDesc({1, 4, 4, 1}, DtypeList[(i + 3) % 4], ACL_FORMAT_ND)
                              .Precision(0.0001, 0.0001);

        auto ut = OP_API_UT(aclnnSmoothL1LossBackward,
                            INPUT(grad_output_desc, self_desc, target_desc, reduction, beta),
                            OUTPUT(grad_input_desc));
        // SAMPLE: only test GetworkspaceSize
        uint64_t workspaceSize = 0;
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
        EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    }
}

// checkDim > 8
TEST_F(l2_smooth_l1_loss_backward_test, case_dim)
{
  auto grad_output_desc = TensorDesc({1,1,4,1,1,1,1,1,1}, ACL_FLOAT, ACL_FORMAT_ND)
                          .ValueRange(-2, 2);
  auto self_desc = TensorDesc({1,1,4,1,1,1,1,1,1}, ACL_FLOAT, ACL_FORMAT_ND)
                   .ValueRange(-2, 2);
  auto target_desc = TensorDesc({1,1,4,1,1,1,1,1,1}, ACL_FLOAT, ACL_FORMAT_ND)
                     .ValueRange(-2, 2);
  int64_t reduction = NONE;
  float beta = 1;
  auto grad_input_desc = TensorDesc(grad_output_desc).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnSmoothL1LossBackward,
                      INPUT(grad_output_desc, self_desc, target_desc, reduction, beta),
                      OUTPUT(grad_input_desc));
  // SAMPLE: only test GetworkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// check reduction
TEST_F(l2_smooth_l1_loss_backward_test, case_reduction)
{
  for (int i = 0; i <= 3; i++) {
    auto grad_output_desc = TensorDesc({1,1,4,1}, ACL_FLOAT, ACL_FORMAT_ND)
                            .ValueRange(-2, 2);
    auto self_desc = TensorDesc({1,1,4,1}, ACL_FLOAT, ACL_FORMAT_ND)
                    .ValueRange(-2, 2);
    auto target_desc = TensorDesc({1,1,4,1}, ACL_FLOAT, ACL_FORMAT_ND)
                      .ValueRange(-2, 2);
    int64_t reduction = (enum Reduction) i;
    float beta = 1;
    auto grad_input_desc = TensorDesc(grad_output_desc).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnSmoothL1LossBackward,
                        INPUT(grad_output_desc, self_desc, target_desc, reduction, beta),
                        OUTPUT(grad_input_desc));
    // SAMPLE: only test GetworkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    if (i != 3) {
      EXPECT_EQ(aclRet, ACLNN_SUCCESS);
      // // SAMPLE: precision simulate
      // ut.TestPrecision();
    }
    else {
      EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    }
  }
}

// check broadcast succ 1 empty
TEST_F(l2_smooth_l1_loss_backward_test, case_broadcast_succ_1)
{
  auto grad_output_desc = TensorDesc({1, 0, 4, 1}, ACL_FLOAT, ACL_FORMAT_ND);
  auto self_desc = TensorDesc({4,1}, ACL_FLOAT, ACL_FORMAT_ND);
  auto target_desc = TensorDesc({1,4,2}, ACL_FLOAT, ACL_FORMAT_ND);
  int64_t reduction = NONE;
  float beta = 1;
  auto grad_input_desc = TensorDesc({1, 0, 4, 2}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnSmoothL1LossBackward,
                      INPUT(grad_output_desc, self_desc, target_desc, reduction, beta),
                      OUTPUT(grad_input_desc));
  // SAMPLE: only test GetworkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// check broadcast succ 2 not empty
TEST_F(l2_smooth_l1_loss_backward_test, case_broadcast_succ_2)
{
  auto grad_output_desc = TensorDesc({1, 2, 4, 1}, ACL_FLOAT, ACL_FORMAT_ND);
  auto self_desc = TensorDesc({4,1}, ACL_FLOAT, ACL_FORMAT_ND);
  auto target_desc = TensorDesc({1,4,2}, ACL_FLOAT, ACL_FORMAT_ND);
  int64_t reduction = NONE;
  float beta = 1;
  auto grad_input_desc = TensorDesc({1, 2, 4, 2}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnSmoothL1LossBackward,
                      INPUT(grad_output_desc, self_desc, target_desc, reduction, beta),
                      OUTPUT(grad_input_desc));
  // SAMPLE: only test GetworkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// check broadcast failed
TEST_F(l2_smooth_l1_loss_backward_test, case_broadcast_fail_1)
{
  auto grad_output_desc = TensorDesc({1, 1, 4, 1}, ACL_FLOAT, ACL_FORMAT_ND);
  auto self_desc = TensorDesc({3,3}, ACL_FLOAT, ACL_FORMAT_ND);
  auto target_desc = TensorDesc({1,4,2}, ACL_FLOAT, ACL_FORMAT_ND);
  int64_t reduction = NONE;
  float beta = 1;
  auto grad_input_desc = TensorDesc({1, 1, 4, 2}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnSmoothL1LossBackward,
                      INPUT(grad_output_desc, self_desc, target_desc, reduction, beta),
                      OUTPUT(grad_input_desc));
  // SAMPLE: only test GetworkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);

  // // SAMPLE: precision simulate
  // ut.TestPrecision();
}

TEST_F(l2_smooth_l1_loss_backward_test, case_broadcast_fail_2)
{
  auto grad_output_desc = TensorDesc({1, 1, 4, 1}, ACL_FLOAT, ACL_FORMAT_ND);
  auto self_desc = TensorDesc({4,3}, ACL_FLOAT, ACL_FORMAT_ND);
  auto target_desc = TensorDesc({1,4,2}, ACL_FLOAT, ACL_FORMAT_ND);
  int64_t reduction = NONE;
  float beta = 1;
  auto grad_input_desc = TensorDesc({1, 1, 4, 3}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnSmoothL1LossBackward,
                      INPUT(grad_output_desc, self_desc, target_desc, reduction, beta),
                      OUTPUT(grad_input_desc));
  // SAMPLE: only test GetworkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);

  // // SAMPLE: precision simulate
  // ut.TestPrecision();
}

//check beta
TEST_F(l2_smooth_l1_loss_backward_test, case_beta)
{
  auto grad_output_desc = TensorDesc({1, 1, 4, 1}, ACL_FLOAT, ACL_FORMAT_ND);
  auto self_desc = TensorDesc({4,2}, ACL_FLOAT, ACL_FORMAT_ND);
  auto target_desc = TensorDesc({1,4,2}, ACL_FLOAT, ACL_FORMAT_ND);
  int64_t reduction = NONE;
  float beta = -1;
  auto grad_input_desc = TensorDesc({1, 1, 4, 2}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnSmoothL1LossBackward,
                      INPUT(grad_output_desc, self_desc, target_desc, reduction, beta),
                      OUTPUT(grad_input_desc));
  // SAMPLE: only test GetworkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// check not contiguous
TEST_F(l2_smooth_l1_loss_backward_test, case_not_contiguous)
{
  auto grad_output_desc = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 5}, 0, {4, 5}).ValueRange(1, 10);
  auto self_desc = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 5}, 0, {4, 5}).ValueRange(1, 10);
  auto target_desc = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 5}, 0, {4, 5}).ValueRange(1, 10);
  int64_t reduction = NONE;
  float beta = 1;
  auto grad_input_desc = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 5}, 0, {4, 5}).ValueRange(1, 10);

  auto ut = OP_API_UT(aclnnSmoothL1LossBackward,
                      INPUT(grad_output_desc, self_desc, target_desc, reduction, beta),
                      OUTPUT(grad_input_desc));
  // SAMPLE: only test GetworkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// check format
TEST_F(l2_smooth_l1_loss_backward_test, case_format)
{
  vector<aclFormat> ValidList = {
      ACL_FORMAT_NCHW,
      ACL_FORMAT_NHWC,
      ACL_FORMAT_ND,
      ACL_FORMAT_HWCN,
      ACL_FORMAT_NDHWC,
      ACL_FORMAT_NCDHW,
      ACL_FORMAT_NCL};
  int length = ValidList.size();
  for (int i = 0; i < length; i++) {
      auto grad_output_desc = TensorDesc({1, 1, 4, 4}, ACL_FLOAT, ValidList[i]);
      auto self_desc = TensorDesc({1, 1, 4, 4}, ACL_FLOAT, ValidList[i]);
      auto target_desc = TensorDesc({1, 1, 4, 4}, ACL_FLOAT, ValidList[i]);
      int64_t reduction = NONE;
      float beta = 1;
      auto grad_input_desc = TensorDesc({1, 1, 4, 4}, ACL_FLOAT, ValidList[i]);

      auto ut = OP_API_UT(aclnnSmoothL1LossBackward,
                          INPUT(grad_output_desc, self_desc, target_desc, reduction, beta),
                          OUTPUT(grad_input_desc));
      // SAMPLE: only test GetworkspaceSize
      uint64_t workspaceSize = 0;
      aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
      EXPECT_EQ(aclRet, ACLNN_SUCCESS);
  }
}

// Checkdtype BFLOAT16
TEST_F(l2_smooth_l1_loss_backward_test, ascend910B2_case_dtype_bfloat16)
{
    auto grad_output_desc = TensorDesc({1, 4, 4, 1}, ACL_BF16, ACL_FORMAT_ND)
                            .ValueRange(-2, 2);
    auto self_desc = TensorDesc({1, 4, 4, 1}, ACL_BF16, ACL_FORMAT_ND)
                     .ValueRange(-2, 2);
    auto target_desc = TensorDesc({1, 4, 4, 1}, ACL_BF16, ACL_FORMAT_ND)
                       .ValueRange(-2, 2);
    int64_t reduction = NONE;
    float beta = 1;
    auto grad_input_desc = TensorDesc(grad_output_desc).Precision(0.004, 0.004);

    auto ut = OP_API_UT(aclnnSmoothL1LossBackward,
                        INPUT(grad_output_desc, self_desc, target_desc, reduction, beta),
                        OUTPUT(grad_input_desc));
    // SAMPLE: only test GetworkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// empty tensor
TEST_F(l2_smooth_l1_loss_backward_test, ascend910_95_case_empty_tensor)
{
  auto grad_output_desc = TensorDesc({ 1, 0, 4, 1 }, ACL_FLOAT, ACL_FORMAT_ND)
    .ValueRange(-2, 2);
  auto self_desc = TensorDesc({ 1, 0, 4, 1 }, ACL_FLOAT, ACL_FORMAT_ND)
    .ValueRange(-2, 2);
  auto target_desc = TensorDesc({ 1, 0, 4, 1 }, ACL_FLOAT, ACL_FORMAT_ND)
    .ValueRange(-2, 2);
  int64_t reduction = NONE;
  float beta = 1;
  auto grad_input_desc = TensorDesc(grad_output_desc).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnSmoothL1LossBackward,
    INPUT(grad_output_desc, self_desc, target_desc, reduction, beta),
    OUTPUT(grad_input_desc));
  // SAMPLE: only test GetworkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_smooth_l1_loss_backward_test, ascend910_95_case_reduction_dtype_float32)
{
  for (int i = 0; i <= 3; i++) {
    auto grad_output_desc = TensorDesc({ 1,1,4,1 }, ACL_FLOAT, ACL_FORMAT_ND)
      .ValueRange(-2, 2);
    auto self_desc = TensorDesc({ 1,1,4,1 }, ACL_FLOAT, ACL_FORMAT_ND)
      .ValueRange(-2, 2);
    auto target_desc = TensorDesc({ 1,1,4,1 }, ACL_FLOAT, ACL_FORMAT_ND)
      .ValueRange(-2, 2);
    int64_t reduction = (enum Reduction)i;
    float beta = 1;
    auto grad_input_desc = TensorDesc(grad_output_desc).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnSmoothL1LossBackward,
      INPUT(grad_output_desc, self_desc, target_desc, reduction, beta),
      OUTPUT(grad_input_desc));
    // SAMPLE: only test GetworkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    if (i != 3) {
      EXPECT_EQ(aclRet, ACLNN_SUCCESS);
    }
    else {
      EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    }
  }
}

TEST_F(l2_smooth_l1_loss_backward_test, ascend910_95_case_reduction_dtype_float16)
{
  for (int i = 0; i <= 3; i++) {
    auto grad_output_desc = TensorDesc({ 1,1,4,1 }, ACL_FLOAT16, ACL_FORMAT_ND)
      .ValueRange(-2, 2);
    auto self_desc = TensorDesc({ 1,1,4,1 }, ACL_FLOAT16, ACL_FORMAT_ND)
      .ValueRange(-2, 2);
    auto target_desc = TensorDesc({ 1,1,4,1 }, ACL_FLOAT16, ACL_FORMAT_ND)
      .ValueRange(-2, 2);
    int64_t reduction = (enum Reduction)i;
    float beta = 1;
    auto grad_input_desc = TensorDesc(grad_output_desc).Precision(0.001, 0.001);

    auto ut = OP_API_UT(aclnnSmoothL1LossBackward,
      INPUT(grad_output_desc, self_desc, target_desc, reduction, beta),
      OUTPUT(grad_input_desc));
    // SAMPLE: only test GetworkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    if (i != 3) {
      EXPECT_EQ(aclRet, ACLNN_SUCCESS);
    }
    else {
      EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    }
  }
}

TEST_F(l2_smooth_l1_loss_backward_test, ascend910_95_case_reduction_dtype_bfloat16)
{
  for (int i = 0; i <= 3; i++) {
    auto grad_output_desc = TensorDesc({ 1,1,4,1 }, ACL_BF16, ACL_FORMAT_ND)
      .ValueRange(-2, 2);
    auto self_desc = TensorDesc({ 1,1,4,1 }, ACL_BF16, ACL_FORMAT_ND)
      .ValueRange(-2, 2);
    auto target_desc = TensorDesc({ 1,1,4,1 }, ACL_BF16, ACL_FORMAT_ND)
      .ValueRange(-2, 2);
    int64_t reduction = (enum Reduction)i;
    float beta = 1;
    auto grad_input_desc = TensorDesc(grad_output_desc).Precision(0.005, 0.005);

    auto ut = OP_API_UT(aclnnSmoothL1LossBackward,
      INPUT(grad_output_desc, self_desc, target_desc, reduction, beta),
      OUTPUT(grad_input_desc));
    // SAMPLE: only test GetworkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    if (i != 3) {
      EXPECT_EQ(aclRet, ACLNN_SUCCESS);
    }
    else {
      EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    }
  }
}

TEST_F(l2_smooth_l1_loss_backward_test, ascend910_95_case_scalar_reduction_dtype_float32)
{
  for (int i = 0; i <= 3; i++) {
    auto grad_output_desc = TensorDesc({ 1, 0, 4, 1 }, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto self_desc = TensorDesc({ 4,1 }, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto target_desc = TensorDesc({ 1,4,2 }, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    int64_t reduction = (enum Reduction)i;
    float beta = 1;
    auto grad_input_desc = TensorDesc({1, 0, 4, 2}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnSmoothL1LossBackward,
      INPUT(grad_output_desc, self_desc, target_desc, reduction, beta),
      OUTPUT(grad_input_desc));
    // SAMPLE: only test GetworkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    if (i != 3) {
      EXPECT_EQ(aclRet, ACLNN_SUCCESS);
    }
    else {
      EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    }
  }
}

TEST_F(l2_smooth_l1_loss_backward_test, ascend910_95_case_scalar_reduction_dtype_float16)
{
  for (int i = 0; i <= 3; i++) {
    auto grad_output_desc = TensorDesc({ 1, 0, 4, 1 }, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto self_desc = TensorDesc({ 4,1 }, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto target_desc = TensorDesc({ 1,4,2 }, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    int64_t reduction = (enum Reduction) i;
    float beta = 1;
    auto grad_input_desc = TensorDesc({1, 0, 4, 2}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(aclnnSmoothL1LossBackward,
                        INPUT(grad_output_desc, self_desc, target_desc, reduction, beta),
                        OUTPUT(grad_input_desc));
    // SAMPLE: only test GetworkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    if (i != 3) {
      EXPECT_EQ(aclRet, ACLNN_SUCCESS);
    }
    else {
      EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    }
  }
}

TEST_F(l2_smooth_l1_loss_backward_test, ascend910_95_case_scalar_reduction_dtype_bfloat16)
{
  for (int i = 0; i <= 3; i++) {
    auto grad_output_desc = TensorDesc({1, 0, 4, 1}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto self_desc = TensorDesc({ 4,1 }, ACL_BF16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto target_desc = TensorDesc({ 1,4,2 }, ACL_BF16, ACL_FORMAT_ND).ValueRange(-2, 2);
    int64_t reduction = (enum Reduction) i;
    float beta = 1;
    auto grad_input_desc = TensorDesc({1, 0, 4, 2}, ACL_BF16, ACL_FORMAT_ND).Precision(0.005, 0.005);

    auto ut = OP_API_UT(aclnnSmoothL1LossBackward,
                        INPUT(grad_output_desc, self_desc, target_desc, reduction, beta),
                        OUTPUT(grad_input_desc));
    // SAMPLE: only test GetworkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    if (i != 3) {
      EXPECT_EQ(aclRet, ACLNN_SUCCESS);
    }
    else {
      EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    }
  }
}
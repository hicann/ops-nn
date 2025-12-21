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

#include "../../../op_host/op_api/aclnn_modulate_backward.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"

using namespace op;
using namespace std;

class l2_modulate_backward_test : public testing::Test {
 protected:
  static void SetUpTestCase() { std::cout << "aclnnModulateBackward_test SetUp" << std::endl; }

  static void TearDownTestCase() { std::cout << "aclnnModulateBackward_test TearDown" << std::endl; }
};

TEST_F(l2_modulate_backward_test, ascend910B_case_0) {
    auto grad_output_desc = TensorDesc({32, 8, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto input_desc = TensorDesc({32, 8, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto scale_desc = TensorDesc({32, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto shift_desc = TensorDesc({32, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto grad_input_desc = TensorDesc({32, 8, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto grad_scale_desc = TensorDesc({32, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto grad_shift_desc = TensorDesc({32, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);

    auto ut = OP_API_UT(aclnnModulateBackward, INPUT(grad_output_desc, input_desc, scale_desc, shift_desc), OUTPUT(grad_input_desc, grad_scale_desc,grad_shift_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
//     EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}
// // checkNotNull
// TEST_F(l2_modulate_backward_test, ascend910B_case_1) {
//     auto scale_desc = TensorDesc({32, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
//     auto shift_desc = TensorDesc({32, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
//     auto grad_scale_desc = TensorDesc({32, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
//     auto grad_shift_desc = TensorDesc({32, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
//     auto ut = OP_API_UT(aclnnModulateBackward, INPUT((aclTensor*)nullptr, (aclTensor*)nullptr, scale_desc, shift_desc), OUTPUT(grad_scale_desc, grad_shift_desc));
//     // SAMPLE: only test GetWorkspaceSize
//     uint64_t workspace_size = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
//     EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
// }

// CheckDtypeValid
// TEST_F(l2_modulate_backward_test, ascend910B_case_2) {
//     vector<aclDataType> ValidList = {
//         ACL_FLOAT,
//         ACL_FLOAT16,
//         ACL_BF16,
//         ACL_DT_UNDEFINED};
//     int length = ValidList.size();
//     for (int i = 0; i < length; i++) {
//         auto grad_output_desc = TensorDesc({32, 8, 1024}, ValidList[i], ACL_FORMAT_ND).ValueRange(-10, 10);
//         auto input_desc = TensorDesc({32, 8, 1024}, ValidList[i], ACL_FORMAT_ND).ValueRange(-10, 10);
//         auto scale_desc = TensorDesc({32, 1024}, ValidList[i], ACL_FORMAT_ND).ValueRange(-10, 10);
//         auto shift_desc = TensorDesc({32, 1024}, ValidList[i], ACL_FORMAT_ND).ValueRange(-10, 10);
//         auto grad_input_desc = TensorDesc({32, 8, 1024}, ValidList[i], ACL_FORMAT_ND).ValueRange(-10, 10);
//         auto grad_scale_desc = TensorDesc({32, 1024}, ValidList[i], ACL_FORMAT_ND).ValueRange(-10, 10);
//         auto grad_shift_desc = TensorDesc({32, 1024}, ValidList[i], ACL_FORMAT_ND).ValueRange(-10, 10);
//         auto ut = OP_API_UT(aclnnModulateBackward, INPUT(grad_output_desc, input_desc, scale_desc, shift_desc), OUTPUT(grad_input_desc, grad_scale_desc, grad_shift_desc));

//         // SAMPLE: only test GetWorkspaceSize
//         uint64_t workspaceSize = 0;
//         aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
        // if (ValidList[i] != ACL_DT_UNDEFINED) {
        //     EXPECT_EQ(aclRet, ACL_SUCCESS);
        //     ut.TestPrecision();
        // } else {
        //     EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
        // }
//     }
// }

// // checkDtype different dtype of input
// TEST_F(l2_modulate_backward_test, ascend910B_case_3) {
//     auto grad_output_desc = TensorDesc({32, 8, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
//     auto input_desc = TensorDesc({32, 8, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
//     auto scale_desc = TensorDesc({32, 1024}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
//     auto shift_desc = TensorDesc({32, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
//     auto grad_input_desc = TensorDesc({32, 8, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
//     auto grad_scale_desc = TensorDesc({32, 1024}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
//     auto grad_shift_desc = TensorDesc({32, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);

//     auto ut = OP_API_UT(aclnnModulateBackward, INPUT(grad_output_desc, input_desc, scale_desc, shift_desc), OUTPUT(grad_input_desc, grad_scale_desc,grad_shift_desc));
//     // SAMPLE: only test GetWorkspaceSize
//     uint64_t workspace_size = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
//     EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
// }

// // checkDtype different dtype of input
// TEST_F(l2_modulate_backward_test, ascend910B_case_4) {
//     auto grad_output_desc = TensorDesc({32, 8, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
//     auto input_desc = TensorDesc({32, 8, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
//     auto scale_desc = TensorDesc({32, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
//     auto shift_desc = TensorDesc({32, 1024}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
//     auto grad_input_desc = TensorDesc({32, 8, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
//     auto grad_scale_desc = TensorDesc({32, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
//     auto grad_shift_desc = TensorDesc({32, 1024}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);

//     auto ut = OP_API_UT(aclnnModulateBackward, INPUT(grad_output_desc, input_desc, scale_desc, shift_desc), OUTPUT(grad_input_desc, grad_scale_desc,grad_shift_desc));
//     // SAMPLE: only test GetWorkspaceSize
//     uint64_t workspace_size = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
//     EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
// }

// // checkShape for self
// TEST_F(l2_modulate_backward_test, ascend910B_case_5) {
//     auto grad_output_desc = TensorDesc({32, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
//     auto input_desc = TensorDesc({32, 8, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
//     auto scale_desc = TensorDesc({32, 1024}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
//     auto shift_desc = TensorDesc({32, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
//     auto grad_input_desc = TensorDesc({32, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
//     auto grad_scale_desc = TensorDesc({32, 1024}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
//     auto grad_shift_desc = TensorDesc({32, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);

//     auto ut = OP_API_UT(aclnnModulateBackward, INPUT(grad_output_desc, input_desc, scale_desc, shift_desc), OUTPUT(grad_input_desc, grad_scale_desc,grad_shift_desc));
//     // SAMPLE: only test GetWorkspaceSize
//     uint64_t workspace_size = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
//     EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
// }

// // checkShape for scale
// TEST_F(l2_modulate_backward_test, ascend910B_case_6) {
//     auto grad_output_desc = TensorDesc({32, 8, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
//     auto input_desc = TensorDesc({32, 8, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
//     auto scale_desc = TensorDesc({32}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
//     auto shift_desc = TensorDesc({32, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
//     auto grad_input_desc = TensorDesc({32, 8, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
//     auto grad_scale_desc = TensorDesc({32}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
//     auto grad_shift_desc = TensorDesc({32, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);

//     auto ut = OP_API_UT(aclnnModulateBackward, INPUT(grad_output_desc, input_desc, scale_desc, shift_desc), OUTPUT(grad_input_desc, grad_scale_desc,grad_shift_desc));
//     // SAMPLE: only test GetWorkspaceSize
//     uint64_t workspace_size = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
//     EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
// }

// // checkShape for scale
// TEST_F(l2_modulate_backward_test, ascend910B_case_7) {
//     auto grad_output_desc = TensorDesc({32, 8, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
//     auto input_desc = TensorDesc({32, 8, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
//     auto scale_desc = TensorDesc({32, 512}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
//     auto shift_desc = TensorDesc({32, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
//     auto grad_input_desc = TensorDesc({32, 8, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
//     auto grad_scale_desc = TensorDesc({32, 512}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
//     auto grad_shift_desc = TensorDesc({32, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);

//     auto ut = OP_API_UT(aclnnModulateBackward, INPUT(grad_output_desc, input_desc, scale_desc, shift_desc), OUTPUT(grad_input_desc, grad_scale_desc,grad_shift_desc));
//     // SAMPLE: only test GetWorkspaceSize
//     uint64_t workspace_size = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
//     EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
// }

// // checkShape for scale 3D检查
// TEST_F(l2_modulate_backward_test, ascend910B_case_8) {
//     auto grad_output_desc = TensorDesc({32, 8, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
//     auto input_desc = TensorDesc({32, 8, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
//     auto scale_desc = TensorDesc({32, 2, 1024}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
//     auto shift_desc = TensorDesc({32, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
//     auto grad_input_desc = TensorDesc({32, 8, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
//     auto grad_scale_desc = TensorDesc({32, 2, 1024}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
//     auto grad_shift_desc = TensorDesc({32, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);

//     auto ut = OP_API_UT(aclnnModulateBackward, INPUT(grad_output_desc, input_desc, scale_desc, shift_desc), OUTPUT(grad_input_desc, grad_scale_desc,grad_shift_desc));
//     // SAMPLE: only test GetWorkspaceSize
//     uint64_t workspace_size = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
//     EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
// }

// // checkShape for shift
// TEST_F(l2_modulate_backward_test, ascend910B_case_9) {
//     auto grad_output_desc = TensorDesc({32, 8, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
//     auto input_desc = TensorDesc({32, 8, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
//     auto scale_desc = TensorDesc({32, 1024}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
//     auto shift_desc = TensorDesc({32}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
//     auto grad_input_desc = TensorDesc({32, 8, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
//     auto grad_scale_desc = TensorDesc({32, 1024}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
//     auto grad_shift_desc = TensorDesc({32}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);

//     auto ut = OP_API_UT(aclnnModulateBackward, INPUT(grad_output_desc, input_desc, scale_desc, shift_desc), OUTPUT(grad_input_desc, grad_scale_desc,grad_shift_desc));
//     // SAMPLE: only test GetWorkspaceSize
//     uint64_t workspace_size = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
//     EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
// }

// // checkShape for shift
// TEST_F(l2_modulate_backward_test, ascend910B_case_10) {
//     auto grad_output_desc = TensorDesc({32, 8, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
//     auto input_desc = TensorDesc({32, 8, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
//     auto scale_desc = TensorDesc({32, 1024}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
//     auto shift_desc = TensorDesc({32, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
//     auto grad_input_desc = TensorDesc({32, 8, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
//     auto grad_scale_desc = TensorDesc({32, 1024}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
//     auto grad_shift_desc = TensorDesc({32, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);

//     auto ut = OP_API_UT(aclnnModulateBackward, INPUT(grad_output_desc, input_desc, scale_desc, shift_desc), OUTPUT(grad_input_desc, grad_scale_desc,grad_shift_desc));
//     // SAMPLE: only test GetWorkspaceSize
//     uint64_t workspace_size = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
//     EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
// }

// // checkShape for shift
// TEST_F(l2_modulate_backward_test, ascend910B_case_11) {
//     auto grad_output_desc = TensorDesc({32, 8, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
//     auto input_desc = TensorDesc({32, 8, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
//     auto scale_desc = TensorDesc({32, 1024}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
//     auto shift_desc = TensorDesc({32, 2, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
//     auto grad_input_desc = TensorDesc({32, 8, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
//     auto grad_scale_desc = TensorDesc({32, 1024}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
//     auto grad_shift_desc = TensorDesc({32, 2, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);

//     auto ut = OP_API_UT(aclnnModulateBackward, INPUT(grad_output_desc, input_desc, scale_desc, shift_desc), OUTPUT(grad_input_desc, grad_scale_desc,grad_shift_desc));
//     // SAMPLE: only test GetWorkspaceSize
//     uint64_t workspace_size = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
//     EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
// }

// checkShape for out
// TEST_F(l2_modulate_backward_test, ascend910B_case_12) {
//     auto grad_output_desc = TensorDesc({32, 8, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
//     auto input_desc = TensorDesc({32, 8, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
//     auto scale_desc = TensorDesc({32, 1024}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
//     auto shift_desc = TensorDesc({32, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
//     auto grad_input_desc = TensorDesc({32, 8, 512}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
//     auto grad_scale_desc = TensorDesc({32, 1024}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
//     auto grad_shift_desc = TensorDesc({32, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);

//     auto ut = OP_API_UT(aclnnModulateBackward, INPUT(grad_output_desc, input_desc, scale_desc, shift_desc), OUTPUT(grad_input_desc, grad_scale_desc,grad_shift_desc));
//     // SAMPLE: only test GetWorkspaceSize
//     uint64_t workspace_size = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
//     EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
// }

// // test optional without scale has shift
// TEST_F(l2_modulate_backward_test, ascend910B_case_13) {
//     auto grad_output_desc = TensorDesc({32, 8, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
//     auto input_desc = TensorDesc({32, 8, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
//     auto shift_desc = TensorDesc({32, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
//     auto grad_input_desc = TensorDesc({32, 8, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
//     auto grad_shift_desc = TensorDesc({32, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);

//     auto ut = OP_API_UT(aclnnModulateBackward, INPUT(grad_output_desc, input_desc, (aclTensor*)nullptr, shift_desc), OUTPUT(grad_input_desc, (aclTensor*)nullptr, grad_shift_desc));
//     // SAMPLE: only test GetWorkspaceSize
//     uint64_t workspace_size = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
//     // EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
// }

// // test optional without shift has scale
// TEST_F(l2_modulate_backward_test, ascend910B_case_14) {
//     auto grad_output_desc = TensorDesc({32, 8, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
//     auto input_desc = TensorDesc({32, 8, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
//     auto scale_desc = TensorDesc({32, 1024}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
//     auto grad_input_desc = TensorDesc({32, 8, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
//     auto grad_scale_desc = TensorDesc({32, 1024}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);

//     auto ut = OP_API_UT(aclnnModulateBackward, INPUT(grad_output_desc, input_desc, scale_desc, (aclTensor*)nullptr), OUTPUT(grad_input_desc, grad_scale_desc,(aclTensor*)nullptr));
//     // SAMPLE: only test GetWorkspaceSize
//     uint64_t workspace_size = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
//     EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
// }

// //only input
// TEST_F(l2_modulate_backward_test, ascend910B_case_15) {
//     auto grad_output_desc = TensorDesc({32, 8, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
//     auto input_desc = TensorDesc({32, 8, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
//     auto grad_input_desc = TensorDesc({32, 8, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);

//     auto ut = OP_API_UT(aclnnModulateBackward, INPUT(grad_output_desc, input_desc, (aclTensor*)nullptr, (aclTensor*)nullptr), OUTPUT(grad_input_desc, (aclTensor*)nullptr, (aclTensor*)nullptr));
//     // SAMPLE: only test GetWorkspaceSize
//     uint64_t workspace_size = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
//     EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
// }

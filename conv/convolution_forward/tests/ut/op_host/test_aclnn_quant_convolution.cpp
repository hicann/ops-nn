/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_quant_convolution.cpp
 * \brief
 */

#include <float.h>

#include <array>
#include <iostream>
#include <vector>

#include "gtest/gtest.h"
#include "../../../op_host/op_api/aclnn_quant_convolution.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

#include "opdev/platform.h"

using namespace op;
using namespace std;

class quant_convolution_test : public testing::Test {
public:
  void SetUp() {
    cout << "quant_convolution_test SetUp" << endl;
  }

  void TearDown() {
    cout << "quant_convolution_test TearDown" << endl;
  }
};

TEST_F(quant_convolution_test, ascend910B2_test_quantConv3D_bias) {
  // test quantConv3d
  vector<aclDataType> ValidList = {ACL_FLOAT, ACL_BF16};
  int length = ValidList.size();

  bool transposed = false;
  const int64_t groups = 1;
  vector<int64_t> inp_dims = {2, 2, 32, 32, 32};
  vector<int64_t> weight_dims = {2, 2 / groups, 3, 3, 3};
  vector<int64_t> bias_dims = {2 / groups};
  vector<int64_t> scale_dims = {2 / groups};
  vector<int64_t> output_dims = {2, 2, 32, 32, 32};
  vector<int64_t> strides_dims = {1, 1, 1};
  vector<int64_t> padding_dims = {1, 1, 1};
  vector<int64_t> dilation_dims = {1, 1, 1};
  vector<int64_t> output_padding_dims = {0, 0, 0};

  for (int i = 0; i < length; i++) {
    auto inp_desc = TensorDesc(inp_dims, ACL_INT8, ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto weight_desc = TensorDesc(weight_dims, ACL_INT8, ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto bias_desc = TensorDesc(bias_dims, ValidList[i], ACL_FORMAT_ND).ValueRange(0, 2);
    auto scale_desc = TensorDesc(scale_dims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto offset_desc = nullptr;
    auto strides_desc = IntArrayDesc(strides_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    auto output_padding_desc = IntArrayDesc(output_padding_dims);
    int32_t offsetx = 0;
    auto roundMode = nullptr;
    auto output_desc = TensorDesc(output_dims, ACL_BF16, ACL_FORMAT_NCDHW).Precision(0.01, 0.01);

    auto ut = OP_API_UT(aclnnQuantConvolution,  // host api第二段接口名称
                        INPUT(inp_desc, weight_desc, bias_desc, scale_desc, offset_desc, strides_desc, padding_desc, dilation_desc, transposed,
                              output_padding_desc, groups, offsetx, roundMode),  // host api输入
                        OUTPUT(output_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
    EXPECT_EQ(aclRet, ACL_SUCCESS);
  }
}

TEST_F(quant_convolution_test, ascend910B2_test_quantConv3D_nonbias) {
  // test quantConv3d
  vector<aclDataType> ValidList = {ACL_FLOAT, ACL_BF16};
  int length = ValidList.size();

  bool transposed = false;
  const int64_t groups = 1;
  vector<int64_t> inp_dims = {2, 2, 32, 32, 32};
  vector<int64_t> weight_dims = {2, 2 / groups, 3, 3, 3};
  vector<int64_t> bias_dims = {2 / groups};
  vector<int64_t> scale_dims = {2 / groups};
  vector<int64_t> output_dims = {2, 2, 32, 32, 32};
  vector<int64_t> strides_dims = {1, 1, 1};
  vector<int64_t> padding_dims = {1, 1, 1};
  vector<int64_t> dilation_dims = {1, 1, 1};
  vector<int64_t> output_padding_dims = {0, 0, 0};

  for (int i = 0; i < length; i++) {
    auto inp_desc = TensorDesc(inp_dims, ACL_INT8, ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto weight_desc = TensorDesc(weight_dims, ACL_INT8, ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto bias_desc = nullptr;
    auto scale_desc = TensorDesc(scale_dims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto offset_desc = nullptr;
    auto strides_desc = IntArrayDesc(strides_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    auto output_padding_desc = IntArrayDesc(output_padding_dims);
    int32_t offsetx = 0;
    auto roundMode = nullptr;
    auto output_desc = TensorDesc(output_dims, ACL_BF16, ACL_FORMAT_NCDHW).Precision(0.01, 0.01);

    auto ut = OP_API_UT(aclnnQuantConvolution,  // host api第二段接口名称
                        INPUT(inp_desc, weight_desc, bias_desc, scale_desc, offset_desc, strides_desc, padding_desc, dilation_desc, transposed,
                              output_padding_desc, groups, offsetx, roundMode),  // host api输入
                        OUTPUT(output_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
  }
}

TEST_F(quant_convolution_test, test_quantConv3d_incorrect_soc_version_negative) {
  // test quantConv3d
  vector<aclDataType> ValidList = {ACL_FLOAT, ACL_BF16};
  int length = ValidList.size();

  bool transposed = false;
  const int64_t groups = 1;
  vector<int64_t> inp_dims = {2, 2, 32, 32, 32};
  vector<int64_t> weight_dims = {2, 2 / groups, 3, 3, 3};
  vector<int64_t> bias_dims = {2 / groups};
  vector<int64_t> scale_dims = {2 / groups};
  vector<int64_t> output_dims = {2, 2, 32, 32, 32};
  vector<int64_t> strides_dims = {1, 1, 1};
  vector<int64_t> padding_dims = {1, 1, 1};
  vector<int64_t> dilation_dims = {1, 1, 1};
  vector<int64_t> output_padding_dims = {0, 0, 0};

  for (int i = 0; i < length; i++) {
    auto inp_desc = TensorDesc(inp_dims, ACL_INT8, ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto weight_desc = TensorDesc(weight_dims, ACL_INT8, ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto bias_desc = TensorDesc(bias_dims, ValidList[i], ACL_FORMAT_ND).ValueRange(0, 2);
    auto scale_desc = TensorDesc(scale_dims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto offset_desc = nullptr;
    auto strides_desc = IntArrayDesc(strides_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    auto output_padding_desc = IntArrayDesc(output_padding_dims);
    int32_t offsetx = 0;
    auto roundMode = nullptr;
    auto output_desc = TensorDesc(output_dims, ACL_BF16, ACL_FORMAT_NCDHW).Precision(0.01, 0.01);

    auto ut = OP_API_UT(aclnnQuantConvolution,  // host api第二段接口名称
                        INPUT(inp_desc, weight_desc, bias_desc, scale_desc, offset_desc, strides_desc, padding_desc, dilation_desc, transposed,
                              output_padding_desc, groups, offsetx, roundMode),  // host api输入
                        OUTPUT(output_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
    auto curSoc = GetCurrentPlatformInfo().GetSocVersion();
    if (curSoc == SocVersion::ASCEND910B) {
        EXPECT_EQ(aclRet, ACL_SUCCESS);
    } else {
        EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    }
  }
}

TEST_F(quant_convolution_test, ascend910B2_quantConv3d_groups_not_equal_1) {
  // test quantConv3d
  vector<aclDataType> ValidList = {ACL_FLOAT, ACL_BF16};
  int length = ValidList.size();

  bool transposed = false;
  const int64_t groups = 2;
  vector<int64_t> inp_dims = {2, 2, 32, 32, 32};
  vector<int64_t> weight_dims = {2, 2 / groups, 3, 3, 3};
  vector<int64_t> bias_dims = {2 / groups};
  vector<int64_t> scale_dims = {2 / groups};
  vector<int64_t> output_dims = {2, 2, 32, 32, 32};
  vector<int64_t> strides_dims = {1, 1, 1};
  vector<int64_t> padding_dims = {1, 1, 1};
  vector<int64_t> dilation_dims = {1, 1, 1};
  vector<int64_t> output_padding_dims = {0, 0, 0};

  for (int i = 0; i < length; i++) {
    auto inp_desc = TensorDesc(inp_dims, ACL_INT8, ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto weight_desc = TensorDesc(weight_dims, ACL_INT8, ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto bias_desc = TensorDesc(bias_dims, ValidList[i], ACL_FORMAT_ND).ValueRange(0, 2);
    auto scale_desc = TensorDesc(scale_dims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto offset_desc = nullptr;
    auto strides_desc = IntArrayDesc(strides_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    auto output_padding_desc = IntArrayDesc(output_padding_dims);
    int32_t offsetx = 0;
    auto roundMode = nullptr;
    auto output_desc = TensorDesc(output_dims, ACL_BF16, ACL_FORMAT_NCDHW).Precision(0.01, 0.01);

    auto ut = OP_API_UT(aclnnQuantConvolution,  // host api第二段接口名称
                        INPUT(inp_desc, weight_desc, bias_desc, scale_desc, offset_desc, strides_desc,
                              padding_desc, dilation_desc, transposed, output_padding_desc, groups, offsetx, roundMode),  // host api输入
                        OUTPUT(output_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
  }
}

TEST_F(quant_convolution_test, ascend910B2_quantConv3d_incorrect_data_format_negative) {
  vector<aclDataType> ValidList = {ACL_BF16};
  int length = ValidList.size();

  bool transposed = false;
  const int64_t groups = 1;
  vector<int64_t> inp_dims = {2, 2, 32, 32, 32};
  vector<int64_t> weight_dims = {2, 2 / groups, 3, 3, 3};
  vector<int64_t> bias_dims = {2 / groups};
  vector<int64_t> scale_dims = {2 / groups};
  vector<int64_t> output_dims = {2, 2, 32, 32, 32};
  vector<int64_t> strides_dims = {1, 1, 1};
  vector<int64_t> padding_dims = {1, 1, 1};
  vector<int64_t> dilation_dims = {1, 1, 1};
  vector<int64_t> output_padding_dims = {0, 0, 0};

  for (int i = 0; i < length; i++) {
    auto inp_desc = TensorDesc(inp_dims, ACL_INT8, ACL_FORMAT_NDHWC).ValueRange(0, 2);
    auto weight_desc = TensorDesc(weight_dims, ACL_INT8, ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto bias_desc = TensorDesc(bias_dims, ValidList[i], ACL_FORMAT_ND).ValueRange(0, 2);
    auto scale_desc = TensorDesc(scale_dims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto offset_desc = nullptr;
    auto strides_desc = IntArrayDesc(strides_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    auto output_padding_desc = IntArrayDesc(output_padding_dims);
    int32_t offsetx = 0;
    auto roundMode = nullptr;
    auto output_desc = TensorDesc(output_dims, ACL_BF16, ACL_FORMAT_NCDHW).Precision(0.01, 0.01);

    auto ut = OP_API_UT(aclnnQuantConvolution,  // host api第二段接口名称
                        INPUT(inp_desc, weight_desc, bias_desc, scale_desc, offset_desc, strides_desc, padding_desc, dilation_desc, transposed,
                              output_padding_desc, groups, offsetx, roundMode),  // host api输入
                        OUTPUT(output_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
  }
}

TEST_F(quant_convolution_test, ascend910B2_test_quantConv3d_incorrect_output_shape) {
  // test conv3d
  vector<aclDataType> ValidList = {ACL_BF16};
  int length = ValidList.size();

  bool transposed = false;
  const int64_t groups = 1;
  vector<int64_t> inp_dims = {2, 2, 32, 32, 32};
  vector<int64_t> weight_dims = {2, 2 / groups, 3, 3, 3};
  vector<int64_t> bias_dims = {2 / groups};
  vector<int64_t> scale_dims = {2 / groups};
  vector<int64_t> output_dims = {2, 2, 30, 30, 30};
  vector<int64_t> strides_dims = {1, 1, 1};
  vector<int64_t> padding_dims = {1, 1, 1};
  vector<int64_t> dilation_dims = {1, 1, 1};
  vector<int64_t> output_padding_dims = {0, 0, 0};

  for (int i = 0; i < length; i++) {
    auto inp_desc = TensorDesc(inp_dims, ACL_INT8, ACL_FORMAT_NDHWC).ValueRange(0, 2);
    auto weight_desc = TensorDesc(weight_dims, ACL_INT8, ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto bias_desc = TensorDesc(bias_dims, ValidList[i], ACL_FORMAT_ND).ValueRange(0, 2);
    auto scale_desc = TensorDesc(scale_dims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto offset_desc = nullptr;
    auto strides_desc = IntArrayDesc(strides_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    auto output_padding_desc = IntArrayDesc(output_padding_dims);
    int32_t offsetx = 0;
    auto roundMode = nullptr;
    auto output_desc = TensorDesc(output_dims, ACL_BF16, ACL_FORMAT_NCDHW).Precision(0.01, 0.01);

    auto ut = OP_API_UT(aclnnQuantConvolution,  // host api第二段接口名称
                        INPUT(inp_desc, weight_desc, bias_desc, scale_desc, offset_desc,
                              strides_desc, padding_desc, dilation_desc, transposed,
                              output_padding_desc, groups, offsetx, roundMode),  // host api输入
                        OUTPUT(output_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
  }
}

TEST_F(quant_convolution_test, ascend910B2_test_quantConv3d_incorrect_scale_shape_value) {
  // test conv3d
  vector<aclDataType> ValidList = {ACL_BF16};
  int length = ValidList.size();

  bool transposed = false;
  const int64_t groups = 1;
  vector<int64_t> inp_dims = {2, 2, 32, 32, 32};
  vector<int64_t> weight_dims = {2, 2 / groups, 3, 3, 3};
  vector<int64_t> bias_dims = {2 / groups};
  vector<int64_t> scale_dims = {1};
  vector<int64_t> output_dims = {2, 2, 32, 32, 32};
  vector<int64_t> strides_dims = {1, 1, 1};
  vector<int64_t> padding_dims = {1, 1, 1};
  vector<int64_t> dilation_dims = {1, 1, 1};
  vector<int64_t> output_padding_dims = {0, 0, 0};

  for (int i = 0; i < length; i++) {
    auto inp_desc = TensorDesc(inp_dims, ACL_INT8, ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto weight_desc = TensorDesc(weight_dims, ACL_INT8, ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto bias_desc = TensorDesc(bias_dims, ValidList[i], ACL_FORMAT_ND).ValueRange(0, 2);
    auto scale_desc = TensorDesc(scale_dims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto offset_desc = nullptr;
    auto strides_desc = IntArrayDesc(strides_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    auto output_padding_desc = IntArrayDesc(output_padding_dims);
    int32_t offsetx = 0;
    auto roundMode = nullptr;
    auto output_desc = TensorDesc(output_dims, ACL_BF16, ACL_FORMAT_NCDHW).Precision(0.01, 0.01);

    auto ut = OP_API_UT(aclnnQuantConvolution,  // host api第二段接口名称
                        INPUT(inp_desc, weight_desc, bias_desc, scale_desc, offset_desc,
                              strides_desc, padding_desc, dilation_desc, transposed,
                              output_padding_desc, groups, offsetx, roundMode),  // host api输入
                        OUTPUT(output_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
  }
}

TEST_F(quant_convolution_test, ascend910B2_test_quantConv3d_incorrect_input_dim) {
  // test conv3d
  vector<aclDataType> ValidList = {ACL_BF16};
  int length = ValidList.size();

  bool transposed = false;
  const int64_t groups = 1;
  vector<int64_t> inp_dims = {2, 2, 32, 32};
  vector<int64_t> weight_dims = {2, 2 / groups, 3, 3, 3};
  vector<int64_t> bias_dims = {2 / groups};
  vector<int64_t> scale_dims = {2 / groups};
  vector<int64_t> output_dims = {2, 2, 32, 32, 32};
  vector<int64_t> strides_dims = {1, 1, 1};
  vector<int64_t> padding_dims = {1, 1, 1};
  vector<int64_t> dilation_dims = {1, 1, 1};
  vector<int64_t> output_padding_dims = {0, 0, 0};

  for (int i = 0; i < length; i++) {
    auto inp_desc = TensorDesc(inp_dims, ACL_INT8, ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto weight_desc = TensorDesc(weight_dims, ACL_INT8, ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto bias_desc = TensorDesc(bias_dims, ValidList[i], ACL_FORMAT_ND).ValueRange(0, 2);
    auto scale_desc = TensorDesc(scale_dims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto offset_desc = nullptr;
    auto strides_desc = IntArrayDesc(strides_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    auto output_padding_desc = IntArrayDesc(output_padding_dims);
    int32_t offsetx = 0;
    auto roundMode = nullptr;
    auto output_desc = TensorDesc(output_dims, ACL_BF16, ACL_FORMAT_NCDHW).Precision(0.01, 0.01);

    auto ut = OP_API_UT(aclnnQuantConvolution,  // host api第二段接口名称
                        INPUT(inp_desc, weight_desc, bias_desc, scale_desc, offset_desc,
                              strides_desc, padding_desc, dilation_desc, transposed,
                              output_padding_desc, groups, offsetx, roundMode),  // host api输入
                        OUTPUT(output_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
  }
}

TEST_F(quant_convolution_test, ascend910B2_test_quantConv3d_incorrect_attr_dim) {
  // test conv3d
  vector<aclDataType> ValidList = {ACL_BF16};
  int length = ValidList.size();

  bool transposed = false;
  const int64_t groups = 1;
  vector<int64_t> inp_dims = {2, 2, 32, 32, 32};
  vector<int64_t> weight_dims = {2, 2 / groups, 3, 3, 3};
  vector<int64_t> bias_dims = {2 / groups};
  vector<int64_t> scale_dims = {2 / groups};
  vector<int64_t> output_dims = {2, 2, 32, 32, 32};
  vector<int64_t> strides_dims = {1, 1, 1};
  vector<int64_t> padding_dims = {1, 1};
  vector<int64_t> dilation_dims = {1, 1, 1};
  vector<int64_t> output_padding_dims = {0, 0, 0};

  for (int i = 0; i < length; i++) {
    auto inp_desc = TensorDesc(inp_dims, ACL_INT8, ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto weight_desc = TensorDesc(weight_dims, ACL_INT8, ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto bias_desc = TensorDesc(bias_dims, ValidList[i], ACL_FORMAT_ND).ValueRange(0, 2);
    auto scale_desc = TensorDesc(scale_dims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto offset_desc = nullptr;
    auto strides_desc = IntArrayDesc(strides_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    auto output_padding_desc = IntArrayDesc(output_padding_dims);
    int32_t offsetx = 0;
    auto roundMode = nullptr;
    auto output_desc = TensorDesc(output_dims, ACL_BF16, ACL_FORMAT_NCDHW).Precision(0.01, 0.01);

    auto ut = OP_API_UT(aclnnQuantConvolution,  // host api第二段接口名称
                        INPUT(inp_desc, weight_desc, bias_desc, scale_desc, offset_desc,
                              strides_desc, padding_desc, dilation_desc, transposed,
                              output_padding_desc, groups, offsetx, roundMode),  // host api输入
                        OUTPUT(output_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
  }
}

TEST_F(quant_convolution_test, ascend910B2_test_quantConv3d_attr_value_has_zero) {
  // test conv3d
  vector<aclDataType> ValidList = {ACL_BF16};
  int length = ValidList.size();

  bool transposed = false;
  const int64_t groups = 1;
  vector<int64_t> inp_dims = {2, 2, 32, 32, 32};
  vector<int64_t> weight_dims = {2, 2 / groups, 3, 3, 3};
  vector<int64_t> bias_dims = {2 / groups};
  vector<int64_t> scale_dims = {2 / groups};
  vector<int64_t> output_dims = {2, 2, 32, 32, 32};
  vector<int64_t> strides_dims = {1, 1, 0};
  vector<int64_t> padding_dims = {1, 1, 1};
  vector<int64_t> dilation_dims = {1, 1, 1};
  vector<int64_t> output_padding_dims = {0, 0, 0};

  for (int i = 0; i < length; i++) {
    auto inp_desc = TensorDesc(inp_dims, ACL_INT8, ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto weight_desc = TensorDesc(weight_dims, ACL_INT8, ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto bias_desc = TensorDesc(bias_dims, ValidList[i], ACL_FORMAT_ND).ValueRange(0, 2);
    auto scale_desc = TensorDesc(scale_dims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto offset_desc = nullptr;
    auto strides_desc = IntArrayDesc(strides_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    auto output_padding_desc = IntArrayDesc(output_padding_dims);
    int32_t offsetx = 0;
    auto roundMode = nullptr;
    auto output_desc = TensorDesc(output_dims, ACL_BF16, ACL_FORMAT_NCDHW).Precision(0.01, 0.01);

    auto ut = OP_API_UT(aclnnQuantConvolution,  // host api第二段接口名称
                        INPUT(inp_desc, weight_desc, bias_desc, scale_desc, offset_desc,
                              strides_desc, padding_desc, dilation_desc, transposed,
                              output_padding_desc, groups, offsetx, roundMode),  // host api输入
                        OUTPUT(output_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
  }
}

TEST_F(quant_convolution_test, ascend910B2_test_quantConv3d_padding_less_zero) {
  // test conv3d
  vector<aclDataType> ValidList = {ACL_BF16};
  int length = ValidList.size();

  bool transposed = false;
  const int64_t groups = 1;
  vector<int64_t> inp_dims = {2, 2, 32, 32, 32};
  vector<int64_t> weight_dims = {2, 2 / groups, 3, 3, 3};
  vector<int64_t> bias_dims = {2 / groups};
  vector<int64_t> scale_dims = {2 / groups};
  vector<int64_t> output_dims = {2, 2, 32, 32, 32};
  vector<int64_t> strides_dims = {1, 1, 1};
  vector<int64_t> padding_dims = {1, 1, -1};
  vector<int64_t> dilation_dims = {1, 1, 1};
  vector<int64_t> output_padding_dims = {0, 0, 0};

  for (int i = 0; i < length; i++) {
    auto inp_desc = TensorDesc(inp_dims, ACL_INT8, ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto weight_desc = TensorDesc(weight_dims, ACL_INT8, ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto bias_desc = TensorDesc(bias_dims, ValidList[i], ACL_FORMAT_ND).ValueRange(0, 2);
    auto scale_desc = TensorDesc(scale_dims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto offset_desc = nullptr;
    auto strides_desc = IntArrayDesc(strides_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    auto output_padding_desc = IntArrayDesc(output_padding_dims);
    int32_t offsetx = 0;
    auto roundMode = nullptr;
    auto output_desc = TensorDesc(output_dims, ACL_BF16, ACL_FORMAT_NCDHW).Precision(0.01, 0.01);

    auto ut = OP_API_UT(aclnnQuantConvolution,  // host api第二段接口名称
                        INPUT(inp_desc, weight_desc, bias_desc, scale_desc, offset_desc,
                              strides_desc, padding_desc, dilation_desc, transposed,
                              output_padding_desc, groups, offsetx, roundMode),  // host api输入
                        OUTPUT(output_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
  }
}

TEST_F(quant_convolution_test, ascend910B2_test_quantConv3d_weight_cin_wrong) {
  // test conv3d
  vector<aclDataType> ValidList = {ACL_BF16};
  int length = ValidList.size();

  bool transposed = false;
  const int64_t groups = 1;
  vector<int64_t> inp_dims = {2, 2, 32, 32, 32};
  vector<int64_t> weight_dims = {2, 1, 3, 3, 3};
  vector<int64_t> bias_dims = {2 / groups};
  vector<int64_t> scale_dims = {2 / groups};
  vector<int64_t> output_dims = {2, 2, 32, 32, 32};
  vector<int64_t> strides_dims = {1, 1, 1};
  vector<int64_t> padding_dims = {1, 1, 0};
  vector<int64_t> dilation_dims = {1, 1, 1};
  vector<int64_t> output_padding_dims = {0, 0, 0};

  for (int i = 0; i < length; i++) {
    auto inp_desc = TensorDesc(inp_dims, ACL_INT8, ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto weight_desc = TensorDesc(weight_dims, ACL_INT8, ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto bias_desc = TensorDesc(bias_dims, ValidList[i], ACL_FORMAT_ND).ValueRange(0, 2);
    auto scale_desc = TensorDesc(scale_dims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto offset_desc = nullptr;
    auto strides_desc = IntArrayDesc(strides_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    auto output_padding_desc = IntArrayDesc(output_padding_dims);
    int32_t offsetx = 0;
    auto roundMode = nullptr;
    auto output_desc = TensorDesc(output_dims, ACL_BF16, ACL_FORMAT_NCDHW).Precision(0.01, 0.01);

    auto ut = OP_API_UT(aclnnQuantConvolution,  // host api第二段接口名称
                        INPUT(inp_desc, weight_desc, bias_desc, scale_desc, offset_desc,
                              strides_desc, padding_desc, dilation_desc, transposed,
                              output_padding_desc, groups, offsetx, roundMode),  // host api输入
                        OUTPUT(output_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
  }
}

TEST_F(quant_convolution_test, ascend910B2_test_quantConv3d_incorrect_bias_dtype) {
  // test conv3d
  vector<aclDataType> ValidList = {ACL_FLOAT16};
  int length = ValidList.size();

  bool transposed = false;
  const int64_t groups = 1;
  vector<int64_t> inp_dims = {2, 2, 32, 32, 32};
  vector<int64_t> weight_dims = {2, 2 / groups, 3, 3, 3};
  vector<int64_t> bias_dims = {2 / groups};
  vector<int64_t> scale_dims = {2 / groups};
  vector<int64_t> output_dims = {2, 2, 32, 32, 32};
  vector<int64_t> strides_dims = {1, 1, 1};
  vector<int64_t> padding_dims = {1, 1, 1};
  vector<int64_t> dilation_dims = {1, 1, 1};
  vector<int64_t> output_padding_dims = {0, 0, 0};

  for (int i = 0; i < length; i++) {
    auto inp_desc = TensorDesc(inp_dims, ACL_INT8, ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto weight_desc = TensorDesc(weight_dims, ACL_INT8, ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto bias_desc = TensorDesc(bias_dims, ValidList[i], ACL_FORMAT_ND).ValueRange(0, 2);
    auto scale_desc = TensorDesc(scale_dims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto offset_desc = nullptr;
    auto strides_desc = IntArrayDesc(strides_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    auto output_padding_desc = IntArrayDesc(output_padding_dims);
    int32_t offsetx = 0;
    auto roundMode = nullptr;
    auto output_desc = TensorDesc(output_dims, ACL_BF16, ACL_FORMAT_NCDHW).Precision(0.01, 0.01);

    auto ut = OP_API_UT(aclnnQuantConvolution,  // host api第二段接口名称
                        INPUT(inp_desc, weight_desc, bias_desc, scale_desc, offset_desc,
                              strides_desc, padding_desc, dilation_desc, transposed,
                              output_padding_desc, groups, offsetx, roundMode),  // host api输入
                        OUTPUT(output_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
  }
}

TEST_F(quant_convolution_test, ascend910B2_test_quantConv3d_incorrect_stride_value) {
  // test conv3d
  vector<aclDataType> ValidList = {ACL_FLOAT16};
  int length = ValidList.size();

  bool transposed = false;
  const int64_t groups = 1;
  vector<int64_t> inp_dims = {2, 2, 32, 32, 32};
  vector<int64_t> weight_dims = {2, 2 / groups, 3, 3, 3};
  vector<int64_t> bias_dims = {2 / groups};
  vector<int64_t> scale_dims = {2 / groups};
  vector<int64_t> output_dims = {2, 2, 32, 32, 32};
  vector<int64_t> strides_dims = {0, 1, 1};
  vector<int64_t> padding_dims = {1, 1, 1};
  vector<int64_t> dilation_dims = {1, 1, 1};
  vector<int64_t> output_padding_dims = {0, 0, 0};

  for (int i = 0; i < length; i++) {
    auto inp_desc = TensorDesc(inp_dims, ACL_INT8, ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto weight_desc = TensorDesc(weight_dims, ACL_INT8, ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto bias_desc = TensorDesc(bias_dims, ValidList[i], ACL_FORMAT_ND).ValueRange(0, 2);
    auto scale_desc = TensorDesc(scale_dims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto offset_desc = nullptr;
    auto strides_desc = IntArrayDesc(strides_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    auto output_padding_desc = IntArrayDesc(output_padding_dims);
    int32_t offsetx = 0;
    auto roundMode = nullptr;
    auto output_desc = TensorDesc(output_dims, ACL_BF16, ACL_FORMAT_NCDHW).Precision(0.01, 0.01);

    auto ut = OP_API_UT(aclnnQuantConvolution,  // host api第二段接口名称
                        INPUT(inp_desc, weight_desc, bias_desc, scale_desc, offset_desc,
                              strides_desc, padding_desc, dilation_desc, transposed,
                              output_padding_desc, groups, offsetx, roundMode),  // host api输入
                        OUTPUT(output_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
  }
}

TEST_F(quant_convolution_test, ascend910B2_test_quantConv3d_input_nullptr) {
  // test conv3d
  vector<aclDataType> ValidList = {ACL_FLOAT16};
  int length = ValidList.size();

  bool transposed = false;
  const int64_t groups = 1;
  vector<int64_t> inp_dims = {2, 2, 32, 32, 32};
  vector<int64_t> weight_dims = {2, 2 / groups, 3, 3, 3};
  vector<int64_t> bias_dims = {2 / groups};
  vector<int64_t> scale_dims = {2 / groups};
  vector<int64_t> output_dims = {2, 2, 32, 32, 32};
  vector<int64_t> strides_dims = {0, 1, 1};
  vector<int64_t> padding_dims = {1, 1, 1};
  vector<int64_t> dilation_dims = {1, 1, 1};
  vector<int64_t> output_padding_dims = {0, 0, 0};

  for (int i = 0; i < length; i++) {
    auto inp_desc = nullptr;
    auto weight_desc = TensorDesc(weight_dims, ACL_INT8, ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto bias_desc = TensorDesc(bias_dims, ValidList[i], ACL_FORMAT_ND).ValueRange(0, 2);
    auto scale_desc = TensorDesc(scale_dims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto offset_desc = nullptr;
    auto strides_desc = IntArrayDesc(strides_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    auto output_padding_desc = IntArrayDesc(output_padding_dims);
    int32_t offsetx = 0;
    auto roundMode = nullptr;
    auto output_desc = TensorDesc(output_dims, ACL_BF16, ACL_FORMAT_NCDHW).Precision(0.01, 0.01);

    auto ut = OP_API_UT(aclnnQuantConvolution,  // host api第二段接口名称
                        INPUT(inp_desc, weight_desc, bias_desc, scale_desc, offset_desc,
                              strides_desc, padding_desc, dilation_desc, transposed,
                              output_padding_desc, groups, offsetx, roundMode),  // host api输入
                        OUTPUT(output_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
  }
}

TEST_F(quant_convolution_test, ascend_test_extend_conv2d_dtypes_group) {
    std::vector<vector<aclDataType>> legalDtype = {
            {aclDataType::ACL_INT8, aclDataType::ACL_INT8, aclDataType::ACL_FLOAT, aclDataType::ACL_INT32},
            {aclDataType::ACL_INT8, aclDataType::ACL_INT8, aclDataType::ACL_FLOAT16, aclDataType::ACL_INT32},
            {aclDataType::ACL_HIFLOAT8, aclDataType::ACL_HIFLOAT8, aclDataType::ACL_FLOAT, aclDataType::ACL_FLOAT},
            {aclDataType::ACL_HIFLOAT8, aclDataType::ACL_HIFLOAT8, aclDataType::ACL_FLOAT16, aclDataType::ACL_FLOAT},
            {aclDataType::ACL_HIFLOAT8, aclDataType::ACL_HIFLOAT8, aclDataType::ACL_BF16, aclDataType::ACL_FLOAT},
            {aclDataType::ACL_HIFLOAT8, aclDataType::ACL_HIFLOAT8, aclDataType::ACL_HIFLOAT8, aclDataType::ACL_FLOAT},
            {aclDataType::ACL_FLOAT8_E4M3FN, aclDataType::ACL_FLOAT8_E4M3FN, aclDataType::ACL_FLOAT, aclDataType::ACL_FLOAT},
            {aclDataType::ACL_FLOAT8_E4M3FN, aclDataType::ACL_FLOAT8_E4M3FN, aclDataType::ACL_FLOAT16, aclDataType::ACL_FLOAT},
            {aclDataType::ACL_FLOAT8_E4M3FN, aclDataType::ACL_FLOAT8_E4M3FN, aclDataType::ACL_BF16, aclDataType::ACL_FLOAT},
            {aclDataType::ACL_FLOAT8_E4M3FN, aclDataType::ACL_FLOAT8_E4M3FN, aclDataType::ACL_FLOAT8_E4M3FN, aclDataType::ACL_FLOAT}
    };
    const int64_t groups = 1;
    vector<int64_t> input_dims = {1, 1, 1, 1};
    vector<int64_t> weight_dims = {1, 1 / groups, 1, 1};
    vector<int64_t> scale_dims = {1};
    vector<int64_t> bias_dims = {1};
    vector<int64_t> output_dims = {1, 1, 1, 1};
    vector<int64_t> strides_dims = {1, 1};
    vector<int64_t> padding_dims_1 = {0, 0};
    vector<int64_t> padding_dims_2 = {0, 0, 0, 0};
    vector<int64_t> dilation_dims = {1, 1};
    vector<int64_t> output_padding_dims = {0, 0};
    auto strides_desc = IntArrayDesc(strides_dims);
    auto padding_desc_1 = IntArrayDesc(padding_dims_1);
    auto padding_desc_2 = IntArrayDesc(padding_dims_2);
    auto dilation_desc = IntArrayDesc(dilation_dims);

    int32_t offsetx = 0;
    bool transposed = false;
    auto output_padding_desc = IntArrayDesc(output_padding_dims);
    auto offset_desc = nullptr;
    for (auto dtypes : legalDtype) {
        auto input_desc = TensorDesc(input_dims, dtypes[0], ACL_FORMAT_NCHW).ValueRange(0, 2);
        auto weight_desc = TensorDesc(weight_dims, dtypes[1], ACL_FORMAT_NCHW).ValueRange(0, 2);
        auto scale_desc = TensorDesc(scale_dims, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 2);
        auto bias_desc = TensorDesc(bias_dims, dtypes[3], ACL_FORMAT_ND).ValueRange(0, 2);
        auto output_desc = TensorDesc(output_dims, dtypes[2], ACL_FORMAT_NCHW).Precision(0, 2);
        string roundMode = (dtypes[0] == aclDataType::ACL_INT8 || dtypes[0] == aclDataType::ACL_FLOAT8_E4M3FN) ? "rint" : "round";
        auto ut = OP_API_UT(aclnnQuantConvolution,  // host api第二段接口名称
                            INPUT(input_desc, weight_desc, bias_desc, scale_desc, offset_desc,
                                  strides_desc, padding_desc_1, dilation_desc, transposed,
                                  output_padding_desc, groups, offsetx, roundMode.c_str()),  // host api输入
                            OUTPUT(output_desc));
        uint64_t workspace_size = 0;
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
        // EXPECT_EQ(aclRet, ACL_SUCCESS);
    }
    for (auto dtypes : legalDtype) {
        auto input_desc = TensorDesc(input_dims, dtypes[0], ACL_FORMAT_NCHW).ValueRange(0, 2);
        auto weight_desc = TensorDesc(weight_dims, dtypes[1], ACL_FORMAT_NCHW).ValueRange(0, 2);
        auto scale_desc = TensorDesc(scale_dims, ACL_UINT64, ACL_FORMAT_ND).ValueRange(0, 2);
        auto output_desc = TensorDesc(output_dims, dtypes[2], ACL_FORMAT_NCHW).Precision(0, 2);
        string roundMode = (dtypes[0] == aclDataType::ACL_INT8 || dtypes[0] == aclDataType::ACL_FLOAT8_E4M3FN) ? "rint" : "round";
        auto ut = OP_API_UT(aclnnQuantConvolution,  // host api第二段接口名称
                            INPUT(input_desc, weight_desc, nullptr, scale_desc, offset_desc,
                                  strides_desc, padding_desc_2, dilation_desc, transposed,
                                  output_padding_desc, groups, offsetx, roundMode.c_str()),  // host api输入
                            OUTPUT(output_desc));
        uint64_t workspace_size = 0;
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
        // EXPECT_EQ(aclRet, ACL_SUCCESS);
    }
}

TEST_F(quant_convolution_test, ascend_test_quant_conv3d_dtypes_group) {
    std::vector<vector<aclDataType>> legalDtype = {
            {aclDataType::ACL_INT8, aclDataType::ACL_INT8, aclDataType::ACL_FLOAT16, aclDataType::ACL_INT32},
            {aclDataType::ACL_HIFLOAT8, aclDataType::ACL_HIFLOAT8, aclDataType::ACL_FLOAT, aclDataType::ACL_FLOAT},
            {aclDataType::ACL_HIFLOAT8, aclDataType::ACL_HIFLOAT8, aclDataType::ACL_FLOAT16, aclDataType::ACL_FLOAT},
            {aclDataType::ACL_HIFLOAT8, aclDataType::ACL_HIFLOAT8, aclDataType::ACL_BF16, aclDataType::ACL_FLOAT},
            {aclDataType::ACL_HIFLOAT8, aclDataType::ACL_HIFLOAT8, aclDataType::ACL_HIFLOAT8, aclDataType::ACL_FLOAT},
            {aclDataType::ACL_FLOAT8_E4M3FN, aclDataType::ACL_FLOAT8_E4M3FN, aclDataType::ACL_FLOAT, aclDataType::ACL_FLOAT},
            {aclDataType::ACL_FLOAT8_E4M3FN, aclDataType::ACL_FLOAT8_E4M3FN, aclDataType::ACL_FLOAT16, aclDataType::ACL_FLOAT},
            {aclDataType::ACL_FLOAT8_E4M3FN, aclDataType::ACL_FLOAT8_E4M3FN, aclDataType::ACL_BF16, aclDataType::ACL_FLOAT},
            {aclDataType::ACL_FLOAT8_E4M3FN, aclDataType::ACL_FLOAT8_E4M3FN, aclDataType::ACL_FLOAT8_E4M3FN, aclDataType::ACL_FLOAT}
    };
    const int64_t groups = 1;
    vector<int64_t> input_dims = {1, 1, 1, 1, 1};
    vector<int64_t> weight_dims = {1, 1 / groups, 1, 1, 1};
    vector<int64_t> scale_dims = {1};
    vector<int64_t> bias_dims = {1};
    vector<int64_t> output_dims = {1, 1, 1, 1, 1};
    vector<int64_t> strides_dims = {1, 1, 1};
    vector<int64_t> padding_dims_1 = {0, 0, 0};
    vector<int64_t> padding_dims_2 = {0, 0, 0, 0, 0, 0};
    vector<int64_t> dilation_dims = {1, 1, 1};
    vector<int64_t> output_padding_dims = {0, 0, 0};
    auto strides_desc = IntArrayDesc(strides_dims);
    auto padding_desc_1 = IntArrayDesc(padding_dims_1);
    auto padding_desc_2 = IntArrayDesc(padding_dims_2);
    auto dilation_desc = IntArrayDesc(dilation_dims);

    int32_t offsetx = 0;
    bool transposed = false;
    auto output_padding_desc = IntArrayDesc(output_padding_dims);
    auto offset_desc = nullptr;
    for (auto dtypes : legalDtype) {
        auto input_desc = TensorDesc(input_dims, dtypes[0], ACL_FORMAT_NCDHW).ValueRange(0, 2);
        auto weight_desc = TensorDesc(weight_dims, dtypes[1], ACL_FORMAT_NCDHW).ValueRange(0, 2);
        auto scale_desc = TensorDesc(scale_dims, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 2);
        auto bias_desc = TensorDesc(bias_dims, dtypes[3], ACL_FORMAT_ND).ValueRange(0, 2);
        auto output_desc = TensorDesc(output_dims, dtypes[2], ACL_FORMAT_NCDHW).Precision(0, 2);
        string roundMode = (dtypes[0] == aclDataType::ACL_INT8 || dtypes[0] == aclDataType::ACL_FLOAT8_E4M3FN) ? "rint" : "round";
        auto ut = OP_API_UT(aclnnQuantConvolution,  // host api第二段接口名称
                            INPUT(input_desc, weight_desc, bias_desc, scale_desc, offset_desc,
                                  strides_desc, padding_desc_1, dilation_desc, transposed,
                                  output_padding_desc, groups, offsetx, roundMode.c_str()),  // host api输入
                            OUTPUT(output_desc));
        uint64_t workspace_size = 0;
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
        // EXPECT_EQ(aclRet, ACL_SUCCESS);
    }
    for (auto dtypes : legalDtype) {
        auto input_desc = TensorDesc(input_dims, dtypes[0], ACL_FORMAT_NCDHW).ValueRange(0, 2);
        auto weight_desc = TensorDesc(weight_dims, dtypes[1], ACL_FORMAT_NCDHW).ValueRange(0, 2);
        auto scale_desc = TensorDesc(scale_dims, ACL_UINT64, ACL_FORMAT_ND).ValueRange(0, 2);
        auto output_desc = TensorDesc(output_dims, dtypes[2], ACL_FORMAT_NCDHW).Precision(0, 2);
        string roundMode = (dtypes[0] == aclDataType::ACL_INT8 || dtypes[0] == aclDataType::ACL_FLOAT8_E4M3FN) ? "rint" : "round";
        auto ut = OP_API_UT(aclnnQuantConvolution,  // host api第二段接口名称
                            INPUT(input_desc, weight_desc, nullptr, scale_desc, offset_desc,
                                  strides_desc, padding_desc_2, dilation_desc, transposed,
                                  output_padding_desc, groups, offsetx, roundMode.c_str()),  // host api输入
                            OUTPUT(output_desc));
        uint64_t workspace_size = 0;
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
        // EXPECT_EQ(aclRet, ACL_SUCCESS);
    }
}

TEST_F(quant_convolution_test, ascend_test_extend_conv2d_scaledtype_check) {
    const int64_t groups = 1;
    vector<int64_t> input_dims = {1, 1, 1, 1};
    vector<int64_t> weight_dims = {1, 1 / groups, 1, 1};
    vector<int64_t> scale_dims = {1};
    vector<int64_t> bias_dims = {1};
    vector<int64_t> output_dims = {1, 1, 1, 1};
    vector<int64_t> strides_dims = {1, 1};
    vector<int64_t> padding_dims_2 = {0, 0, 0, 0};
    vector<int64_t> dilation_dims = {1, 1};
    vector<int64_t> output_padding_dims = {0, 0};
    auto strides_desc = IntArrayDesc(strides_dims);
    auto padding_desc_2 = IntArrayDesc(padding_dims_2);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    int32_t offsetx = 0;
    bool transposed = false;
    auto output_padding_desc = IntArrayDesc(output_padding_dims);
    auto offset_desc = nullptr;
    auto input_desc = TensorDesc(input_dims, aclDataType::ACL_INT8, ACL_FORMAT_NCHW).ValueRange(0, 2);
    auto weight_desc = TensorDesc(weight_dims, aclDataType::ACL_INT8, ACL_FORMAT_NCHW).ValueRange(0, 2);
    auto scale_desc = TensorDesc(scale_dims, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 2);
    auto bias_desc = TensorDesc(bias_dims, aclDataType::ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 2);
    auto output_desc = TensorDesc(output_dims, aclDataType::ACL_FLOAT16, ACL_FORMAT_NCHW).Precision(0, 2);
    string roundMode = "rint";
    auto ut = OP_API_UT(aclnnQuantConvolution,  // host api第二段接口名称
                        INPUT(input_desc, weight_desc, bias_desc, scale_desc, offset_desc,
                              strides_desc, padding_desc_2, dilation_desc, transposed,
                              output_padding_desc, groups, offsetx, roundMode.c_str()),  // host api输入
                        OUTPUT(output_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
    // EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(quant_convolution_test, ascend_test_extend_conv2d_inputdim_check) {
    const int64_t groups = 1;
    vector<int64_t> input_dims = {1, 1, 1};
    vector<int64_t> weight_dims = {1, 1 / groups, 1, 1};
    vector<int64_t> scale_dims = {1};
    vector<int64_t> bias_dims = {1};
    vector<int64_t> output_dims = {1, 1, 1, 1};
    vector<int64_t> strides_dims = {1, 1};
    vector<int64_t> padding_dims_1 = {0, 0};
    vector<int64_t> padding_dims_2 = {0, 0, 0, 0};
    vector<int64_t> dilation_dims = {1, 1};
    vector<int64_t> output_padding_dims = {0, 0};
    auto strides_desc = IntArrayDesc(strides_dims);
    auto padding_desc_1 = IntArrayDesc(padding_dims_1);
    auto padding_desc_2 = IntArrayDesc(padding_dims_2);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    int32_t offsetx = 0;
    bool transposed = false;
    auto output_padding_desc = IntArrayDesc(output_padding_dims);
    auto offset_desc = nullptr;
    auto input_desc = TensorDesc(input_dims, aclDataType::ACL_INT8, ACL_FORMAT_NCHW).ValueRange(0, 2);
    auto weight_desc = TensorDesc(weight_dims, aclDataType::ACL_INT8, ACL_FORMAT_NCHW).ValueRange(0, 2);
    auto scale_desc = TensorDesc(scale_dims, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 2);
    auto bias_desc = TensorDesc(bias_dims, aclDataType::ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 2);
    auto output_desc = TensorDesc(output_dims, aclDataType::ACL_FLOAT16, ACL_FORMAT_NCHW).Precision(0, 2);
    string roundMode = "rint";
    auto ut = OP_API_UT(aclnnQuantConvolution,  // host api第二段接口名称
                        INPUT(input_desc, weight_desc, bias_desc, scale_desc, offset_desc,
                              strides_desc, padding_desc_1, dilation_desc, transposed,
                              output_padding_desc, groups, offsetx, roundMode.c_str()),  // host api输入
                        OUTPUT(output_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
    // EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(quant_convolution_test, ascend_test_extend_conv2d_weightdim_check) {
    const int64_t groups = 1;
    vector<int64_t> input_dims = {1, 1, 1, 1};
    vector<int64_t> weight_dims = {1, 1, 1};
    vector<int64_t> scale_dims = {1};
    vector<int64_t> bias_dims = {1};
    vector<int64_t> output_dims = {1, 1, 1, 1};
    vector<int64_t> strides_dims = {1, 1};
    vector<int64_t> padding_dims_1 = {0, 0};
    vector<int64_t> padding_dims_2 = {0, 0, 0, 0};
    vector<int64_t> dilation_dims = {1, 1};
    vector<int64_t> output_padding_dims = {0, 0};
    auto strides_desc = IntArrayDesc(strides_dims);
    auto padding_desc_1 = IntArrayDesc(padding_dims_1);
    auto padding_desc_2 = IntArrayDesc(padding_dims_2);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    int32_t offsetx = 0;
    bool transposed = false;
    auto output_padding_desc = IntArrayDesc(output_padding_dims);
    auto offset_desc = nullptr;
    auto input_desc = TensorDesc(input_dims, aclDataType::ACL_INT8, ACL_FORMAT_NCHW).ValueRange(0, 2);
    auto weight_desc = TensorDesc(weight_dims, aclDataType::ACL_INT8, ACL_FORMAT_NCHW).ValueRange(0, 2);
    auto scale_desc = TensorDesc(scale_dims, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 2);
    auto bias_desc = TensorDesc(bias_dims, aclDataType::ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 2);
    auto output_desc = TensorDesc(output_dims, aclDataType::ACL_FLOAT16, ACL_FORMAT_NCHW).Precision(0, 2);
    string roundMode = "rint";
    auto ut = OP_API_UT(aclnnQuantConvolution,  // host api第二段接口名称
                        INPUT(input_desc, weight_desc, bias_desc, scale_desc, offset_desc,
                              strides_desc, padding_desc_1, dilation_desc, transposed,
                              output_padding_desc, groups, offsetx, roundMode.c_str()),  // host api输入
                        OUTPUT(output_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
    // EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(quant_convolution_test, ascend_test_extend_conv2d_outputdim_check) {
    const int64_t groups = 1;
    vector<int64_t> input_dims = {1, 1, 1, 1};
    vector<int64_t> weight_dims = {1, 1 / groups, 1, 1};
    vector<int64_t> scale_dims = {1};
    vector<int64_t> bias_dims = {1};
    vector<int64_t> output_dims = {1, 1, 1};
    vector<int64_t> strides_dims = {1, 1};
    vector<int64_t> padding_dims_1 = {0, 0};
    vector<int64_t> padding_dims_2 = {0, 0, 0, 0};
    vector<int64_t> dilation_dims = {1, 1};
    vector<int64_t> output_padding_dims = {0, 0};
    auto strides_desc = IntArrayDesc(strides_dims);
    auto padding_desc_1 = IntArrayDesc(padding_dims_1);
    auto padding_desc_2 = IntArrayDesc(padding_dims_2);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    int32_t offsetx = 0;
    bool transposed = false;
    auto output_padding_desc = IntArrayDesc(output_padding_dims);
    auto offset_desc = nullptr;
    auto input_desc = TensorDesc(input_dims, aclDataType::ACL_INT8, ACL_FORMAT_NCHW).ValueRange(0, 2);
    auto weight_desc = TensorDesc(weight_dims, aclDataType::ACL_INT8, ACL_FORMAT_NCHW).ValueRange(0, 2);
    auto scale_desc = TensorDesc(scale_dims, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 2);
    auto bias_desc = TensorDesc(bias_dims, aclDataType::ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 2);
    auto output_desc = TensorDesc(output_dims, aclDataType::ACL_FLOAT16, ACL_FORMAT_NCHW).Precision(0, 2);
    string roundMode = "rint";
    auto ut = OP_API_UT(aclnnQuantConvolution,  // host api第二段接口名称
                        INPUT(input_desc, weight_desc, bias_desc, scale_desc, offset_desc,
                              strides_desc, padding_desc_1, dilation_desc, transposed,
                              output_padding_desc, groups, offsetx, roundMode.c_str()),  // host api输入
                        OUTPUT(output_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
    // EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(quant_convolution_test, ascend_test_extend_conv2d_dimsame_check) {
    const int64_t groups = 1;
    vector<int64_t> input_dims = {1, 1, 1, 1, 1};
    vector<int64_t> weight_dims = {1, 1 / groups, 1, 1};
    vector<int64_t> scale_dims = {1};
    vector<int64_t> bias_dims = {1};
    vector<int64_t> output_dims = {1, 1, 1, 1};
    vector<int64_t> strides_dims = {1, 1};
    vector<int64_t> padding_dims_1 = {0, 0};
    vector<int64_t> padding_dims_2 = {0, 0, 0, 0};
    vector<int64_t> dilation_dims = {1, 1};
    vector<int64_t> output_padding_dims = {0, 0};
    auto strides_desc = IntArrayDesc(strides_dims);
    auto padding_desc_1 = IntArrayDesc(padding_dims_1);
    auto padding_desc_2 = IntArrayDesc(padding_dims_2);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    int32_t offsetx = 0;
    bool transposed = false;
    auto output_padding_desc = IntArrayDesc(output_padding_dims);
    auto offset_desc = nullptr;
    auto input_desc = TensorDesc(input_dims, aclDataType::ACL_INT8, ACL_FORMAT_NCHW).ValueRange(0, 2);
    auto weight_desc = TensorDesc(weight_dims, aclDataType::ACL_INT8, ACL_FORMAT_NCHW).ValueRange(0, 2);
    auto scale_desc = TensorDesc(scale_dims, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 2);
    auto bias_desc = TensorDesc(bias_dims, aclDataType::ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 2);
    auto output_desc = TensorDesc(output_dims, aclDataType::ACL_FLOAT16, ACL_FORMAT_NCHW).Precision(0, 2);
    string roundMode = "rint";
    auto ut = OP_API_UT(aclnnQuantConvolution,  // host api第二段接口名称
                        INPUT(input_desc, weight_desc, bias_desc, scale_desc, offset_desc,
                              strides_desc, padding_desc_1, dilation_desc, transposed,
                              output_padding_desc, groups, offsetx, roundMode.c_str()),  // host api输入
                        OUTPUT(output_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
    // EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(quant_convolution_test, ascend_test_extend_conv2d_offsetx_check2) {
    const int64_t groups = 1;
    vector<int64_t> input_dims = {1, 1, 1, 1};
    vector<int64_t> weight_dims = {1, 1 / groups, 1, 1};
    vector<int64_t> scale_dims = {1};
    vector<int64_t> bias_dims = {1};
    vector<int64_t> output_dims = {1, 1, 1, 1};
    vector<int64_t> strides_dims = {1, 1};
    vector<int64_t> padding_dims_2 = {0, 0, 0, 0};
    vector<int64_t> dilation_dims = {1, 1};
    vector<int64_t> output_padding_dims = {0, 0};
    auto strides_desc = IntArrayDesc(strides_dims);
    auto padding_desc_2 = IntArrayDesc(padding_dims_2);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    int32_t offsetx = -1;
    bool transposed = false;
    auto output_padding_desc = IntArrayDesc(output_padding_dims);
    auto offset_desc = nullptr;
    auto input_desc = TensorDesc(input_dims, aclDataType::ACL_HIFLOAT8, ACL_FORMAT_NCHW).ValueRange(0, 2);
    auto weight_desc = TensorDesc(weight_dims, aclDataType::ACL_HIFLOAT8, ACL_FORMAT_NCHW).ValueRange(0, 2);
    auto scale_desc = TensorDesc(scale_dims, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 2);
    auto bias_desc = TensorDesc(bias_dims, aclDataType::ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto output_desc = TensorDesc(output_dims, aclDataType::ACL_FLOAT, ACL_FORMAT_NCHW).Precision(0, 2);
    string roundMode = "rint";
    auto ut = OP_API_UT(aclnnQuantConvolution,  // host api第二段接口名称
                        INPUT(input_desc, weight_desc, bias_desc, scale_desc, offset_desc,
                              strides_desc, padding_desc_2, dilation_desc, transposed,
                              output_padding_desc, groups, offsetx, roundMode.c_str()),  // host api输入
                        OUTPUT(output_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
    // EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(quant_convolution_test, ascend_test_extend_conv2d_scaledim_check) {
    const int64_t groups = 1;
    vector<int64_t> input_dims = {1, 1, 1, 1};
    vector<int64_t> weight_dims = {1, 1 / groups, 1, 1};
    vector<int64_t> scale_dims = {1, 1};
    vector<int64_t> bias_dims = {1};
    vector<int64_t> output_dims = {1, 1, 1, 1};
    vector<int64_t> strides_dims = {1, 1};
    vector<int64_t> padding_dims_1 = {0, 0};
    vector<int64_t> padding_dims_2 = {0, 0, 0, 0};
    vector<int64_t> dilation_dims = {1, 1};
    vector<int64_t> output_padding_dims = {0, 0};
    auto strides_desc = IntArrayDesc(strides_dims);
    auto padding_desc_1 = IntArrayDesc(padding_dims_1);
    auto padding_desc_2 = IntArrayDesc(padding_dims_2);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    int32_t offsetx = 0;
    bool transposed = false;
    auto output_padding_desc = IntArrayDesc(output_padding_dims);
    auto offset_desc = nullptr;
    auto input_desc = TensorDesc(input_dims, aclDataType::ACL_INT8, ACL_FORMAT_NCHW).ValueRange(0, 2);
    auto weight_desc = TensorDesc(weight_dims, aclDataType::ACL_INT8, ACL_FORMAT_NCHW).ValueRange(0, 2);
    auto scale_desc = TensorDesc(scale_dims, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 2);
    auto bias_desc = TensorDesc(bias_dims, aclDataType::ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 2);
    auto output_desc = TensorDesc(output_dims, aclDataType::ACL_FLOAT16, ACL_FORMAT_NCHW).Precision(0, 2);
    string roundMode = "rint";
    auto ut = OP_API_UT(aclnnQuantConvolution,  // host api第二段接口名称
                        INPUT(input_desc, weight_desc, bias_desc, scale_desc, offset_desc,
                              strides_desc, padding_desc_1, dilation_desc, transposed,
                              output_padding_desc, groups, offsetx, roundMode.c_str()),  // host api输入
                        OUTPUT(output_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
    // EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(quant_convolution_test, ascend_test_extend_conv2d_biasdim_check) {
    const int64_t groups = 1;
    vector<int64_t> input_dims = {1, 1, 1, 1};
    vector<int64_t> weight_dims = {1, 1 / groups, 1, 1};
    vector<int64_t> scale_dims = {1};
    vector<int64_t> bias_dims = {1, 1};
    vector<int64_t> output_dims = {1, 1, 1, 1};
    vector<int64_t> strides_dims = {1, 1};
    vector<int64_t> padding_dims_1 = {0, 0};
    vector<int64_t> padding_dims_2 = {0, 0, 0, 0};
    vector<int64_t> dilation_dims = {1, 1};
    vector<int64_t> output_padding_dims = {0, 0};
    auto strides_desc = IntArrayDesc(strides_dims);
    auto padding_desc_1 = IntArrayDesc(padding_dims_1);
    auto padding_desc_2 = IntArrayDesc(padding_dims_2);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    int32_t offsetx = 0;
    bool transposed = false;
    auto output_padding_desc = IntArrayDesc(output_padding_dims);
    auto offset_desc = nullptr;
    auto input_desc = TensorDesc(input_dims, aclDataType::ACL_INT8, ACL_FORMAT_NCHW).ValueRange(0, 2);
    auto weight_desc = TensorDesc(weight_dims, aclDataType::ACL_INT8, ACL_FORMAT_NCHW).ValueRange(0, 2);
    auto scale_desc = TensorDesc(scale_dims, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 2);
    auto bias_desc = TensorDesc(bias_dims, aclDataType::ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 2);
    auto output_desc = TensorDesc(output_dims, aclDataType::ACL_FLOAT16, ACL_FORMAT_NCHW).Precision(0, 2);
    string roundMode = "rint";
    auto ut = OP_API_UT(aclnnQuantConvolution,  // host api第二段接口名称
                        INPUT(input_desc, weight_desc, bias_desc, scale_desc, offset_desc,
                              strides_desc, padding_desc_1, dilation_desc, transposed,
                              output_padding_desc, groups, offsetx, roundMode.c_str()),  // host api输入
                        OUTPUT(output_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
    // EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(quant_convolution_test, ascend_test_extend_conv2d_stridedim_check) {
    const int64_t groups = 1;
    vector<int64_t> input_dims = {1, 1, 1, 1};
    vector<int64_t> weight_dims = {1, 1 / groups, 1, 1};
    vector<int64_t> scale_dims = {1};
    vector<int64_t> bias_dims = {1};
    vector<int64_t> output_dims = {1, 1, 1, 1};
    vector<int64_t> strides_dims = {1, 1, 1};
    vector<int64_t> padding_dims_1 = {0, 0};
    vector<int64_t> padding_dims_2 = {0, 0, 0, 0};
    vector<int64_t> dilation_dims = {1, 1};
    vector<int64_t> output_padding_dims = {0, 0};
    auto strides_desc = IntArrayDesc(strides_dims);
    auto padding_desc_1 = IntArrayDesc(padding_dims_1);
    auto padding_desc_2 = IntArrayDesc(padding_dims_2);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    int32_t offsetx = 0;
    bool transposed = false;
    auto output_padding_desc = IntArrayDesc(output_padding_dims);
    auto offset_desc = nullptr;
    auto input_desc = TensorDesc(input_dims, aclDataType::ACL_INT8, ACL_FORMAT_NCHW).ValueRange(0, 2);
    auto weight_desc = TensorDesc(weight_dims, aclDataType::ACL_INT8, ACL_FORMAT_NCHW).ValueRange(0, 2);
    auto scale_desc = TensorDesc(scale_dims, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 2);
    auto bias_desc = TensorDesc(bias_dims, aclDataType::ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 2);
    auto output_desc = TensorDesc(output_dims, aclDataType::ACL_FLOAT16, ACL_FORMAT_NCHW).Precision(0, 2);
    string roundMode = "rint";
    auto ut = OP_API_UT(aclnnQuantConvolution,  // host api第二段接口名称
                        INPUT(input_desc, weight_desc, bias_desc, scale_desc, offset_desc,
                              strides_desc, padding_desc_1, dilation_desc, transposed,
                              output_padding_desc, groups, offsetx, roundMode.c_str()),  // host api输入
                        OUTPUT(output_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
    // EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(quant_convolution_test, ascend_test_extend_conv2d_dilationdim_check) {
    const int64_t groups = 1;
    vector<int64_t> input_dims = {1, 1, 1, 1};
    vector<int64_t> weight_dims = {1, 1 / groups, 1, 1};
    vector<int64_t> scale_dims = {1};
    vector<int64_t> bias_dims = {1};
    vector<int64_t> output_dims = {1, 1, 1, 1};
    vector<int64_t> strides_dims = {1, 1};
    vector<int64_t> padding_dims_1 = {0, 0};
    vector<int64_t> padding_dims_2 = {0, 0, 0, 0};
    vector<int64_t> dilation_dims = {1, 1, 1};
    vector<int64_t> output_padding_dims = {0, 0};
    auto strides_desc = IntArrayDesc(strides_dims);
    auto padding_desc_1 = IntArrayDesc(padding_dims_1);
    auto padding_desc_2 = IntArrayDesc(padding_dims_2);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    int32_t offsetx = 0;
    bool transposed = false;
    auto output_padding_desc = IntArrayDesc(output_padding_dims);
    auto offset_desc = nullptr;
    auto input_desc = TensorDesc(input_dims, aclDataType::ACL_INT8, ACL_FORMAT_NCHW).ValueRange(0, 2);
    auto weight_desc = TensorDesc(weight_dims, aclDataType::ACL_INT8, ACL_FORMAT_NCHW).ValueRange(0, 2);
    auto scale_desc = TensorDesc(scale_dims, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 2);
    auto bias_desc = TensorDesc(bias_dims, aclDataType::ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 2);
    auto output_desc = TensorDesc(output_dims, aclDataType::ACL_FLOAT16, ACL_FORMAT_NCHW).Precision(0, 2);
    string roundMode = "rint";
    auto ut = OP_API_UT(aclnnQuantConvolution,  // host api第二段接口名称
                        INPUT(input_desc, weight_desc, bias_desc, scale_desc, offset_desc,
                              strides_desc, padding_desc_1, dilation_desc, transposed,
                              output_padding_desc, groups, offsetx, roundMode.c_str()),  // host api输入
                        OUTPUT(output_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
    // EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(quant_convolution_test, ascend_test_extend_conv2d_paddim_check) {
    const int64_t groups = 1;
    vector<int64_t> input_dims = {1, 1, 1, 1};
    vector<int64_t> weight_dims = {1, 1 / groups, 1, 1};
    vector<int64_t> scale_dims = {1};
    vector<int64_t> bias_dims = {1};
    vector<int64_t> output_dims = {1, 1, 1, 1};
    vector<int64_t> strides_dims = {1, 1};
    vector<int64_t> padding_dims_2 = {0, 0, 0, 0, 0, 0};
    vector<int64_t> dilation_dims = {1, 1};
    vector<int64_t> output_padding_dims = {0, 0};
    auto strides_desc = IntArrayDesc(strides_dims);
    auto padding_desc_2 = IntArrayDesc(padding_dims_2);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    int32_t offsetx = 0;
    bool transposed = false;
    auto output_padding_desc = IntArrayDesc(output_padding_dims);
    auto offset_desc = nullptr;
    auto input_desc = TensorDesc(input_dims, aclDataType::ACL_INT8, ACL_FORMAT_NCHW).ValueRange(0, 2);
    auto weight_desc = TensorDesc(weight_dims, aclDataType::ACL_INT8, ACL_FORMAT_NCHW).ValueRange(0, 2);
    auto scale_desc = TensorDesc(scale_dims, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 2);
    auto bias_desc = TensorDesc(bias_dims, aclDataType::ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 2);
    auto output_desc = TensorDesc(output_dims, aclDataType::ACL_FLOAT16, ACL_FORMAT_NCHW).Precision(0, 2);
    string roundMode = "rint";
    auto ut = OP_API_UT(aclnnQuantConvolution,  // host api第二段接口名称
                        INPUT(input_desc, weight_desc, bias_desc, scale_desc, offset_desc,
                              strides_desc, padding_desc_2, dilation_desc, transposed,
                              output_padding_desc, groups, offsetx, roundMode.c_str()),  // host api输入
                        OUTPUT(output_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
    // EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(quant_convolution_test, ascend_test_quant_conv3d_paddim_check) {
    const int64_t groups = 1;
    vector<int64_t> input_dims = {1, 1, 1, 1, 1};
    vector<int64_t> weight_dims = {1, 1 / groups, 1, 1, 1};
    vector<int64_t> scale_dims = {1};
    vector<int64_t> bias_dims = {1};
    vector<int64_t> output_dims = {1, 1, 1, 1, 1};
    vector<int64_t> strides_dims = {1, 1, 1};
    vector<int64_t> padding_dims_2 = {0, 0};
    vector<int64_t> dilation_dims = {1, 1, 1};
    vector<int64_t> output_padding_dims = {0, 0, 0};
    auto strides_desc = IntArrayDesc(strides_dims);
    auto padding_desc_2 = IntArrayDesc(padding_dims_2);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    int32_t offsetx = 0;
    bool transposed = false;
    auto output_padding_desc = IntArrayDesc(output_padding_dims);
    auto offset_desc = nullptr;
    auto input_desc = TensorDesc(input_dims, aclDataType::ACL_INT8, ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto weight_desc = TensorDesc(weight_dims, aclDataType::ACL_INT8, ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto scale_desc = TensorDesc(scale_dims, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 2);
    auto bias_desc = TensorDesc(bias_dims, aclDataType::ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 2);
    auto output_desc = TensorDesc(output_dims, aclDataType::ACL_FLOAT16, ACL_FORMAT_NCDHW).Precision(0, 2);
    string roundMode = "rint";
    auto ut = OP_API_UT(aclnnQuantConvolution,  // host api第二段接口名称
                        INPUT(input_desc, weight_desc, bias_desc, scale_desc, offset_desc,
                              strides_desc, padding_desc_2, dilation_desc, transposed,
                              output_padding_desc, groups, offsetx, roundMode.c_str()),  // host api输入
                        OUTPUT(output_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
    // EXPECT_EQ(aclRet, ACL_SUCCESS);

}

TEST_F(quant_convolution_test, ascend_test_extend_conv2d_input_format_check) {
    const int64_t groups = 1;
    vector<int64_t> input_dims = {1, 1, 1, 1};
    vector<int64_t> weight_dims = {1, 1 / groups, 1, 1};
    vector<int64_t> scale_dims = {1};
    vector<int64_t> bias_dims = {1};
    vector<int64_t> output_dims = {1, 1, 1, 1};
    vector<int64_t> strides_dims = {1, 1};
    vector<int64_t> padding_dims_2 = {0, 0, 0, 0};
    vector<int64_t> dilation_dims = {1, 1};
    vector<int64_t> output_padding_dims = {0, 0};
    auto strides_desc = IntArrayDesc(strides_dims);
    auto padding_desc_2 = IntArrayDesc(padding_dims_2);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    int32_t offsetx = 0;
    bool transposed = false;
    auto output_padding_desc = IntArrayDesc(output_padding_dims);
    auto offset_desc = nullptr;
    auto input_desc = TensorDesc(input_dims, aclDataType::ACL_INT8, ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto weight_desc = TensorDesc(weight_dims, aclDataType::ACL_INT8, ACL_FORMAT_NCHW).ValueRange(0, 2);
    auto scale_desc = TensorDesc(scale_dims, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 2);
    auto bias_desc = TensorDesc(bias_dims, aclDataType::ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 2);
    auto output_desc = TensorDesc(output_dims, aclDataType::ACL_FLOAT16, ACL_FORMAT_NCHW).Precision(0, 2);
    string roundMode = "rint";
    auto ut = OP_API_UT(aclnnQuantConvolution,  // host api第二段接口名称
                        INPUT(input_desc, weight_desc, bias_desc, scale_desc, offset_desc,
                              strides_desc, padding_desc_2, dilation_desc, transposed,
                              output_padding_desc, groups, offsetx, roundMode.c_str()),  // host api输入
                        OUTPUT(output_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
    // EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(quant_convolution_test, ascend_test_extend_conv2d_scale_format_check) {
    const int64_t groups = 1;
    vector<int64_t> input_dims = {1, 1, 1, 1};
    vector<int64_t> weight_dims = {1, 1 / groups, 1, 1};
    vector<int64_t> scale_dims = {1};
    vector<int64_t> bias_dims = {1};
    vector<int64_t> output_dims = {1, 1, 1, 1};
    vector<int64_t> strides_dims = {1, 1};
    vector<int64_t> padding_dims_2 = {0, 0, 0, 0};
    vector<int64_t> dilation_dims = {1, 1};
    vector<int64_t> output_padding_dims = {0, 0};
    auto strides_desc = IntArrayDesc(strides_dims);
    auto padding_desc_2 = IntArrayDesc(padding_dims_2);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    int32_t offsetx = 0;
    bool transposed = false;
    auto output_padding_desc = IntArrayDesc(output_padding_dims);
    auto offset_desc = nullptr;
    auto input_desc = TensorDesc(input_dims, aclDataType::ACL_INT8, ACL_FORMAT_NCHW).ValueRange(0, 2);
    auto weight_desc = TensorDesc(weight_dims, aclDataType::ACL_INT8, ACL_FORMAT_NCHW).ValueRange(0, 2);
    auto scale_desc = TensorDesc(scale_dims, ACL_INT64, ACL_FORMAT_NCHW).ValueRange(0, 2);
    auto bias_desc = TensorDesc(bias_dims, aclDataType::ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 2);
    auto output_desc = TensorDesc(output_dims, aclDataType::ACL_FLOAT16, ACL_FORMAT_NCHW).Precision(0, 2);
    string roundMode = "rint";
    auto ut = OP_API_UT(aclnnQuantConvolution,  // host api第二段接口名称
                        INPUT(input_desc, weight_desc, bias_desc, scale_desc, offset_desc,
                              strides_desc, padding_desc_2, dilation_desc, transposed,
                              output_padding_desc, groups, offsetx, roundMode.c_str()),  // host api输入
                        OUTPUT(output_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
    // EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(quant_convolution_test, ascend_test_extend_conv2d_bias_format_check) {
    const int64_t groups = 1;
    vector<int64_t> input_dims = {1, 1, 1, 1};
    vector<int64_t> weight_dims = {1, 1 / groups, 1, 1};
    vector<int64_t> scale_dims = {1};
    vector<int64_t> bias_dims = {1};
    vector<int64_t> output_dims = {1, 1, 1, 1};
    vector<int64_t> strides_dims = {1, 1};
    vector<int64_t> padding_dims_2 = {0, 0, 0, 0};
    vector<int64_t> dilation_dims = {1, 1};
    vector<int64_t> output_padding_dims = {0, 0};
    auto strides_desc = IntArrayDesc(strides_dims);
    auto padding_desc_2 = IntArrayDesc(padding_dims_2);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    int32_t offsetx = 0;
    bool transposed = false;
    auto output_padding_desc = IntArrayDesc(output_padding_dims);
    auto offset_desc = nullptr;
    auto input_desc = TensorDesc(input_dims, aclDataType::ACL_INT8, ACL_FORMAT_NCHW).ValueRange(0, 2);
    auto weight_desc = TensorDesc(weight_dims, aclDataType::ACL_INT8, ACL_FORMAT_NCHW).ValueRange(0, 2);
    auto scale_desc = TensorDesc(scale_dims, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 2);
    auto bias_desc = TensorDesc(bias_dims, aclDataType::ACL_INT32, ACL_FORMAT_NCHW).ValueRange(0, 2);
    auto output_desc = TensorDesc(output_dims, aclDataType::ACL_FLOAT16, ACL_FORMAT_NCHW).Precision(0, 2);
    string roundMode = "rint";
    auto ut = OP_API_UT(aclnnQuantConvolution,  // host api第二段接口名称
                        INPUT(input_desc, weight_desc, bias_desc, scale_desc, offset_desc,
                              strides_desc, padding_desc_2, dilation_desc, transposed,
                              output_padding_desc, groups, offsetx, roundMode.c_str()),  // host api输入
                        OUTPUT(output_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
    // EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(quant_convolution_test, ascend_test_extend_conv2d_roundmode_check1) {
    const int64_t groups = 1;
    vector<int64_t> input_dims = {1, 1, 1, 1};
    vector<int64_t> weight_dims = {1, 1 / groups, 1, 1};
    vector<int64_t> scale_dims = {1};
    vector<int64_t> bias_dims = {1};
    vector<int64_t> output_dims = {1, 1, 1, 1};
    vector<int64_t> strides_dims = {1, 1};
    vector<int64_t> padding_dims_2 = {0, 0, 0, 0};
    vector<int64_t> dilation_dims = {1, 1};
    vector<int64_t> output_padding_dims = {0, 0};
    auto strides_desc = IntArrayDesc(strides_dims);
    auto padding_desc_2 = IntArrayDesc(padding_dims_2);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    int32_t offsetx = 0;
    bool transposed = false;
    auto output_padding_desc = IntArrayDesc(output_padding_dims);
    auto offset_desc = nullptr;
    auto input_desc = TensorDesc(input_dims, aclDataType::ACL_FLOAT8_E4M3FN, ACL_FORMAT_NCHW).ValueRange(0, 2);
    auto weight_desc = TensorDesc(weight_dims, aclDataType::ACL_FLOAT8_E4M3FN, ACL_FORMAT_NCHW).ValueRange(0, 2);
    auto scale_desc = TensorDesc(scale_dims, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 2);
    auto bias_desc = TensorDesc(bias_dims, aclDataType::ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto output_desc = TensorDesc(output_dims, aclDataType::ACL_FLOAT8_E4M3FN, ACL_FORMAT_NCHW).Precision(0, 2);
    string roundMode = "round";
    auto ut = OP_API_UT(aclnnQuantConvolution,  // host api第二段接口名称
                        INPUT(input_desc, weight_desc, bias_desc, scale_desc, offset_desc,
                              strides_desc, padding_desc_2, dilation_desc, transposed,
                              output_padding_desc, groups, offsetx, roundMode.c_str()),  // host api输入
                        OUTPUT(output_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
    // EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(quant_convolution_test, ascend_test_extend_conv2d_roundmode_check2) {
    const int64_t groups = 1;
    vector<int64_t> input_dims = {1, 1, 1, 1};
    vector<int64_t> weight_dims = {1, 1 / groups, 1, 1};
    vector<int64_t> scale_dims = {1};
    vector<int64_t> bias_dims = {1};
    vector<int64_t> output_dims = {1, 1, 1, 1};
    vector<int64_t> strides_dims = {1, 1};
    vector<int64_t> padding_dims_2 = {0, 0, 0, 0};
    vector<int64_t> dilation_dims = {1, 1};
    vector<int64_t> output_padding_dims = {0, 0};
    auto strides_desc = IntArrayDesc(strides_dims);
    auto padding_desc_2 = IntArrayDesc(padding_dims_2);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    int32_t offsetx = 0;
    bool transposed = false;
    auto output_padding_desc = IntArrayDesc(output_padding_dims);
    auto offset_desc = nullptr;
    auto input_desc = TensorDesc(input_dims, aclDataType::ACL_HIFLOAT8, ACL_FORMAT_NCHW).ValueRange(0, 2);
    auto weight_desc = TensorDesc(weight_dims, aclDataType::ACL_HIFLOAT8, ACL_FORMAT_NCHW).ValueRange(0, 2);
    auto scale_desc = TensorDesc(scale_dims, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 2);
    auto bias_desc = TensorDesc(bias_dims, aclDataType::ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto output_desc = TensorDesc(output_dims, aclDataType::ACL_HIFLOAT8, ACL_FORMAT_NCHW).Precision(0, 2);
    string roundMode = "rint";
    auto ut = OP_API_UT(aclnnQuantConvolution,  // host api第二段接口名称
                        INPUT(input_desc, weight_desc, bias_desc, scale_desc, offset_desc,
                              strides_desc, padding_desc_2, dilation_desc, transposed,
                              output_padding_desc, groups, offsetx, roundMode.c_str()),  // host api输入
                        OUTPUT(output_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
    // EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(quant_convolution_test, ascend_test_extend_conv2d_roundmode_check3) {
    const int64_t groups = 1;
    vector<int64_t> input_dims = {1, 1, 1, 1};
    vector<int64_t> weight_dims = {1, 1 / groups, 1, 1};
    vector<int64_t> scale_dims = {1};
    vector<int64_t> bias_dims = {1};
    vector<int64_t> output_dims = {1, 1, 1, 1};
    vector<int64_t> strides_dims = {1, 1};
    vector<int64_t> padding_dims_2 = {0, 0, 0, 0};
    vector<int64_t> dilation_dims = {1, 1};
    vector<int64_t> output_padding_dims = {0, 0};
    auto strides_desc = IntArrayDesc(strides_dims);
    auto padding_desc_2 = IntArrayDesc(padding_dims_2);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    int32_t offsetx = 0;
    bool transposed = false;
    auto output_padding_desc = IntArrayDesc(output_padding_dims);
    auto offset_desc = nullptr;
    string roundMode = "!";
    auto input_desc = TensorDesc(input_dims, aclDataType::ACL_HIFLOAT8, ACL_FORMAT_NCHW).ValueRange(0, 2);
    auto weight_desc = TensorDesc(weight_dims, aclDataType::ACL_HIFLOAT8, ACL_FORMAT_NCHW).ValueRange(0, 2);
    auto scale_desc = TensorDesc(scale_dims, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 2);
    auto bias_desc = TensorDesc(bias_dims, aclDataType::ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto output_desc = TensorDesc(output_dims, aclDataType::ACL_FLOAT16, ACL_FORMAT_NCHW).Precision(0, 2);
    auto ut = OP_API_UT(aclnnQuantConvolution,  // host api第二段接口名称
                        INPUT(input_desc, weight_desc, bias_desc, scale_desc, offset_desc,
                              strides_desc, padding_desc_2, dilation_desc, transposed,
                              output_padding_desc, groups, offsetx, roundMode.c_str()),  // host api输入
                        OUTPUT(output_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
    EXPECT_NE(aclRet, ACL_SUCCESS);
}

TEST_F(quant_convolution_test, ascend_test_extend_conv2d_offsetx_check1) {
    const int64_t groups = 1;
    vector<int64_t> input_dims = {1, 1, 1, 1};
    vector<int64_t> weight_dims = {1, 1 / groups, 1, 1};
    vector<int64_t> scale_dims = {1};
    vector<int64_t> bias_dims = {1};
    vector<int64_t> output_dims = {1, 1, 1, 1};
    vector<int64_t> strides_dims = {1, 1};
    vector<int64_t> padding_dims_2 = {0, 0, 0, 0};
    vector<int64_t> dilation_dims = {1, 1};
    vector<int64_t> output_padding_dims = {0, 0};
    auto strides_desc = IntArrayDesc(strides_dims);
    auto padding_desc_2 = IntArrayDesc(padding_dims_2);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    int32_t offsetx = -129;
    bool transposed = false;
    auto output_padding_desc = IntArrayDesc(output_padding_dims);
    auto offset_desc = nullptr;
    auto input_desc = TensorDesc(input_dims, aclDataType::ACL_INT8, ACL_FORMAT_NCHW).ValueRange(0, 2);
    auto weight_desc = TensorDesc(weight_dims, aclDataType::ACL_INT8, ACL_FORMAT_NCHW).ValueRange(0, 2);
    auto scale_desc = TensorDesc(scale_dims, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 2);
    auto bias_desc = TensorDesc(bias_dims, aclDataType::ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 2);
    auto output_desc = TensorDesc(output_dims, aclDataType::ACL_FLOAT16, ACL_FORMAT_NCHW).Precision(0, 2);
    string roundMode = "rint";
    auto ut = OP_API_UT(aclnnQuantConvolution,  // host api第二段接口名称
                        INPUT(input_desc, weight_desc, bias_desc, scale_desc, offset_desc,
                              strides_desc, padding_desc_2, dilation_desc, transposed,
                              output_padding_desc, groups, offsetx, roundMode.c_str()),  // host api输入
                        OUTPUT(output_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
    // EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(quant_convolution_test, ascend910B2_test_quantConv3D_bias_fp16) {
  // test quantConv3d
  vector<aclDataType> ValidList = {ACL_FLOAT, ACL_FLOAT16};
  int length = ValidList.size();

  bool transposed = false;
  const int64_t groups = 1;
  vector<int64_t> inp_dims = {2, 2, 32, 32, 32};
  vector<int64_t> weight_dims = {2, 2 / groups, 3, 3, 3};
  vector<int64_t> bias_dims = {2 / groups};
  vector<int64_t> scale_dims = {2 / groups};
  vector<int64_t> output_dims = {2, 2, 32, 32, 32};
  vector<int64_t> strides_dims = {1, 1, 1};
  vector<int64_t> padding_dims = {1, 1, 1};
  vector<int64_t> dilation_dims = {1, 1, 1};
  vector<int64_t> output_padding_dims = {0, 0, 0};

  for (int i = 0; i < length; i++) {
    auto inp_desc = TensorDesc(inp_dims, ACL_INT8, ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto weight_desc = TensorDesc(weight_dims, ACL_INT8, ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto bias_desc = TensorDesc(bias_dims, ValidList[i], ACL_FORMAT_ND).ValueRange(0, 2);
    auto scale_desc = TensorDesc(scale_dims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto offset_desc = nullptr;
    auto strides_desc = IntArrayDesc(strides_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    auto output_padding_desc = IntArrayDesc(output_padding_dims);
    int32_t offsetx = 0;
    auto roundMode = nullptr;
    auto output_desc = TensorDesc(output_dims, ACL_FLOAT16, ACL_FORMAT_NCDHW).Precision(0.01, 0.01);

    auto ut = OP_API_UT(aclnnQuantConvolution,  // host api第二段接口名称
                        INPUT(inp_desc, weight_desc, bias_desc, scale_desc, offset_desc, strides_desc, padding_desc, dilation_desc, transposed,
                              output_padding_desc, groups, offsetx, roundMode),  // host api输入
                        OUTPUT(output_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
    EXPECT_EQ(aclRet, ACL_SUCCESS);
  }
}

TEST_F(quant_convolution_test, ascend910B2_test_quantConv3D_nonbias_fp16) {
  // test quantConv3d
  vector<aclDataType> ValidList = {ACL_FLOAT, ACL_FLOAT16};
  int length = ValidList.size();

  bool transposed = false;
  const int64_t groups = 1;
  vector<int64_t> inp_dims = {2, 2, 32, 32, 32};
  vector<int64_t> weight_dims = {2, 2 / groups, 3, 3, 3};
  vector<int64_t> bias_dims = {2 / groups};
  vector<int64_t> scale_dims = {2 / groups};
  vector<int64_t> output_dims = {2, 2, 32, 32, 32};
  vector<int64_t> strides_dims = {1, 1, 1};
  vector<int64_t> padding_dims = {1, 1, 1};
  vector<int64_t> dilation_dims = {1, 1, 1};
  vector<int64_t> output_padding_dims = {0, 0, 0};

  for (int i = 0; i < length; i++) {
    auto inp_desc = TensorDesc(inp_dims, ACL_INT8, ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto weight_desc = TensorDesc(weight_dims, ACL_INT8, ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto bias_desc = nullptr;
    auto scale_desc = TensorDesc(scale_dims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto offset_desc = nullptr;
    auto strides_desc = IntArrayDesc(strides_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    auto output_padding_desc = IntArrayDesc(output_padding_dims);
    int32_t offsetx = 0;
    auto roundMode = nullptr;
    auto output_desc = TensorDesc(output_dims, ACL_FLOAT16, ACL_FORMAT_NCDHW).Precision(0.01, 0.01);

    auto ut = OP_API_UT(aclnnQuantConvolution,  // host api第二段接口名称
                        INPUT(inp_desc, weight_desc, bias_desc, scale_desc, offset_desc, strides_desc, padding_desc, dilation_desc, transposed,
                              output_padding_desc, groups, offsetx, roundMode),  // host api输入
                        OUTPUT(output_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
  }
}

TEST_F(quant_convolution_test, test_quantConv3d_incorrect_soc_version_negative_fp16) {
  // test quantConv3d
  vector<aclDataType> ValidList = {ACL_FLOAT, ACL_FLOAT16};
  int length = ValidList.size();

  bool transposed = false;
  const int64_t groups = 1;
  vector<int64_t> inp_dims = {2, 2, 32, 32, 32};
  vector<int64_t> weight_dims = {2, 2 / groups, 3, 3, 3};
  vector<int64_t> bias_dims = {2 / groups};
  vector<int64_t> scale_dims = {2 / groups};
  vector<int64_t> output_dims = {2, 2, 32, 32, 32};
  vector<int64_t> strides_dims = {1, 1, 1};
  vector<int64_t> padding_dims = {1, 1, 1};
  vector<int64_t> dilation_dims = {1, 1, 1};
  vector<int64_t> output_padding_dims = {0, 0, 0};

  for (int i = 0; i < length; i++) {
    auto inp_desc = TensorDesc(inp_dims, ACL_INT8, ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto weight_desc = TensorDesc(weight_dims, ACL_INT8, ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto bias_desc = TensorDesc(bias_dims, ValidList[i], ACL_FORMAT_ND).ValueRange(0, 2);
    auto scale_desc = TensorDesc(scale_dims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto offset_desc = nullptr;
    auto strides_desc = IntArrayDesc(strides_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    auto output_padding_desc = IntArrayDesc(output_padding_dims);
    int32_t offsetx = 0;
    auto roundMode = nullptr;
    auto output_desc = TensorDesc(output_dims, ACL_FLOAT16, ACL_FORMAT_NCDHW).Precision(0.01, 0.01);

    auto ut = OP_API_UT(aclnnQuantConvolution,  // host api第二段接口名称
                        INPUT(inp_desc, weight_desc, bias_desc, scale_desc, offset_desc, strides_desc, padding_desc, dilation_desc, transposed,
                              output_padding_desc, groups, offsetx, roundMode),  // host api输入
                        OUTPUT(output_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
    auto curSoc = GetCurrentPlatformInfo().GetSocVersion();
    if (curSoc == SocVersion::ASCEND910B) {
        EXPECT_EQ(aclRet, ACL_SUCCESS);
    } else {
        EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    }
  }
}

TEST_F(quant_convolution_test, ascend910B2_quantConv3d_groups_not_equal_1_fp16) {
  // test quantConv3d
  vector<aclDataType> ValidList = {ACL_FLOAT, ACL_FLOAT16};
  int length = ValidList.size();

  bool transposed = false;
  const int64_t groups = 2;
  vector<int64_t> inp_dims = {2, 2, 32, 32, 32};
  vector<int64_t> weight_dims = {2, 2 / groups, 3, 3, 3};
  vector<int64_t> bias_dims = {2 / groups};
  vector<int64_t> scale_dims = {2 / groups};
  vector<int64_t> output_dims = {2, 2, 32, 32, 32};
  vector<int64_t> strides_dims = {1, 1, 1};
  vector<int64_t> padding_dims = {1, 1, 1};
  vector<int64_t> dilation_dims = {1, 1, 1};
  vector<int64_t> output_padding_dims = {0, 0, 0};

  for (int i = 0; i < length; i++) {
    auto inp_desc = TensorDesc(inp_dims, ACL_INT8, ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto weight_desc = TensorDesc(weight_dims, ACL_INT8, ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto bias_desc = TensorDesc(bias_dims, ValidList[i], ACL_FORMAT_ND).ValueRange(0, 2);
    auto scale_desc = TensorDesc(scale_dims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto offset_desc = nullptr;
    auto strides_desc = IntArrayDesc(strides_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    auto output_padding_desc = IntArrayDesc(output_padding_dims);
    int32_t offsetx = 0;
    auto roundMode = nullptr;
    auto output_desc = TensorDesc(output_dims, ACL_FLOAT16, ACL_FORMAT_NCDHW).Precision(0.01, 0.01);

    auto ut = OP_API_UT(aclnnQuantConvolution,  // host api第二段接口名称
                        INPUT(inp_desc, weight_desc, bias_desc, scale_desc, offset_desc, strides_desc,
                              padding_desc, dilation_desc, transposed, output_padding_desc, groups, offsetx, roundMode),  // host api输入
                        OUTPUT(output_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
  }
}

TEST_F(quant_convolution_test, ascend910B2_quantConv3d_incorrect_data_format_negative_fp16) {
  vector<aclDataType> ValidList = {ACL_FLOAT16};
  int length = ValidList.size();

  bool transposed = false;
  const int64_t groups = 1;
  vector<int64_t> inp_dims = {2, 2, 32, 32, 32};
  vector<int64_t> weight_dims = {2, 2 / groups, 3, 3, 3};
  vector<int64_t> bias_dims = {2 / groups};
  vector<int64_t> scale_dims = {2 / groups};
  vector<int64_t> output_dims = {2, 2, 32, 32, 32};
  vector<int64_t> strides_dims = {1, 1, 1};
  vector<int64_t> padding_dims = {1, 1, 1};
  vector<int64_t> dilation_dims = {1, 1, 1};
  vector<int64_t> output_padding_dims = {0, 0, 0};

  for (int i = 0; i < length; i++) {
    auto inp_desc = TensorDesc(inp_dims, ACL_INT8, ACL_FORMAT_NDHWC).ValueRange(0, 2);
    auto weight_desc = TensorDesc(weight_dims, ACL_INT8, ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto bias_desc = TensorDesc(bias_dims, ValidList[i], ACL_FORMAT_ND).ValueRange(0, 2);
    auto scale_desc = TensorDesc(scale_dims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto offset_desc = nullptr;
    auto strides_desc = IntArrayDesc(strides_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    auto output_padding_desc = IntArrayDesc(output_padding_dims);
    int32_t offsetx = 0;
    auto roundMode = nullptr;
    auto output_desc = TensorDesc(output_dims, ACL_FLOAT16, ACL_FORMAT_NCDHW).Precision(0.01, 0.01);

    auto ut = OP_API_UT(aclnnQuantConvolution,  // host api第二段接口名称
                        INPUT(inp_desc, weight_desc, bias_desc, scale_desc, offset_desc, strides_desc, padding_desc, dilation_desc, transposed,
                              output_padding_desc, groups, offsetx, roundMode),  // host api输入
                        OUTPUT(output_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
  }
}

TEST_F(quant_convolution_test, ascend910B2_test_quantConv3d_incorrect_output_shape_fp16) {
  // test conv3d
  vector<aclDataType> ValidList = {ACL_FLOAT16};
  int length = ValidList.size();

  bool transposed = false;
  const int64_t groups = 1;
  vector<int64_t> inp_dims = {2, 2, 32, 32, 32};
  vector<int64_t> weight_dims = {2, 2 / groups, 3, 3, 3};
  vector<int64_t> bias_dims = {2 / groups};
  vector<int64_t> scale_dims = {2 / groups};
  vector<int64_t> output_dims = {2, 2, 30, 30, 30};
  vector<int64_t> strides_dims = {1, 1, 1};
  vector<int64_t> padding_dims = {1, 1, 1};
  vector<int64_t> dilation_dims = {1, 1, 1};
  vector<int64_t> output_padding_dims = {0, 0, 0};

  for (int i = 0; i < length; i++) {
    auto inp_desc = TensorDesc(inp_dims, ACL_INT8, ACL_FORMAT_NDHWC).ValueRange(0, 2);
    auto weight_desc = TensorDesc(weight_dims, ACL_INT8, ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto bias_desc = TensorDesc(bias_dims, ValidList[i], ACL_FORMAT_ND).ValueRange(0, 2);
    auto scale_desc = TensorDesc(scale_dims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto offset_desc = nullptr;
    auto strides_desc = IntArrayDesc(strides_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    auto output_padding_desc = IntArrayDesc(output_padding_dims);
    int32_t offsetx = 0;
    auto roundMode = nullptr;
    auto output_desc = TensorDesc(output_dims, ACL_FLOAT16, ACL_FORMAT_NCDHW).Precision(0.01, 0.01);

    auto ut = OP_API_UT(aclnnQuantConvolution,  // host api第二段接口名称
                        INPUT(inp_desc, weight_desc, bias_desc, scale_desc, offset_desc,
                              strides_desc, padding_desc, dilation_desc, transposed,
                              output_padding_desc, groups, offsetx, roundMode),  // host api输入
                        OUTPUT(output_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
  }
}

TEST_F(quant_convolution_test, ascend910B2_test_quantConv3d_incorrect_scale_shape_value_fp16) {
  // test conv3d
  vector<aclDataType> ValidList = {ACL_FLOAT16};
  int length = ValidList.size();

  bool transposed = false;
  const int64_t groups = 1;
  vector<int64_t> inp_dims = {2, 2, 32, 32, 32};
  vector<int64_t> weight_dims = {2, 2 / groups, 3, 3, 3};
  vector<int64_t> bias_dims = {2 / groups};
  vector<int64_t> scale_dims = {1};
  vector<int64_t> output_dims = {2, 2, 32, 32, 32};
  vector<int64_t> strides_dims = {1, 1, 1};
  vector<int64_t> padding_dims = {1, 1, 1};
  vector<int64_t> dilation_dims = {1, 1, 1};
  vector<int64_t> output_padding_dims = {0, 0, 0};

  for (int i = 0; i < length; i++) {
    auto inp_desc = TensorDesc(inp_dims, ACL_INT8, ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto weight_desc = TensorDesc(weight_dims, ACL_INT8, ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto bias_desc = TensorDesc(bias_dims, ValidList[i], ACL_FORMAT_ND).ValueRange(0, 2);
    auto scale_desc = TensorDesc(scale_dims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto offset_desc = nullptr;
    auto strides_desc = IntArrayDesc(strides_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    auto output_padding_desc = IntArrayDesc(output_padding_dims);
    int32_t offsetx = 0;
    auto roundMode = nullptr;
    auto output_desc = TensorDesc(output_dims, ACL_FLOAT16, ACL_FORMAT_NCDHW).Precision(0.01, 0.01);

    auto ut = OP_API_UT(aclnnQuantConvolution,  // host api第二段接口名称
                        INPUT(inp_desc, weight_desc, bias_desc, scale_desc, offset_desc,
                              strides_desc, padding_desc, dilation_desc, transposed,
                              output_padding_desc, groups, offsetx, roundMode),  // host api输入
                        OUTPUT(output_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
  }
}

TEST_F(quant_convolution_test, ascend910B2_test_quantConv3d_incorrect_input_dim_fp16) {
  // test conv3d
  vector<aclDataType> ValidList = {ACL_FLOAT16};
  int length = ValidList.size();

  bool transposed = false;
  const int64_t groups = 1;
  vector<int64_t> inp_dims = {2, 2, 32, 32};
  vector<int64_t> weight_dims = {2, 2 / groups, 3, 3, 3};
  vector<int64_t> bias_dims = {2 / groups};
  vector<int64_t> scale_dims = {2 / groups};
  vector<int64_t> output_dims = {2, 2, 32, 32, 32};
  vector<int64_t> strides_dims = {1, 1, 1};
  vector<int64_t> padding_dims = {1, 1, 1};
  vector<int64_t> dilation_dims = {1, 1, 1};
  vector<int64_t> output_padding_dims = {0, 0, 0};

  for (int i = 0; i < length; i++) {
    auto inp_desc = TensorDesc(inp_dims, ACL_INT8, ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto weight_desc = TensorDesc(weight_dims, ACL_INT8, ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto bias_desc = TensorDesc(bias_dims, ValidList[i], ACL_FORMAT_ND).ValueRange(0, 2);
    auto scale_desc = TensorDesc(scale_dims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto offset_desc = nullptr;
    auto strides_desc = IntArrayDesc(strides_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    auto output_padding_desc = IntArrayDesc(output_padding_dims);
    int32_t offsetx = 0;
    auto roundMode = nullptr;
    auto output_desc = TensorDesc(output_dims, ACL_FLOAT16, ACL_FORMAT_NCDHW).Precision(0.01, 0.01);

    auto ut = OP_API_UT(aclnnQuantConvolution,  // host api第二段接口名称
                        INPUT(inp_desc, weight_desc, bias_desc, scale_desc, offset_desc,
                              strides_desc, padding_desc, dilation_desc, transposed,
                              output_padding_desc, groups, offsetx, roundMode),  // host api输入
                        OUTPUT(output_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
  }
}

TEST_F(quant_convolution_test, ascend910B2_test_quantConv3d_incorrect_attr_dim_fp16) {
  // test conv3d
  vector<aclDataType> ValidList = {ACL_FLOAT16};
  int length = ValidList.size();

  bool transposed = false;
  const int64_t groups = 1;
  vector<int64_t> inp_dims = {2, 2, 32, 32, 32};
  vector<int64_t> weight_dims = {2, 2 / groups, 3, 3, 3};
  vector<int64_t> bias_dims = {2 / groups};
  vector<int64_t> scale_dims = {2 / groups};
  vector<int64_t> output_dims = {2, 2, 32, 32, 32};
  vector<int64_t> strides_dims = {1, 1, 1};
  vector<int64_t> padding_dims = {1, 1};
  vector<int64_t> dilation_dims = {1, 1, 1};
  vector<int64_t> output_padding_dims = {0, 0, 0};

  for (int i = 0; i < length; i++) {
    auto inp_desc = TensorDesc(inp_dims, ACL_INT8, ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto weight_desc = TensorDesc(weight_dims, ACL_INT8, ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto bias_desc = TensorDesc(bias_dims, ValidList[i], ACL_FORMAT_ND).ValueRange(0, 2);
    auto scale_desc = TensorDesc(scale_dims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto offset_desc = nullptr;
    auto strides_desc = IntArrayDesc(strides_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    auto output_padding_desc = IntArrayDesc(output_padding_dims);
    int32_t offsetx = 0;
    auto roundMode = nullptr;
    auto output_desc = TensorDesc(output_dims, ACL_FLOAT16, ACL_FORMAT_NCDHW).Precision(0.01, 0.01);

    auto ut = OP_API_UT(aclnnQuantConvolution,  // host api第二段接口名称
                        INPUT(inp_desc, weight_desc, bias_desc, scale_desc, offset_desc,
                              strides_desc, padding_desc, dilation_desc, transposed,
                              output_padding_desc, groups, offsetx, roundMode),  // host api输入
                        OUTPUT(output_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
  }
}

TEST_F(quant_convolution_test, ascend910B2_test_quantConv3d_attr_value_has_zero_fp16) {
  // test conv3d
  vector<aclDataType> ValidList = {ACL_FLOAT16};
  int length = ValidList.size();

  bool transposed = false;
  const int64_t groups = 1;
  vector<int64_t> inp_dims = {2, 2, 32, 32, 32};
  vector<int64_t> weight_dims = {2, 2 / groups, 3, 3, 3};
  vector<int64_t> bias_dims = {2 / groups};
  vector<int64_t> scale_dims = {2 / groups};
  vector<int64_t> output_dims = {2, 2, 32, 32, 32};
  vector<int64_t> strides_dims = {1, 1, 0};
  vector<int64_t> padding_dims = {1, 1, 1};
  vector<int64_t> dilation_dims = {1, 1, 1};
  vector<int64_t> output_padding_dims = {0, 0, 0};

  for (int i = 0; i < length; i++) {
    auto inp_desc = TensorDesc(inp_dims, ACL_INT8, ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto weight_desc = TensorDesc(weight_dims, ACL_INT8, ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto bias_desc = TensorDesc(bias_dims, ValidList[i], ACL_FORMAT_ND).ValueRange(0, 2);
    auto scale_desc = TensorDesc(scale_dims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto offset_desc = nullptr;
    auto strides_desc = IntArrayDesc(strides_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    auto output_padding_desc = IntArrayDesc(output_padding_dims);
    int32_t offsetx = 0;
    auto roundMode = nullptr;
    auto output_desc = TensorDesc(output_dims, ACL_FLOAT16, ACL_FORMAT_NCDHW).Precision(0.01, 0.01);

    auto ut = OP_API_UT(aclnnQuantConvolution,  // host api第二段接口名称
                        INPUT(inp_desc, weight_desc, bias_desc, scale_desc, offset_desc,
                              strides_desc, padding_desc, dilation_desc, transposed,
                              output_padding_desc, groups, offsetx, roundMode),  // host api输入
                        OUTPUT(output_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
  }
}

TEST_F(quant_convolution_test, ascend910B2_test_quantConv3d_weight_cin_wrong_fp16) {
  // test conv3d
  vector<aclDataType> ValidList = {ACL_FLOAT16};
  int length = ValidList.size();

  bool transposed = false;
  const int64_t groups = 1;
  vector<int64_t> inp_dims = {2, 2, 32, 32, 32};
  vector<int64_t> weight_dims = {2, 1, 3, 3, 3};
  vector<int64_t> bias_dims = {2 / groups};
  vector<int64_t> scale_dims = {2 / groups};
  vector<int64_t> output_dims = {2, 2, 32, 32, 32};
  vector<int64_t> strides_dims = {1, 1, 1};
  vector<int64_t> padding_dims = {1, 1, 0};
  vector<int64_t> dilation_dims = {1, 1, 1};
  vector<int64_t> output_padding_dims = {0, 0, 0};

  for (int i = 0; i < length; i++) {
    auto inp_desc = TensorDesc(inp_dims, ACL_INT8, ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto weight_desc = TensorDesc(weight_dims, ACL_INT8, ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto bias_desc = TensorDesc(bias_dims, ValidList[i], ACL_FORMAT_ND).ValueRange(0, 2);
    auto scale_desc = TensorDesc(scale_dims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto offset_desc = nullptr;
    auto strides_desc = IntArrayDesc(strides_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    auto output_padding_desc = IntArrayDesc(output_padding_dims);
    int32_t offsetx = 0;
    auto roundMode = nullptr;
    auto output_desc = TensorDesc(output_dims, ACL_FLOAT16, ACL_FORMAT_NCDHW).Precision(0.01, 0.01);

    auto ut = OP_API_UT(aclnnQuantConvolution,  // host api第二段接口名称
                        INPUT(inp_desc, weight_desc, bias_desc, scale_desc, offset_desc,
                              strides_desc, padding_desc, dilation_desc, transposed,
                              output_padding_desc, groups, offsetx, roundMode),  // host api输入
                        OUTPUT(output_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
  }
}

TEST_F(quant_convolution_test, ascend910B2_test_quantConv3d_incorrect_stride_value_fp16) {
  // test conv3d
  vector<aclDataType> ValidList = {ACL_FLOAT16};
  int length = ValidList.size();

  bool transposed = false;
  const int64_t groups = 1;
  vector<int64_t> inp_dims = {2, 2, 32, 32, 32};
  vector<int64_t> weight_dims = {2, 2 / groups, 3, 3, 3};
  vector<int64_t> bias_dims = {2 / groups};
  vector<int64_t> scale_dims = {2 / groups};
  vector<int64_t> output_dims = {2, 2, 32, 32, 32};
  vector<int64_t> strides_dims = {0, 1, 1};
  vector<int64_t> padding_dims = {1, 1, 1};
  vector<int64_t> dilation_dims = {1, 1, 1};
  vector<int64_t> output_padding_dims = {0, 0, 0};

  for (int i = 0; i < length; i++) {
    auto inp_desc = TensorDesc(inp_dims, ACL_INT8, ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto weight_desc = TensorDesc(weight_dims, ACL_INT8, ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto bias_desc = TensorDesc(bias_dims, ValidList[i], ACL_FORMAT_ND).ValueRange(0, 2);
    auto scale_desc = TensorDesc(scale_dims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto offset_desc = nullptr;
    auto strides_desc = IntArrayDesc(strides_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    auto output_padding_desc = IntArrayDesc(output_padding_dims);
    int32_t offsetx = 0;
    auto roundMode = nullptr;
    auto output_desc = TensorDesc(output_dims, ACL_FLOAT16, ACL_FORMAT_NCDHW).Precision(0.01, 0.01);

    auto ut = OP_API_UT(aclnnQuantConvolution,  // host api第二段接口名称
                        INPUT(inp_desc, weight_desc, bias_desc, scale_desc, offset_desc,
                              strides_desc, padding_desc, dilation_desc, transposed,
                              output_padding_desc, groups, offsetx, roundMode),  // host api输入
                        OUTPUT(output_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
  }
}

TEST_F(quant_convolution_test, ascend910B2_test_quantConv3d_input_nullptr_fp16) {
  // test conv3d
  vector<aclDataType> ValidList = {ACL_FLOAT16};
  int length = ValidList.size();

  bool transposed = false;
  const int64_t groups = 1;
  vector<int64_t> inp_dims = {2, 2, 32, 32, 32};
  vector<int64_t> weight_dims = {2, 2 / groups, 3, 3, 3};
  vector<int64_t> bias_dims = {2 / groups};
  vector<int64_t> scale_dims = {2 / groups};
  vector<int64_t> output_dims = {2, 2, 32, 32, 32};
  vector<int64_t> strides_dims = {0, 1, 1};
  vector<int64_t> padding_dims = {1, 1, 1};
  vector<int64_t> dilation_dims = {1, 1, 1};
  vector<int64_t> output_padding_dims = {0, 0, 0};

  for (int i = 0; i < length; i++) {
    auto inp_desc = nullptr;
    auto weight_desc = TensorDesc(weight_dims, ACL_INT8, ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto bias_desc = TensorDesc(bias_dims, ValidList[i], ACL_FORMAT_ND).ValueRange(0, 2);
    auto scale_desc = TensorDesc(scale_dims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto offset_desc = nullptr;
    auto strides_desc = IntArrayDesc(strides_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    auto output_padding_desc = IntArrayDesc(output_padding_dims);
    int32_t offsetx = 0;
    auto roundMode = nullptr;
    auto output_desc = TensorDesc(output_dims, ACL_FLOAT16, ACL_FORMAT_NCDHW).Precision(0.01, 0.01);

    auto ut = OP_API_UT(aclnnQuantConvolution,  // host api第二段接口名称
                        INPUT(inp_desc, weight_desc, bias_desc, scale_desc, offset_desc,
                              strides_desc, padding_desc, dilation_desc, transposed,
                              output_padding_desc, groups, offsetx, roundMode),  // host api输入
                        OUTPUT(output_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
  }
}
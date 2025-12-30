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

#include "../../../op_api/aclnn_masked_scatter.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"

using namespace op;
using namespace std;

class l2_masked_scatter_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    cout << "masked_scatter_test SetUp" << endl;
  }

  static void TearDownTestCase() {
    cout << "masked_scatter_test TearDown" << endl;
  }
};

// 基础用例
TEST_F(l2_masked_scatter_test, aclnnInplaceMaskedScatter_base_case_1) {
  // self input
  const vector<int64_t>& selfShape = {8};
  aclDataType selfDtype = ACL_INT32;
  aclFormat selfFormat = ACL_FORMAT_ND;
  // mask input
  const vector<int64_t>& maskShape = {8};
  aclDataType maskDtype = ACL_BOOL;
  aclFormat maskFormat = ACL_FORMAT_ND;
  // source input
  const vector<int64_t>& sourceShape = {8};
  aclDataType sourceDtype = selfDtype;
  aclFormat sourceFormat = ACL_FORMAT_ND;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat).Value(vector<int32_t>{1, 1, 1, 1, 1, 1, 1, 1});
  auto maskTensorDesc = TensorDesc(maskShape, maskDtype, maskFormat)
                            .Value(vector<bool>{true, true, true, true, false, false, false, false});
  auto sourceTensorDesc =
      TensorDesc(sourceShape, sourceDtype, sourceFormat).Value(vector<int32_t>{0, 0, 0, 0, 0, 0, 0, 0});

  auto ut = OP_API_UT(aclnnInplaceMaskedScatter, INPUT(selfTensorDesc, maskTensorDesc, sourceTensorDesc), OUTPUT());
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  ut.TestPrecision();
}

TEST_F(l2_masked_scatter_test, aclnnInplaceMaskedScatter_base_case_2) {
  // self input
  const vector<int64_t>& selfShape = {2, 4};
  aclDataType selfDtype = ACL_FLOAT;
  aclFormat selfFormat = ACL_FORMAT_ND;
  // mask input
  const vector<int64_t>& maskShape = {2, 4};
  aclDataType maskDtype = ACL_BOOL;
  aclFormat maskFormat = ACL_FORMAT_ND;
  // source input
  const vector<int64_t>& sourceShape = {8};
  aclDataType sourceDtype = ACL_FLOAT;
  aclFormat sourceFormat = ACL_FORMAT_ND;

  auto selfTensorDesc =
      TensorDesc(selfShape, selfDtype, selfFormat).Value(vector<float>{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0});
  auto maskTensorDesc = TensorDesc(maskShape, maskDtype, maskFormat)
                            .Value(vector<bool>{false, false, false, false, false, false, false, false});
  auto sourceTensorDesc =
      TensorDesc(sourceShape, sourceDtype, sourceFormat).Value(vector<float>{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0});

  auto ut = OP_API_UT(aclnnInplaceMaskedScatter, INPUT(selfTensorDesc, maskTensorDesc, sourceTensorDesc), OUTPUT());
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  ut.TestPrecision();
}

// 简单随机用例
TEST_F(l2_masked_scatter_test, aclnnInplaceMaskedScatter_8_int32_nd_and_8_bool_nd_simple_random_case_1) {
  // self input
  const vector<int64_t>& selfShape = {8};
  aclDataType selfDtype = ACL_INT32;
  aclFormat selfFormat = ACL_FORMAT_ND;
  // mask input
  const vector<int64_t>& maskShape = {8};
  aclDataType maskDtype = ACL_BOOL;
  aclFormat maskFormat = ACL_FORMAT_ND;
  // source input
  const vector<int64_t>& sourceShape = {8};
  aclDataType sourceDtype = selfDtype;
  aclFormat sourceFormat = ACL_FORMAT_ND;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
  auto maskTensorDesc = TensorDesc(maskShape, maskDtype, maskFormat);
  auto sourceTensorDesc = TensorDesc(sourceShape, sourceDtype, sourceFormat);

  auto ut = OP_API_UT(aclnnInplaceMaskedScatter, INPUT(selfTensorDesc, maskTensorDesc, sourceTensorDesc), OUTPUT());
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  ut.TestPrecision();
}

TEST_F(l2_masked_scatter_test, aclnnInplaceMaskedScatter_8_8_float_nd_and_8_8_bool_nd_simple_random_case_2) {
  // self input
  const vector<int64_t>& selfShape = {8, 8};
  aclDataType selfDtype = ACL_FLOAT;
  aclFormat selfFormat = ACL_FORMAT_ND;
  // mask input
  const vector<int64_t>& maskShape = {8, 8};

  aclDataType maskDtype = ACL_BOOL;
  aclFormat maskFormat = ACL_FORMAT_ND;
  // source input
  const vector<int64_t>& sourceShape = {8, 8};
  aclDataType sourceDtype = selfDtype;
  aclFormat sourceFormat = ACL_FORMAT_ND;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
  auto maskTensorDesc = TensorDesc(maskShape, maskDtype, maskFormat);
  auto sourceTensorDesc = TensorDesc(sourceShape, sourceDtype, sourceFormat);

  auto ut = OP_API_UT(aclnnInplaceMaskedScatter, INPUT(selfTensorDesc, maskTensorDesc, sourceTensorDesc), OUTPUT());
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  ut.TestPrecision();
}

/* 各元素基本类型覆盖用例
 * 维度：1-8维
 * self基本数据类型：FLOAT、FLOAT16、INT32、INT64、INT16、INT8、UINT8、DOUBLE、COMPLEX64、COMPLEX128，BFLOAT16,BOOL
 * mask基本数据类型：UINT8、BOOL
 * 数据格式：ND、NCHW、NHWC、HWCN、NDHWC、NCDHW
 */

TEST_F(l2_masked_scatter_test, aclnnInplaceMaskedScatter_2_4_6_8_10_12_14_16_int64_nd_and_2_4_6_8_10_12_14_16_bool_nd) {
  // self input
  const vector<int64_t>& selfShape = {2, 4, 6, 8, 10, 12, 14, 16};
  aclDataType selfDtype = ACL_INT64;
  aclFormat selfFormat = ACL_FORMAT_ND;
  // mask input
  const vector<int64_t>& maskShape = {2, 4, 6, 8, 10, 12, 14, 16};
  aclDataType maskDtype = ACL_BOOL;
  aclFormat maskFormat = ACL_FORMAT_ND;
  // source input
  const vector<int64_t>& sourceShape = {2, 4, 6, 8, 10, 12, 14, 16};
  aclDataType sourceDtype = selfDtype;
  aclFormat sourceFormat = ACL_FORMAT_ND;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
  auto maskTensorDesc = TensorDesc(maskShape, maskDtype, maskFormat);
  auto sourceTensorDesc = TensorDesc(sourceShape, sourceDtype, sourceFormat);

  auto ut = OP_API_UT(aclnnInplaceMaskedScatter, INPUT(selfTensorDesc, maskTensorDesc, sourceTensorDesc), OUTPUT());
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_masked_scatter_test, aclnnInplaceMaskedScatter_2_4_6_8_10_12_14_float16_nchw_and_2_4_6_8_10_12_14_bool_nchw) {
  // self input
  const vector<int64_t>& selfShape = {2, 4, 6, 8, 10, 12, 14};
  aclDataType selfDtype = ACL_FLOAT16;
  aclFormat selfFormat = ACL_FORMAT_NCHW;
  // mask input
  const vector<int64_t>& maskShape = {2, 4, 6, 8, 10, 12, 14};
  aclDataType maskDtype = ACL_BOOL;
  aclFormat maskFormat = ACL_FORMAT_NCHW;
  // source input
  const vector<int64_t>& sourceShape = {2, 4, 6, 8, 10, 12, 14};
  aclDataType sourceDtype = selfDtype;
  aclFormat sourceFormat = ACL_FORMAT_NCHW;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
  auto maskTensorDesc = TensorDesc(maskShape, maskDtype, maskFormat);
  auto sourceTensorDesc = TensorDesc(sourceShape, sourceDtype, sourceFormat);

  auto ut = OP_API_UT(aclnnInplaceMaskedScatter, INPUT(selfTensorDesc, maskTensorDesc, sourceTensorDesc), OUTPUT());
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  ut.TestPrecision();
}

TEST_F(l2_masked_scatter_test, aclnnInplaceMaskedScatter_2_2_2_2_2_2_2_float16_nchw_and_2_2_2_2_2_2_2_bool_nchw) {
  // self input
  const vector<int64_t>& selfShape = {2, 2, 2, 2, 2, 2, 2};
  aclDataType selfDtype = ACL_FLOAT16;
  aclFormat selfFormat = ACL_FORMAT_NCHW;
  // mask input
  const vector<int64_t>& maskShape = {2, 2, 2, 2, 2, 2, 2};
  aclDataType maskDtype = ACL_BOOL;
  aclFormat maskFormat = ACL_FORMAT_NCHW;
  // source input
  const vector<int64_t>& sourceShape = {2, 2, 2, 2, 2, 2, 2};
  aclDataType sourceDtype = selfDtype;
  aclFormat sourceFormat = ACL_FORMAT_NCHW;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat).ValueRange(-1, 1);
  auto maskTensorDesc = TensorDesc(maskShape, maskDtype, maskFormat);
  auto sourceTensorDesc = TensorDesc(sourceShape, sourceDtype, sourceFormat);

  auto ut = OP_API_UT(aclnnInplaceMaskedScatter, INPUT(selfTensorDesc, maskTensorDesc, sourceTensorDesc), OUTPUT());
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  ut.TestPrecision();
}

TEST_F(l2_masked_scatter_test, aclnnInplaceMaskedScatter_2_2_2_2_2_2_2_2_float16_nchw_and_2_2_2_2_2_2_2_2_bool_nchw) {
  // self input
  const vector<int64_t>& selfShape = {2, 2, 2, 2, 2, 2, 2, 2};
  aclDataType selfDtype = ACL_FLOAT16;
  aclFormat selfFormat = ACL_FORMAT_NCHW;
  // mask input
  const vector<int64_t>& maskShape = {2, 2, 2, 2, 2, 2, 2, 2};
  aclDataType maskDtype = ACL_BOOL;
  aclFormat maskFormat = ACL_FORMAT_NCHW;
  // source input
  const vector<int64_t>& sourceShape = {2, 2, 2, 2, 2, 2, 2, 2};
  aclDataType sourceDtype = selfDtype;
  aclFormat sourceFormat = ACL_FORMAT_NCHW;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
  auto maskTensorDesc = TensorDesc(maskShape, maskDtype, maskFormat);
  auto sourceTensorDesc = TensorDesc(sourceShape, sourceDtype, sourceFormat);

  auto ut = OP_API_UT(aclnnInplaceMaskedScatter, INPUT(selfTensorDesc, maskTensorDesc, sourceTensorDesc), OUTPUT());
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  ut.TestPrecision();
}

TEST_F(l2_masked_scatter_test, aclnnInplaceMaskedScatter_2_4_6_8_10_12_uint8_nhwc_and_2_4_6_8_10_12_bool_nhwc) {
  // self input
  const vector<int64_t>& selfShape = {2, 4, 6, 8, 10, 12};
  aclDataType selfDtype = ACL_UINT8;
  aclFormat selfFormat = ACL_FORMAT_NHWC;
  // mask input
  const vector<int64_t>& maskShape = {2, 4, 6, 8, 10, 12};
  aclDataType maskDtype = ACL_BOOL;
  aclFormat maskFormat = ACL_FORMAT_NHWC;
  // source input
  const vector<int64_t>& sourceShape = {2, 4, 6, 8, 10, 12};
  aclDataType sourceDtype = selfDtype;
  aclFormat sourceFormat = ACL_FORMAT_NHWC;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
  auto maskTensorDesc = TensorDesc(maskShape, maskDtype, maskFormat);
  auto sourceTensorDesc = TensorDesc(sourceShape, sourceDtype, sourceFormat);

  auto ut = OP_API_UT(aclnnInplaceMaskedScatter, INPUT(selfTensorDesc, maskTensorDesc, sourceTensorDesc), OUTPUT());
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_masked_scatter_test, aclnnInplaceMaskedScatter_2_4_6_8_10_int8_hwcn_and_2_4_6_8_10_bool_hwcn) {
  // self input
  const vector<int64_t>& selfShape = {2, 4, 6, 8, 10};
  aclDataType selfDtype = ACL_INT8;
  aclFormat selfFormat = ACL_FORMAT_HWCN;
  // mask input
  const vector<int64_t>& maskShape = {2, 4, 6, 8, 10};
  aclDataType maskDtype = ACL_BOOL;
  aclFormat maskFormat = ACL_FORMAT_HWCN;
  // source input
  const vector<int64_t>& sourceShape = {2, 4, 6, 8, 10};
  aclDataType sourceDtype = selfDtype;
  aclFormat sourceFormat = ACL_FORMAT_HWCN;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
  auto maskTensorDesc = TensorDesc(maskShape, maskDtype, maskFormat);
  auto sourceTensorDesc = TensorDesc(sourceShape, sourceDtype, sourceFormat);

  auto ut = OP_API_UT(aclnnInplaceMaskedScatter, INPUT(selfTensorDesc, maskTensorDesc, sourceTensorDesc), OUTPUT());
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  ut.TestPrecision();
}

TEST_F(l2_masked_scatter_test, aclnnInplaceMaskedScatter_2_4_6_8_int16_ndhwc_and_2_4_6_8_bool_ndhwc) {
  // self input
  const vector<int64_t>& selfShape = {2, 4, 6, 8};
  aclDataType selfDtype = ACL_INT16;
  aclFormat selfFormat = ACL_FORMAT_NDHWC;
  // mask input
  const vector<int64_t>& maskShape = {2, 4, 6, 8};
  aclDataType maskDtype = ACL_BOOL;
  aclFormat maskFormat = ACL_FORMAT_NDHWC;
  // source input
  const vector<int64_t>& sourceShape = {2, 4, 6, 8};
  aclDataType sourceDtype = selfDtype;
  aclFormat sourceFormat = ACL_FORMAT_NDHWC;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
  auto maskTensorDesc = TensorDesc(maskShape, maskDtype, maskFormat);
  auto sourceTensorDesc = TensorDesc(sourceShape, sourceDtype, sourceFormat);

  auto ut = OP_API_UT(aclnnInplaceMaskedScatter, INPUT(selfTensorDesc, maskTensorDesc, sourceTensorDesc), OUTPUT());
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  ut.TestPrecision();
}

TEST_F(l2_masked_scatter_test, aclnnInplaceMaskedScatter_2_4_6_int32_ncdhw_and_2_4_6_bool_ncdhw) {
  // self input
  const vector<int64_t>& selfShape = {2, 4, 6};
  aclDataType selfDtype = ACL_INT32;
  aclFormat selfFormat = ACL_FORMAT_NCDHW;
  // mask input
  const vector<int64_t>& maskShape = {2, 4, 6};
  aclDataType maskDtype = ACL_BOOL;
  aclFormat maskFormat = ACL_FORMAT_NCDHW;
  // source input
  const vector<int64_t>& sourceShape = {2, 4, 6};
  aclDataType sourceDtype = selfDtype;
  aclFormat sourceFormat = ACL_FORMAT_NCDHW;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
  auto maskTensorDesc = TensorDesc(maskShape, maskDtype, maskFormat);
  auto sourceTensorDesc = TensorDesc(sourceShape, sourceDtype, sourceFormat);

  auto ut = OP_API_UT(aclnnInplaceMaskedScatter, INPUT(selfTensorDesc, maskTensorDesc, sourceTensorDesc), OUTPUT());
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  ut.TestPrecision();
}
/* 各元素特殊类型覆盖用例
 * 空tensor
 * 溢出值
 * 边界值
 * 特殊值：nan,inf
 * 非连续
 */
TEST_F(l2_masked_scatter_test, aclnnInplaceMaskedScatter_0_2_float_nd_and_0_2_bool_nd_empty_tensor) {
  // self input
  const vector<int64_t>& selfShape = {0, 2};
  aclDataType selfDtype = ACL_FLOAT;
  aclFormat selfFormat = ACL_FORMAT_ND;
  // mask input
  const vector<int64_t>& maskShape = {0, 2};

  aclDataType maskDtype = ACL_BOOL;
  aclFormat maskFormat = ACL_FORMAT_ND;
  // source input
  const vector<int64_t>& sourceShape = {2};
  aclDataType sourceDtype = selfDtype;
  aclFormat sourceFormat = ACL_FORMAT_ND;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
  auto maskTensorDesc = TensorDesc(maskShape, maskDtype, maskFormat);
  auto sourceTensorDesc = TensorDesc(sourceShape, sourceDtype, sourceFormat);

  auto ut = OP_API_UT(aclnnInplaceMaskedScatter, INPUT(selfTensorDesc, maskTensorDesc, sourceTensorDesc), OUTPUT());
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  ut.TestPrecision();
}

TEST_F(l2_masked_scatter_test, aclnnInplaceMaskedScatter_1_2_float_nd_and_1_2_bool_nd_empty_tensor) {
  // self input
  const vector<int64_t>& selfShape = {1, 2};
  aclDataType selfDtype = ACL_FLOAT;
  aclFormat selfFormat = ACL_FORMAT_ND;
  // mask input
  const vector<int64_t>& maskShape = {1, 2};

  aclDataType maskDtype = ACL_BOOL;
  aclFormat maskFormat = ACL_FORMAT_ND;
  // source input
  const vector<int64_t>& sourceShape = {0, 2};
  aclDataType sourceDtype = selfDtype;
  aclFormat sourceFormat = ACL_FORMAT_ND;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
  auto maskTensorDesc = TensorDesc(maskShape, maskDtype, maskFormat);
  auto sourceTensorDesc = TensorDesc(sourceShape, sourceDtype, sourceFormat);

  auto ut = OP_API_UT(aclnnInplaceMaskedScatter, INPUT(selfTensorDesc, maskTensorDesc, sourceTensorDesc), OUTPUT());
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_masked_scatter_test, aclnnInplaceMaskedScatter_1_2_float16_nd_and_2_2_bool_nd_boundary_value) {
  // self input
  const vector<int64_t>& selfShape = {1, 2};
  aclDataType selfDtype = ACL_FLOAT16;
  aclFormat selfFormat = ACL_FORMAT_ND;
  // mask input
  const vector<int64_t>& maskShape = {1, 2};
  aclDataType maskDtype = ACL_BOOL;
  aclFormat maskFormat = ACL_FORMAT_ND;
  // source input
  const vector<int64_t>& sourceShape = {2};
  aclDataType sourceDtype = selfDtype;
  aclFormat sourceFormat = ACL_FORMAT_ND;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat).Value(vector<float>{65504.0, -65504.0});
  auto maskTensorDesc = TensorDesc(maskShape, maskDtype, maskFormat);
  auto sourceTensorDesc = TensorDesc(sourceShape, sourceDtype, sourceFormat);

  auto ut = OP_API_UT(aclnnInplaceMaskedScatter, INPUT(selfTensorDesc, maskTensorDesc, sourceTensorDesc), OUTPUT());
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  ut.TestPrecision();
}

// TEST_F(l2_masked_scatter_test, aclnnInplaceMaskedScatter_2_4_float32_nd_and_2_4_bool_nd_special_value) {
//  // self input
//  const vector<int64_t>& selfShape = {2,4};
//  aclDataType selfDtype = ACL_FLOAT;
//  aclFormat selfFormat = ACL_FORMAT_ND;
//  // mask input
//  const vector<int64_t>& maskShape = {2,4};
//  vector<bool> boolMask=genRandomBoolVector(maskShape);
//  aclDataType maskDtype = ACL_BOOL;
//  aclFormat maskFormat = ACL_FORMAT_ND;
//// source input
// const  vector<int64_t>& sourceShape={};
// aclDataType sourceDtype=selfDtype;
// aclFormat sourceFormat=ACL_FORMAT_ND;
//
// auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat).Value(vector<float>{float("inf"), float("-inf"),
// float("nan"), 0, 1, -1}); auto maskTensorDesc = TensorDesc(maskShape, maskDtype, maskFormat); auto sourceTensorDesc =
// TensorDesc(sourceShape, sourceDtype, sourceFormat);
//
// auto ut = OP_API_UT(aclnnInplaceMaskedScatter, INPUT(selfTensorDesc, maskTensorDesc,sourceTensorDesc), OUTPUT());
// // SAMPLE: only test GetWorkspaceSize
// uint64_t workspaceSize = 0;
// aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
// EXPECT_EQ(aclRet, ACL_SUCCESS);
//
// // SAMPLE: precision simulate

//}

TEST_F(l2_masked_scatter_test, aclnnInplaceMaskedScatter_5_4_float_nd_and_5_4_bool_nd_not_contiguous) {
  // self input
  const vector<int64_t>& selfShape = {5, 4};
  aclDataType selfDtype = ACL_FLOAT;
  aclFormat selfFormat = ACL_FORMAT_ND;
  const vector<int64_t>& selfViewDim = {1, 5};
  int64_t selfOffset = 0;
  const vector<int64_t>& selfStorageDim = {4, 5};
  // mask input
  const vector<int64_t>& maskShape = {5, 4};
  aclDataType maskDtype = ACL_BOOL;
  aclFormat maskFormat = ACL_FORMAT_ND;
  const vector<int64_t>& maskViewDim = {1, 5};
  int64_t maskOffset = 0;
  const vector<int64_t>& maskStorageDim = {4, 5};
  // source input
  const vector<int64_t>& sourceShape = {5, 4};
  aclDataType sourceDtype = selfDtype;
  aclFormat sourceFormat = ACL_FORMAT_ND;
  const vector<int64_t>& sourceViewDim = {1, 5};
  int64_t sourceOffset = 0;
  const vector<int64_t>& sourceStorageDim = {4, 5};

  auto selfTensorDesc =
      TensorDesc(selfShape, selfDtype, selfFormat, selfViewDim, selfOffset, selfStorageDim).ValueRange(-2, 2);
  auto maskTensorDesc = TensorDesc(maskShape, maskDtype, maskFormat, maskViewDim, maskOffset, maskStorageDim);
  auto sourceTensorDesc =
      TensorDesc(sourceShape, sourceDtype, sourceFormat, sourceViewDim, sourceOffset, sourceStorageDim);

  auto ut = OP_API_UT(aclnnInplaceMaskedScatter, INPUT(selfTensorDesc, maskTensorDesc, sourceTensorDesc), OUTPUT());
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 报错类型覆盖用例
// 空指针
TEST_F(l2_masked_scatter_test, aclnnInplaceMaskedScatter_input_nullptr) {
  auto tensor_desc = TensorDesc({10, 5}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut_self = OP_API_UT(aclnnInplaceMaskedScatter, INPUT((aclTensor*)nullptr, tensor_desc, tensor_desc), OUTPUT());
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut_self.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

  auto ut_mask = OP_API_UT(aclnnInplaceMaskedScatter, INPUT(tensor_desc, (aclTensor*)nullptr, tensor_desc), OUTPUT());
  // SAMPLE: only test GetWorkspaceSize
  aclRet = ut_mask.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

  auto ut_source = OP_API_UT(aclnnInplaceMaskedScatter, INPUT(tensor_desc, tensor_desc, (aclTensor*)nullptr), OUTPUT());
  // SAMPLE: only test GetWorkspaceSize
  aclRet = ut_source.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// self的维度小于mask
TEST_F(l2_masked_scatter_test, aclnnInplaceMaskedScatter_self_shape_smaller_than_mask) {
  // self input
  const vector<int64_t>& selfShape = {2, 4};
  aclDataType selfDtype = ACL_FLOAT;
  aclFormat selfFormat = ACL_FORMAT_ND;
  // mask input
  const vector<int64_t>& maskShape = {1, 2, 4};
  aclDataType maskDtype = ACL_BOOL;
  aclFormat maskFormat = ACL_FORMAT_ND;
  // source input
  const vector<int64_t>& sourceShape = {1, 2, 4};
  aclDataType sourceDtype = selfDtype;
  aclFormat sourceFormat = ACL_FORMAT_ND;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
  auto maskTensorDesc = TensorDesc(maskShape, maskDtype, maskFormat);
  auto sourceTensorDesc = TensorDesc(sourceShape, sourceDtype, sourceFormat);

  auto ut = OP_API_UT(aclnnInplaceMaskedScatter, INPUT(selfTensorDesc, maskTensorDesc, sourceTensorDesc), OUTPUT());
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// self和mask维度相等，但不能广播
TEST_F(l2_masked_scatter_test, aclnnInplaceMaskedScatter_self_can_not_broadcast_mask) {
  // self input
  const vector<int64_t>& selfShape = {3, 4};
  aclDataType selfDtype = ACL_FLOAT;
  aclFormat selfFormat = ACL_FORMAT_ND;
  // mask input
  const vector<int64_t>& maskShape = {2, 4};
  aclDataType maskDtype = ACL_BOOL;
  aclFormat maskFormat = ACL_FORMAT_ND;
  // source input
  const vector<int64_t>& sourceShape = {3, 4};
  aclDataType sourceDtype = selfDtype;
  aclFormat sourceFormat = ACL_FORMAT_ND;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
  auto maskTensorDesc = TensorDesc(maskShape, maskDtype, maskFormat);
  auto sourceTensorDesc = TensorDesc(sourceShape, sourceDtype, sourceFormat);

  auto ut = OP_API_UT(aclnnInplaceMaskedScatter, INPUT(selfTensorDesc, maskTensorDesc, sourceTensorDesc), OUTPUT());
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// self数据类型不支持uint64
TEST_F(l2_masked_scatter_test, aclnnInplaceMaskedScatter_self_dtype_with_uint64) {
  // self input
  const vector<int64_t>& selfShape = {2, 4};
  aclDataType selfDtype = ACL_UINT64;
  aclFormat selfFormat = ACL_FORMAT_ND;
  // mask input
  const vector<int64_t>& maskShape = {2, 4};
  aclDataType maskDtype = ACL_BOOL;
  aclFormat maskFormat = ACL_FORMAT_ND;
  // source input
  const vector<int64_t>& sourceShape = {2, 4};
  aclDataType sourceDtype = selfDtype;
  aclFormat sourceFormat = ACL_FORMAT_ND;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
  auto maskTensorDesc = TensorDesc(maskShape, maskDtype, maskFormat);
  auto sourceTensorDesc = TensorDesc(sourceShape, sourceDtype, sourceFormat);

  auto ut = OP_API_UT(aclnnInplaceMaskedScatter, INPUT(selfTensorDesc, maskTensorDesc, sourceTensorDesc), OUTPUT());
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// self和source的数据类型不同
TEST_F(l2_masked_scatter_test, aclnnInplaceMaskedScatter_self_and_source_diff_dtype) {
  // self input
  const vector<int64_t>& selfShape = {2, 4};
  aclDataType selfDtype = ACL_INT64;
  aclFormat selfFormat = ACL_FORMAT_NHWC;
  // mask input
  const vector<int64_t>& maskShape = {2, 4};
  aclDataType maskDtype = ACL_BOOL;
  aclFormat maskFormat = ACL_FORMAT_NHWC;
  // source input
  const vector<int64_t>& sourceShape = {2, 4};
  aclDataType sourceDtype = ACL_INT32;
  aclFormat sourceFormat = ACL_FORMAT_NHWC;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
  auto maskTensorDesc = TensorDesc(maskShape, maskDtype, maskFormat);
  auto sourceTensorDesc = TensorDesc(sourceShape, sourceDtype, sourceFormat);

  auto ut = OP_API_UT(aclnnInplaceMaskedScatter, INPUT(selfTensorDesc, maskTensorDesc, sourceTensorDesc), OUTPUT());
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// input的维度大于8
TEST_F(l2_masked_scatter_test, aclnnInplaceMaskedScatter_input_shape_out_of_8) {
  // self input
  const vector<int64_t>& selfShape = {2, 2, 2, 2, 2, 2, 2, 2, 2};
  aclDataType selfDtype = ACL_INT64;
  aclFormat selfFormat = ACL_FORMAT_NHWC;
  // mask input
  const vector<int64_t>& maskShape = {2, 2, 2, 2, 2, 2, 2, 2, 2};
  aclDataType maskDtype = ACL_BOOL;
  aclFormat maskFormat = ACL_FORMAT_NHWC;
  // source input
  const vector<int64_t>& sourceShape = {2, 2, 2, 2, 2, 2, 2, 2, 2};
  aclDataType sourceDtype = ACL_INT32;
  aclFormat sourceFormat = ACL_FORMAT_NHWC;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
  auto maskTensorDesc = TensorDesc(maskShape, maskDtype, maskFormat);
  auto sourceTensorDesc = TensorDesc(sourceShape, sourceDtype, sourceFormat);

  auto ut = OP_API_UT(aclnnInplaceMaskedScatter, INPUT(selfTensorDesc, maskTensorDesc, sourceTensorDesc), OUTPUT());
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}
TEST_F(l2_masked_scatter_test, ascend910B2_aclnnInplaceMaskedScatter_bf16_case) {
  // self input
  const vector<int64_t>& selfShape = {8};
  aclDataType selfDtype = ACL_BF16;
  aclFormat selfFormat = ACL_FORMAT_ND;
  // mask input
  const vector<int64_t>& maskShape = {8};
  aclDataType maskDtype = ACL_BOOL;
  aclFormat maskFormat = ACL_FORMAT_ND;
  // source input
  const vector<int64_t>& sourceShape = {8};
  aclDataType sourceDtype = selfDtype;
  aclFormat sourceFormat = ACL_FORMAT_ND;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
  auto maskTensorDesc = TensorDesc(maskShape, maskDtype, maskFormat);
  auto sourceTensorDesc = TensorDesc(sourceShape, sourceDtype, sourceFormat);

  auto ut = OP_API_UT(aclnnInplaceMaskedScatter, INPUT(selfTensorDesc, maskTensorDesc, sourceTensorDesc), OUTPUT());
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_masked_scatter_test, aclnnInplaceMaskedScatter_large_shape_case) {
  // self input
  const vector<int64_t>& selfShape = {64, 179, 32, 32};
  aclDataType selfDtype = ACL_FLOAT16;
  aclFormat selfFormat = ACL_FORMAT_ND;
  // mask input
  const vector<int64_t>& maskShape = {64, 179, 32, 32};
  aclDataType maskDtype = ACL_BOOL;
  aclFormat maskFormat = ACL_FORMAT_ND;
  // source input
  const vector<int64_t>& sourceShape = {64, 179, 32, 32};
  aclDataType sourceDtype = selfDtype;
  aclFormat sourceFormat = ACL_FORMAT_ND;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
  auto maskTensorDesc = TensorDesc(maskShape, maskDtype, maskFormat);
  auto sourceTensorDesc = TensorDesc(sourceShape, sourceDtype, sourceFormat);

  auto ut = OP_API_UT(aclnnInplaceMaskedScatter, INPUT(selfTensorDesc, maskTensorDesc, sourceTensorDesc), OUTPUT());
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

}
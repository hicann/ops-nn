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

#include "../../../op_host/op_api/aclnn_index_copy.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"
#include <cstdlib>
#include <ctime>

using namespace op;
using namespace std;

class l2_inplace_index_copy_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "inplace_index_copy_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "inplace_index_copy_test TearDown" << std::endl;
  }
};

// 基础用例
TEST_F(l2_inplace_index_copy_test, aclnnInplaceIndexCopy_base_case_1) {
  // self input
  const vector<int64_t>& selfShape = {5, 3};
  aclDataType selfDtype = ACL_INT32;
  aclFormat selfFormat = ACL_FORMAT_ND;
  // index input
  const vector<int64_t>& indexShape = {3};
  aclDataType indexDtype = ACL_INT64;
  aclFormat indexFormat = ACL_FORMAT_ND;
  // source input
  const vector<int64_t>& sourceShape = {3, 3};
  aclDataType sourceDtype = selfDtype;
  aclFormat sourceFormat = ACL_FORMAT_ND;

  auto selfTensorDesc =
      TensorDesc(selfShape, selfDtype, selfFormat).Value(vector<int32_t>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
  auto indexTensorDesc = TensorDesc(indexShape, indexDtype, indexFormat).Value(vector<int32_t>{0, 4, 2});
  auto sourceTensorDesc =
      TensorDesc(sourceShape, sourceDtype, sourceFormat).Value(vector<int32_t>{1, 2, 3, 4, 5, 6, 7, 8, 9});

  auto ut = OP_API_UT(aclnnInplaceIndexCopy, INPUT(selfTensorDesc, 0, indexTensorDesc, sourceTensorDesc), OUTPUT());

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  ut.TestPrecision();
}

TEST_F(l2_inplace_index_copy_test, aclnnInplaceIndexCopy_base_case_2) {
  // self input
  const vector<int64_t>& selfShape = {3, 5};
  aclDataType selfDtype = ACL_FLOAT;
  aclFormat selfFormat = ACL_FORMAT_ND;
  // index input
  const vector<int64_t>& indexShape = {3};
  aclDataType indexDtype = ACL_INT64;
  aclFormat indexFormat = ACL_FORMAT_ND;
  // source input
  const vector<int64_t>& sourceShape = {3, 3};
  aclDataType sourceDtype = ACL_FLOAT;
  aclFormat sourceFormat = ACL_FORMAT_ND;

  auto selfTensorDesc =
      TensorDesc(selfShape, selfDtype, selfFormat)
          .Value(vector<float>{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
  auto indexTensorDesc = TensorDesc(indexShape, indexDtype, indexFormat).Value(vector<int32_t>{0, 4, 2});
  auto sourceTensorDesc = TensorDesc(sourceShape, sourceDtype, sourceFormat)
                              .Value(vector<float>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0});

  auto ut = OP_API_UT(aclnnInplaceIndexCopy, INPUT(selfTensorDesc, 1, indexTensorDesc, sourceTensorDesc), OUTPUT());

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  ut.TestPrecision();
}

/* 各元素基本类型覆盖用例
 * 维度：1-8维
 * index基本数据类型：INT32(pytorch不支持，UT中不能做精度比较),INT64
 * 数据格式：ND、NCHW、NHWC、HWCN、NDHWC、NCDHW
 */

TEST_F(l2_inplace_index_copy_test, aclnnInplaceIndexCopy_1_2_3_4_5_6_7_8_float_nd) {
  // self input
  const vector<int64_t>& selfShape = {1, 2, 3, 4, 5, 6, 7, 8};
  aclDataType selfDtype = ACL_FLOAT;
  aclFormat selfFormat = ACL_FORMAT_ND;
  // index input
  const vector<int64_t>& indexShape = {8};
  aclDataType indexDtype = ACL_INT64;
  aclFormat indexFormat = ACL_FORMAT_ND;
  // source input
  const vector<int64_t>& sourceShape = selfShape;
  aclDataType sourceDtype = selfDtype;
  aclFormat sourceFormat = ACL_FORMAT_ND;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
  auto indexTensorDesc = TensorDesc(indexShape, indexDtype, indexFormat).Value(vector<int32_t>{7, 6, 5, 4, 3, 2, 1, 0});
  auto sourceTensorDesc = TensorDesc(sourceShape, sourceDtype, sourceFormat);

  auto ut = OP_API_UT(aclnnInplaceIndexCopy, INPUT(selfTensorDesc, 7, indexTensorDesc, sourceTensorDesc), OUTPUT());

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  ut.TestPrecision();
}

TEST_F(l2_inplace_index_copy_test, aclnnInplaceIndexCopy_1_2_3_4_5_6_7_float16_nd) {
  // self input
  const vector<int64_t>& selfShape = {1, 2, 3, 4, 5, 6, 7};
  aclDataType selfDtype = ACL_FLOAT16;
  aclFormat selfFormat = ACL_FORMAT_ND;
  // index input
  const vector<int64_t>& indexShape = {7};
  aclDataType indexDtype = ACL_INT64;
  aclFormat indexFormat = ACL_FORMAT_ND;
  // source input
  const vector<int64_t>& sourceShape = selfShape;
  aclDataType sourceDtype = selfDtype;
  aclFormat sourceFormat = ACL_FORMAT_ND;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
  auto indexTensorDesc = TensorDesc(indexShape, indexDtype, indexFormat).Value(vector<int32_t>{6, 5, 4, 3, 2, 1, 0});
  auto sourceTensorDesc = TensorDesc(sourceShape, sourceDtype, sourceFormat);

  auto ut = OP_API_UT(aclnnInplaceIndexCopy, INPUT(selfTensorDesc, 6, indexTensorDesc, sourceTensorDesc), OUTPUT());

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_inplace_index_copy_test, aclnnInplaceIndexCopy_1_2_3_4_int32_nchw) {
  // self input
  const vector<int64_t>& selfShape = {1, 2, 3, 4};
  aclDataType selfDtype = ACL_INT32;
  aclFormat selfFormat = ACL_FORMAT_NCHW;
  // index input
  const vector<int64_t>& indexShape = {3};
  aclDataType indexDtype = ACL_INT64;
  aclFormat indexFormat = ACL_FORMAT_NCHW;
  // source input
  const vector<int64_t>& sourceShape = selfShape;
  aclDataType sourceDtype = selfDtype;
  aclFormat sourceFormat = ACL_FORMAT_NCHW;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat).ValueRange(-1, 1);
  auto indexTensorDesc = TensorDesc(indexShape, indexDtype, indexFormat).Value(vector<int32_t>{2, 1, 0});
  auto sourceTensorDesc = TensorDesc(sourceShape, sourceDtype, sourceFormat);

  auto ut = OP_API_UT(aclnnInplaceIndexCopy, INPUT(selfTensorDesc, 2, indexTensorDesc, sourceTensorDesc), OUTPUT());

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  ut.TestPrecision();
}

TEST_F(l2_inplace_index_copy_test, aclnnInplaceIndexCopy_1_2_3_4_float_hwcn) {
  // self input
  const vector<int64_t>& selfShape = {1, 2, 3};
  aclDataType selfDtype = ACL_FLOAT;
  aclFormat selfFormat = ACL_FORMAT_HWCN;
  // index input
  const vector<int64_t>& indexShape = {3};
  aclDataType indexDtype = ACL_INT64;
  aclFormat indexFormat = ACL_FORMAT_HWCN;
  // source input
  const vector<int64_t>& sourceShape = selfShape;
  aclDataType sourceDtype = selfDtype;
  aclFormat sourceFormat = ACL_FORMAT_HWCN;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
  auto indexTensorDesc = TensorDesc(indexShape, indexDtype, indexFormat).Value(vector<int32_t>{2, 1, 0});
  auto sourceTensorDesc = TensorDesc(sourceShape, sourceDtype, sourceFormat);

  auto ut = OP_API_UT(aclnnInplaceIndexCopy, INPUT(selfTensorDesc, 2, indexTensorDesc, sourceTensorDesc), OUTPUT());

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  ut.TestPrecision();
}

TEST_F(l2_inplace_index_copy_test, aclnnInplaceIndexCopy_1_2_3_4_5_float16_ndhwc) {
  // self input
  const vector<int64_t>& selfShape = {1, 2, 3, 4, 5};
  aclDataType selfDtype = ACL_FLOAT16;
  aclFormat selfFormat = ACL_FORMAT_NDHWC;
  // index input
  const vector<int64_t>& indexShape = {4};
  aclDataType indexDtype = ACL_INT64;
  aclFormat indexFormat = ACL_FORMAT_NDHWC;
  // source input
  const vector<int64_t>& sourceShape = selfShape;
  aclDataType sourceDtype = selfDtype;
  aclFormat sourceFormat = ACL_FORMAT_NDHWC;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
  auto indexTensorDesc = TensorDesc(indexShape, indexDtype, indexFormat).Value(vector<int32_t>{3, 2, 1, 0});
  auto sourceTensorDesc = TensorDesc(sourceShape, sourceDtype, sourceFormat);

  auto ut = OP_API_UT(aclnnInplaceIndexCopy, INPUT(selfTensorDesc, 3, indexTensorDesc, sourceTensorDesc), OUTPUT());

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_inplace_index_copy_test, aclnnInplaceIndexCopy_1_2_3_4_5_double_ncdhw) {
  // self input
  const vector<int64_t>& selfShape = {1, 2, 3, 4, 5};
  aclDataType selfDtype = ACL_INT32;
  aclFormat selfFormat = ACL_FORMAT_NCDHW;
  // index input
  const vector<int64_t>& indexShape = {5};
  aclDataType indexDtype = ACL_INT64;
  aclFormat indexFormat = ACL_FORMAT_NCDHW;
  // source input
  const vector<int64_t>& sourceShape = selfShape;
  aclDataType sourceDtype = selfDtype;
  aclFormat sourceFormat = ACL_FORMAT_NCDHW;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
  auto indexTensorDesc = TensorDesc(indexShape, indexDtype, indexFormat).Value(vector<int32_t>{4, 3, 2, 1, 0});
  auto sourceTensorDesc = TensorDesc(sourceShape, sourceDtype, sourceFormat);

  auto ut = OP_API_UT(aclnnInplaceIndexCopy, INPUT(selfTensorDesc, 4, indexTensorDesc, sourceTensorDesc), OUTPUT());

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  ut.TestPrecision();
}

TEST_F(l2_inplace_index_copy_test, aclnnInplaceIndexCopy_1_2_3_4_5_6_7_8_int32_nd) {
  // self input
  const vector<int64_t>& selfShape = {1, 2, 3, 4, 5, 6, 7, 8};
  aclDataType selfDtype = ACL_INT32;
  aclFormat selfFormat = ACL_FORMAT_ND;
  // index input
  const vector<int64_t>& indexShape = {8};
  aclDataType indexDtype = ACL_INT64;
  aclFormat indexFormat = ACL_FORMAT_ND;
  // source input
  const vector<int64_t>& sourceShape = selfShape;
  aclDataType sourceDtype = selfDtype;
  aclFormat sourceFormat = ACL_FORMAT_ND;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
  auto indexTensorDesc = TensorDesc(indexShape, indexDtype, indexFormat).Value(vector<int32_t>{7, 6, 5, 4, 3, 2, 1, 0});
  auto sourceTensorDesc = TensorDesc(sourceShape, sourceDtype, sourceFormat);

  auto ut = OP_API_UT(aclnnInplaceIndexCopy, INPUT(selfTensorDesc, 7, indexTensorDesc, sourceTensorDesc), OUTPUT());

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  ut.TestPrecision();
}

TEST_F(l2_inplace_index_copy_test, aclnnInplaceIndexCopy_1_2_int64_nd) {
  // self input
  const vector<int64_t>& selfShape = {1, 2};
  aclDataType selfDtype = ACL_INT32;
  aclFormat selfFormat = ACL_FORMAT_ND;
  // index input
  const vector<int64_t>& indexShape = {2};
  aclDataType indexDtype = ACL_INT64;
  aclFormat indexFormat = ACL_FORMAT_ND;
  // source input
  const vector<int64_t>& sourceShape = selfShape;
  aclDataType sourceDtype = selfDtype;
  aclFormat sourceFormat = ACL_FORMAT_ND;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
  auto indexTensorDesc = TensorDesc(indexShape, indexDtype, indexFormat).Value(vector<int32_t>{1, 0});
  auto sourceTensorDesc = TensorDesc(sourceShape, sourceDtype, sourceFormat);

  auto ut = OP_API_UT(aclnnInplaceIndexCopy, INPUT(selfTensorDesc, 1, indexTensorDesc, sourceTensorDesc), OUTPUT());

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  ut.TestPrecision();
}
TEST_F(l2_inplace_index_copy_test, aclnnInplaceIndexCopy_1_2_3_4_5_float16_ndhwc_with_index_int32) {
  // self input
  const vector<int64_t>& selfShape = {1, 2, 3, 4, 5};
  aclDataType selfDtype = ACL_FLOAT16;
  aclFormat selfFormat = ACL_FORMAT_NDHWC;
  // index input
  const vector<int64_t>& indexShape = {4};
  aclDataType indexDtype = ACL_INT64;
  aclFormat indexFormat = ACL_FORMAT_NDHWC;
  // source input
  const vector<int64_t>& sourceShape = selfShape;
  aclDataType sourceDtype = selfDtype;
  aclFormat sourceFormat = ACL_FORMAT_NDHWC;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
  auto indexTensorDesc = TensorDesc(indexShape, indexDtype, indexFormat).Value(vector<int32_t>{3, 2, 1, 0});
  auto sourceTensorDesc = TensorDesc(sourceShape, sourceDtype, sourceFormat);

  auto ut = OP_API_UT(aclnnInplaceIndexCopy, INPUT(selfTensorDesc, 3, indexTensorDesc, sourceTensorDesc), OUTPUT());

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  ut.TestPrecision();
}

// 各元素特殊类型覆盖用例
// 空tensor   self ,index ,source 都为空时，空进空出
TEST_F(l2_inplace_index_copy_test, aclnnInplaceIndexCopy_float_nd_empty_tensor) {
  // self input
  const vector<int64_t>& selfShape = {0};
  aclDataType selfDtype = ACL_FLOAT;
  aclFormat selfFormat = ACL_FORMAT_ND;
  // index input
  const vector<int64_t>& indexShape = {0};

  aclDataType indexDtype = ACL_INT64;
  aclFormat indexFormat = ACL_FORMAT_ND;
  // source input
  const vector<int64_t>& sourceShape = {0};
  aclDataType sourceDtype = selfDtype;
  aclFormat sourceFormat = ACL_FORMAT_ND;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
  auto indexTensorDesc = TensorDesc(indexShape, indexDtype, indexFormat);
  auto sourceTensorDesc = TensorDesc(sourceShape, sourceDtype, sourceFormat);

  auto ut = OP_API_UT(aclnnInplaceIndexCopy, INPUT(selfTensorDesc, 0, indexTensorDesc, sourceTensorDesc), OUTPUT());

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  ut.TestPrecision();
}

// 边界值
TEST_F(l2_inplace_index_copy_test, aclnnInplaceIndexCopy_float32_nd_boundary_value) {
  // self input
  const vector<int64_t>& selfShape = {1, 2};
  aclDataType selfDtype = ACL_FLOAT;
  aclFormat selfFormat = ACL_FORMAT_ND;
  // index input
  const vector<int64_t>& indexShape = {2};
  aclDataType indexDtype = ACL_INT64;
  aclFormat indexFormat = ACL_FORMAT_ND;
  // source input3
  const vector<int64_t>& sourceShape = selfShape;
  aclDataType sourceDtype = selfDtype;
  aclFormat sourceFormat = ACL_FORMAT_ND;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat).Value(vector<float>{65504.0, -65504.0});
  auto indexTensorDesc = TensorDesc(indexShape, indexDtype, indexFormat).Value(vector<int32_t>{1, 0});
  auto sourceTensorDesc = TensorDesc(sourceShape, sourceDtype, sourceFormat);

  auto ut = OP_API_UT(aclnnInplaceIndexCopy, INPUT(selfTensorDesc, 1, indexTensorDesc, sourceTensorDesc), OUTPUT());

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  ut.TestPrecision();
}

// 特殊值：nan,inf
// TEST_F(l2_inplace_index_copy_test, aclnnInplaceIndexCopy_2_4_float32_nd_and_2_4_bool_nd_special_value) {
//  // self input
//  const vector<int64_t>& selfShape = {2,4};
//  aclDataType selfDtype = ACL_FLOAT;
//  aclFormat selfFormat = ACL_FORMAT_ND;
//  // index input
//  const vector<int64_t>& indexShape = {2,4};
//  vector<bool> boolindex=genRandomBoolVector(indexShape);
//  aclDataType indexDtype = ACL_INT64;
//  aclFormat indexFormat = ACL_FORMAT_ND;
//// source input
// const  vector<int64_t>& sourceShape={};
// aclDataType sourceDtype=selfDtype;
// aclFormat sourceFormat=ACL_FORMAT_ND;
//
// auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat).Value(vector<float>{float("inf"), float("-inf"),
// float("nan"), 0, 1, -1}); auto indexTensorDesc = TensorDesc(indexShape, indexDtype, indexFormat); auto
// sourceTensorDesc = TensorDesc(sourceShape, sourceDtype, sourceFormat);
//
// auto ut = OP_API_UT(aclnnInplaceIndexCopy, INPUT(selfTensorDesc, indexTensorDesc,sourceTensorDesc), OUTPUT());
//
// uint64_t workspaceSize = 0;
// aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
// EXPECT_EQ(aclRet, ACL_SUCCESS);
//
//

//}

// 不连续
TEST_F(l2_inplace_index_copy_test, aclnnInplaceIndexCopy_5_4_float_nd_not_contiguous) {
  // self input
  const vector<int64_t>& selfShape = {5, 4};
  aclDataType selfDtype = ACL_FLOAT;
  aclFormat selfFormat = ACL_FORMAT_ND;
  const vector<int64_t>& selfViewDim = {1, 5};
  int64_t selfOffset = 0;
  const vector<int64_t>& selfStorageDim = {4, 5};
  // index input
  const vector<int64_t>& indexShape = {5};
  aclDataType indexDtype = ACL_INT64;
  aclFormat indexFormat = ACL_FORMAT_ND;
  // source input
  const vector<int64_t>& sourceShape = {5, 4};
  aclDataType sourceDtype = selfDtype;
  aclFormat sourceFormat = ACL_FORMAT_ND;
  const vector<int64_t>& sourceViewDim = {1, 5};
  int64_t sourceOffset = 0;
  const vector<int64_t>& sourceStorageDim = {4, 5};

  auto selfTensorDesc =
      TensorDesc(selfShape, selfDtype, selfFormat, selfViewDim, selfOffset, selfStorageDim).ValueRange(-2, 2);

  auto indexTensorDesc = TensorDesc(indexShape, indexDtype, indexFormat).Value(vector<int32_t>{4, 3, 2, 1, 0});

  auto sourceTensorDesc =
      TensorDesc(sourceShape, sourceDtype, sourceFormat, sourceViewDim, sourceOffset, sourceStorageDim);

  auto ut = OP_API_UT(aclnnInplaceIndexCopy, INPUT(selfTensorDesc, 0, indexTensorDesc, sourceTensorDesc), OUTPUT());

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 报错类型覆盖用例
// 空指针
TEST_F(l2_inplace_index_copy_test, aclnnInplaceIndexCopy_input_nullptr) {
  auto tensor_desc = TensorDesc({10}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut_self_nullptr =
      OP_API_UT(aclnnInplaceIndexCopy, INPUT((aclTensor*)nullptr, 0, tensor_desc, tensor_desc), OUTPUT());

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut_self_nullptr.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

  auto ut_index_nullptr =
      OP_API_UT(aclnnInplaceIndexCopy, INPUT(tensor_desc, 0, (aclTensor*)nullptr, tensor_desc), OUTPUT());

  aclRet = ut_index_nullptr.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

  auto ut_source_nullptr =
      OP_API_UT(aclnnInplaceIndexCopy, INPUT(tensor_desc, 0, tensor_desc, (aclTensor*)nullptr), OUTPUT());

  aclRet = ut_source_nullptr.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_inplace_index_copy_test, ascend910B2_aclnnInplaceIndexCopy_self_dtype_error) {
  // self input
  const vector<int64_t>& selfShape = {2, 4};
  aclDataType selfDtype = ACL_BF16;
  aclFormat selfFormat = ACL_FORMAT_ND;
  // index input
  const vector<int64_t>& indexShape = {4};
  aclDataType indexDtype = ACL_INT64;
  aclFormat indexFormat = ACL_FORMAT_ND;
  // source input
  const vector<int64_t>& sourceShape = selfShape;
  aclDataType sourceDtype = selfDtype;
  aclFormat sourceFormat = ACL_FORMAT_ND;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
  auto indexTensorDesc = TensorDesc(indexShape, indexDtype, indexFormat).Value(vector<int32_t>{3, 2, 1, 0});
  auto sourceTensorDesc = TensorDesc(sourceShape, sourceDtype, sourceFormat);

  auto ut = OP_API_UT(aclnnInplaceIndexCopy, INPUT(selfTensorDesc, 1, indexTensorDesc, sourceTensorDesc), OUTPUT());

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// index类型不满足
TEST_F(l2_inplace_index_copy_test, aclnnInplaceIndexCopy_index_dtype_error) {
  // self input
  const vector<int64_t>& selfShape = {2, 4};
  aclDataType selfDtype = ACL_FLOAT;
  aclFormat selfFormat = ACL_FORMAT_ND;
  // index input
  const vector<int64_t>& indexShape = {4};
  aclDataType indexDtype = ACL_INT8;
  aclFormat indexFormat = ACL_FORMAT_ND;
  // source input
  const vector<int64_t>& sourceShape = selfShape;
  aclDataType sourceDtype = selfDtype;
  aclFormat sourceFormat = ACL_FORMAT_ND;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
  auto indexTensorDesc = TensorDesc(indexShape, indexDtype, indexFormat).Value(vector<int32_t>{3, 2, 1, 0});
  auto sourceTensorDesc = TensorDesc(sourceShape, sourceDtype, sourceFormat);

  auto ut = OP_API_UT(aclnnInplaceIndexCopy, INPUT(selfTensorDesc, 1, indexTensorDesc, sourceTensorDesc), OUTPUT());

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// self类型和source不一致
TEST_F(l2_inplace_index_copy_test, aclnnInplaceIndexCopy_self_and_source_dtype_non_consistent) {
  // self input
  const vector<int64_t>& selfShape = {2, 4};
  aclDataType selfDtype = ACL_FLOAT;
  aclFormat selfFormat = ACL_FORMAT_ND;
  // index input
  const vector<int64_t>& indexShape = {4};
  aclDataType indexDtype = ACL_INT64;
  aclFormat indexFormat = ACL_FORMAT_ND;
  // source input
  const vector<int64_t>& sourceShape = selfShape;
  aclDataType sourceDtype = ACL_FLOAT16;
  aclFormat sourceFormat = ACL_FORMAT_ND;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
  auto indexTensorDesc = TensorDesc(indexShape, indexDtype, indexFormat).Value(vector<int32_t>{3, 2, 1, 0});
  auto sourceTensorDesc = TensorDesc(sourceShape, sourceDtype, sourceFormat);

  auto ut = OP_API_UT(aclnnInplaceIndexCopy, INPUT(selfTensorDesc, 1, indexTensorDesc, sourceTensorDesc), OUTPUT());

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// self形状大于8
TEST_F(l2_inplace_index_copy_test, aclnnInplaceIndexCopy_self_shape_out_of_8) {
  // self input
  const vector<int64_t>& selfShape = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  aclDataType selfDtype = ACL_FLOAT;
  aclFormat selfFormat = ACL_FORMAT_ND;
  // index input
  const vector<int64_t>& indexShape = {4};
  aclDataType indexDtype = ACL_INT64;
  aclFormat indexFormat = ACL_FORMAT_ND;
  // source input
  const vector<int64_t>& sourceShape = selfShape;
  aclDataType sourceDtype = selfDtype;
  aclFormat sourceFormat = ACL_FORMAT_ND;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
  auto indexTensorDesc = TensorDesc(indexShape, indexDtype, indexFormat).Value(vector<int32_t>{3, 2, 1, 0});
  auto sourceTensorDesc = TensorDesc(sourceShape, sourceDtype, sourceFormat);

  auto ut = OP_API_UT(aclnnInplaceIndexCopy, INPUT(selfTensorDesc, 3, indexTensorDesc, sourceTensorDesc), OUTPUT());

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// index形状大于1
TEST_F(l2_inplace_index_copy_test, aclnnInplaceIndexCopy_index_shape_out_of_1) {
  // self input
  const vector<int64_t>& selfShape = {1, 2, 3, 4};
  aclDataType selfDtype = ACL_FLOAT;
  aclFormat selfFormat = ACL_FORMAT_ND;
  // index input
  const vector<int64_t>& indexShape = {2, 4};
  aclDataType indexDtype = ACL_INT64;
  aclFormat indexFormat = ACL_FORMAT_ND;
  // source input
  const vector<int64_t>& sourceShape = selfShape;
  aclDataType sourceDtype = selfDtype;
  aclFormat sourceFormat = ACL_FORMAT_ND;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
  auto indexTensorDesc = TensorDesc(indexShape, indexDtype, indexFormat).Value(vector<int32_t>{7, 6, 5, 4, 3, 2, 1, 0});
  auto sourceTensorDesc = TensorDesc(sourceShape, sourceDtype, sourceFormat);

  auto ut = OP_API_UT(aclnnInplaceIndexCopy, INPUT(selfTensorDesc, 3, indexTensorDesc, sourceTensorDesc), OUTPUT());

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// source is scalar,index must have one element
TEST_F(l2_inplace_index_copy_test, aclnnInplaceIndexCopy_source_is_scalar_index_must_have_one_element) {
  // self input
  const vector<int64_t>& selfShape = {1, 2, 3, 4};
  aclDataType selfDtype = ACL_FLOAT;
  aclFormat selfFormat = ACL_FORMAT_ND;
  // index input
  const vector<int64_t>& indexShape = {4};
  aclDataType indexDtype = ACL_INT64;
  aclFormat indexFormat = ACL_FORMAT_ND;
  // source input
  const vector<int64_t>& sourceShape = {0};
  aclDataType sourceDtype = selfDtype;
  aclFormat sourceFormat = ACL_FORMAT_ND;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
  auto indexTensorDesc = TensorDesc(indexShape, indexDtype, indexFormat).Value(vector<int32_t>{3, 2, 1, 0});
  auto sourceTensorDesc = TensorDesc(sourceShape, sourceDtype, sourceFormat).Value(vector<float>{3.0});

  auto ut = OP_API_UT(aclnnInplaceIndexCopy, INPUT(selfTensorDesc, 3, indexTensorDesc, sourceTensorDesc), OUTPUT());

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// when self and source are not scalar,their dimensionality must be same
TEST_F(l2_inplace_index_copy_test, aclnnInplaceIndexCopy_self_and_source_dimensionality_not_same) {
  // self input
  const vector<int64_t>& selfShape = {4, 4, 4};
  aclDataType selfDtype = ACL_FLOAT;
  aclFormat selfFormat = ACL_FORMAT_ND;
  // index input
  const vector<int64_t>& indexShape = {4};
  aclDataType indexDtype = ACL_INT64;
  aclFormat indexFormat = ACL_FORMAT_ND;
  // source input
  const vector<int64_t>& sourceShape = {4, 4};
  aclDataType sourceDtype = selfDtype;
  aclFormat sourceFormat = ACL_FORMAT_ND;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
  auto indexTensorDesc = TensorDesc(indexShape, indexDtype, indexFormat).Value(vector<int32_t>{3, 2, 1, 0});
  auto sourceTensorDesc = TensorDesc(sourceShape, sourceDtype, sourceFormat);

  auto ut = OP_API_UT(aclnnInplaceIndexCopy, INPUT(selfTensorDesc, 0, indexTensorDesc, sourceTensorDesc), OUTPUT());

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

//  self and source  must have same slice shapes
TEST_F(l2_inplace_index_copy_test, aclnnInplaceIndexCopy_self_and_source_slice_shape_not_same) {
  // self input
  const vector<int64_t>& selfShape = {4, 4, 4};
  aclDataType selfDtype = ACL_FLOAT;
  aclFormat selfFormat = ACL_FORMAT_ND;
  // index input
  const vector<int64_t>& indexShape = {4};
  aclDataType indexDtype = ACL_INT64;
  aclFormat indexFormat = ACL_FORMAT_ND;
  // source input
  const vector<int64_t>& sourceShape = {4, 3, 4};
  aclDataType sourceDtype = selfDtype;
  aclFormat sourceFormat = ACL_FORMAT_ND;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
  auto indexTensorDesc = TensorDesc(indexShape, indexDtype, indexFormat).Value(vector<int32_t>{3, 2, 1, 0});
  auto sourceTensorDesc = TensorDesc(sourceShape, sourceDtype, sourceFormat);

  auto ut = OP_API_UT(aclnnInplaceIndexCopy, INPUT(selfTensorDesc, 0, indexTensorDesc, sourceTensorDesc), OUTPUT());

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

//  index element num must be equal to source size in the dim
TEST_F(l2_inplace_index_copy_test, aclnnInplaceIndexCopy_index_element_num_not_equal_to_source_size_in_the_dim) {
  // self input
  const vector<int64_t>& selfShape = {4, 4, 4};
  aclDataType selfDtype = ACL_FLOAT;
  aclFormat selfFormat = ACL_FORMAT_ND;
  // index input
  const vector<int64_t>& indexShape = {3};
  aclDataType indexDtype = ACL_INT64;
  aclFormat indexFormat = ACL_FORMAT_ND;
  // source input
  const vector<int64_t>& sourceShape = {4, 4, 4};
  aclDataType sourceDtype = selfDtype;
  aclFormat sourceFormat = ACL_FORMAT_ND;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
  auto indexTensorDesc = TensorDesc(indexShape, indexDtype, indexFormat).Value(vector<int32_t>{2, 1, 0});
  auto sourceTensorDesc = TensorDesc(sourceShape, sourceDtype, sourceFormat);

  auto ut = OP_API_UT(aclnnInplaceIndexCopy, INPUT(selfTensorDesc, 0, indexTensorDesc, sourceTensorDesc), OUTPUT());

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_inplace_index_copy_test, aclnnInplaceIndexCopy__self_0_dim_tensor_aicore) {
  // self input
  const vector<int64_t>& selfShape = {};
  aclDataType selfDtype = ACL_FLOAT;
  aclFormat selfFormat = ACL_FORMAT_ND;
  // index input
  const vector<int64_t>& indexShape = {1};

  aclDataType indexDtype = ACL_INT64;
  aclFormat indexFormat = ACL_FORMAT_ND;
  // source input
  const vector<int64_t>& sourceShape = {};
  aclDataType sourceDtype = selfDtype;
  aclFormat sourceFormat = ACL_FORMAT_ND;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat).Value(vector<float>{2});
  auto indexTensorDesc = TensorDesc(indexShape, indexDtype, indexFormat).Value(vector<float>{0});
  auto sourceTensorDesc = TensorDesc(sourceShape, sourceDtype, sourceFormat).Value(vector<float>{5});

  auto ut = OP_API_UT(aclnnInplaceIndexCopy, INPUT(selfTensorDesc, 0, indexTensorDesc, sourceTensorDesc), OUTPUT());

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  ut.TestPrecision();
}

TEST_F(l2_inplace_index_copy_test, aclnnInplaceIndexCopy__self_0_dim_tensor_aicpu) {
  // self input
  const vector<int64_t>& selfShape = {};
  aclDataType selfDtype = ACL_INT32;
  aclFormat selfFormat = ACL_FORMAT_ND;
  // index input
  const vector<int64_t>& indexShape = {1};

  aclDataType indexDtype = ACL_INT64;
  aclFormat indexFormat = ACL_FORMAT_ND;
  // source input
  const vector<int64_t>& sourceShape = {};
  aclDataType sourceDtype = selfDtype;
  aclFormat sourceFormat = ACL_FORMAT_ND;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat).Value(vector<int32_t>{2});
  auto indexTensorDesc = TensorDesc(indexShape, indexDtype, indexFormat).Value(vector<int32_t>{0});
  auto sourceTensorDesc = TensorDesc(sourceShape, sourceDtype, sourceFormat).Value(vector<int32_t>{5});

  auto ut = OP_API_UT(aclnnInplaceIndexCopy, INPUT(selfTensorDesc, 0, indexTensorDesc, sourceTensorDesc), OUTPUT());

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  ut.TestPrecision();
}
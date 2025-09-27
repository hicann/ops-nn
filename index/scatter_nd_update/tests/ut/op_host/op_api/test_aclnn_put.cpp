/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <array>
#include <vector>
#include "gtest/gtest.h"

#include "level2/aclnn_put.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"
#include <cstdlib>
#include <ctime>

using namespace op;
using namespace std;

class l2_inplace_put_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "inplace_put_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "inplace_put_test TearDown" << std::endl;
  }
};

// 基础用例
TEST_F(l2_inplace_put_test, Ascend910_9589_aclnnInplacePut_base_case_1) {
  // self input
  const vector<int64_t>& selfShape = {8};
  aclDataType selfDtype = ACL_INT32;
  aclFormat selfFormat = ACL_FORMAT_ND;
  // index input
  const vector<int64_t>& indexShape = {8};
  aclDataType indexDtype = ACL_INT64;
  aclFormat indexFormat = ACL_FORMAT_ND;
  // source input
  const vector<int64_t>& sourceShape = {8};
  aclDataType sourceDtype = selfDtype;
  aclFormat sourceFormat = ACL_FORMAT_ND;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat).Value(vector<int32_t>{1, 1, 1, 1, 1, 1, 1, 1});
  auto indexTensorDesc = TensorDesc(indexShape, indexDtype, indexFormat).Value(vector<int32_t>{0, 1, 2, 3, 4, 5, 6, 7});
  auto sourceTensorDesc =
      TensorDesc(sourceShape, sourceDtype, sourceFormat).Value(vector<int32_t>{0, 0, 0, 0, 0, 0, 0, 0});

  auto ut = OP_API_UT(aclnnInplacePut, INPUT(selfTensorDesc, indexTensorDesc, sourceTensorDesc, false), OUTPUT());

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  ut.TestPrecision();
}

TEST_F(l2_inplace_put_test, aclnnInplacePut_base_case_2) {
  // self input
  const vector<int64_t>& selfShape = {2, 4};
  aclDataType selfDtype = ACL_FLOAT;
  aclFormat selfFormat = ACL_FORMAT_ND;
  // index input
  const vector<int64_t>& indexShape = {4};
  aclDataType indexDtype = ACL_INT64;
  aclFormat indexFormat = ACL_FORMAT_ND;
  // source input
  const vector<int64_t>& sourceShape = {4};
  aclDataType sourceDtype = ACL_FLOAT;
  aclFormat sourceFormat = ACL_FORMAT_ND;

  auto selfTensorDesc =
      TensorDesc(selfShape, selfDtype, selfFormat).Value(vector<float>{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0});
  auto indexTensorDesc = TensorDesc(indexShape, indexDtype, indexFormat).Value(vector<int32_t>{0, 1, 2, 3});
  auto sourceTensorDesc = TensorDesc(sourceShape, sourceDtype, sourceFormat).Value(vector<float>{0.0, 0.0, 0.0, 0.0});

  auto ut = OP_API_UT(aclnnInplacePut, INPUT(selfTensorDesc, indexTensorDesc, sourceTensorDesc, true), OUTPUT());

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  ut.TestPrecision();
}

// 非连续
TEST_F(l2_inplace_put_test, aclnnInplacePut_uncontiguous_case) {
  // self input
  const vector<int64_t>& selfShape = {4, 2};
  aclDataType selfDtype = ACL_FLOAT;
  aclFormat selfFormat = ACL_FORMAT_ND;
  // index input
  const vector<int64_t>& indexShape = {4, 2};
  aclDataType indexDtype = ACL_INT64;
  aclFormat indexFormat = ACL_FORMAT_ND;
  // source input
  const vector<int64_t>& sourceShape = {4 ,2};
  aclDataType sourceDtype = ACL_FLOAT;
  aclFormat sourceFormat = ACL_FORMAT_ND;

  auto selfTensorDesc =
      TensorDesc(selfShape, selfDtype, selfFormat, {1, 2}, 0, {2, 4});
  auto indexTensorDesc = TensorDesc(indexShape, indexDtype, indexFormat);
  auto sourceTensorDesc = TensorDesc(sourceShape, sourceDtype, sourceFormat);

  auto ut = OP_API_UT(aclnnInplacePut, INPUT(selfTensorDesc, indexTensorDesc, sourceTensorDesc, true), OUTPUT());

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

inline int64_t sizes(const vector<int64_t>& arr) {
  int64_t result = 1;
  for (const auto& elem : arr) {
    if (elem) {
      result *= elem;
    }
  }
  return result;
}

vector<int32_t> genInt32Vector(const vector<int64_t>& self, const vector<int64_t>& index) {
  int64_t max = sizes(self);
  int64_t size = sizes(index);
  srand(time(NULL));
  vector<int32_t> result(size);
  for (int i = 0; i < size; ++i) {
    result[i] = rand() % (max);
  }
  return result;
}

vector<int64_t> genInt64Vector(const vector<int64_t>& self, const vector<int64_t>& index) {
  int64_t max = sizes(self);
  int64_t size = sizes(index);
  srand(time(NULL));
  vector<int64_t> result(size);
  for (int i = 0; i < size; ++i) {
    result[i] = rand() % (max);
  }
  return result;
}

// 简单随机用例
TEST_F(l2_inplace_put_test, Ascend910B2_aclnnInplacePut_8_int32_nd_and_8_bool_nd_simple_random_case_1) {
  // self input
  const vector<int64_t>& selfShape = {8};
  aclDataType selfDtype = ACL_INT32;
  aclFormat selfFormat = ACL_FORMAT_ND;
  // index input
  const vector<int64_t>& indexShape = {8};
  aclDataType indexDtype = ACL_INT64;
  aclFormat indexFormat = ACL_FORMAT_ND;
  // source input
  const vector<int64_t>& sourceShape = {8};
  aclDataType sourceDtype = selfDtype;
  aclFormat sourceFormat = ACL_FORMAT_ND;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
  auto indexTensorDesc = TensorDesc(indexShape, indexDtype, indexFormat).Value(genInt64Vector(selfShape, indexShape));
  auto sourceTensorDesc = TensorDesc(sourceShape, sourceDtype, sourceFormat);

  auto ut = OP_API_UT(aclnnInplacePut, INPUT(selfTensorDesc, indexTensorDesc, sourceTensorDesc, true), OUTPUT());

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  ut.TestPrecision();
}

TEST_F(l2_inplace_put_test, aclnnInplacePut_8_8_float_nd_and_8_8_bool_nd_simple_random_case_2) {
  // self input
  const vector<int64_t>& selfShape = {8, 8};
  aclDataType selfDtype = ACL_FLOAT;
  aclFormat selfFormat = ACL_FORMAT_ND;
  // index input
  const vector<int64_t>& indexShape = {4, 4};

  aclDataType indexDtype = ACL_INT64;
  aclFormat indexFormat = ACL_FORMAT_ND;
  // source input
  const vector<int64_t>& sourceShape = {4, 4};
  aclDataType sourceDtype = selfDtype;
  aclFormat sourceFormat = ACL_FORMAT_ND;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
  auto indexTensorDesc = TensorDesc(indexShape, indexDtype, indexFormat).Value(genInt64Vector(selfShape, indexShape));
  auto sourceTensorDesc = TensorDesc(sourceShape, sourceDtype, sourceFormat);

  auto ut = OP_API_UT(aclnnInplacePut, INPUT(selfTensorDesc, indexTensorDesc, sourceTensorDesc, true), OUTPUT());

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  ut.TestPrecision();
}

/* 各元素基本类型覆盖用例
 * 维度：1-8维
 * 根据调用算子不同可分为：
 * ScatterNdAdd算子
 *    AICORE:FLOAT,FLOAT16,INT32
 *    AICPU:INT8,UINT8
 * ScatterNdUpdate算子
 *    AICORE:BOOL,FLOAT,FLOAT16
 *    AICPU:DOUBLE,INT8,INT16,INT32,INT64,UINT8,COMPLEX64,COMPLEX128
 * index基本数据类型：INT32(pytorch不支持，UT中不能做精度比较),INT64
 * 数据格式：ND、NCHW、NHWC、HWCN、NDHWC、NCDHW
 */

TEST_F(l2_inplace_put_test, aclnnInplacePut_scatterNdAdd_1_2_3_4_5_6_7_8_float_nd) {
  // self input
  const vector<int64_t>& selfShape = {1, 2, 3, 4, 5, 6, 7, 8};
  aclDataType selfDtype = ACL_FLOAT;
  aclFormat selfFormat = ACL_FORMAT_ND;
  // index input
  const vector<int64_t>& indexShape = {1, 2, 3};
  aclDataType indexDtype = ACL_INT64;
  aclFormat indexFormat = ACL_FORMAT_ND;
  // source input
  const vector<int64_t>& sourceShape = {1, 2, 3};
  aclDataType sourceDtype = selfDtype;
  aclFormat sourceFormat = ACL_FORMAT_ND;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
  auto indexTensorDesc = TensorDesc(indexShape, indexDtype, indexFormat).Value(genInt64Vector(selfShape, indexShape));
  auto sourceTensorDesc = TensorDesc(sourceShape, sourceDtype, sourceFormat);

  auto ut = OP_API_UT(aclnnInplacePut, INPUT(selfTensorDesc, indexTensorDesc, sourceTensorDesc, true), OUTPUT());

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  ut.TestPrecision();
}

TEST_F(l2_inplace_put_test, aclnnInplacePut_scatterNdAdd_1_2_3_4_5_6_7_float16_nchw) {
  // self input
  const vector<int64_t>& selfShape = {1, 2, 3, 4, 5, 6, 7};
  aclDataType selfDtype = ACL_FLOAT16;
  aclFormat selfFormat = ACL_FORMAT_NCHW;
  // index input
  const vector<int64_t>& indexShape = {1, 2, 3};
  aclDataType indexDtype = ACL_INT64;
  aclFormat indexFormat = ACL_FORMAT_NCHW;
  // source input
  const vector<int64_t>& sourceShape = {1, 2, 3};
  aclDataType sourceDtype = selfDtype;
  aclFormat sourceFormat = ACL_FORMAT_NCHW;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
  auto indexTensorDesc = TensorDesc(indexShape, indexDtype, indexFormat).Value(genInt64Vector(selfShape, indexShape));
  auto sourceTensorDesc = TensorDesc(sourceShape, sourceDtype, sourceFormat);

  auto ut = OP_API_UT(aclnnInplacePut, INPUT(selfTensorDesc, indexTensorDesc, sourceTensorDesc, true), OUTPUT());

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_inplace_put_test, aclnnInplacePut_scatterNdAdd_1_2_3_4_5_6_int32_nchw) {
  // self input
  const vector<int64_t>& selfShape = {1, 2, 3, 4, 5, 6};
  aclDataType selfDtype = ACL_INT32;
  aclFormat selfFormat = ACL_FORMAT_NCHW;
  // index input
  const vector<int64_t>& indexShape = {1, 2, 3, 4};
  aclDataType indexDtype = ACL_INT64;
  aclFormat indexFormat = ACL_FORMAT_NCHW;
  // source input
  const vector<int64_t>& sourceShape = {1, 2, 3, 4};
  aclDataType sourceDtype = selfDtype;
  aclFormat sourceFormat = ACL_FORMAT_NCHW;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat).ValueRange(-1, 1);
  auto indexTensorDesc = TensorDesc(indexShape, indexDtype, indexFormat).Value(genInt64Vector(selfShape, indexShape));
  auto sourceTensorDesc = TensorDesc(sourceShape, sourceDtype, sourceFormat);

  auto ut = OP_API_UT(aclnnInplacePut, INPUT(selfTensorDesc, indexTensorDesc, sourceTensorDesc, true), OUTPUT());

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  ut.TestPrecision();
}

TEST_F(l2_inplace_put_test, Ascend910B2_aclnnInplacePut_scatterNdAdd_1_2_3_4_5_int8_nchw) {
  // self input
  const vector<int64_t>& selfShape = {1, 2, 3, 4, 5};
  aclDataType selfDtype = ACL_INT8;
  aclFormat selfFormat = ACL_FORMAT_NCHW;
  // index input
  const vector<int64_t>& indexShape = {1, 2, 3, 4, 5};
  aclDataType indexDtype = ACL_INT64;
  aclFormat indexFormat = ACL_FORMAT_NCHW;
  // source input
  const vector<int64_t>& sourceShape = {1, 2, 3, 4, 5};
  aclDataType sourceDtype = selfDtype;
  aclFormat sourceFormat = ACL_FORMAT_NCHW;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
  auto indexTensorDesc = TensorDesc(indexShape, indexDtype, indexFormat).Value(genInt64Vector(selfShape, indexShape));
  auto sourceTensorDesc = TensorDesc(sourceShape, sourceDtype, sourceFormat);

  auto ut = OP_API_UT(aclnnInplacePut, INPUT(selfTensorDesc, indexTensorDesc, sourceTensorDesc, false), OUTPUT());

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  ut.TestPrecision();
}

TEST_F(l2_inplace_put_test, Ascend910_9589_aclnnInplacePut_scatterNdAdd_1_2_3_4_uint8_nhwc) {
  // self input
  const vector<int64_t>& selfShape = {1, 2, 3, 4};
  aclDataType selfDtype = ACL_UINT8;
  aclFormat selfFormat = ACL_FORMAT_NHWC;
  // index input
  const vector<int64_t>& indexShape = {1, 2, 3, 4};
  aclDataType indexDtype = ACL_INT64;
  aclFormat indexFormat = ACL_FORMAT_NHWC;
  // source input
  const vector<int64_t>& sourceShape = {1, 2, 3, 4};
  aclDataType sourceDtype = selfDtype;
  aclFormat sourceFormat = ACL_FORMAT_NHWC;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
  auto indexTensorDesc = TensorDesc(indexShape, indexDtype, indexFormat).Value(genInt64Vector(selfShape, indexShape));
  auto sourceTensorDesc = TensorDesc(sourceShape, sourceDtype, sourceFormat);

  auto ut = OP_API_UT(aclnnInplacePut, INPUT(selfTensorDesc, indexTensorDesc, sourceTensorDesc, true), OUTPUT());

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  ut.TestPrecision();
}

TEST_F(l2_inplace_put_test, Ascend910_9589_aclnnInplacePut_scatterNdUpdate_1_2_3_bool_nhwc) {
  // self input
  const vector<int64_t>& selfShape = {1, 2, 3};
  aclDataType selfDtype = ACL_UINT8;
  aclFormat selfFormat = ACL_FORMAT_NHWC;
  // index input
  const vector<int64_t>& indexShape = {1, 2, 3};
  aclDataType indexDtype = ACL_INT64;
  aclFormat indexFormat = ACL_FORMAT_NHWC;
  // source input
  const vector<int64_t>& sourceShape = {1, 2, 3};
  aclDataType sourceDtype = selfDtype;
  aclFormat sourceFormat = ACL_FORMAT_NHWC;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
  auto indexTensorDesc = TensorDesc(indexShape, indexDtype, indexFormat).Value(genInt64Vector(selfShape, indexShape));
  auto sourceTensorDesc = TensorDesc(sourceShape, sourceDtype, sourceFormat);

  auto ut = OP_API_UT(aclnnInplacePut, INPUT(selfTensorDesc, indexTensorDesc, sourceTensorDesc, false), OUTPUT());

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  ut.TestPrecision();
}

TEST_F(l2_inplace_put_test, aclnnInplacePut_scatterNdUpdate_1_2_3_float_hwcn) {
  // self input
  const vector<int64_t>& selfShape = {1, 2, 3};
  aclDataType selfDtype = ACL_FLOAT;
  aclFormat selfFormat = ACL_FORMAT_HWCN;
  // index input
  const vector<int64_t>& indexShape = {1, 2, 3};
  aclDataType indexDtype = ACL_INT64;
  aclFormat indexFormat = ACL_FORMAT_HWCN;
  // source input
  const vector<int64_t>& sourceShape = {1, 2, 3};
  aclDataType sourceDtype = selfDtype;
  aclFormat sourceFormat = ACL_FORMAT_HWCN;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
  auto indexTensorDesc = TensorDesc(indexShape, indexDtype, indexFormat).Value(genInt64Vector(selfShape, indexShape));
  auto sourceTensorDesc = TensorDesc(sourceShape, sourceDtype, sourceFormat);

  auto ut = OP_API_UT(aclnnInplacePut, INPUT(selfTensorDesc, indexTensorDesc, sourceTensorDesc, false), OUTPUT());

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  ut.TestPrecision();
}

TEST_F(l2_inplace_put_test, aclnnInplacePut_scatterNdUpdate_1_2_3_4_float16_ndhwc) {
  // self input
  const vector<int64_t>& selfShape = {1, 2, 3, 4};
  aclDataType selfDtype = ACL_FLOAT16;
  aclFormat selfFormat = ACL_FORMAT_NDHWC;
  // index input
  const vector<int64_t>& indexShape = {1, 2, 3, 4};
  aclDataType indexDtype = ACL_INT64;
  aclFormat indexFormat = ACL_FORMAT_NDHWC;
  // source input
  const vector<int64_t>& sourceShape = {1, 2, 3, 4};
  aclDataType sourceDtype = selfDtype;
  aclFormat sourceFormat = ACL_FORMAT_NDHWC;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
  auto indexTensorDesc = TensorDesc(indexShape, indexDtype, indexFormat).Value(genInt64Vector(selfShape, indexShape));
  auto sourceTensorDesc = TensorDesc(sourceShape, sourceDtype, sourceFormat);

  auto ut = OP_API_UT(aclnnInplacePut, INPUT(selfTensorDesc, indexTensorDesc, sourceTensorDesc, false), OUTPUT());

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_inplace_put_test, Ascend910_9589_aclnnInplacePut_scatterNdUpdate_1_2_3_4_5_double_ncdhw) {
  // self input
  const vector<int64_t>& selfShape = {1, 2, 3, 4, 5};
  aclDataType selfDtype = ACL_INT32;
  aclFormat selfFormat = ACL_FORMAT_NCDHW;
  // index input
  const vector<int64_t>& indexShape = {1, 2, 3};
  aclDataType indexDtype = ACL_INT64;
  aclFormat indexFormat = ACL_FORMAT_NCDHW;
  // source input
  const vector<int64_t>& sourceShape = {1, 2, 3};
  aclDataType sourceDtype = selfDtype;
  aclFormat sourceFormat = ACL_FORMAT_NCDHW;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
  auto indexTensorDesc = TensorDesc(indexShape, indexDtype, indexFormat).Value(genInt64Vector(selfShape, indexShape));
  auto sourceTensorDesc = TensorDesc(sourceShape, sourceDtype, sourceFormat);

  auto ut = OP_API_UT(aclnnInplacePut, INPUT(selfTensorDesc, indexTensorDesc, sourceTensorDesc, false), OUTPUT());

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  ut.TestPrecision();
}

TEST_F(l2_inplace_put_test, Ascend910_9589_aclnnInplacePut_scatterNdUpdate_1_2_3_4_5_6_int8_nd) {
  // self input
  const vector<int64_t>& selfShape = {1, 2, 3, 4, 5, 6};
  aclDataType selfDtype = ACL_INT8;
  aclFormat selfFormat = ACL_FORMAT_ND;
  // index input
  const vector<int64_t>& indexShape = {4};
  aclDataType indexDtype = ACL_INT64;
  aclFormat indexFormat = ACL_FORMAT_ND;
  // source input
  const vector<int64_t>& sourceShape = {4};
  aclDataType sourceDtype = selfDtype;
  aclFormat sourceFormat = ACL_FORMAT_ND;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
  auto indexTensorDesc = TensorDesc(indexShape, indexDtype, indexFormat).Value(genInt64Vector(selfShape, indexShape));
  auto sourceTensorDesc = TensorDesc(sourceShape, sourceDtype, sourceFormat);

  auto ut = OP_API_UT(aclnnInplacePut, INPUT(selfTensorDesc, indexTensorDesc, sourceTensorDesc, false), OUTPUT());

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  ut.TestPrecision();
}

TEST_F(l2_inplace_put_test, Ascend910_9589_aclnnInplacePut_scatterNdUpdate_1_2_3_4_5_6_7_int16_nd) {
  // self input
  const vector<int64_t>& selfShape = {1, 2, 3, 4, 5, 6, 7};
  aclDataType selfDtype = ACL_INT8;
  aclFormat selfFormat = ACL_FORMAT_ND;
  // index input
  const vector<int64_t>& indexShape = {4};
  aclDataType indexDtype = ACL_INT64;
  aclFormat indexFormat = ACL_FORMAT_ND;
  // source input
  const vector<int64_t>& sourceShape = {4};
  aclDataType sourceDtype = selfDtype;
  aclFormat sourceFormat = ACL_FORMAT_ND;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
  auto indexTensorDesc = TensorDesc(indexShape, indexDtype, indexFormat).Value(genInt64Vector(selfShape, indexShape));
  auto sourceTensorDesc = TensorDesc(sourceShape, sourceDtype, sourceFormat);

  auto ut = OP_API_UT(aclnnInplacePut, INPUT(selfTensorDesc, indexTensorDesc, sourceTensorDesc, false), OUTPUT());

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  ut.TestPrecision();
}

TEST_F(l2_inplace_put_test, aclnnInplacePut_scatterNdUpdate_1_2_3_4_5_6_7_8_int32_nd) {
  // self input
  const vector<int64_t>& selfShape = {1, 2, 3, 4, 5, 6, 7, 8};
  aclDataType selfDtype = ACL_INT32;
  aclFormat selfFormat = ACL_FORMAT_ND;
  // index input
  const vector<int64_t>& indexShape = {4};
  aclDataType indexDtype = ACL_INT64;
  aclFormat indexFormat = ACL_FORMAT_ND;
  // source input
  const vector<int64_t>& sourceShape = {4};
  aclDataType sourceDtype = selfDtype;
  aclFormat sourceFormat = ACL_FORMAT_ND;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
  auto indexTensorDesc = TensorDesc(indexShape, indexDtype, indexFormat).Value(genInt64Vector(selfShape, indexShape));
  auto sourceTensorDesc = TensorDesc(sourceShape, sourceDtype, sourceFormat);

  auto ut = OP_API_UT(aclnnInplacePut, INPUT(selfTensorDesc, indexTensorDesc, sourceTensorDesc, false), OUTPUT());

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  ut.TestPrecision();
}

TEST_F(l2_inplace_put_test, Ascend910_9589_aclnnInplacePut_scatterNdUpdate_1_2_int64_nd) {
  // self input
  const vector<int64_t>& selfShape = {1, 2};
  aclDataType selfDtype = ACL_INT32;
  aclFormat selfFormat = ACL_FORMAT_ND;
  // index input
  const vector<int64_t>& indexShape = {2};
  aclDataType indexDtype = ACL_INT64;
  aclFormat indexFormat = ACL_FORMAT_ND;
  // source input
  const vector<int64_t>& sourceShape = {2};
  aclDataType sourceDtype = selfDtype;
  aclFormat sourceFormat = ACL_FORMAT_ND;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
  auto indexTensorDesc = TensorDesc(indexShape, indexDtype, indexFormat).Value(genInt64Vector(selfShape, indexShape));
  auto sourceTensorDesc = TensorDesc(sourceShape, sourceDtype, sourceFormat);

  auto ut = OP_API_UT(aclnnInplacePut, INPUT(selfTensorDesc, indexTensorDesc, sourceTensorDesc, false), OUTPUT());

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  ut.TestPrecision();
}

TEST_F(l2_inplace_put_test, Ascend910_9589_aclnnInplacePut_scatterNdUpdate_2_uint8_nd) {
  // self input
  const vector<int64_t>& selfShape = {2};
  aclDataType selfDtype = ACL_UINT8;
  aclFormat selfFormat = ACL_FORMAT_ND;
  // index input
  const vector<int64_t>& indexShape = {2};
  aclDataType indexDtype = ACL_INT64;
  aclFormat indexFormat = ACL_FORMAT_ND;
  // source input
  const vector<int64_t>& sourceShape = {2};
  aclDataType sourceDtype = selfDtype;
  aclFormat sourceFormat = ACL_FORMAT_ND;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
  auto indexTensorDesc = TensorDesc(indexShape, indexDtype, indexFormat).Value(genInt64Vector(selfShape, indexShape));
  auto sourceTensorDesc = TensorDesc(sourceShape, sourceDtype, sourceFormat);

  auto ut = OP_API_UT(aclnnInplacePut, INPUT(selfTensorDesc, indexTensorDesc, sourceTensorDesc, false), OUTPUT());

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  ut.TestPrecision();
}

TEST_F(l2_inplace_put_test, aclnnInplacePut_scatterNdUpdate_2_complex64_nd) {
  // self input
  const vector<int64_t>& selfShape = {2};
  aclDataType selfDtype = ACL_COMPLEX64;
  aclFormat selfFormat = ACL_FORMAT_ND;
  // index input
  const vector<int64_t>& indexShape = {2};
  aclDataType indexDtype = ACL_INT64;
  aclFormat indexFormat = ACL_FORMAT_ND;
  // source input
  const vector<int64_t>& sourceShape = {2};
  aclDataType sourceDtype = selfDtype;
  aclFormat sourceFormat = ACL_FORMAT_ND;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
  auto indexTensorDesc = TensorDesc(indexShape, indexDtype, indexFormat).Value(genInt64Vector(selfShape, indexShape));
  auto sourceTensorDesc = TensorDesc(sourceShape, sourceDtype, sourceFormat);

  auto ut = OP_API_UT(aclnnInplacePut, INPUT(selfTensorDesc, indexTensorDesc, sourceTensorDesc, false), OUTPUT());

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_inplace_put_test, aclnnInplacePut_scatterNdUpdate_2_complex128_nd) {
  // self input
  const vector<int64_t>& selfShape = {2};
  aclDataType selfDtype = ACL_COMPLEX128;
  aclFormat selfFormat = ACL_FORMAT_ND;
  // index input
  const vector<int64_t>& indexShape = {2};
  aclDataType indexDtype = ACL_INT64;
  aclFormat indexFormat = ACL_FORMAT_ND;
  // source input
  const vector<int64_t>& sourceShape = {2};
  aclDataType sourceDtype = selfDtype;
  aclFormat sourceFormat = ACL_FORMAT_ND;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
  auto indexTensorDesc = TensorDesc(indexShape, indexDtype, indexFormat).Value(genInt64Vector(selfShape, indexShape));
  auto sourceTensorDesc = TensorDesc(sourceShape, sourceDtype, sourceFormat);

  auto ut = OP_API_UT(aclnnInplacePut, INPUT(selfTensorDesc, indexTensorDesc, sourceTensorDesc, false), OUTPUT());

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_inplace_put_test, aclnnInplacePut_scatterNdUpdate_1_2_3_4_float16_ndhwc_with_index_int32) {
  // self input
  const vector<int64_t>& selfShape = {1, 2, 3, 4};
  aclDataType selfDtype = ACL_FLOAT16;
  aclFormat selfFormat = ACL_FORMAT_NDHWC;
  // index input
  const vector<int64_t>& indexShape = {1, 2, 3, 4};
  aclDataType indexDtype = ACL_INT32;
  aclFormat indexFormat = ACL_FORMAT_NDHWC;
  // source input
  const vector<int64_t>& sourceShape = {1, 2, 3, 4};
  aclDataType sourceDtype = selfDtype;
  aclFormat sourceFormat = ACL_FORMAT_NDHWC;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
  auto indexTensorDesc = TensorDesc(indexShape, indexDtype, indexFormat).Value(genInt64Vector(selfShape, indexShape));
  auto sourceTensorDesc = TensorDesc(sourceShape, sourceDtype, sourceFormat);

  auto ut = OP_API_UT(aclnnInplacePut, INPUT(selfTensorDesc, indexTensorDesc, sourceTensorDesc, false), OUTPUT());

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 各元素特殊类型覆盖用例
// 空tensor   self非空,index为空
TEST_F(l2_inplace_put_test, aclnnInplacePut_float_nd_empty_tensor) {
  // self input
  const vector<int64_t>& selfShape = {2};
  aclDataType selfDtype = ACL_FLOAT;
  aclFormat selfFormat = ACL_FORMAT_ND;
  // index input
  const vector<int64_t>& indexShape = {0, 2};

  aclDataType indexDtype = ACL_INT64;
  aclFormat indexFormat = ACL_FORMAT_ND;
  // source input
  const vector<int64_t>& sourceShape = {0, 2};
  aclDataType sourceDtype = selfDtype;
  aclFormat sourceFormat = ACL_FORMAT_ND;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
  auto indexTensorDesc = TensorDesc(indexShape, indexDtype, indexFormat);
  auto sourceTensorDesc = TensorDesc(sourceShape, sourceDtype, sourceFormat);

  auto ut = OP_API_UT(aclnnInplacePut, INPUT(selfTensorDesc, indexTensorDesc, sourceTensorDesc, false), OUTPUT());

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  ut.TestPrecision();
}

// 边界值
TEST_F(l2_inplace_put_test, aclnnInplacePut_1_2_float32_nd_boundary_value) {
  // self input
  const vector<int64_t>& selfShape = {1, 2};
  aclDataType selfDtype = ACL_FLOAT;
  aclFormat selfFormat = ACL_FORMAT_ND;
  // index input
  const vector<int64_t>& indexShape = {1, 2};
  aclDataType indexDtype = ACL_INT64;
  aclFormat indexFormat = ACL_FORMAT_ND;
  // source input
  const vector<int64_t>& sourceShape = {2};
  aclDataType sourceDtype = selfDtype;
  aclFormat sourceFormat = ACL_FORMAT_ND;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat).Value(vector<float>{65504.0, -65504.0});
  auto indexTensorDesc = TensorDesc(indexShape, indexDtype, indexFormat).Value(genInt64Vector(selfShape, indexShape));
  auto sourceTensorDesc = TensorDesc(sourceShape, sourceDtype, sourceFormat);

  auto ut = OP_API_UT(aclnnInplacePut, INPUT(selfTensorDesc, indexTensorDesc, sourceTensorDesc, false), OUTPUT());

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  ut.TestPrecision();
}

// 特殊值：nan,inf
// TEST_F(l2_inplace_put_test, aclnnInplacePut_2_4_float32_nd_and_2_4_bool_nd_special_value) {
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
// auto ut = OP_API_UT(aclnnInplacePut, INPUT(selfTensorDesc, indexTensorDesc,sourceTensorDesc), OUTPUT());
//
// uint64_t workspaceSize = 0;
// aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
// EXPECT_EQ(aclRet, ACL_SUCCESS);
//
//
// ut.TestPrecision();
//}

// 不连续
TEST_F(l2_inplace_put_test, aclnnInplacePut_5_4_float_nd_not_contiguous) {
  // self input
  const vector<int64_t>& selfShape = {5, 4};
  aclDataType selfDtype = ACL_FLOAT;
  aclFormat selfFormat = ACL_FORMAT_ND;
  const vector<int64_t>& selfViewDim = {1, 5};
  int64_t selfOffset = 0;
  const vector<int64_t>& selfStorageDim = {4, 5};
  // index input
  const vector<int64_t>& indexShape = {5, 4};
  aclDataType indexDtype = ACL_INT64;
  aclFormat indexFormat = ACL_FORMAT_ND;
  const vector<int64_t>& indexViewDim = {1, 5};
  int64_t indexOffset = 0;
  const vector<int64_t>& indexStorageDim = {4, 5};
  // source input
  const vector<int64_t>& sourceShape = {5, 4};
  aclDataType sourceDtype = selfDtype;
  aclFormat sourceFormat = ACL_FORMAT_ND;
  const vector<int64_t>& sourceViewDim = {1, 5};
  int64_t sourceOffset = 0;
  const vector<int64_t>& sourceStorageDim = {4, 5};

  auto selfTensorDesc =
      TensorDesc(selfShape, selfDtype, selfFormat, selfViewDim, selfOffset, selfStorageDim).ValueRange(-2, 2);
  auto indexTensorDesc = TensorDesc(indexShape, indexDtype, indexFormat, indexViewDim, indexOffset, indexStorageDim)
                             .Value(genInt64Vector(selfShape, indexShape));
  auto sourceTensorDesc =
      TensorDesc(sourceShape, sourceDtype, sourceFormat, sourceViewDim, sourceOffset, sourceStorageDim);

  auto ut = OP_API_UT(aclnnInplacePut, INPUT(selfTensorDesc, indexTensorDesc, sourceTensorDesc, false), OUTPUT());

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 报错类型覆盖用例
// 空指针
TEST_F(l2_inplace_put_test, aclnnInplacePut_input_nullptr) {
  auto tensor_desc = TensorDesc({10, 5}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut_self_nullptr = OP_API_UT(aclnnInplacePut, INPUT((aclTensor*)nullptr, tensor_desc, tensor_desc,false), OUTPUT());

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut_self_nullptr.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

  auto ut_index_nullptr = OP_API_UT(aclnnInplacePut, INPUT(tensor_desc, (aclTensor*)nullptr, tensor_desc,false), OUTPUT());

  aclRet = ut_index_nullptr.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

  auto ut_source_nullptr = OP_API_UT(aclnnInplacePut, INPUT(tensor_desc, tensor_desc, (aclTensor*)nullptr,false), OUTPUT());

  aclRet = ut_source_nullptr.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// self类型不满足
TEST_F(l2_inplace_put_test, aclnnInplacePut_self_dtype_error) {
  // self input
  const vector<int64_t>& selfShape = {2, 4};
  aclDataType selfDtype = ACL_UINT64;
  aclFormat selfFormat = ACL_FORMAT_ND;
  // index input
  const vector<int64_t>& indexShape = {2, 4};
  aclDataType indexDtype = ACL_INT64;
  aclFormat indexFormat = ACL_FORMAT_ND;
  // source input
  const vector<int64_t>& sourceShape = {2, 4};
  aclDataType sourceDtype = selfDtype;
  aclFormat sourceFormat = ACL_FORMAT_ND;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
  auto indexTensorDesc = TensorDesc(indexShape, indexDtype, indexFormat).Value(genInt64Vector(selfShape, indexShape));
  auto sourceTensorDesc = TensorDesc(sourceShape, sourceDtype, sourceFormat);

  auto ut = OP_API_UT(aclnnInplacePut, INPUT(selfTensorDesc, indexTensorDesc, sourceTensorDesc, false), OUTPUT());

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// index类型不满足
TEST_F(l2_inplace_put_test, aclnnInplacePut_index_dtype_error) {
  // self input
  const vector<int64_t>& selfShape = {2, 4};
  aclDataType selfDtype = ACL_FLOAT;
  aclFormat selfFormat = ACL_FORMAT_ND;
  // index input
  const vector<int64_t>& indexShape = {2, 4};
  aclDataType indexDtype = ACL_INT8;
  aclFormat indexFormat = ACL_FORMAT_ND;
  // source input
  const vector<int64_t>& sourceShape = {2, 4};
  aclDataType sourceDtype = selfDtype;
  aclFormat sourceFormat = ACL_FORMAT_ND;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
  auto indexTensorDesc = TensorDesc(indexShape, indexDtype, indexFormat);
  auto sourceTensorDesc = TensorDesc(sourceShape, sourceDtype, sourceFormat);

  auto ut = OP_API_UT(aclnnInplacePut, INPUT(selfTensorDesc, indexTensorDesc, sourceTensorDesc, false), OUTPUT());

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// self类型和source不一致
TEST_F(l2_inplace_put_test, aclnnInplacePut_self_and_source_dtype_non_consistent) {
  // self input
  const vector<int64_t>& selfShape = {2, 4};
  aclDataType selfDtype = ACL_FLOAT;
  aclFormat selfFormat = ACL_FORMAT_ND;
  // index input
  const vector<int64_t>& indexShape = {2, 4};
  aclDataType indexDtype = ACL_INT64;
  aclFormat indexFormat = ACL_FORMAT_ND;
  // source input
  const vector<int64_t>& sourceShape = {2, 4};
  aclDataType sourceDtype = ACL_FLOAT16;
  aclFormat sourceFormat = ACL_FORMAT_ND;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
  auto indexTensorDesc = TensorDesc(indexShape, indexDtype, indexFormat).Value(genInt64Vector(selfShape, indexShape));
  auto sourceTensorDesc = TensorDesc(sourceShape, sourceDtype, sourceFormat);

  auto ut = OP_API_UT(aclnnInplacePut, INPUT(selfTensorDesc, indexTensorDesc, sourceTensorDesc, false), OUTPUT());

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// self形状大于8
TEST_F(l2_inplace_put_test, aclnnInplacePut_self_shape_out_of_8) {
  // self input
  const vector<int64_t>& selfShape = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  aclDataType selfDtype = ACL_FLOAT;
  aclFormat selfFormat = ACL_FORMAT_ND;
  // index input
  const vector<int64_t>& indexShape = {2, 4};
  aclDataType indexDtype = ACL_INT64;
  aclFormat indexFormat = ACL_FORMAT_ND;
  // source input
  const vector<int64_t>& sourceShape = {2, 4};
  aclDataType sourceDtype = selfDtype;
  aclFormat sourceFormat = ACL_FORMAT_ND;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
  auto indexTensorDesc = TensorDesc(indexShape, indexDtype, indexFormat).Value(genInt64Vector(selfShape, indexShape));
  auto sourceTensorDesc = TensorDesc(sourceShape, sourceDtype, sourceFormat);

  auto ut = OP_API_UT(aclnnInplacePut, INPUT(selfTensorDesc, indexTensorDesc, sourceTensorDesc, false), OUTPUT());

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// index和source的元素数量不等
TEST_F(l2_inplace_put_test, aclnnInplacePut_index_and_source_elements_not_same) {
  // self input
  const vector<int64_t>& selfShape = {1, 2, 3};
  aclDataType selfDtype = ACL_FLOAT;
  aclFormat selfFormat = ACL_FORMAT_ND;
  // index input
  const vector<int64_t>& indexShape = {1, 2, 3};
  aclDataType indexDtype = ACL_INT64;
  aclFormat indexFormat = ACL_FORMAT_ND;
  // source input
  const vector<int64_t>& sourceShape = {3};
  aclDataType sourceDtype = selfDtype;
  aclFormat sourceFormat = ACL_FORMAT_ND;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
  auto indexTensorDesc = TensorDesc(indexShape, indexDtype, indexFormat).Value(genInt64Vector(selfShape, indexShape));
  auto sourceTensorDesc = TensorDesc(sourceShape, sourceDtype, sourceFormat);

  auto ut = OP_API_UT(aclnnInplacePut, INPUT(selfTensorDesc, indexTensorDesc, sourceTensorDesc, false), OUTPUT());

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// self数据类型不支持uint64
TEST_F(l2_inplace_put_test, aclnnInplacePut_self_dtype_with_uint64) {
  // self input
  const vector<int64_t>& selfShape = {2, 4};
  aclDataType selfDtype = ACL_UINT64;
  aclFormat selfFormat = ACL_FORMAT_ND;
  // index input
  const vector<int64_t>& indexShape = {2, 4};
  aclDataType indexDtype = ACL_INT64;
  aclFormat indexFormat = ACL_FORMAT_ND;
  // source input
  const vector<int64_t>& sourceShape = {2, 4};
  aclDataType sourceDtype = selfDtype;
  aclFormat sourceFormat = ACL_FORMAT_ND;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
  auto indexTensorDesc = TensorDesc(indexShape, indexDtype, indexFormat);
  auto sourceTensorDesc = TensorDesc(sourceShape, sourceDtype, sourceFormat);

  auto ut = OP_API_UT(aclnnInplacePut, INPUT(selfTensorDesc, indexTensorDesc, sourceTensorDesc, false), OUTPUT());

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// self为空，index非空
TEST_F(l2_inplace_put_test, aclnnInplacePut_self_empty_tensor_and_index_not_empty) {
  // self input
  const vector<int64_t>& selfShape = {0, 4};
  aclDataType selfDtype = ACL_FLOAT;
  aclFormat selfFormat = ACL_FORMAT_NHWC;
  // index input
  const vector<int64_t>& indexShape = {2, 4};
  aclDataType indexDtype = ACL_INT64;
  aclFormat indexFormat = ACL_FORMAT_NHWC;
  // source input
  const vector<int64_t>& sourceShape = {2, 4};
  aclDataType sourceDtype = selfDtype;
  aclFormat sourceFormat = ACL_FORMAT_NHWC;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
  auto indexTensorDesc = TensorDesc(indexShape, indexDtype, indexFormat);
  auto sourceTensorDesc = TensorDesc(sourceShape, sourceDtype, sourceFormat);

  auto ut = OP_API_UT(aclnnInplacePut, INPUT(selfTensorDesc, indexTensorDesc, sourceTensorDesc, false), OUTPUT());

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}
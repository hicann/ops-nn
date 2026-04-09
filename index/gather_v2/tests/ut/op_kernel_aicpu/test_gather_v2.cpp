/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <complex>
#include <functional>
#include <memory>
#include <numeric>
#include <vector>

#include "gtest/gtest.h"
#ifndef private
#define private public
#define protected public
#endif
#include "utils/aicpu_test_utils.h"
#include "cpu_kernel_utils.h"
#include "node_def_builder.h"
#undef private
#undef protected
#include "Eigen/Core"

using namespace std;
using namespace aicpu;

class TEST_GATHER_V2_UT : public testing::Test {};

#define CREATE_NODEDEF(shapes, data_types, datas, batch_dims)          \
  auto node_def = CpuKernelUtils::CreateNodeDef();                     \
  NodeDefBuilder(node_def.get(), "GatherV2", "GatherV2")               \
      .Input({"x", data_types[0], shapes[0], datas[0]})                \
      .Input({"indices", data_types[1], shapes[1], datas[1]})          \
      .Input({"axis", data_types[2], shapes[2], datas[2]})             \
      .Output({"y", data_types[3], shapes[3], datas[3]})               \
      .Attr("batch_dims", batch_dims)

template <typename T, typename Index, typename AxisType>
void RunGatherV2Kernel(const vector<vector<int64_t>> &shapes, const vector<DataType> &data_types, const T *input_x,
                       const Index *input_indices, const AxisType *input_axis, T *expect_output,
                       int64_t batch_dims = 0) {
  auto calc_size = [](const vector<int64_t> &shape) -> uint64_t {
    return shape.empty() ? 0 : accumulate(shape.begin(), shape.end(), 1LL, multiplies<int64_t>());
  };

  const uint64_t input_x_size = calc_size(shapes[0]);
  const uint64_t input_indices_size = calc_size(shapes[1]);
  const uint64_t output_size = calc_size(shapes[3]);
  auto x_data = make_unique<T[]>(input_x_size);
  auto indices_data = make_unique<Index[]>(input_indices_size);
  auto axis_data = make_unique<AxisType[]>(1);
  auto output_data = make_unique<T[]>(output_size);

  for (uint64_t i = 0; i < input_x_size; ++i) {
    x_data[i] = input_x[i];
  }
  for (uint64_t i = 0; i < input_indices_size; ++i) {
    indices_data[i] = input_indices[i];
  }
  axis_data[0] = input_axis[0];
  for (uint64_t i = 0; i < output_size; ++i) {
    output_data[i] = T();
  }

  vector<void *> datas = {static_cast<void *>(x_data.get()), static_cast<void *>(indices_data.get()),
                          static_cast<void *>(axis_data.get()), static_cast<void *>(output_data.get())};
  CREATE_NODEDEF(shapes, data_types, datas, batch_dims);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  EXPECT_TRUE(CompareResult(output_data.get(), expect_output, output_size));
}

TEST_F(TEST_GATHER_V2_UT, DATA_TYPE_DT_COMPLEX128_INT64_SUCC) {
  vector<DataType> data_types = {DT_COMPLEX128, DT_INT64, DT_INT64, DT_COMPLEX128};
  vector<vector<int64_t>> shapes = {{3, 5}, {2}, {1}, {3, 2}};
  complex<double> input_x[15] = {{0, 10},  {1, 11},  {2, 12},  {3, 13},  {4, 14},
                                 {5, 15},  {6, 16},  {7, 17},  {8, 18},  {9, 19},
                                 {10, 20}, {11, 21}, {12, 22}, {13, 23}, {14, 24}};
  int64_t input_indices[2] = {1, 3};
  int64_t input_axis[1] = {1};
  complex<double> expect_output[6] = {{1, 11}, {3, 13}, {6, 16}, {8, 18}, {11, 21}, {13, 23}};
  RunGatherV2Kernel(shapes, data_types, input_x, input_indices, input_axis, expect_output);
}

TEST_F(TEST_GATHER_V2_UT, DATA_TYPE_DT_DOUBLE_INT32_BATCH_DIMS_SUCC) {
  vector<DataType> data_types = {DT_DOUBLE, DT_INT32, DT_INT32, DT_DOUBLE};
  vector<vector<int64_t>> shapes = {{3, 2, 3}, {3, 2}, {1}, {3, 2, 2}};
  double input_x[18] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18};
  int32_t input_indices[6] = {0, 1, 2, 0, 1, 2};
  int32_t input_axis[1] = {2};
  double expect_output[12] = {1, 2, 4, 5, 9, 7, 12, 10, 14, 15, 17, 18};
  RunGatherV2Kernel(shapes, data_types, input_x, input_indices, input_axis, expect_output, 1);
}

TEST_F(TEST_GATHER_V2_UT, DATA_TYPE_DT_INT32_NEGATIVE_INDEX_SUCC) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{4}, {1}, {1}, {1}};
  int32_t input_x[4] = {0, 1, 2, 3};
  int32_t input_indices[1] = {-1};
  int32_t input_axis[1] = {0};
  int32_t expect_output[1] = {3};
  RunGatherV2Kernel(shapes, data_types, input_x, input_indices, input_axis, expect_output);
}

TEST_F(TEST_GATHER_V2_UT, FAILED_INDICES_TYPE) {
  vector<DataType> data_types = {DT_DOUBLE, DT_DOUBLE, DT_INT64, DT_DOUBLE};
  vector<vector<int64_t>> shapes = {{4}, {1}, {1}, {1}};
  double input_x[4] = {1, 2, 3, 4};
  double input_indices[1] = {1};
  int64_t input_axis[1] = {0};
  double output[1] = {0};
  vector<void *> datas = {static_cast<void *>(input_x), static_cast<void *>(input_indices),
                          static_cast<void *>(input_axis), static_cast<void *>(output)};
  CREATE_NODEDEF(shapes, data_types, datas, 0);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_GATHER_V2_UT, FAILED_OUT_OF_BOUND_INDEX) {
  vector<DataType> data_types = {DT_DOUBLE, DT_INT64, DT_INT64, DT_DOUBLE};
  vector<vector<int64_t>> shapes = {{4}, {1}, {1}, {1}};
  double input_x[4] = {1, 2, 3, 4};
  int64_t input_indices[1] = {5};
  int64_t input_axis[1] = {0};
  double output[1] = {0};
  vector<void *> datas = {static_cast<void *>(input_x), static_cast<void *>(input_indices),
                          static_cast<void *>(input_axis), static_cast<void *>(output)};
  CREATE_NODEDEF(shapes, data_types, datas, 0);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

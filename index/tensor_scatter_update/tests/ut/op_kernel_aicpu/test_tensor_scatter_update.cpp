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

class TEST_TENSOR_SCATTER_UPDATE_UT : public testing::Test {};

auto CreateTensorScatterUpdateNodeDef(const vector<vector<int64_t>> &shapes, const vector<DataType> &data_types,
                                      const vector<void *> &datas) -> decltype(CpuKernelUtils::CreateNodeDef()) {
  auto node_def = CpuKernelUtils::CreateNodeDef();
  NodeDefBuilder(node_def.get(), "TensorScatterUpdate", "TensorScatterUpdate")
      .Input({"x", data_types[0], shapes[0], datas[0]})
      .Input({"indices", data_types[1], shapes[1], datas[1]})
      .Input({"updates", data_types[2], shapes[2], datas[2]})
      .Output({"y", data_types[3], shapes[3], datas[3]});
  return node_def;
}

template <typename T, typename Index>
void RunTensorScatterUpdateKernel(const vector<vector<int64_t>> &shapes, const vector<DataType> &data_types,
                                  const T *input_x, const Index *input_indices, const T *input_updates,
                                  const T *expect_output) {
  const uint64_t input_x_size =
      shapes[0].empty() ? 0 : std::accumulate(shapes[0].begin(), shapes[0].end(), 1LL, std::multiplies<int64_t>());
  const uint64_t input_indices_size =
      shapes[1].empty() ? 0 : std::accumulate(shapes[1].begin(), shapes[1].end(), 1LL, std::multiplies<int64_t>());
  const uint64_t input_updates_size =
      shapes[2].empty() ? 0 : std::accumulate(shapes[2].begin(), shapes[2].end(), 1LL, std::multiplies<int64_t>());
  const uint64_t output_size =
      shapes[3].empty() ? 0 : std::accumulate(shapes[3].begin(), shapes[3].end(), 1LL, std::multiplies<int64_t>());

  auto x_data = std::make_unique<T[]>(input_x_size);
  auto indices_data = std::make_unique<Index[]>(input_indices_size);
  auto updates_data = std::make_unique<T[]>(input_updates_size);
  auto output_data = std::make_unique<T[]>(output_size);
  auto expect = std::make_unique<T[]>(output_size);

  for (uint64_t i = 0; i < input_x_size; ++i) {
    x_data[i] = input_x[i];
  }
  for (uint64_t i = 0; i < input_indices_size; ++i) {
    indices_data[i] = input_indices[i];
  }
  for (uint64_t i = 0; i < input_updates_size; ++i) {
    updates_data[i] = input_updates[i];
  }
  for (uint64_t i = 0; i < output_size; ++i) {
    output_data[i] = T();
    expect[i] = expect_output[i];
  }

  vector<void *> datas = {static_cast<void *>(x_data.get()), static_cast<void *>(indices_data.get()),
                          static_cast<void *>(updates_data.get()), static_cast<void *>(output_data.get())};

  auto node_def = CreateTensorScatterUpdateNodeDef(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  EXPECT_TRUE(CompareResult(output_data.get(), expect.get(), output_size));
}

TEST_F(TEST_TENSOR_SCATTER_UPDATE_UT, DATA_TYPE_DT_BOOL_INT64_SUCC) {
  vector<DataType> data_types = {DT_BOOL, DT_INT64, DT_BOOL, DT_BOOL};
  vector<vector<int64_t>> shapes = {{8}, {4, 1}, {4}, {8}};
  bool input_x[8] = {true, true, true, true, true, true, true, false};
  int64_t input_indices[4] = {4, 3, 1, 7};
  bool input_updates[4] = {false, false, false, true};
  bool expect_output[8] = {true, false, true, false, false, true, true, true};
  RunTensorScatterUpdateKernel(shapes, data_types, input_x, input_indices, input_updates, expect_output);
}

TEST_F(TEST_TENSOR_SCATTER_UPDATE_UT, DATA_TYPE_DT_FLOAT_INT64_SUCC) {
  vector<DataType> data_types = {DT_FLOAT, DT_INT64, DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{8}, {4, 1}, {4}, {8}};
  float input_x[8] = {1, 2, 3, 4, 5, 6, 7, 8};
  int64_t input_indices[4] = {4, 3, 1, 7};
  float input_updates[4] = {9, 10, 11, 12};
  float expect_output[8] = {1, 11, 3, 10, 9, 6, 7, 12};
  RunTensorScatterUpdateKernel(shapes, data_types, input_x, input_indices, input_updates, expect_output);
}

TEST_F(TEST_TENSOR_SCATTER_UPDATE_UT, DATA_TYPE_DT_DOUBLE_INT64_SUCC) {
  vector<DataType> data_types = {DT_DOUBLE, DT_INT64, DT_DOUBLE, DT_DOUBLE};
  vector<vector<int64_t>> shapes = {{2, 2, 2}, {2, 2}, {2, 2}, {2, 2, 2}};
  double input_x[8] = {0, 1, 2, 3, 4, 5, 6, 7};
  int64_t input_indices[4] = {0, 1, 1, 0};
  double input_updates[4] = {20, 21, 30, 31};
  double expect_output[8] = {0, 1, 20, 21, 30, 31, 6, 7};
  RunTensorScatterUpdateKernel(shapes, data_types, input_x, input_indices, input_updates, expect_output);
}

TEST_F(TEST_TENSOR_SCATTER_UPDATE_UT, DATA_TYPE_DT_FLOAT16_INT64_SUCC) {
  vector<DataType> data_types = {DT_FLOAT16, DT_INT64, DT_FLOAT16, DT_FLOAT16};
  vector<vector<int64_t>> shapes = {{3, 5}, {2, 1}, {2, 5}, {3, 5}};
  Eigen::half input_x[15] = {Eigen::half(0), Eigen::half(1), Eigen::half(2), Eigen::half(3), Eigen::half(4),
                              Eigen::half(5), Eigen::half(6), Eigen::half(7), Eigen::half(8), Eigen::half(9),
                              Eigen::half(10), Eigen::half(11), Eigen::half(12), Eigen::half(13), Eigen::half(14)};
  int64_t input_indices[2] = {0, 2};
  Eigen::half input_updates[10] = {Eigen::half(-1), Eigen::half(-2), Eigen::half(-3), Eigen::half(-4), Eigen::half(-5),
                                   Eigen::half(-10), Eigen::half(-9), Eigen::half(-8), Eigen::half(-7), Eigen::half(-6)};
  Eigen::half expect_output[15] = {Eigen::half(-1), Eigen::half(-2), Eigen::half(-3), Eigen::half(-4), Eigen::half(-5),
                                   Eigen::half(5), Eigen::half(6), Eigen::half(7), Eigen::half(8), Eigen::half(9),
                                   Eigen::half(-10), Eigen::half(-9), Eigen::half(-8), Eigen::half(-7), Eigen::half(-6)};
  RunTensorScatterUpdateKernel(shapes, data_types, input_x, input_indices, input_updates, expect_output);
}

TEST_F(TEST_TENSOR_SCATTER_UPDATE_UT, DATA_TYPE_DT_INT32_INT32_SUCC) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{8}, {4, 1}, {4}, {8}};
  int32_t input_x[8] = {1, 2, 3, 4, 5, 6, 7, 8};
  int32_t input_indices[4] = {4, 3, 1, 7};
  int32_t input_updates[4] = {9, 10, 11, 12};
  int32_t expect_output[8] = {1, 11, 3, 10, 9, 6, 7, 12};
  RunTensorScatterUpdateKernel(shapes, data_types, input_x, input_indices, input_updates, expect_output);
}

TEST_F(TEST_TENSOR_SCATTER_UPDATE_UT, DATA_TYPE_DT_COMPLEX64_INT32_SUCC) {
  vector<DataType> data_types = {DT_COMPLEX64, DT_INT32, DT_COMPLEX64, DT_COMPLEX64};
  vector<vector<int64_t>> shapes = {{4}, {2, 1}, {2}, {4}};
  complex<float> input_x[4] = {{1, 1}, {2, 2}, {3, 3}, {4, 4}};
  int32_t input_indices[2] = {1, 3};
  complex<float> input_updates[2] = {{10, -1}, {20, -2}};
  complex<float> expect_output[4] = {{1, 1}, {10, -1}, {3, 3}, {20, -2}};
  RunTensorScatterUpdateKernel(shapes, data_types, input_x, input_indices, input_updates, expect_output);
}

TEST_F(TEST_TENSOR_SCATTER_UPDATE_UT, INPUT_EMPTY_TENSOR_SUCC) {
  vector<DataType> data_types = {DT_DOUBLE, DT_INT64, DT_DOUBLE, DT_DOUBLE};
  vector<vector<int64_t>> shapes = {{0}, {4, 0}, {4, 0}, {0}};
  double input_x[1] = {0};
  int64_t input_indices[1] = {0};
  double input_updates[1] = {0};
  double output[1] = {0};
  vector<void *> datas = {static_cast<void *>(input_x), static_cast<void *>(input_indices),
                          static_cast<void *>(input_updates), static_cast<void *>(output)};
  auto node_def = CreateTensorScatterUpdateNodeDef(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
}

TEST_F(TEST_TENSOR_SCATTER_UPDATE_UT, FAILED_INDICES_TYPE) {
  vector<DataType> data_types = {DT_INT64, DT_DOUBLE, DT_INT64, DT_INT64};
  vector<vector<int64_t>> shapes = {{8}, {4, 1}, {4}, {8}};
  int64_t input_x[8] = {1, 2, 3, 4, 5, 6, 7, 8};
  double input_indices[4] = {4.2, 3.5, 1.7, 7.4};
  int64_t input_updates[4] = {11, 12, 13, 14};
  int64_t output[8] = {0};
  vector<void *> datas = {static_cast<void *>(input_x), static_cast<void *>(input_indices),
                          static_cast<void *>(input_updates), static_cast<void *>(output)};
  auto node_def = CreateTensorScatterUpdateNodeDef(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_TENSOR_SCATTER_UPDATE_UT, FAILED_INDICES_RANK) {
  vector<DataType> data_types = {DT_INT64, DT_INT64, DT_INT64, DT_INT64};
  vector<vector<int64_t>> shapes = {{8}, {4}, {4}, {8}};
  int64_t input_x[8] = {1, 2, 3, 4, 5, 6, 7, 8};
  int64_t input_indices[4] = {0, 1, 2, 3};
  int64_t input_updates[4] = {11, 12, 13, 14};
  int64_t output[8] = {0};
  vector<void *> datas = {static_cast<void *>(input_x), static_cast<void *>(input_indices),
                          static_cast<void *>(input_updates), static_cast<void *>(output)};
  auto node_def = CreateTensorScatterUpdateNodeDef(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_TENSOR_SCATTER_UPDATE_UT, FAILED_SHAPE_MISMATCH) {
  vector<DataType> data_types = {DT_INT64, DT_INT64, DT_INT64, DT_INT64};
  vector<vector<int64_t>> shapes = {{5, 5}, {2, 2, 1}, {2, 3, 5}, {5, 5}};
  int64_t input_x[25] = {0};
  int64_t input_indices[4] = {0, 0, 1, 1};
  int64_t input_updates[30] = {0};
  int64_t output[25] = {0};
  vector<void *> datas = {static_cast<void *>(input_x), static_cast<void *>(input_indices),
                          static_cast<void *>(input_updates), static_cast<void *>(output)};
  auto node_def = CreateTensorScatterUpdateNodeDef(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_TENSOR_SCATTER_UPDATE_UT, FAILED_OUT_OF_BOUNDS_INDEX) {
  vector<DataType> data_types = {DT_DOUBLE, DT_INT64, DT_DOUBLE, DT_DOUBLE};
  vector<vector<int64_t>> shapes = {{8}, {4, 1}, {4}, {8}};
  double input_x[8] = {1, 2, 3, 4, 5, 6, 7, 8};
  int64_t input_indices[4] = {4, -1, -2, 7};
  double input_updates[4] = {2, 2, 2, 2};
  double output[8] = {0};
  vector<void *> datas = {static_cast<void *>(input_x), static_cast<void *>(input_indices),
                          static_cast<void *>(input_updates), static_cast<void *>(output)};
  auto node_def = CreateTensorScatterUpdateNodeDef(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

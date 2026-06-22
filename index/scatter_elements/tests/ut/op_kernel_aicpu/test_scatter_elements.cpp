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

class TEST_SCATTER_ELEMENTS_UT : public testing::Test {};

#define CREATE_NODEDEF(shapes, data_types, datas, axis, reduction)      \
  auto node_def = CpuKernelUtils::CreateNodeDef();                      \
  NodeDefBuilder(node_def.get(), "ScatterElements", "ScatterElements")  \
      .Input({"data", data_types[0], shapes[0], datas[0]})              \
      .Input({"indices", data_types[1], shapes[1], datas[1]})           \
      .Input({"updates", data_types[2], shapes[2], datas[2]})           \
      .Attr("axis", axis)                                               \
      .Attr("reduction", reduction)                                     \
      .Output({"y", data_types[3], shapes[3], datas[3]})

template <typename T, typename Index>
void RunScatterElementsKernel(const vector<vector<int64_t>> &shapes, const vector<DataType> &data_types, const T *input_data,
                              const Index *input_indices, const T *input_updates, T *expect_output, int64_t axis,
                              const string &reduction = "none") {
  auto calc_size = [](const vector<int64_t> &shape) -> uint64_t {
    return shape.empty() ? 0 : accumulate(shape.begin(), shape.end(), 1LL, multiplies<int64_t>());
  };

  const uint64_t data_size = calc_size(shapes[0]);
  const uint64_t indices_size = calc_size(shapes[1]);
  const uint64_t updates_size = calc_size(shapes[2]);
  const uint64_t output_size = calc_size(shapes[3]);
  auto data = make_unique<T[]>(data_size);
  auto indices = make_unique<Index[]>(indices_size);
  auto updates = make_unique<T[]>(updates_size);
  auto output = make_unique<T[]>(output_size);

  for (uint64_t i = 0; i < data_size; ++i) {
    data[i] = input_data[i];
  }
  for (uint64_t i = 0; i < indices_size; ++i) {
    indices[i] = input_indices[i];
  }
  for (uint64_t i = 0; i < updates_size; ++i) {
    updates[i] = input_updates[i];
  }
  for (uint64_t i = 0; i < output_size; ++i) {
    output[i] = T();
  }

  vector<void *> datas = {static_cast<void *>(data.get()), static_cast<void *>(indices.get()),
                          static_cast<void *>(updates.get()), static_cast<void *>(output.get())};
  CREATE_NODEDEF(shapes, data_types, datas, axis, reduction);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  EXPECT_TRUE(CompareResult(output.get(), expect_output, output_size));
}

TEST_F(TEST_SCATTER_ELEMENTS_UT, DATA_TYPE_DT_DOUBLE_NONE_SUCC) {
  vector<DataType> data_types = {DT_DOUBLE, DT_INT64, DT_DOUBLE, DT_DOUBLE};
  vector<vector<int64_t>> shapes = {{1, 5}, {1, 2}, {1, 2}, {1, 5}};
  double input_data[5] = {1, 2, 3, 4, 5};
  int64_t input_indices[2] = {1, 3};
  double input_updates[2] = {10, 20};
  double expect_output[5] = {1, 10, 3, 20, 5};
  RunScatterElementsKernel(shapes, data_types, input_data, input_indices, input_updates, expect_output, 1);
}

TEST_F(TEST_SCATTER_ELEMENTS_UT, DATA_TYPE_DT_COMPLEX64_ADD_SUCC) {
  vector<DataType> data_types = {DT_COMPLEX64, DT_INT32, DT_COMPLEX64, DT_COMPLEX64};
  vector<vector<int64_t>> shapes = {{1, 4}, {1, 2}, {1, 2}, {1, 4}};
  complex<float> input_data[4] = {{1, 1}, {2, 2}, {3, 3}, {4, 4}};
  int32_t input_indices[2] = {1, 3};
  complex<float> input_updates[2] = {{10, -1}, {20, -2}};
  complex<float> expect_output[4] = {{1, 1}, {12, 1}, {3, 3}, {24, 2}};
  RunScatterElementsKernel(shapes, data_types, input_data, input_indices, input_updates, expect_output, 1, "add");
}

TEST_F(TEST_SCATTER_ELEMENTS_UT, DATA_TYPE_DT_BOOL_MUL_SUCC) {
  vector<DataType> data_types = {DT_BOOL, DT_INT32, DT_BOOL, DT_BOOL};
  vector<vector<int64_t>> shapes = {{1, 4}, {1, 2}, {1, 2}, {1, 4}};
  bool input_data[4] = {true, true, false, true};
  int32_t input_indices[2] = {0, 3};
  bool input_updates[2] = {false, true};
  bool expect_output[4] = {false, true, false, true};
  RunScatterElementsKernel(shapes, data_types, input_data, input_indices, input_updates, expect_output, 1, "mul");
}

TEST_F(TEST_SCATTER_ELEMENTS_UT, FAILED_OUT_OF_BOUND_INDEX) {
  vector<DataType> data_types = {DT_DOUBLE, DT_INT64, DT_DOUBLE, DT_DOUBLE};
  vector<vector<int64_t>> shapes = {{1, 5}, {1, 2}, {1, 2}, {1, 5}};
  double input_data[5] = {1, 2, 3, 4, 5};
  int64_t input_indices[2] = {1, 8};
  double input_updates[2] = {10, 20};
  double output[5] = {0};
  vector<void *> datas = {static_cast<void *>(input_data), static_cast<void *>(input_indices),
                          static_cast<void *>(input_updates), static_cast<void *>(output)};
  CREATE_NODEDEF(shapes, data_types, datas, 1, "none");
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

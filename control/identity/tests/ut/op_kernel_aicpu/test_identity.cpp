/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
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

class TEST_IDENTITY_UT : public testing::Test {};

#define CREATE_NODEDEF(shapes, data_types, datas)                   \
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();  \
  NodeDefBuilder(node_def.get(), "Identity", "Identity")            \
      .Input({"x", data_types[0], shapes[0], datas[0]})             \
      .Output({"y", data_types[1], shapes[1], datas[1]})

#define ADD_CASE(base_type, aicpu_type)                                     \
  TEST_F(TEST_IDENTITY_UT, TestIdentifyBroad_##aicpu_type) {                \
    vector<DataType> data_types = {aicpu_type, aicpu_type};                 \
    vector<vector<int64_t>> shapes = {{2, 11}, {2, 11}};                    \
    vector<base_type> input(22);                                            \
    SetRandomValue<base_type>(input.data(), 22);                            \
    vector<base_type> output(22);                                           \
    vector<void *> datas = {(void *)input.data(), (void *)output.data()};   \
    CREATE_NODEDEF(shapes, data_types, datas);                              \
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);                           \
    vector<base_type> expect_out = input;                                   \
    EXPECT_EQ(output, expect_out);                                          \
  }

TEST_F(TEST_IDENTITY_UT, ExpSizeNotMatch) {
  constexpr int32_t INPUT_SIZE = 16;
  constexpr int32_t OUTPUT_SIZE = 20;
  vector<DataType> data_types = {DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{2, 2, 4}, {2, 2, 5}};
  vector<int32_t> input(INPUT_SIZE);
  vector<int32_t> output(OUTPUT_SIZE);
  vector<void *> datas = {(void *)input.data(), (void *)output.data()};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_IDENTITY_UT, ExpInputNull) {
  constexpr int32_t OUTPUT_SIZE = 20;
  vector<DataType> data_types = {DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{2, 11}, {2, 11}};
  vector<int32_t> output(OUTPUT_SIZE);
  vector<void *> datas = {(void *)nullptr, (void *)output.data()};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_IDENTITY_UT, ExpOutputNull) {
  constexpr int32_t OUTPUT_SIZE = 22;
  vector<DataType> data_types = {DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{2, 11}, {2, 11}};
  vector<int32_t> input(OUTPUT_SIZE);
  vector<void *> datas = {(void *)input.data(), (void *)nullptr};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

ADD_CASE(Eigen::half, DT_FLOAT16)

ADD_CASE(float, DT_FLOAT)

ADD_CASE(int8_t, DT_INT8)

ADD_CASE(int16_t, DT_INT16)

ADD_CASE(int32_t, DT_INT32)

ADD_CASE(int64_t, DT_INT64)

ADD_CASE(uint8_t, DT_UINT8)

ADD_CASE(uint16_t, DT_UINT16)

ADD_CASE(uint32_t, DT_UINT32)

ADD_CASE(uint64_t, DT_UINT64)

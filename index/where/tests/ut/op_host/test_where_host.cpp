/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include "utils/status.h"
#include "register/kernel_registry_impl.h"
#include "kernel_run_context_facker.h"
#include "exe_graph/runtime/extended_kernel_context.h"
#include "runtime/aicpu_host_common.h"
#include "aicpu_test_utils.h"
#include <iostream>
#include "Eigen/Core"
#include <complex>

#define WHERE_SUCCESS_CASE(aicpu_type, base_type)                                                                       \
  TEST_F(WhereHostKernelTest, success_##aicpu_type) {                                                                   \
    gert::Shape input1_shape = {2, 2, 2};                                                                               \
    base_type input1_tensor_buffer[] =                                                                                  \
      {(base_type)0, (base_type)2, (base_type)3, (base_type)3, (base_type)1, (base_type)0, (base_type)0, (base_type)1}; \
    gert::TensorData input1_tensor_data{(void*)input1_tensor_buffer, nullptr};                                          \
    gert::Shape output_shape = {5, 3};                                                                                  \
    int64_t output_tensor_buffer[15];                                                                                   \
    gert::TensorData output_tensor_data{(void*)output_tensor_buffer, nullptr};                                          \
    auto kernelHolder =                                                                                                 \
        gert::KernelRunContextFaker()                                                                                   \
            .KernelIONum(2, 2)                                                                                          \
            .Inputs({reinterpret_cast<void*>(&input1_shape),                                                            \
                    reinterpret_cast<void*>(&input1_tensor_data)})                                                      \
            .Outputs({reinterpret_cast<void*>(&output_shape), reinterpret_cast<void*>(&output_tensor_data)})            \
            .NodeIoNum(1, 1)                                                                                            \
            .IrInstanceNum({1})                                                                                         \
            .NodeInputTd(0, ge::aicpu_type, ge::FORMAT_ND, ge::FORMAT_ND)                                               \
            .NodeOutputTd(0, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)                                                \
            .Build();                                                                                                   \
    auto context = kernelHolder.GetContext<KernelContext>();                                                            \
    ASSERT_NE(whereHostKernel, nullptr);                                                                                \
    ASSERT_EQ(whereHostKernel(context), aicpu::KERNEL_STATUS_OK);                                             \
    int64_t output_exp[15] = {0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1};                                             \
    EXPECT_EQ(CompareResult<int64_t>(output_tensor_buffer, output_exp, 15), true);                                      \
  }

namespace gert {
struct WhereHostKernelTest : public testing::Test {
  WhereHostKernelTest() {
    whereHostKernel = aicpu::AicpuHostKernelRegister::GetInstance()->GetFunc("Where");
  }
  aicpu::AicpuHostProcFunc whereHostKernel;
};

WHERE_SUCCESS_CASE(DT_INT32, int32_t)
WHERE_SUCCESS_CASE(DT_FLOAT, float_t)
WHERE_SUCCESS_CASE(DT_FLOAT16, Eigen::half)
WHERE_SUCCESS_CASE(DT_DOUBLE, double_t)
WHERE_SUCCESS_CASE(DT_INT8, int8_t)
WHERE_SUCCESS_CASE(DT_UINT8, uint8_t)
WHERE_SUCCESS_CASE(DT_INT16, int16_t)
WHERE_SUCCESS_CASE(DT_UINT16, uint16_t)
WHERE_SUCCESS_CASE(DT_UINT32, uint32_t)
WHERE_SUCCESS_CASE(DT_INT64, int64_t)
WHERE_SUCCESS_CASE(DT_BOOL, bool)
WHERE_SUCCESS_CASE(DT_COMPLEX64, std::complex<float>)
WHERE_SUCCESS_CASE(DT_COMPLEX128, std::complex<double>)

TEST_F(WhereHostKernelTest, test_empty_tensor) {
  gert::Shape input1_shape = {0, 2};
  gert::TensorData input1_tensor_data{nullptr, nullptr};
  gert::Shape output_shape = {0, 2};
  gert::TensorData output_tensor_data{(void*)nullptr, nullptr};
  auto kernelHolder =
      gert::KernelRunContextFaker()
          .KernelIONum(2, 2)
          .Inputs({reinterpret_cast<void*>(&input1_shape),
                   reinterpret_cast<void*>(&input1_tensor_data)})
          .Outputs({reinterpret_cast<void*>(&output_shape), reinterpret_cast<void*>(&output_tensor_data)})
          .NodeIoNum(1, 1)
          .IrInstanceNum({1})
          .NodeInputTd(0, ge::DT_COMPLEX64, ge::FORMAT_ND, ge::FORMAT_ND)
          .NodeOutputTd(0, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
          .Build();
  auto context = kernelHolder.GetContext<KernelContext>();
  ASSERT_NE(whereHostKernel, nullptr);
  ASSERT_EQ(whereHostKernel(context), aicpu::KERNEL_STATUS_OK);
}

TEST_F(WhereHostKernelTest, test_output_type_int32_err) {
  gert::Shape input1_shape = {2, 2, 2};
  int32_t input1_tensor_buffer[] = {0, 2, 3, 3, 1, 0, 0, 1};
  gert::TensorData input1_tensor_data{(void*)input1_tensor_buffer, nullptr};
  gert::Shape output_shape = {5, 3};
  int32_t output_tensor_buffer[15];
  gert::TensorData output_tensor_data{(void*)output_tensor_buffer, nullptr};
  auto kernelHolder =
      gert::KernelRunContextFaker()
          .KernelIONum(2, 2)
          .Inputs({reinterpret_cast<void*>(&input1_shape),
                   reinterpret_cast<void*>(&input1_tensor_data)})
          .Outputs({reinterpret_cast<void*>(&output_shape), reinterpret_cast<void*>(&output_tensor_data)})
          .NodeIoNum(1, 1)
          .IrInstanceNum({1})
          .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
          .NodeOutputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
          .Build();

  auto context = kernelHolder.GetContext<KernelContext>();
  ASSERT_NE(whereHostKernel, nullptr);
  ASSERT_EQ(whereHostKernel(context), aicpu::KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(WhereHostKernelTest, test_intput_dtype_err) {
  gert::Shape input1_shape = {2, 2, 1, 1};
  std::complex<float> input1_tensor_buffer[] = {0, 1, 0, 0};
  gert::TensorData input1_tensor_data{(void*)input1_tensor_buffer, nullptr};
  gert::Shape output_shape = {1, 4};
  int64_t output_tensor_buffer[4];
  gert::TensorData output_tensor_data{(void*)output_tensor_buffer, nullptr};
  auto kernelHolder =
      gert::KernelRunContextFaker()
          .KernelIONum(2, 2)
          .Inputs({reinterpret_cast<void*>(&input1_shape),
                   reinterpret_cast<void*>(&input1_tensor_data)})
          .Outputs({reinterpret_cast<void*>(&output_shape), reinterpret_cast<void*>(&output_tensor_data)})
          .NodeIoNum(1, 1)
          .IrInstanceNum({1})
          .NodeInputTd(0, ge::DT_MAX, ge::FORMAT_ND, ge::FORMAT_ND)
          .NodeOutputTd(0, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
          .Build();

  auto context = kernelHolder.GetContext<KernelContext>();
  ASSERT_NE(whereHostKernel, nullptr);
  ASSERT_EQ(whereHostKernel(context), ge::GRAPH_FAILED);
}

TEST_F(WhereHostKernelTest, test_intput_dim_exceed_8) {
  gert::Shape input1_shape = {2, 1, 1, 1, 1, 1, 1, 2, 2};
  int32_t input1_tensor_buffer[] = {0, 2, 3, 3, 1, 0, 0, 1};
  gert::TensorData input1_tensor_data{(void*)input1_tensor_buffer, nullptr};
  gert::Shape output_shape = {5, 3};
  int64_t output_tensor_buffer[15];
  gert::TensorData output_tensor_data{(void*)output_tensor_buffer, nullptr};
  auto kernelHolder =
      gert::KernelRunContextFaker()
          .KernelIONum(2, 2)
          .Inputs({reinterpret_cast<void*>(&input1_shape),
                   reinterpret_cast<void*>(&input1_tensor_data)})
          .Outputs({reinterpret_cast<void*>(&output_shape), reinterpret_cast<void*>(&output_tensor_data)})
          .NodeIoNum(1, 1)
          .IrInstanceNum({1})
          .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
          .NodeOutputTd(0, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
          .Build();

  auto context = kernelHolder.GetContext<KernelContext>();
  ASSERT_NE(whereHostKernel, nullptr);
  ASSERT_EQ(whereHostKernel(context), aicpu::KERNEL_STATUS_INNER_ERROR);
}

TEST_F(WhereHostKernelTest, test_intput_rank1) {
  gert::Shape input1_shape = {2};
  std::complex<float> input1_tensor_buffer[] = {0, 2 + 1j};
  gert::TensorData input1_tensor_data{(void*)input1_tensor_buffer, nullptr};
  gert::Shape output_shape = {1, 1};
  int64_t output_tensor_buffer[1];
  gert::TensorData output_tensor_data{(void*)output_tensor_buffer, nullptr};

  auto kernelHolder =
      gert::KernelRunContextFaker()
          .KernelIONum(2, 2)
          .Inputs({reinterpret_cast<void*>(&input1_shape),
                  reinterpret_cast<void*>(&input1_tensor_data)})
          .Outputs({reinterpret_cast<void*>(&output_shape), reinterpret_cast<void*>(&output_tensor_data)})
          .NodeIoNum(1, 1)
          .IrInstanceNum({1})
          .NodeInputTd(0, ge::DT_COMPLEX64, ge::FORMAT_ND, ge::FORMAT_ND)
          .NodeOutputTd(0, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
          .Build();
  auto context = kernelHolder.GetContext<KernelContext>();
  ASSERT_NE(whereHostKernel, nullptr);
  ASSERT_EQ(whereHostKernel(context), aicpu::KERNEL_STATUS_OK);
  int64_t output_exp[1] = {1};
  EXPECT_EQ(CompareResult<int64_t>(output_tensor_buffer, output_exp, 1), true);
}

TEST_F(WhereHostKernelTest, test_intput_rank2) {
  gert::Shape input1_shape = {2, 2};
  std::complex<float> input1_tensor_buffer[] = {0, 2 + 1j, 0, 1};
  gert::TensorData input1_tensor_data{(void*)input1_tensor_buffer, nullptr};
  gert::Shape output_shape = {2, 2};
  int64_t output_tensor_buffer[4];
  gert::TensorData output_tensor_data{(void*)output_tensor_buffer, nullptr};

  auto kernelHolder =
      gert::KernelRunContextFaker()
          .KernelIONum(2, 2)
          .Inputs({reinterpret_cast<void*>(&input1_shape),
                  reinterpret_cast<void*>(&input1_tensor_data)})
          .Outputs({reinterpret_cast<void*>(&output_shape), reinterpret_cast<void*>(&output_tensor_data)})
          .NodeIoNum(1, 1)
          .IrInstanceNum({1})
          .NodeInputTd(0, ge::DT_COMPLEX64, ge::FORMAT_ND, ge::FORMAT_ND)
          .NodeOutputTd(0, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
          .Build();
  auto context = kernelHolder.GetContext<KernelContext>();
  ASSERT_NE(whereHostKernel, nullptr);
  ASSERT_EQ(whereHostKernel(context), aicpu::KERNEL_STATUS_OK);
  int64_t output_exp[4] = {0, 1, 1, 1};
  EXPECT_EQ(CompareResult<int64_t>(output_tensor_buffer, output_exp, 4), true);
}

TEST_F(WhereHostKernelTest, test_intput_rank4) {
  gert::Shape input1_shape = {2, 2, 1, 1};
  std::complex<float> input1_tensor_buffer[] = {0, 1, 0, 0};
  gert::TensorData input1_tensor_data{(void*)input1_tensor_buffer, nullptr};
  gert::Shape output_shape = {1, 4};
  int64_t output_tensor_buffer[4];
  gert::TensorData output_tensor_data{(void*)output_tensor_buffer, nullptr};

  auto kernelHolder =
      gert::KernelRunContextFaker()
          .KernelIONum(2, 2)
          .Inputs({reinterpret_cast<void*>(&input1_shape),
                  reinterpret_cast<void*>(&input1_tensor_data)})
          .Outputs({reinterpret_cast<void*>(&output_shape), reinterpret_cast<void*>(&output_tensor_data)})
          .NodeIoNum(1, 1)
          .IrInstanceNum({1})
          .NodeInputTd(0, ge::DT_COMPLEX64, ge::FORMAT_ND, ge::FORMAT_ND)
          .NodeOutputTd(0, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
          .Build();
  auto context = kernelHolder.GetContext<KernelContext>();
  ASSERT_NE(whereHostKernel, nullptr);
  ASSERT_EQ(whereHostKernel(context), aicpu::KERNEL_STATUS_OK);
  int64_t output_exp[4] = {0, 1, 0, 0};
  EXPECT_EQ(CompareResult<int64_t>(output_tensor_buffer, output_exp, 4), true);
}

TEST_F(WhereHostKernelTest, test_intput_rank5) {
  gert::Shape input1_shape = {2, 2, 1, 1, 1};
  std::complex<float> input1_tensor_buffer[] = {0, 1, 0, 0};
  gert::TensorData input1_tensor_data{(void*)input1_tensor_buffer, nullptr};
  gert::Shape output_shape = {1, 5};
  int64_t output_tensor_buffer[5];
  gert::TensorData output_tensor_data{(void*)output_tensor_buffer, nullptr};

  auto kernelHolder =
      gert::KernelRunContextFaker()
          .KernelIONum(2, 2)
          .Inputs({reinterpret_cast<void*>(&input1_shape),
                  reinterpret_cast<void*>(&input1_tensor_data)})
          .Outputs({reinterpret_cast<void*>(&output_shape), reinterpret_cast<void*>(&output_tensor_data)})
          .NodeIoNum(1, 1)
          .IrInstanceNum({1})
          .NodeInputTd(0, ge::DT_COMPLEX64, ge::FORMAT_ND, ge::FORMAT_ND)
          .NodeOutputTd(0, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
          .Build();
  auto context = kernelHolder.GetContext<KernelContext>();
  ASSERT_NE(whereHostKernel, nullptr);
  ASSERT_EQ(whereHostKernel(context), aicpu::KERNEL_STATUS_OK);
  int64_t output_exp[5] = {0, 1, 0, 0, 0};
  EXPECT_EQ(CompareResult<int64_t>(output_tensor_buffer, output_exp, 5), true);
}

TEST_F(WhereHostKernelTest, test_intput_rank6) {
  gert::Shape input1_shape = {2, 2, 1, 1, 1, 1};
  std::complex<float> input1_tensor_buffer[] = {0, 1, 0, 0};
  gert::TensorData input1_tensor_data{(void*)input1_tensor_buffer, nullptr};
  gert::Shape output_shape = {1, 6};
  int64_t output_tensor_buffer[6];
  gert::TensorData output_tensor_data{(void*)output_tensor_buffer, nullptr};

  auto kernelHolder =
      gert::KernelRunContextFaker()
          .KernelIONum(2, 2)
          .Inputs({reinterpret_cast<void*>(&input1_shape),
                  reinterpret_cast<void*>(&input1_tensor_data)})
          .Outputs({reinterpret_cast<void*>(&output_shape), reinterpret_cast<void*>(&output_tensor_data)})
          .NodeIoNum(1, 1)
          .IrInstanceNum({1})
          .NodeInputTd(0, ge::DT_COMPLEX64, ge::FORMAT_ND, ge::FORMAT_ND)
          .NodeOutputTd(0, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
          .Build();
  auto context = kernelHolder.GetContext<KernelContext>();
  ASSERT_NE(whereHostKernel, nullptr);
  ASSERT_EQ(whereHostKernel(context), aicpu::KERNEL_STATUS_OK);
  int64_t output_exp[6] = {0, 1, 0, 0, 0};
  EXPECT_EQ(CompareResult<int64_t>(output_tensor_buffer, output_exp, 6), true);
}

TEST_F(WhereHostKernelTest, test_intput_rank7) {
  gert::Shape input1_shape = {2, 2, 1, 1, 1, 1, 1};
  std::complex<float> input1_tensor_buffer[] = {0, 1, 0, 0};
  gert::TensorData input1_tensor_data{(void*)input1_tensor_buffer, nullptr};
  gert::Shape output_shape = {1, 7};
  int64_t output_tensor_buffer[7];
  gert::TensorData output_tensor_data{(void*)output_tensor_buffer, nullptr};

  auto kernelHolder =
      gert::KernelRunContextFaker()
          .KernelIONum(2, 2)
          .Inputs({reinterpret_cast<void*>(&input1_shape),
                   reinterpret_cast<void*>(&input1_tensor_data)})
          .Outputs({reinterpret_cast<void*>(&output_shape), reinterpret_cast<void*>(&output_tensor_data)})
          .NodeIoNum(1, 1)
          .IrInstanceNum({1})
          .NodeInputTd(0, ge::DT_COMPLEX64, ge::FORMAT_ND, ge::FORMAT_ND)
          .NodeOutputTd(0, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
          .Build();
  auto context = kernelHolder.GetContext<KernelContext>();
  ASSERT_NE(whereHostKernel, nullptr);
  ASSERT_EQ(whereHostKernel(context), aicpu::KERNEL_STATUS_OK);
  int64_t output_exp[7] = {0, 1, 0, 0, 0, 0, 0};
  EXPECT_EQ(CompareResult<int64_t>(output_tensor_buffer, output_exp, 7), true);
}

TEST_F(WhereHostKernelTest, test_intput_rank8) {
  gert::Shape input1_shape = {2, 2, 1, 1, 1, 1, 1, 1};
  std::complex<float> input1_tensor_buffer[] = {0, 1, 0, 0};
  gert::TensorData input1_tensor_data{(void*)input1_tensor_buffer, nullptr};
  gert::Shape output_shape = {1, 8};
  int64_t output_tensor_buffer[8];
  gert::TensorData output_tensor_data{(void*)output_tensor_buffer, nullptr};

  auto kernelHolder =
      gert::KernelRunContextFaker()
          .KernelIONum(2, 2)
          .Inputs({reinterpret_cast<void*>(&input1_shape),
                  reinterpret_cast<void*>(&input1_tensor_data)})
          .Outputs({reinterpret_cast<void*>(&output_shape), reinterpret_cast<void*>(&output_tensor_data)})
          .NodeIoNum(1, 1)
          .IrInstanceNum({1})
          .NodeInputTd(0, ge::DT_COMPLEX64, ge::FORMAT_ND, ge::FORMAT_ND)
          .NodeOutputTd(0, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
          .Build();
  auto context = kernelHolder.GetContext<KernelContext>();
  ASSERT_NE(whereHostKernel, nullptr);
  ASSERT_EQ(whereHostKernel(context), aicpu::KERNEL_STATUS_OK);
  int64_t output_exp[8] = {0, 1, 0, 0, 0, 0, 0, 0};
  EXPECT_EQ(CompareResult<int64_t>(output_tensor_buffer, output_exp, 8), true);
}
}  // namespace gert
/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AICPU_KERNELS_ASSERT_H_
#define AICPU_KERNELS_ASSERT_H_

#include <string>
#include "cpu_kernel.h"
namespace aicpu {
class AssertCpuKernel : public CpuKernel {
 public:
  AssertCpuKernel() = default;
  ~AssertCpuKernel() = default;
  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  std::string SummarizeValue(const Tensor &t, int64_t max_entries);
  template <typename T>
  std::string SummarizeArray(const int64_t limit, const int64_t num_elts,
                             const Tensor &t);
  template <typename T>
  void PrintOneDim(int dim_index, std::shared_ptr<TensorShape> shape,
                   int64_t limit, int shape_size, const T *data,
                   int64_t *data_index, std::string &result);
};
}

#endif  // AICPU_KERNELS_ASSERT_H_

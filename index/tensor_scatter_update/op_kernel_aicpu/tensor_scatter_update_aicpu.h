/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_NN_INDEX_TENSOR_SCATTER_UPDATE_AICPU_H_
#define OPS_NN_INDEX_TENSOR_SCATTER_UPDATE_AICPU_H_

#include "cpu_kernel.h"
#include "cpu_kernel_utils.h"
#include "status.h"
#include "unsupported/Eigen/CXX11/Tensor"

namespace aicpu {
struct CpuKernelContextInfo {
  CpuKernelContext *ctx;
  Tensor *x;
  Tensor *indices;
  Tensor *updates;
  Tensor *y;

  std::shared_ptr<TensorShape> x_shape;
  std::shared_ptr<TensorShape> indices_shape;
  std::shared_ptr<TensorShape> updates_shape;

  uint64_t x_dims;
  uint64_t indices_dims;
  uint64_t updates_dims;

  uint64_t x_nums;
  uint64_t indices_nums;
  uint64_t updates_nums;

  DataType indices_type;
  DataType x_type;

  uint64_t index_depth;
  uint64_t num_updates;
};

class TensorScatterUpdateCpukernel : public CpuKernel {
 public:
  TensorScatterUpdateCpukernel() = default;
  ~TensorScatterUpdateCpukernel() override = default;
  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  uint32_t GetInputAndCheck(const CpuKernelContextInfo &info);
};
} // namespace aicpu

#endif // OPS_NN_INDEX_TENSOR_SCATTER_UPDATE_AICPU_H_

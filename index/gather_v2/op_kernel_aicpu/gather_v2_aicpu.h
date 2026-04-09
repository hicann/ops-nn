/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_NN_INDEX_GATHER_V2_OP_KERNEL_AICPU_GATHER_V2_AICPU_H_
#define OPS_NN_INDEX_GATHER_V2_OP_KERNEL_AICPU_GATHER_V2_AICPU_H_

#include "cpu_kernel.h"
#include "cpu_kernel_utils.h"
#include "status.h"
#include "unsupported/Eigen/CXX11/Tensor"

namespace aicpu {
class GatherV2CpuKernel : public CpuKernel {
 public:
  GatherV2CpuKernel() = default;
  ~GatherV2CpuKernel() override = default;
  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  uint32_t GetInputAndCheck(const CpuKernelContext &ctx);
};
}  // namespace aicpu

#endif  // OPS_NN_INDEX_GATHER_V2_OP_KERNEL_AICPU_GATHER_V2_AICPU_H_

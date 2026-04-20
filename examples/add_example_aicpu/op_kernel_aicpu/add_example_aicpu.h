/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.

 * The code snippet comes from Huawei's open-source Ascend project.
 * Copyright 2021 Huawei Technologies Co., Ltd
 * Licensed under the Apache License, Version 2.0 (the "License");
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 */

/*!
 * \file add_example_aicpu.h
 * \brief CPU kernel implementation of AddExample operator.
 *        This file defines the AddExampleCpuKernel class which performs element-wise
 *        addition of two input tensors on the CPU (AICPU).
 */
#ifndef AICPU_ADD_EXAMPLE_CPU_KERNELS_H_
#define AICPU_ADD_EXAMPLE_CPU_KERNELS_H_

#include "cpu_kernel.h"

namespace aicpu {
/*!
 * \brief AddExample CPU kernel class.
 *        Implements element-wise tensor addition on CPU.
 */
class AddExampleCpuKernel : public CpuKernel {
 public:
  ~AddExampleCpuKernel() = default;
  /*!
   * \brief Entry point for kernel computation.
   * \param ctx Kernel context containing input/output tensors.
   * \return kSuccess on success, error code on failure.
   */
  uint32_t Compute(CpuKernelContext &ctx) override;
  /*!
   * \brief Template function for type-specific addition computation.
   * \param ctx Kernel context containing input/output tensors.
   * \return kSuccess on success, error code on failure.
   */
  template<typename T>
  uint32_t AddCompute(CpuKernelContext &ctx);
};
}  // namespace aicpu
#endif

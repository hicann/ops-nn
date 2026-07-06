/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AICPU_KERNEL_SCATTER_ND_MAX_H_
#define AICPU_KERNEL_SCATTER_ND_MAX_H_

#include <memory>
#include <string>
#include <vector>

#include "cpu_kernel.h"
#include "cpu_kernel_utils.h"
#include "log.h"
#include "securec.h"

namespace aicpu {
class ScatterNdMaxCpuKernel : public CpuKernel {
public:
    ScatterNdMaxCpuKernel() = default;
    ~ScatterNdMaxCpuKernel() override = default;
    uint32_t Compute(CpuKernelContext& ctx) override;

private:
    template <typename T, typename TI>
    uint32_t DoScatterNdMaxCompute(CpuKernelContext& ctx);
    template <typename TI>
    uint32_t CheckScatterNdMaxShapeAndData(const CpuKernelContext& ctx);
    template <typename T>
    uint32_t InitScatterNdMaxOutput(const CpuKernelContext& ctx);
};
} // namespace aicpu

#endif

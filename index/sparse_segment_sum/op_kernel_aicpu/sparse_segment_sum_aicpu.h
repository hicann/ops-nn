/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef AICPU_KERNEL_SPARSE_SEGMENT_SUM_H_
#define AICPU_KERNEL_SPARSE_SEGMENT_SUM_H_

#include "cpu_kernel.h"
#include "utils/bcast.h"
#include "utils/kernel_util.h"

namespace aicpu {
class SparseSegmentSumCpuKernel : public CpuKernel {
public:
    SparseSegmentSumCpuKernel() = default;
    ~SparseSegmentSumCpuKernel() override = default;
    uint32_t Compute(CpuKernelContext& ctx) override;

private:
    KernelStatus SparseSegmentCheck(const CpuKernelContext& ctx) const;
    KernelStatus SparseSegmentDataCheck(const CpuKernelContext& ctx) const;

    template <typename T1, typename T2>
    KernelStatus SparseSegmentDataCheckWithType(const CpuKernelContext& ctx) const;

    KernelStatus ComputeWithType(const CpuKernelContext& ctx);

    template <typename T>
    KernelStatus ComputeKernel(const CpuKernelContext& ctx) const;

    template <typename T, typename T1, typename T2>
    KernelStatus ComputeKernelWithType(const CpuKernelContext& ctx) const;
};
} // namespace aicpu
#endif

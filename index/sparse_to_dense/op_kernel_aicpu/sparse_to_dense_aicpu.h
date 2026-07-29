/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_BUILT_IN_OP_KERNEL_AICPU_SPARSE_TO_DENSE_AICPU_H_
#define OPS_BUILT_IN_OP_KERNEL_AICPU_SPARSE_TO_DENSE_AICPU_H_

#include "cpu_kernel.h"
#include "status.h"

namespace aicpu {
class SparseToDenseCpuKernel : public CpuKernel {
public:
    ~SparseToDenseCpuKernel() override = default;
    uint32_t Compute(CpuKernelContext& ctx) override;

protected:
    KernelStatus ValidParam(const CpuKernelContext& ctx);
    uint32_t ParallelSetDefaultValue(const CpuKernelContext& ctx, const Tensor* defaultValueTensor,
                                     const Tensor* outputTensor, int64_t outputSize);
    uint32_t SetDefaultValue(const CpuKernelContext& ctx, const Tensor* defaultValueTensor, const Tensor* outputTensor,
                             int64_t outputSize);
};
} // namespace aicpu

#endif // OPS_BUILT_IN_OP_KERNEL_AICPU_SPARSE_TO_DENSE_AICPU_H_

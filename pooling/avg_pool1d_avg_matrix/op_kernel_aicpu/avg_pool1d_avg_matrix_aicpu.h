/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_NN_AVG_POOL1D_AVG_MATRIX_AICPU_H_
#define OPS_NN_AVG_POOL1D_AVG_MATRIX_AICPU_H_

#include "cpu_kernel.h"

namespace aicpu {
class AvgPool1DAvgMatrixCpuKernel : public CpuKernel {
public:
    AvgPool1DAvgMatrixCpuKernel() = default;
    ~AvgPool1DAvgMatrixCpuKernel() = default;
    uint32_t Compute(CpuKernelContext& ctx) override;

private:
    /**
     * @brief Init params
     * @param ctx cpu kernel context
     * @return status if success
     */
    uint32_t CheckParam(CpuKernelContext& ctx);

    /**
     * @brief Init params
     * @param ctx cpu kernel context
     * @return status if success
     */
    template <typename T>
    uint32_t DoCompute(CpuKernelContext& ctx);
};
} // namespace aicpu

#endif // OPS_NN_AVG_POOL1D_AVG_MATRIX_AICPU_H_

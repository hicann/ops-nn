/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under
 * the terms and conditions of CANN Open Software License Agreement Version 2.0
 * (the "License"). Please refer to the License for details. You may not use
 * this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
 * AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
 * FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
 * for the full text of the License.
 */

#ifndef OPS_NN_INDEX_SPARSE_FILL_EMPTY_ROWS_OP_KERNEL_AICPU_SPARSE_FILL_EMPTY_ROWS_AICPU_H_
#define OPS_NN_INDEX_SPARSE_FILL_EMPTY_ROWS_OP_KERNEL_AICPU_SPARSE_FILL_EMPTY_ROWS_AICPU_H_

#include <vector>

#include "cpu_kernel.h"
#include "status.h"
#include "utils/eigen_tensor.h"
#include "unsupported/Eigen/CXX11/Tensor"

namespace aicpu {
class SparseFillEmptyRowsCpuKernel : public CpuKernel {
public:
    SparseFillEmptyRowsCpuKernel() = default;
    ~SparseFillEmptyRowsCpuKernel() override = default;
    uint32_t Compute(CpuKernelContext& ctx) override;

private:
    template <typename T>
    KernelStatus ComputeSparseFillEmptyRows(const CpuKernelContext& ctx, Tensor* indices, const Tensor* values,
                                            const Tensor* denseShape, const Tensor* defaultValueTensor);

    template <typename T>
    void FillValue(const TTypes<int64_t>::Matrix& indicesMatrix, std::vector<int64_t>& csrOffset,
                   TTypes<int64_t>::Matrix& outputIndicesMatrix, const T* valuesData, T* outputValuesData,
                   int64_t* reverseIndexMapData, const T defaultValue);
};
} // namespace aicpu

#endif // OPS_NN_INDEX_SPARSE_FILL_EMPTY_ROWS_OP_KERNEL_AICPU_SPARSE_FILL_EMPTY_ROWS_AICPU_H_

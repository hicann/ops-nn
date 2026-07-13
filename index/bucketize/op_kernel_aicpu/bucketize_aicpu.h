/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_NN_BUCKETIZE_AICPU_H_
#define OPS_NN_BUCKETIZE_AICPU_H_

#include <vector>
#include "cpu_kernel.h"
#include "cpu_types.h"

namespace aicpu {
class BucketizeCpuKernel : public CpuKernel {
public:
    BucketizeCpuKernel() = default;
    ~BucketizeCpuKernel() = default;
    uint32_t Compute(CpuKernelContext& ctx) override;

private:
    std::vector<float> boundaries_data_;

    uint32_t BucketizeParamsCheck(CpuKernelContext& ctx);

    template <typename T, typename T2>
    uint32_t BucketizeCompute(const CpuKernelContext& ctx) const;

    template <typename T, typename T2>
    uint32_t BucketizeComputeParallel(const CpuKernelContext& ctx, const T* input_data, T2* output_data,
                                      int64_t data_num, bool right) const;

    // Upper == true  -> std::upper_bound semantics (right=true)
    // Upper == false -> std::lower_bound semantics (right=false)
    template <typename T, typename T2, bool Upper>
    void DoComputeBound(const T* input_data, T2* output_data, int64_t start, int64_t end) const;
};
} // namespace aicpu

#endif // OPS_NN_BUCKETIZE_AICPU_H_

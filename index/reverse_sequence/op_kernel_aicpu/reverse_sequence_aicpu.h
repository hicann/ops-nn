/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_NN_INDEX_REVERSE_SEQUENCE_AICPU_H
#define OPS_NN_INDEX_REVERSE_SEQUENCE_AICPU_H

#include "cpu_kernel.h"
#include "utils/status.h"

namespace aicpu {
class ReverseSequenceMsCpuKernel : public CpuKernel {
public:
    ~ReverseSequenceMsCpuKernel() = default;
    uint32_t Compute(CpuKernelContext &ctx) override;

private:
    KernelStatus GetInputAndCheck(CpuKernelContext &ctx);

    std::vector<void *> ioAddrs_;
    std::vector<int64_t> xShape_;

    DataType xDtype_ = DT_INT32;
    DataType seqLengthsDtype_ = DT_INT32;
};
}  // namespace aicpu

#endif  //  OPS_NN_INDEX_REVERSE_SEQUENCE_AICPU_H

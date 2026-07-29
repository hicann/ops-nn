/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file batch_matmul_v3_tiling_advanced.h
 * \brief BatchMatMulV3 tiling, inherits MatMulV3Tiling.
 */
#pragma once

#include "matmul/mat_mul_v3/op_host/op_tiling/arch35/matmul_v3_tiling_advanced.h"

namespace optiling {
namespace batch_matmul_v3_advanced {
using namespace matmul_v3_advanced;
class BatchMatMulV3Tiling : public MatMulV3Tiling {
public:
    explicit BatchMatMulV3Tiling(gert::TilingContext* context) : MatMulV3Tiling(context) {};

    ~BatchMatMulV3Tiling() override = default;

protected:
    // ====== Phase 7: ValidateBias (bias shape[-2]==1) ======
    ge::graphStatus ValidateBias() override;

    // ====== Phase 8 sub-steps ======
    ge::graphStatus ExtractMatrixBatchInfo() override;
    ge::graphStatus ValidateMatrixBatchInfo() override;
    ge::graphStatus ExtractOptionalBatchInfo() override;

    // ====== Phase 9: ValidateOptionalBatchInfo ======
    ge::graphStatus ValidateOptionalBatchInfo() override;

    // ====== Phase 10: Registry delegation hooks ======
    const char* GetRegistryOpType() const override { return "BatchMatMulV3"; }
    std::vector<int32_t> GetRegistryPriorities(NpuArch npuArch) const override;

protected:
    MatMulV3BatchInfo batchInfo_{};

private:
    void MergeBatchAndMAxis(MatMulV3BatchInfo& batchInfo);
};
} // namespace batch_matmul_v3_advanced
} // namespace optiling

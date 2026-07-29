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
 * \file fused_matmul_builtin_tiling.h
 * \brief FusedMatMul built-in tiling, inherits BatchMatMulV3Tiling.
 */
#pragma once

#include <string>
#include "fused_matmul_common.h"
#include "fused_matmul_tiling_key.h"
#include "matmul/batch_mat_mul_v3/op_host/op_tiling/arch35/batch_matmul_v3_tiling_advanced.h"
#include "platform/platform_ascendc.h"

namespace optiling {
namespace fused_matmul {
using namespace batch_matmul_v3_advanced;

class FusedMatMulBuiltInTiling : public BatchMatMulV3Tiling {
public:
    explicit FusedMatMulBuiltInTiling(gert::TilingContext* context) : BatchMatMulV3Tiling(context) {};

    ~FusedMatMulBuiltInTiling() override = default;

protected:
    // ====== Phase 2: Input null-check (own attr layout: hf32/opType/innerPrecise) ======
    ge::graphStatus ValidateInputsNotNull() override;

    // ====== Phase 3: Optional input detection (bias + x3) ======
    ge::graphStatus DetectOptionalInputs() override;

    // ====== Phase 5: Dtype & attr-flag extraction (x3Type, hf32 from different attr) ======
    void ExtractDtype() override;
    void ExtractAttrFlags() override;

    // ====== Phase 7: Validation (gelu/add/mul constraints, bias shape, dtype) ======
    ge::graphStatus ValidateOpSpecific() override;
    ge::graphStatus ValidateBias() override;
    ge::graphStatus ValidateDtype() override;

    // ====== Phase 8 sub-steps (no batch bias, no broadcast on non-DAV_RESV) ======
    ge::graphStatus ValidateMatrixBatchInfo() override;
    ge::graphStatus ExtractOptionalBatchInfo() override;

    // ====== Phase 9: ValidateOptionalBatchInfo (x3 batch-axis broadcast only) ======
    ge::graphStatus ValidateOptionalBatchInfo() override;

    // ====== Phase 10: Registry delegation hooks ======
    const char* GetRegistryOpType() const override { return "FusedMatMul"; }
    std::vector<int32_t> GetRegistryPriorities(NpuArch npuArch) const override;
    MatMulV3TilingKey* GetTilingKeyObj() override;

    // ====== Dtype support list (opType + npuArch dependent) ======
    std::vector<std::vector<ge::DataType>> GetDtypeSupportList() const override;

private:
    FusedMatmulTilingKey fusedMatmulTilingKey_{};
    int64_t innerPrecise_ = 0;
    std::string opType_;
};
} // namespace fused_matmul
} // namespace optiling

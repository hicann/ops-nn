/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file gemm_syrk_tiling.h
 * \brief GemmSyrk tiling facade (DoTiling orchestrator).
 *
 * GemmSyrk: C = alpha * (A @ A^T) + beta * C, in-place on the symmetric c
 * ([..., m, m]); transpose_x binds the transposed (..., k, m) storage and
 * computes C = alpha * (A^T @ A) + beta * C. The kernel is the Blaze single-
 * fetch syrk assembly (MIX 1 AIC : 2 AIV). This facade extracts and validates
 * the syrk inputs, maps them onto the matmul arg model (n == m, B == A) and
 * dispatches through the tiling strategy priorities
 * (gemm_syrk_tiling_strategy.h) to the registered GemmSyrkBaseTiling, where
 * IsCapable (MIX ratio plus the mirrored batch/type constraints) and
 * DoOpTiling (the BatchMatMulV3 ASW basic computation followed by the syrk
 * symmetric square clamps) run; the clamped result is packed into the flat
 * GemmSyrkTilingData.
 */

#pragma once

#include <cstring>
#include <string>

#include "exe_graph/runtime/tiling_context.h"
#include "matmul/gemm_syrk/op_kernel/arch35/gemm_syrk_tiling_data.h"
#include "matmul/mat_mul_v3/op_kernel/arch35/mat_mul_tiling_data.h"
#include "matmul/mat_mul_v3/op_host/op_tiling/arch35/matmul_v3_common_advanced.h"
#include "matmul/mat_mul_v3/op_host/op_tiling/matmul_v3_compile_info.h"

namespace optiling {
namespace gemm_syrk {

using matmul_v3_advanced::MatMulV3Args;
using matmul_v3_advanced::MatMulV3BatchInfo;

class GemmSyrkTiling {
public:
    explicit GemmSyrkTiling(gert::TilingContext* context) : context_(context) {}
    ~GemmSyrkTiling() = default;

    ge::graphStatus DoTiling();

private:
    // Platform and shape/attr phases
    ge::graphStatus GetPlatformInfo();
    ge::graphStatus GetShapeAttrsInfo();
    ge::graphStatus ExtractAttrs();
    ge::graphStatus ExtractShape();
    ge::graphStatus ValidateDtype() const;
    ge::graphStatus ValidateFormat() const;
    ge::graphStatus ValidateShape() const;
    ge::graphStatus ValidateAttrs() const;

    // Tiling phases: map the syrk problem onto the matmul arg model, then
    // dispatch through the tiling strategy to the registered basic tiling
    // (IsCapable + the ASW basic computation + syrk clamps run there) and
    // pack the flat tiling data.
    void BuildMatmulArgs();
    ge::graphStatus RunBasicTilingByStrategy();
    void SetTilingData(const BatchMatMulV3BasicTilingData& bmmTiling);
    uint64_t GetTilingKey() const;
    ge::graphStatus PostTiling();

    gert::TilingContext* context_{nullptr};

    // Problem shape: a is [..., m, k] (or the transposed [..., k, m]), c is [..., m, m].
    uint64_t m_{0};
    uint64_t n_{0};
    uint64_t k_{0};
    uint64_t batch_{1};
    float alpha_{1.0F};
    float beta_{1.0F};
    bool transX_{false};
    std::string fillMode_{"full"};
    ge::DataType aType_{ge::DT_FLOAT16};
    uint64_t dtypeSize_{0};

    // The matmul compile info consumed by the basic tiling strategy.
    MatmulV3CompileInfo mmCompileInfo_{};

    // Matmul arg model fed to the basic tiling: B mirrors a, batchC from c.
    MatMulV3Args mmArgs_{};
    MatMulV3BatchInfo mmBatchInfo_{};
    BatchMatMulV3BasicTilingData bmmTiling_{};

    // Flat tiling result
    GemmSyrkTilingData tilingData_{};
};

} // namespace gemm_syrk
} // namespace optiling

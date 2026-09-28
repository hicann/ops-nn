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
 * \file gemm_syrk_base_tiling.h
 * \brief GemmSyrk base tiling template on top of the BatchMatMulV3 ASW basic
 * tiling strategy.
 *
 * Registered through the MM tiling registry (priority strategy::SYRK_BASE);
 * the GemmSyrkTiling facade dispatches here via the strategy priorities.
 * IsCapable (MIX 1 AIC : 2 AIV plus the mirrored batch/type constraints) and
 * DoOpTiling run in this template: DoOpTiling first runs the inherited
 * BatchMatMulV3 ASW basic computation (baseM/baseN/baseK rebalance across
 * cores, L1 tiling, tail handling), then applies the syrk-specific symmetric
 * square clamps to the runInfo so the base GetTilingDataProcess emits the
 * contract directly.
 */

#pragma once

#include "matmul/batch_mat_mul_v3/op_host/op_tiling/arch35/batch_matmul_v3_asw_basic_tiling.h"
#include "matmul/mat_mul_v3/op_host/op_tiling/arch35/matmul_tiling_cfg.h"

namespace optiling {
namespace gemm_syrk {
using batch_matmul_v3_advanced::BatchMatMulV3AswBasicTiling;

class GemmSyrkBaseTiling : public BatchMatMulV3AswBasicTiling {
public:
    GemmSyrkBaseTiling(gert::TilingContext* context, MatMulTilingCfg& cfg)
        : BatchMatMulV3AswBasicTiling(context, cfg) {};

    ~GemmSyrkBaseTiling() override = default;

protected:
    bool IsCapable() override;

    ge::graphStatus DoOpTiling() override;

    uint64_t GetTilingKey() const override;
};
} // namespace gemm_syrk
} // namespace optiling

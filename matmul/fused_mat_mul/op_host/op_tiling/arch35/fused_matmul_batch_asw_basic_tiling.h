/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file fused_matmul_asw_basic_tiling.h
 * \brief
 */
#pragma once

#include "batch_mat_mul_v3/op_host/op_tiling/arch35/batch_matmul_v3_asw_basic_tiling.h"

namespace optiling {
namespace fused_matmul {
using batch_matmul_v3_advanced::BatchMatMulV3AswBasicTiling;

class FusedMatMulBatchAswBasicApiTiling : public BatchMatMulV3AswBasicTiling {
public:
    FusedMatMulBatchAswBasicApiTiling(gert::TilingContext* context, MatMulTilingCfg& cfg)
        : BatchMatMulV3AswBasicTiling(context, cfg){};

    ~FusedMatMulBatchAswBasicApiTiling() override = default;

protected:
    uint64_t GetTilingKey() const override;
};
} // namespace fused_matmul
} // namespace optiling
/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fused_matmul_tiling.cpp
 * \brief A2/A3 relu tiling. 2D reuses MatMulV3 tiling. Batch reuses BatchMatMulV3 tiling.
 */

#include "matmul/batch_mat_mul_v3/op_host/op_tiling/batch_mat_mul_v3_base_tiling.h"
#include "matmul/fused_mat_mul/op_kernel/fused_matmul_tiling_key.h"
#include "op_host/tiling_templates_registry.h"

namespace optiling {
namespace fused_matmul {
namespace {
constexpr uint64_t kMmNoNd2Nz = 1UL;
constexpr size_t kMatmulRank = 2UL;
} // namespace

class FusedMatMulReluTiling : public matmul_v3::MatmulV3BaseTiling {
public:
    explicit FusedMatMulReluTiling(gert::TilingContext* context) : MatmulV3BaseTiling(context) {}

    bool IsCapable() override
    {
        const auto* shape = context_->GetInputShape(0);
        if (shape == nullptr) {
            return false;
        }
        return shape->GetOriginShape().GetDimNum() == kMatmulRank;
    }

    uint64_t GetTilingKey() const override
    {
        // MatmulV3BaseTiling encodes GET_TPL_TILING_KEY with MatMulV3's 6-field schema.
        // FusedMatMul kernels are compiled with 12 fields, so re-pack here.
        auto* self = const_cast<FusedMatMulReluTiling*>(this);
        return GET_TPL_TILING_KEY(
            static_cast<uint64_t>(tilingEnable_.tilingEnableFullLoad),
            static_cast<uint64_t>(tilingEnable_.tilingEnableSplitCore),
            static_cast<uint64_t>(tilingEnable_.tilingEnableFixOpti), static_cast<uint64_t>(self->GetMixNd2nzType()),
            static_cast<uint64_t>(tilingEnable_.tilingEnableSpecialOpti), static_cast<uint64_t>(FMM_FP32_ADDMM_DISABLE),
            static_cast<uint64_t>(FMM_A2_IS_MM), static_cast<uint64_t>(FMM_BMM_L1_FULL_FALSE),
            static_cast<uint64_t>(FMM_BMM_MULTI_FALSE), static_cast<uint64_t>(FMM_BMM_LOAD_BASE),
            static_cast<uint64_t>(FMM_BMM_OUT_FALSE), static_cast<uint64_t>(FMM_BMM_MIX_FALSE));
    }
};

class FusedBatchMatMulReluTiling : public batch_mat_mul_v3::BatchMatmulV3BaseTiling {
public:
    explicit FusedBatchMatMulReluTiling(gert::TilingContext* context) : BatchMatmulV3BaseTiling(context) {}

    bool IsCapable() override
    {
        const auto* shape = context_->GetInputShape(0);
        if (shape == nullptr) {
            return false;
        }
        return shape->GetOriginShape().GetDimNum() > kMatmulRank;
    }

    uint64_t GetTilingKey() const override
    {
        // Keep the MatMulV3 fields at the no-op pattern so they do not alias a 2D key.
        return GET_TPL_TILING_KEY(0UL, 0UL, 0UL, kMmNoNd2Nz, 0UL, 0UL, static_cast<uint64_t>(FMM_A2_IS_BATCH),
                                  static_cast<uint64_t>(tilingEnable_.tilingEnableMultiBatchL1FullLoad),
                                  static_cast<uint64_t>(tilingEnable_.tilingEnableMultiBatch),
                                  static_cast<uint64_t>(tilingEnable_.tilingEnableLoadMode),
                                  static_cast<uint64_t>(tilingEnable_.tilingEnableMultiBatchOut),
                                  static_cast<uint64_t>(tilingEnable_.tilingEnableMixNd2Nz));
    }
};
} // namespace fused_matmul
} // namespace optiling

namespace optiling {
using FusedBatchMatMulReluTiling = fused_matmul::FusedBatchMatMulReluTiling;
using FusedMatMulReluTiling = fused_matmul::FusedMatMulReluTiling;
REGISTER_TILING_TEMPLATE("FusedMatMul", FusedBatchMatMulReluTiling, 0);
REGISTER_TILING_TEMPLATE("FusedMatMul", FusedMatMulReluTiling, 1);
} // namespace optiling

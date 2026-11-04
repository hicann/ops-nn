/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fused_matmul_k_equal_zero_tiling.cpp
 * \brief
 */
#include "fused_matmul_k_equal_zero_tiling.h"

#include "fused_matmul_builtin_tiling_strategy.h"
#include "fused_matmul_common.h"
#include "matmul/mat_mul_v3/op_host/op_tiling/arch35/matmul_tiling_registry.h"
#include "matmul/mat_mul_v3/op_kernel/arch35/mat_mul_tiling_data.h"

namespace optiling {
namespace fused_matmul {
using namespace strategy;
MM_REGISTER_TILING_TEMPLATE(FusedMatMul, FusedMatMulKEqZeroTiling, DAV_3510, INPUT_K_EQUAL_ZERO_INHERITED_FROM_BMMV3);
MM_REGISTER_TILING_TEMPLATE(FusedMatMul, FusedMatMulKEqZeroTiling, DAV_9201, INPUT_K_EQUAL_ZERO_INHERITED_FROM_BMMV3);
MM_REGISTER_TILING_TEMPLATE(FusedMatMul, FusedMatMulKEqZeroTiling, DAV_9202, INPUT_K_EQUAL_ZERO_INHERITED_FROM_BMMV3);

bool FusedMatMulKEqZeroTiling::IsCapable()
{
    auto attrs = context_->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    std::string opType = attrs->GetAttrPointer<char>(ATTR_OP_TYPE_IDX);
    if ((compileInfo_.npuArch == NpuArch::DAV_9201 || compileInfo_.npuArch == NpuArch::DAV_9202) && opType != "relu") {
        return false;
    }
    if (opType != "relu" && opType != "gelu_erf" && opType != "gelu_tanh" && opType != "add" && opType != "mul" &&
        !opType.empty()) {
        return false;
    }
    if ((opType == "gelu_erf" || opType == "gelu_tanh") && batchInfo_ != nullptr && batchInfo_->batchC > 1UL) {
        return false;
    }
    return BatchMatMulV3KEqZeroTiling::IsCapable();
}

uint64_t FusedMatMulKEqZeroTiling::GetTilingKey() const
{
    auto attrs = context_->GetAttrs();
    if (attrs != nullptr) {
        std::string opType = attrs->GetAttrPointer<char>(ATTR_OP_TYPE_IDX);
        if (opType == "add" || opType == "mul") {
            auto it = FUSED_OP_TYPE_MAP.find(opType);
            if (it != FUSED_OP_TYPE_MAP.end()) {
                if (IsFusedMatMulBmmShape(context_)) {
                    // BMM K=0 add/mul stays on the legacy basic-level kernels.
                    return GET_TPL_TILING_KEY(
                        static_cast<uint64_t>(MAT_MUL_BASIC_LEVEL), static_cast<uint64_t>(F_NO_TRANS),
                        MAT_MUL_FOR_FUSED_BATCH, static_cast<uint64_t>(MAT_MUL_K_EQUAL_ZERO),
                        static_cast<uint64_t>(MAT_MUL_NO_FULL_LOAD), static_cast<uint64_t>(MAT_MUL_ON_THE_FLY),
                        static_cast<uint64_t>(it->second), static_cast<uint64_t>(F_INNER_PRECISE_HIGH_PERFORMANCE));
                }
                // MM K=0 add/mul runs on the Tensor-API path (AIV-only copy/zero).
                // 960 无需平台判断: IsCapable() 的 920x 非 relu 排除已保证到达此处仅在 DAV_3510.
                return GET_TPL_TILING_KEY(
                    static_cast<uint64_t>(MAT_MUL_TENSOR_LEVEL), static_cast<uint64_t>(F_NO_TRANS), MAT_MUL_FOR_BATCH,
                    static_cast<uint64_t>(MAT_MUL_K_EQUAL_ZERO), static_cast<uint64_t>(MAT_MUL_NO_FULL_LOAD),
                    static_cast<uint64_t>(MAT_MUL_ON_THE_FLY), static_cast<uint64_t>(it->second),
                    static_cast<uint64_t>(F_INNER_PRECISE_HIGH_PERFORMANCE));
            }
        }
    }
    MatMulV3TilingKey tmp = MatMulV3TilingKey();
    MatMulV3TilingKey& tilingKey = tilingKeyObj == nullptr ? tmp : *tilingKeyObj;
    return tilingKey.SetTrans(false, false)
        .SetApiLevel(MatMulV3ApiLevel::HIGH_LEVEL)
        .SetBatchModel(MatMulV3BatchModel::BATCH_MODEL)
        .SetModel(MatMulV3Model::K_EQUAL_ZERO)
        .SetFullLoad(MatMulV3FullLoad::NONE_FULL_LOAD)
        .SetL0C2Out(MatMulV3L0C2Out::ON_THE_FLY)
        .GetTilingKey();
}

ge::graphStatus FusedMatMulKEqZeroTiling::GetTilingData(TilingResult& tiling) const
{
    auto attrs = context_->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    const std::string opType = attrs->GetAttrPointer<char>(ATTR_OP_TYPE_IDX);
    if (IsFusedMatMulBmmShape(context_) && (opType == "add" || opType == "mul")) {
        return GetTilingDataImpl<BatchMatMulV3BasicTilingData>(tiling);
    }
    return BatchMatMulV3KEqZeroTiling::GetTilingData(tiling);
}
} // namespace fused_matmul
} // namespace optiling

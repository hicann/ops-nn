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

namespace optiling {
namespace fused_matmul {
using namespace strategy;
MM_REGISTER_TILING_TEMPLATE(FusedMatMul, FusedMatMulKEqZeroTiling, DAV_3510, INPUT_K_EQUAL_ZERO_INHERITED_FROM_BMMV3);

bool FusedMatMulKEqZeroTiling::IsCapable()
{
    auto attrs = context_->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    std::string opType = attrs->GetAttrPointer<char>(ATTR_OP_TYPE_IDX);
    if (opType != "relu" && opType != "gelu_erf" && opType != "gelu_tanh" &&
        opType != "add" && opType != "mul" && !opType.empty()) {
        return false;
    }
    if ((opType == "add" || opType == "mul" || opType == "gelu_erf" || opType == "gelu_tanh") &&
        batchInfo_ != nullptr && batchInfo_->batchC > 1UL) {
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
                return GET_TPL_TILING_KEY(
                    static_cast<uint64_t>(MAT_MUL_BASIC_LEVEL),
                    static_cast<uint64_t>(F_NO_TRANS),
                    static_cast<uint64_t>(MAT_MUL_FOR_BATCH),
                    static_cast<uint64_t>(MAT_MUL_K_EQUAL_ZERO),
                    static_cast<uint64_t>(MAT_MUL_NO_FULL_LOAD),
                    static_cast<uint64_t>(MAT_MUL_ON_THE_FLY),
                    static_cast<uint64_t>(it->second));
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
} // namespace fused_matmul
} // namespace optiling

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
 * \file fused_matmul_asw_basic_tiling.cpp
 * \brief
 */
#include "fused_matmul_basic_streamk_tiling.h"
#include "fused_matmul_builtin_tiling_strategy.h"
#include "fused_matmul_common.h"
#include "matmul/mat_mul_v3/op_host/op_tiling/arch35/matmul_tiling_registry.h"
#include "matmul/common/op_host/math_util.h"

namespace optiling {
namespace fused_matmul {
using namespace strategy;
MM_REGISTER_TILING_TEMPLATE(FusedMatMul, FusedMatMulStreamKTiling, DAV_3510, BASIC_STREAM_K_INHERITED_FROM_MMV3);
MM_REGISTER_TILING_TEMPLATE(FusedMatMul, FusedMatMulBatchStreamKTiling, DAV_3510, BATCH_STREAM_K_INHERITED_FROM_BMMV3);

bool FusedMatMulStreamKTiling::IsCapable()
{
    if (args_.batchInfo->batchC > 1) {
        return false;
    }
    auto attrs = context_->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    std::string opType = attrs->GetAttrPointer<char>(ATTR_OP_TYPE_IDX);
    // Only relu and "" and 16cast32 support streamK unconditionally
    if (FusedOpTypeSupportStreamK.find(opType) == FusedOpTypeSupportStreamK.end()) {
        return false;
    }

    // mul and add only supports pure SK scenario (no DP tiles with fixpipe)
    if (opType == "add" || opType == "mul") {
        constexpr uint64_t BASIC_BLOCK_SIZE_256 = 256UL;
        constexpr uint64_t NUM_TWO = 2UL;
        uint64_t mCnt = (args_.mValue + BASIC_BLOCK_SIZE_256 - 1) / BASIC_BLOCK_SIZE_256;
        uint64_t nCnt = (args_.nValue + BASIC_BLOCK_SIZE_256 - 1) / BASIC_BLOCK_SIZE_256;
        uint64_t totalMNCnt = mCnt * nCnt;
        // Pure SK requires totalMNCnt <= aicNum / 2
        // If totalMNCnt > aicNum / 2, will generate DP tiles that cannot do mul and add fusion
        if (totalMNCnt > compileInfo_.aicNum / NUM_TWO) {
            return false;
        }
    }

    return MatMulV3BasicStreamKTiling::IsCapable();
}

bool FusedMatMulBatchStreamKTiling::IsCapable()
{
    if (!IsFusedMatMulBmmShape(context_)) {
        return false;
    }
    auto attrs = context_->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    std::string opType = attrs->GetAttrPointer<char>(ATTR_OP_TYPE_IDX);
    if (opType != "relu" && !opType.empty()) {
        return false;
    }
    return BatchMatMulV3BasicStreamKTiling::IsCapable();
}

uint64_t FusedMatMulBatchStreamKTiling::GetTilingKey() const
{
    MatMulV3TilingKey tmp = MatMulV3TilingKey();
    MatMulV3TilingKey& tilingKey = tilingKeyObj == nullptr ? tmp : *tilingKeyObj;
    return tilingKey.SetTrans(args_.isATrans, args_.isBTrans)
        .SetBatchModel(MatMulV3BatchModel::FUSED_BATCH_MODEL)
        .SetModel(MatMulV3Model::STREAM_K)
        .SetFullLoad(MatMulV3FullLoad::NONE_FULL_LOAD)
        .SetL0C2Out(GetL0C2OutFlag())
        .SetApiLevel(MatMulV3ApiLevel::BASIC_LEVEL)
        .GetTilingKey();
}

} // namespace fused_matmul
} // namespace optiling

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
 * \file fused_matmul_batch_asw_basic_tiling.cpp
 * \brief
 */
#include "fused_matmul_batch_asw_basic_tiling.h"

#include <memory>
#include <new>

#include "fused_matmul_builtin_tiling.h"
#include "fused_matmul_builtin_tiling_strategy.h"
#include "fused_matmul_common.h"
#include "matmul/mat_mul_v3/op_host/op_tiling/arch35/matmul_tiling_registry.h"
#include "matmul/common/op_host/math_util.h"

namespace optiling {
namespace fused_matmul {
using namespace strategy;
MM_REGISTER_TILING_TEMPLATE(FusedMatMul, FusedMatMulBatchAswBasicApiTiling, DAV_3510, ASWT_BASIC_INHERITED_FROM_BMMV3);
MM_REGISTER_TILING_TEMPLATE(FusedMatMul, FusedMatMulBatchAswBasicApiTiling, DAV_RESV, ASWT_BASIC_INHERITED_FROM_BMMV3);

bool FusedMatMulBatchAswBasicApiTiling::IsCapable()
{
    if (!IsFusedMatMulBmmShape(context_)) {
        return false;
    }
    if (compileInfo_.npuArch == NpuArch::DAV_RESV) {
        return BatchMatMulV3AswBasicTiling::IsCapable();
    }
    auto attrs = context_->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    std::string opType = attrs->GetAttrPointer<char>(ATTR_OP_TYPE_IDX);
    if (opType == "scale_add") {
        return true;
    }
    if (opType != "relu" && opType != "add" && opType != "mul" && !opType.empty()) {
        return false;
    }
    return BatchMatMulV3AswBasicTiling::IsCapable();
}

uint64_t FusedMatMulBatchAswBasicApiTiling::GetTilingKey() const
{
    const auto* attrs = context_->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    const std::string opType = attrs->GetAttrPointer<char>(ATTR_OP_TYPE_IDX);
    if (opType == "scale_add") {
        return GET_TPL_TILING_KEY(MAT_MUL_BASIC_LEVEL, F_NO_TRANS, MAT_MUL_FOR_FUSED_BATCH, MAT_MUL_BASIC,
                                  MAT_MUL_NO_FULL_LOAD, MAT_MUL_1V2_ND_ALIG_FIXPIPE, F_OPTYPE_SCALE_ADD,
                                  F_INNER_PRECISE_HIGH_PERFORMANCE);
    }
    MatMulV3TilingKey tmp = MatMulV3TilingKey();
    MatMulV3TilingKey& tilingKey = tilingKeyObj == nullptr ? tmp : *tilingKeyObj;
    return tilingKey.SetTrans(args_.isATrans, args_.isBTrans)
        .SetModel(MatMulV3Model::BASIC)
        .SetBatchModel(MatMulV3BatchModel::FUSED_BATCH_MODEL)
        .SetFullLoad(MatMulV3FullLoad::NONE_FULL_LOAD)
        .SetL0C2Out(MatMulV3L0C2Out::ON_THE_FLY)
        .SetApiLevel(MatMulV3ApiLevel::BASIC_LEVEL)
        .GetTilingKey();
}

ge::graphStatus FusedMatMulBatchAswBasicApiTiling::GetTilingDataProcess(FusedMatMulTilingData& tilingData) const
{
    const ge::graphStatus ret = BatchMatMulV3AswBasicTiling::GetTilingDataProcess(tilingData.matMulTilingData);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    const auto* attrs = context_->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    const std::string opType = attrs->GetAttrPointer<char>(ATTR_OP_TYPE_IDX);
    if (opType != "scale_add") {
        return ge::GRAPH_SUCCESS;
    }

    const auto* alpha = attrs->GetAttrPointer<float>(ATTR_ALPHA_IDX);
    const auto* beta = attrs->GetAttrPointer<float>(ATTR_BETA_IDX);
    tilingData.alpha = alpha == nullptr ? 1.0F : *alpha;
    tilingData.beta = beta == nullptr ? 1.0F : *beta;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedMatMulBatchAswBasicApiTiling::GetTilingData(TilingResult& tiling) const
{
    std::shared_ptr<FusedMatMulTilingData> tilingData;
    try {
        tilingData = std::make_shared<FusedMatMulTilingData>();
    } catch (const std::bad_alloc&) {
        OP_LOGE(context_->GetNodeName(), "Failed to allocate memory for tilingData");
        return ge::GRAPH_FAILED;
    }
    const ge::graphStatus ret = GetTilingDataProcess(*tilingData);
    tiling.tilingData = tilingData;
    tiling.tilingDataSize = sizeof(FusedMatMulTilingData);
    return ret;
}

} // namespace fused_matmul
} // namespace optiling

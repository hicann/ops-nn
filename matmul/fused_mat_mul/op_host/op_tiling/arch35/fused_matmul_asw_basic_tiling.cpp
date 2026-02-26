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
#include "fused_matmul_asw_basic_tiling.h"
#include "fused_matmul_builtin_tiling_strategy.h"
#include "fused_matmul_common.h"
#include "matmul/mat_mul_v3/op_host/op_tiling/arch35/matmul_tiling_registry.h"
#include "matmul/common/op_host/math_util.h"

namespace optiling {
namespace fused_matmul {
using matmul_v3_advanced::strategy::BASIC_ASWT;
MM_REGISTER_TILING_TEMPLATE(FusedMatMul, FusedMatMulAswBasicApiTiling, DAV_3510, BASIC_ASWT);

bool FusedMatMulAswBasicApiTiling::IsCapable()
{
    if (args_.batchInfo->batchC > 1) {
        OP_LOGD(args_.opName, "bmm only support IterBatch shape");
        return false;
    }
    if (args_.bFormat != ge::FORMAT_ND || args_.aFormat != ge::FORMAT_ND) {
        OP_LOGD(args_.opName, "ND is the only supported format for basic api");
        return false;
    }

    OP_LOGI(args_.opName, "FusedMatMul tiling enable state basic api");
    return true;
}

ge::graphStatus FusedMatMulAswBasicApiTiling::DoOpTiling() {
    auto attrs = context_->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    std::string opType = attrs->GetAttrPointer<char>(ATTR_OP_TYPE_IDX);
    if (opType == "add" || opType == "mul") {
        OP_TILING_CHECK(
            (MatMulV3AswTiling::DoOpTiling() != ge::GRAPH_SUCCESS),
            CUBE_INNER_ERR_REPORT(args_.opName, "Do MatMul AswTiling failed in FusedMatMul."), return ge::GRAPH_FAILED);
        uint64_t remainSizeForAL1BL1 =
            args_.hasBias ? (compileInfo_.l1Size - BIAS_TABLE_NUM * DATA_SIZE_FP32) : compileInfo_.l1Size;
        runInfo_.stepKa =
            remainSizeForAL1BL1 / NUM_TWO / ((runInfo_.baseM + runInfo_.baseN) * runInfo_.baseK) / args_.aDtypeSize;
        runInfo_.stepKb = runInfo_.stepKa; // has bias, adjust stepK to suitable value
        runInfo_.depthA1 = runInfo_.stepKa * DB_SIZE;
        runInfo_.depthB1 = runInfo_.stepKb * DB_SIZE;
        return ge::GRAPH_SUCCESS;
    }
    ge::graphStatus status = MatMulV3BasicAswtTiling::DoOpTiling();
    if (l0C2Out_ == MatMulV3L0C2Out::ND_FIXPIPE_1_1) {
        l0C2Out_ = MatMulV3L0C2Out::ON_THE_FLY;
    }
    return status;
}

uint64_t FusedMatMulAswBasicApiTiling::GetTilingKey() const
{
    MatMulV3TilingKey tmp = MatMulV3TilingKey();
    MatMulV3TilingKey& tilingKey = tilingKeyObj == nullptr ? tmp : *tilingKeyObj;
    auto attrs = context_->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    std::string opType = attrs->GetAttrPointer<char>(ATTR_OP_TYPE_IDX);
    if (opType == "add" || opType == "mul") {
        return tilingKey.SetTrans(args_.isATrans, args_.isBTrans)
            .SetL0C2Out(MatMulV3L0C2Out::ON_THE_FLY)
            .SetFullLoad(MatMulV3FullLoad::NONE_FULL_LOAD)
            .SetApiLevel(MatMulV3ApiLevel::BASIC_LEVEL)
            .GetTilingKey();
    }
    return tilingKey.SetTrans(args_.isATrans, args_.isBTrans)
        .SetFullLoad(fullLoad_)
        .SetModel(MatMulV3Model::BASIC)
        .SetL0C2Out(l0C2Out_)
        .SetApiLevel(MatMulV3ApiLevel::BASIC_LEVEL)
        .GetTilingKey();
}

} // namespace fused_matmul
} // namespace optiling
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
 * \file fused_matmul_mergebatch_basic_tiling.cpp
 * \brief
 */

#include "fused_matmul_mergebatch_basic_tiling.h"
#include "fused_matmul_builtin_tiling.h"
#include "fused_matmul_builtin_tiling_strategy.h"
#include "fused_matmul_common.h"
#include "matmul/mat_mul_v3/op_host/op_tiling/arch35/matmul_tiling_registry.h"
#include "matmul/mat_mul_v3/op_host/op_tiling/arch35/matmul_v3_tiling_helper.h"

namespace optiling {
namespace fused_matmul {
using namespace matmul_v3_advanced;
using namespace strategy;

MM_REGISTER_TILING_TEMPLATE(FusedMatMul, FusedMatMulMergeBatchBasicApiTiling, DAV_3510,
                            MERGE_BATCH_BASICAPI_INHERITED_FROM_BMMV3);

bool FusedMatMulMergeBatchBasicApiTiling::IsCapable()
{
    auto attrs = context_->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    std::string opType = attrs->GetAttrPointer<char>(ATTR_OP_TYPE_IDX);
    if ((opType == "add" || opType == "mul") && context_->InputIsView(1) &&
        MatMulV3TilingHelper::IsTransposeNonContiguous(context_, 1)) {
        OP_LOGD(args_.opName, "MergeBatch add/mul does not support a non-contiguous x2 view");
        return false;
    }
    if (opType != "relu" && opType != "add" && opType != "mul" && !opType.empty()) {
        OP_LOGD(args_.opName, "MergeBatch model only supports add, mul, relu or empty op type in FusedMatMul");
        return false;
    }
    bool status = BatchMatMulV3MergeBatchBasicApiTiling::IsCapable();
    if (!status) {
        OP_LOGD(args_.opName, "MergeBatch model is not supported for this shape");
        return false;
    }
    if (opType == "add" || opType == "mul") {
        auto innerPrecise = attrs->GetAttrPointer<int64_t>(ATTR_INNER_PRECISE_IDX);
        OPS_CHECK_NULL_WITH_CONTEXT(context_, innerPrecise);
        const bool useFloatMmadOut = *innerPrecise == INNER_PRECISE_HIGH_PRECISION;
        const uint64_t mmOutDtypeSize = useFloatMmadOut ? sizeof(float) :
                                                          static_cast<uint64_t>(ge::GetSizeByDataType(args_.cType));
        const uint64_t x3DtypeSize = static_cast<uint64_t>(ge::GetSizeByDataType(args_.x3Type));
        const uint64_t nAlign = ops::CeilAlign(args_.nValue, BASIC_BLOCK_SIZE_16);
        const bool needCastBuffer = useFloatMmadOut && args_.x3Type != ge::DT_FLOAT;
        const uint64_t resultUbSize = args_.mValue * nAlign * mmOutDtypeSize;
        const uint64_t minStageUbSize = nAlign * NUM_TWO * (x3DtypeSize + (needCastBuffer ? mmOutDtypeSize : 0UL));
        if (resultUbSize + minStageUbSize > compileInfo_.ubSize) {
            OP_LOGD(args_.opName, "FusedMatMul mergebatch fusion requires %lu bytes UB, but only %lu are available",
                    resultUbSize + minStageUbSize, compileInfo_.ubSize);
            return false;
        }
    }
    OP_LOGI(args_.opName, "FusedMatMul tiling enable mergebatch basic api");
    return true;
}

uint64_t FusedMatMulMergeBatchBasicApiTiling::GetTilingKey() const
{
    MatMulV3TilingKey tmp = MatMulV3TilingKey();
    MatMulV3TilingKey& tilingKey = tilingKeyObj == nullptr ? tmp : *tilingKeyObj;
    auto attrs = context_->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    std::string opType = attrs->GetAttrPointer<char>(ATTR_OP_TYPE_IDX);
    bool transA = args_.isATrans && args_.mValue > 1;
    return tilingKey.SetTrans(transA, args_.isBTrans)
        .SetBatchModel(MatMulV3BatchModel::MERGE_BATCH_MODEL)
        .SetModel(MatMulV3Model::BASIC)
        .SetFullLoad(MatMulV3FullLoad::NONE_FULL_LOAD)
        .SetL0C2Out((opType == "add" || opType == "mul") ? MatMulV3L0C2Out::ND_FIXPIPE_1_2 :
                                                           MatMulV3L0C2Out::ON_THE_FLY)
        .SetApiLevel(MatMulV3ApiLevel::BASIC_LEVEL)
        .GetTilingKey();
}

} // namespace fused_matmul
} // namespace optiling

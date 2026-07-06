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
#include "fused_matmul_builtin_tiling_strategy.h"
#include "fused_matmul_common.h"
#include "matmul/mat_mul_v3/op_host/op_tiling/arch35/matmul_tiling_registry.h"

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
    if (opType != "relu" && !opType.empty()) {
        OP_LOGD(args_.opName, "MergeBatch model only supports relu or empty op type in FusedMatMul");
        return false;
    }
    bool status = BatchMatMulV3MergeBatchBasicApiTiling::IsCapable();
    if (!status) {
        OP_LOGD(args_.opName, "MergeBatch model is not supported for this shape");
        return false;
    }
    OP_LOGI(args_.opName, "FusedMatMul tiling enable mergebatch basic api");
    return true;
}

uint64_t FusedMatMulMergeBatchBasicApiTiling::GetTilingKey() const
{
    MatMulV3TilingKey tmp = MatMulV3TilingKey();
    MatMulV3TilingKey& tilingKey = tilingKeyObj == nullptr ? tmp : *tilingKeyObj;
    bool transA = args_.isATrans && args_.mValue > 1;
    return tilingKey.SetTrans(transA, args_.isBTrans)
        .SetBatchModel(MatMulV3BatchModel::MERGE_BATCH_MODEL)
        .SetModel(MatMulV3Model::BASIC)
        .SetFullLoad(MatMulV3FullLoad::NONE_FULL_LOAD)
        .SetL0C2Out(MatMulV3L0C2Out::ON_THE_FLY)
        .SetApiLevel(MatMulV3ApiLevel::BASIC_LEVEL)
        .GetTilingKey();
}

} // namespace fused_matmul
} // namespace optiling

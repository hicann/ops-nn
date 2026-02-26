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
 * \file fused_matmul_iterbatch_basic_tiling.cpp
 * \brief
 */
#include "fused_matmul_iterbatch_basic_tiling.h"
#include "fused_matmul_builtin_tiling_strategy.h"
#include "matmul/mat_mul_v3/op_host/op_tiling/arch35/matmul_tiling_registry.h"
#include "matmul/common/op_host/math_util.h"

namespace optiling {
namespace fused_matmul {
using strategy::ITER_BATCH_BASICAPI;
MM_REGISTER_TILING_TEMPLATE(FusedMatMul, FusedMatMulIterBatchApiTiling, DAV_3510, ITER_BATCH_BASICAPI);

bool FusedMatMulIterBatchApiTiling::IsCapable()
{
    bool status = BatchMatMulV3IterBatchBasicApiTiling::IsCapable();
    if (!status) {
        OP_LOGD(args_.opName, "IterBatch model is not supported for this shape");
        return false;
    }
    OP_LOGI(args_.opName, "FusedMatMul tiling enable iterbatch basic api");
    return true;
}

uint64_t FusedMatMulIterBatchApiTiling::GetTilingKey() const
{
    MatMulV3TilingKey tmp = MatMulV3TilingKey();
    MatMulV3TilingKey& tilingKey = tilingKeyObj == nullptr ? tmp : *tilingKeyObj;
    // has x3 must be add or mul
    if (args_.hasX3Input) {
        return tilingKey.SetTrans(args_.isATrans, args_.isBTrans)
            .SetBatchModel(MatMulV3BatchModel::SINGLE_BIAS_MODEL)
            .SetL0C2Out(MatMulV3L0C2Out::ND_FIXPIPE_1_2)
            .SetApiLevel(MatMulV3ApiLevel::BASIC_LEVEL)
            .GetTilingKey();
    }
    return tilingKey.SetTrans(args_.isATrans, args_.isBTrans)
        .SetBatchModel(MatMulV3BatchModel::SINGLE_BIAS_MODEL)
        .SetL0C2Out(l0C2Out_)
        .SetApiLevel(MatMulV3ApiLevel::BASIC_LEVEL)
        .GetTilingKey();
}

ge::graphStatus FusedMatMulIterBatchApiTiling::DoOpTiling() {
    BatchMatMulV3IterBatchBasicApiTiling::DoOpTiling();
    // calculate iterBatchX3 for rest ub
    uint64_t restUB = compileInfo_.ubSize - runInfo_.iterBatchL0 * runInfo_.baseM * runInfo_.baseN;
    uint64_t singleBatchSize = args_.mValue * runInfo_.baseN;
    uint64_t iterBatchX3 = std::min(runInfo_.iterBatchL0, restUB / NUM_TWO / singleBatchSize);
    runInfo_.needNdDma = (iterBatchX3 > 1) && (args_.batchX3 == 1);
    return ge::GRAPH_SUCCESS;
}

} // namespace fused_matmul
} // namespace optiling
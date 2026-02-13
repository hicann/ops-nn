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
 * \file batch_matmul_v3_k_equal_zero_tiling.cc
 * \brief
 */

#include "batch_matmul_v3_k_equal_zero_tiling.h"
#include "batch_matmul_v3_tiling_strategy.h"
#include "matmul/mat_mul_v3/op_host/op_tiling/arch35/matmul_tiling_registry.h"
#include "batch_matmul_v3_tiling_key.h"

namespace optiling {
namespace batch_matmul_v3_advanced {
using namespace strategy;

MM_REGISTER_TILING_TEMPLATE(BatchMatMulV3, BatchMatMulV3KEqZeroTiling, DAV_3510, BATCH_MATMUL_INPUT_K_EQUAL_ZERO);

bool BatchMatMulV3KEqZeroTiling::IsCapable()
{
    if (args_.hasBias) {
        return false;
    }

    if (args_.kValue != 0UL) {
        return false;
    }
    return true;
}

ge::graphStatus BatchMatMulV3KEqZeroTiling::DoOpTiling()
{
    runInfo_.totalDataAmount = args_.mValue * args_.nValue * batchInfo_->batchC;
    runInfo_.usedCoreNum = compileInfo_.aivNum;
    return ge::GRAPH_SUCCESS;
}

uint64_t BatchMatMulV3KEqZeroTiling::GetNumBlocks() const
{
    return compileInfo_.aivNum;
}

uint64_t BatchMatMulV3KEqZeroTiling::GetTilingKey() const
{
    return BatchMatMulV3TilingKey().SetTrans(false, false)
        .SetApiLevel(MatMulV3ApiLevel::HIGH_LEVEL)
        .SetBatchModel(MatMulV3BatchModel::BATCH_MODEL)
        .SetModel(MatMulV3Model::K_EQUAL_ZERO)
        .SetFullLoad(MatMulV3FullLoad::NONE_FULL_LOAD)
        .SetL0C2Out(MatMulV3L0C2Out::ON_THE_FLY)
        .GetTilingKey();
}

ge::graphStatus BatchMatMulV3KEqZeroTiling::GetTilingData(TilingResult& tiling) const
{
    return GetTilingDataImpl<MatMulV3KEqZeroBasicTilingData>(tiling);
}
}
}
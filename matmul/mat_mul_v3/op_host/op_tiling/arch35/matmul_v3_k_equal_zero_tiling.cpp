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
 * \file matmul_v3_k_equal_zero_tiling.cc
 * \brief
 */

#include "matmul_v3_k_equal_zero_tiling.h"
#include "matmul_tiling_registry.h"
#include "matmul_v3_tiling_strategy.h"
#include "matmul/common/op_host/math_util.h"

namespace optiling {
namespace matmul_v3_advanced {
using namespace strategy;

MM_REGISTER_TILING_TEMPLATE(MatMulV3, MatMulV3KEqZeroTiling, ASCEND910_95, MATMUL_INPUT_K_EQUAL_ZERO);

bool MatMulV3KEqZeroTiling::IsCapable()
{
    if (args_.hasBias) {
        return false;
    }

    if (args_.kValue != 0UL) {
        return false;
    }
    return true;
}

ge::graphStatus MatMulV3KEqZeroTiling::DoOpTiling()
{
    runInfo_.totalDataAmount = args_.mValue * args_.nValue;
    runInfo_.usedCoreNum = compileInfo_.aivNum;
    return ge::GRAPH_SUCCESS;
}

uint64_t MatMulV3KEqZeroTiling::GetNumBlocks() const
{
    return compileInfo_.aivNum;
}

uint64_t MatMulV3KEqZeroTiling::GetTilingKey() const
{
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

ge::graphStatus MatMulV3KEqZeroTiling::GetTilingData(TilingResult& tiling) const
{
    return GetTilingDataImpl<MatMulV3KEqZeroBasicTilingData>(tiling);
}
}
}
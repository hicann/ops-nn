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
 * \file batch_matmul_v3_asw_basic_tiling.cc
 * \brief
 */

#include "batch_matmul_v3_asw_basic_tiling.h"
#include "batch_matmul_v3_tiling_strategy.h"
#include "batch_matmul_v3_common_advanced.h"
#include "matmul/mat_mul_v3/op_host/op_tiling/arch35/matmul_tiling_registry.h"
#include "batch_matmul_v3_tiling_key.h"

namespace optiling {
namespace batch_matmul_v3_advanced {
using namespace strategy;
MM_REGISTER_TILING_TEMPLATE(BatchMatMulV3, BatchMatMulV3AswBasicTiling, ASCEND910_95, ASW_BASIC);

bool BatchMatMulV3AswBasicTiling::IsCapable()
{
    bool isEqualBatch = batchInfo_->batchA0 == batchInfo_->batchB0 && batchInfo_->batchA1 == batchInfo_->batchB1 &&
                           batchInfo_->batchA2 == batchInfo_->batchB2 && batchInfo_->batchA3 == batchInfo_->batchB3;
    if (!isEqualBatch) {
        return false;
    }
    return true;
}

ge::graphStatus BatchMatMulV3AswBasicTiling::DoOpTiling()
{
    MatMulV3TilingHelper::ResetBase(compileInfo_, args_, runInfo_);
    MatMulV3TilingHelper::CalL1Tiling(compileInfo_, args_, runInfo_);

    // l1开2db后依然只使用了一半的空间，则开启4 db。该字段仅在基础api场景生效
    uint64_t abL1TensorSize = runInfo_.baseK * runInfo_.stepKa * (runInfo_.baseM + runInfo_.baseN) * args_.aDtypeSize;
    if (args_.hasBias) {
        abL1TensorSize +=  runInfo_.baseN * sizeof(args_.biasType);
    }
    if (abL1TensorSize * NUM_FOUR <= compileInfo_.l1Size) {
        runInfo_.l1BufferNum = NUM_FOUR;
    } else {
        runInfo_.l1BufferNum = NUM_TWO;
    }

    return ge::GRAPH_SUCCESS;
}

uint64_t BatchMatMulV3AswBasicTiling::GetTilingKey() const
{
    return BatchMatMulV3TilingKey()
        .SetTrans(args_.isATrans, args_.isBTrans)
        .SetModel(MatMulV3Model::BASIC)
        .SetApiLevel(MatMulV3ApiLevel::BASIC_LEVEL)
        .GetTilingKey();
}

ge::graphStatus BatchMatMulV3AswBasicTiling::GetTilingData(TilingResult& tiling) const
{
    return GetTilingDataImpl<BatchMatMulV3BasicTilingData>(tiling);
}

uint64_t BatchMatMulV3AswBasicTiling::GetNumBlocks() const
{
    return compileInfo_.aicNum;
}
} // namespace batch_matmul_v3_advanced
} // namespace optiling
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
 * \file batch_matmul_v3_asw_bl1_full_load_basic_tiling.cc
 * \brief
 */

#include "batch_matmul_v3_asw_bl1_full_load_basic_tiling.h"
#include "batch_matmul_v3_tiling_strategy.h"
#include "batch_matmul_v3_common_advanced.h"
#include "matmul/mat_mul_v3/op_host/op_tiling/arch35/matmul_tiling_registry.h"
#include "batch_matmul_v3_tiling_key.h"

namespace optiling {
namespace batch_matmul_v3_advanced {
using namespace strategy;
MM_REGISTER_TILING_TEMPLATE(BatchMatMulV3, BatchMatMulV3AswBL1FullLoadBasicTiling, ASCEND950, BL1_FULL_LOAD_BASIC);


bool BatchMatMulV3AswBL1FullLoadBasicTiling::IsCapable()
{
    if (batchInfo_->batchB > 1UL) { // matrix B should not have batch when BL1FullLoad
        return false;
    }
    if (args_.nValue > BASIC_BLOCK_SIZE_256) {
        return false;
    }
    // get align m,k,n value
    uint64_t alignMValue = ops::CeilAlign(args_.mValue, BASIC_BLOCK_SIZE_16);
    uint64_t alignKaValue = ops::CeilAlign(args_.kValue, BASIC_BLOCK_SIZE_16);
    uint64_t alignKbValue = ops::CeilAlign(args_.kValue, BASIC_BLOCK_SIZE_16);
    uint64_t alignNValue = ops::CeilAlign(args_.nValue, BASIC_BLOCK_SIZE_16);
    uint64_t alignMatASize = batchInfo_->batchA * alignMValue * alignKaValue * args_.aDtypeSize;
    uint64_t alignMatBSize = batchInfo_->batchB * alignKbValue * alignNValue * args_.bDtypeSize;
    if (alignMatASize < (compileInfo_.l1Size) * compileInfo_.aicNum &&
        batchInfo_->batchA * ops::CeilDiv(args_.mValue, BASIC_BLOCK_SIZE_256) <
            4UL * compileInfo_.aicNum) {  // each core needs to loop at least 4 batch
        return false;
    }
    if (alignMatBSize * NUM_TWO > compileInfo_.l1Size) {
        return false;
    }
    return true;
}

ge::graphStatus BatchMatMulV3AswBL1FullLoadBasicTiling::DoOpTiling()
{
    MatMulV3TilingHelper::ResetBase(compileInfo_, args_, runInfo_);
    MatMulV3TilingHelper::CalL1Tiling(compileInfo_, args_, runInfo_);
    // 复用基础API B全载
    MatMulV3BasicAswtTiling::DoBL1FullLoad(args_.batchInfo->batchA, args_.batchInfo->batchBias);

    // l1开2db后依然只使用了一半的空间，则开启4 db。该字段仅在基础api场景生效
    uint64_t alignKbValue = ops::CeilAlign(args_.kValue, BASIC_BLOCK_SIZE_16);
    uint64_t alignNValue = ops::CeilAlign(args_.nValue, BASIC_BLOCK_SIZE_16);
    uint64_t bL1TensorSize = alignKbValue * alignNValue * args_.bDtypeSize;

    uint64_t aL1TensorSize = runInfo_.baseM * runInfo_.baseK * runInfo_.stepKa * args_.aDtypeSize;
    uint64_t dtypeSize = GetSizeByDataType(ge::DT_FLOAT);
    runInfo_.dbL0C = runInfo_.baseM * runInfo_.baseN * dtypeSize * DB_SIZE <= compileInfo_.l0CSize ? DB_SIZE : 1UL;
    runInfo_.stepN = ops::CeilAlign(args_.nValue, runInfo_.baseN);
    if (aL1TensorSize * NUM_FOUR + bL1TensorSize <= compileInfo_.l1Size) {
        runInfo_.l1BufferNum = NUM_FOUR;
    } else {
        runInfo_.l1BufferNum = NUM_TWO;
    }
    return ge::GRAPH_SUCCESS;
}

uint64_t BatchMatMulV3AswBL1FullLoadBasicTiling::GetTilingKey() const
{
    return BatchMatMulV3TilingKey()
        .SetTrans(args_.isATrans, args_.isBTrans)
        .SetModel(MatMulV3Model::BASIC)
        .SetFullLoad(MatMulV3FullLoad::B_FULL_LOAD)
        .SetApiLevel(MatMulV3ApiLevel::BASIC_LEVEL)
        .GetTilingKey();
}

ge::graphStatus BatchMatMulV3AswBL1FullLoadBasicTiling::GetTilingData(TilingResult& tiling) const
{
    return GetTilingDataImpl<BatchMatMulV3BasicTilingData>(tiling);
}

} // namespace batch_matmul_v3_advanced
} // namespace optiling
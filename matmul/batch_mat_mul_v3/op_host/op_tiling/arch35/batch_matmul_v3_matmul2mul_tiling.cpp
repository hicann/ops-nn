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
 * \file batch_matmul_v3_matmul2mul_tiling.cc
 * \brief
 */

#include "batch_matmul_v3_matmul2mul_tiling.h"
#include "batch_matmul_v3_tiling_strategy.h"
#include "matmul/mat_mul_v3/op_host/op_tiling/arch35/matmul_tiling_registry.h"
#include "matmul/mat_mul_v3/op_host/op_tiling/arch35/matmul_v3_tiling_key.h"
#include "batch_matmul_v3_tiling_key.h"

namespace optiling {
namespace batch_matmul_v3_advanced {

using namespace strategy;
MM_REGISTER_TILING_TEMPLATE(BatchMatMulV3, BatchMatMulV3ToMulTiling, ASCEND910_95, BATCH_MATMUL_TO_MUL);

ge::graphStatus BatchMatMulV3ToMulTiling::DoOpTiling()
{
    uint64_t size = args_.aDtypeSize;
    uint64_t m = args_.mValue;
    uint64_t n = args_.nValue;
    uint64_t alignM = ops::CeilAlign(m, alignNum_);
    uint64_t alignN = ops::CeilAlign(n, alignNum_);
    uint64_t usedCoreNum = compileInfo_.aivNum;
    uint64_t b = batchInfo_ == nullptr ? 1 : batchInfo_->batchC;
    uint64_t singleBatchSize = (m + alignN + m * alignN) * size;
    // alignN 小于 VL/sizeof(dtypeSize) 走特殊优化分支
    if (alignN <= alignNum_) {
        singleBatchSize = (alignM + alignN + alignM * alignN) * size;
    }
    uint64_t singleCoreBatch = ops::CeilDiv(b, usedCoreNum);
    uint64_t ubLimitBatchNum = compileInfo_.ubSize / singleBatchSize;
    uint64_t batchNum = singleCoreBatch <= ubLimitBatchNum ? singleCoreBatch : ubLimitBatchNum;
    uint64_t batchNumLastRound = singleCoreBatch % batchNum == 0UL ? batchNum : singleCoreBatch % batchNum;
    uint64_t lastCoreNum = b % (batchNum * usedCoreNum) / batchNumLastRound;
    uint64_t batchNumLastRoundTail = b % (batchNum * usedCoreNum) % batchNumLastRound;
    if (batchNumLastRoundTail == 0UL) {
        batchNumLastRoundTail = batchNumLastRound;
        lastCoreNum -= 1UL;
    }

    runInfo_.toMulInfo.m = m;
    runInfo_.toMulInfo.n = n;
    runInfo_.toMulInfo.b = b;
    runInfo_.toMulInfo.usedCoreNum = usedCoreNum;
    runInfo_.toMulInfo.singleCoreBatch = singleCoreBatch;
    runInfo_.toMulInfo.batchNum = batchNum;
    runInfo_.toMulInfo.batchNumLastRound = batchNumLastRound;
    runInfo_.toMulInfo.batchNumLastRoundTail = batchNumLastRoundTail;
    runInfo_.toMulInfo.lastCoreNum = lastCoreNum;
    runInfo_.toMulInfo.alignNum = alignNum_;
    return ge::GRAPH_SUCCESS;
}

bool BatchMatMulV3ToMulTiling::IsCapable()
{
    bool isNotEqualBatch = batchInfo_->batchA0 != batchInfo_->batchB0 || batchInfo_->batchA1 != batchInfo_->batchB1 ||
                           batchInfo_->batchA2 != batchInfo_->batchB2 || batchInfo_->batchA3 != batchInfo_->batchB3;
    if (isNotEqualBatch) {
        return false;
    }
    // batch数大于等于128(AIV CoreNum *2) 才能开pingpong
    if (batchInfo_->batchC < BASIC_BLOCK_SIZE_128) {
        return false;
    }
    if (args_.hasBias) {
        return false;
    }
    if (args_.kValue != 1UL) {
        return false;
    }
    // N>256/DtypeSize才能用满Vector的计算能力，小N走特殊优化
    if (args_.nValue > BLOCK_BYTE_SIZE / args_.aDtypeSize &&
        args_.nValue <= BASIC_BLOCK_SIZE_256 / args_.aDtypeSize) {
        return false;
    }
    // N=1时部分case劣化
    if (args_.nValue == 1) {
        return false;
    }
    if ((ops::CeilAlign(args_.mValue, alignNum_) + ops::CeilAlign(args_.nValue, alignNum_) +
         ops::CeilAlign(args_.mValue, alignNum_) * ops::CeilAlign(args_.nValue, alignNum_)) *
            args_.aDtypeSize >
        compileInfo_.ubSize) {
        return false;
    }
    if (args_.nValue % (BASIC_BLOCK_SIZE_256 / args_.aDtypeSize) == 0) {
        return false;
    }
    OP_LOGI(args_.opName, "Enter BatchMatMulV3 matmul2mul module.");
    return true;
}

uint64_t BatchMatMulV3ToMulTiling::GetTilingKey() const
{
    return BatchMatMulV3TilingKey()
        .SetTrans(false, false)
        .SetBatchModel(MatMulV3BatchModel::BATCH_MATMUL_TO_MUL)
        .SetApiLevel(MatMulV3ApiLevel::BASIC_LEVEL)
        .GetTilingKey();
}

uint64_t BatchMatMulV3ToMulTiling::GetBlockDim() const
{
    return runInfo_.toMulInfo.usedCoreNum;
}

ge::graphStatus BatchMatMulV3ToMulTiling::GetTilingData(TilingResult& tiling) const
{
    return GetTilingDataImpl<BatchMatMulToMulBasicTilingData>(tiling);
}
} // namespace batch_matmul_v3_advanced
} // namespace optiling
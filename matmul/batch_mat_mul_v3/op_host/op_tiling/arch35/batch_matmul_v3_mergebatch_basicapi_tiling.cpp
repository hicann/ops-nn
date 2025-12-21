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
 * \file batch_matmul_v3_mergebatch_basicapi_tiling.cc
 * \brief
 */
#include <cmath>

#include "batch_matmul_v3_mergebatch_basicapi_tiling.h"
#include "batch_matmul_v3_tiling_strategy.h"
#include "matmul/mat_mul_v3/op_host/op_tiling/arch35/matmul_tiling_registry.h"
#include "batch_matmul_v3_tiling_key.h"

namespace optiling {
namespace batch_matmul_v3_advanced {
using namespace strategy;
MM_REGISTER_TILING_TEMPLATE(BatchMatMulV3, BatchMatMulV3MergeBatchBasicApiTiling, ASCEND910_95, MERGE_BATCH_BASICAPI);

ge::graphStatus BatchMatMulV3MergeBatchBasicApiTiling::DoOpTiling()
{
    uint64_t batchNumPerCore = ops::CeilDiv(batchInfo_->batchC, compileInfo_.aicNum);
    uint64_t maxBasek = compileInfo_.l0BSize / MIN_BATCH_L0 / alignNValue_ / args_.bDtypeSize / DB_SIZE;
    maxBasek = std::max(maxBasek / BASIC_BLOCK_SIZE_16 * BASIC_BLOCK_SIZE_16, BASIC_BLOCK_SIZE_16);
    // threshold of basek is 64
    runInfo_.baseK = std::min(maxBasek, 64UL);
    uint64_t maxBaseN = compileInfo_.l0BSize / runInfo_.baseK / args_.bDtypeSize / DB_SIZE;
    uint64_t maxBatchL0 = std::max(maxBaseN / alignNValue_, 1UL);

    double l0cElemSize = static_cast<double>(compileInfo_.l0CSize) / DB_SIZE / sizeof(float);
    uint64_t mValue = args_.isATrans ? alignMValue_ : args_.mValue;
    runInfo_.mergeBatchL0 = std::min({CalBatchL0WithPolynomial(l0cElemSize, mValue), maxBatchL0, batchNumPerCore});
    uint64_t baseM = args_.isATrans ? runInfo_.mergeBatchL0 * alignMValue_:
        ops::CeilAlign(runInfo_.mergeBatchL0 * args_.mValue, BASIC_BLOCK_SIZE_16);
    uint64_t baseN = runInfo_.mergeBatchL0 * alignNValue_;
    // 4 bufer for al1_db and bl1_db
    uint64_t stepKaMax = std::min(ops::CeilDiv(batchNumPerCore, runInfo_.mergeBatchL0),
        compileInfo_.l1Size / NUM_FOUR / (baseM * runInfo_.baseK * args_.aDtypeSize));
    uint64_t stepKbMax = std::min(ops::CeilDiv(batchNumPerCore, runInfo_.mergeBatchL0),
        compileInfo_.l1Size / NUM_FOUR / (baseN * runInfo_.baseK * args_.bDtypeSize));
    runInfo_.stepKa = std::min(std::min(stepKaMax, stepKbMax), ops::CeilDiv(alignKValue_, runInfo_.baseK));
    // calculate batchAL1 & batchBL1
    runInfo_.mergeBatchAL1 = runInfo_.mergeBatchL0;
    runInfo_.mergeBatchBL1 = runInfo_.mergeBatchL0;

    OP_LOGI(args_.opName, "In MergeBatchBasicApi module, temp mergeBatchAL1 is %lu, temp mergeBatchBL1 is %lu, \
            temp mergeBatchL0 is %lu, kL1 is %lu, baseK is %lu", runInfo_.mergeBatchAL1, runInfo_.mergeBatchBL1,
            runInfo_.mergeBatchL0, runInfo_.stepKa * runInfo_.baseK, runInfo_.baseK);
    return ge::GRAPH_SUCCESS;
}

bool BatchMatMulV3MergeBatchBasicApiTiling::IsCapable()
{
    if (args_.hasBias) {
        return false;
    }
    bool isNotEqualBatch = batchInfo_->batchA0 != batchInfo_->batchB0 || batchInfo_->batchA1 != batchInfo_->batchB1 ||
                           batchInfo_->batchA2 != batchInfo_->batchB2 || batchInfo_->batchA3 != batchInfo_->batchB3;
    if (isNotEqualBatch)  {
        return false;
    }
    // each aic should process at least 4 batchs
    if (batchInfo_->batchC < MIN_BATCH_L0 * compileInfo_.aicNum) {
        return false;
    }
    c0Size_ = BLOCK_BYTE_SIZE / args_.aDtypeSize;
    alignKValue_ =  ops::CeilAlign(args_.kValue, c0Size_);
    alignMValue_ =  ops::CeilAlign(args_.mValue, BASIC_BLOCK_SIZE_16);
    alignNValue_ = ops::CeilAlign(args_.nValue, BASIC_BLOCK_SIZE_16);
    // shape check, aligned k shape should be at least 64
    if (alignKValue_ < 64UL || args_.mValue > args_.nValue) {
        return false;
    }
    uint64_t tempAlignM = ops::CeilAlign(args_.mValue * MIN_BATCH_L0, BASIC_BLOCK_SIZE_16);
    if (args_.isATrans && args_.mValue > 1) {
        tempAlignM = MIN_BATCH_L0 * ops::CeilAlign(args_.mValue, BASIC_BLOCK_SIZE_16);
    }
    uint64_t tempAlignN = MIN_BATCH_L0 * alignNValue_;
    // l0 buffer check
    uint64_t minBaseK = c0Size_;
    if (args_.isATrans || !args_.isBTrans) {
        minBaseK = ops::CeilAlign(minBaseK, BASIC_BLOCK_SIZE_16);
    }
    uint64_t al0Size = tempAlignM * minBaseK * args_.aDtypeSize * DB_SIZE;
    uint64_t bl0Size = tempAlignN * minBaseK * args_.bDtypeSize * DB_SIZE;
    if (al0Size > compileInfo_.l0ASize || bl0Size > compileInfo_.l0BSize ||
        tempAlignM * tempAlignN * sizeof(float) * DB_SIZE > compileInfo_.l0CSize) {
        return false;
    }

    OP_LOGI(args_.opName, "Enter BatchMatmul basicapi mergebatch module.");
    return true;
}

uint64_t BatchMatMulV3MergeBatchBasicApiTiling::CalBatchL0WithPolynomial(double singleBatchSize, uint64_t mValue) const
{
    // 多项式求解最佳batchL0
    double a = double(mValue) / BASIC_BLOCK_SIZE_16;
    double p = singleBatchSize / BASIC_BLOCK_SIZE_16 / alignNValue_ * a;
    double t = double(BASIC_BLOCK_SIZE_16 - 1) / BASIC_BLOCK_SIZE_16;
    double y = sqrt(p + t * t / NUM_FOUR) - t / NUM_TWO;
    double x = floor(std::min(p / ceil(y) / a, ceil(y) / a));
    return std::max(static_cast<uint64_t>(x), 1UL);
}

uint64_t BatchMatMulV3MergeBatchBasicApiTiling::GetTilingKey() const
{
    // trans_a=true && m==1 equals to trans_a=False
    bool transA = args_.isATrans && args_.mValue > 1;
    return BatchMatMulV3TilingKey()
        .SetTrans(transA, args_.isBTrans)
        .SetBatchModel(MatMulV3BatchModel::MERGE_BATCH_MODEL)
        .SetApiLevel(MatMulV3ApiLevel::BASIC_LEVEL)
        .GetTilingKey();
}

uint64_t BatchMatMulV3MergeBatchBasicApiTiling::GetBlockDim() const
{
    return compileInfo_.aicNum;
}
} // namespace batch_matmul_v3_advanced
} // namespace optiling
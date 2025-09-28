/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file batch_matmul_v3_iterbatch_basicapi_tiling.cc
 * \brief
 */

#include "batch_matmul_v3_iterbatch_basicapi_tiling.h"
#include "batch_matmul_v3_tiling_strategy.h"
#include "matmul/mat_mul_v3/op_host/op_tiling/arch35/matmul_tiling_registry.h"

namespace optiling {
namespace batch_matmul_v3_advanced {
using namespace strategy;
MM_REGISTER_TILING_TEMPLATE(BatchMatMulV3, BatchMatMulV3IterBatchBasicApiTiling, ASCEND910_95, ITER_BATCH_BASICAPI);

bool BatchMatMulV3IterBatchBasicApiTiling::IsCapable()
{
    if (args_.hasBias) {
        return false;
    }
    bool isNotEqualBatch = batchInfo_->batchA0 != batchInfo_->batchB0 || batchInfo_->batchA1 != batchInfo_->batchB1 ||
                           batchInfo_->batchA2 != batchInfo_->batchB2 || batchInfo_->batchA3 != batchInfo_->batchB3;
    if (isNotEqualBatch)  {
        return false;
    }
    if (batchInfo_->batchC <= compileInfo_.aicNum) {
        return false;
    }
    c0Size_ = BLOCK_BYTE_SIZE / args_.aDtypeSize;
    // when fp16 or (fp32 and m,k), m align to 16; when fp32 and k,m, m align to 8 * 2 for frac combine in loadtol0a
    alignMValue_ = ops::CeilAlign(args_.mValue, BASIC_BLOCK_SIZE_16);
    alignKaValue_ = args_.isATrans ? ops::CeilAlign(args_.kValue, BASIC_BLOCK_SIZE_16) :
                                     ops::CeilAlign(args_.kValue, c0Size_);
    alignKbValue_ = args_.isBTrans ? ops::CeilAlign(args_.kValue, c0Size_) :
                                     ops::CeilAlign(args_.kValue, BASIC_BLOCK_SIZE_16);
    // when fp16 or (fp32 and n,k), m align to 16; when fp32 and k,n, n align to 8 * 2 for frac combine in loadtol0b
    alignNValue_ = ops::CeilAlign(args_.nValue, BASIC_BLOCK_SIZE_16);
    if (alignMValue_ * alignKaValue_ * args_.aDtypeSize * DB_SIZE > compileInfo_.l0ASize) {
        return false;
    }
    if (alignKbValue_ * alignNValue_ * args_.aDtypeSize * DB_SIZE > compileInfo_.l0BSize) {
        return false;
    }
    if (alignMValue_ * alignNValue_ * DATA_SIZE_FP32 * DB_SIZE > compileInfo_.l0CSize) {
        return false;
    }
    if ((alignMValue_ * alignKaValue_ + alignKbValue_ * alignNValue_) * args_.aDtypeSize * DB_SIZE >
        compileInfo_.l1Size) {
        return false;
    }
    OP_LOGI(args_.opName, "Enter BatchMatmul basicapi iterbatch module.");
    return true;
}

ge::graphStatus BatchMatMulV3IterBatchBasicApiTiling::DoOpTiling()
{
    constexpr uint64_t mmadCount = 8UL; // cube count which will cause issuequene
    constexpr uint64_t fullCopySize = 64 * 1024UL; // datasize moving once which can use full of bandwith
    uint64_t iterBatchL0A = ops::FloorDiv(compileInfo_.l0ASize / DB_SIZE,
                                          alignMValue_ * alignKaValue_ * args_.aDtypeSize);
    uint64_t iterBatchL0B = ops::FloorDiv(compileInfo_.l0BSize / DB_SIZE,
                                          alignKbValue_ * alignNValue_ * args_.aDtypeSize);
    uint64_t iterBatchL0C = ops::FloorDiv(compileInfo_.l0CSize / DB_SIZE,
                                          alignMValue_ * alignNValue_ * DATA_SIZE_FP32);
    uint64_t iterBatchL1 = ops::FloorDiv(compileInfo_.l1Size / DB_SIZE, (alignMValue_ * alignKaValue_ +
                                         alignKbValue_ * alignNValue_) * args_.aDtypeSize);
    if (mmadCount * (alignMValue_ * alignKaValue_ + alignKbValue_ * alignNValue_) * args_.aDtypeSize > fullCopySize) {
        runInfo_.iterBatchL1 = std::min({iterBatchL1, mmadCount, ops::CeilDiv(batchInfo_->batchC,
                                         compileInfo_.aicNum)});
        runInfo_.iterBatchL0 = std::min({iterBatchL0A, iterBatchL0B, iterBatchL0C, runInfo_.iterBatchL1, mmadCount});
    } else {
        runInfo_.iterBatchL1 = std::min({iterBatchL1, ops::CeilDiv(batchInfo_->batchC, compileInfo_.aicNum)});
        runInfo_.iterBatchL0 = std::min({iterBatchL0A, iterBatchL0B, iterBatchL0C, runInfo_.iterBatchL1});
        // now iterBatchL0c equals to L0a,b, which could be optimized
    }
    OP_LOGI(args_.opName, "In IterBatchBasicApi module, temp iterBatchL0A is %lu, temp iterBatchL0B is %lu, \
        temp iterBatchL0C is %lu, temp iterBatchL1 is %lu, after calculation actual runInfo_.iterBatchL0 is %lu, \
        runInfo_.iterBatchL1 is %lu,", iterBatchL0A, iterBatchL0B, iterBatchL0C, iterBatchL1, runInfo_.iterBatchL0,
        runInfo_.iterBatchL1);
    return ge::GRAPH_SUCCESS;
}

uint64_t BatchMatMulV3IterBatchBasicApiTiling::GetTilingKey() const
{
    return MatMulV3TilingKey()
        .SetTrans(args_.isATrans, args_.isBTrans)
        .SetModel(MatMulV3Model::ITER_BATCH_BATCH_BIAS)
        .SetApiLevel(MatMulV3ApiLevel::BASIC_LEVEL)
        .GetTilingKey();
}

uint64_t BatchMatMulV3IterBatchBasicApiTiling::GetBlockDim() const
{
    return compileInfo_.aicNum;
}
}
}
/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "batch_matmul_v3_iterbatch_broadcast_basicapi_tiling.h"
#include "batch_matmul_v3_iterbatch_basicapi_tiling.h"
#include "batch_matmul_v3_tiling_strategy.h"
#include "matmul/mat_mul_v3/op_host/op_tiling/arch35/matmul_tiling_registry.h"
#include "batch_matmul_v3_tiling_key.h"

namespace optiling {
namespace batch_matmul_v3_advanced {
constexpr uint64_t MAX_BATCH_DIM = 4UL;
constexpr uint64_t FIRST_BATCH_IDX = 0UL;
constexpr uint64_t SECOND_BATCH_IDX = 1UL;
constexpr uint64_t THIRD_BATCH_IDX = 2UL;
constexpr uint64_t FOURTH_BATCH_IDX = 3UL;
using namespace strategy;
MM_REGISTER_TILING_TEMPLATE(BatchMatMulV3, BatchMatMulV3IterBatchBroadcastBasicApiTiling, DAV_3510,
    ITER_BATCH_BROADCAST_BASICAPI);

bool BatchMatMulV3IterBatchBroadcastBasicApiTiling::IsContiguousStride(StrideIndexPairs& strideIndexPairs) const
{
    int64_t expectStride = 1;
    for (auto it = strideIndexPairs.rbegin(); it != strideIndexPairs.rend(); it++) {
        if (it->first != expectStride) {
            return false;
        }
        expectStride *= it->second.second;
    }
    return true;
}

uint64_t BatchMatMulV3IterBatchBroadcastBasicApiTiling::GetTilingKey() const
{
    MatMulV3BatchModel batchModel = (broadcastAxisA_ < MAX_BATCH_DIM)
        ? MatMulV3BatchModel::ITER_BATCH_BROADCAST_A_MODEL
        : MatMulV3BatchModel::ITER_BATCH_BROADCAST_B_MODEL;
    return BatchMatMulV3TilingKey()
        .SetTrans(args_.isATrans, args_.isBTrans)
        .SetBatchModel(batchModel)
        .SetL0C2Out(MatMulV3L0C2Out::ON_THE_FLY)
        .SetApiLevel(MatMulV3ApiLevel::TENSOR_LEVEL)
        .GetTilingKey();
}

bool BatchMatMulV3IterBatchBroadcastBasicApiTiling::IsInputNonContiguous(uint32_t inputIdx) const
{
    if (!context_->InputIsView(inputIdx)) {
        return false;
    }
    auto strides = context_->GetInputStride(inputIdx);
    auto viewShape = context_->GetInputShape(inputIdx)->GetOriginShape();
    int64_t dimNum = strides->GetDimNum();
    StrideIndexPairs pairs;
    pairs.reserve(dimNum);
    for (int64_t i = 0; i < dimNum; i++) {
        int64_t curStride = strides->GetStride(i);
        if (curStride == 0 || viewShape[i] == 1) {
            continue;
        }
        pairs.emplace_back(std::make_pair(curStride, std::make_pair(i, viewShape[i])));
    }
    if (pairs.empty()) {
        return false;
    }
    std::sort(pairs.rbegin(), pairs.rend());
    return !IsContiguousStride(pairs);
}

uint64_t BatchMatMulV3IterBatchBroadcastBasicApiTiling::GetABatchDim(int32_t idx) const
{
    return (idx == 0) ? batchInfo_->batchA0 : (idx == 1) ? batchInfo_->batchA1 :
           (idx == 2) ? batchInfo_->batchA2 : batchInfo_->batchA3;
}

uint64_t BatchMatMulV3IterBatchBroadcastBasicApiTiling::GetBBatchDim(int32_t idx) const
{
    return (idx == 0) ? batchInfo_->batchB0 : (idx == 1) ? batchInfo_->batchB1 :
           (idx == 2) ? batchInfo_->batchB2 : batchInfo_->batchB3;
}

bool BatchMatMulV3IterBatchBroadcastBasicApiTiling::hasBroadcastAxis()
{
    uint64_t broadCastA = 0;
    uint64_t broadCastB = 0;
    broadcastAxisA_ = MAX_BATCH_DIM;
    broadcastAxisB_ = MAX_BATCH_DIM;

    for (uint64_t i = 0; i < MAX_BATCH_DIM; ++i) {
        uint64_t aDim = GetABatchDim(i);
        uint64_t bDim = GetBBatchDim(i);
        if (aDim == 1UL && bDim != 1UL) {
            broadCastA++;
            broadcastAxisA_ = i;
        } else if (bDim == 1UL && aDim != 1UL) {
            broadCastB++;
            broadcastAxisB_ = i;
        }
    }

    if (broadCastA > 0 && broadCastB > 0) {
        OP_LOGD(args_.opName, "[iterbatch_broadcast_basicapi] Dual-side broadcast is not supported.");
        return false;
    }
    if (broadCastA > 1 || broadCastB > 1) {
        OP_LOGD(args_.opName, "[iterbatch_broadcast_basicapi] Multi-axis broadcast is not supported.");
        return false;
    }
    if (broadCastA == 0 && broadCastB == 0) {
        OP_LOGD(args_.opName, "[iterbatch_broadcast_basicapi] No broadcast axis detected, should use ITER_BATCH.");
        return false;
    }
    return true;
}

bool BatchMatMulV3IterBatchBroadcastBasicApiTiling::CheckNonBroadcastAxisMatch() const
{
    uint64_t bcAxis = (broadcastAxisA_ < MAX_BATCH_DIM) ? broadcastAxisA_ : broadcastAxisB_;
    for (uint64_t i = 0; i < MAX_BATCH_DIM; ++i) {
        if (static_cast<uint32_t>(i) == bcAxis) {
            continue;
        }
        uint64_t aDim = GetABatchDim(i);
        uint64_t bDim = GetBBatchDim(i);
        if (aDim != bDim) {
            OP_LOGD(args_.opName, "[iterbatch_broadcast_basicapi] Non-broadcast axis mismatch on axis %d.", i);
            return false;
        }
    }
    return true;
}

bool BatchMatMulV3IterBatchBroadcastBasicApiTiling::CheckL1IterBatch()
{
    alignMValue_ = ops::CeilAlign(args_.mValue, BASIC_BLOCK_SIZE_16);
    alignKValue_ = ops::CeilAlign(args_.kValue, BASIC_BLOCK_SIZE_16);
    alignNValue_ = ops::CeilAlign(args_.nValue, BASIC_BLOCK_SIZE_16);

    const uint64_t c0Size = 32UL / args_.aDtypeSize; // 32 bytes per block
    const uint64_t dtypeSize = args_.aDtypeSize;
    sizeAOneBatch_ = alignMValue_ * alignKValue_ * dtypeSize;
    sizeBOneBatch_ = alignNValue_ * alignKValue_ * dtypeSize;
    sizeCOneBatch_ = alignMValue_ * alignNValue_ * DATA_SIZE_FP32;
    uint64_t alignBiasSize = 0UL;
    if (args_.hasBias) {
        alignBiasSize = alignNValue_ * GetSizeByDataType(args_.biasType);
    }
    if ((sizeAOneBatch_ + sizeBOneBatch_ + alignBiasSize) * DB_SIZE > compileInfo_.l1Size) {
        return false;
    }

    uint64_t l1Avail = compileInfo_.l1Size / DB_SIZE;
    if (broadcastAxisA_ == FOURTH_BATCH_IDX) {
        iterBatchL1_ = ops::FloorDiv(l1Avail - sizeAOneBatch_ - alignBiasSize, sizeBOneBatch_);
    } else if (broadcastAxisB_ == FOURTH_BATCH_IDX) {
        iterBatchL1_ = ops::FloorDiv(l1Avail - sizeBOneBatch_ - alignBiasSize, sizeAOneBatch_);
    } else {
        iterBatchL1_ = ops::FloorDiv(l1Avail - alignBiasSize, sizeAOneBatch_ + sizeBOneBatch_);
    }
    // iterBatchL1 expected to be no less than 2
    if (iterBatchL1_ < 2UL) {
        OP_LOGD(args_.opName, "[iterbatch_broadcast_basicapi] iterBatchL1 < 2, insufficient L1 capacity.");
        return false;
    }

    uint64_t preCoreBatch = ops::CeilDiv(batchInfo_->batchC, compileInfo_.aicNum);
    iterBatchL1_ = std::min(iterBatchL1_, preCoreBatch);

    uint64_t innerDimsProduct = 1;
    bool isBroadcastA = broadcastAxisA_ < MAX_BATCH_DIM;
    uint32_t bcAxis = isBroadcastA ? broadcastAxisA_ : broadcastAxisB_;
    if (bcAxis < FOURTH_BATCH_IDX) {
        for (uint64_t i = bcAxis + 1; i < MAX_BATCH_DIM; ++i) {
            innerDimsProduct *= isBroadcastA ? GetABatchDim(i) : GetBBatchDim(i);
        }
    } else {
        innerDimsProduct = std::max(GetABatchDim(FOURTH_BATCH_IDX), GetBBatchDim(FOURTH_BATCH_IDX));
    }
    if (innerDimsProduct > 1) {
        uint64_t candidate = std::min(iterBatchL1_, innerDimsProduct);
        // iterBatchL1 expected to be no less than 2
        while (candidate >= 2UL) {
            if (innerDimsProduct % candidate == 0) {
                break;
            }
            --candidate;
        }
        iterBatchL1_ = candidate;
    }
    // iterBatchL1 expected to be no less than 2
    if (iterBatchL1_ < 2UL) {
        OP_LOGD(args_.opName, "[iterbatch_broadcast_basicapi] iterBatchL1 < 2 after contiguity constraint.");
        return false;
    }
    return true;
}

bool BatchMatMulV3IterBatchBroadcastBasicApiTiling::CheckL0IterBatch()
{
    iterBatchL0A_ = ops::FloorDiv(compileInfo_.l0ASize / DB_SIZE, sizeAOneBatch_);
    iterBatchL0B_ = ops::FloorDiv(compileInfo_.l0BSize / DB_SIZE, sizeBOneBatch_);
    iterBatchL0C_ = ops::FloorDiv(compileInfo_.l0CSize / DB_SIZE, sizeCOneBatch_);
    uint64_t iterBatchL0AWithoutDB = ops::FloorDiv(compileInfo_.l0ASize, sizeAOneBatch_);
    uint64_t iterBatchL0BWithoutDB = ops::FloorDiv(compileInfo_.l0BSize, sizeBOneBatch_);
    l0CanLoadBatch_ = (std::min({iterBatchL0A_, iterBatchL0B_, iterBatchL0C_}) >= 1UL) ||
        (iterBatchL0AWithoutDB >= 1 && iterBatchL0BWithoutDB >= 1 && iterBatchL0C_ > 1); // try to reduce fixpipe instr
    constexpr static double defaultBalanceOfBatch = 0.8;
    if (!l0CanLoadBatch_) {
        double avgIterBatch = static_cast<double>(batchInfo_->batchC) / static_cast<double>(compileInfo_.aicNum);
        double actualMaxIterBatch = static_cast<double>(ops::CeilDiv(ops::CeilDiv(batchInfo_->batchC, iterBatchL1_),
                                    compileInfo_.aicNum) * iterBatchL1_);
        double balanceRateOfBatch = avgIterBatch / actualMaxIterBatch;
        if (balanceRateOfBatch < defaultBalanceOfBatch) {
            OP_LOGI(args_.opName, "FormulateBalanceRate lower than 0.8, unable to enter in bmm iterbatch module");
            return false;
        }
    }
    return true;
}

bool BatchMatMulV3IterBatchBroadcastBasicApiTiling::IsCapable()
{
    if (args_.aFormat == ge::FORMAT_FRACTAL_NZ || args_.bFormat == ge::FORMAT_FRACTAL_NZ) {
        OP_LOGD(args_.opName, "[iterbatch_broadcast_basicapi] The NZ format is not supported.");
        return false;
    }
    if (IsInputNonContiguous(0) || IsInputNonContiguous(1)) {
        OP_LOGD(args_.opName,
            "[iterbatch_broadcast_basicapi] Non-contiguous input strides not supported.");
        return false;
    }
    if (batchInfo_->batchBias > 1UL) {
        OP_LOGD(args_.opName, "[iterbatch_broadcast_basicapi] Multi-batch bias is not supported.");
        return false;
    }
    if (!hasBroadcastAxis()) {
        return false;
    }
    if (!CheckNonBroadcastAxisMatch()) {
        return false;
    }
    if (batchInfo_->batchC <= compileInfo_.aicNum) {
        OP_LOGD(args_.opName, "[iterbatch_broadcast_basicapi] batchC <= aicNum, no need for iterbatch.");
        return false;
    }
    if (!CheckL1IterBatch() || !CheckL0IterBatch()) {
        return false;
    }
    OP_LOGI(args_.opName, "Enter BatchMatmul basicapi iterbatch_broadcast module.");
    return true;
}

uint64_t BatchMatMulV3IterBatchBroadcastBasicApiTiling::GetNumBlocks() const
{
    return compileInfo_.aicNum;
}

ge::graphStatus BatchMatMulV3IterBatchBroadcastBasicApiTiling::DoOpTiling()
{
    runInfo_.iterBatchL1 = iterBatchL1_;
    runInfo_.iterBatchL0 = std::max(std::min({iterBatchL0A_, iterBatchL0B_, iterBatchL0C_, iterBatchL1_}), 1UL);
    runInfo_.bmmRunInfo.broadcastAxisA = broadcastAxisA_;
    runInfo_.bmmRunInfo.broadcastAxisB = broadcastAxisB_;

    if (l0CanLoadBatch_) {
        runInfo_.baseM = alignMValue_;
        runInfo_.baseN = alignNValue_;
        runInfo_.baseK = alignKValue_;
    } else {
        if ((alignMValue_ < alignNValue_) && (alignMValue_ > alignKValue_)) {
            runInfo_.baseK = alignKValue_;
            runInfo_.baseM = std::min(ops::FloorDiv(compileInfo_.l0ASize / DB_SIZE / args_.aDtypeSize,
                              runInfo_.baseK), alignMValue_);
            runInfo_.baseN = std::min(ops::FloorDiv(compileInfo_.l0BSize / DB_SIZE / args_.aDtypeSize,
                              runInfo_.baseK), alignNValue_);
        } else if ((alignMValue_ < alignNValue_) && (alignMValue_ <= alignKValue_)) {
            runInfo_.baseM = alignMValue_;
            runInfo_.baseK = std::min(ops::FloorDiv(compileInfo_.l0ASize / DB_SIZE / args_.aDtypeSize,
                              runInfo_.baseM), alignKValue_);
            runInfo_.baseN = std::min(ops::FloorDiv(compileInfo_.l0BSize / DB_SIZE / args_.aDtypeSize,
                              runInfo_.baseK), alignNValue_);
        } else if ((alignMValue_ >= alignNValue_) && (alignNValue_ > alignKValue_)) {
            runInfo_.baseK = alignKValue_;
            runInfo_.baseN = std::min(ops::FloorDiv(compileInfo_.l0BSize / DB_SIZE / args_.aDtypeSize,
                              runInfo_.baseK), alignNValue_);
            runInfo_.baseM = std::min(ops::FloorDiv(compileInfo_.l0ASize / DB_SIZE / args_.aDtypeSize,
                              runInfo_.baseK), alignMValue_);
        } else if ((alignMValue_ >= alignNValue_) && (alignNValue_ <= alignKValue_)) {
            runInfo_.baseN = alignNValue_;
            runInfo_.baseK = std::min(ops::FloorDiv(compileInfo_.l0BSize / DB_SIZE / args_.aDtypeSize,
                              runInfo_.baseN), alignKValue_);
            runInfo_.baseM = std::min(ops::FloorDiv(compileInfo_.l0ASize / DB_SIZE / args_.aDtypeSize,
                              runInfo_.baseK), alignMValue_);
        }
        if (args_.hasBias) {
            runInfo_.baseN = std::min(runInfo_.baseN, compileInfo_.btSize / DB_SIZE / DATA_SIZE_FP32);
        }
        if (runInfo_.baseM >= runInfo_.baseN) {
            runInfo_.baseM = std::min(runInfo_.baseM, ops::FloorDiv(compileInfo_.l0CSize / DB_SIZE / NUM_FOUR,
                              runInfo_.baseN));
        } else {
            runInfo_.baseN = std::min(runInfo_.baseN, ops::FloorDiv(compileInfo_.l0CSize / DB_SIZE / NUM_FOUR,
                              runInfo_.baseM));
        }
        runInfo_.baseM = ops::FloorAlign(std::max(runInfo_.baseM, BASIC_BLOCK_SIZE_16), BASIC_BLOCK_SIZE_16);
        runInfo_.baseN = ops::FloorAlign(std::max(runInfo_.baseN, BASIC_BLOCK_SIZE_16), BASIC_BLOCK_SIZE_16);
        runInfo_.baseK = ops::FloorAlign(std::max(runInfo_.baseK, BASIC_BLOCK_SIZE_16), BASIC_BLOCK_SIZE_16);
    }
    runInfo_.usedCoreNum = compileInfo_.aicNum;

    OP_LOGI(args_.opName,
        "In IterBatchBroadcastBasicApi module, iterBatchL0A=%lu, iterBatchL0B=%lu, iterBatchL0C=%lu, "
        "iterBatchL1=%lu, runInfo_.iterBatchL0=%lu, runInfo_.iterBatchL1=%lu, "
        "runInfo_.baseM=%lu, runInfo_.baseN=%lu, broadcastAxisA=%u, broadcastAxisB=%u",
        iterBatchL0A_, iterBatchL0B_, iterBatchL0C_, iterBatchL1_,
        runInfo_.iterBatchL0, runInfo_.iterBatchL1,
        runInfo_.baseM, runInfo_.baseN, broadcastAxisA_, broadcastAxisB_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BatchMatMulV3IterBatchBroadcastBasicApiTiling::GetTilingData(TilingResult& tiling) const
{
    return GetTilingDataImpl<BatchMatMulV3TilingData>(tiling);
}

} // namespace batch_matmul_v3_advanced
} // namespace optiling

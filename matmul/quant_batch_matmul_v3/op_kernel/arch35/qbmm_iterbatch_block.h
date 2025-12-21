/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file qbmm_iterbatch_block.h
 * \brief
 */

#ifndef QBMM_ITERBATCH_BLOCK_H
#define QBMM_ITERBATCH_BLOCK_H

#include "../quant_batch_matmul_v3_base.h"
#include "quant_batch_matmul_v3_tiling_data.h"

namespace QuantBatchMatmulV3 {
struct QbmmMultiBatchBaseBlockOffset {
    uint64_t offsetA;
    uint64_t offsetB;
    uint64_t offsetC;
    uint64_t offsetBias;
    uint64_t offsetScale;
    uint64_t scaleScalar;
    uint64_t batchCOffset;
};

struct QbmmMultiBatchBaseBlockArgs {
    uint64_t singleASize;
    uint64_t singleBSize;
    uint64_t singleCSize;
    uint64_t mainLoopPreCoreBatchNum;
    uint64_t lastLoopAllBatchNum;
    uint64_t lastLoopPreCoreBatchNum;
    uint64_t lastLoopBlockNum;
    uint64_t loopTimes;
    uint64_t batchIndex;
    uint64_t batchAIndex;
    uint64_t batchBIndex;
    uint64_t batchANum;
    uint64_t batchBNum;
    uint64_t useCoreNum;
    uint64_t nBatchOutNum;
    // outshape = (batch1, batch2, batch3, batch4, m, n)
    uint64_t iterBatch3;
    uint64_t iterBatch2;
    uint64_t iterBatch1;
    uint64_t calcBatchNum;
    uint64_t innerBatchNum;
    uint64_t broadcastFlag;
};

class QbmmMultiBatchBaseBlock {
public:
    __aicore__ inline QbmmMultiBatchBaseBlock()
    {
    }
    // tilingData改为内部结构体
    __aicore__ inline void Init(const DequantBmm::QuantBatchMatmulV3TilingDataParams *tilingData, bool bTrans,
                                bool isWeightNz);
    __aicore__ inline void GetMultiBatchInfo(uint64_t loopIndex, uint32_t blockIdx);
    __aicore__ inline void GetCalcBatchOffset(uint64_t b1Index, uint64_t b2Index, uint64_t b3Index);
    __aicore__ inline void CalcGMOffset();

public:
    QbmmMultiBatchBaseBlockOffset offset_;
    QbmmMultiBatchBaseBlockArgs params_;
    const DequantBmm::QuantBatchMatmulV3TilingDataParams *tilingData_;

protected:
    __aicore__ inline void UpdateBatchInfo();
    __aicore__ inline void GetBatchInfo(uint64_t batchNum);
};
__aicore__ inline void QbmmMultiBatchBaseBlock::Init(const DequantBmm::QuantBatchMatmulV3TilingDataParams *tilingData,
                                                     bool bTrans, bool isWeightNz)
{
    tilingData_ = tilingData;
    params_.singleASize =
        static_cast<uint64_t>(tilingData_->matmulTiling.M) * static_cast<uint64_t>(tilingData_->matmulTiling.Ka);
    uint64_t nAlign = DequantBmm::Align(static_cast<uint64_t>(tilingData_->matmulTiling.N),
                                        static_cast<uint64_t>(bTrans ? BMM_BLOCK_NUM : K0_INT8));
    uint64_t kAlign = DequantBmm::Align(static_cast<uint64_t>(tilingData_->matmulTiling.Kb),
                                        static_cast<uint64_t>(bTrans ? K0_INT8 : BMM_BLOCK_NUM));
    params_.singleBSize = isWeightNz ? nAlign * kAlign
                                     : static_cast<uint64_t>(tilingData_->matmulTiling.N) *
                                           static_cast<uint64_t>(tilingData_->matmulTiling.Kb);
    params_.singleCSize =
        static_cast<uint64_t>(tilingData_->matmulTiling.M) * static_cast<uint64_t>(tilingData_->matmulTiling.N);

    params_.useCoreNum = tilingData_->matmulTiling.usedCoreNum;
    params_.mainLoopPreCoreBatchNum = tilingData_->matmulTiling.BatchNum;
    params_.nBatchOutNum =
        DequantBmm::Min(L0C_SIZE_256K / (tilingData_->matmulTiling.baseM * tilingData_->matmulTiling.baseN *
                                         tilingData_->matmulTiling.dbL0C * sizeof(int32_t)),
                        static_cast<uint64_t>(tilingData_->matmulTiling.BatchNum));
    UpdateBatchInfo();
    params_.loopTimes = DequantBmm::CeilDiv(params_.calcBatchNum, params_.mainLoopPreCoreBatchNum * params_.useCoreNum);

    params_.lastLoopAllBatchNum = params_.calcBatchNum % (params_.mainLoopPreCoreBatchNum * params_.useCoreNum);
    // 当前仅支持最后一个loop batch整除的场景
    params_.lastLoopAllBatchNum = params_.lastLoopAllBatchNum == 0
                                      ? params_.mainLoopPreCoreBatchNum * params_.useCoreNum
                                      : params_.lastLoopAllBatchNum;

    params_.lastLoopPreCoreBatchNum = params_.lastLoopAllBatchNum / params_.useCoreNum;
    params_.lastLoopBlockNum = params_.lastLoopAllBatchNum % params_.useCoreNum;
    params_.batchIndex = 0;
    params_.batchAIndex = 0;
    params_.batchBIndex = 0;
    params_.batchANum = 1;
    params_.batchBNum = 1;
}

__aicore__ inline void QbmmMultiBatchBaseBlock::UpdateBatchInfo()
{
    if (tilingData_->params.batchA == 1 || tilingData_->params.batchB == 1 ||
        tilingData_->params.batchA == tilingData_->params.batchB) {
        // no need broadcast or batchA = 1 or batchB = 1
        params_.innerBatchNum = 0;
        params_.calcBatchNum = tilingData_->params.batchC;
        params_.broadcastFlag = 0;
    } else if (tilingData_->params.batchA4 != tilingData_->params.batchB4) {
        // batch4 need broadcast
        params_.iterBatch3 = tilingData_->params.batchC3;
        params_.iterBatch2 = tilingData_->params.batchC2;
        params_.iterBatch1 = tilingData_->params.batchC1;
        params_.calcBatchNum = tilingData_->params.batchC4;
        params_.innerBatchNum = 1;
        params_.broadcastFlag = (tilingData_->params.batchA4 == 1) ? A_NEED_BROADCAST : B_NEED_BROADCAST;
    } else if (tilingData_->params.batchA3 != tilingData_->params.batchB3) {
        // batch3 need broadcast
        params_.iterBatch3 = 1;
        params_.iterBatch2 = tilingData_->params.batchC2;
        params_.iterBatch1 = tilingData_->params.batchC1;
        params_.innerBatchNum = tilingData_->params.batchC4;
        params_.calcBatchNum = tilingData_->params.batchC4 * tilingData_->params.batchC3;
        params_.broadcastFlag = (tilingData_->params.batchA3 == 1) ? A_NEED_BROADCAST : B_NEED_BROADCAST;
    } else if (tilingData_->params.batchA2 != tilingData_->params.batchB2) {
        // batch2 need broadcast
        params_.iterBatch3 = 1;
        params_.iterBatch2 = 1;
        params_.iterBatch1 = tilingData_->params.batchC1;
        params_.innerBatchNum = tilingData_->params.batchC4 * tilingData_->params.batchC3;
        params_.calcBatchNum = tilingData_->params.batchC4 * tilingData_->params.batchC3 * tilingData_->params.batchC2;
        params_.broadcastFlag = (tilingData_->params.batchA2 == 1) ? A_NEED_BROADCAST : B_NEED_BROADCAST;
    } else {
        // batch1 need broadcast
        params_.iterBatch3 = 1;
        params_.iterBatch2 = 1;
        params_.iterBatch1 = 1;
        params_.innerBatchNum = tilingData_->params.batchC4 * tilingData_->params.batchC3 * tilingData_->params.batchC2;
        params_.calcBatchNum = tilingData_->params.batchC;
        params_.broadcastFlag = (tilingData_->params.batchA1 == 1) ? A_NEED_BROADCAST : B_NEED_BROADCAST;
    }
}

__aicore__ inline void QbmmMultiBatchBaseBlock::GetBatchInfo(uint64_t batchNum)
{
    if (params_.innerBatchNum == 0) {
        params_.batchANum = DequantBmm::Min(batchNum, static_cast<uint64_t>(tilingData_->params.batchA));
        params_.batchBNum = DequantBmm::Min(batchNum, static_cast<uint64_t>(tilingData_->params.batchB));
        params_.batchAIndex =
            DequantBmm::Min(params_.batchIndex, static_cast<uint64_t>(tilingData_->params.batchA - 1));
        params_.batchBIndex =
            DequantBmm::Min(params_.batchIndex, static_cast<uint64_t>(tilingData_->params.batchB - 1));
    } else if (params_.batchIndex >= (params_.innerBatchNum - 1) && params_.broadcastFlag == A_NEED_BROADCAST) {
        // 当本次计算需要使用的batch数量超过需要broadcast的batch轴的内轴数量，且A矩阵的batch需要broadcast
        params_.batchANum = DequantBmm::Min(params_.innerBatchNum, batchNum);
        params_.batchBNum = batchNum;
        params_.batchAIndex = params_.batchIndex % params_.innerBatchNum;
        params_.batchBIndex = params_.batchIndex;
    } else if (params_.batchIndex >= (params_.innerBatchNum - 1) && params_.broadcastFlag == B_NEED_BROADCAST) {
        // 当本次计算需要使用的batch数量超过需要broadcast的batch轴的内轴数量，且B矩阵的batch需要broadcast
        params_.batchANum = batchNum;
        params_.batchBNum = DequantBmm::Min(params_.innerBatchNum, batchNum);
        params_.batchAIndex = params_.batchIndex;
        params_.batchBIndex = params_.batchIndex % params_.innerBatchNum;
    } else {
        params_.batchANum = batchNum;
        params_.batchBNum = batchNum;
        params_.batchAIndex = params_.batchIndex;
        params_.batchBIndex = params_.batchIndex;
    }
}

__aicore__ inline void QbmmMultiBatchBaseBlock::GetMultiBatchInfo(uint64_t loopIndex, uint32_t blockIdx)
{
    // main loop
    if (loopIndex + 1 < params_.loopTimes) {
        params_.batchIndex = loopIndex * params_.mainLoopPreCoreBatchNum * params_.useCoreNum +
                             blockIdx * params_.mainLoopPreCoreBatchNum;
        GetBatchInfo(params_.mainLoopPreCoreBatchNum);
    } else if (blockIdx < params_.lastLoopBlockNum) {
        params_.batchIndex = loopIndex * params_.mainLoopPreCoreBatchNum * params_.useCoreNum +
                             blockIdx * (params_.lastLoopPreCoreBatchNum + 1);
        GetBatchInfo(params_.lastLoopPreCoreBatchNum + 1);
    } else {
        params_.batchIndex = loopIndex * params_.mainLoopPreCoreBatchNum * params_.useCoreNum +
                             params_.lastLoopBlockNum * (params_.lastLoopPreCoreBatchNum + 1) +
                             (blockIdx - params_.lastLoopBlockNum) * params_.lastLoopPreCoreBatchNum;

        GetBatchInfo(params_.lastLoopPreCoreBatchNum);
    }
}

__aicore__ inline void QbmmMultiBatchBaseBlock::GetCalcBatchOffset(uint64_t b1Index, uint64_t b2Index, uint64_t b3Index)
{
    if (params_.iterBatch1 != tilingData_->params.batchC1) {
        // batch1 need broadcast
        offset_.batchCOffset = 0;
    } else if (params_.iterBatch2 != tilingData_->params.batchC2) {
        // batch2 need broadcast
        offset_.batchCOffset = b1Index;
    } else if (params_.iterBatch3 != tilingData_->params.batchC3) {
        // batch3 need broadcast
        offset_.batchCOffset = b2Index + b1Index * params_.iterBatch2;
    } else {
        // batch4 need broadcast
        offset_.batchCOffset = b3Index + b2Index * params_.iterBatch3 + b1Index * params_.iterBatch2;
    }
}

__aicore__ inline void QbmmMultiBatchBaseBlock::CalcGMOffset()
{
    if (params_.broadcastFlag == A_NEED_BROADCAST) {
        offset_.offsetA = params_.batchAIndex * params_.singleASize +
                          offset_.batchCOffset * params_.innerBatchNum * params_.singleASize;
        offset_.offsetB = params_.batchBIndex * params_.singleBSize +
                          offset_.batchCOffset * params_.calcBatchNum * params_.singleBSize;
        offset_.offsetC = params_.batchIndex * params_.singleCSize +
                          offset_.batchCOffset * params_.calcBatchNum * params_.singleCSize;
    } else if (params_.broadcastFlag == B_NEED_BROADCAST) {
        offset_.offsetA = params_.batchAIndex * params_.singleASize +
                          offset_.batchCOffset * params_.calcBatchNum * params_.singleASize;
        offset_.offsetB = params_.batchBIndex * params_.singleBSize +
                          offset_.batchCOffset * params_.innerBatchNum * params_.singleBSize;
        offset_.offsetC = params_.batchIndex * params_.singleCSize +
                          offset_.batchCOffset * params_.calcBatchNum * params_.singleCSize;
    } else {
        offset_.offsetA = params_.batchAIndex * params_.singleASize;
        offset_.offsetB = params_.batchBIndex * params_.singleBSize;
        offset_.offsetC = params_.batchIndex * params_.singleCSize;
    }
    offset_.offsetScale = 0;
    offset_.offsetBias = 0;
}
}  // namespace QuantBatchMatmulV3

#endif  // QBMM_ITERBATCH_BLOCK_H
/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file qbmm_int4_to_int8_preprocess.h
 * \brief Preprocess class for converting int4 inputs (x1, x2) to int8 before matrix multiplication.
 *        Core splitting: cores are split between x1 and x2 by m:n ratio.
 *        Each core processes ONLY x1 or x2.
 *        Pipeline: CopyIn -> Compute (int4->half->int8) -> CopyOut
 */

#pragma once

#include "../quant_batch_matmul_v3_base.h"
#include "quant_batch_matmul_v3_tiling_data.h"

using namespace AscendC;

namespace {
constexpr uint64_t ALIGN_SIZE_128 = 128u;
constexpr uint64_t TILE_ELEMS_16K = 16 * 1024u;
constexpr uint32_t ELEM_ALIGN_64 = 64u;
constexpr uint64_t NUM_2 = 2u;
}

class QbmmInt4ToInt8Preprocess {
public:
    __aicore__ inline QbmmInt4ToInt8Preprocess() {}

    __aicore__ inline void Init(GM_ADDR x1In, GM_ADDR x2In, GM_ADDR workspace,
                                TPipe& pipe,  uint64_t m, uint64_t n,
                                uint64_t k, uint64_t batchC);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(uint32_t progress, uint32_t currentNum);
    __aicore__ inline void Compute(uint32_t currentNum);
    __aicore__ inline void CopyOut(uint32_t progress, uint32_t currentNum);

private:
    // Queues
    TQue<QuePosition::VECIN,   BUFFER_NUM> inQueueInt4_;
    TQue<QuePosition::VECCALC, BUFFER_NUM> computeQueueHalf_;
    TQue<QuePosition::VECOUT,  BUFFER_NUM> outQueueInt8_;

    GlobalTensor<int8_t> srcInt4Global_;
    GlobalTensor<int8_t> dstInt8Global_;

    GM_ADDR x1Out_;
    GM_ADDR x2Out_;

    bool    isX1Core_;
    uint64_t blockLength_ = 0;
    uint32_t ubLength_ = 0;
};

__aicore__ inline void QbmmInt4ToInt8Preprocess::Init(GM_ADDR x1In, GM_ADDR x2In, GM_ADDR workspace,
                                                        TPipe& pipe, uint64_t m, uint64_t n, uint64_t k,
                                                        uint64_t batchC)
{
    uint64_t x1TotalElems = batchC * m * k;
    uint64_t x2TotalElems = k * n;

    x1Out_ = workspace;
    x2Out_ = workspace + DequantBmm::Align(x1TotalElems * sizeof(int8_t), ALIGN_SIZE_128);

    // ---- core assignment by m:n ratio ----
    uint64_t totalCores = GetBlockNum();
    if ASCEND_IS_AIV {
        totalCores = totalCores * NUM_2;
    }
    uint64_t coresForX1 = (totalCores * batchC * m + (batchC * m + n) / NUM_2) / (batchC * m + n);
    if (coresForX1 < 1)           coresForX1 = 1;
    if (coresForX1 >= totalCores) coresForX1 = totalCores - 1;
    uint64_t coresForX2 = totalCores - coresForX1;

    uint64_t coreIdx = GetBlockIdx();
    if (coreIdx >= totalCores) {
        return;
    }
    isX1Core_ = (coreIdx < coresForX1);

    // ---- compute this core's element range ----
    uint64_t totalElems = isX1Core_ ? x1TotalElems : x2TotalElems;
    uint64_t groupCores = isX1Core_ ? coresForX1   : coresForX2;
    uint64_t localId    = isX1Core_ ? coreIdx       : coreIdx - coresForX1;

    uint64_t totalBlocks = DequantBmm::Align(totalElems, static_cast<uint64_t>(ELEM_ALIGN_64));
    uint64_t numChunks = totalBlocks / ELEM_ALIGN_64;
    uint64_t baseChunks = numChunks / groupCores;
    uint64_t remainChunks = numChunks % groupCores;

    uint64_t chunkStart = localId * baseChunks + DequantBmm::Min(localId, remainChunks);
    uint64_t chunkEnd = chunkStart + baseChunks + (localId < remainChunks ? 1u : 0u);

    uint64_t elemStart = chunkStart * ELEM_ALIGN_64;
    uint64_t elemEnd   = chunkEnd * ELEM_ALIGN_64;

    blockLength_ = elemEnd - elemStart;

    uint64_t tileSize = DequantBmm::Min(blockLength_, TILE_ELEMS_16K);
    uint64_t alignUb = DequantBmm::Align(tileSize, static_cast<uint64_t>(ELEM_ALIGN_64));
    ubLength_ = DequantBmm::Max(ELEM_ALIGN_64, static_cast<uint32_t>(alignUb));

    // ---- bind global buffers to this core's slice ----
    if (isX1Core_) {
        srcInt4Global_.SetGlobalBuffer((__gm__ int8_t*)x1In   + elemStart / NUM_2, blockLength_ / NUM_2);
        dstInt8Global_.SetGlobalBuffer((__gm__ int8_t*)x1Out_ + elemStart, blockLength_);
    } else {
        srcInt4Global_.SetGlobalBuffer((__gm__ int8_t*)x2In   + elemStart / NUM_2, blockLength_ / NUM_2);
        dstInt8Global_.SetGlobalBuffer((__gm__ int8_t*)x2Out_ + elemStart, blockLength_);
    }

    // ---- init queues ----
    pipe.InitBuffer(inQueueInt4_,      BUFFER_NUM, static_cast<uint32_t>(ubLength_) / NUM_2);
    pipe.InitBuffer(computeQueueHalf_, BUFFER_NUM, static_cast<uint32_t>(ubLength_) * sizeof(half));
    pipe.InitBuffer(outQueueInt8_,     BUFFER_NUM, static_cast<uint32_t>(ubLength_) * sizeof(int8_t));
}

__aicore__ inline void QbmmInt4ToInt8Preprocess::Process()
{
    if (blockLength_ == 0) return;

    uint64_t loopCount = DequantBmm::CeilDiv(blockLength_, static_cast<uint64_t>(ubLength_));
    for (uint32_t i = 0; i < loopCount; i++) {
        uint64_t remaining  = blockLength_ - ubLength_ * i;
        uint32_t currentNum = DequantBmm::Min(static_cast<uint32_t>(remaining), ubLength_);

        currentNum = DequantBmm::FloorAlign(currentNum, ELEM_ALIGN_64);
        if (currentNum == 0) break;

        CopyIn(i, currentNum);
        Compute(currentNum);
        CopyOut(i, currentNum);
    }
}

__aicore__ inline void QbmmInt4ToInt8Preprocess::CopyIn(uint32_t progress, uint32_t currentNum)
{
    LocalTensor<int8_t> int4Local = inQueueInt4_.AllocTensor<int8_t>();
    DataCopy(int4Local, srcInt4Global_[progress * ubLength_ / NUM_2], static_cast<uint32_t>(currentNum) / NUM_2);
    inQueueInt4_.EnQue<int8_t>(int4Local);
}

__aicore__ inline void QbmmInt4ToInt8Preprocess::Compute(uint32_t currentNum)
{
    LocalTensor<int8_t> int4Local = inQueueInt4_.DeQue<int8_t>();
    LocalTensor<int4b_t> int4View = int4Local.ReinterpretCast<int4b_t>();

    // ---- int4 -> half ----
    LocalTensor<half> halfLocal = computeQueueHalf_.AllocTensor<half>();
    Cast<half, int4b_t>(halfLocal, int4View, RoundMode::CAST_NONE, static_cast<uint32_t>(currentNum));
    inQueueInt4_.FreeTensor(int4Local);
    computeQueueHalf_.EnQue<half>(halfLocal);

    // ---- half -> int8 ----
    LocalTensor<half> halfSrc = computeQueueHalf_.DeQue<half>();
    LocalTensor<int8_t> int8Local = outQueueInt8_.AllocTensor<int8_t>();
    Cast<int8_t, half>(int8Local, halfSrc, RoundMode::CAST_ROUND, static_cast<uint32_t>(currentNum));
    computeQueueHalf_.FreeTensor(halfSrc);
    outQueueInt8_.EnQue<int8_t>(int8Local);
}

__aicore__ inline void QbmmInt4ToInt8Preprocess::CopyOut(uint32_t progress, uint32_t currentNum)
{
    LocalTensor<int8_t> int8Local = outQueueInt8_.DeQue<int8_t>();
    DataCopy(dstInt8Global_[progress * ubLength_], int8Local, static_cast<uint32_t>(currentNum * sizeof(int8_t)));
    outQueueInt8_.FreeTensor(int8Local);
}


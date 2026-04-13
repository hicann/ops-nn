/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file unsorted_segment_sort_simt.h
 * \brief
 */

#ifndef UNSORTED_SEGMENT_SORT_SIMT_H
#define UNSORTED_SEGMENT_SORT_SIMT_H

#include "unsorted_segment_base.h"

namespace UnsortedSegmentSum
{
constexpr int64_t DOUBLE = 2;
constexpr int64_t MAX_INDEX_NUM = 1024;
using namespace AscendC;

template <typename TX, typename Index, typename CAST_T, uint32_t castType>
class KernelUnsortedSegmentSortSimt
{
public:
    __aicore__ inline KernelUnsortedSegmentSortSimt(const UnsortedSegmentSumSortSimtTilingData* tiling, TPipe* pipe)
            : tilingData_(tiling), pipe_(pipe){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR segmentIds, GM_ADDR output);
    __aicore__ inline void Process();
    __aicore__ inline void CopyInX(int64_t baseCoreOffset, int64_t loopIdx, int64_t stride, int64_t length);
    __aicore__ inline void CopyInIndex(int64_t baseCoreOffset, int64_t loopIdx, int64_t stride, int64_t length);
    __aicore__ inline void ProcessEachLoop(uint32_t maxIndexNum);
private:
    TPipe* pipe_;
    const UnsortedSegmentSumSortSimtTilingData* tilingData_;
    TQue<QuePosition::VECIN, DOUBLE> inQueueX_;
    TQue<QuePosition::VECIN, DOUBLE> inQueueIndex_;
    GlobalTensor<TX> xGm_;
    GlobalTensor<TX> outputGm_;
    GlobalTensor<Index> segmentIdsGm_;
    TBuf<TPosition::VECCALC> sortedIndexBuf_;
    TBuf<TPosition::VECCALC> sortedSengmentIdsBuf_;
    TBuf<TPosition::VECCALC> sortTmpBuf_;
    TBuf<TPosition::VECCALC> sortedKeyBuf_;
    TBuf<TPosition::VECCALC> castKeyIdxBuf_;

    uint32_t shiftOffset_ = 0;
};
template <typename TX, typename Index, typename CAST_T, uint32_t castType>
__aicore__ inline void KernelUnsortedSegmentSortSimt<TX, Index, CAST_T, castType>::Init(GM_ADDR x, GM_ADDR segmentIds, GM_ADDR output)
{
    InitGm<TX>(output, tilingData_->outputOuterDim * tilingData_->innerDim);

    xGm_.SetGlobalBuffer((__gm__ TX*)(x));
    segmentIdsGm_.SetGlobalBuffer((__gm__ Index*)(segmentIds));
    outputGm_.SetGlobalBuffer((__gm__ TX*)(output));
    uint32_t ubBlockSize = Ops::Base::GetUbBlockSize();
    shiftOffset_ = ubBlockSize / sizeof(CAST_T);
    uint32_t inputOutputSize = Ops::Base::CeilAlign(
            static_cast<uint32_t>(tilingData_->innerDim * tilingData_->maxIndexNum * sizeof(TX)), ubBlockSize);
    uint32_t alignIndexSize =
            Ops::Base::CeilAlign(static_cast<uint32_t>(tilingData_->maxIndexNum * sizeof(Index)), ubBlockSize);
    uint32_t alignCastIndexSize =
            Ops::Base::CeilAlign(static_cast<uint32_t>(tilingData_->maxIndexNum * sizeof(CAST_T)), ubBlockSize);
    uint32_t sortedIndexSize =
            Ops::Base::CeilAlign(static_cast<uint32_t>(tilingData_->maxIndexNum * sizeof(uint32_t)), ubBlockSize);
    pipe_->InitBuffer(inQueueX_, DOUBLE, inputOutputSize);
    pipe_->InitBuffer(inQueueIndex_, DOUBLE, alignIndexSize);
    pipe_->InitBuffer(sortedIndexBuf_, sortedIndexSize);

    pipe_->InitBuffer(sortTmpBuf_, tilingData_->sortTmpSize);

    if constexpr (castType == CAST_NO) {
        pipe_->InitBuffer(sortedSengmentIdsBuf_, alignIndexSize + DOUBLE * ubBlockSize);
    } else {
        pipe_->InitBuffer(sortedSengmentIdsBuf_, alignCastIndexSize + DOUBLE * ubBlockSize);
        pipe_->InitBuffer(castKeyIdxBuf_, alignCastIndexSize);
    }
    return;
}
template <typename TX, typename Index, typename CAST_T, uint32_t castType>
__aicore__ inline void KernelUnsortedSegmentSortSimt<TX, Index, CAST_T, castType>::CopyInX(
        int64_t baseCoreOffset, int64_t loopIdx, int64_t stride, int64_t length)
{
    LocalTensor<TX> xLocal = inQueueX_.AllocTensor<TX>();
    int64_t offset = baseCoreOffset + loopIdx * stride;
    DataCopyPadExtParams<TX> dataCopyPadExtParams = {false, 0, 0, 0};
    DataCopyExtParams dataCoptExtParams;
    dataCoptExtParams.blockCount = 1;
    dataCoptExtParams.blockLen = length * sizeof(TX);
    dataCoptExtParams.srcStride = 0;
    dataCoptExtParams.dstStride = 0;

    DataCopyPad(xLocal, xGm_[offset], dataCoptExtParams, dataCopyPadExtParams);
    inQueueX_.EnQue(xLocal);
    return;
}
template <typename TX, typename Index, typename CAST_T, uint32_t castType>
__aicore__ inline void KernelUnsortedSegmentSortSimt<TX, Index, CAST_T, castType>::CopyInIndex(
        int64_t baseCoreOffset, int64_t loopIdx, int64_t stride, int64_t length)
{
    LocalTensor<Index> indexLocal = inQueueIndex_.AllocTensor<Index>();
    int64_t offset = baseCoreOffset + loopIdx * stride;
    DataCopyPadExtParams<Index> dataCopyPadExtParams = {false, 0, 0, 0};
    DataCopyExtParams dataCoptExtParams;
    dataCoptExtParams.blockCount = 1;
    dataCoptExtParams.blockLen = length * sizeof(Index);
    dataCoptExtParams.srcStride = 0;
    dataCoptExtParams.dstStride = 0;

    DataCopyPad(indexLocal, segmentIdsGm_[offset], dataCoptExtParams, dataCopyPadExtParams);
    inQueueIndex_.EnQue(indexLocal);
    return;
}

template <typename TX, typename Index, typename CAST_T, uint32_t castType>
__aicore__ inline void KernelUnsortedSegmentSortSimt<TX, Index, CAST_T, castType>::ProcessEachLoop(uint32_t maxIndexNum)
{
    LocalTensor<Index> indexLocal = inQueueIndex_.DeQue<Index>();
    LocalTensor<TX> xLocal = inQueueX_.DeQue<TX>();
    LocalTensor<uint32_t> sortedIndexLocal = sortedIndexBuf_.Get<uint32_t>();
    LocalTensor<CAST_T> sortedSegmentLocal = sortedSengmentIdsBuf_.Get<CAST_T>();

    Duplicate(sortedSegmentLocal, static_cast<CAST_T>(-1), shiftOffset_ * DOUBLE + maxIndexNum);
    static constexpr SortConfig config{SortType::RADIX_SORT, false};
    LocalTensor<CAST_T> dstSortedResult = sortedSegmentLocal[shiftOffset_];

    if constexpr (castType == CAST_NO) {
        LocalTensor<uint8_t> shareTmpBufferLocal = sortTmpBuf_.Get<uint8_t>();
        AscendC::Sort<CAST_T, false, config>(
            dstSortedResult, sortedIndexLocal, indexLocal, shareTmpBufferLocal, maxIndexNum);
    } else {
        LocalTensor<CAST_T> indicesCastLocal = castKeyIdxBuf_.Get<CAST_T>();
        LocalTensor<int32_t> noDupRes = sortTmpBuf_.Get<int32_t>();
        IndicesSortCast<Index, CAST_T, castType>(indexLocal, indicesCastLocal, noDupRes, maxIndexNum);
        LocalTensor<uint8_t> shareTmpBufferLocal = sortTmpBuf_.Get<uint8_t>();
        AscendC::Sort<CAST_T, false, config>(
            dstSortedResult, sortedIndexLocal, indicesCastLocal, shareTmpBufferLocal, maxIndexNum);
    }

    LocalTensor<int32_t> cumSumLocal = sortTmpBuf_.Get<int32_t>();
    int32_t uniqueIndexNum = ComputeUniqueIdNum<CAST_T>(sortedSegmentLocal, cumSumLocal, maxIndexNum);

    uint32_t currentMaxThread =
            (tilingData_->maxThread) >= SORT_THREAD_DIM_LAUNCH_BOUND ? SORT_THREAD_DIM_LAUNCH_BOUND : tilingData_->maxThread;
    int32_t threadBlock = currentMaxThread / tilingData_->innerDim;
    threadBlock =
    threadBlock < uniqueIndexNum ? threadBlock : uniqueIndexNum;
    __ubuf__ uint32_t* sortedIndexAddr = (__ubuf__ uint32_t*)sortedIndexLocal.GetPhyAddr();
    __ubuf__ CAST_T* sortedSengmentAddr = (__ubuf__ CAST_T*)dstSortedResult.GetPhyAddr();
    __ubuf__ TX* inputAddr = (__ubuf__ TX*)xLocal.GetPhyAddr();
    __ubuf__ uint32_t* cumSumAddr = (__ubuf__ uint32_t*)cumSumLocal.GetPhyAddr();
    __gm__ TX* outputGm = (__gm__ TX*)outputGm_.GetPhyAddr();

    AscendC::Simt::VF_CALL<SegmentReduceSortSimt<TX, CAST_T>>(
        Simt::Dim3({static_cast<uint32_t>(tilingData_->innerDim), static_cast<uint32_t>(threadBlock)}), inputAddr,
        sortedIndexAddr, sortedSengmentAddr, cumSumAddr, outputGm, uniqueIndexNum, tilingData_->innerDim,
        tilingData_->outputOuterDim);

    inQueueIndex_.FreeTensor(indexLocal);
    inQueueX_.FreeTensor(xLocal);
    return;
}
template <typename TX, typename Index, typename CAST_T, uint32_t castType>
__aicore__ inline void KernelUnsortedSegmentSortSimt<TX, Index, CAST_T, castType>::Process()
{
    int64_t blockIdx = GetBlockIdx();
    if (blockIdx >= tilingData_->usedCoreNum) {
        return;
    }

    int64_t currLoopTimes =
         (blockIdx == tilingData_->usedCoreNum - 1) ? tilingData_->tailCoreUbLoopTimes : tilingData_->oneCoreUbLoopTimes;
    int64_t baseCoreOffset =
        blockIdx * tilingData_->oneCoreUbLoopTimes * tilingData_->maxIndexNum * tilingData_->innerDim;
    int64_t baseCoreOffsetIndex = blockIdx * tilingData_->oneCoreUbLoopTimes * tilingData_->maxIndexNum;
    int64_t length = tilingData_->maxIndexNum * tilingData_->innerDim;
    uint32_t tailSize =
        (blockIdx == tilingData_->usedCoreNum - 1) ? tilingData_->tailIndexNum : tilingData_->maxIndexNum;
    for (int64_t i = 0; i < currLoopTimes - 1; i++) {
        CopyInX(baseCoreOffset, i, length, length);
        CopyInIndex(baseCoreOffsetIndex, i, tilingData_->maxIndexNum, tilingData_->maxIndexNum);
        ProcessEachLoop(static_cast<uint32_t>(tilingData_->maxIndexNum));
    }
    CopyInX(baseCoreOffset, currLoopTimes - 1, length, tailSize * tilingData_->innerDim);
    CopyInIndex(baseCoreOffsetIndex, currLoopTimes - 1, tilingData_->maxIndexNum, tailSize);
    ProcessEachLoop(tailSize);
    return;
}
}
#endif
/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef UNSORTED_SEGMENT_SORT_SIMT_H
#define UNSORTED_SEGMENT_SORT_SIMT_H

#include "unsorted_segment_base.h"

namespace UnsortedSegment
{
constexpr int64_t DOUBLE = 2;
constexpr int64_t MAX_INDEX_NUM = 1024;
using namespace AscendC;

template <typename TX, typename Index, uint8_t Mode>
class KernelUnsortedSegmentSortSimt
{
public:
    __aicore__ inline KernelUnsortedSegmentSortSimt(const UnsortedSegmentSortSimtTilingData* tiling, TPipe* pipe)
        : tilingData_(tiling), pipe_(pipe){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR segmentIds, GM_ADDR output);
    __aicore__ inline void Process();
    __aicore__ inline void CopyInX(int64_t baseCoreOffset, int64_t loopIdx, int64_t stride, int64_t length);
    __aicore__ inline void CopyInIndex(int64_t baseCoreOffset, int64_t loopIdx, int64_t stride, int64_t length);
    __aicore__ inline int32_t GetUniqueCount(
        LocalTensor<uint32_t>& cumSumLocal, LocalTensor<Index>& sortedSegmentLocal);
    __aicore__ inline void ProcessEachLoop(uint32_t maxIndexNum);
private:
    TPipe* pipe_;
    const UnsortedSegmentSortSimtTilingData* tilingData_;
    TQue<QuePosition::VECIN, DOUBLE> inQueueX_;
    TQue<QuePosition::VECIN, DOUBLE> inQueueIndex_;
    GlobalTensor<TX> xGm_;
    GlobalTensor<TX> outputGm_;
    GlobalTensor<Index> segmentIdsGm_;
    TBuf<TPosition::VECCALC> sortedIndexBuf_;
    TBuf<TPosition::VECCALC> sortedSengmentIdsBuf_;
    TBuf<TPosition::VECCALC> sortTmpBuf_;
    uint32_t shiftOffset_ = 0;
};
template <typename TX, typename Index, uint8_t Mode>
__aicore__ inline void KernelUnsortedSegmentSortSimt<TX, Index, Mode>::Init(GM_ADDR x, GM_ADDR segmentIds, GM_ADDR output)
{
    InitGm<TX, Mode>(output, tilingData_->outputOuterDim * tilingData_->innerDim);

    xGm_.SetGlobalBuffer((__gm__ TX*)(x));
    segmentIdsGm_.SetGlobalBuffer((__gm__ Index*)(segmentIds));
    outputGm_.SetGlobalBuffer((__gm__ TX*)(output));
    uint32_t ubBlockSize = ONE_BLOCK_SIZE;
    shiftOffset_ = ubBlockSize / sizeof(Index);
    uint32_t inputOutputSize = ops::CeilAlign(
        static_cast<uint32_t>(tilingData_->innerDim * tilingData_->maxIndexNum * sizeof(TX)), ubBlockSize);
    uint32_t alignIndexSize =
        ops::CeilAlign(static_cast<uint32_t>(tilingData_->maxIndexNum * sizeof(Index)), ubBlockSize);
    uint32_t sortedIndexSize =
        ops::CeilAlign(static_cast<uint32_t>(tilingData_->maxIndexNum * sizeof(uint32_t)), ubBlockSize);
    pipe_->InitBuffer(inQueueX_, DOUBLE, inputOutputSize);
    pipe_->InitBuffer(inQueueIndex_, DOUBLE, alignIndexSize);
    pipe_->InitBuffer(sortedIndexBuf_, sortedIndexSize);
    pipe_->InitBuffer(sortedSengmentIdsBuf_, alignIndexSize + DOUBLE * ubBlockSize);
    pipe_->InitBuffer(sortTmpBuf_, tilingData_->sortTmpSize);
    return;
}
template <typename TX, typename Index, uint8_t Mode>
__aicore__ inline void KernelUnsortedSegmentSortSimt<TX, Index, Mode>::CopyInX(
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
template <typename TX, typename Index, uint8_t Mode>
__aicore__ inline void KernelUnsortedSegmentSortSimt<TX, Index, Mode>::CopyInIndex(
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
template <typename TX, typename Index, uint8_t Mode>
__aicore__ inline int32_t KernelUnsortedSegmentSortSimt<TX, Index, Mode>::GetUniqueCount(
    LocalTensor<uint32_t>& cumSumLocal, LocalTensor<Index>& sortedSegmentLocal)
{
    __ubuf__ Index* sortedSengmentAddr = (__ubuf__ Index*)sortedSegmentLocal.GetPhyAddr();
    __ubuf__ int32_t* cumSumAddr = (__ubuf__ int32_t*)cumSumLocal.GetPhyAddr();
    uint32_t vl = platform::GetVRegSize() / sizeof(Index);
    uint16_t loopCnt = (uint16_t)(ops::CeilDiv(static_cast<uint32_t>(tilingData_->maxIndexNum + 1), vl));
    uint32_t maskCount = tilingData_->maxIndexNum + 1;
    uint32_t offset = shiftOffset_;
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<int32_t> orderReg;
        AscendC::MicroAPI::RegTensor<int32_t> selReg;
        AscendC::MicroAPI::RegTensor<Index> indicesReg;
        AscendC::MicroAPI::RegTensor<Index> indicesShiftOneReg;
        AscendC::MicroAPI::MaskReg cmpMask;
        AscendC::MicroAPI::MaskReg maskRegUpdate;
        AscendC::MicroAPI::UnalignReg u0;
        MicroAPI::UnalignReg ureg;
        AscendC::MicroAPI::ClearSpr<AscendC::SpecialPurposeReg::AR>();
        int32_t vciStart = 0;
        for (uint16_t i = 0; i < loopCnt; ++i) {
            vciStart = i * vl;
            auto sortedIndicesAddrUpdate = sortedSengmentAddr + offset + i * vl;
            AscendC::MicroAPI::Arange(orderReg, vciStart);
            maskRegUpdate = AscendC::MicroAPI::UpdateMask<Index>(maskCount);
            AscendC::MicroAPI::DataCopy(indicesReg, sortedIndicesAddrUpdate);
            AscendC::MicroAPI::DataCopyUnAlignPre(u0, sortedIndicesAddrUpdate - 1);
            AscendC::MicroAPI::DataCopyUnAlign<Index>(indicesShiftOneReg, u0, sortedIndicesAddrUpdate - 1);
            AscendC::MicroAPI::Compare<Index, CMPMODE::NE>(cmpMask, indicesReg, indicesShiftOneReg, maskRegUpdate);

            if constexpr (IsSameType<Index, int64_t>::value) {
                AscendC::MicroAPI::MaskReg maskHalf;
                AscendC::MicroAPI::MaskPack<AscendC::MicroAPI::HighLowPart::LOWEST>(maskHalf, cmpMask);
                // vSQZ
                AscendC::MicroAPI::GatherMask<int32_t, AscendC::MicroAPI::GatherMaskMode::STORE_REG>(
                    selReg, orderReg, maskHalf);
            } else {
                // vSQZ
                AscendC::MicroAPI::GatherMask<int32_t, AscendC::MicroAPI::GatherMaskMode::STORE_REG>(
                    selReg, orderReg, cmpMask);
            }
            AscendC::MicroAPI::DataCopyUnAlign<int32_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                cumSumAddr, selReg, ureg);
        }
        AscendC::MicroAPI::DataCopyUnAlignPost(cumSumAddr, ureg);
    }
    return ((AscendC::MicroAPI::GetSpr<AscendC::SpecialPurposeReg::AR>()) / sizeof(int32_t)) - 1;
}
template <typename TX, typename Index, uint8_t Mode>
__aicore__ inline void KernelUnsortedSegmentSortSimt<TX, Index, Mode>::ProcessEachLoop(uint32_t maxIndexNum)
{
    LocalTensor<Index> indexLocal = inQueueIndex_.DeQue<Index>();
    LocalTensor<TX> xLocal = inQueueX_.DeQue<TX>();
    LocalTensor<uint32_t> sortedIndexLocal = sortedIndexBuf_.Get<uint32_t>();
    LocalTensor<Index> sortedSegmentLocal = sortedSengmentIdsBuf_.Get<Index>();
    LocalTensor<uint8_t> shareTmpBufferLocal = sortTmpBuf_.Get<uint8_t>();
    LocalTensor<Index> dstSortedResult = sortedSegmentLocal[shiftOffset_];

    Duplicate(sortedSegmentLocal, static_cast<Index>(-1), shiftOffset_ * DOUBLE + tilingData_->maxIndexNum);
    static constexpr SortConfig config{SortType::RADIX_SORT, false};
    AscendC::Sort<Index, false, config>(
        dstSortedResult, sortedIndexLocal, indexLocal, shareTmpBufferLocal, maxIndexNum);
    LocalTensor<uint32_t> cumSumLocal = sortTmpBuf_.Get<uint32_t>();
    int32_t uniqueIndexNum = GetUniqueCount(cumSumLocal, sortedSegmentLocal);

    uint32_t currentMaxThread = 
        (tilingData_->maxThread) >= SORT_THREAD_DIM_LAUNCH_BOUND ? SORT_THREAD_DIM_LAUNCH_BOUND : tilingData_->maxThread;
    int32_t threadBlock = currentMaxThread / tilingData_->innerDim;
    threadBlock = 
        threadBlock < uniqueIndexNum ? threadBlock : uniqueIndexNum;
    __ubuf__ uint32_t* sortedIndexAddr = (__ubuf__ uint32_t*)sortedIndexLocal.GetPhyAddr();
    __ubuf__ Index* sortedSengmentAddr = (__ubuf__ Index*)dstSortedResult.GetPhyAddr();
    __ubuf__ TX* inputAddr = (__ubuf__ TX*)xLocal.GetPhyAddr();
    __ubuf__ uint32_t* cumSumAddr = (__ubuf__ uint32_t*)cumSumLocal.GetPhyAddr();
    __gm__ TX* outputGm = (__gm__ TX*)outputGm_.GetPhyAddr();

    AscendC::Simt::VF_CALL<SegmentReduceSortSimt<TX, Index, Mode>>(
        Simt::Dim3({static_cast<uint32_t>(tilingData_->innerDim), static_cast<uint32_t>(threadBlock)}), inputAddr,
        sortedIndexAddr, sortedSengmentAddr, cumSumAddr, outputGm, uniqueIndexNum, tilingData_->innerDim,
        tilingData_->outputOuterDim);

    inQueueIndex_.FreeTensor(indexLocal);
    inQueueX_.FreeTensor(xLocal);
    return;
}
template <typename TX, typename Index, uint8_t Mode>
__aicore__ inline void KernelUnsortedSegmentSortSimt<TX, Index, Mode>::Process()
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
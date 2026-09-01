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

namespace UnsortedSegment {
constexpr int64_t DOUBLE = 2;
constexpr int64_t MAX_INDEX_NUM = 1024;
constexpr uint32_t ONE_BYTE_BIT_NUM = 8;
using namespace AscendC;

template <typename TX, typename Index, typename SimtGatherFunc, typename SimtAtomicFunc, typename InitValueType,
          typename GmInitFunc>
class KernelUnsortedSegmentSortSimt {
public:
    using KEY_T = typename std::conditional<IsSameType<Index, int64_t>::value, int32_t, Index>::type;
    static constexpr bool CAN_NARROW = !IsSameType<KEY_T, Index>::value;

    __aicore__ inline KernelUnsortedSegmentSortSimt(const UnsortedSegmentSortSimtTilingData* tiling, TPipe* pipe)
        : tilingData_(tiling), pipe_(pipe){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR segmentIds, GM_ADDR output);
    __aicore__ inline void Process();
    __aicore__ inline void CopyInX(int64_t baseCoreOffset, int64_t loopIdx, int64_t stride, int64_t length);
    __aicore__ inline void CopyInIndex(int64_t baseCoreOffset, int64_t loopIdx, int64_t stride, int64_t length);
    template <typename KT>
    __aicore__ inline int32_t GetUniqueCount(LocalTensor<uint32_t>& cumSumLocal, LocalTensor<KT>& sortedSegmentLocal);
    __aicore__ inline void ProcessEachLoop(uint32_t maxIndexNum);
    __aicore__ inline void NarrowSortKey(LocalTensor<Index>& idsLocal, LocalTensor<KEY_T>& keyLocal, uint32_t count);
    template <typename KT>
    static __simd_vf__ inline void NarrowSortKeyVf(__ubuf__ KT* idsAddr, uint32_t vl, uint16_t loopCnt, uint32_t count,
                                                   KT upperBound);
    template <typename KT>
    static __simd_vf__ inline void GetUniqueCountVf(__ubuf__ KT* sortedSengmentAddr, __ubuf__ int32_t* cumSumAddr,
                                                    uint32_t vl, uint16_t loopCnt, uint32_t maskCount, uint32_t offset);

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
    TBuf<TPosition::VECCALC> narrowKeyBuf_;
    TBuf<TPosition::VECCALC> maskBuf_;
    uint32_t shiftOffset_ = 0;
    bool narrow_ = false;
};
template <typename TX, typename Index, typename SimtGatherFunc, typename SimtAtomicFunc, typename InitValueType,
          typename GmInitFunc>
__aicore__ inline void KernelUnsortedSegmentSortSimt<TX, Index, SimtGatherFunc, SimtAtomicFunc, InitValueType,
                                                     GmInitFunc>::Init(GM_ADDR x, GM_ADDR segmentIds, GM_ADDR output)
{
    InitGm<TX, GmInitFunc>(output, tilingData_->outputOuterDim * tilingData_->innerDim);

    xGm_.SetGlobalBuffer((__gm__ TX*)(x));
    segmentIdsGm_.SetGlobalBuffer((__gm__ Index*)(segmentIds));
    outputGm_.SetGlobalBuffer((__gm__ TX*)(output));
    uint32_t ubBlockSize = ONE_BLOCK_SIZE;
    narrow_ = CAN_NARROW && (tilingData_->narrowSortKey == 1UL);
    uint32_t sortKeyBytes = narrow_ ? static_cast<uint32_t>(sizeof(KEY_T)) : static_cast<uint32_t>(sizeof(Index));
    shiftOffset_ = ubBlockSize / sortKeyBytes;
    uint32_t inputOutputSize = ops::CeilAlign(
        static_cast<uint32_t>(tilingData_->innerDim * tilingData_->maxIndexNum * sizeof(TX)), ubBlockSize);
    uint32_t alignIndexSize = ops::CeilAlign(static_cast<uint32_t>(tilingData_->maxIndexNum * sizeof(Index)),
                                             ubBlockSize);
    uint32_t alignKeySize = ops::CeilAlign(static_cast<uint32_t>(tilingData_->maxIndexNum * sortKeyBytes), ubBlockSize);
    uint32_t sortedIndexSize = ops::CeilAlign(static_cast<uint32_t>(tilingData_->maxIndexNum * sizeof(uint32_t)),
                                              ubBlockSize);
    pipe_->InitBuffer(inQueueX_, DOUBLE, inputOutputSize);
    pipe_->InitBuffer(inQueueIndex_, DOUBLE, alignIndexSize);
    pipe_->InitBuffer(sortedIndexBuf_, sortedIndexSize);
    pipe_->InitBuffer(sortedSengmentIdsBuf_, alignKeySize + DOUBLE * ubBlockSize);
    pipe_->InitBuffer(sortTmpBuf_, tilingData_->sortTmpSize);
    if (narrow_) {
        pipe_->InitBuffer(narrowKeyBuf_, alignKeySize);
        pipe_->InitBuffer(
            maskBuf_,
            ops::CeilAlign(static_cast<uint32_t>(tilingData_->maxIndexNum / ONE_BYTE_BIT_NUM + 1), ubBlockSize) +
                ubBlockSize);
    }
    return;
}
template <typename TX, typename Index, typename SimtGatherFunc, typename SimtAtomicFunc, typename InitValueType,
          typename GmInitFunc>
__aicore__ inline void KernelUnsortedSegmentSortSimt<TX, Index, SimtGatherFunc, SimtAtomicFunc, InitValueType,
                                                     GmInitFunc>::CopyInX(int64_t baseCoreOffset, int64_t loopIdx,
                                                                          int64_t stride, int64_t length)
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
template <typename TX, typename Index, typename SimtGatherFunc, typename SimtAtomicFunc, typename InitValueType,
          typename GmInitFunc>
__aicore__ inline void KernelUnsortedSegmentSortSimt<TX, Index, SimtGatherFunc, SimtAtomicFunc, InitValueType,
                                                     GmInitFunc>::CopyInIndex(int64_t baseCoreOffset, int64_t loopIdx,
                                                                              int64_t stride, int64_t length)
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
template <typename TX, typename Index, typename SimtGatherFunc, typename SimtAtomicFunc, typename InitValueType,
          typename GmInitFunc>
template <typename KT>
__simd_vf__ inline void KernelUnsortedSegmentSortSimt<TX, Index, SimtGatherFunc, SimtAtomicFunc, InitValueType,
                                                      GmInitFunc>::GetUniqueCountVf(__ubuf__ KT* sortedSengmentAddr,
                                                                                    __ubuf__ int32_t* cumSumAddr,
                                                                                    uint32_t vl, uint16_t loopCnt,
                                                                                    uint32_t maskCount, uint32_t offset)
{
    AscendC::Reg::RegTensor<int32_t> orderReg;
    AscendC::Reg::RegTensor<int32_t> selReg;
    AscendC::Reg::RegTensor<KT> indicesReg;
    AscendC::Reg::RegTensor<KT> indicesShiftOneReg;
    AscendC::Reg::MaskReg cmpMask;
    AscendC::Reg::MaskReg maskRegUpdate;
    AscendC::Reg::UnalignRegForLoad u0;
    Reg::UnalignRegForStore ureg;
    AscendC::Reg::ClearSpr<AscendC::SpecialPurposeReg::AR>();
    int32_t vciStart = 0;
    for (uint16_t i = 0; i < loopCnt; ++i) {
        vciStart = i * vl;
        auto sortedIndicesAddrUpdate = sortedSengmentAddr + offset + i * vl;
        AscendC::Reg::Arange(orderReg, vciStart);
        maskRegUpdate = AscendC::Reg::UpdateMask<KT>(maskCount);
        AscendC::Reg::LoadAlign(indicesReg, sortedIndicesAddrUpdate);
        AscendC::Reg::LoadUnAlignPre(u0, sortedIndicesAddrUpdate - 1);
        AscendC::Reg::LoadUnAlign<KT>(indicesShiftOneReg, u0, sortedIndicesAddrUpdate - 1);
        AscendC::Reg::Compare<KT, CMPMODE::NE>(cmpMask, indicesReg, indicesShiftOneReg, maskRegUpdate);

        if constexpr (IsSameType<KT, int64_t>::value) {
            AscendC::Reg::MaskReg maskHalf;
            AscendC::Reg::Pack<AscendC::Reg::HighLowPart::LOWEST>(maskHalf, cmpMask);
            AscendC::Reg::Squeeze<int32_t, AscendC::Reg::GatherMaskMode::STORE_REG>(selReg, orderReg, maskHalf);
        } else {
            AscendC::Reg::Squeeze<int32_t, AscendC::Reg::GatherMaskMode::STORE_REG>(selReg, orderReg, cmpMask);
        }
        AscendC::Reg::StoreUnAlign<int32_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(cumSumAddr, selReg, ureg);
    }
    AscendC::Reg::StoreUnAlignPost(cumSumAddr, ureg);
}

template <typename TX, typename Index, typename SimtGatherFunc, typename SimtAtomicFunc, typename InitValueType,
          typename GmInitFunc>
template <typename KT>
__simd_vf__ inline void KernelUnsortedSegmentSortSimt<TX, Index, SimtGatherFunc, SimtAtomicFunc, InitValueType,
                                                      GmInitFunc>::NarrowSortKeyVf(__ubuf__ KT* idsAddr, uint32_t vl,
                                                                                   uint16_t loopCnt, uint32_t count,
                                                                                   KT upperBound)
{
    AscendC::Reg::RegTensor<KT> idsReg;
    AscendC::Reg::RegTensor<KT> zeroReg;
    AscendC::Reg::RegTensor<KT> upperReg;
    AscendC::Reg::RegTensor<KT> invalidReg;
    uint32_t remain = count;
    for (uint16_t i = 0; i < loopCnt; ++i) {
        AscendC::Reg::MaskReg active = AscendC::Reg::UpdateMask<KT>(remain);
        AscendC::Reg::LoadAlign(idsReg, idsAddr + i * vl);
        AscendC::Reg::Duplicate(zeroReg, static_cast<KT>(0), active);
        AscendC::Reg::Duplicate(upperReg, upperBound, active);
        AscendC::Reg::Duplicate(invalidReg, static_cast<KT>(-1), active);
        AscendC::Reg::MaskReg geZero;
        AscendC::Reg::MaskReg ltUpper;
        AscendC::Reg::MaskReg valid;
        AscendC::Reg::Compare<KT, CMPMODE::GE>(geZero, idsReg, zeroReg, active);
        AscendC::Reg::Compare<KT, CMPMODE::LT>(ltUpper, idsReg, upperReg, active);
        AscendC::Reg::MaskAnd(valid, geZero, ltUpper, active);
        AscendC::Reg::Select(idsReg, idsReg, invalidReg, valid);
        AscendC::Reg::StoreAlign(idsAddr + i * vl, idsReg, active);
        remain = (remain > vl) ? (remain - vl) : 0U;
    }
}

template <typename TX, typename Index, typename SimtGatherFunc, typename SimtAtomicFunc, typename InitValueType,
          typename GmInitFunc>
template <typename KT>
__aicore__ inline int32_t KernelUnsortedSegmentSortSimt<TX, Index, SimtGatherFunc, SimtAtomicFunc, InitValueType,
                                                        GmInitFunc>::GetUniqueCount(LocalTensor<uint32_t>& cumSumLocal,
                                                                                    LocalTensor<KT>& sortedSegmentLocal)
{
    __ubuf__ KT* sortedSengmentAddr = (__ubuf__ KT*)sortedSegmentLocal.GetPhyAddr();
    __ubuf__ int32_t* cumSumAddr = (__ubuf__ int32_t*)cumSumLocal.GetPhyAddr();
    uint32_t vl = platform::GetVRegSize() / sizeof(KT);
    uint16_t loopCnt = (uint16_t)(ops::CeilDiv(static_cast<uint32_t>(tilingData_->maxIndexNum + 1), vl));
    uint32_t maskCount = tilingData_->maxIndexNum + 1;
    uint32_t offset = shiftOffset_;
    GetUniqueCountVf<KT>(sortedSengmentAddr, cumSumAddr, vl, loopCnt, maskCount, offset);
    return ((AscendC::Reg::GetSpr<AscendC::SpecialPurposeReg::AR>()) / sizeof(int32_t)) - 1;
}
template <typename TX, typename Index, typename SimtGatherFunc, typename SimtAtomicFunc, typename InitValueType,
          typename GmInitFunc>
__aicore__ inline void KernelUnsortedSegmentSortSimt<TX, Index, SimtGatherFunc, SimtAtomicFunc, InitValueType,
                                                     GmInitFunc>::NarrowSortKey(LocalTensor<Index>& idsLocal,
                                                                                LocalTensor<KEY_T>& keyLocal,
                                                                                uint32_t count)
{
    __ubuf__ Index* idsAddr = (__ubuf__ Index*)idsLocal.GetPhyAddr();
    uint32_t vl = platform::GetVRegSize() / sizeof(Index);
    uint16_t loopCnt = static_cast<uint16_t>((count + vl - 1U) / vl);
    NarrowSortKeyVf<Index>(idsAddr, vl, loopCnt, count, static_cast<Index>(tilingData_->outputOuterDim));
    Cast<KEY_T, Index>(keyLocal, idsLocal, RoundMode::CAST_NONE, count);
}

template <typename TX, typename Index, typename SimtGatherFunc, typename SimtAtomicFunc, typename InitValueType,
          typename GmInitFunc>
__aicore__ inline void KernelUnsortedSegmentSortSimt<TX, Index, SimtGatherFunc, SimtAtomicFunc, InitValueType,
                                                     GmInitFunc>::ProcessEachLoop(uint32_t maxIndexNum)
{
    LocalTensor<Index> indexLocal = inQueueIndex_.DeQue<Index>();
    LocalTensor<TX> xLocal = inQueueX_.DeQue<TX>();
    LocalTensor<uint32_t> sortedIndexLocal = sortedIndexBuf_.Get<uint32_t>();
    LocalTensor<uint8_t> shareTmpBufferLocal = sortTmpBuf_.Get<uint8_t>();
    static constexpr SortConfig config{SortType::RADIX_SORT, false};
    int32_t uniqueIndexNum = 0;
    uint32_t currentMaxThread = (tilingData_->maxThread) >= SORT_THREAD_DIM_LAUNCH_BOUND ?
                                    SORT_THREAD_DIM_LAUNCH_BOUND :
                                    tilingData_->maxThread;
    int32_t threadBlock = 0;
    __ubuf__ uint32_t* sortedIndexAddr = (__ubuf__ uint32_t*)sortedIndexLocal.GetPhyAddr();
    __ubuf__ TX* inputAddr = (__ubuf__ TX*)xLocal.GetPhyAddr();
    __gm__ TX* outputGm = (__gm__ TX*)outputGm_.GetPhyAddr();

    if constexpr (CAN_NARROW) {
        if (narrow_) {
            LocalTensor<KEY_T> narrowKeyLocal = narrowKeyBuf_.Get<KEY_T>();
            LocalTensor<KEY_T> sortedSegmentLocal = sortedSengmentIdsBuf_.Get<KEY_T>();
            LocalTensor<KEY_T> dstSortedResult = sortedSegmentLocal[shiftOffset_];

            Duplicate(sortedSegmentLocal, static_cast<KEY_T>(-1), shiftOffset_ * DOUBLE + tilingData_->maxIndexNum);
            NarrowSortKey(indexLocal, narrowKeyLocal, maxIndexNum);
            PipeBarrier<PIPE_V>();
            AscendC::Sort<KEY_T, false, config>(dstSortedResult, sortedIndexLocal, narrowKeyLocal, shareTmpBufferLocal,
                                                maxIndexNum);
            LocalTensor<uint32_t> cumSumLocal = sortTmpBuf_.Get<uint32_t>();
            uniqueIndexNum = GetUniqueCount<KEY_T>(cumSumLocal, sortedSegmentLocal);
            if (uniqueIndexNum <= 0) {
                inQueueIndex_.FreeTensor(indexLocal);
                inQueueX_.FreeTensor(xLocal);
                return;
            }
            threadBlock = currentMaxThread / tilingData_->innerDim;
            threadBlock = threadBlock < uniqueIndexNum ? threadBlock : uniqueIndexNum;
            asc_vf_call<SegmentReduceSortSimt<TX, KEY_T, SimtGatherFunc, SimtAtomicFunc, InitValueType>>(
                dim3({static_cast<uint32_t>(tilingData_->innerDim), static_cast<uint32_t>(threadBlock)}), inputAddr,
                sortedIndexAddr, (__ubuf__ KEY_T*)dstSortedResult.GetPhyAddr(),
                (__ubuf__ uint32_t*)cumSumLocal.GetPhyAddr(), outputGm, uniqueIndexNum, tilingData_->innerDim,
                tilingData_->outputOuterDim);
            inQueueIndex_.FreeTensor(indexLocal);
            inQueueX_.FreeTensor(xLocal);
            return;
        }
    }

    LocalTensor<Index> sortedSegmentLocal = sortedSengmentIdsBuf_.Get<Index>();
    LocalTensor<Index> dstSortedResult = sortedSegmentLocal[shiftOffset_];

    Duplicate(sortedSegmentLocal, static_cast<Index>(-1), shiftOffset_ * DOUBLE + tilingData_->maxIndexNum);
    AscendC::Sort<Index, false, config>(dstSortedResult, sortedIndexLocal, indexLocal, shareTmpBufferLocal,
                                        maxIndexNum);
    LocalTensor<uint32_t> cumSumLocal = sortTmpBuf_.Get<uint32_t>();
    uniqueIndexNum = GetUniqueCount<Index>(cumSumLocal, sortedSegmentLocal);

    threadBlock = currentMaxThread / tilingData_->innerDim;
    threadBlock = threadBlock < uniqueIndexNum ? threadBlock : uniqueIndexNum;
    __ubuf__ Index* sortedSengmentAddr = (__ubuf__ Index*)dstSortedResult.GetPhyAddr();
    __ubuf__ uint32_t* cumSumAddr = (__ubuf__ uint32_t*)cumSumLocal.GetPhyAddr();

    asc_vf_call<SegmentReduceSortSimt<TX, Index, SimtGatherFunc, SimtAtomicFunc, InitValueType>>(
        dim3({static_cast<uint32_t>(tilingData_->innerDim), static_cast<uint32_t>(threadBlock)}), inputAddr,
        sortedIndexAddr, sortedSengmentAddr, cumSumAddr, outputGm, uniqueIndexNum, tilingData_->innerDim,
        tilingData_->outputOuterDim);

    inQueueIndex_.FreeTensor(indexLocal);
    inQueueX_.FreeTensor(xLocal);
    return;
}
template <typename TX, typename Index, typename SimtGatherFunc, typename SimtAtomicFunc, typename InitValueType,
          typename GmInitFunc>
__aicore__ inline void
KernelUnsortedSegmentSortSimt<TX, Index, SimtGatherFunc, SimtAtomicFunc, InitValueType, GmInitFunc>::Process()
{
    int64_t block_idx = GetBlockIdx();
    if (block_idx >= tilingData_->usedCoreNum) {
        return;
    }

    int64_t currLoopTimes = (block_idx == tilingData_->usedCoreNum - 1) ? tilingData_->tailCoreUbLoopTimes :
                                                                          tilingData_->oneCoreUbLoopTimes;
    int64_t baseCoreOffset = block_idx * tilingData_->oneCoreUbLoopTimes * tilingData_->maxIndexNum *
                             tilingData_->innerDim;
    int64_t baseCoreOffsetIndex = block_idx * tilingData_->oneCoreUbLoopTimes * tilingData_->maxIndexNum;
    int64_t length = tilingData_->maxIndexNum * tilingData_->innerDim;
    uint32_t tailSize = (block_idx == tilingData_->usedCoreNum - 1) ? tilingData_->tailIndexNum :
                                                                      tilingData_->maxIndexNum;
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
} // namespace UnsortedSegment
#endif

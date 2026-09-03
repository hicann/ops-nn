/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* Internal continuation of bn_training_reduce.h. Not a standalone header. */
__aicore__ inline void GetCoreALoopRange(int64_t blockIdx, int64_t& start, int64_t& end) const
{
    if (blockIdx < static_cast<int64_t>(aBigCoreCnt_)) {
        start = blockIdx * aBigCoreLoopCnt_;
        end = start + aBigCoreLoopCnt_;
    } else {
        start = static_cast<int64_t>(aBigCoreCnt_) * aBigCoreLoopCnt_ +
                (blockIdx - static_cast<int64_t>(aBigCoreCnt_)) * aSmallCoreLoopCnt_;
        end = start + aSmallCoreLoopCnt_;
    }
}

__aicore__ inline void UnravelALoop(int64_t aLoopIdx, int64_t& aLen, int64_t& inputBaseOff, int64_t& outputOff) const
{
    int64_t rem = aLoopIdx;
    const int64_t aChunkIdx = rem % aSplitChunkCnt_;
    rem /= aSplitChunkCnt_;
    inputBaseOff = 0;
    outputOff = 0;
    for (int32_t i = aSplit_ - 2; i >= 0; i -= 2) {
        const int64_t coord = rem % axisShape_[i];
        rem /= axisShape_[i];
        inputBaseOff += coord * axisStride_[i];
        outputOff += coord * outStride_[i];
    }
    const int64_t aStart = aChunkIdx * aUbFactor_;
    const int64_t remain = axisShape_[aSplit_] - aStart;
    aLen = (remain < aUbFactor_) ? remain : aUbFactor_;
    inputBaseOff += aStart * axisStride_[aSplit_];
    outputOff += aStart * outStride_[aSplit_];
}

__aicore__ inline int64_t UnravelR(int64_t rLoopIdx, int64_t& rLen) const
{
    const int64_t rChunkCount = (axisShape_[rSplit_] + rUbFactor_ - 1) / rUbFactor_;
    const int64_t rChunkIdx = rLoopIdx % rChunkCount;
    int64_t rem = rLoopIdx / rChunkCount;
    int64_t inputOff = 0;
    for (int32_t i = rSplit_ - 2; i >= 1; i -= 2) {
        const int64_t coord = rem % axisShape_[i];
        rem /= axisShape_[i];
        inputOff += coord * axisStride_[i];
    }
    const int64_t rStart = rChunkIdx * rUbFactor_;
    const int64_t remain = axisShape_[rSplit_] - rStart;
    rLen = (remain < rUbFactor_) ? remain : rUbFactor_;
    return inputOff + rStart * axisStride_[rSplit_];
}

__aicore__ inline int32_t BuildUBAxes(int64_t aLen, int64_t rLen, UBAxisDesc out[]) const
{
    int32_t count = 0;
    const int32_t lastA = LastAAxis();
    const int32_t lastR = LastRAxis();
    const int64_t blockElems = kBlockBytes / static_cast<int64_t>(sizeof(D_T));
    if constexpr (isTailR) {
        for (int32_t i = axisNum_ - 1; i >= rSplit_; --i) {
            if ((i & 1) == 0) {
                continue;
            }
            const int64_t actual = (i == rSplit_) ? rLen : axisShape_[i];
            int64_t padded = actual;
            if (i == rSplit_) {
                padded = rUbFactorAlign_;
            } else if (i == lastR) {
                padded = (actual + blockElems - 1) / blockElems * blockElems;
            }
            out[count++] = UBAxisDesc{i, actual, padded, axisStride_[i]};
        }
        for (int32_t i = axisNum_ - 1; i >= aSplit_; --i) {
            if ((i & 1) != 0) {
                continue;
            }
            const int64_t actual = (i == aSplit_) ? aLen : axisShape_[i];
            const int64_t padded = (i == aSplit_) ? aUbFactor_ : actual;
            out[count++] = UBAxisDesc{i, actual, padded, axisStride_[i]};
        }
    } else {
        for (int32_t i = axisNum_ - 1; i >= aSplit_; --i) {
            if ((i & 1) != 0) {
                continue;
            }
            const int64_t actual = (i == aSplit_) ? aLen : axisShape_[i];
            int64_t padded = actual;
            if (i == aSplit_) {
                padded = (i == lastA) ? aUbFactorAlign_ : aUbFactor_;
            } else if (i == lastA) {
                padded = (actual + blockElems - 1) / blockElems * blockElems;
            }
            out[count++] = UBAxisDesc{i, actual, padded, axisStride_[i]};
        }
        for (int32_t i = axisNum_ - 1; i >= rSplit_; --i) {
            if ((i & 1) == 0) {
                continue;
            }
            const int64_t actual = (i == rSplit_) ? rLen : axisShape_[i];
            const int64_t padded = (i == rSplit_) ? rUbFactorAlign_ : actual;
            out[count++] = UBAxisDesc{i, actual, padded, axisStride_[i]};
        }
    }
    return count;
}

__aicore__ inline void DoCopyInTile(int64_t baseGmOff, int64_t aLen, int64_t rLen, LocalTensor<D_T>& preInLocal)
{
    UBAxisDesc ubAxes[MAX_PATTERN_RANK];
    const int32_t axisCount = BuildUBAxes(aLen, rLen, ubAxes);
    const int64_t typeBytes = static_cast<int64_t>(sizeof(D_T));

    DataCopyExtParams ext = {};
    ext.blockLen = static_cast<uint32_t>(ubAxes[0].ubSize * typeBytes);
    ext.rsv = 0;
    const uint32_t misalign = ext.blockLen & (kBlockBytes - 1U);
    const uint32_t gapBytes = (misalign == 0U) ? 0U : kBlockBytes - misalign;
    const uint8_t rightPad = static_cast<uint8_t>(gapBytes / sizeof(D_T));
    DataCopyPadExtParams<D_T> pad = {true, 0, rightPad, static_cast<D_T>(0)};

    const int64_t copyPadBytes = (static_cast<int64_t>(ext.blockLen) + kBlockBytes - 1) / kBlockBytes * kBlockBytes;
    const int64_t targetRowBytes = ubAxes[0].paddedSize * typeBytes;
    ext.dstStride = (targetRowBytes - copyPadBytes) / kBlockBytes;
    if (axisCount >= 2) {
        ext.blockCount = static_cast<uint16_t>(ubAxes[1].ubSize);
        ext.srcStride = ubAxes[1].gmStride * typeBytes - ext.blockLen;
    } else {
        ext.blockCount = 1;
        ext.srcStride = 0;
    }

    int64_t ubStride[MAX_PATTERN_RANK];
    ubStride[0] = typeBytes;
    for (int32_t i = 1; i < axisCount; ++i) {
        ubStride[i] = ubStride[i - 1] * ubAxes[i - 1].paddedSize;
    }

    LoopModeParams loop = {};
    loop.loop1Size = 1;
    loop.loop2Size = 1;
    if (axisCount >= 3) {
        loop.loop1Size = static_cast<uint32_t>(ubAxes[2].ubSize);
        loop.loop1SrcStride = static_cast<uint64_t>(ubAxes[2].gmStride * typeBytes);
        loop.loop1DstStride = static_cast<uint64_t>(ubStride[2]);
    }
    if (axisCount >= 4) {
        loop.loop2Size = static_cast<uint32_t>(ubAxes[3].ubSize);
        loop.loop2SrcStride = static_cast<uint64_t>(ubAxes[3].gmStride * typeBytes);
        loop.loop2DstStride = static_cast<uint64_t>(ubStride[3]);
    }
    const bool useLoopMode = axisCount >= 3;
    if (useLoopMode) {
        SetLoopModePara(loop, DataCopyMVType::OUT_TO_UB);
    }

    int64_t outerCount = 1;
    for (int32_t i = 4; i < axisCount; ++i) {
        outerCount *= ubAxes[i].ubSize;
    }
    for (int64_t flat = 0; flat < outerCount; ++flat) {
        int64_t rem = flat;
        int64_t gmDelta = 0;
        int64_t ubDeltaBytes = 0;
        for (int32_t i = 4; i < axisCount; ++i) {
            const int64_t coord = rem % ubAxes[i].ubSize;
            rem /= ubAxes[i].ubSize;
            gmDelta += coord * ubAxes[i].gmStride;
            ubDeltaBytes += coord * ubStride[i];
        }
        const int64_t ubDelta = ubDeltaBytes / typeBytes;
        DataCopyPad(preInLocal[ubDelta], xGm_[baseGmOff + gmDelta], ext, pad);
    }
    if (useLoopMode) {
        ResetLoopModePara(DataCopyMVType::OUT_TO_UB);
    }
}

template <bool groupPhase>
__aicore__ inline void PreElewise(LocalTensor<D_T>& src, uint32_t slot, int32_t outputIdx)
{
    __ubuf__ D_T* srcPtr = reinterpret_cast<__ubuf__ D_T*>(src.GetPhyAddr());
    auto tmp = tmpBuf_.Get<float>();
    __ubuf__ float* dstPtr = reinterpret_cast<__ubuf__ float*>(
        tmp[static_cast<int64_t>(slot) * tmpSlotElems_].GetPhyAddr());
    const uint32_t count = static_cast<uint32_t>(aUbFactorAlign_ * innerAProdAlign_ * rUbFactorAlign_ *
                                                 innerRProdAlign_);
    const uint16_t repeats = static_cast<uint16_t>((count + kRepF32 - 1U) / kRepF32);
    if (outputIdx == 0) {
        if constexpr (groupPhase && isTailR) {
            asc_vf_call<CastToF32VfImpl<D_T>>(srcPtr, dstPtr, count, repeats);
        } else {
            if (sumInputScale_ == 1.0F) {
                asc_vf_call<CastToF32VfImpl<D_T>>(srcPtr, dstPtr, count, repeats);
            } else {
                asc_vf_call<CastScaleToF32VfImpl<D_T>>(srcPtr, dstPtr, sumInputScale_, count, repeats);
            }
        }
    } else {
        asc_vf_call<CastSquareVfImpl<D_T>>(srcPtr, dstPtr, count, repeats);
    }
}

__aicore__ inline void ClearChunkExtensionVf(uint32_t slot, int64_t aLen, int64_t rLen)
{
    if (rLen >= rUbFactor_ && aLen >= aUbFactor_) {
        return;
    }
    auto tmp = tmpBuf_.Get<float>();
    __ubuf__ float* base = reinterpret_cast<__ubuf__ float*>(
        tmp[static_cast<int64_t>(slot) * tmpSlotElems_].GetPhyAddr());
    if constexpr (isTailR) {
        const uint32_t aEntries = static_cast<uint32_t>(aUbFactorAlign_ * innerAProdAlign_);
        const uint32_t innerR = static_cast<uint32_t>(innerRProdAlign_);
        const uint32_t validEnd = static_cast<uint32_t>(rLen) * innerR;
        const uint32_t extStart = (validEnd + kBlockF32 - 1U) / kBlockF32 * kBlockF32;
        const uint32_t aStride = static_cast<uint32_t>(rUbFactorAlign_ * innerRProdAlign_);
        if (extStart < aStride) {
            const uint32_t extLanes = aStride - extStart;
            const uint16_t repeats = static_cast<uint16_t>((extLanes + kRepF32 - 1U) / kRepF32);
            asc_vf_call<ClearChunkExtTailRVfImpl>(base, extStart, aStride, extLanes, static_cast<uint16_t>(aEntries),
                                                  repeats);
        }
    } else {
        const uint32_t cellElems = static_cast<uint32_t>(aUbFactorAlign_ * innerAProdAlign_ * innerRProdAlign_);
        const uint32_t startElem = static_cast<uint32_t>(rLen) * cellElems;
        const uint32_t clearElems = (static_cast<uint32_t>(rUbFactor_) - static_cast<uint32_t>(rLen)) * cellElems;
        if (clearElems > 0U) {
            const uint16_t repeats = static_cast<uint16_t>((clearElems + kRepF32 - 1U) / kRepF32);
            asc_vf_call<ClearChunkExtTailAVfImpl>(base, startElem, clearElems, repeats);
        }
    }
}

__aicore__ inline void MergeTmpBufVf()
{
    auto tmp = tmpBuf_.Get<float>();
    __ubuf__ float* slot0 = reinterpret_cast<__ubuf__ float*>(tmp.GetPhyAddr());
    __ubuf__ float* slot1 = slot0 + tmpSlotElems_;
    const uint32_t totalElems = static_cast<uint32_t>(aUbFactorAlign_ * innerAProdAlign_ * rUbFactorAlign_ *
                                                      innerRProdAlign_);
    const uint16_t repeats = static_cast<uint16_t>((totalElems + kRepF32 - 1U) / kRepF32);
    asc_vf_call<MergeTmpBufVfImpl>(slot0, slot1, totalElems, repeats);
}

__aicore__ inline void ReduceRPattern(uint32_t srcSlot, uint32_t dstSlot)
{
    auto tmp = tmpBuf_.Get<float>();
    auto src = tmp[static_cast<int64_t>(srcSlot) * tmpSlotElems_];
    auto dst = tmp[static_cast<int64_t>(dstSlot) * tmpSlotElems_];
    const uint32_t aBundle = static_cast<uint32_t>(aUbFactorAlign_ * innerAProdAlign_);
    const uint32_t rBundle = static_cast<uint32_t>(rUbFactorAlign_ * innerRProdAlign_);
    if constexpr (isTailR) {
        uint32_t srcShape[2] = {aBundle, rBundle};
        AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR, true>(dst, src, srcShape, true);
    } else {
        uint32_t srcShape[2] = {rBundle, aBundle};
        AscendC::ReduceSum<float, AscendC::Pattern::Reduce::RA, true>(dst, src, srcShape, true);
    }
}

__aicore__ inline void ClearCacheTreeVf()
{
    __ubuf__ float* cache = reinterpret_cast<__ubuf__ float*>(cacheBuf_.Get<float>().GetPhyAddr());
    const uint32_t totalElems = static_cast<uint32_t>(cacheBufElems_);
    const uint16_t repeats = static_cast<uint16_t>((totalElems + kRepF32 - 1U) / kRepF32);
    asc_vf_call<ClearCacheTreeVfImpl>(cache, totalElems, repeats);
}

__aicore__ inline void DoCaching(uint16_t cacheID, uint32_t srcSlot)
{
    auto tmp = tmpBuf_.Get<float>();
    __ubuf__ float* src = reinterpret_cast<__ubuf__ float*>(
        tmp[static_cast<int64_t>(srcSlot) * tmpSlotElems_].GetPhyAddr());
    __ubuf__ float* cache = reinterpret_cast<__ubuf__ float*>(cacheBuf_.Get<float>().GetPhyAddr());
    const uint32_t lanes = static_cast<uint32_t>(aUbFactorAlign_ * innerAProdAlign_);
    const uint32_t levelStride = (lanes + kBlockF32 - 1U) / kBlockF32 * kBlockF32;
    const int32_t levelOffset = static_cast<int32_t>(cacheID) * static_cast<int32_t>(levelStride);
    const uint16_t repeats = static_cast<uint16_t>((lanes + kRepF32 - 1U) / kRepF32);
    asc_vf_call<DoCachingVfImpl>(src, cache, lanes, levelStride, levelOffset, repeats, cacheID);
}

__aicore__ inline void CopyInRChunk(int64_t inputBaseOff, int64_t aLen, int64_t rLoopIdx, int64_t& rLen)
{
    const int64_t rGmOff = UnravelR(rLoopIdx, rLen);
    auto preIn = preInQue_.AllocTensor<D_T>();
    DoCopyInTile(inputBaseOff + rGmOff, aLen, rLen, preIn);
    preInQue_.EnQue(preIn);
}

template <bool groupPhase>
__aicore__ inline void ConsumeRChunk(int64_t aLen, int64_t rLen, uint32_t slot, int32_t outputIdx)
{
    auto deq = preInQue_.DeQue<D_T>();
    PreElewise<groupPhase>(deq, slot, outputIdx);
    preInQue_.FreeTensor(deq);
    if (rLen < rUbFactor_ || aLen < aUbFactor_) {
        ClearChunkExtensionVf(slot, aLen, rLen);
    }
}

__aicore__ inline int64_t GetScheduledRLoopIdx(int64_t flatIdx, int64_t rStart, int64_t bisectionPos,
                                               int64_t bisectionTail) const
{
    const int64_t pairChunkCount = 2 * bisectionTail;
    if (flatIdx < pairChunkCount) {
        const int64_t unitIdx = flatIdx / 2;
        return rStart + unitIdx + ((flatIdx & 1) ? bisectionPos : 0);
    }
    return rStart + bisectionTail + flatIdx - pairChunkCount;
}

template <bool groupPhase>
__aicore__ inline void ComputeRRange(int64_t inputBaseOff, int64_t aLen, int64_t rStart, int64_t rCount,
                                     int32_t outputIdx)
{
    const int64_t bisectionPos = static_cast<int64_t>(FindNearestPower2(static_cast<uint64_t>(rCount)));
    const int64_t bisectionTail = rCount - bisectionPos;
    const int64_t pairChunkCount = 2 * bisectionTail;
    int64_t queuedRLen[2] = {};

    const int64_t initialPrefetch = (rCount < 2) ? rCount : 2;
    for (int64_t flatIdx = 0; flatIdx < initialPrefetch; ++flatIdx) {
        const int64_t rLoopIdx = GetScheduledRLoopIdx(flatIdx, rStart, bisectionPos, bisectionTail);
        CopyInRChunk(inputBaseOff, aLen, rLoopIdx, queuedRLen[flatIdx & 1]);
    }

    for (int64_t flatIdx = 0; flatIdx < rCount; ++flatIdx) {
        const bool isPairedUnit = flatIdx < pairChunkCount;
        const uint32_t slot = isPairedUnit ? static_cast<uint32_t>(flatIdx & 1) : 0U;
        const int64_t cacheIdx = isPairedUnit ? flatIdx / 2 : bisectionTail + flatIdx - pairChunkCount;
        const bool isUnitEnd = !isPairedUnit || slot == 1U;
        ConsumeRChunk<groupPhase>(aLen, queuedRLen[flatIdx & 1], slot, outputIdx);

        const int64_t nextFlatIdx = flatIdx + 2;
        if (nextFlatIdx < rCount) {
            const int64_t nextRLoopIdx = GetScheduledRLoopIdx(nextFlatIdx, rStart, bisectionPos, bisectionTail);
            CopyInRChunk(inputBaseOff, aLen, nextRLoopIdx, queuedRLen[nextFlatIdx & 1]);
        }

        if (isUnitEnd) {
            if (isPairedUnit) {
                MergeTmpBufVf();
            }
            ReduceRPattern(0U, 1U);
            DoCaching(GetCacheID(cacheIdx), 1U);
        }
    }
}

__aicore__ inline void ComputeR(int64_t inputBaseOff, int64_t aLen, int32_t outputIdx)
{
    ComputeRRange<false>(inputBaseOff, aLen, 0, rLoopCntTotal_, outputIdx);
}

__aicore__ inline void DoOneAChunkGroup(int64_t inputBaseOff, int64_t aLen, int64_t rStart, int64_t rEnd,
                                        int32_t outputIdx)
{
    const int64_t rCount = rEnd - rStart;
    ClearCacheTreeVf();
    ComputeRRange<true>(inputBaseOff, aLen, rStart, rCount, outputIdx);
}

__aicore__ inline void Phase1OutputToWorkspace(int64_t outputOff, int64_t aLen, int64_t rGroupIdx, int64_t rCount)
{
    const uint32_t lanes = static_cast<uint32_t>(aUbFactorAlign_ * innerAProdAlign_);
    const uint32_t levelStride = (lanes + kBlockF32 - 1U) / kBlockF32 * kBlockF32;
    const int64_t rootLevel = static_cast<int64_t>(CalLog2(FindNearestPower2(static_cast<uint64_t>(rCount))));
    const int64_t rootOff = rootLevel * static_cast<int64_t>(levelStride);
    auto cache = cacheBuf_.Get<float>();

    event_t vToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(vToMte3);
    WaitFlag<HardEvent::V_MTE3>(vToMte3);

    DataCopyExtParams params = {};
    if constexpr (isTailR) {
        params.blockLen = static_cast<uint32_t>(aLen * innerAProd_ * static_cast<int64_t>(sizeof(float)));
        params.blockCount = 1;
        params.srcStride = 0;
    } else {
        const int32_t lastA = LastAAxis();
        const int64_t lastASize = axisShape_[lastA];
        if (aSplit_ == lastA) {
            params.blockLen = static_cast<uint32_t>(aLen * static_cast<int64_t>(sizeof(float)));
            params.blockCount = 1;
            params.srcStride = 0;
        } else {
            const int64_t inputBlockElems = kBlockBytes / static_cast<int64_t>(sizeof(D_T));
            const int64_t lastASizeAlign = (lastASize + inputBlockElems - 1) / inputBlockElems * inputBlockElems;
            params.blockLen = static_cast<uint32_t>(lastASize * static_cast<int64_t>(sizeof(float)));
            params.blockCount = static_cast<uint16_t>(aLen * innerAProd_ / lastASize);
            params.srcStride = (lastASizeAlign - lastASize) * static_cast<int64_t>(sizeof(float)) / kBlockBytes;
        }
    }
    params.dstStride = 0;
    const int64_t workspaceOff = rGroupIdx * aTotal_ + outputOff;
    DataCopyPad(wsGm_[workspaceOff], cache[rootOff], params);
}

#include "bn_training_reduce_part2.h"

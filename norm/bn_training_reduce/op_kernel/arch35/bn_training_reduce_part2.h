/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* Internal implementation section of bn_training_reduce.h. Include only from bn_training_reduce.h. */

__aicore__ inline void PostGroupPartial(int64_t rCount, int32_t outputIdx)
{
    const uint32_t lanes = static_cast<uint32_t>(aUbFactorAlign_ * innerAProdAlign_);
    const uint32_t levelStride = (lanes + kBlockF32 - 1U) / kBlockF32 * kBlockF32;
    const int64_t rootLevel = static_cast<int64_t>(CalLog2(FindNearestPower2(static_cast<uint64_t>(rCount))));
    const int64_t rootOff = rootLevel * static_cast<int64_t>(levelStride);
    auto cache = cacheBuf_.Get<float>();
    auto out = outQue_.AllocTensor<float>();
    __ubuf__ float* srcPtr = reinterpret_cast<__ubuf__ float*>(cache[rootOff].GetPhyAddr());
    __ubuf__ float* dstPtr = reinterpret_cast<__ubuf__ float*>(out.GetPhyAddr());
    const uint16_t repeats = static_cast<uint16_t>((lanes + kRepF32 - 1U) / kRepF32);
    if constexpr (isTailR) {
        asc_vf_call<Phase2PostElewiseVfImpl>(srcPtr, dstPtr, lanes, repeats);
    } else if (outputIdx == 0) {
        asc_vf_call<RestoreSumVfImpl>(srcPtr, dstPtr, sumOutputScale_, lanes, repeats);
    } else {
        asc_vf_call<Phase2PostElewiseVfImpl>(srcPtr, dstPtr, lanes, repeats);
    }
    outQue_.EnQue(out);
}

__aicore__ inline void Phase1Process(int32_t outputIdx)
{
    const int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
    if (blockIdx >= static_cast<int64_t>(usedCoreNum_)) {
        return;
    }
    const int64_t aLoopIdx = blockIdx / rGroupCnt_;
    const int64_t rGroupIdx = blockIdx % rGroupCnt_;
    // R 方向大小核式均匀分配：rGroupCnt_ ≤ rLoopCntTotal_ 恒成立（tiling 保证），每组 ≥1 chunk，无空组
    const int64_t rSmallGroupLoopCnt = rLoopCntTotal_ / rGroupCnt_;
    const int64_t rBigGroupCnt = rLoopCntTotal_ % rGroupCnt_;
    const int64_t rBigGroupLoopCnt = rSmallGroupLoopCnt + (rBigGroupCnt > 0 ? 1 : 0);
    int64_t rStart = 0;
    int64_t rCount = 0;
    if (rGroupIdx < rBigGroupCnt) {
        rStart = rGroupIdx * rBigGroupLoopCnt;
        rCount = rBigGroupLoopCnt;
    } else {
        rStart = rBigGroupCnt * rBigGroupLoopCnt + (rGroupIdx - rBigGroupCnt) * rSmallGroupLoopCnt;
        rCount = rSmallGroupLoopCnt;
    }
    const int64_t rEnd = rStart + rCount;
    if (rStart >= rLoopCntTotal_) {
        return; // 防御性早退（理论上不可达）
    }
    if (aLoopIdx >= aLoopCntTotal_) {
        return;
    }

    int64_t aLen = 0;
    int64_t inputBaseOff = 0;
    int64_t outputOff = 0;
    UnravelALoop(aLoopIdx, aLen, inputBaseOff, outputOff);
    DoOneAChunkGroup(inputBaseOff, aLen, rStart, rEnd, outputIdx);
    if constexpr (isDeterministic) {
        Phase1OutputToWorkspace(outputOff, aLen, rGroupIdx, rCount);
    } else {
        PostGroupPartial(rCount, outputIdx);
        CopyOut<true>(outputOff, aLen, outputIdx);
    }
}

__aicore__ inline void Phase2Process(int32_t outputIdx)
{
    const int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
    const int64_t preInElems = preReduceUbSize_ / static_cast<int64_t>(sizeof(float));
    int64_t aUbFactorP2 = preInElems / rGroupCnt_;
    if (aUbFactorP2 >= static_cast<int64_t>(kBlockF32)) {
        aUbFactorP2 = aUbFactorP2 / static_cast<int64_t>(kBlockF32) * static_cast<int64_t>(kBlockF32);
    }
    const int64_t outLimit = postReduceUbSize_ / static_cast<int64_t>(sizeof(float));
    aUbFactorP2 = (aUbFactorP2 < outLimit) ? aUbFactorP2 : outLimit;
    aUbFactorP2 = (aUbFactorP2 < tmpSlotElems_) ? aUbFactorP2 : tmpSlotElems_;
    aUbFactorP2 = (aUbFactorP2 < aTotal_) ? aUbFactorP2 : aTotal_;
    if (aUbFactorP2 <= 0) {
        return;
    }

    const int64_t aLoopTotal = (aTotal_ + aUbFactorP2 - 1) / aUbFactorP2;
    const int64_t smallLoops = aLoopTotal / usedCoreNum_;
    const int64_t bigCoreCount = aLoopTotal % usedCoreNum_;
    const int64_t bigLoops = smallLoops + (bigCoreCount > 0 ? 1 : 0);
    const int64_t usedP2 = (smallLoops > 0) ? static_cast<int64_t>(usedCoreNum_) : bigCoreCount;
    if (blockIdx >= usedP2) {
        return;
    }

    int64_t loopStart = 0;
    int64_t loopEnd = 0;
    if (blockIdx < bigCoreCount) {
        loopStart = blockIdx * bigLoops;
        loopEnd = loopStart + bigLoops;
    } else {
        loopStart = bigCoreCount * bigLoops + (blockIdx - bigCoreCount) * smallLoops;
        loopEnd = loopStart + smallLoops;
    }

    for (int64_t aLoop = loopStart; aLoop < loopEnd; ++aLoop) {
        const int64_t aOff = aLoop * aUbFactorP2;
        const int64_t remain = aTotal_ - aOff;
        const int64_t aLen = (remain < aUbFactorP2) ? remain : aUbFactorP2;
        const int64_t aLenUb = (aLen + static_cast<int64_t>(kBlockF32) - 1) / static_cast<int64_t>(kBlockF32) *
                               static_cast<int64_t>(kBlockF32);

        auto preIn = preInQue_.AllocTensor<float>();
        DataCopyExtParams inParams = {};
        inParams.blockLen = static_cast<uint32_t>(aLen * static_cast<int64_t>(sizeof(float)));
        inParams.blockCount = static_cast<uint16_t>(rGroupCnt_);
        inParams.srcStride = aTotal_ * static_cast<int64_t>(sizeof(float)) - inParams.blockLen;
        inParams.dstStride = 0;
        DataCopyPadExtParams<float> inPad = {true, 0, static_cast<uint8_t>(aLenUb - aLen), 0.0f};
        DataCopyPad(preIn, wsGm_[aOff], inParams, inPad);
        preInQue_.EnQue(preIn);
        auto src = preInQue_.DeQue<float>();

        auto tmp = tmpBuf_.Get<float>();
        uint32_t srcShape[2] = {static_cast<uint32_t>(rGroupCnt_), static_cast<uint32_t>(aLenUb)};
        AscendC::ReduceSum<float, AscendC::Pattern::Reduce::RA, true>(tmp, src, srcShape, true);
        preInQue_.FreeTensor(src);

        auto out = outQue_.AllocTensor<float>();
        __ubuf__ float* srcPtr = reinterpret_cast<__ubuf__ float*>(tmp.GetPhyAddr());
        __ubuf__ float* dstPtr = reinterpret_cast<__ubuf__ float*>(out.GetPhyAddr());
        const uint32_t elems = static_cast<uint32_t>(aLen);
        const uint16_t repeats = static_cast<uint16_t>((elems + kRepF32 - 1U) / kRepF32);
        if constexpr (isTailR) {
            asc_vf_call<Phase2PostElewiseVfImpl>(srcPtr, dstPtr, elems, repeats);
        } else if (outputIdx == 0) {
            asc_vf_call<RestoreSumVfImpl>(srcPtr, dstPtr, sumOutputScale_, elems, repeats);
        } else {
            asc_vf_call<Phase2PostElewiseVfImpl>(srcPtr, dstPtr, elems, repeats);
        }
        outQue_.EnQue(out);

        auto deq = outQue_.DeQue<float>();
        DataCopyExtParams outParams = {};
        outParams.blockLen = static_cast<uint32_t>(aLen * static_cast<int64_t>(sizeof(float)));
        outParams.blockCount = 1;
        outParams.srcStride = 0;
        outParams.dstStride = 0;
        if (outputIdx == 0) {
            DataCopyPad(sumGm_[aOff], deq, outParams);
        } else {
            DataCopyPad(squareSumGm_[aOff], deq, outParams);
        }
        outQue_.FreeTensor(deq);
    }
}

__aicore__ inline void PostElewise(int32_t outputIdx)
{
    const uint32_t lanes = static_cast<uint32_t>(aUbFactorAlign_ * innerAProdAlign_);
    const uint32_t levelStride = (lanes + kBlockF32 - 1U) / kBlockF32 * kBlockF32;
    const int32_t rootOffset = static_cast<int32_t>(cacheCount_ - 1) * static_cast<int32_t>(levelStride);
    __ubuf__ float* root = reinterpret_cast<__ubuf__ float*>(cacheBuf_.Get<float>().GetPhyAddr()) + rootOffset;
    auto out = outQue_.AllocTensor<float>();
    __ubuf__ float* output = reinterpret_cast<__ubuf__ float*>(out.GetPhyAddr());
    const uint16_t repeats = static_cast<uint16_t>((lanes + kRepF32 - 1U) / kRepF32);
    if (outputIdx == 0) {
        asc_vf_call<RestoreSumVfImpl>(root, output, sumOutputScale_, lanes, repeats);
    } else {
        asc_vf_call<PostElewiseVfImpl>(root, output, lanes, repeats);
    }
    outQue_.EnQue(out);
}

template <bool atomicAdd>
__aicore__ inline void CopyOut(int64_t outputOff, int64_t aLen, int32_t outputIdx)
{
    auto out = outQue_.DeQue<float>();
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
    params.rsv = 0;
    if constexpr (atomicAdd) {
        SetAtomicAdd<float>();
    }
    if (outputIdx == 0) {
        DataCopyPad(sumGm_[outputOff], out, params);
    } else {
        DataCopyPad(squareSumGm_[outputOff], out, params);
    }
    if constexpr (atomicAdd) {
        SetAtomicNone();
    }
    outQue_.FreeTensor(out);
}

int32_t axisNum_ = 0;
int64_t axisShape_[MAX_PATTERN_RANK] = {};
int64_t axisStride_[MAX_PATTERN_RANK] = {};
int64_t outStride_[MAX_PATTERN_RANK] = {};
int64_t aLoopCntTotal_ = 0;
int64_t aSplitChunkCnt_ = 0;
int64_t aBigCoreLoopCnt_ = 0;
int64_t aSmallCoreLoopCnt_ = 0;
int32_t aBigCoreCnt_ = 0;
int32_t usedCoreNum_ = 0;
int32_t aSplit_ = 0;
int32_t rSplit_ = 0;
int64_t aUbFactor_ = 0;
int64_t aUbFactorAlign_ = 0;
int64_t rUbFactor_ = 0;
int64_t rUbFactorAlign_ = 0;
int64_t innerAProd_ = 0;
int64_t innerAProdAlign_ = 0;
int64_t innerRProdAlign_ = 0;
int64_t rLoopCntTotal_ = 0;
int64_t cacheCount_ = 0;
int64_t preReduceUbSize_ = 0;
int64_t postReduceUbSize_ = 0;
int64_t tmpSlotElems_ = 0;
int64_t cacheBufElems_ = 0;
int64_t rGroupCnt_ = 0;
int64_t aTotal_ = 0;
float sumInputScale_ = 1.0F;
float sumOutputScale_ = 1.0F;

GlobalTensor<D_T> xGm_;
GlobalTensor<float> sumGm_;
GlobalTensor<float> squareSumGm_;
GlobalTensor<float> wsGm_;
TPipe pipe_;
TQue<QuePosition::VECIN, 2> preInQue_;
TBuf<QuePosition::VECCALC> tmpBuf_;
TBuf<QuePosition::VECCALC> cacheBuf_;
TQue<QuePosition::VECOUT, 1> outQue_;

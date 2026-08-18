/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BN_TRAINING_REDUCE_H_
#define BN_TRAINING_REDUCE_H_

#include "adv_api/reduce/reduce.h"
#include "bn_training_reduce_tiling_data.h"
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"

namespace NsBNTrainingReduce {

using namespace AscendC;

constexpr uint32_t kVlBytes = 256U;
constexpr uint32_t kRepF32 = kVlBytes / sizeof(float);
constexpr uint32_t kBlockBytes = 32U;
constexpr uint32_t kBlockF32 = kBlockBytes / sizeof(float);

constexpr AscendC::Reg::CastTrait kCastTraitToFp32{AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN,
                                                   AscendC::Reg::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_NONE};

template <typename DType>
__simd_vf__ inline void CastToF32VfImpl(__ubuf__ DType* src, __ubuf__ float* dst, uint32_t totalElems,
                                        uint16_t repeatTime)
{
    constexpr bool kNeedCast = !std::is_same_v<DType, float>;
    AscendC::Reg::MaskReg mask;
    uint32_t remaining = totalElems;
    for (uint16_t i = 0; i < repeatTime; ++i) {
        const int32_t off = static_cast<int32_t>(i) * static_cast<int32_t>(kRepF32);
        mask = AscendC::Reg::UpdateMask<float>(remaining);
        AscendC::Reg::RegTensor<float> f32Reg;
        if constexpr (kNeedCast) {
            AscendC::Reg::RegTensor<DType> b16Reg;
            AscendC::Reg::LoadAlign<DType, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(b16Reg, src + off);
            AscendC::Reg::Cast<float, DType, kCastTraitToFp32>(f32Reg, b16Reg, mask);
        } else {
            AscendC::Reg::LoadAlign(f32Reg, src + off);
        }
        AscendC::Reg::StoreAlign(dst + off, f32Reg, mask);
    }
}

template <typename DType>
__simd_vf__ inline void CastScaleToF32VfImpl(__ubuf__ DType* src, __ubuf__ float* dst, float scale, uint32_t totalElems,
                                             uint16_t repeatTime)
{
    constexpr bool kNeedCast = !std::is_same_v<DType, float>;
    AscendC::Reg::RegTensor<float> scaleReg;
    AscendC::Reg::Duplicate(scaleReg, scale);
    AscendC::Reg::MaskReg mask;
    uint32_t remaining = totalElems;
    for (uint16_t i = 0; i < repeatTime; ++i) {
        const int32_t off = static_cast<int32_t>(i) * static_cast<int32_t>(kRepF32);
        mask = AscendC::Reg::UpdateMask<float>(remaining);
        AscendC::Reg::RegTensor<float> f32Reg;
        if constexpr (kNeedCast) {
            AscendC::Reg::RegTensor<DType> b16Reg;
            AscendC::Reg::LoadAlign<DType, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(b16Reg, src + off);
            AscendC::Reg::Cast<float, DType, kCastTraitToFp32>(f32Reg, b16Reg, mask);
        } else {
            AscendC::Reg::LoadAlign(f32Reg, src + off);
        }
        AscendC::Reg::Mul(f32Reg, f32Reg, scaleReg, mask);
        AscendC::Reg::StoreAlign(dst + off, f32Reg, mask);
    }
}

template <typename DType>
__simd_vf__ inline void CastSquareVfImpl(__ubuf__ DType* src, __ubuf__ float* dst, uint32_t totalElems,
                                         uint16_t repeatTime)
{
    constexpr bool kNeedCast = !std::is_same_v<DType, float>;
    AscendC::Reg::MaskReg mask;
    uint32_t remaining = totalElems;
    for (uint16_t i = 0; i < repeatTime; ++i) {
        const int32_t off = static_cast<int32_t>(i) * static_cast<int32_t>(kRepF32);
        mask = AscendC::Reg::UpdateMask<float>(remaining);
        AscendC::Reg::RegTensor<float> f32Reg;
        if constexpr (kNeedCast) {
            AscendC::Reg::RegTensor<DType> b16Reg;
            AscendC::Reg::LoadAlign<DType, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(b16Reg, src + off);
            AscendC::Reg::Cast<float, DType, kCastTraitToFp32>(f32Reg, b16Reg, mask);
        } else {
            AscendC::Reg::LoadAlign(f32Reg, src + off);
        }
        AscendC::Reg::Mul(f32Reg, f32Reg, f32Reg, mask);
        AscendC::Reg::StoreAlign(dst + off, f32Reg, mask);
    }
}

__simd_vf__ inline void ClearChunkExtTailAVfImpl(__ubuf__ float* base, uint32_t startElem, uint32_t clearElems,
                                                 uint16_t repeats)
{
    AscendC::Reg::RegTensor<float> zero;
    AscendC::Reg::Duplicate(zero, 0.0f);
    AscendC::Reg::MaskReg mask;
    uint32_t remaining = clearElems;
    for (uint16_t i = 0; i < repeats; ++i) {
        const int32_t off = static_cast<int32_t>(startElem) + static_cast<int32_t>(i) * static_cast<int32_t>(kRepF32);
        mask = AscendC::Reg::UpdateMask<float>(remaining);
        AscendC::Reg::StoreAlign(base + off, zero, mask);
    }
}

__simd_vf__ inline void ClearChunkExtTailRVfImpl(__ubuf__ float* base, uint32_t extStart, uint32_t aStride,
                                                 uint32_t extLanes, uint16_t aCount, uint16_t repeatsPerA)
{
    AscendC::Reg::RegTensor<float> zero;
    AscendC::Reg::Duplicate(zero, 0.0f);
    for (uint16_t a = 0; a < aCount; ++a) {
        const int32_t aOff = static_cast<int32_t>(a) * static_cast<int32_t>(aStride);
        uint32_t remaining = extLanes;
        for (uint16_t r = 0; r < repeatsPerA; ++r) {
            const int32_t off = aOff + static_cast<int32_t>(extStart) +
                                static_cast<int32_t>(r) * static_cast<int32_t>(kRepF32);
            auto mask = AscendC::Reg::UpdateMask<float>(remaining);
            AscendC::Reg::StoreAlign(base + off, zero, mask);
        }
    }
}

__simd_vf__ inline void MergeTmpBufVfImpl(__ubuf__ float* slot0, __ubuf__ float* slot1, uint32_t totalElems,
                                          uint16_t repeats)
{
    AscendC::Reg::MaskReg mask;
    uint32_t remaining = totalElems;
    for (uint16_t i = 0; i < repeats; ++i) {
        const int32_t off = static_cast<int32_t>(i) * static_cast<int32_t>(kRepF32);
        mask = AscendC::Reg::UpdateMask<float>(remaining);
        AscendC::Reg::RegTensor<float> left;
        AscendC::Reg::RegTensor<float> right;
        AscendC::Reg::LoadAlign(left, slot0 + off);
        AscendC::Reg::LoadAlign(right, slot1 + off);
        AscendC::Reg::Add(left, left, right, mask);
        AscendC::Reg::StoreAlign(slot0 + off, left, mask);
    }
}

__simd_vf__ inline void ClearCacheTreeVfImpl(__ubuf__ float* cache, uint32_t totalElems, uint16_t repeats)
{
    AscendC::Reg::RegTensor<float> zero;
    AscendC::Reg::Duplicate(zero, 0.0f);
    AscendC::Reg::MaskReg mask;
    uint32_t remaining = totalElems;
    for (uint16_t i = 0; i < repeats; ++i) {
        const int32_t off = static_cast<int32_t>(i) * static_cast<int32_t>(kRepF32);
        mask = AscendC::Reg::UpdateMask<float>(remaining);
        AscendC::Reg::StoreAlign(cache + off, zero, mask);
    }
}

__simd_vf__ inline void DoCachingVfImpl(__ubuf__ float* src, __ubuf__ float* cache, uint32_t laneCount,
                                        uint32_t levelStride, int32_t levelOffset, uint16_t repeats,
                                        uint16_t cacheLevel)
{
    AscendC::Reg::MaskReg mask;
    uint32_t remaining = laneCount;
    for (uint16_t i = 0; i < repeats; ++i) {
        const int32_t off = static_cast<int32_t>(i) * static_cast<int32_t>(kRepF32);
        mask = AscendC::Reg::UpdateMask<float>(remaining);
        AscendC::Reg::RegTensor<float> acc;
        AscendC::Reg::LoadAlign(acc, src + off);
        for (uint16_t level = 0; level < cacheLevel; ++level) {
            const int32_t cacheOff = static_cast<int32_t>(level) * static_cast<int32_t>(levelStride) + off;
            AscendC::Reg::RegTensor<float> old;
            AscendC::Reg::LoadAlign(old, cache + cacheOff);
            AscendC::Reg::Add(acc, acc, old, mask);
        }
        AscendC::Reg::StoreAlign(cache + levelOffset + off, acc, mask);
    }
}

__simd_vf__ inline void PostElewiseVfImpl(__ubuf__ float* root, __ubuf__ float* output, uint32_t lanes,
                                          uint16_t repeats)
{
    AscendC::Reg::MaskReg mask;
    uint32_t remaining = lanes;
    for (uint16_t i = 0; i < repeats; ++i) {
        const int32_t off = static_cast<int32_t>(i) * static_cast<int32_t>(kRepF32);
        mask = AscendC::Reg::UpdateMask<float>(remaining);
        AscendC::Reg::RegTensor<float> value;
        AscendC::Reg::LoadAlign(value, root + off);
        AscendC::Reg::StoreAlign(output + off, value, mask);
    }
}

__simd_vf__ inline void Phase2PostElewiseVfImpl(__ubuf__ float* src, __ubuf__ float* dst, uint32_t elems,
                                                uint16_t repeats)
{
    AscendC::Reg::MaskReg mask;
    uint32_t remaining = elems;
    for (uint16_t i = 0; i < repeats; ++i) {
        const int32_t off = static_cast<int32_t>(i) * static_cast<int32_t>(kRepF32);
        mask = AscendC::Reg::UpdateMask<float>(remaining);
        AscendC::Reg::RegTensor<float> value;
        AscendC::Reg::LoadAlign(value, src + off);
        AscendC::Reg::StoreAlign(dst + off, value, mask);
    }
}

__simd_vf__ inline void RestoreSumVfImpl(__ubuf__ float* src, __ubuf__ float* dst, float scale, uint32_t elems,
                                         uint16_t repeats)
{
    AscendC::Reg::RegTensor<float> scaleReg;
    AscendC::Reg::Duplicate(scaleReg, scale);
    AscendC::Reg::MaskReg mask;
    uint32_t remaining = elems;
    for (uint16_t i = 0; i < repeats; ++i) {
        const int32_t off = static_cast<int32_t>(i) * static_cast<int32_t>(kRepF32);
        mask = AscendC::Reg::UpdateMask<float>(remaining);
        AscendC::Reg::RegTensor<float> value;
        AscendC::Reg::LoadAlign(value, src + off);
        AscendC::Reg::Mul(value, value, scaleReg, mask);
        AscendC::Reg::StoreAlign(dst + off, value, mask);
    }
}

struct UBAxisDesc {
    int32_t gmIdx;
    int64_t ubSize;
    int64_t paddedSize;
    int64_t gmStride;
};

template <typename DType, bool isTailR, bool isDeterministic>
class BNTrainingReduceKernel {
public:
    using D_T = DType;

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR sum, GM_ADDR squareSum, const BNTrainingReduceTilingData* td)
    {
        axisNum_ = td->axisNum;
        for (int32_t i = 0; i < MAX_PATTERN_RANK; ++i) {
            axisShape_[i] = td->axisShape[i];
            axisStride_[i] = td->axisStride[i];
        }
        aLoopCntTotal_ = td->aLoopCntTotal;
        aSplitChunkCnt_ = td->aSplitChunkCnt;
        aBigCoreLoopCnt_ = td->aBigCoreLoopCnt;
        aSmallCoreLoopCnt_ = td->aSmallCoreLoopCnt;
        aBigCoreCnt_ = td->aBigCoreCnt;
        usedCoreNum_ = td->usedCoreNum;
        aSplit_ = td->aSplitAxisIdx;
        rSplit_ = td->rSplitAxisIdx;
        aUbFactor_ = td->aUbFactor;
        aUbFactorAlign_ = td->aUbFactorAlign;
        rUbFactor_ = td->rUbFactor;
        rUbFactorAlign_ = td->rUbFactorAlign;
        innerAProd_ = td->innerAProd;
        innerAProdAlign_ = td->innerAProdAlign;
        innerRProdAlign_ = td->innerRProdAlign;
        rLoopCntTotal_ = td->rLoopCntTotal;
        preReduceUbSize_ = td->preReduceUbSize;
        postReduceUbSize_ = td->postReduceUbSize;
        tmpSlotElems_ = td->tmpBufUbSize / static_cast<int64_t>(sizeof(float));
        cacheBufElems_ = td->cacheBufUbSize / static_cast<int64_t>(sizeof(float));

        int64_t reductionTotal = 1;
        for (int32_t i = 1; i < axisNum_; i += 2) {
            reductionTotal *= axisShape_[i];
        }
        uint64_t scaleSteps = static_cast<uint64_t>(reductionTotal - 1);
        while (scaleSteps > 0U) {
            sumInputScale_ *= 0.5F;
            sumOutputScale_ *= 2.0F;
            scaleSteps >>= 1U;
        }

        const int64_t bisectionPos = static_cast<int64_t>(FindNearestPower2(static_cast<uint64_t>(rLoopCntTotal_)));
        cacheCount_ = static_cast<int64_t>(CalLog2(static_cast<uint64_t>(bisectionPos))) + 1;

        int64_t outputStride = 1;
        for (int32_t i = axisNum_ - 1; i >= 0; --i) {
            if ((i & 1) == 0) {
                outStride_[i] = outputStride;
                outputStride *= axisShape_[i];
            }
        }

        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ D_T*>(x));
        sumGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(sum));
        squareSumGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(squareSum));
        pipe_.InitBuffer(preInQue_, 2, td->preReduceUbSize);
        pipe_.InitBuffer(tmpBuf_, 2 * td->tmpBufUbSize);
        pipe_.InitBuffer(cacheBuf_, td->cacheBufUbSize);
        pipe_.InitBuffer(outQue_, 1, td->postReduceUbSize);
    }

    __aicore__ inline void InitGroup(GM_ADDR x, GM_ADDR sum, GM_ADDR squareSum, GM_ADDR workspace,
                                     const BNTrainingReduceTilingData* td)
    {
        Init(x, sum, squareSum, td);
        if constexpr (isDeterministic) {
            wsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace));
        }
        rGroupCnt_ = td->rGroupCnt;
        aTotal_ = 1;
        for (int32_t i = 0; i < axisNum_; i += 2) {
            aTotal_ *= axisShape_[i];
        }
        if constexpr (!isDeterministic) {
            if (GetBlockIdx() == 0) {
                InitOutput<float>(sumGm_, aTotal_, 0.0F);
                InitOutput<float>(squareSumGm_, aTotal_, 0.0F);
            }
            SyncAll();
        }
    }

    __aicore__ inline void Process(int32_t outputIdx)
    {
        const int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
        if (blockIdx >= static_cast<int64_t>(usedCoreNum_)) {
            return;
        }

        int64_t aLoopStart = 0;
        int64_t aLoopEnd = 0;
        GetCoreALoopRange(blockIdx, aLoopStart, aLoopEnd);
        for (int64_t aLoopIdx = aLoopStart; aLoopIdx < aLoopEnd && aLoopIdx < aLoopCntTotal_; ++aLoopIdx) {
            int64_t aLen = 0;
            int64_t inputBaseOff = 0;
            int64_t outputOff = 0;
            UnravelALoop(aLoopIdx, aLen, inputBaseOff, outputOff);
            ClearCacheTreeVf();
            ComputeR(inputBaseOff, aLen, outputIdx);
            PostElewise(outputIdx);
            CopyOut<false>(outputOff, aLen, outputIdx);
        }
    }

    __aicore__ inline void ProcessGroup(int32_t outputIdx)
    {
        Phase1Process(outputIdx);
        if constexpr (isDeterministic) {
            SyncAll();
            Phase2Process(outputIdx);
        }
    }

private:
    __aicore__ inline uint64_t FindNearestPower2(uint64_t value) const
    {
        if (value == 0) {
            return 0;
        }
        if (value <= 2) {
            return 1;
        }
        if (value <= 4) {
            return 2;
        }
        const uint64_t reduced = value - 1;
        const uint64_t power = 63 - AscendC::ScalarCountLeadingZero(reduced);
        return static_cast<uint64_t>(1) << power;
    }

    __aicore__ inline uint64_t CalLog2(uint64_t value) const
    {
        uint64_t result = 0;
        while (value > 1) {
            value >>= 1;
            ++result;
        }
        return result;
    }

    __aicore__ inline uint16_t GetCacheID(int64_t rIdx) const
    {
        const uint64_t value = static_cast<uint64_t>(rIdx);
        return static_cast<uint16_t>(AscendC::ScalarGetCountOfValue<1>(value ^ (value + 1)) - 1);
    }

    __aicore__ inline int32_t LastAAxis() const
    {
        for (int32_t i = axisNum_ - 1; i >= 0; --i) {
            if ((i & 1) == 0) {
                return i;
            }
        }
        return 0;
    }

    __aicore__ inline int32_t LastRAxis() const
    {
        for (int32_t i = axisNum_ - 1; i >= 0; --i) {
            if ((i & 1) != 0) {
                return i;
            }
        }
        return 1;
    }

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

    __aicore__ inline void UnravelALoop(int64_t aLoopIdx, int64_t& aLen, int64_t& inputBaseOff,
                                        int64_t& outputOff) const
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
                asc_vf_call<ClearChunkExtTailRVfImpl>(base, extStart, aStride, extLanes,
                                                      static_cast<uint16_t>(aEntries), repeats);
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
        const int64_t rPerGroup = (rLoopCntTotal_ + rGroupCnt_ - 1) / rGroupCnt_;
        const int64_t rStart = rGroupIdx * rPerGroup;
        int64_t rEnd = rStart + rPerGroup;
        if (rEnd > rLoopCntTotal_) {
            rEnd = rLoopCntTotal_;
        }
        if (aLoopIdx >= aLoopCntTotal_ || rStart >= rEnd) {
            return;
        }

        int64_t aLen = 0;
        int64_t inputBaseOff = 0;
        int64_t outputOff = 0;
        UnravelALoop(aLoopIdx, aLen, inputBaseOff, outputOff);
        const int64_t rCount = rEnd - rStart;
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
};

} // namespace NsBNTrainingReduce

#endif // BN_TRAINING_REDUCE_H_

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

#include "bn_training_reduce_part1.h"

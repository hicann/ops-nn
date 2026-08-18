/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BN_TRAINING_REDUCE_SMALL_R_H_
#define BN_TRAINING_REDUCE_SMALL_R_H_

#include "bn_training_reduce_local_deps.h"
#include "bn_training_reduce_tiling_data.h"

#if !defined(__NPU_HOST__)

namespace NsBNTrainingReduce {

using namespace AscendC;

constexpr uint32_t kSmallRVlBytes = 256U;
constexpr uint32_t kSmallRRepF32 = kSmallRVlBytes / sizeof(float);

constexpr AscendC::Reg::CastTrait kSmallRCastTraitToFp32{AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN,
                                                         AscendC::Reg::MaskMergeMode::ZEROING,
                                                         AscendC::RoundMode::CAST_NONE};

// The input tile is already transposed by MultiCopy from [C, R] to [R, C].
// Accumulating both statistics in one VF traversal avoids the generic
// per-output NDDMA/cache-tree path entirely. R=1 naturally degenerates to one
// cast/add and one square, with no ReduceSum or cache operation.
template <typename DType>
__simd_vf__ inline void SmallRFusedStatsVfImpl(__ubuf__ DType* src, __ubuf__ float* sumDst,
                                               __ubuf__ float* squareSumDst, uint32_t channelCount,
                                               uint32_t channelStride, uint16_t reduceLen, uint16_t repeatTime)
{
    constexpr bool kNeedCast = !std::is_same_v<DType, float>;
    uint32_t remaining = channelCount;
    for (uint16_t i = 0; i < repeatTime; ++i) {
        const int32_t channelOff = static_cast<int32_t>(i) * static_cast<int32_t>(kSmallRRepF32);
        auto mask = AscendC::Reg::UpdateMask<float>(remaining);
        AscendC::Reg::RegTensor<float> sumReg;
        AscendC::Reg::RegTensor<float> squareSumReg;
        AscendC::Reg::Duplicate(sumReg, 0.0F);
        AscendC::Reg::Duplicate(squareSumReg, 0.0F);

        for (uint16_t r = 0; r < reduceLen; ++r) {
            const int32_t srcOff = static_cast<int32_t>(r) * static_cast<int32_t>(channelStride) + channelOff;
            AscendC::Reg::RegTensor<float> valueReg;
            if constexpr (kNeedCast) {
                AscendC::Reg::RegTensor<DType> b16Reg;
                AscendC::Reg::LoadAlign<DType, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(b16Reg, src + srcOff);
                AscendC::Reg::Cast<float, DType, kSmallRCastTraitToFp32>(valueReg, b16Reg, mask);
            } else {
                AscendC::Reg::LoadAlign(valueReg, src + srcOff);
            }
            AscendC::Reg::Add(sumReg, sumReg, valueReg, mask);
            AscendC::Reg::RegTensor<float> squareReg;
            AscendC::Reg::Mul(squareReg, valueReg, valueReg, mask);
            AscendC::Reg::Add(squareSumReg, squareSumReg, squareReg, mask);
        }

        AscendC::Reg::StoreAlign(sumDst + channelOff, sumReg, mask);
        AscendC::Reg::StoreAlign(squareSumDst + channelOff, squareSumReg, mask);
    }
}

template <typename DType>
class BNTrainingReduceSmallRKernel {
public:
    using D_T = DType;

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR sum, GM_ADDR squareSum, const BNTrainingReduceTilingData* td)
    {
        channels_ = td->axisShape[0];
        reduceLen_ = td->axisShape[1];
        tileChannels_ = td->aUbFactor;
        tileChannelsAlign_ = td->aUbFactorAlign;
        tileCount_ = td->aLoopCntTotal;
        bigCoreLoopCount_ = td->aBigCoreLoopCnt;
        smallCoreLoopCount_ = td->aSmallCoreLoopCnt;
        bigCoreCount_ = td->aBigCoreCnt;
        usedCoreCount_ = td->usedCoreNum;

        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ D_T*>(x));
        sumGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(sum));
        squareSumGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(squareSum));
        pipe_.InitBuffer(inputBuf_, td->preReduceUbSize);
        pipe_.InitBuffer(sumBuf_, td->postReduceUbSize);
        pipe_.InitBuffer(squareSumBuf_, td->tmpBufUbSize);
    }

    __aicore__ inline void Process()
    {
        const int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
        if (blockIdx >= static_cast<int64_t>(usedCoreCount_)) {
            return;
        }

        int64_t tileStart = 0;
        int64_t tileEnd = 0;
        if (blockIdx < static_cast<int64_t>(bigCoreCount_)) {
            tileStart = blockIdx * bigCoreLoopCount_;
            tileEnd = tileStart + bigCoreLoopCount_;
        } else {
            tileStart = static_cast<int64_t>(bigCoreCount_) * bigCoreLoopCount_ +
                        (blockIdx - static_cast<int64_t>(bigCoreCount_)) * smallCoreLoopCount_;
            tileEnd = tileStart + smallCoreLoopCount_;
        }

        const event_t mte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        const event_t vToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        for (int64_t tileIdx = tileStart; tileIdx < tileEnd && tileIdx < tileCount_; ++tileIdx) {
            const int64_t channelStart = tileIdx * tileChannels_;
            const int64_t remain = channels_ - channelStart;
            const int64_t channelCount = (remain < tileChannels_) ? remain : tileChannels_;
            CopyInTranspose(channelStart, channelCount);
            SetFlag<HardEvent::MTE2_V>(mte2ToV);
            WaitFlag<HardEvent::MTE2_V>(mte2ToV);

            auto inputLocal = inputBuf_.Get<D_T>();
            auto sumLocal = sumBuf_.Get<float>();
            auto squareSumLocal = squareSumBuf_.Get<float>();
            __ubuf__ D_T* src = reinterpret_cast<__ubuf__ D_T*>(inputLocal.GetPhyAddr());
            __ubuf__ float* sumDst = reinterpret_cast<__ubuf__ float*>(sumLocal.GetPhyAddr());
            __ubuf__ float* squareSumDst = reinterpret_cast<__ubuf__ float*>(squareSumLocal.GetPhyAddr());
            const uint16_t repeats = static_cast<uint16_t>((channelCount + static_cast<int64_t>(kSmallRRepF32) - 1) /
                                                           static_cast<int64_t>(kSmallRRepF32));
            asc_vf_call<SmallRFusedStatsVfImpl<D_T>>(src, sumDst, squareSumDst, static_cast<uint32_t>(channelCount),
                                                     static_cast<uint32_t>(tileChannelsAlign_),
                                                     static_cast<uint16_t>(reduceLen_), repeats);

            SetFlag<HardEvent::V_MTE3>(vToMte3);
            WaitFlag<HardEvent::V_MTE3>(vToMte3);
            DataCopyExtParams outParams = {};
            outParams.blockCount = 1;
            outParams.blockLen = static_cast<uint32_t>(channelCount * static_cast<int64_t>(sizeof(float)));
            outParams.srcStride = 0;
            outParams.dstStride = 0;
            outParams.rsv = 0;
            DataCopyPad(sumGm_[channelStart], sumLocal, outParams);
            DataCopyPad(squareSumGm_[channelStart], squareSumLocal, outParams);
            PipeBarrier<PIPE_ALL>();
        }
    }

private:
    __aicore__ inline void CopyInTranspose(int64_t channelStart, int64_t channelCount)
    {
        static constexpr MultiCopyConfig config = {false};
        MultiCopyLoopInfo<2> loopInfo;
        loopInfo.loopSrcStride[0] = 1;
        loopInfo.loopSrcStride[1] = reduceLen_;
        loopInfo.loopDstStride[0] = tileChannelsAlign_;
        loopInfo.loopDstStride[1] = 1;
        loopInfo.loopSize[0] = reduceLen_;
        loopInfo.loopSize[1] = channelCount;
        MultiCopyParams<D_T, 2> params = {loopInfo, 0};
        auto inputLocal = inputBuf_.Get<D_T>();
        DataCopy<D_T, 2, config>(inputLocal, xGm_[channelStart * reduceLen_], params);
    }

    int64_t channels_ = 0;
    int64_t reduceLen_ = 0;
    int64_t tileChannels_ = 0;
    int64_t tileChannelsAlign_ = 0;
    int64_t tileCount_ = 0;
    int64_t bigCoreLoopCount_ = 0;
    int64_t smallCoreLoopCount_ = 0;
    int32_t bigCoreCount_ = 0;
    int32_t usedCoreCount_ = 0;
    GlobalTensor<D_T> xGm_;
    GlobalTensor<float> sumGm_;
    GlobalTensor<float> squareSumGm_;
    TPipe pipe_;
    TBuf<TPosition::VECCALC> inputBuf_;
    TBuf<TPosition::VECCALC> sumBuf_;
    TBuf<TPosition::VECCALC> squareSumBuf_;
};

} // namespace NsBNTrainingReduce

#endif // !defined(__NPU_HOST__)

#endif // BN_TRAINING_REDUCE_SMALL_R_H_

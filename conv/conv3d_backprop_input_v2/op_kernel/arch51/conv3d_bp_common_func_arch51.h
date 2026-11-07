/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file conv3d_bp_common_func_arch51.h
 * \brief
 */

#ifndef CONV3D_BP_COMMON_FUNC_ARCH51_H
#define CONV3D_BP_COMMON_FUNC_ARCH51_H

namespace Convolution3DBackpropFunc {

using TypeFalse = struct {
    __uint128_t _[1024];
};

static __aicore__ inline void CalcL0KIdx(uint32_t& l0aKIdx, uint32_t& l0bKIdx, uint32_t curStepKa, uint32_t curStepKb)
{
    if (unlikely(l0aKIdx + 1 == curStepKa)) {
        l0aKIdx = 0;
    } else {
        ++l0aKIdx;
    }
    if (unlikely(l0bKIdx + 1 == curStepKb)) {
        l0bKIdx = 0;
    } else {
        ++l0bKIdx;
    }
}

template <class Intf>
static __aicore__ inline void ComputeL1A(Intf* self, uint64_t kIdx, int64_t curDoutIdx, uint32_t l0aKIdx)
{
    bool loadFlag = false;
    if (self->ctx.tiling_->al1Pbuffer > 1) {
        loadFlag = (kIdx == 0);
    } else {
        loadFlag = (l0aKIdx == 0);
    }
    Convolution3DBackpropFunc::LoadToA1<Intf, typename Intf::SrcAT>(self, kIdx, curDoutIdx, loadFlag);
}

template <class Intf>
static __aicore__ inline void ComputeL1B(Intf* self, uint64_t kIdx, int64_t curKdIdx, uint32_t l0bKIdx)
{
    bool loadFlag = false;
    if (self->ctx.tiling_->bl1Pbuffer > 1) {
        loadFlag = (kIdx == 0);
    } else {
        loadFlag = (l0bKIdx == 0);
    }
    Convolution3DBackpropFunc::LoadToB1<Intf, typename Intf::SrcBT>(self, kIdx, curKdIdx, loadFlag);
}

template <class Intf>
static __aicore__ inline void ComputeL1APreLoad(Intf* self, uint64_t kIdx, int64_t curDoutIdx, uint32_t l0aKIdx)
{
    if (self->ctx.tiling_->al1Pbuffer <= 1) {
        return;
    }
    Convolution3DBackpropFunc::LoadToA1<Intf, typename Intf::SrcAT>(self, self->ctx.curStepKa_ + kIdx, curDoutIdx,
                                                                    l0aKIdx == 0);
}

template <class Intf>
static __aicore__ inline void ComputeL1BPreLoad(Intf* self, uint64_t kIdx, int64_t curKdIdx, uint32_t l0bKIdx)
{
    if (self->ctx.tiling_->bl1Pbuffer <= 1) {
        return;
    }
    Convolution3DBackpropFunc::LoadToB1<Intf, typename Intf::SrcBT>(self, self->ctx.curStepKb_ + kIdx, curKdIdx,
                                                                    l0bKIdx == 0);
}

template <class Intf>
static __aicore__ inline void ComputeBTBias(Intf* self)
{
    if (self->ctx.tiling_->isBiasFullLoad) {
        return;
    }
    Convolution3DBackpropFunc::LoadBiasToBT<Intf>(self);
}

template <class Intf, bool hasBias>
static __aicore__ inline void ComputeL0BForTQueData(Intf* self, uint32_t l0bKIdx, uint32_t curStepKb,
                                                    LocalTensor<typename Intf::SrcBT>& l0b)
{
    if (unlikely(l0bKIdx == 0 &&
                 (!self->ctx.isB1FullLoadFlag_ || (self->ctx.isB1FullLoadFlag_ && self->ctx.isLoadB1_)))) {
        self->ctx.isLoadB1_ = false;
        self->ctx.cacheB1Buf_ = self->ctx.inQueL1B_.template DeQue<typename Intf::SrcBT>();
    }

    LoadToB2<Intf>(self, self->ctx.cacheB1Buf_, l0bKIdx, l0b);

    if ((l0bKIdx == curStepKb - 1) &&
        (!self->ctx.isB1FullLoadFlag_ || (self->ctx.isB1FullLoadFlag_ && self->ctx.isFreeB1_))) {
        self->ctx.isLoadB1_ = true;
        self->ctx.inQueL1B_.FreeTensor(self->ctx.cacheB1Buf_);
        if (hasBias && (!self->ctx.tiling_->isBiasFullLoad)) {
            self->ctx.biasBTQue_.FreeTensor(self->ctx.biasBTBuf_);
        }
    }
}

template <class Intf, bool hasBias>
static __aicore__ inline void ComputeL0B(Intf* self, uint32_t l0bKIdx, uint64_t curKdIdx, uint32_t curStepKb,
                                         LocalTensor<typename Intf::SrcBT>& l0b, uint64_t kIdx)
{
    UpdateLoadToB2ParamsK<Intf>(self);

    ComputeL0BForTQueData<Intf, hasBias>(self, l0bKIdx, curStepKb, l0b);
}

template <class Intf>
static __aicore__ inline void ComputeL0A(Intf* self, uint32_t l0aKIdx)
{
    if (unlikely(l0aKIdx == 0)) {
        self->ctx.cacheA1Buf_ = self->ctx.inQueL1A_.template DeQue<typename Intf::SrcAT>();
    }
}

template <class Intf, bool hasBias>
static __aicore__ inline void ComputeForKIter(Intf* self, LocalTensor<typename Intf::SrcAT>& l0a,
                                              LocalTensor<typename Intf::SrcBT>& l0b,
                                              LocalTensor<typename Intf::L0cT>& l0c, uint64_t curInnerKdIdx,
                                              int64_t curDoutIdx, bool isFirstDk, uint8_t& l0PingPongFlag)
{
    uint32_t curStepKa = self->ctx.curStepKa_;
    uint32_t curStepKb = self->ctx.curStepKb_;
    uint32_t l0aKIdx = 0;
    uint32_t l0bKIdx = 0;
    for (uint64_t kIdx = 0; kIdx < self->ctx.kIter_; kIdx++) {
        UpdateL1KParams<Intf>(self, kIdx, curStepKa, curStepKb);
        ComputeL1A<Intf>(self, kIdx, curDoutIdx, l0aKIdx);
        ComputeL1B<Intf>(self, kIdx, curInnerKdIdx, l0bKIdx);
        ComputeL1APreLoad<Intf>(self, kIdx, curDoutIdx, l0aKIdx);
        ComputeL1BPreLoad<Intf>(self, kIdx, curInnerKdIdx, l0bKIdx);
        if (hasBias) {
            ComputeBTBias<Intf>(self);
        }
        WaitFlag<HardEvent::M_MTE1>(l0PingPongFlag & 1);
        l0b = self->ctx.l0bBuf_.template Get<typename Intf::SrcBT>();
        if (l0PingPongFlag) {
            l0b = l0b[self->ctx.l0bPingPongAddr_];
        }
        ComputeL0A<Intf>(self, l0aKIdx);
        ComputeL0B<Intf, hasBias>(self, l0bKIdx, curInnerKdIdx, curStepKb, l0b, kIdx);
        SetFlag<HardEvent::MTE1_M>(l0PingPongFlag);
        WaitFlag<HardEvent::MTE1_M>(l0PingPongFlag);
        self->ctx.curKIdx_ = static_cast<uint32_t>(kIdx);
        ConvToL1Local<Intf, hasBias>(self, self->ctx.cacheA1Buf_, l0b);
        self->ctx.l1ResPingPongFlag_++;
        if (l0aKIdx == curStepKa - 1) {
            self->ctx.inQueL1A_.FreeTensor(self->ctx.cacheA1Buf_);
        }

        SetFlag<HardEvent::M_MTE1>(l0PingPongFlag);
        l0PingPongFlag ^= self->ctx.enableL0PingPong_;
        CalcL0KIdx(l0aKIdx, l0bKIdx, curStepKa, curStepKb);
    }
}

template <class Intf, bool hasBias>
static __aicore__ inline void ComputeForBias(Intf* self, LocalTensor<typename Intf::SrcBT>& l0b,
                                             uint8_t& l0PingPongFlag)
{
    ComputeBTBias<Intf>(self);
    LocalTensor<typename Intf::SrcAT> a1ForBias = self->ctx.inQueL1A_.template AllocTensor<typename Intf::SrcAT>();
    InitZeroValue<Intf, typename Intf::SrcAT>(self, a1ForBias);
    l0b = self->ctx.l0bBuf_.template Get<typename Intf::SrcBT>();
    if (l0PingPongFlag) {
        l0b = l0b[self->ctx.l0bPingPongAddr_];
    }
    SetFlag<HardEvent::MTE1_M>(l0PingPongFlag);
    WaitFlag<HardEvent::MTE1_M>(l0PingPongFlag);
    self->ctx.curKIdx_ = 0;
    ConvParams<Intf>(self);
    auto quantPre = GetQuantPre<Intf, typename Intf::DstT>(self);
    self->ctx.convFixParams_.SetPreQuantMode(MapQuantModeToPreQuant(quantPre));
    ConvStrideBitModeParams strideParams;
    const uint32_t hiCal = DivCeil(self->ctx.baseUseM_ + self->ctx.load3d_.mStartPt, self->ctx.tiling_->wi);
    strideParams.SetLoop3DstStride(self->ctx.tiling_->wi);
    strideParams.SetLoop4DstStride(self->ctx.tiling_->wi * hiCal);
    SetConvStridePara(strideParams);
    if constexpr (hasBias) {
        const uint8_t biasFlag = self->ctx.l1ResPingPongFlag_ & 1;
        constexpr uint32_t halfResBytes = V516_L1_RES_HALF_BYTES;
        uint32_t biasOffset = biasFlag * halfResBytes / sizeof(typename Intf::DstT);
        if (self->ctx.fixmEventSeen_[biasFlag]) {
            WaitFlag<HardEvent::FIX_M>(biasFlag);
            self->ctx.fixmEventSeen_[biasFlag] = false;
        }
        ConvFix<typename Intf::DstT, typename Intf::SrcAT, typename Intf::SrcBT, typename Intf::L0cT>(
            self->ctx.wholeL1ResTensor_[biasOffset], a1ForBias, l0b, self->ctx.biasBTBuf_, self->ctx.convFixParams_);
        SetFlag<HardEvent::M_FIX>(biasFlag);
        self->ctx.l1ResPingPongFlag_++;
    }
    self->ctx.inQueL1A_.FreeTensor(a1ForBias);
    SetFlag<HardEvent::M_MTE1>(l0PingPongFlag);
    l0PingPongFlag ^= self->ctx.enableL0PingPong_;
}

template <class Intf>
__aicore__ inline uint32_t CalFmapH(Intf* self, uint32_t mL1Size)
{
    uint32_t hiCal;
    if (mL1Size >= self->ctx.tiling_->wi) {
        hiCal = mL1Size / self->ctx.tiling_->wi;
        if (mL1Size != hiCal * self->ctx.tiling_->wi) {
            hiCal += 2; // 多搬运首尾两行
        }
    } else {
        hiCal = self->ctx.tiling_->wi % mL1Size == 0 ? 1 : 2; // 不能整除要跨行搬运
    }
    return (hiCal - 1) + self->ctx.tiling_->khDilation;
}

template <class Intf>
static __aicore__ inline void UpdateCurHoSizeCore(Intf* self, uint32_t endHoIdx)
{
    if (self->ctx.curHoIdx_ < static_cast<int32_t>(self->ctx.hoExpand_) && endHoIdx > 0) {
        uint32_t realStartHo = self->ctx.curHoIdx_ < 0 ? 0 : self->ctx.curHoIdx_;
        uint32_t realEndHoIdx = endHoIdx > self->ctx.hoExpand_ ? self->ctx.hoExpand_ : endHoIdx;
        self->ctx.curHoSize_ = realEndHoIdx - realStartHo;
    } else {
        self->ctx.needComputeFlag_ = false;
        self->ctx.curHoSize_ = 0;
    }
}

} // namespace Convolution3DBackpropFunc

#endif // CONV3D_BP_COMMON_FUNC_ARCH51_H

/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file conv3d_bp_common_func.h
 * \brief
 */

#ifndef CONV3D_BP_COMMON_FUNC_ADVANCE_H
#define CONV3D_BP_COMMON_FUNC_ADVANCE_H

#include "conv3d_bp_util.h"
#include "basic_api/kernel_basic_intf.h"
#include "../conv3d_backprop_input_v2/conv3d_backprop_input_v2_tiling_data.h"
#include "../../conv3d_backprop_input_v2_arch35_tiling_key.h"

#if defined(__DAV_C310__) || defined(__DAV_310R6__)
#include "impl/dav_v310/conv_bp_sub_func.h"
#endif
namespace Convolution3DBackpropFunc {

using TypeFalse = struct {
    __uint128_t _[1024];
};

static __aicore__ inline void CalcL0KIdx(uint32_t &l0aKIdx, uint32_t &l0bKIdx,
    uint32_t curStepKa, uint32_t curStepKb)
{
    // 用局部变量更新取代频繁的除法和取余操作
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
__aicore__ inline uint32_t CalFmapH(Intf *self, uint32_t mL1Size)
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
    uint32_t curHk = self->ctx.tiling_->hk;
    if constexpr (Intf::conv3dConfig.kernelSplitMode == TPL_SPLIT_KERNEL_H) {
        curHk = self->ctx.splitHkList_[0];
    }

    if constexpr (Intf::conv3dConfig.loadB1Condition != TPL_GM_TO_L1) {
        curHk = 1;
    }

    uint32_t khDilation = (curHk - 1) * self->ctx.tiling_->dilationH + 1;
    return (hiCal - 1) + khDilation;
}

template <class Intf>
static __aicore__ inline void ComputeL1A(Intf *self, uint64_t kIdx, int64_t curDoutIdx, uint32_t l0aKIdx)
{
    if ASCEND_IS_AIV {
        return;
    }
    bool loadFlag = false;
    if (self->ctx.tiling_->al1Pbuffer > 1) {
        loadFlag = (kIdx == 0);
    } else {
        loadFlag = (l0aKIdx == 0);
    }
    Convolution3DBackpropFunc::LoadToA1<Intf, typename Intf::SrcT>(self, kIdx, curDoutIdx, loadFlag);
}

template <class Intf>
static __aicore__ inline void ComputeL1B(Intf *self, uint64_t kIdx, int64_t curKdIdx, uint32_t l0bKIdx)
{
    bool loadFlag = false;
    if (self->ctx.tiling_->bl1Pbuffer > 1) {
        loadFlag = (kIdx == 0);
    } else {
        loadFlag = (l0bKIdx == 0);
    }
    Convolution3DBackpropFunc::LoadToB1<Intf, typename Intf::SrcT>(self, kIdx, curKdIdx, loadFlag);
}

template <class Intf>
static __aicore__ inline void ComputeL1APreLoad(Intf *self, uint64_t kIdx, int64_t curDoutIdx, uint32_t l0aKIdx)
{
    if ASCEND_IS_AIV {
        return;
    }
    if (self->ctx.tiling_->al1Pbuffer <= 1) {
        return;
    }
    Convolution3DBackpropFunc::LoadToA1<Intf, typename Intf::SrcT>(self, self->ctx.curStepKa_ + kIdx,
        curDoutIdx, l0aKIdx == 0);
}

template <class Intf>
static __aicore__ inline void ComputeL1BPreLoad(Intf *self, uint64_t kIdx, int64_t curKdIdx, uint32_t l0bKIdx)
{
    if (self->ctx.tiling_->bl1Pbuffer <= 1) {
        return;
    }
    Convolution3DBackpropFunc::LoadToB1<Intf, typename Intf::SrcT>(self, self->ctx.curStepKb_ + kIdx,
        curKdIdx, l0bKIdx == 0);
}

#if defined(__DAV_310R6__)
template <class Intf>
static __aicore__ inline void ComputeBTBias(Intf *self)
{
    if (self->ctx.tiling_->isBiasFullLoad) {
        return;
    }
    Convolution3DBackpropFunc::LoadBiasToBT<Intf>(self);
}
#endif

template <class Intf>
static __aicore__ inline void ComputeL0AForKernelSplit(Intf *self, uint32_t l0aKIdx,
    uint64_t curKdIdx, uint32_t curStepKa, LocalTensor<typename Intf::SrcT> &l0a)
{
    if (unlikely(l0aKIdx == 0 && (!self->ctx.isA1FullLoadFlag_ || (self->ctx.isA1FullLoadFlag_ &&
        self->ctx.isLoadA1_)))) {
        self->ctx.isLoadA1_ = false;
        self->ctx.cacheA1Buf_ = self->ctx.inQueL1A_.template DeQue<typename Intf::SrcT>();
    }

    LoadToA2<Intf>(self, self->ctx.cacheA1Buf_, l0a);

    if ((l0aKIdx == curStepKa - 1) && (!self->ctx.isA1FullLoadFlag_ ||
        (self->ctx.isA1FullLoadFlag_ && self->ctx.isFreeA1_))) {
        self->ctx.isLoadA1_ = true;
        self->ctx.inQueL1A_.FreeTensor(self->ctx.cacheA1Buf_);
    }
}

template <class Intf>
static __aicore__ inline void ComputeL0A(Intf *self, uint32_t l0aKIdx, uint32_t curStepKa, LocalTensor<typename Intf::SrcT> &l0a)
{
    if (unlikely(l0aKIdx == 0)) {
        self->ctx.cacheA1Buf_ = self->ctx.inQueL1A_.template DeQue<typename Intf::SrcT>();
    }

    LoadToA2<Intf>(self, self->ctx.cacheA1Buf_, l0a);

    if (l0aKIdx == curStepKa - 1) {
        self->ctx.inQueL1A_.FreeTensor(self->ctx.cacheA1Buf_);
    }
}

template <class Intf>
static __aicore__ inline void ComputeL0BForTQueData(Intf *self, uint32_t l0bKIdx,
        uint32_t curStepKb, LocalTensor<typename Intf::SrcT> &l0b)
{
    if (unlikely(l0bKIdx == 0 && (!self->ctx.isB1FullLoadFlag_ || (self->ctx.isB1FullLoadFlag_ &&
        self->ctx.isLoadB1_)))) {
        self->ctx.isLoadB1_ = false;
        self->ctx.cacheB1Buf_ = self->ctx.inQueL1B_.template DeQue<typename Intf::SrcT>();
    }

    LoadToB2<Intf>(self, self->ctx.cacheB1Buf_, l0bKIdx, l0b);

    if ((l0bKIdx == curStepKb - 1) && (!self->ctx.isB1FullLoadFlag_ ||
        (self->ctx.isB1FullLoadFlag_ && self->ctx.isFreeB1_))) {
        self->ctx.isLoadB1_ = true;
        self->ctx.inQueL1B_.FreeTensor(self->ctx.cacheB1Buf_);

#if defined(__DAV_310R6__)
        if (!self->ctx.tiling_->isBiasFullLoad) {
            self->ctx.biasBTQue_.FreeTensor(self->ctx.biasBTBuf_);
        }
#endif
    }
}

template <class Intf>
static __aicore__ inline void ComputeL0BForTBufData(Intf *self, uint32_t l0bKIdx,
    uint64_t curKdIdx, uint32_t curStepKb, LocalTensor<typename Intf::SrcT> &l0b, uint64_t kIdx)
{
    if (unlikely(l0bKIdx == 0 && (!self->ctx.isB1FullLoadFlag_ ||
        (self->ctx.isB1FullLoadFlag_ && self->ctx.isLoadB1_)))) {
        self->ctx.isLoadB1_ = false;
        if ASCEND_IS_AIC {
            WaitForVecBeforeLoadToB2<Intf>(self);
        }
        self->ctx.cacheB1Buf_ = GetB1Tbuf<Intf>(self, kIdx);
    }

    if ASCEND_IS_AIC {
        LoadToB2<Intf>(self, self->ctx.cacheB1Buf_, l0bKIdx, l0b);
    }

    if ((l0bKIdx == curStepKb - 1) && (!self->ctx.isB1FullLoadFlag_ ||
        (self->ctx.isB1FullLoadFlag_ && self->ctx.isFreeB1_))) {
        self->ctx.isLoadB1_ = true;
        if ASCEND_IS_AIC {
            NotifyVecAfterLoadToB2<Intf>(self);
        }
    }
}

template <class Intf>
static __aicore__ inline void ComputeL0B(Intf *self, uint32_t l0bKIdx,
    uint64_t curKdIdx, uint32_t curStepKb, LocalTensor<typename Intf::SrcT> &l0b, uint64_t kIdx)
{
    if constexpr (Intf::conv3dConfig.kernelSplitMode == TPL_SPLIT_KERNEL_HW) {
        if (self->ctx.kSCoutFullLoad_) {
            UpdateLoadToB2ParamsKForKernelSplit<Intf>(self, l0bKIdx);
        }
    } else {
        UpdateLoadToB2ParamsK<Intf>(self, l0bKIdx);
    }

    if (IsL1bUseTQue<Intf>(self)) {
        if ASCEND_IS_AIV {
            return;
        }
        ComputeL0BForTQueData<Intf>(self, l0bKIdx, curStepKb, l0b);
    } else {
        ComputeL0BForTBufData<Intf>(self, l0bKIdx, curKdIdx, curStepKb, l0b, kIdx);
    }
}

template <class Intf>
static __aicore__ inline void ComputeForKIter(Intf *self, LocalTensor<typename Intf::SrcT> &l0a,
    LocalTensor<typename Intf::SrcT> &l0b, LocalTensor<typename Intf::L0cT> &l0c,
    uint64_t curInnerKdIdx, int64_t curDoutIdx, bool isFirstDk, uint8_t &l0PingPongFlag)
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
#if defined(__DAV_310R6__)
        ComputeBTBias<Intf>(self);
#endif

        if ASCEND_IS_AIC {
            WaitFlag<HardEvent::M_MTE1>(l0PingPongFlag & 1);
            l0a = self->ctx.l0aBuf_.template Get<typename Intf::SrcT>();
            l0b = self->ctx.l0bBuf_.template Get<typename Intf::SrcT>();
            if (l0PingPongFlag) {
                l0a = l0a[self->ctx.l0aPingPongAddr_];
                l0b = l0b[self->ctx.l0bPingPongAddr_];
            }
            UpdateLoadToA2ParamsK<Intf>(self, l0aKIdx);
            // kernel拆分后，A1全载处理
            if constexpr (Intf::conv3dConfig.kernelSplitMode == TPL_SPLIT_KERNEL_HW) {
                ComputeL0AForKernelSplit<Intf>(self, l0aKIdx, curInnerKdIdx, curStepKa, l0a);
            } else {
                ComputeL0A<Intf>(self, l0aKIdx, curStepKa, l0a);
            }
        }

        ComputeL0B<Intf>(self, l0bKIdx, curInnerKdIdx, curStepKb, l0b, kIdx);
        if ASCEND_IS_AIC {
            SetFlag<HardEvent::MTE1_M>(l0PingPongFlag);
            WaitFlag<HardEvent::MTE1_M>(l0PingPongFlag);
            CalcParamsMmad<Intf>(self, kIdx, isFirstDk);
            MmadLocal<Intf>(self, l0a, l0b, l0c);
            SetFlag<HardEvent::M_MTE1>(l0PingPongFlag);

            l0PingPongFlag ^= self->ctx.enableL0PingPong_;
        }
        CalcL0KIdx(l0aKIdx, l0bKIdx, curStepKa, curStepKb);
    }
}

template <class Intf>
static __aicore__ inline bool CalcCurDoutIdx(Intf *self, uint64_t curInnerKdIdx, int64_t &curDoutIdx)
{
    // 由于膨胀卷积使dk的位置发生改变，求解dout_idx时，dk_idx需乘上膨胀系数再参与计算，才能求取正确的索引
    int64_t dTmp = self->ctx.curDinIdx_ + self->ctx.tiling_->padFront - curInnerKdIdx * self->ctx.tiling_->dilationD;
    if (dTmp < 0 || (self->ctx.tiling_->strideD > 1 && (dTmp % self->ctx.tiling_->strideD > 0)) ||
        dTmp >= self->ctx.tiling_->dout * self->ctx.tiling_->strideD) {
        return false;
    }

    // 当代码走到这，说明continue未生效，需要加载L1 L0参与mmad计算
    curDoutIdx = dTmp;
    if (self->ctx.tiling_->strideD > 1) {
        curDoutIdx = dTmp / self->ctx.tiling_->strideD;
    }

    return true;
}

template <class Intf>
static __aicore__ inline void UpdateCurHoSizeCore(Intf *self, uint32_t endHoIdx)
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
}  // namespace Convolution3DBackpropFunc
#endif
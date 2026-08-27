/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file conv_bp_compute.h
 * \brief
 */

#ifndef CONV_BP_COMPUTE_H
#define CONV_BP_COMPUTE_H

#include "conv_bp_config_base.h"
#include "conv_bp_util.h"
#include "kernel_operator.h"
#include "../conv3d_backprop_filter_v2_tiling_data.h"
#if __CCE_AICORE__ == 220
#include "conv_bp_sub_func.h"
#endif
namespace ConvolutionBackpropFunc {

template <class Intf>
__aicore__ inline void CalcParamsL12L0b(Intf* self, uint64_t kPos)
{
    // load3dStepK
    self->ctx.load3dB_.kExtension = self->ctx.baseUseN_;
    // posK
    uint32_t localN = ShiftDivChannelSize<Intf>(self->ctx.tiling_->baseN, self->ctx.tiling_->channelSize);
    uint32_t localUseN = ShiftDivChannelSize<Intf>(self->ctx.baseUseN_, self->ctx.tiling_->channelSize);
    uint32_t kStartLocal = RemainderOfHkWk(self->ctx.curNL1Idx_ * localN, self->ctx.hwK_) +
                           RemainderStepN(self->ctx.curNL0Idx_, self->ctx.tiling_->stepN) * localN;
    self->ctx.load3dB_.kStartPt = kStartLocal * self->ctx.tiling_->channelSize;
    self->ctx.load3dB_.channelSize = CeilHkWk(kStartLocal + localUseN, self->ctx.hwK_) * self->ctx.tiling_->channelSize;
}

template <class Intf>
__aicore__ inline void CalcParamsL12L0a(Intf* self, uint64_t kPos)
{
    uint32_t alignedBaseUseM = ShiftCeilM0(self->ctx.baseUseM_, self->ctx.tiling_->m0) * self->ctx.tiling_->m0;
    self->ctx.load3dA_.kExtension = alignedBaseUseM;
    self->ctx.load3dA_.channelSize = alignedBaseUseM;
}

template <class Intf>
__aicore__ inline void LoadL12L0b(Intf* self, const LocalTensor<typename Intf::SrcT>& l1BMatrix,
                                  LocalTensor<typename Intf::SrcT>& l0b)
{
    static constexpr IsResetLoad3dConfig LOAD3D_CONFIG_220 = {false, true};
    SetFmatrix(self->ctx.load3dB_.l1H, self->ctx.load3dB_.l1W, self->ctx.load3dB_.padList, FmatrixMode::FMATRIX_RIGHT);
    LoadData<typename Intf::SrcT, LOAD3D_CONFIG_220>(l0b, l1BMatrix, self->ctx.load3dB_);
}

template <class Intf>
__aicore__ inline void FreeA1Tensor(Intf* self, bool a1PingPongFlag)
{
    if (a1PingPongFlag) {
        self->ctx.a1Ping_.FreeTensor(self->ctx.cacheA1BufPing_);
#ifdef ASCENDC_CPU_DEBUG
        self->ctx.cacheA1BufPing_.SetSize(0);
#endif
    } else {
        self->ctx.a1Pong_.FreeTensor(self->ctx.cacheA1BufPong_);
#ifdef ASCENDC_CPU_DEBUG
        self->ctx.cacheA1BufPong_.SetSize(0);
#endif
    }
}

template <class Intf>
__aicore__ inline void FreeB1Tensor(Intf* self, bool b1PingPongFlag)
{
    if (b1PingPongFlag) {
        self->ctx.b1Ping_.FreeTensor(self->ctx.cacheB1BufPing_);
#ifdef ASCENDC_CPU_DEBUG
        self->ctx.cacheB1BufPing_.SetSize(0);
#endif
    } else {
        self->ctx.b1Pong_.FreeTensor(self->ctx.cacheB1BufPong_);
#ifdef ASCENDC_CPU_DEBUG
        self->ctx.cacheB1BufPong_.SetSize(0);
#endif
    }
}

template <class Intf>
__aicore__ inline void Compute(Intf* self, Out2L1ScalarParams& out2L1Params)
{
    CalcParamsL12L0b<Intf>(self, 0);
    CalcParamsL12L0a<Intf>(self, 0);
    CalcParamsMmad<Intf>(self, 0);
    LocalTensor<typename Intf::SrcT> l0a;
    LocalTensor<typename Intf::SrcT> l0b;
    LocalTensor<typename Intf::L0cT> l0c;
    uint32_t curML0IdxModstepKaMulBaseM = RemainderStepM(self->ctx.curML0Idx_, self->ctx.tiling_->stepM) *
                                          self->ctx.tiling_->baseM;
    constexpr uint32_t l0aPingPongAddr = TOTAL_L0A_SIZE / 2 / sizeof(typename Intf::SrcT);
    constexpr uint32_t l0bPingPongAddr = TOTAL_L0B_SIZE / 2 / sizeof(typename Intf::SrcT);
    CalOut2L1ScalarParams(self, out2L1Params);

    if (self->ctx.l0cPingPongFlag_) {
        l0c = self->ctx.l0cPing_.template AllocTensor<typename Intf::L0cT>();
    } else {
        l0c = self->ctx.l0cPong_.template AllocTensor<typename Intf::L0cT>();
    }

    bool a1PingPongFlag = true;
    bool b1PingPongFlag = true;
    bool isAL1PingPong = self->ctx.tiling_->al1Pbuffer > 1;
    bool isBL1PingPong = self->ctx.tiling_->bl1Pbuffer > 1;
    uint32_t kaIdx = 0;
    uint32_t kbIdx = 0;
    uint64_t kaStepIdx = 0;
    uint64_t kbStepIdx = 0;
    uint64_t curMKL1Idx = self->ctx.stepKaRound * DivStepM(self->ctx.curML1Idx_, self->ctx.tiling_->stepM);
    uint64_t curNKL1Idx = self->ctx.stepKbRound * DivStepN(self->ctx.curNL1Idx_, self->ctx.tiling_->stepN);

    for (uint64_t k = 0; k < self->ctx.kIter_; k++) {
        bool isLastKIter = k + 1 == self->ctx.kIter_;
        bool isLastStepKa = kaIdx + 1 == self->ctx.tiling_->stepKa;
        bool isLastStepKb = kbIdx + 1 == self->ctx.tiling_->stepKb;
        bool isLoadA1 = kaIdx == 0;
        bool isLoadB1 = kbIdx == 0;
        self->ctx.baseUseK_ = isLastKIter ? self->ctx.tailK_ : self->ctx.tiling_->baseK;

        /*
        通过M*K的奇偶判断load到L1A ping还是L1A pong, BL1同理
                    kL1Idx=0  kL1Idx=1 kL1Idx=2
                    ----------------------------
        mL1Idx=0    |  ping  |  pong  |  ping  |
                    ----------------------------
        mL1Idx=1    |  pong  |  ping  |  pong  |
                    ----------------------------
        mL1Idx=2    |  ping  |  pong  |  ping  |
                    ----------------------------
        */
        if (self->ctx.tiling_->bl1Pbuffer > 1) {
            b1PingPongFlag = (curNKL1Idx + kbStepIdx + 1) & 1;
        }
        ConvolutionBackpropFunc::LoadToB1<Intf, typename Intf::SrcT>(self, b1PingPongFlag, k, out2L1Params, isLoadB1,
                                                                     kbStepIdx);

        if (self->ctx.tiling_->al1Pbuffer > 1) {
            a1PingPongFlag = (curMKL1Idx + kaStepIdx + 1) & 1;
        }
        ConvolutionBackpropFunc::LoadToA1<Intf, typename Intf::SrcT>(self, a1PingPongFlag, k, out2L1Params, isLoadA1,
                                                                     kaStepIdx);

        WaitFlag<HardEvent::M_MTE1>(self->ctx.l0aPingPongFlag_);

        uint32_t alignedBaseUseK = ShiftCeilChannelSize<Intf>(self->ctx.baseUseK_, self->ctx.tiling_->k0) *
                                   self->ctx.tiling_->k0;
        l0b = self->ctx.l0bBuf_.template Get<typename Intf::SrcT>();
        if (self->ctx.l0aPingPongFlag_) {
            l0b = l0b[l0bPingPongAddr];
        }
        // posM
        self->ctx.load3dB_.mStartPt = (k - kbIdx) * self->ctx.tiling_->baseK % self->ctx.tiling_->wo +
                                      kbIdx * self->ctx.tiling_->baseK;
        // load3dStepM
        self->ctx.load3dB_.mExtension = alignedBaseUseK;
        if (unlikely(out2L1Params.isLoad2L1B && isLoadB1)) {
            if (b1PingPongFlag) {
                self->ctx.cacheB1BufPing_ = self->ctx.b1Ping_.template DeQue<typename Intf::SrcT>();
            } else {
                self->ctx.cacheB1BufPong_ = self->ctx.b1Pong_.template DeQue<typename Intf::SrcT>();
            }
        }

        if (b1PingPongFlag) {
            self->ctx.load3dB_.l1H = self->ctx.bL1HiCopyLenPing;
            self->ctx.load3dB_.padList[2] = self->ctx.bL1PadUpPing;
            LoadL12L0b<Intf>(self, self->ctx.cacheB1BufPing_, l0b);
        } else {
            self->ctx.load3dB_.l1H = self->ctx.bL1HiCopyLenPong;
            self->ctx.load3dB_.padList[2] = self->ctx.bL1PadUpPong;
            LoadL12L0b<Intf>(self, self->ctx.cacheB1BufPong_, l0b);
        }

        if (out2L1Params.isFreeBL1 && (isLastStepKb || isLastKIter)) {
            FreeB1Tensor(self, b1PingPongFlag);
        }

        l0a = self->ctx.l0aBuf_.template Get<typename Intf::SrcT>();
        if (self->ctx.l0aPingPongFlag_) {
            l0a = l0a[l0aPingPongAddr];
        }
        uint32_t mOffset = curML0IdxModstepKaMulBaseM * self->ctx.curLoadKal1_;
        self->ctx.srcL12L0aOffset_ = kaIdx * self->ctx.tiling_->baseK * self->ctx.tiling_->channelSize + mOffset;
        self->ctx.load3dA_.mExtension = alignedBaseUseK;
        if (unlikely(out2L1Params.isLoad2L1A && isLoadA1)) {
            if (a1PingPongFlag) {
                self->ctx.cacheA1BufPing_ = self->ctx.a1Ping_.template DeQue<typename Intf::SrcT>();
            } else {
                self->ctx.cacheA1BufPong_ = self->ctx.a1Pong_.template DeQue<typename Intf::SrcT>();
            }
        }

        if (a1PingPongFlag) {
            LoadL12L0a<Intf>(self, self->ctx.cacheA1BufPing_, k, l0a, out2L1Params, kaStepIdx);
        } else {
            LoadL12L0a<Intf>(self, self->ctx.cacheA1BufPong_, k, l0a, out2L1Params, kaStepIdx);
        }

        if (out2L1Params.isFreeAL1 && (isLastStepKa || isLastKIter)) {
            FreeA1Tensor(self, a1PingPongFlag);
        }

        SetFlag<HardEvent::MTE1_M>(self->ctx.l0aPingPongFlag_);
        WaitFlag<HardEvent::MTE1_M>(self->ctx.l0aPingPongFlag_);
        self->ctx.mmad_.cmatrixInitVal = k == 0;
        self->ctx.mmad_.k = self->ctx.baseUseK_;
        MmadLocal<Intf>(self, l0a, l0b, l0c);
        SetFlag<HardEvent::M_MTE1>(self->ctx.l0aPingPongFlag_);

        self->ctx.l0aPingPongFlag_ ^= self->ctx.useL0PingPong_;
        if (isLastStepKa) {
            ++kaStepIdx;
            kaIdx = 0;
        } else {
            ++kaIdx;
        }
        if (isLastStepKb) {
            ++kbStepIdx;
            kbIdx = 0;
        } else {
            ++kbIdx;
        }
    }

    if (self->ctx.l0cPingPongFlag_) {
        self->ctx.l0cPing_.EnQue(l0c);
    } else {
        self->ctx.l0cPong_.EnQue(l0c);
    }
}
} // namespace ConvolutionBackpropFunc
#endif

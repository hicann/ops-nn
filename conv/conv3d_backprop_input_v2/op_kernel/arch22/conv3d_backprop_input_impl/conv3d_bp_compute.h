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
 * \file conv3d_bp_compute.h
 * \brief
 */

#ifndef CONV3D_BP_COMPUTE_H
#define CONV3D_BP_COMPUTE_H

#include "../conv3d_backprop_input_v2_tiling_data.h"
#include "conv3d_bp_util.h"
#include "kernel_operator.h"

#if __CCE_AICORE__ == 220
#include "conv_bp_sub_func.h"
#include "conv3d_bp_kernel_split.h"
#endif
namespace Convolution3DBackpropFunc {

template <class Intf>
static __aicore__ inline void Compute(Intf* self)
{
    // 先刷新h方向的值，方便判断是否为有效计算
    UpdateLoadToA2ParamsM<Intf>(self);

    // 在跳过计算逻辑中，如果有部分无需跳过的操作逻辑。后续如果有类似逻辑，可以在此处继续增加
    // 当前已存在的操作为isFreeB1_为true的情况(B1全加载且循环至最后一块运算空间，需要释放B1空间)。此时预期的只有ping空间被使用
    if (!self->ctx.needComputeFlag_) {
        if (self->ctx.isFreeB1_ && !self->ctx.isLoadB1_) {
            self->ctx.b1Ping_.FreeTensor(self->ctx.cacheB1BufPing_);
        }
        return;
    }

    if ASCEND_IS_AIV {
        return;
    }

    InitMmadParams<Intf>(self);
    if constexpr ((std::is_same<typename Intf::SrcT, bfloat16_t>::value) ||
                  (std::is_same<typename Intf::SrcT, half>::value)) {
        if (unlikely(self->ctx.curNL0Idx_ == 0 || self->ctx.curNL0Idx_ + 1 == self->ctx.nIter_)) {
            UpdateLoadToB2ParamsN<Intf>(self);
        }
    }

    bool isFirstDk = true;
    bool a1PingPongFlag = true;
    bool b1PingPongFlag = true;
    LocalTensor<typename Intf::SrcT> l0a;
    LocalTensor<typename Intf::SrcT> l0b;
    LocalTensor<typename Intf::L0cT> l0c;
    uint8_t l0aPingPongFlag = self->ctx.l0aPingPongFlag_;
    constexpr uint32_t l0aPingPongAddr = TOTAL_L0A_SIZE / 2 / sizeof(typename Intf::SrcT);
    constexpr uint32_t l0bPingPongAddr = TOTAL_L0B_SIZE / 2 / sizeof(typename Intf::SrcT);

    if (self->ctx.usingCacheC1Ping_) {
        l0c = self->ctx.c1Ping_.template AllocTensor<typename Intf::L0cT>();
    } else {
        l0c = self->ctx.c1Pong_.template AllocTensor<typename Intf::L0cT>();
    }
    self->ctx.needComputeKFlag_ = false;
    for (uint64_t curKdIdx = 0; curKdIdx < self->ctx.tiling_->dk; curKdIdx++) {
        int64_t dTmp = 0;
        if (unlikely(self->ctx.tiling_->strideD > self->ctx.tiling_->dk)) {
            dTmp = self->ctx.curDinIdx_ + self->ctx.tiling_->padFront;
            if (CalcRemainder(dTmp, self->ctx.tiling_->strideD) >= self->ctx.tiling_->dk ||
                dTmp / self->ctx.tiling_->strideD >= self->ctx.tiling_->dout) {
                continue;
            }
        } else {
            // 由于膨胀卷积使dk的位置发生改变，求解dout_idx时，dk_idx需乘上膨胀系数再参与计算，才能求取正确的索引
            dTmp = self->ctx.curDinIdx_ + self->ctx.tiling_->padFront - curKdIdx * self->ctx.tiling_->dilationD;
            if (dTmp < 0 || CalcRemainder(dTmp, self->ctx.tiling_->strideD) > 0 ||
                dTmp >= self->ctx.tiling_->dout * self->ctx.tiling_->strideD) {
                continue;
            }
        }
        self->ctx.needComputeKFlag_ = true;
        int64_t curDoutIdx = dTmp;
        if (self->ctx.tiling_->strideD > 1) {
            curDoutIdx = dTmp / self->ctx.tiling_->strideD;
        }

        uint32_t kaIdx = 0;
        uint32_t kbIdx = 0;
        uint64_t kaStepIdx = 0;
        uint64_t kbStepIdx = 0;
        self->ctx.load3d_.kExtension = self->ctx.tiling_->baseK;
        self->ctx.mmad_.k = self->ctx.tiling_->baseK;
        self->ctx.curLoadKal1_ = self->ctx.tiling_->stepKa * self->ctx.tiling_->baseK;
        self->ctx.curLoadKbl1_ = self->ctx.tiling_->stepKb * self->ctx.tiling_->baseK;
        for (uint64_t kIdx = 0; kIdx < self->ctx.kIter_; kIdx++) {
            bool isLastKIdx = (kIdx + 1 == self->ctx.kIter_);
            if (isLastKIdx) {
                self->ctx.load3d_.kExtension = self->ctx.tailK_;
                self->ctx.mmad_.k = self->ctx.tailK_;
            }
            if (kIdx == self->ctx.kIterStepKaTail) {
                self->ctx.curLoadKal1_ = (self->ctx.stepKaTail - 1) * self->ctx.tiling_->baseK + self->ctx.tailK_;
            }
            if (kIdx == self->ctx.kIterStepKbTail) {
                self->ctx.curLoadKbl1_ = (self->ctx.stepKbTail - 1) * self->ctx.tiling_->baseK + self->ctx.tailK_;
            }
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
            bool isLoadA1 = kaIdx == 0;
            if (isLoadA1 && self->ctx.tiling_->al1Pbuffer > 1) {
                // 此处默认stepM = 1
                a1PingPongFlag = ((self->ctx.curML1Idx_ * self->ctx.stepKaRound_ + kaStepIdx + 1) & 1);
            }
            Convolution3DBackpropFunc::LoadToA1<Intf, typename Intf::SrcT>(self, kIdx, curDoutIdx, a1PingPongFlag,
                                                                           isLoadA1);

            bool isLoadB1 = kbIdx == 0;
            if (isLoadB1 && self->ctx.tiling_->bl1Pbuffer > 1) {
                // 此处默认stepN = 1
                b1PingPongFlag = ((self->ctx.curNL1Idx_ * self->ctx.stepKbRound_ + kbStepIdx + 1) & 1);
            }
            Convolution3DBackpropFunc::LoadToB1<Intf, typename Intf::SrcT>(self, kIdx, curKdIdx, b1PingPongFlag,
                                                                           isLoadB1);

            l0a = self->ctx.l0aBuf_.template Get<typename Intf::SrcT>();
            if (l0aPingPongFlag) {
                l0a = l0a[l0aPingPongAddr];
            }

            self->ctx.load3d_.kStartPt = kaIdx * self->ctx.tiling_->baseK;

            if (unlikely(isLoadA1)) {
                if (a1PingPongFlag) {
                    self->ctx.cacheA1BufPing_ = self->ctx.a1Ping_.template DeQue<typename Intf::SrcT>();
                } else {
                    self->ctx.cacheA1BufPong_ = self->ctx.a1Pong_.template DeQue<typename Intf::SrcT>();
                }
            }

            WaitFlag<HardEvent::M_MTE1>(l0aPingPongFlag);
            if (a1PingPongFlag) {
                LoadToA2<Intf>(self, self->ctx.cacheA1BufPing_, l0a);
            } else {
                LoadToA2<Intf>(self, self->ctx.cacheA1BufPong_, l0a);
            }

            bool isLastStepKa = kaIdx + 1 == self->ctx.tiling_->stepKa;
            if (isLastStepKa || isLastKIdx) {
                if (a1PingPongFlag) {
                    self->ctx.a1Ping_.FreeTensor(self->ctx.cacheA1BufPing_);
                } else {
                    self->ctx.a1Pong_.FreeTensor(self->ctx.cacheA1BufPong_);
                }
            }

            l0b = self->ctx.l0bBuf_.template Get<typename Intf::SrcT>();
            if (l0aPingPongFlag) {
                l0b = l0b[l0bPingPongAddr];
            }

            if (unlikely(isLoadB1 &&
                         (!self->ctx.isB1FullLoadFlag_ || (self->ctx.isB1FullLoadFlag_ && self->ctx.isLoadB1_)))) {
                if (b1PingPongFlag) {
                    self->ctx.cacheB1BufPing_ = self->ctx.b1Ping_.template DeQue<typename Intf::SrcT>();
                } else {
                    self->ctx.cacheB1BufPong_ = self->ctx.b1Pong_.template DeQue<typename Intf::SrcT>();
                }
                if (self->ctx.isLoadB1_) {
                    self->ctx.isLoadB1_ = false;
                }
            }

            if constexpr ((std::is_same<typename Intf::SrcT, bfloat16_t>::value) ||
                          (std::is_same<typename Intf::SrcT, half>::value)) {
                if (unlikely(kIdx == 0 || kIdx == self->ctx.kIterStepKbTail)) {
                    UpdateLoadToB2ParamsK<Intf>(self);
                }
            }
            if (b1PingPongFlag) {
                LoadToB2<Intf>(self, self->ctx.cacheB1BufPing_, kbIdx, kIdx, b1PingPongFlag, l0b);
            } else {
                LoadToB2<Intf>(self, self->ctx.cacheB1BufPong_, kbIdx, kIdx, b1PingPongFlag, l0b);
            }

            bool isLastStepKb = kbIdx + 1 == self->ctx.tiling_->stepKb;
            if ((isLastStepKb || isLastKIdx) &&
                (!self->ctx.isB1FullLoadFlag_ || (self->ctx.isB1FullLoadFlag_ && self->ctx.isFreeB1_))) {
                if (b1PingPongFlag) {
                    self->ctx.b1Ping_.FreeTensor(self->ctx.cacheB1BufPing_);
                } else {
                    self->ctx.b1Pong_.FreeTensor(self->ctx.cacheB1BufPong_);
                }
            }

            SetFlag<HardEvent::MTE1_M>(l0aPingPongFlag);
            WaitFlag<HardEvent::MTE1_M>(l0aPingPongFlag);

            MmadLocal<Intf>(self, l0a, l0b, l0c);
            SetFlag<HardEvent::M_MTE1>(l0aPingPongFlag);
            if (unlikely(isFirstDk && kIdx == 0)) {
                self->ctx.mmad_.cmatrixInitVal = 0;
            }

            l0aPingPongFlag ^= self->ctx.useL0PingPong_;
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
        isFirstDk = false;
    }
    if (self->ctx.usingCacheC1Ping_) {
        self->ctx.c1Ping_.EnQue(l0c);
    } else {
        self->ctx.c1Pong_.EnQue(l0c);
    }
    self->ctx.l0aPingPongFlag_ = l0aPingPongFlag;

    if constexpr (Intf::conv3dConfig.enableKernelSplit) {
        SetFlag<HardEvent::M_FIX>(EVENT_ID2);
        WaitFlag<HardEvent::M_FIX>(EVENT_ID2);
        SetFlag<HardEvent::FIX_M>(EVENT_ID2);
        WaitFlag<HardEvent::FIX_M>(EVENT_ID2);
    }
}
} // namespace Convolution3DBackpropFunc
#endif

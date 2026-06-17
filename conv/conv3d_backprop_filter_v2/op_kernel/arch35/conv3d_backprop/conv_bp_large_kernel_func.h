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
 * \file conv_bp_large_kernel_func.h
 * \brief
 */

#ifndef CONV_BP_LARGE_KERNEL_FUNC_H
#define CONV_BP_LARGE_KERNEL_FUNC_H

#include "conv_bp_config_base.h"
#include "conv_bp_util.h"
#include "basic_api/kernel_basic_intf.h"
#include "../conv3d_backprop_filter_v2/conv3d_backprop_filter_v2_tiling_data.h"
#include "conv_bp_func_common.h"

namespace ConvolutionBackpropFunc {
template <class Intf>
__aicore__ inline void updateParasForSplitKernelHW(
    Intf* self, Out2L1ScalarParams& out2L1Params, uint32_t startWo, uint64_t out2B1SrcAddrStart, uint32_t wkIdx)
{
    int64_t padLeft = 0;
    int64_t padRight = 0;
    int64_t leftValidAddrOffset = 0;
    int64_t b1SrcWiLeftOffGm = static_cast<int64_t>(startWo) * self->ctx.tiling_->strideW - self->ctx.tiling_->padLeft +
                               wkIdx * self->ctx.tiling_->dilationW;
    if (b1SrcWiLeftOffGm < 0) {
        padLeft = -b1SrcWiLeftOffGm;
    }
    leftValidAddrOffset = Ceil(padLeft, self->ctx.tiling_->strideW) * self->ctx.tiling_->strideW + b1SrcWiLeftOffGm;

    int64_t b1SrcWiRightOffGm = static_cast<int64_t>(startWo + self->ctx.singleShapeWo_) * self->ctx.tiling_->strideW +
                                self->ctx.strideKernelDilationW - (self->ctx.tiling_->padLeft + self->ctx.tiling_->wi);
    if (b1SrcWiRightOffGm > 0) {
        padRight = b1SrcWiRightOffGm;
    }
    padRight = padRight - (self->ctx.tiling_->wk - wkIdx - 1) * self->ctx.tiling_->dilationW;

    padLeft = padLeft < 0 ? (0) : Ceil(padLeft, self->ctx.tiling_->strideW);
    padRight = padRight < 0 ? (0) : Ceil(padRight, self->ctx.tiling_->strideW);

    if (Intf::Config::xType::format == ConvolutionBackprop::CubeFormat::NCDHW) {
        if (padLeft) {
            out2L1Params.out2B1SrcAddr = out2B1SrcAddrStart + leftValidAddrOffset;
        } else {
            out2L1Params.out2B1SrcAddr = out2B1SrcAddrStart + b1SrcWiLeftOffGm;
        }
    } else if (Intf::Config::xType::format == ConvolutionBackprop::CubeFormat::NDHWC) {
        if (padLeft) {
            out2L1Params.out2B1SrcAddr = out2B1SrcAddrStart + leftValidAddrOffset * self->ctx.tiling_->cin;
        } else {
            out2L1Params.out2B1SrcAddr = out2B1SrcAddrStart + b1SrcWiLeftOffGm * self->ctx.tiling_->cin;
        }
    }

    out2L1Params.singleShapeWi = self->ctx.singleShapeWo_;
    if (out2L1Params.singleShapeWi > (padLeft + padRight)) {
        self->ctx.load3d_.l1W = out2L1Params.singleShapeWi - padLeft - padRight;
    } else {
        self->ctx.load3d_.l1W = 0;
    }
    // singleShapeWi已经被限制到256，因此padLeft和padRight最大可能为256，
    // 当padLeft或者padRight大于uint8的最大值255，此时l1w等于0，因此padList值溢出也不影响结果
    self->ctx.load3d_.padList[0] = padLeft;
    self->ctx.load3d_.padList[1] = padRight;
}

template <class Intf>
__aicore__ inline void initParasSplitKernelHW(Intf* self)
{
    // 矩阵计算的值，默认为baseUseN_，baseUseN_可能等于baseN或TailN，但都一定是n0对齐的,splitkernel场景，
    //  每次只循环一个hkwk=1,因此一定是n0
    self->ctx.mmad_.n = self->ctx.tiling_->n0;

    // kExtension是N轴，由于切了Kernel，不管是fp16和32(两个C0的Wk连续)场景均为16,正好一个n0单元;
    self->ctx.load3d_.kExtension = self->ctx.tiling_->n0;
    self->ctx.load3d_.kStartPt = 0; // stepM=stepN=1，每次N都是从0开始读取

    self->ctx.load3d_.channelSize = 16; // cin等于16，避免循环cin1
    self->ctx.load3d_.filterW = 1;
    self->ctx.load3d_.filterH = 1;

    self->ctx.load3d_.dilationFilterW = 1;
    self->ctx.load3d_.dilationFilterH = 1;

    self->ctx.load3d_.filterSizeW = false;
    self->ctx.load3d_.filterSizeH = false;

    // 跳着搬运数据，stride等价于1
    self->ctx.load3d_.strideW = 1;
    self->ctx.load3d_.strideH = 1;
}

template <class Intf>
__aicore__ inline void getHWkIdx(Intf* self, uint64_t hwkLoopIdx, uint64_t& hkIdx, uint64_t& wkIdx)
{
    hkIdx = hwkLoopIdx / self->ctx.tiling_->wk;
    wkIdx = hwkLoopIdx % self->ctx.tiling_->wk;
}

template <class Intf>
__aicore__ inline void ComputeSplitKernelHW(Intf* self, Out2L1ScalarParams& out2L1Params)
{
    if ASCEND_IS_AIV {
        return;
    }
    LocalTensor<typename Intf::L0cT> l0c;
    if (self->ctx.l0cPingPongFlag_) {
        l0c = self->ctx.l0cPing_.template AllocTensor<typename Intf::L0cT>();
    } else {
        l0c = self->ctx.l0cPong_.template AllocTensor<typename Intf::L0cT>();
    }
    ClearBaseMNL0C<Intf>(self, l0c);

    CalcParamsL12L0a<Intf>(self);
    //todo: baseUseM_=1会走到特殊的处理分支，使用Gemv实现mmad,当前按照m0进行对其，走通用分支使用GEMM，后续需要评估特殊分支是否有性能收益决定特殊分支是否有必要存在；
    uint32_t baseUseMBak = self->ctx.baseUseM_;
    if (self->ctx.baseUseM_ == 1) {
        self->ctx.baseUseM_ = ShiftCeilM0(self->ctx.baseUseM_, self->ctx.tiling_->m0) * self->ctx.tiling_->m0;
        self->ctx.mmad_.m = self->ctx.baseUseM_;
        CalcParamsL12L0a<Intf>(self);
    }
    LocalTensor<typename Intf::SrcT> l0a;
    LocalTensor<typename Intf::SrcT> l0b;

    constexpr uint32_t l0aPingPongAddr = TOTAL_L0A_SIZE / 2 / sizeof(typename Intf::SrcT);
    constexpr uint32_t l0bPingPongAddr = TOTAL_L0B_SIZE / 2 / sizeof(typename Intf::SrcT);

    CalcParamsL12L0b<Intf>(self);
    CalcParamsMmad<Intf>(self);
    CalOut2L1ScalarParams(self, out2L1Params);
    bool isFirstMmad = true;
    uint64_t dstL0cOffsetBase = self->ctx.dstL0cOffset_;
    // 基本块模板中stepN一定等于1，此处使用curNL1Idx和curNL0Idx均可
    uint64_t usedN = self->ctx.curNIdx_ * self->ctx.tiling_->baseN;
    uint64_t hwkLoopStart = usedN / self->ctx.tiling_->n0;
    uint64_t hwkLoopEnd = (usedN + self->ctx.baseUseN_) / self->ctx.tiling_->n0;
    uint64_t hkIdx = 0;
    uint64_t wkIdx = 0;
    for (uint64_t hwkLoopIdx = hwkLoopStart; hwkLoopIdx < hwkLoopEnd; hwkLoopIdx++) {
        getHWkIdx(self, hwkLoopIdx, hkIdx, wkIdx);
        self->ctx.dstL0cOffset_ =
            dstL0cOffsetBase + (hwkLoopIdx - hwkLoopStart) * self->ctx.tiling_->baseM * self->ctx.tiling_->n0;
        initParasSplitKernelHW(self);
        isFirstMmad = true;

        uint64_t out2A1BatchDoutSrcAddrStart = out2L1Params.out2A1SrcAddr;
        uint64_t out2B1BatchDoutSrcAddrStart = out2L1Params.out2B1SrcAddr;
        uint64_t batchDoutEndIdx = self->ctx.batchDoutStartIdx_ + self->ctx.singleShapeBatch_;
        for (uint64_t batchDoutIdx = self->ctx.batchDoutStartIdx_; batchDoutIdx < batchDoutEndIdx; batchDoutIdx++) {
            bool skipCurrentDinCompute = false; // dinIdx小于padFront或大于din+padFront则跳过本轮计算
            UpdateSrcAddrBaseOnBatchDoutIdx<Intf>(self, batchDoutIdx, out2L1Params, skipCurrentDinCompute);
            if (skipCurrentDinCompute) {
                continue;
            }

            const int32_t splitWo = self->ctx.tiling_->splitWo;
            uint64_t out2A1SrcAddrStart = out2L1Params.out2A1SrcAddr;
            uint64_t out2B1SrcAddrStart = out2L1Params.out2B1SrcAddr;

            int32_t woIterateTimes = 1;
            calculateWoIterTimes(self, woIterateTimes, splitWo);
            for (int32_t splitWoIdx = 0; splitWoIdx < woIterateTimes; splitWoIdx++) {
                updateSingleShapeWoI(self, out2L1Params, woIterateTimes, splitWoIdx, splitWo);
                if (unlikely(self->ctx.isSplitWo_)) {
                    updateParasForSplitW(
                        self, out2L1Params, splitWoIdx * splitWo, out2A1SrcAddrStart, out2B1SrcAddrStart);
                }
                updateParasForSplitKernelHW(self, out2L1Params, splitWoIdx * splitWo, out2B1SrcAddrStart, wkIdx);
                if (!self->ctx.load3d_.l1W) {
                    PipeBarrier<PIPE_ALL>();
                    continue;
                }
                bool a1PingPongFlag = true;
                bool b1PingPongFlag = true;
                bool isAL1PingPong = self->ctx.tiling_->al1Pbuffer > 1;
                bool isBL1PingPong = self->ctx.tiling_->bl1Pbuffer > 1;
                uint32_t kaIdx = 0;
                uint32_t kbIdx = 0;
                uint64_t kaStepIdx = 0;
                uint64_t kbStepIdx = 0;
                uint64_t curMKL1Idx = self->ctx.stepKaRound * self->ctx.curMIdx_;
                uint64_t curNKL1Idx = self->ctx.stepKbRound * self->ctx.curNIdx_;

                bool skipCurrentHiCompute = false;
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
                    if (isBL1PingPong) {
                        b1PingPongFlag = (curNKL1Idx + kbStepIdx + 1) & 1;
                    }

                    if (isLoadB1) {
                        LoadToB1SplitKernelHW<Intf, typename Intf::SrcT>(
                            self, b1PingPongFlag, out2L1Params, kbStepIdx, hkIdx, skipCurrentHiCompute);
                    }
                    if (skipCurrentHiCompute) {
                        UpdateIdx(isLastStepKa, isLastStepKb, kaIdx, kbIdx, kaStepIdx, kbStepIdx);
                        continue;
                    }
                    if (isAL1PingPong) {
                        a1PingPongFlag = (curMKL1Idx + kaStepIdx + 1) & 1;
                    }
                    ConvolutionBackpropFunc::LoadToA1<Intf, typename Intf::SrcT>(
                        self, a1PingPongFlag, k, out2L1Params, isLoadA1, kaStepIdx);

                    WaitFlag<HardEvent::M_MTE1>(self->ctx.l0aPingPongFlag_ & 1);

                    l0b = self->ctx.l0bBuf_.template Get<typename Intf::SrcT>();
                    if (self->ctx.l0aPingPongFlag_) {
                        l0b = l0b[l0bPingPongAddr];
                    }
                    // posM
                    self->ctx.load3d_.mStartPt = (k - kbIdx) * self->ctx.tiling_->baseK % self->ctx.singleShapeWo_ +
                                                 kbIdx * self->ctx.tiling_->baseK;

                    if (unlikely(out2L1Params.isLoad2L1B && isLoadB1)) {
                        if (b1PingPongFlag) {
                            self->ctx.cacheB1BufPing_ = self->ctx.b1Ping_.template DeQue<typename Intf::SrcT>();
                        } else {
                            self->ctx.cacheB1BufPong_ = self->ctx.b1Pong_.template DeQue<typename Intf::SrcT>();
                        }
                    }

                    if (b1PingPongFlag) {
                        self->ctx.load3d_.l1H = self->ctx.bL1HiCopyLenPing;
                        self->ctx.load3d_.padList[2] = self->ctx.bL1PadUpPing;
                        LoadL12L0b<Intf>(self, self->ctx.cacheB1BufPing_, l0b);
                    } else {
                        self->ctx.load3d_.l1H = self->ctx.bL1HiCopyLenPong;
                        self->ctx.load3d_.padList[2] = self->ctx.bL1PadUpPong;
                        LoadL12L0b<Intf>(self, self->ctx.cacheB1BufPong_, l0b);
                    }
                    if (out2L1Params.isFreeBL1 && (isLastStepKb || isLastKIter)) {
                        FreeB1Tensor(self, b1PingPongFlag);
                    }

                    l0a = self->ctx.l0aBuf_.template Get<typename Intf::SrcT>();
                    if (self->ctx.l0aPingPongFlag_) {
                        l0a = l0a[l0aPingPongAddr];
                    }
                    if (unlikely(out2L1Params.isLoad2L1A && isLoadA1)) {
                        if (a1PingPongFlag) {
                            self->ctx.cacheA1BufPing_ = self->ctx.a1Ping_.template DeQue<typename Intf::SrcT>();
                        } else {
                            self->ctx.cacheA1BufPong_ = self->ctx.a1Pong_.template DeQue<typename Intf::SrcT>();
                        }
                    }
                    if (a1PingPongFlag) {
                        LoadL12L0a<Intf>(self, self->ctx.cacheA1BufPing_, k, l0a);
                    } else {
                        LoadL12L0a<Intf>(self, self->ctx.cacheA1BufPong_, k, l0a);
                    }

                    if (out2L1Params.isFreeAL1 && (isLastStepKa || isLastKIter)) {
                        FreeA1Tensor(self, a1PingPongFlag);
                    }

                    SetFlag<HardEvent::MTE1_M>(self->ctx.l0aPingPongFlag_);
                    WaitFlag<HardEvent::MTE1_M>(self->ctx.l0aPingPongFlag_);
                    // isFirstMmadd为true时，compute还没有被完整执行过，此时dkIdx第一次滑窗到din的有效位置上进行mmad计算，因此重置L0C
                    self->ctx.mmad_.cmatrixInitVal = isFirstMmad;
                    self->ctx.mmad_.k = self->ctx.baseUseK_;
                    MmadLocal<Intf>(self, l0a, l0b, l0c);
                    isFirstMmad = false;
                    SetFlag<HardEvent::M_MTE1>(self->ctx.l0aPingPongFlag_);

                    self->ctx.l0aPingPongFlag_ ^= self->ctx.useL0PingPong_;
                    UpdateIdx(isLastStepKa, isLastStepKb, kaIdx, kbIdx, kaStepIdx, kbStepIdx);
                }
            }
            out2L1Params.out2A1SrcAddr = out2A1SrcAddrStart;
            out2L1Params.out2B1SrcAddr = out2B1SrcAddrStart;
        }
        // batchout 偏移后的地址要还原回来
        out2L1Params.out2A1SrcAddr = out2A1BatchDoutSrcAddrStart;
        out2L1Params.out2B1SrcAddr = out2B1BatchDoutSrcAddrStart;
    }
    self->ctx.dstL0cOffset_ = dstL0cOffsetBase;
    self->ctx.baseUseM_ = baseUseMBak;
    if (self->ctx.l0cPingPongFlag_) {
        self->ctx.l0cPing_.EnQue(l0c);
    } else {
        self->ctx.l0cPong_.EnQue(l0c);
    }
}
} // namespace ConvolutionBackpropFunc
#endif
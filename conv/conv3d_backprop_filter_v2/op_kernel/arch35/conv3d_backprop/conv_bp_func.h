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
 * \file conv_bp_func.h
 * \brief
 */

#ifndef CONV_BP_FUNC_H
#define CONV_BP_FUNC_H

#include "conv_bp_config_base.h"
#include "conv_bp_util.h"
#include "kernel_operator.h"
#include "../conv3d_backprop_filter_v2/conv3d_backprop_filter_v2_tiling_data.h"
#include "impl/dav_v310/conv_bp_sub_func.h"
#include "impl/dav_v310/conv_bp_sub_func_deterministic_calculate.h"
#include "conv_bp_large_kernel_func.h"
#include "conv_bp_func_common.h"

DECLARE_CHECK_IMPL(Init);
DECLARE_CHECK_IMPL(SetFmap);
DECLARE_CHECK_IMPL(SetOutBackprop);
DECLARE_CHECK_IMPL(SetSingleShape);
DECLARE_CHECK_IMPL(SetStartIdx);
DECLARE_CHECK_IMPL(SetDeterministicCoreInfo);
DECLARE_CHECK_SYNC_IMPL(DeterministicReduceKInUb);
DECLARE_CHECK_SYNC_IMPL(UpdateMNIdx);
DECLARE_CHECK_SYNC_IMPL(Compute);
DECLARE_CHECK_SYNC_IMPL(Iterate);
DECLARE_CHECK_SYNC_IMPL(IterateAll);
DECLARE_CHECK_SYNC_IMPL(GetTensorC);
DECLARE_CHECK_SYNC_IMPL(VecPostProcess);
DECLARE_CHECK_IMPL(End);
namespace ConvolutionBackpropFunc {

using TypeFalse = struct {
    __uint128_t _[1024];
};

enum class IterateOrder {
    ORDER_M = 0,
    ORDER_N,
    UNDEF,
};

template <class Intf>
__aicore__ inline void CheckTiling(Intf *self)
{
#ifdef __CCE_KT_TEST__
    ascendc_assert((self->ctx.tiling_->batch > 0),
        "orignal batch is %d , which should be larger than 0", self->ctx.tiling_->batch);
    ascendc_assert((self->ctx.tiling_->cin > 0),
        "orignal cin is %d , which should be larger than 0", self->ctx.tiling_->cin);
    ascendc_assert((self->ctx.tiling_->cout > 0),
        "orignal cout is %d , which should be larger than 0", self->ctx.tiling_->cout);
    ascendc_assert((self->ctx.tiling_->cin1G > 0),
        "orignal cin1G is %d , which should be larger than 0", self->ctx.tiling_->cin1G);
    ascendc_assert((self->ctx.tiling_->cout1G > 0),
        "orignal cout1G is %d , which should be larger than 0", self->ctx.tiling_->cout1G);
    ascendc_assert((self->ctx.tiling_->dout > 0),
        "orignal dout is %d , which should be larger than 0", self->ctx.tiling_->dout);
    ascendc_assert((self->ctx.tiling_->ho > 0),
        "orignal ho is %d , which should be larger than 0", self->ctx.tiling_->ho);
    ascendc_assert((self->ctx.tiling_->wo > 0),
        "orignal wo is %d , which should be larger than 0", self->ctx.tiling_->wo);
    ascendc_assert((self->ctx.tiling_->di > 0),
        "orignal di is %d , which should be larger than 0", self->ctx.tiling_->di);
    ascendc_assert((self->ctx.tiling_->hi > 0),
        "orignal hi is %d , which should be larger than 0", self->ctx.tiling_->hi);
    ascendc_assert((self->ctx.tiling_->wi > 0),
        "orignal wi is %d , which should be larger than 0", self->ctx.tiling_->wi);
    ascendc_assert((self->ctx.tiling_->dk > 0),
        "orignal dk is %d , which should be larger than 0", self->ctx.tiling_->dk);
    ascendc_assert((self->ctx.tiling_->hk > 0),
        "orignal hk is %d , which should be larger than 0", self->ctx.tiling_->hk);
    ascendc_assert((self->ctx.tiling_->wk > 0),
        "orignal wk is %d , which should be larger than 0", self->ctx.tiling_->wk);
    ascendc_assert((self->ctx.tiling_->singleCoreBatch > 0),
        "singleCoreBatch is %d , which should be larger than 0", self->ctx.tiling_->singleCoreBatch);
    ascendc_assert((self->ctx.tiling_->singleCoreCout > 0),
        "singleCoreCout is %d , which should be larger than 0", self->ctx.tiling_->singleCoreCout);
    ascendc_assert((self->ctx.tiling_->singleCoreHo > 0),
        "singleCoreHo is %d , which should be larger than 0", self->ctx.tiling_->singleCoreHo);
    ascendc_assert((self->ctx.tiling_->singleCoreCin > 0),
        "singleCoreCin is %d , which should be larger than 0", self->ctx.tiling_->singleCoreCin);
    ascendc_assert((self->ctx.tiling_->baseM > 0),
        "baseM is %d , which should be larger than 0", self->ctx.tiling_->baseM);
    ascendc_assert((self->ctx.tiling_->baseK > 0),
        "baseK is %d , which should be larger than 0", self->ctx.tiling_->baseK);
    ascendc_assert((self->ctx.tiling_->baseN > 0),
        "baseN is %d , which should be larger than 0", self->ctx.tiling_->baseN);
    ascendc_assert((self->ctx.tiling_->stepM > 0),
        "stepM is %d , which should be larger than 0", self->ctx.tiling_->stepM);
    ascendc_assert((self->ctx.tiling_->stepN > 0),
        "stepN is %d , which should be larger than 0", self->ctx.tiling_->stepN);
    ascendc_assert((self->ctx.tiling_->stepKa > 0),
        "stepKa is %d , which should be larger than 0", self->ctx.tiling_->stepKa);
    ascendc_assert((self->ctx.tiling_->stepKb > 0),
        "stepKb is %d , which should be larger than 0", self->ctx.tiling_->stepKb);
#endif
}

template <class Intf>
__aicore__ inline void InitStepMParams(Intf *self)
{
    self->ctx.mIter_ = Ceil(self->ctx.singleShapeCout_, self->ctx.tiling_->baseM);
    self->ctx.tailM_ = self->ctx.singleShapeCout_ - self->ctx.tiling_->baseM * (self->ctx.mIter_ - 1);
#ifdef __CCE_KT_TEST__
    ascendc_assert((self->ctx.mIter_ > 0),
        "self->ctx.mIter_ is %d , which should be larger than 0", self->ctx.mIter_);
#endif
}

template <class Intf>
__aicore__ inline void InitStepKParams(Intf *self)
{
    uint64_t singleCoreHoWo = static_cast<uint64_t>(self->ctx.singleShapeHo_) * self->ctx.tiling_->wo;
    uint64_t kIter = Ceil(singleCoreHoWo, self->ctx.tiling_->baseK);
    self->ctx.kIter_ = kIter;
    self->ctx.tailK_ = singleCoreHoWo - self->ctx.tiling_->baseK * (kIter - 1);
    self->ctx.stepKaRound = Ceil(kIter, self->ctx.tiling_->stepKa);
    self->ctx.stepKbRound = Ceil(kIter, self->ctx.tiling_->stepKb);
#ifdef __CCE_KT_TEST__
    ascendc_assert((self->ctx.kIter_ > 0),
        "self->ctx.kIter_ is %d , which should be larger than 0", self->ctx.kIter_);
#endif
}

template <class Intf>
__aicore__ inline void InitStepNParams(Intf *self)
{
    if (Intf::Config::xType::format == ConvolutionBackprop::CubeFormat::NDC1HWC0) {
        self->ctx.nIter_ = Ceil(ShiftCeilChannelSize<Intf>(self->ctx.singleShapeCin_, self->ctx.tiling_->channelSize) *
            self->ctx.tiling_->channelSize * self->ctx.hwK_, self->ctx.tiling_->baseN);
        self->ctx.tailN_ =
            self->ctx.singleShapeCin_ * self->ctx.hwK_ - self->ctx.tiling_->baseN * (self->ctx.nIter_ - 1);
    } else {
        uint64_t curNWithoutDk = ShiftCeilM0(self->ctx.singleShapeCin_, self->ctx.tiling_->n0) *
            self->ctx.tiling_->n0 * self->ctx.hwK_;
        self->ctx.nIter_ = Ceil(curNWithoutDk * self->ctx.singleShapeDk_, self->ctx.tiling_->baseN);
        self->ctx.tailN_ = curNWithoutDk % self->ctx.tiling_->baseN;
        bool baseNIncludeDkNoCin1Flag1 = (self->ctx.nIter_ > self->ctx.singleShapeDk_ &&
                                    self->ctx.nIter_ % self->ctx.singleShapeDk_ != 0);
        bool baseNIncludeDkNoCin1Flag2 = (self->ctx.nIter_ < self->ctx.singleShapeDk_ &&
                                    self->ctx.singleShapeDk_ % self->ctx.nIter_ != 0);
        // 当baseN上不包含cin1但包含全部的dk时，需手动干预，使baseN上先载cin后载dk
        if (baseNIncludeDkNoCin1Flag1 || baseNIncludeDkNoCin1Flag2) {
            self->ctx.nIter_ = Ceil(curNWithoutDk, self->ctx.tiling_->baseN) * self->ctx.singleShapeDk_;
        }
        // nIter_包含cinhkwk循环和dk循环，d上有padding时不计算
        self->ctx.cinHkWkLoop_ = Ceil(self->ctx.nIter_, self->ctx.singleShapeDk_);
        // 当SingleCoreDk_大于nIter_时，表明每次循环时SingleCoreDk_>1，否则SingleCoreDk_=1
        self->ctx.curSingleCoreDk_ = Ceil(self->ctx.singleShapeDk_, self->ctx.nIter_);
        self->ctx.tailN_ = self->ctx.tailN_ * self->ctx.curSingleCoreDk_;
    }
    if (self->ctx.tailN_ == 0) {
        self->ctx.tailN_ = self->ctx.tiling_->baseN;
    }
#ifdef __CCE_KT_TEST__
    ascendc_assert((self->ctx.nIter_ > 0),
        "self->ctx.nIter_ is %d , which should be larger than 0", self->ctx.nIter_);
#endif
}

template <class Intf>
__aicore__ inline void InitParamsPart2(Intf *self)
{
    self->ctx.stepKaRound = 0;
    self->ctx.stepKbRound = 0;
    self->ctx.curStepM_ = 0;
    self->ctx.curStepN_ = 0;

    self->ctx.tailN_ = 0;
    self->ctx.tailK_ = 0;
    self->ctx.tailM_ = 0;

    self->ctx.baseUseM_ = 0;
    self->ctx.baseUseN_ = 0;
    self->ctx.baseUseK_ = 0;

    self->ctx.curML0Idx_ = 0;
    self->ctx.curNL0Idx_ = 0;
    self->ctx.curML1Idx_ = 0;
    self->ctx.curNL1Idx_ = 0;

    self->ctx.mIter_ = 0;
    self->ctx.nIter_ = 0;
    self->ctx.kIter_ = 0;

#if defined(__DAV_C310__)
    self->ctx.enableStepNIncludeDkNocinhwk_ = false;
    self->ctx.enableStepNTail_ = false;
    self->ctx.isSplitWo_ = false;
    self->ctx.bL1cin1CopyLen = 0;
    self->ctx.singleShapeDk_ = 0;
    self->ctx.curSingleCoreDk_ = 0;
    self->ctx.cinHkWkLoop_ = 0;
    self->ctx.cinRemainLen_ = 0;
    self->ctx.nLoopCinRemainLen_ = 0;
    self->ctx.head_ = 0;
    self->ctx.tail_ = 0;
    self->ctx.nLoopHead_ = 0;
    self->ctx.lastNIdx_ = -1;
#endif
}
template <class Intf>
__aicore__ inline void InitParams(Intf *self)
{
    self->ctx.baseMN_ = self->ctx.tiling_->baseM * self->ctx.tiling_->baseN;
    self->ctx.hwK_ = static_cast<uint64_t>(self->ctx.tiling_->hk) * self->ctx.tiling_->wk;
    self->ctx.hwO_ = static_cast<uint64_t>(self->ctx.tiling_->ho) * self->ctx.tiling_->wo;
    self->ctx.hwI_ = static_cast<uint64_t>(self->ctx.tiling_->hi) * self->ctx.tiling_->wi;
    self->ctx.dhwK_ = self->ctx.hwK_ * self->ctx.tiling_->dk;
    self->ctx.dhwO_ = self->ctx.hwO_ * self->ctx.tiling_->dout;
    self->ctx.dhwI_ = self->ctx.hwI_ * self->ctx.tiling_->di;
    self->ctx.kal1_ = self->ctx.tiling_->stepKa * self->ctx.tiling_->baseK;
    self->ctx.kbl1_ = self->ctx.tiling_->stepKb * self->ctx.tiling_->baseK;

    // simplify the calculation of hi = (ho - 1) * strideh + (hk - 1) * dilationh + 1
    self->ctx.strideKernelDilationH = static_cast<int64_t>(self->ctx.tiling_->hk - 1) * self->ctx.tiling_->dilationH
        + 1 - self->ctx.tiling_->strideH;
    // simplify the calculation of wi = (wo - 1) * stridew + (wk - 1) * dilationw + 1
    self->ctx.strideKernelDilationW = static_cast<int64_t>(self->ctx.tiling_->wk - 1) * self->ctx.tiling_->dilationW
        + 1 - self->ctx.tiling_->strideW;
    self->ctx.isFirstIter_ = true;
    self->ctx.l0aPingPongFlag_ = 0;
    self->ctx.l0bPingPongFlag_ = 0;
    self->ctx.l0cPingPongFlag_ = 1;
    self->ctx.useL0PingPong_ = (self->ctx.tiling_->al0Pbuffer - 1) & (self->ctx.tiling_->bl0Pbuffer - 1);

    self->ctx.singleShapeWo_ = 0;
    self->ctx.singleShapeCout_ = 0;
    self->ctx.singleShapeBatch_ = 0;
    self->ctx.singleShapeCin_ = 0;
    self->ctx.singleShapeHo_ = 0;
    self->ctx.batchDoutStartIdx_ = 0;
    self->ctx.hoStartIdx_ = 0;
    self->ctx.hiStartIdx_ = 0;
    self->ctx.dkStartIdx_ = 0;
    InitParamsPart2<Intf>(self);
    InitLoadToB2Params<Intf>(self);
    InitLoadToA2Params<Intf>(self);
    InitSetFmatrixParams<Intf>(self);
    InitMmadParams<Intf>(self);
}

template <class Intf>
__aicore__ inline void InitTque(Intf *self)
{
#if defined(__DAV_C310__)
    // streamK场景，AIV需要申请UB空间，扩维场景AIC,AIV需要申请UB空间
    self->ctx.pipe_.InitBuffer(self->ctx.vecBuf_, AscendC::TOTAL_UB_SIZE);

#endif
    if ASCEND_IS_AIC {
        uint32_t al1BoundByteSize = self->ctx.tiling_->al1Bound * sizeof(typename Intf::SrcT);
        self->ctx.pipe_.InitBuffer(self->ctx.a1Ping_, 1, al1BoundByteSize);
        if (self->ctx.tiling_->al1Pbuffer > 1) {
            self->ctx.pipe_.InitBuffer(self->ctx.a1Pong_, 1, al1BoundByteSize);
        }

        uint32_t bl1BoundByteSize = self->ctx.tiling_->bl1Bound * sizeof(typename Intf::SrcT);
        self->ctx.pipe_.InitBuffer(self->ctx.b1Ping_, 1, bl1BoundByteSize);
        if (self->ctx.tiling_->bl1Pbuffer > 1) {
            self->ctx.pipe_.InitBuffer(self->ctx.b1Pong_, 1, bl1BoundByteSize);
        }

        uint32_t cMatrixByteSize = self->ctx.baseMN_ * sizeof(typename Intf::L0cT);
        self->ctx.pipe_.InitBuffer(self->ctx.l0cPing_, 1, cMatrixByteSize);
        if (self->ctx.tiling_->cl0Pbuffer > 1) {
            self->ctx.pipe_.InitBuffer(self->ctx.l0cPong_, 1, cMatrixByteSize);
        }

        self->ctx.pipe_.InitBuffer(self->ctx.l0aBuf_, TOTAL_L0A_SIZE);
        self->ctx.pipe_.InitBuffer(self->ctx.l0bBuf_, TOTAL_L0B_SIZE);
    }
}

template <class Intf>
__aicore__ inline void ComputeNormal(Intf *self, Out2L1ScalarParams& out2L1Params)
{
    if ASCEND_IS_AIV {
        return;
    }

    CalcParamsL12L0a<Intf>(self);
    CalcParamsL12L0b<Intf>(self);
    CalcParamsMmad<Intf>(self);
    LocalTensor<typename Intf::SrcT> l0a;
    LocalTensor<typename Intf::SrcT> l0b;
    LocalTensor<typename Intf::L0cT> l0c;
    constexpr uint32_t l0aPingPongAddr = TOTAL_L0A_SIZE / 2 / sizeof(typename Intf::SrcT);
    constexpr uint32_t l0bPingPongAddr = TOTAL_L0B_SIZE / 2 / sizeof(typename Intf::SrcT);
    CalOut2L1ScalarParams(self, out2L1Params);

    if (self->ctx.l0cPingPongFlag_) {
        l0c = self->ctx.l0cPing_.template AllocTensor<typename Intf::L0cT>();
    } else {
        l0c = self->ctx.l0cPong_.template AllocTensor<typename Intf::L0cT>();
    }
    uint64_t batchDoutEndIdx = self->ctx.batchDoutStartIdx_ + self->ctx.singleShapeBatch_;
    bool isFirstMmad = true;

    //todo: baseUseM_=1会走到特殊的处理分支，使用Gemv实现mmad,当前按照m0进行对其，走通用分支使用GEMM，后续需要评估特殊分支是否有性能收益决定特殊分支是否有必要存在；
    uint32_t baseUseMBak = self->ctx.baseUseM_;
    if (self->ctx.baseUseM_ == 1) {
        self->ctx.baseUseM_ = ShiftCeilM0(self->ctx.baseUseM_, self->ctx.tiling_->m0) * self->ctx.tiling_->m0;
        self->ctx.mmad_.m = self->ctx.baseUseM_;
        CalcParamsL12L0a<Intf>(self);
    }

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
            if (self->ctx.isSplitWo_) {
                updateParasForSplitW(self, out2L1Params, splitWoIdx * splitWo, out2A1SrcAddrStart, out2B1SrcAddrStart);
            }
            if (!self->ctx.load3d_.l1W) {
                    continue ;
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
                ConvolutionBackpropFunc::LoadToB1<Intf, typename Intf::SrcT>(self, b1PingPongFlag, out2L1Params,
                            isLoadB1, kbStepIdx, skipCurrentHiCompute);
                if (skipCurrentHiCompute) {
                    UpdateIdx(isLastStepKa, isLastStepKb, kaIdx, kbIdx, kaStepIdx, kbStepIdx);
                    continue;
                }
                if (isAL1PingPong) {
                    a1PingPongFlag = (curMKL1Idx + kaStepIdx + 1) & 1;
                }
                ConvolutionBackpropFunc::LoadToA1<Intf, typename Intf::SrcT>(self, a1PingPongFlag, k,
                                                                            out2L1Params, isLoadA1, kaStepIdx);

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
    if (isFirstMmad == true) {
        ClearBaseMNL0C<Intf>(self, l0c);
    }
    self->ctx.baseUseM_ = baseUseMBak;
    if (self->ctx.l0cPingPongFlag_) {
        self->ctx.l0cPing_.EnQue(l0c);
    } else {
        self->ctx.l0cPong_.EnQue(l0c);
    }
}

template <class Intf, bool sync>
struct Compute {
    DECLARE_DEFAULT_OVERLOADING_FUN(Intf, ConvolutionBackpropFunc);
    static __aicore__ inline void call(Intf* self, Out2L1ScalarParams& out2L1Params)
    {
        self->ctx.baseUseM_ =
            (self->ctx.curML0Idx_ + 1 == self->ctx.mIter_) ? self->ctx.tailM_ : self->ctx.tiling_->baseM;
        // 由于N循环可能包含cinhkwk循环和dk循环，因此，self->ctx.baseUseN_的判断条件需改变
        self->ctx.baseUseN_ =
            ((self->ctx.curNL0Idx_ + 1) % self->ctx.cinHkWkLoop_ == 0) ? self->ctx.tailN_ : self->ctx.tiling_->baseN;
        if (!self->ctx.tiling_->isSplitKernelHW) {
            ComputeNormal<Intf>(self, out2L1Params);
        } else {
            ComputeSplitKernelHW<Intf>(self, out2L1Params);
        }
    }
};

template <class Intf>
__aicore__ inline void UpdateIdxAndStep(Intf *self)
{
    self->ctx.curML0Idx_ = self->ctx.curML1Idx_;
    self->ctx.curNL0Idx_ = self->ctx.curNL1Idx_;
    self->ctx.curStepM_ = (self->ctx.mIter_ - self->ctx.curML0Idx_) > self->ctx.tiling_->stepM ?
        self->ctx.tiling_->stepM : (self->ctx.mIter_ - self->ctx.curML1Idx_);
    // 是否不是最后一次循环
    bool notLastLoop = (self->ctx.nIter_ - self->ctx.curNL0Idx_) > self->ctx.cinHkWkLoop_;

    uint32_t nL0RemainLen = self->ctx.nIter_ - self->ctx.curNL0Idx_;
    uint32_t nL1RemainLen = self->ctx.nIter_ - self->ctx.curNL1Idx_;
    if (self->ctx.enableStepNIncludeDkNocinhwk_) {
        self->ctx.curStepN_ = nL0RemainLen > (self->ctx.nIter_ % self->ctx.curStepN_) ?
            self->ctx.curStepN_ : nL1RemainLen;
    } else if (self->ctx.enableStepNTail_ && notLastLoop) {
        // 若stepN有尾块，且非最后一次循环，需考虑dk循环，先循环cin后循环dk
        self->ctx.curStepN_ = (nL0RemainLen % self->ctx.cinHkWkLoop_) > self->ctx.tiling_->stepN ?
            self->ctx.tiling_->stepN : (nL1RemainLen % self->ctx.cinHkWkLoop_);
    } else {
        self->ctx.curStepN_ = nL0RemainLen > self->ctx.tiling_->stepN ? self->ctx.tiling_->stepN : nL1RemainLen;
    }
}

template <class Intf>
struct Init {
    // 定义call函数的默认重载函数，支持任意类型任意数量的参数
    DECLARE_DEFAULT_OVERLOADING_FUN(Intf, ConvolutionBackpropFunc);
    static __aicore__ inline void call(Intf *self, const AscendC::conv_bp_v2_kernel::TConv3DDwTiling *__restrict tiling, bool seperateDk)
    {
        self->ctx.tiling_ = tiling;
        self->ctx.seperateDk_ = seperateDk;
        CheckTiling<Intf>(self);
        InitParams<Intf>(self);
        InitTque<Intf>(self);
        if (self->ctx.tiling_->hf32Flag) {
            SetHF32Mode(true);
        }
    }
};

template <class Intf>
struct SetFmap {
    DECLARE_DEFAULT_OVERLOADING_FUN(Intf, ConvolutionBackpropFunc);
    static __aicore__ inline void call(Intf *self, const GlobalTensor<typename Intf::SrcT> &fmap)
    {
        self->ctx.fmapGlobal_ = fmap;
    }
};

template <class Intf>
struct SetOutBackprop {
    DECLARE_DEFAULT_OVERLOADING_FUN(Intf, ConvolutionBackpropFunc);
    static __aicore__ inline void call(Intf *self, const GlobalTensor<typename Intf::SrcT> &outBackprop)
    {
        self->ctx.outBackPropGlobal_ = outBackprop;
    }
};

template <class Intf>
struct SetSingleShape {
    DECLARE_DEFAULT_OVERLOADING_FUN(Intf, ConvolutionBackpropFunc);
    static __aicore__ inline void call(Intf *self, uint64_t singleShapeM, uint64_t singleShapeN, uint64_t singleShapeK,
                                                   uint32_t singleShapeCin, uint32_t singleShapeBatch)
    {
        self->ctx.singleShapeCout_ = singleShapeM;
        self->ctx.singleShapeBatch_ = singleShapeBatch;
        self->ctx.singleShapeCin_ = DivHkWk(singleShapeN, self->ctx.hwK_);
        if (Intf::Config::xType::format == ConvolutionBackprop::CubeFormat::NDC1HWC0) {
            self->ctx.singleShapeCin_ =
            ShiftCeilChannelSize<Intf>(self->ctx.singleShapeCin_, self->ctx.tiling_->channelSize) * self->ctx.tiling_->channelSize;
        } else {
            // self->ctx.singleShapeCin_包含singleShapeCin和singleShapeDk
            self->ctx.singleShapeDk_ = Ceil(self->ctx.singleShapeCin_, singleShapeCin);
            self->ctx.singleShapeCin_ = singleShapeCin;
        }

        self->ctx.singleShapeHo_ = singleShapeK / self->ctx.tiling_->wo;
        InitStepMParams<Intf>(self);
        InitStepKParams<Intf>(self);
        InitStepNParams<Intf>(self);
        // 是否stepN上包含dk， 未包含cinhkwk
        self->ctx.enableStepNIncludeDkNocinhwk_ = (self->ctx.tiling_->stepN != 1) && (self->ctx.tiling_->stepN >
            self->ctx.cinHkWkLoop_) && (self->ctx.tiling_->stepN  % self->ctx.cinHkWkLoop_ != 0);
        // 是否stepN有尾块，例如nIter_=6，stepN为4，则存在尾块2，默认stepN大于cinHkWkLoop时不存在stepN尾块情况
        self->ctx.enableStepNTail_ = (self->ctx.tiling_->stepN != 1) && (self->ctx.cinHkWkLoop_ >
            self->ctx.tiling_->stepN) && (self->ctx.cinHkWkLoop_ % self->ctx.tiling_->stepN != 0);
        // 每次iterall之前都需将如下变量初始化赋值一次，否则会出现尾块精度问题
        self->ctx.head_ = 0;
        self->ctx.tail_ = 0;
        self->ctx.nLoopHead_ = 0;
        self->ctx.lastNIdx_ = -1;
        self->ctx.cinRemainLen_ = self->ctx.singleShapeCin_;
        self->ctx.nLoopCinRemainLen_ = self->ctx.singleShapeCin_;
    }
};

template <class Intf>
struct SetStartIdx {
    DECLARE_DEFAULT_OVERLOADING_FUN(Intf, ConvolutionBackpropFunc);
    static __aicore__ inline void call(Intf *self, uint64_t batchDoutIdx, uint32_t hoStartIdx, int32_t dkIdx)
    {
        self->ctx.batchDoutStartIdx_ = batchDoutIdx;
        self->ctx.hoStartIdx_ = hoStartIdx;
        self->ctx.hiStartIdx_ = static_cast<int32_t>(hoStartIdx) * self->ctx.tiling_->strideH - self->ctx.tiling_->padUp;
        self->ctx.dkStartIdx_ = dkIdx;
    }
};

template <class Intf, bool sync>
struct UpdateMNIdx {
    DECLARE_DEFAULT_OVERLOADING_FUN(Intf, ConvolutionBackpropFunc);
    static __aicore__ inline bool call(Intf* self, Out2L1ScalarParams& out2L1Params)
    {
        /*
        |   <---------singleShapeM------->        |
        |  <---L1A_ping--->  |  <---L1A_pong--->  |
        |_L0A1_|_L0A2_|_L0A3_|_L0A4_|_L0A5_|_L0A6_|
        ↑  <--curStepM_-->    |                    ↑
        curML0Idx_            ↑                  mIter_
        curML1Idx_        next_curML1Idx

        |   <---------singleShapeN------->        |
        |  <---L1B_ping--->  |  <---L1B_pong--->  |
        |_L0B1_|_L0B2_|_L0B3_|_L0B4_|_L0B5_|_L0B6_|
        ↑  <--curStepN_-->    |                    ↑
        curNL0Idx_            ↑                   nIter_
        curNL1Idx_       next_curNL1Idx

        order_N表示L1上驻留B循环A，顺序为L1A_ping * L1B_ping, L1A_pong * L1B_ping，L1A_ping * L1B_pong，L1A_pong * L1B_pong
        L0上也是驻留B，循环A
        order_N: L0A1*L0B1, L0A2*L0B1, L0A3*L0B1, L0A1*L0B2 ………… L0A3*L0B3，L0A4*L0B1，L0A5*L0B1 …… L0A6*L0B6
        order_M: L0A1*L0B1, L0A1*L0B2, L0A1*L0B3, L0A2*L0B1 ………… L0A3*L0B3，L0A1*L0B4，L0A1*L0B5 …… L0A6*L0B6
        */
        // 更新idx，用L1、L1step、L0三个指针控制走位和计算offset，表示计算第几个mL0 * baseN

        // 以开DB为例，循环顺序NMK时，如果K方向需要两块buffer，M轴每循环到stepM最后一次时需要释放AL1上的ping pong buffer，第一次循环时载入
        // 如果循环顺序为MNK时，如果K方向需要两块buffer，M轴最后一次循环时需要释放AL1上的ping pong buffer，第一次循环时载入
        // 如果K方向需要的buffer数量大于bl1Pbuffer，当K循环到stepKa时就需要置换AL1
        // B矩阵计算思路同A矩阵，区别是MN反过来

        bool kIterCeilStepKaGreaterAl1Pbuffer =
            self->ctx.kIter_ > self->ctx.tiling_->stepKa * self->ctx.tiling_->al1Pbuffer;
        bool kIterCeilStepKbGreaterBl1Pbuffer =
            self->ctx.kIter_ > self->ctx.tiling_->stepKb * self->ctx.tiling_->bl1Pbuffer;
        self->ctx.isSplitWo_ = (self->ctx.tiling_->splitWo == self->ctx.tiling_->wo) ? 0 : 1;
        // 当singleShapeBatch大于1时，batchDout循环内移动，L1不驻留数据;当启动splitWo时,L1不能驻留数据
        out2L1Params.isLoad2L1A = kIterCeilStepKaGreaterAl1Pbuffer || self->ctx.singleShapeBatch_ > 1
                                    || self->ctx.isSplitWo_ == 1 || self->ctx.tiling_->isSplitKernelHW;
        out2L1Params.isFreeAL1 = kIterCeilStepKaGreaterAl1Pbuffer || self->ctx.singleShapeBatch_ > 1
                                    || self->ctx.isSplitWo_ == 1 || self->ctx.tiling_->isSplitKernelHW;
        out2L1Params.isLoad2L1B = kIterCeilStepKbGreaterBl1Pbuffer || self->ctx.singleShapeBatch_ > 1
                                    || self->ctx.isSplitWo_ == 1 || self->ctx.tiling_->isSplitKernelHW;
        out2L1Params.isFreeBL1 = kIterCeilStepKbGreaterBl1Pbuffer || self->ctx.singleShapeBatch_ > 1
                                    || self->ctx.isSplitWo_ == 1 || self->ctx.tiling_->isSplitKernelHW;
    
        if (unlikely(self->ctx.isFirstIter_)) {
            self->ctx.curML0Idx_ = 0;
            self->ctx.curNL0Idx_ = 0;
            self->ctx.curML1Idx_ = 0;
            self->ctx.curNL1Idx_ = 0;
            self->ctx.isFirstIter_ = false;
            self->ctx.curStepM_ = self->ctx.mIter_ > self->ctx.tiling_->stepM
                                        ? self->ctx.tiling_->stepM
                                        : self->ctx.mIter_;
            if (self->ctx.enableStepNIncludeDkNocinhwk_) {
                uint32_t curStepN = self->ctx.tiling_->stepN / self->ctx.cinHkWkLoop_ * self->ctx.cinHkWkLoop_;
                self->ctx.curStepN_ = self->ctx.nIter_ > curStepN ? curStepN : self->ctx.nIter_;
            } else {
                self->ctx.curStepN_ = self->ctx.nIter_ > self->ctx.tiling_->stepN ?
                    self->ctx.tiling_->stepN : self->ctx.nIter_;
            }
            bool isLastNLoop = self->ctx.nIter_ == 1;
            bool isLastMLoop = self->ctx.mIter_ == 1;
            bool isNLastStep = isLastNLoop || self->ctx.tiling_->stepN == 1;
            bool isMLastStep = isLastMLoop || self->ctx.tiling_->stepM == 1;
            out2L1Params.isLoad2L1A = true;
            out2L1Params.isFreeAL1 = kIterCeilStepKaGreaterAl1Pbuffer || self->ctx.singleShapeBatch_ > 1  || self->ctx.isSplitWo_ == 1 ||
                self->ctx.tiling_->isSplitKernelHW || (self->ctx.tiling_->iterateOrder && isMLastStep) ||                  // OrderN
                (!(self->ctx.tiling_->iterateOrder) && isLastNLoop && isMLastStep);  // OrderM
            out2L1Params.isLoad2L1B = true;
            out2L1Params.isFreeBL1 = kIterCeilStepKbGreaterBl1Pbuffer || self->ctx.singleShapeBatch_ > 1  || self->ctx.isSplitWo_ == 1 ||
                self->ctx.tiling_->isSplitKernelHW || (self->ctx.tiling_->iterateOrder && isLastMLoop && isNLastStep) ||    // OrderN
                (!(self->ctx.tiling_->iterateOrder) && isNLastStep);                  // OrderM
        } else if (likely(self->ctx.tiling_->iterateOrder == static_cast<int>(IterateOrder::ORDER_N))) {
            if (++self->ctx.curML0Idx_ >= self->ctx.curML1Idx_ + self->ctx.curStepM_) {
                self->ctx.curML0Idx_ = self->ctx.curML1Idx_;
                if (++self->ctx.curNL0Idx_ >= self->ctx.curNL1Idx_ + self->ctx.curStepN_) {
                    self->ctx.curML1Idx_ += self->ctx.curStepM_;
                    if (self->ctx.curNL0Idx_ >= self->ctx.nIter_ && self->ctx.curML1Idx_ >= self->ctx.mIter_) {
                        return false;
                    }
                    if (self->ctx.curML1Idx_ >= self->ctx.mIter_) {
                        self->ctx.curML1Idx_ = 0;
                        self->ctx.curNL1Idx_ += self->ctx.curStepN_;
                    }
                    UpdateIdxAndStep<Intf>(self);
                    if (self->ctx.curML0Idx_ == 0) {
                        out2L1Params.isLoad2L1B = true; // OrderN, N轴循环结束，需要置换BL1
                    }
                }
                out2L1Params.isLoad2L1A = true; // OrderN, M轴循环结束，需要置换AL1
            }
            if (unlikely(self->ctx.curML0Idx_ == self->ctx.mIter_ - 1) ||
                self->ctx.curML0Idx_ == self->ctx.curML1Idx_ + self->ctx.curStepM_ - 1) {
                out2L1Params.isFreeAL1 = true; // OrderN, M轴最后一次循环，需要释放AL1
            }
            if (unlikely(self->ctx.curML0Idx_ == self->ctx.mIter_ - 1) &&
                (unlikely(self->ctx.curNL0Idx_ == self->ctx.nIter_ - 1) ||
                self->ctx.curNL0Idx_ == self->ctx.curNL1Idx_ + self->ctx.curStepN_ - 1)) {
                out2L1Params.isFreeBL1 = true; // OrderN, N轴最后一次循环，需要释放BL1
            }
        } else {  // order_M
            if (++self->ctx.curNL0Idx_ >= self->ctx.curNL1Idx_ + self->ctx.curStepN_) {
                self->ctx.curNL0Idx_ = self->ctx.curNL1Idx_;
                if (++self->ctx.curML0Idx_ >= self->ctx.curML1Idx_ + self->ctx.curStepM_) {
                    self->ctx.curNL1Idx_ += self->ctx.curStepN_;
                    if (self->ctx.curML0Idx_ >= self->ctx.mIter_ && self->ctx.curNL1Idx_ >= self->ctx.nIter_) {
                        return false;
                    }
                    if (self->ctx.curNL1Idx_ >= self->ctx.nIter_) {
                        self->ctx.curNL1Idx_ = 0;
                        self->ctx.curML1Idx_ += self->ctx.curStepM_;
                    }
                    UpdateIdxAndStep<Intf>(self);
                    if (self->ctx.curNL0Idx_ == 0) {
                        out2L1Params.isLoad2L1A = true; // OrderM, M轴循环结束，需要置换AL1
                    }
                }
                out2L1Params.isLoad2L1B = true; // OrderM, N轴循环结束，需要置换BL1
            }
            if (unlikely(self->ctx.curNL0Idx_ == self->ctx.nIter_ - 1) &&
                (unlikely(self->ctx.curML0Idx_ == self->ctx.mIter_ - 1) ||
                self->ctx.curML0Idx_ == self->ctx.curML1Idx_ + self->ctx.curStepM_ - 1)) {
                out2L1Params.isFreeAL1 = true; // OrderM, M轴最后一次循环，需要释放AL1
            }
            if (unlikely(self->ctx.curNL0Idx_ == self->ctx.nIter_ - 1) ||
                self->ctx.curNL0Idx_ == self->ctx.curNL1Idx_ + self->ctx.curStepN_ - 1) {
                out2L1Params.isFreeBL1 = true; // OrderM, N轴最后一次循环，需要释放BL1
            }
        }
        return true;
    }
};

template <class Intf, bool sync>
struct Iterate {
    DECLARE_DEFAULT_OVERLOADING_FUN(Intf, ConvolutionBackpropFunc);
    static __aicore__ inline bool call(Intf *self, bool enPartialSum)
    {
        Out2L1ScalarParams out2L1Params;
        if(!self->template UpdateMNIdx<sync>(out2L1Params)){
            return false;
        }
        self->template Compute(out2L1Params);
        return true;
    }
};

template <class Intf, bool sync>
struct IterateAll {
    DECLARE_DEFAULT_OVERLOADING_FUN(Intf, ConvolutionBackpropFunc);
    static __aicore__ inline void call(Intf *self, const GlobalTensor<typename Intf::DstT> &output, uint8_t enAtomic)
    {
        if (self->ctx.tiling_->group == self->ctx.tiling_->realGroup) {
            while (self->template Iterate<sync>()) {
                self->template GetTensorC<sync>(output, enAtomic);
            }
        } else {
            while (self->template Iterate<sync>()) {
                if ASCEND_IS_AIC {
                    CrossCoreWaitFlag<SYNC_MODE4, PIPE_FIX>(FLAG_FIXP_ID);
                    self->template GetTensorC<sync>(output, enAtomic);
                    CrossCoreSetFlag<SYNC_MODE4, PIPE_FIX>(FLAG_MTE3_ID);
                }

                if ASCEND_IS_AIV {
                    CrossCoreSetFlag<SYNC_MODE4, PIPE_MTE3>(FLAG_FIXP_ID);
                    CrossCoreWaitFlag<SYNC_MODE4, PIPE_V>(FLAG_MTE3_ID);
                    self->template VecPostProcess<sync>(output, enAtomic);
                }
            }
        }
        self->ctx.isFirstIter_ = true;
    }
};

template <class Intf, bool sync>
struct GetTensorC {
    DECLARE_DEFAULT_OVERLOADING_FUN(Intf, ConvolutionBackpropFunc);
    static __aicore__ inline void call(Intf *self, const GlobalTensor<typename Intf::DstT> &output,
        uint8_t enAtomic = 0, bool enSequentialWrite = false)
    {
        LoadL0c2Gm<Intf>(self, output, enAtomic, enSequentialWrite);
    }
};

template <class Intf, bool sync>
struct VecPostProcess {
    DECLARE_DEFAULT_OVERLOADING_FUN(Intf, ConvolutionBackpropFunc);
    static __aicore__ inline void call(Intf *self, const GlobalTensor<typename Intf::DstT> &output,
        uint8_t enAtomic = 0)
    {
        Rearrange2Gm<Intf>(self, output, enAtomic);
    }
};

template <class Intf>
struct End {
    DECLARE_DEFAULT_OVERLOADING_FUN(Intf, ConvolutionBackpropFunc);
    static __aicore__ inline void call(Intf *self)
    {
        self->ctx.a1Ping_.FreeAllEvent();
        if (self->ctx.tiling_->al1Pbuffer > 1) {
            self->ctx.a1Pong_.FreeAllEvent();
        }
        self->ctx.b1Ping_.FreeAllEvent();
        if (self->ctx.tiling_->bl1Pbuffer > 1) {
            self->ctx.b1Pong_.FreeAllEvent();
        }
        self->ctx.l0cPing_.FreeAllEvent();
        if (self->ctx.tiling_->cl0Pbuffer > 1) {
            self->ctx.l0cPong_.FreeAllEvent();
        }

        if (self->ctx.tiling_->hf32Flag) {
            SetHF32Mode(false);
        }
    }
};

template <class Intf>
struct SetDeterministicCoreInfo {
    DECLARE_DEFAULT_OVERLOADING_FUN(Intf, ConvolutionBackpropFunc);
    static __aicore__ inline void call(Intf *self, uint32_t addCoreNum, uint32_t addCoreIndex, uint32_t addCoreIndexTotal,
        bool isNoDeter)
    {
        self->ctx.deterAddCoreNum_ = addCoreNum;
        self->ctx.deterAddCoreIndex_ = addCoreIndex;
        self->ctx.coreStartIndexTotal_ = addCoreIndexTotal;
        self->ctx.isNoDeterministic_ = isNoDeter;
    }
};

template <class Intf, bool sync>
struct DeterministicReduceKInUb {
    DECLARE_DEFAULT_OVERLOADING_FUN(Intf, ConvolutionBackpropFunc);
    static __aicore__ inline void call(Intf *self, const GlobalTensor<typename Intf::DstT> &output,
        const GlobalTensor<typename Intf::DstT> &userGm)
    {
        // 确定性计算累加
        DeterministicAddFunc<Intf>(self, output, userGm);
    }
};
}  // namespace ConvolutionBackpropFunc
#endif